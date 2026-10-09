//! gpt-oss's routed experts, as both engines hold and run them.
//!
//! Three checkpoint layouts load, each to the `[E, in, out]`-applied stack
//! [`crate::native_moe::switch_glu`] dispatches over:
//!
//! * the release's MXFP4 (`experts.gate_up_proj_blocks` `[E, 2I, H/32, 16]`
//!   `uint8` with `_scales` `[E, 2I, H/32]` E8M0, `down_proj_blocks`
//!   `[E, H, I/32, 16]`), kept packed: the blocks' bytes viewed as `uint32`
//!   are MLX's mxfp4 layout (two E2M1 values a byte, the low nibble first;
//!   one power-of-two scale per 32 values, bias 127), run by `gather_qmm`,
//! * transformers' dense layout (`experts.gate_up_proj` `[E, H, 2I]`,
//!   `down_proj` `[E, I, H]`, applied as `x @ W`),
//! * the stacked per-projection layout MLX conversions ship
//!   (`experts.{gate,up,down}_proj.weight`, `[E, out, in]`).
//!
//! Gate and up are interleaved along the fused projection's outputs
//! (`gate_up[..., ::2]`, `gate_up[..., 1::2]`), biases included.

use std::collections::HashMap;

use crate::InlineArray;
use crate::QuantizedMode;
use crate::native_weight::{LayerWeight, QuantParams};

/// One layer's routed experts with their biases.
#[derive(Clone)]
pub struct GptOssExperts {
    /// Gate projection, `hidden → intermediate` per expert.
    pub gate: LayerWeight,
    /// Up projection, `hidden → intermediate` per expert.
    pub up: LayerWeight,
    /// Down projection, `intermediate → hidden` per expert.
    pub down: LayerWeight,
    /// `[E, intermediate]`.
    pub gate_bias: InlineArray,
    /// `[E, intermediate]`.
    pub up_bias: InlineArray,
    /// `[E, hidden]`.
    pub down_bias: InlineArray,
}

impl std::fmt::Debug for GptOssExperts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GptOssExperts")
            .field("experts", &self.gate_bias.dim(0))
            .field("packed", &self.is_packed())
            .finish()
    }
}

/// The even (`0`) or odd (`1`) entries of `x` along `axis`, transformers'
/// `x[..., ::2]` / `x[..., 1::2]` there, as a contiguous array.
fn interleaved(x: &InlineArray, axis: usize, which: i32) -> InlineArray {
    let mut shape: Vec<i32> = (0..x.ndim()).map(|i| x.dim(i)).collect();
    let n = shape[axis];
    shape[axis] = n / 2;
    shape.insert(axis + 1, 2);
    let mut start = vec![0; shape.len()];
    let mut stop = shape.clone();
    start[axis + 1] = which;
    stop[axis + 1] = which + 1;
    let half = x
        .reshape(&shape)
        .slice(&start, &stop)
        .squeeze(axis as i32 + 1);
    // `+ 0` is a copy for every dtype, integer payloads included, and makes
    // the strided half contiguous for the kernels.
    half.add(&InlineArray::zeros(&[1], half.dtype_raw()))
}

/// MXFP4 blocks `[..., out, G, 16]` `uint8` and scales `[..., out, G]` as an
/// MLX mxfp4 weight: the bytes viewed as `uint32`, `[..., out, G * 4]`.
fn mxfp4(blocks: &InlineArray, scales: InlineArray) -> Result<LayerWeight, String> {
    const UINT32: i32 = 3;
    let n = blocks.ndim();
    if n < 3 || blocks.dim(n - 1) != 16 {
        return Err(format!(
            "gpt_oss: MXFP4 blocks have shape {:?}; expected [..., out, groups, 16]",
            blocks.shape()
        ));
    }
    let mut shape: Vec<i32> = (0..n - 2).map(|i| blocks.dim(i)).collect();
    shape.push(blocks.dim(n - 2) * 4);
    let weight = blocks.view(UINT32).reshape(&shape);
    Ok(LayerWeight::Quantized {
        weight,
        scales,
        biases: None,
        tensor_scale: None,
        params: QuantParams::defaults_for(QuantizedMode::Mxfp4),
    })
}

impl GptOssExperts {
    /// Read the experts under `prefix` (`model.layers.N.mlp.experts`) from
    /// any of the layouts in the module docs.
    pub fn from_checkpoint(
        raw: &HashMap<String, InlineArray>,
        prefix: &str,
    ) -> Result<Self, String> {
        let get = |name: &str| {
            raw.get(&format!("{prefix}.{name}"))
                .cloned()
                .ok_or_else(|| format!("gpt_oss: missing weight key: {prefix}.{name}"))
        };
        let swap = |x: InlineArray| LayerWeight::Dense(x.transpose_axes(&[0, 2, 1]));
        let experts = if raw.contains_key(&format!("{prefix}.gate_up_proj_blocks")) {
            // [E, 2I, G, 16] / [E, 2I, G]: gate and up interleaved on axis 1.
            let blocks = get("gate_up_proj_blocks")?;
            let scales = get("gate_up_proj_scales")?;
            let bias = get("gate_up_proj_bias")?;
            Self {
                gate: mxfp4(&interleaved(&blocks, 1, 0), interleaved(&scales, 1, 0))?,
                up: mxfp4(&interleaved(&blocks, 1, 1), interleaved(&scales, 1, 1))?,
                down: mxfp4(&get("down_proj_blocks")?, get("down_proj_scales")?)?,
                gate_bias: interleaved(&bias, 1, 0),
                up_bias: interleaved(&bias, 1, 1),
                down_bias: get("down_proj_bias")?,
            }
        } else if raw.contains_key(&format!("{prefix}.gate_up_proj")) {
            // [E, H, 2I], applied `x @ W`: gate and up interleaved on axis 2.
            let gate_up = get("gate_up_proj")?;
            let bias = get("gate_up_proj_bias")?;
            Self {
                gate: LayerWeight::Dense(interleaved(&gate_up, 2, 0)),
                up: LayerWeight::Dense(interleaved(&gate_up, 2, 1)),
                down: LayerWeight::Dense(get("down_proj")?),
                gate_bias: interleaved(&bias, 1, 0),
                up_bias: interleaved(&bias, 1, 1),
                down_bias: get("down_proj_bias")?,
            }
        } else {
            // Stacked `[E, out, in]`.
            Self {
                gate: swap(get("gate_proj.weight")?),
                up: swap(get("up_proj.weight")?),
                down: swap(get("down_proj.weight")?),
                gate_bias: get("gate_proj.bias")?,
                up_bias: get("up_proj.bias")?,
                down_bias: get("down_proj.bias")?,
            }
        };
        crate::check_last_error()
            .map_err(|e| format!("gpt_oss: reading the experts under {prefix} failed: {e}"))?;
        Ok(experts)
    }

    /// Whether the expert weights are packed (MXFP4).
    pub fn is_packed(&self) -> bool {
        !self.gate.is_dense()
    }

    /// Re-materialise every array into a fresh buffer.
    pub fn copy_fresh(&self, zero: &InlineArray) -> Self {
        let fresh = |a: &InlineArray| crate::native_weight::copy_fresh_arr(a, zero);
        Self {
            gate: self.gate.copy_fresh(zero),
            up: self.up.copy_fresh(zero),
            down: self.down.copy_fresh(zero),
            gate_bias: fresh(&self.gate_bias),
            up_bias: fresh(&self.up_bias),
            down_bias: fresh(&self.down_bias),
        }
    }

    /// The same experts with packed weights unpacked to `dtype`, `[E, in,
    /// out]`, out of the autograd graph.
    pub fn unpacked(&self, dtype: i32) -> Self {
        let dense = |w: &LayerWeight| match w {
            LayerWeight::Quantized {
                weight,
                scales,
                biases,
                params,
                ..
            } => LayerWeight::Dense(
                weight
                    .dequantize_mode(
                        scales,
                        biases.as_ref(),
                        params.group_size,
                        params.bits,
                        params.mode,
                    )
                    .as_dtype(dtype)
                    .transpose_axes(&[0, 2, 1])
                    .stop_gradient(),
            ),
            dense => dense.clone(),
        };
        Self {
            gate: dense(&self.gate),
            up: dense(&self.up),
            down: dense(&self.down),
            ..self.clone()
        }
    }

    /// `Σ_k scores[s, k] · (down(glu(gate(x) + b_g, up(x) + b_u)) + b_d)` over
    /// each row's experts `inds[s, k]`, with gpt-oss's clamped GLU
    /// (`alpha`, `limit`). `x_flat` is `[S, hidden]`, `inds` and `scores`
    /// `[S, top_k]`; the result is `[S, hidden]`.
    pub fn forward(
        &self,
        x_flat: &InlineArray,
        inds: &InlineArray,
        scores: &InlineArray,
        alpha: f32,
        limit: f32,
    ) -> InlineArray {
        let (rows, top_k) = (inds.dim(0), inds.dim(1));
        let inter = self.gate_bias.dim(1);
        let bias_for =
            |bias: &InlineArray| bias.take_axis(inds, 0).reshape(&[rows, top_k, 1, inter]);
        let (gate_b, up_b) = (bias_for(&self.gate_bias), bias_for(&self.up_bias));
        let routed = crate::native_moe::switch_glu(
            x_flat,
            &self.gate,
            &self.up,
            &self.down,
            inds,
            scores,
            |gate, up| glu(&gate.add(&gate_b), &up.add(&up_b), alpha, limit),
        );
        // The down bias, weighted like the expert outputs it belongs to.
        let down_b = self
            .down_bias
            .take_axis(inds, 0)
            .multiply(&scores.reshape(&[rows, top_k, 1]))
            .sum_axis(1, false);
        routed.add(&down_b)
    }
}

/// gpt-oss's clamped GLU, transformers' `GptOssExperts._apply_gate`:
///
/// ```text
///   gate = min(gate, limit)
///   up   = clip(up, -limit, limit)
///   out  = gate · σ(alpha · gate) · (up + 1)
/// ```
pub fn glu(gate: &InlineArray, up: &InlineArray, alpha: f32, limit: f32) -> InlineArray {
    let gate = gate.minimum(&InlineArray::scalar_like(limit, gate));
    let up = up.clip(
        Some(&InlineArray::scalar_like(-limit, up)),
        Some(&InlineArray::scalar_like(limit, up)),
    );
    let glu = gate.multiply(
        &gate
            .multiply(&InlineArray::scalar_like(alpha, &gate))
            .sigmoid(),
    );
    glu.multiply(&up.add(&InlineArray::scalar_like(1.0, &up)))
}
