//! Projection and embedding weights that may be stored packed.
//!
//! MLX models quantization as a property of the *layer*, not of the
//! architecture: `nn.QuantizedLinear` and `nn.QuantizedEmbedding` are drop-in
//! replacements, and a model calls `embed_tokens.as_linear(x)` without knowing
//! which it holds. That is why mlx-lm's architectures contain no `if
//! quantized` branch anywhere.
//!
//! The native engines are hand-rolled rather than module-based, so the
//! equivalent is these two enums plus their dispatchers. An architecture asks
//! for `matmul_from` or `lookup` and gets the right kernel.
//!
//! Layout note: MLX stores a quantized weight row-major as
//! `[out, in / (32 / bits)]` and expects `transpose = true` to signal that it
//! logically needs transposing. Dense weights are pre-transposed to `[in, out]`
//! at load time instead, so the per-step matmul is contiguous. The two arrive
//! at the same product by different routes, which is why only the dense arm
//! calls `.t()` at load.

use crate::{InlineArray, QuantizedMode};

/// How a packed tensor was quantized.
///
/// Mirrors the trailing arguments of `mlx::core::quantized_matmul`, so a
/// checkpoint's `quantization` block maps onto it field for field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantParams {
    pub group_size: i32,
    pub bits: i32,
    pub mode: QuantizedMode,
}

impl QuantParams {
    /// MLX's defaults for a mode, used when a config names the mode but omits
    /// the widths. Taken from `_defaults_for_mode` in `mlx/nn/layers/quantized.py`.
    pub fn defaults_for(mode: QuantizedMode) -> Self {
        let (group_size, bits) = match mode {
            QuantizedMode::Affine => (64, 4),
            QuantizedMode::Mxfp4 => (32, 4),
            QuantizedMode::Nvfp4 => (16, 4),
            QuantizedMode::Mxfp8 => (32, 8),
        };
        Self {
            group_size,
            bits,
            mode,
        }
    }
}

/// A projection weight, dense or packed.
///
/// One `Quantized` variant covers every mode: `biases` is `None` for the
/// floating-point modes (mxfp4, nvfp4, mxfp8), which carry scales only. That
/// matches `mlx::core::quantized_matmul`, where `biases` is
/// `std::optional<array>`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum LayerWeight {
    /// Pre-transposed to `[in, out]`; used with `x.matmul(w)`.
    Dense(InlineArray),
    /// Packed `uint32` as the checkpoint stores it, with `transpose = true`.
    Quantized {
        weight: InlineArray,
        scales: InlineArray,
        biases: Option<InlineArray>,
        params: QuantParams,
    },
}

impl LayerWeight {
    /// Build from the three tensors a checkpoint stores for one module.
    ///
    /// `scales` absent means the module was left dense, which is how a
    /// mixed-precision checkpoint marks the layers it did not quantize.
    pub fn new(
        weight: InlineArray,
        scales: Option<InlineArray>,
        biases: Option<InlineArray>,
        params: QuantParams,
    ) -> Self {
        match scales {
            Some(scales) => Self::Quantized {
                weight,
                scales,
                biases,
                params,
            },
            // Only the dense arm transposes: see the layout note above.
            None => Self::Dense(weight.t()),
        }
    }

    /// The packed or dense payload, for pointer export and buffer refresh.
    pub fn weight_arr(&self) -> &InlineArray {
        match self {
            Self::Dense(w) => w,
            Self::Quantized { weight, .. } => weight,
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// The quantization parameters, or `None` when dense.
    pub fn params(&self) -> Option<QuantParams> {
        match self {
            Self::Dense(_) => None,
            Self::Quantized { params, .. } => Some(*params),
        }
    }

    /// `x @ self`.
    #[inline(always)]
    pub fn matmul_from(&self, x: &InlineArray) -> InlineArray {
        match self {
            Self::Dense(w) => x.matmul(w),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => x.quantized_matmul_mode(
                weight,
                scales,
                biases.as_ref(),
                true,
                params.group_size,
                params.bits,
                params.mode,
            ),
        }
    }

    /// Gather-matmul for MoE expert dispatch.
    ///
    /// Every expert tensor in a layer has to be the same variant; a checkpoint
    /// that quantized some experts and not others is not something MLX's
    /// gather kernels can express.
    #[inline(always)]
    pub fn gather_mm_from(
        &self,
        x: &InlineArray,
        lhs_indices: Option<&InlineArray>,
        rhs_indices: Option<&InlineArray>,
        sorted: bool,
    ) -> InlineArray {
        match self {
            Self::Dense(w) => x.gather_mm(w, lhs_indices, rhs_indices, sorted),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => x.gather_qmm_mode(
                weight,
                scales,
                biases.as_ref(),
                lhs_indices,
                rhs_indices,
                true,
                params.group_size,
                params.bits,
                sorted,
                params.mode,
            ),
        }
    }

    /// Re-materialise every array into a fresh buffer.
    pub fn copy_fresh(&self, zero: &InlineArray) -> Self {
        match self {
            Self::Dense(w) => Self::Dense(copy_fresh_arr(w, zero)),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => Self::Quantized {
                weight: copy_fresh_arr(weight, zero),
                scales: copy_fresh_arr(scales, zero),
                biases: biases.as_ref().map(|b| copy_fresh_arr(b, zero)),
                params: *params,
            },
        }
    }

    /// Queue every array for evaluation without blocking.
    pub fn async_eval_ref(&self) {
        match self {
            Self::Dense(w) => w.async_eval_ref(),
            Self::Quantized {
                weight,
                scales,
                biases,
                ..
            } => {
                weight.async_eval_ref();
                scales.async_eval_ref();
                if let Some(b) = biases {
                    b.async_eval_ref();
                }
            }
        }
    }
}

/// An embedding table, dense or packed.
///
/// The two methods are `mlx.nn.Embedding`'s: a lookup, and `as_linear` for the
/// tied LM head. A quantized lookup gathers the packed rows *and* their scales
/// and biases, then dequantizes only those rows, which is why it cannot go
/// through `LayerWeight`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum EmbeddingWeight {
    /// `[vocab, hidden]`.
    Dense(InlineArray),
    Quantized {
        weight: InlineArray,
        scales: InlineArray,
        biases: Option<InlineArray>,
        params: QuantParams,
    },
}

impl EmbeddingWeight {
    pub fn new(
        weight: InlineArray,
        scales: Option<InlineArray>,
        biases: Option<InlineArray>,
        params: QuantParams,
    ) -> Self {
        match scales {
            Some(scales) => Self::Quantized {
                weight,
                scales,
                biases,
                params,
            },
            // No transpose: an embedding is indexed by row, and `as_linear`
            // takes the transpose at the point of use.
            None => Self::Dense(weight),
        }
    }

    pub fn weight_arr(&self) -> &InlineArray {
        match self {
            Self::Dense(w) => w,
            Self::Quantized { weight, .. } => weight,
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// Row lookup: `[.., T] -> [.., T, hidden]`.
    ///
    /// The quantized arm is `mlx.nn.QuantizedEmbedding.__call__`:
    /// `dequantize(w[x], scales[x], biases[x])`. Gathering first means only the
    /// rows this batch touches are unpacked, rather than the whole table.
    pub fn lookup(&self, ids: &InlineArray) -> InlineArray {
        match self {
            Self::Dense(w) => w.take_axis(ids, 0),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => {
                let rows = weight.take_axis(ids, 0);
                let row_scales = scales.take_axis(ids, 0);
                let row_biases = biases.as_ref().map(|b| b.take_axis(ids, 0));
                rows.dequantize_mode(
                    &row_scales,
                    row_biases.as_ref(),
                    params.group_size,
                    params.bits,
                    params.mode,
                )
            }
        }
    }

    /// The table used as the output projection, for a model with tied weights.
    ///
    /// `mlx.nn.QuantizedEmbedding.as_linear`.
    pub fn as_linear(&self, x: &InlineArray) -> InlineArray {
        match self {
            Self::Dense(w) => x.matmul(&w.t()),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => x.quantized_matmul_mode(
                weight,
                scales,
                biases.as_ref(),
                true,
                params.group_size,
                params.bits,
                params.mode,
            ),
        }
    }

    pub fn async_eval_ref(&self) {
        match self {
            Self::Dense(w) => w.async_eval_ref(),
            Self::Quantized {
                weight,
                scales,
                biases,
                ..
            } => {
                weight.async_eval_ref();
                scales.async_eval_ref();
                if let Some(b) = biases {
                    b.async_eval_ref();
                }
            }
        }
    }
}

/// The dtype a checkpoint's trunk computes in.
///
/// ⚠️ **Not the embedding's.** On a quantized checkpoint `embed_tokens.weight`
/// is the packed `uint32` payload, so reading its dtype makes every scalar
/// derived from it, and the whole KV cache, an integer type. The model then
/// loads, runs at the right speed, and decodes noise.
///
/// Norm weights are one-dimensional and no quantizer touches them, so they
/// carry the real compute dtype. Scales are the second choice: they are float
/// by construction and present exactly when the weights are packed.
pub fn detect_model_dtype(lookup: impl Fn(&str) -> Option<i32>) -> i32 {
    const CANDIDATES: [&str; 6] = [
        "model.norm.weight",
        "backbone.norm_f.weight",
        "language_model.model.norm.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    ];
    CANDIDATES
        .iter()
        .filter_map(|key| lookup(key))
        .find(|dt| is_float_dtype(*dt))
        .unwrap_or(crate::dtype::BF16)
}

/// Whether a raw dtype code is one the trunk can compute in.
fn is_float_dtype(dt: i32) -> bool {
    use crate::dtype;
    matches!(dt, dtype::F16 | dtype::F32 | dtype::BF16)
}

/// Force an array into a fresh Metal buffer (add zero, eval, detach).
pub fn copy_fresh_arr(w: &InlineArray, _hint_zero: &InlineArray) -> InlineArray {
    // A packed weight is already `uint32` in a buffer of its own, and adding
    // zero to it would be an integer op on the payload rather than a copy.
    const UINT32: i32 = 3;
    let dt = w.dtype_raw();
    if dt == UINT32 {
        let mut fresh = w.clone();
        fresh.eval();
        fresh.detach();
        return fresh;
    }
    let own_zero = InlineArray::zeros(&[1], dt);
    let mut fresh = w.add(&own_zero);
    fresh.eval();
    fresh.detach();
    fresh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;
    use std::collections::HashMap;

    fn lookup<'a>(map: &'a HashMap<&'a str, i32>) -> impl Fn(&str) -> Option<i32> + 'a {
        move |key: &str| map.get(key).copied()
    }

    /// ⚠️ The regression this exists for: a quantized checkpoint's
    /// `embed_tokens.weight` is packed `uint32`, and taking the model dtype
    /// from it made every derived scalar and the KV cache integer. The model
    /// loaded, ran at the right speed, and decoded noise.
    #[test]
    fn a_packed_embedding_does_not_set_the_model_dtype() {
        let map = HashMap::from([
            ("model.embed_tokens.weight", dtype::U32),
            ("model.embed_tokens.scales", dtype::BF16),
            ("model.norm.weight", dtype::BF16),
        ]);
        assert_eq!(detect_model_dtype(lookup(&map)), dtype::BF16);
    }

    /// With no norm weight to read, the scales still carry a float dtype.
    #[test]
    fn scales_answer_when_no_norm_weight_is_present() {
        let map = HashMap::from([
            ("model.embed_tokens.weight", dtype::U32),
            ("model.embed_tokens.scales", dtype::F16),
        ]);
        assert_eq!(detect_model_dtype(lookup(&map)), dtype::F16);
    }

    #[test]
    fn a_dense_checkpoint_reports_its_own_dtype() {
        let map = HashMap::from([
            ("model.norm.weight", dtype::F32),
            ("model.embed_tokens.weight", dtype::F32),
        ]);
        assert_eq!(detect_model_dtype(lookup(&map)), dtype::F32);
    }

    /// A checkpoint whose keys are all unfamiliar still has to produce a dtype
    /// the trunk can compute in, rather than whatever integer came first.
    #[test]
    fn an_unrecognised_layout_falls_back_to_bfloat16() {
        let map = HashMap::from([("something.else.weight", dtype::U32)]);
        assert_eq!(detect_model_dtype(lookup(&map)), dtype::BF16);
    }

    #[test]
    fn mode_defaults_match_mlx() {
        // `_defaults_for_mode` in mlx/nn/layers/quantized.py.
        for (mode, group_size, bits) in [
            (QuantizedMode::Affine, 64, 4),
            (QuantizedMode::Mxfp4, 32, 4),
            (QuantizedMode::Nvfp4, 16, 4),
            (QuantizedMode::Mxfp8, 32, 8),
        ] {
            let p = QuantParams::defaults_for(mode);
            assert_eq!((p.group_size, p.bits), (group_size, bits), "{mode:?}");
        }
    }
}
