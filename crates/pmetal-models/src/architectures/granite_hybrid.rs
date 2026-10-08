//! The pieces of the Granite family beyond plain Granite: the Mamba-2 mixer
//! of Granite 4.0-H, the routed experts of `granitemoe` / `granitemoehybrid`,
//! the shared MLP every `granitemoeshared` / `granitemoehybrid` layer runs, and
//! the weight loader that maps a released checkpoint onto them.
//!
//! Reference: `transformers.models.granitemoehybrid.modeling_granitemoehybrid`
//! (`GraniteMoeHybridMambaLayer`, `GraniteMoeHybridMoE`, `GraniteMoeHybridMLP`).
//! `granitemoe` and `granitemoeshared` are the same layer with the Mamba and
//! shared-MLP branches removed, so one implementation serves all three.
//!
//! The Mamba-2 scan itself is not reimplemented here: it is the SSD form in
//! [`super::nemotron_h`] ([`ssm_attention`] for a sequence, [`ssm_update_single`]
//! for one cached token), and the conv / SSM state lives in the shared
//! [`MambaCache`](pmetal_mlx::kv_cache::MambaCache). What differs from
//! Nemotron-H is around the scan: Granite normalizes the gated output over the
//! whole intermediate width (not per group), and runs the scan in float32 the
//! way the reference's chunked path does.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use pmetal_bridge::compat::{Array, Exception, Module, ModuleParametersExt, Param, nn, ops};
use pmetal_bridge::impl_module_params;
use pmetal_mlx::kv_cache::MambaCacheEntry;

use super::granite::{GraniteConfig, GraniteForCausalLM};
use super::nemotron_h::{ssm_attention, ssm_update_single};

// =============================================================================
// Mamba-2 mixer
// =============================================================================

/// Granite 4.0-H's Mamba-2 mixer (`GraniteMoeHybridMambaLayer`).
///
/// `in_proj` produces `[gate | x B C | dt]`; `x B C` goes through a causal
/// depthwise conv and SiLU, the SSM scans it, and the output is normalized
/// together with `silu(gate)` before `out_proj`. Field names are the
/// checkpoint's, `A_log` and `D` included, so a loaded or saved state dict
/// round-trips without a rename table.
#[allow(non_snake_case)]
#[derive(Debug)]
pub struct GraniteMamba {
    pub in_proj: nn::Linear,
    /// Depthwise causal conv. MLX layout `[conv_dim, kernel, 1]`; the loader
    /// reshapes a released `[conv_dim, 1, kernel]` weight into it.
    pub conv1d: nn::Conv1d,
    pub dt_bias: Param<Array>,
    pub A_log: Param<Array>,
    pub D: Param<Array>,
    /// Gated RMSNorm over the full `intermediate` width.
    pub norm: nn::RmsNorm,
    pub out_proj: nn::Linear,

    pub num_heads: i32,
    pub head_dim: i32,
    pub n_groups: i32,
    pub state_size: i32,
    pub intermediate: i32,
    pub conv_dim: i32,
    pub conv_kernel: i32,
    pub chunk_size: i32,
    pub norm_eps: f32,
}
impl_module_params!(GraniteMamba; in_proj, conv1d, dt_bias, A_log, D, norm, out_proj);

impl GraniteMamba {
    pub fn new(config: &GraniteConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size;
        let intermediate = config.mamba_intermediate_size();
        let conv_dim = config.mamba_conv_dim();
        let num_heads = config.mamba_n_heads;
        let projection = intermediate + conv_dim + num_heads;

        let in_proj = nn::LinearBuilder::new(hidden, projection)
            .bias(config.mamba_proj_bias)
            .build()?;
        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, config.mamba_d_conv)
            .groups(conv_dim)
            .bias(config.mamba_conv_bias)
            .padding(0)
            .build()?;
        let out_proj = nn::LinearBuilder::new(intermediate, hidden)
            .bias(config.mamba_proj_bias)
            .build()?;
        let norm = nn::RmsNormBuilder::new(intermediate)
            .eps(config.rms_norm_eps)
            .build()?;

        // The reference's own initialisation: `A = arange(1, H + 1)`, unit
        // `D` and `dt_bias`. Loaded checkpoints overwrite all three.
        let a: Vec<f32> = (1..=num_heads).map(|h| (h as f32).ln()).collect();

        Ok(Self {
            in_proj,
            conv1d,
            dt_bias: Param::new(Array::ones_f32(&[num_heads])),
            A_log: Param::new(Array::from_slice(&a, &[num_heads])),
            D: Param::new(Array::ones_f32(&[num_heads])),
            norm,
            out_proj,
            num_heads,
            head_dim: config.mamba_head_dim(),
            n_groups: config.mamba_n_groups,
            state_size: config.mamba_d_state,
            intermediate,
            conv_dim,
            conv_kernel: config.mamba_d_conv,
            chunk_size: config.mamba_chunk_size.max(1),
            norm_eps: config.rms_norm_eps,
        })
    }

    /// Mix `x` `[B, L, hidden]`, advancing `cache` when one is given.
    ///
    /// With no cache the sequence starts from zero conv and SSM state. With
    /// one, the conv reads the entry's last `kernel - 1` inputs and the scan
    /// starts from its SSM state; both are written back. A single token
    /// against existing state takes the recurrent step, anything else the
    /// chunked scan, as the reference does.
    pub fn forward(
        &mut self,
        x: &Array,
        mut cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let seq_len = x.dim(1);
        let input_dtype = x.dtype();

        let projected = Module::forward(&mut self.in_proj, x)?;
        let parts = ops::split_sections(
            &projected,
            &[self.intermediate, self.intermediate + self.conv_dim],
            -1,
        );
        let (gate, xbc, dt) = (&parts[0], &parts[1], &parts[2]);

        // Causal depthwise conv: zero history for a fresh sequence, the
        // cached history otherwise.
        let padded = match cache.as_deref_mut() {
            Some(entry) => entry.update_conv_state(xbc, self.conv_kernel)?,
            None => ops::pad(
                xbc,
                &[(0, 0), (self.conv_kernel - 1, 0), (0, 0)],
                None,
                Some(0.0),
            ),
        };
        let conv = Module::forward(&mut self.conv1d, &padded)?;
        let conv = ops::slice_axis(&conv, 1, conv.dim(1) - seq_len, conv.dim(1));
        let conv = nn::silu(&conv);

        let bc = self.n_groups * self.state_size;
        let conv_parts =
            ops::split_sections(&conv, &[self.intermediate, self.intermediate + bc], -1);

        // The reference's chunked scan runs in float32 whatever the model
        // dtype; so does this one, on both paths.
        let f32 = |a: &Array| a.as_type::<f32>();
        let xs = f32(&conv_parts[0]).reshape(&[batch, seq_len, self.num_heads, self.head_dim]);
        let b = f32(&conv_parts[1]).reshape(&[batch, seq_len, self.n_groups, self.state_size]);
        let c = f32(&conv_parts[2]).reshape(&[batch, seq_len, self.n_groups, self.state_size]);
        let dt = f32(dt);
        let a_log = f32(self.A_log.as_ref());
        let d = f32(self.D.as_ref());
        let dt_bias = f32(self.dt_bias.as_ref());

        let previous = cache
            .as_ref()
            .and_then(|entry| entry.get_ssm_state().cloned());
        // `time_step_limit` is (0, inf) on every config this accepts (see
        // `GraniteConfig::validate`), so the floor is 0 and softplus never
        // reaches it.
        let (y, state) = match (seq_len, previous) {
            (1, Some(state)) => {
                ssm_update_single(&xs, &a_log, &b, &c, &d, &dt, &dt_bias, &state, 0.0)?
            }
            (_, state) => self.chunked_scan(&xs, &a_log, &b, &c, &d, &dt, &dt_bias, state)?,
        };
        if let Some(entry) = cache {
            entry.set_ssm_state(state);
        }

        let y = y.reshape(&[batch, seq_len, self.intermediate]);
        // Gated RMSNorm with the gate applied first (`norm_before_gate=False`)
        // and the statistics taken over the whole width, not per group.
        let gated = y.multiply(&nn::silu(&f32(gate)));
        let normed = gated.rms_norm(Some(&f32(self.norm.weight.as_ref())), self.norm_eps);
        Module::forward(&mut self.out_proj, &normed.as_dtype(input_dtype.as_i32()))
    }

    /// The SSD scan in `chunk_size` pieces, carrying the state across.
    ///
    /// [`ssm_attention`] materialises an `[L, L]` decay matrix per head, so a
    /// long prompt in one piece is quadratic in memory. Chunking is exactly the
    /// reference's own decomposition (`mamba_chunk_size`): each piece starts
    /// from the state the previous one ended in.
    #[allow(clippy::too_many_arguments)]
    fn chunked_scan(
        &self,
        xs: &Array,
        a_log: &Array,
        b: &Array,
        c: &Array,
        d: &Array,
        dt: &Array,
        dt_bias: &Array,
        mut state: Option<Array>,
    ) -> Result<(Array, Array), Exception> {
        let seq_len = xs.dim(1);
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < seq_len {
            let end = (start + self.chunk_size).min(seq_len);
            let piece = |a: &Array| ops::slice_axis(a, 1, start, end);
            let (y, next) = ssm_attention(
                &piece(xs),
                a_log,
                &piece(b),
                &piece(c),
                d,
                &piece(dt),
                dt_bias,
                state.as_ref(),
                0.0,
            )?;
            outputs.push(y);
            state = Some(next);
            start = end;
        }
        let state = state.ok_or_else(|| Exception::custom("Granite Mamba: empty sequence"))?;
        let y = if outputs.len() == 1 {
            outputs.pop().expect("one chunk")
        } else {
            let refs: Vec<&Array> = outputs.iter().collect();
            ops::concatenate_axis(&refs, 1)
        };
        Ok((y, state))
    }
}

// =============================================================================
// Shared MLP
// =============================================================================

/// The dense SwiGLU every `granitemoeshared` / `granitemoehybrid` layer runs:
/// one fused `input_linear` whose output halves are `[gate | up]`.
#[derive(Debug)]
pub struct GraniteSharedMlp {
    pub input_linear: nn::Linear,
    pub output_linear: nn::Linear,
    pub size: i32,
}
impl_module_params!(GraniteSharedMlp; input_linear, output_linear);

impl GraniteSharedMlp {
    pub fn new(hidden: i32, size: i32) -> Result<Self, Exception> {
        Ok(Self {
            input_linear: nn::LinearBuilder::new(hidden, 2 * size)
                .bias(false)
                .build()?,
            output_linear: nn::LinearBuilder::new(size, hidden).bias(false).build()?,
            size,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let fused = Module::forward(&mut self.input_linear, x)?;
        let gate = ops::slice_axis(&fused, -1, 0, self.size);
        let up = ops::slice_axis(&fused, -1, self.size, 2 * self.size);
        Module::forward(&mut self.output_linear, &nn::silu(&gate).multiply(&up))
    }
}

// =============================================================================
// Routed experts
// =============================================================================

/// A stacked expert weight, `[experts, out, in]`, under a `weight` key so it
/// loads from `block_sparse_moe.input_linear.weight` as released.
#[derive(Debug)]
pub struct GraniteExpertBank {
    pub weight: Param<Array>,
}
impl_module_params!(GraniteExpertBank; weight);

/// The router: one bias-free linear, `router.layer.weight` `[experts, hidden]`.
#[derive(Debug)]
pub struct GraniteRouter {
    pub layer: nn::Linear,
}
impl_module_params!(GraniteRouter; layer);

/// `block_sparse_moe`: top-k routing over SwiGLU experts.
///
/// The router takes the top `k` logits and softmaxes over just those
/// (`GraniteMoeHybridTopKRouter`), so the weights always sum to one. The
/// experts are `input_linear` `[E, 2I, H]` (`[gate | up]` on the output axis)
/// and `output_linear` `[E, H, I]`, gathered per token by `gather_mm`.
#[derive(Debug)]
pub struct GraniteMoe {
    pub router: GraniteRouter,
    pub input_linear: GraniteExpertBank,
    pub output_linear: GraniteExpertBank,
    pub num_experts: i32,
    pub top_k: i32,
    pub intermediate: i32,
}
impl_module_params!(GraniteMoe; router, input_linear, output_linear);

impl GraniteMoe {
    pub fn new(
        hidden: i32,
        intermediate: i32,
        num_experts: i32,
        top_k: i32,
    ) -> Result<Self, Exception> {
        let bank = |shape: &[i32]| GraniteExpertBank {
            weight: Param::new(
                pmetal_bridge::compat::random::normal(shape, pmetal_bridge::compat::Dtype::Float32)
                    .multiply(&Array::from_f32(0.02)),
            ),
        };
        Ok(Self {
            router: GraniteRouter {
                layer: nn::LinearBuilder::new(hidden, num_experts)
                    .bias(false)
                    .build()?,
            },
            input_linear: bank(&[num_experts, 2 * intermediate, hidden]),
            output_linear: bank(&[num_experts, hidden, intermediate]),
            num_experts,
            top_k,
            intermediate,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape().to_vec();
        let hidden = *shape.last().expect("rank >= 1");
        let tokens: i32 = shape[..shape.len() - 1].iter().product();
        let flat = x.reshape(&[tokens, hidden]);

        // `F.linear(...).float()`, `topk`, then softmax over the k winners.
        let logits = Module::forward(&mut self.router.layer, &flat)?.as_type::<f32>();
        let (indices, top_logits) = crate::moe_routing::topk_normalize(&logits, self.top_k, false)?;
        let weights = ops::softmax_axis(&top_logits, -1).as_dtype(x.dtype().as_i32());

        // [N, 1, 1, H] against [E, H, 2I] gathered per row -> [N, k, 1, 2I].
        let rows = flat.expand_dims(1).expand_dims(2);
        let gate_up = rows.gather_mm(
            &self.input_linear.weight.as_ref().transpose_axes(&[0, 2, 1]),
            None,
            Some(&indices),
            false,
        );
        let i = self.intermediate;
        let gate = ops::slice_axis(&gate_up, -1, 0, i);
        let up = ops::slice_axis(&gate_up, -1, i, 2 * i);
        let activated = nn::silu(&gate).multiply(&up);
        // [N, k, 1, I] against [E, I, H] -> [N, k, H].
        let down = activated
            .gather_mm(
                &self
                    .output_linear
                    .weight
                    .as_ref()
                    .transpose_axes(&[0, 2, 1]),
                None,
                Some(&indices),
                false,
            )
            .squeeze_axes(&[2]);
        let mixed = down
            .multiply(&weights.reshape(&[tokens, self.top_k, 1]))
            .sum_axis(1, false);
        Ok(mixed.reshape(&shape))
    }
}

// =============================================================================
// Weight loading
// =============================================================================

/// What [`load_granite_weights`] did with a checkpoint.
#[derive(Debug, Default)]
pub struct GraniteLoadReport {
    /// Parameters assigned from the checkpoint.
    pub loaded: usize,
    /// Checkpoint keys deliberately not used (a tied checkpoint's spare
    /// `lm_head.weight`, rotary buffers).
    pub ignored: Vec<String>,
}

/// Load a released Granite checkpoint into `model`, all of it or not at all.
///
/// Every checkpoint tensor must land on a parameter of the same shape, and
/// every parameter must be supplied. The generic loader drops what it cannot
/// place, which for this family is the difference between a model and a
/// random one: a `granitemoe` checkpoint used to load into a dense layer with
/// every expert silently discarded. A layout this does not know is an error
/// naming the keys, not a model that parses and computes noise.
///
/// Two layouts are accepted: the released one (`input_linear`, `A_log`,
/// `conv1d.weight` as `[C, 1, K]`), and the MLX conversion of it, which splits
/// the fused expert projection into `switch_mlp.{gate,up,down}_proj`, renames
/// a dense model's `shared_mlp` to `mlp.{gate,up,down}_proj`, and stores the
/// conv as `[C, K, 1]`. MLX-quantized checkpoints are dequantized by the shared
/// reader before any of this sees them.
pub fn load_granite_weights(
    model: &mut GraniteForCausalLM,
    model_dir: &Path,
) -> Result<GraniteLoadReport, Exception> {
    let loaded = crate::loader::load_weights(model_dir)
        .map_err(|e| Exception::custom(format!("Granite: reading weights: {e:?}")))?;
    assign_granite_weights(model, loaded)
}

/// [`load_granite_weights`] on tensors already in memory.
pub fn assign_granite_weights(
    model: &mut GraniteForCausalLM,
    loaded: HashMap<String, Array>,
) -> Result<GraniteLoadReport, Exception> {
    let tied = model.lm_head.is_none();
    let dense_family = model.config.family()? == super::granite::GraniteFamily::Dense;
    let mut report = GraniteLoadReport::default();
    let weights = normalize_checkpoint_layout(loaded, dense_family, tied, &mut report)?;

    let mut params = model.flatten_params_mut();
    let mut unmatched = Vec::new();
    let mut assigned = HashSet::new();
    for (key, value) in weights {
        let Some(param) = params.get_mut(key.as_str()) else {
            unmatched.push(key);
            continue;
        };
        if param.shape() != value.shape() {
            return Err(Exception::custom(format!(
                "Granite: checkpoint tensor {key} is {:?}, the model expects {:?}",
                value.shape(),
                param.shape()
            )));
        }
        **param = value;
        assigned.insert(key);
        report.loaded += 1;
    }
    let mut missing: Vec<String> = params
        .keys()
        .filter(|k| !assigned.contains(k.as_str()))
        .cloned()
        .collect();

    if !unmatched.is_empty() || !missing.is_empty() {
        unmatched.sort();
        missing.sort();
        let sample = |v: &[String]| v.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
        return Err(Exception::custom(format!(
            "Granite: checkpoint does not match the model the config describes \
             ({} unmatched checkpoint tensors: [{}]; {} parameters not in the checkpoint: [{}]). \
             Refusing to run with weights missing.",
            unmatched.len(),
            sample(&unmatched),
            missing.len(),
            sample(&missing),
        )));
    }
    Ok(report)
}

/// Rewrite an MLX-converted Granite checkpoint into the released layout, and
/// put the conv weights in MLX's `[C, K, 1]`.
fn normalize_checkpoint_layout(
    mut weights: HashMap<String, Array>,
    dense_family: bool,
    tied: bool,
    report: &mut GraniteLoadReport,
) -> Result<HashMap<String, Array>, Exception> {
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in &keys {
        if key.ends_with("rotary_emb.inv_freq") || (tied && key == "lm_head.weight") {
            weights.remove(key);
            report.ignored.push(key.clone());
        }
    }

    // Fused expert projection, split by the MLX conversion.
    let fuse = |weights: &mut HashMap<String, Array>,
                gate_key: &str,
                up_key: &str,
                fused_key: String,
                axis: i32|
     -> Result<(), Exception> {
        let gate = weights.remove(gate_key);
        let up = weights.remove(up_key);
        match (gate, up) {
            (Some(gate), Some(up)) => {
                weights.insert(fused_key, ops::concatenate_axis(&[&gate, &up], axis));
                Ok(())
            }
            (None, None) => Ok(()),
            _ => Err(Exception::custom(format!(
                "Granite: checkpoint has one of {gate_key} / {up_key} without the other"
            ))),
        }
    };
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in &keys {
        if let Some(prefix) = key.strip_suffix(".block_sparse_moe.switch_mlp.gate_proj.weight") {
            fuse(
                &mut weights,
                key,
                &format!("{prefix}.block_sparse_moe.switch_mlp.up_proj.weight"),
                format!("{prefix}.block_sparse_moe.input_linear.weight"),
                1,
            )?;
        } else if let Some(prefix) =
            key.strip_suffix(".block_sparse_moe.switch_mlp.down_proj.weight")
        {
            let value = weights.remove(key).expect("listed key");
            weights.insert(
                format!("{prefix}.block_sparse_moe.output_linear.weight"),
                value,
            );
        } else if !dense_family {
            // Plain Granite *is* `mlp.{gate,up,down}_proj`; only the MoE
            // family's dense layers were renamed from `shared_mlp`.
            if let Some(prefix) = key.strip_suffix(".mlp.gate_proj.weight") {
                fuse(
                    &mut weights,
                    key,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    format!("{prefix}.shared_mlp.input_linear.weight"),
                    0,
                )?;
            } else if let Some(prefix) = key.strip_suffix(".mlp.down_proj.weight") {
                let value = weights.remove(key).expect("listed key");
                weights.insert(format!("{prefix}.shared_mlp.output_linear.weight"), value);
            }
        }
    }

    for (key, value) in weights.iter_mut() {
        if key.ends_with(".conv1d.weight") && value.ndim() == 3 && value.dim(1) == 1 {
            // [C, 1, K] -> [C, K, 1]: the middle axis is a singleton, so this
            // is a reshape, not a transpose.
            let (channels, kernel) = (value.dim(0), value.dim(2));
            *value = value.reshape(&[channels, kernel, 1]);
        }
    }
    Ok(weights)
}
