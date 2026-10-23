//! Safetensors loading + shape sanitization + quantized expert stacking.

use crate::{RuntimeError, RuntimeResult};
use std::collections::HashMap;

use crate::{InlineArray, QuantParams, QuantizedMode};

use super::weights::{EmbeddingWeight, LayerWeight, LayerWeights, NativeWeights, copy_fresh_arr};
use super::{Qwen3Config, validate_quantization_runtime_support};

// ============================================================================
// Weight loading
// ============================================================================

// ============================================================================
// Diagnostics helpers
// ============================================================================

/// Returns `true` if any projection weight in the loaded layers is `Quantized`.
/// Used to confirm at load time that quantized models were loaded correctly.
fn weights_are_quantized(layers: &[LayerWeights]) -> bool {
    for lw in layers {
        if matches!(lw.mlp_gate_w.as_ref(), Some(LayerWeight::Quantized { .. })) {
            return true;
        }
        if matches!(lw.attn_q_w.as_ref(), Some(LayerWeight::Quantized { .. })) {
            return true;
        }
        if matches!(lw.gdn_qkv_w.as_ref(), Some(LayerWeight::Quantized { .. })) {
            return true;
        }
    }
    false
}

/// Quantization for the module at `base_key` (the tensor key without
/// `.weight`), bits and group size both.
///
/// Checked in order: pmetal's legacy `per_tensor_overrides` (bits only,
/// keyed `{module}.weight`; what `mlx_quant` wrote before #30, and nothing
/// else writes it), then MLX's own per-module override or file default, then
/// `default`. The legacy map goes first because MLX's parser always answers,
/// with the file default when a module isn't named.
fn quant_params_for(config: &Qwen3Config, base_key: &str, default: QuantParams) -> QuantParams {
    let legacy = config.quantization().and_then(|qc| {
        qc.per_tensor_overrides
            .get(&format!("{base_key}.weight"))
            .copied()
    });
    if let Some(bits) = legacy {
        return QuantParams { bits, ..default };
    }
    config
        .mlx_quantization
        .as_ref()
        .and_then(|q| q.params_for(base_key))
        .unwrap_or(default)
}

use super::family::{self, canonical_checkpoint_key as canonical_key};

const MXFP8_GROUP_SIZE: i32 = 32;
const MXFP8_BITS: i32 = 8;

// Load model weights from a directory containing safetensors shards.
//
// Supports all three Qwen variants:
//   - Qwen3 dense (model_type = "qwen3"): standard attention, full RoPE.
//   - Qwen3.5 dense (model_type = "qwen3_5" / "qwen3_5_text", num_experts = 0).
//   - Qwen3.5 MoE (same type but num_experts > 0): routes through SwitchGLU +
//     shared expert. Per-expert weights are stacked into [E, in, out] tensors at
//     load time (matches Python's sanitize() stacking).
//
// Applies the same sanitization as the pmetal-models loader:
// - VLM prefix stripping (model.language_model. -> model.)
// - A_log -> a_log rename
// - mtp.* key drop
// - conv1d weight transpose (when shape is [out, k, in] not [out, k, 1])
// ============================================================================
// Hadamard preconditioning — absorb random rotation into Q/K/V/O weights
// ============================================================================

/// - norm `(1+w)` offset when the model has `mtp.*` keys or unsanitized conv shapes
/// - Q/K scale synthesis for GDN (not stored in safetensors)
/// - MoE expert weight stacking into `[E, in, out]` for `gather_mm`
/// - `copy_fresh` on all weights (add zero + eval + detach) for fresh Metal buffers
pub fn load_model(
    model_dir: &std::path::Path,
    config: &Qwen3Config,
) -> RuntimeResult<NativeWeights> {
    // ── Step 1+2: Shard discovery and bulk-load ─────────────────────────────
    let shard_paths = crate::native_loader::discover_safetensors_shards(model_dir)?;
    let mut raw = crate::native_loader::load_shards_into_map(&shard_paths, model_dir)?;

    // How to read a module nothing overrides: MLX's block (which names the
    // mode, so mxfp4 and nvfp4 checkpoints are read as such), else the plain
    // bits and group size of a pmetal-written one, else MLX's affine defaults.
    // Only consulted for a module that stores `scales`.
    let default_params = config
        .mlx_quantization
        .as_ref()
        .and_then(|q| q.default_params())
        .or_else(|| {
            config.quantization().map(|qc| QuantParams {
                group_size: qc.group_size,
                bits: qc.bits,
                mode: QuantizedMode::Affine,
            })
        })
        .unwrap_or_else(|| QuantParams::defaults_for(QuantizedMode::Affine));
    validate_quantization_runtime_support(default_params.bits)?;

    // ── Step 3: Sanitization ────────────────────────────────────────────────

    // Detect whether norm shift is needed before any renaming.
    let has_mtp = raw.keys().any(|k| k.contains("mtp."));
    let has_unsanitized_conv = raw
        .iter()
        .any(|(k, v)| k.contains("conv1d.weight") && v.ndim() == 3 && v.dim(2) != 1);
    let should_shift_norms = has_mtp || has_unsanitized_conv;

    // 3a. Key renaming: strip VLM prefix; A_log → a_log.
    // Qwen3.5-27B uses "language_model.model." prefix (no leading "model.").
    // Older checkpoints use "model.language_model." — strip both to "model.".
    // Qwen3 uses plain "model.layers.N..." with no prefix.
    let original_keys: Vec<String> = raw.keys().cloned().collect();
    for old_key in original_keys {
        let new_key = canonical_key(&old_key);
        if new_key != old_key {
            if let Some(v) = raw.remove(&old_key) {
                raw.insert(new_key, v);
            }
        }
    }

    // 3b. Drop what the text model never reads: the bundled MTP predictor
    // (`--mtp` loads it separately) and the vision tower.
    raw.retain(|k, _| !k.contains("mtp.") && !family::is_unused_by_text_model(k));

    // 3c. Drop lm_head.weight when embeddings are tied.
    if config.tie_word_embeddings {
        raw.remove("lm_head.weight");
    }

    // Norm suffixes that receive the (1+w) shift.
    // Excludes `.linear_attn.norm.weight` (the GDN sub-norm, which is NOT shifted).
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];

    // 3d. Conv1d transpose + norm shift + f32→model_dtype casts.
    let detected_model_dtype =
        crate::native_weight::detect_model_dtype(|k| raw.get(k).map(|w| w.dtype_raw()));

    // 3d. FP8 normalization.
    //
    // HF Qwen FP8 checkpoints store projection weights as E4M3 with sibling
    // `.weight_scale_inv` tensors. Repack them into MLX's mxfp8 quantized
    // matmul format so decode keeps the bandwidth advantage instead of
    // expanding the checkpoint to bf16 runtime weights.
    normalize_fp8_weight_scale_inv_sidecars(&mut raw, detected_model_dtype)?;

    // 3d. NVIDIA ModelOpt normalization (nvfp4 and per-tensor fp8).
    normalize_modelopt_sidecars(&mut raw, detected_model_dtype)?;

    let one = InlineArray::scalar_with_dtype(1.0, detected_model_dtype);

    // GDN-specific f32 weights that must be cast to model dtype to prevent f32
    // dtype propagation through the residual stream.
    let f32_gdn_suffixes = [
        "linear_attn.a_log",
        "linear_attn.norm.weight",
        "linear_attn.q_norm.weight",
        "linear_attn.k_norm.weight",
    ];

    let all_keys: Vec<String> = raw.keys().cloned().collect();
    for k in &all_keys {
        if k.contains("conv1d.weight") {
            if let Some(v) = raw.get(k) {
                if v.ndim() == 3 && v.dim(2) != 1 {
                    let transposed = v.transpose_axes(&[0, 2, 1]);
                    raw.insert(k.clone(), transposed);
                }
            }
        }
        if f32_gdn_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            if let Some(v) = raw.get(k) {
                if v.dtype_raw() != detected_model_dtype {
                    let cast = v.as_dtype(detected_model_dtype);
                    raw.insert(k.clone(), cast);
                }
            }
        }
        if should_shift_norms && norm_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            if let Some(v) = raw.get(k) {
                if v.ndim() == 1 {
                    let shifted = v.add(&one);
                    raw.insert(k.clone(), shifted);
                }
            }
        }
    }

    // 3e. MoE experts: Hugging Face's fused layout.
    //
    // Experts reach the model in one of three layouts, and all of them end up
    // as stacked `[E, out, in]` projections (dense, or packed with their
    // scales) that `gather_mm` / `gather_qmm` run:
    //
    //   * `{prefix}.experts.gate_up_proj` `[E, 2I, in]` and
    //     `{prefix}.experts.down_proj` `[E, out, I]`: transformers 5's fused
    //     layout, split here into the next layout's keys;
    //   * `{prefix}.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`:
    //     the layout MLX's converter writes, read as it stands;
    //   * `{prefix}.experts.{e}.{gate,up,down}_proj.*`: one module per expert
    //     (transformers 4, NVIDIA ModelOpt), each loaded like any projection
    //     and stacked when the layer is built.
    if config.is_moe() {
        for li in 0..config.num_hidden_layers as usize {
            if config.is_dense_mlp_layer(li) {
                continue;
            }
            let prefix = format!("model.layers.{li}.mlp");
            let packed_gate_up_key = format!("{prefix}.experts.gate_up_proj");
            let packed_down_key = format!("{prefix}.experts.down_proj");
            if raw.contains_key(&packed_gate_up_key) && raw.contains_key(&packed_down_key) {
                let gate_up = raw.remove(&packed_gate_up_key).ok_or_else(|| {
                    RuntimeError::Weights(format!(
                        "MoE: missing packed expert tensor {packed_gate_up_key}"
                    ))
                })?;
                let down = raw.remove(&packed_down_key).ok_or_else(|| {
                    RuntimeError::Weights(format!(
                        "MoE: missing packed expert tensor {packed_down_key}"
                    ))
                })?;
                let hidden = config.moe_intermediate_size;
                if hidden <= 0 {
                    return Err(RuntimeError::Weights(format!(
                        "MoE: invalid moe_intermediate_size={} for packed expert weights at layer {li}",
                        hidden
                    )));
                }

                // gate_up: [E, 2I, in] splits into gate and up, [E, I, in]
                // each, gate first.
                let gate = gate_up.index((.., 0..hidden, ..));
                let up = gate_up.index((.., hidden..(hidden * 2), ..));
                raw.insert(format!("{prefix}.switch_mlp.gate_proj.weight"), gate);
                raw.insert(format!("{prefix}.switch_mlp.up_proj.weight"), up);
                raw.insert(format!("{prefix}.switch_mlp.down_proj.weight"), down);
            }
        }
    }

    // ── Step 4: Build per-layer weight structs ──────────────────────────────
    //
    // Every tensor read is recorded, so that after the build any checkpoint
    // tensor nobody read is an error rather than a silently ignored weight.
    let consumed = std::cell::RefCell::new(std::collections::HashSet::<String>::new());
    let take = |key: &str| -> Option<InlineArray> {
        let value = raw.get(key).cloned();
        if value.is_some() {
            consumed.borrow_mut().insert(key.to_owned());
        }
        value
    };
    let get = |key: &str| -> RuntimeResult<InlineArray> {
        take(key).ok_or_else(|| {
            let parts: Vec<&str> = key.rsplitn(2, '.').collect();
            let suffix = parts[0];
            let close: Vec<&String> = raw.keys().filter(|k| k.ends_with(suffix)).take(5).collect();
            RuntimeError::Weights(format!(
                "missing weight key: {key} (close matches: {close:?})"
            ))
        })
    };

    // A module's quantization, as MLX decides it: the module is packed
    // exactly when `{base_key}.scales` is in the weights. `scales` and
    // `biases` are the tensors' own sibling keys (affine has both, the float
    // modes `scales` alone), and how to read them (bits, group size, mode)
    // comes from the checkpoint's `quantization` block, a module's own entry
    // first. `None` when the module is dense.
    let packed_params = |base_key: &str| -> RuntimeResult<Option<QuantParams>> {
        if !raw.contains_key(&format!("{base_key}.scales")) {
            return Ok(None);
        }
        let has_biases = raw.contains_key(&format!("{base_key}.biases"));
        let params =
            quant_params_for(config, base_key, default_params).for_tensors(base_key, has_biases)?;
        validate_quantization_runtime_support(params.bits)?;
        Ok(Some(params))
    };

    // Load a projection weight as `LayerWeight`.
    //
    // Packed (see `packed_params`) it is `LayerWeight::Quantized`, with no
    // transpose: the kernel takes the checkpoint's `[out, in_packed]` and
    // `transpose = true`. Otherwise it is `LayerWeight::Dense`, pre-transposed
    // to `[in, out]`. The staged `{base_key}.mxfp8_scales` and `.nvfp4_scales`
    // of the FP8 and ModelOpt normalizations above are packed too, with a
    // per-tensor scale where ModelOpt gave one.
    //
    // `base_key` is the full key WITHOUT the trailing `.weight` suffix.
    let get_layer_weight = |base_key: &str| -> RuntimeResult<LayerWeight> {
        let w_key = format!("{base_key}.weight");
        let s_key = format!("{base_key}.scales");
        let b_key = format!("{base_key}.biases");
        let mxfp8_s_key = format!("{base_key}.mxfp8_scales");
        let nvfp4_s_key = format!("{base_key}.nvfp4_scales");
        let tensor_scale = || take(&format!("{base_key}.tensor_scale"));

        if raw.contains_key(&w_key)
            && let Some(scales) = take(&mxfp8_s_key)
        {
            return Ok(LayerWeight::Quantized {
                tensor_scale: tensor_scale(),
                weight: get(&w_key)?,
                scales,
                biases: None,
                params: QuantParams {
                    group_size: MXFP8_GROUP_SIZE,
                    bits: MXFP8_BITS,
                    mode: QuantizedMode::Mxfp8,
                },
            });
        }
        if raw.contains_key(&w_key)
            && let Some(scales) = take(&nvfp4_s_key)
        {
            return Ok(LayerWeight::Quantized {
                tensor_scale: tensor_scale(),
                weight: get(&w_key)?,
                scales,
                biases: None,
                params: QuantParams::defaults_for(QuantizedMode::Nvfp4),
            });
        }

        match packed_params(base_key)? {
            Some(params) => Ok(LayerWeight::Quantized {
                tensor_scale: None,
                weight: get(&w_key)?,
                scales: get(&s_key)?,
                biases: take(&b_key),
                params,
            }),
            None => {
                // Dense path: load `.weight` and pre-transpose for matmul.
                let w = get(&w_key)?;
                Ok(LayerWeight::Dense(w.t()))
            }
        }
    };

    // A stack of experts at `{base_key}`, `[E, out, in]` as MLX's converter saves it
    // and as the fused layout was split into above: packed as it stands,
    // dense pre-transposed to `[E, in, out]` for `gather_mm`.
    let stacked_projection = |base_key: &str| -> RuntimeResult<LayerWeight> {
        let w_key = format!("{base_key}.weight");
        match packed_params(base_key)? {
            Some(params) => Ok(LayerWeight::Quantized {
                tensor_scale: None,
                weight: get(&w_key)?,
                scales: get(&format!("{base_key}.scales"))?,
                biases: take(&format!("{base_key}.biases")),
                params,
            }),
            None => Ok(LayerWeight::Dense(get(&w_key)?.transpose_axes(&[0, 2, 1]))),
        }
    };

    let embed_w = {
        let base = "model.embed_tokens";
        let params = packed_params(base)?
            .unwrap_or_else(|| QuantParams::defaults_for(QuantizedMode::Affine));
        EmbeddingWeight::new(
            get(&format!("{base}.weight"))?,
            take(&format!("{base}.scales")),
            take(&format!("{base}.biases")),
            params,
        )
    };
    let final_norm_w = get("model.norm.weight")?;
    let final_norm_eps = config.rms_norm_eps;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        Some(get_layer_weight("lm_head")?)
    };

    // Deliberately not `embed_w.dtype_raw()`: see `detect_model_dtype`. On a
    // quantized checkpoint that tensor is packed `uint32`.
    let model_dtype =
        crate::native_weight::detect_model_dtype(|k| raw.get(k).map(|w| w.dtype_raw()));

    // GDN dimensions (identical across all GDN layers; only meaningful for Qwen3.5).
    let nv = config.gdn_nv();
    let nk = config.gdn_nk();
    let dk = config.gdn_dk();
    let dv = config.gdn_dv();
    let ck = config.linear_conv_kernel_dim;
    let kd = nk * dk; // total key dim
    let cd = kd * 2 + nv * dv; // conv projection dim

    // Attention dimensions.
    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.get_num_kv_heads();
    let head_dim = config.get_head_dim();
    let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let rope_dims = config.rope_dims();
    let rope_base = config.rope_theta as f32;
    let rope_scale = 1.0_f32;
    let scaled_rope = config.scaled_rope();
    let attn_gated = !config.has_plain_attention();
    let gdn_gate = if config.has_plain_attention() {
        family::GdnGateActivation::Silu
    } else {
        config.gdn_gate()?
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);

    for li in 0..config.num_hidden_layers as usize {
        let p = format!("model.layers.{li}");
        let is_linear = config.is_linear_layer(li);
        let is_moe_layer = !config.is_dense_mlp_layer(li);

        let input_ln_w = get(&format!("{p}.input_layernorm.weight"))?;
        let post_ln_w = get(&format!("{p}.post_attention_layernorm.weight"))?;

        // MLP weights — dense or MoE.
        let (
            mlp_gate_w,
            mlp_up_w,
            mlp_down_w,
            moe_router_w,
            moe_gate_w,
            moe_up_w,
            moe_down_w,
            shared_gate_w,
            shared_up_w,
            shared_down_w,
            shared_expert_gate_w,
        ) = if is_moe_layer {
            // MoE layer: router + stacked experts + shared expert.
            // The router, `[num_experts, hidden]` in the checkpoint, is a
            // projection like the rest, and MLX's converter quantizes it (at 8 bits).
            let router = get_layer_weight(&format!("{p}.mlp.gate"))?;

            // Stacked as the checkpoint ships them (`switch_mlp`, or the fused
            // layout split above), else one module per expert, stacked here.
            let stacked = |proj: &str| -> RuntimeResult<LayerWeight> {
                let base = format!("{p}.mlp.switch_mlp.{proj}");
                if raw.contains_key(&format!("{base}.weight")) {
                    return stacked_projection(&base);
                }
                let experts = (0..config.num_experts)
                    .map(|e| get_layer_weight(&format!("{p}.mlp.experts.{e}.{proj}")))
                    .collect::<Result<Vec<_>, _>>()?;
                LayerWeight::stack(experts)
                    .map_err(|e| RuntimeError::Weights(format!("{p}.mlp.experts.*.{proj}: {e}")))
            };
            let moe_gate = stacked("gate_proj")?;
            let moe_up = stacked("up_proj")?;
            let moe_down = stacked("down_proj")?;

            // The shared expert (the Qwen3.5 family; Qwen3-MoE has none),
            // may be quantized. Its gate is `[1, hidden]`, quantized by
            // MLX's converter like the router.
            let shared = if config.has_shared_expert() {
                (
                    Some(get_layer_weight(&format!(
                        "{p}.mlp.shared_expert.gate_proj"
                    ))?),
                    Some(get_layer_weight(&format!("{p}.mlp.shared_expert.up_proj"))?),
                    Some(get_layer_weight(&format!(
                        "{p}.mlp.shared_expert.down_proj"
                    ))?),
                    Some(get_layer_weight(&format!("{p}.mlp.shared_expert_gate"))?),
                )
            } else {
                (None, None, None, None)
            };
            (
                None,
                None,
                None,
                Some(router),
                Some(moe_gate),
                Some(moe_up),
                Some(moe_down),
                shared.0,
                shared.1,
                shared.2,
                shared.3,
            )
        } else {
            // Dense MLP — may be quantized.
            let gate = get_layer_weight(&format!("{p}.mlp.gate_proj"))?;
            let up = get_layer_weight(&format!("{p}.mlp.up_proj"))?;
            let down = get_layer_weight(&format!("{p}.mlp.down_proj"))?;
            (
                Some(gate),
                Some(up),
                Some(down),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        };

        let mut lw = LayerWeights {
            is_linear,
            input_ln_w,
            input_ln_eps: config.rms_norm_eps,
            post_ln_w,
            post_ln_eps: config.rms_norm_eps,
            mlp_gate_w,
            mlp_up_w,
            mlp_down_w,
            moe_router_w,
            moe_gate_w,
            moe_up_w,
            moe_down_w,
            shared_gate_w,
            shared_up_w,
            shared_down_w,
            shared_expert_gate_w,
            moe_top_k: config.num_experts_per_tok,
            moe_norm_topk_prob: config.norm_topk_prob,
            is_moe_layer,
            // Attention — filled below when !is_linear
            attn_q_w: None,
            attn_k_w: None,
            attn_v_w: None,
            attn_o_w: None,
            attn_q_norm_w: None,
            attn_q_norm_eps: config.rms_norm_eps,
            attn_k_norm_w: None,
            attn_k_norm_eps: config.rms_norm_eps,
            attn_n_heads: 0,
            attn_n_kv_heads: 0,
            attn_head_dim: 0,
            attn_scale: 0.0,
            attn_rope_dims: 0,
            attn_rope_base: 0.0,
            attn_rope_scale: 0.0,
            attn_scaled_rope: None,
            attn_gated,
            // GDN — filled below when is_linear
            gdn_qkv_w: None,
            gdn_z_w: None,
            gdn_b_w: None,
            gdn_a_w: None,
            gdn_conv_w: None,
            gdn_q_nw: None,
            gdn_k_nw: None,
            gdn_a_log: None,
            gdn_dt_bias: None,
            gdn_norm_w: None,
            gdn_norm_eps: config.rms_norm_eps,
            gdn_out_w: None,
            gdn_nv: 0,
            gdn_nk: 0,
            gdn_dk: 0,
            gdn_dv: 0,
            gdn_kd: 0,
            gdn_cd: 0,
            gdn_ck: 0,
            gdn_gate,
        };

        if is_linear {
            let la = format!("{p}.linear_attn");
            // in_proj and out_proj are large matrices — can be quantized.
            lw.gdn_qkv_w = Some(get_layer_weight(&format!("{la}.in_proj_qkv"))?);
            lw.gdn_z_w = Some(get_layer_weight(&format!("{la}.in_proj_z"))?);
            lw.gdn_b_w = Some(get_layer_weight(&format!("{la}.in_proj_b"))?);
            lw.gdn_a_w = Some(get_layer_weight(&format!("{la}.in_proj_a"))?);
            // conv1d weight is a small 3D tensor — never quantized.
            lw.gdn_conv_w = Some(get(&format!("{la}.conv1d.weight"))?);

            // Q/K scale weights are SYNTHETIC — not present in safetensors.
            // They encode the Q/K normalization scaling factor:
            //   q_norm_weight = ones(dk) * inv_scale²
            //   k_norm_weight = ones(dk) * inv_scale
            // where inv_scale = 1/sqrt(dk).
            // IMPORTANT: the scalar MUST be cast to model_dtype before multiply.
            let inv_scale = (dk as f32).sqrt().recip();
            let q_scale_arr = {
                let a = InlineArray::ones(&[dk], model_dtype);
                let scale = InlineArray::scalar_with_dtype(inv_scale * inv_scale, model_dtype);
                a.multiply(&scale)
            };
            let k_scale_arr = {
                let a = InlineArray::ones(&[dk], model_dtype);
                let scale = InlineArray::scalar_with_dtype(inv_scale, model_dtype);
                a.multiply(&scale)
            };
            lw.gdn_q_nw = Some(
                get(&format!("{la}.q_norm_weight"))
                    .or_else(|_| get(&format!("{la}.q_norm.weight")))
                    .unwrap_or(q_scale_arr),
            );
            lw.gdn_k_nw = Some(
                get(&format!("{la}.k_norm_weight"))
                    .or_else(|_| get(&format!("{la}.k_norm.weight")))
                    .unwrap_or(k_scale_arr),
            );
            lw.gdn_a_log = Some(get(&format!("{la}.a_log"))?);
            lw.gdn_dt_bias = Some(get(&format!("{la}.dt_bias"))?);
            lw.gdn_norm_w = Some(get(&format!("{la}.norm.weight"))?);
            // out_proj is a large matrix — can be quantized.
            lw.gdn_out_w = Some(get_layer_weight(&format!("{la}.out_proj"))?);
            lw.gdn_nv = nv;
            lw.gdn_nk = nk;
            lw.gdn_dk = dk;
            lw.gdn_dv = dv;
            lw.gdn_kd = kd;
            lw.gdn_cd = cd;
            lw.gdn_ck = ck;

            // Log GDN config once (first linear-attention layer only).
            // Uses stderr to avoid interleaving with streamed stdout output.
        } else {
            let sa = format!("{p}.self_attn");
            // Q projection width differs between Qwen3 and Qwen3.5:
            //   Qwen3:   [n_heads * head_dim, hidden]
            //   Qwen3.5: [n_heads * head_dim * 2, hidden]  (queries + gate concatenated)
            // We just load whatever is in the checkpoint; the forward pass
            // uses `attn_gated` to decide how to interpret the output.
            lw.attn_q_w = Some(get_layer_weight(&format!("{sa}.q_proj"))?);
            lw.attn_k_w = Some(get_layer_weight(&format!("{sa}.k_proj"))?);
            lw.attn_v_w = Some(get_layer_weight(&format!("{sa}.v_proj"))?);
            lw.attn_o_w = Some(get_layer_weight(&format!("{sa}.o_proj"))?);
            // Q/K norms are 1D scale vectors — never quantized.
            lw.attn_q_norm_w = Some(get(&format!("{sa}.q_norm.weight"))?);
            lw.attn_k_norm_w = Some(get(&format!("{sa}.k_norm.weight"))?);
            lw.attn_n_heads = n_heads;
            lw.attn_n_kv_heads = n_kv_heads;
            lw.attn_head_dim = head_dim;
            lw.attn_scale = attn_scale;
            lw.attn_rope_dims = rope_dims;
            lw.attn_rope_base = rope_base;
            lw.attn_rope_scale = rope_scale;
            lw.attn_scaled_rope = scaled_rope.clone();
        }

        layers.push(lw);
    }

    // A Qwen3.5-family checkpoint tensor that nothing above read is a weight
    // the model would run without. The vision tower and the MTP predictor
    // were dropped by name in step 3b; anything else left over is refused.
    if !config.has_plain_attention() {
        let consumed = consumed.borrow();
        let mut unread: Vec<&String> = raw.keys().filter(|k| !consumed.contains(*k)).collect();
        if !unread.is_empty() {
            unread.sort();
            return Err(RuntimeError::Weights(format!(
                "{} checkpoint tensor(s) were not consumed by the Qwen3.5 loader \
             (first {}): {:?}",
                unread.len(),
                unread.len().min(10),
                &unread[..unread.len().min(10)]
            )));
        }
    }

    // ── Step 5: copy_fresh — force all weights into fresh Metal buffers ─────
    //
    // For dense InlineArray weights: `w.add(zero).eval().detach()` creates a
    // new buffer with use_count=1.  Without this, weights share data with the
    // raw tensors (use_count=2), preventing optimal Metal buffer scheduling.
    //
    // For LayerWeight: `LayerWeight::copy_fresh` handles each tensor (weight,
    // scales, biases) independently using a same-dtype zero.
    //
    // Drop the raw safetensors handle map first. Large checkpoints like
    // Qwen3.5-35B-A3B otherwise keep both the raw handle set and the copied
    // weights live through the entire pass, needlessly spiking load-time peak
    // memory and causing benchmark/process kills under pressure.
    drop(raw);
    let zero = InlineArray::scalar_with_dtype(0.0, model_dtype);
    let cf_arr = |w: &InlineArray| -> InlineArray { copy_fresh_arr(w, &zero) };
    let cf_lw = |w: &LayerWeight| -> LayerWeight { w.copy_fresh(&zero) };

    let embed_w = embed_w.copy_fresh(&zero);
    let final_norm_w = cf_arr(&final_norm_w);
    let lm_head_w = lm_head_w.map(|w| cf_lw(&w));

    for lw in &mut layers {
        lw.input_ln_w = cf_arr(&lw.input_ln_w);
        lw.post_ln_w = cf_arr(&lw.post_ln_w);
        // Dense MLP (LayerWeight)
        if let Some(ref w) = lw.mlp_gate_w {
            lw.mlp_gate_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.mlp_up_w {
            lw.mlp_up_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.mlp_down_w {
            lw.mlp_down_w = Some(cf_lw(w));
        }
        // MoE — the router, the stacked experts and the shared expert's gate
        // are all LayerWeights
        if let Some(ref w) = lw.moe_router_w {
            lw.moe_router_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.moe_gate_w {
            lw.moe_gate_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.moe_up_w {
            lw.moe_up_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.moe_down_w {
            lw.moe_down_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.shared_gate_w {
            lw.shared_gate_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.shared_up_w {
            lw.shared_up_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.shared_down_w {
            lw.shared_down_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.shared_expert_gate_w {
            lw.shared_expert_gate_w = Some(cf_lw(w));
        }
        // Attention projections (LayerWeight); norms are InlineArray
        if let Some(ref w) = lw.attn_q_w {
            lw.attn_q_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.attn_k_w {
            lw.attn_k_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.attn_v_w {
            lw.attn_v_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.attn_o_w {
            lw.attn_o_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.attn_q_norm_w {
            lw.attn_q_norm_w = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.attn_k_norm_w {
            lw.attn_k_norm_w = Some(cf_arr(w));
        }
        // GDN projections (LayerWeight); small tensors are InlineArray
        if let Some(ref w) = lw.gdn_qkv_w {
            lw.gdn_qkv_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.gdn_z_w {
            lw.gdn_z_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.gdn_b_w {
            lw.gdn_b_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.gdn_a_w {
            lw.gdn_a_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.gdn_conv_w {
            lw.gdn_conv_w = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_q_nw {
            lw.gdn_q_nw = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_k_nw {
            lw.gdn_k_nw = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_a_log {
            lw.gdn_a_log = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_dt_bias {
            lw.gdn_dt_bias = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_norm_w {
            lw.gdn_norm_w = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_out_w {
            lw.gdn_out_w = Some(cf_lw(w));
        }
    }

    // Determine quantization mode for diagnostic output.
    let _quant_mode = if let Some(qc) = config.quantization() {
        // Inspect the first projection weight to confirm quantized loading succeeded.
        let confirmed = weights_are_quantized(&layers);
        format!(
            "quantized {}-bit (group_size={}) confirmed={}",
            qc.bits, qc.group_size, confirmed,
        )
    } else {
        "dense bf16".to_string()
    };

    Ok(NativeWeights {
        embed_w,
        final_norm_w,
        final_norm_eps,
        lm_head_w,
        tie_word_embeddings: config.tie_word_embeddings,
        layers,
        model_dtype,
        qjl_matrix: None, // populated by apply_qjl_matrix() when --kv-qjl is set
    })
}

#[derive(Default, Debug, Clone, Copy, Eq, PartialEq)]
struct Fp8SidecarLoadStats {
    repacked_mxfp8: usize,
    dequantized_dense: usize,
}

fn normalize_fp8_weight_scale_inv_sidecars(
    raw: &mut HashMap<String, InlineArray>,
    target_dtype: i32,
) -> RuntimeResult<Fp8SidecarLoadStats> {
    let scale_inv_keys: Vec<String> = raw
        .keys()
        .filter(|k| {
            k.ends_with(".weight_scale_inv")
                && (k.starts_with("model.layers.") || k.as_str() == "lm_head.weight_scale_inv")
        })
        .cloned()
        .collect();

    let mut stats = Fp8SidecarLoadStats::default();
    for scale_inv_key in scale_inv_keys {
        let Some(weight_key) = scale_inv_key
            .strip_suffix("_scale_inv")
            .map(ToOwned::to_owned)
        else {
            continue;
        };

        let scale_inv = raw.remove(&scale_inv_key).ok_or_else(|| {
            RuntimeError::Weights(format!(
                "FP8 sidecar disappeared during load: {scale_inv_key}"
            ))
        })?;
        let Some(weight) = raw.remove(&weight_key) else {
            raw.insert(scale_inv_key.clone(), scale_inv);
            return Err(RuntimeError::Weights(format!(
                "FP8 scale sidecar {scale_inv_key} has no matching weight {weight_key}"
            )));
        };

        let dequant = crate::native_loader::dequantize_fp8_e4m3_scaled_weight(
            &weight,
            &scale_inv,
            target_dtype,
        )
        .map_err(|e| RuntimeError::Weights(format!("{weight_key}: {e}")))?;

        if can_repack_as_mxfp8(&dequant) {
            let Some(base_key) = weight_key.strip_suffix(".weight") else {
                raw.insert(weight_key, dequant);
                stats.dequantized_dense += 1;
                continue;
            };
            let mxfp8_scale_key = format!("{base_key}.mxfp8_scales");
            let (packed, scales) = dequant
                .try_quantize_weights_mode(MXFP8_GROUP_SIZE, MXFP8_BITS, QuantizedMode::Mxfp8)
                .map_err(|e| {
                    RuntimeError::Weights(format!(
                        "{weight_key}: failed to repack FP8 as mxfp8: {e}"
                    ))
                })?;
            raw.insert(weight_key, packed);
            raw.insert(mxfp8_scale_key, scales);
            stats.repacked_mxfp8 += 1;
        } else {
            raw.insert(weight_key, dequant);
            stats.dequantized_dense += 1;
        }
    }

    Ok(stats)
}

#[derive(Default, Debug, Clone, Copy, Eq, PartialEq)]
struct ModelOptLoadStats {
    nvfp4: usize,
    fp8_packed: usize,
    fp8_dense: usize,
}

/// Put NVIDIA ModelOpt's quantized tensors on the packed kernels, keeping each
/// one's per-tensor factor as `{base}.tensor_scale`. Staged as keys rather than
/// built straight into `LayerWeight`s because MoE experts are stacked from
/// these keys afterwards. See [`crate::native_loader::ModelOptWeight::into_native`]
/// for which kernel each tensor gets.
fn normalize_modelopt_sidecars(
    raw: &mut HashMap<String, InlineArray>,
    target_dtype: i32,
) -> RuntimeResult<ModelOptLoadStats> {
    use crate::native_loader::NativeModelOpt;

    let mut stats = ModelOptLoadStats::default();
    for (base, weight) in crate::native_loader::take_modelopt_weights(raw)? {
        let native = weight
            .into_native(target_dtype)
            .map_err(|e| RuntimeError::Weights(format!("{base}.weight: {e}")))?;
        match native {
            NativeModelOpt::Packed {
                weight,
                scales,
                tensor_scale,
                params,
            } => {
                let scales_name = match params.mode {
                    QuantizedMode::Nvfp4 => {
                        stats.nvfp4 += 1;
                        "nvfp4_scales"
                    }
                    _ => {
                        stats.fp8_packed += 1;
                        "mxfp8_scales"
                    }
                };
                raw.insert(format!("{base}.weight"), weight);
                raw.insert(format!("{base}.{scales_name}"), scales);
                raw.insert(format!("{base}.tensor_scale"), tensor_scale);
            }
            NativeModelOpt::Dense(dense) => {
                raw.insert(format!("{base}.weight"), dense);
                stats.fp8_dense += 1;
            }
        }
    }
    Ok(stats)
}

fn can_repack_as_mxfp8(weight: &InlineArray) -> bool {
    weight.ndim() >= 2
        && weight
            .shape()
            .last()
            .is_some_and(|last_dim| *last_dim > 0 && *last_dim % MXFP8_GROUP_SIZE == 0)
}

#[cfg(test)]
mod quant_params_tests {
    use super::*;

    fn config_with(quantization: &str) -> Qwen3Config {
        super::super::parse_config_text(&format!(
            r#"{{"model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2,
                "num_attention_heads": 4, "quantization": {quantization}}}"#
        ))
        .expect("config parses")
    }

    const DEFAULT: QuantParams = QuantParams {
        group_size: 64,
        bits: 4,
        mode: QuantizedMode::Affine,
    };

    /// #30: MLX writes per-module overrides as sibling keys of the
    /// `quantization` object, and both bits and group size can change.
    #[test]
    fn mlx_per_module_overrides_set_bits_and_group_size() {
        let config = config_with(
            r#"{"group_size": 64, "bits": 4, "mode": "affine",
                "model.layers.0.self_attn.q_proj": {"group_size": 32, "bits": 8}}"#,
        );
        let q = quant_params_for(&config, "model.layers.0.self_attn.q_proj", DEFAULT);
        assert_eq!((q.group_size, q.bits), (32, 8));
        let other = quant_params_for(&config, "model.layers.1.self_attn.q_proj", DEFAULT);
        assert_eq!((other.group_size, other.bits), (64, 4));
    }

    /// Override paths go through the same prefix stripping as tensor keys.
    #[test]
    fn vlm_prefixed_override_paths_match_canonical_keys() {
        let config = config_with(
            r#"{"group_size": 64, "bits": 4,
                "language_model.model.layers.0.mlp.gate_proj": {"group_size": 64, "bits": 8}}"#,
        );
        assert_eq!(
            quant_params_for(&config, "model.layers.0.mlp.gate_proj", DEFAULT).bits,
            8
        );
    }

    /// Checkpoints quantized by pmetal before #30 keep loading.
    #[test]
    fn legacy_per_tensor_overrides_still_apply() {
        let config = config_with(
            r#"{"group_size": 64, "bits": 4,
                "per_tensor_overrides": {"model.layers.0.mlp.down_proj.weight": 8}}"#,
        );
        let q = quant_params_for(&config, "model.layers.0.mlp.down_proj", DEFAULT);
        assert_eq!((q.group_size, q.bits), (64, 8));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::Dtype;
    use crate::native_weight::LayerWeight as Lw;

    fn max_abs_diff(mut a: InlineArray, mut b: InlineArray, n: usize) -> f32 {
        let va = a.to_f32_vec(n).expect("read lhs");
        let vb = b.to_f32_vec(n).expect("read rhs");
        va.iter()
            .zip(vb.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    }

    fn modelopt_weight(raw: &HashMap<String, InlineArray>, base: &str, mode: QuantizedMode) -> Lw {
        let scales_key = match mode {
            QuantizedMode::Nvfp4 => "nvfp4_scales",
            _ => "mxfp8_scales",
        };
        Lw::new_scaled(
            raw[&format!("{base}.weight")].clone(),
            Some(raw[&format!("{base}.{scales_key}")].clone()),
            None,
            Some(raw[&format!("{base}.tensor_scale")].clone()),
            QuantParams::defaults_for(mode),
        )
    }

    /// NVIDIA's per-tensor fp8 bytes run on MLX's mxfp8 kernel with every
    /// block scale at 2^0. If the unit byte or the element format disagreed,
    /// every attention projection of a ModelOpt checkpoint would be off.
    #[test]
    fn modelopt_fp8_runs_exactly_on_the_mxfp8_kernel() {
        let (out, in_dim) = (4i32, 64i32);
        let values: Vec<f32> = (0..out * in_dim)
            .map(|i| ((i % 19) as f32 - 9.0) / 3.0)
            .collect();
        let fp8 = InlineArray::from_f32_slice(&values, &[out, in_dim]).to_fp8();
        let base = "model.layers.0.self_attn.q_proj";
        let mut raw = HashMap::from([
            (format!("{base}.weight"), fp8.clone()),
            (
                format!("{base}.weight_scale"),
                InlineArray::from_f32_slice(&[0.37], &[]),
            ),
            (
                format!("{base}.input_scale"),
                InlineArray::from_f32_slice(&[0.2], &[]),
            ),
        ]);

        let stats = normalize_modelopt_sidecars(&mut raw, Dtype::Bfloat16.as_i32())
            .expect("normalize ModelOpt fp8");
        assert_eq!(stats.fp8_packed, 1);
        assert!(!raw.contains_key(&format!("{base}.input_scale")));

        let x_values: Vec<f32> = (0..in_dim).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let x = InlineArray::from_f32_slice(&x_values, &[1, in_dim]);
        let got = modelopt_weight(&raw, base, QuantizedMode::Mxfp8).matmul_from(&x);
        crate::check_unobserved_error().expect("mxfp8 matmul");

        let dense = fp8
            .from_fp8(Dtype::Float32.as_i32())
            .multiply(&InlineArray::from_f32_slice(&[0.37], &[]));
        let want = x.matmul(&dense.t());
        let diff = max_abs_diff(got, want, out as usize);
        assert!(
            diff < 1e-4,
            "fp8 on the mxfp8 kernel drifted: max|diff| = {diff}"
        );
    }

    /// ModelOpt's nvfp4 bytes are MLX's layout read as `uint8`, so the loader
    /// only bitcasts them, and the per-tensor scale lands on the product.
    #[test]
    fn modelopt_nvfp4_is_a_bitcast_plus_a_tensor_scale() {
        let (out, in_dim) = (4i32, 64i32);
        let values: Vec<f32> = (0..out * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) / 6.0)
            .collect();
        let w = InlineArray::from_f32_slice(&values, &[out, in_dim]);
        let (packed, scales) = w.quantize_weights_mode(16, 4, QuantizedMode::Nvfp4);
        crate::check_unobserved_error().expect("nvfp4 quantize");

        let base = "model.layers.0.mlp.down_proj";
        let mut raw = HashMap::from([
            // What ModelOpt stores: the same bytes, typed as uint8.
            (format!("{base}.weight"), packed.view(crate::dtype::U8)),
            (format!("{base}.weight_scale"), scales.clone()),
            (
                format!("{base}.weight_scale_2"),
                InlineArray::from_f32_slice(&[0.37], &[]),
            ),
            (
                format!("{base}.input_scale"),
                InlineArray::from_f32_slice(&[0.2], &[]),
            ),
        ]);

        let stats = normalize_modelopt_sidecars(&mut raw, Dtype::Bfloat16.as_i32())
            .expect("normalize ModelOpt nvfp4");
        assert_eq!(stats.nvfp4, 1);
        assert_eq!(
            raw[&format!("{base}.weight")].dtype_raw(),
            crate::dtype::U32
        );

        let x_values: Vec<f32> = (0..in_dim).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let x = InlineArray::from_f32_slice(&x_values, &[1, in_dim]);
        let got = modelopt_weight(&raw, base, QuantizedMode::Nvfp4).matmul_from(&x);
        crate::check_unobserved_error().expect("nvfp4 matmul");

        let dense = packed
            .dequantize_mode(&scales, None, 16, 4, QuantizedMode::Nvfp4)
            .as_dtype(Dtype::Float32.as_i32())
            .multiply(&InlineArray::from_f32_slice(&[0.37], &[]));
        let want = x.matmul(&dense.t());
        let diff = max_abs_diff(got, want, out as usize);
        assert!(diff < 1e-3, "nvfp4 product misscaled: max|diff| = {diff}");
    }

    /// A width the mxfp8 kernel cannot tile still loads, as a dense weight.
    #[test]
    fn modelopt_fp8_falls_back_to_dense_off_the_block_size() {
        let (out, in_dim) = (2i32, 48i32);
        let values: Vec<f32> = (0..out * in_dim).map(|i| (i % 5) as f32 - 2.0).collect();
        let base = "model.layers.0.linear_attn.in_proj_qkv";
        let mut raw = HashMap::from([
            (
                format!("{base}.weight"),
                InlineArray::from_f32_slice(&values, &[out, in_dim]).to_fp8(),
            ),
            (
                format!("{base}.weight_scale"),
                InlineArray::from_f32_slice(&[0.5], &[]),
            ),
        ]);

        let stats = normalize_modelopt_sidecars(&mut raw, Dtype::Bfloat16.as_i32())
            .expect("normalize ModelOpt fp8");
        assert_eq!(stats.fp8_dense, 1);
        assert!(!raw.contains_key(&format!("{base}.tensor_scale")));
        assert_eq!(
            raw[&format!("{base}.weight")].dtype_raw(),
            Dtype::Bfloat16.as_i32()
        );
    }

    #[test]
    fn dequantizes_qwen_fp8_weight_scale_inv_sidecars_when_mxfp8_is_not_shape_legal() {
        let dense = InlineArray::from_f32_slice(&[1.0, -2.0, 0.5, 4.0], &[2, 2]);
        let fp8 = dense.to_fp8();
        let scale_inv =
            InlineArray::from_f32_slice(&[2.0], &[1, 1]).as_dtype(Dtype::Bfloat16.as_i32());

        let weight_key = "model.layers.0.mlp.gate_proj.weight".to_string();
        let scale_key = "model.layers.0.mlp.gate_proj.weight_scale_inv".to_string();
        let mut raw = HashMap::new();
        raw.insert(weight_key.clone(), fp8);
        raw.insert(scale_key.clone(), scale_inv);

        let stats = normalize_fp8_weight_scale_inv_sidecars(&mut raw, Dtype::Bfloat16.as_i32())
            .expect("normalize sidecars");

        assert_eq!(
            stats,
            Fp8SidecarLoadStats {
                repacked_mxfp8: 0,
                dequantized_dense: 1
            }
        );
        assert!(!raw.contains_key(&scale_key));
        assert_eq!(raw[&weight_key].dtype_raw(), Dtype::Bfloat16.as_i32());

        let mut got = raw.get(&weight_key).expect("dequantized weight").clone();
        got.eval();
        let values = got.to_f32_vec(4).expect("weight values");
        let expected = [2.0, -4.0, 1.0, 8.0];
        for (got, expected) in values.iter().zip(expected) {
            assert!(
                (got - expected).abs() <= 0.05,
                "got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn repacks_qwen_fp8_weight_scale_inv_sidecars_as_mxfp8() {
        let values: Vec<f32> = (0..64).map(|i| ((i as f32 % 17.0) - 8.0) / 4.0).collect();
        let dense = InlineArray::from_f32_slice(&values, &[2, 32]);
        let fp8 = dense.to_fp8();
        let scale_inv =
            InlineArray::from_f32_slice(&[1.0], &[1, 1]).as_dtype(Dtype::Bfloat16.as_i32());

        let weight_key = "model.layers.0.mlp.gate_proj.weight".to_string();
        let scale_key = "model.layers.0.mlp.gate_proj.weight_scale_inv".to_string();
        let mxfp8_scale_key = "model.layers.0.mlp.gate_proj.mxfp8_scales".to_string();
        let mut raw = HashMap::new();
        raw.insert(weight_key.clone(), fp8);
        raw.insert(scale_key.clone(), scale_inv);

        let stats = normalize_fp8_weight_scale_inv_sidecars(&mut raw, Dtype::Bfloat16.as_i32())
            .expect("normalize sidecars");

        assert_eq!(
            stats,
            Fp8SidecarLoadStats {
                repacked_mxfp8: 1,
                dequantized_dense: 0
            }
        );
        assert!(!raw.contains_key(&scale_key));
        assert_eq!(raw[&weight_key].dtype_raw(), 3);
        assert_eq!(raw[&mxfp8_scale_key].dtype_raw(), 1);

        let x_values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let x = InlineArray::from_f32_slice(&x_values, &[1, 32]).as_dtype(Dtype::Bfloat16.as_i32());
        let got = x.quantized_matmul_mode(
            raw.get(&weight_key).expect("packed weight"),
            raw.get(&mxfp8_scale_key).expect("mxfp8 scales"),
            None,
            true,
            MXFP8_GROUP_SIZE,
            MXFP8_BITS,
            QuantizedMode::Mxfp8,
        );
        let expected = x.matmul(&dense.as_dtype(Dtype::Bfloat16.as_i32()).t());

        let mut got = got;
        let mut expected = expected;
        let got = got.to_f32_vec(2).expect("got qmm");
        let expected = expected.to_f32_vec(2).expect("expected dense");
        for (got, expected) in got.iter().zip(expected) {
            assert!(
                (got - expected).abs() <= 0.75,
                "got {got}, expected {expected}"
            );
        }
    }
}
