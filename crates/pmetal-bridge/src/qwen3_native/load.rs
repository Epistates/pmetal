//! Safetensors loading + shape sanitization + quantized expert stacking.

use crate::InlineArray;

use super::{Qwen3Config, validate_quantization_runtime_support};
use super::weights::{LayerWeight, LayerWeights, NativeWeights, copy_fresh_arr};

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
        if let Some(LayerWeight::Quantized { .. }) = lw.mlp_gate_w {
            return true;
        }
        if let Some(LayerWeight::Quantized { .. }) = lw.attn_q_w {
            return true;
        }
        if let Some(LayerWeight::Quantized { .. }) = lw.gdn_qkv_w {
            return true;
        }
    }
    false
}

// ============================================================================
// MoE quantized expert stacking helper
// ============================================================================

/// Load pre-stacked MoE expert weights as a `LayerWeight`.
///
/// After Step 3e, the stacked weight tensor lives at `{base_key}.weight`.
/// For quantized models the stacking loop also stored:
///   `{base_key}.scales`  — [E, out, in/group_size] stacked scales
///   `{base_key}.biases`  — [E, out, in/group_size] stacked biases
///
/// If both scales and biases are present → `LayerWeight::Quantized`.
/// Otherwise → `LayerWeight::Dense` (the weight is already in [E, in, out] form
/// from the stacking step, so no additional transpose is needed).
fn get_stacked_expert_weight(
    raw: &std::collections::HashMap<String, InlineArray>,
    base_key: &str,
    group_size: i32,
    bits: i32,
) -> Result<LayerWeight, String> {
    let w_key = format!("{base_key}.weight");
    let s_key = format!("{base_key}.scales");
    let b_key = format!("{base_key}.biases");

    let w = raw
        .get(&w_key)
        .cloned()
        .ok_or_else(|| format!("missing stacked expert weight: {w_key}"))?;

    match (raw.get(&s_key), raw.get(&b_key)) {
        (Some(scales), Some(biases)) => Ok(LayerWeight::Quantized {
            weight: w,
            scales: scales.clone(),
            biases: biases.clone(),
            group_size,
            bits,
        }),
        _ => {
            // Dense — already [E, in, out]; no transpose needed.
            Ok(LayerWeight::Dense(w))
        }
    }
}

// Load model weights from a directory containing safetensors shards.
//
// Supports all three Qwen variants:
//   - Qwen3 dense (model_type = "qwen3"): standard attention, full RoPE.
//   - Qwen3.5 dense (model_type = "qwen3_5" / "qwen3_5_text", num_experts = 0).
//   - Qwen3.5 MoE (same type but num_experts > 0): routes through SwitchGLU +
//     shared expert. Per-expert weights are stacked into [E, in, out] tensors at
//     load time (matches Python's sanitize() stacking).
//
// Applies the same sanitization as the mlx-rs loader:
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
) -> Result<NativeWeights, String> {
    // ── Step 1+2: Shard discovery and bulk-load ─────────────────────────────
    let shard_paths = crate::native_loader::discover_safetensors_shards(model_dir)?;
    let mut raw = crate::native_loader::load_shards_into_map(&shard_paths, model_dir)?;

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
        let mut new_key = old_key.clone();
        if new_key.starts_with("language_model.model.") {
            // "language_model.model.X" → "model.X"
            new_key = new_key.replacen("language_model.", "", 1);
        } else if new_key.starts_with("language_model.") {
            // "language_model.lm_head.weight" → "lm_head.weight"
            new_key = new_key.replacen("language_model.", "", 1);
        } else if new_key.starts_with("model.language_model.") {
            new_key = new_key.replacen("model.language_model.", "model.", 1);
        }
        if new_key.contains(".A_log") {
            new_key = new_key.replace(".A_log", ".a_log");
        }
        if new_key != old_key {
            if let Some(v) = raw.remove(&old_key) {
                raw.insert(new_key, v);
            }
        }
    }

    // 3b. Drop mtp.* keys.
    raw.retain(|k, _| !k.contains("mtp."));

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
    let detected_model_dtype = raw
        .get("model.embed_tokens.weight")
        .map(|w| w.dtype_raw())
        .unwrap_or(11); // 11 = bfloat16 fallback
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

    // 3e. MoE expert weight stacking / normalization.
    //
    // Python's sanitize() stacks per-expert weights into [E, out, in]:
    //   to_join = [weights[f"{prefix}.experts.{e}.{n}.weight"] for e in range(E)]
    //   weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)
    //
    // Newer Qwen3.5 MoE checkpoints already ship experts in a packed form:
    //   - `{prefix}.experts.gate_up_proj` : [E, 2H, in]
    //   - `{prefix}.experts.down_proj`    : [E, out, H]
    // We normalize both layouts into the same `switch_mlp.*` keys so the hot
    // path stays identical.
    //
    // For dense weights:
    //   `gather_mm` in SwitchLinear calls `weight.swapaxes(-1,-2)` at forward
    //   time (so it uses [E, in, out] at runtime).  We pre-transpose to
    //   [E, in, out] here so the forward pass can call gather_mm directly
    //   without the extra transpose node.
    //
    // For quantized weights:
    //   Each expert has `.weight`, `.scales`, `.biases`.
    //   We stack weight: [E, out, in_packed] — no transpose (gather_qmm handles layout).
    //   We stack scales: [E, out, in/group_size].
    //   We stack biases: [E, out, in/group_size].
    //   These are stored under "switch_mlp.{proj}.weight/.scales/.biases".
    //
    // We store the stacked tensors under "switch_mlp.{n}.weight" keys matching
    // the post-sanitize Python layout for clarity, then look them up by layer.
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
                    format!("MoE: missing packed expert tensor {packed_gate_up_key}")
                })?;
                let down = raw.remove(&packed_down_key).ok_or_else(|| {
                    format!("MoE: missing packed expert tensor {packed_down_key}")
                })?;
                let hidden = config.moe_intermediate_size;
                if hidden <= 0 {
                    return Err(format!(
                        "MoE: invalid moe_intermediate_size={} for packed expert weights at layer {li}",
                        hidden
                    ));
                }

                // gate_up: [E, 2H, in] -> split into gate/up [E, H, in], then
                // transpose to [E, in, H] for gather_mm.
                let gate = gate_up
                    .index((.., 0..hidden, ..))
                    .transpose_axes(&[0, 2, 1]);
                let up = gate_up
                    .index((.., hidden..(hidden * 2), ..))
                    .transpose_axes(&[0, 2, 1]);
                // down: [E, out, H] -> [E, H, out] for gather_mm.
                let down = down.transpose_axes(&[0, 2, 1]);

                raw.insert(format!("{prefix}.switch_mlp.gate_proj.weight"), gate);
                raw.insert(format!("{prefix}.switch_mlp.up_proj.weight"), up);
                raw.insert(format!("{prefix}.switch_mlp.down_proj.weight"), down);
                continue;
            }

            for proj in &["gate_proj", "up_proj", "down_proj"] {
                // Detect whether first expert is quantized.
                let is_quantized = raw.contains_key(&format!("{prefix}.experts.0.{proj}.scales"));

                if is_quantized {
                    // Quantized: stack weight, scales, biases separately.
                    let mut w_shards: Vec<InlineArray> =
                        Vec::with_capacity(config.num_experts as usize);
                    let mut s_shards: Vec<InlineArray> =
                        Vec::with_capacity(config.num_experts as usize);
                    let mut b_shards: Vec<InlineArray> =
                        Vec::with_capacity(config.num_experts as usize);

                    for e in 0..config.num_experts as usize {
                        let wk = format!("{prefix}.experts.{e}.{proj}.weight");
                        let sk = format!("{prefix}.experts.{e}.{proj}.scales");
                        let bk = format!("{prefix}.experts.{e}.{proj}.biases");
                        w_shards.push(
                            raw.remove(&wk)
                                .ok_or_else(|| format!("MoE quant: missing {wk}"))?,
                        );
                        s_shards.push(
                            raw.remove(&sk)
                                .ok_or_else(|| format!("MoE quant: missing {sk}"))?,
                        );
                        b_shards.push(
                            raw.remove(&bk)
                                .ok_or_else(|| format!("MoE quant: missing {bk}"))?,
                        );
                    }

                    // Stack: weight [E, out, in_packed], scales [E, out, in/g], biases same.
                    // For quantized gather_qmm the weight layout is [E, out, in_packed]
                    // (no pre-transpose — gather_qmm handles the implicit transpose via
                    // the `transpose=true` flag we pass in LayerWeight::gather_mm_from).
                    let w_stacked = stack_arrays(w_shards, 0)?;
                    let s_stacked = stack_arrays(s_shards, 0)?;
                    let b_stacked = stack_arrays(b_shards, 0)?;

                    raw.insert(format!("{prefix}.switch_mlp.{proj}.weight"), w_stacked);
                    raw.insert(format!("{prefix}.switch_mlp.{proj}.scales"), s_stacked);
                    raw.insert(format!("{prefix}.switch_mlp.{proj}.biases"), b_stacked);
                } else {
                    // Dense: collect and pre-transpose to [E, in, out] for gather_mm.
                    let mut shards: Vec<InlineArray> =
                        Vec::with_capacity(config.num_experts as usize);
                    for e in 0..config.num_experts as usize {
                        let key = format!("{prefix}.experts.{e}.{proj}.weight");
                        let w = raw
                            .remove(&key)
                            .ok_or_else(|| format!("MoE: missing expert weight {key}"))?;
                        shards.push(w);
                    }
                    // Stack [E, out, in] → then transpose to [E, in, out] for gather_mm.
                    let stacked = stack_arrays(shards, 0)?;
                    let transposed = stacked.transpose_axes(&[0, 2, 1]);
                    raw.insert(format!("{prefix}.switch_mlp.{proj}.weight"), transposed);
                }
            }
        }
    }

    // ── Step 4: Build per-layer weight structs ──────────────────────────────
    // Quantization params — present only in quantized checkpoints.
    let (q_bits, q_group_size) = config
        .quantization_config
        .as_ref()
        .map(|qc| (qc.bits, qc.group_size))
        .unwrap_or((4, 64));
    validate_quantization_runtime_support(q_bits)?;

    let get = |key: &str| -> Result<InlineArray, String> {
        raw.get(key).cloned().ok_or_else(|| {
            let parts: Vec<&str> = key.rsplitn(2, '.').collect();
            let suffix = parts[0];
            let close: Vec<&String> = raw.keys().filter(|k| k.ends_with(suffix)).take(5).collect();
            format!("missing weight key: {key} (close matches: {close:?})")
        })
    };

    // Load a projection weight as `LayerWeight`, auto-detecting quantized format.
    //
    // For a dense weight stored at `{base_key}.weight`, looks up sibling keys
    // `{base_key}.scales` and `{base_key}.biases`.  When both are present the
    // weight is loaded as `LayerWeight::Quantized` (no transpose — the caller
    // passes `transpose=true` to `quantized_matmul` at runtime).  Otherwise
    // falls back to `LayerWeight::Dense` with the weight pre-transposed.
    //
    // `base_key` is the full key WITHOUT the trailing `.weight` suffix.
    let get_layer_weight = |base_key: &str| -> Result<LayerWeight, String> {
        let w_key = format!("{base_key}.weight");
        let s_key = format!("{base_key}.scales");
        let b_key = format!("{base_key}.biases");
        match (raw.get(&w_key), raw.get(&s_key), raw.get(&b_key)) {
            (Some(w), Some(s), Some(b)) => Ok(LayerWeight::Quantized {
                weight: w.clone(),
                scales: s.clone(),
                biases: b.clone(),
                group_size: q_group_size,
                bits: q_bits,
            }),
            _ => {
                // Dense path: load `.weight` and pre-transpose for matmul.
                let w = get(&w_key)?;
                Ok(LayerWeight::Dense(w.t()))
            }
        }
    };

    let embed_w = get("model.embed_tokens.weight")?;
    let embed_scales = get("model.embed_tokens.scales").ok();
    let embed_biases = get("model.embed_tokens.biases").ok();
    let final_norm_w = get("model.norm.weight")?;
    let final_norm_eps = config.rms_norm_eps;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        Some(get_layer_weight("lm_head")?)
    };

    let model_dtype = embed_w.dtype_raw();

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
    let attn_gated = !config.is_qwen3_dense();

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
            // router gate: stored as [num_experts, hidden]; transpose to [hidden, num_experts].
            // The router is a tiny matrix and is never quantized.
            let router = get(&format!("{p}.mlp.gate.weight"))?.t();

            // Stacked expert weights were placed under "switch_mlp.{proj}.weight" during
            // Step 3e above.  For quantized models the stacking loop also placed
            // "switch_mlp.{proj}.scales" and ".biases" — check for those here.
            let moe_gate = get_stacked_expert_weight(
                &raw,
                &format!("{p}.mlp.switch_mlp.gate_proj"),
                q_group_size,
                q_bits,
            )?;
            let moe_up = get_stacked_expert_weight(
                &raw,
                &format!("{p}.mlp.switch_mlp.up_proj"),
                q_group_size,
                q_bits,
            )?;
            let moe_down = get_stacked_expert_weight(
                &raw,
                &format!("{p}.mlp.switch_mlp.down_proj"),
                q_group_size,
                q_bits,
            )?;

            // Shared expert weights — may be quantized.
            let sh_gate = get_layer_weight(&format!("{p}.mlp.shared_expert.gate_proj"))?;
            let sh_up = get_layer_weight(&format!("{p}.mlp.shared_expert.up_proj"))?;
            let sh_down = get_layer_weight(&format!("{p}.mlp.shared_expert.down_proj"))?;
            // Shared expert gate: [1, hidden]; tiny matrix, never quantized.
            let sh_eg = get(&format!("{p}.mlp.shared_expert_gate.weight"))?.t();
            (
                None,
                None,
                None,
                Some(router),
                Some(moe_gate),
                Some(moe_up),
                Some(moe_down),
                Some(sh_gate),
                Some(sh_up),
                Some(sh_down),
                Some(sh_eg),
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
        }

        layers.push(lw);
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

    let embed_w = cf_arr(&embed_w);
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
        // MoE — router is InlineArray; stacked expert weights are LayerWeight
        if let Some(ref w) = lw.moe_router_w {
            lw.moe_router_w = Some(cf_arr(w));
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
            lw.shared_expert_gate_w = Some(cf_arr(w));
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
    let _quant_mode = if let Some(ref qc) = config.quantization_config {
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
        embed_scales,
        embed_biases,
        final_norm_w,
        final_norm_eps,
        lm_head_w,
        tie_word_embeddings: config.tie_word_embeddings,
        quantization_config: config.quantization_config.clone(),
        layers,
        model_dtype,
        qjl_matrix: None, // populated by apply_qjl_matrix() when --kv-qjl is set
    })
}

// ============================================================================
// Stack helper — concatenates arrays along a new axis
// ============================================================================
//
// MLX does not expose a standalone `stack` in the bridge; we implement it as
// expand_dims(axis=0) on each shard + successive concatenate_2 calls.
// For small E (e.g. 512 experts) this is done ONCE at load time and is not
// on the hot path.

fn stack_arrays(arrays: Vec<InlineArray>, axis: i32) -> Result<InlineArray, String> {
    if arrays.is_empty() {
        return Err("stack_arrays: empty input".to_string());
    }
    // Expand each shard: [out, in] → [1, out, in]
    let mut expanded: Vec<InlineArray> = arrays.into_iter().map(|a| a.expand_dims(axis)).collect();

    // Concatenate along the new axis: [1, out, in] × E → [E, out, in]
    let mut acc = expanded.remove(0);
    for e in expanded {
        acc = acc.concatenate_2(&e, axis);
    }
    Ok(acc)
}
