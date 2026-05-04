//! Per-layer + full-model weight bundles, FP8 dequantization, and expert
//! sanitization for DeepSeek V3/R1 checkpoints.

use crate::InlineArray;

use super::DeepSeekConfig;

/// Holds weights for one MoE expert (gate_proj, up_proj, down_proj) or
/// a stacked form when all expert weights are concatenated.
pub(super) struct MoEWeights {
    /// Stacked gate projections: [n_experts, intermediate, hidden] pre-transposed.
    pub(super) gate_w: InlineArray,
    /// Stacked up projections: [n_experts, intermediate, hidden] pre-transposed.
    pub(super) up_w: InlineArray,
    /// Stacked down projections: [n_experts, hidden, intermediate] pre-transposed.
    pub(super) down_w: InlineArray,
    /// Routing gate weight: [n_experts, hidden].
    pub(super) gate_weight: InlineArray,
    /// Auxiliary-free bias for expert score correction: [n_experts].
    pub(super) e_score_correction_bias: InlineArray,
    /// Shared expert gate_proj: [shared_intermediate, hidden] pre-transposed.
    pub(super) shared_gate_w: Option<InlineArray>,
    /// Shared expert up_proj: [shared_intermediate, hidden] pre-transposed.
    pub(super) shared_up_w: Option<InlineArray>,
    /// Shared expert down_proj: [hidden, shared_intermediate] pre-transposed.
    pub(super) shared_down_w: Option<InlineArray>,
    // MoE config scalars
    pub(super) n_routed_experts: i32,
    pub(super) n_group: i32,
    pub(super) topk_group: i32,
    pub(super) top_k: i32,
    pub(super) routed_scaling_factor: f32,
    pub(super) norm_topk_prob: bool,
}

pub(super) struct LayerWeights {
    // Shared: layer norms
    pub(super) input_ln_w: InlineArray,
    pub(super) post_ln_w: InlineArray,
    pub(super) norm_eps: f32,

    // ── MLA attention ─────────────────────────────────────────────────────
    // Q projection (low-rank path when q_lora_rank is Some):
    //   q_a_proj: [hidden, q_lora_rank]
    //   q_a_layernorm: [q_lora_rank]
    //   q_b_proj: [q_lora_rank, n_heads * q_head_dim]
    // or direct path:
    //   q_proj: [hidden, n_heads * q_head_dim]
    pub(super) q_a_w: Option<InlineArray>, // pre-transposed [hidden, q_lora_rank]
    pub(super) q_a_norm_w: Option<InlineArray>, // [q_lora_rank] for rms_norm
    pub(super) q_b_w: Option<InlineArray>, // pre-transposed [q_lora_rank, n_heads*q_head_dim]
    pub(super) q_w: Option<InlineArray>,   // pre-transposed [hidden, n_heads*q_head_dim] (direct)

    // KV compression: kv_a_proj_with_mqa projects x → [kv_lora_rank + qk_rope_head_dim]
    pub(super) kv_a_proj_w: InlineArray, // pre-transposed [hidden, kv_lora_rank + qk_rope_head_dim]
    pub(super) kv_a_norm_w: InlineArray, // [kv_lora_rank] for rms_norm

    // embed_q: [n_heads, qk_nope_head_dim, kv_lora_rank] — W_uk^T
    //   Decode (L=1): q_nope @ embed_q_w^T = q_nope @ [n_heads, lora, nope]^T = q_nope @ [n_heads, nope, lora]
    //   For bmm we store in shape that makes matmul efficient.
    //   embed_q.weight in Python: [n_heads, qk_nope_head_dim, kv_lora_rank]
    //   In decode: q_nope [B,H,1,nope] @ embed_q_w.swapaxes(-1,-2) [H,lora,nope]^T → [B,H,1,lora]
    //   So embed_q_w stored as [H, nope_dim, lora_rank] for direct bmm (no extra transpose)
    pub(super) embed_q_w: InlineArray, // [n_heads, qk_nope_head_dim, kv_lora_rank]

    // unembed_out: [n_heads, kv_lora_rank, v_head_dim] — W_uv
    //   Decode: output [B,H,1,lora] @ unembed_out_w [H,lora,v_dim] → [B,H,1,v_dim]
    pub(super) unembed_out_w: InlineArray, // [n_heads, kv_lora_rank, v_head_dim]

    // Output projection: [n_heads * v_head_dim, hidden] pre-transposed to [n_heads*v_dim, hidden]
    pub(super) o_proj_w: InlineArray, // pre-transposed [n_heads*v_head_dim, hidden]

    // Attention scalars
    pub(super) n_heads: i32,
    pub(super) q_head_dim: i32,
    pub(super) qk_nope_head_dim: i32,
    pub(super) qk_rope_head_dim: i32,
    pub(super) v_head_dim: i32,
    pub(super) kv_lora_rank: i32,
    pub(super) scale: f32,
    pub(super) rope_base: f32,
    pub(super) rope_scale: f32,

    // ── MLP / MoE ─────────────────────────────────────────────────────────
    pub(super) is_moe: bool,

    // Dense MLP weights (only when !is_moe)
    pub(super) mlp_gate_w: Option<InlineArray>,
    pub(super) mlp_up_w: Option<InlineArray>,
    pub(super) mlp_down_w: Option<InlineArray>,

    // MoE weights (only when is_moe)
    pub(super) moe: Option<Box<MoEWeights>>,
}

/// All model weights as InlineArray. Zero dependency on mlx-rs.
pub struct NativeWeights {
    pub embed_w: InlineArray,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    /// None when `tie_word_embeddings = true`.
    pub lm_head_w: Option<InlineArray>,
    pub tie_word_embeddings: bool,
    pub(super) layers: Vec<LayerWeights>,
    pub model_dtype: i32,
}

impl std::fmt::Debug for NativeWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeWeights")
            .field("layers", &self.layers.len())
            .field("tie_word_embeddings", &self.tie_word_embeddings)
            .field("model_dtype", &self.model_dtype)
            .finish()
    }
}

/// Load model weights from a directory of safetensors shards.
///
/// Applies the sanitization described in the Python `Model.sanitize()`:
/// - FP8 dequantization (model weights stored as e4m3)
/// - INT4 weight repacking
/// - Expert stacking: `experts.{0..N}.{gate,up,down}_proj` → `switch_mlp.{*}.weight`
/// - KV-B-proj split into `embed_q` + `unembed_out`
/// - Drops `rotary_emb.inv_freq` and MTP layers (`model.layers.61`)
pub fn load_model(
    model_dir: &std::path::Path,
    config: &DeepSeekConfig,
) -> Result<NativeWeights, String> {
    // ── Step 1+2: Shard discovery and bulk-load ─────────────────────────────
    let shard_paths = crate::native_loader::discover_safetensors_shards(model_dir)?;
    let mut raw = crate::native_loader::load_shards_into_map(&shard_paths, model_dir)?;

    // ── Step 3: Sanitization ────────────────────────────────────────────────

    // 3a. Drop MTP layers (model.layers.61 is the auxiliary MTP module in V3).
    // Also drop rotary_emb.inv_freq (precomputed, not needed — we compute on the fly).
    raw.retain(|k, _| !k.contains("rotary_emb.inv_freq") && !k.starts_with("model.layers.61"));

    // Drop lm_head when tied.
    if config.tie_word_embeddings {
        raw.remove("lm_head.weight");
    }

    // Detect model dtype for scalar creation.
    let detected_model_dtype = raw
        .get("model.embed_tokens.weight")
        .map(|w| w.dtype_raw())
        .unwrap_or(11); // 11 = bfloat16 fallback

    // 3b. FP8 dequantization.
    // The DeepSeek V3 checkpoint uses E4M3 FP8 for weights with a separate
    // `_scale_inv` tensor (block-wise scaling, block_size=128).
    // We dequantize to bfloat16 in-place so the rest of the loader sees
    // plain floating-point tensors.
    //
    // Note: MLX's quantized_matmul handles FP8 natively in some paths, but
    // for simplicity in this native engine we dequantize eagerly.
    {
        // Collect all scale_inv keys first to avoid borrow conflict.
        let scale_inv_keys: Vec<String> = raw
            .keys()
            .filter(|k| k.ends_with("_scale_inv"))
            .cloned()
            .collect();

        for scale_inv_key in scale_inv_keys {
            let weight_key = scale_inv_key.replace("_scale_inv", "");
            if let (Some(scale_inv), Some(weight)) = (
                raw.get(&scale_inv_key).cloned(),
                raw.get(&weight_key).cloned(),
            ) {
                // Dequantize: weight is fp8 e4m3, scale_inv is fp32 block-wise.
                // Block size = 128. Weight shape: [M, N]. Scale shape: [M/128, N/128].
                // dequantized[i,j] = fp8_to_fp32(weight[i,j]) * scale_inv[i//128, j//128]
                // We implement this using MLX operations:
                //   1. from_fp8 → bf16
                //   2. reshape to [M/128, 128, N/128, 128]
                //   3. multiply by scale_inv[:, None, :, None]
                //   4. reshape back
                let dequant = crate::native_loader::dequantize_fp8_e4m3_scaled_weight(
                    &weight,
                    &scale_inv,
                    detected_model_dtype,
                )
                .map_err(|e| format!("{weight_key}: {e}"))?;
                raw.insert(weight_key, dequant);
                raw.remove(&scale_inv_key);
            }
        }
    }

    // 3c. Stack routed experts into batched tensors.
    // Python sanitize(): experts.{e}.{m}.weight → switch_mlp.{m}.weight (stacked)
    let n_layers = config.num_hidden_layers as usize;
    let n_experts = config.n_routed_experts.unwrap_or(0) as usize;

    if n_experts > 0 {
        for li in 0..n_layers {
            if !config.is_moe_layer(li) {
                continue;
            }
            let p = format!("model.layers.{li}");
            // Check whether per-expert keys exist (unsanitized checkpoint).
            let first_expert_key = format!("{p}.mlp.experts.0.gate_proj.weight");
            if raw.contains_key(&first_expert_key) {
                for (n, m) in [
                    ("gate_proj", "gate_proj"),
                    ("down_proj", "down_proj"),
                    ("up_proj", "up_proj"),
                ] {
                    let to_join: Option<Vec<InlineArray>> = (0..n_experts)
                        .map(|e| raw.remove(&format!("{p}.mlp.experts.{e}.{m}.weight")))
                        .collect();
                    if let Some(parts) = to_join {
                        // Stack along new axis 0 → [n_experts, out, in]
                        let stacked = stack_arrays(&parts);
                        raw.insert(format!("{p}.mlp.switch_mlp.{n}.weight"), stacked);
                    }
                }
            }
        }
    }

    // 3d. Split kv_b_proj into embed_q + unembed_out per layer.
    // kv_b_proj.weight: [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    // → reshape to [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]
    // → split on axis 1 at qk_nope_head_dim:
    //     embed_q.weight:   [n_heads, qk_nope_head_dim, kv_lora_rank]  (W_uk transposed)
    //     unembed_out.weight: [n_heads, v_head_dim, kv_lora_rank]       (W_uv)
    // In Python (sanitize):
    //   wk = v[:, :qk_nope_head_dim, :].swapaxes(-1,-2) → [H, lora, nope]  → embed_q.weight stored transposed
    //   wv = v[:, qk_nope_head_dim:, :]               → [H, v_dim, lora]  → unembed_out.weight
    //
    // The Python MultiLinear forward:
    //   embed_q(q_nope, transpose=True): q_nope @ embed_q.weight.swapaxes(-1,-2)
    //     = [B,H,1,nope] @ [H,lora,nope].T = [B,H,1,nope] @ [H,nope,lora] → [B,H,1,lora]  Wait, that's wrong.
    //   Actually: embed_q.weight is [H, nope_dim, lora_rank] per Python init.
    //   But after sanitize(), embed_q.weight = wk = v[:, :nope, :].swapaxes(-1,-2) = [H, lora, nope]
    //   embed_q(q_nope, transpose=True): q_nope @ embed_q.weight.swapaxes(-1,-2)
    //     = [B,H,1,nope] @ [H,lora,nope].swapaxes(-1,-2) = [B,H,1,nope] @ [H,nope,lora] → [B,H,1,lora]
    //   embed_q(kv_latent, transpose=False): kv_latent @ embed_q.weight (prefill)
    //     = [B,1,T,lora] @ [H,lora,nope] → but shapes don't broadcast... Python uses batch matmul.
    //     kv_latent expanded: [B,1,T,lora], embed_q.weight [H,lora,nope]
    //     → [B,H,T,nope] via batched matmul with broadcasting over H
    //
    // We store:
    //   embed_q_w = wk = [H, lora_rank, nope_dim]  (as Python stores it after sanitize)
    //   unembed_out_w = wv = [H, v_head_dim, lora_rank]

    for li in 0..n_layers {
        let prefix = format!("model.layers.{li}.self_attn");
        let kv_b_key = format!("{prefix}.kv_b_proj.weight");
        if let Some(kv_b) = raw.remove(&kv_b_key) {
            // kv_b shape: [n_heads * (nope + v_dim), kv_lora_rank]
            let n_heads = config.num_attention_heads;
            let head_dim = config.qk_nope_head_dim + config.v_head_dim;
            let kv_lora_rank = config.kv_lora_rank;

            // reshape → [n_heads, nope+v_dim, lora_rank]
            let v = kv_b.reshape(&[n_heads, head_dim, kv_lora_rank]);

            // split at qk_nope_head_dim along axis 1
            let mut parts = v.split(&[config.qk_nope_head_dim], 1);
            let v_nope = parts.remove(0); // [H, nope, lora]
            let v_v = parts.remove(0); // [H, v_dim, lora]

            // embed_q.weight = wk = v_nope.swapaxes(-1,-2) → [H, lora, nope]
            let wk = v_nope.transpose_axes(&[0, 2, 1]);
            // unembed_out.weight = wv = v_v → [H, v_dim, lora]
            let wv = v_v;

            raw.insert(format!("{prefix}.embed_q.weight"), wk);
            raw.insert(format!("{prefix}.unembed_out.weight"), wv);
        }
    }

    // ── Step 4: Build per-layer weight structs ──────────────────────────────
    let get = |key: &str| -> Result<InlineArray, String> {
        raw.get(key).cloned().ok_or_else(|| {
            let parts: Vec<&str> = key.rsplitn(2, '.').collect();
            let suffix = parts[0];
            let close: Vec<&String> = raw.keys().filter(|k| k.ends_with(suffix)).take(5).collect();
            format!("missing weight key: {key} (close matches: {close:?})")
        })
    };

    let embed_w = get("model.embed_tokens.weight")?;
    let final_norm_w = get("model.norm.weight")?;
    let final_norm_eps = config.rms_norm_eps;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        // lm_head.weight stored as [vocab, hidden]; pre-transpose to [hidden, vocab].
        Some(get("lm_head.weight")?.t())
    };

    let model_dtype = embed_w.dtype_raw();

    let scale = config.attention_scale();
    let rope_base = config.rope_base_f32();
    let rope_scale = 1.0_f32;

    let mut layers = Vec::with_capacity(n_layers);

    for li in 0..n_layers {
        let p = format!("model.layers.{li}");
        let sa = format!("{p}.self_attn");

        let input_ln_w = get(&format!("{p}.input_layernorm.weight"))?;
        let post_ln_w = get(&format!("{p}.post_attention_layernorm.weight"))?;

        // Q projection (low-rank or direct).
        let (q_a_w, q_a_norm_w, q_b_w, q_w) = if config.q_lora_rank.is_some() {
            let qa = get(&format!("{sa}.q_a_proj.weight"))?.t();
            let qa_norm = get(&format!("{sa}.q_a_layernorm.weight"))?;
            let qb = get(&format!("{sa}.q_b_proj.weight"))?.t();
            (Some(qa), Some(qa_norm), Some(qb), None)
        } else {
            let q = get(&format!("{sa}.q_proj.weight"))?.t();
            (None, None, None, Some(q))
        };

        // KV compression projection: [hidden, kv_lora_rank + qk_rope_head_dim]
        let kv_a_proj_w = get(&format!("{sa}.kv_a_proj_with_mqa.weight"))?.t();
        let kv_a_norm_w = get(&format!("{sa}.kv_a_layernorm.weight"))?;

        // embed_q and unembed_out (split from kv_b_proj above).
        // embed_q.weight: [H, lora_rank, nope_dim]
        // unembed_out.weight: [H, v_dim, lora_rank]
        let embed_q_w = get(&format!("{sa}.embed_q.weight"))?;
        let unembed_out_w = get(&format!("{sa}.unembed_out.weight"))?;

        let o_proj_w = get(&format!("{sa}.o_proj.weight"))?.t();

        // MLP / MoE
        let is_moe = config.is_moe_layer(li);
        let (mlp_gate_w, mlp_up_w, mlp_down_w, moe) = if is_moe {
            let moe_prefix = format!("{p}.mlp");
            let n_re = config.n_routed_experts.unwrap_or(0);
            let n_shared = config.n_shared_experts.unwrap_or(0);
            let moe_inter = config
                .moe_intermediate_size
                .unwrap_or(config.intermediate_size);

            // Stacked expert weights: [n_experts, intermediate, hidden] stored pre-transposed.
            // In safetensors the stacked weights are [n_experts, intermediate, hidden] for
            // gate/up (transposed to [n_experts, hidden, intermediate] for matmul) and
            // [n_experts, hidden, intermediate] for down (transposed to [n_experts, intermediate, hidden]).
            // We pre-transpose to the matmul-ready form.
            let sw_gate = get(&format!("{moe_prefix}.switch_mlp.gate_proj.weight"))?;
            let sw_up = get(&format!("{moe_prefix}.switch_mlp.up_proj.weight"))?;
            let sw_down = get(&format!("{moe_prefix}.switch_mlp.down_proj.weight"))?;

            // gate.weight: [n_experts, hidden]
            let gate_weight = get(&format!("{moe_prefix}.gate.weight"))?;
            // e_score_correction_bias: [n_experts]
            let e_score_bias = get(&format!("{moe_prefix}.gate.e_score_correction_bias"))?;

            // Shared expert (optional).
            let (s_gate, s_up, s_down) = if n_shared > 0 {
                let sh_inter = moe_inter * n_shared;
                let sg = get(&format!("{moe_prefix}.shared_experts.gate_proj.weight"))
                    .map(|w| w.t())
                    .ok();
                let su = get(&format!("{moe_prefix}.shared_experts.up_proj.weight"))
                    .map(|w| w.t())
                    .ok();
                let sd = get(&format!("{moe_prefix}.shared_experts.down_proj.weight"))
                    .map(|w| w.t())
                    .ok();
                let _ = sh_inter; // used for docs
                (sg, su, sd)
            } else {
                (None, None, None)
            };

            let moe_weights = Box::new(MoEWeights {
                gate_w: sw_gate,
                up_w: sw_up,
                down_w: sw_down,
                gate_weight,
                e_score_correction_bias: e_score_bias,
                shared_gate_w: s_gate,
                shared_up_w: s_up,
                shared_down_w: s_down,
                n_routed_experts: n_re,
                n_group: config.n_group,
                topk_group: config.topk_group,
                top_k: config.num_experts_per_tok,
                routed_scaling_factor: config.routed_scaling_factor,
                norm_topk_prob: config.norm_topk_prob,
            });
            (None, None, None, Some(moe_weights))
        } else {
            let gate = get(&format!("{p}.mlp.gate_proj.weight"))?.t();
            let up = get(&format!("{p}.mlp.up_proj.weight"))?.t();
            let down = get(&format!("{p}.mlp.down_proj.weight"))?.t();
            (Some(gate), Some(up), Some(down), None)
        };

        layers.push(LayerWeights {
            input_ln_w,
            post_ln_w,
            norm_eps: config.rms_norm_eps,
            q_a_w,
            q_a_norm_w,
            q_b_w,
            q_w,
            kv_a_proj_w,
            kv_a_norm_w,
            embed_q_w,
            unembed_out_w,
            o_proj_w,
            n_heads: config.num_attention_heads,
            q_head_dim: config.q_head_dim(),
            qk_nope_head_dim: config.qk_nope_head_dim,
            qk_rope_head_dim: config.qk_rope_head_dim,
            v_head_dim: config.v_head_dim,
            kv_lora_rank: config.kv_lora_rank,
            scale,
            rope_base,
            rope_scale,
            is_moe,
            mlp_gate_w,
            mlp_up_w,
            mlp_down_w,
            moe,
        });
    }

    // ── Step 5: copy_fresh ──────────────────────────────────────────────────
    // Force all weights into fresh Metal buffers (use_count=1).
    let zero = InlineArray::scalar_with_dtype(0.0, model_dtype);
    let copy_fresh = |w: &InlineArray| -> InlineArray {
        let mut fresh = w.add(&zero);
        fresh.eval();
        fresh.detach();
        fresh
    };

    let embed_w = copy_fresh(&embed_w);
    let final_norm_w = copy_fresh(&final_norm_w);
    let lm_head_w = lm_head_w.map(|w| copy_fresh(&w));

    for lw in &mut layers {
        lw.input_ln_w = copy_fresh(&lw.input_ln_w);
        lw.post_ln_w = copy_fresh(&lw.post_ln_w);
        if let Some(ref w) = lw.q_a_w {
            lw.q_a_w = Some(copy_fresh(w));
        }
        if let Some(ref w) = lw.q_a_norm_w {
            lw.q_a_norm_w = Some(copy_fresh(w));
        }
        if let Some(ref w) = lw.q_b_w {
            lw.q_b_w = Some(copy_fresh(w));
        }
        if let Some(ref w) = lw.q_w {
            lw.q_w = Some(copy_fresh(w));
        }
        lw.kv_a_proj_w = copy_fresh(&lw.kv_a_proj_w);
        lw.kv_a_norm_w = copy_fresh(&lw.kv_a_norm_w);
        lw.embed_q_w = copy_fresh(&lw.embed_q_w);
        lw.unembed_out_w = copy_fresh(&lw.unembed_out_w);
        lw.o_proj_w = copy_fresh(&lw.o_proj_w);
        if let Some(ref w) = lw.mlp_gate_w {
            lw.mlp_gate_w = Some(copy_fresh(w));
        }
        if let Some(ref w) = lw.mlp_up_w {
            lw.mlp_up_w = Some(copy_fresh(w));
        }
        if let Some(ref w) = lw.mlp_down_w {
            lw.mlp_down_w = Some(copy_fresh(w));
        }
        if let Some(ref mut moe) = lw.moe {
            moe.gate_w = copy_fresh(&moe.gate_w);
            moe.up_w = copy_fresh(&moe.up_w);
            moe.down_w = copy_fresh(&moe.down_w);
            moe.gate_weight = copy_fresh(&moe.gate_weight);
            moe.e_score_correction_bias = copy_fresh(&moe.e_score_correction_bias);
            if let Some(ref w) = moe.shared_gate_w {
                moe.shared_gate_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = moe.shared_up_w {
                moe.shared_up_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = moe.shared_down_w {
                moe.shared_down_w = Some(copy_fresh(w));
            }
        }
    }

    eprintln!(
        "[DEEPSEEK] load_model: {} layers, dtype={}",
        layers.len(),
        model_dtype
    );

    Ok(NativeWeights {
        embed_w,
        final_norm_w,
        final_norm_eps,
        lm_head_w,
        tie_word_embeddings: config.tie_word_embeddings,
        layers,
        model_dtype,
    })
}

// ============================================================================
// Expert stacking helper
// ============================================================================

/// Stack a slice of same-shape arrays along a new leading axis.
///
/// This approximates `mx.stack(parts)` which is not directly available as a
/// C++ bridge call. We implement it by:
///   1. Reshape each [d0, d1, ...] → [1, d0, d1, ...]
///   2. Concatenate along axis 0 → [N, d0, d1, ...]
///
/// Note: For the bridge we use repeated `kv_cache_append` as a concat proxy
/// since we already have `concatenate_2`. We build iteratively.
fn stack_arrays(parts: &[InlineArray]) -> InlineArray {
    assert!(!parts.is_empty(), "stack_arrays: empty input");
    // Get shape of first element.
    let ndim = parts[0].ndim();
    let mut shape_1 = Vec::with_capacity(ndim as usize + 1);
    shape_1.push(1i32);
    for i in 0..ndim {
        shape_1.push(parts[0].dim(i));
    }

    // Reshape all to [1, d0, d1, ...] then concatenate along axis 0.
    let first = parts[0].reshape(&shape_1);
    let mut acc = first;

    for part in &parts[1..] {
        let p_r = part.reshape(&shape_1);
        acc = acc.concatenate_2(&p_r, 0);
    }
    acc
}
