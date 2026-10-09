//! Per-layer + full-model weight bundles and safetensors loading.
//!
//! Two dense expert layouts load:
//!
//! * transformers' own (a bf16 release): `mlp.experts.gate_up_proj
//!   [E, H, 2I]`, applied as `x @ W`, with gate and up **interleaved** on the
//!   last axis (`[..., ::2]` / `[..., 1::2]`), its `gate_up_proj_bias [E, 2I]`
//!   interleaved the same way, and `down_proj [E, I, H]` / `down_proj_bias`;
//! * the stacked per-projection layout MLX conversions ship:
//!   `mlp.experts.{gate,up}_proj.weight [E, I, H]`,
//!   `mlp.experts.down_proj.weight [E, H, I]`, each with its `.bias`.
//!
//! Packed checkpoints (the release's MXFP4 `*_blocks` / `*_scales`, or an MLX
//! quantization) are refused by name: the expert kernels here are dense.

use crate::InlineArray;
use crate::native_weight::LayerWeight;

use super::{AttentionLayerType, GptOssConfig};

/// GPT-OSS layer weights — attention + MoE.
pub(super) struct LayerWeights {
    // Layer norms
    pub(super) input_ln_w: InlineArray,
    pub(super) input_ln_eps: f32,
    pub(super) post_ln_w: InlineArray,
    pub(super) post_ln_eps: f32,

    // Attention projections (pre-transposed [in, out] for direct matmul)
    pub(super) attn_q_w: InlineArray, // [hidden, n_heads * head_dim]
    pub(super) attn_q_b: Option<InlineArray>, // [n_heads * head_dim]
    pub(super) attn_k_w: InlineArray, // [hidden, n_kv_heads * head_dim]
    pub(super) attn_k_b: Option<InlineArray>, // [n_kv_heads * head_dim]
    pub(super) attn_v_w: InlineArray, // [hidden, n_kv_heads * head_dim]
    pub(super) attn_v_b: Option<InlineArray>, // [n_kv_heads * head_dim]
    pub(super) attn_o_w: InlineArray, // [n_heads * head_dim, hidden]
    pub(super) attn_o_b: Option<InlineArray>, // [hidden]
    /// Per-head attention sink logits, `[n_heads]`, in the model dtype: one
    /// more logit in each softmax row that attends to nothing.
    pub(super) attn_sinks: InlineArray,

    // Attention dims
    pub(super) attn_n_heads: i32,
    pub(super) attn_n_kv_heads: i32,
    pub(super) attn_head_dim: i32,
    pub(super) attn_scale: f32,
    /// The rotary embedding (YaRN on every release).
    pub(super) attn_rotary: crate::rope::RotaryEmbedding,
    pub(super) attn_is_sliding: bool,
    pub(super) attn_sliding_window: i32,

    // MoE
    /// Router `[hidden, num_experts]` and its bias `[num_experts]`.
    pub(super) moe_router_w: InlineArray,
    pub(super) moe_router_b: InlineArray,
    /// Experts stacked `[E, in, out]` for `gather_mm`.
    pub(super) moe_gate_w: LayerWeight, // [E, hidden, intermediate]
    pub(super) moe_up_w: LayerWeight,   // [E, hidden, intermediate]
    pub(super) moe_down_w: LayerWeight, // [E, intermediate, hidden]
    pub(super) moe_gate_b: InlineArray, // [E, intermediate]
    pub(super) moe_up_b: InlineArray,   // [E, intermediate]
    pub(super) moe_down_b: InlineArray, // [E, hidden]

    pub(super) moe_num_experts: i32,
    pub(super) moe_top_k: i32,

    // GLU parameters
    pub(super) swiglu_alpha: f32,
    pub(super) swiglu_limit: f32,
}

/// All GPT-OSS model weights as InlineArray. Zero dependency on mlx-rs.
pub struct NativeWeights {
    pub embed_w: InlineArray,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    /// None when `tie_word_embeddings = true`.
    pub lm_head_w: Option<InlineArray>,
    pub tie_word_embeddings: bool,
    /// Per-layer weights — opaque to callers.
    pub(super) layers: Vec<LayerWeights>,
    /// Model activation dtype (e.g., 11 = bfloat16).
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

/// One layer's experts in the `[E, in, out]` layout `gather_mm` takes, with
/// their biases.
struct Experts {
    gate_w: InlineArray,
    up_w: InlineArray,
    down_w: InlineArray,
    gate_b: InlineArray,
    up_b: InlineArray,
    down_b: InlineArray,
}

/// The even (`0`) or odd (`1`) entries of `x`'s last axis: transformers'
/// `x[..., ::2]` / `x[..., 1::2]`.
fn interleaved(x: &InlineArray, which: i32) -> InlineArray {
    let mut shape: Vec<i32> = (0..x.ndim()).map(|i| x.dim(i)).collect();
    let last = shape.pop().expect("an expert tensor has axes");
    let pairs: Vec<i32> = shape.iter().copied().chain([last / 2, 2]).collect();
    let mut start = vec![0; pairs.len()];
    let mut stop = pairs.clone();
    *start.last_mut().unwrap() = which;
    *stop.last_mut().unwrap() = which + 1;
    x.reshape(&pairs).slice(&start, &stop).squeeze(-1)
}

/// Read one layer's experts from either dense layout (see the module docs).
fn load_experts(
    raw: &std::collections::HashMap<String, InlineArray>,
    exp: &str,
) -> Result<Experts, String> {
    let get = |key: &str| {
        raw.get(key)
            .cloned()
            .ok_or_else(|| format!("missing weight key: {key}"))
    };
    let swap = |x: InlineArray| x.transpose_axes(&[0, 2, 1]);
    if let Some(gate_up) = raw.get(&format!("{exp}.gate_up_proj")) {
        // [E, H, 2I], gate and up interleaved on the last axis.
        let gate_up_b = get(&format!("{exp}.gate_up_proj_bias"))?;
        return Ok(Experts {
            gate_w: interleaved(gate_up, 0),
            up_w: interleaved(gate_up, 1),
            down_w: get(&format!("{exp}.down_proj"))?,
            gate_b: interleaved(&gate_up_b, 0),
            up_b: interleaved(&gate_up_b, 1),
            down_b: get(&format!("{exp}.down_proj_bias"))?,
        });
    }
    // Stacked `[E, out, in]`.
    Ok(Experts {
        gate_w: swap(get(&format!("{exp}.gate_proj.weight"))?),
        up_w: swap(get(&format!("{exp}.up_proj.weight"))?),
        down_w: swap(get(&format!("{exp}.down_proj.weight"))?),
        gate_b: get(&format!("{exp}.gate_proj.bias"))?,
        up_b: get(&format!("{exp}.up_proj.bias"))?,
        down_b: get(&format!("{exp}.down_proj.bias"))?,
    })
}

/// Load GPT-OSS model weights from a directory containing safetensors shards.
pub fn load_model(
    model_dir: &std::path::Path,
    config: &GptOssConfig,
) -> Result<NativeWeights, String> {
    let shard_paths = crate::native_loader::discover_safetensors_shards(model_dir)?;
    let mut raw = crate::native_loader::load_shards_into_map(&shard_paths, model_dir)?;

    if let Some(packed) = raw
        .keys()
        .find(|k| k.ends_with("_blocks") || k.ends_with("_scales") || k.ends_with(".scales"))
    {
        return Err(format!(
            "gpt_oss: `{packed}` is a packed (MXFP4 or MLX-quantized) tensor, which the \
             native gpt-oss engine does not run; use a dense (bf16 / f16 / f32) checkpoint"
        ));
    }

    // Drop lm_head when embeddings are tied.
    if config.tie_word_embeddings {
        raw.remove("lm_head.weight");
    }

    let get = |key: &str| -> Result<InlineArray, String> {
        raw.get(key).cloned().ok_or_else(|| {
            let parts: Vec<&str> = key.rsplitn(2, '.').collect();
            let suffix = parts[0];
            let close: Vec<&String> = raw.keys().filter(|k| k.ends_with(suffix)).take(5).collect();
            format!("missing weight key: {key} (close matches: {close:?})")
        })
    };
    let get_opt = |key: &str| -> Option<InlineArray> { raw.get(key).cloned() };

    let embed_w = get("model.embed_tokens.weight")?;
    let final_norm_w = get("model.norm.weight")?;
    let final_norm_eps = config.rms_norm_eps;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        // lm_head.weight stored as [vocab, hidden]; pre-transpose to [hidden, vocab]
        Some(get("lm_head.weight")?.t())
    };

    let model_dtype = embed_w.dtype_raw();

    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let rotary = config.rotary()?;
    let n_experts = config.num_local_experts;
    let top_k = config.experts_per_tok();
    let use_bias = config.attention_bias;

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);

    for li in 0..config.num_hidden_layers as usize {
        let p = format!("model.layers.{li}");
        let sa = format!("{p}.self_attn");
        let mlp = format!("{p}.mlp");
        let layer_type = config.layer_type(li);
        let is_sliding = layer_type == AttentionLayerType::SlidingAttention;

        let bias = |name: &str| {
            if use_bias {
                get_opt(&format!("{sa}.{name}.bias"))
            } else {
                None
            }
        };
        let experts = load_experts(&raw, &format!("{mlp}.experts"))?;

        layers.push(LayerWeights {
            input_ln_w: get(&format!("{p}.input_layernorm.weight"))?,
            input_ln_eps: config.rms_norm_eps,
            post_ln_w: get(&format!("{p}.post_attention_layernorm.weight"))?,
            post_ln_eps: config.rms_norm_eps,

            // Stored [out, in]; pre-transposed to [in, out].
            attn_q_w: get(&format!("{sa}.q_proj.weight"))?.t(),
            attn_q_b: bias("q_proj"),
            attn_k_w: get(&format!("{sa}.k_proj.weight"))?.t(),
            attn_k_b: bias("k_proj"),
            attn_v_w: get(&format!("{sa}.v_proj.weight"))?.t(),
            attn_v_b: bias("v_proj"),
            attn_o_w: get(&format!("{sa}.o_proj.weight"))?.t(),
            attn_o_b: bias("o_proj"),
            // MLX's SDPA takes sinks only in the output dtype.
            attn_sinks: get(&format!("{sa}.sinks"))?.as_dtype(model_dtype),

            attn_n_heads: n_heads,
            attn_n_kv_heads: n_kv_heads,
            attn_head_dim: head_dim,
            attn_scale,
            attn_rotary: rotary.clone(),
            attn_is_sliding: is_sliding,
            attn_sliding_window: config.sliding_window,

            // Router stored [E, hidden]; pre-transposed to [hidden, E].
            moe_router_w: get(&format!("{mlp}.router.weight"))?.t(),
            moe_router_b: get(&format!("{mlp}.router.bias"))?,
            moe_gate_w: LayerWeight::Dense(experts.gate_w),
            moe_up_w: LayerWeight::Dense(experts.up_w),
            moe_down_w: LayerWeight::Dense(experts.down_w),
            moe_gate_b: experts.gate_b,
            moe_up_b: experts.up_b,
            moe_down_b: experts.down_b,
            moe_num_experts: n_experts,
            moe_top_k: top_k,

            swiglu_alpha: config.swiglu_alpha,
            swiglu_limit: config.swiglu_limit,
        });

        if li == 0 {
            eprintln!(
                "[GPT-OSS] layer 0: type={:?} n_heads={n_heads} n_kv={n_kv_heads} \
                 head_dim={head_dim} experts={n_experts} top_k={top_k}",
                layer_type,
            );
        }
    }

    // Force every weight into a fresh, contiguous Metal buffer (the expert
    // splits above are strided views of the fused tensors).
    let zero = InlineArray::scalar_with_dtype(0.0, model_dtype);
    let copy_fresh = |w: &InlineArray| -> InlineArray {
        let mut fresh = w.add(&zero);
        fresh.eval();
        fresh.detach();
        fresh
    };
    let copy_fresh_opt =
        |w: Option<InlineArray>| -> Option<InlineArray> { w.map(|w| copy_fresh(&w)) };

    let embed_w = copy_fresh(&embed_w);
    let final_norm_w = copy_fresh(&final_norm_w);
    let lm_head_w = lm_head_w.map(|w| copy_fresh(&w));

    for lw in &mut layers {
        lw.input_ln_w = copy_fresh(&lw.input_ln_w);
        lw.post_ln_w = copy_fresh(&lw.post_ln_w);
        lw.attn_q_w = copy_fresh(&lw.attn_q_w);
        lw.attn_k_w = copy_fresh(&lw.attn_k_w);
        lw.attn_v_w = copy_fresh(&lw.attn_v_w);
        lw.attn_o_w = copy_fresh(&lw.attn_o_w);
        lw.attn_q_b = copy_fresh_opt(lw.attn_q_b.take());
        lw.attn_k_b = copy_fresh_opt(lw.attn_k_b.take());
        lw.attn_v_b = copy_fresh_opt(lw.attn_v_b.take());
        lw.attn_o_b = copy_fresh_opt(lw.attn_o_b.take());
        lw.attn_sinks = copy_fresh(&lw.attn_sinks);
        lw.moe_router_w = copy_fresh(&lw.moe_router_w);
        lw.moe_router_b = copy_fresh(&lw.moe_router_b);
        lw.moe_gate_w = lw.moe_gate_w.copy_fresh(&zero);
        lw.moe_up_w = lw.moe_up_w.copy_fresh(&zero);
        lw.moe_down_w = lw.moe_down_w.copy_fresh(&zero);
        lw.moe_gate_b = copy_fresh(&lw.moe_gate_b);
        lw.moe_up_b = copy_fresh(&lw.moe_up_b);
        lw.moe_down_b = copy_fresh(&lw.moe_down_b);
    }
    crate::check_last_error().map_err(|e| format!("gpt_oss: loading the weights failed: {e}"))?;

    eprintln!("[GPT-OSS] load_model: all weights force-copied into fresh Metal buffers");

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
