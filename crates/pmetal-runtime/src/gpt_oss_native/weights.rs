//! Per-layer + full-model weight bundles and the checkpoint loader.
//!
//! The attention projections, the router, the embeddings and the experts load
//! dense or packed, as the checkpoint stores them (see
//! [`crate::native_checkpoint`]). The experts load from the release's MXFP4
//! (kept packed), transformers' dense layout, the stacked layout MLX
//! conversions ship, or that layout packed (MLX's `gpt-oss-*-MXFP4-Q8`
//! conversions: experts in `mxfp4`, attention, router and embeddings 8-bit
//! affine); see [`GptOssExperts`].

use crate::InlineArray;
use crate::native_checkpoint::Checkpoint;
use crate::native_weight::{EmbeddingWeight, LayerWeight, copy_fresh_arr};

use super::{AttentionLayerType, GptOssConfig, GptOssExperts};

/// GPT-OSS layer weights — attention + MoE.
pub(super) struct LayerWeights {
    // Layer norms
    pub(super) input_ln_w: InlineArray,
    pub(super) input_ln_eps: f32,
    pub(super) post_ln_w: InlineArray,
    pub(super) post_ln_eps: f32,

    // Attention projections
    pub(super) attn_q: LayerWeight,
    pub(super) attn_q_b: Option<InlineArray>, // [n_heads * head_dim]
    pub(super) attn_k: LayerWeight,
    pub(super) attn_k_b: Option<InlineArray>, // [n_kv_heads * head_dim]
    pub(super) attn_v: LayerWeight,
    pub(super) attn_v_b: Option<InlineArray>, // [n_kv_heads * head_dim]
    pub(super) attn_o: LayerWeight,
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
    /// The router, `hidden -> experts`, and its bias `[num_experts]`.
    pub(super) moe_router: LayerWeight,
    pub(super) moe_router_b: InlineArray,
    /// The routed experts, packed or dense.
    pub(super) moe_experts: GptOssExperts,
    pub(super) moe_top_k: i32,

    // GLU parameters
    pub(super) swiglu_alpha: f32,
    pub(super) swiglu_limit: f32,
}

impl LayerWeights {
    /// The four attention projections when all are dense: the compiled
    /// single-token graph takes dense matrices, so a packed layer decodes
    /// through the per-op path instead.
    pub(super) fn dense_attention(
        &self,
    ) -> Option<(&InlineArray, &InlineArray, &InlineArray, &InlineArray)> {
        match (&self.attn_q, &self.attn_k, &self.attn_v, &self.attn_o) {
            (
                LayerWeight::Dense(q),
                LayerWeight::Dense(k),
                LayerWeight::Dense(v),
                LayerWeight::Dense(o),
            ) => Some((q, k, v, o)),
            _ => None,
        }
    }
}

/// All GPT-OSS model weights.
pub struct NativeWeights {
    pub embed_w: EmbeddingWeight,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    /// None when `tie_word_embeddings = true`.
    pub lm_head_w: Option<LayerWeight>,
    pub tie_word_embeddings: bool,
    /// Per-layer weights — opaque to callers.
    pub(super) layers: Vec<LayerWeights>,
    /// The dtype the trunk computes in (11 = bfloat16, 1 = float16, 0 = float32).
    pub model_dtype: i32,
}

impl NativeWeights {
    /// Whether the routed experts are held packed (the release's MXFP4).
    pub fn experts_packed(&self) -> bool {
        self.layers.iter().all(|lw| lw.moe_experts.is_packed())
    }

    /// Whether every attention projection is held packed.
    pub fn attention_packed(&self) -> bool {
        self.layers.iter().all(|lw| lw.dense_attention().is_none())
    }
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

/// Load a GPT-OSS checkpoint directory.
pub fn load_model(
    model_dir: &std::path::Path,
    config: &GptOssConfig,
) -> Result<NativeWeights, String> {
    let mut ckpt = Checkpoint::open("gpt_oss", model_dir, |key| Some(key.to_string()))?;
    let model_dtype = ckpt.dtype();

    let embed_w = ckpt.embedding("model.embed_tokens")?;
    let final_norm_w = ckpt.take("model.norm.weight")?;
    let lm_head_w = if config.tie_word_embeddings {
        ckpt.take_opt("lm_head.weight");
        None
    } else {
        Some(ckpt.linear("lm_head")?)
    };

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

        let bias = |ckpt: &mut Checkpoint, name: &str| {
            if use_bias {
                ckpt.take_opt(&format!("{sa}.{name}.bias"))
            } else {
                None
            }
        };
        let (attn_q_b, attn_k_b, attn_v_b, attn_o_b) = (
            bias(&mut ckpt, "q_proj"),
            bias(&mut ckpt, "k_proj"),
            bias(&mut ckpt, "v_proj"),
            bias(&mut ckpt, "o_proj"),
        );
        let experts = GptOssExperts::from_tensors(&mut ckpt, &format!("{mlp}.experts"))?;

        layers.push(LayerWeights {
            input_ln_w: ckpt.take(&format!("{p}.input_layernorm.weight"))?,
            input_ln_eps: config.rms_norm_eps,
            post_ln_w: ckpt.take(&format!("{p}.post_attention_layernorm.weight"))?,
            post_ln_eps: config.rms_norm_eps,

            attn_q: ckpt.linear(&format!("{sa}.q_proj"))?,
            attn_q_b,
            attn_k: ckpt.linear(&format!("{sa}.k_proj"))?,
            attn_k_b,
            attn_v: ckpt.linear(&format!("{sa}.v_proj"))?,
            attn_v_b,
            attn_o: ckpt.linear(&format!("{sa}.o_proj"))?,
            attn_o_b,
            // MLX's SDPA takes sinks only in the output dtype.
            attn_sinks: ckpt.take(&format!("{sa}.sinks"))?.as_dtype(model_dtype),

            attn_n_heads: n_heads,
            attn_n_kv_heads: n_kv_heads,
            attn_head_dim: head_dim,
            attn_scale,
            attn_rotary: rotary.clone(),
            attn_is_sliding: is_sliding,
            attn_sliding_window: config.sliding_window,

            moe_router: ckpt.linear(&format!("{mlp}.router"))?,
            moe_router_b: ckpt.take(&format!("{mlp}.router.bias"))?,
            moe_experts: experts,
            moe_top_k: top_k,

            swiglu_alpha: config.swiglu_alpha,
            swiglu_limit: config.swiglu_limit,
        });

        if li == 0 {
            tracing::info!(
                "[GPT-OSS] layer 0: type={:?} n_heads={n_heads} n_kv={n_kv_heads} \
                 head_dim={head_dim} experts={n_experts} top_k={top_k}",
                layer_type,
            );
        }
    }

    // A tensor of a layer nothing above read belongs to a feature this
    // engine does not run; loading past it would run a different model.
    let unread: Vec<&str> = ckpt
        .unclaimed()
        .into_iter()
        .filter(|k| k.starts_with("model.layers."))
        .collect();
    if !unread.is_empty() {
        return Err(format!(
            "gpt_oss: {} checkpoint tensors in the layers are not part of the model this \
             engine runs, first: {:?}",
            unread.len(),
            &unread[..unread.len().min(5)]
        ));
    }

    // Force every weight into a fresh, contiguous Metal buffer (the expert
    // splits above are strided views of the fused tensors).
    let zero = InlineArray::scalar_with_dtype(0.0, model_dtype);
    let fresh = |w: &InlineArray| copy_fresh_arr(w, &zero);
    let embed_w = match embed_w {
        EmbeddingWeight::Dense(w) => EmbeddingWeight::Dense(fresh(&w)),
        packed @ EmbeddingWeight::Quantized { .. } => packed,
    };
    let final_norm_w = fresh(&final_norm_w);
    let lm_head_w = lm_head_w.map(|w| w.copy_fresh(&zero));
    for lw in &mut layers {
        lw.input_ln_w = fresh(&lw.input_ln_w);
        lw.post_ln_w = fresh(&lw.post_ln_w);
        for w in [
            &mut lw.attn_q,
            &mut lw.attn_k,
            &mut lw.attn_v,
            &mut lw.attn_o,
            &mut lw.moe_router,
        ] {
            *w = w.copy_fresh(&zero);
        }
        for b in [
            &mut lw.attn_q_b,
            &mut lw.attn_k_b,
            &mut lw.attn_v_b,
            &mut lw.attn_o_b,
        ] {
            *b = b.as_ref().map(fresh);
        }
        lw.attn_sinks = fresh(&lw.attn_sinks);
        lw.moe_router_b = fresh(&lw.moe_router_b);
        lw.moe_experts = lw.moe_experts.copy_fresh(&zero);
    }
    crate::check_unobserved_error()
        .map_err(|e| format!("gpt_oss: loading the weights failed: {e}"))?;

    tracing::info!("[GPT-OSS] load_model: all weights force-copied into fresh Metal buffers");

    Ok(NativeWeights {
        embed_w,
        final_norm_w,
        final_norm_eps: config.rms_norm_eps,
        lm_head_w,
        tie_word_embeddings: config.tie_word_embeddings,
        layers,
        model_dtype,
    })
}
