//! Per-layer + full-model weight bundles and the checkpoint loader.
//!
//! Every projection, the experts and the embeddings load dense or packed, as
//! the checkpoint stores them (see [`crate::native_checkpoint`]). Keys are
//! read in the text model's namespace: vision and projector tensors are
//! dropped and the multimodal prefixes (`language_model.model.X`,
//! `model.language_model.X`) fold onto `model.X`.
//!
//! Three expert layouts load, each to `[E, in, out]` (what `gather_mm` takes):
//!
//! * transformers' (`Llama4TextExperts`, every release): `experts.gate_up_proj`
//!   `[E, hidden, 2 * expert_hidden]`, gate then up along the last axis, and
//!   `experts.down_proj` `[E, expert_hidden, hidden]`, both applied as `x @ W`,
//! * the stacked per-projection layout of a Linear (`experts.{gate,up,down}_proj.weight`,
//!   `[E, out, in]`), dense or packed, which is what MLX conversions ship.

use crate::InlineArray;
use crate::native_checkpoint::Checkpoint;
use crate::native_weight::{EmbeddingWeight, LayerWeight, copy_fresh_arr};

use super::Llama4Config;

/// A gated MLP: the dense layers' and the shared expert's.
pub(super) struct Mlp {
    pub(super) gate: LayerWeight,
    pub(super) up: LayerWeight,
    pub(super) down: LayerWeight,
}

/// MoE layer: the router, the routed experts and the shared expert.
pub(super) struct MoeWeights {
    /// Routed experts, `[E, in, out]` dense or packed as stored.
    pub(super) experts_gate: LayerWeight,
    pub(super) experts_up: LayerWeight,
    pub(super) experts_down: LayerWeight,
    /// `hidden -> experts`.
    pub(super) router: LayerWeight,
    pub(super) top_k: i32,
    pub(super) shared: Mlp,
}

// A layer holds exactly one of these, loaded once: the size difference is a
// few KB per layer, not worth an indirection on the hot path.
#[allow(clippy::large_enum_variant)]
pub(super) enum Ffn {
    Dense(Mlp),
    Moe(MoeWeights),
}

pub(super) struct LayerWeights {
    // ── iRoPE meta ──
    pub(super) use_rope: bool, // false for NoPE (global) layers

    // ── Layer norms ──
    pub(super) input_ln_w: InlineArray,
    pub(super) post_ln_w: InlineArray,
    pub(super) norm_eps: f32,

    // ── Attention projections ──
    pub(super) attn_q: LayerWeight,
    pub(super) attn_k: LayerWeight,
    pub(super) attn_v: LayerWeight,
    pub(super) attn_o: LayerWeight,

    // Optional bias (when attention_bias=true)
    pub(super) attn_q_b: Option<InlineArray>,
    pub(super) attn_k_b: Option<InlineArray>,
    pub(super) attn_v_b: Option<InlineArray>,
    pub(super) attn_o_b: Option<InlineArray>,

    // QK-norm (only on RoPE layers — same flag for both Q and K, no learned weight)
    pub(super) attn_qk_norm: bool, // true → weightless rms_norm(eps = norm_eps) on Q and K

    // Attention shape config (stored per-layer for self-containedness)
    pub(super) n_heads: i32,
    pub(super) n_kv_heads: i32,
    pub(super) head_dim: i32,
    pub(super) attn_scale: f32,
    /// Interleaved RoPE of the RoPE layers, Llama 3 bands included.
    pub(super) rotary: crate::rope::RotaryEmbedding,

    // Temperature tuning for NoPE layers
    pub(super) attn_temperature_tuning: i32,
    pub(super) floor_scale: i32,
    pub(super) layer_attn_scale: f32,

    pub(super) ffn: Ffn,
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

/// All model weights.
pub struct NativeWeights {
    pub embed_w: EmbeddingWeight,
    pub final_norm_w: InlineArray, // [hidden]
    pub final_norm_eps: f32,
    /// `None` when `tie_word_embeddings = true`.
    pub lm_head_w: Option<LayerWeight>,
    pub tie_word_embeddings: bool,
    /// Per-layer weights — only accessed via [`super::forward_step`].
    pub(super) layers: Vec<LayerWeights>,
    /// The dtype the trunk computes in (11 = bfloat16, 1 = float16, 0 = float32).
    pub model_dtype: i32,
    /// Chunk size for local attention (from config).
    pub attention_chunk_size: i32,
}

impl NativeWeights {
    /// Whether any projection or expert tensor is held packed.
    pub fn is_packed(&self) -> bool {
        self.layers.iter().any(|lw| {
            lw.dense_attention().is_none()
                || match &lw.ffn {
                    Ffn::Moe(m) => !m.experts_gate.is_dense(),
                    Ffn::Dense(m) => !m.gate.is_dense(),
                }
        })
    }
}

impl std::fmt::Debug for NativeWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeWeights")
            .field("layers", &self.layers.len())
            .field("tie_word_embeddings", &self.tie_word_embeddings)
            .field("model_dtype", &self.model_dtype)
            .field("attention_chunk_size", &self.attention_chunk_size)
            .finish()
    }
}

/// A checkpoint key in the text model's namespace, or `None` for what the
/// engine never loads (the vision tower and its projector).
fn text_key(key: &str) -> Option<String> {
    if key.contains("vision_model") || key.contains("multi_modal_projector") {
        return None;
    }
    // `language_model.model.X` and `language_model.lm_head.X` fold onto `X`'s
    // text-only names; `model.language_model.X` (transformers 5) onto `model.X`.
    if let Some(rest) = key.strip_prefix("language_model.") {
        return Some(rest.to_string());
    }
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        return Some(format!("model.{rest}"));
    }
    Some(key.to_string())
}

fn load_mlp(ckpt: &mut Checkpoint, prefix: &str) -> Result<Mlp, String> {
    Ok(Mlp {
        gate: ckpt.linear(&format!("{prefix}.gate_proj"))?,
        up: ckpt.linear(&format!("{prefix}.up_proj"))?,
        down: ckpt.linear(&format!("{prefix}.down_proj"))?,
    })
}

/// The routed experts under `{ff}.experts`, from either layout in the module
/// docs.
fn load_experts(
    ckpt: &mut Checkpoint,
    ff: &str,
    expert_hidden: i32,
) -> Result<(LayerWeight, LayerWeight, LayerWeight), String> {
    let experts = format!("{ff}.experts");
    // Already `[E, in, out]`: no transpose, and gate comes first.
    if let Some(gate_up) = ckpt.take_opt(&format!("{experts}.gate_up_proj")) {
        let mut halves = gate_up.split(&[expert_hidden], -1);
        let up = halves.pop().expect("split yields two parts");
        let gate = halves.pop().expect("split yields two parts");
        let down = ckpt.take(&format!("{experts}.down_proj"))?;
        return Ok((
            LayerWeight::Dense(gate),
            LayerWeight::Dense(up),
            LayerWeight::Dense(down),
        ));
    }
    Ok((
        ckpt.stacked(&format!("{experts}.gate_proj"))?,
        ckpt.stacked(&format!("{experts}.up_proj"))?,
        ckpt.stacked(&format!("{experts}.down_proj"))?,
    ))
}

/// Load a Llama 4 checkpoint directory.
pub fn load_model(
    model_dir: &std::path::Path,
    config: &Llama4Config,
) -> Result<NativeWeights, String> {
    let tc = config.text();
    let mut ckpt = Checkpoint::open("llama4", model_dir, text_key)?;
    let model_dtype = ckpt.dtype();

    let embed_w = ckpt.embedding("model.embed_tokens")?;
    let final_norm_w = ckpt.take("model.norm.weight")?;
    let lm_head_w = if tc.tie_word_embeddings {
        ckpt.take_opt("lm_head.weight");
        None
    } else {
        Some(ckpt.linear("lm_head")?)
    };

    let n_heads = tc.num_attention_heads;
    let n_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let attn_scale = (head_dim as f32).powi(-1).sqrt(); // 1/sqrt(head_dim)
    // The RoPE layers' rotation: rope_theta (500_000) with the Llama 3 bands
    // Scout's `rope_scaling` gives. The bands divide the low frequencies at
    // every position, not only past the chunk, so they are not optional.
    let rotary = config.rotary()?;

    let mut layers = Vec::with_capacity(tc.num_hidden_layers as usize);
    for li in 0..tc.num_hidden_layers as usize {
        let p = format!("model.layers.{li}");
        let sa = format!("{p}.self_attn");
        let ff = format!("{p}.feed_forward");
        let use_rope = config.use_rope(li);

        let ffn = if config.is_moe_layer(li) {
            let (experts_gate, experts_up, experts_down) =
                load_experts(&mut ckpt, &ff, tc.intermediate_size)?;
            Ffn::Moe(MoeWeights {
                experts_gate,
                experts_up,
                experts_down,
                router: ckpt.linear(&format!("{ff}.router"))?,
                top_k: tc.num_experts_per_tok,
                shared: load_mlp(&mut ckpt, &format!("{ff}.shared_expert"))?,
            })
        } else {
            Ffn::Dense(load_mlp(&mut ckpt, &ff)?)
        };

        layers.push(LayerWeights {
            use_rope,
            input_ln_w: ckpt.take(&format!("{p}.input_layernorm.weight"))?,
            post_ln_w: ckpt.take(&format!("{p}.post_attention_layernorm.weight"))?,
            norm_eps: tc.rms_norm_eps,
            attn_q: ckpt.linear(&format!("{sa}.q_proj"))?,
            attn_k: ckpt.linear(&format!("{sa}.k_proj"))?,
            attn_v: ckpt.linear(&format!("{sa}.v_proj"))?,
            attn_o: ckpt.linear(&format!("{sa}.o_proj"))?,
            attn_q_b: ckpt.take_opt(&format!("{sa}.q_proj.bias")),
            attn_k_b: ckpt.take_opt(&format!("{sa}.k_proj.bias")),
            attn_v_b: ckpt.take_opt(&format!("{sa}.v_proj.bias")),
            attn_o_b: ckpt.take_opt(&format!("{sa}.o_proj.bias")),
            // QK-norm only on RoPE layers (and only when use_qk_norm=true in config)
            attn_qk_norm: use_rope && tc.use_qk_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rotary: rotary.clone(),
            attn_temperature_tuning: tc.attn_temperature_tuning,
            floor_scale: tc.floor_scale,
            layer_attn_scale: tc.attn_scale,
            ffn,
        });
    }

    // A tensor of a text layer nothing above read belongs to a feature this
    // engine does not run; loading past it would run a different model.
    let unread: Vec<&str> = ckpt
        .unclaimed()
        .into_iter()
        .filter(|k| k.starts_with("model.layers."))
        .collect();
    if !unread.is_empty() {
        return Err(format!(
            "llama4: {} checkpoint tensors in the text layers are not part of the model this \
             engine runs, first: {:?}",
            unread.len(),
            &unread[..unread.len().min(5)]
        ));
    }

    // Force every weight into a fresh Metal buffer (the expert split above is
    // a strided view of the fused tensor).
    let zero = InlineArray::scalar_with_dtype(0.0, model_dtype);
    let fresh = |a: &InlineArray| copy_fresh_arr(a, &zero);
    let opt = |a: &mut Option<InlineArray>| *a = a.as_ref().map(fresh);
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
        ] {
            *w = w.copy_fresh(&zero);
        }
        for b in [
            &mut lw.attn_q_b,
            &mut lw.attn_k_b,
            &mut lw.attn_v_b,
            &mut lw.attn_o_b,
        ] {
            opt(b);
        }
        let mlp = |m: &mut Mlp| {
            m.gate = m.gate.copy_fresh(&zero);
            m.up = m.up.copy_fresh(&zero);
            m.down = m.down.copy_fresh(&zero);
        };
        match &mut lw.ffn {
            Ffn::Dense(m) => mlp(m),
            Ffn::Moe(moe) => {
                moe.experts_gate = moe.experts_gate.copy_fresh(&zero);
                moe.experts_up = moe.experts_up.copy_fresh(&zero);
                moe.experts_down = moe.experts_down.copy_fresh(&zero);
                moe.router = moe.router.copy_fresh(&zero);
                mlp(&mut moe.shared);
            }
        }
    }
    crate::check_unobserved_error()
        .map_err(|e| format!("llama4: loading the weights failed: {e}"))?;

    eprintln!(
        "[LLAMA4_NATIVE] load_model: {} layers, dtype={}, chunk_size={}",
        layers.len(),
        model_dtype,
        tc.attention_chunk_size
    );

    Ok(NativeWeights {
        embed_w,
        final_norm_w,
        final_norm_eps: tc.rms_norm_eps,
        lm_head_w,
        tie_word_embeddings: tc.tie_word_embeddings,
        layers,
        model_dtype,
        attention_chunk_size: tc.attention_chunk_size,
    })
}
