//! Per-layer + full-model weight bundles and the checkpoint loader for
//! DeepSeek V2 / V3 / R1.
//!
//! Every projection, the routed experts and the embeddings load dense or
//! packed, as the checkpoint stores them (see [`crate::native_checkpoint`]).
//! Three expert layouts load:
//!
//! * the releases': every expert its own `mlp.experts.E.{gate,up,down}_proj`,
//! * transformers' 5.x fused `mlp.experts.gate_up_proj` `[E, 2I, H]` (gate
//!   rows first) and `mlp.experts.down_proj` `[E, H, I]`,
//! * the stacked layout MLX conversions ship, `mlp.switch_mlp.*_proj`.
//!
//! `kv_b_proj` is split per head into the two matrices MLA's decode absorbs
//! (`embed_q`, `unembed_out`); a packed one is dequantized first, as MLX's own
//! conversion does.

use crate::InlineArray;
use crate::dtype;
use crate::native_checkpoint::Checkpoint;
use crate::native_weight::{EmbeddingWeight, LayerWeight, copy_fresh_arr};

use super::DeepSeekConfig;
use super::moe::{DeepseekRouting, Router};

/// One gated MLP: the dense layers' and the shared expert's.
pub(super) struct Mlp {
    pub(super) gate: LayerWeight,
    pub(super) up: LayerWeight,
    pub(super) down: LayerWeight,
}

/// A MoE layer: the router, the routed experts and the shared expert.
pub(super) struct MoeWeights {
    pub(super) router: Router,
    /// Routed experts, `[E, in, out]` dense or packed as stored.
    pub(super) gate: LayerWeight,
    pub(super) up: LayerWeight,
    pub(super) down: LayerWeight,
    pub(super) shared: Option<Mlp>,
}

// A layer holds exactly one of these, loaded once: the size difference is a
// few KB per layer, not worth an indirection on the hot path.
#[allow(clippy::large_enum_variant)]
pub(super) enum Ffn {
    Dense(Mlp),
    Moe(MoeWeights),
}

/// How a layer computes Q.
#[allow(clippy::large_enum_variant)]
pub(super) enum QProj {
    /// `x -> q_a_proj -> rms_norm -> q_b_proj`.
    LowRank {
        a: LayerWeight,
        a_norm: InlineArray,
        b: LayerWeight,
    },
    /// `x -> q_proj` (V2-Lite).
    Direct(LayerWeight),
}

pub(super) struct LayerWeights {
    pub(super) input_ln_w: InlineArray,
    pub(super) post_ln_w: InlineArray,
    pub(super) norm_eps: f32,

    // ── MLA attention ─────────────────────────────────────────────────────
    pub(super) q: QProj,
    /// `x -> [kv_lora_rank + qk_rope_head_dim]`: the latent and the shared
    /// rotary key.
    pub(super) kv_a: LayerWeight,
    pub(super) kv_a_norm_w: InlineArray,
    /// `W_UK`, `[n_heads, qk_nope_head_dim, kv_lora_rank]`: decode multiplies
    /// the query's nope part by it to attend in the latent space; prefill
    /// expands the keys with its transpose.
    pub(super) w_uk: InlineArray,
    /// `W_UV^T`, `[n_heads, kv_lora_rank, v_head_dim]`: the latent's values
    /// per head.
    pub(super) w_uv_t: InlineArray,
    pub(super) o_proj: LayerWeight,

    // Attention scalars
    pub(super) n_heads: i32,
    pub(super) q_head_dim: i32,
    pub(super) qk_nope_head_dim: i32,
    pub(super) qk_rope_head_dim: i32,
    pub(super) v_head_dim: i32,
    pub(super) kv_lora_rank: i32,
    pub(super) scale: f32,
    /// RoPE of `q_pe` / `k_pe`, YaRN included.
    pub(super) rotary: crate::rope::RotaryEmbedding,

    pub(super) ffn: Ffn,
}

/// All model weights.
pub struct NativeWeights {
    pub embed_w: EmbeddingWeight,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    /// `None` when `tie_word_embeddings = true`.
    pub lm_head_w: Option<LayerWeight>,
    pub tie_word_embeddings: bool,
    pub(super) layers: Vec<LayerWeights>,
    /// The dtype the trunk computes in (11 = bfloat16, 1 = float16, 0 = float32).
    pub model_dtype: i32,
}

impl NativeWeights {
    /// Whether any layer's routed experts are held packed.
    pub fn experts_packed(&self) -> bool {
        self.layers.iter().any(|lw| match &lw.ffn {
            Ffn::Moe(moe) => !(moe.gate.is_dense() && moe.up.is_dense() && moe.down.is_dense()),
            Ffn::Dense(_) => false,
        })
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

/// A checkpoint key as the engine reads it: `None` for what it never loads,
/// that is the rotary table and every layer past the trunk (V3's
/// multi-token-prediction layer is layer 61 of 61).
fn trunk_key(key: &str, n_layers: usize) -> Option<String> {
    if key.contains("rotary_emb.inv_freq") {
        return None;
    }
    if let Some(rest) = key.strip_prefix("model.layers.")
        && let Some((index, _)) = rest.split_once('.')
        && index.parse::<usize>().is_ok_and(|i| i >= n_layers)
    {
        return None;
    }
    Some(key.to_string())
}

/// Dequantize every FP8 (E4M3) weight that has a `{weight}_scale_inv` beside
/// it, block-wise, to the model dtype.
fn dequantize_fp8(ckpt: &mut Checkpoint, dtype: i32) -> Result<(), String> {
    let scale_keys: Vec<String> = ckpt
        .keys()
        .filter(|k| k.ends_with("_scale_inv"))
        .map(str::to_string)
        .collect();
    for scale_key in scale_keys {
        let weight_key = scale_key.replace("_scale_inv", "");
        let (Some(scale_inv), Some(weight)) =
            (ckpt.take_opt(&scale_key), ckpt.take_opt(&weight_key))
        else {
            continue;
        };
        let dense =
            crate::native_loader::dequantize_fp8_e4m3_scaled_weight(&weight, &scale_inv, dtype)
                .map_err(|e| format!("deepseek: {weight_key}: {e}"))?;
        ckpt.put(weight_key, dense);
    }
    Ok(())
}

/// `[E, out, in]` expert projections of one layer, from whichever of the
/// three layouts the checkpoint holds.
fn load_experts(
    ckpt: &mut Checkpoint,
    mlp: &str,
    config: &DeepSeekConfig,
) -> Result<(LayerWeight, LayerWeight, LayerWeight), String> {
    let experts = config.n_routed_experts.unwrap_or(0) as usize;
    let module = |proj: &str| format!("{mlp}.switch_mlp.{proj}");

    // MLX's stacked layout: each projection a (possibly packed) module.
    if ckpt.contains(&format!("{}.weight", module("gate_proj"))) {
        return Ok((
            ckpt.stacked(&module("gate_proj"))?,
            ckpt.stacked(&module("up_proj"))?,
            ckpt.stacked(&module("down_proj"))?,
        ));
    }

    // transformers 5.x: gate and up fused along the output rows.
    let fused = format!("{mlp}.experts.gate_up_proj");
    if let Some(gate_up) = ckpt.take_opt(&fused) {
        let inter = config
            .moe_intermediate_size
            .unwrap_or(config.intermediate_size);
        let mut halves = gate_up.split(&[inter], 1);
        let up = halves.pop().expect("split yields two parts");
        let gate = halves.pop().expect("split yields two parts");
        let down = ckpt.take(&format!("{mlp}.experts.down_proj"))?;
        let dense = |w| ckpt.stacked_from("", w, None, None);
        return Ok((dense(gate), dense(up), dense(down)));
    }

    // The releases': one module per expert, stacked here.
    let mut stacked = |proj: &str| -> Result<LayerWeight, String> {
        let prefix = format!("{mlp}.experts");
        let part = |ckpt: &mut Checkpoint, what: &str| {
            ckpt.stack_experts(&prefix, experts, &format!("{proj}.{what}"))
        };
        let weight = part(ckpt, "weight")?.ok_or_else(|| {
            format!(
                "deepseek: no routed experts under {prefix} (expected {prefix}.0.{proj}.weight)"
            )
        })?;
        let (scales, biases) = (part(ckpt, "scales")?, part(ckpt, "biases")?);
        Ok(ckpt.stacked_from(&module(proj), weight, scales, biases))
    };
    Ok((
        stacked("gate_proj")?,
        stacked("up_proj")?,
        stacked("down_proj")?,
    ))
}

fn load_mlp(ckpt: &mut Checkpoint, prefix: &str) -> Result<Mlp, String> {
    Ok(Mlp {
        gate: ckpt.linear(&format!("{prefix}.gate_proj"))?,
        up: ckpt.linear(&format!("{prefix}.up_proj"))?,
        down: ckpt.linear(&format!("{prefix}.down_proj"))?,
    })
}

/// The router's gate as `[hidden, experts]` `f32`, dequantized if packed
/// (MLX conversions quantize it at 8 bits).
fn load_gate(ckpt: &mut Checkpoint, module: &str) -> Result<InlineArray, String> {
    let weight = ckpt.take(&format!("{module}.weight"))?;
    let scales = ckpt.take_opt(&format!("{module}.scales"));
    let biases = ckpt.take_opt(&format!("{module}.biases"));
    let dense = match scales {
        Some(scales) => {
            let p = ckpt.params_for(module);
            weight.dequantize_mode(&scales, biases.as_ref(), p.group_size, p.bits, p.mode)
        }
        None => weight,
    };
    Ok(dense.as_dtype(dtype::F32).t())
}

/// `kv_b_proj` of one layer split per head into `(W_UK, W_UV^T)`; see
/// [`LayerWeights`].
fn load_kv_b(
    ckpt: &mut Checkpoint,
    sa: &str,
    config: &DeepSeekConfig,
    model_dtype: i32,
) -> Result<(InlineArray, InlineArray), String> {
    let module = format!("{sa}.kv_b_proj");
    let weight = ckpt.take(&format!("{module}.weight"))?;
    let scales = ckpt.take_opt(&format!("{module}.scales"));
    let biases = ckpt.take_opt(&format!("{module}.biases"));
    let weight = match scales {
        Some(scales) => {
            let p = ckpt.params_for(&module);
            weight
                .dequantize_mode(&scales, biases.as_ref(), p.group_size, p.bits, p.mode)
                .as_dtype(model_dtype)
        }
        None => weight,
    };
    // [n_heads * (nope + v), lora] -> [n_heads, nope + v, lora].
    let per_head = weight.reshape(&[
        config.num_attention_heads,
        config.qk_nope_head_dim + config.v_head_dim,
        config.kv_lora_rank,
    ]);
    let mut parts = per_head.split(&[config.qk_nope_head_dim], 1);
    let w_uv = parts.pop().expect("split yields two parts"); // [H, v, lora]
    let w_uk = parts.pop().expect("split yields two parts"); // [H, nope, lora]
    Ok((w_uk, w_uv.transpose_axes(&[0, 2, 1])))
}

/// Load a DeepSeek checkpoint directory.
pub fn load_model(
    model_dir: &std::path::Path,
    config: &DeepSeekConfig,
) -> Result<NativeWeights, String> {
    config.validate()?;
    let n_layers = config.num_hidden_layers as usize;
    let mut ckpt = Checkpoint::open("deepseek", model_dir, |key| trunk_key(key, n_layers))?;
    let model_dtype = ckpt.dtype();
    dequantize_fp8(&mut ckpt, model_dtype)?;

    let embed_w = ckpt.embedding("model.embed_tokens")?;
    let final_norm_w = ckpt.take("model.norm.weight")?;
    let lm_head_w = if config.tie_word_embeddings {
        ckpt.take_opt("lm_head.weight");
        None
    } else {
        Some(ckpt.linear("lm_head")?)
    };

    let rotary = config.rotary()?;
    let scale = config.attention_scale(&rotary);
    let n_routed = config.n_routed_experts.unwrap_or(0);

    let mut layers = Vec::with_capacity(n_layers);
    for li in 0..n_layers {
        let p = format!("model.layers.{li}");
        let sa = format!("{p}.self_attn");

        let q = if config.q_lora_rank.is_some() {
            QProj::LowRank {
                a: ckpt.linear(&format!("{sa}.q_a_proj"))?,
                a_norm: ckpt.take(&format!("{sa}.q_a_layernorm.weight"))?,
                b: ckpt.linear(&format!("{sa}.q_b_proj"))?,
            }
        } else {
            QProj::Direct(ckpt.linear(&format!("{sa}.q_proj"))?)
        };
        let (w_uk, w_uv_t) = load_kv_b(&mut ckpt, &sa, config, model_dtype)?;

        let mlp = format!("{p}.mlp");
        let ffn = if config.is_moe_layer(li) {
            let (gate, up, down) = load_experts(&mut ckpt, &mlp, config)?;
            let method = config.topk();
            let bias = ckpt.take_opt(&format!("{mlp}.gate.e_score_correction_bias"));
            if bias.is_none() && method == super::TopkMethod::NoauxTc {
                return Err(format!(
                    "deepseek: missing weight key: {mlp}.gate.e_score_correction_bias \
                     (topk_method noaux_tc needs it)"
                ));
            }
            let shared = if config.n_shared_experts.unwrap_or(0) > 0 {
                Some(load_mlp(&mut ckpt, &format!("{mlp}.shared_experts"))?)
            } else {
                None
            };
            Ffn::Moe(MoeWeights {
                router: Router {
                    weight: load_gate(&mut ckpt, &format!("{mlp}.gate"))?,
                    bias: bias.map(|b| b.as_dtype(dtype::F32)),
                    routing: DeepseekRouting {
                        scoring: config.scoring(),
                        method,
                        n_group: config.n_group(),
                        topk_group: config.topk_group(),
                        top_k: config.num_experts_per_tok,
                        routed_scaling_factor: config.routed_scaling_factor,
                        norm_topk_prob: config.norm_topk_prob,
                    },
                },
                gate,
                up,
                down,
                shared,
            })
        } else {
            Ffn::Dense(load_mlp(&mut ckpt, &mlp)?)
        };

        layers.push(LayerWeights {
            input_ln_w: ckpt.take(&format!("{p}.input_layernorm.weight"))?,
            post_ln_w: ckpt.take(&format!("{p}.post_attention_layernorm.weight"))?,
            norm_eps: config.rms_norm_eps,
            q,
            kv_a: ckpt.linear(&format!("{sa}.kv_a_proj_with_mqa"))?,
            kv_a_norm_w: ckpt.take(&format!("{sa}.kv_a_layernorm.weight"))?,
            w_uk,
            w_uv_t,
            o_proj: ckpt.linear(&format!("{sa}.o_proj"))?,
            n_heads: config.num_attention_heads,
            q_head_dim: config.q_head_dim(),
            qk_nope_head_dim: config.qk_nope_head_dim,
            qk_rope_head_dim: config.qk_rope_head_dim,
            v_head_dim: config.v_head_dim,
            kv_lora_rank: config.kv_lora_rank,
            scale,
            rotary: rotary.clone(),
            ffn,
        });
    }

    // A tensor of a trunk layer nothing above read belongs to a feature this
    // engine does not run; loading past it would run a different model.
    let unread: Vec<&str> = ckpt
        .unclaimed()
        .into_iter()
        .filter(|k| k.starts_with("model.layers."))
        .collect();
    if !unread.is_empty() {
        return Err(format!(
            "deepseek: {} checkpoint tensors in the trunk are not part of the model this \
             engine runs, first: {:?}",
            unread.len(),
            &unread[..unread.len().min(5)]
        ));
    }

    // Force every weight into a fresh Metal buffer (the splits above are
    // strided views of the checkpoint's tensors).
    let zero = InlineArray::scalar_with_dtype(0.0, model_dtype);
    let fresh = |a: &InlineArray| copy_fresh_arr(a, &zero);
    let embed_w = match embed_w {
        EmbeddingWeight::Dense(w) => EmbeddingWeight::Dense(fresh(&w)),
        packed @ EmbeddingWeight::Quantized { .. } => packed,
    };
    let final_norm_w = fresh(&final_norm_w);
    let lm_head_w = lm_head_w.map(|w| w.copy_fresh(&zero));
    for lw in &mut layers {
        lw.make_fresh(&zero);
    }
    crate::check_unobserved_error()
        .map_err(|e| format!("deepseek: loading the weights failed: {e}"))?;

    tracing::info!(
        "[DEEPSEEK] load_model: {} layers ({} routed experts), dtype={}",
        layers.len(),
        n_routed,
        model_dtype
    );

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

impl LayerWeights {
    /// Re-materialise every array into a fresh buffer.
    fn make_fresh(&mut self, zero: &InlineArray) {
        let fresh = |a: &InlineArray| copy_fresh_arr(a, zero);
        let mlp = |m: &mut Mlp| {
            m.gate = m.gate.copy_fresh(zero);
            m.up = m.up.copy_fresh(zero);
            m.down = m.down.copy_fresh(zero);
        };
        self.input_ln_w = fresh(&self.input_ln_w);
        self.post_ln_w = fresh(&self.post_ln_w);
        match &mut self.q {
            QProj::LowRank { a, a_norm, b } => {
                *a = a.copy_fresh(zero);
                *a_norm = fresh(a_norm);
                *b = b.copy_fresh(zero);
            }
            QProj::Direct(w) => *w = w.copy_fresh(zero),
        }
        self.kv_a = self.kv_a.copy_fresh(zero);
        self.kv_a_norm_w = fresh(&self.kv_a_norm_w);
        self.w_uk = fresh(&self.w_uk);
        self.w_uv_t = fresh(&self.w_uv_t);
        self.o_proj = self.o_proj.copy_fresh(zero);
        match &mut self.ffn {
            Ffn::Dense(m) => mlp(m),
            Ffn::Moe(moe) => {
                moe.router.weight = fresh(&moe.router.weight);
                moe.router.bias = moe.router.bias.as_ref().map(fresh);
                moe.gate = moe.gate.copy_fresh(zero);
                moe.up = moe.up.copy_fresh(zero);
                moe.down = moe.down.copy_fresh(zero);
                if let Some(shared) = &mut moe.shared {
                    mlp(shared);
                }
            }
        }
    }
}
