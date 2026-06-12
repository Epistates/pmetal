//! DiffusionGemma — block-autoregressive discrete-diffusion LM on a Gemma 4
//! MoE trunk (`google/diffusiongemma-26B-A4B-it`, `model_type:
//! diffusion_gemma`).
//!
//! DiffusionGemma is an encoder–decoder model. The **encoder** consumes the
//! prompt causally (writing a KV cache) and the **decoder** denoises a fixed
//! `canvas_length` block bidirectionally, reading — but never writing — the
//! encoder KV cache. Both towers share the same Gemma 4 MoE layer stack.
//!
//! This module ports the **text path**. Its trunk reuses the parity-verified
//! Gemma 4 building blocks wholesale ([`Gemma4Attention`], [`Gemma4Mlp`],
//! [`Gemma4RmsNorm`], the proportional-RoPE helpers) and adds the pieces
//! Gemma 4's text tower omits:
//!
//! * **Per-layer MoE block** that runs *in parallel* with the dense MLP (both
//!   branches feed the same residual): a 7-RMSNorm layer where the router
//!   reads the **raw** post-attention residual and the experts read a
//!   separately-normed copy. See [`DiffusionGemmaTextLayer`].
//! * **Router** ([`DiffusionGemmaRouter`]): weight-less RMSNorm → per-channel
//!   `scale` → `hidden^-0.5` → linear → fp32 softmax over experts → top-k →
//!   renormalise → per-expert learned scale.
//! * **Grouped experts** ([`DiffusionGemmaExperts`]): fused
//!   `gate_up_proj [E, 2·I, H]` + `down_proj [E, H, I]` SwiGLU (gelu-tanh).
//! * **Self-conditioning** ([`DiffusionGemmaSelfConditioning`]): folds the
//!   previous denoising step's soft embeddings into the decoder input.
//!
//! Per-layer attention geometry (sliding vs. full head_dim / KV heads /
//! proportional RoPE / K=V on full layers) is delegated to [`Gemma4Config`]
//! via [`DiffusionGemmaTextConfig::geometry`] — DiffusionGemma's full layers
//! are exactly Gemma 4's `attention_k_eq_v` layers.
//!
//! # Scope
//!
//! Landed: config, the leaf MoE blocks, the 7-norm layer, and the **encoder**
//! model (causal, auto-masked, KV-collecting). The **decoder** model
//! (bidirectional canvas, read-only encoder-KV concat via
//! [`Gemma4Attention::forward_with_encoder_kv`], self-conditioning) and the
//! discrete-diffusion generation engine are built in later phases against the
//! transformers parity oracle.

use pmetal_bridge::compat::{Array, Exception, Module, Param, nn, ops};
use pmetal_bridge::impl_module_params;
use serde::{Deserialize, Serialize};

use super::gemma4::{
    Gemma4Attention, Gemma4Config, Gemma4Mlp, Gemma4RmsNorm, Gemma4RopeConfig,
    Gemma4RopeLayerConfig, rms_norm_noscale,
};

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

fn default_model_type() -> String {
    "diffusion_gemma_text".to_string()
}
fn default_vocab_size() -> i32 {
    262_144
}
fn default_hidden_size() -> i32 {
    2304
}
fn default_intermediate_size() -> i32 {
    9216
}
fn default_num_hidden_layers() -> i32 {
    30
}
fn default_num_attention_heads() -> i32 {
    8
}
fn default_num_key_value_heads() -> i32 {
    4
}
fn default_head_dim() -> i32 {
    256
}
fn default_global_head_dim() -> i32 {
    512
}
fn default_max_position_embeddings() -> i32 {
    131_072
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_sliding_window() -> i32 {
    512
}
fn default_sliding_window_pattern() -> i32 {
    6
}
fn default_final_logit_softcapping() -> Option<f32> {
    Some(30.0)
}
fn default_hidden_activation() -> String {
    "gelu_pytorch_tanh".to_string()
}
fn default_num_experts() -> i32 {
    128
}
fn default_top_k_experts() -> i32 {
    8
}
fn default_moe_intermediate_size() -> i32 {
    704
}
fn default_canvas_length() -> i32 {
    256
}
fn default_tie_word_embeddings() -> bool {
    true
}

/// DiffusionGemma text-tower configuration (`DiffusionGemmaTextConfig`).
///
/// Mirrors `Gemma4TextConfig` (per-layer-type attention geometry +
/// proportional RoPE) and adds the always-on MoE fields. The vision tower
/// and audio/PLE/double-wide-MLP knobs of Gemma 4 are intentionally absent —
/// DiffusionGemma drops them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionGemmaTextConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Head dim for full (global) attention layers.
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: i32,
    /// KV heads for full (global) attention layers. `None` reuses
    /// `num_key_value_heads`.
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// Period of the `5 sliding : 1 full` layer-type pattern. Only used to
    /// synthesise `layer_types` when the checkpoint omits it.
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: i32,
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: Option<f32>,
    /// Per-layer attention mode: `"full_attention"` or `"sliding_attention"`.
    /// Synthesised from `sliding_window_pattern` when empty.
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<Gemma4RopeConfig>,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    /// Controls encoder bidirectionality. `"all"` makes the encoder fully
    /// bidirectional; anything else (the released checkpoint uses `"vision"`)
    /// keeps text-token attention causal.
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,
    #[serde(default = "default_num_experts")]
    pub num_experts: i32,
    #[serde(default = "default_top_k_experts")]
    pub top_k_experts: i32,
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: i32,
    /// Block length used by the diffusion decoder canvas.
    #[serde(default = "default_canvas_length")]
    pub canvas_length: i32,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

impl Default for DiffusionGemmaTextConfig {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            head_dim: default_head_dim(),
            global_head_dim: default_global_head_dim(),
            num_global_key_value_heads: None,
            max_position_embeddings: default_max_position_embeddings(),
            rms_norm_eps: default_rms_norm_eps(),
            sliding_window: default_sliding_window(),
            sliding_window_pattern: default_sliding_window_pattern(),
            final_logit_softcapping: default_final_logit_softcapping(),
            layer_types: Vec::new(),
            rope_parameters: None,
            hidden_activation: default_hidden_activation(),
            use_bidirectional_attention: None,
            num_experts: default_num_experts(),
            top_k_experts: default_top_k_experts(),
            moe_intermediate_size: default_moe_intermediate_size(),
            canvas_length: default_canvas_length(),
            tie_word_embeddings: default_tie_word_embeddings(),
        }
    }
}

impl DiffusionGemmaTextConfig {
    /// Resolve `layer_types`, synthesising the `5 sliding : 1 full` pattern
    /// (last layer forced to `full_attention`) when the checkpoint omits it.
    pub fn resolved_layer_types(&self) -> Vec<String> {
        if !self.layer_types.is_empty() {
            return self.layer_types.clone();
        }
        let period = self.sliding_window_pattern.max(1);
        let n = self.num_hidden_layers.max(0);
        let mut types: Vec<String> = (0..n)
            .map(|i| {
                if (i + 1) % period == 0 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect();
        if let Some(last) = types.last_mut() {
            *last = "full_attention".to_string();
        }
        types
    }

    /// Default per-layer-type RoPE parameters: sliding layers use plain RoPE
    /// at θ=1e4; full layers use proportional RoPE (`partial_rotary_factor`
    /// 0.25) at θ=1e6.
    fn resolved_rope_parameters(&self) -> Gemma4RopeConfig {
        self.rope_parameters.clone().unwrap_or(Gemma4RopeConfig {
            full_attention: Gemma4RopeLayerConfig {
                partial_rotary_factor: 0.25,
                rope_theta: Some(1_000_000.0),
                rope_type: Some("proportional".to_string()),
            },
            sliding_attention: Gemma4RopeLayerConfig {
                partial_rotary_factor: 1.0,
                rope_theta: Some(10_000.0),
                rope_type: Some("default".to_string()),
            },
        })
    }

    /// Build the [`Gemma4Config`] that describes this tower's per-layer
    /// attention + dense-MLP geometry. DiffusionGemma's full-attention layers
    /// are exactly Gemma 4's `attention_k_eq_v` layers, so the verified
    /// [`Gemma4Attention`] / [`Gemma4Mlp`] constructors can be reused directly.
    pub fn geometry(&self) -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4_text".to_string(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            global_head_dim: Some(self.global_head_dim),
            num_global_key_value_heads: self.num_global_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            attention_k_eq_v: true,
            tie_word_embeddings: self.tie_word_embeddings,
            sliding_window: self.sliding_window,
            final_logit_softcapping: self.final_logit_softcapping,
            layer_types: self.resolved_layer_types(),
            rope_parameters: Some(self.resolved_rope_parameters()),
            _raw_rope_parameters: None,
            hidden_size_per_layer_input: None,
            vocab_size_per_layer_input: None,
            hidden_activation: Some(self.hidden_activation.clone()),
            num_kv_shared_layers: None,
            use_double_wide_mlp: None,
            enable_moe_block: None,
        }
    }

    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.resolved_layer_types()
            .get(layer_idx)
            .map(|s| s == "full_attention")
            .unwrap_or(false)
    }
}

// ----------------------------------------------------------------------------
// Router
// ----------------------------------------------------------------------------

/// MoE router (`DiffusionGemmaTextRouter`).
///
/// `softmax(proj(norm(x) · scale · hidden^-0.5))` over experts, then top-k,
/// renormalise to sum 1, and multiply by the learned `per_expert_scale`.
/// The softmax runs in fp32. `norm` is weight-less RMSNorm.
#[derive(Debug)]
pub struct DiffusionGemmaRouter {
    pub proj: nn::Linear,
    /// Per-channel pre-projection scale `[hidden_size]`.
    pub scale: Param<Array>,
    /// Per-expert output scale `[num_experts]`.
    pub per_expert_scale: Param<Array>,
    pub top_k: i32,
    pub scalar_root_size: f32,
    pub eps: f32,
}
impl_module_params!(DiffusionGemmaRouter; proj, scale, per_expert_scale);

impl DiffusionGemmaRouter {
    pub fn new(config: &DiffusionGemmaTextConfig) -> Result<Self, Exception> {
        Ok(Self {
            proj: nn::LinearBuilder::new(config.hidden_size, config.num_experts)
                .bias(false)
                .build()?,
            scale: Param::new(Array::ones_f32(&[config.hidden_size])),
            per_expert_scale: Param::new(Array::ones_f32(&[config.num_experts])),
            top_k: config.top_k_experts,
            scalar_root_size: (config.hidden_size as f32).powf(-0.5),
            eps: config.rms_norm_eps,
        })
    }

    /// Route a `[N, hidden]` tensor of (flattened) tokens. Returns
    /// `(top_indices, top_weights)`, both `[N, top_k]` — indices `i32`,
    /// weights already renormalised and per-expert-scaled.
    pub fn route(&mut self, hidden_flat: &Array) -> Result<(Array, Array), Exception> {
        let normed = rms_norm_noscale(hidden_flat, self.eps);
        let scaled = normed
            .multiply(self.scale.as_ref())
            .multiply(&Array::from_f32(self.scalar_root_size));
        let scores = self.proj.forward(&scaled);
        let scores_f32 = scores.as_type::<f32>();
        let probs = ops::softmax_axis(&scores_f32, -1);

        let (top_indices, top_weights) =
            crate::moe_routing::topk_normalize(&probs, self.top_k, true)?;

        // Gather the per-expert scale for each selected expert and apply it.
        let n = top_indices.dim(0);
        let k = top_indices.dim(1);
        let idx_flat = top_indices.reshape(&[n * k]);
        let gathered = self
            .per_expert_scale
            .as_ref()
            .take_axis(&idx_flat, 0)
            .reshape(&[n, k]);
        let top_weights = top_weights.multiply(&gathered);
        Ok((top_indices, top_weights))
    }
}

// ----------------------------------------------------------------------------
// Experts
// ----------------------------------------------------------------------------

/// Grouped expert FFNs (`DiffusionGemmaTextExperts` / `Gemma4TextExperts`).
///
/// Weights are stored as fused 3-D parameters: `gate_up_proj [E, 2·I, H]`
/// (gate and up concatenated along the output axis) and `down_proj [E, H, I]`,
/// matching `nn.Linear` weight layout (`out_features` first). The activation
/// is gelu-tanh.
#[derive(Debug)]
pub struct DiffusionGemmaExperts {
    pub gate_up_proj: Param<Array>,
    pub down_proj: Param<Array>,
    pub num_experts: i32,
    pub moe_intermediate_size: i32,
    pub hidden_size: i32,
}
impl_module_params!(DiffusionGemmaExperts; gate_up_proj, down_proj);

impl DiffusionGemmaExperts {
    pub fn new(config: &DiffusionGemmaTextConfig) -> Result<Self, Exception> {
        let e = config.num_experts;
        let i = config.moe_intermediate_size;
        let h = config.hidden_size;
        Ok(Self {
            gate_up_proj: Param::new(Array::zeros_f32(&[e, 2 * i, h])),
            down_proj: Param::new(Array::zeros_f32(&[e, h, i])),
            num_experts: e,
            moe_intermediate_size: i,
            hidden_size: h,
        })
    }

    /// Apply the experts to a `[N, hidden]` tensor, dispatching each token to
    /// its `top_indices` experts and weighting by `top_weights`
    /// (both `[N, top_k]`). Returns `[N, hidden]`.
    pub fn forward(
        &self,
        hidden_flat: &Array,
        top_indices: &Array,
        top_weights: &Array,
    ) -> Result<Array, Exception> {
        let n = hidden_flat.dim(0);
        let h = self.hidden_size;
        let i = self.moe_intermediate_size;
        let k = top_indices.dim(1);

        let mut out = ops::zeros_dtype(&[n, h], hidden_flat.dtype());
        for slot in 0..k {
            let slot_experts = ops::slice_axis(top_indices, -1, slot, slot + 1).reshape(&[n]);
            let slot_weights = ops::slice_axis(top_weights, -1, slot, slot + 1); // [N, 1]

            // gate_up_proj[e]: [2I, H] -> need [H, 2I] for x[N,1,H] @ w[N,H,2I].
            let gate_up_w = self
                .gate_up_proj
                .as_ref()
                .take_axis(&slot_experts, 0)
                .transpose_axes(&[0, 2, 1]);
            let x_b = hidden_flat.reshape(&[n, 1, h]);
            let gate_up = ops::matmul(&x_b, &gate_up_w).squeeze_axes(&[1]); // [N, 2I]
            let gate = ops::slice_axis(&gate_up, -1, 0, i);
            let up = ops::slice_axis(&gate_up, -1, i, 2 * i);
            let activated = nn::gelu_tanh_approximate(&gate).multiply(&up); // [N, I]

            // down_proj[e]: [H, I] -> need [I, H] for h[N,1,I] @ w[N,I,H].
            let down_w = self
                .down_proj
                .as_ref()
                .take_axis(&slot_experts, 0)
                .transpose_axes(&[0, 2, 1]);
            let act_b = activated.reshape(&[n, 1, i]);
            let down = ops::matmul(&act_b, &down_w).squeeze_axes(&[1]); // [N, H]

            out = out.add(&down.multiply(&slot_weights));
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// Self-conditioning
// ----------------------------------------------------------------------------

/// Self-conditioning FFN (`DiffusionGemmaSelfConditioning`).
///
/// Folds the previous denoising step's soft embeddings into the decoder's
/// input embeddings: `post_norm(inputs_embeds + down(act(gate(pre_norm(s))) *
/// up(pre_norm(s))))`. `pre_norm` is a learnable RMSNorm; `post_norm` is
/// weight-less.
#[derive(Debug)]
pub struct DiffusionGemmaSelfConditioning {
    pub pre_norm: Gemma4RmsNorm,
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
    pub eps: f32,
}
impl_module_params!(DiffusionGemmaSelfConditioning; pre_norm, gate_proj, up_proj, down_proj);

impl DiffusionGemmaSelfConditioning {
    pub fn new(config: &DiffusionGemmaTextConfig) -> Result<Self, Exception> {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        Ok(Self {
            pre_norm: Gemma4RmsNorm::new(h, config.rms_norm_eps),
            gate_proj: nn::LinearBuilder::new(h, i).bias(false).build()?,
            up_proj: nn::LinearBuilder::new(h, i).bias(false).build()?,
            down_proj: nn::LinearBuilder::new(i, h).bias(false).build()?,
            eps: config.rms_norm_eps,
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Array,
        self_conditioning_signal: &Array,
    ) -> Result<Array, Exception> {
        let normed = self.pre_norm.forward(self_conditioning_signal);
        let gate = nn::gelu_tanh_approximate(&self.gate_proj.forward(&normed));
        let sc = self
            .down_proj
            .forward(&gate.multiply(&self.up_proj.forward(&normed)));
        let combined = inputs_embeds.add(&sc);
        Ok(rms_norm_noscale(&combined, self.eps))
    }
}

// ----------------------------------------------------------------------------
// 7-norm parallel dense+MoE layer
// ----------------------------------------------------------------------------

/// A DiffusionGemma text layer (shared by encoder and decoder).
///
/// Pre-norm attention, then a feed-forward stage where a **dense MLP** and a
/// **MoE block** run in parallel off the same post-attention residual and are
/// summed. Critically, the router sees the **raw** residual while the experts
/// see a separately-normed copy:
///
/// ```text
///   h  = residual + post_attention_layernorm(attn(input_layernorm(x)))
///   r  = h                                              (raw residual)
///   m1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h)))
///   m2 = post_feedforward_layernorm_2(
///            experts(pre_feedforward_layernorm_2(r), router(r)))
///   h  = r + post_feedforward_layernorm(m1 + m2)
///   h  = h * layer_scalar
/// ```
#[derive(Debug)]
pub struct DiffusionGemmaTextLayer {
    pub input_layernorm: Gemma4RmsNorm,
    pub self_attn: Gemma4Attention,
    pub post_attention_layernorm: Gemma4RmsNorm,
    pub pre_feedforward_layernorm: Gemma4RmsNorm,
    pub mlp: Gemma4Mlp,
    pub post_feedforward_layernorm: Gemma4RmsNorm,
    pub post_feedforward_layernorm_1: Gemma4RmsNorm,
    pub pre_feedforward_layernorm_2: Gemma4RmsNorm,
    pub post_feedforward_layernorm_2: Gemma4RmsNorm,
    pub router: DiffusionGemmaRouter,
    pub experts: DiffusionGemmaExperts,
    pub layer_scalar: Param<Array>,
}
impl_module_params!(
    DiffusionGemmaTextLayer;
    input_layernorm,
    self_attn,
    post_attention_layernorm,
    pre_feedforward_layernorm,
    mlp,
    post_feedforward_layernorm,
    post_feedforward_layernorm_1,
    pre_feedforward_layernorm_2,
    post_feedforward_layernorm_2,
    router,
    experts,
    layer_scalar
);

impl DiffusionGemmaTextLayer {
    pub fn new(config: &DiffusionGemmaTextConfig, layer_idx: usize) -> Result<Self, Exception> {
        let geometry = config.geometry();
        let h = config.hidden_size;
        let eps = config.rms_norm_eps;
        Ok(Self {
            input_layernorm: Gemma4RmsNorm::new(h, eps),
            self_attn: Gemma4Attention::new(&geometry, layer_idx)?,
            post_attention_layernorm: Gemma4RmsNorm::new(h, eps),
            pre_feedforward_layernorm: Gemma4RmsNorm::new(h, eps),
            mlp: Gemma4Mlp::new(&geometry)?,
            post_feedforward_layernorm: Gemma4RmsNorm::new(h, eps),
            post_feedforward_layernorm_1: Gemma4RmsNorm::new(h, eps),
            pre_feedforward_layernorm_2: Gemma4RmsNorm::new(h, eps),
            post_feedforward_layernorm_2: Gemma4RmsNorm::new(h, eps),
            router: DiffusionGemmaRouter::new(config)?,
            experts: DiffusionGemmaExperts::new(config)?,
            layer_scalar: Param::new(Array::ones_f32(&[1])),
        })
    }

    /// The shared parallel dense+MoE feed-forward tail. `h` is the
    /// post-attention hidden state (`residual + post_attn_ln(attn)`).
    fn feed_forward(&mut self, h: &Array) -> Result<Array, Exception> {
        let residual = h.clone();
        let b = h.dim(0);
        let s = h.dim(1);
        let hidden = h.dim(2);

        // Dense branch.
        let dense_in = self.pre_feedforward_layernorm.forward(h);
        let dense = self.mlp.forward(&dense_in)?;
        let dense = self.post_feedforward_layernorm_1.forward(&dense);

        // MoE branch — router reads the RAW residual; experts read a
        // separately-normed copy of it.
        let flat = residual.reshape(&[b * s, hidden]);
        let (top_indices, top_weights) = self.router.route(&flat)?;
        let experts_in = self.pre_feedforward_layernorm_2.forward(&flat);
        let moe = self
            .experts
            .forward(&experts_in, &top_indices, &top_weights)?;
        let moe = moe.reshape(&[b, s, hidden]);
        let moe = self.post_feedforward_layernorm_2.forward(&moe);

        let combined = dense.add(&moe);
        let combined = self.post_feedforward_layernorm.forward(&combined);
        let out = residual.add(&combined);
        Ok(out.multiply(self.layer_scalar.as_ref()))
    }

    /// Encoder forward: causal self-attention that *collects* this layer's
    /// post-norm/post-rope K/V (for the decoder to read later). `mask` is
    /// `None` to use Gemma 4's automatic causal / sliding-window masking.
    pub fn forward_encoder(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        offset: i32,
    ) -> Result<(Array, Array, Array), Exception> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let (attn_out, keys, values) = self.self_attn.forward_collect_kv(&h, mask, offset)?;
        let h = self.post_attention_layernorm.forward(&attn_out);
        let h = residual.add(&h);
        let out = self.feed_forward(&h)?;
        Ok((out, keys, values))
    }

    /// Decoder forward: bidirectional self-attention over the canvas that
    /// also reads the encoder's (read-only) K/V for this layer. `mask` must
    /// be supplied explicitly (the decoder is never auto-causal).
    pub fn forward_decoder(
        &mut self,
        x: &Array,
        encoder_keys: &Array,
        encoder_values: &Array,
        mask: Option<&Array>,
        offset: i32,
    ) -> Result<Array, Exception> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward_with_encoder_kv(
            &h,
            encoder_keys,
            encoder_values,
            mask,
            offset,
        )?;
        let h = self.post_attention_layernorm.forward(&attn_out);
        let h = residual.add(&h);
        self.feed_forward(&h)
    }
}

// ----------------------------------------------------------------------------
// Encoder model
// ----------------------------------------------------------------------------

/// DiffusionGemma encoder text tower (`DiffusionGemmaEncoderTextModel`).
///
/// Embeds the prompt (scaled by `sqrt(hidden)`), runs the causal layer stack,
/// and returns the final-normed hidden states alongside each layer's K/V so
/// the decoder can attend to them.
#[derive(Debug)]
pub struct DiffusionGemmaEncoderModel {
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<DiffusionGemmaTextLayer>,
    pub norm: Gemma4RmsNorm,
    pub config: DiffusionGemmaTextConfig,
    pub embed_scale: f32,
}
impl_module_params!(DiffusionGemmaEncoderModel; embed_tokens, layers, norm);

impl DiffusionGemmaEncoderModel {
    pub fn new(config: DiffusionGemmaTextConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| DiffusionGemmaTextLayer::new(&config, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = Gemma4RmsNorm::new(config.hidden_size, config.rms_norm_eps);
        let embed_scale = (config.hidden_size as f32).sqrt();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            config,
            embed_scale,
        })
    }

    /// Run the encoder. Returns `(hidden_states, per_layer_kv)` where
    /// `per_layer_kv[i]` is the `(keys, values)` of layer `i`, each
    /// `[B, n_kv_heads, seq, head_dim]`.
    pub fn forward(
        &mut self,
        input_ids: &Array,
    ) -> Result<(Array, Vec<(Array, Array)>), Exception> {
        let mut h = self
            .embed_tokens
            .forward(input_ids)
            .multiply(&Array::from_f32(self.embed_scale));
        let mut kvs = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter_mut() {
            // mask=None lets Gemma 4's attention apply per-layer causal /
            // sliding-window masking automatically.
            let (next, keys, values) = layer.forward_encoder(&h, None, 0)?;
            kvs.push((keys, values));
            h = next;
        }
        Ok((self.norm.forward(&h), kvs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn tiny_config() -> DiffusionGemmaTextConfig {
        DiffusionGemmaTextConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 48,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            global_head_dim: 16,
            num_global_key_value_heads: Some(1),
            sliding_window: 8,
            sliding_window_pattern: 3,
            num_experts: 4,
            top_k_experts: 2,
            moe_intermediate_size: 16,
            canvas_length: 8,
            ..Default::default()
        }
    }

    #[test]
    fn layer_types_pattern_forces_last_full() {
        let cfg = tiny_config();
        let types = cfg.resolved_layer_types();
        assert_eq!(types.len(), 3);
        // pattern 3: idx 2 (i+1=3) is full; last forced full.
        assert_eq!(types[0], "sliding_attention");
        assert_eq!(types[1], "sliding_attention");
        assert_eq!(types[2], "full_attention");
        assert!(cfg.is_full_attention(2));
        assert!(!cfg.is_full_attention(0));
    }

    #[test]
    fn geometry_marks_full_layers_k_eq_v() {
        let geometry = tiny_config().geometry();
        assert!(geometry.attention_k_eq_v);
        // Full layer uses global head dim / global KV heads.
        assert_eq!(geometry.layer_head_dim(2), 16);
        assert_eq!(geometry.layer_num_kv_heads(2), 1);
        assert!(geometry.layer_uses_k_eq_v(2));
        // Sliding layer uses the base geometry with a v_proj.
        assert_eq!(geometry.layer_head_dim(0), 8);
        assert_eq!(geometry.layer_num_kv_heads(0), 2);
        assert!(!geometry.layer_uses_k_eq_v(0));
    }

    #[test]
    #[serial]
    fn router_shapes() {
        let cfg = tiny_config();
        let mut router = DiffusionGemmaRouter::new(&cfg).unwrap();
        let x = pmetal_bridge::compat::random::uniform_range(
            -1.0,
            1.0,
            &[6, cfg.hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let (idx, w) = router.route(&x).unwrap();
        assert_eq!(idx.shape(), &[6, cfg.top_k_experts]);
        assert_eq!(w.shape(), &[6, cfg.top_k_experts]);
    }

    #[test]
    #[serial]
    fn experts_shapes() {
        let cfg = tiny_config();
        let mut router = DiffusionGemmaRouter::new(&cfg).unwrap();
        let experts = DiffusionGemmaExperts::new(&cfg).unwrap();
        let x = pmetal_bridge::compat::random::uniform_range(
            -1.0,
            1.0,
            &[6, cfg.hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let (idx, w) = router.route(&x).unwrap();
        let out = experts.forward(&x, &idx, &w).unwrap();
        assert_eq!(out.shape(), &[6, cfg.hidden_size]);
    }

    #[test]
    #[serial]
    fn encoder_forward_shape_and_kv() {
        let cfg = tiny_config();
        let kv_layer_0 = cfg.geometry().layer_num_kv_heads(0);
        let mut model = DiffusionGemmaEncoderModel::new(cfg.clone()).unwrap();
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4, 5], &[1, 5]);
        let (hidden, kvs) = model.forward(&input_ids).unwrap();
        assert_eq!(hidden.shape(), &[1, 5, cfg.hidden_size]);
        assert_eq!(kvs.len(), cfg.num_hidden_layers as usize);
        // Layer 0 is sliding: [B, n_kv_heads, seq, head_dim].
        assert_eq!(kvs[0].0.shape(), &[1, kv_layer_0, 5, cfg.head_dim]);
    }
}
