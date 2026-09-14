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
//! Landed: config, the leaf MoE blocks, the 7-norm layer, the **encoder**
//! model (causal, auto-masked, KV-collecting), and the **decoder** model
//! (fully-bidirectional canvas over `[encoder_kv | canvas]`, read-only
//! encoder-KV concat via [`Gemma4Attention::forward_with_encoder_kv`],
//! self-conditioning). Both are numerically parity-verified against the
//! transformers oracle. The discrete-diffusion generation engine (block loop,
//! entropy-bound sampler, stopping) is built in a later phase.

use pmetal_bridge::compat::{Array, Dtype, Exception, Module, Param, nn, ops};
use pmetal_bridge::impl_module_params;
use pmetal_mlx::kernels::rope::RopePositions;
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::gemma4::{
    Gemma4Attention, Gemma4Config, Gemma4Mlp, Gemma4RmsNorm, Gemma4RopeConfig,
    Gemma4RopeLayerConfig, LoadReport, rms_norm_noscale,
};
use super::gemma4_vision::{
    Gemma4MultimodalEmbedder, Gemma4VisionConfig, Gemma4VisionModel, load_gemma4_vision_weights,
};

/// Default `image_token_id` for `google/diffusiongemma-26B-A4B-it` — the token
/// whose embedding slots are replaced by projected vision soft tokens.
pub const DIFFUSION_GEMMA_IMAGE_TOKEN_ID: i32 = 258_880;

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
            enable_moe_block: Some(true),
            num_experts: Some(self.num_experts),
            top_k_experts: Some(self.top_k_experts),
            moe_intermediate_size: Some(self.moe_intermediate_size),
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
// Router + Experts (canonical Gemma 4 MoE blocks)
// ----------------------------------------------------------------------------

// DiffusionGemma's MoE router and grouped experts ARE the Gemma 4 MoE blocks
// (`Gemma4TextRouter` / `Gemma4TextExperts`) — same fp32-softmax top-k router
// with per-channel `scale` + per-expert scale, and the same fused
// `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]` gelu-tanh SwiGLU experts.
// They live canonically in [`super::gemma4`]; we re-export them here under the
// DiffusionGemma names so call sites and weight-loader keys read naturally.
pub use super::gemma4::{
    Gemma4Experts as DiffusionGemmaExperts, Gemma4Router as DiffusionGemmaRouter,
};

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
            router: DiffusionGemmaRouter::new(
                config.hidden_size,
                config.num_experts,
                config.top_k_experts,
                config.rms_norm_eps,
            )?,
            experts: DiffusionGemmaExperts::new(
                config.num_experts,
                config.moe_intermediate_size,
                config.hidden_size,
            )?,
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
        let (attn_out, keys, values) =
            self.self_attn
                .forward_collect_kv(&h, mask, RopePositions::Offset(offset))?;
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
            RopePositions::Offset(offset),
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
    /// Optional image backbone (`None` for the text-only path, which is then
    /// byte-identical to the pre-vision encoder — preserving text parity by
    /// construction). Attached via [`DiffusionGemmaEncoderModel::attach_vision`].
    pub vision_tower: Option<Gemma4VisionModel>,
    /// Optional vision→text projector, paired with `vision_tower`.
    pub embed_vision: Option<Gemma4MultimodalEmbedder>,
    /// Token id whose embedding slots are replaced by projected vision soft
    /// tokens (default [`DIFFUSION_GEMMA_IMAGE_TOKEN_ID`]).
    pub image_token_id: i32,
}
// `vision_tower` / `embed_vision` are intentionally outside the parameter tree
// (like the QLoRA `qbase` fields): they are loaded by a bespoke path and their
// absence keeps the text-tower parameter set byte-identical.
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
            vision_tower: None,
            embed_vision: None,
            image_token_id: DIFFUSION_GEMMA_IMAGE_TOKEN_ID,
        })
    }

    /// Attach the image backbone: a [`Gemma4VisionModel`] tower and the
    /// vision→text [`Gemma4MultimodalEmbedder`]. Enables
    /// [`DiffusionGemmaEncoderModel::forward_multimodal`]; the text-only
    /// [`forward`](Self::forward) path is unaffected.
    pub fn attach_vision(
        &mut self,
        vision_config: &Gemma4VisionConfig,
        image_token_id: i32,
    ) -> Result<(), Exception> {
        let embedder = Gemma4MultimodalEmbedder::new(
            vision_config.hidden_size,
            self.config.hidden_size,
            vision_config.rms_norm_eps,
        )?;
        self.vision_tower = Some(Gemma4VisionModel::new(vision_config)?);
        self.embed_vision = Some(embedder);
        self.image_token_id = image_token_id;
        Ok(())
    }

    /// Run the encoder. Returns `(hidden_states, per_layer_kv)` where
    /// `per_layer_kv[i]` is the `(keys, values)` of layer `i`, each
    /// `[B, n_kv_heads, seq, head_dim]`.
    pub fn forward(
        &mut self,
        input_ids: &Array,
    ) -> Result<(Array, Vec<(Array, Array)>), Exception> {
        let embeds = self
            .embed_tokens
            .forward(input_ids)
            .multiply(&Array::from_f32(self.embed_scale));
        self.forward_from_embeds(&embeds, None)
    }

    /// Run the encoder layer stack from precomputed input embeddings (the shared
    /// tail of [`forward`](Self::forward) and
    /// [`forward_multimodal`](Self::forward_multimodal)). Returns the
    /// final-normed hidden states and each layer's K/V.
    ///
    /// `masks` supplies explicit per-layer-type additive masks
    /// `(full_layer_mask, sliding_layer_mask)` — used for the bidirectional-image
    /// band. `None` lets Gemma 4's attention apply its automatic per-layer
    /// causal / sliding-window masking.
    fn forward_from_embeds(
        &mut self,
        inputs_embeds: &Array,
        masks: Option<(&Array, &Array)>,
    ) -> Result<(Array, Vec<(Array, Array)>), Exception> {
        // Precompute layer-type flags to avoid borrowing `self.config` inside the
        // `self.layers` mutable loop.
        let fulls: Vec<bool> = (0..self.layers.len())
            .map(|i| self.config.is_full_attention(i))
            .collect();
        let mut h = inputs_embeds.clone();
        let mut kvs = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let mask = masks.map(|(full, sliding)| if fulls[i] { full } else { sliding });
            let (next, keys, values) = layer.forward_encoder(&h, mask, 0)?;
            kvs.push((keys, values));
            h = next;
        }
        Ok((self.norm.forward(&h), kvs))
    }

    /// Multimodal encoder forward: embed `input_ids` (with `image_token_id`
    /// slots temporarily zeroed to stay in-vocab, then `√hidden`-scaled),
    /// encode `pixel_values` into projected soft tokens, scatter those into the
    /// image slots, and run the causal stack. Requires
    /// [`attach_vision`](Self::attach_vision). `image_position_ids` are the
    /// `[B, num_patches, 2]` patch coordinates from the image processor.
    ///
    /// Numerically mirrors the oracle `DiffusionGemmaEncoderModel.forward(
    /// input_ids, pixel_values, image_position_ids)`: the projected vision
    /// features carry their own `√vision_hidden` pooling scale + projection and
    /// are *not* re-scaled by the text embed scale (they replace the scaled
    /// placeholder embeddings after the fact).
    ///
    /// `bidirectional_images` selects the attention mask over the merged
    /// sequence:
    ///
    /// * `true` (production / design-correct): tokens within the same image
    ///   span attend **bidirectionally** while text stays causal — the
    ///   architecture's `use_bidirectional_attention="vision"` intent.
    /// * `false`: the whole sequence is causal.
    ///
    /// **pmetal deliberately diverges from `transformers` here.** In
    /// `transformers` (≤ 5.10.0.dev0) `DiffusionGemmaEncoderModel.forward`
    /// builds the bidirectional-vision mask via `create_masks_for_generate(...)`
    /// but **discards the result**, so `mm_token_type_ids` has no effect and the
    /// encoder is silently causal (`pooled_bidir == pooled_causal`). We
    /// implement the intended bidirectional behaviour; the `false` path exists
    /// to reproduce the transformers oracle for the vision-tower / projector /
    /// merge parity test.
    pub fn forward_multimodal(
        &mut self,
        input_ids: &Array,
        pixel_values: &Array,
        image_position_ids: &Array,
        bidirectional_images: bool,
    ) -> Result<(Array, Vec<(Array, Array)>), Exception> {
        let image_token = Array::from_f32(self.image_token_id as f32).as_type::<i32>();
        let image_mask = ops::equal(input_ids, &image_token); // [B, seq] bool

        // Zero image-token ids before the embedding lookup (they may be OOV),
        // then apply the √hidden scale.
        let safe_ids = ops::where_fn(&image_mask, &ops::zeros_like(input_ids), input_ids);
        let text_embeds = self
            .embed_tokens
            .forward(&safe_ids)
            .multiply(&Array::from_f32(self.embed_scale));

        // Pooled vision soft tokens → projected into text space.
        let feats = {
            let vision = self.vision_tower.as_mut().ok_or_else(|| {
                Exception::custom("forward_multimodal: vision_tower not attached")
            })?;
            vision.forward(pixel_values, image_position_ids)?
        };
        let vhidden = feats.dim(feats.shape().len() as i32 - 1);
        let feats_flat = feats.reshape(&[-1, vhidden]); // [B·out_len, vision_hidden]
        let image_features = {
            let embedder = self.embed_vision.as_mut().ok_or_else(|| {
                Exception::custom("forward_multimodal: embed_vision not attached")
            })?;
            embedder.forward(&feats_flat) // [B·out_len, text_hidden]
        };

        let merged = merge_image_features(&text_embeds, &image_mask, &image_features);

        if bidirectional_images {
            let block_ids = derive_image_block_ids(&image_mask);
            let (full_mask, sliding_mask) =
                build_multimodal_masks(&block_ids, self.config.sliding_window, merged.dtype());
            self.forward_from_embeds(&merged, Some((&full_mask, &sliding_mask)))
        } else {
            self.forward_from_embeds(&merged, None)
        }
    }
}

/// Per-token image block ids (`get_block_sequence_ids_for_mask`): each
/// contiguous run of image tokens gets an increasing id (0, 1, …); text tokens
/// get −1. Two tokens attend bidirectionally iff they share a non-negative id.
fn derive_image_block_ids(image_mask: &Array) -> Array {
    let seq = image_mask.dim(1);
    let is_vision = image_mask.as_type::<f32>(); // [B, seq], 0/1
    // prev = is_vision shifted right by one (first column zeroed).
    let zeros_col = ops::slice_axis(&is_vision, 1, 0, 1).multiply(&Array::from_f32(0.0));
    let head = ops::slice_axis(&is_vision, 1, 0, seq - 1);
    let prev = ops::concatenate_axis(&[&zeros_col, &head], 1); // [B, seq]
    // new run start = is_vision AND NOT prev.
    let new_start = is_vision.multiply(&Array::from_f32(1.0).subtract(&prev));
    let group = ops::cumsum(&new_start, 1).subtract(&Array::from_f32(1.0)); // [B, seq]
    ops::where_fn(image_mask, &group, &Array::from_f32(-1.0))
}

/// Additive attention masks for a merged multimodal sequence: causal
/// (respecting the sliding window on sliding layers) OR bidirectional within an
/// image block. Returns `(full_layer_mask, sliding_layer_mask)`, each
/// `[B, 1, seq, seq]` in `dtype`. The sliding-window term is a no-op when
/// `seq <= sliding_window` (as in the parity fixture); it mirrors Gemma 4's
/// `q − k < sliding_window` convention for longer sequences.
fn build_multimodal_masks(block_ids: &Array, sliding_window: i32, dtype: Dtype) -> (Array, Array) {
    let b = block_ids.dim(0);
    let seq = block_ids.dim(1);

    // Bidirectional band: same block id and not text (−1).
    let bi = block_ids.reshape(&[b, seq, 1]);
    let bj = block_ids.reshape(&[b, 1, seq]);
    let same = ops::equal(&bi, &bj)
        .as_type::<f32>()
        .multiply(&ops::greater(&bi, &Array::from_f32(-0.5)).as_type::<f32>()); // [B, seq, seq]

    // Causal + window terms (in {0, 1}).
    let idx = ops::arange(seq, Dtype::Float32);
    let qi = idx.reshape(&[seq, 1]);
    let kj = idx.reshape(&[1, seq]);
    let causal = ops::less_equal(&kj, &qi).as_type::<f32>(); // [seq, seq]
    let within =
        ops::greater(&Array::from_f32(sliding_window as f32), &qi.subtract(&kj)).as_type::<f32>();

    let full_ok = ops::maximum(&causal, &same); // OR
    let sliding_ok = ops::maximum(&causal.multiply(&within), &same);

    // Additive mask: 0 where allowed, −inf where blocked (matching MLX's own
    // `create_sliding_window_mask`; a finite fill can leak through the masked
    // softmax fast path for non-triangular masks).
    let zero = Array::from_f32(0.0);
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let to_additive = |ok: &Array| {
        ops::where_fn(&ops::greater(ok, &Array::from_f32(0.5)), &zero, &neg_inf)
            .reshape(&[b, 1, seq, seq])
            .as_dtype(dtype.as_i32())
    };
    (to_additive(&full_ok), to_additive(&sliding_ok))
}

/// Scatter `image_features` `[n_img, hidden]` into the `image_mask`-true slots
/// of `inputs_embeds` `[B, seq, hidden]`, in row-major order (the MLX analogue
/// of `torch.Tensor.masked_scatter`). Non-image positions keep their text
/// embedding.
fn merge_image_features(
    inputs_embeds: &Array,
    image_mask: &Array,
    image_features: &Array,
) -> Array {
    let b = inputs_embeds.dim(0);
    let seq = inputs_embeds.dim(1);
    let h = inputs_embeds.dim(2);
    let n_img = image_features.dim(0).max(1);

    let text_flat = inputs_embeds.reshape(&[b * seq, h]);
    let mask_flat = image_mask.reshape(&[b * seq]).as_type::<f32>();

    // slot[i] = (#image tokens up to and including i) − 1: the row of
    // `image_features` destined for position i (garbage where non-image, but
    // masked out below). Clamp into range so the gather is always valid.
    let slot = ops::cumsum(&mask_flat, 0).subtract(&Array::from_f32(1.0));
    let slot = ops::clip(
        &slot,
        Some(&Array::from_f32(0.0)),
        Some(&Array::from_f32((n_img - 1) as f32)),
    )
    .as_type::<i32>();
    let gathered = ops::take_axis(image_features, &slot, 0); // [B·seq, hidden]

    let is_image = ops::greater(&mask_flat.expand_dims(-1), &Array::from_f32(0.5));
    ops::where_fn(&is_image, &gathered, &text_flat).reshape(&[b, seq, h])
}

// ----------------------------------------------------------------------------
// Decoder model
// ----------------------------------------------------------------------------

/// DiffusionGemma decoder text tower (`DiffusionGemmaDecoderModel`).
///
/// Refines a fixed `canvas_length` block of tokens with **bidirectional**
/// self-attention while reading — but never writing — the encoder's per-layer
/// K/V cache. The canvas attends over the whole `[encoder_kv | canvas]`
/// sequence with no causal or sliding-window restriction (the no-padding
/// generation path: see `sdpa`/`eager` with `is_causal=False`, `mask=None`).
///
/// Each step folds the previous denoising step's soft embeddings into the
/// input via [`DiffusionGemmaSelfConditioning`]; on the first step
/// (`self_conditioning_logits = None`) the signal is zero and the block just
/// re-normalises the input embeddings.
///
/// The trunk (`layers`, `embed_tokens`, `norm`) is weight-identical to — and
/// in the full checkpoint tied to — the encoder; only `self_conditioning` is
/// decoder-only.
#[derive(Debug)]
pub struct DiffusionGemmaDecoderModel {
    pub embed_tokens: nn::Embedding,
    pub self_conditioning: DiffusionGemmaSelfConditioning,
    pub layers: Vec<DiffusionGemmaTextLayer>,
    pub norm: Gemma4RmsNorm,
    pub config: DiffusionGemmaTextConfig,
    pub embed_scale: f32,
}
impl_module_params!(
    DiffusionGemmaDecoderModel;
    embed_tokens,
    self_conditioning,
    layers,
    norm
);

impl DiffusionGemmaDecoderModel {
    pub fn new(config: DiffusionGemmaTextConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let self_conditioning = DiffusionGemmaSelfConditioning::new(&config)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| DiffusionGemmaTextLayer::new(&config, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = Gemma4RmsNorm::new(config.hidden_size, config.rms_norm_eps);
        let embed_scale = (config.hidden_size as f32).sqrt();
        Ok(Self {
            embed_tokens,
            self_conditioning,
            layers,
            norm,
            config,
            embed_scale,
        })
    }

    /// Run the decoder over one canvas block.
    ///
    /// * `decoder_input_ids` — `[B, canvas_length]` canvas token ids.
    /// * `encoder_kvs` — per-layer `(keys, values)` from the encoder, each
    ///   `[B, n_kv_heads, enc_len, head_dim]` (post-norm, post-rope). Length
    ///   must equal `num_hidden_layers`.
    /// * `self_conditioning_logits` — `[B, canvas_length, vocab]` logits from
    ///   the previous denoising step, or `None` on the first step (zeroed
    ///   signal).
    pub fn forward(
        &mut self,
        decoder_input_ids: &Array,
        encoder_kvs: &[(Array, Array)],
        self_conditioning_logits: Option<&Array>,
    ) -> Result<Array, Exception> {
        let inputs_embeds = self
            .embed_tokens
            .forward(decoder_input_ids)
            .multiply(&Array::from_f32(self.embed_scale));

        // Soft embeddings from the previous step's logits (zeros on step 0):
        // `softmax(logits, fp32) @ embed_weight * embed_scale`.
        let soft = match self_conditioning_logits {
            Some(logits) => {
                let probs = ops::softmax_axis(&logits.as_type::<f32>(), -1);
                let weight = self.embed_tokens.weight.as_ref();
                let probs = probs.as_dtype(weight.dtype().as_i32());
                ops::matmul(&probs, weight).multiply(&Array::from_f32(self.embed_scale))
            }
            None => ops::zeros_like(&inputs_embeds),
        };
        let mut h = self.self_conditioning.forward(&inputs_embeds, &soft)?;

        // Canvas RoPE offset = the cumulative encoder length. Sliding-attention
        // layers keep only the last `sliding_window - 1` encoder keys in their
        // cache, so their KV length is shorter; full-attention layers always
        // retain the whole sequence. The true sequence length (and hence the
        // canvas position offset) is therefore the longest per-layer cache —
        // the last layer is always full, so this equals the prompt length.
        let canvas = decoder_input_ids.dim(1);
        let canvas_offset = encoder_kvs.iter().map(|(k, _)| k.dim(2)).max().unwrap_or(0);

        for (layer, (enc_k, enc_v)) in self.layers.iter_mut().zip(encoder_kvs.iter()) {
            // Fully-bidirectional mask over this layer's `[encoder_kv | canvas]`:
            // an all-zeros additive mask sized to this layer's (possibly
            // truncated) cache. Passing an explicit mask stops `Gemma4Attention`
            // from synthesising a causal / sliding-window mask, so every canvas
            // query attends to every key — matching the oracle's
            // `is_causal=False`, `mask=None` SDPA path.
            let enc_len = enc_k.dim(2);
            let mask = ops::zeros_dtype(&[1, 1, canvas, enc_len + canvas], h.dtype());
            h = layer.forward_decoder(&h, enc_k, enc_v, Some(&mask), canvas_offset)?;
        }
        Ok(self.norm.forward(&h))
    }
}

// ----------------------------------------------------------------------------
// Discrete-diffusion generation
// ----------------------------------------------------------------------------

/// Generation / sampler configuration (`DiffusionGemmaGenerationConfig` +
/// `EntropyBoundSamplerConfig`). Defaults match the released checkpoint's
/// `generation_config.json`.
#[derive(Debug, Clone)]
pub struct DiffusionGemmaGenerationConfig {
    pub max_denoising_steps: i32,
    pub t_min: f32,
    pub t_max: f32,
    pub entropy_bound: f32,
    pub confidence_threshold: f32,
    pub stability_threshold: i32,
    pub max_new_tokens: i32,
    pub eos_token_ids: Vec<i32>,
    pub pad_token_id: i32,
}

impl Default for DiffusionGemmaGenerationConfig {
    fn default() -> Self {
        Self {
            max_denoising_steps: 48,
            t_min: 0.4,
            t_max: 0.8,
            entropy_bound: 0.1,
            confidence_threshold: 0.005,
            stability_threshold: 1,
            max_new_tokens: 256,
            eos_token_ids: vec![1, 106, 50],
            pad_token_id: 0,
        }
    }
}

/// Linear temperature schedule (`LinearTemperatureScheduleLogitsProcessor`):
/// `scores / (t_min + (t_max - t_min) * cur_step / max_steps)`. Applied
/// *after* the LM-head softcap. As a positive monotone scaling it leaves the
/// argmax unchanged.
pub fn linear_temperature(
    logits: &Array,
    cur_step: i32,
    t_min: f32,
    t_max: f32,
    max_steps: i32,
) -> Array {
    let temperature = t_min + (t_max - t_min) * (cur_step as f32 / max_steps as f32);
    logits.divide(&Array::from_f32(temperature))
}

/// Categorical token entropy over the last (vocab) axis: `-Σ p·log p` where
/// `p = softmax(logits)`. Returns the input with its last axis reduced.
pub fn categorical_entropy(logits: &Array) -> Array {
    let logp = nn::log_softmax(logits, -1);
    let p = ops::exp(&logp);
    p.multiply(&logp)
        .sum_axis(-1, false)
        .multiply(&Array::from_f32(-1.0))
}

/// Entropy-bound acceptance (`EntropyBoundSampler.accept_canvas`). Accepts the
/// lowest-entropy canvas positions while the cumulative entropy *excluding the
/// current position* stays within `entropy_bound`, then takes the denoiser's
/// token there (and keeps the current token elsewhere).
///
/// Returns `(accepted_canvas, accepted_mask)` — `accepted_mask` is a boolean
/// `[B, canvas]` array (true where the denoiser token was taken).
pub fn entropy_bound_accept(
    processed_logits: &Array,
    current_canvas: &Array,
    denoiser_canvas: &Array,
    entropy_bound: f32,
) -> (Array, Array) {
    let ent = categorical_entropy(processed_logits); // [B, canvas]
    let sorted_idx = ops::argsort_axis(&ent, -1); // ascending
    let sorted_ent = ops::take_along_axis(&ent, &sorted_idx, -1);
    let cum = ops::cumsum(&sorted_ent, -1);
    // cumulative entropy excluding the current (max) position.
    let excl = ops::subtract(&cum, &sorted_ent);
    let sel = ops::less_equal(&excl, &Array::from_f32(entropy_bound)).as_type::<f32>();
    // Scatter the sorted-order selection back to original positions
    // (`torch.scatter(zeros, -1, sorted_idx, sel)`): place `sel[j]` at
    // `sorted_idx[j]`.
    let zeros = ops::zeros_like(&sel);
    let mask_f = zeros.put_along_axis_op(&sorted_idx, &sel, -1);
    let mask = ops::greater(&mask_f, &Array::from_f32(0.5));
    let accepted = ops::where_fn(&mask, denoiser_canvas, current_canvas);
    (accepted, mask)
}

/// Stable-and-confident adaptive stopping (`StableAndConfidentStoppingCriteria`),
/// specialised to a single sequence. Stops once the argmax canvas has been
/// identical for `stability_threshold` consecutive steps *and* the mean token
/// entropy of the processed logits is below `confidence_threshold`.
#[derive(Debug)]
struct StableConfidentStopper {
    stability_threshold: i32,
    confidence_threshold: f32,
    history: Vec<Vec<u32>>,
}

impl StableConfidentStopper {
    fn new(stability_threshold: i32, confidence_threshold: f32) -> Self {
        Self {
            stability_threshold,
            confidence_threshold,
            history: Vec::new(),
        }
    }

    /// `argmax_canvas` is `[1, canvas]`, `processed_logits` is `[1, canvas, vocab]`.
    fn should_stop(&mut self, argmax_canvas: &Array, processed_logits: &Array) -> bool {
        let argmax = {
            let a = argmax_canvas.as_type::<u32>();
            a.eval();
            a.as_slice::<u32>().to_vec()
        };
        let stable = if self.stability_threshold <= 0 {
            true
        } else {
            let full = self.history.len() >= self.stability_threshold as usize;
            full && self.history.iter().all(|h| *h == argmax)
        };
        self.history.push(argmax);
        while self.history.len() > self.stability_threshold.max(0) as usize {
            self.history.remove(0);
        }

        let mean_entropy = categorical_entropy(processed_logits).mean_all().item_f32();
        let confident = mean_entropy < self.confidence_threshold;
        stable && confident
    }
}

/// Top-level block-diffusion model (`DiffusionGemmaForBlockDiffusion`): the
/// encoder + decoder trunk plus the (tied) LM head and discrete-diffusion
/// generation loop.
#[derive(Debug)]
pub struct DiffusionGemmaForBlockDiffusion {
    pub encoder: DiffusionGemmaEncoderModel,
    pub decoder: DiffusionGemmaDecoderModel,
    pub final_logit_softcapping: Option<f32>,
    pub canvas_length: i32,
    pub sliding_window: i32,
    pub layer_types: Vec<String>,
    pub vocab_size: i32,
}
impl_module_params!(DiffusionGemmaForBlockDiffusion; encoder, decoder);

impl DiffusionGemmaForBlockDiffusion {
    pub fn new(config: DiffusionGemmaTextConfig) -> Result<Self, Exception> {
        let layer_types = config.resolved_layer_types();
        let canvas_length = config.canvas_length;
        let sliding_window = config.sliding_window;
        let vocab_size = config.vocab_size;
        let final_logit_softcapping = config.final_logit_softcapping;
        let encoder = DiffusionGemmaEncoderModel::new(config.clone())?;
        let decoder = DiffusionGemmaDecoderModel::new(config)?;
        Ok(Self {
            encoder,
            decoder,
            final_logit_softcapping,
            canvas_length,
            sliding_window,
            layer_types,
            vocab_size,
        })
    }

    /// LM head: tied to the decoder's input embedding, followed by the fp32
    /// final-logit softcap.
    pub fn lm_logits(&self, hidden: &Array) -> Array {
        let raw = self.decoder.embed_tokens.as_linear(hidden).as_type::<f32>();
        match self.final_logit_softcapping {
            Some(cap) => {
                let c = Array::from_f32(cap);
                ops::tanh(&raw.divide(&c)).multiply(&c)
            }
            None => raw,
        }
    }

    /// Encoder trunk hidden states `[B, seq, hidden]` — the natural
    /// pre-LM-head representation for sentence-embedding / pooling endpoints
    /// (`/v1/embeddings`). Runs the causal encoder and discards the per-layer
    /// KV cache. (DiffusionGemma has no single causal next-token forward, so
    /// the encoder output is the meaningful "hidden states" to pool over.)
    pub fn encode_hidden(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        let (hidden, _kvs) = self.encoder.forward(input_ids)?;
        Ok(hidden)
    }

    /// Attach the image backbone (delegates to
    /// [`DiffusionGemmaEncoderModel::attach_vision`]). Enables
    /// [`encode_hidden_multimodal`](Self::encode_hidden_multimodal); the base
    /// weights load through [`load_diffusion_gemma_weights`] afterwards.
    pub fn attach_vision(
        &mut self,
        vision_config: &Gemma4VisionConfig,
        image_token_id: i32,
    ) -> Result<(), Exception> {
        self.encoder.attach_vision(vision_config, image_token_id)
    }

    /// Multimodal encoder hidden states `[B, seq, hidden]`: embed `input_ids`,
    /// merge projected vision soft tokens into the `image_token_id` slots, and
    /// run the encoder with image spans attending bidirectionally (the
    /// design-correct default — see
    /// [`DiffusionGemmaEncoderModel::forward_multimodal`]).
    pub fn encode_hidden_multimodal(
        &mut self,
        input_ids: &Array,
        pixel_values: &Array,
        image_position_ids: &Array,
    ) -> Result<Array, Exception> {
        let (hidden, _kvs) =
            self.encoder
                .forward_multimodal(input_ids, pixel_values, image_position_ids, true)?;
        Ok(hidden)
    }

    /// Truncate sliding-attention layers' encoder K/V to the last
    /// `sliding_window - 1` positions, matching transformers'
    /// `DynamicSlidingWindowLayer` (non-compiled path). Full-attention layers
    /// keep the whole sequence. The decoder's per-layer masking already copes
    /// with the resulting ragged KV lengths.
    fn truncate_sliding_kvs(&self, kvs: Vec<(Array, Array)>) -> Vec<(Array, Array)> {
        let keep = (self.sliding_window - 1).max(0);
        kvs.into_iter()
            .enumerate()
            .map(|(i, (k, v))| {
                let is_sliding = self
                    .layer_types
                    .get(i)
                    .map(|t| t == "sliding_attention")
                    .unwrap_or(false);
                let enc_len = k.dim(2);
                if is_sliding && enc_len > keep {
                    let start = enc_len - keep;
                    (
                        ops::slice_axis(&k, 2, start, enc_len),
                        ops::slice_axis(&v, 2, start, enc_len),
                    )
                } else {
                    (k, v)
                }
            })
            .collect()
    }

    /// Differentiable **training** forward: encode `context_ids` into a
    /// read-only KV cache, then denoise the (already-noised) `canvas_ids` in one
    /// decoder pass and return softcapped canvas logits `[B, canvas, vocab]`.
    ///
    /// This is the gradient-carrying counterpart of a single `generate`
    /// denoising step and is numerically identical to the oracle
    /// `DiffusionGemmaForBlockDiffusion.forward(input_ids=context,
    /// decoder_input_ids=canvas, self_conditioning_logits=...)`: it composes the
    /// P4-verified encoder forward, the P5-verified sliding-KV truncation +
    /// decoder forward, and the tied-head fp32 softcap — without the sampling
    /// loop. The block-diffusion training objective (noise the canvas, run this,
    /// cross-entropy against the clean targets) lives in `pmetal-trainer`.
    ///
    /// `self_conditioning_logits` carries the previous step's `[B, canvas,
    /// vocab]` logits (detached during training) or `None` (step 0 / dropped).
    pub fn forward_train(
        &mut self,
        context_ids: &Array,
        canvas_ids: &Array,
        self_conditioning_logits: Option<&Array>,
    ) -> Result<Array, Exception> {
        let (_enc_hidden, kvs_full) = self.encoder.forward(context_ids)?;
        let kvs = self.truncate_sliding_kvs(kvs_full);
        let hidden = self
            .decoder
            .forward(canvas_ids, &kvs, self_conditioning_logits)?;
        Ok(self.lm_logits(&hidden))
    }

    /// Attach LoRA adapters to every encoder and decoder attention layer, for
    /// the projections named in `config.target_modules`. Adapters initialise to
    /// a no-op (`B = 0`), so `forward_train` / `generate` are numerically
    /// unchanged until the adapters are trained.
    ///
    /// The encoder and decoder trunks carry *independent* adapters — they are
    /// separate module instances in pmetal even though their base weights are
    /// tied — so a fine-tune adapts both the causal context encoder and the
    /// bidirectional denoising decoder.
    pub fn attach_lora(&mut self, config: &pmetal_core::LoraConfig) -> Result<(), Exception> {
        for layer in &mut self.encoder.layers {
            layer.self_attn.attach_lora(config)?;
        }
        for layer in &mut self.decoder.layers {
            layer.self_attn.attach_lora(config)?;
        }
        Ok(())
    }

    /// Quantize the base weights of every encoder + decoder attention projection
    /// and MoE expert block to `bits`-bit affine (group size `group_size`) for
    /// QLoRA. The frozen base then runs through MLX's fused quantized matmuls
    /// (`quantized_matmul` / `gather_qmm`) while the LoRA adapters stay in f32 —
    /// so `attach_lora` composes with this in either order, and the block-
    /// diffusion trainer's gradients still flow only to the f32 adapters.
    ///
    /// `hidden_size`, `moe_intermediate_size`, and each attention projection's
    /// input dimension must be multiples of `group_size ∈ {32, 64, 128}`; a
    /// violation returns a clean error rather than aborting.
    ///
    /// `for_training` selects the quantized experts' backward path: `true` uses
    /// the exact dequantize-to-dense forward (required for QLoRA training, since
    /// the fused `gather_qmm` has no input-activation vjp); `false` uses the
    /// fast fused `gather_qmm` inference path.
    pub fn quantize_base(
        &mut self,
        group_size: i32,
        bits: i32,
        for_training: bool,
    ) -> Result<(), Exception> {
        for layer in &mut self.encoder.layers {
            layer.self_attn.quantize_projections(group_size, bits)?;
            layer.experts.quantize(group_size, bits)?;
            layer.experts.set_dequant_backward(for_training);
        }
        for layer in &mut self.decoder.layers {
            layer.self_attn.quantize_projections(group_size, bits)?;
            layer.experts.quantize(group_size, bits)?;
            layer.experts.set_dequant_backward(for_training);
        }
        Ok(())
    }

    /// All LoRA parameters, namespaced
    /// `{encoder|decoder}.layers.{i}.self_attn.{proj}.lora_{a,b}`.
    pub fn lora_parameters(&self) -> Vec<(String, &Array)> {
        let mut out = Vec::new();
        for (tower, layers) in [
            ("encoder", &self.encoder.layers),
            ("decoder", &self.decoder.layers),
        ] {
            for (i, layer) in layers.iter().enumerate() {
                for (name, arr) in layer.self_attn.lora_parameters() {
                    out.push((format!("{tower}.layers.{i}.self_attn.{name}"), arr));
                }
            }
        }
        out
    }

    /// Mutable LoRA parameters for the optimiser (same namespacing). Two
    /// sequential loops keep the encoder/decoder mutable borrows disjoint.
    pub fn lora_parameters_mut(&mut self) -> Vec<(String, &mut Array)> {
        let mut out = Vec::new();
        for (i, layer) in self.encoder.layers.iter_mut().enumerate() {
            for (name, arr) in layer.self_attn.lora_parameters_mut() {
                out.push((format!("encoder.layers.{i}.self_attn.{name}"), arr));
            }
        }
        for (i, layer) in self.decoder.layers.iter_mut().enumerate() {
            for (name, arr) in layer.self_attn.lora_parameters_mut() {
                out.push((format!("decoder.layers.{i}.self_attn.{name}"), arr));
            }
        }
        out
    }

    /// Generate by block-autoregressive discrete diffusion (batch size 1).
    ///
    /// For each canvas block: encode the running sequence into a read-only KV
    /// cache, denoise a fresh uniform-random canvas over `max_denoising_steps`
    /// (decoder → softcapped logits → linear-temperature → multinomial proposal
    /// → entropy-bound accept → renoise rejected; processed logits feed the
    /// next step's self-conditioning), stopping early once stable & confident,
    /// then append the argmax canvas. Stops at `max_new_tokens` or when the
    /// canvas contains an EOS token.
    ///
    /// The trajectory is stochastic (multinomial + uniform renoise), seeded via
    /// `seed`; the per-step *transforms* are the parity-verified deterministic
    /// functions above.
    pub fn generate(
        &mut self,
        input_ids: &Array,
        config: &DiffusionGemmaGenerationConfig,
        seed: u64,
    ) -> Result<Array, Exception> {
        pmetal_bridge::compat::random::seed(seed);
        let canvas = self.canvas_length;
        let vocab = self.vocab_size;
        let max_new_canvases = (config.max_new_tokens + canvas - 1) / canvas;

        let mut sequence = input_ids.clone();
        for _block in 0..max_new_canvases {
            let (_enc_hidden, kvs_full) = self.encoder.forward(&sequence)?;
            let kvs = self.truncate_sliding_kvs(kvs_full);

            let mut current = pmetal_bridge::compat::random::randint(
                0,
                vocab,
                &[1, canvas],
                pmetal_bridge::compat::Dtype::Int32,
            );
            let mut argmax_canvas = current.clone();
            let mut sc_logits: Option<Array> = None;
            let mut stopper = StableConfidentStopper::new(
                config.stability_threshold,
                config.confidence_threshold,
            );

            for cur_step in (1..=config.max_denoising_steps).rev() {
                let hidden = self.decoder.forward(&current, &kvs, sc_logits.as_ref())?;
                let raw = self.lm_logits(&hidden);
                let processed = linear_temperature(
                    &raw,
                    cur_step,
                    config.t_min,
                    config.t_max,
                    config.max_denoising_steps,
                );
                argmax_canvas = ops::argmax(&processed, -1)
                    .as_dtype(pmetal_bridge::compat::Dtype::Int32.as_i32());

                // Multinomial proposal: categorical over processed logits is
                // equivalent to multinomial over softmax(processed).
                let denoiser = pmetal_bridge::compat::random::categorical(&processed, -1)
                    .as_dtype(pmetal_bridge::compat::Dtype::Int32.as_i32());

                let (accepted, mask) =
                    entropy_bound_accept(&processed, &current, &denoiser, config.entropy_bound);
                // Renoise: rejected positions get fresh uniform tokens.
                let renoise_mask = ops::logical_not(&mask);
                let random_canvas = pmetal_bridge::compat::random::randint(
                    0,
                    vocab,
                    &[1, canvas],
                    pmetal_bridge::compat::Dtype::Int32,
                );
                current = ops::where_fn(&renoise_mask, &random_canvas, &accepted);

                let stop = stopper.should_stop(&argmax_canvas, &processed);
                sc_logits = Some(processed);
                if stop {
                    break;
                }
            }

            sequence = ops::concatenate_axis(&[&sequence, &argmax_canvas], -1);
            if canvas_contains_eos(&argmax_canvas, &config.eos_token_ids) {
                break;
            }
        }
        Ok(sequence)
    }
}

/// True if any token in the `[1, canvas]` argmax canvas is an EOS id.
fn canvas_contains_eos(argmax_canvas: &Array, eos_token_ids: &[i32]) -> bool {
    if eos_token_ids.is_empty() {
        return false;
    }
    let ids = {
        let a = argmax_canvas.as_type::<i32>();
        a.eval();
        a.as_slice::<i32>().to_vec()
    };
    ids.iter().any(|t| eos_token_ids.contains(t))
}

// ----------------------------------------------------------------------------
// Weight loading
// ----------------------------------------------------------------------------

fn dg_load_linear(
    linear: &mut nn::Linear,
    weights: &HashMap<String, Array>,
    prefix: &str,
    report: &mut LoadReport,
) {
    if let Some(w) = weights.get(&format!("{prefix}.weight")) {
        linear.weight = Param::new(w.clone());
        report.loaded += 1;
    } else {
        report.skipped.push(format!("{prefix}.weight"));
    }
}

fn dg_load_norm(
    slot: &mut Param<Array>,
    weights: &HashMap<String, Array>,
    key: &str,
    report: &mut LoadReport,
) {
    if let Some(w) = weights.get(key) {
        *slot = Param::new(w.clone());
        report.loaded += 1;
    } else {
        report.skipped.push(key.to_string());
    }
}

fn dg_load_param(
    slot: &mut Param<Array>,
    weights: &HashMap<String, Array>,
    key: &str,
    report: &mut LoadReport,
) {
    if let Some(w) = weights.get(key) {
        *slot = Param::new(w.clone());
        report.loaded += 1;
    } else {
        report.skipped.push(key.to_string());
    }
}

/// Load HF `DiffusionGemmaEncoderTextModel` weights into a
/// [`DiffusionGemmaEncoderModel`]. Keys are the raw transformers state-dict
/// names (`embed_tokens.weight`, `layers.{i}.…`, `norm.weight`) — i.e. the
/// encoder text tower as its own root, with no `model.` prefix. Both the
/// fused expert tensors and `nn.Linear` weights are stored in the same layout
/// the Python checkpoint uses, so no transpose happens at load time.
pub fn load_diffusion_gemma_encoder_weights(
    model: &mut DiffusionGemmaEncoderModel,
    weights: &HashMap<String, Array>,
) -> Result<LoadReport, Exception> {
    let mut report = LoadReport::default();

    dg_load_norm(
        &mut model.embed_tokens.weight,
        weights,
        "embed_tokens.weight",
        &mut report,
    );
    dg_load_norm(&mut model.norm.weight, weights, "norm.weight", &mut report);

    for (i, layer) in model.layers.iter_mut().enumerate() {
        dg_load_text_layer(layer, weights, &format!("layers.{i}"), &mut report);
    }

    Ok(report)
}

/// Load a single [`DiffusionGemmaTextLayer`] (the 7 RMSNorms, attention
/// projections + QK norms, dense MLP, router, fused experts, and the
/// `layer_scalar`). Shared by the encoder and decoder loaders — encoder and
/// decoder layers are weight-identical (tied in the full model), differing
/// only in how attention reads the KV cache.
fn dg_load_text_layer(
    layer: &mut DiffusionGemmaTextLayer,
    weights: &HashMap<String, Array>,
    prefix: &str,
    report: &mut LoadReport,
) {
    let p = prefix;

    // The seven RMSNorms.
    dg_load_norm(
        &mut layer.input_layernorm.weight,
        weights,
        &format!("{p}.input_layernorm.weight"),
        report,
    );
    dg_load_norm(
        &mut layer.post_attention_layernorm.weight,
        weights,
        &format!("{p}.post_attention_layernorm.weight"),
        report,
    );
    dg_load_norm(
        &mut layer.pre_feedforward_layernorm.weight,
        weights,
        &format!("{p}.pre_feedforward_layernorm.weight"),
        report,
    );
    dg_load_norm(
        &mut layer.post_feedforward_layernorm.weight,
        weights,
        &format!("{p}.post_feedforward_layernorm.weight"),
        report,
    );
    dg_load_norm(
        &mut layer.post_feedforward_layernorm_1.weight,
        weights,
        &format!("{p}.post_feedforward_layernorm_1.weight"),
        report,
    );
    dg_load_norm(
        &mut layer.pre_feedforward_layernorm_2.weight,
        weights,
        &format!("{p}.pre_feedforward_layernorm_2.weight"),
        report,
    );
    dg_load_norm(
        &mut layer.post_feedforward_layernorm_2.weight,
        weights,
        &format!("{p}.post_feedforward_layernorm_2.weight"),
        report,
    );

    // Attention projections + QK norms. Full layers have no `v_proj`.
    dg_load_linear(
        &mut layer.self_attn.q_proj,
        weights,
        &format!("{p}.self_attn.q_proj"),
        report,
    );
    dg_load_linear(
        &mut layer.self_attn.k_proj,
        weights,
        &format!("{p}.self_attn.k_proj"),
        report,
    );
    if let Some(ref mut v) = layer.self_attn.v_proj {
        dg_load_linear(v, weights, &format!("{p}.self_attn.v_proj"), report);
    }
    dg_load_linear(
        &mut layer.self_attn.o_proj,
        weights,
        &format!("{p}.self_attn.o_proj"),
        report,
    );
    dg_load_norm(
        &mut layer.self_attn.q_norm.weight,
        weights,
        &format!("{p}.self_attn.q_norm.weight"),
        report,
    );
    dg_load_norm(
        &mut layer.self_attn.k_norm.weight,
        weights,
        &format!("{p}.self_attn.k_norm.weight"),
        report,
    );

    // Dense MLP.
    dg_load_linear(
        &mut layer.mlp.gate_proj,
        weights,
        &format!("{p}.mlp.gate_proj"),
        report,
    );
    dg_load_linear(
        &mut layer.mlp.up_proj,
        weights,
        &format!("{p}.mlp.up_proj"),
        report,
    );
    dg_load_linear(
        &mut layer.mlp.down_proj,
        weights,
        &format!("{p}.mlp.down_proj"),
        report,
    );

    // Router.
    dg_load_linear(
        &mut layer.router.proj,
        weights,
        &format!("{p}.router.proj"),
        report,
    );
    dg_load_param(
        &mut layer.router.scale,
        weights,
        &format!("{p}.router.scale"),
        report,
    );
    dg_load_param(
        &mut layer.router.per_expert_scale,
        weights,
        &format!("{p}.router.per_expert_scale"),
        report,
    );

    // Experts (fused 3-D tensors, stored in HF layout).
    dg_load_param(
        &mut layer.experts.gate_up_proj,
        weights,
        &format!("{p}.experts.gate_up_proj"),
        report,
    );
    dg_load_param(
        &mut layer.experts.down_proj,
        weights,
        &format!("{p}.experts.down_proj"),
        report,
    );

    // Per-layer scalar (persistent buffer in HF).
    dg_load_param(
        &mut layer.layer_scalar,
        weights,
        &format!("{p}.layer_scalar"),
        report,
    );
}

/// Load HF `DiffusionGemmaDecoderModel` weights into a
/// [`DiffusionGemmaDecoderModel`]. The decoder shares its trunk (`layers.*`,
/// `embed_tokens`, `norm`) with the encoder (tied in the full model) and adds
/// the decoder-only `self_conditioning` block (`post_norm` is weight-less, so
/// it is absent from the checkpoint).
pub fn load_diffusion_gemma_decoder_weights(
    model: &mut DiffusionGemmaDecoderModel,
    weights: &HashMap<String, Array>,
) -> Result<LoadReport, Exception> {
    let mut report = LoadReport::default();

    dg_load_norm(
        &mut model.embed_tokens.weight,
        weights,
        "embed_tokens.weight",
        &mut report,
    );
    dg_load_norm(&mut model.norm.weight, weights, "norm.weight", &mut report);

    dg_load_norm(
        &mut model.self_conditioning.pre_norm.weight,
        weights,
        "self_conditioning.pre_norm.weight",
        &mut report,
    );
    dg_load_linear(
        &mut model.self_conditioning.gate_proj,
        weights,
        "self_conditioning.gate_proj",
        &mut report,
    );
    dg_load_linear(
        &mut model.self_conditioning.up_proj,
        weights,
        "self_conditioning.up_proj",
        &mut report,
    );
    dg_load_linear(
        &mut model.self_conditioning.down_proj,
        weights,
        "self_conditioning.down_proj",
        &mut report,
    );

    for (i, layer) in model.layers.iter_mut().enumerate() {
        dg_load_text_layer(layer, weights, &format!("layers.{i}"), &mut report);
    }

    Ok(report)
}

/// Parse a released `DiffusionGemmaConfig` (`model_type: diffusion_gemma`) JSON
/// string into the text-tower [`DiffusionGemmaTextConfig`].
///
/// The checkpoint nests the text-tower fields under `text_config` and carries
/// `canvas_length` (the block-diffusion canvas size) at the top level. This
/// unwraps the former and folds the latter in. A bare `DiffusionGemmaTextConfig`
/// (no `text_config` key — used by the synthetic parity fixtures) is also
/// accepted. Vision / audio sub-configs are ignored (text path only).
pub fn parse_diffusion_gemma_config(
    config_content: &str,
) -> Result<DiffusionGemmaTextConfig, Exception> {
    let root: serde_json::Value =
        serde_json::from_str(config_content).map_err(|e| Exception::custom(e.to_string()))?;
    let text_json = root.get("text_config").unwrap_or(&root);
    let mut config: DiffusionGemmaTextConfig =
        serde_json::from_value(text_json.clone()).map_err(|e| Exception::custom(e.to_string()))?;
    // `canvas_length` lives on the outer config; mirror it into the text config
    // (which owns the decoder's canvas geometry).
    if let Some(canvas) = root.get("canvas_length").and_then(|v| v.as_i64()) {
        config.canvas_length = canvas as i32;
    }
    Ok(config)
}

/// Extract the vision sub-config + `image_token_id` from a full DiffusionGemma
/// config, or `None` for a text-only checkpoint (absent / null `vision_config`).
/// The checkpoint expresses the vision RoPE base as a `rope_parameters:
/// {rope_theta}` dict, which is folded into the flat `rope_theta` field.
pub fn parse_diffusion_gemma_vision_config(
    config_content: &str,
) -> Result<Option<(Gemma4VisionConfig, i32)>, Exception> {
    let root: serde_json::Value =
        serde_json::from_str(config_content).map_err(|e| Exception::custom(e.to_string()))?;
    let vision_json = match root.get("vision_config") {
        Some(v) if !v.is_null() => v,
        _ => return Ok(None),
    };
    let mut vision: Gemma4VisionConfig = serde_json::from_value(vision_json.clone())
        .map_err(|e| Exception::custom(e.to_string()))?;
    if let Some(theta) = vision_json
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
    {
        vision.rope_theta = theta as f32;
    }
    let image_token_id = root
        .get("image_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(DIFFUSION_GEMMA_IMAGE_TOKEN_ID as i64) as i32;
    Ok(Some((vision, image_token_id)))
}

/// Load the encoder's vision backbone (`vision_tower.*` + `embed_vision.*`) from
/// checkpoint `weights`, accepting either bare (`vision_tower.…`) or
/// full-checkpoint (`model.encoder.vision_tower.…`) key roots. A no-op unless
/// the encoder has a tower attached ([`DiffusionGemmaEncoderModel::attach_vision`]).
fn load_encoder_vision(
    encoder: &mut DiffusionGemmaEncoderModel,
    weights: &HashMap<String, Array>,
    report: &mut LoadReport,
) -> Result<(), Exception> {
    if encoder.vision_tower.is_none() {
        return Ok(());
    }
    let mut vision: HashMap<String, Array> = HashMap::new();
    let mut embed_vision: HashMap<String, Array> = HashMap::new();
    for (key, value) in weights {
        let k = key.strip_prefix("model.").unwrap_or(key.as_str());
        if let Some(rest) = k
            .strip_prefix("encoder.vision_tower.")
            .or_else(|| k.strip_prefix("vision_tower."))
        {
            vision.insert(rest.to_string(), value.clone());
        } else if let Some(rest) = k
            .strip_prefix("encoder.embed_vision.")
            .or_else(|| k.strip_prefix("embed_vision."))
        {
            embed_vision.insert(rest.to_string(), value.clone());
        }
    }
    if let Some(vt) = encoder.vision_tower.as_mut() {
        let r = load_gemma4_vision_weights(vt, &vision)?;
        report.loaded += r.loaded;
        report.skipped.extend(r.skipped);
    }
    if let Some(ev) = encoder.embed_vision.as_mut() {
        ev.load_weights(&embed_vision, report);
    }
    Ok(())
}

/// Load a DiffusionGemma **multimodal** encoder (text tower + optional vision
/// tower + projector). Accepts keys rooted at the encoder — either bare
/// (`language_model.*`, `vision_tower.*`, `embed_vision.*`) or full-checkpoint
/// (`model.encoder.*`). Text loads via [`load_diffusion_gemma_encoder_weights`];
/// vision loads only when a tower is attached.
pub fn load_diffusion_gemma_encoder_multimodal_weights(
    encoder: &mut DiffusionGemmaEncoderModel,
    weights: &HashMap<String, Array>,
) -> Result<LoadReport, Exception> {
    let mut text: HashMap<String, Array> = HashMap::new();
    for (key, value) in weights {
        let k = key.strip_prefix("model.").unwrap_or(key.as_str());
        if let Some(rest) = k
            .strip_prefix("encoder.language_model.")
            .or_else(|| k.strip_prefix("language_model."))
        {
            text.insert(rest.to_string(), value.clone());
        }
    }
    let mut report = load_diffusion_gemma_encoder_weights(encoder, &text)?;
    load_encoder_vision(encoder, weights, &mut report)?;
    Ok(report)
}

/// Load a full `DiffusionGemmaForBlockDiffusion` checkpoint into the model.
///
/// The HF checkpoint stores the trunk once and ties three ways:
/// `lm_head.weight` ⇄ `model.decoder.embed_tokens.weight`, and the whole
/// encoder text tower (`model.encoder.language_model.{embed_tokens,layers,norm}`)
/// ⇄ the decoder trunk (`model.decoder.{embed_tokens,layers,norm}`). Only the
/// decoder carries the extra `self_conditioning.*` block.
///
/// Because tying lets `save_pretrained` keep a single physical copy of each
/// shared tensor, this remapper resolves every trunk slot from whichever copy
/// is present — decoder first, then the encoder copy, then (for the embedding
/// only) the tied `lm_head.weight` — and feeds the reconstructed per-tower
/// state dicts to [`load_diffusion_gemma_encoder_weights`] /
/// [`load_diffusion_gemma_decoder_weights`]. The two towers share the same
/// (reference-counted) tensors, matching the model's weight tying.
pub fn load_diffusion_gemma_weights(
    model: &mut DiffusionGemmaForBlockDiffusion,
    weights: &HashMap<String, Array>,
) -> Result<LoadReport, Exception> {
    // Trunk slots keyed relative to a tower root (e.g.
    // `layers.3.self_attn.q_proj.weight`, `embed_tokens.weight`, `norm.weight`).
    let mut trunk: HashMap<String, Array> = HashMap::new(); // decoder copies (authoritative)
    let mut trunk_enc: HashMap<String, Array> = HashMap::new(); // encoder copies (fallback)
    let mut self_cond: HashMap<String, Array> = HashMap::new(); // decoder-only
    let mut lm_head: Option<Array> = None;

    for (key, value) in weights {
        let k = key.strip_prefix("model.").unwrap_or(key.as_str());
        if let Some(rest) = k.strip_prefix("encoder.language_model.") {
            trunk_enc.insert(rest.to_string(), value.clone());
        } else if let Some(rest) = k.strip_prefix("decoder.") {
            if rest.starts_with("self_conditioning.") {
                self_cond.insert(rest.to_string(), value.clone());
            } else {
                trunk.insert(rest.to_string(), value.clone());
            }
        } else if k == "lm_head.weight" {
            lm_head = Some(value.clone());
        }
        // Other keys (vision / audio towers) are intentionally dropped.
    }

    // Fill any trunk gaps from the encoder copy (tied ⇒ identical tensors).
    for (slot, value) in trunk_enc {
        trunk.entry(slot).or_insert(value);
    }
    // The embedding may only survive physically as the tied `lm_head.weight`.
    if !trunk.contains_key("embed_tokens.weight") {
        if let Some(w) = lm_head {
            trunk.insert("embed_tokens.weight".to_string(), w);
        }
    }

    // Encoder state dict = the resolved trunk; decoder = trunk + self-conditioning.
    let encoder_weights = trunk.clone();
    let mut decoder_weights = trunk;
    decoder_weights.extend(self_cond);

    let enc_report = load_diffusion_gemma_encoder_weights(&mut model.encoder, &encoder_weights)?;
    let dec_report = load_diffusion_gemma_decoder_weights(&mut model.decoder, &decoder_weights)?;

    let mut report = LoadReport {
        loaded: enc_report.loaded + dec_report.loaded,
        skipped: enc_report.skipped,
    };
    report.skipped.extend(dec_report.skipped);

    // Vision backbone (encoder-only; no-op unless a tower was attached).
    load_encoder_vision(&mut model.encoder, weights, &mut report)?;
    Ok(report)
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

    fn tiny_vision_config() -> Gemma4VisionConfig {
        Gemma4VisionConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 8,
            pooling_kernel_size: 2,
            patch_size: 2,
            position_embedding_size: 32,
            rope_theta: 100.0,
            ..Default::default()
        }
    }

    #[test]
    fn multimodal_mask_is_causal_or_image_block() {
        // Image tokens at positions 2..=5 (block 0); text elsewhere (−1).
        let block_ids =
            Array::from_slice(&[-1.0f32, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], &[1, 8]);
        let (full, _sliding) = build_multimodal_masks(&block_ids, 8, Dtype::Float32);
        let mut full = full.reshape(&[8, 8]);
        full.eval();
        let v = full.to_f32_vec(64).unwrap();
        let allowed: Vec<Vec<i32>> = (0..8)
            .map(|i| {
                (0..8)
                    .map(|j| if v[i * 8 + j] > -1.0 { 1 } else { 0 })
                    .collect()
            })
            .collect();
        // Causal OR same-image-block (rows 2..=5 additionally see the whole
        // 2..=5 span, including "future" image tokens).
        let expected = vec![
            vec![1, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 1, 0, 0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1, 1, 0, 0],
            vec![1, 1, 1, 1, 1, 1, 0, 0],
            vec![1, 1, 1, 1, 1, 1, 0, 0],
            vec![1, 1, 1, 1, 1, 1, 0, 0],
            vec![1, 1, 1, 1, 1, 1, 1, 0],
            vec![1, 1, 1, 1, 1, 1, 1, 1],
        ];
        assert_eq!(allowed, expected);
    }

    #[test]
    fn merge_image_features_scatters_in_order() {
        // text embeds [1,4,2]; image mask marks positions 1 and 2.
        let text = Array::from_slice(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], &[1, 4, 2]);
        let mask = ops::greater(
            &Array::from_slice(&[0.0f32, 1.0, 1.0, 0.0], &[1, 4]),
            &Array::from_f32(0.5),
        );
        let feats = Array::from_slice(&[10.0, 10.0, 20.0, 20.0], &[2, 2]);

        let mut merged = merge_image_features(&text, &mask, &feats);
        assert_eq!(merged.shape(), &[1, 4, 2]);
        merged.eval();
        assert_eq!(
            merged.to_f32_vec(8).unwrap(),
            vec![1.0, 1.0, 10.0, 10.0, 20.0, 20.0, 4.0, 4.0]
        );
    }

    #[test]
    fn forward_multimodal_runs_and_shapes() {
        let cfg = tiny_config();
        let n_layers = cfg.num_hidden_layers as usize;
        let hidden = cfg.hidden_size;
        let mut enc = DiffusionGemmaEncoderModel::new(cfg).unwrap();
        enc.attach_vision(&tiny_vision_config(), 5).unwrap();

        // 4x4 patch grid → pool k=2 → 4 soft tokens → 4 image tokens (id 5).
        let grid = 4;
        let n = grid * grid;
        let patch_dim = 3 * 2 * 2;
        let pixel_values = pmetal_bridge::compat::random::uniform_f32(&[1, n, patch_dim]);
        let mut coords = Vec::with_capacity((n * 2) as usize);
        for y in 0..grid {
            for x in 0..grid {
                coords.push(x);
                coords.push(y);
            }
        }
        let position_ids = Array::from_slice(&coords, &[1, n, 2]);
        let input_ids = Array::from_slice(&[1, 2, 5, 5, 5, 5, 3], &[1, 7]);

        let (mut out, kvs) = enc
            .forward_multimodal(&input_ids, &pixel_values, &position_ids, true)
            .unwrap();
        assert_eq!(out.shape(), &[1, 7, hidden]);
        assert_eq!(kvs.len(), n_layers);
        out.eval();
        let v = out.to_f32_vec((7 * hidden) as usize).unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "multimodal output non-finite"
        );
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
    fn forward_train_shape_and_finite() {
        let cfg = tiny_config();
        let vocab = cfg.vocab_size;
        let canvas = cfg.canvas_length;
        let mut model = DiffusionGemmaForBlockDiffusion::new(cfg).unwrap();

        // Deterministic context [1, 5] and canvas [1, canvas], all ids < vocab.
        let ctx_ids: Vec<i32> = (0..5).map(|i| (i * 3 + 1) % vocab).collect();
        let context = Array::from_slice(&ctx_ids, &[1, 5]);
        let canvas_ids: Vec<i32> = (0..canvas).map(|i| (i * 7 + 2) % vocab).collect();
        let canvas_arr = Array::from_slice(&canvas_ids, &[1, canvas]);

        let mut logits = model.forward_train(&context, &canvas_arr, None).unwrap();
        assert_eq!(logits.shape(), &[1, canvas, vocab]);
        let v = logits.to_f32_vec((canvas * vocab) as usize).unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "forward_train produced non-finite logits"
        );
    }

    /// A QLoRA-quantized DiffusionGemma (4-bit base + f32 LoRA) must run
    /// `forward_train` end-to-end and produce finite, correctly-shaped logits.
    /// Uses a config whose `hidden`/`moe_intermediate` are multiples of the
    /// minimum group size (32).
    #[test]
    #[serial]
    fn quantize_base_forward_train_finite() {
        use pmetal_core::LoraConfig;
        let cfg = DiffusionGemmaTextConfig {
            moe_intermediate_size: 32,
            ..tiny_config()
        };
        let vocab = cfg.vocab_size;
        let canvas = cfg.canvas_length;
        let mut model = DiffusionGemmaForBlockDiffusion::new(cfg).unwrap();

        let lora_cfg = LoraConfig {
            r: 4,
            alpha: 8.0,
            target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        };
        model.attach_lora(&lora_cfg).unwrap();
        // for_training = false exercises the fused gather_qmm inference path.
        model.quantize_base(32, 4, false).unwrap();

        let ctx_ids: Vec<i32> = (0..5).map(|i| (i * 3 + 1) % vocab).collect();
        let context = Array::from_slice(&ctx_ids, &[1, 5]);
        let canvas_ids: Vec<i32> = (0..canvas).map(|i| (i * 7 + 2) % vocab).collect();
        let canvas_arr = Array::from_slice(&canvas_ids, &[1, canvas]);

        let mut logits = model.forward_train(&context, &canvas_arr, None).unwrap();
        assert_eq!(logits.shape(), &[1, canvas, vocab]);
        let v = logits.to_f32_vec((canvas * vocab) as usize).unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "quantized forward_train produced non-finite logits"
        );
    }

    #[test]
    #[serial]
    fn lora_attach_is_noop_until_trained() {
        use pmetal_core::LoraConfig;
        let cfg = tiny_config();
        let vocab = cfg.vocab_size;
        let canvas = cfg.canvas_length;
        let mut model = DiffusionGemmaForBlockDiffusion::new(cfg).unwrap();

        let ctx_ids: Vec<i32> = (0..5).map(|i| (i * 3 + 1) % vocab).collect();
        let context = Array::from_slice(&ctx_ids, &[1, 5]);
        let canvas_ids: Vec<i32> = (0..canvas).map(|i| (i * 7 + 2) % vocab).collect();
        let canvas_arr = Array::from_slice(&canvas_ids, &[1, canvas]);

        let mut before = model.forward_train(&context, &canvas_arr, None).unwrap();
        let before_v = before.to_f32_vec((canvas * vocab) as usize).unwrap();
        assert!(
            model.lora_parameters().is_empty(),
            "no adapters before attach"
        );

        let mut lora_cfg = LoraConfig::default();
        lora_cfg.r = 4;
        lora_cfg.alpha = 8.0;
        lora_cfg.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        model.attach_lora(&lora_cfg).unwrap();

        let n = model.lora_parameters().len();
        assert!(n > 0, "expected LoRA params after attach");
        assert_eq!(model.lora_parameters_mut().len(), n, "mut count matches");

        // B is zero-initialised, so the adapters must not change the output yet.
        let mut after = model.forward_train(&context, &canvas_arr, None).unwrap();
        let after_v = after.to_f32_vec((canvas * vocab) as usize).unwrap();
        let max_diff = before_v
            .iter()
            .zip(&after_v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "attaching zero-init LoRA changed the output by {max_diff}"
        );
    }

    #[test]
    #[serial]
    fn router_shapes() {
        let cfg = tiny_config();
        let mut router = DiffusionGemmaRouter::new(
            cfg.hidden_size,
            cfg.num_experts,
            cfg.top_k_experts,
            cfg.rms_norm_eps,
        )
        .unwrap();
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
        let mut router = DiffusionGemmaRouter::new(
            cfg.hidden_size,
            cfg.num_experts,
            cfg.top_k_experts,
            cfg.rms_norm_eps,
        )
        .unwrap();
        let mut experts =
            DiffusionGemmaExperts::new(cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size)
                .unwrap();
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

    #[test]
    #[serial]
    fn decoder_forward_shape() {
        let cfg = tiny_config();
        // Encoder produces the read-only KV the decoder reads.
        let mut encoder = DiffusionGemmaEncoderModel::new(cfg.clone()).unwrap();
        let prompt = Array::from_slice(&[1i32, 2, 3, 4, 5, 6], &[1, 6]);
        let (_enc_hidden, kvs) = encoder.forward(&prompt).unwrap();

        let mut decoder = DiffusionGemmaDecoderModel::new(cfg.clone()).unwrap();
        let canvas_ids = Array::from_slice(&[7i32, 8, 9, 10], &[1, 4]);
        // Step 0: no self-conditioning signal.
        let out = decoder.forward(&canvas_ids, &kvs, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, cfg.hidden_size]);

        // Step >0: with self-conditioning logits over the vocab.
        let sc_logits = pmetal_bridge::compat::random::uniform_range(
            -1.0,
            1.0,
            &[1, 4, cfg.vocab_size],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let out2 = decoder
            .forward(&canvas_ids, &kvs, Some(&sc_logits))
            .unwrap();
        assert_eq!(out2.shape(), &[1, 4, cfg.hidden_size]);
    }

    #[test]
    #[serial]
    fn entropy_bound_accept_basic() {
        // One near-deterministic position (entropy ≈ 0) and two uniform
        // (entropy ≈ log 8). Sorted ascending the entropies are ≈ [0, 2.08,
        // 2.08]; the acceptance test `cum_entropy_excluding_self <= 0.1` gives
        // excl ≈ [0, 0, 2.08], so the lowest slot and the next (preceded only
        // by the ≈0 slot) accept, and the last (preceded by 2.08) rejects.
        let vocab = 8;
        let mut rows: Vec<f32> = vec![10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0];
        rows.extend_from_slice(&[0.0; 8]); // uniform (entropy ≈ log 8)
        rows.extend_from_slice(&[0.0; 8]); // uniform (entropy ≈ log 8)
        let logits = Array::from_slice(&rows, &[1, 3, vocab]);
        let current = Array::from_slice(&[100i32, 101, 102], &[1, 3]);
        let denoiser = Array::from_slice(&[200i32, 201, 202], &[1, 3]);
        let (accepted, mask) = entropy_bound_accept(&logits, &current, &denoiser, 0.1);

        let mask_v: Vec<f32> = {
            let m = mask.as_type::<f32>();
            m.eval();
            m.as_slice::<f32>().to_vec()
        };
        assert_eq!(mask_v, vec![1.0, 1.0, 0.0]);
        let acc_v: Vec<i32> = {
            let a = accepted.as_type::<i32>();
            a.eval();
            a.as_slice::<i32>().to_vec()
        };
        // Accepted slots take the denoiser token; rejected keeps current.
        assert_eq!(acc_v, vec![200, 201, 102]);
    }

    #[test]
    #[serial]
    fn generate_structural_and_deterministic() {
        let cfg = tiny_config();
        let mut model = DiffusionGemmaForBlockDiffusion::new(cfg.clone()).unwrap();
        let prompt = Array::from_slice(&[1i32, 2, 3, 4, 5], &[1, 5]);
        let gen_cfg = DiffusionGemmaGenerationConfig {
            max_new_tokens: cfg.canvas_length,
            max_denoising_steps: 3,
            eos_token_ids: vec![], // no early EOS stop
            ..Default::default()
        };

        let out = model.generate(&prompt, &gen_cfg, 7).unwrap();
        // One canvas block appended (max_new_tokens == canvas_length).
        assert_eq!(out.shape(), &[1, 5 + cfg.canvas_length]);

        // Same seed → identical trajectory.
        let out_a = model.generate(&prompt, &gen_cfg, 42).unwrap();
        let out_b = model.generate(&prompt, &gen_cfg, 42).unwrap();
        let va: Vec<f32> = {
            let a = out_a.as_type::<f32>();
            a.eval();
            a.as_slice::<f32>().to_vec()
        };
        let vb: Vec<f32> = {
            let b = out_b.as_type::<f32>();
            b.eval();
            b.as_slice::<f32>().to_vec()
        };
        assert_eq!(
            va, vb,
            "generation must be deterministic under a fixed seed"
        );
    }
}
