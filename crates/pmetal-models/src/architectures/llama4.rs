//! Llama 4 architecture with Mixture of Experts and iRoPE.
//!
//! Key features:
//! - **iRoPE**: Interleaved RoPE/NoPE layers for long context (10M+ tokens)
//! - **MoE with shared expert**: Each token routed to 1 expert + shared expert
//! - **Interleaved MoE/Dense**: Scout is full MoE, Maverick alternates
//! - **QK norm**: Layer normalization on Q and K for stable attention
//! - **Temperature scaling**: Dynamic attention scaling for long sequences
//!
//! Variants:
//! - **Llama 4 Scout**: 109B total params (16 experts), 17B active, 10M context
//! - **Llama 4 Maverick**: 402B total params (128 experts), 17B active, 1M context
use pmetal_runtime::compat::{
    Array, Dtype, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, nn, ops,
};
use pmetal_runtime::decode::last_token_hidden;
use pmetal_runtime::impl_module_params;
use pmetal_runtime::llama4_native::route_logits;
use pmetal_runtime::native_moe;
use pmetal_runtime::native_weight::LayerWeight;

use pmetal_cache::kv_cache::KVCache;
use pmetal_runtime::nn::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopePositions, rope_embedding},
};
use pmetal_runtime::rope::{RopeConfig, RotaryEmbedding};
use serde::{Deserialize, Serialize};

use crate::checkpointing::checkpointed_layer;
use crate::traits::ModelConfig;

/// Llama 4 text configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Llama4TextConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    /// Intermediate size for MLP layers (distinct from MoE intermediate size).
    #[serde(default = "default_intermediate_size_mlp")]
    pub intermediate_size_mlp: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // MoE configuration
    /// Number of experts per token (typically 1).
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    /// Total number of routed experts.
    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: i32,
    /// Step for interleaving MoE layers (1 = every layer, 2 = every other layer).
    #[serde(default = "default_interleave_moe_layer_step")]
    pub interleave_moe_layer_step: i32,
    /// Specific layers that are MoE (if set, overrides interleave_moe_layer_step).
    #[serde(default)]
    pub moe_layers: Option<Vec<i32>>,

    // iRoPE configuration
    /// Interval for NoPE layers (e.g., 4 = NoPE every 4th layer).
    #[serde(default = "default_no_rope_layer_interval")]
    pub no_rope_layer_interval: i32,
    /// Explicit list of which layers use RoPE (1) vs NoPE (0).
    #[serde(default)]
    pub no_rope_layers: Option<Vec<i32>>,
    /// Attention chunk size for RoPE layers.
    #[serde(default = "default_attention_chunk_size")]
    pub attention_chunk_size: i32,

    // Attention configuration
    /// Whether to use QK normalization.
    #[serde(default = "default_use_qk_norm")]
    pub use_qk_norm: bool,
    /// Whether to use temperature tuning for long context.
    #[serde(default = "default_attn_temperature_tuning")]
    pub attn_temperature_tuning: bool,
    /// Floor scale for temperature computation.
    #[serde(default = "default_floor_scale")]
    pub floor_scale: i32,
    /// Attention scale factor.
    #[serde(default = "default_attn_scale")]
    pub attn_scale: f32,

    /// Router auxiliary loss coefficient for load balancing.
    #[serde(default = "default_router_aux_loss_coef")]
    pub router_aux_loss_coef: f32,

    /// RoPE scaling (Scout: `rope_type: "llama3"`, factor 16), read by
    /// [`pmetal_runtime::rope`].
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// transformers v5's spelling of the same, with `rope_theta` inside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<serde_json::Value>,
}

fn default_intermediate_size_mlp() -> i32 {
    16384
}
fn default_num_experts_per_tok() -> i32 {
    1
}
fn default_num_local_experts() -> i32 {
    16
}
fn default_interleave_moe_layer_step() -> i32 {
    1
}
fn default_no_rope_layer_interval() -> i32 {
    4
}
fn default_attention_chunk_size() -> i32 {
    8192
}
fn default_use_qk_norm() -> bool {
    true
}
fn default_attn_temperature_tuning() -> bool {
    true
}
fn default_floor_scale() -> i32 {
    8192
}
fn default_attn_scale() -> f32 {
    0.1
}
fn default_router_aux_loss_coef() -> f32 {
    0.001
}

impl Default for Llama4TextConfig {
    fn default() -> Self {
        // Default for Llama 4 Scout 109B
        Self {
            vocab_size: 202048,
            hidden_size: 5120,
            intermediate_size: 8192,
            intermediate_size_mlp: 16384,
            num_hidden_layers: 48,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 131072,
            tie_word_embeddings: false,
            num_experts_per_tok: 1,
            num_local_experts: 16,
            interleave_moe_layer_step: 1,
            moe_layers: None,
            no_rope_layer_interval: 4,
            no_rope_layers: None,
            attention_chunk_size: 8192,
            use_qk_norm: true,
            attn_temperature_tuning: true,
            floor_scale: 8192,
            attn_scale: 0.1,
            router_aux_loss_coef: 0.001,
            rope_scaling: None,
            rope_parameters: None,
        }
    }
}

impl Llama4TextConfig {
    /// Check if a given layer is an MoE layer.
    pub fn is_moe_layer(&self, layer_idx: i32) -> bool {
        if let Some(ref moe_layers) = self.moe_layers {
            moe_layers.contains(&layer_idx)
        } else {
            // The LAST layer in each interleave group is MoE (HF default
            // `moe_layers = range(step-1, n, step)`).
            // For step == 1 every layer is MoE; for step == 2 the odd layers are.
            layer_idx % self.interleave_moe_layer_step == self.interleave_moe_layer_step - 1
        }
    }

    /// Check if a layer uses RoPE (true) or NoPE (false).
    pub fn uses_rope(&self, layer_idx: i32) -> bool {
        if let Some(ref no_rope_layers) = self.no_rope_layers {
            if (layer_idx as usize) < no_rope_layers.len() {
                return no_rope_layers[layer_idx as usize] == 1;
            }
        }
        // NoPE every no_rope_layer_interval layers. HF uses the 1-based
        // index `(layer_idx + 1) % interval != 0`, so NoPE lands on layers
        // 3, 7, 11, ... (not 0, 4, 8, ...).
        (layer_idx + 1) % self.no_rope_layer_interval != 0
    }
}

impl ModelConfig for Llama4TextConfig {
    fn model_type(&self) -> &str {
        "llama4"
    }
    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }
    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.head_dim
    }
    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }
    fn max_position_embeddings(&self) -> i32 {
        self.max_position_embeddings
    }
    fn norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

// =============================================================================
// MLP, Expert and MoE Components
// =============================================================================

/// A gated MLP (SwiGLU): a dense layer's feed-forward and the MoE layers'
/// shared expert.
#[derive(Debug)]
pub struct Llama4Mlp {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(Llama4Mlp; gate_proj, up_proj, down_proj);

impl Llama4Mlp {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let gate = nn::silu(&gate);
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up);
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// The routed experts of a layer, as transformers' `Llama4TextExperts` (and so
/// every release) holds them: two stacked weights, each applied as `x @ W`.
///
/// They are the checkpoint's tensors as they are, not a copy per expert, so a
/// layer is two arrays whatever the expert count, the gate and up halves
/// come out of one gather matmul, and no weight is copied to run them.
#[derive(Debug)]
pub struct Llama4Experts {
    /// `[experts, hidden, 2 * expert_hidden]`, gate then up along the last
    /// axis (`gate_up.chunk(2, dim=-1)`).
    pub gate_up_proj: Array,
    /// `[experts, expert_hidden, hidden]`.
    pub down_proj: Array,
}
impl_module_params!(Llama4Experts; gate_up_proj, down_proj);

impl Llama4Experts {
    /// Randomly initialised experts (lazily: a checkpoint's weights replace
    /// them before anything computes them).
    pub fn new(num_experts: i32, hidden_size: i32, intermediate_size: i32) -> Self {
        let init = |shape: &[i32], fan_in: i32| {
            let bound = (fan_in as f32).sqrt().recip();
            pmetal_runtime::compat::random::uniform_range(-bound, bound, shape, Dtype::Float32)
        };
        Self {
            gate_up_proj: init(
                &[num_experts, hidden_size, 2 * intermediate_size],
                hidden_size,
            ),
            down_proj: init(
                &[num_experts, intermediate_size, hidden_size],
                intermediate_size,
            ),
        }
    }
}

/// Mixture of Experts layer with shared expert.
///
/// Parameter paths are the checkpoint's: `router.weight`,
/// `experts.{gate_up_proj,down_proj}`, `shared_expert.*`.
#[derive(Debug)]
pub struct Llama4MoE {
    /// Experts per token.
    pub top_k: i32,
    /// `hidden -> experts`. A router's argmax picks which computation runs,
    /// so no adapter attaches to it (`pmetal-lora` skips a path ending
    /// `router`).
    pub router: nn::Linear,
    pub experts: Llama4Experts,
    pub shared_expert: Llama4Mlp,
}
impl_module_params!(Llama4MoE; router, experts, shared_expert);

impl Llama4MoE {
    pub fn new(config: &Llama4TextConfig) -> Result<Self, Exception> {
        let router = nn::LinearBuilder::new(config.hidden_size, config.num_local_experts)
            .bias(false)
            .build()?;
        Ok(Self {
            top_k: config.num_experts_per_tok,
            router,
            experts: Llama4Experts::new(
                config.num_local_experts,
                config.hidden_size,
                config.intermediate_size,
            ),
            shared_expert: Llama4Mlp::new(config.hidden_size, config.intermediate_size)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape().to_vec();
        let hidden_size = *shape.last().unwrap();

        // Flatten to [total_tokens, hidden]
        let total_tokens = shape.iter().take(shape.len() - 1).product::<i32>();
        let flat_x = x.reshape(&[total_tokens, hidden_size]);

        // Llama4 routing (HF `Llama4TextMoe`): the top-k experts by RAW
        // logit, gated with `sigmoid(logit)`, not a softmax. The router is
        // `route_logits`, the native engine's.
        let router_logits = Module::forward(&mut self.router, &flat_x)?;
        let (expert_indices, gates) = route_logits(&router_logits, self.top_k, x.dtype().as_i32());

        // Llama4 scales the expert *input* by the gate (`experts(x * scores)`),
        // not the output: the expert MLP is SwiGLU (nonlinear), so the two
        // are not equivalent. `switch_glu_gated_input` is that, as grouped
        // gather matmuls with no host readback, shared with the native engine.
        let gate_up = LayerWeight::Dense(self.experts.gate_up_proj.clone());
        let down = LayerWeight::Dense(self.experts.down_proj.clone());
        let routed = native_moe::switch_glu_gated_input(
            &flat_x,
            native_moe::GateUp::Fused(&gate_up),
            &down,
            &expert_indices,
            &gates,
            |gate, up| nn::silu(gate).multiply(up),
        );

        // Add the shared expert's contribution (applied to every token).
        let shared_out = self.shared_expert.forward(&flat_x)?;
        Ok(shared_out.add(&routed).reshape(&shape))
    }
}

/// A layer's feed-forward: a dense MLP or the mixture of experts. Parameter
/// paths are the checkpoint's, `feed_forward.*`.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per layer, never moved; boxing would add an indirection to every forward"
)]
pub enum Llama4FeedForward {
    Dense(Llama4Mlp),
    Moe(Llama4MoE),
}

impl pmetal_runtime::compat::VisitLinears for Llama4FeedForward {
    fn visit_linears_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut nn::Linear)) {
        match self {
            Self::Dense(m) => m.visit_linears_mut(prefix, f),
            Self::Moe(m) => m.visit_linears_mut(prefix, f),
        }
    }
}

impl ModuleParameters for Llama4FeedForward {
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.parameters(),
            Self::Moe(m) => m.parameters(),
        }
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.trainable_parameters(),
            Self::Moe(m) => m.trainable_parameters(),
        }
    }
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(m) => m.num_parameters(),
            Self::Moe(m) => m.num_parameters(),
        }
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Dense(m) => m.parameters_mut(),
            Self::Moe(m) => m.parameters_mut(),
        }
    }
}

impl Llama4FeedForward {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Moe(moe) => moe.forward(x),
        }
    }
}

// =============================================================================
// Attention with iRoPE and QK Norm
// =============================================================================

/// Llama 4 attention with iRoPE (interleaved RoPE/NoPE) and QK norm.
#[derive(Debug)]
pub struct Llama4Attention {
    pub layer_idx: usize,
    pub uses_rope: bool,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    /// Interleaved RoPE (Llama 3 bands on Scout); unused on NoPE layers.
    pub rotary: RotaryEmbedding,
    /// RoPE layers attend within chunks of this many positions.
    pub attention_chunk_size: i32,
    pub attn_temperature_tuning: bool,
    pub floor_scale: f32,
    pub attn_scale: f32,

    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,

    // QK normalization (optional)
    pub q_norm: Option<nn::WeightlessRmsNorm>,
    pub k_norm: Option<nn::WeightlessRmsNorm>,
}
impl_module_params!(Llama4Attention; q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

impl Llama4Attention {
    pub fn new(config: &Llama4TextConfig, layer_idx: usize) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

        let q_proj = nn::LinearBuilder::new(hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, hidden_size)
            .bias(false)
            .build()?;

        let uses_rope = config.uses_rope(layer_idx as i32);

        // QK norm: weightless RMS norm applied AFTER RoPE, and only on RoPE
        // layers, at the model's `rms_norm_eps` (transformers'
        // `Llama4TextL2Norm(config.rms_norm_eps)`, the reference's
        // `rmsnorm(x, norm_eps)`). It has no weight, so it holds none: a
        // weight at ones would be a trainable parameter that a full
        // fine-tune moves off the identity.
        let (q_norm, k_norm) = if config.use_qk_norm && uses_rope {
            (
                Some(nn::WeightlessRmsNorm::new(config.rms_norm_eps)),
                Some(nn::WeightlessRmsNorm::new(config.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            layer_idx,
            uses_rope,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            rotary: crate::common::rotary_embedding(
                "llama4",
                head_dim,
                RopeConfig {
                    rope_scaling: config.rope_scaling.as_ref(),
                    rope_parameters: config.rope_parameters.as_ref(),
                    rope_theta: Some(config.rope_theta as f64),
                    max_position_embeddings: Some(config.max_position_embeddings as f64),
                    ..RopeConfig::default()
                },
                1.0,
                true,
            )?,
            attention_chunk_size: config.attention_chunk_size,
            attn_temperature_tuning: config.attn_temperature_tuning,
            floor_scale: config.floor_scale as f32,
            attn_scale: config.attn_scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        })
    }

    /// The additive `[L, K]` mask (0 / -inf, f32) that keeps each query within
    /// its chunk of `attention_chunk_size` positions; `causal` adds `k <= q`,
    /// for when no caller's mask carries it. Keys sit at `0..kv_len` with a
    /// cache, and at the queries' own positions without one. `None` when every
    /// position is in chunk 0, where the chunk changes nothing.
    fn chunk_mask(
        &self,
        positions: RopePositions<'_>,
        seq_len: i32,
        kv_len: Option<i32>,
        causal: bool,
    ) -> Option<Array> {
        let chunk = self.attention_chunk_size;
        let f32_dtype = Dtype::Float32.as_i32();
        let q = match positions {
            RopePositions::Offset(offset) => {
                if offset + seq_len <= chunk {
                    return None;
                }
                ops::arange_from(offset, offset + seq_len).as_dtype(f32_dtype)
            }
            RopePositions::Explicit(ids) => ids.as_dtype(f32_dtype),
        };
        let k = match kv_len {
            Some(n) => ops::arange_from(0, n).as_dtype(f32_dtype),
            None => q.clone(),
        };
        let (q, k) = (q.reshape(&[-1, 1]), k.reshape(&[1, -1]));
        let size = Array::from_f32(chunk as f32);
        let mut keep = ops::equal(&ops::floor(&q.divide(&size)), &ops::floor(&k.divide(&size)));
        if causal {
            keep = ops::logical_and(&keep, &ops::less_equal(&k, &q));
        }
        Some(ops::r#where(
            &keep,
            &Array::from_f32(0.0),
            &Array::from_f32(f32::NEG_INFINITY),
        ))
    }

    /// Forward pass without a cache; see [`Self::forward_with_cache`].
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None, position_ids)
    }

    /// Forward pass with an optional KV cache.
    ///
    /// Each token sits at its explicit position when `positions` (`[seq_len]`)
    /// is given (a packed batch), else at the cache's offset plus its index,
    /// so a cached decode rotates new keys where they belong. The NoPE
    /// layers' temperature reads the same positions. `mask` is an additive
    /// mask; without one, attention is causal.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let heads = |x: Array, n: i32| {
            x.reshape(&[batch, seq_len, n, self.head_dim])
                .transpose_axes(&[0, 2, 1, 3])
        };
        let q = heads(Module::forward(&mut self.q_proj, x)?, self.n_heads);
        let k = heads(Module::forward(&mut self.k_proj, x)?, self.n_kv_heads);
        let v = heads(Module::forward(&mut self.v_proj, x)?, self.n_kv_heads);

        let offset = cache
            .as_ref()
            .map_or(0, |(cache, layer)| cache.rope_offset_for(*layer));
        let rope_positions = RopePositions::resolve(positions, offset);

        // RoPE layers rotate (interleaved, Llama 3 bands when configured);
        // NoPE layers do not.
        let (mut q, mut k) = if self.uses_rope {
            (
                rope_embedding(&q, rope_positions, &self.rotary),
                rope_embedding(&k, rope_positions, &self.rotary),
            )
        } else {
            (q, k)
        };

        // QK normalization: weightless RMS norm applied AFTER RoPE, only on RoPE
        // layers (q_norm/k_norm are None on NoPE layers per construction).
        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            q = qn.forward(&q);
            k = kn.forward(&k);
        }

        // Attention temperature on the NoPE layers (arXiv 2501.19399):
        // `log(floor((p + 1) / floor_scale) + 1) * attn_scale + 1` at each
        // token's position p, in f32, back in the query dtype.
        if !self.uses_rope && self.attn_temperature_tuning {
            let p = match rope_positions {
                RopePositions::Explicit(ids) => ids.as_dtype(Dtype::Float32.as_i32()),
                RopePositions::Offset(offset) => {
                    ops::arange_from(offset, offset + seq_len).as_dtype(Dtype::Float32.as_i32())
                }
            };
            let one = Array::from_f32(1.0);
            let floored = ops::floor(&p.add(&one).divide(&Array::from_f32(self.floor_scale)));
            let scales = ops::log(&floored.add(&one))
                .multiply(&Array::from_f32(self.attn_scale))
                .add(&one)
                .reshape(&[1, 1, seq_len, 1]);
            q = q.multiply(&scales).as_dtype(x.dtype().as_i32());
        }

        let cached = cache.is_some();
        let (k, v) = match cache {
            Some((cache, layer)) => cache.update_and_fetch(layer, &k, &v)?,
            None => (k, v),
        };

        // A single decoded token on a RoPE layer attends to the keys of its
        // own chunk only, `[p - p % chunk, p]`: read exactly those, so a
        // decode costs the chunk and not the whole cache under a mask. The
        // cache keeps every position from 0.
        let mut chunk_read = false;
        let (k, v) = match rope_positions {
            RopePositions::Offset(p)
                if self.uses_rope
                    && cached
                    && seq_len == 1
                    && mask.is_none()
                    && self.attention_chunk_size > 0 =>
            {
                chunk_read = true;
                let start = (p / self.attention_chunk_size) * self.attention_chunk_size;
                if start > 0 {
                    (
                        ops::slice_axis(&k, 2, start, k.dim(2)),
                        ops::slice_axis(&v, 2, start, v.dim(2)),
                    )
                } else {
                    (k, v)
                }
            }
            _ => (k, v),
        };

        // RoPE layers attend within their chunk of `attention_chunk_size`
        // positions (transformers' `create_chunked_causal_mask`, the
        // reference's `block_pos == 0`).
        let chunk_mask = if self.uses_rope && !chunk_read {
            self.chunk_mask(
                rope_positions,
                seq_len,
                cached.then(|| k.dim(2)),
                mask.is_none(),
            )
        } else {
            None
        };
        let mask = match (mask, &chunk_mask) {
            (Some(m), Some(c)) => Some(m.add(c)),
            (Some(m), None) => Some(m.clone()),
            (None, c) => c.clone(),
        };
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });
        let output = fused_sdpa(&q, &k, &v, &attn_config, mask.as_ref())?
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        Module::forward(&mut self.o_proj, &output)
    }
}

// =============================================================================
// Decoder Layer
// =============================================================================

/// Llama 4 decoder layer (dense or MoE).
#[derive(Debug)]
pub struct Llama4DecoderLayer {
    pub layer_idx: usize,

    pub self_attn: Llama4Attention,
    /// The dense MLP or the mixture of experts: `feed_forward` in a checkpoint.
    pub feed_forward: Llama4FeedForward,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}
impl_module_params!(Llama4DecoderLayer; self_attn, feed_forward, input_layernorm, post_attention_layernorm);

impl Llama4DecoderLayer {
    pub fn new(config: &Llama4TextConfig, layer_idx: usize) -> Result<Self, Exception> {
        let self_attn = Llama4Attention::new(config, layer_idx)?;

        let feed_forward = if config.is_moe_layer(layer_idx as i32) {
            Llama4FeedForward::Moe(Llama4MoE::new(config)?)
        } else {
            Llama4FeedForward::Dense(Llama4Mlp::new(
                config.hidden_size,
                config.intermediate_size_mlp,
            )?)
        };

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            layer_idx,
            self_attn,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None, position_ids)
    }

    /// Forward pass with an optional KV cache (`(cache, layer index)`).
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Self attention with residual
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self
            .self_attn
            .forward_with_cache(&normed, mask, cache, position_ids)?;
        let h = x.add(&attn_out);

        // FFN with residual (MoE or dense)
        let normed = Module::forward(&mut self.post_attention_layernorm, &h)?;
        let ffn_out = self.feed_forward.forward(&normed)?;
        Ok(h.add(&ffn_out))
    }
}

// =============================================================================
// Full Model
// =============================================================================

/// Llama 4 text model.
#[derive(Debug)]
pub struct Llama4TextModel {
    pub config: Llama4TextConfig,

    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Llama4DecoderLayer>,
    pub norm: nn::RmsNorm,
    /// Recompute each layer's activations during the backward pass instead of
    /// holding them. Training only; see `crate::checkpointing`.
    pub grad_checkpoint: bool,
}
impl_module_params!(Llama4TextModel; embed_tokens, layers, norm);

impl Llama4TextModel {
    pub fn new(config: Llama4TextConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| Llama4DecoderLayer::new(&config, i as usize))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            grad_checkpoint: false,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None, position_ids)
    }

    /// Forward pass with an optional KV cache: each layer rotates and
    /// attends from the cache's offset, and appends its keys and values.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Hoisted: the loop below borrows `self.layers` mutably.
        let grad_checkpoint = self.grad_checkpoint;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache.as_deref_mut().map(|c| (c, idx));
            // A cache means generation, which has no backward pass for the
            // recompute to pay for.
            hidden_states = if grad_checkpoint && layer_cache.is_none() {
                checkpointed_layer(
                    layer,
                    &hidden_states,
                    mask,
                    position_ids,
                    |layer, h, mask, positions| layer.forward(h, mask, positions),
                )?
            } else {
                layer.forward_with_cache(&hidden_states, mask, layer_cache, position_ids)?
            };
        }

        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Llama 4 for causal language modeling.
#[derive(Debug)]
pub struct Llama4ForCausalLM {
    pub config: Llama4TextConfig,

    pub model: Llama4TextModel,
    /// `None` when `tie_word_embeddings` ties the head to the token
    /// embedding.
    pub lm_head: Option<nn::Linear>,
}
impl_module_params!(Llama4ForCausalLM; model, lm_head);

impl Llama4ForCausalLM {
    pub fn new(config: Llama4TextConfig) -> Result<Self, Exception> {
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };

        let model = Llama4TextModel::new(config.clone())?;

        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    /// Project hidden states to vocabulary logits, through the LM head when
    /// the checkpoint has one and the tied embedding otherwise.
    fn lm_head_forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(lm_head) => Module::forward(lm_head, hidden_states),
            None => Ok(self.model.embed_tokens.as_linear(hidden_states)),
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids, mask, position_ids)?;
        self.lm_head_forward(&hidden_states)
    }

    /// Forward pass with one rotary position per token, `[seq_len]`.
    ///
    /// Llama 4 already carries positions through its forward, so this is the
    /// same call under the name every architecture answers to.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward(input_ids, mask, positions)
    }

    /// Forward pass with an optional KV cache. New tokens rotate and attend
    /// from the cache's offset, so a prefill followed by one token at a time
    /// gives the logits of one forward over the whole sequence.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self
            .model
            .forward_with_cache(input_ids, mask, cache, None)?;
        self.lm_head_forward(&hidden_states)
    }

    /// [`forward_with_cache`](Self::forward_with_cache) returning the last
    /// position's logits, `[B, 1, vocab]`: the LM head runs on one hidden state
    /// per sequence instead of every prompt token's, for a caller that samples
    /// from the last position alone (a prefill, a decode step).
    pub fn forward_with_cache_last(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self
            .model
            .forward_with_cache(input_ids, mask, cache, None)?;
        self.lm_head_forward(&last_token_hidden(&hidden_states))
    }
}

// =============================================================================
// Checkpoint layout
// =============================================================================

/// A Llama 4 checkpoint's text-model tensors, named as this module names its
/// parameters.
///
/// The parameter tree is the checkpoint's own layout: each layer's
/// `feed_forward` is a dense MLP or a `router`, the fused `experts`
/// (`gate_up_proj` `[E, hidden, 2 * expert_hidden]`, applied as `x @ W`, gate
/// then up; `down_proj` `[E, expert_hidden, hidden]`) and a `shared_expert`.
/// What differs is the multimodal wrapper: a release names the text model
/// `language_model.model.*` (or `model.language_model.*`), and ships the
/// vision tower and projector beside it, which a text model does not read.
pub fn sanitize_checkpoint(
    weights: std::collections::HashMap<String, Array>,
) -> std::collections::HashMap<String, Array> {
    weights
        .into_iter()
        .filter_map(|(key, value)| Some((sanitize_key(&key)?, value)))
        .collect()
}

/// The text model's name for a checkpoint key (see [`sanitize_checkpoint`]), or
/// `None` for a key of the vision tower or projector.
pub fn sanitize_key(key: &str) -> Option<String> {
    if key.contains("vision_model") || key.contains("multi_modal_projector") {
        return None;
    }
    Some(if let Some(rest) = key.strip_prefix("language_model.") {
        rest.to_string()
    } else if let Some(rest) = key.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else {
        key.to_string()
    })
}

// =============================================================================
// Preset Configurations
// =============================================================================

impl Llama4TextConfig {
    /// Create config for Llama 4 Scout (109B, 16 experts).
    pub fn scout() -> Self {
        Self {
            num_local_experts: 16,
            interleave_moe_layer_step: 1, // All layers are MoE
            ..Default::default()
        }
    }

    /// Create config for Llama 4 Maverick (402B, 128 experts, interleaved).
    pub fn maverick() -> Self {
        Self {
            num_local_experts: 128,
            interleave_moe_layer_step: 2, // MoE every other layer
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::ModuleParametersExt;
    use serial_test::serial;

    #[test]
    fn test_llama4_config_moe_layers() {
        let config = Llama4TextConfig::scout();

        // Scout: all layers are MoE
        assert!(config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(config.is_moe_layer(47));

        let maverick = Llama4TextConfig::maverick();

        // Maverick (step=2): the LAST layer of each pair is MoE -> odd layers.
        assert!(!maverick.is_moe_layer(0));
        assert!(maverick.is_moe_layer(1));
        assert!(!maverick.is_moe_layer(2));
        assert!(maverick.is_moe_layer(3));
    }

    #[test]
    fn test_llama4_config_irope() {
        let config = Llama4TextConfig::default();

        // NoPE on layers where (idx + 1) % 4 == 0, i.e. layers 3, 7, 11, ...
        assert!(config.uses_rope(0)); // RoPE
        assert!(config.uses_rope(1)); // RoPE
        assert!(config.uses_rope(2)); // RoPE
        assert!(!config.uses_rope(3)); // NoPE
        assert!(config.uses_rope(4)); // RoPE
        assert!(!config.uses_rope(7)); // NoPE
    }

    #[test]
    #[serial]
    fn test_llama4_mlp() {
        let mut mlp = Llama4Mlp::new(64, 256).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 10, 64],
            pmetal_runtime::compat::Dtype::Float32,
        );

        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[1, 10, 64]);
    }

    fn host(a: &Array) -> Vec<f64> {
        let a = a.as_dtype(Dtype::Float32.as_i32());
        a.eval();
        a.as_slice::<f32>().iter().map(|&v| f64::from(v)).collect()
    }

    fn small_moe_config(num_experts: i32, top_k: i32) -> Llama4TextConfig {
        Llama4TextConfig {
            hidden_size: 12,
            intermediate_size: 10,
            intermediate_size_mlp: 10,
            num_local_experts: num_experts,
            num_experts_per_tok: top_k,
            ..Default::default()
        }
    }

    /// `Llama4TextMoe` in f64 on the host, from the layer's own weights: the
    /// `top_k` experts of each token by raw router logit (sorted here, not by
    /// the layer's argpartition), each run on the token scaled by its
    /// `sigmoid(logit)`, the outputs summed unnormalised, plus the shared
    /// expert on the token.
    fn host_moe(moe: &Llama4MoE, x: &Array) -> Vec<f64> {
        let (hidden, inter) = (x.dim(-1) as usize, moe.experts.down_proj.dim(1) as usize);
        let tokens = x.size() / hidden;
        let x = host(x);
        let router = host(&moe.router.weight); // [E, hidden]
        let experts = moe.experts.gate_up_proj.dim(0) as usize;
        let gate_up = host(&moe.experts.gate_up_proj); // [E, hidden, 2 * inter]
        let down = host(&moe.experts.down_proj); // [E, inter, hidden]
        let (sg, su, sd) = (
            host(&moe.shared_expert.gate_proj.weight), // [inter, hidden]
            host(&moe.shared_expert.up_proj.weight),
            host(&moe.shared_expert.down_proj.weight), // [hidden, inter]
        );
        let silu = |v: f64| v / (1.0 + (-v).exp());
        let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());

        let mut out = vec![0.0; tokens * hidden];
        for t in 0..tokens {
            let row = &x[t * hidden..(t + 1) * hidden];
            let logits: Vec<f64> = (0..experts)
                .map(|e| (0..hidden).map(|h| router[e * hidden + h] * row[h]).sum())
                .collect();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            for &e in &order[..moe.top_k as usize] {
                let gate = sigmoid(logits[e]);
                let scaled: Vec<f64> = row.iter().map(|v| v * gate).collect();
                let w = &gate_up[e * hidden * 2 * inter..(e + 1) * hidden * 2 * inter];
                let activated: Vec<f64> = (0..inter)
                    .map(|i| {
                        let g: f64 = (0..hidden).map(|h| scaled[h] * w[h * 2 * inter + i]).sum();
                        let u: f64 = (0..hidden)
                            .map(|h| scaled[h] * w[h * 2 * inter + inter + i])
                            .sum();
                        silu(g) * u
                    })
                    .collect();
                let d = &down[e * inter * hidden..(e + 1) * inter * hidden];
                for h in 0..hidden {
                    out[t * hidden + h] += (0..inter)
                        .map(|i| activated[i] * d[i * hidden + h])
                        .sum::<f64>();
                }
            }
            let shared: Vec<f64> = (0..inter)
                .map(|i| {
                    let g: f64 = (0..hidden).map(|h| row[h] * sg[i * hidden + h]).sum();
                    let u: f64 = (0..hidden).map(|h| row[h] * su[i * hidden + h]).sum();
                    silu(g) * u
                })
                .collect();
            for h in 0..hidden {
                out[t * hidden + h] += (0..inter)
                    .map(|i| shared[i] * sd[h * inter + i])
                    .sum::<f64>();
            }
        }
        out
    }

    /// The layer against `Llama4TextMoe` written out in f64, for one and two
    /// experts per token, and for token counts on both sides of the point
    /// where the dispatch groups the `(token, expert)` pairs by expert. The
    /// gate halves of the fused weight, the gate on the expert's input and the
    /// unnormalised sum are all load-bearing: a swapped gate and up, a gate on
    /// the output or a softmax each change the result by far more than the
    /// tolerance.
    #[test]
    #[serial]
    fn test_llama4_moe_matches_host_reference() {
        for (experts, top_k, tokens) in [(4, 1, 9), (4, 2, 9), (4, 2, 80), (6, 1, 80)] {
            let config = small_moe_config(experts, top_k);
            let mut moe = Llama4MoE::new(&config).unwrap();
            let x = pmetal_runtime::compat::random::normal(
                &[1, tokens, config.hidden_size],
                Dtype::Float32,
            );

            let want = host_moe(&moe, &x);
            let got_array = moe.forward(&x).unwrap();
            pmetal_runtime::check_unobserved_error().expect("runtime error");
            assert_eq!(got_array.shape(), &[1, tokens, config.hidden_size]);
            let got = host(&got_array);
            let diff = want
                .iter()
                .zip(&got)
                .map(|(w, g)| (w - g).abs())
                .fold(0.0, f64::max);
            let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
            assert!(
                diff <= 1e-4 * scale.max(1.0),
                "{experts} experts, top-{top_k}, {tokens} tokens: max |diff| {diff:e} \
                 against an output of up to {scale:e}"
            );
        }
    }

    /// The router sends each token to its top-k experts by raw logit and
    /// gates them `sigmoid(logit)`: not a softmax, so the gates of one token
    /// need not sum to 1.
    #[test]
    #[serial]
    fn test_llama4_router_gates_are_sigmoids_of_the_top_logits() {
        let logits = Array::from_slice(&[0.5f32, -1.0, 2.0, 0.1, -3.0, 1.5], &[2, 3]);
        let (inds, gates) = route_logits(&logits, 2, Dtype::Float32.as_i32());
        let (inds, gates) = (host(&inds), host(&gates));
        let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
        // Row 0 [0.5, -1, 2]: experts 2 and 0. Row 1 [0.1, -3, 1.5]: 2 and 0.
        for (row, want) in [
            (0usize, [(2usize, 2.0f64), (0, 0.5)]),
            (1, [(2, 1.5), (0, 0.1)]),
        ] {
            for (slot_expert, logit) in want {
                let slot = (0..2)
                    .find(|&s| inds[row * 2 + s] as usize == slot_expert)
                    .unwrap_or_else(|| panic!("row {row}: expert {slot_expert} not chosen"));
                assert!((gates[row * 2 + slot] - sigmoid(logit)).abs() < 1e-6);
            }
        }
        assert!(
            gates[0] + gates[1] > 1.0,
            "the gates of a token are not normalised"
        );
    }

    /// The loss reaches the experts' weights and the layer's input through
    /// the grouped gather matmuls and the fused gate-up slices: the same
    /// gradients as the experts written as per-token weight lookups.
    #[test]
    #[serial]
    fn test_llama4_moe_gradients_match_per_token_lookups() {
        let config = small_moe_config(4, 2);
        let (hidden, inter) = (config.hidden_size, config.intermediate_size);
        let mut moe = Llama4MoE::new(&config).unwrap();
        // 40 tokens x 2 experts = 80 pairs: the grouped dispatch.
        let x = pmetal_runtime::compat::random::normal(&[40, hidden], Dtype::Float32);
        let gate_up = moe.experts.gate_up_proj.clone();
        let down = moe.experts.down_proj.clone();

        let layer = |moe: &mut Llama4MoE, arrays: &[Array]| {
            moe.experts.gate_up_proj = arrays[1].clone();
            moe.experts.down_proj = arrays[2].clone();
            moe.forward(&arrays[0]).unwrap().square().sum(None)
        };
        let (loss, grads) = pmetal_runtime::compat::nn::value_and_grad_explicit(
            |arrays: &[Array]| layer(&mut moe, arrays),
            &[x.clone(), gate_up.clone(), down.clone()],
            &[],
        )
        .unwrap();
        pmetal_runtime::check_unobserved_error().expect("runtime error");
        assert!(loss.item::<f32>().is_finite());

        // The same layer from per-token lookups: `take_axis` picks each
        // token's expert weights, and ordinary matmuls run them.
        let router = moe.router.weight.clone();
        let shared = (
            moe.shared_expert.gate_proj.weight.clone(),
            moe.shared_expert.up_proj.weight.clone(),
            moe.shared_expert.down_proj.weight.clone(),
        );
        let reference = |arrays: &[Array]| {
            let (x, gate_up, down) = (&arrays[0], &arrays[1], &arrays[2]);
            let logits = x.matmul(&router.t());
            let (inds, gates) = route_logits(&logits, 2, Dtype::Float32.as_i32());
            let mut out = {
                let g = nn::silu(&x.matmul(&shared.0.t())).multiply(&x.matmul(&shared.1.t()));
                g.matmul(&shared.2.t())
            };
            for slot in 0..2 {
                let ids = ops::slice_axis(&inds, 1, slot, slot + 1).reshape(&[-1]);
                let gate = ops::slice_axis(&gates, 1, slot, slot + 1);
                let scaled = x.multiply(&gate).reshape(&[-1, 1, hidden]);
                let w = gate_up.take_axis(&ids, 0); // [N, hidden, 2 * inter]
                let both = scaled.matmul(&w).squeeze_axes(&[1]);
                let parts = ops::split_sections(&both, &[inter], -1);
                let activated = nn::silu(&parts[0])
                    .multiply(&parts[1])
                    .reshape(&[-1, 1, inter]);
                let d = down.take_axis(&ids, 0); // [N, inter, hidden]
                out = out.add(&activated.matmul(&d).squeeze_axes(&[1]));
            }
            out.square().sum(None)
        };
        let (ref_loss, ref_grads) = pmetal_runtime::compat::nn::value_and_grad_explicit(
            reference,
            &[x, gate_up, down],
            &[],
        )
        .unwrap();
        pmetal_runtime::check_unobserved_error().expect("runtime error");

        let rel = |a: &Array, b: &Array| {
            let (a, b) = (host(a), host(b));
            let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-6);
            a.iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max)
                / scale
        };
        let loss_diff = f64::from((loss.item::<f32>() - ref_loss.item::<f32>()).abs())
            / f64::from(ref_loss.item::<f32>().abs()).max(1.0);
        assert!(loss_diff < 1e-4, "loss differs by {loss_diff:e}");
        for (name, (g, r)) in ["input", "gate_up_proj", "down_proj"]
            .iter()
            .zip(grads.iter().zip(&ref_grads))
        {
            assert!(
                host(g).iter().any(|v| *v != 0.0),
                "the {name} gradient is zero"
            );
            let diff = rel(g, r);
            assert!(
                diff < 1e-3,
                "{name} gradient differs by {diff:e} (relative)"
            );
        }
    }

    #[test]
    #[serial]
    fn test_llama4_moe_accepts_float16_inputs() {
        let config = Llama4TextConfig {
            hidden_size: 16,
            intermediate_size: 32,
            intermediate_size_mlp: 32,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            ..Default::default()
        };

        let mut moe = Llama4MoE::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 4, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        )
        .as_dtype(pmetal_runtime::compat::Dtype::Float16.as_i32())
        .unwrap();

        let output = moe.forward(&x).unwrap();
        output.eval().unwrap();

        assert_eq!(output.shape(), &[1, 4, config.hidden_size]);
        for value in output.as_slice::<f32>().to_vec() {
            assert!(value.is_finite(), "non-finite Llama4 MoE output");
        }
    }

    /// A config that ties the head to the embedding builds no head, and the
    /// logits are the hidden states against the embedding table.
    #[test]
    #[serial]
    fn test_llama4_tied_head_projects_through_the_embedding() {
        let mut model = Llama4ForCausalLM::new(Llama4TextConfig {
            hidden_size: 32,
            intermediate_size: 24,
            intermediate_size_mlp: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            num_local_experts: 2,
            vocab_size: 50,
            tie_word_embeddings: true,
            ..Default::default()
        })
        .unwrap();
        assert!(model.lm_head.is_none());
        let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let hidden = model.model.forward(&ids, None, None).unwrap();
        let want = hidden.matmul(&model.model.embed_tokens.weight.value.t());
        let got = model.forward(&ids, None, None).unwrap();
        let diff = got.subtract(&want).abs().max(None);
        diff.eval();
        assert!(diff.item::<f32>() < 1e-5);
    }

    #[test]
    #[serial]
    fn test_llama4_model_instantiation() {
        let config = Llama4TextConfig {
            hidden_size: 64,
            intermediate_size: 256,
            intermediate_size_mlp: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            num_local_experts: 4,
            vocab_size: 1000,
            ..Default::default()
        };

        let model = Llama4ForCausalLM::new(config).unwrap();

        let params = model.flatten_params();
        assert!(!params.is_empty());
    }
}
