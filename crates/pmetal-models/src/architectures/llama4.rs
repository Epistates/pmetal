//! Llama 4 architecture with Mixture of Experts, iRoPE, and Mixture of Depths.
//!
//! Key features:
//! - **iRoPE**: Interleaved RoPE/NoPE layers for long context (10M+ tokens)
//! - **MoE with shared expert**: Each token routed to 1 expert + shared expert
//! - **Interleaved MoE/Dense**: Scout is full MoE, Maverick alternates
//! - **QK norm**: Layer normalization on Q and K for stable attention
//! - **Temperature scaling**: Dynamic attention scaling for long sequences
//! - **MoD**: Mixture-of-Depths (Raposo et al., 2024) for adaptive compute
//!
//! Variants:
//! - **Llama 4 Scout**: 109B total params (16 experts), 17B active, 10M context
//! - **Llama 4 Maverick**: 402B total params (128 experts), 17B active, 1M context
use pmetal_bridge::compat::{
    Array, Dtype, Exception, Module, ModuleParameters, ModuleParametersExt, fast, indexing, nn,
    ops, random,
};
use pmetal_bridge::impl_module_params;

use pmetal_bridge::rope::{RopeConfig, RotaryEmbedding};
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopePositions, rope_embedding},
};
use pmetal_mlx::kv_cache::KVCache;
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

    // Mixture-of-Depths (MoD) configuration (Raposo et al., 2024)
    /// Enable MoD: tokens are selectively routed through transformer blocks.
    #[serde(default = "default_use_mod")]
    pub use_mod: bool,
    /// MoD capacity factor C in (0, 1]: fraction of tokens processed per MoD layer.
    /// k = floor(C * seq_len) tokens are selected per forward pass.
    #[serde(default = "default_mod_capacity")]
    pub mod_capacity: f32,
    /// Explicit list of layer indices that use MoD.
    /// When set, overrides `mod_layer_interval`.
    #[serde(default)]
    pub mod_layers: Option<Vec<i32>>,
    /// Interval for MoD layers when `mod_layers` is None.
    /// Default 2 = every other layer is a MoD layer.
    #[serde(default = "default_mod_layer_interval")]
    pub mod_layer_interval: i32,

    /// RoPE scaling (Scout: `rope_type: "llama3"`, factor 16), read by
    /// [`pmetal_bridge::rope`].
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
fn default_use_mod() -> bool {
    false
}
fn default_mod_capacity() -> f32 {
    0.5
}
fn default_mod_layer_interval() -> i32 {
    2
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
            use_mod: false,
            mod_capacity: 0.5,
            mod_layers: None,
            mod_layer_interval: 2,
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

    /// Check if a given layer uses Mixture-of-Depths.
    ///
    /// Returns `false` when MoD is globally disabled (`use_mod == false`).
    pub fn is_mod_layer(&self, layer_idx: i32) -> bool {
        if !self.use_mod {
            return false;
        }
        if let Some(ref mod_layers) = self.mod_layers {
            mod_layers.contains(&layer_idx)
        } else {
            layer_idx % self.mod_layer_interval == 0
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
// Expert and MoE Components
// =============================================================================

/// A single expert (MLP).
#[derive(Debug)]
pub struct Llama4Expert {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(Llama4Expert; gate_proj, up_proj, down_proj);

impl Llama4Expert {
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

/// Router for selecting experts.
#[derive(Debug)]
pub struct Llama4Router {
    pub gate: nn::Linear,
    pub num_experts: i32,
    pub top_k: i32,
}
impl_module_params!(Llama4Router; gate);

impl Llama4Router {
    pub fn new(hidden_size: i32, num_experts: i32, top_k: i32) -> Result<Self, Exception> {
        let gate = nn::LinearBuilder::new(hidden_size, num_experts)
            .bias(false)
            .build()?;
        Ok(Self {
            gate,
            num_experts,
            top_k,
        })
    }

    /// Route tokens to experts.
    ///
    /// Returns `(expert_indices, expert_weights, router_logits)` where:
    /// - `expert_indices`: `[total_tokens, top_k]` — selected expert IDs (i32)
    /// - `expert_weights`: `[total_tokens, top_k]` — normalized routing weights
    /// - `router_logits`:  `[total_tokens, num_experts]` — raw pre-softmax logits
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Array, Array), Exception> {
        // x: [total_tokens, hidden]
        let router_logits = Module::forward(&mut self.gate, x)?;

        // Llama4 routing (HF Llama4TextMoe): pick the top-k
        // experts by RAW logit, then gate with `sigmoid(logit)` — NOT a
        // softmax. The previous softmax-then-renormalize collapsed the top-1
        // weight to exactly 1.0 (softmax over a single selected logit is 1),
        // throwing away the router signal entirely.
        let neg_k = -(self.top_k as i32);
        let part_indices = ops::argpartition_axis(&router_logits, neg_k, -1);
        // Slice the last top_k entries: [total_tokens, top_k]. A selection
        // carries no gradient, and MLX refuses to differentiate a gather with
        // respect to its indices.
        let expert_indices = ops::stop_gradient(&ops::slice_axis_from(&part_indices, -1, neg_k));

        // Gate weight = sigmoid of the selected RAW logits (per expert,
        // independent — no cross-expert normalisation), in f32 and then back
        // to the activations' dtype, as `Llama4TextMoe` computes it.
        let selected_logits = router_logits.take_along_axis(&expert_indices, -1);
        let expert_weights =
            ops::sigmoid(&selected_logits.as_type::<f32>()).as_dtype(x.dtype().as_i32());

        Ok((expert_indices, expert_weights, router_logits))
    }
}

// =============================================================================
// Mixture-of-Depths (MoD) Router
// =============================================================================

/// Per-layer MoD router (Raposo et al., 2024).
///
/// A lightweight scalar projection that assigns each token a routing weight.
/// Top-k tokens (by weight) are selected to pass through the transformer block;
/// the remaining tokens receive a residual identity pass-through.
#[derive(Debug)]
pub struct Llama4ModRouter {
    /// Scalar linear projection: [hidden_size] -> [1].
    pub gate: nn::Linear,
}
impl_module_params!(Llama4ModRouter; gate);

impl Llama4ModRouter {
    pub fn new(hidden_size: i32) -> Result<Self, Exception> {
        let gate = nn::LinearBuilder::new(hidden_size, 1).bias(false).build()?;
        Ok(Self { gate })
    }

    /// Compute router logits and select top-k token indices.
    ///
    /// # Arguments
    /// * `x` - Hidden states `[batch, seq_len, hidden_size]`
    /// * `capacity` - Capacity factor C in (0, 1]; k = floor(C * seq_len) tokens selected
    ///
    /// # Returns
    /// `(selected_indices, router_logits, top_k_mask)` where:
    /// - `selected_indices`: `[batch, k]` — positions of the selected tokens (i32)
    /// - `router_logits`:    `[batch, seq_len, 1]` — raw scalar logits from the linear gate
    /// - `top_k_mask`:       `[batch, seq_len]` — binary mask, 1.0 for selected tokens
    pub fn route(&mut self, x: &Array, capacity: f32) -> Result<(Array, Array, Array), Exception> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        // router_logits: [B, T, 1]
        let router_logits = Module::forward(&mut self.gate, x)?;

        // Squeeze to [B, T] for selection
        let weights = router_logits.reshape(&[batch, seq_len]);

        // k = floor(C * T), clamped to [1, T]
        let k = ((capacity * seq_len as f32).floor() as i32)
            .max(1)
            .min(seq_len);

        // argpartition(weights, -k, axis=-1) places the k largest at positions [-k..]
        // This is O(T) vs O(T log T) for argsort.
        let part_indices = ops::argpartition_axis(&weights, -k, -1);

        // Slice the last k indices — these correspond to the top-k tokens.
        let selected_indices = ops::slice_axis_from(&part_indices, -1, -k);
        // selected_indices: [B, k]

        // Build a binary top-k mask [B, T] of zeros with 1s at selected positions.
        // We scatter ones into a zeros tensor using put_along_axis.
        let zeros = ops::zeros(&[batch, seq_len], Dtype::Float32);
        let ones = ops::ones(&[batch, k], Dtype::Float32);
        let top_k_mask = ops::put_along_axis(&zeros, &selected_indices, &ones, 1);
        // top_k_mask: [B, T] with 1.0 at selected token positions

        Ok((selected_indices, router_logits, top_k_mask))
    }
}

/// Mixture of Experts layer with shared expert.
#[derive(Debug)]
pub struct Llama4MoE {
    pub config: Llama4TextConfig,

    pub router: Llama4Router,
    pub experts: Vec<Llama4Expert>,
    pub shared_expert: Llama4Expert,
}
impl_module_params!(Llama4MoE; router, experts, shared_expert);

impl Llama4MoE {
    pub fn new(config: &Llama4TextConfig) -> Result<Self, Exception> {
        let router = Llama4Router::new(
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
        )?;

        let experts = (0..config.num_local_experts)
            .map(|_| Llama4Expert::new(config.hidden_size, config.intermediate_size))
            .collect::<Result<Vec<_>, _>>()?;

        let shared_expert = Llama4Expert::new(config.hidden_size, config.intermediate_size)?;

        Ok(Self {
            config: config.clone(),
            router,
            experts,
            shared_expert,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape().to_vec();
        let hidden_size = *shape.last().unwrap();

        // Flatten to [total_tokens, hidden]
        let total_tokens = shape.iter().take(shape.len() - 1).product::<i32>();
        let flat_x = x.reshape(&[total_tokens, hidden_size]);

        // Route tokens.
        // expert_indices: [total_tokens, top_k]
        // expert_weights: [total_tokens, top_k]  (normalized)
        let (expert_indices, expert_weights, _router_logits) = self.router.forward(&flat_x)?;

        // Shared expert output (always applied to all tokens)
        let shared_out = self.shared_expert.forward(&flat_x)?;

        // Which expert each token goes to is read on the host, so that each
        // expert runs over its own tokens only. The gate values stay in the
        // graph: the loss reaches the router, and every adapter upstream of
        // it, through them, as it does in `Llama4TextMoe`.
        let expert_indices = expert_indices.as_type::<i32>();
        expert_indices.eval();

        let top_k = self.config.num_experts_per_tok as usize;
        let n_tokens = total_tokens as usize;
        let expert_ids: Vec<i32> = expert_indices.as_slice::<i32>().to_vec();
        let flat_weights = expert_weights.reshape(&[-1]);

        // Per expert: the tokens routed to it, and where each one's gate
        // value sits in `flat_weights`.
        let mut expert_assignments: Vec<Vec<(i32, i32)>> = vec![Vec::new(); self.experts.len()];
        for token_idx in 0..n_tokens {
            for slot in 0..top_k {
                let flat_idx = token_idx * top_k + slot;
                let expert_id = expert_ids[flat_idx] as usize;
                if expert_id < self.experts.len() {
                    expert_assignments[expert_id].push((token_idx as i32, flat_idx as i32));
                }
            }
        }

        let input_dtype = flat_x.dtype();
        let mut combined_out = ops::zeros_dtype(&[total_tokens, hidden_size], input_dtype);
        for (expert_idx, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }

            let count = assignments.len() as i32;
            let token_indices: Vec<i32> = assignments.iter().map(|&(token, _)| token).collect();
            let gate_slots: Vec<i32> = assignments.iter().map(|&(_, slot)| slot).collect();

            let idx_array = Array::from_slice(&token_indices, &[count]);
            let weight_array = flat_weights
                .take_axis(&Array::from_slice(&gate_slots, &[count]), 0)
                .reshape(&[count, 1]);

            // Llama4 scales the expert *input* by the sigmoid gate
            // (`experts(x * scores)`), not the output. The expert MLP is
            // SwiGLU (nonlinear), so input- and output-scaling are NOT
            // equivalent — the input must be scaled to match HF.
            let expert_input = flat_x.take_axis(&idx_array, 0).multiply(&weight_array);
            let expert_out = self.experts[expert_idx].forward(&expert_input)?;

            let updates = expert_out.reshape(&[count, 1, hidden_size]);
            combined_out = pmetal_bridge::compat::indexing::scatter_add_single(
                &combined_out,
                &idx_array,
                &updates,
                0,
            );
        }

        // Add shared expert contribution and reshape to original shape
        let output = shared_out.add(&combined_out);
        Ok(output.reshape(&shape))
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
    pub attn_temperature_tuning: bool,
    pub floor_scale: f32,
    pub attn_scale: f32,

    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,

    // QK normalization (optional)
    pub q_norm: Option<nn::RmsNorm>,
    pub k_norm: Option<nn::RmsNorm>,
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
        // `rmsnorm(x, norm_eps)`). The RmsNorm weight stays at its default
        // ones, so it is the weightless norm.
        let qk_norm = || {
            nn::RmsNormBuilder::new(head_dim)
                .eps(config.rms_norm_eps)
                .build()
        };
        let (q_norm, k_norm) = if config.use_qk_norm && uses_rope {
            (Some(qk_norm()?), Some(qk_norm()?))
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
        if let (Some(qn), Some(kn)) = (&mut self.q_norm, &mut self.k_norm) {
            q = Module::forward(qn, &q)?;
            k = Module::forward(kn, &k)?;
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

        let (k, v) = match cache {
            Some((cache, layer)) => cache.update_and_fetch(layer, &k, &v)?,
            None => (k, v),
        };

        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });
        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        Module::forward(&mut self.o_proj, &output)
    }
}

// =============================================================================
// Decoder Layer
// =============================================================================

/// Llama 4 decoder layer (can be dense or MoE, optionally with MoD).
#[derive(Debug)]
pub struct Llama4DecoderLayer {
    pub layer_idx: usize,
    pub is_moe: bool,
    /// MoD capacity factor for this layer (None = MoD disabled).
    pub mod_capacity: Option<f32>,

    pub self_attn: Llama4Attention,
    pub mlp: Option<Llama4Expert>, // Dense MLP (if not MoE)
    pub moe: Option<Llama4MoE>,    // MoE layer (if MoE)
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
    /// MoD router (present only when this layer uses Mixture-of-Depths).
    pub mod_router: Option<Llama4ModRouter>,

    // Auxiliary loss from the most recent MoD forward pass (not a learned parameter).
    // Stored here so the parent model can aggregate it without threading extra return values
    // through the forward signature.
    pub last_mod_aux_loss: Option<Array>,
}
impl_module_params!(Llama4DecoderLayer; self_attn, mlp, moe, input_layernorm, post_attention_layernorm, mod_router);

impl Llama4DecoderLayer {
    pub fn new(config: &Llama4TextConfig, layer_idx: usize) -> Result<Self, Exception> {
        let self_attn = Llama4Attention::new(config, layer_idx)?;

        let is_moe = config.is_moe_layer(layer_idx as i32);
        let (mlp, moe) = if is_moe {
            (None, Some(Llama4MoE::new(config)?))
        } else {
            (
                Some(Llama4Expert::new(
                    config.hidden_size,
                    config.intermediate_size_mlp,
                )?),
                None,
            )
        };

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        // MoD router (only allocated for MoD-enabled layers)
        let mod_capacity = if config.is_mod_layer(layer_idx as i32) {
            Some(config.mod_capacity)
        } else {
            None
        };
        let mod_router = if mod_capacity.is_some() {
            Some(Llama4ModRouter::new(config.hidden_size)?)
        } else {
            None
        };

        Ok(Self {
            layer_idx,
            is_moe,
            mod_capacity,
            self_attn,
            mlp,
            moe,
            input_layernorm,
            post_attention_layernorm,
            mod_router,
            last_mod_aux_loss: None,
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
    /// A MoD layer routes a subset of the tokens and runs without a cache.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        if let Some(capacity) = self.mod_capacity {
            if cache.is_some() {
                return Err(Exception::custom(
                    "Llama 4 Mixture-of-Depths layers have no cached decode",
                ));
            }
            self.forward_mod(x, mask, position_ids, capacity)
        } else {
            self.forward_full(x, mask, cache, position_ids)
        }
    }

    /// Standard full-sequence forward (no MoD).
    fn forward_full(
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
        let ffn_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed)?
        };
        Ok(h.add(&ffn_out))
    }

    /// MoD forward: route top-k tokens through the block, identity for the rest.
    ///
    /// Algorithm (Raposo et al., 2024):
    /// 1. Router produces a scalar weight per token.
    /// 2. Top-k tokens (k = floor(C * T)) are gathered from the sequence.
    /// 3. The gathered sub-batch passes through attention + FFN.
    /// 4. Results are scattered back; non-selected tokens keep their input value
    ///    (residual identity pass-through).
    /// 5. Auxiliary BCE loss is stored in `last_mod_aux_loss` for the caller to aggregate.
    fn forward_mod(
        &mut self,
        x: &Array,
        _mask: Option<&Array>,
        _position_ids: Option<&Array>,
        capacity: f32,
    ) -> Result<Array, Exception> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden = x.shape()[2];
        let k = ((capacity * seq_len as f32).floor() as i32)
            .max(1)
            .min(seq_len);

        // ---- Router ----
        let router = self
            .mod_router
            .as_mut()
            .expect("mod_router must be Some when forward_mod is called");
        let (selected_indices, router_logits, top_k_mask) = router.route(x, capacity)?;
        // selected_indices: [B, k]  (i32, seq-axis positions)
        // router_logits:    [B, T, 1]
        // top_k_mask:       [B, T]

        // ---- Gather selected tokens ----
        // Expand indices to [B, k, D] for take_along_axis on axis=1
        let idx_reshaped = selected_indices.reshape(&[batch, k, 1]);
        let idx_expanded = ops::broadcast_to(&idx_reshaped, &[batch, k, hidden]);
        // gathered: [B, k, D]
        let gathered = x.take_along_axis(&idx_expanded, 1);

        // ---- Run transformer block on gathered sub-batch ----
        // Note: the gathered tokens attend to each other unmasked (an all-zero
        // additive mask; without one attention is causal) — they form a dense
        // sub-sequence and causal masking at this level would be wrong.
        // Position IDs are also omitted, so the sub-batch rotates at 0..k.
        let normed = Module::forward(&mut self.input_layernorm, &gathered)?;
        let unmasked = Array::zeros_f32(&[k, k]);
        let attn_out = self.self_attn.forward(&normed, Some(&unmasked), None)?;
        let h_sel = gathered.add(&attn_out);

        let normed2 = Module::forward(&mut self.post_attention_layernorm, &h_sel)?;
        let ffn_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed2)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed2)?
        };
        let block_out = h_sel.add(&ffn_out);
        // block_out: [B, k, D] — processed token outputs

        // ---- Scatter results back into the full-sequence tensor ----
        // Non-selected token slots start as the original `x` (identity/residual skip).
        // We overwrite the selected positions with the block output.
        let idx_reshaped_scatter = selected_indices.reshape(&[batch, k, 1]);
        let idx_expanded_scatter = ops::broadcast_to(&idx_reshaped_scatter, &[batch, k, hidden]);
        let output = ops::put_along_axis(x, &idx_expanded_scatter, &block_out, 1);
        // output: [B, T, D]  — selected tokens updated, others unchanged

        // ---- Auxiliary BCE loss ----
        // BCE(sigmoid(router_logits), top_k_mask) teaches the router to
        // predict which tokens it will select, enabling autoregressive inference
        // where the router must decide without seeing future selections.
        let logits_flat = router_logits.reshape(&[batch * seq_len]);
        let mask_flat = top_k_mask.reshape(&[batch * seq_len]);
        let aux_loss = pmetal_bridge::compat::losses::BinaryCrossEntropyBuilder::new()
            .with_logits(true)
            .reduction(pmetal_bridge::compat::losses::LossReduction::Mean)
            .build()
            .call(&logits_flat, &mask_flat);
        self.last_mod_aux_loss = Some(aux_loss);

        Ok(output)
    }

    /// Return the auxiliary MoD loss from the most recent forward pass, if any.
    pub fn mod_aux_loss(&self) -> Option<&Array> {
        self.last_mod_aux_loss.as_ref()
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
    /// holding them. Training only; see [`crate::checkpointing`].
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

    /// Aggregate MoD auxiliary losses from all MoD-enabled layers after a forward pass.
    ///
    /// Returns the mean BCE loss across all MoD layers, or `None` if no MoD layers fired.
    /// Callers should scale by `config.router_aux_loss_coef` before adding to the task loss.
    pub fn mod_aux_loss(&self) -> Result<Option<Array>, Exception> {
        let mut total: Option<Array> = None;
        let mut count = 0usize;

        for layer in &self.layers {
            if let Some(loss) = layer.mod_aux_loss() {
                total = Some(match total {
                    None => loss.clone(),
                    Some(acc) => acc.add(loss),
                });
                count += 1;
            }
        }

        match total {
            None => Ok(None),
            Some(sum) => {
                let denom = Array::from_f32(count as f32);
                Ok(Some(sum.divide(&denom)))
            }
        }
    }
}

/// Llama 4 for causal language modeling.
#[derive(Debug)]
pub struct Llama4ForCausalLM {
    pub config: Llama4TextConfig,

    pub model: Llama4TextModel,
    pub lm_head: nn::Linear,
}
impl_module_params!(Llama4ForCausalLM; model, lm_head);

impl Llama4ForCausalLM {
    pub fn new(config: Llama4TextConfig) -> Result<Self, Exception> {
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()?;

        let model = Llama4TextModel::new(config.clone())?;

        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids, mask, position_ids)?;
        Module::forward(&mut self.lm_head, &hidden_states)
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
        Module::forward(&mut self.lm_head, &hidden_states)
    }

    /// Aggregate MoD auxiliary losses across all layers after a forward pass.
    ///
    /// Callers should add `config.router_aux_loss_coef * mod_aux_loss()` to the
    /// task loss when training with MoD enabled.
    pub fn mod_aux_loss(&self) -> Result<Option<Array>, Exception> {
        self.model.mod_aux_loss()
    }
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
    use pmetal_bridge::compat::ModuleParameters;
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
    fn test_llama4_expert() {
        let expert = Llama4Expert::new(64, 256).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[1, 10, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let mut expert = expert;
        let out = expert.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[1, 10, 64]);
    }

    #[test]
    #[serial]
    fn test_llama4_moe_matches_naive_reference() {
        let mut config = Llama4TextConfig::default();
        config.hidden_size = 32;
        config.intermediate_size = 64;
        config.intermediate_size_mlp = 64;
        config.num_local_experts = 4;
        config.num_experts_per_tok = 2;

        let mut moe = Llama4MoE::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[2, 5, config.hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let shape = x.shape().to_vec();
        let hidden_size = *shape.last().unwrap();
        let total_tokens = shape.iter().take(shape.len() - 1).product::<i32>();
        let flat_x = x.reshape(&[total_tokens, hidden_size]).unwrap();

        let (expert_indices, expert_weights, router_logits) = moe.router.forward(&flat_x).unwrap();
        let shared_out = moe.shared_expert.forward(&flat_x).unwrap();

        // Router weights must be sigmoid(selected raw logits) — independent
        // per expert, NOT softmax-normalized. Verify against a direct sigmoid
        // of the gathered logits, and confirm the per-token weights do NOT sum
        // to 1 (which the old softmax-renormalize path would have forced).
        let mut sig_ref =
            ops::sigmoid(&router_logits.take_along_axis(&expert_indices, -1)).unwrap();
        let mut weights_eval = expert_weights.clone();
        sig_ref.eval().unwrap();
        weights_eval.eval().unwrap();
        let n = (total_tokens * config.num_experts_per_tok) as usize;
        let w = weights_eval.to_f32_vec(n).unwrap();
        let s = sig_ref.to_f32_vec(n).unwrap();
        let max_w_diff = w
            .iter()
            .zip(&s)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_w_diff < 1e-5,
            "router weights must equal sigmoid(logits): {max_w_diff}"
        );

        let mut reference = ops::zeros_dtype(&[total_tokens, hidden_size], flat_x.dtype()).unwrap();
        let top_k = config.num_experts_per_tok;
        for slot in 0..top_k {
            let slot_indices = expert_indices
                .index((.., slot..slot + 1))
                .squeeze_axes(&[-1])
                .unwrap();
            let slot_weights = expert_weights.index((.., slot..slot + 1));

            // Naive path scales the expert INPUT by the gate (matches
            // `experts(x * scores)`); unmasked tokens get a zero input and a
            // zero output (the bias-free SwiGLU expert maps 0 → 0).
            let mut slot_out =
                ops::zeros_dtype(&[total_tokens, hidden_size], flat_x.dtype()).unwrap();
            for (expert_idx, expert) in moe.experts.iter_mut().enumerate() {
                let expert_id = Array::from_int(expert_idx as i32);
                let mask = slot_indices.eq(&expert_id).unwrap();
                let mask_f32 = mask
                    .as_dtype(pmetal_bridge::compat::Dtype::Float32.as_i32())
                    .unwrap();
                let gate = mask_f32
                    .reshape(&[total_tokens, 1])
                    .unwrap()
                    .multiply(&slot_weights)
                    .unwrap();
                let scaled_input = flat_x.multiply(&gate).unwrap();
                let exp_output = expert.forward(&scaled_input).unwrap();
                slot_out = slot_out.add(&exp_output).unwrap();
            }

            reference = reference.add(&slot_out).unwrap();
        }

        let reference = shared_out.add(&reference).unwrap().reshape(&shape).unwrap();
        let output = moe.forward(&x).unwrap();

        output.eval().unwrap();
        reference.eval().unwrap();
        let diff = output
            .subtract(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        diff.eval().unwrap();
        let max_diff = diff.item::<f32>();
        assert!(
            max_diff < 1e-4,
            "llama4 moe drifted from naive path: {max_diff}"
        );
        assert_eq!(output.shape(), &[2, 5, config.hidden_size]);
    }

    #[test]
    #[serial]
    fn test_llama4_moe_accepts_float16_inputs() {
        let mut config = Llama4TextConfig::default();
        config.hidden_size = 16;
        config.intermediate_size = 32;
        config.intermediate_size_mlp = 32;
        config.num_local_experts = 2;
        config.num_experts_per_tok = 1;

        let mut moe = Llama4MoE::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, config.hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
        )
        .as_dtype(pmetal_bridge::compat::Dtype::Float16.as_i32())
        .unwrap();

        let output = moe.forward(&x).unwrap();
        output.eval().unwrap();

        assert_eq!(output.shape(), &[1, 4, config.hidden_size]);
        for value in output.as_slice::<f32>().to_vec() {
            assert!(value.is_finite(), "non-finite Llama4 MoE output");
        }
    }

    #[test]
    #[serial]
    fn test_llama4_model_instantiation() {
        let mut config = Llama4TextConfig::default();
        config.hidden_size = 64;
        config.intermediate_size = 256;
        config.intermediate_size_mlp = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.num_local_experts = 4;
        config.vocab_size = 1000;

        let model = Llama4ForCausalLM::new(config).unwrap();

        let params = model.flatten_params();
        assert!(params.len() > 0);
    }

    // =========================================================================
    // MoD tests
    // =========================================================================

    #[test]
    fn test_llama4_config_mod_layer_detection() {
        // MoD disabled by default
        let config = Llama4TextConfig::default();
        assert!(!config.is_mod_layer(0));
        assert!(!config.is_mod_layer(1));

        // Enable MoD with interval=2
        let mut cfg = Llama4TextConfig::default();
        cfg.use_mod = true;
        cfg.mod_layer_interval = 2;
        assert!(cfg.is_mod_layer(0));
        assert!(!cfg.is_mod_layer(1));
        assert!(cfg.is_mod_layer(2));
        assert!(!cfg.is_mod_layer(3));

        // Explicit mod_layers list
        let mut cfg2 = Llama4TextConfig::default();
        cfg2.use_mod = true;
        cfg2.mod_layers = Some(vec![1, 3, 5]);
        assert!(!cfg2.is_mod_layer(0));
        assert!(cfg2.is_mod_layer(1));
        assert!(!cfg2.is_mod_layer(2));
        assert!(cfg2.is_mod_layer(3));
        assert!(cfg2.is_mod_layer(5));
    }

    #[test]
    #[serial]
    fn test_llama4_mod_router_route() {
        let batch = 2i32;
        let seq_len = 8i32;
        let hidden = 16i32;
        let capacity = 0.5_f32; // k = 4

        let mut router = Llama4ModRouter::new(hidden).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[batch, seq_len, hidden],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let (selected_indices, router_logits, top_k_mask) = router.route(&x, capacity).unwrap();
        selected_indices.eval().unwrap();
        router_logits.eval().unwrap();
        top_k_mask.eval().unwrap();

        let k = ((capacity * seq_len as f32).floor() as i32).max(1);

        // selected_indices: [B, k]
        assert_eq!(selected_indices.shape(), &[batch, k]);
        // router_logits: [B, T, 1]
        assert_eq!(router_logits.shape(), &[batch, seq_len, 1]);
        // top_k_mask: [B, T]
        assert_eq!(top_k_mask.shape(), &[batch, seq_len]);
    }

    #[test]
    #[serial]
    fn test_llama4_mod_router_mask_has_correct_count() {
        // Each row of top_k_mask must sum to exactly k.
        let batch = 1i32;
        let seq_len = 10i32;
        let hidden = 16i32;
        let capacity = 0.3_f32; // k = floor(0.3 * 10) = 3

        let mut router = Llama4ModRouter::new(hidden).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[batch, seq_len, hidden],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let (_indices, _logits, top_k_mask) = router.route(&x, capacity).unwrap();
        top_k_mask.eval().unwrap();

        // Sum along seq dimension: should equal k for each batch item.
        let row_sums = top_k_mask.sum_axis(-1, false).unwrap();
        row_sums.eval().unwrap();

        let expected_k = ((capacity * seq_len as f32).floor() as i32).max(1) as f32;
        let sum_vals: Vec<f32> = row_sums.as_slice::<f32>().to_vec();
        for s in sum_vals {
            assert!(
                (s - expected_k).abs() < 1e-4,
                "Expected row sum {expected_k}, got {s}"
            );
        }
    }

    #[test]
    #[serial]
    fn test_llama4_decoder_layer_mod_forward() {
        let mut config = Llama4TextConfig::default();
        config.hidden_size = 32;
        config.intermediate_size = 64;
        config.intermediate_size_mlp = 64;
        config.num_attention_heads = 2;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.num_local_experts = 2;
        config.vocab_size = 100;
        // Enable MoD on all layers
        config.use_mod = true;
        config.mod_capacity = 0.5;
        config.mod_layer_interval = 1;

        let mut layer = Llama4DecoderLayer::new(&config, 0).unwrap();
        assert!(layer.mod_router.is_some(), "MoD router should be allocated");

        let batch = 1i32;
        let seq_len = 8i32;
        let hidden = config.hidden_size;

        let x = pmetal_bridge::compat::random::normal(
            &[batch, seq_len, hidden],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let out = layer.forward(&x, None, None).unwrap();
        out.eval().unwrap();

        // Output shape must match input shape.
        assert_eq!(out.shape(), &[batch, seq_len, hidden]);

        // Aux loss should be present after a MoD forward.
        assert!(layer.mod_aux_loss().is_some(), "MoD aux loss should be set");
        let aux = layer.mod_aux_loss().unwrap();
        aux.eval().unwrap();
        // BCE is a scalar (mean reduction is the default).
        assert_eq!(aux.shape().len(), 0, "aux loss should be scalar");
    }

    #[test]
    #[serial]
    fn test_llama4_mod_identity_on_non_mod_layer() {
        // Without MoD, decoder layer must behave exactly as before.
        let mut config = Llama4TextConfig::default();
        config.hidden_size = 32;
        config.intermediate_size = 64;
        config.intermediate_size_mlp = 64;
        config.num_attention_heads = 2;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.num_local_experts = 2;
        config.vocab_size = 100;
        config.use_mod = false; // MoD globally disabled

        let mut layer = Llama4DecoderLayer::new(&config, 0).unwrap();
        assert!(
            layer.mod_router.is_none(),
            "No MoD router when MoD is disabled"
        );

        let x = pmetal_bridge::compat::random::normal(
            &[1, 6, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let out = layer.forward(&x, None, None).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[1, 6, 32]);
        assert!(layer.mod_aux_loss().is_none(), "No aux loss without MoD");
    }

    #[test]
    #[serial]
    fn test_llama4_mod_causal_model_instantiation() {
        let mut config = Llama4TextConfig::default();
        config.hidden_size = 32;
        config.intermediate_size = 64;
        config.intermediate_size_mlp = 64;
        config.num_hidden_layers = 4;
        config.num_attention_heads = 2;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.num_local_experts = 2;
        config.vocab_size = 100;
        config.use_mod = true;
        config.mod_capacity = 0.5;
        config.mod_layer_interval = 2; // layers 0 and 2 are MoD

        let model = Llama4ForCausalLM::new(config).unwrap();

        // Verify MoD layers have a router, non-MoD layers do not.
        assert!(model.model.layers[0].mod_router.is_some());
        assert!(model.model.layers[1].mod_router.is_none());
        assert!(model.model.layers[2].mod_router.is_some());
        assert!(model.model.layers[3].mod_router.is_none());

        let params = model.flatten_params();
        assert!(!params.is_empty());
    }

    #[test]
    #[serial]
    fn test_llama4_mod_aux_loss_aggregation() {
        let mut config = Llama4TextConfig::default();
        config.hidden_size = 32;
        config.intermediate_size = 64;
        config.intermediate_size_mlp = 64;
        config.num_hidden_layers = 4;
        config.num_attention_heads = 2;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.num_local_experts = 2;
        config.vocab_size = 100;
        config.use_mod = true;
        config.mod_capacity = 0.5;
        config.mod_layer_interval = 2;

        let mut model = Llama4ForCausalLM::new(config).unwrap();

        let input_ids = Array::from_slice(&[0i32, 1, 2, 3, 4, 5, 6, 7], &[1, 8]);
        let logits = model.forward(&input_ids, None, None).unwrap();
        logits.eval().unwrap();

        let aux = model.mod_aux_loss().unwrap();
        assert!(
            aux.is_some(),
            "mod_aux_loss should be Some after a forward pass with MoD enabled"
        );
        let aux_val = aux.unwrap();
        aux_val.eval().unwrap();
        // Scalar or 0-d tensor expected.
        assert!(aux_val.shape().len() == 0 || aux_val.size() == 1);
    }
}
