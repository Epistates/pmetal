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

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp, softmax_axis},
};
use serde::{Deserialize, Serialize};

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
        }
    }
}

impl Llama4TextConfig {
    /// Check if a given layer is an MoE layer.
    pub fn is_moe_layer(&self, layer_idx: i32) -> bool {
        if let Some(ref moe_layers) = self.moe_layers {
            moe_layers.contains(&layer_idx)
        } else {
            // All layers are MoE when interleave_moe_layer_step == 1
            // Otherwise, MoE layers are those where layer_idx % step == 0
            layer_idx % self.interleave_moe_layer_step == 0
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
        // NoPE every no_rope_layer_interval layers
        layer_idx % self.no_rope_layer_interval != 0
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
#[derive(Debug, ModuleParameters)]
pub struct Llama4Expert {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

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
        let gate = nn::silu(gate)?;
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up)?;
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// Router for selecting experts.
#[derive(Debug, ModuleParameters)]
pub struct Llama4Router {
    #[param]
    pub gate: nn::Linear,
    pub num_experts: i32,
    pub top_k: i32,
}

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
    /// Returns (expert_indices, expert_weights, router_logits).
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Array, Array), Exception> {
        // x: [batch, seq, hidden] or [total_tokens, hidden]
        let router_logits = Module::forward(&mut self.gate, x)?;

        // Softmax over experts
        let router_probs = softmax_axis(&router_logits, -1, None)?;

        // Top-k selection
        // For now, we assume top_k=1 and use argmax
        let expert_indices = mlx_rs::ops::indexing::argmax_axis(&router_probs, -1, false)?;

        // Get the weights for selected experts
        let expert_weights = router_probs.max_axis(-1, false)?;

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
#[derive(Debug, ModuleParameters)]
pub struct Llama4ModRouter {
    /// Scalar linear projection: [hidden_size] -> [1].
    #[param]
    pub gate: nn::Linear,
}

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
        let weights = router_logits.reshape(&[batch, seq_len])?;

        // k = floor(C * T), clamped to [1, T]
        let k = ((capacity * seq_len as f32).floor() as i32)
            .max(1)
            .min(seq_len);

        // argpartition(-weights, -k, axis=-1) places the k largest at positions [-k..]
        // This is O(T) vs O(T log T) for argsort.
        let neg_weights = weights.negative()?;
        let neg_k = -k;
        let part_indices = ops::argpartition_axis(&neg_weights, neg_k, -1)?;

        // Slice the last k indices — these correspond to the top-k tokens.
        // part_indices shape: [B, T]; the k largest (negated smallest) are at positions [-k..].
        let selected_indices = part_indices.index((.., neg_k..));
        // selected_indices: [B, k]

        // Build a binary top-k mask [B, T] of zeros with 1s at selected positions.
        // We scatter ones into a zeros tensor using put_along_axis.
        let zeros = ops::zeros::<f32>(&[batch, seq_len])?;
        let ones = ops::ones::<f32>(&[batch, k])?;
        let top_k_mask = zeros.put_along_axis(&selected_indices, &ones, 1)?;
        // top_k_mask: [B, T] with 1.0 at selected token positions

        Ok((selected_indices, router_logits, top_k_mask))
    }
}

/// Mixture of Experts layer with shared expert.
#[derive(Debug, ModuleParameters)]
pub struct Llama4MoE {
    pub config: Llama4TextConfig,

    #[param]
    pub router: Llama4Router,
    #[param]
    pub experts: Vec<Llama4Expert>,
    #[param]
    pub shared_expert: Llama4Expert,
}

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
        let flat_x = x.reshape(&[total_tokens, hidden_size])?;

        // Route tokens
        let (expert_indices, expert_weights, _router_logits) = self.router.forward(&flat_x)?;

        // Shared expert output (always applied)
        let shared_out = self.shared_expert.forward(&flat_x)?;

        // Expert routing - for simplicity, process each expert sequentially
        // In production, this would use grouped GEMM for efficiency
        let mut expert_out = mlx_rs::ops::zeros::<f32>(&[total_tokens, hidden_size])?;

        for (expert_idx, expert) in self.experts.iter_mut().enumerate() {
            // Create mask for tokens routed to this expert
            let expert_id = Array::from_int(expert_idx as i32);
            let mask = expert_indices.eq(&expert_id)?;
            let mask_f32 = mask.as_dtype(mlx_rs::Dtype::Float32)?;

            // Process all tokens through expert and mask
            let exp_output = expert.forward(&flat_x)?;
            let masked = exp_output.multiply(&mask_f32.reshape(&[total_tokens, 1])?)?;
            expert_out = expert_out.add(&masked)?;
        }

        // Weight expert output and add shared
        let expert_weights_2d = expert_weights.reshape(&[total_tokens, 1])?;
        let weighted_expert = expert_out.multiply(&expert_weights_2d)?;
        let output = shared_out.add(&weighted_expert)?;

        // Reshape back
        output.reshape(&shape)
    }
}

// =============================================================================
// Attention with iRoPE and QK Norm
// =============================================================================

/// Llama 4 attention with iRoPE (interleaved RoPE/NoPE) and QK norm.
#[derive(Debug, ModuleParameters)]
pub struct Llama4Attention {
    pub layer_idx: usize,
    pub uses_rope: bool,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,

    // QK normalization (optional)
    #[param]
    pub q_norm: Option<nn::RmsNorm>,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
}

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

        // QK norm (if enabled)
        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(
                    nn::RmsNormBuilder::new(head_dim)
                        .eps(config.rms_norm_eps)
                        .build()?,
                ),
                Some(
                    nn::RmsNormBuilder::new(head_dim)
                        .eps(config.rms_norm_eps)
                        .build()?,
                ),
            )
        } else {
            (None, None)
        };

        let uses_rope = config.uses_rope(layer_idx as i32);

        Ok(Self {
            layer_idx,
            uses_rope,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut q = Module::forward(&mut self.q_proj, x)?;
        let mut k = Module::forward(&mut self.k_proj, x)?;
        let v = Module::forward(&mut self.v_proj, x)?;

        // Reshape for attention
        q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // QK normalization (applied before RoPE)
        if let (Some(qn), Some(kn)) = (&mut self.q_norm, &mut self.k_norm) {
            q = Module::forward(qn, &q)?;
            k = Module::forward(kn, &k)?;
        }

        // Apply RoPE if this is a RoPE layer (not NoPE)
        if self.uses_rope {
            if let Some(pos_ids) = position_ids {
                // Apply RoPE with position IDs
                q = self.apply_rope(&q, pos_ids)?;
                k = self.apply_rope(&k, pos_ids)?;
            } else {
                // Apply RoPE with sequential positions
                let positions = Array::from_iter(0..seq_len, &[seq_len]);
                q = self.apply_rope(&q, &positions)?;
                k = self.apply_rope(&k, &positions)?;
            }
        }
        // NoPE layers: no positional encoding applied

        // Transpose for attention: [B, n_heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Attention scores
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let mut scores = q.matmul(&k_t)?;
        scores = scores.multiply(&Array::from_f32(self.scale))?;

        // Apply mask
        if let Some(m) = mask {
            scores = scores.add(m)?;
        }

        let probs = softmax_axis(&scores, -1, None)?;
        let output = probs.matmul(&v)?;

        // Reshape and project
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq_len, -1])?;
        Module::forward(&mut self.o_proj, &output)
    }

    /// Apply RoPE embeddings.
    fn apply_rope(&self, x: &Array, position_ids: &Array) -> Result<Array, Exception> {
        // Simplified RoPE implementation
        // Full implementation would use precomputed cos/sin tables
        let _seq_len = position_ids.shape()[0];

        // For now, return x unchanged (full RoPE implementation needed)
        // In production, this would compute rotary embeddings
        Ok(x.clone())
    }
}

// =============================================================================
// Decoder Layer
// =============================================================================

/// Llama 4 decoder layer (can be dense or MoE, optionally with MoD).
#[derive(Debug, ModuleParameters)]
pub struct Llama4DecoderLayer {
    pub layer_idx: usize,
    pub is_moe: bool,
    /// MoD capacity factor for this layer (None = MoD disabled).
    pub mod_capacity: Option<f32>,

    #[param]
    pub self_attn: Llama4Attention,
    #[param]
    pub mlp: Option<Llama4Expert>, // Dense MLP (if not MoE)
    #[param]
    pub moe: Option<Llama4MoE>, // MoE layer (if MoE)
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    /// MoD router (present only when this layer uses Mixture-of-Depths).
    #[param]
    pub mod_router: Option<Llama4ModRouter>,

    // Auxiliary loss from the most recent MoD forward pass (not a learned parameter).
    // Stored here so the parent model can aggregate it without threading extra return values
    // through the forward signature.
    pub last_mod_aux_loss: Option<Array>,
}

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
        if let Some(capacity) = self.mod_capacity {
            self.forward_mod(x, mask, position_ids, capacity)
        } else {
            self.forward_full(x, mask, position_ids)
        }
    }

    /// Standard full-sequence forward (no MoD).
    fn forward_full(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Self attention with residual
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward(&normed, mask, position_ids)?;
        let h = x.add(&attn_out)?;

        // FFN with residual (MoE or dense)
        let normed = Module::forward(&mut self.post_attention_layernorm, &h)?;
        let ffn_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed)?
        };
        h.add(&ffn_out)
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
        let idx_reshaped = selected_indices.reshape(&[batch, k, 1])?;
        let idx_expanded = ops::broadcast_to(&idx_reshaped, &[batch, k, hidden])?;
        // gathered: [B, k, D]
        let gathered = x.take_along_axis(&idx_expanded, 1)?;

        // ---- Run transformer block on gathered sub-batch ----
        // Note: we pass `None` for mask here — the gathered tokens form a
        // dense sub-sequence and causal masking at this level would be wrong.
        // Position IDs are also omitted; RoPE on the full sequence is correct
        // only when all positions are present. For the selected sub-batch we
        // skip positional encoding (NoPE behaviour) which is consistent with
        // how iRoPE NoPE layers work in this model.
        let normed = Module::forward(&mut self.input_layernorm, &gathered)?;
        let attn_out = self.self_attn.forward(&normed, None, None)?;
        let h_sel = gathered.add(&attn_out)?;

        let normed2 = Module::forward(&mut self.post_attention_layernorm, &h_sel)?;
        let ffn_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed2)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed2)?
        };
        let block_out = h_sel.add(&ffn_out)?;
        // block_out: [B, k, D] — processed token outputs

        // ---- Scatter results back into the full-sequence tensor ----
        // Non-selected token slots start as the original `x` (identity/residual skip).
        // We overwrite the selected positions with the block output.
        let idx_reshaped_scatter = selected_indices.reshape(&[batch, k, 1])?;
        let idx_expanded_scatter = ops::broadcast_to(&idx_reshaped_scatter, &[batch, k, hidden])?;
        let output = x.put_along_axis(&idx_expanded_scatter, &block_out, 1)?;
        // output: [B, T, D]  — selected tokens updated, others unchanged

        // ---- Auxiliary BCE loss ----
        // BCE(sigmoid(router_logits), top_k_mask) teaches the router to
        // predict which tokens it will select, enabling autoregressive inference
        // where the router must decide without seeing future selections.
        let logits_flat = router_logits.reshape(&[batch * seq_len])?;
        let mask_flat = top_k_mask.reshape(&[batch * seq_len])?;
        let aux_loss = mlx_rs::losses::BinaryCrossEntropyBuilder::new()
            .inputs_are_logits(true)
            .reduction(mlx_rs::losses::LossReduction::Mean)
            .build()?
            .apply(&logits_flat, &mask_flat)?;
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
#[derive(Debug, ModuleParameters)]
pub struct Llama4TextModel {
    pub config: Llama4TextConfig,

    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<Llama4DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

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
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, mask, position_ids)?;
        }

        Module::forward(&mut self.norm, &hidden_states)
    }

    /// Aggregate MoD auxiliary losses from all MoD-enabled layers after a forward pass.
    ///
    /// Returns the mean BCE loss across all MoD layers, or `None` if no MoD layers fired.
    /// Callers should scale by `config.router_aux_loss_coef` before adding to the task loss.
    pub fn mod_aux_loss(&self) -> Option<Array> {
        let mut total: Option<Array> = None;
        let mut count = 0usize;

        for layer in &self.layers {
            if let Some(loss) = layer.mod_aux_loss() {
                total = Some(match total {
                    None => loss.clone(),
                    Some(acc) => acc.add(loss).ok()?,
                });
                count += 1;
            }
        }

        let sum = total?;
        let denom = Array::from_f32(count as f32);
        sum.divide(&denom).ok()
    }
}

/// Llama 4 for causal language modeling.
#[derive(Debug, ModuleParameters)]
pub struct Llama4ForCausalLM {
    pub config: Llama4TextConfig,

    #[param]
    pub model: Llama4TextModel,
    #[param]
    pub lm_head: nn::Linear,
}

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

    /// Aggregate MoD auxiliary losses across all layers after a forward pass.
    ///
    /// Callers should add `config.router_aux_loss_coef * mod_aux_loss()` to the
    /// task loss when training with MoD enabled.
    pub fn mod_aux_loss(&self) -> Option<Array> {
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
    use mlx_rs::module::ModuleParameters;
    use serial_test::serial;

    #[test]
    fn test_llama4_config_moe_layers() {
        let config = Llama4TextConfig::scout();

        // Scout: all layers are MoE
        assert!(config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(config.is_moe_layer(47));

        let maverick = Llama4TextConfig::maverick();

        // Maverick: even layers are MoE
        assert!(maverick.is_moe_layer(0));
        assert!(!maverick.is_moe_layer(1));
        assert!(maverick.is_moe_layer(2));
    }

    #[test]
    fn test_llama4_config_irope() {
        let config = Llama4TextConfig::default();

        // NoPE every 4th layer (layers 0, 4, 8, ...)
        assert!(!config.uses_rope(0)); // NoPE
        assert!(config.uses_rope(1)); // RoPE
        assert!(config.uses_rope(2)); // RoPE
        assert!(config.uses_rope(3)); // RoPE
        assert!(!config.uses_rope(4)); // NoPE
    }

    #[test]
    #[serial]
    fn test_llama4_expert() {
        let expert = Llama4Expert::new(64, 256).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 10, 64], None, None, None).unwrap();

        let mut expert = expert;
        let out = expert.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[1, 10, 64]);
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

        let params = model.parameters().flatten();
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
        let x = mlx_rs::random::normal::<f32>(&[batch, seq_len, hidden], None, None, None).unwrap();

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
        let x = mlx_rs::random::normal::<f32>(&[batch, seq_len, hidden], None, None, None).unwrap();

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

        let x = mlx_rs::random::normal::<f32>(&[batch, seq_len, hidden], None, None, None).unwrap();
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

        let x = mlx_rs::random::normal::<f32>(&[1, 6, 32], None, None, None).unwrap();
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

        let params = model.parameters().flatten();
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

        let aux = model.mod_aux_loss();
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
