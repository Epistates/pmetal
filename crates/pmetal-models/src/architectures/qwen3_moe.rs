//! Qwen3-MoE model architecture.
//!
//! Implements Qwen3-MoE with:
//! - Mixture of Experts with top-k routing (softmax-based)
//! - Configurable sparse step (decoder_sparse_step) to control MoE layer frequency
//! - RMSNorm applied to Q and K before RoPE (q_norm, k_norm)
//! - SwitchGLU-style expert MLP with gather_mm

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param},
    nn,
    ops::indexing::IndexOp,
};
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopeScaling, apply_rope},
};
use pmetal_mlx::kv_cache::KVCache;
// MoE block uses pmetal_mlx::moe::Expert directly for individual expert MLPs
use serde::{Deserialize, Serialize};

/// Qwen3-MoE model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3MoEConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Intermediate size for dense MLP layers.
    pub intermediate_size: i32,
    /// Intermediate size for MoE expert MLPs.
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    /// Head dimension (fixed at 128 for Qwen3).
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Number of experts.
    #[serde(default = "default_num_experts")]
    pub num_experts: i32,
    /// Number of experts to route to per token.
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    /// Sparse decoder step - MoE layers are used every N-th layer.
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: i32,
    /// Layers that should use dense MLP instead of MoE.
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    /// Whether to normalize top-k routing probabilities.
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<std::collections::HashMap<String, serde_json::Value>>,
}

fn default_model_type() -> String {
    "qwen3_moe".to_string()
}
fn default_vocab_size() -> i32 {
    151936
}
fn default_head_dim() -> i32 {
    128
}
fn default_max_position_embeddings() -> i32 {
    32768
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_num_experts() -> i32 {
    64
}
fn default_num_experts_per_tok() -> i32 {
    8
}
fn default_decoder_sparse_step() -> i32 {
    1
}
fn default_norm_topk_prob() -> bool {
    true
}

impl Qwen3MoEConfig {
    /// Get the number of KV heads (defaults to num_attention_heads if not specified).
    pub fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the MoE intermediate size (defaults to intermediate_size if not specified).
    pub fn get_moe_intermediate_size(&self) -> i32 {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    /// Check if layer at index should use MoE.
    /// Returns true if:
    /// 1. The layer is not in mlp_only_layers
    /// 2. The layer index + 1 is divisible by decoder_sparse_step
    pub fn use_moe_at(&self, layer_idx: usize) -> bool {
        let layer_idx_i32 = layer_idx as i32;
        // Check if this layer is in the mlp_only list
        if self.mlp_only_layers.contains(&layer_idx_i32) {
            return false;
        }
        // Check if this is a sparse step layer
        self.num_experts > 0 && ((layer_idx_i32 + 1) % self.decoder_sparse_step == 0)
    }
}

impl Default for Qwen3MoEConfig {
    fn default() -> Self {
        Self {
            model_type: "qwen3_moe".to_string(),
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 5632,
            moe_intermediate_size: Some(1408),
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            head_dim: 128,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            num_experts: 64,
            num_experts_per_tok: 8,
            decoder_sparse_step: 1,
            mlp_only_layers: vec![],
            norm_topk_prob: true,
            rope_scaling: None,
        }
    }
}

/// Qwen3-MoE attention with Q/K normalization before RoPE.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoEAttention {
    /// Configuration.
    #[allow(dead_code)]
    // Retained for future dynamic config access (e.g. sliding window, rope scaling)
    config: Qwen3MoEConfig,
    /// Number of attention heads.
    n_heads: i32,
    /// Number of KV heads.
    n_kv_heads: i32,
    /// Head dimension.
    head_dim: i32,
    /// Attention scale.
    scale: f32,
    /// RoPE theta.
    #[allow(dead_code)]
    // Retained as authoritative source; effective_base is the derived working value
    rope_theta: f32,
    /// RoPE position scale (from rope_scaling config).
    rope_scale: f32,
    /// Effective RoPE base after scaling.
    effective_base: f32,
    /// Query projection.
    #[param]
    pub q_proj: nn::Linear,
    /// Key projection.
    #[param]
    pub k_proj: nn::Linear,
    /// Value projection.
    #[param]
    pub v_proj: nn::Linear,
    /// Output projection.
    #[param]
    pub o_proj: nn::Linear,
    /// Query normalization.
    #[param]
    pub q_norm: nn::RmsNorm,
    /// Key normalization.
    #[param]
    pub k_norm: nn::RmsNorm,
}

impl Qwen3MoEAttention {
    /// Create a new attention layer.
    pub fn new(config: Qwen3MoEConfig) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let scale = (head_dim as f32).powf(-0.5);

        // Parse rope_scaling from config
        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(RopeScaling::from_config_map)
            .unwrap_or(RopeScaling::None);
        let rope_scale = rope_scaling.scale();
        let effective_base = rope_scaling.effective_base(config.rope_theta, head_dim);

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

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            rope_theta: config.rope_theta,
            rope_scale,
            effective_base,
            config,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        })
    }

    /// Forward pass through attention.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape and apply per-head normalization
        q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Apply Q/K normalization (Qwen3 specific)
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;

        // Transpose to [batch, heads, seq, head_dim]
        q = q.transpose_axes(&[0, 2, 1, 3])?;
        k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(
            &q,
            self.head_dim,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;
        let k = apply_rope(
            &k,
            self.head_dim,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;

        // Update cache if provided
        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        // Use fused SDPA
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Transpose back and project
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        self.o_proj.forward(&output)
    }
}

/// Dense MLP for non-MoE layers.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoEDenseMLP {
    /// Gate projection.
    #[param]
    pub gate_proj: nn::Linear,
    /// Up projection.
    #[param]
    pub up_proj: nn::Linear,
    /// Down projection.
    #[param]
    pub down_proj: nn::Linear,
}

impl Qwen3MoEDenseMLP {
    /// Create a new MLP.
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

    /// Forward pass through MLP.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // SwiGLU: silu(gate) * up
        let activated = nn::silu(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }
}

/// MoE block with softmax routing for Qwen3-MoE.
///
/// Uses cached stacked expert weights plus GPU batched gathers/matmuls for the main inference path.
/// The cache is rebuilt automatically when the underlying expert weight handles change.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoEBlock {
    /// Number of experts.
    pub num_experts: usize,
    /// Top-k experts per token.
    pub top_k: usize,
    /// Whether to normalize top-k probabilities.
    pub norm_topk_prob: bool,
    /// Gate projection (routes to experts).
    #[param]
    pub gate: nn::Linear,
    /// Expert MLPs.
    #[param]
    pub experts: Vec<pmetal_mlx::moe::Expert>,
    /// Stacked gate projection weights `[num_experts, hidden, intermediate]`.
    pub stacked_gate_proj: Option<Array>,
    /// Stacked up projection weights `[num_experts, hidden, intermediate]`.
    pub stacked_up_proj: Option<Array>,
    /// Stacked down projection weights `[num_experts, intermediate, hidden]`.
    pub stacked_down_proj: Option<Array>,
    /// Signature of the currently stacked expert weight handles.
    pub stacked_weight_signature: Option<Vec<usize>>,
}

impl Qwen3MoEBlock {
    /// Create a new MoE block.
    pub fn new(config: &Qwen3MoEConfig) -> Result<Self, Exception> {
        let num_experts = config.num_experts as usize;
        let moe_intermediate = config.get_moe_intermediate_size();

        let gate = nn::LinearBuilder::new(config.hidden_size, config.num_experts)
            .bias(false)
            .build()?;

        let experts = (0..num_experts)
            .map(|_| pmetal_mlx::moe::Expert::new(config.hidden_size, moe_intermediate))
            .collect();

        Ok(Self {
            num_experts,
            top_k: config.num_experts_per_tok as usize,
            norm_topk_prob: config.norm_topk_prob,
            gate,
            experts,
            stacked_gate_proj: None,
            stacked_up_proj: None,
            stacked_down_proj: None,
            stacked_weight_signature: None,
        })
    }

    fn current_stacked_weight_signature(&self) -> Vec<usize> {
        let mut signature = Vec::with_capacity(self.experts.len() * 3);
        for expert in &self.experts {
            signature.push(expert.w1.weight.as_ref().as_ptr().ctx as usize);
            signature.push(expert.w3.weight.as_ref().as_ptr().ctx as usize);
            signature.push(expert.w2.weight.as_ref().as_ptr().ctx as usize);
        }
        signature
    }

    fn stack_expert_weights(&self) -> Result<(Array, Array, Array), Exception> {
        let gate_weights: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| expert.w1.weight.as_ref().t())
            .collect();
        let up_weights: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| expert.w3.weight.as_ref().t())
            .collect();
        let down_weights: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| expert.w2.weight.as_ref().t())
            .collect();
        let gate_weight_refs: Vec<&Array> = gate_weights.iter().collect();
        let up_weight_refs: Vec<&Array> = up_weights.iter().collect();
        let down_weight_refs: Vec<&Array> = down_weights.iter().collect();

        Ok((
            mlx_rs::ops::stack_axis(&gate_weight_refs, 0)?,
            mlx_rs::ops::stack_axis(&up_weight_refs, 0)?,
            mlx_rs::ops::stack_axis(&down_weight_refs, 0)?,
        ))
    }

    fn ensure_stacked_moe(&mut self) -> Result<(), Exception> {
        let signature = self.current_stacked_weight_signature();
        let needs_refresh = self.stacked_gate_proj.is_none()
            || self.stacked_up_proj.is_none()
            || self.stacked_down_proj.is_none()
            || self.stacked_weight_signature.as_ref() != Some(&signature);

        if needs_refresh {
            let (stacked_gate_proj, stacked_up_proj, stacked_down_proj) =
                self.stack_expert_weights()?;
            stacked_gate_proj.eval()?;
            stacked_up_proj.eval()?;
            stacked_down_proj.eval()?;
            self.stacked_gate_proj = Some(stacked_gate_proj);
            self.stacked_up_proj = Some(stacked_up_proj);
            self.stacked_down_proj = Some(stacked_down_proj);
            self.stacked_weight_signature = Some(signature);
        }

        Ok(())
    }

    /// Eagerly build or refresh the stacked expert-weight cache.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        self.ensure_stacked_moe()
    }

    /// Whether the stacked expert-weight cache is currently populated.
    pub fn has_stacked_moe(&self) -> bool {
        self.stacked_gate_proj.is_some()
            && self.stacked_up_proj.is_some()
            && self.stacked_down_proj.is_some()
    }

    fn route_topk(
        &mut self,
        hidden_flat: &Array,
    ) -> Result<(i32, i32, Array, Array), Exception> {
        let batch_seq = hidden_flat.dim(0);
        let hidden_size = hidden_flat.dim(1);

        let gate_logits = self.gate.forward(hidden_flat)?;
        let gate_logits_f32 = if gate_logits.dtype() != mlx_rs::Dtype::Float32 {
            gate_logits.as_type::<f32>()?
        } else {
            gate_logits
        };
        let routing_probs = mlx_rs::ops::softmax_axis(&gate_logits_f32, -1, None)?;

        let top_k = self.top_k as i32;
        let neg_k = -top_k;
        let part_indices = mlx_rs::ops::argpartition_axis(&routing_probs, neg_k, -1)?;
        let top_indices = part_indices.index((.., neg_k..)).as_type::<i32>()?;
        let top_weights = routing_probs.take_along_axis(&top_indices, -1)?;

        let normalized_weights = if self.norm_topk_prob {
            let weight_sum = top_weights.sum_axis(-1, Some(true))?;
            let safe_sum = mlx_rs::ops::maximum(&weight_sum, &Array::from_f32(1e-8))?;
            top_weights.divide(&safe_sum)?
        } else {
            top_weights
        };

        Ok((batch_seq, hidden_size, top_indices, normalized_weights))
    }

    #[cfg(test)]
    fn forward_per_expert(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden_size = shape[shape.len() - 1];
        let hidden_flat = x.reshape(&[batch_seq, hidden_size])?;
        let (_batch_seq, _hidden_size, top_indices, normalized_weights) =
            self.route_topk(&hidden_flat)?;

        top_indices.eval()?;
        normalized_weights.eval()?;

        let n_tokens = batch_seq as usize;
        let expert_indices: Vec<i32> = top_indices.as_slice().to_vec();
        let expert_weights: Vec<f32> = normalized_weights.as_slice().to_vec();

        let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.num_experts];
        for token_idx in 0..n_tokens {
            for slot in 0..self.top_k {
                let flat_idx = token_idx * self.top_k + slot;
                let expert_id = expert_indices[flat_idx] as usize;
                let weight = expert_weights[flat_idx];
                if expert_id < self.num_experts {
                    expert_assignments[expert_id].push((token_idx, weight));
                }
            }
        }

        let mut final_output = mlx_rs::ops::zeros_dtype(&[batch_seq, hidden_size], hidden_flat.dtype())?;
        for (expert_idx, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }

            let token_indices: Vec<i32> = assignments.iter().map(|&(idx, _)| idx as i32).collect();
            let weights: Vec<f32> = assignments.iter().map(|&(_, w)| w).collect();

            let idx_array = Array::from_slice(&token_indices, &[token_indices.len() as i32]);
            let weight_array = Array::from_slice(&weights, &[weights.len() as i32, 1]);
            let expert_input = hidden_flat.take_axis(&idx_array, 0)?;
            let expert_out = self.experts[expert_idx].forward(&expert_input)?;
            let weighted_out = expert_out.multiply(&weight_array)?;
            let updates_3d = weighted_out.reshape(&[token_indices.len() as i32, 1, hidden_size])?;
            final_output = mlx_rs::ops::indexing::scatter_add_single(
                &final_output,
                &idx_array,
                &updates_3d,
                0,
            )?;
        }

        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        final_output.reshape(&output_shape)
    }

    fn batched_matmul(&self, x: &Array, w: &Array) -> Result<Array, Exception> {
        let x_expanded = x.reshape(&[x.dim(0), 1, x.dim(1)])?;
        let result = mlx_rs::ops::matmul(&x_expanded, w)?;
        result.squeeze_axes(&[1])
    }

    fn forward_stacked(&mut self, x: &Array) -> Result<Array, Exception> {
        self.ensure_stacked_moe()?;

        let shape = x.shape();
        let hidden_flat = x.reshape(&[shape[..shape.len() - 1].iter().product(), shape[shape.len() - 1]])?;
        let (batch_seq, hidden_size, top_indices, normalized_weights) = self.route_topk(&hidden_flat)?;
        let top_k = self.top_k as i32;
        let mut output = mlx_rs::ops::zeros_dtype(&[batch_seq, hidden_size], hidden_flat.dtype())?;
        for slot in 0..top_k {
            let slot_experts = top_indices.index((.., slot));
            let slot_weights = normalized_weights.index((.., slot..slot + 1));

            let gate_weights = self
                .stacked_gate_proj
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0)?;
            let up_weights = self
                .stacked_up_proj
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0)?;
            let down_weights = self
                .stacked_down_proj
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0)?;

            let gate_out = self.batched_matmul(&hidden_flat, &gate_weights)?;
            let up_out = self.batched_matmul(&hidden_flat, &up_weights)?;
            let activated = nn::silu(&gate_out)?.multiply(&up_out)?;
            let slot_out = self.batched_matmul(&activated, &down_weights)?;
            output = output.add(&slot_out.multiply(&slot_weights)?)?;
        }

        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        output.reshape(&output_shape)
    }

    /// Forward pass using the stacked batched-gather path.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        self.forward_stacked(x)
    }
}

/// Feed-forward for a decoder layer - either dense MLP or MoE.
///
/// C8: Implements `ModuleParameters` so that expert weights are visible to the
/// optimizer and parameter loading. The derive macro cannot handle enums, so
/// we implement the trait manually, delegating to the inner variant.
#[derive(Debug)]
pub enum Qwen3MoEFeedForward {
    /// Dense MLP.
    Dense(Qwen3MoEDenseMLP),
    /// Mixture of Experts.
    MoE(Qwen3MoEBlock),
}

impl ModuleParameters for Qwen3MoEFeedForward {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(mlp) => mlp.num_parameters(),
            Self::MoE(moe) => moe.num_parameters(),
        }
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(mlp) => mlp.parameters(),
            Self::MoE(moe) => moe.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Dense(mlp) => mlp.parameters_mut(),
            Self::MoE(moe) => moe.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(mlp) => mlp.trainable_parameters(),
            Self::MoE(moe) => moe.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(mlp) => mlp.freeze_parameters(recursive),
            Self::MoE(moe) => moe.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(mlp) => mlp.unfreeze_parameters(recursive),
            Self::MoE(moe) => moe.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(mlp) => mlp.all_frozen(),
            Self::MoE(moe) => moe.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(mlp) => mlp.any_frozen(),
            Self::MoE(moe) => moe.any_frozen(),
        }
    }
}

impl Qwen3MoEFeedForward {
    /// Forward pass.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::MoE(moe) => moe.forward(x),
        }
    }

    /// Eagerly build stacked expert caches for MoE variants.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        match self {
            Self::Dense(_) => Ok(()),
            Self::MoE(moe) => moe.init_stacked_moe(),
        }
    }
}

/// Qwen3-MoE decoder layer.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoEDecoderLayer {
    /// Configuration.
    pub config: Qwen3MoEConfig,
    /// Self-attention.
    #[param]
    pub self_attn: Qwen3MoEAttention,
    /// Input layer norm.
    #[param]
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    /// Feed-forward (MLP or MoE).
    #[param]
    pub ffn: Qwen3MoEFeedForward,
}

impl Qwen3MoEDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: Qwen3MoEConfig, layer_idx: usize) -> Result<Self, Exception> {
        let self_attn = Qwen3MoEAttention::new(config.clone())?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        // Determine if this layer uses MoE
        let ffn = if config.use_moe_at(layer_idx) {
            Qwen3MoEFeedForward::MoE(Qwen3MoEBlock::new(&config)?)
        } else {
            Qwen3MoEFeedForward::Dense(Qwen3MoEDenseMLP::new(
                config.hidden_size,
                config.intermediate_size,
            )?)
        };

        Ok(Self {
            config,
            self_attn,
            input_layernorm,
            post_attention_layernorm,
            ffn,
        })
    }

    /// Forward pass through decoder layer.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        // Self-attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(&attn_out)?;

        // FFN with residual
        let normed = self.post_attention_layernorm.forward(&h)?;
        let ffn_out = self.ffn.forward(&normed)?;
        h.add(&ffn_out)
    }

    /// Eagerly build stacked expert caches for this layer's MoE path.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        self.ffn.init_stacked_moe()
    }
}

/// Qwen3-MoE base model (without LM head).
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoEModel {
    /// Configuration.
    pub config: Qwen3MoEConfig,
    /// Token embeddings.
    #[param]
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    #[param]
    pub layers: Vec<Qwen3MoEDecoderLayer>,
    /// Final layer norm.
    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3MoEModel {
    /// Create a new model.
    pub fn new(config: Qwen3MoEConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for layer_idx in 0..config.num_hidden_layers as usize {
            layers.push(Qwen3MoEDecoderLayer::new(config.clone(), layer_idx)?);
        }

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

    /// Forward pass through the model.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(input_ids)?;

        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    h = layer.forward(&h, mask, Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in self.layers.iter_mut() {
                    h = layer.forward(&h, mask, None)?;
                }
            }
        }

        self.norm.forward(&h)
    }

    /// Eagerly build stacked expert caches for all MoE layers.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        for layer in &mut self.layers {
            layer.init_stacked_moe()?;
        }
        Ok(())
    }
}

/// Full Qwen3-MoE model with LM head.
///
/// C7: When `tie_word_embeddings` is true, `lm_head` is `None` and the
/// forward pass uses the embedding weight directly for the language model head.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoE {
    /// Configuration.
    pub config: Qwen3MoEConfig,
    /// Base model.
    #[param]
    pub model: Qwen3MoEModel,
    /// LM head (None when tied to embedding weights).
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl Qwen3MoE {
    /// Create a new model.
    pub fn new(config: Qwen3MoEConfig) -> Result<Self, Exception> {
        let model = Qwen3MoEModel::new(config.clone())?;

        let lm_head = if config.tie_word_embeddings {
            // When tied, we use embed_tokens weight in forward - no separate lm_head
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };

        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    /// Forward pass through full model.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids, mask, cache)?;

        if let Some(ref mut head) = self.lm_head {
            head.forward(&hidden_states)
        } else {
            // Tied embeddings: use embed_tokens weight as linear projection
            // embed_tokens.weight is [vocab, hidden], so logits = hidden @ weight.T
            let embed_weight = self.model.embed_tokens.weight.value.as_ref();
            hidden_states.matmul(&embed_weight.t())
        }
    }

    /// Eagerly build stacked expert caches for all Qwen3-MoE layers.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        self.model.init_stacked_moe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_config_defaults() {
        let config = Qwen3MoEConfig::default();
        assert_eq!(config.num_experts, 64);
        assert_eq!(config.num_experts_per_tok, 8);
        assert!(config.use_moe_at(0)); // Layer 0 should use MoE with sparse_step=1
    }

    #[test]
    fn test_use_moe_at() {
        let mut config = Qwen3MoEConfig::default();

        // With sparse_step=2, only odd layers (0-indexed layer 1, 3, 5...) should be MoE
        config.decoder_sparse_step = 2;
        assert!(!config.use_moe_at(0)); // Layer 1: 0+1=1, 1%2=1 != 0
        assert!(config.use_moe_at(1)); // Layer 2: 1+1=2, 2%2=0
        assert!(!config.use_moe_at(2)); // Layer 3: 2+1=3, 3%2=1 != 0
        assert!(config.use_moe_at(3)); // Layer 4: 3+1=4, 4%2=0

        // Test mlp_only_layers
        config.mlp_only_layers = vec![1];
        assert!(!config.use_moe_at(1)); // Forced to dense
    }

    #[test]
    #[serial]
    fn test_create_attention() {
        let config = Qwen3MoEConfig::default();
        let attn = Qwen3MoEAttention::new(config);
        assert!(attn.is_ok());
    }

    #[test]
    #[serial]
    fn test_create_dense_mlp() {
        let mlp = Qwen3MoEDenseMLP::new(2048, 5632);
        assert!(mlp.is_ok());
    }

    fn tiny_moe_config() -> Qwen3MoEConfig {
        Qwen3MoEConfig {
            hidden_size: 32,
            intermediate_size: 64,
            moe_intermediate_size: Some(32),
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: 16,
            vocab_size: 100,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    #[test]
    #[serial]
    fn test_feed_forward_dense_dispatch() {
        let config = tiny_moe_config();
        let mut ffn = Qwen3MoEFeedForward::Dense(
            Qwen3MoEDenseMLP::new(config.hidden_size, config.intermediate_size).unwrap(),
        );
        let x = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[1, 4, config.hidden_size], None)
            .unwrap();
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, config.hidden_size]);
    }

    #[test]
    #[serial]
    fn test_feed_forward_moe_dispatch() {
        let config = tiny_moe_config();
        let mut ffn = Qwen3MoEFeedForward::MoE(Qwen3MoEBlock::new(&config).unwrap());
        let x = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[1, 4, config.hidden_size], None)
            .unwrap();
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, config.hidden_size]);
    }

    #[test]
    #[serial]
    fn test_tie_word_embeddings_none_lm_head() {
        let mut config = tiny_moe_config();
        config.tie_word_embeddings = true;
        let model = Qwen3MoE::new(config).unwrap();
        assert!(
            model.lm_head.is_none(),
            "lm_head should be None when tie_word_embeddings is true"
        );
    }

    #[test]
    #[serial]
    fn test_separate_lm_head() {
        let mut config = tiny_moe_config();
        config.tie_word_embeddings = false;
        let model = Qwen3MoE::new(config).unwrap();
        assert!(
            model.lm_head.is_some(),
            "lm_head should be Some when tie_word_embeddings is false"
        );
    }

    #[test]
    #[serial]
    fn test_module_parameters_delegation() {
        let config = tiny_moe_config();

        // Test Dense variant exposes parameters
        let ffn_dense = Qwen3MoEFeedForward::Dense(
            Qwen3MoEDenseMLP::new(config.hidden_size, config.intermediate_size).unwrap(),
        );
        assert!(
            ffn_dense.num_parameters() > 0,
            "Dense FFN should have parameters"
        );
        let params = ffn_dense.parameters();
        assert!(
            !params.entries.is_empty(),
            "Dense FFN parameters() should be non-empty"
        );

        // Test MoE variant exposes parameters
        let ffn_moe = Qwen3MoEFeedForward::MoE(Qwen3MoEBlock::new(&config).unwrap());
        assert!(
            ffn_moe.num_parameters() > 0,
            "MoE FFN should have parameters"
        );
        let params = ffn_moe.parameters();
        assert!(
            !params.entries.is_empty(),
            "MoE FFN parameters() should be non-empty"
        );

        // MoE should have more parameters than Dense (4 experts × 3 linear each + gate)
        assert!(
            ffn_moe.num_parameters() > ffn_dense.num_parameters(),
            "MoE should have more parameters than Dense"
        );
    }

    #[test]
    #[serial]
    fn test_moe_block_forward_shape() {
        let config = tiny_moe_config();
        let mut block = Qwen3MoEBlock::new(&config).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[2, 5, config.hidden_size], None)
            .unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 5, config.hidden_size]);
    }

    #[test]
    #[serial]
    fn test_moe_block_stacked_matches_per_expert_path() {
        let config = tiny_moe_config();
        let mut block = Qwen3MoEBlock::new(&config).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[2, 5, config.hidden_size], None)
            .unwrap();

        let reference = block.forward_per_expert(&x).unwrap();
        let fast = block.forward(&x).unwrap();
        reference.eval().unwrap();
        fast.eval().unwrap();

        assert!(block.has_stacked_moe());

        let max_diff = fast
            .subtract(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        max_diff.eval().unwrap();
        assert!(
            max_diff.item::<f32>() < 1e-4,
            "stacked qwen3-moe path drifted from per-expert reference"
        );
    }

    #[test]
    #[serial]
    fn test_moe_block_stacked_cache_refreshes_after_weight_change() {
        let config = tiny_moe_config();
        let mut block = Qwen3MoEBlock::new(&config).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[1, 4, config.hidden_size], None)
            .unwrap();

        let _ = block.forward(&x).unwrap();
        assert!(block.has_stacked_moe());

        block.experts[0].w1.weight = Param::new(
            Array::zeros::<f32>(&[config.get_moe_intermediate_size(), config.hidden_size]).unwrap(),
        );

        let reference = block.forward_per_expert(&x).unwrap();
        let fast = block.forward(&x).unwrap();
        reference.eval().unwrap();
        fast.eval().unwrap();

        let max_diff = fast
            .subtract(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        max_diff.eval().unwrap();
        assert!(
            max_diff.item::<f32>() < 1e-4,
            "stacked cache failed to refresh after expert weight change"
        );
    }

    #[test]
    #[serial]
    fn test_full_model_forward_shape() {
        let config = tiny_moe_config();
        let mut model = Qwen3MoE::new(config.clone()).unwrap();
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let out = model.forward(&input_ids, None, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, config.vocab_size]);
    }

    #[test]
    #[serial]
    fn test_full_model_init_stacked_moe_preserves_forward() {
        let config = tiny_moe_config();
        let mut model = Qwen3MoE::new(config.clone()).unwrap();
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);

        let reference = model.forward(&input_ids, None, None).unwrap();
        model.init_stacked_moe().unwrap();
        let fast = model.forward(&input_ids, None, None).unwrap();
        reference.eval().unwrap();
        fast.eval().unwrap();

        let max_diff = fast
            .subtract(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        max_diff.eval().unwrap();
        assert!(
            max_diff.item::<f32>() < 1e-4,
            "qwen3-moe full model drifted after stacked init"
        );
    }

    #[test]
    #[serial]
    fn test_decoder_layer_moe_vs_dense() {
        let config = tiny_moe_config();

        // Layer 0 with sparse_step=1 should be MoE
        let layer_moe = Qwen3MoEDecoderLayer::new(config.clone(), 0).unwrap();
        assert!(
            matches!(layer_moe.ffn, Qwen3MoEFeedForward::MoE(_)),
            "Layer 0 should be MoE with sparse_step=1"
        );

        // Force layer 0 to be dense via mlp_only_layers
        let mut config_dense = config.clone();
        config_dense.mlp_only_layers = vec![0];
        let layer_dense = Qwen3MoEDecoderLayer::new(config_dense, 0).unwrap();
        assert!(
            matches!(layer_dense.ffn, Qwen3MoEFeedForward::Dense(_)),
            "Layer 0 should be Dense when in mlp_only_layers"
        );
    }

    #[test]
    #[serial]
    fn test_freeze_unfreeze_delegation() {
        let config = tiny_moe_config();
        let mut ffn = Qwen3MoEFeedForward::MoE(Qwen3MoEBlock::new(&config).unwrap());

        // Freeze all parameters
        ffn.freeze_parameters(true);
        assert_eq!(ffn.all_frozen(), Some(true));

        // Unfreeze all parameters
        ffn.unfreeze_parameters(true);
        assert_eq!(ffn.all_frozen(), Some(false));
    }
}
