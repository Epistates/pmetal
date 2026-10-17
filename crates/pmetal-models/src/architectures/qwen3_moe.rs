//! Qwen3-MoE model architecture.
//!
//! Implements Qwen3-MoE with:
//! - Mixture of Experts with top-k routing (softmax-based)
//! - Configurable sparse step (decoder_sparse_step) to control MoE layer frequency
//! - RMSNorm applied to Q and K before RoPE (q_norm, k_norm)
//! - SwitchGLU-style expert MLP with gather_mm

use pmetal_cache::kv_cache::KVCache;
use pmetal_runtime::compat::{
    Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, VisitLinears, nn,
};
use pmetal_runtime::decode::last_token_hidden;
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopePositions, rope_embedding},
};
use pmetal_runtime::rope::{RopeConfig, RotaryEmbedding};
// MoE block uses pmetal_runtime::nn::Expert directly for individual expert MLPs
use serde::{Deserialize, Serialize};

use crate::checkpointing::checkpointed_layer;
use crate::common::switch_mlp::SwitchMlp;
use crate::decoder_layer::{AttentionModule, DecoderLayer, MlpModule, std_pre_norm_forward};

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
    /// RoPE scaling (YaRN on the Qwen3 cards), read by
    /// [`pmetal_runtime::rope`].
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// transformers v5's spelling of the same, with `rope_theta` inside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<serde_json::Value>,
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
    /// The rotary embedding `rope_scaling` / `rope_parameters` describe; an
    /// unknown `rope_type` is an error naming it.
    pub fn rotary(&self) -> Result<RotaryEmbedding, Exception> {
        crate::common::rotary_embedding(
            &self.model_type,
            self.head_dim,
            RopeConfig {
                rope_scaling: self.rope_scaling.as_ref(),
                rope_parameters: self.rope_parameters.as_ref(),
                rope_theta: Some(self.rope_theta as f64),
                max_position_embeddings: Some(self.max_position_embeddings as f64),
                ..RopeConfig::default()
            },
            1.0,
            false,
        )
    }

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
            rope_parameters: None,
        }
    }
}

/// Qwen3-MoE attention with Q/K normalization before RoPE.
#[derive(Debug)]
pub struct Qwen3MoEAttention {
    /// Number of attention heads.
    n_heads: i32,
    /// Number of KV heads.
    n_kv_heads: i32,
    /// Head dimension.
    head_dim: i32,
    /// Attention scale.
    scale: f32,
    /// The rotary embedding, YaRN or other scaling included.
    pub rotary: RotaryEmbedding,
    /// Query projection.
    pub q_proj: nn::Linear,
    /// Key projection.
    pub k_proj: nn::Linear,
    /// Value projection.
    pub v_proj: nn::Linear,
    /// Output projection.
    pub o_proj: nn::Linear,
    /// Query normalization.
    pub q_norm: nn::RmsNorm,
    /// Key normalization.
    pub k_norm: nn::RmsNorm,
}
impl_module_params!(Qwen3MoEAttention; q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

impl Qwen3MoEAttention {
    /// Create a new attention layer.
    pub fn new(config: Qwen3MoEConfig) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let scale = (head_dim as f32).powf(-0.5);

        let rotary = config.rotary()?;

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
            rotary,
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
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let mut cache = cache;

        // Project Q, K, V
        let mut q = self.q_proj.forward(x);
        let mut k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape and apply per-head normalization
        q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply Q/K normalization (Qwen3 specific)
        q = self.q_norm.forward(&q);
        k = self.k_norm.forward(&k);

        // Transpose to [batch, heads, seq, head_dim]
        q = q.transpose_axes(&[0, 2, 1, 3]);
        k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        // Apply RoPE
        let rope_positions = RopePositions::resolve(
            positions,
            cache
                .as_ref()
                .map_or(0, |(c, layer)| c.rope_offset_for(*layer)),
        );
        let q = rope_embedding(&q, rope_positions, &self.rotary);
        let k = rope_embedding(&k, rope_positions, &self.rotary);

        // Use fused SDPA
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });

        if mask.is_none() {
            if let Some((cache_ref, layer_idx)) = cache.as_mut() {
                if let Some(output) =
                    (*cache_ref).try_compressed_attention(*layer_idx, &q, &k, &v, &attn_config)?
                {
                    let output = output.transpose_axes(&[0, 2, 1, 3]);
                    let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);
                    return Ok(self.o_proj.forward(&output));
                }
            }
        }

        // Update cache if provided
        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Transpose back and project
        let output = output.transpose_axes(&[0, 2, 1, 3]);
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);

        Ok(self.o_proj.forward(&output))
    }
}

/// Dense MLP for non-MoE layers.
#[derive(Debug)]
pub struct Qwen3MoEDenseMLP {
    /// Gate projection.
    pub gate_proj: nn::Linear,
    /// Up projection.
    pub up_proj: nn::Linear,
    /// Down projection.
    pub down_proj: nn::Linear,
}
impl_module_params!(Qwen3MoEDenseMLP; gate_proj, up_proj, down_proj);

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
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        // SwiGLU: silu(gate) * up
        let activated = nn::silu(&gate).multiply(&up);
        Ok(self.down_proj.forward(&activated))
    }
}

/// Qwen3-MoE's router: the gate matrix, `[num_experts, hidden]`.
///
/// A bare matrix and not a `Linear`, so its parameter path is the checkpoint's
/// (`mlp.gate.weight`) and no adapter can attach to it: a router's argmax picks
/// which computation runs.
#[derive(Debug)]
pub struct Qwen3MoERouter {
    /// `[num_experts, hidden]`.
    pub weight: Array,
}
impl_module_params!(Qwen3MoERouter; weight);

impl Qwen3MoERouter {
    fn new(hidden_size: i32, num_experts: i32) -> Self {
        let bound = (hidden_size as f32).sqrt().recip();
        Self {
            weight: pmetal_runtime::compat::random::uniform_range(
                -bound,
                bound,
                &[num_experts, hidden_size],
                pmetal_runtime::compat::Dtype::Float32,
            ),
        }
    }

    /// `[tokens, hidden]` to each token's experts and their weights, both
    /// `[tokens, top_k]`, the weights in the dtype of `x`.
    ///
    /// The logits are computed in the model's dtype and the softmax in `f32`,
    /// as transformers' `Qwen3MoeTopKRouter` does; the weights are
    /// renormalized over the chosen experts when `norm_topk_prob` asks, and
    /// cast to the activation dtype last
    /// ([`native_moe::route_top_k`](pmetal_runtime::native_moe::route_top_k), the
    /// router the native engine runs too).
    fn route(&self, x: &Array, top_k: i32, norm_topk_prob: bool) -> (Array, Array) {
        let logits = x.matmul(&self.weight.as_dtype(x.dtype().as_i32()).t());
        pmetal_runtime::native_moe::route_top_k(&logits, top_k, norm_topk_prob, x.dtype().as_i32())
    }
}

/// MoE block of Qwen3-MoE: a softmax top-k router over stacked SwiGLU experts.
///
/// The experts are the three `switch_mlp` weights an MLX conversion stores
/// (`[experts, out, in]`); a released checkpoint's expert-per-key layout and
/// transformers' fused `gate_up_proj` are stacked at load
/// ([`crate::loader::experts::stack_experts`]). Nothing is derived from them:
/// the next forward runs the weights the block holds.
#[derive(Debug)]
pub struct Qwen3MoEBlock {
    /// Top-k experts per token.
    pub top_k: usize,
    /// Whether to normalize top-k probabilities.
    pub norm_topk_prob: bool,
    /// The router.
    pub gate: Qwen3MoERouter,
    /// The routed experts.
    pub switch_mlp: SwitchMlp,
}
impl_module_params!(Qwen3MoEBlock; gate, switch_mlp);

impl Qwen3MoEBlock {
    /// Create a new MoE block.
    pub fn new(config: &Qwen3MoEConfig) -> Result<Self, Exception> {
        let num_experts = config.num_experts;
        if config.num_experts_per_tok < 1 || config.num_experts_per_tok > num_experts {
            return Err(Exception::custom(format!(
                "qwen3_moe config: num_experts_per_tok {} is not within the {num_experts} experts",
                config.num_experts_per_tok
            )));
        }
        Ok(Self {
            top_k: config.num_experts_per_tok as usize,
            norm_topk_prob: config.norm_topk_prob,
            gate: Qwen3MoERouter::new(config.hidden_size, num_experts),
            switch_mlp: SwitchMlp::new(
                num_experts,
                config.hidden_size,
                config.get_moe_intermediate_size(),
                config.num_experts_per_tok as usize,
            ),
        })
    }

    /// Every expert over every token, each weighted by the router's weight for
    /// it where the router picked it and zero elsewhere: the routed sum written
    /// out densely, from primitives.
    #[cfg(test)]
    fn forward_reference(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape().to_vec();
        let hidden = shape[shape.len() - 1];
        let flat = x.reshape(&[-1, hidden]);
        let (indices, weights) = self
            .gate
            .route(&flat, self.top_k as i32, self.norm_topk_prob);
        let mut routed =
            pmetal_runtime::compat::ops::zeros_dtype(&[flat.dim(0), hidden], flat.dtype());
        let expert = |linear: &pmetal_runtime::nn::moe::Linear, e: i32| {
            pmetal_runtime::compat::ops::slice_axis(&linear.weight, 0, e, e + 1).squeeze(0)
        };
        for e in 0..self.switch_mlp.num_experts() {
            let picked = indices
                .equal(&Array::from_int(e))
                .as_dtype(weights.dtype().as_i32());
            let weight = picked.multiply(&weights).sum_axis(-1, true);
            let activated = nn::silu(&flat.matmul(&expert(&self.switch_mlp.gate_proj, e).t()))
                .multiply(&flat.matmul(&expert(&self.switch_mlp.up_proj, e).t()));
            let out = activated.matmul(&expert(&self.switch_mlp.down_proj, e).t());
            routed = routed.add(&out.multiply(&weight));
        }
        Ok(routed.reshape(&shape))
    }

    /// Forward pass: `[.., hidden]` in, the same shape out.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape().to_vec();
        let hidden = shape[shape.len() - 1];
        let flat = x.reshape(&[-1, hidden]);
        let (indices, weights) = self
            .gate
            .route(&flat, self.top_k as i32, self.norm_topk_prob);
        let routed = self.switch_mlp.forward(&flat, &indices, &weights)?;
        Ok(routed.reshape(&shape))
    }
}

/// Feed-forward for a decoder layer - either dense MLP or MoE.
///
/// C8: Implements `ModuleParameters` so that expert weights are visible to the
/// optimizer and parameter loading. The derive macro cannot handle enums, so
/// we implement the trait manually, delegating to the inner variant.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per layer, never moved; boxing would add an indirection to every forward"
)]
pub enum Qwen3MoEFeedForward {
    /// Dense MLP.
    Dense(Qwen3MoEDenseMLP),
    /// Mixture of Experts.
    MoE(Qwen3MoEBlock),
}

impl VisitLinears for Qwen3MoEFeedForward {
    fn visit_linears_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut nn::Linear)) {
        match self {
            Self::Dense(mlp) => mlp.visit_linears_mut(prefix, f),
            Self::MoE(moe) => moe.visit_linears_mut(prefix, f),
        }
    }

    fn visit_experts_mut(
        &mut self,
        prefix: &str,
        top_k: usize,
        f: &mut dyn FnMut(&str, usize, &mut pmetal_runtime::nn::moe::Linear),
    ) {
        match self {
            Self::Dense(mlp) => mlp.visit_experts_mut(prefix, top_k, f),
            Self::MoE(moe) => moe.visit_experts_mut(prefix, top_k, f),
        }
    }
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
}

/// Qwen3-MoE decoder layer.
#[derive(Debug)]
pub struct Qwen3MoEDecoderLayer {
    /// Configuration.
    pub config: Qwen3MoEConfig,
    /// Self-attention.
    pub self_attn: Qwen3MoEAttention,
    /// Input layer norm.
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: nn::RmsNorm,
    /// Feed-forward (MLP or MoE).
    pub mlp: Qwen3MoEFeedForward,
}
impl_module_params!(Qwen3MoEDecoderLayer; self_attn, input_layernorm, post_attention_layernorm, mlp);

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
        let mlp = if config.use_moe_at(layer_idx) {
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
            mlp,
        })
    }

    /// Forward pass through decoder layer.
    ///
    /// Delegates to `crate::decoder_layer::std_pre_norm_forward` — the
    /// standard pre-norm skeleton, with `self_attn: Qwen3MoEAttention` and
    /// `mlp: Qwen3MoEFeedForward` plugging into the `AttentionModule` /
    /// `MlpModule` traits.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        std_pre_norm_forward(
            &mut self.input_layernorm,
            &mut self.self_attn,
            &mut self.post_attention_layernorm,
            &mut self.mlp,
            x,
            mask,
            cache,
            positions,
        )
    }
}

impl AttentionModule for Qwen3MoEAttention {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        Qwen3MoEAttention::forward(self, x, mask, cache, positions)
    }
}

impl MlpModule for Qwen3MoEFeedForward {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        Qwen3MoEFeedForward::forward(self, x)
    }
}

impl DecoderLayer for Qwen3MoEDecoderLayer {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        Qwen3MoEDecoderLayer::forward(self, x, mask, cache, positions)
    }
}

/// Qwen3-MoE base model (without LM head).
#[derive(Debug)]
pub struct Qwen3MoEModel {
    /// Configuration.
    pub config: Qwen3MoEConfig,
    /// Token embeddings.
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    pub layers: Vec<Qwen3MoEDecoderLayer>,
    /// Final layer norm.
    pub norm: nn::RmsNorm,
    /// Recompute each layer's activations during the backward pass instead of
    /// holding them. Training only; see `crate::checkpointing`.
    pub grad_checkpoint: bool,
}
impl_module_params!(Qwen3MoEModel; embed_tokens, layers, norm);

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
            grad_checkpoint: false,
        })
    }

    /// Forward pass through the model.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_with_capture(input_ids, mask, cache, None, None)
    }

    /// Forward pass with one rotary position per token, `[seq_len]`.
    ///
    /// Packed training concatenates several sequences into one row, so the
    /// positions have to restart at each boundary instead of running through
    /// it. The block-diagonal mask keeps the content apart; this keeps the
    /// positions apart.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_capture(input_ids, mask, None, positions, None)
    }

    /// Forward pass with optional hidden-state capture for DFlash
    /// speculative decoding.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        positions: Option<&Array>,
        mut capture: Option<&mut pmetal_cache::speculative::SpecCapture>,
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(input_ids);

        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    h = layer.forward(&h, mask, Some((cache, layer_idx)), positions)?;
                    if let Some(buf) = capture.as_deref_mut()
                        && buf.wants_hidden_for(layer_idx)
                    {
                        buf.record_hidden(layer_idx, h.clone());
                    }
                }
            }
            None => {
                // Hoisted: the loop below borrows `self.layers` mutably.
                let grad_checkpoint = self.grad_checkpoint;
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    h = if grad_checkpoint {
                        checkpointed_layer(
                            layer,
                            &h,
                            mask,
                            positions,
                            |layer, h, mask, positions| layer.forward(h, mask, None, positions),
                        )?
                    } else {
                        layer.forward(&h, mask, None, positions)?
                    };
                    if let Some(buf) = capture.as_deref_mut()
                        && buf.wants_hidden_for(layer_idx)
                    {
                        buf.record_hidden(layer_idx, h.clone());
                    }
                }
            }
        }

        Ok(self.norm.forward(&h))
    }
}

/// Full Qwen3-MoE model with LM head.
///
/// C7: When `tie_word_embeddings` is true, `lm_head` is `None` and the
/// forward pass uses the embedding weight directly for the language model head.
#[derive(Debug)]
pub struct Qwen3MoE {
    /// Configuration.
    pub config: Qwen3MoEConfig,
    /// Base model.
    pub model: Qwen3MoEModel,
    /// LM head (None when tied to embedding weights).
    pub lm_head: Option<nn::Linear>,
}
impl_module_params!(Qwen3MoE; model, lm_head);

impl Qwen3MoE {
    /// Whether RoPE is a scalar base and position scale, which the fused
    /// batched decode path carries; YaRN is not.
    pub fn has_scalar_rope(&self) -> bool {
        self.model
            .layers
            .first()
            .is_some_and(|layer| layer.self_attn.rotary.scalar().is_some())
    }

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
        self.project_logits(&hidden_states)
    }

    /// [`forward`](Self::forward) with a cache, returning the last position's
    /// logits, `[B, 1, vocab]`: the LM head runs on one hidden state per
    /// sequence instead of every prompt token's, for a caller that samples from
    /// the last position alone (a prefill, a decode step).
    pub fn forward_with_cache_last(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids, mask, cache)?;
        self.project_logits(&last_token_hidden(&hidden_states))
    }

    /// Forward pass with one rotary position per token; see
    /// [`Qwen3MoEModel::forward_with_positions`].
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self
            .model
            .forward_with_positions(input_ids, mask, positions)?;
        self.project_logits(&hidden_states)
    }

    fn project_logits(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        if let Some(ref mut head) = self.lm_head {
            Ok(head.forward(hidden_states))
        } else {
            // Tied embeddings: use embed_tokens weight as linear projection
            // embed_tokens.weight is [vocab, hidden], so logits = hidden @ weight.T
            let embed_weight = self.model.embed_tokens.weight.value.as_ref();
            Ok(hidden_states.matmul(&embed_weight.t()))
        }
    }

    /// Forward pass that records hidden states into a DFlash capture
    /// buffer at every tapped layer index.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        capture: &mut pmetal_cache::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        let hidden_states =
            self.model
                .forward_with_capture(input_ids, mask, cache, None, Some(capture))?;
        if let Some(ref mut head) = self.lm_head {
            Ok(head.forward(&hidden_states))
        } else {
            let embed_weight = self.model.embed_tokens.weight.value.as_ref();
            Ok(hidden_states.matmul(&embed_weight.t()))
        }
    }

    /// Fused batched decode forward.
    ///
    /// Mirrors [`crate::architectures::llama::LlamaForCausalLM::forward_batched_impl`]
    /// but routes each layer's FFN through the existing
    /// [`Qwen3MoEFeedForward`] (dense or MoE). The MoE block's token-level
    /// router already handles the `[N_active, 1, H]` tensor natively —
    /// `forward_stacked` flattens to `[N_active, hidden]` and back.
    pub fn forward_batched_impl(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut pmetal_cache::kv_cache::FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        use crate::common::{BatchedGqaAttnCfg, batched_prenorm_layer};

        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let attn_cfg = BatchedGqaAttnCfg::for_rotary(
            cfg.num_attention_heads,
            cfg.num_kv_heads(),
            head_dim,
            &self.model.layers[0].self_attn.rotary,
        )?;
        let mut hidden =
            pmetal_runtime::compat::Module::forward(&mut self.model.embed_tokens, input_ids)?;
        for (layer_idx, layer) in self.model.layers.iter_mut().enumerate() {
            hidden = batched_prenorm_layer(
                &hidden,
                &mut layer.input_layernorm,
                &mut layer.self_attn.q_proj,
                &mut layer.self_attn.k_proj,
                &mut layer.self_attn.v_proj,
                &mut layer.self_attn.o_proj,
                Some(&mut layer.self_attn.q_norm),
                Some(&mut layer.self_attn.k_norm),
                &mut layer.post_attention_layernorm,
                &mut layer.mlp,
                &attn_cfg,
                cache,
                active_indices,
                layer_idx,
            )?;
        }
        let hidden = pmetal_runtime::compat::Module::forward(&mut self.model.norm, &hidden)?;
        if let Some(ref mut head) = self.lm_head {
            pmetal_runtime::compat::Module::forward(head, &hidden)
        } else {
            let embed_weight = self.model.embed_tokens.weight.value.as_ref();
            Ok(hidden.matmul(&embed_weight.t()))
        }
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
        // With sparse_step=2, only odd layers (0-indexed layer 1, 3, 5...) should be MoE
        let mut config = Qwen3MoEConfig {
            decoder_sparse_step: 2,
            ..Default::default()
        };
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
        let mut mlp = Qwen3MoEFeedForward::Dense(
            Qwen3MoEDenseMLP::new(config.hidden_size, config.intermediate_size).unwrap(),
        );
        let x = pmetal_runtime::compat::random::uniform_range(
            -1.0,
            1.0,
            &[1, 4, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, config.hidden_size]);
    }

    #[test]
    #[serial]
    fn test_feed_forward_moe_dispatch() {
        let config = tiny_moe_config();
        let mut mlp = Qwen3MoEFeedForward::MoE(Qwen3MoEBlock::new(&config).unwrap());
        let x = pmetal_runtime::compat::random::uniform_range(
            -1.0,
            1.0,
            &[1, 4, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let out = mlp.forward(&x).unwrap();
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
            !params.is_empty(),
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
            !params.is_empty(),
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
        let x = pmetal_runtime::compat::random::uniform_range(
            -1.0,
            1.0,
            &[2, 5, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let out = block.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 5, config.hidden_size]);
    }

    fn max_diff(a: &Array, b: &Array) -> f32 {
        let diff = a.subtract(b).abs().max(None);
        diff.eval();
        diff.item::<f32>()
    }

    /// The routed sum through the grouped gather matmuls against every expert
    /// over every token written out densely. 80 tokens x 2 experts is past the
    /// point where the dispatch groups the (token, expert) pairs by expert; 5
    /// tokens run the router-order path.
    #[test]
    #[serial]
    fn test_moe_block_matches_dense_reference() {
        pmetal_runtime::compat::random::seed(11);
        let config = tiny_moe_config();
        let mut block = Qwen3MoEBlock::new(&config).unwrap();
        for tokens in [5, 80] {
            let x = pmetal_runtime::compat::random::uniform_range(
                -1.0,
                1.0,
                &[2, tokens, config.hidden_size],
                pmetal_runtime::compat::Dtype::Float32,
            );
            let reference = block.forward_reference(&x).unwrap();
            let fast = block.forward(&x).unwrap();
            pmetal_runtime::check_unobserved_error().expect("runtime error");
            assert!(
                max_diff(&fast, &reference) < 1e-4,
                "qwen3-moe block drifted from the dense reference at {tokens} tokens"
            );
        }
    }

    /// The experts are the parameters, with nothing derived from them: a
    /// replaced weight is what the next forward runs.
    #[test]
    #[serial]
    fn test_moe_block_runs_the_weights_it_holds() {
        pmetal_runtime::compat::random::seed(12);
        let config = tiny_moe_config();
        let mut block = Qwen3MoEBlock::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::uniform_range(
            -1.0,
            1.0,
            &[1, 4, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        );

        let before = block.forward(&x).unwrap();
        block.switch_mlp.gate_proj.weight = Array::zeros_f32(&[
            config.num_experts,
            config.get_moe_intermediate_size(),
            config.hidden_size,
        ]);
        let reference = block.forward_reference(&x).unwrap();
        let after = block.forward(&x).unwrap();
        assert!(max_diff(&after, &reference) < 1e-4);
        assert!(
            max_diff(&after, &before) > 1e-3,
            "zeroing every gate projection changed nothing"
        );
    }

    /// Per-expert adapters, by definition: the gather matmuls that add
    /// `s·(x·A_eᵀ)·B_eᵀ` for the expert each row was routed to compute what the
    /// dense loop computes over experts whose weights are `W_e + s·B_e·A_e`,
    /// in both dispatch orders, and folding the adapters in (`merge_lora`)
    /// gives the same output. The block starts out unchanged (`B = 0`).
    #[test]
    #[serial]
    fn test_expert_adapters_match_the_dense_reference_and_merge() {
        use pmetal_runtime::compat::{Dtype, random};
        random::seed(15);
        let config = tiny_moe_config();
        let mut block = Qwen3MoEBlock::new(&config).unwrap();
        let draw = |tokens| {
            random::uniform_range(-1.0, 1.0, &[2, tokens, config.hidden_size], Dtype::Float32)
        };
        let (short, long) = (draw(5), draw(80));
        let plain = block.forward(&short).unwrap();

        for projection in [
            &mut block.switch_mlp.gate_proj,
            &mut block.switch_mlp.up_proj,
            &mut block.switch_mlp.down_proj,
        ] {
            projection.attach_lora(2, 8.0, false).unwrap();
        }
        assert!(
            max_diff(&block.forward(&short).unwrap(), &plain) < 1e-6,
            "B = 0 changed the block"
        );

        // Give every adapter a value, and build the dense twin from W + s·B·A.
        let mut twin = Qwen3MoEBlock::new(&config).unwrap();
        twin.gate.weight = block.gate.weight.clone();
        for (live, dense) in [
            (
                &mut block.switch_mlp.gate_proj,
                &mut twin.switch_mlp.gate_proj,
            ),
            (&mut block.switch_mlp.up_proj, &mut twin.switch_mlp.up_proj),
            (
                &mut block.switch_mlp.down_proj,
                &mut twin.switch_mlp.down_proj,
            ),
        ] {
            let adapter = live.adapter.as_mut().unwrap();
            adapter.b = random::uniform_range(-0.3, 0.3, adapter.b.shape(), Dtype::Float32);
            dense.weight = live.weight.add(&adapter.delta());
        }
        for (tokens, x) in [(5, &short), (80, &long)] {
            let adapted = block.forward(x).unwrap();
            let reference = twin.forward_reference(x).unwrap();
            pmetal_runtime::check_unobserved_error().expect("runtime error");
            assert!(
                max_diff(&adapted, &reference) < 2e-4,
                "expert adapters differ from the dense reference at {tokens} tokens"
            );
            assert!(
                max_diff(&adapted, &block_without_adapters(&block, x)) > 1e-3,
                "the adapters changed nothing, so this test proves nothing"
            );
        }

        // Merge: the weights now carry the update and the output is the same.
        let before = block.forward(&long).unwrap();
        for projection in [
            &mut block.switch_mlp.gate_proj,
            &mut block.switch_mlp.up_proj,
            &mut block.switch_mlp.down_proj,
        ] {
            projection.merge_lora().unwrap();
        }
        assert!(max_diff(&block.forward(&long).unwrap(), &before) < 2e-4);
    }

    /// `block`'s output with its adapters ignored.
    fn block_without_adapters(block: &Qwen3MoEBlock, x: &Array) -> Array {
        let mut copy = Qwen3MoEBlock::new(&tiny_moe_config()).unwrap();
        copy.gate.weight = block.gate.weight.clone();
        copy.switch_mlp.gate_proj.weight = block.switch_mlp.gate_proj.weight.clone();
        copy.switch_mlp.up_proj.weight = block.switch_mlp.up_proj.weight.clone();
        copy.switch_mlp.down_proj.weight = block.switch_mlp.down_proj.weight.clone();
        copy.forward(x).unwrap()
    }

    /// The routing weights are in the activation dtype, as the experts'
    /// outputs they scale are, so a bf16 model's MoE output stays bf16 and
    /// agrees with the f32 run to bf16 resolution.
    #[test]
    #[serial]
    fn test_moe_block_keeps_the_activation_dtype() {
        pmetal_runtime::compat::random::seed(13);
        let config = tiny_moe_config();
        let mut block = Qwen3MoEBlock::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::uniform_range(
            -1.0,
            1.0,
            &[1, 6, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let bf16 = pmetal_runtime::compat::Dtype::Bfloat16.as_i32();
        for proj in [
            &mut block.switch_mlp.gate_proj,
            &mut block.switch_mlp.up_proj,
            &mut block.switch_mlp.down_proj,
        ] {
            proj.weight = proj.weight.as_dtype(bf16);
        }
        block.gate.weight = block.gate.weight.as_dtype(bf16);
        let out = block.forward(&x.as_dtype(bf16)).unwrap();
        pmetal_runtime::check_unobserved_error().expect("runtime error");
        assert_eq!(out.dtype().as_i32(), bf16);
    }

    /// Training reaches the router and every expert projection, with the
    /// gradients of the dense reference. A gather that dropped the experts'
    /// gradient would train a model whose routed layers never moved.
    #[test]
    #[serial]
    fn test_moe_block_gradients_match_the_dense_reference() {
        pmetal_runtime::compat::random::seed(14);
        let config = tiny_moe_config();
        let mut block = Qwen3MoEBlock::new(&config).unwrap();
        // 40 tokens x 2 experts = 80 pairs: the grouped dispatch.
        let x = pmetal_runtime::compat::random::normal(
            &[1, 40, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let inputs = [
            x,
            block.gate.weight.clone(),
            block.switch_mlp.gate_proj.weight.clone(),
            block.switch_mlp.up_proj.weight.clone(),
            block.switch_mlp.down_proj.weight.clone(),
        ];
        let loss = |block: &mut Qwen3MoEBlock, arrays: &[Array], dense: bool| {
            block.gate.weight = arrays[1].clone();
            block.switch_mlp.gate_proj.weight = arrays[2].clone();
            block.switch_mlp.up_proj.weight = arrays[3].clone();
            block.switch_mlp.down_proj.weight = arrays[4].clone();
            let out = if dense {
                block.forward_reference(&arrays[0]).unwrap()
            } else {
                block.forward(&arrays[0]).unwrap()
            };
            out.square().sum(None)
        };
        let (fast_loss, fast) = pmetal_runtime::compat::nn::value_and_grad_explicit(
            |arrays: &[Array]| loss(&mut block, arrays, false),
            &inputs,
            &[],
        )
        .unwrap();
        let mut reference_block = Qwen3MoEBlock::new(&config).unwrap();
        let (ref_loss, reference) = pmetal_runtime::compat::nn::value_and_grad_explicit(
            |arrays: &[Array]| loss(&mut reference_block, arrays, true),
            &inputs,
            &[],
        )
        .unwrap();
        pmetal_runtime::check_unobserved_error().expect("runtime error");

        let host = |a: &Array| -> Vec<f32> {
            let a = a.as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
            a.eval();
            a.as_slice::<f32>().to_vec()
        };
        let loss_diff = (fast_loss.item::<f32>() - ref_loss.item::<f32>()).abs()
            / ref_loss.item::<f32>().abs().max(1.0);
        assert!(loss_diff < 1e-4, "loss differs by {loss_diff:e}");
        for (name, (g, r)) in ["input", "router", "gate_proj", "up_proj", "down_proj"]
            .iter()
            .zip(fast.iter().zip(&reference))
        {
            let (g, r) = (host(g), host(r));
            assert!(
                g.iter().all(|v| v.is_finite()),
                "the {name} gradient is not finite"
            );
            assert!(g.iter().any(|v| *v != 0.0), "the {name} gradient is zero");
            let scale = r.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            let diff = g
                .iter()
                .zip(&r)
                .fold(0.0f32, |m, (p, q)| m.max((p - q).abs()))
                / scale;
            assert!(diff < 1e-3, "the {name} gradient differs by {diff:e}");
        }
    }

    #[test]
    fn test_moe_block_rejects_an_impossible_top_k() {
        let mut config = tiny_moe_config();
        config.num_experts_per_tok = config.num_experts + 1;
        assert!(Qwen3MoEBlock::new(&config).is_err());
        config.num_experts_per_tok = 0;
        assert!(Qwen3MoEBlock::new(&config).is_err());
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
    fn test_decoder_layer_moe_vs_dense() {
        let config = tiny_moe_config();

        // Layer 0 with sparse_step=1 should be MoE
        let layer_moe = Qwen3MoEDecoderLayer::new(config.clone(), 0).unwrap();
        assert!(
            matches!(layer_moe.mlp, Qwen3MoEFeedForward::MoE(_)),
            "Layer 0 should be MoE with sparse_step=1"
        );

        // Force layer 0 to be dense via mlp_only_layers
        let mut config_dense = config.clone();
        config_dense.mlp_only_layers = vec![0];
        let layer_dense = Qwen3MoEDecoderLayer::new(config_dense, 0).unwrap();
        assert!(
            matches!(layer_dense.mlp, Qwen3MoEFeedForward::Dense(_)),
            "Layer 0 should be Dense when in mlp_only_layers"
        );
    }

    #[test]
    #[serial]
    #[ignore = "bridge ModuleParameters default freeze/unfreeze are no-ops — impl_module_params! does not generate them"]
    fn test_freeze_unfreeze_delegation() {
        let config = tiny_moe_config();
        let mut mlp = Qwen3MoEFeedForward::MoE(Qwen3MoEBlock::new(&config).unwrap());

        // Freeze all parameters
        mlp.freeze_parameters(true);
        assert_eq!(mlp.all_frozen(), Some(true));

        // Unfreeze all parameters
        mlp.unfreeze_parameters(true);
        assert_eq!(mlp.all_frozen(), Some(false));
    }
}
