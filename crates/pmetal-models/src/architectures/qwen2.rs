//! Qwen2 model architecture.
//!
//! Supports Qwen2 0.5B, 1.5B, 7B, 14B, 32B, and 72B variants.
//! Qwen2 is architecturally similar to Llama but with key differences:
//! - Fixed head_dim = 128 (always)
//! - Higher RoPE theta (1,000,000)
//! - No RoPE scaling support
//! - Optional sliding window attention (usually disabled)

use crate::decoder_layer::{AttentionModule, DecoderLayer, MlpModule, std_pre_norm_forward};
use crate::traits::ModelConfig;
use pmetal_bridge::compat::{Array, Dtype, Exception, Module, ModuleParameters, nn, ops, random};
use pmetal_bridge::impl_module_params;
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopeScaling, apply_rope},
};
use pmetal_mlx::kv_cache::KVCache;
use serde::{Deserialize, Serialize};

/// Qwen2 model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen2Config {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Intermediate size (MLP).
    pub intermediate_size: i32,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    /// Head dimension (fixed at 128 for Qwen2).
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency (much higher for Qwen2).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Hidden activation function.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Sliding window attention size (None = disabled).
    #[serde(default)]
    pub sliding_window: Option<i32>,
    /// Use sliding window attention.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Attention dropout (unused in inference).
    #[serde(default)]
    pub attention_dropout: f32,
    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<std::collections::HashMap<String, serde_json::Value>>,
}

fn default_model_type() -> String {
    "qwen2".to_string()
}
fn default_vocab_size() -> i32 {
    152064
}
fn default_head_dim() -> i32 {
    128 // Always 128 for Qwen2
}
fn default_max_position_embeddings() -> i32 {
    131072 // 128K context
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0 // Much higher than Llama's 10000.0
}
fn default_hidden_act() -> String {
    "silu".to_string()
}

impl crate::traits::ModelConfig for Qwen2Config {
    fn model_type(&self) -> &str {
        &self.model_type
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
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
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

impl Qwen2Config {
    /// Get the head dimension (always 128 for Qwen2).
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim
    }

    /// Get GQA group count.
    pub fn num_groups(&self) -> i32 {
        self.num_attention_heads / self.num_kv_heads()
    }
}

impl Default for Qwen2Config {
    fn default() -> Self {
        // Qwen2-1.5B defaults
        Self {
            model_type: "qwen2".to_string(),
            vocab_size: 152064,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: Some(2),
            head_dim: 128,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
            sliding_window: None,
            use_sliding_window: false,
            attention_dropout: 0.0,
            rope_scaling: None,
        }
    }
}

/// Qwen2 1.5B configuration.
impl Qwen2Config {
    /// Create Qwen2-0.5B configuration.
    pub fn qwen2_0_5b() -> Self {
        Self {
            hidden_size: 896,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: Some(2),
            ..Default::default()
        }
    }

    /// Create Qwen2-1.5B configuration.
    pub fn qwen2_1_5b() -> Self {
        Self::default()
    }

    /// Create Qwen2-7B configuration.
    pub fn qwen2_7b() -> Self {
        Self {
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: Some(4),
            ..Default::default()
        }
    }

    /// Create Qwen2-14B configuration.
    pub fn qwen2_14b() -> Self {
        Self {
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: Some(8),
            ..Default::default()
        }
    }

    /// Create Qwen2-72B configuration.
    pub fn qwen2_72b() -> Self {
        Self {
            hidden_size: 8192,
            intermediate_size: 29568,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: Some(8),
            rms_norm_eps: 1e-5, // Slightly different for 72B
            ..Default::default()
        }
    }
}

/// Qwen2 attention layer.
#[derive(Debug)]
pub struct Qwen2Attention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension (always 128).
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RoPE position scale (from rope_scaling config).
    pub rope_scale: f32,
    /// Effective RoPE base after scaling.
    pub effective_base: f32,
    /// Whether to use sliding window attention.
    pub use_sliding_window: bool,
    /// Sliding window size.
    pub sliding_window: Option<i32>,

    /// Query projection.
    pub q_proj: nn::Linear,
    /// Key projection.
    pub k_proj: nn::Linear,
    /// Value projection.
    pub v_proj: nn::Linear,
    /// Output projection.
    pub o_proj: nn::Linear,
    /// RoPE layer.
    pub rope: nn::Rope,
}
impl_module_params!(Qwen2Attention; q_proj, k_proj, v_proj, o_proj, rope);

impl Qwen2Attention {
    /// Create a new attention layer.
    pub fn new(config: &Qwen2Config) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = (head_dim as f32).sqrt().recip();
        let rope_theta = config.rope_theta;

        // Parse rope_scaling from config
        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(RopeScaling::from_config_map)
            .unwrap_or(RopeScaling::None);
        let rope_scale = rope_scaling.scale();
        let effective_base = rope_scaling.effective_base(rope_theta, head_dim);

        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(true) // Qwen2 uses bias in attention projections
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(true)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(true)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, config.hidden_size)
            .bias(false) // Output projection has no bias
            .build()?;

        // Initialize RoPE with scaled parameters
        let rope = nn::RopeBuilder::new(head_dim)
            .base(effective_base)
            .scale(rope_scale)
            .traditional(false)
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            rope_theta,
            rope_scale,
            effective_base,
            use_sliding_window: config.use_sliding_window,
            sliding_window: config.sliding_window,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    /// Forward pass through attention using fused kernels.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None)
    }

    /// Forward pass with optional KV cache.
    ///
    /// Uses MLX's optimized `scaled_dot_product_attention` with:
    /// - Fixed head_dim=128 (Qwen2 standard)
    /// - Native GQA support
    /// - Optional sliding window attention
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional (KVCache, layer_idx) tuple for cached generation
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_size]
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let mut cache = cache;

        // Project to Q, K, V
        let queries = Module::forward(&mut self.q_proj, x)?;
        let keys = Module::forward(&mut self.k_proj, x)?;
        let values = Module::forward(&mut self.v_proj, x)?;

        // Reshape for multi-head attention: [B, L, heads, head_dim] -> [B, heads, L, head_dim]
        let queries = queries
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        // Get RoPE offset and apply RoPE
        let (queries, keys, values) = if let Some((cache_ref, _layer_idx)) = cache.as_ref() {
            let offset = cache_ref.rope_offset();
            let queries = apply_rope(
                &queries,
                self.head_dim,
                false,
                self.effective_base,
                self.rope_scale,
                offset,
            )?;
            let keys = apply_rope(
                &keys,
                self.head_dim,
                false,
                self.effective_base,
                self.rope_scale,
                offset,
            )?;
            (queries, keys, values)
        } else {
            let queries = Module::forward(&mut self.rope, &queries)?;
            let keys = Module::forward(&mut self.rope, &keys)?;
            (queries, keys, values)
        };

        // Determine mask type for fused attention
        // Gate on use_sliding_window flag before checking sliding_window size
        let mask_type = if mask.is_some() {
            AttentionMaskType::None // Custom mask provided
        } else if self.use_sliding_window {
            if let Some(window) = self.sliding_window {
                AttentionMaskType::SlidingWindow(window)
            } else {
                AttentionMaskType::Causal
            }
        } else {
            AttentionMaskType::Causal
        };

        // Use fused attention kernel - handles GQA natively
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(mask_type);

        if mask.is_none() {
            if let Some((cache_ref, layer_idx)) = cache.as_mut() {
                if let Some(output) = (*cache_ref).try_turboquant_attention(
                    *layer_idx,
                    &queries,
                    &keys,
                    &values,
                    &attn_config,
                )? {
                    let output = output
                        .transpose_axes(&[0, 2, 1, 3])
                        .reshape(&[batch, seq_len, -1]);
                    return Module::forward(&mut self.o_proj, &output);
                }
            }
        }

        // Handle KV cache update - keys/values are already in [B, heads, seq, head_dim] format
        // No transpose needed - cache uses attention format directly (SOTA performance matching mlx_lm)
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, &values)?
        } else {
            (keys, values)
        };

        // Fused SDPA
        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);

        // Output projection
        Module::forward(&mut self.o_proj, &output)
    }
}

/// Qwen2 MLP layer (SwiGLU).
#[derive(Debug)]
pub struct Qwen2MLP {
    /// Gate projection.
    pub gate_proj: nn::Linear,
    /// Up projection.
    pub up_proj: nn::Linear,
    /// Down projection.
    pub down_proj: nn::Linear,
}
impl_module_params!(Qwen2MLP; gate_proj, up_proj, down_proj);

impl Qwen2MLP {
    /// Create a new MLP layer.
    pub fn new(config: &Qwen2Config) -> Result<Self, Exception> {
        // Qwen2 MLP uses no bias
        let gate_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass (SwiGLU activation).
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let gate = nn::silu(&gate);
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up);
        Module::forward(&mut self.down_proj, &hidden)
    }
}

impl MlpModule for Qwen2MLP {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        Qwen2MLP::forward(self, x)
    }
}

/// Qwen2 transformer block.
#[derive(Debug)]
pub struct Qwen2DecoderLayer {
    /// Self-attention layer.
    pub self_attn: Qwen2Attention,
    /// MLP layer.
    pub mlp: Qwen2MLP,
    /// Input layer norm.
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: nn::RmsNorm,
}
impl_module_params!(Qwen2DecoderLayer; self_attn, mlp, input_layernorm, post_attention_layernorm);

impl Qwen2DecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: &Qwen2Config) -> Result<Self, Exception> {
        let self_attn = Qwen2Attention::new(config)?;

        let mlp = Qwen2MLP::new(config)?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None)
    }

    /// Forward pass with optional KV cache.
    ///
    /// Delegates to the shared pre-norm skeleton —
    /// see `crate::decoder_layer::std_pre_norm_forward`.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        std_pre_norm_forward(
            &mut self.input_layernorm,
            &mut self.self_attn,
            &mut self.post_attention_layernorm,
            &mut self.mlp,
            x,
            mask,
            cache,
        )
    }
}

impl AttentionModule for Qwen2Attention {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        Qwen2Attention::forward_with_cache(self, x, mask, cache)
    }
}

impl DecoderLayer for Qwen2DecoderLayer {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        Qwen2DecoderLayer::forward_with_cache(self, x, mask, cache)
    }
}

/// Qwen2 base model (without LM head).
#[derive(Debug)]
pub struct Qwen2Model {
    /// Configuration (not a parameter).
    pub config: Qwen2Config,
    /// Token embeddings.
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    pub layers: Vec<Qwen2DecoderLayer>,
    /// Final layer norm.
    pub norm: nn::RmsNorm,
}
impl_module_params!(Qwen2Model; embed_tokens, layers, norm);

impl Qwen2Model {
    /// Create a new Qwen2 model.
    pub fn new(config: Qwen2Config) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| Qwen2DecoderLayer::new(&config))
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

    /// Forward pass through the model.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None)
    }

    /// Forward pass with optional KV cache.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KV cache for efficient generation
    ///
    /// # Returns
    /// Hidden states of shape [batch, seq_len, hidden_size]
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_with_capture(input_ids, mask, cache, None)
    }

    /// Forward pass with optional hidden-state capture for DFlash
    /// speculative decoding. Matches [`forward_with_cache`] exactly when
    /// `capture` is `None`.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        mut capture: Option<&mut pmetal_mlx::speculative::SpecCapture>,
    ) -> Result<Array, Exception> {
        // Get embeddings
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Create causal mask if not provided and not using cache
        let mask_owned;
        let mask = if mask.is_none() && cache.is_none() {
            let seq_len = input_ids.dim(1);
            mask_owned = if self.config.use_sliding_window {
                let window = self.config.sliding_window.unwrap_or(4096);
                create_sliding_window_mask(seq_len, window)?
            } else {
                create_causal_mask(seq_len)?
            };
            Some(&mask_owned)
        } else {
            mask
        };

        // Pass through transformer layers
        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden_states =
                        layer.forward_with_cache(&hidden_states, mask, Some((cache, layer_idx)))?;
                    if let Some(buf) = capture.as_deref_mut()
                        && buf.wants_hidden_for(layer_idx)
                    {
                        buf.record_hidden(layer_idx, hidden_states.clone());
                    }
                }
            }
            None => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden_states = layer.forward(&hidden_states, mask)?;
                    if let Some(buf) = capture.as_deref_mut()
                        && buf.wants_hidden_for(layer_idx)
                    {
                        buf.record_hidden(layer_idx, hidden_states.clone());
                    }
                }
            }
        }

        // Final norm
        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Qwen2 model with language modeling head.
#[derive(Debug)]
pub struct Qwen2ForCausalLM {
    /// Base model.
    pub model: Qwen2Model,
    /// LM head (optional, may share weights with embeddings).
    pub lm_head: Option<nn::Linear>,
}
impl_module_params!(Qwen2ForCausalLM; model, lm_head);

impl Qwen2ForCausalLM {
    /// Create a new Qwen2 model with LM head.
    pub fn new(config: Qwen2Config) -> Result<Self, Exception> {
        let tie_weights = config.tie_word_embeddings;
        let model = Qwen2Model::new(config.clone())?;

        let lm_head = if !tie_weights {
            let head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                .bias(false)
                .build()?;
            Some(head)
        } else {
            None
        };

        Ok(Self { model, lm_head })
    }

    /// Forward pass producing logits.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None)
    }

    /// Forward pass with optional KV cache.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KV cache
    ///
    /// # Returns
    /// Logits of shape [batch, seq_len, vocab_size]
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;

        // Get logits from LM head or shared embeddings
        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden_states)
        } else {
            // Tie weights: use embedding weight transposed
            Ok(self.model.embed_tokens.as_linear(&hidden_states))
        }
    }

    /// Forward pass that records hidden states into a DFlash capture
    /// buffer for every requested layer index.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        capture: &mut pmetal_mlx::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        let hidden_states =
            self.model
                .forward_with_capture(input_ids, mask, cache, Some(capture))?;
        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden_states)
        } else {
            Ok(self.model.embed_tokens.as_linear(&hidden_states))
        }
    }

    /// Fused batched-decode forward (sliding-window must be disabled —
    /// dispatcher gates via `supports_fused_batched`).
    pub fn forward_batched_impl(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut pmetal_mlx::kv_cache::FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        use crate::common::{BatchedGqaAttnCfg, batched_prenorm_layer};

        let cfg = &self.model.config;
        let head_dim = cfg.get_head_dim();
        let attn_cfg = BatchedGqaAttnCfg::new(
            cfg.num_attention_heads,
            cfg.num_kv_heads(),
            head_dim,
            self.model.layers[0].self_attn.effective_base,
            self.model.layers[0].self_attn.rope_scale,
        );
        let mut hidden = Module::forward(&mut self.model.embed_tokens, input_ids)?;
        for (layer_idx, layer) in self.model.layers.iter_mut().enumerate() {
            hidden = batched_prenorm_layer(
                &hidden,
                &mut layer.input_layernorm,
                &mut layer.self_attn.q_proj,
                &mut layer.self_attn.k_proj,
                &mut layer.self_attn.v_proj,
                &mut layer.self_attn.o_proj,
                None,
                None,
                &mut layer.post_attention_layernorm,
                &mut layer.mlp,
                &attn_cfg,
                cache,
                active_indices,
                layer_idx,
            )?;
        }
        let hidden = Module::forward(&mut self.model.norm, &hidden)?;
        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden)
        } else {
            Ok(self.model.embed_tokens.as_linear(&hidden))
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &Qwen2Config {
        &self.model.config
    }
}

use super::utils::create_causal_mask;

/// Create a sliding window causal attention mask.
fn create_sliding_window_mask(seq_len: i32, window_size: i32) -> Result<Array, Exception> {
    use pmetal_bridge::compat::Dtype;
    // Create lower triangular mask
    let causal_mask = pmetal_bridge::compat::ops::tri(seq_len, seq_len, 0, Dtype::Float32);

    // Create window mask: positions within window_size distance
    let indices =
        pmetal_bridge::compat::ops::arange_from(0, seq_len).as_dtype(Dtype::Float32.as_i32());
    let row_indices = indices.reshape(&[seq_len, 1]);
    let col_indices = indices.reshape(&[1, seq_len]);
    let distance = row_indices.subtract(&col_indices);

    // Valid if distance >= 0 and distance < window_size
    let window_size_arr = Array::from_f32(window_size as f32);
    let zero = Array::from_f32(0.0);
    let in_window = pmetal_bridge::compat::ops::logical_and(
        &distance.greater_equal(&zero),
        &distance.less(&window_size_arr),
    );

    // Combine with causal mask
    let combined = causal_mask.multiply(&in_window.as_dtype(Dtype::Float32.as_i32()));

    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    Ok(pmetal_bridge::compat::ops::where_fn(
        &combined.equal(&zero),
        &neg_inf,
        &zero,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn small_config() -> Qwen2Config {
        Qwen2Config {
            vocab_size: 1000,
            hidden_size: 128, // Must be divisible by head_dim * heads
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 2, // 2 heads * 64 head_dim = 128
            num_key_value_heads: Some(1),
            head_dim: 64, // Smaller for testing
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_qwen2_config_defaults() {
        let config = Qwen2Config::default();
        assert_eq!(config.model_type, "qwen2");
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.vocab_size, 152064);
    }

    #[test]
    fn test_qwen2_presets() {
        let config_7b = Qwen2Config::qwen2_7b();
        assert_eq!(config_7b.hidden_size, 3584);
        assert_eq!(config_7b.num_attention_heads, 28);
        assert_eq!(config_7b.num_key_value_heads, Some(4));

        let config_72b = Qwen2Config::qwen2_72b();
        assert_eq!(config_72b.hidden_size, 8192);
        assert_eq!(config_72b.num_hidden_layers, 80);
    }

    #[test]
    #[serial]
    fn test_qwen2_attention() {
        let config = small_config();
        let mut attn = Qwen2Attention::new(&config).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 128],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 128]);
    }

    #[test]
    #[serial]
    fn test_qwen2_mlp() {
        let config = small_config();
        let mut mlp = Qwen2MLP::new(&config).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 128],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 4, 128]);
    }

    #[test]
    #[serial]
    fn test_qwen2_decoder_layer() {
        let config = small_config();
        let mut layer = Qwen2DecoderLayer::new(&config).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 128],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = layer.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 128]);
    }

    #[test]
    #[serial]
    fn test_qwen2_model() {
        let config = small_config();
        let mut model = Qwen2Model::new(config).unwrap();

        let input_ids = pmetal_bridge::compat::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let output = model.forward(&input_ids, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 128]);
    }

    #[test]
    #[serial]
    fn test_qwen2_causal_lm() {
        let config = small_config();
        let mut model = Qwen2ForCausalLM::new(config).unwrap();

        let input_ids = pmetal_bridge::compat::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]); // [batch, seq, vocab]
    }

    #[test]
    #[serial]
    fn test_sliding_window_mask() {
        let mask = create_sliding_window_mask(8, 4).unwrap();
        mask.eval().unwrap();
        assert_eq!(mask.shape(), &[8, 8]);
    }

    #[test]
    fn test_qwen2_gqa_groups() {
        let config = Qwen2Config::qwen2_7b();
        assert_eq!(config.num_groups(), 7); // 28 heads / 4 kv_heads = 7 groups
    }
}
