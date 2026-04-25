//! Llama model architecture.
//!
//! Supports Llama 2, 3, 3.1, 3.2, 3.3, and 4 variants.

use std::collections::HashMap;

use pmetal_bridge::compat::{Array, Exception, Module, ModuleParameters, nn, random};
use pmetal_bridge::impl_module_params;
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, differentiable_attention, fused_sdpa,
    get_training_context,
    rope::{RopeScaling, apply_rope, apply_rope_scaled},
};
use pmetal_mlx::kv_cache::KVCache;
use serde::{Deserialize, Serialize};

use crate::decoder_layer::{AttentionModule, DecoderLayer, MlpModule, std_pre_norm_forward};

/// Llama model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
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
    /// Head dimension (computed if not provided).
    #[serde(default)]
    pub head_dim: Option<i32>,
    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, RopeScalingValue>>,
    /// Hidden activation function.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_model_type() -> String {
    "llama".to_string()
}
fn default_max_position_embeddings() -> i32 {
    4096
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_hidden_act() -> String {
    "silu".to_string()
}

/// RoPE scaling configuration value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RopeScalingValue {
    /// Floating point value.
    Float(f32),
    /// String value.
    String(String),
}

impl LlamaConfig {
    /// Get the number of KV heads (defaults to num_attention_heads if not specified).
    pub fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the head dimension.
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // Llama 3.2 1B defaults
        Self {
            model_type: "llama".to_string(),
            vocab_size: 128256,
            hidden_size: 2048,
            intermediate_size: 8192,
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            head_dim: None,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            rope_scaling: None,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
        }
    }
}

/// Llama attention layer.
#[derive(Debug)]
pub struct LlamaAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// RoPE base frequency (for apply_rope with offset).
    pub rope_theta: f32,
    /// RoPE position scale (from rope_scaling config).
    pub rope_scale: f32,
    /// Effective RoPE base after scaling.
    pub effective_base: f32,
    /// Layer ID for training cache (set during model construction).
    pub layer_id: usize,

    /// Query projection.
    pub q_proj: nn::Linear,
    /// Key projection.
    pub k_proj: nn::Linear,
    /// Value projection.
    pub v_proj: nn::Linear,
    /// Output projection.
    pub o_proj: nn::Linear,
    /// RoPE layer (used for non-cached forward).
    pub rope: nn::Rope,
}
impl_module_params!(LlamaAttention; q_proj, k_proj, v_proj, o_proj, rope);

impl LlamaAttention {
    /// Create a new attention layer.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `layer_id` - Layer index (used for training cache)
    pub fn new(config: &LlamaConfig, layer_id: usize) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = (head_dim as f32).sqrt().recip();
        let rope_theta = config.rope_theta;

        // Parse rope_scaling from config
        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(|map| {
                // Convert HashMap<String, RopeScalingValue> to HashMap<String, serde_json::Value>
                let json_map: std::collections::HashMap<String, serde_json::Value> = map
                    .iter()
                    .map(|(k, v)| {
                        let json_v = match v {
                            RopeScalingValue::Float(f) => serde_json::Value::from(*f as f64),
                            RopeScalingValue::String(s) => serde_json::Value::from(s.clone()),
                        };
                        (k.clone(), json_v)
                    })
                    .collect();
                RopeScaling::from_config_map(&json_map)
            })
            .unwrap_or(RopeScaling::None);

        let rope_scale = rope_scaling.scale();
        let effective_base = rope_scaling.effective_base(rope_theta, head_dim);

        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, config.hidden_size)
            .bias(false)
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
            layer_id,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    /// Forward pass through attention using fused Metal kernels.
    ///
    /// Uses MLX's optimized `scaled_dot_product_attention` which provides:
    /// - Metal kernel optimization for single-token generation
    /// - Native GQA support (no manual KV head expansion overhead)
    /// - Reduced memory bandwidth
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask (additive, -inf for masked positions)
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_size]
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        // Non-cached forward - delegate to cached version with None
        self.forward_with_cache(x, mask, None)
    }

    /// Forward pass with optional KV cache for efficient generation.
    ///
    /// When `cache` is provided:
    /// - RoPE is applied with position offset from cache
    /// - K/V tensors are concatenated with cached values
    /// - Only new tokens need to be processed (O(1) per token instead of O(n))
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
            // Cached path: use apply_rope with offset and scaling
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
            // Non-cached path: use RoPE module (offset=0)
            let queries = Module::forward(&mut self.rope, &queries)?;
            let keys = Module::forward(&mut self.rope, &keys)?;
            (queries, keys, values)
        };

        // Use fused attention kernel - handles GQA natively (no KV head expansion needed)
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None // Custom mask provided
            } else {
                AttentionMaskType::Causal // Use built-in causal mask
            });

        // Check if training mode is enabled for efficient backward pass
        let is_training = get_training_context()
            .map(|ctx| ctx.lock().map(|c| c.is_training()).unwrap_or(false))
            .unwrap_or(false);

        if !is_training && mask.is_none() {
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

        // Use differentiable attention for training (O(n) backward pass via Metal FlashAttention)
        // or fused SDPA for inference
        let output = if is_training && mask.is_none() {
            // Training mode: use Metal FlashAttention with proper backward pass
            differentiable_attention(self.layer_id, &queries, &keys, &values, &attn_config)
                .map_err(|e| Exception::custom(e.to_string()))?
        } else {
            // Inference mode: use fused SDPA (faster for single token generation)
            fused_sdpa(&queries, &keys, &values, &attn_config, mask)?
        };

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);

        // Output projection
        Module::forward(&mut self.o_proj, &output)
    }
}

/// Llama MLP layer (SwiGLU).
#[derive(Debug)]
pub struct LlamaMLP {
    /// Gate projection.
    pub gate_proj: nn::Linear,
    /// Up projection.
    pub up_proj: nn::Linear,
    /// Down projection.
    pub down_proj: nn::Linear,
}
impl_module_params!(LlamaMLP; gate_proj, up_proj, down_proj);

impl LlamaMLP {
    /// Create a new MLP layer.
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
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

impl MlpModule for LlamaMLP {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        LlamaMLP::forward(self, x)
    }
}

/// Llama transformer block.
#[derive(Debug)]
pub struct LlamaDecoderLayer {
    /// Self-attention layer.
    pub self_attn: LlamaAttention,
    /// MLP layer.
    pub mlp: LlamaMLP,
    /// Input layer norm.
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: nn::RmsNorm,
}
impl_module_params!(LlamaDecoderLayer; self_attn, mlp, input_layernorm, post_attention_layernorm);

impl LlamaDecoderLayer {
    /// Create a new decoder layer.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `layer_id` - Layer index (used for training cache)
    pub fn new(config: &LlamaConfig, layer_id: usize) -> Result<Self, Exception> {
        let self_attn = LlamaAttention::new(config, layer_id)?;

        let mlp = LlamaMLP::new(config)?;

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
    /// Delegates to the shared pre-norm skeleton in
    /// [`crate::decoder_layer::std_pre_norm_forward`] — the 6-line
    /// `norm → attn → +residual → norm → mlp → +residual` pattern is
    /// defined once and reused across every dense-transformer arch
    /// that matches the standard shape.
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

impl AttentionModule for LlamaAttention {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        LlamaAttention::forward_with_cache(self, x, mask, cache)
    }
}

impl DecoderLayer for LlamaDecoderLayer {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        LlamaDecoderLayer::forward_with_cache(self, x, mask, cache)
    }
}

/// Llama base model (without LM head).
#[derive(Debug)]
pub struct LlamaModel {
    /// Configuration (not a parameter).
    pub config: LlamaConfig,
    /// Token embeddings.
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    pub layers: Vec<LlamaDecoderLayer>,
    /// Final layer norm.
    pub norm: nn::RmsNorm,
}
impl_module_params!(LlamaModel; embed_tokens, layers, norm);

impl LlamaModel {
    /// Create a new Llama model.
    pub fn new(config: LlamaConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|layer_id| LlamaDecoderLayer::new(&config, layer_id as usize))
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
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch, seq_len]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Hidden states of shape [batch, seq_len, hidden_size]
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
    /// speculative decoding. `capture = None` is identical to
    /// [`forward_with_cache`]; otherwise the post-layer hidden state is
    /// cloned into `capture` for every layer index the caller requested.
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
            mask_owned = create_causal_mask(seq_len)?;
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

/// Llama model with language modeling head.
#[derive(Debug)]
pub struct LlamaForCausalLM {
    /// Base model.
    pub model: LlamaModel,
    /// LM head (optional, may share weights with embeddings).
    pub lm_head: Option<nn::Linear>,
}
impl_module_params!(LlamaForCausalLM; model, lm_head);

impl LlamaForCausalLM {
    /// Create a new Llama model with LM head.
    pub fn new(config: LlamaConfig) -> Result<Self, Exception> {
        let tie_weights = config.tie_word_embeddings;
        let model = LlamaModel::new(config.clone())?;

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
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch, seq_len]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Logits of shape [batch, seq_len, vocab_size]
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None)
    }

    /// Forward pass with optional KV cache for efficient generation.
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

    /// Forward pass that records hidden states into a DFlash speculative
    /// capture buffer at every tapped layer index.
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

    /// Fused batched decode forward.
    ///
    /// Runs one `[N_active, 1]` forward across every active slot,
    /// reusing the shared [`FusedBatchKVCache`] for all layers. Shape
    /// contract matches [`crate::dispatcher::DynamicModel::forward_batched`].
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

    /// Create a KV cache for this model.
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length to cache
    ///
    /// # Returns
    /// A new KV cache configured for this model
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        use pmetal_mlx::kv_cache::KVCacheConfig;

        let config = &self.model.config;
        KVCache::new(KVCacheConfig::new(
            config.num_hidden_layers as usize,
            max_seq_len,
            config.num_kv_heads() as usize,
            config.get_head_dim() as usize,
        ))
    }

    /// Get configuration.
    pub fn config(&self) -> &LlamaConfig {
        &self.model.config
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl crate::traits::ModelConfig for LlamaConfig {
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
        self.num_kv_heads()
    }

    fn head_dim(&self) -> i32 {
        self.get_head_dim()
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

impl crate::traits::CausalLMModel for LlamaForCausalLM {
    type Config = LlamaConfig;

    fn new(config: Self::Config) -> Result<Self, Exception> {
        LlamaForCausalLM::new(config)
    }

    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        LlamaForCausalLM::forward(self, input_ids, mask)
    }

    fn config(&self) -> &Self::Config {
        LlamaForCausalLM::config(self)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), Exception> {
        crate::loader::load_llama_weights(self, weights)
            .map_err(|e: crate::loader::LoadError| Exception::custom(e.to_string()))
    }

    fn eval(&self) -> Result<(), Exception> {
        pmetal_bridge::compat::ModuleParametersExt::eval(self)
    }
}

/// Re-export the shared causal mask utility.
use super::utils::create_causal_mask;

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn small_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: None,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            ..Default::default()
        }
    }

    #[test]
    #[serial]
    fn test_llama_attention() {
        let config = small_config();
        let mut attn = LlamaAttention::new(&config, 0).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_llama_mlp() {
        let config = small_config();
        let mut mlp = LlamaMLP::new(&config).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_llama_decoder_layer() {
        let config = small_config();
        let mut layer = LlamaDecoderLayer::new(&config, 0).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = layer.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    /// Canary: `LlamaDecoderLayer` must work through the shared
    /// [`crate::decoder_layer::DecoderLayer`] trait — not just via its
    /// inherent `forward_with_cache`. Dispatching through `dyn DecoderLayer`
    /// proves the trait impl is live and the trait object is object-safe
    /// (which is what generation code relies on).
    #[test]
    #[serial]
    fn test_llama_decoder_layer_dispatches_via_trait() {
        use crate::decoder_layer::DecoderLayer;

        let config = small_config();
        let mut layer = LlamaDecoderLayer::new(&config, 0).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let via_trait = {
            let dyn_layer: &mut dyn DecoderLayer = &mut layer;
            dyn_layer.forward(&x, None).unwrap()
        };
        assert_eq!(via_trait.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_llama_model() {
        let config = small_config();
        let mut model = LlamaModel::new(config).unwrap();

        let input_ids = pmetal_bridge::compat::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let output = model.forward(&input_ids, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_llama_causal_lm() {
        let config = small_config();
        let mut model = LlamaForCausalLM::new(config).unwrap();

        let input_ids = pmetal_bridge::compat::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]); // [batch, seq, vocab]
    }
}
