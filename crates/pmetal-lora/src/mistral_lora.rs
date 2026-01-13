//! LoRA-enabled Mistral model architecture.
//!
//! Implements Mistral with LoRA adapters on attention and MLP projections for efficient fine-tuning.
//! Supports sliding window attention configurations.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::{ModuleParamMut, ModuleParamRef, ModuleParameters},
    nested::NestedValue,
    nn, Array,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{fused_sdpa, rope::apply_rope, AttentionMaskType, FusedAttentionConfig};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::mistral::MistralConfig;

use crate::{LoraError, LoraLinear, TrainableModel};

/// LoRA-enabled attention layer for Mistral.
///
/// Applies LoRA to q_proj, k_proj, v_proj, and o_proj.
/// Supports sliding window attention for efficient long-context handling.
#[derive(Debug)]
pub struct MistralLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads (for GQA).
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// Sliding window size (None for full attention).
    pub sliding_window: Option<i32>,

    /// Query projection with LoRA.
    pub q_proj: LoraLinear,
    /// Key projection with LoRA.
    pub k_proj: LoraLinear,
    /// Value projection with LoRA.
    pub v_proj: LoraLinear,
    /// Output projection with LoRA.
    pub o_proj: LoraLinear,
    /// RoPE layer.
    pub rope: nn::Rope,
}

impl MistralLoraAttention {
    /// Create a new LoRA attention layer.
    pub fn new(config: &MistralConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = (head_dim as f32).sqrt().recip();

        let rank = lora_config.r as i32;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        // Create LoRA linear layers for projections
        let q_proj = LoraLinear::new(
            config.hidden_size,
            n_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let k_proj = LoraLinear::new(
            config.hidden_size,
            n_kv_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let v_proj = LoraLinear::new(
            config.hidden_size,
            n_kv_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let o_proj = LoraLinear::new(
            n_heads * head_dim,
            config.hidden_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;

        // Initialize RoPE
        let rope = nn::RopeBuilder::new(head_dim)
            .base(config.rope_theta)
            .traditional(false)
            .build()
            .unwrap();

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            sliding_window: config.sliding_window,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    /// Forward pass through attention.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project to Q, K, V using LoRA layers
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape for multi-head attention: [B, L, heads, head_dim]
        let queries = queries
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B, heads, L, head_dim]
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let queries = mlx_rs::module::Module::forward(&mut self.rope, &queries)?;
        let keys = mlx_rs::module::Module::forward(&mut self.rope, &keys)?;

        // Expand KV heads for GQA if needed
        let keys = if self.n_kv_heads < self.n_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            expand_kv_heads(&keys, repeats)?
        } else {
            keys
        };
        let values = if self.n_kv_heads < self.n_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            expand_kv_heads(&values, repeats)?
        } else {
            values
        };

        // Scaled dot-product attention
        let scores = queries.matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?;
        let scores = scores.multiply(Array::from_f32(self.scale))?;

        // Apply mask if provided
        let scores = if let Some(m) = mask {
            scores.add(m)?
        } else {
            scores
        };

        // Softmax
        let weights = mlx_rs::ops::softmax_axis(&scores, -1, None)?;

        // Attention output
        let output = weights.matmul(&values)?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        // Output projection
        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    /// Forward pass with KV cache for efficient inference.
    ///
    /// This method is optimized for autoregressive generation with O(n)
    /// complexity per token instead of O(n²).
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional (KVCache, layer_idx) tuple for cached generation
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project to Q, K, V using LoRA layers
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape for multi-head attention: [B, L, heads, head_dim]
        let queries = queries
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B, heads, L, head_dim]
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Get RoPE offset and apply RoPE
        let (queries, keys, values) = if let Some((ref cache_ref, _layer_idx)) = cache {
            let offset = cache_ref.rope_offset();
            let queries = apply_rope(&queries, self.head_dim, false, self.rope.base, 1.0, offset)?;
            let keys = apply_rope(&keys, self.head_dim, false, self.rope.base, 1.0, offset)?;
            (queries, keys, values)
        } else {
            let queries = mlx_rs::module::Module::forward(&mut self.rope, &queries)?;
            let keys = mlx_rs::module::Module::forward(&mut self.rope, &keys)?;
            (queries, keys, values)
        };

        // Handle KV cache update - keys/values are already in [B, heads, seq, head_dim] format
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache
                .update_and_fetch(layer_idx, &keys, &values)
                .map_err(|e| LoraError::Mlx(e))?
        } else {
            (keys, values)
        };

        // Use fused attention kernel for inference (more efficient than standard attention)
        // Mistral supports sliding window attention if configured
        let mask_type = if let Some(window_size) = self.sliding_window {
            AttentionMaskType::SlidingWindow(window_size)
        } else {
            AttentionMaskType::Causal
        };

        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(mask_type);

        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)
            .map_err(|e| LoraError::Mlx(e))?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        // Output projection
        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

/// Expand KV heads for grouped query attention.
fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    let x = x.reshape(&[batch, n_kv_heads, 1, seq_len, head_dim])?;
    let x = mlx_rs::ops::broadcast_to(&x, &[batch, n_kv_heads, repeats, seq_len, head_dim])?;
    x.reshape(&[batch, n_kv_heads * repeats, seq_len, head_dim])
}

/// LoRA-enabled MLP layer for Mistral (SwiGLU).
#[derive(Debug)]
pub struct MistralLoraMLP {
    /// Gate projection with LoRA.
    pub gate_proj: LoraLinear,
    /// Up projection with LoRA.
    pub up_proj: LoraLinear,
    /// Down projection with LoRA.
    pub down_proj: LoraLinear,
}

impl MistralLoraMLP {
    /// Create a new LoRA MLP layer.
    pub fn new(config: &MistralConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let rank = lora_config.r as i32;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let gate_proj = LoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let up_proj = LoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let down_proj = LoraLinear::new(
            config.intermediate_size,
            config.hidden_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass (SwiGLU activation).
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden).map_err(LoraError::from)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

/// LoRA-enabled decoder layer for Mistral.
#[derive(Debug)]
pub struct MistralLoraDecoderLayer {
    /// Self-attention layer.
    pub self_attn: MistralLoraAttention,
    /// MLP layer.
    pub mlp: MistralLoraMLP,
    /// Input layer norm.
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl MistralLoraDecoderLayer {
    /// Create a new LoRA decoder layer.
    pub fn new(config: &MistralConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let self_attn = MistralLoraAttention::new(config, lora_config)?;
        let mlp = MistralLoraMLP::new(config, lora_config)?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        // Pre-norm + attention + residual
        let normed = mlx_rs::module::Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        let h = x.add(&attn_out)?;

        // Pre-norm + MLP + residual
        let normed = mlx_rs::module::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out)?)
    }

    /// Forward pass with KV cache for efficient inference.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional (KVCache, layer_idx) tuple for cached generation
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        // Pre-norm + attention + residual
        let normed = mlx_rs::module::Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let h = x.add(&attn_out)?;

        // Pre-norm + MLP + residual
        let normed = mlx_rs::module::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out)?)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.mlp.num_trainable_params()
    }
}

/// LoRA-enabled Mistral model (without LM head).
#[derive(Debug)]
pub struct MistralLoraModel {
    /// Configuration.
    pub config: MistralConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    pub layers: Vec<MistralLoraDecoderLayer>,
    /// Final layer norm.
    pub norm: nn::RmsNorm,
}

impl MistralLoraModel {
    /// Create a new LoRA Mistral model.
    pub fn new(config: MistralConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| MistralLoraDecoderLayer::new(&config, &lora_config))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let mut hidden_states = mlx_rs::module::Module::forward(&mut self.embed_tokens, input_ids)?;

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, mask)?;
        }

        Ok(mlx_rs::module::Module::forward(
            &mut self.norm,
            &hidden_states,
        )?)
    }

    /// Forward with explicit position IDs (for packed sequences).
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        // For now, use standard forward - position IDs not used directly
        // RoPE handles positions implicitly
        self.forward(input_ids, mask)
    }

    /// Forward pass with KV cache for efficient inference.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional mutable reference to KV cache
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        // Get embeddings
        let mut hidden_states = mlx_rs::module::Module::forward(&mut self.embed_tokens, input_ids)?;

        // Don't create explicit causal mask - fused SDPA handles it internally
        // with proper dtype handling. Only pass through user-provided masks.

        // Pass through transformer layers
        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden_states =
                        layer.forward_with_cache(&hidden_states, mask, Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    hidden_states = layer.forward(&hidden_states, mask)?;
                }
            }
        }

        // Final norm
        Ok(mlx_rs::module::Module::forward(
            &mut self.norm,
            &hidden_states,
        )?)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }
}

/// LoRA-enabled Mistral model with LM head.
#[derive(Debug)]
pub struct MistralLoraForCausalLM {
    /// Base model.
    pub model: MistralLoraModel,
    /// LM head (optional, may share weights with embeddings).
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl MistralLoraForCausalLM {
    /// Create a new LoRA Mistral model with LM head.
    pub fn new(config: MistralConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let tie_weights = config.tie_word_embeddings;
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let model = MistralLoraModel::new(config, lora_config)?;

        let lm_head = if !tie_weights {
            let head = nn::LinearBuilder::new(hidden_size, vocab_size)
                .bias(false)
                .build()
                .unwrap();
            Some(head)
        } else {
            None
        };

        Ok(Self {
            model,
            lm_head,
            checkpoint_config: None,
        })
    }

    /// Forward pass producing logits.
    pub fn forward_internal(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let hidden_states = self.model.forward(input_ids, mask)?;

        if let Some(ref mut lm_head) = self.lm_head {
            Ok(mlx_rs::module::Module::forward(lm_head, &hidden_states)?)
        } else {
            Ok(self.model.embed_tokens.as_linear(&hidden_states)?)
        }
    }

    /// Forward pass with KV cache for efficient inference.
    ///
    /// KV caching provides O(n) complexity per token instead of O(n²),
    /// enabling fast autoregressive generation.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional mutable reference to KV cache
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;

        // Get logits from LM head or shared embeddings
        if let Some(ref mut lm_head) = self.lm_head {
            Ok(mlx_rs::module::Module::forward(lm_head, &hidden_states)?)
        } else {
            // Tie weights: use embedding weight transposed
            Ok(self.model.embed_tokens.as_linear(&hidden_states)?)
        }
    }

    /// Create a KV cache for this model.
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length to cache
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let config = KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_kv_heads() as usize,
            self.model.config.get_head_dim() as usize,
        );
        KVCache::new(config)
    }

    /// Load base model weights from safetensors.
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: &std::path::Path,
    ) -> Result<(), LoraError> {
        use mlx_rs::error::Exception;
        use mlx_rs::module::Param;

        // Check for single file model
        let single_file = model_dir.join("model.safetensors");
        if single_file.exists() {
            let weights = Array::load_safetensors(&single_file)?;
            return self.load_base_weights(&weights);
        }

        // Load sharded model
        let index_path = model_dir.join("model.safetensors.index.json");
        if !index_path.exists() {
            return Err(LoraError::Mlx(Exception::custom(
                "No model.safetensors or model.safetensors.index.json found".to_string(),
            )));
        }

        let index_content = std::fs::read_to_string(&index_path)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        #[derive(serde::Deserialize)]
        struct WeightIndex {
            weight_map: std::collections::HashMap<String, String>,
        }

        let index: WeightIndex = serde_json::from_str(&index_content)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        // Get unique shard files
        let shard_files: std::collections::HashSet<&String> = index.weight_map.values().collect();

        // Load all shards
        let mut all_weights: HashMap<String, Array> = HashMap::new();
        for shard_file in shard_files {
            let shard_path = model_dir.join(shard_file);
            let weights = Array::load_safetensors(&shard_path)?;
            for (key, value) in weights {
                all_weights.insert(key.to_string(), value);
            }
        }

        self.load_base_weights(&all_weights)
    }

    /// Load base model weights from a HashMap of weight tensors.
    pub fn load_base_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        use mlx_rs::module::Param;

        // Load embed_tokens
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        // Load layers
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            // Attention weights (load into LoraLinear.weight)
            if let Some(w) = weights.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                layer.self_attn.q_proj.weight = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                layer.self_attn.k_proj.weight = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                layer.self_attn.v_proj.weight = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                layer.self_attn.o_proj.weight = w.clone();
            }

            // MLP weights
            if let Some(w) = weights.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                layer.mlp.gate_proj.weight = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                layer.mlp.up_proj.weight = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                layer.mlp.down_proj.weight = w.clone();
            }

            // Layer norms
            if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }
        }

        // Load final norm
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }

        // Load LM head (if not tied)
        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                lm_head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }

    /// Freeze all non-LoRA parameters.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        use mlx_rs::transforms::eval;

        // Evaluate all parameters
        let params: Vec<&Array> = self
            .parameters()
            .flatten()
            .into_iter()
            .map(|(_, arr)| arr)
            .collect();
        eval(params)?;
        Ok(())
    }

    /// Enable gradient checkpointing.
    pub fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        self.checkpoint_config = Some(CheckpointConfig {
            enabled: true,
            layers_per_block,
            eval_at_boundaries: true,
        });
    }

    /// Disable gradient checkpointing.
    pub fn disable_gradient_checkpointing(&mut self) {
        self.checkpoint_config = None;
    }

    /// Get configuration.
    pub fn config(&self) -> &MistralConfig {
        &self.model.config
    }
}

impl ModuleParameters for MistralLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        // Return only LoRA parameters as that's what we're training
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // Attention LoRA params
            let mut attn_params = HashMap::new();
            let mut q_params = HashMap::new();
            q_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.q_proj.lora_a),
            );
            q_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.q_proj.lora_b),
            );
            attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

            let mut k_params = HashMap::new();
            k_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.k_proj.lora_a),
            );
            k_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.k_proj.lora_b),
            );
            attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

            let mut v_params = HashMap::new();
            v_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.v_proj.lora_a),
            );
            v_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.v_proj.lora_b),
            );
            attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

            let mut o_params = HashMap::new();
            o_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.o_proj.lora_a),
            );
            o_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.o_proj.lora_b),
            );
            attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // MLP LoRA params
            let mut mlp_params = HashMap::new();
            let mut gate_params = HashMap::new();
            gate_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.mlp.gate_proj.lora_a),
            );
            gate_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.mlp.gate_proj.lora_b),
            );
            mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

            let mut up_params = HashMap::new();
            up_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.mlp.up_proj.lora_a),
            );
            up_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.mlp.up_proj.lora_b),
            );
            mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

            let mut down_params = HashMap::new();
            down_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.mlp.down_proj.lora_a),
            );
            down_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.mlp.down_proj.lora_b),
            );
            mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            params.insert(prefix, NestedValue::Map(layer_params));
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // Attention LoRA params
            let mut attn_params = HashMap::new();
            let mut q_params = HashMap::new();
            q_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.q_proj.lora_a),
            );
            q_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.q_proj.lora_b),
            );
            attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

            let mut k_params = HashMap::new();
            k_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.k_proj.lora_a),
            );
            k_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.k_proj.lora_b),
            );
            attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

            let mut v_params = HashMap::new();
            v_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.v_proj.lora_a),
            );
            v_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.v_proj.lora_b),
            );
            attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

            let mut o_params = HashMap::new();
            o_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.o_proj.lora_a),
            );
            o_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.o_proj.lora_b),
            );
            attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // MLP LoRA params
            let mut mlp_params = HashMap::new();
            let mut gate_params = HashMap::new();
            gate_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.mlp.gate_proj.lora_a),
            );
            gate_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.mlp.gate_proj.lora_b),
            );
            mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

            let mut up_params = HashMap::new();
            up_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.mlp.up_proj.lora_a),
            );
            up_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.mlp.up_proj.lora_b),
            );
            mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

            let mut down_params = HashMap::new();
            down_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.mlp.down_proj.lora_a),
            );
            down_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.mlp.down_proj.lora_b),
            );
            mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            params.insert(prefix, NestedValue::Map(layer_params));
        }

        params
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        // All LoRA parameters are trainable
        self.parameters()
    }

    fn freeze_parameters(&mut self, _recursive: bool) {
        // LoRA params can't be frozen - they're always trainable
    }

    fn unfreeze_parameters(&mut self, _recursive: bool) {
        // LoRA params are always unfrozen
    }

    fn all_frozen(&self) -> Option<bool> {
        // LoRA params are never frozen
        Some(false)
    }

    fn any_frozen(&self) -> Option<bool> {
        // No LoRA params are frozen
        Some(false)
    }
}

impl TrainableModel for MistralLoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_internal(input_ids, mask)
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        let hidden_states = self
            .model
            .forward_with_positions(input_ids, mask, position_ids)?;

        if let Some(ref mut lm_head) = self.lm_head {
            Ok(mlx_rs::module::Module::forward(lm_head, &hidden_states)?)
        } else {
            Ok(self.model.embed_tokens.as_linear(&hidden_states)?)
        }
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        MistralLoraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(MistralLoraForCausalLM::create_cache(self, max_seq_len))
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("model.layers.{}", i);

            // Attention LoRA
            params.insert(
                Rc::from(format!("{}.self_attn.q_proj.lora_A.weight", prefix)),
                layer.self_attn.q_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.q_proj.lora_B.weight", prefix)),
                layer.self_attn.q_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.k_proj.lora_A.weight", prefix)),
                layer.self_attn.k_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.k_proj.lora_B.weight", prefix)),
                layer.self_attn.k_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.v_proj.lora_A.weight", prefix)),
                layer.self_attn.v_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.v_proj.lora_B.weight", prefix)),
                layer.self_attn.v_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.o_proj.lora_A.weight", prefix)),
                layer.self_attn.o_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.o_proj.lora_B.weight", prefix)),
                layer.self_attn.o_proj.lora_b.clone(),
            );

            // MLP LoRA
            params.insert(
                Rc::from(format!("{}.mlp.gate_proj.lora_A.weight", prefix)),
                layer.mlp.gate_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.gate_proj.lora_B.weight", prefix)),
                layer.mlp.gate_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.up_proj.lora_A.weight", prefix)),
                layer.mlp.up_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.up_proj.lora_B.weight", prefix)),
                layer.mlp.up_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.down_proj.lora_A.weight", prefix)),
                layer.mlp.down_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.down_proj.lora_B.weight", prefix)),
                layer.mlp.down_proj.lora_b.clone(),
            );
        }

        params
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            // Attention LoRA
            if let Some(p) = params.get(&Rc::from(format!(
                "{}.self_attn.q_proj.lora_A.weight",
                prefix
            ))) {
                layer.self_attn.q_proj.lora_a = p.clone();
            }
            if let Some(p) = params.get(&Rc::from(format!(
                "{}.self_attn.q_proj.lora_B.weight",
                prefix
            ))) {
                layer.self_attn.q_proj.lora_b = p.clone();
            }
            if let Some(p) = params.get(&Rc::from(format!(
                "{}.self_attn.k_proj.lora_A.weight",
                prefix
            ))) {
                layer.self_attn.k_proj.lora_a = p.clone();
            }
            if let Some(p) = params.get(&Rc::from(format!(
                "{}.self_attn.k_proj.lora_B.weight",
                prefix
            ))) {
                layer.self_attn.k_proj.lora_b = p.clone();
            }
            if let Some(p) = params.get(&Rc::from(format!(
                "{}.self_attn.v_proj.lora_A.weight",
                prefix
            ))) {
                layer.self_attn.v_proj.lora_a = p.clone();
            }
            if let Some(p) = params.get(&Rc::from(format!(
                "{}.self_attn.v_proj.lora_B.weight",
                prefix
            ))) {
                layer.self_attn.v_proj.lora_b = p.clone();
            }
            if let Some(p) = params.get(&Rc::from(format!(
                "{}.self_attn.o_proj.lora_A.weight",
                prefix
            ))) {
                layer.self_attn.o_proj.lora_a = p.clone();
            }
            if let Some(p) = params.get(&Rc::from(format!(
                "{}.self_attn.o_proj.lora_B.weight",
                prefix
            ))) {
                layer.self_attn.o_proj.lora_b = p.clone();
            }

            // MLP LoRA
            if let Some(p) =
                params.get(&Rc::from(format!("{}.mlp.gate_proj.lora_A.weight", prefix)))
            {
                layer.mlp.gate_proj.lora_a = p.clone();
            }
            if let Some(p) =
                params.get(&Rc::from(format!("{}.mlp.gate_proj.lora_B.weight", prefix)))
            {
                layer.mlp.gate_proj.lora_b = p.clone();
            }
            if let Some(p) = params.get(&Rc::from(format!("{}.mlp.up_proj.lora_A.weight", prefix)))
            {
                layer.mlp.up_proj.lora_a = p.clone();
            }
            if let Some(p) = params.get(&Rc::from(format!("{}.mlp.up_proj.lora_B.weight", prefix)))
            {
                layer.mlp.up_proj.lora_b = p.clone();
            }
            if let Some(p) =
                params.get(&Rc::from(format!("{}.mlp.down_proj.lora_A.weight", prefix)))
            {
                layer.mlp.down_proj.lora_a = p.clone();
            }
            if let Some(p) =
                params.get(&Rc::from(format!("{}.mlp.down_proj.lora_B.weight", prefix)))
            {
                layer.mlp.down_proj.lora_b = p.clone();
            }
        }
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        let params_ref: Vec<(Rc<str>, &Array)> =
            params.iter().map(|(k, v)| (k.clone(), v)).collect();
        mlx_rs::Array::save_safetensors(params_ref, None, path)?;
        Ok(())
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let loaded = mlx_rs::Array::load_safetensors(path)?;
        let params: HashMap<Rc<str>, Array> = loaded
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        self.set_lora_parameters(&params);
        Ok(())
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        self.checkpoint_config = Some(CheckpointConfig {
            enabled: true,
            layers_per_block,
            eval_at_boundaries: true,
        });
    }

    fn disable_gradient_checkpointing(&mut self) {
        self.checkpoint_config = None;
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> MistralConfig {
        MistralConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: None,
            max_position_embeddings: 512,
            sliding_window: Some(128),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            ..Default::default()
        }
    }

    fn small_lora_config() -> LoraConfig {
        LoraConfig {
            r: 4,
            alpha: 8.0,
            use_rslora: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_mistral_lora_attention() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut attn = MistralLoraAttention::new(&config, &lora_config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_mistral_lora_mlp() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut mlp = MistralLoraMLP::new(&config, &lora_config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_mistral_lora_model() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = MistralLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);
    }

    #[test]
    fn test_mistral_lora_trainable_params() {
        let config = small_config();
        let lora_config = small_lora_config();
        let model = MistralLoraForCausalLM::new(config, lora_config).unwrap();

        // 2 layers × (4 attention + 3 MLP) × 2 (A + B) matrices
        // Each A is [rank, in_features], each B is [out_features, rank]
        assert!(model.num_trainable_params() > 0);
    }

    #[test]
    fn test_lora_parameters() {
        let config = small_config();
        let lora_config = small_lora_config();
        let model = MistralLoraForCausalLM::new(config, lora_config).unwrap();

        let params = model.lora_parameters();
        // 2 layers × 7 projections × 2 (A + B) = 28 parameters
        assert_eq!(params.len(), 28);
    }

    #[test]
    fn test_kv_cache_support() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = MistralLoraForCausalLM::new(config, lora_config).unwrap();

        // Check that model supports KV cache
        use crate::TrainableModel;
        assert!(model.supports_kv_cache());

        // Create a cache (via trait method which returns Option<KVCache>)
        let mut cache = model.create_cache(128);
        assert_eq!(cache.rope_offset(), 0);

        // Test forward pass with cache
        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model
            .forward_with_cache(&input_ids, None, Some(&mut cache))
            .unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);
        assert_eq!(cache.rope_offset(), 4);

        // Test incremental generation
        let next_token = mlx_rs::Array::from_slice(&[5_i32], &[1, 1]);
        let next_logits = model
            .forward_with_cache(&next_token, None, Some(&mut cache))
            .unwrap();

        assert_eq!(next_logits.shape(), &[1, 1, 1000]);
        assert_eq!(cache.rope_offset(), 5);
    }
}
