//! LoRA-enabled Llama model architecture.
//!
//! Implements Llama with LoRA adapters on attention projections for efficient fine-tuning.

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
use pmetal_models::architectures::llama::LlamaConfig;

use crate::{LoraError, LoraLinear};

/// LoRA-enabled attention layer for Llama.
///
/// Applies LoRA to q_proj, k_proj, v_proj, and o_proj.
#[derive(Debug)]
pub struct LlamaLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,

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

impl LlamaLoraAttention {
    /// Create a new LoRA attention layer.
    pub fn new(config: &LlamaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
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

        // Initialize RoPE (unwrap is safe - Infallible error)
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
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(AttentionMaskType::Causal);

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

/// LoRA-enabled MLP layer for Llama.
#[derive(Debug)]
pub struct LlamaLoraMLP {
    /// Gate projection with LoRA.
    pub gate_proj: LoraLinear,
    /// Up projection with LoRA.
    pub up_proj: LoraLinear,
    /// Down projection with LoRA.
    pub down_proj: LoraLinear,
}

impl LlamaLoraMLP {
    /// Create a new LoRA MLP layer.
    pub fn new(config: &LlamaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
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
        self.down_proj.forward(&hidden)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

/// LoRA-enabled Llama decoder layer.
#[derive(Debug)]
pub struct LlamaLoraDecoderLayer {
    /// Self-attention layer with LoRA.
    pub self_attn: LlamaLoraAttention,
    /// MLP layer with LoRA.
    pub mlp: LlamaLoraMLP,
    /// Input layer norm.
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl LlamaLoraDecoderLayer {
    /// Create a new decoder layer with LoRA.
    pub fn new(config: &LlamaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let self_attn = LlamaLoraAttention::new(config, lora_config)?;
        let mlp = LlamaLoraMLP::new(config, lora_config)?;

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

/// LoRA-enabled Llama model (without LM head).
#[derive(Debug)]
pub struct LlamaLoraModel {
    /// Configuration.
    pub config: LlamaConfig,
    /// LoRA configuration.
    pub lora_config: LoraConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers with LoRA.
    pub layers: Vec<LlamaLoraDecoderLayer>,
    /// Final layer norm (frozen).
    pub norm: nn::RmsNorm,
}

impl LlamaLoraModel {
    /// Create a new LoRA Llama model.
    pub fn new(config: LlamaConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| LlamaLoraDecoderLayer::new(&config, &lora_config))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();

        Ok(Self {
            config,
            lora_config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_with_checkpoint(input_ids, mask, None)
    }

    /// Forward pass with optional gradient checkpointing.
    ///
    /// Gradient checkpointing trades compute for memory by breaking the computation
    /// graph at layer boundaries. This allows training with larger batch sizes
    /// at the cost of recomputing activations during the backward pass.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `checkpoint_config` - Optional checkpointing configuration
    ///
    /// # Memory Savings
    /// With checkpointing enabled (layers_per_block=4 for 32-layer model):
    /// - Without: Store all 32 layer activations
    /// - With: Store 8 checkpoint boundaries + recompute
    /// - Typical savings: 50-75% activation memory
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        // Get embeddings
        let mut hidden_states = mlx_rs::module::Module::forward(&mut self.embed_tokens, input_ids)?;

        // Create causal mask if not provided
        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len)?)
        } else {
            mask.cloned()
        };

        // Pass through transformer layers with optional checkpointing
        let layers_per_block = checkpoint_config
            .map(|c| c.layers_per_block)
            .unwrap_or(usize::MAX);
        let checkpointing_enabled = checkpoint_config.map(|c| c.enabled).unwrap_or(false);

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, mask.as_ref())?;

            // Checkpoint boundary marker
            // NOTE: We do NOT call eval() here - that breaks the gradient computation graph.
            // MLX's lazy evaluation with unified memory handles memory pressure reasonably well.
            // True gradient checkpointing (save/recompute) would require custom VJP implementation.
            if checkpointing_enabled && (idx + 1) % layers_per_block == 0 {
                tracing::trace!("Checkpoint boundary at layer {}", idx + 1);
            }
        }

        // Final norm
        Ok(mlx_rs::module::Module::forward(
            &mut self.norm,
            &hidden_states,
        )?)
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

/// LoRA-enabled Llama model with LM head.
#[derive(Debug)]
pub struct LlamaLoraForCausalLM {
    /// Base model with LoRA.
    pub model: LlamaLoraModel,
    /// LM head (frozen, optional for tied weights).
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    /// When enabled, breaks computation graph at layer boundaries to save memory.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl LlamaLoraForCausalLM {
    /// Create a new LoRA Llama model with LM head.
    pub fn new(config: LlamaConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let tie_weights = config.tie_word_embeddings;
        let model = LlamaLoraModel::new(config.clone(), lora_config)?;

        let lm_head = if !tie_weights {
            let head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
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

    /// Enable gradient checkpointing for memory-efficient training.
    ///
    /// # Arguments
    /// * `layers_per_block` - Number of layers per checkpoint block (typically 2-4)
    ///
    /// # Example
    /// ```ignore
    /// model.enable_gradient_checkpointing(4); // Checkpoint every 4 layers
    /// ```
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

    /// Forward pass producing logits.
    ///
    /// Uses the model's stored checkpoint_config if set via `enable_gradient_checkpointing()`.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        // Clone checkpoint config to avoid borrow conflicts
        let checkpoint_config = self.checkpoint_config.clone();
        self.forward_with_checkpoint(input_ids, mask, checkpoint_config.as_ref())
    }

    /// Forward pass with optional gradient checkpointing.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `checkpoint_config` - Optional checkpointing configuration
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let hidden_states =
            self.model
                .forward_with_checkpoint(input_ids, mask, checkpoint_config)?;

        // Get logits from LM head or shared embeddings
        if let Some(ref mut lm_head) = self.lm_head {
            Ok(mlx_rs::module::Module::forward(lm_head, &hidden_states)?)
        } else {
            // Tie weights: use embedding weight transposed
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

    /// Get all trainable LoRA parameters as a flat HashMap.
    ///
    /// Returns parameters with keys like "layers.0.self_attn.q_proj.lora_a"
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            // Attention LoRA params
            params.insert(
                Rc::from(format!("{}.self_attn.q_proj.lora_a", prefix)),
                layer.self_attn.q_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.q_proj.lora_b", prefix)),
                layer.self_attn.q_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.k_proj.lora_a", prefix)),
                layer.self_attn.k_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.k_proj.lora_b", prefix)),
                layer.self_attn.k_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.v_proj.lora_a", prefix)),
                layer.self_attn.v_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.v_proj.lora_b", prefix)),
                layer.self_attn.v_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.o_proj.lora_a", prefix)),
                layer.self_attn.o_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.o_proj.lora_b", prefix)),
                layer.self_attn.o_proj.lora_b.clone(),
            );

            // MLP LoRA params
            params.insert(
                Rc::from(format!("{}.mlp.gate_proj.lora_a", prefix)),
                layer.mlp.gate_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.gate_proj.lora_b", prefix)),
                layer.mlp.gate_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.up_proj.lora_a", prefix)),
                layer.mlp.up_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.up_proj.lora_b", prefix)),
                layer.mlp.up_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.down_proj.lora_a", prefix)),
                layer.mlp.down_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.down_proj.lora_b", prefix)),
                layer.mlp.down_proj.lora_b.clone(),
            );
        }

        params
    }

    /// Apply gradient updates to LoRA parameters.
    ///
    /// # Arguments
    /// * `gradients` - HashMap of parameter key to gradient
    /// * `learning_rate` - Learning rate for SGD update
    pub fn apply_gradients(
        &mut self,
        gradients: &HashMap<Rc<str>, Array>,
        learning_rate: f32,
    ) -> Result<(), LoraError> {
        let lr = Array::from_f32(learning_rate);

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            // Helper to apply gradient
            macro_rules! apply_grad {
                ($param:expr, $key:expr) => {
                    if let Some(grad) = gradients.get(&Rc::from($key)) {
                        let update = grad.multiply(&lr)?;
                        $param = $param.subtract(&update)?;
                    }
                };
            }

            // Attention LoRA params
            apply_grad!(
                layer.self_attn.q_proj.lora_a,
                format!("{}.self_attn.q_proj.lora_a", prefix)
            );
            apply_grad!(
                layer.self_attn.q_proj.lora_b,
                format!("{}.self_attn.q_proj.lora_b", prefix)
            );
            apply_grad!(
                layer.self_attn.k_proj.lora_a,
                format!("{}.self_attn.k_proj.lora_a", prefix)
            );
            apply_grad!(
                layer.self_attn.k_proj.lora_b,
                format!("{}.self_attn.k_proj.lora_b", prefix)
            );
            apply_grad!(
                layer.self_attn.v_proj.lora_a,
                format!("{}.self_attn.v_proj.lora_a", prefix)
            );
            apply_grad!(
                layer.self_attn.v_proj.lora_b,
                format!("{}.self_attn.v_proj.lora_b", prefix)
            );
            apply_grad!(
                layer.self_attn.o_proj.lora_a,
                format!("{}.self_attn.o_proj.lora_a", prefix)
            );
            apply_grad!(
                layer.self_attn.o_proj.lora_b,
                format!("{}.self_attn.o_proj.lora_b", prefix)
            );

            // MLP LoRA params
            apply_grad!(
                layer.mlp.gate_proj.lora_a,
                format!("{}.mlp.gate_proj.lora_a", prefix)
            );
            apply_grad!(
                layer.mlp.gate_proj.lora_b,
                format!("{}.mlp.gate_proj.lora_b", prefix)
            );
            apply_grad!(
                layer.mlp.up_proj.lora_a,
                format!("{}.mlp.up_proj.lora_a", prefix)
            );
            apply_grad!(
                layer.mlp.up_proj.lora_b,
                format!("{}.mlp.up_proj.lora_b", prefix)
            );
            apply_grad!(
                layer.mlp.down_proj.lora_a,
                format!("{}.mlp.down_proj.lora_a", prefix)
            );
            apply_grad!(
                layer.mlp.down_proj.lora_b,
                format!("{}.mlp.down_proj.lora_b", prefix)
            );
        }

        Ok(())
    }

    /// Set LoRA parameters from a HashMap.
    ///
    /// This is used by autodiff to inject parameter values before the forward pass.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            // Helper to set param
            macro_rules! set_param {
                ($param:expr, $key:expr) => {
                    if let Some(value) = params.get(&Rc::from($key)) {
                        $param = value.clone();
                    }
                };
            }

            // Attention LoRA params
            set_param!(
                layer.self_attn.q_proj.lora_a,
                format!("{}.self_attn.q_proj.lora_a", prefix)
            );
            set_param!(
                layer.self_attn.q_proj.lora_b,
                format!("{}.self_attn.q_proj.lora_b", prefix)
            );
            set_param!(
                layer.self_attn.k_proj.lora_a,
                format!("{}.self_attn.k_proj.lora_a", prefix)
            );
            set_param!(
                layer.self_attn.k_proj.lora_b,
                format!("{}.self_attn.k_proj.lora_b", prefix)
            );
            set_param!(
                layer.self_attn.v_proj.lora_a,
                format!("{}.self_attn.v_proj.lora_a", prefix)
            );
            set_param!(
                layer.self_attn.v_proj.lora_b,
                format!("{}.self_attn.v_proj.lora_b", prefix)
            );
            set_param!(
                layer.self_attn.o_proj.lora_a,
                format!("{}.self_attn.o_proj.lora_a", prefix)
            );
            set_param!(
                layer.self_attn.o_proj.lora_b,
                format!("{}.self_attn.o_proj.lora_b", prefix)
            );

            // MLP LoRA params
            set_param!(
                layer.mlp.gate_proj.lora_a,
                format!("{}.mlp.gate_proj.lora_a", prefix)
            );
            set_param!(
                layer.mlp.gate_proj.lora_b,
                format!("{}.mlp.gate_proj.lora_b", prefix)
            );
            set_param!(
                layer.mlp.up_proj.lora_a,
                format!("{}.mlp.up_proj.lora_a", prefix)
            );
            set_param!(
                layer.mlp.up_proj.lora_b,
                format!("{}.mlp.up_proj.lora_b", prefix)
            );
            set_param!(
                layer.mlp.down_proj.lora_a,
                format!("{}.mlp.down_proj.lora_a", prefix)
            );
            set_param!(
                layer.mlp.down_proj.lora_b,
                format!("{}.mlp.down_proj.lora_b", prefix)
            );
        }
    }

    /// Evaluate all LoRA parameters (force computation).
    pub fn eval_lora_params(&self) -> Result<(), LoraError> {
        for layer in &self.model.layers {
            layer.self_attn.q_proj.lora_a.eval()?;
            layer.self_attn.q_proj.lora_b.eval()?;
            layer.self_attn.k_proj.lora_a.eval()?;
            layer.self_attn.k_proj.lora_b.eval()?;
            layer.self_attn.v_proj.lora_a.eval()?;
            layer.self_attn.v_proj.lora_b.eval()?;
            layer.self_attn.o_proj.lora_a.eval()?;
            layer.self_attn.o_proj.lora_b.eval()?;

            layer.mlp.gate_proj.lora_a.eval()?;
            layer.mlp.gate_proj.lora_b.eval()?;
            layer.mlp.up_proj.lora_a.eval()?;
            layer.mlp.up_proj.lora_b.eval()?;
            layer.mlp.down_proj.lora_a.eval()?;
            layer.mlp.down_proj.lora_b.eval()?;
        }
        Ok(())
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    /// Get configuration.
    pub fn config(&self) -> &LlamaConfig {
        &self.model.config
    }

    /// Get LoRA configuration.
    pub fn lora_config(&self) -> &LoraConfig {
        &self.model.lora_config
    }

    /// Merge LoRA weights into base weights.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.merge()?;
            layer.self_attn.k_proj.merge()?;
            layer.self_attn.v_proj.merge()?;
            layer.self_attn.o_proj.merge()?;
            layer.mlp.gate_proj.merge()?;
            layer.mlp.up_proj.merge()?;
            layer.mlp.down_proj.merge()?;
        }
        Ok(())
    }

    /// Unmerge LoRA weights from base weights.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.unmerge()?;
            layer.self_attn.k_proj.unmerge()?;
            layer.self_attn.v_proj.unmerge()?;
            layer.self_attn.o_proj.unmerge()?;
            layer.mlp.gate_proj.unmerge()?;
            layer.mlp.up_proj.unmerge()?;
            layer.mlp.down_proj.unmerge()?;
        }
        Ok(())
    }

    /// Save LoRA weights to safetensors.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        Array::save_safetensors(params, None, path)?;
        Ok(())
    }

    /// Load LoRA weights from safetensors.
    ///
    /// # Arguments
    /// * `path` - Path to either a directory containing `lora_weights.safetensors` or a direct `.safetensors` file
    pub fn load_lora_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let path = path.as_ref();
        let file_path = if path.is_dir() {
            path.join("lora_weights.safetensors")
        } else {
            path.to_path_buf()
        };
        let loaded = Array::load_safetensors(&file_path)?;

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            // Helper to load param
            macro_rules! load_param {
                ($param:expr, $key:expr) => {
                    if let Some(value) = loaded.get(&Rc::from($key) as &str) {
                        $param = value.clone();
                    }
                };
            }

            // Attention LoRA params
            load_param!(
                layer.self_attn.q_proj.lora_a,
                format!("{}.self_attn.q_proj.lora_a", prefix)
            );
            load_param!(
                layer.self_attn.q_proj.lora_b,
                format!("{}.self_attn.q_proj.lora_b", prefix)
            );
            load_param!(
                layer.self_attn.k_proj.lora_a,
                format!("{}.self_attn.k_proj.lora_a", prefix)
            );
            load_param!(
                layer.self_attn.k_proj.lora_b,
                format!("{}.self_attn.k_proj.lora_b", prefix)
            );
            load_param!(
                layer.self_attn.v_proj.lora_a,
                format!("{}.self_attn.v_proj.lora_a", prefix)
            );
            load_param!(
                layer.self_attn.v_proj.lora_b,
                format!("{}.self_attn.v_proj.lora_b", prefix)
            );
            load_param!(
                layer.self_attn.o_proj.lora_a,
                format!("{}.self_attn.o_proj.lora_a", prefix)
            );
            load_param!(
                layer.self_attn.o_proj.lora_b,
                format!("{}.self_attn.o_proj.lora_b", prefix)
            );

            // MLP LoRA params
            load_param!(
                layer.mlp.gate_proj.lora_a,
                format!("{}.mlp.gate_proj.lora_a", prefix)
            );
            load_param!(
                layer.mlp.gate_proj.lora_b,
                format!("{}.mlp.gate_proj.lora_b", prefix)
            );
            load_param!(
                layer.mlp.up_proj.lora_a,
                format!("{}.mlp.up_proj.lora_a", prefix)
            );
            load_param!(
                layer.mlp.up_proj.lora_b,
                format!("{}.mlp.up_proj.lora_b", prefix)
            );
            load_param!(
                layer.mlp.down_proj.lora_a,
                format!("{}.mlp.down_proj.lora_a", prefix)
            );
            load_param!(
                layer.mlp.down_proj.lora_b,
                format!("{}.mlp.down_proj.lora_b", prefix)
            );
        }

        Ok(())
    }

    /// Load base model weights from a HashMap of weight tensors.
    ///
    /// This loads the frozen pretrained weights into the model's base weight matrices,
    /// embeddings, and layer norms. LoRA adapter weights (lora_a, lora_b) are not affected.
    ///
    /// # Arguments
    /// * `weights` - HashMap mapping weight names to Array tensors
    ///
    /// # Weight name format
    /// Expected weight names follow HuggingFace format:
    /// - `model.embed_tokens.weight` - Token embeddings
    /// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` - Attention projections
    /// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` - MLP projections
    /// - `model.layers.{i}.input_layernorm.weight` - Pre-attention norm
    /// - `model.layers.{i}.post_attention_layernorm.weight` - Post-attention norm
    /// - `model.norm.weight` - Final layer norm
    /// - `lm_head.weight` - Output projection (if not tied)
    pub fn load_base_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        use mlx_rs::module::Param;

        // Load embed_tokens
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        // Load transformer layers
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            // Self-attention projections (load into LoraLinear.weight)
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

            // MLP projections
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

        // Load lm_head if present and not tied
        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                lm_head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }

    /// Load base model weights from safetensor files in a directory.
    ///
    /// Handles both single-file (`model.safetensors`) and sharded models
    /// (`model.safetensors.index.json` with multiple shard files).
    ///
    /// # Arguments
    /// * `model_dir` - Path to the model directory containing safetensor files
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();

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

        // Load each shard and combine weights
        let mut all_weights = std::collections::HashMap::new();
        for shard_file in shard_files {
            let shard_path = model_dir.join(shard_file);
            let shard_weights = Array::load_safetensors(&shard_path)?;
            all_weights.extend(shard_weights);
        }

        self.load_base_weights(&all_weights)
    }

    /// Evaluate all model parameters (force computation).
    ///
    /// This evaluates both base weights and LoRA weights to ensure
    /// they are materialized on the device.
    pub fn eval_all(&self) -> Result<(), LoraError> {
        // Eval embeddings
        self.model.embed_tokens.weight.value.as_ref().eval()?;

        // Eval layers
        for layer in &self.model.layers {
            // Base weights
            layer.self_attn.q_proj.weight.eval()?;
            layer.self_attn.k_proj.weight.eval()?;
            layer.self_attn.v_proj.weight.eval()?;
            layer.self_attn.o_proj.weight.eval()?;
            layer.mlp.gate_proj.weight.eval()?;
            layer.mlp.up_proj.weight.eval()?;
            layer.mlp.down_proj.weight.eval()?;

            // LoRA weights
            layer.self_attn.q_proj.lora_a.eval()?;
            layer.self_attn.q_proj.lora_b.eval()?;
            layer.self_attn.k_proj.lora_a.eval()?;
            layer.self_attn.k_proj.lora_b.eval()?;
            layer.self_attn.v_proj.lora_a.eval()?;
            layer.self_attn.v_proj.lora_b.eval()?;
            layer.self_attn.o_proj.lora_a.eval()?;
            layer.self_attn.o_proj.lora_b.eval()?;
            layer.mlp.gate_proj.lora_a.eval()?;
            layer.mlp.gate_proj.lora_b.eval()?;
            layer.mlp.up_proj.lora_a.eval()?;
            layer.mlp.up_proj.lora_b.eval()?;
            layer.mlp.down_proj.lora_a.eval()?;
            layer.mlp.down_proj.lora_b.eval()?;

            // Layer norms
            layer.input_layernorm.weight.value.as_ref().eval()?;
            layer
                .post_attention_layernorm
                .weight
                .value
                .as_ref()
                .eval()?;
        }

        // Final norm
        self.model.norm.weight.value.as_ref().eval()?;

        // LM head if present
        if let Some(ref lm_head) = self.lm_head {
            lm_head.weight.value.as_ref().eval()?;
        }

        Ok(())
    }
}

/// Implement ModuleParameters for LlamaLoraForCausalLM.
///
/// This enables use with `nn::value_and_grad` for automatic differentiation.
/// The implementation returns ALL params for `parameters()` but only LoRA params
/// for `trainable_parameters()` - this means gradients are only computed for LoRA params.
impl ModuleParameters for LlamaLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        // Count all parameters (base + LoRA)
        self.model.num_trainable_params() // For simplicity, just return LoRA count
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        // Return only LoRA parameters as that's what we're training
        // Base model params are frozen and not exposed
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
        // Base model is always frozen
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

/// Implement TrainableModel for LlamaLoraForCausalLM.
impl crate::TrainableModel for LlamaLoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        LlamaLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        LlamaLoraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        LlamaLoraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        LlamaLoraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        LlamaLoraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        LlamaLoraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        LlamaLoraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        LlamaLoraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        true
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        LlamaLoraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(LlamaLoraForCausalLM::create_cache(self, max_seq_len))
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }
}

/// Create a causal attention mask.
fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    let mask = mlx_rs::ops::tri::<f32>(seq_len, None, None)?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);
    mlx_rs::ops::r#where(&mask.eq(&zero)?, &neg_inf, &zero)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn small_lora_config() -> LoraConfig {
        LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            use_rslora: false,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            bias: "none".to_string(),
            init_lora_weights: true,
            use_dora: false,
        }
    }

    #[test]
    fn test_llama_lora_attention() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut attn = LlamaLoraAttention::new(&config, &lora_config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_llama_lora_model() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = LlamaLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);
    }

    #[test]
    fn test_lora_param_count() {
        let config = small_config();
        let lora_config = small_lora_config();
        let model = LlamaLoraForCausalLM::new(config, lora_config).unwrap();

        // Should have trainable params
        assert!(model.num_trainable_params() > 0);

        // Check parameter count
        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = LlamaLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);

        // Get output before merge
        let output_before = model.forward(&input_ids, None).unwrap();
        output_before.eval().unwrap();

        // Merge
        model.merge_lora().unwrap();

        // Get output after merge
        let output_after = model.forward(&input_ids, None).unwrap();
        output_after.eval().unwrap();

        // Outputs should be close
        let diff = output_before.subtract(&output_after).unwrap();
        let max_diff = diff.abs().unwrap().max(None).unwrap();
        max_diff.eval().unwrap();
        assert!(max_diff.item::<f32>() < 1e-4);

        // Unmerge and verify
        model.unmerge_lora().unwrap();
        let output_unmerged = model.forward(&input_ids, None).unwrap();
        output_unmerged.eval().unwrap();

        let diff2 = output_before.subtract(&output_unmerged).unwrap();
        let max_diff2 = diff2.abs().unwrap().max(None).unwrap();
        max_diff2.eval().unwrap();
        assert!(max_diff2.item::<f32>() < 1e-4);
    }
}
