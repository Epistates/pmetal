//! QLoRA-enabled Qwen3 model architecture.
//!
//! Implements Qwen3 with QLoRA (Quantized LoRA) for memory-efficient fine-tuning.
//! Base weights are stored in 4-bit NF4 format, reducing memory by ~87.5%.
//! LoRA adapters (A, B matrices) remain in full precision for training.
//!
//! Key Qwen3-specific features:
//! - Q/K normalization before RoPE
//! - Higher default vocab size (151936)
//! - Higher default rope_theta (1_000_000)

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::{ModuleParamMut, ModuleParamRef, ModuleParameters, Param},
    nested::NestedValue,
    nn, Array,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_models::architectures::qwen3::Qwen3Config;

use crate::{LoraError, QLoraConfig, QLoraLinear};

/// QLoRA-enabled attention layer for Qwen3.
///
/// Uses quantized base weights (NF4) with full-precision LoRA adapters.
/// Includes Qwen3-specific Q/K normalization before RoPE.
#[derive(Debug)]
pub struct Qwen3QLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,

    /// Query projection with QLoRA.
    pub q_proj: QLoraLinear,
    /// Key projection with QLoRA.
    pub k_proj: QLoraLinear,
    /// Value projection with QLoRA.
    pub v_proj: QLoraLinear,
    /// Output projection with QLoRA.
    pub o_proj: QLoraLinear,
    /// Query normalization (Qwen3 specific).
    pub q_norm: nn::RmsNorm,
    /// Key normalization (Qwen3 specific).
    pub k_norm: nn::RmsNorm,
    /// RoPE layer.
    pub rope: nn::Rope,
}

impl Qwen3QLoraAttention {
    /// Create a new QLoRA attention layer with random weights.
    pub fn new(config: &Qwen3Config, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = (head_dim as f32).sqrt().recip();

        // Create QLoRA linear layers for projections
        let q_proj = QLoraLinear::new(config.hidden_size, n_heads * head_dim, qlora_config, false)?;
        let k_proj = QLoraLinear::new(
            config.hidden_size,
            n_kv_heads * head_dim,
            qlora_config,
            false,
        )?;
        let v_proj = QLoraLinear::new(
            config.hidden_size,
            n_kv_heads * head_dim,
            qlora_config,
            false,
        )?;
        let o_proj = QLoraLinear::new(n_heads * head_dim, config.hidden_size, qlora_config, false)?;

        // Qwen3-specific: Q and K normalization before RoPE
        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();

        // Initialize RoPE with Qwen3's higher theta
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
            q_norm,
            k_norm,
            rope,
        })
    }

    /// Forward pass through attention.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `position_ids` - Optional position IDs for packed sequences
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project to Q, K, V using QLoRA layers (dequantization happens here)
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape for multi-head attention: [B, L, heads, head_dim]
        let queries = queries.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let keys = keys.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let values = values.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Qwen3-specific: Apply Q/K normalization before RoPE
        let queries = mlx_rs::module::Module::forward(&mut self.q_norm, &queries)?;
        let keys = mlx_rs::module::Module::forward(&mut self.k_norm, &keys)?;

        // Transpose for attention: [B, heads, L, head_dim]
        let queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys.transpose_axes(&[0, 2, 1, 3])?;
        let values = values.transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE - use position_ids if provided (for packed sequences)
        let (queries, keys) = if let Some(pos_ids) = position_ids {
            // Use custom RoPE with explicit position IDs
            use pmetal_mlx::kernels::rope::apply_rope_with_positions;
            let q =
                apply_rope_with_positions(&queries, pos_ids, self.head_dim, self.rope.base, 1.0)?;
            let k = apply_rope_with_positions(&keys, pos_ids, self.head_dim, self.rope.base, 1.0)?;
            (q, k)
        } else {
            // Use standard RoPE for sequential positions
            let q = mlx_rs::module::Module::forward(&mut self.rope, &queries)?;
            let k = mlx_rs::module::Module::forward(&mut self.rope, &keys)?;
            (q, k)
        };

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

    /// Get number of trainable parameters (LoRA only).
    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (q_quant, q_lora, q_total) = self.q_proj.memory_usage();
        let (k_quant, k_lora, k_total) = self.k_proj.memory_usage();
        let (v_quant, v_lora, v_total) = self.v_proj.memory_usage();
        let (o_quant, o_lora, o_total) = self.o_proj.memory_usage();

        (
            q_quant + k_quant + v_quant + o_quant,
            q_lora + k_lora + v_lora + o_lora,
            q_total + k_total + v_total + o_total,
        )
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

/// QLoRA-enabled MLP layer for Qwen3.
#[derive(Debug)]
pub struct Qwen3QloraMLP {
    /// Gate projection with QLoRA.
    pub gate_proj: QLoraLinear,
    /// Up projection with QLoRA.
    pub up_proj: QLoraLinear,
    /// Down projection with QLoRA.
    pub down_proj: QLoraLinear,
}

impl Qwen3QloraMLP {
    /// Create a new QLoRA MLP layer with random weights.
    pub fn new(config: &Qwen3Config, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let gate_proj = QLoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            qlora_config,
            false,
        )?;
        let up_proj = QLoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            qlora_config,
            false,
        )?;
        let down_proj = QLoraLinear::new(
            config.intermediate_size,
            config.hidden_size,
            qlora_config,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass (SwiGLU activation).
    pub fn forward(&self, x: &Array) -> Result<Array, LoraError> {
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

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (g_quant, g_lora, g_total) = self.gate_proj.memory_usage();
        let (u_quant, u_lora, u_total) = self.up_proj.memory_usage();
        let (d_quant, d_lora, d_total) = self.down_proj.memory_usage();

        (
            g_quant + u_quant + d_quant,
            g_lora + u_lora + d_lora,
            g_total + u_total + d_total,
        )
    }
}

/// QLoRA-enabled Qwen3 decoder layer.
#[derive(Debug)]
pub struct Qwen3QloraDecoderLayer {
    /// Self-attention layer with QLoRA.
    pub self_attn: Qwen3QLoraAttention,
    /// MLP layer with QLoRA.
    pub mlp: Qwen3QloraMLP,
    /// Input layer norm.
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl Qwen3QloraDecoderLayer {
    /// Create a new decoder layer with QLoRA (random weights).
    pub fn new(config: &Qwen3Config, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let self_attn = Qwen3QLoraAttention::new(config, qlora_config)?;
        let mlp = Qwen3QloraMLP::new(config, qlora_config)?;

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
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, LoraError> {
        // Pre-norm + attention + residual
        let normed = mlx_rs::module::Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward(&normed, mask, position_ids)?;
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

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (attn_quant, attn_lora, attn_total) = self.self_attn.memory_usage();
        let (mlp_quant, mlp_lora, mlp_total) = self.mlp.memory_usage();
        (
            attn_quant + mlp_quant,
            attn_lora + mlp_lora,
            attn_total + mlp_total,
        )
    }
}

/// QLoRA-enabled Qwen3 model (without LM head).
#[derive(Debug)]
pub struct Qwen3QloraModel {
    /// Configuration.
    pub config: Qwen3Config,
    /// QLoRA configuration.
    pub qlora_config: QLoraConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers with QLoRA.
    pub layers: Vec<Qwen3QloraDecoderLayer>,
    /// Final layer norm (frozen).
    pub norm: nn::RmsNorm,
}

impl Qwen3QloraModel {
    /// Create a new QLoRA Qwen3 model with random weights.
    pub fn new(config: Qwen3Config, qlora_config: QLoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| Qwen3QloraDecoderLayer::new(&config, &qlora_config))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();

        Ok(Self {
            config,
            qlora_config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.forward_with_checkpoint(input_ids, mask, position_ids, None)
    }

    /// Forward pass with optional gradient checkpointing.
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
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
            hidden_states = layer.forward(&hidden_states, mask.as_ref(), position_ids)?;

            // Checkpoint at block boundaries
            if checkpointing_enabled && (idx + 1) % layers_per_block == 0 {
                hidden_states.eval().map_err(LoraError::Mlx)?;
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

    /// Get total memory usage.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let mut total_quant = 0;
        let mut total_lora = 0;
        let mut total = 0;

        for layer in &self.layers {
            let (quant, lora, layer_total) = layer.memory_usage();
            total_quant += quant;
            total_lora += lora;
            total += layer_total;
        }

        (total_quant, total_lora, total)
    }
}

/// QLoRA-enabled Qwen3 model with LM head.
///
/// Memory-efficient fine-tuning with 4-bit quantized base weights.
/// Typical memory usage for a 0.6B model: ~0.5GB (vs 2.4GB for full precision).
#[derive(Debug)]
pub struct Qwen3QloraForCausalLM {
    /// Base model with QLoRA.
    pub model: Qwen3QloraModel,
    /// LM head (frozen, optional for tied weights).
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl Qwen3QloraForCausalLM {
    /// Create a new QLoRA Qwen3 model with random weights.
    pub fn new(config: Qwen3Config, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let qlora_config = QLoraConfig::from_lora(lora_config);
        Self::with_qlora_config(config, qlora_config)
    }

    /// Create with explicit QLoRA configuration.
    pub fn with_qlora_config(
        config: Qwen3Config,
        qlora_config: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let tie_weights = config.tie_word_embeddings;
        let model = Qwen3QloraModel::new(config.clone(), qlora_config)?;

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
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let checkpoint_config = self.checkpoint_config.clone();
        self.forward_with_checkpoint(input_ids, mask, position_ids, checkpoint_config.as_ref())
    }

    /// Forward pass with optional gradient checkpointing.
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let hidden_states =
            self.model
                .forward_with_checkpoint(input_ids, mask, position_ids, checkpoint_config)?;

        // Get logits from LM head or shared embeddings
        if let Some(ref mut lm_head) = self.lm_head {
            Ok(mlx_rs::module::Module::forward(lm_head, &hidden_states)?)
        } else {
            // Tie weights: use embedding weight transposed
            Ok(self.model.embed_tokens.as_linear(&hidden_states)?)
        }
    }

    /// Get all trainable LoRA parameters as a flat HashMap.
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

    /// Set LoRA parameters from a HashMap.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            macro_rules! set_param {
                ($layer:expr, $key:expr) => {
                    if let Some(value) = params.get(&Rc::from($key)) {
                        $layer = value.clone();
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

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    /// Get memory usage in bytes: (quantized_bytes, lora_bytes, total_bytes).
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    /// Get configuration.
    pub fn config(&self) -> &Qwen3Config {
        &self.model.config
    }

    /// Get QLoRA configuration.
    pub fn qlora_config(&self) -> &QLoraConfig {
        &self.model.qlora_config
    }

    /// Save LoRA weights to safetensors.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        Array::save_safetensors(params, None, path)?;
        Ok(())
    }

    /// Load LoRA weights from safetensors.
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

    /// Load and quantize base model weights from a HashMap.
    pub fn load_and_quantize_weights(
        &mut self,
        weights: &HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        // Load embed_tokens (kept in full precision)
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        // For transformer layers, quantize the weights
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            macro_rules! quantize_weight {
                ($proj:expr, $key:expr) => {
                    if let Some(w) = weights.get(&$key) {
                        *$proj = QLoraLinear::from_weight(w, None, &self.model.qlora_config)?;
                    }
                };
            }

            // Attention projections
            quantize_weight!(
                &mut layer.self_attn.q_proj,
                format!("{}.self_attn.q_proj.weight", prefix)
            );
            quantize_weight!(
                &mut layer.self_attn.k_proj,
                format!("{}.self_attn.k_proj.weight", prefix)
            );
            quantize_weight!(
                &mut layer.self_attn.v_proj,
                format!("{}.self_attn.v_proj.weight", prefix)
            );
            quantize_weight!(
                &mut layer.self_attn.o_proj,
                format!("{}.self_attn.o_proj.weight", prefix)
            );

            // MLP projections
            quantize_weight!(
                &mut layer.mlp.gate_proj,
                format!("{}.mlp.gate_proj.weight", prefix)
            );
            quantize_weight!(
                &mut layer.mlp.up_proj,
                format!("{}.mlp.up_proj.weight", prefix)
            );
            quantize_weight!(
                &mut layer.mlp.down_proj,
                format!("{}.mlp.down_proj.weight", prefix)
            );

            // Q/K norms (full precision)
            if let Some(w) = weights.get(&format!("{}.self_attn.q_norm.weight", prefix)) {
                layer.self_attn.q_norm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{}.self_attn.k_norm.weight", prefix)) {
                layer.self_attn.k_norm.weight = Param::new(w.clone());
            }

            // Layer norms (full precision)
            if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }
        }

        // Final norm (full precision)
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }

        // LM head if present (full precision)
        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                lm_head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }
}

/// Implement ModuleParameters for Qwen3QloraForCausalLM.
impl ModuleParameters for Qwen3QloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
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
        self.parameters()
    }

    fn freeze_parameters(&mut self, _recursive: bool) {}
    fn unfreeze_parameters(&mut self, _recursive: bool) {}
    fn all_frozen(&self) -> Option<bool> {
        Some(false)
    }
    fn any_frozen(&self) -> Option<bool> {
        Some(false)
    }
}

/// Implement TrainableModel for Qwen3QloraForCausalLM.
impl crate::TrainableModel for Qwen3QloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        Qwen3QloraForCausalLM::forward(self, input_ids, mask, None)
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        Qwen3QloraForCausalLM::forward(self, input_ids, mask, Some(position_ids))
    }

    fn num_trainable_params(&self) -> usize {
        Qwen3QloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        Qwen3QloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        Qwen3QloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        Qwen3QloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        Qwen3QloraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        Qwen3QloraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        Qwen3QloraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
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

    fn small_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: 16,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            ..Default::default()
        }
    }

    fn small_qlora_config() -> QLoraConfig {
        QLoraConfig {
            lora: LoraConfig {
                r: 4,
                alpha: 8.0,
                dropout: 0.0,
                use_rslora: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_qwen3_qlora_attention() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut attn = Qwen3QLoraAttention::new(&config, &qlora_config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_qwen3_qlora_model_forward() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = Qwen3QloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 256]);
    }

    #[test]
    fn test_qwen3_qlora_param_count() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = Qwen3QloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        assert!(model.num_trainable_params() > 0);
        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_qwen3_qlora_memory_usage() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = Qwen3QloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let (quantized, lora, total) = model.memory_usage();
        assert!(quantized > 0);
        assert!(lora > 0);
        assert_eq!(total, quantized + lora);
    }
}
