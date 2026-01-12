//! QLoRA-enabled Mistral model architecture.
//!
//! Implements Mistral with QLoRA (Quantized LoRA) for memory-efficient fine-tuning.
//! Base weights are stored in 4-bit NF4 format, reducing memory by ~87.5%.
//! LoRA adapters (A, B matrices) remain in full precision for training.
//!
//! Supports sliding window attention for efficient long-context handling.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::{ModuleParamMut, ModuleParamRef, ModuleParameters},
    nested::NestedValue,
    nn,
    Array,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_models::architectures::mistral::MistralConfig;

use crate::{LoraError, QLoraConfig, QLoraLinear, TrainableModel};

/// QLoRA-enabled attention layer for Mistral.
///
/// Uses quantized base weights (NF4) with full-precision LoRA adapters.
/// Supports sliding window attention for efficient long-context handling.
#[derive(Debug)]
pub struct MistralQLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// Sliding window size (None for full attention).
    pub sliding_window: Option<i32>,

    /// Query projection with QLoRA.
    pub q_proj: QLoraLinear,
    /// Key projection with QLoRA.
    pub k_proj: QLoraLinear,
    /// Value projection with QLoRA.
    pub v_proj: QLoraLinear,
    /// Output projection with QLoRA.
    pub o_proj: QLoraLinear,
    /// RoPE layer.
    pub rope: nn::Rope,
}

impl MistralQLoraAttention {
    /// Create a new QLoRA attention layer with random weights.
    pub fn new(config: &MistralConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = (head_dim as f32).sqrt().recip();

        // Create QLoRA linear layers for projections
        let q_proj = QLoraLinear::new(
            config.hidden_size,
            n_heads * head_dim,
            qlora_config,
            false,
        )?;
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
        let o_proj = QLoraLinear::new(
            n_heads * head_dim,
            config.hidden_size,
            qlora_config,
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

    /// Create QLoRA attention layer from pre-trained weights.
    ///
    /// Quantizes the provided weights to NF4 format.
    pub fn from_weights(
        config: &MistralConfig,
        qlora_config: &QLoraConfig,
        q_weight: &Array,
        k_weight: &Array,
        v_weight: &Array,
        o_weight: &Array,
    ) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = (head_dim as f32).sqrt().recip();

        // Quantize pre-trained weights
        let q_proj = QLoraLinear::from_weight(q_weight, None, qlora_config)?;
        let k_proj = QLoraLinear::from_weight(k_weight, None, qlora_config)?;
        let v_proj = QLoraLinear::from_weight(v_weight, None, qlora_config)?;
        let o_proj = QLoraLinear::from_weight(o_weight, None, qlora_config)?;

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

        // Project to Q, K, V using QLoRA layers (dequantization happens here)
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

/// QLoRA-enabled MLP layer for Mistral (SwiGLU).
#[derive(Debug)]
pub struct MistralQloraMLP {
    /// Gate projection with QLoRA.
    pub gate_proj: QLoraLinear,
    /// Up projection with QLoRA.
    pub up_proj: QLoraLinear,
    /// Down projection with QLoRA.
    pub down_proj: QLoraLinear,
}

impl MistralQloraMLP {
    /// Create a new QLoRA MLP layer with random weights.
    pub fn new(config: &MistralConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
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

    /// Create QLoRA MLP from pre-trained weights.
    pub fn from_weights(
        qlora_config: &QLoraConfig,
        gate_weight: &Array,
        up_weight: &Array,
        down_weight: &Array,
    ) -> Result<Self, LoraError> {
        let gate_proj = QLoraLinear::from_weight(gate_weight, None, qlora_config)?;
        let up_proj = QLoraLinear::from_weight(up_weight, None, qlora_config)?;
        let down_proj = QLoraLinear::from_weight(down_weight, None, qlora_config)?;

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

/// QLoRA-enabled Mistral decoder layer.
#[derive(Debug)]
pub struct MistralQloraDecoderLayer {
    /// Self-attention layer with QLoRA.
    pub self_attn: MistralQLoraAttention,
    /// MLP layer with QLoRA.
    pub mlp: MistralQloraMLP,
    /// Input layer norm.
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl MistralQloraDecoderLayer {
    /// Create a new decoder layer with QLoRA (random weights).
    pub fn new(config: &MistralConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let self_attn = MistralQLoraAttention::new(config, qlora_config)?;
        let mlp = MistralQloraMLP::new(config, qlora_config)?;

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

/// QLoRA-enabled Mistral model (without LM head).
#[derive(Debug)]
pub struct MistralQloraModel {
    /// Configuration.
    pub config: MistralConfig,
    /// QLoRA configuration.
    pub qlora_config: QLoraConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers with QLoRA.
    pub layers: Vec<MistralQloraDecoderLayer>,
    /// Final layer norm (frozen).
    pub norm: nn::RmsNorm,
}

impl MistralQloraModel {
    /// Create a new QLoRA Mistral model with random weights.
    pub fn new(config: MistralConfig, qlora_config: QLoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| MistralQloraDecoderLayer::new(&config, &qlora_config))
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
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        // Get embeddings
        let mut hidden_states = mlx_rs::module::Module::forward(&mut self.embed_tokens, input_ids)?;

        // Create causal mask if not provided
        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len, self.config.sliding_window)?)
        } else {
            mask.cloned()
        };

        // Pass through transformer layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
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

/// QLoRA-enabled Mistral model with LM head.
///
/// Memory-efficient fine-tuning with 4-bit quantized base weights.
/// Typical memory usage for a 7B model: ~4GB (vs 28GB for full precision).
#[derive(Debug)]
pub struct MistralQloraForCausalLM {
    /// Base model with QLoRA.
    pub model: MistralQloraModel,
    /// LM head (frozen, optional for tied weights).
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl MistralQloraForCausalLM {
    /// Create a new QLoRA Mistral model with random weights.
    pub fn new(config: MistralConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let qlora_config = QLoraConfig::from_lora(lora_config);
        Self::with_qlora_config(config, qlora_config)
    }

    /// Create with explicit QLoRA configuration.
    pub fn with_qlora_config(config: MistralConfig, qlora_config: QLoraConfig) -> Result<Self, LoraError> {
        let tie_weights = config.tie_word_embeddings;
        let model = MistralQloraModel::new(config.clone(), qlora_config)?;

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

    /// Forward pass producing logits.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let hidden_states = self.model.forward(input_ids, mask)?;

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
            let prefix = format!("model.layers.{}", i);

            // Attention LoRA params
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

            // MLP LoRA params
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

    /// Set LoRA parameters from a HashMap.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            macro_rules! set_param {
                ($layer:expr, $key:expr) => {
                    if let Some(value) = params.get(&Rc::from($key)) {
                        $layer = value.clone();
                    }
                };
            }

            // Attention LoRA params
            set_param!(layer.self_attn.q_proj.lora_a, format!("{}.self_attn.q_proj.lora_A.weight", prefix));
            set_param!(layer.self_attn.q_proj.lora_b, format!("{}.self_attn.q_proj.lora_B.weight", prefix));
            set_param!(layer.self_attn.k_proj.lora_a, format!("{}.self_attn.k_proj.lora_A.weight", prefix));
            set_param!(layer.self_attn.k_proj.lora_b, format!("{}.self_attn.k_proj.lora_B.weight", prefix));
            set_param!(layer.self_attn.v_proj.lora_a, format!("{}.self_attn.v_proj.lora_A.weight", prefix));
            set_param!(layer.self_attn.v_proj.lora_b, format!("{}.self_attn.v_proj.lora_B.weight", prefix));
            set_param!(layer.self_attn.o_proj.lora_a, format!("{}.self_attn.o_proj.lora_A.weight", prefix));
            set_param!(layer.self_attn.o_proj.lora_b, format!("{}.self_attn.o_proj.lora_B.weight", prefix));

            // MLP LoRA params
            set_param!(layer.mlp.gate_proj.lora_a, format!("{}.mlp.gate_proj.lora_A.weight", prefix));
            set_param!(layer.mlp.gate_proj.lora_b, format!("{}.mlp.gate_proj.lora_B.weight", prefix));
            set_param!(layer.mlp.up_proj.lora_a, format!("{}.mlp.up_proj.lora_A.weight", prefix));
            set_param!(layer.mlp.up_proj.lora_b, format!("{}.mlp.up_proj.lora_B.weight", prefix));
            set_param!(layer.mlp.down_proj.lora_a, format!("{}.mlp.down_proj.lora_A.weight", prefix));
            set_param!(layer.mlp.down_proj.lora_b, format!("{}.mlp.down_proj.lora_B.weight", prefix));
        }
    }

    /// Evaluate all LoRA parameters.
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

    /// Get memory usage in bytes: (quantized_bytes, lora_bytes, total_bytes).
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    /// Get memory savings compared to full-precision model.
    pub fn memory_savings(&self) -> f32 {
        let (quantized, lora, _) = self.memory_usage();
        let full_precision = self.model.layers.iter()
            .map(|l| {
                // Each QLoraLinear has frozen params
                l.self_attn.q_proj.num_frozen_params() * 4
                    + l.self_attn.k_proj.num_frozen_params() * 4
                    + l.self_attn.v_proj.num_frozen_params() * 4
                    + l.self_attn.o_proj.num_frozen_params() * 4
                    + l.mlp.gate_proj.num_frozen_params() * 4
                    + l.mlp.up_proj.num_frozen_params() * 4
                    + l.mlp.down_proj.num_frozen_params() * 4
            })
            .sum::<usize>() + lora;

        (quantized + lora) as f32 / full_precision as f32
    }

    /// Get configuration.
    pub fn config(&self) -> &MistralConfig {
        &self.model.config
    }

    /// Get QLoRA configuration.
    pub fn qlora_config(&self) -> &QLoraConfig {
        &self.model.qlora_config
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
    pub fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let path = path.as_ref();
        let file_path = if path.is_dir() {
            path.join("lora_weights.safetensors")
        } else {
            path.to_path_buf()
        };
        let loaded = Array::load_safetensors(&file_path)?;

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            macro_rules! load_param {
                ($param:expr, $key:expr) => {
                    if let Some(value) = loaded.get(&Rc::from($key) as &str) {
                        $param = value.clone();
                    }
                };
            }

            // Attention LoRA params
            load_param!(layer.self_attn.q_proj.lora_a, format!("{}.self_attn.q_proj.lora_A.weight", prefix));
            load_param!(layer.self_attn.q_proj.lora_b, format!("{}.self_attn.q_proj.lora_B.weight", prefix));
            load_param!(layer.self_attn.k_proj.lora_a, format!("{}.self_attn.k_proj.lora_A.weight", prefix));
            load_param!(layer.self_attn.k_proj.lora_b, format!("{}.self_attn.k_proj.lora_B.weight", prefix));
            load_param!(layer.self_attn.v_proj.lora_a, format!("{}.self_attn.v_proj.lora_A.weight", prefix));
            load_param!(layer.self_attn.v_proj.lora_b, format!("{}.self_attn.v_proj.lora_B.weight", prefix));
            load_param!(layer.self_attn.o_proj.lora_a, format!("{}.self_attn.o_proj.lora_A.weight", prefix));
            load_param!(layer.self_attn.o_proj.lora_b, format!("{}.self_attn.o_proj.lora_B.weight", prefix));

            // MLP LoRA params
            load_param!(layer.mlp.gate_proj.lora_a, format!("{}.mlp.gate_proj.lora_A.weight", prefix));
            load_param!(layer.mlp.gate_proj.lora_b, format!("{}.mlp.gate_proj.lora_B.weight", prefix));
            load_param!(layer.mlp.up_proj.lora_a, format!("{}.mlp.up_proj.lora_A.weight", prefix));
            load_param!(layer.mlp.up_proj.lora_b, format!("{}.mlp.up_proj.lora_B.weight", prefix));
            load_param!(layer.mlp.down_proj.lora_a, format!("{}.mlp.down_proj.lora_A.weight", prefix));
            load_param!(layer.mlp.down_proj.lora_b, format!("{}.mlp.down_proj.lora_B.weight", prefix));
        }

        Ok(())
    }

    /// Load and quantize base model weights from a HashMap.
    ///
    /// Takes full-precision weights and quantizes them to NF4.
    pub fn load_and_quantize_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        use mlx_rs::module::Param;

        // Load embed_tokens (kept in full precision)
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        // For transformer layers, we need to quantize the weights
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            // Helper to quantize a weight
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

            // Layer norms (kept in full precision)
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

    /// Load and quantize base model from safetensor files.
    pub fn load_and_quantize_from_dir(&mut self, model_dir: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();

        // Check for single file model
        let single_file = model_dir.join("model.safetensors");
        if single_file.exists() {
            let weights = Array::load_safetensors(&single_file)?;
            return self.load_and_quantize_weights(&weights);
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
            weight_map: HashMap<String, String>,
        }

        let index: WeightIndex = serde_json::from_str(&index_content)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        // Get unique shard files
        let shard_files: std::collections::HashSet<&String> = index.weight_map.values().collect();

        // Load each shard and combine weights
        let mut all_weights = HashMap::new();
        for shard_file in shard_files {
            let shard_path = model_dir.join(shard_file);
            let shard_weights = Array::load_safetensors(&shard_path)?;
            all_weights.extend(shard_weights);
        }

        self.load_and_quantize_weights(&all_weights)
    }
}

/// Create a causal attention mask with optional sliding window.
fn create_causal_mask(seq_len: i32, sliding_window: Option<i32>) -> Result<Array, Exception> {
    let mask = mlx_rs::ops::tri::<f32>(seq_len, None, None)?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);
    let mut causal_mask = mlx_rs::ops::r#where(&mask.eq(&zero)?, &neg_inf, &zero)?;

    // Apply sliding window if specified
    if let Some(window) = sliding_window {
        // Create lower triangular mask with window offset
        let window_mask = mlx_rs::ops::tri::<f32>(seq_len, Some(seq_len), Some(-window))?;
        let one = Array::from_f32(1.0);
        causal_mask = mlx_rs::ops::r#where(&window_mask.eq(&one)?, &neg_inf, &causal_mask)?;
    }

    Ok(causal_mask)
}

/// Implement ModuleParameters for MistralQloraForCausalLM.
impl ModuleParameters for MistralQloraForCausalLM {
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
            q_params.insert(Rc::from("lora_a"), NestedValue::Value(&layer.self_attn.q_proj.lora_a));
            q_params.insert(Rc::from("lora_b"), NestedValue::Value(&layer.self_attn.q_proj.lora_b));
            attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

            let mut k_params = HashMap::new();
            k_params.insert(Rc::from("lora_a"), NestedValue::Value(&layer.self_attn.k_proj.lora_a));
            k_params.insert(Rc::from("lora_b"), NestedValue::Value(&layer.self_attn.k_proj.lora_b));
            attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

            let mut v_params = HashMap::new();
            v_params.insert(Rc::from("lora_a"), NestedValue::Value(&layer.self_attn.v_proj.lora_a));
            v_params.insert(Rc::from("lora_b"), NestedValue::Value(&layer.self_attn.v_proj.lora_b));
            attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

            let mut o_params = HashMap::new();
            o_params.insert(Rc::from("lora_a"), NestedValue::Value(&layer.self_attn.o_proj.lora_a));
            o_params.insert(Rc::from("lora_b"), NestedValue::Value(&layer.self_attn.o_proj.lora_b));
            attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // MLP LoRA params
            let mut mlp_params = HashMap::new();
            let mut gate_params = HashMap::new();
            gate_params.insert(Rc::from("lora_a"), NestedValue::Value(&layer.mlp.gate_proj.lora_a));
            gate_params.insert(Rc::from("lora_b"), NestedValue::Value(&layer.mlp.gate_proj.lora_b));
            mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

            let mut up_params = HashMap::new();
            up_params.insert(Rc::from("lora_a"), NestedValue::Value(&layer.mlp.up_proj.lora_a));
            up_params.insert(Rc::from("lora_b"), NestedValue::Value(&layer.mlp.up_proj.lora_b));
            mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

            let mut down_params = HashMap::new();
            down_params.insert(Rc::from("lora_a"), NestedValue::Value(&layer.mlp.down_proj.lora_a));
            down_params.insert(Rc::from("lora_b"), NestedValue::Value(&layer.mlp.down_proj.lora_b));
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
            q_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.self_attn.q_proj.lora_a));
            q_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.self_attn.q_proj.lora_b));
            attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

            let mut k_params = HashMap::new();
            k_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.self_attn.k_proj.lora_a));
            k_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.self_attn.k_proj.lora_b));
            attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

            let mut v_params = HashMap::new();
            v_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.self_attn.v_proj.lora_a));
            v_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.self_attn.v_proj.lora_b));
            attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

            let mut o_params = HashMap::new();
            o_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.self_attn.o_proj.lora_a));
            o_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.self_attn.o_proj.lora_b));
            attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // MLP LoRA params
            let mut mlp_params = HashMap::new();
            let mut gate_params = HashMap::new();
            gate_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.mlp.gate_proj.lora_a));
            gate_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.mlp.gate_proj.lora_b));
            mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

            let mut up_params = HashMap::new();
            up_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.mlp.up_proj.lora_a));
            up_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.mlp.up_proj.lora_b));
            mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

            let mut down_params = HashMap::new();
            down_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.mlp.down_proj.lora_a));
            down_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.mlp.down_proj.lora_b));
            mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            params.insert(prefix, NestedValue::Map(layer_params));
        }

        params
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.parameters()
    }

    fn freeze_parameters(&mut self, _recursive: bool) {
        // LoRA params can't be frozen
    }

    fn unfreeze_parameters(&mut self, _recursive: bool) {
        // LoRA params are always unfrozen
    }

    fn all_frozen(&self) -> Option<bool> {
        Some(false)
    }

    fn any_frozen(&self) -> Option<bool> {
        Some(false)
    }
}

/// Implement TrainableModel for MistralQloraForCausalLM.
impl TrainableModel for MistralQloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        MistralQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        MistralQloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        MistralQloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        MistralQloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        MistralQloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        MistralQloraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        MistralQloraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        MistralQloraForCausalLM::disable_gradient_checkpointing(self)
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
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: None,
            max_position_embeddings: 128,
            sliding_window: Some(64),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
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
    fn test_mistral_qlora_attention() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut attn = MistralQLoraAttention::new(&config, &qlora_config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_mistral_qlora_model_forward() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = MistralQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 256]);
    }

    #[test]
    fn test_mistral_qlora_param_count() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = MistralQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        // Should have trainable params
        assert!(model.num_trainable_params() > 0);

        // Check parameter count
        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_mistral_qlora_memory_savings() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = MistralQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let savings = model.memory_savings();
        // QLoRA should provide significant memory savings (< 0.35 of full precision)
        assert!(savings < 0.35, "Expected significant memory savings, got {}", savings);
    }

    #[test]
    fn test_mistral_qlora_sliding_window() {
        let config = small_config();
        assert_eq!(config.sliding_window, Some(64));

        let qlora_config = small_qlora_config();
        let mut model = MistralQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        // Verify sliding window is passed through
        assert_eq!(model.model.layers[0].self_attn.sliding_window, Some(64));

        // Forward pass should work with sliding window
        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 8, 256]);
    }

    #[test]
    fn test_mistral_qlora_set_parameters() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = MistralQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        // Get initial parameters
        let params = model.lora_parameters();

        // Modify a parameter
        let mut new_params = params.clone();
        if let Some(key) = new_params.keys().next().cloned() {
            let new_val = mlx_rs::ops::ones::<f32>(&[4, 64]).unwrap();
            new_params.insert(key.clone(), new_val);
        }

        // Set modified parameters
        model.set_lora_parameters(&new_params);

        // Verify parameters were updated
        let updated_params = model.lora_parameters();
        assert_eq!(params.len(), updated_params.len());
    }
}
