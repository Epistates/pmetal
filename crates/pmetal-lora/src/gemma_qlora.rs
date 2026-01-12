//! QLoRA-enabled Gemma model architecture.
//!
//! Implements Gemma and Gemma2 with QLoRA (Quantized LoRA) for memory-efficient fine-tuning.
//! Base weights are stored in 4-bit NF4 format, reducing memory by ~87.5%.
//! LoRA adapters (A, B matrices) remain in full precision for training.
//!
//! Key Gemma differences from Llama:
//! - GemmaRMSNorm: output = x * (1 + weight) instead of x * weight
//! - GeGLU instead of SwiGLU (uses GELU instead of SiLU)
//! - Embedding scaling by sqrt(hidden_size)
//! - Gemma2: Extra pre/post feedforward normalization layers
//! - Gemma2: Attention logit softcapping

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param},
    nested::NestedValue,
    nn,
    Array,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_models::architectures::gemma::GemmaConfig;

use crate::{LoraError, QLoraConfig, QLoraLinear};

/// GELU activation with tanh approximation.
fn gelu_tanh(x: &Array) -> Result<Array, Exception> {
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let coef = Array::from_f32(0.044715);
    let half = Array::from_f32(0.5);
    let one = Array::from_f32(1.0);
    let two = Array::from_f32(2.0);
    let sqrt_2_pi = Array::from_f32(sqrt_2_over_pi);

    let x_cubed = x.multiply(x)?.multiply(x)?;
    let inner = x.add(&x_cubed.multiply(&coef)?)?;
    let inner = inner.multiply(&sqrt_2_pi)?;

    let exp_2x = inner.multiply(&two)?.exp()?;
    let tanh_val = exp_2x.subtract(&one)?.divide(&exp_2x.add(&one)?)?;

    let gate = one.add(&tanh_val)?.multiply(&half)?;
    x.multiply(&gate)
}

/// Gemma-style RMSNorm with +1 offset.
#[derive(Debug)]
pub struct GemmaQloraRmsNorm {
    /// Weight parameter.
    pub weight: Param<Array>,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl GemmaQloraRmsNorm {
    /// Create a new GemmaRmsNorm layer.
    pub fn new(hidden_size: i32, eps: f32) -> Result<Self, Exception> {
        let weight = mlx_rs::ops::zeros::<f32>(&[hidden_size])?;
        Ok(Self {
            weight: Param::new(weight),
            eps,
        })
    }

    /// Forward pass.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let x_sq = x.multiply(x)?;
        let mean_sq = x_sq.mean_axis(-1, Some(true))?;
        let eps_arr = Array::from_f32(self.eps);
        let rms = mean_sq.add(&eps_arr)?.sqrt()?;
        let normed = x.divide(&rms)?;

        // Apply weight with +1 offset: output = normed * (1 + weight)
        let one = Array::from_f32(1.0);
        let scale = self.weight.as_ref().add(&one)?;
        normed.multiply(&scale)
    }
}

/// QLoRA-enabled attention layer for Gemma.
///
/// Uses quantized base weights (NF4) with full-precision LoRA adapters.
#[derive(Debug)]
pub struct GemmaQLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// Attention logit softcapping (Gemma2).
    pub logit_softcapping: Option<f32>,

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

impl GemmaQLoraAttention {
    /// Create a new QLoRA attention layer with random weights.
    pub fn new(config: &GemmaConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = config.attention_scale();

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
            logit_softcapping: config.attn_logit_softcapping,
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

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries = queries
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let queries = Module::forward(&mut self.rope, &queries)?;
        let keys = Module::forward(&mut self.rope, &keys)?;

        // Expand KV heads for GQA
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
        let mut scores = queries.matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?;
        scores = scores.multiply(Array::from_f32(self.scale))?;

        // Apply logit softcapping (Gemma2)
        if let Some(cap) = self.logit_softcapping {
            let cap_val = Array::from_f32(cap);
            scores = scores.divide(&cap_val)?;
            scores = mlx_rs::ops::tanh(&scores)?;
            scores = scores.multiply(&cap_val)?;
        }

        let scores = if let Some(m) = mask {
            scores.add(m)?
        } else {
            scores
        };

        let weights = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
        let output = weights.matmul(&values)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    /// Get number of trainable parameters.
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

/// QLoRA-enabled MLP layer for Gemma (GeGLU).
#[derive(Debug)]
pub struct GemmaQloraMLP {
    /// Gate projection with QLoRA.
    pub gate_proj: QLoraLinear,
    /// Up projection with QLoRA.
    pub up_proj: QLoraLinear,
    /// Down projection with QLoRA.
    pub down_proj: QLoraLinear,
}

impl GemmaQloraMLP {
    /// Create a new QLoRA MLP layer.
    pub fn new(config: &GemmaConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
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

    /// Forward pass (GeGLU activation).
    pub fn forward(&self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = gelu_tanh(&gate)?;
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

/// QLoRA-enabled Gemma decoder layer.
#[derive(Debug)]
pub struct GemmaQloraDecoderLayer {
    /// Self-attention layer with QLoRA.
    pub self_attn: GemmaQLoraAttention,
    /// MLP layer with QLoRA.
    pub mlp: GemmaQloraMLP,
    /// Input layer norm.
    pub input_layernorm: GemmaQloraRmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: GemmaQloraRmsNorm,
}

impl GemmaQloraDecoderLayer {
    /// Create a new QLoRA decoder layer.
    pub fn new(config: &GemmaConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let self_attn = GemmaQLoraAttention::new(config, qlora_config)?;
        let mlp = GemmaQloraMLP::new(config, qlora_config)?;

        let input_layernorm = GemmaQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = GemmaQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        let h = x.add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
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

/// QLoRA-enabled Gemma2 decoder layer with extra normalization.
#[derive(Debug)]
pub struct Gemma2QloraDecoderLayer {
    /// Self-attention layer with QLoRA.
    pub self_attn: GemmaQLoraAttention,
    /// MLP layer with QLoRA.
    pub mlp: GemmaQloraMLP,
    /// Input layer norm.
    pub input_layernorm: GemmaQloraRmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: GemmaQloraRmsNorm,
    /// Pre-feedforward layer norm (Gemma2 specific).
    pub pre_feedforward_layernorm: GemmaQloraRmsNorm,
    /// Post-feedforward layer norm (Gemma2 specific).
    pub post_feedforward_layernorm: GemmaQloraRmsNorm,
}

impl Gemma2QloraDecoderLayer {
    /// Create a new Gemma2 QLoRA decoder layer.
    pub fn new(config: &GemmaConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let self_attn = GemmaQLoraAttention::new(config, qlora_config)?;
        let mlp = GemmaQloraMLP::new(config, qlora_config)?;

        let input_layernorm = GemmaQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = GemmaQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let pre_feedforward_layernorm = GemmaQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_feedforward_layernorm = GemmaQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    /// Forward pass with extra normalization (Gemma2 style).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        // Pre-norm + attention
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        // Post-attention norm before residual (Gemma2 specific)
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let h = x.add(&attn_out)?;

        // Pre-feedforward norm + MLP
        let normed = self.pre_feedforward_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        // Post-feedforward norm before residual (Gemma2 specific)
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
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

/// Container for Gemma QLoRA layers (supports both Gemma and Gemma2).
#[derive(Debug)]
pub enum GemmaQloraLayers {
    /// Gemma v1 layers.
    Gemma1(Vec<GemmaQloraDecoderLayer>),
    /// Gemma v2 layers with extra normalization.
    Gemma2(Vec<Gemma2QloraDecoderLayer>),
}

impl GemmaQloraLayers {
    /// Get the number of layers.
    pub fn len(&self) -> usize {
        match self {
            Self::Gemma1(layers) => layers.len(),
            Self::Gemma2(layers) => layers.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        match self {
            Self::Gemma1(layers) => layers.iter().map(|l| l.num_trainable_params()).sum(),
            Self::Gemma2(layers) => layers.iter().map(|l| l.num_trainable_params()).sum(),
        }
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let mut total_quant = 0;
        let mut total_lora = 0;
        let mut total = 0;

        match self {
            Self::Gemma1(layers) => {
                for layer in layers {
                    let (q, l, t) = layer.memory_usage();
                    total_quant += q;
                    total_lora += l;
                    total += t;
                }
            }
            Self::Gemma2(layers) => {
                for layer in layers {
                    let (q, l, t) = layer.memory_usage();
                    total_quant += q;
                    total_lora += l;
                    total += t;
                }
            }
        }

        (total_quant, total_lora, total)
    }
}

/// QLoRA-enabled Gemma model (without LM head).
#[derive(Debug)]
pub struct GemmaQloraModel {
    /// Configuration.
    pub config: GemmaConfig,
    /// QLoRA configuration.
    pub qlora_config: QLoraConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers with QLoRA.
    pub layers: GemmaQloraLayers,
    /// Final layer norm.
    pub norm: GemmaQloraRmsNorm,
    /// Embedding scale factor.
    pub embedding_scale: f32,
}

impl GemmaQloraModel {
    /// Create a new QLoRA Gemma model.
    pub fn new(config: GemmaConfig, qlora_config: QLoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let embedding_scale = config.embedding_scale();

        let layers = if config.is_gemma2 {
            GemmaQloraLayers::Gemma2(
                (0..config.num_hidden_layers)
                    .map(|_| Gemma2QloraDecoderLayer::new(&config, &qlora_config))
                    .collect::<Result<Vec<_>, _>>()?
            )
        } else {
            GemmaQloraLayers::Gemma1(
                (0..config.num_hidden_layers)
                    .map(|_| GemmaQloraDecoderLayer::new(&config, &qlora_config))
                    .collect::<Result<Vec<_>, _>>()?
            )
        };

        let norm = GemmaQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            qlora_config,
            embed_tokens,
            layers,
            norm,
            embedding_scale,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        // Get embeddings and scale
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;
        let scale = Array::from_f32(self.embedding_scale);
        hidden_states = hidden_states.multiply(&scale)?;

        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len)?)
        } else {
            mask.cloned()
        };

        match &mut self.layers {
            GemmaQloraLayers::Gemma1(layers) => {
                for layer in layers.iter_mut() {
                    hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
                }
            }
            GemmaQloraLayers::Gemma2(layers) => {
                for layer in layers.iter_mut() {
                    hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
                }
            }
        }

        Ok(self.norm.forward(&hidden_states)?)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.layers.num_trainable_params()
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.layers.memory_usage()
    }
}

/// QLoRA-enabled Gemma model with LM head.
///
/// Memory-efficient fine-tuning with 4-bit quantized base weights.
#[derive(Debug)]
pub struct GemmaQloraForCausalLM {
    /// Base model with QLoRA.
    pub model: GemmaQloraModel,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl GemmaQloraForCausalLM {
    /// Create a new QLoRA Gemma model.
    pub fn new(config: GemmaConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let qlora_config = QLoraConfig::from_lora(lora_config);
        Self::with_qlora_config(config, qlora_config)
    }

    /// Create with explicit QLoRA configuration.
    pub fn with_qlora_config(config: GemmaConfig, qlora_config: QLoraConfig) -> Result<Self, LoraError> {
        let model = GemmaQloraModel::new(config, qlora_config)?;

        Ok(Self {
            model,
            checkpoint_config: None,
        })
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

    /// Forward pass producing logits.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let hidden_states = self.model.forward(input_ids, mask)?;
        // Gemma always ties embeddings
        Ok(self.model.embed_tokens.as_linear(&hidden_states)?)
    }

    /// Get all trainable LoRA parameters.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();

        macro_rules! add_layer_params {
            ($layer:expr, $i:expr) => {
                let prefix = format!("model.layers.{}", $i);
                params.insert(Rc::from(format!("{}.self_attn.q_proj.lora_A.weight", prefix)), $layer.self_attn.q_proj.lora_a.clone());
                params.insert(Rc::from(format!("{}.self_attn.q_proj.lora_B.weight", prefix)), $layer.self_attn.q_proj.lora_b.clone());
                params.insert(Rc::from(format!("{}.self_attn.k_proj.lora_A.weight", prefix)), $layer.self_attn.k_proj.lora_a.clone());
                params.insert(Rc::from(format!("{}.self_attn.k_proj.lora_B.weight", prefix)), $layer.self_attn.k_proj.lora_b.clone());
                params.insert(Rc::from(format!("{}.self_attn.v_proj.lora_A.weight", prefix)), $layer.self_attn.v_proj.lora_a.clone());
                params.insert(Rc::from(format!("{}.self_attn.v_proj.lora_B.weight", prefix)), $layer.self_attn.v_proj.lora_b.clone());
                params.insert(Rc::from(format!("{}.self_attn.o_proj.lora_A.weight", prefix)), $layer.self_attn.o_proj.lora_a.clone());
                params.insert(Rc::from(format!("{}.self_attn.o_proj.lora_B.weight", prefix)), $layer.self_attn.o_proj.lora_b.clone());
                params.insert(Rc::from(format!("{}.mlp.gate_proj.lora_A.weight", prefix)), $layer.mlp.gate_proj.lora_a.clone());
                params.insert(Rc::from(format!("{}.mlp.gate_proj.lora_B.weight", prefix)), $layer.mlp.gate_proj.lora_b.clone());
                params.insert(Rc::from(format!("{}.mlp.up_proj.lora_A.weight", prefix)), $layer.mlp.up_proj.lora_a.clone());
                params.insert(Rc::from(format!("{}.mlp.up_proj.lora_B.weight", prefix)), $layer.mlp.up_proj.lora_b.clone());
                params.insert(Rc::from(format!("{}.mlp.down_proj.lora_A.weight", prefix)), $layer.mlp.down_proj.lora_a.clone());
                params.insert(Rc::from(format!("{}.mlp.down_proj.lora_B.weight", prefix)), $layer.mlp.down_proj.lora_b.clone());
            };
        }

        match &self.model.layers {
            GemmaQloraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter().enumerate() {
                    add_layer_params!(layer, i);
                }
            }
            GemmaQloraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter().enumerate() {
                    add_layer_params!(layer, i);
                }
            }
        }

        params
    }

    /// Set LoRA parameters.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        macro_rules! set_layer_params {
            ($layer:expr, $i:expr) => {
                let prefix = format!("model.layers.{}", $i);
                macro_rules! set_param {
                    ($param:expr, $key:expr) => {
                        if let Some(value) = params.get(&Rc::from($key)) {
                            $param = value.clone();
                        }
                    };
                }
                set_param!($layer.self_attn.q_proj.lora_a, format!("{}.self_attn.q_proj.lora_A.weight", prefix));
                set_param!($layer.self_attn.q_proj.lora_b, format!("{}.self_attn.q_proj.lora_B.weight", prefix));
                set_param!($layer.self_attn.k_proj.lora_a, format!("{}.self_attn.k_proj.lora_A.weight", prefix));
                set_param!($layer.self_attn.k_proj.lora_b, format!("{}.self_attn.k_proj.lora_B.weight", prefix));
                set_param!($layer.self_attn.v_proj.lora_a, format!("{}.self_attn.v_proj.lora_A.weight", prefix));
                set_param!($layer.self_attn.v_proj.lora_b, format!("{}.self_attn.v_proj.lora_B.weight", prefix));
                set_param!($layer.self_attn.o_proj.lora_a, format!("{}.self_attn.o_proj.lora_A.weight", prefix));
                set_param!($layer.self_attn.o_proj.lora_b, format!("{}.self_attn.o_proj.lora_B.weight", prefix));
                set_param!($layer.mlp.gate_proj.lora_a, format!("{}.mlp.gate_proj.lora_A.weight", prefix));
                set_param!($layer.mlp.gate_proj.lora_b, format!("{}.mlp.gate_proj.lora_B.weight", prefix));
                set_param!($layer.mlp.up_proj.lora_a, format!("{}.mlp.up_proj.lora_A.weight", prefix));
                set_param!($layer.mlp.up_proj.lora_b, format!("{}.mlp.up_proj.lora_B.weight", prefix));
                set_param!($layer.mlp.down_proj.lora_a, format!("{}.mlp.down_proj.lora_A.weight", prefix));
                set_param!($layer.mlp.down_proj.lora_b, format!("{}.mlp.down_proj.lora_B.weight", prefix));
            };
        }

        match &mut self.model.layers {
            GemmaQloraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    set_layer_params!(layer, i);
                }
            }
            GemmaQloraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    set_layer_params!(layer, i);
                }
            }
        }
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    /// Get memory savings compared to full-precision model.
    pub fn memory_savings(&self) -> f32 {
        let (quantized, lora, _) = self.memory_usage();

        // Calculate full precision memory
        let full_precision = match &self.model.layers {
            GemmaQloraLayers::Gemma1(layers) => {
                layers.iter().map(|l| {
                    l.self_attn.q_proj.num_frozen_params() * 4
                        + l.self_attn.k_proj.num_frozen_params() * 4
                        + l.self_attn.v_proj.num_frozen_params() * 4
                        + l.self_attn.o_proj.num_frozen_params() * 4
                        + l.mlp.gate_proj.num_frozen_params() * 4
                        + l.mlp.up_proj.num_frozen_params() * 4
                        + l.mlp.down_proj.num_frozen_params() * 4
                }).sum::<usize>()
            }
            GemmaQloraLayers::Gemma2(layers) => {
                layers.iter().map(|l| {
                    l.self_attn.q_proj.num_frozen_params() * 4
                        + l.self_attn.k_proj.num_frozen_params() * 4
                        + l.self_attn.v_proj.num_frozen_params() * 4
                        + l.self_attn.o_proj.num_frozen_params() * 4
                        + l.mlp.gate_proj.num_frozen_params() * 4
                        + l.mlp.up_proj.num_frozen_params() * 4
                        + l.mlp.down_proj.num_frozen_params() * 4
                }).sum::<usize>()
            }
        } + lora;

        (quantized + lora) as f32 / full_precision as f32
    }

    /// Get configuration.
    pub fn config(&self) -> &GemmaConfig {
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
        let params: HashMap<Rc<str>, Array> = loaded
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        self.set_lora_parameters(&params);
        Ok(())
    }

    /// Load and quantize base model weights.
    pub fn load_and_quantize_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        // Load embed_tokens (kept in full precision)
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        macro_rules! quantize_layer_weights {
            ($layer:expr, $prefix:expr, $qlora_config:expr) => {
                if let Some(w) = weights.get(&format!("{}.self_attn.q_proj.weight", $prefix)) {
                    $layer.self_attn.q_proj = QLoraLinear::from_weight(w, None, $qlora_config)?;
                }
                if let Some(w) = weights.get(&format!("{}.self_attn.k_proj.weight", $prefix)) {
                    $layer.self_attn.k_proj = QLoraLinear::from_weight(w, None, $qlora_config)?;
                }
                if let Some(w) = weights.get(&format!("{}.self_attn.v_proj.weight", $prefix)) {
                    $layer.self_attn.v_proj = QLoraLinear::from_weight(w, None, $qlora_config)?;
                }
                if let Some(w) = weights.get(&format!("{}.self_attn.o_proj.weight", $prefix)) {
                    $layer.self_attn.o_proj = QLoraLinear::from_weight(w, None, $qlora_config)?;
                }
                if let Some(w) = weights.get(&format!("{}.mlp.gate_proj.weight", $prefix)) {
                    $layer.mlp.gate_proj = QLoraLinear::from_weight(w, None, $qlora_config)?;
                }
                if let Some(w) = weights.get(&format!("{}.mlp.up_proj.weight", $prefix)) {
                    $layer.mlp.up_proj = QLoraLinear::from_weight(w, None, $qlora_config)?;
                }
                if let Some(w) = weights.get(&format!("{}.mlp.down_proj.weight", $prefix)) {
                    $layer.mlp.down_proj = QLoraLinear::from_weight(w, None, $qlora_config)?;
                }
                if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", $prefix)) {
                    $layer.input_layernorm.weight = Param::new(w.clone());
                }
                if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", $prefix)) {
                    $layer.post_attention_layernorm.weight = Param::new(w.clone());
                }
            };
        }

        match &mut self.model.layers {
            GemmaQloraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    let prefix = format!("model.layers.{}", i);
                    quantize_layer_weights!(layer, prefix, &self.model.qlora_config);
                }
            }
            GemmaQloraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    let prefix = format!("model.layers.{}", i);
                    quantize_layer_weights!(layer, prefix, &self.model.qlora_config);
                    // Gemma2 extra layernorms
                    if let Some(w) = weights.get(&format!("{}.pre_feedforward_layernorm.weight", prefix)) {
                        layer.pre_feedforward_layernorm.weight = Param::new(w.clone());
                    }
                    if let Some(w) = weights.get(&format!("{}.post_feedforward_layernorm.weight", prefix)) {
                        layer.post_feedforward_layernorm.weight = Param::new(w.clone());
                    }
                }
            }
        }

        // Load final norm
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }

        Ok(())
    }

    /// Load and quantize base model from safetensor files.
    pub fn load_and_quantize_from_dir(&mut self, model_dir: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();

        let single_file = model_dir.join("model.safetensors");
        if single_file.exists() {
            let weights = Array::load_safetensors(&single_file)?;
            return self.load_and_quantize_weights(&weights);
        }

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

        let shard_files: std::collections::HashSet<&String> = index.weight_map.values().collect();

        let mut all_weights = HashMap::new();
        for shard_file in shard_files {
            let shard_path = model_dir.join(shard_file);
            let shard_weights = Array::load_safetensors(&shard_path)?;
            all_weights.extend(shard_weights);
        }

        self.load_and_quantize_weights(&all_weights)
    }
}

fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    let mask = mlx_rs::ops::tri::<f32>(seq_len, None, None)?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);
    mlx_rs::ops::r#where(&mask.eq(&zero)?, &neg_inf, &zero)
}

/// Implement ModuleParameters for GemmaQloraForCausalLM.
impl ModuleParameters for GemmaQloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        macro_rules! build_layer_params_ref {
            ($layer:expr, $i:expr, $params:expr) => {{
                let prefix: Rc<str> = Rc::from(format!("layers.{}", $i));
                let mut layer_params = HashMap::new();

                let mut attn_params = HashMap::new();
                let mut q_params = HashMap::new();
                q_params.insert(Rc::from("lora_a"), NestedValue::Value(&$layer.self_attn.q_proj.lora_a));
                q_params.insert(Rc::from("lora_b"), NestedValue::Value(&$layer.self_attn.q_proj.lora_b));
                attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

                let mut k_params = HashMap::new();
                k_params.insert(Rc::from("lora_a"), NestedValue::Value(&$layer.self_attn.k_proj.lora_a));
                k_params.insert(Rc::from("lora_b"), NestedValue::Value(&$layer.self_attn.k_proj.lora_b));
                attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

                let mut v_params = HashMap::new();
                v_params.insert(Rc::from("lora_a"), NestedValue::Value(&$layer.self_attn.v_proj.lora_a));
                v_params.insert(Rc::from("lora_b"), NestedValue::Value(&$layer.self_attn.v_proj.lora_b));
                attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

                let mut o_params = HashMap::new();
                o_params.insert(Rc::from("lora_a"), NestedValue::Value(&$layer.self_attn.o_proj.lora_a));
                o_params.insert(Rc::from("lora_b"), NestedValue::Value(&$layer.self_attn.o_proj.lora_b));
                attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

                layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

                let mut mlp_params = HashMap::new();
                let mut gate_params = HashMap::new();
                gate_params.insert(Rc::from("lora_a"), NestedValue::Value(&$layer.mlp.gate_proj.lora_a));
                gate_params.insert(Rc::from("lora_b"), NestedValue::Value(&$layer.mlp.gate_proj.lora_b));
                mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

                let mut up_params = HashMap::new();
                up_params.insert(Rc::from("lora_a"), NestedValue::Value(&$layer.mlp.up_proj.lora_a));
                up_params.insert(Rc::from("lora_b"), NestedValue::Value(&$layer.mlp.up_proj.lora_b));
                mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

                let mut down_params = HashMap::new();
                down_params.insert(Rc::from("lora_a"), NestedValue::Value(&$layer.mlp.down_proj.lora_a));
                down_params.insert(Rc::from("lora_b"), NestedValue::Value(&$layer.mlp.down_proj.lora_b));
                mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

                layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));
                $params.insert(prefix, NestedValue::Map(layer_params));
            }};
        }

        match &self.model.layers {
            GemmaQloraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter().enumerate() {
                    build_layer_params_ref!(layer, i, params);
                }
            }
            GemmaQloraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter().enumerate() {
                    build_layer_params_ref!(layer, i, params);
                }
            }
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        macro_rules! build_layer_params_mut {
            ($layer:expr, $i:expr, $params:expr) => {{
                let prefix: Rc<str> = Rc::from(format!("layers.{}", $i));
                let mut layer_params = HashMap::new();

                let mut attn_params = HashMap::new();
                let mut q_params = HashMap::new();
                q_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut $layer.self_attn.q_proj.lora_a));
                q_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut $layer.self_attn.q_proj.lora_b));
                attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

                let mut k_params = HashMap::new();
                k_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut $layer.self_attn.k_proj.lora_a));
                k_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut $layer.self_attn.k_proj.lora_b));
                attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

                let mut v_params = HashMap::new();
                v_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut $layer.self_attn.v_proj.lora_a));
                v_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut $layer.self_attn.v_proj.lora_b));
                attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

                let mut o_params = HashMap::new();
                o_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut $layer.self_attn.o_proj.lora_a));
                o_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut $layer.self_attn.o_proj.lora_b));
                attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

                layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

                let mut mlp_params = HashMap::new();
                let mut gate_params = HashMap::new();
                gate_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut $layer.mlp.gate_proj.lora_a));
                gate_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut $layer.mlp.gate_proj.lora_b));
                mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

                let mut up_params = HashMap::new();
                up_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut $layer.mlp.up_proj.lora_a));
                up_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut $layer.mlp.up_proj.lora_b));
                mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

                let mut down_params = HashMap::new();
                down_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut $layer.mlp.down_proj.lora_a));
                down_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut $layer.mlp.down_proj.lora_b));
                mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

                layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));
                $params.insert(prefix, NestedValue::Map(layer_params));
            }};
        }

        match &mut self.model.layers {
            GemmaQloraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    build_layer_params_mut!(layer, i, params);
                }
            }
            GemmaQloraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    build_layer_params_mut!(layer, i, params);
                }
            }
        }

        params
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.parameters()
    }

    fn freeze_parameters(&mut self, _recursive: bool) {}
    fn unfreeze_parameters(&mut self, _recursive: bool) {}
    fn all_frozen(&self) -> Option<bool> { Some(false) }
    fn any_frozen(&self) -> Option<bool> { Some(false) }
}

/// Implement TrainableModel for GemmaQloraForCausalLM.
impl crate::TrainableModel for GemmaQloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        GemmaQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        GemmaQloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        GemmaQloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        GemmaQloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        GemmaQloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        GemmaQloraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        GemmaQloraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        GemmaQloraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(16),
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
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
    fn test_gemma_qlora_attention() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut attn = GemmaQLoraAttention::new(&config, &qlora_config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_gemma_qlora_model_forward() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = GemmaQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 256]);
    }

    #[test]
    fn test_gemma_qlora_param_count() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = GemmaQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        assert!(model.num_trainable_params() > 0);
        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_gemma_qlora_memory_savings() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = GemmaQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let savings = model.memory_savings();
        assert!(savings < 0.35, "Expected significant memory savings, got {}", savings);
    }

    fn small_gemma2_config() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(16),
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            is_gemma2: true,
            attn_logit_softcapping: Some(50.0),
            sliding_window: Some(64),
            ..Default::default()
        }
    }

    #[test]
    fn test_gemma2_qlora_model() {
        let config = small_gemma2_config();
        let qlora_config = small_qlora_config();
        let mut model = GemmaQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        // Verify Gemma2 layers were created
        match &model.model.layers {
            GemmaQloraLayers::Gemma2(layers) => {
                assert_eq!(layers.len(), 2);
            }
            _ => panic!("Expected Gemma2 layers"),
        }

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 256]);
    }

    #[test]
    fn test_gemma2_qlora_set_parameters() {
        let config = small_gemma2_config();
        let qlora_config = small_qlora_config();
        let mut model = GemmaQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let params = model.lora_parameters();
        let mut new_params = params.clone();
        if let Some(key) = new_params.keys().next().cloned() {
            let new_val = mlx_rs::ops::ones::<f32>(&[4, 64]).unwrap();
            new_params.insert(key.clone(), new_val);
        }

        model.set_lora_parameters(&new_params);
        let updated_params = model.lora_parameters();
        assert_eq!(params.len(), updated_params.len());
    }
}
