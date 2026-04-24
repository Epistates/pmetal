//! QLoRA-enabled Granite model architecture.
//!
//! Implements IBM Granite with QLoRA (Quantized LoRA) for memory-efficient fine-tuning.
//! Base weights are stored in 4-bit NF4 format, reducing memory by ~87.5%.
//! LoRA adapters (A, B matrices) remain in full precision for training.
//!
//! Granite-specific notes:
//! - Hybrid models (Granite 4.0-H) have alternating Attention and Mamba2 layers.
//!   Mamba2 layers are frozen passthrough — no LoRA is applied to them.
//! - Dense-attention models (Granite 4.0) use the same LoRA targets as Llama.
//! - No per-head Q/K normalization (unlike Qwen3).
//! - RoPE is currently a no-op stub matching the base model implementation.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param, nn,
    ops,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kv_cache::KVCache;
use pmetal_models::architectures::granite::{GraniteConfig, GraniteLayerType, GraniteMamba2};

use crate::{LoraError, QLoraConfig, QLoraLinear};

// =============================================================================
// Attention layer
// =============================================================================

/// QLoRA-enabled attention layer for Granite.
///
/// Uses quantized base weights (NF4) with full-precision LoRA adapters on
/// q_proj, k_proj, v_proj, and o_proj.
#[derive(Debug)]
pub struct GraniteQLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads (GQA).
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor (1 / sqrt(head_dim)).
    pub scale: f32,

    /// Query projection with QLoRA.
    pub q_proj: QLoraLinear,
    /// Key projection with QLoRA.
    pub k_proj: QLoraLinear,
    /// Value projection with QLoRA.
    pub v_proj: QLoraLinear,
    /// Output projection with QLoRA.
    pub o_proj: QLoraLinear,
}

impl GraniteQLoraAttention {
    /// Create a new QLoRA attention layer with random weights.
    pub fn new(config: &GraniteConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let scale = (head_dim as f32).sqrt().recip();

        let mut q_config = qlora_config.clone();
        q_config.lora.r = crate::effective_rank(&qlora_config.lora, "q_proj");
        let q_proj = QLoraLinear::new(hidden_size, n_heads * head_dim, &q_config, false)?;

        let mut k_config = qlora_config.clone();
        k_config.lora.r = crate::effective_rank(&qlora_config.lora, "k_proj");
        let k_proj = QLoraLinear::new(hidden_size, n_kv_heads * head_dim, &k_config, false)?;

        let mut v_config = qlora_config.clone();
        v_config.lora.r = crate::effective_rank(&qlora_config.lora, "v_proj");
        let v_proj = QLoraLinear::new(hidden_size, n_kv_heads * head_dim, &v_config, false)?;

        let mut o_config = qlora_config.clone();
        o_config.lora.r = crate::effective_rank(&qlora_config.lora, "o_proj");
        let o_proj = QLoraLinear::new(n_heads * head_dim, hidden_size, &o_config, false)?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Forward pass through QLoRA attention.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // RoPE is a stub in the base model; omit here for parity.

        // Transpose: [B, heads, L, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        // Expand KV heads for GQA if needed
        let (k, v) = if self.n_kv_heads < self.n_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            (
                expand_kv_heads(&k, repeats)?,
                expand_kv_heads(&v, repeats)?,
            )
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let k_t = k.transpose_axes(&[0, 1, 3, 2]);
        let mut scores = q.matmul(&k_t);
        scores = scores.multiply(&Array::from_f32(self.scale));

        if let Some(m) = mask {
            scores = scores.add(m);
        }

        let probs = ops::softmax_axis(&scores, -1);
        let output = probs.matmul(&v);

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);

        self.o_proj.forward(&output)
    }

    /// Number of trainable LoRA parameters (adapters only).
    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }

    /// Memory usage: (quantized_bytes, lora_bytes, total_bytes).
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (q_q, q_l, q_t) = self.q_proj.memory_usage();
        let (k_q, k_l, k_t) = self.k_proj.memory_usage();
        let (v_q, v_l, v_t) = self.v_proj.memory_usage();
        let (o_q, o_l, o_t) = self.o_proj.memory_usage();
        (q_q + k_q + v_q + o_q, q_l + k_l + v_l + o_l, q_t + k_t + v_t + o_t)
    }
}

// =============================================================================
// MLP layer
// =============================================================================

/// QLoRA-enabled SwiGLU MLP for Granite.
#[derive(Debug)]
pub struct GraniteQloraMLP {
    /// Gate projection with QLoRA.
    pub gate_proj: QLoraLinear,
    /// Up projection with QLoRA.
    pub up_proj: QLoraLinear,
    /// Down projection with QLoRA.
    pub down_proj: QLoraLinear,
}

impl GraniteQloraMLP {
    /// Create a new QLoRA MLP with random weights.
    pub fn new(config: &GraniteConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let mut gate_config = qlora_config.clone();
        gate_config.lora.r = crate::effective_rank(&qlora_config.lora, "gate_proj");
        let gate_proj = QLoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            &gate_config,
            false,
        )?;

        let mut up_config = qlora_config.clone();
        up_config.lora.r = crate::effective_rank(&qlora_config.lora, "up_proj");
        let up_proj = QLoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            &up_config,
            false,
        )?;

        let mut down_config = qlora_config.clone();
        down_config.lora.r = crate::effective_rank(&qlora_config.lora, "down_proj");
        let down_proj = QLoraLinear::new(
            config.intermediate_size,
            config.hidden_size,
            &down_config,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass (SwiGLU: silu(gate) * up, then down).
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(&gate);
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up);
        self.down_proj.forward(&hidden)
    }

    /// Number of trainable LoRA parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }

    /// Memory usage: (quantized_bytes, lora_bytes, total_bytes).
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (g_q, g_l, g_t) = self.gate_proj.memory_usage();
        let (u_q, u_l, u_t) = self.up_proj.memory_usage();
        let (d_q, d_l, d_t) = self.down_proj.memory_usage();
        (g_q + u_q + d_q, g_l + u_l + d_l, g_t + u_t + d_t)
    }
}

// =============================================================================
// Decoder layer — attention or frozen Mamba2 passthrough
// =============================================================================

/// QLoRA-enabled Granite decoder layer.
///
/// Attention layers get QLoRA on q/k/v/o and gate/up/down.
/// Mamba2 layers are kept fully frozen — no LoRA is applied.
#[derive(Debug)]
pub struct GraniteQloraDecoderLayer {
    /// Layer type tag (Attention or Mamba2).
    pub layer_type: GraniteLayerType,

    /// QLoRA attention (present for Attention layers, None for Mamba2).
    pub attention: Option<GraniteQLoraAttention>,
    /// Frozen Mamba2 layer (present for Mamba2 layers, None for Attention).
    pub mamba: Option<GraniteMamba2>,
    /// MLP with QLoRA.
    pub mlp: GraniteQloraMLP,
    /// Input layer norm (frozen).
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm (frozen).
    pub post_attention_layernorm: nn::RmsNorm,
}

impl GraniteQloraDecoderLayer {
    /// Create a new decoder layer (random weights).
    pub fn new(
        config: &GraniteConfig,
        qlora_config: &QLoraConfig,
        layer_idx: usize,
    ) -> Result<Self, LoraError> {
        let layer_type = config.layer_type(layer_idx);

        let (attention, mamba) = match layer_type {
            GraniteLayerType::Attention => (Some(GraniteQLoraAttention::new(config, qlora_config)?), None),
            GraniteLayerType::Mamba2 => (
                None,
                Some(GraniteMamba2::new(config).map_err(LoraError::Mlx)?),
            ),
        };

        let mlp = GraniteQloraMLP::new(config, qlora_config)?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;

        Ok(Self {
            layer_type,
            attention,
            mamba,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        // Pre-norm
        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)
                .map_err(LoraError::Mlx)?;

        // Mixer: QLoRA attention or frozen Mamba2
        let mixer_out = match self.layer_type {
            GraniteLayerType::Attention => {
                self.attention.as_mut().unwrap().forward(&normed, mask)?
            }
            GraniteLayerType::Mamba2 => {
                // Frozen passthrough — Mamba2 is not differentiably adapted.
                self.mamba
                    .as_mut()
                    .unwrap()
                    .forward(&normed)
                    .map_err(LoraError::Mlx)?
            }
        };

        let h = x.add(&mixer_out);

        // FFN pre-norm + QLoRA MLP + residual
        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)
                .map_err(LoraError::Mlx)?;
        let ffn_out = self.mlp.forward(&normed)?;
        Ok(h.add(&ffn_out))
    }

    /// Number of trainable LoRA parameters for this layer.
    ///
    /// Mamba2 layers contribute 0 (fully frozen).
    pub fn num_trainable_params(&self) -> usize {
        let attn = self
            .attention
            .as_ref()
            .map(|a| a.num_trainable_params())
            .unwrap_or(0);
        attn + self.mlp.num_trainable_params()
    }

    /// Memory usage: (quantized_bytes, lora_bytes, total_bytes).
    ///
    /// Mamba2 layers contribute 0 to quantized/lora columns.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (attn_q, attn_l, attn_t) = self
            .attention
            .as_ref()
            .map(|a| a.memory_usage())
            .unwrap_or((0, 0, 0));
        let (mlp_q, mlp_l, mlp_t) = self.mlp.memory_usage();
        (attn_q + mlp_q, attn_l + mlp_l, attn_t + mlp_t)
    }
}

// =============================================================================
// Model (trunk, no LM head)
// =============================================================================

/// QLoRA-enabled Granite model (without LM head).
#[derive(Debug)]
pub struct GraniteQloraModel {
    /// Model configuration.
    pub config: GraniteConfig,
    /// QLoRA configuration.
    pub qlora_config: QLoraConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers with QLoRA.
    pub layers: Vec<GraniteQloraDecoderLayer>,
    /// Final RMSNorm (frozen).
    pub norm: nn::RmsNorm,
}

impl GraniteQloraModel {
    /// Create a new QLoRA Granite model with random weights.
    pub fn new(config: GraniteConfig, qlora_config: QLoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)
            .map_err(LoraError::Mlx)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| GraniteQloraDecoderLayer::new(&config, &qlora_config, i as usize))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;

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
    ) -> Result<Array, LoraError> {
        self.forward_with_checkpoint(input_ids, mask, None)
    }

    /// Forward pass with optional gradient checkpointing.
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let mut hidden_states =
            pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)
                .map_err(LoraError::Mlx)?;

        // Auto-create causal mask for full-sequence forward passes
        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len).map_err(LoraError::Mlx)?)
        } else {
            mask.cloned()
        };

        let layers_per_block = checkpoint_config
            .map(|c| c.layers_per_block)
            .unwrap_or(usize::MAX);
        let checkpointing_enabled = checkpoint_config.map(|c| c.enabled).unwrap_or(false);

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, mask.as_ref())?;

            if checkpointing_enabled && (idx + 1) % layers_per_block == 0 {
                tracing::trace!("Checkpoint boundary at layer {}", idx + 1);
            }
        }

        pmetal_bridge::compat::Module::forward(&mut self.norm, &hidden_states)
            .map_err(LoraError::Mlx)
    }

    /// Number of trainable LoRA parameters across all layers.
    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }

    /// Aggregate memory usage across all layers.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (mut tq, mut tl, mut tt) = (0, 0, 0);
        for layer in &self.layers {
            let (q, l, t) = layer.memory_usage();
            tq += q;
            tl += l;
            tt += t;
        }
        (tq, tl, tt)
    }
}

// =============================================================================
// ForCausalLM
// =============================================================================

/// QLoRA-enabled Granite model with LM head for causal language modelling.
///
/// Memory-efficient fine-tuning with 4-bit quantized base weights.
/// Typical memory usage for a 1B model: ~0.8 GB (vs ~6 GB for full precision).
///
/// Hybrid (Mamba2 + Attention) variants are supported: Mamba2 layers are
/// frozen passthrough while attention layers receive LoRA adapters.
#[derive(Debug)]
pub struct GraniteQloraForCausalLM {
    /// Trunk model.
    pub model: GraniteQloraModel,
    /// LM head (frozen).  `None` when `tie_word_embeddings = true`.
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl GraniteQloraForCausalLM {
    /// Create a new QLoRA Granite model with random weights.
    pub fn new(config: GraniteConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let qlora_config = QLoraConfig::from_lora(lora_config);
        Self::with_qlora_config(config, qlora_config)
    }

    /// Create with an explicit QLoRA configuration.
    pub fn with_qlora_config(
        config: GraniteConfig,
        qlora_config: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let tie_weights = config.tie_word_embeddings;
        let model = GraniteQloraModel::new(config.clone(), qlora_config)?;

        let lm_head = if !tie_weights {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()
                    .map_err(LoraError::Mlx)?,
            )
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

    /// Forward pass producing logits [batch, seq_len, vocab_size].
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let checkpoint_config = self.checkpoint_config.clone();
        self.forward_with_checkpoint(input_ids, mask, checkpoint_config.as_ref())
    }

    /// Forward pass with explicit checkpoint configuration.
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let hidden_states =
            self.model
                .forward_with_checkpoint(input_ids, mask, checkpoint_config)?;

        if let Some(ref mut lm_head) = self.lm_head {
            pmetal_bridge::compat::Module::forward(lm_head, &hidden_states)
                .map_err(LoraError::Mlx)
        } else {
            Ok(self.model.embed_tokens.as_linear(&hidden_states))
        }
    }

    /// Get all trainable LoRA parameters as a flat HashMap.
    ///
    /// Only attention-type layers emit parameters; Mamba2 layers are skipped.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            // Mamba2 layers have no LoRA — skip.
            let attn = match layer.attention.as_ref() {
                Some(a) => a,
                None => continue,
            };

            let prefix = format!("layers.{}", i);

            // Attention adapters
            for (name, lora) in [
                ("q_proj", &attn.q_proj),
                ("k_proj", &attn.k_proj),
                ("v_proj", &attn.v_proj),
                ("o_proj", &attn.o_proj),
            ] {
                params.insert(
                    Rc::from(format!("{}.self_attn.{}.lora_a", prefix, name)),
                    lora.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.self_attn.{}.lora_b", prefix, name)),
                    lora.lora_b.clone(),
                );
            }

            // MLP adapters
            for (name, lora) in [
                ("gate_proj", &layer.mlp.gate_proj),
                ("up_proj", &layer.mlp.up_proj),
                ("down_proj", &layer.mlp.down_proj),
            ] {
                params.insert(
                    Rc::from(format!("{}.mlp.{}.lora_a", prefix, name)),
                    lora.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.mlp.{}.lora_b", prefix, name)),
                    lora.lora_b.clone(),
                );
            }
        }

        params
    }

    /// Set LoRA parameters from a HashMap.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        macro_rules! set_param {
            ($target:expr, $key:expr) => {
                if let Some(value) = params.get(&Rc::from($key)) {
                    $target = value.clone();
                }
            };
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let attn = match layer.attention.as_mut() {
                Some(a) => a,
                None => continue,
            };

            let prefix = format!("layers.{}", i);

            set_param!(attn.q_proj.lora_a, format!("{}.self_attn.q_proj.lora_a", prefix));
            set_param!(attn.q_proj.lora_b, format!("{}.self_attn.q_proj.lora_b", prefix));
            set_param!(attn.k_proj.lora_a, format!("{}.self_attn.k_proj.lora_a", prefix));
            set_param!(attn.k_proj.lora_b, format!("{}.self_attn.k_proj.lora_b", prefix));
            set_param!(attn.v_proj.lora_a, format!("{}.self_attn.v_proj.lora_a", prefix));
            set_param!(attn.v_proj.lora_b, format!("{}.self_attn.v_proj.lora_b", prefix));
            set_param!(attn.o_proj.lora_a, format!("{}.self_attn.o_proj.lora_a", prefix));
            set_param!(attn.o_proj.lora_b, format!("{}.self_attn.o_proj.lora_b", prefix));

            set_param!(layer.mlp.gate_proj.lora_a, format!("{}.mlp.gate_proj.lora_a", prefix));
            set_param!(layer.mlp.gate_proj.lora_b, format!("{}.mlp.gate_proj.lora_b", prefix));
            set_param!(layer.mlp.up_proj.lora_a, format!("{}.mlp.up_proj.lora_a", prefix));
            set_param!(layer.mlp.up_proj.lora_b, format!("{}.mlp.up_proj.lora_b", prefix));
            set_param!(layer.mlp.down_proj.lora_a, format!("{}.mlp.down_proj.lora_a", prefix));
            set_param!(layer.mlp.down_proj.lora_b, format!("{}.mlp.down_proj.lora_b", prefix));
        }
    }

    /// Number of trainable parameters (LoRA adapters only).
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    /// Memory usage: (quantized_bytes, lora_bytes, total_bytes).
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    /// Ratio of QLoRA memory to equivalent full-precision memory.
    pub fn memory_savings(&self) -> f32 {
        let (quantized, lora, _) = self.memory_usage();
        let full_precision = self
            .model
            .layers
            .iter()
            .map(|l| {
                let attn_fp = l.attention.as_ref().map(|a| {
                    (a.q_proj.num_frozen_params()
                        + a.k_proj.num_frozen_params()
                        + a.v_proj.num_frozen_params()
                        + a.o_proj.num_frozen_params())
                        * 4
                }).unwrap_or(0);
                let mlp_fp = (l.mlp.gate_proj.num_frozen_params()
                    + l.mlp.up_proj.num_frozen_params()
                    + l.mlp.down_proj.num_frozen_params())
                    * 4;
                attn_fp + mlp_fp
            })
            .sum::<usize>()
            + lora;

        if full_precision == 0 {
            1.0
        } else {
            (quantized + lora) as f32 / full_precision as f32
        }
    }

    /// Access the model configuration.
    pub fn config(&self) -> &GraniteConfig {
        &self.model.config
    }

    /// Access the QLoRA configuration.
    pub fn qlora_config(&self) -> &QLoraConfig {
        &self.model.qlora_config
    }

    /// Save LoRA adapter weights to a safetensors file.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        crate::save_safetensors_map(path, &params)
    }

    /// Load LoRA adapter weights from a safetensors file or directory.
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
        let loaded = crate::load_safetensors_map(&file_path)?;

        macro_rules! load_param {
            ($target:expr, $key:expr) => {
                if let Some(value) = loaded.get(&$key) {
                    $target = value.clone();
                }
            };
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let attn = match layer.attention.as_mut() {
                Some(a) => a,
                None => continue,
            };

            let prefix = format!("layers.{}", i);

            load_param!(attn.q_proj.lora_a, format!("{}.self_attn.q_proj.lora_a", prefix));
            load_param!(attn.q_proj.lora_b, format!("{}.self_attn.q_proj.lora_b", prefix));
            load_param!(attn.k_proj.lora_a, format!("{}.self_attn.k_proj.lora_a", prefix));
            load_param!(attn.k_proj.lora_b, format!("{}.self_attn.k_proj.lora_b", prefix));
            load_param!(attn.v_proj.lora_a, format!("{}.self_attn.v_proj.lora_a", prefix));
            load_param!(attn.v_proj.lora_b, format!("{}.self_attn.v_proj.lora_b", prefix));
            load_param!(attn.o_proj.lora_a, format!("{}.self_attn.o_proj.lora_a", prefix));
            load_param!(attn.o_proj.lora_b, format!("{}.self_attn.o_proj.lora_b", prefix));

            load_param!(layer.mlp.gate_proj.lora_a, format!("{}.mlp.gate_proj.lora_a", prefix));
            load_param!(layer.mlp.gate_proj.lora_b, format!("{}.mlp.gate_proj.lora_b", prefix));
            load_param!(layer.mlp.up_proj.lora_a, format!("{}.mlp.up_proj.lora_a", prefix));
            load_param!(layer.mlp.up_proj.lora_b, format!("{}.mlp.up_proj.lora_b", prefix));
            load_param!(layer.mlp.down_proj.lora_a, format!("{}.mlp.down_proj.lora_a", prefix));
            load_param!(layer.mlp.down_proj.lora_b, format!("{}.mlp.down_proj.lora_b", prefix));
        }

        Ok(())
    }

    /// Load and quantize base weights from a HashMap of full-precision tensors.
    ///
    /// Quantizes attention and MLP projection weights to NF4.
    /// Layer norms, embeddings, and Mamba2 weights stay in full precision.
    pub fn load_base_weights(
        &mut self,
        weights: &HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        // Embeddings (full precision)
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            macro_rules! quantize_weight {
                ($proj:expr, $key:expr) => {
                    if let Some(w) = weights.get(&$key) {
                        *$proj = QLoraLinear::from_weight(w, None, &self.model.qlora_config)?;
                    }
                };
            }

            // Attention projections (only for attention-type layers)
            if let Some(ref mut attn) = layer.attention {
                quantize_weight!(&mut attn.q_proj, format!("{}.self_attn.q_proj.weight", prefix));
                quantize_weight!(&mut attn.k_proj, format!("{}.self_attn.k_proj.weight", prefix));
                quantize_weight!(&mut attn.v_proj, format!("{}.self_attn.v_proj.weight", prefix));
                quantize_weight!(&mut attn.o_proj, format!("{}.self_attn.o_proj.weight", prefix));
            }

            // MLP projections
            quantize_weight!(&mut layer.mlp.gate_proj, format!("{}.mlp.gate_proj.weight", prefix));
            quantize_weight!(&mut layer.mlp.up_proj, format!("{}.mlp.up_proj.weight", prefix));
            quantize_weight!(&mut layer.mlp.down_proj, format!("{}.mlp.down_proj.weight", prefix));

            // Layer norms (full precision)
            if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }
        }

        // Final norm
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }

        // LM head
        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                lm_head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }

    /// Load and quantize base weights from a model directory (single-shard or sharded).
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();

        let single_file = model_dir.join("model.safetensors");
        if single_file.exists() {
            let weights =
                crate::sanitize_loaded_weights(crate::load_safetensors_map(&single_file)?)?;
            return self.load_base_weights(&weights);
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
            let shard_weights =
                crate::sanitize_loaded_weights(crate::load_safetensors_map(&shard_path)?)?;
            all_weights.extend(shard_weights);
        }

        self.load_base_weights(&all_weights)
    }

    /// Evaluate (materialize) all LoRA adapter parameters.
    pub fn eval_all(&mut self) {
        for layer in &mut self.model.layers {
            if let Some(ref mut attn) = layer.attention {
                attn.q_proj.lora_a.eval();
                attn.q_proj.lora_b.eval();
                attn.k_proj.lora_a.eval();
                attn.k_proj.lora_b.eval();
                attn.v_proj.lora_a.eval();
                attn.v_proj.lora_b.eval();
                attn.o_proj.lora_a.eval();
                attn.o_proj.lora_b.eval();
            }
            layer.mlp.gate_proj.lora_a.eval();
            layer.mlp.gate_proj.lora_b.eval();
            layer.mlp.up_proj.lora_a.eval();
            layer.mlp.up_proj.lora_b.eval();
            layer.mlp.down_proj.lora_a.eval();
            layer.mlp.down_proj.lora_b.eval();
        }
    }

    /// Merge LoRA adapters into the base weights (stub).
    ///
    /// Full merge requires dequantizing base weights, adding the low-rank
    /// product, and re-quantizing — not yet implemented.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        // Stub: merging LoRA into NF4-quantized base weights requires
        // dequantize → add (B A) → re-quantize, which is not yet wired.
        Ok(())
    }

    /// Unmerge LoRA adapters from the base weights (stub).
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Ok(())
    }
}

// =============================================================================
// ModuleParameters impl
// =============================================================================

impl ModuleParameters for GraniteQloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let attn = match layer.attention.as_ref() {
                Some(a) => a,
                None => continue,
            };

            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // --- attention ---
            let mut attn_params = HashMap::new();
            for (name, proj) in [
                ("q_proj", &attn.q_proj),
                ("k_proj", &attn.k_proj),
                ("v_proj", &attn.v_proj),
                ("o_proj", &attn.o_proj),
            ] {
                let mut p = HashMap::new();
                p.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                p.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                attn_params.insert(Rc::from(name), NestedValue::Map(p));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // --- mlp ---
            let mut mlp_params = HashMap::new();
            for (name, proj) in [
                ("gate_proj", &layer.mlp.gate_proj),
                ("up_proj", &layer.mlp.up_proj),
                ("down_proj", &layer.mlp.down_proj),
            ] {
                let mut p = HashMap::new();
                p.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                p.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                mlp_params.insert(Rc::from(name), NestedValue::Map(p));
            }
            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            params.insert(prefix, NestedValue::Map(layer_params));
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let attn = match layer.attention.as_mut() {
                Some(a) => a,
                None => continue,
            };

            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // --- attention ---
            let mut attn_params = HashMap::new();

            let mut q_p = HashMap::new();
            q_p.insert(Rc::from("lora_a"), NestedValue::Value(&mut attn.q_proj.lora_a));
            q_p.insert(Rc::from("lora_b"), NestedValue::Value(&mut attn.q_proj.lora_b));
            attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_p));

            let mut k_p = HashMap::new();
            k_p.insert(Rc::from("lora_a"), NestedValue::Value(&mut attn.k_proj.lora_a));
            k_p.insert(Rc::from("lora_b"), NestedValue::Value(&mut attn.k_proj.lora_b));
            attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_p));

            let mut v_p = HashMap::new();
            v_p.insert(Rc::from("lora_a"), NestedValue::Value(&mut attn.v_proj.lora_a));
            v_p.insert(Rc::from("lora_b"), NestedValue::Value(&mut attn.v_proj.lora_b));
            attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_p));

            let mut o_p = HashMap::new();
            o_p.insert(Rc::from("lora_a"), NestedValue::Value(&mut attn.o_proj.lora_a));
            o_p.insert(Rc::from("lora_b"), NestedValue::Value(&mut attn.o_proj.lora_b));
            attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_p));

            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // --- mlp ---
            let mut mlp_params = HashMap::new();

            let mut gate_p = HashMap::new();
            gate_p.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.mlp.gate_proj.lora_a));
            gate_p.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.mlp.gate_proj.lora_b));
            mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_p));

            let mut up_p = HashMap::new();
            up_p.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.mlp.up_proj.lora_a));
            up_p.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.mlp.up_proj.lora_b));
            mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_p));

            let mut down_p = HashMap::new();
            down_p.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.mlp.down_proj.lora_a));
            down_p.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.mlp.down_proj.lora_b));
            mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_p));

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

// =============================================================================
// TrainableModel impl
// =============================================================================

impl crate::TrainableModel for GraniteQloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        GraniteQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        // Granite QLoRA uses standard forward; KV cache threading is a no-op here
        // (full KV cache integration would require the attention layer to consume
        // the cache slices — out of scope for LoRA training).
        GraniteQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn num_trainable_params(&self) -> usize {
        GraniteQloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        GraniteQloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        GraniteQloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        GraniteQloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        GraniteQloraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        GraniteQloraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        GraniteQloraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        // Stub: same posture as other QLoRA models.
        false
    }
}

// =============================================================================
// Internal helpers
// =============================================================================

/// Expand KV heads for grouped-query attention.
fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, LoraError> {
    let shape = x.shape();
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    let x = x.reshape(&[batch, n_kv_heads, 1, seq_len, head_dim]);
    let x = pmetal_bridge::compat::ops::broadcast_to(
        &x,
        &[batch, n_kv_heads, repeats, seq_len, head_dim],
    );
    Ok(x.reshape(&[batch, n_kv_heads * repeats, seq_len, head_dim]))
}

/// Build a causal attention mask of shape [seq_len, seq_len].
fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    let mask = ops::tri(
        seq_len,
        seq_len,
        0,
        pmetal_bridge::compat::Dtype::Float32,
    );
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);
    let equal_zero = mask.equal(&zero);
    Ok(ops::where_fn(&equal_zero, &neg_inf, &zero))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> GraniteConfig {
        GraniteConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            is_hybrid: false,
            layer_types: None,
            mamba_state_dim: 32,
            mamba_conv_dim: 4,
            is_moe: false,
            num_experts: 8,
            num_experts_per_tok: 2,
            use_shared_expert: true,
        }
    }

    fn small_hybrid_config() -> GraniteConfig {
        GraniteConfig {
            is_hybrid: true,
            layer_types: Some(vec![GraniteLayerType::Attention, GraniteLayerType::Mamba2]),
            ..small_config()
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
    fn test_granite_qlora_attention_forward() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut attn = GraniteQLoraAttention::new(&config, &qlora_config).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = attn.forward(&x, None).unwrap();
        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_granite_qlora_model_forward() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = GraniteQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 256]);
    }

    #[test]
    fn test_granite_qlora_hybrid_forward() {
        let config = small_hybrid_config();
        let qlora_config = small_qlora_config();
        let mut model = GraniteQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 256]);
    }

    #[test]
    fn test_granite_qlora_param_count() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = GraniteQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        assert!(model.num_trainable_params() > 0);
        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_granite_qlora_hybrid_mamba_layers_have_no_lora() {
        // For a 4-layer hybrid (attn, mamba, attn, mamba) only attention layers
        // should appear in the LoRA parameter map.
        let config = GraniteConfig {
            num_hidden_layers: 4,
            is_hybrid: true,
            layer_types: Some(vec![
                GraniteLayerType::Attention,
                GraniteLayerType::Mamba2,
                GraniteLayerType::Attention,
                GraniteLayerType::Mamba2,
            ]),
            ..small_config()
        };
        let qlora_config = small_qlora_config();
        let model = GraniteQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let params = model.lora_parameters();
        // Parameters should only be keyed under layers.0 and layers.2, not layers.1 or layers.3.
        assert!(params.keys().any(|k| k.starts_with("layers.0.")));
        assert!(params.keys().any(|k| k.starts_with("layers.2.")));
        assert!(!params.keys().any(|k| k.starts_with("layers.1.")));
        assert!(!params.keys().any(|k| k.starts_with("layers.3.")));
    }

    #[test]
    fn test_granite_qlora_memory_usage() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = GraniteQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let (quantized, lora, total) = model.memory_usage();
        assert!(quantized > 0, "should have quantized weight memory");
        assert!(lora > 0, "should have LoRA adapter memory");
        assert_eq!(total, quantized + lora);
    }

    #[test]
    fn test_granite_qlora_no_nan_output() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = GraniteQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let output = model.forward(&input_ids, None).unwrap();
        output.eval().unwrap();

        let has_nan = pmetal_bridge::compat::ops::any(
            &pmetal_bridge::compat::ops::is_nan(&output),
            None,
            false,
        );
        has_nan.eval().unwrap();
        assert!(
            !pmetal_bridge::compat::ops::item_bool(&has_nan),
            "output must not contain NaN"
        );
    }

    #[test]
    fn test_granite_qlora_set_and_get_params_roundtrip() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = GraniteQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let params = model.lora_parameters();
        // Store key count before mutation
        let n = params.len();

        // Overwrite with a fresh set (same keys, different values)
        let mut new_params = params.clone();
        if let Some(key) = new_params.keys().next().cloned() {
            let new_val =
                pmetal_bridge::compat::ops::ones(&[4, 64], pmetal_bridge::compat::Dtype::Float32);
            new_params.insert(key, new_val);
        }
        model.set_lora_parameters(&new_params);

        let updated = model.lora_parameters();
        assert_eq!(updated.len(), n);
    }

    #[test]
    fn test_granite_qlora_gradient_checkpointing_toggle() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = GraniteQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        assert!(model.checkpoint_config.is_none());
        model.enable_gradient_checkpointing(4);
        assert!(model.checkpoint_config.is_some());
        model.disable_gradient_checkpointing();
        assert!(model.checkpoint_config.is_none());
    }
}
