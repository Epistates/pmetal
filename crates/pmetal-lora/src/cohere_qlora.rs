//! QLoRA-enabled Cohere (Command R / R+ / A) model architecture.
//!
//! Implements Cohere with QLoRA (Quantized LoRA) for memory-efficient fine-tuning.
//! Base weights are stored in 4-bit NF4 format, reducing memory by ~87.5%.
//! LoRA adapters (A, B matrices) remain in full precision for training.
//!
//! Cohere architectural distinctives vs. Llama/Mistral:
//! - LayerNorm (not RmsNorm) with both weight AND bias in attention input norm and
//!   final model norm.
//! - Parallel residual: `x + attn(normed) + ffn(normed)` — both branches share the
//!   same pre-norm output rather than applying sequentially.
//! - Per-layer sliding-window attention (every 4th layer uses global attention).
//! - Standard SwiGLU MLP (gate/up/down, no bias).
//! - Configurable `tie_word_embeddings`.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param,
    nn, ops,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kv_cache::KVCache;
use pmetal_models::architectures::cohere::CohereConfig;
use pmetal_models::architectures::utils::create_causal_mask;

use crate::{LoraError, QLoraConfig, QLoraLinear, TrainableModel};

// =============================================================================
// Attention
// =============================================================================

/// QLoRA-enabled attention layer for Cohere.
///
/// Uses quantized NF4 base weights with full-precision LoRA adapters.
/// Mirrors the RoPE-before-transpose layout from the base `CohereAttention`.
#[derive(Debug)]
pub struct CohereQloraAttention {
    pub layer_idx: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
    pub use_sliding_window: bool,
    pub sliding_window: i32,

    pub q_proj: QLoraLinear,
    pub k_proj: QLoraLinear,
    pub v_proj: QLoraLinear,
    pub o_proj: QLoraLinear,
}

// Manual ModuleParameters — NF4 frozen base is NOT a parameter; only lora_a/lora_b.
impl ModuleParameters for CohereQloraAttention {
    fn num_parameters(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut p = ModuleParamRef::new();
        for (name, proj) in [
            ("q_proj", &self.q_proj),
            ("k_proj", &self.k_proj),
            ("v_proj", &self.v_proj),
            ("o_proj", &self.o_proj),
        ] {
            let mut m = HashMap::new();
            m.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
            m.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
            p.insert(Rc::from(name), NestedValue::Map(m));
        }
        p
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut p = ModuleParamMut::new();
        for (name, proj) in [
            ("q_proj", &mut self.q_proj),
            ("k_proj", &mut self.k_proj),
            ("v_proj", &mut self.v_proj),
            ("o_proj", &mut self.o_proj),
        ] {
            let mut m = HashMap::new();
            m.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
            m.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
            p.insert(Rc::from(name), NestedValue::Map(m));
        }
        p
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

impl CohereQloraAttention {
    pub fn new(
        config: &CohereConfig,
        layer_idx: usize,
        qlora_config: &QLoraConfig,
    ) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

        let mut q_cfg = qlora_config.clone();
        q_cfg.lora.r = crate::effective_rank(&qlora_config.lora, "q_proj");
        let q_proj = QLoraLinear::new(hidden_size, n_heads * head_dim, &q_cfg, false)?;

        let mut k_cfg = qlora_config.clone();
        k_cfg.lora.r = crate::effective_rank(&qlora_config.lora, "k_proj");
        let k_proj = QLoraLinear::new(hidden_size, n_kv_heads * head_dim, &k_cfg, false)?;

        let mut v_cfg = qlora_config.clone();
        v_cfg.lora.r = crate::effective_rank(&qlora_config.lora, "v_proj");
        let v_proj = QLoraLinear::new(hidden_size, n_kv_heads * head_dim, &v_cfg, false)?;

        let mut o_cfg = qlora_config.clone();
        o_cfg.lora.r = crate::effective_rank(&qlora_config.lora, "o_proj");
        let o_proj = QLoraLinear::new(n_heads * head_dim, hidden_size, &o_cfg, false)?;

        let use_sliding_window =
            config.use_sliding_window && !config.uses_global_attention(layer_idx as i32);

        Ok(Self {
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            rope_theta: config.rope_theta,
            use_sliding_window,
            sliding_window: config.sliding_window,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Forward pass.
    ///
    /// RoPE is applied in `[B, S, H, D]` layout (before the `[0,2,1,3]` transpose),
    /// matching the base `CohereAttention`.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head layout [B, S, H, D]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply RoPE BEFORE transpose — [B, S, H, D] matches apply_rope convention
        // Cohere uses traditional (interleaved) RoPE — see CohereAttention.
        let q =
            pmetal_mlx::kernels::rope::apply_rope(&q, self.head_dim, true, self.rope_theta, 1.0, 0)
                .map_err(LoraError::Mlx)?;
        let k =
            pmetal_mlx::kernels::rope::apply_rope(&k, self.head_dim, true, self.rope_theta, 1.0, 0)
                .map_err(LoraError::Mlx)?;

        // Transpose to [B, H, S, D]
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        // GQA expansion
        let k = if self.n_kv_heads < self.n_heads {
            expand_kv_heads(&k, self.n_heads / self.n_kv_heads)?
        } else {
            k
        };
        let v = if self.n_kv_heads < self.n_heads {
            expand_kv_heads(&v, self.n_heads / self.n_kv_heads)?
        } else {
            v
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

        let output = output.transpose_axes(&[0, 2, 1, 3]);
        let output = output.reshape(&[batch, seq_len, -1]);
        self.o_proj.forward(&output)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (q0, q1, q2) = self.q_proj.memory_usage();
        let (k0, k1, k2) = self.k_proj.memory_usage();
        let (v0, v1, v2) = self.v_proj.memory_usage();
        let (o0, o1, o2) = self.o_proj.memory_usage();
        (q0 + k0 + v0 + o0, q1 + k1 + v1 + o1, q2 + k2 + v2 + o2)
    }
}

fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, LoraError> {
    let s = x.shape();
    let (b, h, t, d) = (s[0], s[1], s[2], s[3]);
    let x = x.reshape(&[b, h, 1, t, d]);
    let x = ops::broadcast_to(&x, &[b, h, repeats, t, d]);
    Ok(x.reshape(&[b, h * repeats, t, d]))
}

// =============================================================================
// MLP
// =============================================================================

/// QLoRA-enabled SwiGLU MLP for Cohere.
#[derive(Debug)]
pub struct CohereQloraMLP {
    pub gate_proj: QLoraLinear,
    pub up_proj: QLoraLinear,
    pub down_proj: QLoraLinear,
}

// Manual ModuleParameters — only lora_a/lora_b exposed.
impl ModuleParameters for CohereQloraMLP {
    fn num_parameters(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut p = ModuleParamRef::new();
        for (name, proj) in [
            ("gate_proj", &self.gate_proj),
            ("up_proj", &self.up_proj),
            ("down_proj", &self.down_proj),
        ] {
            let mut m = HashMap::new();
            m.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
            m.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
            p.insert(Rc::from(name), NestedValue::Map(m));
        }
        p
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut p = ModuleParamMut::new();
        for (name, proj) in [
            ("gate_proj", &mut self.gate_proj),
            ("up_proj", &mut self.up_proj),
            ("down_proj", &mut self.down_proj),
        ] {
            let mut m = HashMap::new();
            m.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
            m.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
            p.insert(Rc::from(name), NestedValue::Map(m));
        }
        p
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

impl CohereQloraMLP {
    pub fn new(config: &CohereConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;

        let mut gate_cfg = qlora_config.clone();
        gate_cfg.lora.r = crate::effective_rank(&qlora_config.lora, "gate_proj");
        let gate_proj = QLoraLinear::new(hidden, inter, &gate_cfg, false)?;

        let mut up_cfg = qlora_config.clone();
        up_cfg.lora.r = crate::effective_rank(&qlora_config.lora, "up_proj");
        let up_proj = QLoraLinear::new(hidden, inter, &up_cfg, false)?;

        let mut down_cfg = qlora_config.clone();
        down_cfg.lora.r = crate::effective_rank(&qlora_config.lora, "down_proj");
        let down_proj = QLoraLinear::new(inter, hidden, &down_cfg, false)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// SwiGLU forward: `silu(gate) * up → down`.
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(&gate);
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up);
        self.down_proj.forward(&hidden)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (g0, g1, g2) = self.gate_proj.memory_usage();
        let (u0, u1, u2) = self.up_proj.memory_usage();
        let (d0, d1, d2) = self.down_proj.memory_usage();
        (g0 + u0 + d0, g1 + u1 + d1, g2 + u2 + d2)
    }
}

// =============================================================================
// Decoder layer
// =============================================================================

/// QLoRA-enabled Cohere decoder layer with parallel residual.
#[derive(Debug)]
pub struct CohereQloraDecoderLayer {
    pub layer_idx: usize,
    pub self_attn: CohereQloraAttention,
    pub mlp: CohereQloraMLP,
    /// LayerNorm with weight AND bias (shared input norm for both branches).
    pub input_layernorm: nn::LayerNorm,
}

impl CohereQloraDecoderLayer {
    pub fn new(
        config: &CohereConfig,
        layer_idx: usize,
        qlora_config: &QLoraConfig,
    ) -> Result<Self, LoraError> {
        let self_attn = CohereQloraAttention::new(config, layer_idx, qlora_config)?;
        let mlp = CohereQloraMLP::new(config, qlora_config)?;
        let input_layernorm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;
        Ok(Self {
            layer_idx,
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    /// Parallel residual forward: `x + attn(norm(x)) + ffn(norm(x))`.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.input_layernorm, x).map_err(LoraError::Mlx)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        let ffn_out = self.mlp.forward(&normed)?;
        Ok(x.add(&attn_out).add(&ffn_out))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.mlp.num_trainable_params()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (a0, a1, a2) = self.self_attn.memory_usage();
        let (m0, m1, m2) = self.mlp.memory_usage();
        (a0 + m0, a1 + m1, a2 + m2)
    }
}

// =============================================================================
// Model trunk
// =============================================================================

/// QLoRA-enabled Cohere model (without LM head).
#[derive(Debug)]
pub struct CohereQloraModel {
    pub config: CohereConfig,
    pub qlora_config: QLoraConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<CohereQloraDecoderLayer>,
    /// Final LayerNorm with weight AND bias.
    pub norm: nn::LayerNorm,
}

impl CohereQloraModel {
    pub fn new(config: CohereConfig, qlora_config: QLoraConfig) -> Result<Self, LoraError> {
        let embed_tokens =
            nn::Embedding::new(config.vocab_size, config.hidden_size).map_err(LoraError::Mlx)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| CohereQloraDecoderLayer::new(&config, i as usize, &qlora_config))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
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

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let mut hidden =
            Module::forward(&mut self.embed_tokens, input_ids).map_err(LoraError::Mlx)?;

        // Build causal mask when none is provided
        let owned_mask: Option<Array>;
        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            owned_mask = Some(create_causal_mask(seq_len).map_err(LoraError::Mlx)?);
            owned_mask.as_ref()
        } else {
            mask
        };

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, mask)?;
        }

        Module::forward(&mut self.norm, &hidden).map_err(LoraError::Mlx)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.layers.iter().fold((0, 0, 0), |(q, l, t), layer| {
            let (lq, ll, lt) = layer.memory_usage();
            (q + lq, l + ll, t + lt)
        })
    }
}

// =============================================================================
// Causal LM head
// =============================================================================

/// QLoRA-enabled Cohere model for causal language modelling.
#[derive(Debug)]
pub struct CohereQloraForCausalLM {
    pub model: CohereQloraModel,
    /// Separate LM head when `tie_word_embeddings == false`.
    pub lm_head: Option<nn::Linear>,
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl CohereQloraForCausalLM {
    /// Construct from `LoraConfig` (wraps into default `QLoraConfig`).
    pub fn new(config: CohereConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let qlora_config = QLoraConfig::from_lora(lora_config);
        Self::with_qlora_config(config, qlora_config)
    }

    /// Construct with explicit QLoRA configuration.
    pub fn with_qlora_config(
        config: CohereConfig,
        qlora_config: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let tie_weights = config.tie_word_embeddings;

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

        let model = CohereQloraModel::new(config, qlora_config)?;
        Ok(Self {
            model,
            lm_head,
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

    /// Forward pass producing logits `[B, S, vocab_size]`.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let hidden = self.model.forward(input_ids, mask)?;
        let logits = if let Some(ref mut head) = self.lm_head {
            Module::forward(head, &hidden).map_err(LoraError::Mlx)?
        } else {
            // Tied embeddings: embed_tokens.weight is the LM head transposed
            self.model.embed_tokens.as_linear(&hidden)
        };
        // Cohere scales LM-head logits by `logit_scale` (0.0625 for Command-R).
        Ok(logits.multiply(&Array::from_f32(self.model.config.logit_scale)))
    }

    // -------------------------------------------------------------------------
    // Trainable parameter accessors
    // -------------------------------------------------------------------------

    /// Collect all trainable LoRA parameters keyed by their weight-file path.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("model.layers.{}", i);
            macro_rules! insert {
                ($proj:expr, $name:literal) => {
                    params.insert(
                        Rc::from(format!("{}.{}.lora_A.weight", prefix, $name)),
                        $proj.lora_a.clone(),
                    );
                    params.insert(
                        Rc::from(format!("{}.{}.lora_B.weight", prefix, $name)),
                        $proj.lora_b.clone(),
                    );
                };
            }
            insert!(layer.self_attn.q_proj, "self_attn.q_proj");
            insert!(layer.self_attn.k_proj, "self_attn.k_proj");
            insert!(layer.self_attn.v_proj, "self_attn.v_proj");
            insert!(layer.self_attn.o_proj, "self_attn.o_proj");
            insert!(layer.mlp.gate_proj, "mlp.gate_proj");
            insert!(layer.mlp.up_proj, "mlp.up_proj");
            insert!(layer.mlp.down_proj, "mlp.down_proj");
        }
        params
    }

    /// Restore LoRA parameters from a HashMap.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);
            macro_rules! maybe_set {
                ($param:expr, $key:expr) => {
                    if let Some(v) = params.get(&Rc::from($key)) {
                        $param = v.clone();
                    }
                };
            }
            maybe_set!(
                layer.self_attn.q_proj.lora_a,
                format!("{}.self_attn.q_proj.lora_A.weight", prefix)
            );
            maybe_set!(
                layer.self_attn.q_proj.lora_b,
                format!("{}.self_attn.q_proj.lora_B.weight", prefix)
            );
            maybe_set!(
                layer.self_attn.k_proj.lora_a,
                format!("{}.self_attn.k_proj.lora_A.weight", prefix)
            );
            maybe_set!(
                layer.self_attn.k_proj.lora_b,
                format!("{}.self_attn.k_proj.lora_B.weight", prefix)
            );
            maybe_set!(
                layer.self_attn.v_proj.lora_a,
                format!("{}.self_attn.v_proj.lora_A.weight", prefix)
            );
            maybe_set!(
                layer.self_attn.v_proj.lora_b,
                format!("{}.self_attn.v_proj.lora_B.weight", prefix)
            );
            maybe_set!(
                layer.self_attn.o_proj.lora_a,
                format!("{}.self_attn.o_proj.lora_A.weight", prefix)
            );
            maybe_set!(
                layer.self_attn.o_proj.lora_b,
                format!("{}.self_attn.o_proj.lora_B.weight", prefix)
            );
            maybe_set!(
                layer.mlp.gate_proj.lora_a,
                format!("{}.mlp.gate_proj.lora_A.weight", prefix)
            );
            maybe_set!(
                layer.mlp.gate_proj.lora_b,
                format!("{}.mlp.gate_proj.lora_B.weight", prefix)
            );
            maybe_set!(
                layer.mlp.up_proj.lora_a,
                format!("{}.mlp.up_proj.lora_A.weight", prefix)
            );
            maybe_set!(
                layer.mlp.up_proj.lora_b,
                format!("{}.mlp.up_proj.lora_B.weight", prefix)
            );
            maybe_set!(
                layer.mlp.down_proj.lora_a,
                format!("{}.mlp.down_proj.lora_A.weight", prefix)
            );
            maybe_set!(
                layer.mlp.down_proj.lora_b,
                format!("{}.mlp.down_proj.lora_B.weight", prefix)
            );
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    /// Memory savings as a ratio vs. full-precision equivalent.
    ///
    /// Returns `(quantized_bytes + lora_bytes) / full_fp32_bytes`.  A value < 1.0
    /// means the QLoRA model is smaller than the full-precision version.
    pub fn memory_savings(&self) -> f32 {
        let (quantized, lora, _) = self.memory_usage();

        let full_precision: usize = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.self_attn.q_proj.num_frozen_params() * 4
                    + l.self_attn.k_proj.num_frozen_params() * 4
                    + l.self_attn.v_proj.num_frozen_params() * 4
                    + l.self_attn.o_proj.num_frozen_params() * 4
                    + l.mlp.gate_proj.num_frozen_params() * 4
                    + l.mlp.up_proj.num_frozen_params() * 4
                    + l.mlp.down_proj.num_frozen_params() * 4
            })
            .sum::<usize>()
            + lora; // lora params stay at full precision in both cases

        (quantized + lora) as f32 / full_precision as f32
    }

    pub fn config(&self) -> &CohereConfig {
        &self.model.config
    }

    pub fn qlora_config(&self) -> &QLoraConfig {
        &self.model.qlora_config
    }

    // -------------------------------------------------------------------------
    // Save / load LoRA weights
    // -------------------------------------------------------------------------

    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        crate::save_safetensors_map(path, &params)
    }

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
        let params: HashMap<Rc<str>, Array> = loaded
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        self.set_lora_parameters(&params);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Merge / unmerge (NF4 restriction)
    // -------------------------------------------------------------------------

    /// NF4 base weights cannot be losslessly merged.
    ///
    /// Dequantize to fp16/fp32 first, then use a standard LoRA merge path.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "NF4 base cannot be losslessly merged — dequantize first".into(),
        ))
    }

    /// NF4 base weights cannot be losslessly unmerged.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "NF4 base cannot be losslessly merged — dequantize first".into(),
        ))
    }

    // -------------------------------------------------------------------------
    // Weight loading
    // -------------------------------------------------------------------------

    /// Load full-precision weights from a HashMap and NF4-quantize each projection.
    ///
    /// Norm weight/bias (LayerNorm) are kept in full precision.
    pub fn load_and_quantize_weights(
        &mut self,
        weights: &HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        // Embeddings (full precision)
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);
            let qcfg = &self.model.qlora_config.clone();

            macro_rules! quantize {
                ($proj:expr, $key:literal) => {
                    if let Some(w) = weights.get(&format!("{}.{}", prefix, $key)) {
                        $proj = QLoraLinear::from_weight(w, None, qcfg)?;
                    }
                };
            }

            quantize!(layer.self_attn.q_proj, "self_attn.q_proj.weight");
            quantize!(layer.self_attn.k_proj, "self_attn.k_proj.weight");
            quantize!(layer.self_attn.v_proj, "self_attn.v_proj.weight");
            quantize!(layer.self_attn.o_proj, "self_attn.o_proj.weight");
            quantize!(layer.mlp.gate_proj, "mlp.gate_proj.weight");
            quantize!(layer.mlp.up_proj, "mlp.up_proj.weight");
            quantize!(layer.mlp.down_proj, "mlp.down_proj.weight");

            // LayerNorm weight + bias (full precision, Option<Array> inner type)
            if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.input_layernorm.weight = Param::new(Some(w.clone()));
            }
            if let Some(b) = weights.get(&format!("{}.input_layernorm.bias", prefix)) {
                layer.input_layernorm.bias = Param::new(Some(b.clone()));
            }
        }

        // Final LayerNorm weight + bias (full precision)
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(Some(w.clone()));
        }
        if let Some(b) = weights.get("model.norm.bias") {
            self.model.norm.bias = Param::new(Some(b.clone()));
        }

        // LM head (full precision, only when not tied)
        if let Some(ref mut head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }

    /// Load and NF4-quantize base weights from a safetensors directory.
    pub fn load_and_quantize_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();

        let single = model_dir.join("model.safetensors");
        if single.exists() {
            let weights = crate::sanitize_loaded_weights(crate::load_safetensors_map(&single)?)?;
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

        let shards: std::collections::HashSet<&String> = index.weight_map.values().collect();
        let mut all_weights = HashMap::new();
        for shard in shards {
            let shard_path = model_dir.join(shard);
            let shard_weights =
                crate::sanitize_loaded_weights(crate::load_safetensors_map(&shard_path)?)?;
            all_weights.extend(shard_weights);
        }

        self.load_and_quantize_weights(&all_weights)
    }

    /// Reload a merged (full-precision) checkpoint over the current QLoRA model.
    ///
    /// This is useful after running an external dequantize+merge step: you can
    /// reconstruct the model in its unquantized state by loading the merged weights
    /// back in and re-quantizing them.
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        self.load_and_quantize_from_dir(model_dir)
    }
}

// =============================================================================
// ModuleParameters
// =============================================================================

impl ModuleParameters for CohereQloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // Attention
            let mut attn_params = HashMap::new();
            for (name, proj) in [
                ("q_proj", &layer.self_attn.q_proj),
                ("k_proj", &layer.self_attn.k_proj),
                ("v_proj", &layer.self_attn.v_proj),
                ("o_proj", &layer.self_attn.o_proj),
            ] {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                attn_params.insert(Rc::from(name), NestedValue::Map(m));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // MLP
            let mut mlp_params = HashMap::new();
            for (name, proj) in [
                ("gate_proj", &layer.mlp.gate_proj),
                ("up_proj", &layer.mlp.up_proj),
                ("down_proj", &layer.mlp.down_proj),
            ] {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                mlp_params.insert(Rc::from(name), NestedValue::Map(m));
            }
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

            // Attention
            let mut attn_params = HashMap::new();
            for (name, proj) in [
                ("q_proj", &mut layer.self_attn.q_proj),
                ("k_proj", &mut layer.self_attn.k_proj),
                ("v_proj", &mut layer.self_attn.v_proj),
                ("o_proj", &mut layer.self_attn.o_proj),
            ] {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                attn_params.insert(Rc::from(name), NestedValue::Map(m));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // MLP
            let mut mlp_params = HashMap::new();
            for (name, proj) in [
                ("gate_proj", &mut layer.mlp.gate_proj),
                ("up_proj", &mut layer.mlp.up_proj),
                ("down_proj", &mut layer.mlp.down_proj),
            ] {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                mlp_params.insert(Rc::from(name), NestedValue::Map(m));
            }
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
// TrainableModel
// =============================================================================

impl TrainableModel for CohereQloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        CohereQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        CohereQloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        CohereQloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        CohereQloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        CohereQloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        CohereQloraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        CohereQloraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        CohereQloraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        true
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        use pmetal_mlx::kv_cache::KVCacheConfig;
        let n_layers = self.model.config.num_hidden_layers as usize;
        let n_kv = self.model.config.num_key_value_heads as usize;
        let head_dim = self.model.config.head_dim as usize;

        let config = KVCacheConfig::new(n_layers, max_seq_len, n_kv, head_dim);
        Some(KVCache::new(config))
    }
}

// =============================================================================
// Helpers
// =============================================================================

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> CohereConfig {
        CohereConfig {
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            max_position_embeddings: 128,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            logit_scale: 0.0625,
            tie_word_embeddings: false,
            use_sliding_window: true,
            sliding_window: 64,
            global_attention_layers: None,
        }
    }

    fn tiny_qlora_config() -> QLoraConfig {
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
    fn test_cohere_qlora_construction() {
        // 2-layer tiny model constructs without error.
        let config = tiny_config();
        let qlora_config = tiny_qlora_config();
        let model = CohereQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        // Both decoder layers created
        assert_eq!(model.model.layers.len(), 2);

        // lm_head exists (tie_word_embeddings = false)
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn test_cohere_qlora_trainable_params() {
        let config = tiny_config();
        let qlora_config = tiny_qlora_config();
        let model = CohereQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        // At least one trainable parameter
        let n = model.num_trainable_params();
        assert!(n > 0, "Expected trainable params, got 0");

        // lora_parameters() returns the same count (2 per projection × 7 projs × 2 layers)
        let param_map = model.lora_parameters();
        assert_eq!(param_map.len(), 2 * 7 * 2);

        // Frozen NF4 base weights are NOT in the parameter map
        // (every key ends in lora_A.weight or lora_B.weight)
        for key in param_map.keys() {
            assert!(
                key.contains("lora_A") || key.contains("lora_B"),
                "Unexpected non-LoRA key in parameter map: {}",
                key
            );
        }
    }

    #[test]
    fn test_cohere_qlora_memory_savings() {
        let config = tiny_config();
        let qlora_config = tiny_qlora_config();
        let model = CohereQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let ratio = model.memory_savings();
        // NF4 gives ~8× weight compression; the ratio must be less than 1.0
        assert!(
            ratio > 0.0,
            "memory_savings() should return a positive ratio"
        );
        assert!(
            ratio < 1.0,
            "QLoRA must use less memory than full-precision equivalent (got ratio {})",
            ratio
        );
    }

    #[test]
    fn test_cohere_qlora_tied_embeddings() {
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let qlora_config = tiny_qlora_config();
        let model = CohereQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        // With tied embeddings, lm_head should be None
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn test_cohere_qlora_merge_unmerge_returns_err() {
        let config = tiny_config();
        let qlora_config = tiny_qlora_config();
        let mut model = CohereQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        assert!(model.merge_lora().is_err());
        assert!(model.unmerge_lora().is_err());
    }
}
