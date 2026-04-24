//! QLoRA-enabled Nemotron-H hybrid architecture.
//!
//! Mirrors the LoRA placement strategy from `nemotron_h_lora`:
//! - **Attention layers** (`*` blocks): `q_proj`, `k_proj`, `v_proj`, `o_proj` quantized + LoRA.
//! - **Mamba layers** (`M` blocks): ALL components stay fp16 frozen — no LoRA, no quantization.
//! - **MLP layers** (`-` blocks): `up_proj`, `down_proj` quantized + LoRA.
//!   Note: NemotronH MLP uses relu² — no gate_proj.
//! - **MoE layers** (`E` blocks): shared expert (`up_proj`, `down_proj`) quantized + LoRA;
//!   routed experts stay fp16 frozen.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Once;

use pmetal_bridge::compat::{
    Array, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param, nn,
};
use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::{KVCache, MambaCache, MambaCacheEntry};
use pmetal_models::architectures::nemotron_h::{
    MoELayer, NemotronHConfig, NemotronHMixer, load_nemotron_weights,
};

use crate::{
    LoraError, QLoraConfig, QLoraLinear, TrainableModel,
    nemotron_h_lora::{
        NemotronHLoraAttention, NemotronHLoraBlock, NemotronHLoraForCausalLM,
        NemotronHLoraMLP, NemotronHLoraMixer, NemotronHLoraMoE, NemotronHLoraModel,
        NemotronHLoraSharedExpert,
    },
    qlora::quantize_lora_layer,
};

static GRAD_CKPT_WARN: Once = Once::new();

// ============================================================================
// NemotronHQloraAttention
// ============================================================================

/// QLoRA-enabled attention mixer for Nemotron-H `*` blocks.
///
/// Base weights are quantized; LoRA adapters on q/k/v/o stay in full precision.
#[derive(Debug)]
pub struct NemotronHQloraAttention {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
    pub q_proj: QLoraLinear,
    pub k_proj: QLoraLinear,
    pub v_proj: QLoraLinear,
    pub o_proj: QLoraLinear,
}

impl NemotronHQloraAttention {
    fn from_lora(attn: NemotronHLoraAttention, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            num_heads: attn.num_heads,
            num_kv_heads: attn.num_kv_heads,
            head_dim: attn.head_dim,
            scale: attn.scale,
            rope_theta: attn.rope_theta,
            q_proj: quantize_lora_layer(&attn.q_proj, qcfg)?,
            k_proj: quantize_lora_layer(&attn.k_proj, qcfg)?,
            v_proj: quantize_lora_layer(&attn.v_proj, qcfg)?,
            o_proj: quantize_lora_layer(&attn.o_proj, qcfg)?,
        })
    }

    /// Training forward (no KV cache).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_impl(x, mask, None)
    }

    /// Cache-aware forward for inference.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        self.forward_impl(x, mask, cache)
    }

    fn forward_impl(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim]);

        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, offset)
            .map_err(LoraError::Mlx)?;
        let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, offset)
            .map_err(LoraError::Mlx)?;

        let (k, v) = if let Some((kvcache, layer_idx)) = cache {
            kvcache
                .update_and_fetch(layer_idx, &k, &v)
                .map_err(LoraError::Mlx)?
        } else {
            (k, v)
        };

        let attn_config =
            FusedAttentionConfig::new(self.num_heads, self.num_kv_heads, self.head_dim)
                .with_scale(self.scale)
                .with_mask_type(AttentionMaskType::Causal);

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask).map_err(LoraError::Mlx)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(&output)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (qq, ql, qt) = self.q_proj.memory_usage();
        let (kq, kl, kt) = self.k_proj.memory_usage();
        let (vq, vl, vt) = self.v_proj.memory_usage();
        let (oq, ol, ot) = self.o_proj.memory_usage();
        (qq + kq + vq + oq, ql + kl + vl + ol, qt + kt + vt + ot)
    }
}

// ============================================================================
// NemotronHQloraMLP — relu² MLP with LoRA on up/down
// ============================================================================

/// QLoRA-enabled MLP for Nemotron-H `-` blocks.
///
/// relu² activation: `down_proj(relu(up_proj(x))^2)`. No gate_proj.
#[derive(Debug)]
pub struct NemotronHQloraMLP {
    pub up_proj: QLoraLinear,
    pub down_proj: QLoraLinear,
}

impl NemotronHQloraMLP {
    fn from_lora(mlp: NemotronHLoraMLP, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            up_proj: quantize_lora_layer(&mlp.up_proj, qcfg)?,
            down_proj: quantize_lora_layer(&mlp.down_proj, qcfg)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let up = self.up_proj.forward(x)?;
        let activated = nn::relu(&up).square();
        self.down_proj.forward(&activated)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.up_proj.num_trainable_params() + self.down_proj.num_trainable_params()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (uq, ul, ut) = self.up_proj.memory_usage();
        let (dq, dl, dt) = self.down_proj.memory_usage();
        (uq + dq, ul + dl, ut + dt)
    }
}

// ============================================================================
// NemotronHQloraSharedExpert
// ============================================================================

/// QLoRA-enabled shared expert for Nemotron-H `E` blocks.
///
/// relu² activation, no gate_proj. Routed experts stay fp16 frozen.
#[derive(Debug)]
pub struct NemotronHQloraSharedExpert {
    pub up_proj: QLoraLinear,
    pub down_proj: QLoraLinear,
}

impl NemotronHQloraSharedExpert {
    fn from_lora(se: NemotronHLoraSharedExpert, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            up_proj: quantize_lora_layer(&se.up_proj, qcfg)?,
            down_proj: quantize_lora_layer(&se.down_proj, qcfg)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let up = self.up_proj.forward(x)?;
        let activated = nn::relu(&up).square();
        self.down_proj.forward(&activated)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.up_proj.num_trainable_params() + self.down_proj.num_trainable_params()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (uq, ul, ut) = self.up_proj.memory_usage();
        let (dq, dl, dt) = self.down_proj.memory_usage();
        (uq + dq, ul + dl, ut + dt)
    }
}

// ============================================================================
// NemotronHQloraMoE — frozen routed experts + QLoRA'd shared expert
// ============================================================================

/// QLoRA MoE block: frozen routed experts, quantized+LoRA shared expert.
#[derive(Debug)]
pub struct NemotronHQloraMoE {
    /// Frozen router + routed experts from base model.
    pub moe_layer: MoELayer,
    pub stacked_moe_up: Option<Array>,
    pub stacked_moe_down: Option<Array>,
    /// QLoRA'd shared expert (optional — absent when n_shared_experts == 0).
    pub shared_expert: Option<NemotronHQloraSharedExpert>,
}

impl NemotronHQloraMoE {
    fn from_lora(moe: NemotronHLoraMoE, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        let shared_expert = moe
            .shared_expert
            .map(|se| NemotronHQloraSharedExpert::from_lora(se, qcfg))
            .transpose()?;
        Ok(Self {
            moe_layer: moe.moe_layer,
            stacked_moe_up: moe.stacked_moe_up,
            stacked_moe_down: moe.stacked_moe_down,
            shared_expert,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let routed_out = if let (Some(stacked_up), Some(stacked_down)) =
            (&self.stacked_moe_up, &self.stacked_moe_down)
        {
            self.moe_layer
                .forward_stacked(x, stacked_up, stacked_down)
                .map_err(LoraError::Mlx)?
        } else {
            self.moe_layer.forward(x).map_err(LoraError::Mlx)?
        };

        let out = if let Some(ref mut shared) = self.shared_expert {
            let orig_shape = x.shape();
            let batch_seq: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            let hidden = orig_shape[orig_shape.len() - 1];
            let x_flat = x.reshape(&[batch_seq, hidden]);
            let shared_out = shared.forward(&x_flat)?;
            let shared_out = shared_out.reshape(routed_out.shape());
            routed_out.add(&shared_out)
        } else {
            routed_out
        };

        Ok(out)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.shared_expert
            .as_ref()
            .map(|s| s.num_trainable_params())
            .unwrap_or(0)
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.shared_expert
            .as_ref()
            .map(|s| s.memory_usage())
            .unwrap_or((0, 0, 0))
    }
}

// ============================================================================
// NemotronHQloraMixer — per-block dispatch enum
// ============================================================================

/// Mixer dispatch for a single NemotronH QLoRA block.
///
/// Mamba variant wraps the base frozen mixer unchanged.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum NemotronHQloraMixer {
    /// Mamba-2 SSM — all components frozen and fp16.
    Mamba(NemotronHMixer),
    /// Full attention — q/k/v/o quantized with LoRA adapters.
    Attention(NemotronHQloraAttention),
    /// Dense MLP — up/down quantized with LoRA adapters.
    Mlp(NemotronHQloraMLP),
    /// Mixture-of-experts — shared expert quantized with LoRA adapters.
    MoE(NemotronHQloraMoE),
}

impl NemotronHQloraMixer {
    fn from_lora(mixer: NemotronHLoraMixer, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        match mixer {
            NemotronHLoraMixer::Mamba(m) => Ok(Self::Mamba(m)),
            NemotronHLoraMixer::Attention(a) => {
                Ok(Self::Attention(NemotronHQloraAttention::from_lora(a, qcfg)?))
            }
            NemotronHLoraMixer::Mlp(m) => {
                Ok(Self::Mlp(NemotronHQloraMLP::from_lora(m, qcfg)?))
            }
            NemotronHLoraMixer::MoE(m) => {
                Ok(Self::MoE(NemotronHQloraMoE::from_lora(m, qcfg)?))
            }
        }
    }

    pub fn is_mamba(&self) -> bool {
        matches!(self, Self::Mamba(_))
    }

    /// Training forward (no caches).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        match self {
            Self::Mamba(m) => m
                .forward_with_cache(x, mask, None, None)
                .map_err(LoraError::Mlx),
            Self::Attention(a) => a.forward(x, mask),
            Self::Mlp(m) => m.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }

    /// Cache-aware forward for inference.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Mamba(m) => m
                .forward_with_cache(x, mask, None, mamba_cache)
                .map_err(LoraError::Mlx),
            Self::Attention(a) => a.forward_with_cache(x, mask, kv_cache),
            Self::Mlp(m) => m.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        match self {
            Self::Mamba(_) => 0,
            Self::Attention(a) => a.num_trainable_params(),
            Self::Mlp(m) => m.num_trainable_params(),
            Self::MoE(m) => m.num_trainable_params(),
        }
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        match self {
            Self::Mamba(_) => (0, 0, 0),
            Self::Attention(a) => a.memory_usage(),
            Self::Mlp(m) => m.memory_usage(),
            Self::MoE(m) => m.memory_usage(),
        }
    }
}

// ============================================================================
// NemotronHQloraBlock — single transformer block
// ============================================================================

/// A single NemotronH QLoRA block: pre-norm + mixer + residual.
#[derive(Debug)]
pub struct NemotronHQloraBlock {
    /// Pre-norm applied before the mixer (frozen).
    pub norm: nn::RmsNorm,
    pub mixer: NemotronHQloraMixer,
}

impl NemotronHQloraBlock {
    fn from_lora(block: NemotronHLoraBlock, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            norm: block.norm,
            mixer: NemotronHQloraMixer::from_lora(block.mixer, qcfg)?,
        })
    }

    /// Training forward.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.norm, x)?;
        let r = self.mixer.forward(&normed, mask)?;
        Ok(x.add(&r))
    }

    /// Cache-aware forward for inference.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.norm, x)?;
        let r = self
            .mixer
            .forward_with_cache(&normed, mask, kv_cache, mamba_cache)?;
        Ok(x.add(&r))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.mixer.num_trainable_params()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.mixer.memory_usage()
    }
}

// ============================================================================
// NemotronHQloraModel — backbone
// ============================================================================

/// Nemotron-H backbone with QLoRA adapters.
#[derive(Debug)]
pub struct NemotronHQloraModel {
    pub config: NemotronHConfig,
    pub qlora_config: QLoraConfig,
    pub embeddings: nn::Embedding,
    pub layers: Vec<NemotronHQloraBlock>,
    pub norm_f: nn::RmsNorm,
}

impl NemotronHQloraModel {
    fn from_lora(model: NemotronHLoraModel, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            config: model.config,
            qlora_config: qcfg.clone(),
            embeddings: model.embeddings,
            layers: model
                .layers
                .into_iter()
                .map(|block| NemotronHQloraBlock::from_lora(block, qcfg))
                .collect::<Result<Vec<_>, _>>()?,
            norm_f: model.norm_f,
        })
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_with_checkpoint(input_ids, mask, None)
    }

    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let mut hidden = Module::forward(&mut self.embeddings, input_ids)?;

        let layers_per_block = checkpoint_config
            .map(|c| c.layers_per_block)
            .unwrap_or(usize::MAX);
        let checkpointing_enabled = checkpoint_config.map(|c| c.enabled).unwrap_or(false);

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_mask = if layer.mixer.is_mamba() { None } else { mask };
            hidden = layer.forward(&hidden, layer_mask)?;

            if checkpointing_enabled && (idx + 1) % layers_per_block == 0 {
                GRAD_CKPT_WARN.call_once(|| {
                    tracing::info!(
                        "NemotronH QLoRA uses eager evaluation for memory management \
                         (gradient checkpointing requires custom_vjp not yet in MLX-rs)"
                    );
                });
            }
        }

        Ok(Module::forward(&mut self.norm_f, &hidden)?)
    }

    /// Cache-aware forward for inference.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, LoraError> {
        let mut hidden = Module::forward(&mut self.embeddings, input_ids)?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv = if matches!(layer.mixer, NemotronHQloraMixer::Attention(_)) {
                kv_cache.as_deref_mut().map(|c| (c, layer_idx))
            } else {
                None
            };
            let mamba = if layer.mixer.is_mamba() {
                mamba_cache
                    .as_deref_mut()
                    .and_then(|c| c.get_mut(layer_idx))
            } else {
                None
            };
            let layer_mask = if layer.mixer.is_mamba() { None } else { mask };
            hidden = layer.forward_with_cache(&hidden, layer_mask, kv, mamba)?;
        }

        Ok(Module::forward(&mut self.norm_f, &hidden)?)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.layers.iter().fold((0, 0, 0), |acc, layer| {
            let (q, l, t) = layer.memory_usage();
            (acc.0 + q, acc.1 + l, acc.2 + t)
        })
    }
}

// ============================================================================
// NemotronHQloraForCausalLM — top-level model
// ============================================================================

/// Nemotron-H causal language model with QLoRA adapters.
#[derive(Debug)]
pub struct NemotronHQloraForCausalLM {
    pub model: NemotronHQloraModel,
    /// LM head — absent when `tie_word_embeddings = true`.
    pub lm_head: Option<nn::Linear>,
    /// Interface-only gradient checkpointing parity.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl NemotronHQloraForCausalLM {
    pub fn new(config: NemotronHConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        Self::with_qlora_config(config, QLoraConfig::from_lora(lora_config))
    }

    pub fn with_qlora_config(
        config: NemotronHConfig,
        qcfg: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let lora = NemotronHLoraForCausalLM::new(config, qcfg.lora.clone())?;
        Self::from_lora(lora, qcfg)
    }

    fn from_lora(lora: NemotronHLoraForCausalLM, qcfg: QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            model: NemotronHQloraModel::from_lora(lora.model, &qcfg)?,
            lm_head: lora.lm_head,
            checkpoint_config: None,
        })
    }

    pub fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        self.checkpoint_config = Some(CheckpointConfig {
            enabled: true,
            layers_per_block,
            eval_at_boundaries: true,
        });
    }

    pub fn disable_gradient_checkpointing(&mut self) {
        self.checkpoint_config = None;
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let checkpoint_config = self.checkpoint_config.clone();
        let hidden =
            self.model
                .forward_with_checkpoint(input_ids, mask, checkpoint_config.as_ref())?;
        self.lm_head_forward(&hidden)
    }

    /// Cache-aware forward for autoregressive inference.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, LoraError> {
        let h = self
            .model
            .forward_with_cache(input_ids, mask, kv_cache, mamba_cache)?;
        self.lm_head_forward(&h)
    }

    fn lm_head_forward(&mut self, h: &Array) -> Result<Array, LoraError> {
        if let Some(ref mut lm_head) = self.lm_head {
            Ok(Module::forward(lm_head, h)?)
        } else {
            Ok(self.model.embeddings.as_linear(h))
        }
    }

    /// Forward returning hidden states before lm_head.
    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let checkpoint_config = self.checkpoint_config.clone();
        self.model
            .forward_with_checkpoint(input_ids, mask, checkpoint_config.as_ref())
    }

    pub fn get_lm_head_weight(&self) -> Option<Array> {
        if let Some(ref lm_head) = self.lm_head {
            Some(lm_head.weight.value.clone())
        } else {
            Some(self.model.embeddings.weight.value.clone())
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    // -------------------------------------------------------------------------
    // LoRA parameter utilities
    // -------------------------------------------------------------------------

    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("layers.{i}");

            match &layer.mixer {
                NemotronHQloraMixer::Mamba(_) => {
                    // No trainable parameters.
                }
                NemotronHQloraMixer::Attention(attn) => {
                    for (proj_name, lora) in [
                        ("q_proj", &attn.q_proj),
                        ("k_proj", &attn.k_proj),
                        ("v_proj", &attn.v_proj),
                        ("o_proj", &attn.o_proj),
                    ] {
                        params.insert(
                            Rc::from(format!("{prefix}.mixer.{proj_name}.lora_a")),
                            lora.lora_a.clone(),
                        );
                        params.insert(
                            Rc::from(format!("{prefix}.mixer.{proj_name}.lora_b")),
                            lora.lora_b.clone(),
                        );
                    }
                }
                NemotronHQloraMixer::Mlp(mlp) => {
                    for (proj_name, lora) in
                        [("up_proj", &mlp.up_proj), ("down_proj", &mlp.down_proj)]
                    {
                        params.insert(
                            Rc::from(format!("{prefix}.mixer.{proj_name}.lora_a")),
                            lora.lora_a.clone(),
                        );
                        params.insert(
                            Rc::from(format!("{prefix}.mixer.{proj_name}.lora_b")),
                            lora.lora_b.clone(),
                        );
                    }
                }
                NemotronHQloraMixer::MoE(moe) => {
                    if let Some(ref se) = moe.shared_expert {
                        for (proj_name, lora) in
                            [("up_proj", &se.up_proj), ("down_proj", &se.down_proj)]
                        {
                            params.insert(
                                Rc::from(format!(
                                    "{prefix}.mixer.shared_expert.{proj_name}.lora_a"
                                )),
                                lora.lora_a.clone(),
                            );
                            params.insert(
                                Rc::from(format!(
                                    "{prefix}.mixer.shared_expert.{proj_name}.lora_b"
                                )),
                                lora.lora_b.clone(),
                            );
                        }
                    }
                }
            }
        }

        params
    }

    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        macro_rules! set_param {
            ($dst:expr, $key:expr) => {
                if let Some(v) = params.get(&Rc::from($key) as &Rc<str>) {
                    $dst = v.clone();
                }
            };
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{i}");

            match &mut layer.mixer {
                NemotronHQloraMixer::Mamba(_) => {}
                NemotronHQloraMixer::Attention(attn) => {
                    set_param!(attn.q_proj.lora_a, format!("{prefix}.mixer.q_proj.lora_a"));
                    set_param!(attn.q_proj.lora_b, format!("{prefix}.mixer.q_proj.lora_b"));
                    set_param!(attn.k_proj.lora_a, format!("{prefix}.mixer.k_proj.lora_a"));
                    set_param!(attn.k_proj.lora_b, format!("{prefix}.mixer.k_proj.lora_b"));
                    set_param!(attn.v_proj.lora_a, format!("{prefix}.mixer.v_proj.lora_a"));
                    set_param!(attn.v_proj.lora_b, format!("{prefix}.mixer.v_proj.lora_b"));
                    set_param!(attn.o_proj.lora_a, format!("{prefix}.mixer.o_proj.lora_a"));
                    set_param!(attn.o_proj.lora_b, format!("{prefix}.mixer.o_proj.lora_b"));
                }
                NemotronHQloraMixer::Mlp(mlp) => {
                    set_param!(mlp.up_proj.lora_a, format!("{prefix}.mixer.up_proj.lora_a"));
                    set_param!(mlp.up_proj.lora_b, format!("{prefix}.mixer.up_proj.lora_b"));
                    set_param!(
                        mlp.down_proj.lora_a,
                        format!("{prefix}.mixer.down_proj.lora_a")
                    );
                    set_param!(
                        mlp.down_proj.lora_b,
                        format!("{prefix}.mixer.down_proj.lora_b")
                    );
                }
                NemotronHQloraMixer::MoE(moe) => {
                    if let Some(ref mut se) = moe.shared_expert {
                        set_param!(
                            se.up_proj.lora_a,
                            format!("{prefix}.mixer.shared_expert.up_proj.lora_a")
                        );
                        set_param!(
                            se.up_proj.lora_b,
                            format!("{prefix}.mixer.shared_expert.up_proj.lora_b")
                        );
                        set_param!(
                            se.down_proj.lora_a,
                            format!("{prefix}.mixer.shared_expert.down_proj.lora_a")
                        );
                        set_param!(
                            se.down_proj.lora_b,
                            format!("{prefix}.mixer.shared_expert.down_proj.lora_b")
                        );
                    }
                }
            }
        }
    }

    pub fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        crate::save_safetensors_map(path, &self.lora_parameters())
    }

    pub fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
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
    // Weight loading — delegates to the lora variant then re-quantizes
    // -------------------------------------------------------------------------

    /// Load base model weights from a HashMap, re-quantizing projection layers.
    ///
    /// Strategy: construct a fresh `NemotronHLoraForCausalLM`, load weights into
    /// it, then convert to QLoRA. This reuses the thoroughly-tested weight loading
    /// logic from the unquantized variant.
    pub fn load_base_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        let mut lora =
            NemotronHLoraForCausalLM::new(self.model.config.clone(), self.model.qlora_config.lora.clone())?;
        lora.load_base_weights(weights)?;
        let qlora_cfg = self.model.qlora_config.clone();
        let new_self = Self::from_lora(lora, qlora_cfg)?;
        *self = new_self;
        Ok(())
    }

    /// Load base weights from SafeTensors files in a directory.
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
    ) -> Result<(), LoraError> {
        let mut lora = NemotronHLoraForCausalLM::new(
            self.model.config.clone(),
            self.model.qlora_config.lora.clone(),
        )?;
        lora.load_base_weights_from_dir(model_dir)?;
        let qlora_cfg = self.model.qlora_config.clone();
        let new_self = Self::from_lora(lora, qlora_cfg)?;
        *self = new_self;
        Ok(())
    }

    /// Force evaluation of all parameters.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.model.embeddings.weight.value.eval();

        for layer in &mut self.model.layers {
            layer.norm.weight.value.eval();

            match &mut layer.mixer {
                NemotronHQloraMixer::Mamba(_) => {}
                NemotronHQloraMixer::Attention(attn) => {
                    attn.q_proj.lora_a.eval();
                    attn.q_proj.lora_b.eval();
                    attn.k_proj.lora_a.eval();
                    attn.k_proj.lora_b.eval();
                    attn.v_proj.lora_a.eval();
                    attn.v_proj.lora_b.eval();
                    attn.o_proj.lora_a.eval();
                    attn.o_proj.lora_b.eval();
                }
                NemotronHQloraMixer::Mlp(mlp) => {
                    mlp.up_proj.lora_a.eval();
                    mlp.up_proj.lora_b.eval();
                    mlp.down_proj.lora_a.eval();
                    mlp.down_proj.lora_b.eval();
                }
                NemotronHQloraMixer::MoE(moe) => {
                    if let Some(ref mut se) = moe.shared_expert {
                        se.up_proj.lora_a.eval();
                        se.up_proj.lora_b.eval();
                        se.down_proj.lora_a.eval();
                        se.down_proj.lora_b.eval();
                    }
                }
            }
        }

        self.model.norm_f.weight.value.eval();
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.weight.value.eval();
        }

        Ok(())
    }

    /// Merge LoRA weights into the dequantized base weights.
    ///
    /// Note: QLoRA merge is non-reversible. Reload from disk to undo.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            match &mut layer.mixer {
                NemotronHQloraMixer::Mamba(_) => {}
                NemotronHQloraMixer::Attention(attn) => {
                    attn.q_proj.merge()?;
                    attn.k_proj.merge()?;
                    attn.v_proj.merge()?;
                    attn.o_proj.merge()?;
                }
                NemotronHQloraMixer::Mlp(mlp) => {
                    mlp.up_proj.merge()?;
                    mlp.down_proj.merge()?;
                }
                NemotronHQloraMixer::MoE(moe) => {
                    if let Some(ref mut se) = moe.shared_expert {
                        se.up_proj.merge()?;
                        se.down_proj.merge()?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Unmerge is not reversible — reload base weights to undo.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".to_string(),
        ))
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    pub fn memory_savings(&self) -> f32 {
        let (quantized, lora, _) = self.memory_usage();
        let full_precision: usize = self
            .model
            .layers
            .iter()
            .map(|layer| match &layer.mixer {
                NemotronHQloraMixer::Mamba(_) => 0,
                NemotronHQloraMixer::Attention(attn) => {
                    attn.q_proj.num_frozen_params() * 4
                        + attn.k_proj.num_frozen_params() * 4
                        + attn.v_proj.num_frozen_params() * 4
                        + attn.o_proj.num_frozen_params() * 4
                }
                NemotronHQloraMixer::Mlp(mlp) => {
                    mlp.up_proj.num_frozen_params() * 4 + mlp.down_proj.num_frozen_params() * 4
                }
                NemotronHQloraMixer::MoE(moe) => moe
                    .shared_expert
                    .as_ref()
                    .map(|se| {
                        se.up_proj.num_frozen_params() * 4 + se.down_proj.num_frozen_params() * 4
                    })
                    .unwrap_or(0),
            })
            .sum::<usize>()
            + lora;
        if full_precision == 0 {
            return 1.0;
        }
        (quantized + lora) as f32 / full_precision as f32
    }
}

// ============================================================================
// ModuleParameters for NemotronHQloraForCausalLM
// ============================================================================

impl ModuleParameters for NemotronHQloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let layer_key: Rc<str> = Rc::from(format!("layers.{i}"));
            let mut mixer_map = HashMap::new();

            match &layer.mixer {
                NemotronHQloraMixer::Mamba(_) => {}
                NemotronHQloraMixer::Attention(attn) => {
                    for (proj_name, lora) in [
                        ("q_proj", &attn.q_proj),
                        ("k_proj", &attn.k_proj),
                        ("v_proj", &attn.v_proj),
                        ("o_proj", &attn.o_proj),
                    ] {
                        let mut p = HashMap::new();
                        p.insert(Rc::from("lora_a"), NestedValue::Value(&lora.lora_a));
                        p.insert(Rc::from("lora_b"), NestedValue::Value(&lora.lora_b));
                        mixer_map.insert(Rc::from(proj_name), NestedValue::Map(p));
                    }
                }
                NemotronHQloraMixer::Mlp(mlp) => {
                    for (proj_name, lora) in
                        [("up_proj", &mlp.up_proj), ("down_proj", &mlp.down_proj)]
                    {
                        let mut p = HashMap::new();
                        p.insert(Rc::from("lora_a"), NestedValue::Value(&lora.lora_a));
                        p.insert(Rc::from("lora_b"), NestedValue::Value(&lora.lora_b));
                        mixer_map.insert(Rc::from(proj_name), NestedValue::Map(p));
                    }
                }
                NemotronHQloraMixer::MoE(moe) => {
                    if let Some(ref se) = moe.shared_expert {
                        let mut se_map = HashMap::new();
                        for (proj_name, lora) in
                            [("up_proj", &se.up_proj), ("down_proj", &se.down_proj)]
                        {
                            let mut p = HashMap::new();
                            p.insert(Rc::from("lora_a"), NestedValue::Value(&lora.lora_a));
                            p.insert(Rc::from("lora_b"), NestedValue::Value(&lora.lora_b));
                            se_map.insert(Rc::from(proj_name), NestedValue::Map(p));
                        }
                        mixer_map.insert(Rc::from("shared_expert"), NestedValue::Map(se_map));
                    }
                }
            }

            if !mixer_map.is_empty() {
                let mut layer_map = HashMap::new();
                layer_map.insert(Rc::from("mixer"), NestedValue::Map(mixer_map));
                params.insert(layer_key, NestedValue::Map(layer_map));
            }
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let layer_key: Rc<str> = Rc::from(format!("layers.{i}"));
            let mut mixer_map = HashMap::new();

            match &mut layer.mixer {
                NemotronHQloraMixer::Mamba(_) => {}
                NemotronHQloraMixer::Attention(attn) => {
                    let mut q = HashMap::new();
                    q.insert(Rc::from("lora_a"), NestedValue::Value(&mut attn.q_proj.lora_a));
                    q.insert(Rc::from("lora_b"), NestedValue::Value(&mut attn.q_proj.lora_b));
                    mixer_map.insert(Rc::from("q_proj"), NestedValue::Map(q));

                    let mut k = HashMap::new();
                    k.insert(Rc::from("lora_a"), NestedValue::Value(&mut attn.k_proj.lora_a));
                    k.insert(Rc::from("lora_b"), NestedValue::Value(&mut attn.k_proj.lora_b));
                    mixer_map.insert(Rc::from("k_proj"), NestedValue::Map(k));

                    let mut v = HashMap::new();
                    v.insert(Rc::from("lora_a"), NestedValue::Value(&mut attn.v_proj.lora_a));
                    v.insert(Rc::from("lora_b"), NestedValue::Value(&mut attn.v_proj.lora_b));
                    mixer_map.insert(Rc::from("v_proj"), NestedValue::Map(v));

                    let mut o = HashMap::new();
                    o.insert(Rc::from("lora_a"), NestedValue::Value(&mut attn.o_proj.lora_a));
                    o.insert(Rc::from("lora_b"), NestedValue::Value(&mut attn.o_proj.lora_b));
                    mixer_map.insert(Rc::from("o_proj"), NestedValue::Map(o));
                }
                NemotronHQloraMixer::Mlp(mlp) => {
                    let mut up = HashMap::new();
                    up.insert(Rc::from("lora_a"), NestedValue::Value(&mut mlp.up_proj.lora_a));
                    up.insert(Rc::from("lora_b"), NestedValue::Value(&mut mlp.up_proj.lora_b));
                    mixer_map.insert(Rc::from("up_proj"), NestedValue::Map(up));

                    let mut down = HashMap::new();
                    down.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut mlp.down_proj.lora_a),
                    );
                    down.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut mlp.down_proj.lora_b),
                    );
                    mixer_map.insert(Rc::from("down_proj"), NestedValue::Map(down));
                }
                NemotronHQloraMixer::MoE(moe) => {
                    if let Some(ref mut se) = moe.shared_expert {
                        let mut se_map = HashMap::new();

                        let mut up = HashMap::new();
                        up.insert(Rc::from("lora_a"), NestedValue::Value(&mut se.up_proj.lora_a));
                        up.insert(Rc::from("lora_b"), NestedValue::Value(&mut se.up_proj.lora_b));
                        se_map.insert(Rc::from("up_proj"), NestedValue::Map(up));

                        let mut down = HashMap::new();
                        down.insert(
                            Rc::from("lora_a"),
                            NestedValue::Value(&mut se.down_proj.lora_a),
                        );
                        down.insert(
                            Rc::from("lora_b"),
                            NestedValue::Value(&mut se.down_proj.lora_b),
                        );
                        se_map.insert(Rc::from("down_proj"), NestedValue::Map(down));

                        mixer_map.insert(Rc::from("shared_expert"), NestedValue::Map(se_map));
                    }
                }
            }

            if !mixer_map.is_empty() {
                let mut layer_map = HashMap::new();
                layer_map.insert(Rc::from("mixer"), NestedValue::Map(mixer_map));
                params.insert(layer_key, NestedValue::Map(layer_map));
            }
        }

        params
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.parameters()
    }

    fn freeze_parameters(&mut self, _recurse: bool) {}
    fn unfreeze_parameters(&mut self, _recurse: bool) {}
    fn all_frozen(&self) -> Option<bool> {
        Some(false)
    }
    fn any_frozen(&self) -> Option<bool> {
        Some(false)
    }
}

// ============================================================================
// TrainableModel for NemotronHQloraForCausalLM
// ============================================================================

impl TrainableModel for NemotronHQloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        NemotronHQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        // Hybrid models do not use explicit position IDs — Mamba layers use
        // recurrent state, attention layers use implicit RoPE offsets.
        NemotronHQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        NemotronHQloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        NemotronHQloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        NemotronHQloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        NemotronHQloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        NemotronHQloraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        NemotronHQloraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        NemotronHQloraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        false
    }

    /// NemotronH is a hybrid arch — KV cache alone is insufficient.
    fn supports_kv_cache(&self) -> bool {
        false
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(NemotronHQloraForCausalLM::forward_hidden_states(
            self, input_ids, mask,
        ))
    }

    fn lm_head_weight(&self) -> Option<Array> {
        NemotronHQloraForCausalLM::get_lm_head_weight(self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> NemotronHConfig {
        NemotronHConfig {
            model_type: "nemotron_h".to_string(),
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 3,
            max_position_embeddings: 512,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            attention_bias: false,
            head_dim: Some(16),
            mamba_num_heads: 4,
            mamba_head_dim: 8,
            mamba_proj_bias: false,
            ssm_state_size: 8,
            conv_kernel: 4,
            n_groups: 1,
            time_step_limit: (0.001, 0.1),
            time_step_min: None,
            time_step_max: None,
            mlp_bias: false,
            mlp_hidden_act: "relu2".to_string(),
            layer_norm_epsilon: 1e-5,
            use_bias: false,
            use_conv_bias: true,
            tie_word_embeddings: true,
            // Pattern: Mamba, Attention, MLP
            hybrid_override_pattern: Some("M*-".to_string()),
            moe_intermediate_size: None,
            moe_shared_expert_intermediate_size: None,
            n_group: None,
            n_routed_experts: None,
            n_shared_experts: None,
            topk_group: None,
            num_experts_per_tok: None,
            norm_topk_prob: None,
            routed_scaling_factor: None,
            rope_theta: 10000.0,
        }
    }

    fn tiny_lora_config() -> LoraConfig {
        LoraConfig {
            r: 4,
            alpha: 8.0,
            dropout: 0.0,
            use_rslora: false,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            bias: pmetal_core::LoraBias::None,
            init_lora_weights: true,
            loraplus_lr_ratio: None,
            use_dora: false,
        }
    }

    #[test]
    fn test_nemotron_h_qlora_builds() {
        let model = NemotronHQloraForCausalLM::with_qlora_config(
            tiny_config(),
            QLoraConfig::default(),
        );
        assert!(model.is_ok(), "QLoRA model construction should succeed");
        let model = model.unwrap();
        assert!(
            model.num_trainable_params() > 0,
            "Should have trainable parameters"
        );
    }

    #[test]
    fn test_nemotron_h_qlora_from_lora_config() {
        let model = NemotronHQloraForCausalLM::new(tiny_config(), tiny_lora_config());
        assert!(model.is_ok(), "Construction from LoraConfig should succeed");
    }

    #[test]
    fn test_layer_type_dispatch() {
        let model =
            NemotronHQloraForCausalLM::new(tiny_config(), tiny_lora_config()).unwrap();

        // Pattern "M*-": layer 0 = Mamba, layer 1 = Attention, layer 2 = MLP
        assert!(
            matches!(model.model.layers[0].mixer, NemotronHQloraMixer::Mamba(_)),
            "Layer 0 should be Mamba"
        );
        assert!(
            matches!(model.model.layers[1].mixer, NemotronHQloraMixer::Attention(_)),
            "Layer 1 should be Attention"
        );
        assert!(
            matches!(model.model.layers[2].mixer, NemotronHQloraMixer::Mlp(_)),
            "Layer 2 should be MLP"
        );
    }

    #[test]
    fn test_mamba_layers_have_no_lora_params() {
        let model =
            NemotronHQloraForCausalLM::new(tiny_config(), tiny_lora_config()).unwrap();
        let params = model.lora_parameters();
        let mamba_keys: Vec<_> = params
            .keys()
            .filter(|k| k.starts_with("layers.0."))
            .collect();
        assert!(
            mamba_keys.is_empty(),
            "Mamba layer should have no LoRA params, found: {:?}",
            mamba_keys
        );
    }

    #[test]
    fn test_attention_lora_param_keys() {
        let model =
            NemotronHQloraForCausalLM::new(tiny_config(), tiny_lora_config()).unwrap();
        let params = model.lora_parameters();

        // Layer 1 is Attention
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            for ab in &["lora_a", "lora_b"] {
                let key = format!("layers.1.mixer.{proj}.{ab}");
                assert!(
                    params.contains_key(&Rc::from(key.as_str())),
                    "Missing key: {key}"
                );
            }
        }
    }

    #[test]
    fn test_mlp_lora_param_keys() {
        let model =
            NemotronHQloraForCausalLM::new(tiny_config(), tiny_lora_config()).unwrap();
        let params = model.lora_parameters();

        // Layer 2 is MLP — only up_proj/down_proj, no gate_proj
        for proj in &["up_proj", "down_proj"] {
            for ab in &["lora_a", "lora_b"] {
                let key = format!("layers.2.mixer.{proj}.{ab}");
                assert!(
                    params.contains_key(&Rc::from(key.as_str())),
                    "Missing key: {key}"
                );
            }
        }
        // gate_proj must NOT appear
        assert!(
            !params.contains_key(&Rc::from("layers.2.mixer.gate_proj.lora_a")),
            "MLP should not have gate_proj (relu² not SwiGLU)"
        );
    }

    #[test]
    fn test_lora_param_roundtrip() {
        let mut model =
            NemotronHQloraForCausalLM::new(tiny_config(), tiny_lora_config()).unwrap();

        let original = model.lora_parameters();
        model.set_lora_parameters(&original);
        let restored = model.lora_parameters();

        assert_eq!(
            original.len(),
            restored.len(),
            "Parameter count should be preserved after roundtrip"
        );
    }

    #[test]
    fn test_supports_kv_cache_is_false() {
        let model =
            NemotronHQloraForCausalLM::new(tiny_config(), tiny_lora_config()).unwrap();
        assert!(
            !TrainableModel::supports_kv_cache(&model),
            "Hybrid model must report supports_kv_cache = false"
        );
    }

    #[test]
    fn test_nemotron_h_qlora_forward() {
        let mut model =
            NemotronHQloraForCausalLM::new(tiny_config(), tiny_lora_config()).unwrap();

        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let result = model.forward(&input_ids, None);
        assert!(
            result.is_ok(),
            "Forward pass should succeed: {:?}",
            result.err()
        );
        let logits = result.unwrap();
        assert_eq!(logits.shape(), &[1, 4, 64], "Logits shape mismatch");
    }

    #[test]
    fn test_memory_savings_in_range() {
        let model =
            NemotronHQloraForCausalLM::with_qlora_config(tiny_config(), QLoraConfig::default())
                .unwrap();
        let savings = model.memory_savings();
        // QLoRA should produce a ratio in (0.0, 1.0] — tiny config has few params
        // so the exact value varies, but should not be negative or > 1.0.
        assert!(savings > 0.0 && savings <= 1.0, "Unexpected savings: {savings}");
    }
}
