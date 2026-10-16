//! Nemotron-H hybrid Mamba-Transformer architecture.
//!
//! Supports NVIDIA Nemotron-H and the Nemotron 3 family, which combine:
//! - Mamba-2 SSM layers (the shared [`Mamba2Mixer`])
//! - attention layers, with no positional encoding (the Mamba layers carry
//!   the order)
//! - MLP layers with squared-ReLU activation
//! - mixture-of-experts layers: a sigmoid router with a load-balancing bias
//!   ([`DeepseekRouting`], the DeepSeek-V3 router), squared-ReLU experts and a
//!   shared expert
//!
//! The layer kinds are given by `hybrid_override_pattern` (`M` Mamba, `*`
//! attention, `-` MLP, `E` MoE) or by `layers_block_type`.
//!
//! Reference: <https://arxiv.org/abs/2504.03624>

use std::collections::HashMap;

use pmetal_cache::kv_cache::{KVCache, MambaCache, MambaCacheEntry};
use pmetal_runtime::compat::{
    Array, Buffer, Dtype, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters,
    VisitLinears, nn,
};
use pmetal_runtime::deepseek_native::{DeepseekRouting, ScoringFunc, TopkMethod};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use serde::{Deserialize, Serialize};

use crate::common::mamba2::{Mamba2Config, Mamba2Mixer, segments_from_positions};
use crate::common::switch_mlp::PlainSwitchMlp;

// ============================================================================
// Configuration
// ============================================================================

/// Nemotron-H model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NemotronHConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Intermediate size for MLP.
    pub intermediate_size: i32,
    /// Number of hidden layers; the layer pattern is authoritative when it
    /// disagrees.
    #[serde(default)]
    pub num_hidden_layers: i32,
    /// Maximum position embeddings.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,

    // Attention config
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of key-value heads (GQA).
    pub num_key_value_heads: i32,
    /// Attention bias.
    #[serde(default)]
    pub attention_bias: bool,
    /// Head dimension for attention.
    #[serde(default)]
    pub head_dim: Option<i32>,
    /// Head dimension for attention in the first releases, which state it
    /// under this name.
    #[serde(default, rename = "attention_head_dim")]
    pub legacy_head_dim: Option<i32>,

    // Mamba config
    /// Number of Mamba heads.
    #[serde(default = "default_mamba_num_heads")]
    pub mamba_num_heads: i32,
    /// Mamba head dimension.
    #[serde(default = "default_mamba_head_dim")]
    pub mamba_head_dim: i32,
    /// SSM state size.
    #[serde(default = "default_ssm_state_size")]
    pub ssm_state_size: i32,
    /// Convolution kernel size.
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: i32,
    /// Number of groups for Mamba.
    #[serde(default = "default_n_groups")]
    pub n_groups: i32,
    /// Tokens per chunk of the Mamba scan.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: i32,
    /// The limits the scan clamps the step size to, `[min, max]`: `(0, inf)`,
    /// no limit, in every release.
    #[serde(default = "default_time_step_limit")]
    pub time_step_limit: (f32, f32),
    /// The lower end of the range `dt_bias` was initialised from. Read by
    /// nothing at inference: the reference kernels clamp to
    /// `time_step_limit`, and flooring the step size at this value instead
    /// (as `transformers` does) moves a trained model's output.
    #[serde(default)]
    pub time_step_min: Option<f32>,
    /// The upper end of the range `dt_bias` was initialised from. Read by
    /// nothing at inference; kept so a config round-trips.
    #[serde(default)]
    pub time_step_max: Option<f32>,

    // MLP config
    /// MLP bias.
    #[serde(default)]
    pub mlp_bias: bool,
    /// MLP hidden activation (`relu2`, the only one released).
    #[serde(default = "default_mlp_hidden_act")]
    pub mlp_hidden_act: String,

    // General config
    /// Epsilon of every norm.
    #[serde(default)]
    pub layer_norm_epsilon: Option<f32>,
    /// Epsilon of every norm in the Nemotron 3.5 releases, which state no
    /// `layer_norm_epsilon`; the Nemotron 3 Nano releases state both, equal.
    #[serde(default)]
    pub norm_eps: Option<f32>,
    /// Bias in the Mamba input and output projections.
    #[serde(default)]
    pub use_bias: bool,
    /// Use bias in conv1d.
    #[serde(default = "default_use_conv_bias")]
    pub use_conv_bias: bool,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Layer kinds as a string: `M` Mamba, `*` attention, `-` MLP, `E` MoE.
    #[serde(default)]
    pub hybrid_override_pattern: Option<String>,
    /// Layer kinds as a list: `mamba` / `linear_attention`, `attention` /
    /// `full_attention`, `mlp`, `moe`. Wins over the pattern when both are
    /// given.
    #[serde(default)]
    pub layers_block_type: Option<Vec<String>>,

    // MoE config
    /// MoE intermediate size per expert.
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    /// MoE shared expert intermediate size.
    #[serde(default)]
    pub moe_shared_expert_intermediate_size: Option<i32>,
    /// Width the routed experts work in, when it is not the hidden size (the
    /// latent MoE of Nemotron 3 Super): the experts' input is projected down
    /// to it and their output back up.
    #[serde(default)]
    pub moe_latent_size: Option<i32>,
    /// Number of groups for MoE routing.
    #[serde(default)]
    pub n_group: Option<i32>,
    /// Number of routed experts.
    #[serde(default)]
    pub n_routed_experts: Option<i32>,
    /// Number of shared experts (the shared expert is one MLP of
    /// `moe_shared_expert_intermediate_size`, present when this is not 0).
    #[serde(default)]
    pub n_shared_experts: Option<i32>,
    /// Top-k groups for expert selection.
    #[serde(default)]
    pub topk_group: Option<i32>,
    /// Number of experts per token.
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    /// Normalize top-k probabilities (default on, as in the reference).
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
    /// Routed scaling factor.
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,

    /// RoPE theta. Nemotron-H's attention is unrotated; this is only what the
    /// shared config trait reports.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_model_type() -> String {
    "nemotron_h".to_string()
}
fn default_max_position_embeddings() -> i32 {
    262144
}
fn default_mamba_num_heads() -> i32 {
    64
}
fn default_mamba_head_dim() -> i32 {
    64
}
fn default_ssm_state_size() -> i32 {
    128
}
fn default_conv_kernel() -> i32 {
    4
}
fn default_n_groups() -> i32 {
    8
}
fn default_chunk_size() -> i32 {
    128
}
fn default_time_step_limit() -> (f32, f32) {
    (0.0, f32::INFINITY)
}
fn default_mlp_hidden_act() -> String {
    "relu2".to_string()
}
fn default_use_conv_bias() -> bool {
    true
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl crate::traits::ModelConfig for NemotronHConfig {
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
        self.layer_types().len() as i32
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.attention_head_dim()
    }
    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }
    fn max_position_embeddings(&self) -> i32 {
        self.max_position_embeddings
    }
    fn norm_eps(&self) -> f32 {
        self.eps()
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

impl NemotronHConfig {
    /// The kind of every layer: `M`, `*`, `-` or `E`.
    ///
    /// From `layers_block_type` when the config has one, else from
    /// `hybrid_override_pattern`, else every layer is attention. A kind this
    /// does not know is kept as it is, and [`validate`](Self::validate)
    /// refuses it.
    pub fn layer_types(&self) -> Vec<char> {
        if let Some(blocks) = &self.layers_block_type {
            return blocks
                .iter()
                .map(|kind| match kind.as_str() {
                    "mamba" | "linear_attention" => 'M',
                    "attention" | "full_attention" => '*',
                    "mlp" => '-',
                    "moe" => 'E',
                    other => other.chars().next().unwrap_or('?'),
                })
                .collect();
        }
        self.hybrid_override_pattern
            .as_ref()
            .map(|p| p.chars().collect())
            .unwrap_or_else(|| vec!['*'; self.num_hidden_layers.max(0) as usize])
    }

    /// Epsilon of every norm: `layer_norm_epsilon`, else `norm_eps`, else the
    /// reference's `1e-5`.
    pub fn eps(&self) -> f32 {
        self.layer_norm_epsilon.or(self.norm_eps).unwrap_or(1e-5)
    }

    /// Get attention head dimension.
    pub fn attention_head_dim(&self) -> i32 {
        self.head_dim
            .or(self.legacy_head_dim)
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Check if MoE is enabled.
    pub fn has_moe(&self) -> bool {
        self.n_routed_experts.is_some_and(|n| n > 0)
    }

    /// Get Mamba intermediate size.
    pub fn mamba_intermediate_size(&self) -> i32 {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// Get Mamba conv dimension.
    pub fn mamba_conv_dim(&self) -> i32 {
        self.mamba_intermediate_size() + 2 * self.n_groups * self.ssm_state_size
    }

    /// The Mamba-2 mixer this config describes.
    ///
    /// The step size is clamped to `time_step_limit`, which no release sets
    /// to anything but `(0, inf)`; `time_step_min` and `time_step_max` are
    /// where `dt_bias` was initialised, not limits. The gated norm
    /// normalizes each of the `n_groups` groups of channels on its own.
    pub fn mamba_config(&self) -> Mamba2Config {
        Mamba2Config {
            hidden_size: self.hidden_size,
            num_heads: self.mamba_num_heads,
            head_dim: self.mamba_head_dim,
            n_groups: self.n_groups,
            state_size: self.ssm_state_size,
            conv_kernel: self.conv_kernel,
            chunk_size: self.chunk_size,
            norm_group_size: self.mamba_intermediate_size() / self.n_groups.max(1),
            norm_eps: self.eps(),
            dt_min: self.time_step_limit.0,
            dt_max: self.time_step_limit.1,
            proj_bias: self.use_bias,
            conv_bias: self.use_conv_bias,
        }
    }

    /// The router's settings: a sigmoid score, the DeepSeek-V3 top-k over
    /// groups with a bias on the choice, weights renormalized (by default)
    /// and scaled.
    pub fn routing(&self) -> DeepseekRouting {
        DeepseekRouting {
            scoring: ScoringFunc::Sigmoid,
            method: TopkMethod::NoauxTc,
            n_group: self.n_group.unwrap_or(1),
            topk_group: self.topk_group.unwrap_or(1),
            top_k: self.num_experts_per_tok.unwrap_or(2),
            routed_scaling_factor: self.routed_scaling_factor.unwrap_or(1.0),
            norm_topk_prob: self.norm_topk_prob.unwrap_or(true),
        }
    }

    /// Refuse a config the model cannot be built from: an unknown layer kind,
    /// an activation other than squared ReLU, a MoE layer without MoE
    /// settings, a Mamba layer whose shape does not divide.
    pub fn validate(&self) -> Result<(), Exception> {
        let types = self.layer_types();
        if types.is_empty() {
            return Err(Exception::custom(
                "Nemotron-H: the config names no layers (hybrid_override_pattern or layers_block_type)",
            ));
        }
        if let Some(bad) = types.iter().find(|c| !matches!(c, 'M' | '*' | '-' | 'E')) {
            return Err(Exception::custom(format!(
                "Nemotron-H: unknown layer kind {bad:?} (expected M, *, - or E)"
            )));
        }
        if self.mlp_hidden_act != "relu2" {
            return Err(Exception::custom(format!(
                "Nemotron-H: mlp_hidden_act {:?} is not supported (only relu2)",
                self.mlp_hidden_act
            )));
        }
        if types.contains(&'M') {
            self.mamba_config().validate()?;
        }
        if types.contains(&'E') {
            let experts = self.n_routed_experts.unwrap_or(0);
            self.routing()
                .validate(experts)
                .map_err(|e| Exception::custom(format!("Nemotron-H config: {e}")))?;
            if self.moe_intermediate_size.is_none() {
                return Err(Exception::custom(
                    "Nemotron-H: a MoE layer needs moe_intermediate_size",
                ));
            }
        }
        if types.contains(&'*') && self.num_attention_heads % self.num_key_value_heads.max(1) != 0 {
            return Err(Exception::custom(format!(
                "Nemotron-H: {} attention heads do not divide over {} key-value heads",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        Ok(())
    }
}

// ============================================================================
// Blocks
// ============================================================================

/// Squared ReLU, `relu(x)²`.
fn relu2(x: &Array) -> Array {
    nn::relu(x).square()
}

/// An MLP: `down(relu2(up(x)))`.
#[derive(Debug)]
pub struct NemotronHMlp {
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(NemotronHMlp; up_proj, down_proj);

impl NemotronHMlp {
    pub fn new(hidden_size: i32, intermediate_size: i32, bias: bool) -> Result<Self, Exception> {
        Ok(Self {
            up_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(bias)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(bias)
                .build()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let up = Module::forward(&mut self.up_proj, x)?;
        Module::forward(&mut self.down_proj, &relu2(&up))
    }
}

/// Attention with no positional encoding: `NemotronHAttention.forward`
/// projects Q, K and V and goes straight to the attention, because the
/// interleaved Mamba layers are what supply position.
#[derive(Debug)]
pub struct NemotronHAttention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
}
impl_module_params!(NemotronHAttention; q_proj, k_proj, v_proj, o_proj);

impl NemotronHAttention {
    pub fn new(config: &NemotronHConfig) -> Result<Self, Exception> {
        let head_dim = config.attention_head_dim();
        let (heads, kv_heads) = (config.num_attention_heads, config.num_key_value_heads);
        let linear = |from: i32, to: i32| {
            nn::LinearBuilder::new(from, to)
                .bias(config.attention_bias)
                .build()
        };
        Ok(Self {
            q_proj: linear(config.hidden_size, heads * head_dim)?,
            k_proj: linear(config.hidden_size, kv_heads * head_dim)?,
            v_proj: linear(config.hidden_size, kv_heads * head_dim)?,
            o_proj: linear(heads * head_dim, config.hidden_size)?,
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let (batch, seq_len) = (x.dim(0), x.dim(1));
        let mut cache = cache;

        let heads = |a: &Array, n: i32| {
            a.reshape(&[batch, seq_len, n, self.head_dim])
                .transpose_axes(&[0, 2, 1, 3])
        };
        let q = heads(&Module::forward(&mut self.q_proj, x)?, self.num_heads);
        let k = heads(&Module::forward(&mut self.k_proj, x)?, self.num_kv_heads);
        let v = heads(&Module::forward(&mut self.v_proj, x)?, self.num_kv_heads);

        let attn_config =
            FusedAttentionConfig::new(self.num_heads, self.num_kv_heads, self.head_dim)
                .with_scale((self.head_dim as f32).sqrt().recip())
                .with_mask_type(AttentionMaskType::Causal);
        let merge = |out: Array| {
            out.transpose_axes(&[0, 2, 1, 3])
                .reshape(&[batch, seq_len, -1])
        };

        if mask.is_none()
            && let Some((cache_ref, layer_idx)) = cache.as_mut()
            && let Some(output) =
                (*cache_ref).try_compressed_attention(*layer_idx, &q, &k, &v, &attn_config)?
        {
            return Module::forward(&mut self.o_proj, &merge(output));
        }

        let (k, v) = match cache {
            Some((cache, layer_idx)) => cache.update_and_fetch(layer_idx, &k, &v)?,
            None => (k, v),
        };
        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;
        Module::forward(&mut self.o_proj, &merge(output))
    }
}

/// The MoE router: a bare gate matrix and the load-balancing bias, so the
/// parameter paths are the checkpoint's (`gate.weight`,
/// `gate.e_score_correction_bias`) and no adapter attaches to a router, whose
/// argmax picks which computation runs.
#[derive(Debug)]
pub struct NemotronHRouter {
    /// `[experts, hidden]`.
    pub weight: Array,
    /// `[experts]`. A buffer: the checkpoint carries it and the choice of
    /// experts reads it, but no gradient reaches it.
    pub e_score_correction_bias: Buffer,
    pub routing: DeepseekRouting,
}
impl_module_params!(NemotronHRouter; weight, e_score_correction_bias);

impl NemotronHRouter {
    pub fn new(
        hidden_size: i32,
        experts: i32,
        routing: DeepseekRouting,
    ) -> Result<Self, Exception> {
        routing
            .validate(experts)
            .map_err(|e| Exception::custom(format!("Nemotron-H config: {e}")))?;
        let bound = (hidden_size as f32).sqrt().recip();
        Ok(Self {
            weight: pmetal_runtime::compat::random::uniform_range(
                -bound,
                bound,
                &[experts, hidden_size],
                Dtype::Float32,
            ),
            e_score_correction_bias: Buffer(Array::zeros_f32(&[experts])),
            routing,
        })
    }

    /// `[tokens, hidden]` to each token's experts, `[tokens, top_k]` indices,
    /// and their weights, `[tokens, top_k]` float32.
    ///
    /// The logits are computed in float32 from a float32 hidden state and
    /// gate, as the reference does, so the near-ties the choice of experts
    /// turns on resolve as they do there.
    pub fn route(&self, x_flat: &Array) -> (Array, Array) {
        let f32 = Dtype::Float32.as_i32();
        let logits = x_flat.as_dtype(f32).matmul(&self.weight.as_dtype(f32).t());
        self.routing
            .route(&logits, Some(&self.e_score_correction_bias.as_dtype(f32)))
    }
}

/// A mixture-of-experts layer: squared-ReLU experts stacked over a leading
/// expert axis (`switch_mlp.fc1`, `switch_mlp.fc2`), the router, a shared
/// expert every token runs, and optionally the latent projections the experts
/// work between.
#[derive(Debug)]
pub struct NemotronHMoe {
    pub gate: NemotronHRouter,
    pub switch_mlp: PlainSwitchMlp,
    pub shared_experts: Option<NemotronHMlp>,
    pub fc1_latent_proj: Option<nn::Linear>,
    pub fc2_latent_proj: Option<nn::Linear>,
}
impl_module_params!(
    NemotronHMoe;
    gate,
    switch_mlp,
    shared_experts,
    fc1_latent_proj,
    fc2_latent_proj
);

impl NemotronHMoe {
    pub fn new(config: &NemotronHConfig) -> Result<Self, Exception> {
        let experts = config.n_routed_experts.unwrap_or(0);
        let intermediate = config.moe_intermediate_size.ok_or_else(|| {
            Exception::custom("Nemotron-H: a MoE layer needs moe_intermediate_size")
        })?;
        let width = config.moe_latent_size.unwrap_or(config.hidden_size);
        let shared_experts = if config.n_shared_experts.unwrap_or(1) > 0 {
            let size = config
                .moe_shared_expert_intermediate_size
                .unwrap_or(intermediate);
            Some(NemotronHMlp::new(
                config.hidden_size,
                size,
                config.mlp_bias,
            )?)
        } else {
            None
        };
        let latent = |from: i32, to: i32| -> Result<Option<nn::Linear>, Exception> {
            match config.moe_latent_size {
                Some(_) => Ok(Some(
                    nn::LinearBuilder::new(from, to)
                        .bias(config.mlp_bias)
                        .build()?,
                )),
                None => Ok(None),
            }
        };
        Ok(Self {
            gate: NemotronHRouter::new(config.hidden_size, experts, config.routing())?,
            switch_mlp: PlainSwitchMlp::new(experts, width, intermediate),
            shared_experts,
            fc1_latent_proj: latent(config.hidden_size, width)?,
            fc2_latent_proj: latent(width, config.hidden_size)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape().to_vec();
        let hidden = *shape.last().expect("rank >= 1");
        let flat = x.reshape(&[-1, hidden]);

        let (inds, weights) = self.gate.route(&flat);
        let into_experts = match self.fc1_latent_proj.as_mut() {
            Some(proj) => Module::forward(proj, &flat)?,
            None => flat.clone(),
        };
        // The weights stay float32 so the weighted sum of the experts' outputs
        // is taken there and rounded to the model dtype once, as the
        // reference does.
        let routed = self
            .switch_mlp
            .forward(&into_experts, &inds, &weights, relu2)?
            .as_dtype(x.dtype().as_i32());
        let routed = match self.fc2_latent_proj.as_mut() {
            Some(proj) => Module::forward(proj, &routed)?,
            None => routed,
        };
        let routed = routed.reshape(&shape);
        match self.shared_experts.as_mut() {
            Some(shared) => Ok(routed.add(&shared.forward(x)?)),
            None => Ok(routed),
        }
    }
}

/// One layer's mixer. The variants hold their parameters directly, so a
/// layer's parameter paths are the checkpoint's whichever kind it is
/// (`mixer.in_proj.weight`, `mixer.q_proj.weight`, `mixer.up_proj.weight`,
/// `mixer.gate.weight`, ...).
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per layer, never moved; boxing would add an indirection to every forward"
)]
pub enum NemotronHMixer {
    Mamba(Mamba2Mixer),
    Attention(NemotronHAttention),
    Mlp(NemotronHMlp),
    Moe(NemotronHMoe),
}

macro_rules! delegate {
    ($self:ident, $m:ident => $body:expr) => {
        match $self {
            Self::Mamba($m) => $body,
            Self::Attention($m) => $body,
            Self::Mlp($m) => $body,
            Self::Moe($m) => $body,
        }
    };
}

impl ModuleParameters for NemotronHMixer {
    fn num_parameters(&self) -> usize {
        delegate!(self, m => m.num_parameters())
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        delegate!(self, m => m.parameters())
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        delegate!(self, m => m.parameters_mut())
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        delegate!(self, m => m.trainable_parameters())
    }
    fn freeze_parameters(&mut self, recurse: bool) {
        delegate!(self, m => m.freeze_parameters(recurse))
    }
    fn unfreeze_parameters(&mut self, recurse: bool) {
        delegate!(self, m => m.unfreeze_parameters(recurse))
    }
    fn all_frozen(&self) -> Option<bool> {
        delegate!(self, m => m.all_frozen())
    }
    fn any_frozen(&self) -> Option<bool> {
        delegate!(self, m => m.any_frozen())
    }
}

impl VisitLinears for NemotronHMixer {
    fn visit_linears_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut nn::Linear)) {
        delegate!(self, m => m.visit_linears_mut(prefix, f))
    }
}

impl NemotronHMixer {
    /// The mixer of a layer of kind `kind` (`M`, `*`, `-` or `E`).
    pub fn new(config: &NemotronHConfig, kind: char) -> Result<Self, Exception> {
        match kind {
            'M' => Ok(Self::Mamba(Mamba2Mixer::new(config.mamba_config())?)),
            '*' => Ok(Self::Attention(NemotronHAttention::new(config)?)),
            '-' => Ok(Self::Mlp(NemotronHMlp::new(
                config.hidden_size,
                config.intermediate_size,
                config.mlp_bias,
            )?)),
            'E' => Ok(Self::Moe(NemotronHMoe::new(config)?)),
            other => Err(Exception::custom(format!(
                "Nemotron-H: unknown layer kind {other:?}"
            ))),
        }
    }

    /// The layer's kind: `M`, `*`, `-` or `E`.
    pub fn kind(&self) -> char {
        match self {
            Self::Mamba(_) => 'M',
            Self::Attention(_) => '*',
            Self::Mlp(_) => '-',
            Self::Moe(_) => 'E',
        }
    }

    /// Forward with the state the layer's kind reads: attention takes `mask`
    /// and `kv_cache`, a Mamba layer `mamba_cache`; the others are stateless.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        self.forward_segmented(x, mask, kv_cache, mamba_cache, None)
    }

    /// [`forward`](Self::forward) over a row that packs several sequences:
    /// `segments` (`[B, L]`, see [`segments_from_positions`]) is what keeps a
    /// Mamba layer from carrying one sequence into the next. Attention is
    /// kept apart by `mask`; the other layers act on each token alone. Only
    /// meaningful without caches.
    pub fn forward_segmented(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
        segments: Option<&Array>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Mamba(m) => match (segments, mamba_cache) {
                (Some(segments), None) => m.forward_packed(x, segments),
                (_, cache) => m.forward(x, cache),
            },
            Self::Attention(m) => m.forward(x, mask, kv_cache),
            Self::Mlp(m) => m.forward(x),
            Self::Moe(m) => m.forward(x),
        }
    }
}

/// Nemotron-H hybrid block: a pre-norm and a residual around the mixer.
#[derive(Debug)]
pub struct NemotronHBlock {
    pub norm: nn::RmsNorm,
    pub mixer: NemotronHMixer,
}
impl_module_params!(NemotronHBlock; norm, mixer);

impl NemotronHBlock {
    pub fn new(config: &NemotronHConfig, kind: char) -> Result<Self, Exception> {
        Ok(Self {
            norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.eps())
                .build()?,
            mixer: NemotronHMixer::new(config, kind)?,
        })
    }

    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        self.forward_segmented(x, mask, kv_cache, mamba_cache, None)
    }

    /// [`forward_with_cache`](Self::forward_with_cache) over a packed row; see
    /// [`NemotronHMixer::forward_segmented`].
    pub fn forward_segmented(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
        segments: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden = Module::forward(&mut self.norm, x)?;
        let output =
            self.mixer
                .forward_segmented(&hidden, mask, kv_cache, mamba_cache, segments)?;
        Ok(x.add(&output))
    }

    pub fn block_type(&self) -> char {
        self.mixer.kind()
    }
}

// ============================================================================
// Model
// ============================================================================

/// Nemotron-H model backbone.
#[derive(Debug)]
pub struct NemotronHModel {
    pub embeddings: nn::Embedding,
    pub layers: Vec<NemotronHBlock>,
    pub norm_f: nn::RmsNorm,
    pub config: NemotronHConfig,
}
impl_module_params!(NemotronHModel; embeddings, layers, norm_f);

impl NemotronHModel {
    pub fn new(config: NemotronHConfig) -> Result<Self, Exception> {
        config.validate()?;
        let embeddings = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let layers = config
            .layer_types()
            .into_iter()
            .map(|kind| NemotronHBlock::new(&config, kind))
            .collect::<Result<Vec<_>, _>>()?;
        let norm_f = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.eps())
            .build()?;
        Ok(Self {
            embeddings,
            layers,
            norm_f,
            config,
        })
    }

    /// The hidden states before the final norm.
    fn forward_layers(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        // A packed row restarts its position ids at every sequence; the Mamba
        // layers, which no attention mask reaches, reset there too.
        let segments = positions
            .filter(|_| kv_cache.is_none() && mamba_cache.is_none())
            .map(|p| segments_from_positions(p, input_ids.dim(0)));
        let mut hidden = Module::forward(&mut self.embeddings, input_ids)?;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let attends = layer.block_type() == '*';
            let kv = if attends {
                kv_cache.as_deref_mut().map(|c| (c, layer_idx))
            } else {
                None
            };
            let mamba = if layer.block_type() == 'M' {
                mamba_cache
                    .as_deref_mut()
                    .and_then(|c| c.get_mut(layer_idx))
            } else {
                None
            };
            hidden = layer.forward_segmented(
                &hidden,
                mask.filter(|_| attends),
                kv,
                mamba,
                segments.as_ref(),
            )?;
        }
        Ok(hidden)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let hidden = self.forward_layers(input_ids, mask, kv_cache, mamba_cache, None)?;
        Module::forward(&mut self.norm_f, &hidden)
    }

    /// [`forward_with_cache`](Self::forward_with_cache) for the last position
    /// alone, `[B, 1, hidden]`: every layer still runs over the whole input
    /// (the caches fill as they would), the final norm on one row.
    pub fn forward_with_cache_last(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let hidden = self.forward_layers(input_ids, mask, kv_cache, mamba_cache, None)?;
        Module::forward(
            &mut self.norm_f,
            &pmetal_runtime::decode::last_token_hidden(&hidden),
        )
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None, None)
    }

    /// Forward over a row that packs several sequences, with one position id
    /// per token `[L]` (or `[B, L]`) restarting at 0 for each. Attention is
    /// kept within a sequence by `mask`; the Mamba layers reset their state
    /// and their convolution's history at each restart. Nothing else reads
    /// the positions: Nemotron-H's attention is unrotated.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden = self.forward_layers(input_ids, mask, None, None, positions)?;
        Module::forward(&mut self.norm_f, &hidden)
    }
}

/// Nemotron-H for causal language modeling.
#[derive(Debug)]
pub struct NemotronHForCausalLM {
    pub backbone: NemotronHModel,
    pub lm_head: Option<nn::Linear>,
}
impl_module_params!(NemotronHForCausalLM; backbone, lm_head);

impl NemotronHForCausalLM {
    pub fn new(config: NemotronHConfig) -> Result<Self, Exception> {
        let tie_weights = config.tie_word_embeddings;
        let (vocab_size, hidden_size) = (config.vocab_size, config.hidden_size);
        let backbone = NemotronHModel::new(config)?;
        let lm_head = if tie_weights {
            None
        } else {
            Some(
                nn::LinearBuilder::new(hidden_size, vocab_size)
                    .bias(false)
                    .build()?,
            )
        };
        Ok(Self { backbone, lm_head })
    }

    /// Get config reference.
    pub fn config(&self) -> &NemotronHConfig {
        &self.backbone.config
    }

    /// Project hidden states to vocabulary logits, through the LM head when
    /// the checkpoint has one and the tied embedding otherwise.
    fn lm_head_forward(&mut self, hidden: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(lm_head) => Module::forward(lm_head, hidden),
            None => Ok(self.backbone.embeddings.as_linear(hidden)),
        }
    }

    /// Forward pass producing logits.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None, None)
    }

    /// Forward pass with optional KV cache and Mamba cache.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let hidden = self
            .backbone
            .forward_with_cache(input_ids, mask, kv_cache, mamba_cache)?;
        self.lm_head_forward(&hidden)
    }

    /// Logits for a row that packs several sequences; see
    /// [`NemotronHModel::forward_with_positions`].
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden = self
            .backbone
            .forward_with_positions(input_ids, mask, positions)?;
        self.lm_head_forward(&hidden)
    }

    /// [`forward_with_cache`](Self::forward_with_cache) returning the last
    /// position's logits, `[B, 1, vocab]`: the LM head runs on one hidden state
    /// per sequence instead of every prompt token's.
    pub fn forward_with_cache_last(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let hidden =
            self.backbone
                .forward_with_cache_last(input_ids, mask, kv_cache, mamba_cache)?;
        self.lm_head_forward(&hidden)
    }

    /// Evaluate all parameters to materialize them on device.
    pub fn eval(&self) -> Result<(), Exception> {
        pmetal_runtime::compat::ModuleParametersExt::eval(self)
    }
}

/// Bring a released Nemotron-H checkpoint to the layout the model holds: each
/// MoE layer's experts stacked into `mixer.switch_mlp.{fc1,fc2}` (what an MLX
/// conversion stores) and the depthwise convolutions in the runtime's
/// `[channels, kernel, 1]`. Already converted tensors pass through.
pub fn sanitize_nemotron_h_checkpoint(
    weights: HashMap<String, Array>,
) -> Result<HashMap<String, Array>, Exception> {
    let mut weights = crate::loader::experts::stack_plain_experts("nemotron_h", weights)?;
    crate::common::mamba2::conv_weights_to_runtime_layout(&mut weights);
    Ok(weights)
}

/// Assign a released Nemotron-H checkpoint's tensors (already unpacked to dense
/// by the shared reader) to `model`, all of it or not at all: the experts are
/// stacked and the convolutions reshaped
/// ([`sanitize_nemotron_h_checkpoint`]), then every tensor must find a
/// parameter and every parameter a tensor, bar what
/// [`ModelArchitecture::load_policy`](crate::dispatcher::ModelArchitecture::load_policy)
/// exempts (the bundled `mtp.*` head).
pub fn assign_nemotron_h_weights(
    model: &mut NemotronHForCausalLM,
    weights: HashMap<String, Array>,
) -> Result<(), crate::loader::LoadError> {
    let weights = sanitize_nemotron_h_checkpoint(weights)
        .map_err(|e| crate::loader::LoadError::SafeTensors(e.to_string()))?;
    crate::loader::assign_weights(
        model,
        weights,
        &crate::dispatcher::ModelArchitecture::NemotronH.load_policy(),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn small_config() -> NemotronHConfig {
        NemotronHConfig {
            model_type: "nemotron_h".to_string(),
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 4,
            max_position_embeddings: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            attention_bias: false,
            head_dim: Some(32),
            legacy_head_dim: None,
            mamba_num_heads: 4,
            mamba_head_dim: 32,
            ssm_state_size: 16,
            conv_kernel: 4,
            n_groups: 2,
            chunk_size: 16,
            time_step_limit: (0.0, f32::INFINITY),
            time_step_min: None,
            time_step_max: None,
            mlp_bias: false,
            mlp_hidden_act: "relu2".to_string(),
            layer_norm_epsilon: Some(1e-5),
            norm_eps: None,
            use_bias: false,
            use_conv_bias: true,
            tie_word_embeddings: true,
            hybrid_override_pattern: Some("M*-E".to_string()),
            layers_block_type: None,
            moe_intermediate_size: Some(64),
            moe_shared_expert_intermediate_size: Some(96),
            moe_latent_size: None,
            n_group: None,
            n_routed_experts: Some(4),
            n_shared_experts: Some(1),
            topk_group: None,
            num_experts_per_tok: Some(2),
            norm_topk_prob: Some(true),
            routed_scaling_factor: Some(2.5),
            rope_theta: 10000.0,
        }
    }

    #[test]
    fn the_layer_kinds_come_from_the_pattern_or_the_block_list() {
        let config = small_config();
        assert_eq!(config.layer_types(), vec!['M', '*', '-', 'E']);

        let listed = NemotronHConfig {
            hybrid_override_pattern: None,
            layers_block_type: Some(
                [
                    "mamba",
                    "moe",
                    "attention",
                    "mlp",
                    "linear_attention",
                    "full_attention",
                ]
                .map(String::from)
                .to_vec(),
            ),
            ..config
        };
        assert_eq!(listed.layer_types(), vec!['M', 'E', '*', '-', 'M', '*']);
        listed.validate().unwrap();
    }

    #[test]
    fn a_config_it_cannot_build_is_refused_by_name() {
        let mut config = small_config();
        config.hybrid_override_pattern = Some("M?E".to_string());
        assert!(config.validate().unwrap_err().to_string().contains("'?'"));

        let mut config = small_config();
        config.mlp_hidden_act = "silu".to_string();
        assert!(config.validate().unwrap_err().to_string().contains("silu"));

        let mut config = small_config();
        config.num_experts_per_tok = Some(9);
        assert!(NemotronHForCausalLM::new(config).is_err());
    }

    #[test]
    fn the_released_spellings_parse() {
        // Nemotron 3.5: `norm_eps`, a block list, no `num_hidden_layers`.
        let config: NemotronHConfig = json5::from_str(
            r#"{
                "vocab_size": 10, "hidden_size": 8, "intermediate_size": 16,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "norm_eps": 1e-6, "attention_head_dim": 4,
                "layers_block_type": ["mamba", "moe"],
                "time_step_limit": [0.0, Infinity]
            }"#,
        )
        .unwrap();
        assert_eq!(config.eps(), 1e-6);

        // Nemotron 3 Nano states both spellings of the epsilon.
        let both: NemotronHConfig = json5::from_str(
            r#"{
                "vocab_size": 10, "hidden_size": 8, "intermediate_size": 16,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "norm_eps": 1e-5, "layer_norm_epsilon": 1e-5, "head_dim": 4,
                "attention_head_dim": 4, "hybrid_override_pattern": "ME"
            }"#,
        )
        .unwrap();
        assert_eq!(both.eps(), 1e-5);
        assert_eq!(config.attention_head_dim(), 4);
        assert_eq!(config.layer_types(), vec!['M', 'E']);
        assert!(config.time_step_limit.1.is_infinite());
    }

    /// FP8 weight storage reaches every projection of every kind of layer
    /// (the generic pass, which reads the weights back through `Linear`) and
    /// leaves the router, which is read as a float matrix, alone.
    #[test]
    fn fp8_storage_covers_the_projections_and_spares_the_router() {
        use pmetal_runtime::compat::ModuleParametersExt;

        pmetal_runtime::compat::random::seed(3);
        let mut model = NemotronHForCausalLM::new(small_config()).unwrap();
        let ids = Array::from_slice(&[1i32, 5, 9, 2, 7, 3, 8, 4], &[1, 8]);
        let dense = model.forward(&ids, None).unwrap();
        crate::fp8_utils::quantize_model_linears(&mut model).unwrap();
        let packed = model.forward(&ids, None).unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();

        let params = model.flatten_params();
        let dtype = |key: &str| params[key].dtype();
        for packed_key in [
            "backbone.layers.0.mixer.in_proj.weight",
            "backbone.layers.0.mixer.out_proj.weight",
            "backbone.layers.1.mixer.q_proj.weight",
            "backbone.layers.2.mixer.up_proj.weight",
            "backbone.layers.3.mixer.shared_experts.up_proj.weight",
        ] {
            assert_eq!(dtype(packed_key), Dtype::Uint8, "{packed_key}");
        }
        assert_eq!(
            dtype("backbone.layers.3.mixer.gate.weight"),
            Dtype::Float32,
            "the router is not stored in FP8"
        );
        assert_eq!(packed.shape(), dense.shape());
        let finite = packed.abs().max(None).item::<f32>();
        assert!(finite.is_finite() && finite > 0.0, "{finite}");
    }
}
