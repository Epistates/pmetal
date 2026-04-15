//! DFlash block-diffusion draft model.
//!
//! DFlash (Chen et al., 2026) is a block-diffusion drafter for flash
//! speculative decoding. Instead of autoregressively proposing tokens one at
//! a time, a small transformer takes a block of mask-token noise embeddings
//! plus intermediate hidden states tapped from the target model's previous
//! forward pass and proposes every token in the block in a single pass.
//!
//! The target model then verifies the entire block with one forward pass
//! (via [`crate::architectures::qwen3::Qwen3ForCausalLM::forward_with_capture`]
//! or the Qwen3.5 equivalent), accepts the longest matching prefix, and
//! feeds the next draft with the just-captured verifier hidden states.
//!
//! # Weights
//!
//! This model loads checkpoints from the `z-lab/*-DFlash*` family on Hugging
//! Face (currently `z-lab/Qwen3-4B-DFlash-b16` and
//! `z-lab/Qwen3.5-4B-DFlash`). Weight naming follows the upstream Python
//! implementation at `dflash_mlx/draft.py`:
//! * `layers.{i}.self_attn.{q,k,v,o}_proj.weight`
//! * `layers.{i}.self_attn.{q,k}_norm.weight`
//! * `layers.{i}.{input_layernorm,post_attention_layernorm}.weight`
//! * `layers.{i}.mlp.{gate,up,down}_proj.weight`
//! * `fc.weight` (shape `[hidden, L * hidden]`)
//! * `hidden_norm.weight`, `norm.weight`
//!
//! The `dflash_config.target_layer_ids` list in the checkpoint's
//! `config.json` enumerates which target-model layers the drafter taps. The
//! verifier must request hidden-state capture at exactly those indices.

use std::collections::HashMap;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParameters, ModuleParametersExt, Param, nn, ops,
};
use pmetal_bridge::impl_module_params;
use serde::{Deserialize, Serialize};

use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopeScaling, apply_rope},
};
use pmetal_mlx::kv_cache::KVCache;

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_block_size() -> i32 {
    16
}

/// Extra DFlash-specific config section (stored under `dflash_config` in
/// the upstream config.json).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DFlashExtras {
    /// Target-model layer indices whose hidden states the drafter consumes.
    pub target_layer_ids: Vec<i32>,
    /// Token id of the `[MASK]` token used for block-diffusion noise.
    pub mask_token_id: i32,
}

/// Configuration for [`DFlashDraftModel`].
///
/// Matches the Python `DraftArgs` struct in `dflash_mlx/draft.py`. Fields
/// that have sensible defaults in the upstream implementation are given
/// `#[serde(default)]` so a config.json that omits them still loads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DFlashDraftConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default)]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub head_dim: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<std::collections::HashMap<String, serde_json::Value>>,
    #[serde(default = "default_block_size")]
    pub block_size: i32,
    /// DFlash-specific hyperparameters.
    pub dflash_config: DFlashExtras,
}

impl DFlashDraftConfig {
    /// Target layer indices as `usize` — the form needed by
    /// [`pmetal_mlx::speculative::SpecCapture::with_layers`].
    pub fn target_layer_ids(&self) -> Vec<usize> {
        self.dflash_config
            .target_layer_ids
            .iter()
            .map(|&id| id as usize)
            .collect()
    }

    /// Number of layers the drafter conditions on.
    pub fn num_target_layers(&self) -> usize {
        self.dflash_config.target_layer_ids.len()
    }
}

// ----------------------------------------------------------------------------
// MLP
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct DFlashMlp {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(DFlashMlp; gate_proj, up_proj, down_proj);

impl DFlashMlp {
    pub fn new(config: &DFlashDraftConfig) -> Result<Self, Exception> {
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

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        Ok(self.down_proj.forward(&nn::silu(&gate).multiply(&up)))
    }
}

// ----------------------------------------------------------------------------
// Attention
// ----------------------------------------------------------------------------

/// DFlash cross-attention.
///
/// Structurally this is a standard Q/K/V attention, but the K/V projection
/// input is the concatenation `[target_hidden || query_hidden_states]`
/// rather than just the query's own hidden states. Queries only see the
/// positions they contribute; keys and values are taken from every position
/// in the concatenated sequence. RoPE is applied with an offset that
/// accounts for the target-hidden prefix length.
#[derive(Debug)]
pub struct DFlashAttention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_scale: f32,
    pub effective_base: f32,
}
impl_module_params!(DFlashAttention; q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

impl DFlashAttention {
    pub fn new(config: &DFlashDraftConfig) -> Result<Self, Exception> {
        let head_dim = config.head_dim;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;

        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(|map| RopeScaling::from_config_map(map))
            .unwrap_or(RopeScaling::None);
        let rope_scale = rope_scaling.scale();
        let effective_base = rope_scaling.effective_base(config.rope_theta, head_dim);

        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, config.hidden_size)
            .bias(config.attention_bias)
            .build()?;
        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_scale,
            effective_base,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        target_hidden: &Array,
        mut cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let batch = hidden_states.dim(0);
        let query_len = hidden_states.dim(1);
        let context_len = target_hidden.dim(1);

        // Queries come from the draft's own hidden states only.
        let queries = self.q_proj.forward(hidden_states);
        let mut queries = queries.reshape(&[batch, query_len, self.n_heads, self.head_dim]);
        queries = self.q_norm.forward(&queries);
        let queries = queries.transpose_axes(&[0, 2, 1, 3]);

        // K/V projection input is [target_hidden || hidden_states].
        let kv_input = ops::concatenate_axis(&[target_hidden, hidden_states], 1);
        let kv_len = context_len + query_len;
        let keys = self.k_proj.forward(&kv_input);
        let values = self.v_proj.forward(&kv_input);

        let mut keys = keys.reshape(&[batch, kv_len, self.n_kv_heads, self.head_dim]);
        keys = self.k_norm.forward(&keys);
        let keys = keys.transpose_axes(&[0, 2, 1, 3]);
        let values = values
            .reshape(&[batch, kv_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        // RoPE: queries start at offset `cache.offset + context_len`, keys
        // start at offset `cache.offset` (the context rows sit at the front
        // of the KV sequence). This matches dflash_mlx/draft.py:141-148.
        let cache_offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let queries = apply_rope(
            &queries,
            self.head_dim,
            false,
            self.effective_base,
            self.rope_scale,
            cache_offset + context_len,
        )?;
        let keys = apply_rope(
            &keys,
            self.head_dim,
            false,
            self.effective_base,
            self.rope_scale,
            cache_offset,
        )?;

        let (keys, values) = if let Some((cache_ref, layer_idx)) = cache.as_mut() {
            (*cache_ref).update_and_fetch(*layer_idx, &keys, &values)?
        } else {
            (keys, values)
        };

        // DFlash draft runs with `mask_mode = "none"` by default — every
        // query position can see every key position (including future ones
        // from the same proposed block). The upstream implementation only
        // flips this to causal for debugging.
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(AttentionMaskType::None);
        let output = fused_sdpa(&queries, &keys, &values, &attn_config, None)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, query_len, self.n_heads * self.head_dim]);
        Ok(self.o_proj.forward(&output))
    }
}

// ----------------------------------------------------------------------------
// Decoder layer
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct DFlashDecoderLayer {
    pub input_layernorm: nn::RmsNorm,
    pub self_attn: DFlashAttention,
    pub post_attention_layernorm: nn::RmsNorm,
    pub mlp: DFlashMlp,
}
impl_module_params!(DFlashDecoderLayer; input_layernorm, self_attn, post_attention_layernorm, mlp);

impl DFlashDecoderLayer {
    pub fn new(config: &DFlashDraftConfig) -> Result<Self, Exception> {
        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let self_attn = DFlashAttention::new(config)?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let mlp = DFlashMlp::new(config)?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        target_hidden: &Array,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let residual = hidden_states.clone();
        let normed = self.input_layernorm.forward(hidden_states);
        let attn = self.self_attn.forward(&normed, target_hidden, cache)?;
        let hidden_states = residual.add(&attn);

        let residual = hidden_states.clone();
        let normed = self.post_attention_layernorm.forward(&hidden_states);
        let mlp = self.mlp.forward(&normed)?;
        Ok(residual.add(&mlp))
    }
}

// ----------------------------------------------------------------------------
// Top-level draft model
// ----------------------------------------------------------------------------

/// DFlash draft model.
///
/// Unlike a standard causal LM the draft does not own token embeddings or
/// an lm_head — the DFlash pipeline shares both with the target model. The
/// draft is therefore a stack of decoder layers plus the `fc` + `hidden_norm`
/// projection that conditions on target hidden states.
#[derive(Debug)]
pub struct DFlashDraftModel {
    pub layers: Vec<DFlashDecoderLayer>,
    /// Projects `[B, T, L * hidden]` target hidden states down to `[B, T, hidden]`.
    pub fc: nn::Linear,
    /// RMSNorm applied to the projected target hidden states.
    pub hidden_norm: nn::RmsNorm,
    /// Final RMSNorm over the draft hidden states.
    pub norm: nn::RmsNorm,
    pub config: DFlashDraftConfig,
}
impl_module_params!(DFlashDraftModel; layers, fc, hidden_norm, norm);

impl DFlashDraftModel {
    pub fn new(config: DFlashDraftConfig) -> Result<Self, Exception> {
        let l = config.num_target_layers() as i32;
        if l == 0 {
            return Err(Exception::custom(
                "DFlashDraftModel requires at least one target_layer_id",
            ));
        }

        let layers = (0..config.num_hidden_layers)
            .map(|_| DFlashDecoderLayer::new(&config))
            .collect::<Result<Vec<_>, _>>()?;

        let fc = nn::LinearBuilder::new(l * config.hidden_size, config.hidden_size)
            .bias(false)
            .build()?;
        let hidden_norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            layers,
            fc,
            hidden_norm,
            norm,
            config,
        })
    }

    /// DFlash block size — how many tokens the drafter proposes per step.
    pub fn block_size(&self) -> usize {
        self.config.block_size as usize
    }

    /// Token id used to fill proposal slots in the noise embedding.
    pub fn mask_token_id(&self) -> i32 {
        self.config.dflash_config.mask_token_id
    }

    /// Number of layers in the draft stack — handy when constructing
    /// per-layer `KVCache`s.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass.
    ///
    /// * `noise_embedding`: `[B, T_block, hidden]` — the target model's
    ///   token embedding of the mask-token block. The DFlash pipeline
    ///   computes this via `target.embed_tokens(block_input)` and passes
    ///   the result in directly, so the draft does not need its own token
    ///   embeddings.
    /// * `target_hidden`: `[B, T_ctx, L * hidden]` — the hidden states
    ///   captured from the target model's most recent forward pass,
    ///   concatenated along the hidden dimension in the order of
    ///   [`DFlashExtras::target_layer_ids`].
    /// * `cache`: optional per-layer KV cache. Must have one entry per
    ///   layer; pass `None` for cacheless operation.
    pub fn forward(
        &mut self,
        noise_embedding: &Array,
        target_hidden: &Array,
        mut cache: Option<&mut [KVCache]>,
    ) -> Result<Array, Exception> {
        // Project target_hidden [B, T, L*hidden] → [B, T, hidden] and norm.
        let projected = self.fc.forward(target_hidden);
        let target_hidden = self.hidden_norm.forward(&projected);

        let mut hidden = noise_embedding.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache
                .as_deref_mut()
                .and_then(|caches| caches.get_mut(i))
                .map(|c| (c, 0_usize));
            hidden = layer.forward(&hidden, &target_hidden, layer_cache)?;
        }
        Ok(self.norm.forward(&hidden))
    }
}

// ----------------------------------------------------------------------------
// Weight loading
// ----------------------------------------------------------------------------

impl DFlashDraftModel {
    /// Load weights from a flat `name → tensor` map.
    ///
    /// Accepts both the upstream dflash_mlx naming (`layers.{i}.…`) and a
    /// `model.layers.{i}.…` prefixed variant so a safetensors file that
    /// follows either convention drops in.
    pub fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
    ) -> Result<LoadReport, Exception> {
        let mut report = LoadReport::default();
        for (name, weight) in weights {
            let stripped = name.strip_prefix("model.").unwrap_or(name.as_str());
            match stripped {
                "fc.weight" => {
                    self.fc.weight = Param::new(weight.clone());
                    report.loaded += 1;
                }
                "hidden_norm.weight" => {
                    self.hidden_norm.weight = Param::new(weight.clone());
                    report.loaded += 1;
                }
                "norm.weight" => {
                    self.norm.weight = Param::new(weight.clone());
                    report.loaded += 1;
                }
                other if other.starts_with("layers.") => {
                    // layers.{i}.{rest}
                    let parts: Vec<&str> = other.splitn(3, '.').collect();
                    if parts.len() < 3 {
                        report.skipped.push(name.clone());
                        continue;
                    }
                    let Ok(layer_idx) = parts[1].parse::<usize>() else {
                        report.skipped.push(name.clone());
                        continue;
                    };
                    if layer_idx >= self.layers.len() {
                        report.skipped.push(name.clone());
                        continue;
                    }
                    if assign_layer_weight(&mut self.layers[layer_idx], parts[2], weight.clone()) {
                        report.loaded += 1;
                    } else {
                        report.skipped.push(name.clone());
                    }
                }
                _ => report.skipped.push(name.clone()),
            }
        }
        Ok(report)
    }
}

fn assign_layer_weight(layer: &mut DFlashDecoderLayer, suffix: &str, weight: Array) -> bool {
    match suffix {
        "input_layernorm.weight" => {
            layer.input_layernorm.weight = Param::new(weight);
        }
        "post_attention_layernorm.weight" => {
            layer.post_attention_layernorm.weight = Param::new(weight);
        }
        "self_attn.q_proj.weight" => layer.self_attn.q_proj.weight = Param::new(weight),
        "self_attn.k_proj.weight" => layer.self_attn.k_proj.weight = Param::new(weight),
        "self_attn.v_proj.weight" => layer.self_attn.v_proj.weight = Param::new(weight),
        "self_attn.o_proj.weight" => layer.self_attn.o_proj.weight = Param::new(weight),
        "self_attn.q_norm.weight" => layer.self_attn.q_norm.weight = Param::new(weight),
        "self_attn.k_norm.weight" => layer.self_attn.k_norm.weight = Param::new(weight),
        "mlp.gate_proj.weight" => layer.mlp.gate_proj.weight = Param::new(weight),
        "mlp.up_proj.weight" => layer.mlp.up_proj.weight = Param::new(weight),
        "mlp.down_proj.weight" => layer.mlp.down_proj.weight = Param::new(weight),
        _ => return false,
    }
    true
}

/// Summary of a [`DFlashDraftModel::load_weights`] call.
#[derive(Debug, Default, Clone)]
pub struct LoadReport {
    /// Number of weights successfully assigned.
    pub loaded: usize,
    /// Names of weights that did not match a known parameter.
    pub skipped: Vec<String>,
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn tiny_config() -> DFlashDraftConfig {
        DFlashDraftConfig {
            model_type: "dflash_qwen3".to_string(),
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            vocab_size: 128,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            head_dim: 8,
            tie_word_embeddings: false,
            attention_bias: false,
            rope_scaling: None,
            block_size: 4,
            dflash_config: DFlashExtras {
                target_layer_ids: vec![1, 3],
                mask_token_id: 7,
            },
        }
    }

    #[test]
    #[serial]
    fn test_dflash_draft_forward_shape() {
        let config = tiny_config();
        let hidden = config.hidden_size;
        let block = config.block_size;
        let num_target = config.num_target_layers() as i32;

        let mut model = DFlashDraftModel::new(config).unwrap();

        // Simulate the DFlash pipeline: noise embedding is [B, block, hidden]
        // and target_hidden is [B, ctx, num_target * hidden].
        let noise = pmetal_bridge::compat::random::normal(
            &[1, block, hidden],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let target_hidden = pmetal_bridge::compat::random::normal(
            &[1, 6, num_target * hidden],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let out = model.forward(&noise, &target_hidden, None).unwrap();
        assert_eq!(out.shape(), &[1, block, hidden]);
    }

    #[test]
    #[serial]
    fn test_dflash_draft_block_size_and_mask_token() {
        let model = DFlashDraftModel::new(tiny_config()).unwrap();
        assert_eq!(model.block_size(), 4);
        assert_eq!(model.mask_token_id(), 7);
        assert_eq!(model.num_layers(), 2);
    }

    #[test]
    fn test_dflash_draft_config_requires_target_layers() {
        let mut config = tiny_config();
        config.dflash_config.target_layer_ids.clear();
        let err = DFlashDraftModel::new(config).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("target_layer_id"),
            "expected target_layer_id error, got: {msg}"
        );
    }
}
