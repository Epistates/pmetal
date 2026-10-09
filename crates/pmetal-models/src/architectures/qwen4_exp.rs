//! Qwen4-Exp (`model_type: "qwen4_exp"`): the text tower of Qwen3.8-Flash-Next.
//!
//! A Qwen 3.5-style hybrid (Gated DeltaNet linear attention, a gated
//! full-attention layer every `full_attention_interval` layers, a routed MoE
//! with a sigmoid-gated shared expert) with four additions:
//!
//! - **Hyper-connections.** The residual stream is `hc_count` copies of the
//!   hidden state wide. Before each mixer and each MoE a [`Qwen4ExpGatedResidual`]
//!   normalizes the streams, mixes them down to one input with learned
//!   per-channel weights, and afterwards injects the block's output back into
//!   every stream with a learned per-stream weight. There are no pre-norms and
//!   no final norm: the LM head reads the last mix directly.
//! - **The QSA indexer.** Each attention layer pools its keys into blocks of
//!   `indexer_compress_ratio` tokens, scores the blocks against a small
//!   multi-head query, and lets a token attend only to the best
//!   `indexer_budget / indexer_compress_ratio` complete blocks plus the
//!   incomplete tail. Below that budget it keeps everything, so short contexts
//!   run plain causal attention ([`Qwen4ExpIndexer::select`] returns `None`).
//! - **PLE n-gram embeddings.** On the layers listed in `ple_layer_ids`, the
//!   bigrams and trigrams ending at each token are hashed into a 320M-row table
//!   (51B parameters in the release) and injected into every stream through a
//!   query-gated value and a dilated depthwise convolution ([`Qwen4ExpPle`]).
//!   The rows are addressed on the CPU, so the table can be served from disk
//!   ([`NgramTable`]).
//! - **A sigmoid GDN output gate** (`output_gate_type`).
//!
//! The GDN, gated-attention and MoE blocks are `qwen3_next`'s own, built from a
//! [`Qwen3NextConfig`] view of this config ([`Qwen4ExpConfig::qwen3_next_view`]).
//!
//! # Cache layout
//!
//! The [`KVCache`] holds attention keys and values per layer, as for any model.
//! The [`MambaCache`] (see [`Qwen4ExpCacheLayout`]) holds one entry per decoder
//! layer plus two per PLE layer:
//!
//! - linear-attention layer `i`: the GDN conv and recurrent state, as Qwen 3.5;
//! - attention layer `i`: the indexer's keys. `conv_state` holds the raw keys of
//!   the incomplete trailing block, `ssm_state` the pooled, normalized and
//!   rotated key of every complete block;
//! - `num_layers + 2k`: PLE layer `k`'s dilated short-conv state;
//! - `num_layers + 2k + 1`: PLE layer `k`'s last `ngram_size - 1` token ids.
//!
//! Reference: `transformers.models.qwen4_exp` (the definition the released
//! weights are trained against). The vision tower and the bundled MTP head are
//! not ported; the loader skips them by name.

use std::collections::{HashMap, HashSet};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pmetal_bridge::compat::ops::{slice_axis, slice_axis_from, slice_last_from, slice_last_to};
use pmetal_bridge::compat::{
    Array, Dtype, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, ModuleParametersExt,
    Param, Parameter, VisitLinears, child_path, fast, nn, ops,
};
use pmetal_bridge::impl_module_params;
use pmetal_mlx::kernels::rope::{RopePositions, rope};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig, MambaCache, MambaCacheEntry};
use pmetal_mlx::speculative::SpecCapture;
use serde::{Deserialize, Serialize};

use super::qwen3_next::{
    ExpertOffloadAttachment, GateActivation, PackedRoutedExperts, Qwen3NextAttention,
    Qwen3NextConfig, Qwen3NextGatedDeltaNet, Qwen3NextRoutedExpertMode, Qwen3NextSanitizeOptions,
    Qwen3NextSparseMoeBlock, RopeParameters, attach_expert_offload, sanitize_weights,
};
use super::utils::LoadReport;
use crate::loader::LoadError;
use crate::traits::ModelConfig;
use pmetal_bridge::native_weight::LayerWeight;

// ============================================================================
// Configuration
// ============================================================================

/// `eos_token_id` as configs spell it: one id or a list of them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenIds {
    One(i64),
    Many(Vec<i64>),
}

impl TokenIds {
    /// The first id, which is the one the n-gram hash pads with.
    pub fn first(&self) -> Option<i64> {
        match self {
            Self::One(id) => Some(*id),
            Self::Many(ids) => ids.first().copied(),
        }
    }
}

/// Qwen4-Exp text configuration (`qwen4_exp_text`).
///
/// Defaults are the reference configuration class's. [`Qwen4ExpConfig::from_json`]
/// applies the reference's normalisation (`full_attention` layers are indexed
/// attention) and refuses every value this port does not compute.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Qwen4ExpConfig {
    pub model_type: String,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub hidden_act: String,
    pub max_position_embeddings: i32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub rope_parameters: Option<RopeParameters>,
    /// Legacy spelling of `rope_parameters`; when set it replaces them, as in
    /// the reference.
    pub rope_scaling: Option<RopeParameters>,
    /// Top-level fallbacks the reference folds into `rope_parameters`.
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f32>,
    pub layer_types: Option<Vec<String>>,
    pub full_attention_interval: i32,

    pub linear_conv_kernel_dim: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_num_key_heads: i32,
    pub linear_num_value_heads: i32,
    pub output_gate_type: Option<String>,

    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub norm_topk_prob: bool,

    pub hc_count: i32,
    pub hc_lowrank: i32,

    pub ple_layer_ids: Vec<i32>,
    pub ple_embed_dim: Option<i32>,
    pub ple_conv_kernel_size: i32,
    pub ngram_size: i32,
    pub heads_per_ngram: i32,
    pub ngram_vocab_size_base: i64,
    pub make_ngram_vocab_size_divisible_by: i64,
    pub seed: i64,
    pub split_ngram_parts: i32,

    pub indexer_n_heads: Option<i32>,
    pub indexer_kv_heads: Option<i32>,
    pub indexer_head_dim: Option<i32>,
    pub indexer_budget: Option<i32>,
    pub indexer_compress_ratio: Option<i32>,

    pub eos_token_id: Option<TokenIds>,
    /// Bundled MTP predictor layers. Not loaded; the reference ignores them too.
    pub mtp_num_hidden_layers: Option<i32>,
}

impl Default for Qwen4ExpConfig {
    fn default() -> Self {
        Self {
            model_type: "qwen4_exp_text".to_string(),
            vocab_size: 248_320,
            hidden_size: 2048,
            num_hidden_layers: 40,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 256,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 32_768,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            tie_word_embeddings: false,
            rope_parameters: None,
            rope_scaling: None,
            rope_theta: None,
            partial_rotary_factor: None,
            layer_types: None,
            full_attention_interval: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            output_gate_type: None,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            num_experts: 512,
            num_experts_per_tok: 10,
            norm_topk_prob: true,
            hc_count: 4,
            hc_lowrank: 320,
            ple_layer_ids: Vec::new(),
            ple_embed_dim: None,
            ple_conv_kernel_size: 4,
            ngram_size: 3,
            heads_per_ngram: 8,
            ngram_vocab_size_base: 20_000_000,
            make_ngram_vocab_size_divisible_by: 128,
            seed: 1234,
            split_ngram_parts: 512,
            indexer_n_heads: None,
            indexer_kv_heads: None,
            indexer_head_dim: None,
            indexer_budget: None,
            indexer_compress_ratio: None,
            eos_token_id: None,
            mtp_num_hidden_layers: None,
        }
    }
}

/// The QSA indexer's geometry, resolved from the five `indexer_*` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerGeometry {
    pub n_heads: i32,
    pub head_dim: i32,
    pub budget: i32,
    pub compress_ratio: i32,
}

impl IndexerGeometry {
    /// Complete blocks a query keeps once it can see more than that many.
    pub fn block_topk(&self) -> i32 {
        self.budget / self.compress_ratio
    }
}

impl Qwen4ExpConfig {
    /// Parse a text config (the `text_config` of a wrapper, already unwrapped)
    /// and validate it.
    pub fn from_json(text_config: &str) -> Result<Self, Exception> {
        let mut config: Self =
            json5::from_str(text_config).map_err(|e| Exception::custom(e.to_string()))?;
        config.normalize();
        config.validate()?;
        Ok(config)
    }

    /// The reference's `__post_init__`: derive missing layer types from
    /// `full_attention_interval`, read `full_attention` as indexed attention,
    /// sort and dedupe the PLE layer ids, default `ple_embed_dim`.
    pub fn normalize(&mut self) {
        let layer_types = match self.layer_types.take() {
            Some(types) => types,
            None => (0..self.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % self.full_attention_interval.max(1) == 0 {
                        "indexed_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect(),
        };
        self.layer_types = Some(
            layer_types
                .into_iter()
                .map(|t| {
                    if t == "full_attention" {
                        "indexed_attention".to_string()
                    } else {
                        t
                    }
                })
                .collect(),
        );
        self.ple_layer_ids.sort_unstable();
        self.ple_layer_ids.dedup();
        if self.ple_embed_dim.is_none() {
            self.ple_embed_dim = Some(self.hidden_size);
        }
    }

    /// Refuse every value whose math this port does not implement.
    pub fn validate(&self) -> Result<(), Exception> {
        let mut problems: Vec<String> = Vec::new();
        let layer_types = self.layer_types();
        if layer_types.len() != self.num_hidden_layers as usize {
            problems.push(format!(
                "layer_types has {} entries for {} layers",
                layer_types.len(),
                self.num_hidden_layers
            ));
        }
        for t in &layer_types {
            if t != "linear_attention" && t != "indexed_attention" {
                problems.push(format!("unsupported layer type {t:?}"));
            }
        }
        if !matches!(
            self.hidden_act.to_ascii_lowercase().as_str(),
            "silu" | "swish"
        ) {
            problems.push(format!(
                "hidden_act {:?}: the GDN conv, experts and shared expert are SiLU here",
                self.hidden_act
            ));
        }
        if let Err(e) = self.gate_activation() {
            problems.push(e.to_string());
        }
        if self.hc_count <= 1 {
            problems.push(format!("hc_count must be > 1, got {}", self.hc_count));
        }
        if self.num_experts <= 0
            || self.num_experts_per_tok <= 0
            || self.num_experts_per_tok > self.num_experts
        {
            problems.push(format!(
                "num_experts_per_tok {} must be in [1, num_experts {}]",
                self.num_experts_per_tok, self.num_experts
            ));
        }
        if self.moe_intermediate_size <= 0 || self.shared_expert_intermediate_size <= 0 {
            problems.push("moe and shared expert intermediate sizes must be > 0".to_string());
        }
        match self.rope() {
            Ok(rope) => {
                let rotary = (self.head_dim as f32 * rope.partial_rotary_factor) as i32;
                if let Ok(indexer) = self.indexer() {
                    if rotary > indexer.head_dim {
                        problems.push(format!(
                            "rotary dims {rotary} do not fit the indexer head dim {}",
                            indexer.head_dim
                        ));
                    }
                }
            }
            Err(e) => problems.push(e.to_string()),
        }
        match self.indexer() {
            Ok(indexer) => {
                if indexer.budget % indexer.compress_ratio != 0 {
                    problems.push(format!(
                        "indexer_budget {} is not a multiple of indexer_compress_ratio {}",
                        indexer.budget, indexer.compress_ratio
                    ));
                }
            }
            Err(e) => problems.push(e.to_string()),
        }
        if !self.ple_layer_ids.is_empty() {
            let heads = self.ngram_heads();
            let embed = self.ple_embed_dim();
            if self.ngram_size < 2 || heads <= 0 || embed <= 0 || embed % heads != 0 {
                problems.push(format!(
                    "ple_embed_dim {embed} must be a positive multiple of the {heads} n-gram heads"
                ));
            }
            for &id in &self.ple_layer_ids {
                if id < 1 || id > self.num_hidden_layers {
                    problems.push(format!("ple layer id {id} is not a one-indexed layer"));
                } else if layer_types.get(id as usize - 1).map(String::as_str)
                    != Some("linear_attention")
                {
                    problems.push(format!("ple layer id {id} is not a linear-attention layer"));
                }
            }
            if self.eos_id().is_none() {
                problems.push("eos_token_id must be set when PLE layers are enabled".to_string());
            }
            if self.ple_conv_kernel_size < 1 || self.make_ngram_vocab_size_divisible_by < 1 {
                problems.push("ple_conv_kernel_size and the n-gram divisor must be >= 1".into());
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Exception::custom(format!(
                "unsupported qwen4_exp config: {}",
                problems.join("; ")
            )))
        }
    }

    /// Per-layer types after [`normalize`](Self::normalize).
    pub fn layer_types(&self) -> Vec<String> {
        self.layer_types.clone().unwrap_or_default()
    }

    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        self.layer_types
            .as_ref()
            .and_then(|t| t.get(layer_idx))
            .is_some_and(|t| t == "linear_attention")
    }

    /// The GDN output gate's activation (`output_gate_type`, else
    /// `hidden_act`), resolved as for the rest of the Qwen3.5 family.
    pub fn gate_activation(&self) -> Result<GateActivation, Exception> {
        GateActivation::resolve(self.output_gate_type.as_deref(), Some(&self.hidden_act))
            .map_err(Exception::custom)
    }

    /// RoPE as the reference resolves it: `rope_scaling` replaces
    /// `rope_parameters`; a value missing from the dict falls back to the
    /// top-level key, then to the default (theta 10000, full rotation).
    pub fn rope(&self) -> Result<ResolvedRope, Exception> {
        let params = self.rope_scaling.as_ref().or(self.rope_parameters.as_ref());
        let rope_type = params
            .and_then(|p| p.rope_type.clone())
            .unwrap_or_else(|| "default".to_string());
        if rope_type != "default" {
            return Err(Exception::custom(format!(
                "rope_type {rope_type:?}: only the default rotation is implemented"
            )));
        }
        let theta = params
            .and_then(|p| p.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10_000.0);
        let partial_rotary_factor = params
            .and_then(|p| p.partial_rotary_factor)
            .or(self.partial_rotary_factor)
            .unwrap_or(1.0);
        Ok(ResolvedRope {
            theta: theta as f32,
            partial_rotary_factor,
        })
    }

    /// The QSA indexer geometry. The reference builds an indexer on every
    /// attention layer, so it is required.
    pub fn indexer(&self) -> Result<IndexerGeometry, Exception> {
        let fields = [
            ("indexer_n_heads", self.indexer_n_heads),
            ("indexer_kv_heads", self.indexer_kv_heads),
            ("indexer_head_dim", self.indexer_head_dim),
            ("indexer_budget", self.indexer_budget),
            ("indexer_compress_ratio", self.indexer_compress_ratio),
        ];
        let missing: Vec<&str> = fields
            .iter()
            .filter(|(_, v)| v.is_none())
            .map(|(n, _)| *n)
            .collect();
        if !missing.is_empty() {
            return Err(Exception::custom(format!(
                "the QSA indexer needs {missing:?}"
            )));
        }
        if fields.iter().any(|(_, v)| v.unwrap_or(0) <= 0) {
            return Err(Exception::custom("indexer fields must be positive"));
        }
        if self.indexer_kv_heads != Some(1) {
            return Err(Exception::custom("the QSA indexer has one key head"));
        }
        Ok(IndexerGeometry {
            n_heads: self.indexer_n_heads.unwrap_or_default(),
            head_dim: self.indexer_head_dim.unwrap_or_default(),
            budget: self.indexer_budget.unwrap_or_default(),
            compress_ratio: self.indexer_compress_ratio.unwrap_or_default(),
        })
    }

    pub fn ple_embed_dim(&self) -> i32 {
        self.ple_embed_dim.unwrap_or(self.hidden_size)
    }

    /// Hashed heads per PLE layer: `heads_per_ngram` for each order 2..=`ngram_size`.
    pub fn ngram_heads(&self) -> i32 {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// The token the n-gram hash pads segment starts with.
    pub fn eos_id(&self) -> Option<i64> {
        self.eos_token_id.as_ref().and_then(TokenIds::first)
    }

    /// Position of zero-indexed `layer_idx` among the PLE layers, if it has one.
    pub fn ple_index(&self, layer_idx: usize) -> Option<usize> {
        self.ple_layer_ids
            .iter()
            .position(|&id| id as usize == layer_idx + 1)
    }

    /// The config the reused `qwen3_next` blocks are built from.
    pub fn qwen3_next_view(&self) -> Qwen3NextConfig {
        let rope = self.rope().unwrap_or(ResolvedRope {
            theta: 10_000.0,
            partial_rotary_factor: 1.0,
        });
        Qwen3NextConfig {
            model_type: self.model_type.clone(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.shared_expert_intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: Some(self.num_key_value_heads),
            head_dim: Some(self.head_dim),
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: rope.theta,
            tie_word_embeddings: self.tie_word_embeddings,
            linear_num_value_heads: self.linear_num_value_heads,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            full_attention_interval: self.full_attention_interval,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            decoder_sparse_step: 1,
            moe_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size,
            mlp_only_layers: Vec::new(),
            norm_topk_prob: self.norm_topk_prob,
            partial_rotary_factor: rope.partial_rotary_factor,
            attention_bias: self.attention_bias,
            rope_scaling: None,
            rope_parameters: None,
            layer_types: Some(
                self.layer_types()
                    .into_iter()
                    .map(|t| {
                        if t == "linear_attention" {
                            t
                        } else {
                            "full_attention".to_string()
                        }
                    })
                    .collect(),
            ),
            ..Qwen3NextConfig::default()
        }
    }

    /// Bytes the n-gram tables of every PLE layer take in bf16, to decide
    /// whether they can be resident.
    pub fn ngram_table_bytes(&self) -> u64 {
        let dim = (self.ple_embed_dim() / self.ngram_heads().max(1)) as u64;
        (0..self.ple_layer_ids.len())
            .map(|k| NgramHash::new(self, k).rows as u64 * dim * 2)
            .sum()
    }

    /// Where each recurrent state lives in the [`MambaCache`].
    pub fn cache_layout(&self) -> Qwen4ExpCacheLayout {
        Qwen4ExpCacheLayout {
            num_layers: self.num_hidden_layers as usize,
            ple_layers: self.ple_layer_ids.len(),
        }
    }
}

/// RoPE settings after the reference's fallback rules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedRope {
    pub theta: f32,
    pub partial_rotary_factor: f32,
}

impl ModelConfig for Qwen4ExpConfig {
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
        self.num_hidden_layers
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.head_dim
    }
    fn intermediate_size(&self) -> i32 {
        self.shared_expert_intermediate_size
    }
    fn max_position_embeddings(&self) -> i32 {
        self.max_position_embeddings
    }
    fn norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
    fn rope_theta(&self) -> f32 {
        self.rope().map(|r| r.theta).unwrap_or(10_000.0)
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

/// Which [`MambaCache`] entry holds which state; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen4ExpCacheLayout {
    pub num_layers: usize,
    pub ple_layers: usize,
}

impl Qwen4ExpCacheLayout {
    /// Entries a [`MambaCache`] for this model needs.
    pub fn entries(&self) -> usize {
        self.num_layers + 2 * self.ple_layers
    }
    /// PLE layer `k`'s short-conv state.
    pub fn ple_conv(&self, k: usize) -> usize {
        self.num_layers + 2 * k
    }
    /// PLE layer `k`'s n-gram token context.
    pub fn ple_tokens(&self, k: usize) -> usize {
        self.num_layers + 2 * k + 1
    }
}

// ============================================================================
// Norms and hyper-connections
// ============================================================================

/// `x * rsqrt(mean(x^2) + eps) * (1 + w)`, computed in f32 like the reference,
/// optionally over groups of `group` channels (one per hyper-connection stream).
///
/// The weight is stored as the checkpoint ships it, centred on zero.
#[derive(Debug)]
pub struct Qwen4ExpRmsNorm {
    pub weight: Param<Array>,
    pub group: Option<i32>,
    pub eps: f32,
}
impl_module_params!(Qwen4ExpRmsNorm; weight);

impl Qwen4ExpRmsNorm {
    pub fn new(dim: i32, group: Option<i32>, eps: f32) -> Self {
        Self {
            weight: Param::new(Array::zeros_f32(&[dim])),
            group,
            eps,
        }
    }

    pub fn forward(&self, x: &Array) -> Array {
        let shape = x.shape().to_vec();
        let last = *shape.last().expect("norm input has a channel axis");
        let width = self.group.unwrap_or(last);
        let xf = x.cast(Dtype::Float32).reshape(&[-1, width]);
        let normed = fast::rms_norm_opt(&xf, None, self.eps).reshape(&shape);
        let scale = self.weight.as_ref().cast(Dtype::Float32).add_scalar(1.0);
        normed.multiply(&scale).as_dtype(x.dtype().as_i32())
    }
}

/// One hyper-connection: mixes the `hc_count` residual streams into a block's
/// input and, when `block_inject_weight` is present, weights the block's
/// output back into each stream.
#[derive(Debug)]
pub struct Qwen4ExpGatedResidual {
    pub hc_norm: Qwen4ExpRmsNorm,
    pub input_mix_weight_down: nn::Linear,
    pub input_mix_weight_up: nn::Linear,
    pub block_inject_weight: Option<nn::Linear>,
    pub hc_count: i32,
    pub hidden_size: i32,
}
impl_module_params!(Qwen4ExpGatedResidual; hc_norm, input_mix_weight_down, input_mix_weight_up, block_inject_weight);

impl Qwen4ExpGatedResidual {
    pub fn new(config: &Qwen4ExpConfig, with_inject: bool) -> Result<Self, Exception> {
        let hc = config.hc_count;
        let hidden = config.hidden_size;
        let width = hc * hidden;
        let linear = |i: i32, o: i32| nn::LinearBuilder::new(i, o).bias(false).build();
        Ok(Self {
            hc_norm: Qwen4ExpRmsNorm::new(width, Some(hidden), config.rms_norm_eps),
            input_mix_weight_down: linear(width, config.hc_lowrank)?,
            input_mix_weight_up: linear(config.hc_lowrank, width)?,
            block_inject_weight: if with_inject {
                Some(linear(width, hc)?)
            } else {
                None
            },
            hc_count: hc,
            hidden_size: hidden,
        })
    }

    /// `(block input [B, L, H], per-stream injection weights [B, L, hc])`.
    pub fn mix(&self, streams: &Array) -> (Array, Option<Array>) {
        let b = streams.dim(0);
        let l = streams.dim(1);
        let hc = self.hc_count as f32;
        let normed = self.hc_norm.forward(streams);
        let down = self.input_mix_weight_down.forward(&normed).div_scalar(hc);
        let weights = nn::sigmoid(&self.input_mix_weight_up.forward(&nn::silu(&down)));
        let grid = [b, l, self.hc_count, self.hidden_size];
        let mixed = weights
            .reshape(&grid)
            .multiply(&normed.reshape(&grid))
            .mean_axis(2, false);
        let inject = self
            .block_inject_weight
            .as_ref()
            .map(|linear| nn::sigmoid(&linear.forward(&normed).div_scalar(hc)).mul_scalar(2.0));
        (mixed, inject)
    }

    /// `streams + out ⊗ inject`, the block output weighted into every stream.
    pub fn inject(&self, streams: &Array, out: &Array, inject: &Array) -> Array {
        let b = streams.dim(0);
        let l = streams.dim(1);
        let per_stream = out
            .reshape(&[b, l, 1, self.hidden_size])
            .multiply(&inject.reshape(&[b, l, self.hc_count, 1]))
            .reshape(&[b, l, self.hc_count * self.hidden_size]);
        streams.add(&per_stream)
    }
}

// ============================================================================
// QSA indexer
// ============================================================================

/// Picks, per query, the key blocks an attention layer may read.
///
/// Keys are pooled into blocks of `compress_ratio` tokens (mean, then
/// `k_layernorm`, then RoPE at the block's first position). A query scores a
/// block as `sum_h relu(q_h . k) / sqrt(head_dim)` over its `n_heads` heads and
/// keeps the `budget / compress_ratio` best complete blocks it can see plus its
/// incomplete tail.
#[derive(Debug)]
pub struct Qwen4ExpIndexer {
    pub index_qk_proj: nn::Linear,
    pub q_layernorm: Qwen4ExpRmsNorm,
    pub k_layernorm: Qwen4ExpRmsNorm,
    pub geometry: IndexerGeometry,
    pub rope_dims: i32,
    pub rope_base: f32,
}
impl_module_params!(Qwen4ExpIndexer; index_qk_proj, q_layernorm, k_layernorm);

impl Qwen4ExpIndexer {
    pub fn new(config: &Qwen4ExpConfig) -> Result<Self, Exception> {
        let geometry = config.indexer()?;
        let rope = config.rope()?;
        let d = geometry.head_dim;
        Ok(Self {
            index_qk_proj: nn::LinearBuilder::new(config.hidden_size, (geometry.n_heads + 1) * d)
                .bias(false)
                .build()?,
            q_layernorm: Qwen4ExpRmsNorm::new(d, None, config.rms_norm_eps),
            k_layernorm: Qwen4ExpRmsNorm::new(d, None, config.rms_norm_eps),
            geometry,
            rope_dims: (config.head_dim as f32 * rope.partial_rotary_factor) as i32,
            rope_base: rope.theta,
        })
    }

    /// Pool `n` complete blocks of raw keys `[B, n·c, D]` into block keys
    /// `[B, n, D]`, the first of which is block number `first`.
    fn compress(&self, raw: &Array, first: i32, n: i32) -> Result<Array, Exception> {
        let b = raw.dim(0);
        let c = self.geometry.compress_ratio;
        let d = self.geometry.head_dim;
        let pooled = raw
            .cast(Dtype::Float32)
            .reshape(&[b, n, c, d])
            .mean_axis(2, false)
            .as_dtype(raw.dtype().as_i32());
        let normed = self.k_layernorm.forward(&pooled);
        let starts: Vec<i32> = (first..first + n).map(|j| j * c).collect();
        let starts = Array::from_i32_slice(&starts);
        let rotated = rope(
            &normed.reshape(&[b, 1, n, d]),
            RopePositions::Explicit(&starts),
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
        )?;
        Ok(rotated.reshape(&[b, n, d]))
    }

    /// Fold this chunk's keys into the block cache and return the additive
    /// attention mask `[B, 1, L, offset + L]`, or `None` when every query still
    /// sees few enough blocks to keep all of them (plain causal attention).
    ///
    /// `offset` is the absolute position of the chunk's first token. `state`
    /// is the layer's indexer entry (see the module docs); `None` for an
    /// uncached forward.
    pub fn select(
        &self,
        x: &Array,
        offset: i32,
        state: Option<&mut MambaCacheEntry>,
    ) -> Result<Option<Array>, Exception> {
        let b = x.dim(0);
        let l = x.dim(1);
        let IndexerGeometry {
            n_heads,
            head_dim: d,
            compress_ratio: c,
            ..
        } = self.geometry;

        let qk = self.index_qk_proj.forward(x);
        let q = slice_last_to(&qk, n_heads * d).reshape(&[b, l, n_heads, d]);
        let raw_keys = slice_last_from(&qk, n_heads * d);

        let (pending, mut blocks) = match &state {
            Some(entry) => (entry.conv_state.clone(), entry.ssm_state.clone()),
            None => (None, None),
        };
        let seen =
            blocks.as_ref().map_or(0, |k| k.dim(1)) * c + pending.as_ref().map_or(0, |p| p.dim(1));
        if seen != offset {
            return Err(Exception::custom(format!(
                "QSA indexer cache holds {seen} tokens but the chunk starts at {offset}"
            )));
        }
        let pending = match pending {
            Some(p) => ops::concatenate_axis(&[&p, &raw_keys], 1),
            None => raw_keys,
        };
        let first = blocks.as_ref().map_or(0, |k| k.dim(1));
        let completed = pending.dim(1) / c;
        if completed > 0 {
            let full = slice_axis(&pending, 1, 0, completed * c);
            let new_blocks = self.compress(&full, first, completed)?;
            blocks = Some(match blocks {
                Some(old) => ops::concatenate_axis(&[&old, &new_blocks], 1),
                None => new_blocks,
            });
        }
        let tail =
            (pending.dim(1) > completed * c).then(|| slice_axis_from(&pending, 1, completed * c));
        if let Some(entry) = state {
            entry.conv_state = tail;
            entry.ssm_state = blocks.clone();
        }

        let total = offset + l;
        let nb = total / c;
        let keep_blocks = self.geometry.block_topk();
        if nb <= keep_blocks {
            return Ok(None);
        }
        let blocks = blocks.expect("complete blocks exist past the budget");

        // Scores [B, L, nb], f32 like the reference.
        let q = self.q_layernorm.forward(&q).transpose_axes(&[0, 2, 1, 3]);
        let q = rope(
            &q,
            RopePositions::Offset(offset),
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
        )?;
        let keys_t = blocks
            .cast(Dtype::Float32)
            .reshape(&[b, 1, nb, d])
            .transpose_axes(&[0, 1, 3, 2]);
        let scores = ops::matmul(&q.cast(Dtype::Float32), &keys_t)
            .relu()
            .sum_axis(1, false)
            .div_scalar((d as f32).sqrt());

        // Per-query geometry: query `i` sits at `offset + i` and sees
        // `(offset + i + 1) / c` complete blocks; the rest of its prefix is tail.
        let positions: Vec<i32> = (0..l).map(|i| offset + i).collect();
        let visible: Vec<i32> = positions.iter().map(|p| (p + 1) / c).collect();
        let tail_start: Vec<i32> = visible.iter().map(|v| v * c).collect();
        let visible = Array::from_i32_slice_shaped(&visible, &[l, 1]);
        let block_ids = Array::from_i32_slice_shaped(&(0..nb).collect::<Vec<_>>(), &[1, nb]);
        let valid = block_ids.less(&visible);

        let neg_inf = Array::scalar_with_dtype(f32::NEG_INFINITY, Dtype::Float32.as_i32());
        let masked = ops::r#where(&valid, &scores, &neg_inf);
        // The `keep_blocks` best blocks; with fewer valid ones the extras are
        // invalid and dropped below.
        let order = ops::argpartition_axis(&masked.negative(), keep_blocks - 1, -1);
        let top = slice_last_to(&order, keep_blocks);
        let chosen = ops::put_along_axis(
            &Array::zeros_f32(&[b, l, nb]),
            &top,
            &Array::ones_f32(&[b, l, keep_blocks]),
            -1,
        );
        let chosen = ops::logical_and(&chosen.greater(&Array::from_f32(0.0)), &valid);

        // Tokens: a block's choice covers its tokens; the tail is always kept.
        let mut tokens = chosen.repeat(c, -1);
        if total > nb * c {
            let rest = Array::ones(&[b, l, total - nb * c], Dtype::Bool.as_i32());
            tokens = ops::concatenate_axis(&[&tokens, &rest], -1);
        }
        let token_ids = Array::from_i32_slice_shaped(&(0..total).collect::<Vec<_>>(), &[1, total]);
        let in_tail = token_ids.greater_equal(&Array::from_i32_slice_shaped(&tail_start, &[l, 1]));
        let causal = token_ids.less_equal(&Array::from_i32_slice_shaped(&positions, &[l, 1]));
        let keep = ops::logical_and(&ops::logical_or(&tokens, &in_tail), &causal);
        let zero = Array::scalar_with_dtype(0.0, Dtype::Float32.as_i32());
        Ok(Some(
            ops::r#where(&keep, &zero, &neg_inf).reshape(&[b, 1, l, total]),
        ))
    }
}

/// Gated attention (`qwen3_next`'s) with its QSA indexer.
///
/// Flattens the two into one parameter namespace so the paths are the
/// checkpoint's: `self_attn.q_proj.weight` and `self_attn.indexer.*`.
#[derive(Debug)]
pub struct Qwen4ExpAttention {
    pub attn: Qwen3NextAttention,
    pub indexer: Qwen4ExpIndexer,
}

impl ModuleParameters for Qwen4ExpAttention {
    fn num_parameters(&self) -> usize {
        self.attn.num_parameters() + self.indexer.num_parameters()
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut out = self.attn.parameters();
        Parameter::collect_params(&self.indexer, "indexer", &mut out);
        out
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut out = self.attn.trainable_parameters();
        Parameter::collect_trainable_params(&self.indexer, "indexer", &mut out);
        out
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut out = self.attn.parameters_mut();
        Parameter::collect_params_mut(&mut self.indexer, "indexer", &mut out);
        out
    }
}

impl VisitLinears for Qwen4ExpAttention {
    fn visit_linears_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut nn::Linear)) {
        self.attn.visit_linears_mut(prefix, f);
        self.indexer
            .visit_linears_mut(&child_path(prefix, "indexer"), f);
    }
}

// ============================================================================
// PLE n-gram embeddings
// ============================================================================

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: i128 = 10_007;

fn splitmix64(value: u64) -> u64 {
    let v = value.wrapping_add(SPLITMIX_GAMMA);
    let v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    let v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    v ^ (v >> 31)
}

fn is_prime(value: i64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

/// The `count`-th prime greater than `start`.
fn nth_prime_after(start: i64, count: usize) -> i64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// How one PLE layer hashes the n-grams ending at a token into table rows.
///
/// Each order `n` in `2..=ngram_size` gets `heads_per_ngram` heads, each its own
/// prime-sized slice of the table. An n-gram's id is
/// `XOR_p token[t - p] * multiplier[p]` (for `p < n`), reduced modulo the
/// head's prime. Tokens before a segment start (the previous EOS) and before
/// the sequence read as EOS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHash {
    pub multipliers: Vec<i64>,
    pub head_vocab_sizes: Vec<i64>,
    pub head_offsets: Vec<i64>,
    /// Table rows: the heads' total, padded to the configured divisor.
    pub rows: i64,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub eos: i64,
}

impl NgramHash {
    /// The hash for PLE layer `ple_index` (its position among `ple_layer_ids`).
    pub fn new(config: &Qwen4ExpConfig, ple_index: usize) -> Self {
        let ngram_size = config.ngram_size as usize;
        let heads = config.ngram_heads() as usize;

        let multiplier_max = i64::MAX / i64::from(config.vocab_size.max(1));
        let half_bound = (multiplier_max / 2).max(1) as u64;
        let base_seed = i128::from(config.seed) + PRIME_1 * ple_index as i128;
        let multipliers = (0..ngram_size)
            .map(|index| {
                let value = (base_seed + i128::from(SPLITMIX_GAMMA) * (index as i128 + 1))
                    .rem_euclid(1i128 << 64) as u64;
                (2 * (splitmix64(value) % half_bound) + 1) as i64
            })
            .collect();

        let mut head_vocab_sizes = Vec::with_capacity(heads);
        let mut head_offsets = Vec::with_capacity(heads);
        let mut total = 0i64;
        for head in 0..heads {
            let global_head = ple_index * heads + head;
            let size = nth_prime_after(config.ngram_vocab_size_base - 1, global_head + 1);
            head_vocab_sizes.push(size);
            head_offsets.push(total);
            total += size;
        }
        let divisor = config.make_ngram_vocab_size_divisible_by;
        Self {
            multipliers,
            head_vocab_sizes,
            head_offsets,
            rows: (total + divisor - 1) / divisor * divisor,
            ngram_size,
            heads_per_ngram: config.heads_per_ngram as usize,
            eos: config.eos_id().unwrap_or_default(),
        }
    }

    pub fn heads(&self) -> usize {
        self.head_vocab_sizes.len()
    }

    /// Table rows `[tokens.len() * heads]` for `tokens`, preceded by `context`
    /// (the `ngram_size - 1` tokens before them, EOS-padded at the start).
    pub fn rows_for(&self, context: &[i64], tokens: &[i64]) -> Vec<i32> {
        let history: Vec<i64> = context.iter().chain(tokens).copied().collect();
        let ctx = context.len();
        // Most recent EOS strictly before each history position.
        let mut last_eos_before = Vec::with_capacity(history.len());
        let mut last: Option<usize> = None;
        for (i, &token) in history.iter().enumerate() {
            last_eos_before.push(last);
            if token == self.eos {
                last = Some(i);
            }
        }
        let shifted = |t: usize, shift: usize| -> i64 {
            if shift == 0 {
                return history[t];
            }
            let segment_start = last_eos_before[t].map_or(0, |e| e + 1);
            if t >= shift && t - segment_start >= shift {
                history[t - shift]
            } else {
                self.eos
            }
        };

        let mut rows = Vec::with_capacity(tokens.len() * self.heads());
        for t in ctx..history.len() {
            for order in 2..=self.ngram_size {
                let mut mixed = shifted(t, 0).wrapping_mul(self.multipliers[0]);
                for position in 1..order {
                    mixed ^= shifted(t, position).wrapping_mul(self.multipliers[position]);
                }
                let first_head = (order - 2) * self.heads_per_ngram;
                for head in first_head..first_head + self.heads_per_ngram {
                    let row =
                        mixed.rem_euclid(self.head_vocab_sizes[head]) + self.head_offsets[head];
                    rows.push(row as i32);
                }
            }
        }
        rows
    }
}

/// How a disk-served table's rows are encoded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NgramRowEncoding {
    Bf16,
    F16,
    F32,
    /// E4M3 bytes times a per-tensor scale.
    Fp8E4m3 {
        scale: f32,
    },
}

impl NgramRowEncoding {
    fn bytes_per_value(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::Fp8E4m3 { .. } => 1,
        }
    }
}

/// One file region holding consecutive table rows.
#[derive(Debug)]
pub struct NgramRowRange {
    pub file: Arc<std::fs::File>,
    /// Absolute byte offset of the range's first row.
    pub start: u64,
    pub rows: u64,
}

/// An n-gram table read row by row from the checkpoint's own safetensors
/// shards, so the 51B-parameter table never has to be resident.
///
/// Each lookup reads `heads` rows of `dim` values per token with positional
/// reads (the OS page cache keeps hot rows warm); nothing is mapped or loaded
/// up front.
#[derive(Debug)]
pub struct DiskNgramRows {
    pub ranges: Vec<NgramRowRange>,
    pub dim: usize,
    pub encoding: NgramRowEncoding,
    /// Dtype rows are returned in.
    pub out_dtype: Dtype,
    row_starts: Vec<u64>,
}

impl DiskNgramRows {
    pub fn new(
        ranges: Vec<NgramRowRange>,
        dim: usize,
        encoding: NgramRowEncoding,
        out_dtype: Dtype,
    ) -> Self {
        let mut row_starts = Vec::with_capacity(ranges.len());
        let mut total = 0;
        for range in &ranges {
            row_starts.push(total);
            total += range.rows;
        }
        Self {
            ranges,
            dim,
            encoding,
            out_dtype,
            row_starts,
        }
    }

    pub fn rows(&self) -> u64 {
        self.ranges.iter().map(|r| r.rows).sum()
    }

    /// Rows `ids` as `[ids.len(), dim]`.
    pub fn gather(&self, ids: &[i32]) -> Result<Array, Exception> {
        let width = self.dim * self.encoding.bytes_per_value();
        let mut bytes = vec![0u8; ids.len() * width];
        for (slot, &id) in ids.iter().enumerate() {
            let id = id as u64;
            let range_idx = self.row_starts.partition_point(|&s| s <= id) - 1;
            let range = &self.ranges[range_idx];
            let local = id - self.row_starts[range_idx];
            if local >= range.rows {
                return Err(Exception::custom(format!(
                    "n-gram row {id} is past the table's {} rows",
                    self.rows()
                )));
            }
            range
                .file
                .read_exact_at(
                    &mut bytes[slot * width..(slot + 1) * width],
                    range.start + local * width as u64,
                )
                .map_err(|e| Exception::custom(format!("n-gram row {id}: {e}")))?;
        }
        let shape = [ids.len() as i32, self.dim as i32];
        let rows = match self.encoding {
            NgramRowEncoding::Bf16 | NgramRowEncoding::F16 => {
                let bits: Vec<u16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| u16::from_le_bytes(*b))
                    .collect();
                let dtype = if self.encoding == NgramRowEncoding::Bf16 {
                    Dtype::Bfloat16
                } else {
                    Dtype::Float16
                };
                Array::from_u16_bits_slice(&bits, &shape, dtype.as_i32())
            }
            NgramRowEncoding::F32 => {
                let values: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                Array::from_f32_slice(&values, &shape)
            }
            NgramRowEncoding::Fp8E4m3 { scale } => Array::from_u8_slice(&bytes, &shape)
                .from_fp8(Dtype::Float32.as_i32())
                .mul_scalar(scale),
        };
        Ok(if rows.dtype() == self.out_dtype {
            rows
        } else {
            rows.as_dtype(self.out_dtype.as_i32())
        })
    }
}

/// An n-gram embedding table, resident or served from disk.
///
/// `weight` is the resident table `[rows, dim]` under the reference's runtime
/// name (`ngram_embedding.weight`, the checkpoint's shards concatenated). With
/// rows on disk it stays `None` and contributes no parameter.
#[derive(Debug)]
pub struct NgramTable {
    pub weight: Param<Option<Array>>,
    pub disk: Option<Arc<DiskNgramRows>>,
    pub dim: i32,
}
impl_module_params!(NgramTable; weight);

impl NgramTable {
    /// Rows `ids` as `[ids.len(), dim]`.
    pub fn lookup(&self, ids: &[i32]) -> Result<Array, Exception> {
        if let Some(weight) = self.weight.value.as_ref() {
            let ids = Array::from_i32_slice(ids);
            return Ok(weight.take_axis(&ids, 0));
        }
        match &self.disk {
            Some(disk) => disk.gather(ids),
            None => Err(Exception::custom(
                "the n-gram embedding table was never loaded",
            )),
        }
    }
}

/// PLE's n-gram embedding: hash, then look up one row per head.
#[derive(Debug)]
pub struct Qwen4ExpNgramEmbedding {
    pub ngram_embedding: NgramTable,
    pub hash: NgramHash,
}
impl_module_params!(Qwen4ExpNgramEmbedding; ngram_embedding);

/// Per-Layer Embedding: injects the n-gram features into every stream.
///
/// Each stream gates the n-gram value by how well its normalized activations
/// match the n-gram's key for that stream, then a causal depthwise convolution
/// (kernel `ple_conv_kernel_size`, dilation `ngram_size`) adds local context.
#[derive(Debug)]
pub struct Qwen4ExpPle {
    pub ple_embedding: Qwen4ExpNgramEmbedding,
    pub key_proj: nn::Linear,
    pub value_proj: nn::Linear,
    pub norm_key: Qwen4ExpRmsNorm,
    pub norm_query: Qwen4ExpRmsNorm,
    pub norm_conv: Qwen4ExpRmsNorm,
    pub conv1d: nn::Conv1d,
    pub hc_count: i32,
    pub hidden_size: i32,
    /// Inputs the dilated conv looks back over: `(kernel - 1) * dilation`.
    pub conv_history: i32,
}
impl_module_params!(Qwen4ExpPle; ple_embedding, key_proj, value_proj, norm_key, norm_query, norm_conv, conv1d);

impl Qwen4ExpPle {
    pub fn new(
        config: &Qwen4ExpConfig,
        ple_index: usize,
        resident_table: bool,
    ) -> Result<Self, Exception> {
        let hidden = config.hidden_size;
        let width = hidden * config.hc_count;
        let embed = config.ple_embed_dim();
        let hash = NgramHash::new(config, ple_index);
        let dim = embed / config.ngram_heads();
        let weight = resident_table.then(|| {
            let scale = (1.0 / dim as f32).sqrt();
            pmetal_bridge::compat::random::uniform_range(
                -scale,
                scale,
                &[hash.rows as i32, dim],
                Dtype::Float32,
            )
        });
        let linear = |i: i32, o: i32| nn::LinearBuilder::new(i, o).bias(false).build();
        let kernel = config.ple_conv_kernel_size;
        let dilation = config.ngram_size;
        Ok(Self {
            ple_embedding: Qwen4ExpNgramEmbedding {
                ngram_embedding: NgramTable {
                    weight: Param::new(weight),
                    disk: None,
                    dim,
                },
                hash,
            },
            key_proj: linear(embed, width)?,
            value_proj: linear(embed, hidden)?,
            norm_key: Qwen4ExpRmsNorm::new(width, Some(hidden), config.rms_norm_eps),
            norm_query: Qwen4ExpRmsNorm::new(width, Some(hidden), config.rms_norm_eps),
            norm_conv: Qwen4ExpRmsNorm::new(width, Some(hidden), config.rms_norm_eps),
            conv1d: nn::Conv1dBuilder::new(width, width, kernel)
                .bias(false)
                .groups(width)
                .dilation(dilation)
                .padding(0)
                .build()?,
            hc_count: config.hc_count,
            hidden_size: hidden,
            conv_history: (kernel - 1) * dilation,
        })
    }

    /// N-gram embeddings `[B, L, ple_embed_dim]` for `tokens` (one row of ids
    /// per batch entry), advancing the token context in `context`.
    fn embed(
        &self,
        tokens: &[Vec<i64>],
        context: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        let hash = &self.ple_embedding.hash;
        let ctx_len = hash.ngram_size - 1;
        let batch = tokens.len();
        let seq = tokens.first().map_or(0, Vec::len);

        let previous: Vec<i64> = match context.as_ref().and_then(|c| c.conv_state.as_ref()) {
            Some(state) => {
                state.eval();
                state
                    .as_slice::<i32>()
                    .iter()
                    .map(|&t| i64::from(t))
                    .collect()
            }
            None => vec![hash.eos; batch * ctx_len],
        };
        let mut rows = Vec::with_capacity(batch * seq * hash.heads());
        let mut next_context = Vec::with_capacity(batch * ctx_len);
        for (row, ids) in tokens.iter().enumerate() {
            let ctx = &previous[row * ctx_len..(row + 1) * ctx_len];
            rows.extend(hash.rows_for(ctx, ids));
            let history: Vec<i64> = ctx.iter().chain(ids).copied().collect();
            next_context.extend(history[history.len() - ctx_len..].iter().map(|&t| t as i32));
        }
        if let Some(entry) = context {
            entry.conv_state = Some(Array::from_i32_slice_shaped(
                &next_context,
                &[batch as i32, ctx_len as i32],
            ));
        }

        let embedded = self.ple_embedding.ngram_embedding.lookup(&rows)?;
        Ok(embedded.reshape(&[
            batch as i32,
            seq as i32,
            hash.heads() as i32 * self.ple_embedding.ngram_embedding.dim,
        ]))
    }

    /// The PLE contribution `[B, L, hc·H]` to add to `streams`.
    pub fn forward(
        &self,
        streams: &Array,
        tokens: &[Vec<i64>],
        context: Option<&mut MambaCacheEntry>,
        conv_state: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        let b = streams.dim(0);
        let l = streams.dim(1);
        let hc = self.hc_count;
        let hidden = self.hidden_size;
        let embeddings = self
            .embed(tokens, context)?
            .as_dtype(streams.dtype().as_i32());

        let grid = [b, l, hc, hidden];
        let key = self
            .norm_key
            .forward(&self.key_proj.forward(&embeddings))
            .reshape(&grid);
        let value = self
            .value_proj
            .forward(&embeddings)
            .reshape(&[b, l, 1, hidden]);
        let query = self.norm_query.forward(streams).reshape(&grid);
        let gate = key
            .multiply(&query)
            .sum_axis(-1, true)
            .div_scalar((hidden as f32).sqrt());
        // Signed square root, floored away from zero.
        let gate = ops::maximum(&gate.abs(), &Array::scalar_like(1e-6, &gate))
            .sqrt()
            .multiply(&gate.sign());
        let gated = nn::sigmoid(&gate)
            .multiply(&value)
            .reshape(&[b, l, hc * hidden]);
        let gated_normed = self.norm_conv.forward(&gated);

        let padded = match conv_state {
            Some(entry) => entry.update_conv_state(&gated_normed, self.conv_history + 1)?,
            None => {
                let zeros =
                    ops::zeros_dtype(&[b, self.conv_history, hc * hidden], gated_normed.dtype());
                ops::concatenate_axis(&[&zeros, &gated_normed], 1)
            }
        };
        let conv = nn::silu(&self.conv1d.forward(&padded));
        Ok(gated.add(&conv))
    }
}

// ============================================================================
// Decoder
// ============================================================================

/// One decoder layer: optional PLE, then mixer and MoE, each between a pair of
/// hyper-connection mixes.
#[derive(Debug)]
pub struct Qwen4ExpDecoderLayer {
    pub linear_attn: Option<Qwen3NextGatedDeltaNet>,
    pub self_attn: Option<Qwen4ExpAttention>,
    pub mlp: Qwen3NextSparseMoeBlock,
    pub ple: Option<Qwen4ExpPle>,
    pub attn_hyper_connection: Qwen4ExpGatedResidual,
    pub mlp_hyper_connection: Qwen4ExpGatedResidual,
    /// Position among the PLE layers, when this is one.
    pub ple_index: Option<usize>,
}
impl_module_params!(Qwen4ExpDecoderLayer; linear_attn, self_attn, mlp, ple, attn_hyper_connection, mlp_hyper_connection);

impl Qwen4ExpDecoderLayer {
    pub fn new(
        config: &Qwen4ExpConfig,
        view: &Qwen3NextConfig,
        layer_idx: usize,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
        resident_tables: bool,
    ) -> Result<Self, Exception> {
        let (linear_attn, self_attn) = if config.is_linear_layer(layer_idx) {
            let mut gdn = Qwen3NextGatedDeltaNet::new(view)?;
            gdn.norm.gate_activation = config.gate_activation()?;
            // The reference's q/k `l2norm` (eps on the sum of squares).
            gdn.qk_norm_eps = 1e-6 / config.linear_key_head_dim as f32;
            (Some(gdn), None)
        } else {
            let attention = Qwen4ExpAttention {
                attn: Qwen3NextAttention::new(view)?,
                indexer: Qwen4ExpIndexer::new(config)?,
            };
            (None, Some(attention))
        };
        let mut mlp =
            Qwen3NextSparseMoeBlock::new_with_routed_expert_mode(view, routed_expert_mode)?;
        mlp.layer_idx = layer_idx;
        let ple_index = config.ple_index(layer_idx);
        let ple = ple_index
            .map(|k| Qwen4ExpPle::new(config, k, resident_tables))
            .transpose()?;
        Ok(Self {
            linear_attn,
            self_attn,
            mlp,
            ple,
            attn_hyper_connection: Qwen4ExpGatedResidual::new(config, true)?,
            mlp_hyper_connection: Qwen4ExpGatedResidual::new(config, true)?,
            ple_index,
        })
    }

    /// `(output streams, MoE input)`; the MoE input feeds the expert
    /// prefetcher's guess for the next layer.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        streams: &Array,
        tokens: &[Vec<i64>],
        kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
        layer_idx: usize,
        layout: Qwen4ExpCacheLayout,
    ) -> Result<(Array, Array), Exception> {
        let mut h = streams.clone();
        if let (Some(ple), Some(k)) = (&self.ple, self.ple_index) {
            let out = match mamba_cache.as_deref_mut() {
                Some(cache) => {
                    let mut context = cache.get(layout.ple_tokens(k)).cloned().unwrap_or_default();
                    let conv = cache
                        .get_mut(layout.ple_conv(k))
                        .ok_or_else(|| Exception::custom("mamba cache has no PLE conv entry"))?;
                    let out = ple.forward(&h, tokens, Some(&mut context), Some(conv))?;
                    *cache
                        .get_mut(layout.ple_tokens(k))
                        .ok_or_else(|| Exception::custom("mamba cache has no PLE token entry"))? =
                        context;
                    out
                }
                None => ple.forward(&h, tokens, None, None)?,
            };
            h = h.add(&out);
        }

        let (mixed, inject) = self.attn_hyper_connection.mix(&h);
        let out = if let Some(gdn) = self.linear_attn.as_mut() {
            let state = mamba_cache
                .as_deref_mut()
                .and_then(|c| c.get_mut(layer_idx));
            gdn.forward(&mixed, None, state)?
        } else {
            let attention = self
                .self_attn
                .as_mut()
                .expect("attention layers carry self_attn");
            let offset = kv_cache
                .as_deref()
                .map_or(0, |c| c.rope_offset_for(layer_idx));
            let index_state = match kv_cache {
                Some(_) => mamba_cache.and_then(|c| c.get_mut(layer_idx)),
                None => None,
            };
            let mask = attention.indexer.select(&mixed, offset, index_state)?;
            let kv = kv_cache.map(|c| (c, layer_idx));
            attention.attn.forward(&mixed, mask.as_ref(), kv, None)?
        };
        let inject = inject.expect("decoder hyper-connections inject");
        h = self.attn_hyper_connection.inject(&h, &out, &inject);

        let (mixed, inject) = self.mlp_hyper_connection.mix(&h);
        let out = self.mlp.forward(&mixed)?;
        let inject = inject.expect("decoder hyper-connections inject");
        Ok((self.mlp_hyper_connection.inject(&h, &out, &inject), mixed))
    }
}

/// The text trunk: embedding, decoder layers, final hyper-connection mix.
#[derive(Debug)]
pub struct Qwen4ExpModel {
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Qwen4ExpDecoderLayer>,
    pub hyper_connection_mixer: Qwen4ExpGatedResidual,
    pub hc_count: i32,
    pub layout: Qwen4ExpCacheLayout,
    /// SSD expert offloading, once enabled.
    pub(crate) offload: Option<ExpertOffloadAttachment>,
}
impl_module_params!(Qwen4ExpModel; embed_tokens, layers, hyper_connection_mixer);

impl Qwen4ExpModel {
    pub fn new(
        config: &Qwen4ExpConfig,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
        resident_tables: bool,
    ) -> Result<Self, Exception> {
        let view = config.qwen3_next_view();
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| {
                Qwen4ExpDecoderLayer::new(config, &view, i, routed_expert_mode, resident_tables)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size, config.hidden_size)?,
            layers,
            hyper_connection_mixer: Qwen4ExpGatedResidual::new(config, false)?,
            hc_count: config.hc_count,
            layout: config.cache_layout(),
            offload: None,
        })
    }

    /// Hidden states `[B, L, H]` after the final mix (what the LM head reads).
    ///
    /// The attention mask is the indexer's to build, so an explicit one is
    /// refused rather than silently dropped. With a KV cache the hybrid state
    /// needs the [`MambaCache`] too.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache_and_capture(input_ids, mask, kv_cache, mamba_cache, None)
    }

    /// [`forward_with_cache`](Self::forward_with_cache), also recording the
    /// embedding and the requested layers' output streams `[B, L, hc·H]` into
    /// `capture`.
    pub fn forward_with_cache_and_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
        mut capture: Option<&mut SpecCapture>,
    ) -> Result<Array, Exception> {
        if mask.is_some() {
            return Err(Exception::custom(
                "qwen4_exp builds its attention mask from the QSA indexer; explicit masks \
                 are not supported",
            ));
        }
        if kv_cache.is_some() != mamba_cache.is_some() {
            return Err(Exception::custom(
                "qwen4_exp needs both caches or neither: the KV cache for attention and the \
                 Mamba cache for GDN, PLE and indexer state",
            ));
        }
        if let Some(cache) = mamba_cache.as_deref() {
            if cache.num_layers() < self.layout.entries() {
                return Err(Exception::custom(format!(
                    "qwen4_exp needs a Mamba cache of {} entries (create it with \
                     create_mamba_cache), got {}",
                    self.layout.entries(),
                    cache.num_layers()
                )));
            }
        }

        let tokens = if self.layers.iter().any(|l| l.ple.is_some()) {
            token_rows(input_ids)
        } else {
            Vec::new()
        };
        let embedded = self.embed_tokens.forward(input_ids);
        if let Some(buf) = capture.as_deref_mut()
            && buf.wants_embedding()
        {
            buf.record_embedding(embedded.clone());
        }
        let mut h = embedded.tile(&[1, 1, self.hc_count]);
        let layout = self.layout;
        let decode = input_ids.dim(input_ids.ndim() - 1) == 1;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let (next, moe_input) = layer.forward(
                &h,
                &tokens,
                kv_cache.as_deref_mut(),
                mamba_cache.as_deref_mut(),
                layer_idx,
                layout,
            )?;
            h = next;
            if let Some(buf) = capture.as_deref_mut()
                && buf.wants_hidden_for(layer_idx)
            {
                buf.record_hidden(layer_idx, h.clone());
            }
            // With experts on SSD, start reading the next layer's likely
            // experts, guessed from this layer's MoE input, while the GPU
            // works. Decode only: prefill routes too many tokens to guess.
            if decode && let Some(offload) = &self.offload {
                offload
                    .prefetcher
                    .predict_and_prefetch(layer_idx + 1, &moe_input, &offload.ctx);
            }
        }
        let (mixed, _) = self.hyper_connection_mixer.mix(&h);
        Ok(mixed)
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None, None)
    }
}

/// `input_ids` as one row of token ids per batch entry, on the CPU (the
/// n-gram hash runs there).
fn token_rows(input_ids: &Array) -> Vec<Vec<i64>> {
    let ids = if input_ids.ndim() == 1 {
        input_ids.reshape(&[1, -1])
    } else {
        input_ids.clone()
    };
    let (batch, seq) = (ids.dim(0) as usize, ids.dim(1) as usize);
    // `as_slice` reads memory order, and a chunk sliced out of a longer batch
    // is a strided view, so every row past the first came back as the first's
    // continuation (or, for one column, as an unrelated token). An elementwise
    // op writes a strided input out row-major, and the reshape copies whatever
    // layout is left that a flat view cannot express.
    let ids = ids
        .as_dtype(Dtype::Int32.as_i32())
        .add(&Array::from_i32(0))
        .reshape(&[-1]);
    ids.eval();
    let flat = ids.as_slice::<i32>();
    (0..batch)
        .map(|b| {
            flat[b * seq..(b + 1) * seq]
                .iter()
                .map(|&t| i64::from(t))
                .collect()
        })
        .collect()
}

/// Qwen4-Exp causal language model (text).
#[derive(Debug)]
pub struct Qwen4ExpForCausalLM {
    pub model: Qwen4ExpModel,
    pub lm_head: Option<nn::Linear>,
    pub config: Qwen4ExpConfig,
}
impl_module_params!(Qwen4ExpForCausalLM; model, lm_head);

impl Qwen4ExpForCausalLM {
    /// A randomly initialised model with resident n-gram tables.
    pub fn new(config: Qwen4ExpConfig) -> Result<Self, Exception> {
        Self::build(config, Qwen3NextRoutedExpertMode::Resident, true)
    }

    /// A model for [`load_qwen4_exp_weights`] to fill: the n-gram tables are
    /// left empty for the loader (resident or on disk), and routed experts are
    /// placeholders when they will be offloaded.
    pub fn new_for_loading(
        config: Qwen4ExpConfig,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
    ) -> Result<Self, Exception> {
        Self::build(config, routed_expert_mode, false)
    }

    fn build(
        config: Qwen4ExpConfig,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
        resident_tables: bool,
    ) -> Result<Self, Exception> {
        config.validate()?;
        let model = Qwen4ExpModel::new(&config, routed_expert_mode, resident_tables)?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };
        Ok(Self {
            model,
            lm_head,
            config,
        })
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.config
    }

    fn lm_head_forward(&self, h: &Array) -> Array {
        match &self.lm_head {
            Some(head) => head.forward(h),
            None => self.model.embed_tokens.as_linear(h),
        }
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let h = self.model.forward(input_ids, mask)?;
        Ok(self.lm_head_forward(&h))
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let h = self
            .model
            .forward_with_cache(input_ids, mask, kv_cache, mamba_cache)?;
        Ok(self.lm_head_forward(&h))
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        KVCache::new(KVCacheConfig::new(
            self.config.num_hidden_layers as usize,
            max_seq_len,
            self.config.num_key_value_heads as usize,
            self.config.head_dim as usize,
        ))
    }

    pub fn create_mamba_cache(&self) -> MambaCache {
        MambaCache::new(self.model.layout.entries())
    }

    /// Whether some routed experts are placeholders waiting for an offload
    /// directory.
    pub fn requires_expert_offloading(&self) -> bool {
        self.model
            .layers
            .iter()
            .any(|l| !l.mlp.routed_experts_loaded && l.mlp.offload_ctx.is_none())
    }

    /// Serve routed experts from `experts_dir` (written by
    /// `pmetal pack-experts`) instead of memory, as for Qwen 3.5.
    pub fn enable_expert_offloading(&mut self, experts_dir: &Path) -> Result<(), Exception> {
        let mut blocks: Vec<(usize, &mut Qwen3NextSparseMoeBlock)> = self
            .model
            .layers
            .iter_mut()
            .enumerate()
            .map(|(layer_idx, layer)| (layer_idx, &mut layer.mlp))
            .collect();
        let offload = attach_expert_offload(
            experts_dir,
            &mut blocks,
            self.config.num_experts as usize,
            self.config.hidden_size as usize,
            self.config.num_experts_per_tok as usize,
        )?;
        self.model.offload = Some(offload);
        Ok(())
    }

    /// Prefetch hit/miss statistics, when experts are offloaded.
    pub fn prefetch_stats(&self) -> Option<crate::expert_prefetch::PrefetchStats> {
        self.model.offload.as_ref().map(|o| o.prefetcher.stats())
    }

    pub fn reset_prefetch_stats(&self) {
        if let Some(offload) = &self.model.offload {
            offload.prefetcher.reset_stats();
        }
    }
}

// ============================================================================
// Weight loading
// ============================================================================

/// What a checkpoint tensor is to this port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointKeyRole {
    /// Loaded into a parameter, after `qwen3_next`'s sanitization (prefix,
    /// `A_log`, conv layout, routed-expert stacking).
    Weight,
    /// A shard (or the whole) of PLE layer `layer`'s n-gram table.
    NgramTable { layer: usize },
    /// The per-tensor FP8 scale of PLE layer `layer`'s n-gram table.
    NgramScale { layer: usize },
    /// An int64 hash buffer, checked against the value derived from the config.
    HashBuffer { layer: usize },
    /// Deliberately not loaded.
    Skipped(&'static str),
}

const NGRAM_PREFIX: &str = ".ple.ple_embedding.";

/// Classify a checkpoint key. Quantization sidecars (`weight_scale_inv`,
/// `weight_scale`, `weight_scale_2`, `input_scale`) classify with the weight
/// they belong to.
pub fn checkpoint_key_role(key: &str) -> CheckpointKeyRole {
    if key.starts_with("model.visual.") || key.starts_with("visual.") {
        return CheckpointKeyRole::Skipped("vision tower (text path only)");
    }
    if key.starts_with("mtp.") || key.contains(".mtp.") {
        return CheckpointKeyRole::Skipped(
            "MTP predictor head (not used for generation; the reference ignores it too)",
        );
    }
    if let Some(at) = key.find(NGRAM_PREFIX) {
        let layer = key[..at]
            .rsplit('.')
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(usize::MAX);
        let tail = &key[at + NGRAM_PREFIX.len()..];
        return match tail {
            "layer_multipliers" | "ngram_heads_offsets" | "ngram_heads_vocab_sizes" => {
                CheckpointKeyRole::HashBuffer { layer }
            }
            "ngram_embedding.weight_scale" => CheckpointKeyRole::NgramScale { layer },
            _ => CheckpointKeyRole::NgramTable { layer },
        };
    }
    CheckpointKeyRole::Weight
}

/// Options for [`load_qwen4_exp_weights`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen4ExpLoadOptions {
    /// Leave routed experts out (they will be offloaded with `--experts-dir`).
    pub skip_routed_experts: bool,
    /// Serve n-gram table rows from the checkpoint files instead of loading
    /// the tables (51B parameters in the release).
    pub ngram_rows_on_disk: bool,
    /// Unpack NVFP4 routed experts to dense instead of keeping them on packed
    /// kernels. Dense is 3.6x the memory (241 GB in bf16 for the release
    /// against 68 GB packed); it exists to check the packed path against.
    pub unpack_quantized_experts: bool,
}

impl Qwen4ExpLoadOptions {
    /// Above this the dispatcher serves n-gram rows from disk. The released
    /// tables are 102 GB in bf16 (51 GB in FP8), and a token touches 16 rows of
    /// 320 bytes per PLE layer, so residency buys nothing worth that.
    pub const RESIDENT_NGRAM_LIMIT_BYTES: u64 = 4 << 30;
}

/// Safetensors tensor metadata, read from the file header alone.
#[derive(Debug, Clone, Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

/// Where each checkpoint tensor lives, from the index and the shard headers.
struct CheckpointFiles {
    dir: PathBuf,
    key_to_file: HashMap<String, String>,
    headers: HashMap<String, (u64, HashMap<String, TensorHeader>)>,
}

impl CheckpointFiles {
    fn open(dir: &Path) -> Result<Self, LoadError> {
        let index_path = dir.join("model.safetensors.index.json");
        let key_to_file: HashMap<String, String> = if index_path.exists() {
            let index: crate::loader::WeightIndex =
                serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            index.weight_map
        } else {
            let (_, tensors) = read_header(&dir.join("model.safetensors"))?;
            tensors
                .into_keys()
                .map(|k| (k, "model.safetensors".to_string()))
                .collect()
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            key_to_file,
            headers: HashMap::new(),
        })
    }

    fn keys(&self) -> impl Iterator<Item = &String> {
        self.key_to_file.keys()
    }

    /// `(file path, absolute data start, header)` for `key`.
    fn locate(&mut self, key: &str) -> Result<(PathBuf, u64, TensorHeader), LoadError> {
        let file = self
            .key_to_file
            .get(key)
            .ok_or_else(|| LoadError::MissingWeight(key.to_string()))?
            .clone();
        let path = crate::loader::validate_shard_path(&self.dir, &file)?;
        if !self.headers.contains_key(&file) {
            let header = read_header(&path)?;
            self.headers.insert(file.clone(), header);
        }
        let (data_start, tensors) = &self.headers[&file];
        let header = tensors
            .get(key)
            .ok_or_else(|| LoadError::MissingWeight(format!("{key} (not in {file})")))?
            .clone();
        Ok((path, data_start + header.data_offsets.0, header))
    }

    fn read(&mut self, key: &str) -> Result<(TensorHeader, Vec<u8>), LoadError> {
        let (path, start, header) = self.locate(key)?;
        let len = (header.data_offsets.1 - header.data_offsets.0) as usize;
        let mut bytes = vec![0u8; len];
        std::fs::File::open(&path)?.read_exact_at(&mut bytes, start)?;
        Ok((header, bytes))
    }

    fn read_i64s(&mut self, key: &str) -> Result<Vec<i64>, LoadError> {
        let (header, bytes) = self.read(key)?;
        if header.dtype != "I64" {
            return Err(LoadError::SafeTensors(format!(
                "{key}: expected I64, got {}",
                header.dtype
            )));
        }
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect())
    }

    fn read_scalar_f32(&mut self, key: &str) -> Result<f32, LoadError> {
        let (header, bytes) = self.read(key)?;
        let value = match header.dtype.as_str() {
            "BF16" => f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16),
            "F16" => half_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
            "F32" => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            other => {
                return Err(LoadError::SafeTensors(format!(
                    "{key}: unsupported scale dtype {other}"
                )));
            }
        };
        Ok(value)
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x3ff);
    let magnitude = match exponent {
        0 => (mantissa as f32) * 2f32.powi(-24),
        0x1f => f32::INFINITY,
        e => f32::from_bits((u32::from(e) + 112) << 23 | mantissa << 13),
    };
    f32::from_bits(sign | magnitude.to_bits())
}

/// A safetensors file's header: `(absolute data start, tensors)`.
fn read_header(path: &Path) -> Result<(u64, HashMap<String, TensorHeader>), LoadError> {
    let file = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    file.read_exact_at(&mut len, 0)?;
    let len = u64::from_le_bytes(len);
    let mut json = vec![0u8; len as usize];
    file.read_exact_at(&mut json, 8)?;
    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&json)?;
    let mut tensors = HashMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        tensors.insert(name, serde_json::from_value(value)?);
    }
    Ok((8 + len, tensors))
}

/// How an n-gram table shard of `dtype` and `shape` encodes its rows, given the
/// table's per-tensor `scale` (FP8 checkpoints carry one). Refuses a shard whose
/// row width is not `ple_embed_dim / ngram_heads` or whose encoding is unknown.
pub fn ngram_shard_encoding(
    config: &Qwen4ExpConfig,
    key: &str,
    dtype: &str,
    shape: &[u64],
    scale: Option<f32>,
) -> Result<NgramRowEncoding, LoadError> {
    let dim = (config.ple_embed_dim() / config.ngram_heads()) as u64;
    if shape.len() != 2 || shape[1] != dim {
        return Err(LoadError::ShapeMismatch {
            key: key.to_string(),
            expected: vec![-1, dim as i32],
            actual: shape.iter().map(|&d| d as i32).collect(),
        });
    }
    match (dtype, scale) {
        ("BF16", None) => Ok(NgramRowEncoding::Bf16),
        ("F16", None) => Ok(NgramRowEncoding::F16),
        ("F32", None) => Ok(NgramRowEncoding::F32),
        ("F8_E4M3", Some(scale)) => Ok(NgramRowEncoding::Fp8E4m3 { scale }),
        (dtype, scale) => Err(LoadError::SafeTensors(format!(
            "{key}: n-gram rows of dtype {dtype} with scale {scale:?} are not supported"
        ))),
    }
}

/// The n-gram table rows for PLE layer `layer`, as file ranges in table order.
fn ngram_row_source(
    files: &mut CheckpointFiles,
    config: &Qwen4ExpConfig,
    layer: usize,
    out_dtype: Dtype,
) -> Result<DiskNgramRows, LoadError> {
    let base = format!("model.language_model.layers.{layer}{NGRAM_PREFIX}ngram_embedding");
    let alt_base = format!("model.layers.{layer}{NGRAM_PREFIX}ngram_embedding");
    let has = |files: &CheckpointFiles, k: &str| files.key_to_file.contains_key(k);
    let base =
        if has(files, &format!("{base}.shard_0.weight")) || has(files, &format!("{base}.weight")) {
            base
        } else {
            alt_base
        };
    let mut keys: Vec<String> = if has(files, &format!("{base}.weight")) {
        vec![format!("{base}.weight")]
    } else {
        (0..config.split_ngram_parts)
            .map(|k| format!("{base}.shard_{k}.weight"))
            .collect()
    };
    let shard_count = files
        .keys()
        .filter(|k| k.starts_with(&format!("{base}.shard_")))
        .count();
    if keys.len() > 1 && shard_count != keys.len() {
        return Err(LoadError::SafeTensors(format!(
            "{base}: config says {} shards, checkpoint has {shard_count}",
            keys.len()
        )));
    }
    let scale_key = format!("{base}.weight_scale");
    let scale = if has(files, &scale_key) {
        Some(files.read_scalar_f32(&scale_key)?)
    } else {
        None
    };

    let dim = (config.ple_embed_dim() / config.ngram_heads()) as u64;
    let mut ranges = Vec::with_capacity(keys.len());
    let mut encoding = None;
    let mut opened: HashMap<PathBuf, Arc<std::fs::File>> = HashMap::new();
    for key in keys.drain(..) {
        let (path, start, header) = files.locate(&key)?;
        let this = ngram_shard_encoding(config, &key, &header.dtype, &header.shape, scale)?;
        if encoding.is_some_and(|e| e != this) {
            return Err(LoadError::SafeTensors(format!(
                "{base}: shards disagree on their encoding"
            )));
        }
        encoding = Some(this);
        let file = match opened.get(&path) {
            Some(file) => file.clone(),
            None => {
                let file = Arc::new(std::fs::File::open(&path)?);
                opened.insert(path.clone(), file.clone());
                file
            }
        };
        ranges.push(NgramRowRange {
            file,
            start,
            rows: header.shape[0],
        });
    }
    let encoding = encoding.ok_or_else(|| LoadError::MissingWeight(format!("{base}.*")))?;
    Ok(DiskNgramRows::new(
        ranges,
        dim as usize,
        encoding,
        out_dtype,
    ))
}

/// The tensor half of [`load_qwen4_exp_weights`]: keep ModelOpt NVFP4 routed
/// experts packed (unless `unpack_quantized_experts`), unpack every other
/// quantization sidecar (block FP8, ModelOpt FP8), apply `qwen3_next`'s
/// sanitization, and assign every tensor in `weights` (checkpoint-named,
/// [`CheckpointKeyRole::Weight`] only) to its parameter. Returns how many
/// parameters and packed expert stacks it filled.
///
/// Strict both ways: a tensor that matches no parameter, a shape that differs,
/// or a parameter left unfilled is an error. The n-gram tables are the one
/// parameter it does not fill (the caller does), and routed experts are
/// allowed to stay empty when `skip_routed_experts` (they will be offloaded).
///
/// Public so a checkpoint's layout can be checked against the model from its
/// safetensors headers alone, with lazy placeholders standing in for the data.
pub fn assign_qwen4_exp_tensors(
    model: &mut Qwen4ExpForCausalLM,
    mut weights: HashMap<String, Array>,
    options: Qwen4ExpLoadOptions,
) -> Result<usize, LoadError> {
    let skip_routed_experts = options.skip_routed_experts;
    let packed = if options.unpack_quantized_experts {
        HashMap::new()
    } else {
        take_packed_experts(&mut weights, &model.config)?
    };
    crate::loader::dequantize_sidecar_weights(&mut weights)?;
    let dtypes: HashMap<String, i32> = weights
        .iter()
        .filter(|(k, _)| k.ends_with(".q_norm.weight") || k.ends_with(".k_norm.weight"))
        .map(|(k, v)| (k.clone(), v.dtype_raw()))
        .collect();
    sanitize_weights(
        &mut weights,
        &model.config.qwen3_next_view(),
        Qwen3NextSanitizeOptions {
            skip_routed_experts,
        },
    )
    .map_err(LoadError::from)?;
    // `sanitize_weights` shifts the attention q/k norms to `1 + w` (they run as
    // plain RMSNorms) and promotes them to f32 doing it; keep the checkpoint's
    // dtype so a bf16 model's attention stays bf16.
    for (key, dtype) in dtypes {
        let renamed = key.replacen("model.language_model.", "model.", 1);
        if let Some(w) = weights.get_mut(&renamed) {
            *w = w.as_dtype(dtype);
        }
    }

    // Packed layers keep their experts outside the parameter tree; their
    // stacked arrays become placeholders so nothing ever materialises them.
    for (&layer_idx, experts) in &packed {
        let block = &mut model
            .model
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| {
                LoadError::SafeTensors(format!("packed experts for missing layer {layer_idx}"))
            })?
            .mlp;
        *block.switch_mlp_gate_proj = Array::zeros_f32(&[1]);
        *block.switch_mlp_up_proj = Array::zeros_f32(&[1]);
        *block.switch_mlp_down_proj = Array::zeros_f32(&[1]);
        block.packed_experts = Some(experts.clone());
        block.routed_experts_loaded = true;
    }
    let packed_prefixes: Vec<String> = packed
        .keys()
        .map(|l| format!("model.layers.{l}.mlp.switch_mlp_"))
        .collect();

    let mut params = model.flatten_params_mut();
    let expected: HashSet<String> = params.keys().map(|k| k.to_string()).collect();
    let mut unmatched = Vec::new();
    let mut loaded: HashSet<String> = HashSet::new();
    for (key, value) in weights {
        match params.get_mut(key.as_str()) {
            Some(param) => {
                if param.shape() != value.shape() {
                    return Err(LoadError::ShapeMismatch {
                        key,
                        expected: param.shape().to_vec(),
                        actual: value.shape().to_vec(),
                    });
                }
                **param = value;
                loaded.insert(key);
            }
            None => unmatched.push(key),
        }
    }
    if !unmatched.is_empty() {
        unmatched.sort();
        return Err(LoadError::SafeTensors(format!(
            "qwen4_exp: {} checkpoint tensors match no parameter (first: {:?})",
            unmatched.len(),
            &unmatched[..unmatched.len().min(10)]
        )));
    }
    let mut missing: Vec<&String> = expected
        .iter()
        .filter(|k| !loaded.contains(*k))
        .filter(|k| !k.contains(".ngram_embedding."))
        .filter(|k| !(skip_routed_experts && k.contains(".mlp.switch_mlp_")))
        .filter(|k| !packed_prefixes.iter().any(|p| k.starts_with(p.as_str())))
        .collect();
    if !missing.is_empty() {
        missing.sort();
        return Err(LoadError::SafeTensors(format!(
            "qwen4_exp: {} parameters missing from the checkpoint (first: {:?})",
            missing.len(),
            &missing[..missing.len().min(10)]
        )));
    }
    Ok(loaded.len() + 3 * packed.len())
}

/// Lift ModelOpt-quantized routed experts out of `weights` and stack them,
/// per layer and projection, onto MLX's packed kernels. Any other ModelOpt
/// tensor goes back in dense, as the shared dequantizer would leave it.
fn take_packed_experts(
    weights: &mut HashMap<String, Array>,
    config: &Qwen4ExpConfig,
) -> Result<HashMap<usize, PackedRoutedExperts>, LoadError> {
    use pmetal_bridge::native_loader::take_modelopt_weights;
    use pmetal_bridge::native_weight::detect_model_dtype;

    if !weights
        .keys()
        .any(|k| k.contains(".mlp.experts.") && k.ends_with(".weight_scale"))
    {
        return Ok(HashMap::new());
    }
    let dtype = detect_model_dtype(|key| weights.get(key).map(|w| w.dtype_raw()));
    let experts = config.num_experts as usize;
    let mut parts: HashMap<(usize, usize), Vec<Option<LayerWeight>>> = HashMap::new();
    for (base, weight) in take_modelopt_weights(weights).map_err(LoadError::SafeTensors)? {
        match routed_expert_module(&base) {
            Some((layer, expert, proj)) if expert < experts => {
                let packed = weight
                    .into_native(dtype)
                    .map_err(|e| LoadError::SafeTensors(format!("{base}: {e}")))?
                    .into_layer_weight();
                parts
                    .entry((layer, proj))
                    .or_insert_with(|| vec![None; experts])[expert] = Some(packed);
            }
            _ => {
                let dense = weight
                    .to_dense(dtype)
                    .map_err(|e| LoadError::SafeTensors(format!("{base}: {e}")))?;
                weights.insert(format!("{base}.weight"), dense);
            }
        }
    }

    let layers: HashSet<usize> = parts.keys().map(|&(layer, _)| layer).collect();
    let mut packed = HashMap::new();
    for layer in layers {
        let mut stack = |proj: usize| -> Result<LayerWeight, LoadError> {
            let name = ["gate_proj", "up_proj", "down_proj"][proj];
            let experts = parts.remove(&(layer, proj)).ok_or_else(|| {
                LoadError::MissingWeight(format!("layer {layer} packed experts' {name}"))
            })?;
            let experts = experts
                .into_iter()
                .enumerate()
                .map(|(e, w)| {
                    w.ok_or_else(|| {
                        LoadError::MissingWeight(format!("layer {layer} expert {e} {name}"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            LayerWeight::stack(experts)
                .map_err(|e| LoadError::SafeTensors(format!("layer {layer} {name}: {e}")))
        };
        let experts = PackedRoutedExperts {
            gate: stack(0)?,
            up: stack(1)?,
            down: stack(2)?,
        };
        packed.insert(layer, experts);
    }
    Ok(packed)
}

/// `(layer, expert, projection)` of a per-expert routed module path
/// (`...layers.{l}.mlp.experts.{e}.{gate,up,down}_proj`).
fn routed_expert_module(base: &str) -> Option<(usize, usize, usize)> {
    let (head, tail) = base.split_once(".mlp.experts.")?;
    let layer = head.rsplit('.').next()?.parse().ok()?;
    let (expert, proj) = tail.split_once('.')?;
    let proj = match proj {
        "gate_proj" => 0,
        "up_proj" => 1,
        "down_proj" => 2,
        _ => return None,
    };
    Some((layer, expert.parse().ok()?, proj))
}

/// Load a Qwen4-Exp checkpoint (the released `Qwen4ExpForConditionalGeneration`
/// layout, bf16, block-FP8 or ModelOpt NVFP4) into `model`.
///
/// Strict both ways: every checkpoint tensor is loaded, checked, or skipped by
/// [`checkpoint_key_role`], and every parameter must be filled (routed experts
/// excepted when they will be offloaded). The int64 hash buffers are compared
/// against the values derived from the config, so a config that does not
/// match the checkpoint fails here instead of hashing into the wrong rows.
pub fn load_qwen4_exp_weights(
    model: &mut Qwen4ExpForCausalLM,
    model_dir: &Path,
    options: Qwen4ExpLoadOptions,
) -> Result<LoadReport, LoadError> {
    let config = model.config.clone();
    let mut files = CheckpointFiles::open(model_dir)?;

    let mut report = LoadReport::default();
    let mut ngram_layers: HashSet<usize> = HashSet::new();
    let mut hash_layers: HashSet<usize> = HashSet::new();
    for key in files.keys() {
        match checkpoint_key_role(key) {
            CheckpointKeyRole::Skipped(_) => report.skipped.push(key.clone()),
            CheckpointKeyRole::NgramTable { layer } | CheckpointKeyRole::NgramScale { layer } => {
                ngram_layers.insert(layer);
            }
            CheckpointKeyRole::HashBuffer { layer } => {
                hash_layers.insert(layer);
            }
            CheckpointKeyRole::Weight => {}
        }
    }

    let skip_routed = options.skip_routed_experts;
    // Raw, so NVFP4 experts arrive still packed; the assignment decides what
    // to unpack.
    let mut weights = crate::loader::load_weights_filtered_raw(model_dir, |key| {
        checkpoint_key_role(key) == CheckpointKeyRole::Weight
            && !(skip_routed && key.contains(".mlp.experts."))
    })?;
    crate::loader::unpack_mlx_quantized_weights(model_dir, &mut weights)?;
    report.loaded += assign_qwen4_exp_tensors(model, weights, options)?;

    let dtype = model.model.embed_tokens.weight.as_ref().dtype();
    for (layer_idx, layer) in model.model.layers.iter_mut().enumerate() {
        let Some(ple) = layer.ple.as_mut() else {
            if ngram_layers.contains(&layer_idx) || hash_layers.contains(&layer_idx) {
                return Err(LoadError::SafeTensors(format!(
                    "checkpoint has PLE tensors for layer {layer_idx}, which the config does \
                     not list in ple_layer_ids"
                )));
            }
            continue;
        };
        verify_hash_buffers(&mut files, layer_idx, &ple.ple_embedding.hash)?;
        report.loaded += 3;
        let rows = ngram_row_source(&mut files, &config, layer_idx, dtype)?;
        if rows.rows() as i64 != ple.ple_embedding.hash.rows {
            return Err(LoadError::SafeTensors(format!(
                "layer {layer_idx}: n-gram table has {} rows, the config's hash needs {}",
                rows.rows(),
                ple.ple_embedding.hash.rows
            )));
        }
        report.loaded += rows.ranges.len();
        let table = &mut ple.ple_embedding.ngram_embedding;
        if options.ngram_rows_on_disk {
            table.weight = Param::new(None);
            table.disk = Some(Arc::new(rows));
        } else {
            let all: Vec<i32> = (0..rows.rows() as i32).collect();
            table.weight = Param::new(Some(rows.gather(&all).map_err(LoadError::from)?));
            table.disk = None;
        }
    }
    Ok(report)
}

fn verify_hash_buffers(
    files: &mut CheckpointFiles,
    layer: usize,
    hash: &NgramHash,
) -> Result<(), LoadError> {
    let expected: [(&str, &[i64]); 3] = [
        ("layer_multipliers", &hash.multipliers),
        ("ngram_heads_offsets", &hash.head_offsets),
        ("ngram_heads_vocab_sizes", &hash.head_vocab_sizes),
    ];
    for (name, want) in expected {
        let key = [
            format!("model.language_model.layers.{layer}{NGRAM_PREFIX}{name}"),
            format!("model.layers.{layer}{NGRAM_PREFIX}{name}"),
        ]
        .into_iter()
        .find(|k| files.key_to_file.contains_key(k));
        // Older exports may omit the buffers; the hash is fully determined by
        // the config, so there is nothing to check against.
        let Some(key) = key else { continue };
        let got = files.read_i64s(&key)?;
        if got != want {
            return Err(LoadError::SafeTensors(format!(
                "{key} is {got:?} but the config derives {want:?}; the checkpoint was hashed \
                 with a different seed, vocabulary or n-gram geometry"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released text tower's geometry. Written out here rather than
    /// committed as the released `config.json`, which carries the model's own
    /// (non-permissive) license.
    fn released_text_config() -> String {
        serde_json::json!({
            "model_type": "qwen4_exp_text",
            "vocab_size": 248_320,
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "full_attention_interval": 4,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "hidden_act": "silu",
            "output_gate_type": "sigmoid",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262_144,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            },
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "hc_count": 4,
            "hc_lowrank": 320,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "eos_token_id": 248_044,
            "tie_word_embeddings": false
        })
        .to_string()
    }

    fn released_config() -> Qwen4ExpConfig {
        Qwen4ExpConfig::from_json(&released_text_config()).expect("released config parses")
    }

    #[test]
    fn released_config_resolves_like_the_reference() {
        let config = released_config();
        let types = config.layer_types();
        assert_eq!(types.len(), 48);
        assert_eq!(
            types.iter().filter(|t| *t == "indexed_attention").count(),
            12
        );
        assert!(config.is_linear_layer(0) && !config.is_linear_layer(3));
        let rope = config.rope().unwrap();
        assert_eq!(rope.theta, 1e7);
        assert_eq!(rope.partial_rotary_factor, 0.25);
        assert_eq!(config.gate_activation().unwrap(), GateActivation::Sigmoid);
        assert_eq!(config.indexer().unwrap().block_topk(), 512);
        assert_eq!(config.ple_index(1), Some(0));
        assert_eq!(config.ple_index(2), None);
        assert_eq!(config.ngram_heads(), 16);
        assert_eq!(config.eos_id(), Some(248_044));
        assert_eq!(
            config.cache_layout(),
            Qwen4ExpCacheLayout {
                num_layers: 48,
                ple_layers: 1
            }
        );
        let view = config.qwen3_next_view();
        assert_eq!(view.rope_dims(), 64);
        assert_eq!(view.get_num_kv_heads(), 2);
        assert!(view.norm_topk_prob);
    }

    /// The hash buffers shipped in `Qwen/Qwen3.8-Flash-Next` (read from the
    /// checkpoint with HTTP range requests), reproduced from the config alone.
    #[test]
    fn ngram_hash_reproduces_the_released_buffers() {
        let hash = NgramHash::new(&released_config(), 0);
        assert_eq!(
            hash.multipliers,
            vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
        assert_eq!(
            hash.head_vocab_sizes,
            vec![
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171,
            ]
        );
        assert_eq!(
            hash.head_offsets,
            vec![
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655,
                240_000_802,
                260_000_955,
                280_001_114,
                300_001_275,
            ]
        );
        // 128 shards of 2,500,012 rows each in the checkpoint.
        assert_eq!(hash.rows, 128 * 2_500_012);
    }

    #[test]
    fn ngram_rows_reset_at_eos_and_pad_with_it() {
        let mut config = released_config();
        config.vocab_size = 64;
        config.ngram_vocab_size_base = 37;
        config.heads_per_ngram = 1;
        config.make_ngram_vocab_size_divisible_by = 8;
        config.eos_token_id = Some(TokenIds::One(1));
        let hash = NgramHash::new(&config, 0);
        let row = |a: i64, b: i64, c: i64| -> Vec<i32> {
            let m = &hash.multipliers;
            let bigram = (a * m[0]) ^ (b * m[1]);
            let trigram = bigram ^ (c * m[2]);
            vec![
                (bigram.rem_euclid(hash.head_vocab_sizes[0]) + hash.head_offsets[0]) as i32,
                (trigram.rem_euclid(hash.head_vocab_sizes[1]) + hash.head_offsets[1]) as i32,
            ]
        };
        // Tokens 5 6 1(EOS) 7 8 from an empty context: an n-gram reaching back
        // past the start or across the EOS reads EOS instead.
        let rows = hash.rows_for(&[1, 1], &[5, 6, 1, 7, 8]);
        let expected: Vec<i32> = [
            row(5, 1, 1),
            row(6, 5, 1),
            row(1, 6, 5),
            row(7, 1, 1),
            row(8, 7, 1),
        ]
        .concat();
        assert_eq!(rows, expected);
        // A chunk continuing from cached context hashes as the whole would.
        assert_eq!(hash.rows_for(&[1, 7], &[8]), row(8, 7, 1));
    }

    /// `"swish"` is SiLU, as Qwen3.6 and 3.8 spell it, for the gate and for
    /// `hidden_act` alike.
    #[test]
    fn gate_accepts_the_family_spellings() {
        let base: serde_json::Value = serde_json::from_str(&released_text_config()).unwrap();
        let gate = |patch: serde_json::Value| {
            let mut config = base.clone();
            for (k, v) in patch.as_object().unwrap() {
                config[k] = v.clone();
            }
            Qwen4ExpConfig::from_json(&config.to_string())
                .expect("config parses")
                .gate_activation()
                .unwrap()
        };
        assert_eq!(
            gate(serde_json::json!({"output_gate_type": "swish"})),
            GateActivation::Silu
        );
        assert_eq!(
            gate(serde_json::json!({"output_gate_type": null, "hidden_act": "swish"})),
            GateActivation::Silu
        );
        assert_eq!(
            gate(serde_json::json!({"output_gate_type": null, "hidden_act": "silu"})),
            GateActivation::Silu
        );
    }

    #[test]
    fn config_refuses_math_it_does_not_implement() {
        let base: serde_json::Value = serde_json::from_str(&released_text_config()).unwrap();
        let refuses = |patch: serde_json::Value, needle: &str| {
            let mut config = base.clone();
            for (k, v) in patch.as_object().unwrap() {
                config[k] = v.clone();
            }
            let err = Qwen4ExpConfig::from_json(&config.to_string())
                .expect_err("config should be refused")
                .to_string();
            assert!(err.contains(needle), "{needle:?} not in {err:?}");
        };
        refuses(serde_json::json!({"hidden_act": "gelu"}), "hidden_act");
        refuses(
            serde_json::json!({"output_gate_type": "tanh"}),
            "gate activation",
        );
        refuses(
            serde_json::json!({"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e7}}),
            "rope_type",
        );
        refuses(
            serde_json::json!({"indexer_budget": null}),
            "indexer_budget",
        );
        refuses(serde_json::json!({"indexer_kv_heads": 2}), "one key head");
        refuses(serde_json::json!({"indexer_budget": 2047}), "multiple");
        refuses(
            serde_json::json!({"ple_layer_ids": [4]}),
            "not a linear-attention",
        );
        refuses(serde_json::json!({"hc_count": 1}), "hc_count");
        refuses(serde_json::json!({"eos_token_id": null}), "eos_token_id");
    }

    /// E4M3 (bias 7, no infinities) to f32, written out independently of MLX.
    fn e4m3(byte: u8) -> f32 {
        let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((byte >> 3) & 0xf);
        let mantissa = f32::from(byte & 0x7);
        let magnitude = if exponent == 0 {
            mantissa / 8.0 * 2f32.powi(-6)
        } else {
            (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
        };
        sign * magnitude
    }

    /// The FP8 release stores the n-gram table as E4M3 shards plus one
    /// per-tensor `weight_scale`; served rows must be the bytes times it, in
    /// table order across the shards.
    #[test]
    fn fp8_ngram_rows_decode_with_their_scale() {
        let mut config = released_config();
        config.ple_embed_dim = Some(16);
        config.heads_per_ngram = 2;
        config.split_ngram_parts = 2;
        let (rows_per_shard, dim) = (6usize, 4usize);
        let bytes: Vec<u8> = (0..2 * rows_per_shard * dim)
            .map(|i| ((i * 5) % 0x70) as u8 | if i % 3 == 0 { 0x80 } else { 0 })
            .collect();
        let scale_bits = 0x3e80u16.to_le_bytes(); // bf16 0.25
        let base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
        let half = rows_per_shard * dim;
        let views = vec![
            (
                format!("{base}.shard_0.weight"),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F8_E4M3,
                    vec![rows_per_shard, dim],
                    &bytes[..half],
                )
                .unwrap(),
            ),
            (
                format!("{base}.shard_1.weight"),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F8_E4M3,
                    vec![rows_per_shard, dim],
                    &bytes[half..],
                )
                .unwrap(),
            ),
            (
                format!("{base}.weight_scale"),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    vec![1],
                    &scale_bits,
                )
                .unwrap(),
            ),
        ];
        let dir = std::env::temp_dir().join(format!("qwen4_exp_fp8_rows_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();

        let mut files = CheckpointFiles::open(&dir).unwrap();
        let rows = ngram_row_source(&mut files, &config, 1, Dtype::Float32).unwrap();
        assert_eq!(rows.encoding, NgramRowEncoding::Fp8E4m3 { scale: 0.25 });
        let ids: Vec<i32> = vec![11, 0, 6, 5, 7];
        let got = rows.gather(&ids).unwrap();
        got.eval();
        let got: Vec<f32> = got.as_slice::<f32>().to_vec();
        let expected: Vec<f32> = ids
            .iter()
            .flat_map(|&r| {
                let r = r as usize;
                bytes[r * dim..(r + 1) * dim]
                    .iter()
                    .map(|&b| e4m3(b) * 0.25)
                    .collect::<Vec<_>>()
            })
            .collect();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(got, expected);
    }

    #[test]
    fn checkpoint_keys_classify_by_role() {
        use CheckpointKeyRole::*;
        let p = "model.language_model.layers.1.ple.ple_embedding.";
        assert_eq!(
            checkpoint_key_role(&format!("{p}ngram_embedding.shard_7.weight")),
            NgramTable { layer: 1 }
        );
        assert_eq!(
            checkpoint_key_role(&format!("{p}ngram_embedding.weight_scale")),
            NgramScale { layer: 1 }
        );
        assert_eq!(
            checkpoint_key_role(&format!("{p}layer_multipliers")),
            HashBuffer { layer: 1 }
        );
        assert_eq!(
            checkpoint_key_role("model.language_model.layers.1.ple.key_proj.weight"),
            Weight
        );
        assert_eq!(
            checkpoint_key_role(
                "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"
            ),
            Weight
        );
        assert!(matches!(
            checkpoint_key_role("mtp.layers.0.mlp.gate.weight"),
            Skipped(_)
        ));
        assert!(matches!(
            checkpoint_key_role("model.visual.blocks.0.attn.qkv.weight"),
            Skipped(_)
        ));
    }
}
