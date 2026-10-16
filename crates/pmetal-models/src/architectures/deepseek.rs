//! DeepSeek V3 / V3.2 / V3.2-Speciale model architecture.
//!
//! Implements DeepSeek V3 and V3.2 variants with:
//! - Multi-Latent Attention (MLA): LoRA-style Q/K/V compression for 28x KV cache reduction
//! - Mixture of Experts (MoE) with shared experts and aux-free load balancing
//! - Sparse Attention (DSA) for V3.2 (Lightning Indexer + Token Selector)
//! - Multi-Token Prediction (MTP) for densified training signals
//! - Multi-token prediction lookahead modules

// ModuleParameters derive via impl_module_params!
use crate::checkpointing::checkpointed_layer;
use crate::common::switch_mlp::SwitchMlp;
use crate::decoder_layer::{AttentionModule, DecoderLayer, MlpModule, std_pre_norm_forward};
use pmetal_cache::kv_cache::KVCache;
use pmetal_runtime::compat::{
    Array, Buffer, Dtype, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters,
    VisitLinears, nn, ops,
};
use pmetal_runtime::deepseek_native::{DeepseekRouting, ScoringFunc, TopkMethod};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopePositions, rope_embedding},
};
use pmetal_runtime::rope::{RopeConfig, RotaryEmbedding};
use serde::{Deserialize, Serialize};

/// Result type for DeepSeek operations.
pub type Result<T, E = Exception> = std::result::Result<T, E>;

/// DeepSeek model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeepSeekVariant {
    /// DeepSeek V3 (Standard MLA + MoE).
    #[default]
    V3,
    /// DeepSeek V3.2 (MLA + MoE + Sparse Attention).
    V32,
    /// DeepSeek V3.2-Speciale (Thinking-optimized variant).
    V32Speciale,
}

/// DeepSeek model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub moe_intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: Option<i32>,
    pub n_shared_experts: Option<i32>,
    pub n_routed_experts: Option<i32>,
    pub routed_scaling_factor: f32,
    pub kv_lora_rank: i32,
    pub q_lora_rank: Option<i32>,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub qk_nope_head_dim: i32,
    /// How the router picks experts. Absent: the model type's (V2 `greedy`,
    /// V3 `noaux_tc`).
    #[serde(default)]
    pub topk_method: Option<TopkMethod>,
    /// How the router scores them. Absent: the model type's (V2 `softmax`,
    /// V3 `sigmoid`).
    #[serde(default)]
    pub scoring_func: Option<ScoringFunc>,
    pub norm_topk_prob: bool,
    /// Expert groups of the group-limited methods; absent or `null` is 1.
    #[serde(default = "default_one", deserialize_with = "one_if_null")]
    pub n_group: i32,
    /// Groups those methods keep; absent or `null` is 1.
    #[serde(default = "default_one", deserialize_with = "one_if_null")]
    pub topk_group: i32,
    pub num_experts_per_tok: i32,
    pub moe_layer_freq: i32,
    pub first_k_dense_replace: i32,
    pub max_position_embeddings: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<serde_json::Value>,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_sparse_attention: bool,
    #[serde(default = "default_lightning_indexer_heads")]
    pub lightning_indexer_heads: i32,
    #[serde(default = "default_sparse_top_k")]
    pub sparse_top_k: i32,
    #[serde(default = "default_true")]
    pub indexer_non_interleaved_rope: bool,
    #[serde(default)]
    pub indexer_use_fp8: bool,
    #[serde(default)]
    pub variant: DeepSeekVariant,
    #[serde(default)]
    pub thinking_mode: bool,
    #[serde(default)]
    pub max_thinking_tokens: Option<i32>,
    #[serde(default)]
    pub thinking_start_token_id: Option<i32>,
    #[serde(default)]
    pub thinking_end_token_id: Option<i32>,
    #[serde(default)]
    pub use_mtp: bool,
    #[serde(default = "default_num_nextn_predict_layers")]
    pub num_nextn_predict_layers: i32,
    #[serde(default = "default_hidden_size")]
    pub mtp_hidden_size: i32,
    #[serde(default = "default_mtp_loss_weight")]
    pub mtp_loss_weight: f32,
}

impl DeepSeekConfig {
    pub fn q_head_dim(&self) -> i32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// The rotary embedding of the `qk_rope_head_dim` channels, YaRN as
    /// `rope_scaling` gives it (frequencies and attention factor); an unknown
    /// `rope_type` is an error naming it. MLA pairs them interleaved
    /// (`traditional`); the V3.2 indexer may not.
    pub fn rotary(&self, traditional: bool) -> Result<RotaryEmbedding> {
        crate::common::rotary_embedding(
            "deepseek",
            self.qk_rope_head_dim,
            RopeConfig {
                rope_scaling: self.rope_scaling.as_ref(),
                rope_theta: Some(self.rope_theta as f64),
                max_position_embeddings: Some(self.max_position_embeddings as f64),
                ..RopeConfig::default()
            },
            1.0,
            traditional,
        )
    }
    /// The router's settings: the config's `topk_method`, `scoring_func`,
    /// `n_group`, `topk_group`, `num_experts_per_tok`, `routed_scaling_factor`
    /// and `norm_topk_prob`, with the model type's defaults filled in.
    pub fn routing(&self) -> DeepseekRouting {
        DeepseekRouting {
            scoring: self
                .scoring_func
                .unwrap_or_else(|| ScoringFunc::default_for(&self.model_type)),
            method: self
                .topk_method
                .unwrap_or_else(|| TopkMethod::default_for(&self.model_type)),
            n_group: self.n_group,
            topk_group: self.topk_group,
            top_k: self.num_experts_per_tok,
            routed_scaling_factor: self.routed_scaling_factor,
            norm_topk_prob: self.norm_topk_prob,
        }
    }
    pub fn is_moe_layer(&self, layer_id: i32) -> bool {
        if layer_id < self.first_k_dense_replace {
            return false;
        }
        layer_id % self.moe_layer_freq == 0
    }
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            model_type: "deepseek_v3".to_string(),
            vocab_size: 102400,
            hidden_size: 4096,
            intermediate_size: 11008,
            moe_intermediate_size: 1407,
            num_hidden_layers: 30,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            n_shared_experts: None,
            n_routed_experts: None,
            routed_scaling_factor: 1.0,
            kv_lora_rank: 512,
            q_lora_rank: Some(1536),
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            qk_nope_head_dim: 128,
            topk_method: None,
            scoring_func: None,
            norm_topk_prob: true,
            n_group: 1,
            topk_group: 1,
            num_experts_per_tok: 1,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            tie_word_embeddings: false,
            use_sparse_attention: false,
            lightning_indexer_heads: 4,
            sparse_top_k: 2048,
            indexer_non_interleaved_rope: true,
            indexer_use_fp8: false,
            variant: DeepSeekVariant::V3,
            thinking_mode: false,
            max_thinking_tokens: None,
            thinking_start_token_id: None,
            thinking_end_token_id: None,
            use_mtp: false,
            num_nextn_predict_layers: 1,
            mtp_hidden_size: 4096,
            mtp_loss_weight: 0.3,
        }
    }
}

fn default_num_nextn_predict_layers() -> i32 {
    1
}
fn default_mtp_loss_weight() -> f32 {
    0.3
}
fn default_lightning_indexer_heads() -> i32 {
    4
}
fn default_sparse_top_k() -> i32 {
    2048
}
fn default_model_type() -> String {
    "deepseek_v3".to_string()
}
fn default_vocab_size() -> i32 {
    102400
}
fn default_hidden_size() -> i32 {
    4096
}
fn default_true() -> bool {
    true
}
fn default_one() -> i32 {
    1
}
/// A `null` count is the default one, as transformers reads it.
fn one_if_null<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<i32, D::Error> {
    Ok(Option::<i32>::deserialize(deserializer)?.unwrap_or(1))
}

#[derive(Debug)]
pub struct DeepSeekAttention {
    pub config: DeepSeekConfig,
    pub n_heads: i32,
    pub scale: f32,
    pub layer_id: usize,
    pub q_a_proj: Option<nn::Linear>,
    pub q_a_layernorm: Option<nn::RmsNorm>,
    pub q_b_proj: Option<nn::Linear>,
    pub q_proj: Option<nn::Linear>,
    pub kv_a_proj_with_mqa: nn::Linear,
    pub kv_a_layernorm: nn::RmsNorm,
    pub kv_b_proj: nn::Linear,
    pub o_proj: nn::Linear,
    /// The interleaved rotary embedding of `q_pe` / `k_pe`, YaRN included.
    rotary: RotaryEmbedding,
}
impl_module_params!(DeepSeekAttention; q_a_proj, q_a_layernorm, q_b_proj, q_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj);

/// The softmax scale and rotary embedding an MLA layer runs with.
///
/// These are one decision, not two: a scaled RoPE folds
/// `get_mscale(factor, mscale_all_dim)²` into the softmax scale (transformers'
/// `deepseek_v3` `yarn_apply_mscale`) on top of the YaRN frequencies and
/// attention factor the rotation carries. Returning them together is what
/// stops a caller taking the rotation and forgetting the scale.
///
/// Public because `pmetal-lora` builds the same attention and needs the same
/// answer.
pub fn mla_scale_and_rotary(config: &DeepSeekConfig) -> Result<(f32, RotaryEmbedding)> {
    let rotary = config.rotary(true)?;
    let scale = (config.q_head_dim() as f32).powf(-0.5)
        * rotary.rotary().scaling.mla_softmax_mscale() as f32;
    Ok((scale, rotary))
}

impl DeepSeekAttention {
    pub fn new(config: &DeepSeekConfig, layer_id: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let q_head_dim = config.q_head_dim();
        let (scale, rotary) = mla_scale_and_rotary(config)?;
        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) =
            if let Some(q_lora_rank) = config.q_lora_rank {
                let q_a = nn::LinearBuilder::new(hidden_size, q_lora_rank)
                    .bias(config.attention_bias)
                    .build()?;
                let q_a_norm = nn::RmsNormBuilder::new(q_lora_rank).eps(1e-6).build()?;
                let q_b = nn::LinearBuilder::new(q_lora_rank, n_heads * q_head_dim)
                    .bias(false)
                    .build()?;
                (Some(q_a), Some(q_a_norm), Some(q_b), None)
            } else {
                let q = nn::LinearBuilder::new(hidden_size, n_heads * q_head_dim)
                    .bias(false)
                    .build()?;
                (None, None, None, Some(q))
            };
        let kv_a_proj_with_mqa =
            nn::LinearBuilder::new(hidden_size, config.kv_lora_rank + config.qk_rope_head_dim)
                .bias(config.attention_bias)
                .build()?;
        let kv_a_layernorm = nn::RmsNormBuilder::new(config.kv_lora_rank)
            .eps(1e-6)
            .build()?;
        let kv_b_proj = nn::LinearBuilder::new(
            config.kv_lora_rank,
            n_heads * (config.qk_nope_head_dim + config.v_head_dim),
        )
        .bias(false)
        .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * config.v_head_dim, hidden_size)
            .bias(config.attention_bias)
            .build()?;
        Ok(Self {
            config: config.clone(),
            n_heads,
            scale,
            layer_id,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary,
        })
    }
    fn project_qkv_uncached(
        &mut self,
        x: &Array,
        rope_positions: RopePositions<'_>,
    ) -> Result<(Array, Array, Array)> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let q = if let Some(ref q_a) = self.q_a_proj {
            let q_a_out = q_a.forward(x);
            let q_a_norm = self.q_a_layernorm.as_ref().unwrap().forward(&q_a_out);
            self.q_b_proj.as_ref().unwrap().forward(&q_a_norm)
        } else {
            self.q_proj.as_ref().unwrap().forward(x)
        };
        let q_head_dim = self.config.q_head_dim();
        let q = q
            .reshape(&[batch, seq_len, self.n_heads, q_head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let q_parts = ops::split_sections(&q, &[self.config.qk_nope_head_dim], -1);
        let q_nope = &q_parts[0];
        let q_pe = &q_parts[1];
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);
        let kv_parts = ops::split_sections(&compressed_kv, &[self.config.kv_lora_rank], -1);
        let compressed_latent = &kv_parts[0];
        let k_pe = &kv_parts[1]
            .reshape(&[batch, seq_len, 1, self.config.qk_rope_head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let kv_normalized = self.kv_a_layernorm.forward(compressed_latent);
        let kv = self.kv_b_proj.forward(&kv_normalized);
        let kv_dim = self.config.qk_nope_head_dim + self.config.v_head_dim;
        let kv = kv
            .reshape(&[batch, seq_len, self.n_heads, kv_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let kv_split = ops::split_sections(&kv, &[self.config.qk_nope_head_dim], -1);
        let k_nope = &kv_split[0];
        let values = &kv_split[1];
        // DeepSeek MLA rotates `q_pe` / `k_pe` interleaved (traditional),
        // with YaRN's frequencies and attention factor when configured.
        let q_pe = rope_embedding(q_pe, rope_positions, &self.rotary);
        let k_pe = rope_embedding(k_pe, rope_positions, &self.rotary);
        let k_pe_repeated = pmetal_runtime::compat::ops::broadcast_to(
            &k_pe,
            &[batch, self.n_heads, seq_len, self.config.qk_rope_head_dim],
        );
        let keys = pmetal_runtime::compat::ops::concatenate_axis(&[k_nope, &k_pe_repeated], -1);
        let queries = pmetal_runtime::compat::ops::concatenate_axis(&[q_nope, &q_pe], -1);
        Ok((queries, keys, values.clone()))
    }

    pub fn project_qkv(
        &mut self,
        x: &Array,
        mut cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<(Array, Array, Array)> {
        let rope_positions = RopePositions::resolve(
            positions,
            cache
                .as_ref()
                .map_or(0, |(c, layer)| c.rope_offset_for(*layer)),
        );
        let (queries, keys, values) = self.project_qkv_uncached(x, rope_positions)?;
        let (keys, values) = if let Some((ref mut cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, &values)?
        } else {
            (keys, values)
        };
        Ok((queries, keys, values))
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array> {
        let mut cache = cache;
        let rope_positions = RopePositions::resolve(
            positions,
            cache
                .as_ref()
                .map_or(0, |(c, layer)| c.rope_offset_for(*layer)),
        );
        let (queries, keys, values) = self.project_qkv_uncached(x, rope_positions)?;
        let batch = queries.shape()[0];
        let seq_len = queries.shape()[2];
        let attn_config =
            FusedAttentionConfig::new(self.n_heads, self.n_heads, self.config.q_head_dim())
                .with_scale(self.scale)
                .with_mask_type(if mask.is_some() {
                    AttentionMaskType::None
                } else {
                    AttentionMaskType::Causal
                });

        if mask.is_none() {
            if let Some((cache_ref, layer_idx)) = cache.as_mut() {
                if let Some(output) = (*cache_ref).try_compressed_attention(
                    *layer_idx,
                    &queries,
                    &keys,
                    &values,
                    &attn_config,
                )? {
                    let output = output
                        .transpose_axes(&[0, 2, 1, 3])
                        .reshape(&[batch, seq_len, -1]);
                    return Ok(self.o_proj.forward(&output));
                }
            }
        }

        let (keys, values) = if let Some((ref mut cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, &values)?
        } else {
            (keys, values)
        };

        // `attn_config` carries the causal mask type, so hand it to the kernel
        // rather than recomputing attention here. The previous hand-rolled
        // `matmul -> scale -> add(mask) -> softmax` added only the *caller's*
        // mask; with `mask = None`, which is how generation calls this, no mask
        // was applied at all and every token attended to its own future.
        //
        // `fused_sdpa` reaches MLX's SDPA for MLA: `flash_attention_supported`
        // declines when the value head dim differs from the key head dim, which
        // is exactly the DeepSeek case, and MLX sizes its output from
        // `v.shape(-1)`.
        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)?
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        Ok(self.o_proj.forward(&output))
    }
}

#[derive(Debug)]
pub struct LightningIndexer {
    pub n_heads: i32,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub head_dim: i32,
    pub scale: f32,
    /// The model's rotary embedding (YaRN included, as transformers'
    /// `deepseek_v32` indexer takes the model's `cos`/`sin`), in the
    /// indexer's own pair layout.
    pub rotary: RotaryEmbedding,
}
impl_module_params!(LightningIndexer; q_proj, k_proj);

impl LightningIndexer {
    pub fn new(config: &DeepSeekConfig) -> Result<Self> {
        let n_heads = config.lightning_indexer_heads;
        let head_dim = config.qk_rope_head_dim;
        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        Ok(Self {
            n_heads,
            q_proj,
            k_proj,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rotary: config.rotary(!config.indexer_non_interleaved_rope)?,
        })
    }
    pub fn compute_scores(
        &mut self,
        x: &Array,
        rope_positions: RopePositions<'_>,
    ) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let q = self
            .q_proj
            .forward(x)
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let k = self
            .k_proj
            .forward(x)
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let q = rope_embedding(&q, rope_positions, &self.rotary);
        let k = rope_embedding(&k, rope_positions, &self.rotary);
        let scores = q
            .matmul(&k.transpose_axes(&[0, 1, 3, 2]))
            .multiply(&Array::from_f32(self.scale));
        Ok(scores.mean_axis(1, true).squeeze_axes(&[1]))
    }
}

#[derive(Debug)]
pub struct TokenSelector {
    pub top_k: i32,
}
impl TokenSelector {
    pub fn new(config: &DeepSeekConfig) -> Self {
        Self {
            top_k: config.sparse_top_k,
        }
    }
    pub fn select_tokens(&self, scores: &Array, mask: Option<&Array>) -> Result<Array> {
        let mask_2d = mask.map(|m| m.squeeze_axes(&[0, 1]));
        let masked_scores = if let Some(m) = mask_2d {
            scores.add(&m)
        } else {
            scores.clone()
        };
        let neg_k = -self.top_k;
        // Indices carry no gradient, and MLX refuses to differentiate a gather
        // with respect to them.
        Ok(pmetal_runtime::compat::ops::stop_gradient(
            &pmetal_runtime::compat::ops::slice_axis_from(
                &pmetal_runtime::compat::ops::argpartition_axis(&masked_scores, neg_k, -1),
                -1,
                neg_k,
            ),
        ))
    }
}

#[derive(Debug)]
pub struct DeepSeekSparseAttention {
    pub base_attention: DeepSeekAttention,
    pub indexer: LightningIndexer,
    pub selector: TokenSelector,
    pub store_indices: bool,
}
impl_module_params!(DeepSeekSparseAttention; base_attention, indexer);

impl DeepSeekSparseAttention {
    pub fn new(config: &DeepSeekConfig, layer_id: usize) -> Result<Self> {
        Ok(Self {
            base_attention: DeepSeekAttention::new(config, layer_id)?,
            indexer: LightningIndexer::new(config)?,
            selector: TokenSelector::new(config),
            store_indices: false,
        })
    }
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array> {
        let seq_len = x.shape()[1];
        if seq_len < 2 * self.selector.top_k {
            return self.base_attention.forward(x, mask, cache, positions);
        }
        let rope_positions = RopePositions::resolve(
            positions,
            cache
                .as_ref()
                .map_or(0, |(c, layer)| c.rope_offset_for(*layer)),
        );
        let scores = self.indexer.compute_scores(x, rope_positions)?;
        let selected_indices = self.selector.select_tokens(&scores, mask)?;
        let (queries, keys, values) = self.base_attention.project_qkv(x, cache, positions)?;
        let (batch, n_heads, query_len, key_dim) = (
            queries.shape()[0],
            queries.shape()[1],
            queries.shape()[2],
            keys.shape()[3],
        );
        let val_dim = values.shape()[3];
        let top_k = self.selector.top_k;
        let idx = selected_indices.reshape(&[batch, 1, query_len, top_k]);
        let idx =
            pmetal_runtime::compat::ops::broadcast_to(&idx, &[batch, n_heads, query_len, top_k]);
        let idx_for_keys = idx.reshape(&[batch, n_heads, query_len * top_k, 1]);
        let idx_for_keys = pmetal_runtime::compat::ops::broadcast_to(
            &idx_for_keys,
            &[batch, n_heads, query_len * top_k, key_dim],
        );
        let gathered_keys =
            pmetal_runtime::compat::indexing::take_along_axis(&keys, &idx_for_keys, 2)
                .reshape(&[batch, n_heads, query_len, top_k, key_dim]);
        let idx_for_vals = idx.reshape(&[batch, n_heads, query_len * top_k, 1]);
        let idx_for_vals = pmetal_runtime::compat::ops::broadcast_to(
            &idx_for_vals,
            &[batch, n_heads, query_len * top_k, val_dim],
        );
        let gathered_values =
            pmetal_runtime::compat::indexing::take_along_axis(&values, &idx_for_vals, 2)
                .reshape(&[batch, n_heads, query_len, top_k, val_dim]);
        let q_expanded = queries.reshape(&[batch, n_heads, query_len, 1, key_dim]);
        let attn_scores = q_expanded
            .matmul(&gathered_keys.transpose_axes(&[0, 1, 2, 4, 3]))
            .squeeze_axes(&[3])
            .multiply(&Array::from_f32(self.base_attention.scale));
        let attn_weights = pmetal_runtime::compat::ops::softmax_axis(&attn_scores, -1)
            .reshape(&[batch, n_heads, query_len, 1, top_k]);
        let output = attn_weights
            .matmul(&gathered_values)
            .squeeze_axes(&[3])
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, query_len, -1]);
        Ok(self.base_attention.o_proj.forward(&output))
    }
}

#[derive(Debug)]
pub struct DeepSeekMLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(DeepSeekMLP; gate_proj, up_proj, down_proj);

impl DeepSeekMLP {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            up_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()?,
        })
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        Ok(self.down_proj.forward(&nn::silu(&gate).multiply(&up)))
    }
}

/// DeepSeek's router: the gate matrix, V3's load-balancing bias, and the
/// settings of the top-k ([`DeepseekRouting`], shared with the native engine).
///
/// The gate is a bare matrix and not a `Linear`, so its parameter paths are the
/// checkpoint's (`mlp.gate.weight`, `mlp.gate.e_score_correction_bias`) and no
/// adapter can attach to it: a router's argmax picks which computation runs.
#[derive(Debug)]
pub struct DeepSeekMoEGate {
    /// `[num_experts, hidden]`.
    pub weight: Array,
    /// `[num_experts]`, V3's `e_score_correction_bias`. A buffer, not a
    /// weight: the checkpoint carries it and the choice of experts reads it, but
    /// no gradient reaches it. A model that left it at zero would pick other
    /// experts than the checkpoint was trained with.
    pub e_score_correction_bias: Buffer,
    /// How scores, the top-k and the weights are computed.
    pub routing: DeepseekRouting,
}
impl_module_params!(DeepSeekMoEGate; weight, e_score_correction_bias);

impl DeepSeekMoEGate {
    pub fn new(config: &DeepSeekConfig) -> Result<Self> {
        let num_experts = config.n_routed_experts.unwrap_or(8);
        let routing = config.routing();
        routing
            .validate(num_experts)
            .map_err(|e| Exception::custom(format!("deepseek config: {e}")))?;
        let bound = (config.hidden_size as f32).sqrt().recip();
        Ok(Self {
            weight: pmetal_runtime::compat::random::uniform_range(
                -bound,
                bound,
                &[num_experts, config.hidden_size],
                Dtype::Float32,
            ),
            e_score_correction_bias: Buffer(Array::zeros_f32(&[num_experts])),
            routing,
        })
    }

    /// `[.., hidden]` to each token's experts, `[tokens, top_k]` indices, and
    /// their weights, `[tokens, top_k]` `f32`.
    ///
    /// The logits are computed in `f32` from an `f32` hidden state and gate,
    /// as transformers does (`F.linear(x.float(), W.float())`), so the
    /// near-ties the choice of experts turns on resolve as they do there.
    pub fn forward(&self, x: &Array) -> Result<(Array, Array)> {
        let shape = x.shape();
        let hidden_size = *shape.last().unwrap_or(&0);
        let token_count: i32 = shape[..shape.len().saturating_sub(1)].iter().product();
        let f32 = Dtype::Float32.as_i32();
        let hidden_flat = x.reshape(&[token_count, hidden_size]).as_dtype(f32);
        let logits = hidden_flat.matmul(&self.weight.as_dtype(f32).t());
        Ok(self
            .routing
            .route(&logits, Some(&self.e_score_correction_bias.as_dtype(f32))))
    }
}

#[derive(Debug)]
pub struct DeepSeekMoE {
    pub gate: DeepSeekMoEGate,
    pub switch_mlp: SwitchMlp,
    pub shared_experts: Option<DeepSeekMLP>,
}
// The layout of a checkpoint after `loader::experts::stack_experts`: `gate.weight`,
// `gate.e_score_correction_bias`, `switch_mlp.{gate,up,down}_proj.weight`,
// `shared_experts.*`.
impl_module_params!(DeepSeekMoE; gate, switch_mlp, shared_experts);

impl DeepSeekMoE {
    pub fn new(config: &DeepSeekConfig) -> Result<Self> {
        let shared_experts = if let Some(n_shared) = config.n_shared_experts {
            Some(DeepSeekMLP::new(
                config.hidden_size,
                config.moe_intermediate_size * n_shared,
            )?)
        } else {
            None
        };
        Ok(Self {
            gate: DeepSeekMoEGate::new(config)?,
            switch_mlp: SwitchMlp::new(
                config.n_routed_experts.unwrap_or(8),
                config.hidden_size,
                config.moe_intermediate_size,
            ),
            shared_experts,
        })
    }

    /// Every expert over every token, each weighted by the gate's weight for
    /// it where the gate picked it and zero elsewhere: the routed sum written
    /// out densely, from primitives.
    #[cfg(test)]
    fn forward_reference(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let hidden = shape[shape.len() - 1];
        let flat = x.reshape(&[-1, hidden]);
        let (expert_indices, expert_weights) = self.gate.forward(&flat)?;
        let mut routed = ops::zeros_dtype(&[flat.dim(0), hidden], flat.dtype());
        let expert = |linear: &pmetal_runtime::nn::moe::Linear, e: i32| {
            ops::slice_axis(&linear.weight, 0, e, e + 1).squeeze(0)
        };
        for e in 0..self.switch_mlp.gate_proj.weight.dim(0) {
            let picked = expert_indices
                .equal(&Array::from_int(e))
                .as_dtype(expert_weights.dtype().as_i32());
            let weight = picked.multiply(&expert_weights).sum_axis(-1, true);
            let hidden_act = nn::silu(&flat.matmul(&expert(&self.switch_mlp.gate_proj, e).t()))
                .multiply(&flat.matmul(&expert(&self.switch_mlp.up_proj, e).t()));
            let out = hidden_act.matmul(&expert(&self.switch_mlp.down_proj, e).t());
            routed = routed.add(&out.multiply(&weight));
        }
        let moe_out = routed.reshape(&shape);
        if let Some(ref mut shared) = self.shared_experts {
            Ok(moe_out.add(&shared.forward(x)?))
        } else {
            Ok(moe_out)
        }
    }

    /// The routed experts through grouped gather matmuls
    /// ([`SwitchMlp::forward`], the `switch_glu` the native engine runs too),
    /// plus the shared expert.
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let hidden_size = shape[shape.len() - 1];
        let hidden_flat = x.reshape(&[-1, hidden_size]);
        let (expert_indices, expert_weights) = self.gate.forward(&hidden_flat)?;
        let routed = self.switch_mlp.forward(
            &hidden_flat,
            &expert_indices,
            &expert_weights.as_dtype(hidden_flat.dtype().as_i32()),
        )?;
        let moe_out = routed.reshape(&shape);

        if let Some(ref mut shared) = self.shared_experts {
            Ok(moe_out.add(&shared.forward(x)?))
        } else {
            Ok(moe_out)
        }
    }
}

#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per layer, never moved; boxing would add an indirection to every forward"
)]
pub enum DeepSeekMLPType {
    Dense(DeepSeekMLP),
    MoE(DeepSeekMoE),
}
impl VisitLinears for DeepSeekMLPType {
    fn visit_linears_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut nn::Linear)) {
        match self {
            Self::Dense(mlp) => mlp.visit_linears_mut(prefix, f),
            Self::MoE(moe) => moe.visit_linears_mut(prefix, f),
        }
    }
}

impl ModuleParameters for DeepSeekMLPType {
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.parameters(),
            Self::MoE(m) => m.parameters(),
        }
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.trainable_parameters(),
            Self::MoE(m) => m.trainable_parameters(),
        }
    }
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(m) => m.num_parameters(),
            Self::MoE(m) => m.num_parameters(),
        }
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Dense(m) => m.parameters_mut(),
            Self::MoE(m) => m.parameters_mut(),
        }
    }
    fn freeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Dense(m) => m.freeze_parameters(recurse),
            Self::MoE(m) => m.freeze_parameters(recurse),
        }
    }
    fn unfreeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Dense(m) => m.unfreeze_parameters(recurse),
            Self::MoE(m) => m.unfreeze_parameters(recurse),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(m) => m.all_frozen(),
            Self::MoE(m) => m.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(m) => m.any_frozen(),
            Self::MoE(m) => m.any_frozen(),
        }
    }
}
impl DeepSeekMLPType {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        match self {
            DeepSeekMLPType::Dense(mlp) => mlp.forward(x),
            DeepSeekMLPType::MoE(moe) => moe.forward(x),
        }
    }
}

#[derive(Debug)]
pub struct DeepSeekDecoderLayer {
    pub layer_id: usize,
    pub self_attn: DeepSeekAttention,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
    pub mlp: DeepSeekMLPType,
}
impl_module_params!(DeepSeekDecoderLayer; self_attn, input_layernorm, post_attention_layernorm, mlp);

impl DeepSeekDecoderLayer {
    pub fn new(config: &DeepSeekConfig, layer_id: usize) -> Result<Self> {
        let self_attn = DeepSeekAttention::new(config, layer_id)?;
        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let mlp = if config.is_moe_layer(layer_id as i32) {
            DeepSeekMLPType::MoE(DeepSeekMoE::new(config)?)
        } else {
            DeepSeekMLPType::Dense(DeepSeekMLP::new(
                config.hidden_size,
                config.intermediate_size,
            )?)
        };
        Ok(Self {
            layer_id,
            self_attn,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array> {
        std_pre_norm_forward(
            &mut self.input_layernorm,
            &mut self.self_attn,
            &mut self.post_attention_layernorm,
            &mut self.mlp,
            x,
            mask,
            cache,
            positions,
        )
    }
}

impl AttentionModule for DeepSeekAttention {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        DeepSeekAttention::forward(self, x, mask, cache, positions)
    }
}

impl MlpModule for DeepSeekMLPType {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        DeepSeekMLPType::forward(self, x)
    }
}

impl DecoderLayer for DeepSeekDecoderLayer {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        DeepSeekDecoderLayer::forward(self, x, mask, cache, positions)
    }
}

#[derive(Debug)]
pub struct DeepSeekModel {
    pub config: DeepSeekConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<DeepSeekDecoderLayer>,
    pub norm: nn::RmsNorm,
    /// Recompute each layer's activations during the backward pass instead of
    /// holding them. Training only; see `crate::checkpointing`.
    pub grad_checkpoint: bool,
}
impl_module_params!(DeepSeekModel; embed_tokens, layers, norm);

impl DeepSeekModel {
    pub fn new(config: DeepSeekConfig) -> Result<Self> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| DeepSeekDecoderLayer::new(&config, i))
            .collect::<Result<Vec<_>>>()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            grad_checkpoint: false,
        })
    }
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        self.forward_with_capture(input_ids, mask, cache, None, None)
    }

    /// Forward pass with one rotary position per token, `[seq_len]`.
    ///
    /// Packed training concatenates several sequences into one row, so the
    /// positions have to restart at each boundary instead of running through
    /// it. The block-diagonal mask keeps the content apart; this keeps the
    /// positions apart.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array> {
        self.forward_with_capture(input_ids, mask, None, positions, None)
    }

    /// Forward pass with optional hidden-state capture for DFlash
    /// speculative decoding.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
        positions: Option<&Array>,
        mut capture: Option<&mut pmetal_cache::speculative::SpecCapture>,
    ) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids);
        // Hoisted: the loop below borrows `self.layers` mutably.
        let grad_checkpoint = self.grad_checkpoint;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache.as_mut().map(|c| (&mut **c, i));
            // A cache means generation, which has no backward pass for the
            // recompute to pay for.
            h = if grad_checkpoint && layer_cache.is_none() {
                checkpointed_layer(layer, &h, mask, positions, |layer, h, mask, positions| {
                    layer.forward(h, mask, None, positions)
                })?
            } else {
                layer.forward(&h, mask, layer_cache, positions)?
            };
            if let Some(buf) = capture.as_deref_mut()
                && buf.wants_hidden_for(i)
            {
                buf.record_hidden(i, h.clone());
            }
        }
        Ok(self.norm.forward(&h))
    }

    pub fn forward_with_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Vec<Array>)> {
        let mut h = self.embed_tokens.forward(input_ids);
        let mut all_hidden = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, mask, None, None)?;
            all_hidden.push(h.clone());
        }
        let out = self.norm.forward(&h);
        Ok((out, all_hidden))
    }
}

#[derive(Debug)]
pub struct DeepSeekMTPModule {
    pub eh_proj: nn::Linear,
    pub enorm: nn::RmsNorm,
    pub hnorm: nn::RmsNorm,
    pub layer: DeepSeekDecoderLayer,
}
impl_module_params!(DeepSeekMTPModule; eh_proj, enorm, hnorm, layer);

impl DeepSeekMTPModule {
    pub fn new(config: &DeepSeekConfig, layer_idx: usize) -> Result<Self> {
        Ok(Self {
            eh_proj: nn::LinearBuilder::new(config.hidden_size * 2, config.hidden_size)
                .bias(false)
                .build()?,
            enorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            hnorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            layer: DeepSeekDecoderLayer::new(config, layer_idx)?,
        })
    }
    pub fn forward(
        &mut self,
        h_prev: &Array,
        e_curr: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let hn = self.hnorm.forward(h_prev);
        let en = self.enorm.forward(e_curr);
        let cat = pmetal_runtime::compat::ops::concatenate_axis(&[&hn, &en], -1);
        self.layer
            .forward(&self.eh_proj.forward(&cat), mask, None, None)
    }
}

#[derive(Debug)]
pub struct DeepSeek {
    pub config: DeepSeekConfig,
    pub model: DeepSeekModel,
    /// `None` when `tie_word_embeddings` ties the head to the token
    /// embedding.
    pub lm_head: Option<nn::Linear>,
    pub mtp_modules: Vec<DeepSeekMTPModule>,
}
impl_module_params!(DeepSeek; model, lm_head, mtp_modules);

impl DeepSeek {
    pub fn new(config: DeepSeekConfig) -> Result<Self> {
        let mut mtp_modules = Vec::new();
        if config.use_mtp {
            for i in 0..config.num_nextn_predict_layers {
                mtp_modules.push(DeepSeekMTPModule::new(
                    &config,
                    config.num_hidden_layers as usize + i as usize,
                )?);
            }
        }
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };
        let config_clone = config.clone();
        Ok(Self {
            config,
            model: DeepSeekModel::new(config_clone)?,
            lm_head,
            mtp_modules,
        })
    }

    /// Project hidden states to vocabulary logits, through the LM head when
    /// the checkpoint has one and the tied embedding otherwise.
    pub fn lm_head_forward(&self, hidden: &Array) -> Array {
        Self::project(&self.lm_head, &self.model.embed_tokens, hidden)
    }

    /// [`Self::lm_head_forward`] over the fields it reads, for callers that
    /// hold another field of the model mutably.
    fn project(
        lm_head: &Option<nn::Linear>,
        embed_tokens: &nn::Embedding,
        hidden: &Array,
    ) -> Array {
        match lm_head {
            Some(lm_head) => lm_head.forward(hidden),
            None => embed_tokens.as_linear(hidden),
        }
    }

    pub fn forward_mtp(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Vec<Array>> {
        let (hidden_final, _) = self.model.forward_with_hidden(input_ids, mask)?;
        let mut all_logits = vec![self.lm_head_forward(&hidden_final)];
        if !self.config.use_mtp {
            return Ok(all_logits);
        }
        let mut h_prev = hidden_final;
        let embeddings = self.model.embed_tokens.forward(input_ids);
        for mtp_module in &mut self.mtp_modules {
            let e_curr = pmetal_runtime::compat::ops::slice_axis_from(&embeddings, 1, 1);
            let e_curr_padded = pmetal_runtime::compat::ops::concatenate_axis(
                &[
                    &e_curr,
                    &Array::zeros_f32(&[embeddings.dim(0), 1, self.config.hidden_size]),
                ],
                1,
            );
            h_prev = mtp_module.forward(&h_prev, &e_curr_padded, mask)?;
            all_logits.push(Self::project(
                &self.lm_head,
                &self.model.embed_tokens,
                &h_prev,
            ));
        }
        Ok(all_logits)
    }
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        let hidden = self.model.forward(input_ids, mask, cache)?;
        Ok(self.lm_head_forward(&hidden))
    }

    /// Forward pass with one rotary position per token; see
    /// [`DeepSeekModel::forward_with_positions`].
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array> {
        let hidden = self
            .model
            .forward_with_positions(input_ids, mask, positions)?;
        Ok(self.lm_head_forward(&hidden))
    }

    /// Forward pass that records hidden states into a DFlash capture
    /// buffer for every requested layer index.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        capture: &mut pmetal_cache::speculative::SpecCapture,
    ) -> Result<Array> {
        let hidden =
            self.model
                .forward_with_capture(input_ids, mask, cache, None, Some(capture))?;
        Ok(self.lm_head_forward(&hidden))
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        KVCache::new(
            pmetal_cache::kv_cache::KVCacheConfig::new(
                self.config.num_hidden_layers as usize,
                max_seq_len,
                self.config.num_attention_heads as usize,
                self.config.q_head_dim() as usize,
            )
            .with_value_head_dim(self.config.v_head_dim as usize),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serial_test::serial;

    fn tiny_deepseek_moe_config() -> DeepSeekConfig {
        // All attention dimensions are proportional to hidden_size=16 so that
        // projection shapes are consistent when the full model is constructed.
        // Without these overrides the inherited defaults (kv_lora_rank=512,
        // q_lora_rank=1536, qk_nope_head_dim=128, v_head_dim=128) produce
        // cache sizes that don't match the actual tensor shapes, causing SIGSEGV
        // in the turboquant Metal kernels.
        DeepSeekConfig {
            vocab_size: 512,
            hidden_size: 16,
            intermediate_size: 32,
            moe_intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            n_shared_experts: Some(1),
            n_routed_experts: Some(4),
            num_experts_per_tok: 2,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            // Attention dimensions: q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 32+32=64.
            // cache head_dim = 64, value_head_dim = 32.
            kv_lora_rank: 8,
            q_lora_rank: Some(16),
            qk_nope_head_dim: 32,
            qk_rope_head_dim: 32,
            v_head_dim: 32,
            ..DeepSeekConfig::default()
        }
    }

    /// A random router bias (zero until a checkpoint loads it), so the
    /// choice of experts below depends on it.
    fn randomize_bias(moe: &mut DeepSeekMoE) {
        let experts = moe.switch_mlp.gate_proj.weight.dim(0);
        moe.gate.e_score_correction_bias = Buffer(pmetal_runtime::compat::random::uniform_range(
            -0.5,
            0.5,
            &[experts],
            Dtype::Float32,
        ));
    }

    fn max_diff(a: &Array, b: &Array) -> f32 {
        let diff = a.subtract(b).abs().max(None);
        diff.eval();
        diff.item::<f32>()
    }

    #[test]
    #[serial]
    fn test_deepseek_moe_matches_reference() {
        let config = tiny_deepseek_moe_config();
        let mut moe = DeepSeekMoE::new(&config).unwrap();
        randomize_bias(&mut moe);
        // 80 tokens x 2 experts: past the point where the dispatch groups the
        // (token, expert) pairs by expert; the parity fixtures (12 tokens) run
        // the router-order path.
        let x = pmetal_runtime::compat::random::uniform_range(
            -1.0,
            1.0,
            &[2, 40, config.hidden_size],
            Dtype::Float32,
        );

        let reference = moe.forward_reference(&x).unwrap();
        let fast = moe.forward(&x).unwrap();
        pmetal_runtime::check_unobserved_error().expect("runtime error");
        assert!(
            max_diff(&fast, &reference) < 1e-4,
            "deepseek MoE drifted from reference"
        );
    }

    /// The experts are the parameters, with nothing derived from them: a
    /// replaced weight is what the next forward runs.
    #[test]
    #[serial]
    fn test_deepseek_moe_runs_the_weights_it_holds() {
        let config = tiny_deepseek_moe_config();
        let mut moe = DeepSeekMoE::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::uniform_range(
            -1.0,
            1.0,
            &[1, 4, config.hidden_size],
            Dtype::Float32,
        );

        let before = moe.forward(&x).unwrap();
        moe.switch_mlp.gate_proj.weight = Array::zeros_f32(&[
            config.n_routed_experts.unwrap(),
            config.moe_intermediate_size,
            config.hidden_size,
        ]);
        let reference = moe.forward_reference(&x).unwrap();
        let after = moe.forward(&x).unwrap();
        assert!(max_diff(&after, &reference) < 1e-4);
        assert!(
            max_diff(&after, &before) > 1e-3,
            "zeroing every gate projection changed nothing"
        );
    }

    /// A config that ties the head to the embedding builds no head, and the
    /// logits are the hidden states against the embedding table: a model that
    /// built one anyway would answer with a head no checkpoint fills.
    #[test]
    #[serial]
    fn test_deepseek_tied_head_projects_through_the_embedding() {
        let mut model = DeepSeek::new(DeepSeekConfig {
            tie_word_embeddings: true,
            ..tiny_deepseek_moe_config()
        })
        .unwrap();
        assert!(model.lm_head.is_none());
        let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let hidden = model.model.forward(&ids, None, None).unwrap();
        let want = hidden.matmul(&model.model.embed_tokens.weight.value.t());
        let got = model.forward(&ids, None, None).unwrap();
        let diff = got.subtract(&want).abs().max(None);
        diff.eval();
        assert!(diff.item::<f32>() < 1e-5);
    }

    #[test]
    fn test_deepseek_cache_uses_asymmetric_value_head_dim() {
        let config = DeepSeekConfig::default();
        let model = DeepSeek::new(config.clone()).unwrap();
        let cache = model.create_cache(128);

        assert_eq!(cache.config().head_dim, config.q_head_dim() as usize);
        assert_eq!(cache.config().value_head_dim, config.v_head_dim as usize);
    }

    #[test]
    #[serial]
    fn test_deepseek_forward_with_turboquant_cache_handles_asymmetric_kv_dims() {
        let config = tiny_deepseek_moe_config();
        let mut model = DeepSeek::new(config).unwrap();
        let base_cache = model.create_cache(16);
        let mut cache = KVCache::new(base_cache.config().clone().with_turboquant(4, 3));

        let first = Array::from_slice(&[1i32], &[1, 1]);
        let second = Array::from_slice(&[2i32], &[1, 1]);

        let first_logits = model.forward(&first, None, Some(&mut cache)).unwrap();
        let second_logits = model.forward(&second, None, Some(&mut cache)).unwrap();
        first_logits.eval().unwrap();
        second_logits.eval().unwrap();

        assert_eq!(cache.seq_len(), 2);
        assert_eq!(cache.config().head_dim, model.config.q_head_dim() as usize);
        assert_eq!(
            cache.config().value_head_dim,
            model.config.v_head_dim as usize
        );
        assert_eq!(first_logits.dim(0), 1);
        assert_eq!(second_logits.dim(0), 1);
    }
}
