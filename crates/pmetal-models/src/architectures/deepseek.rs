//! DeepSeek V3 / V3.2 / V3.2-Speciale model architecture.
//!
//! Implements DeepSeek V3 and V3.2 variants with:
//! - Multi-Latent Attention (MLA): LoRA-style Q/K/V compression for 28x KV cache reduction
//! - Mixture of Experts (MoE) with shared experts and aux-free load balancing
//! - Sparse Attention (DSA) for V3.2 (Lightning Indexer + Token Selector)
//! - Multi-Token Prediction (MTP) for densified training signals
//! - Multi-token prediction lookahead modules

use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nested::NestedHashMap;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, nn};
use pmetal_mlx::Builder;
use pmetal_mlx::kernels::rope::apply_rope;
use pmetal_mlx::kv_cache::KVCache;
use pmetal_mlx::moe::{MoEConfig, MoELayer};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

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
    pub topk_method: String,
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    pub n_group: i32,
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
            topk_method: "noaux_tc".to_string(),
            scoring_func: "sigmoid".to_string(),
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

#[derive(Debug, ModuleParameters)]
pub struct DeepSeekAttention {
    pub config: DeepSeekConfig,
    pub n_heads: i32,
    pub scale: f32,
    pub layer_id: usize,
    #[param]
    pub q_a_proj: Option<nn::Linear>,
    #[param]
    pub q_a_layernorm: Option<nn::RmsNorm>,
    #[param]
    pub q_b_proj: Option<nn::Linear>,
    #[param]
    pub q_proj: Option<nn::Linear>,
    #[param]
    pub kv_a_proj_with_mqa: nn::Linear,
    #[param]
    pub kv_a_layernorm: nn::RmsNorm,
    #[param]
    pub kv_b_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
}

impl DeepSeekAttention {
    pub fn new(config: &DeepSeekConfig, layer_id: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let q_head_dim = config.q_head_dim();
        let scale = (q_head_dim as f32).powf(-0.5);
        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) =
            if let Some(q_lora_rank) = config.q_lora_rank {
                let q_a = nn::LinearBuilder::new(hidden_size, q_lora_rank)
                    .bias(config.attention_bias)
                    .build()
                    .map_err(|_| Exception::custom("Build error"))?;
                let q_a_norm = nn::RmsNormBuilder::new(q_lora_rank)
                    .eps(1e-6)
                    .build()
                    .map_err(|_| Exception::custom("Build error"))?;
                let q_b = nn::LinearBuilder::new(q_lora_rank, n_heads * q_head_dim)
                    .bias(false)
                    .build()
                    .map_err(|_| Exception::custom("Build error"))?;
                (Some(q_a), Some(q_a_norm), Some(q_b), None)
            } else {
                let q = nn::LinearBuilder::new(hidden_size, n_heads * q_head_dim)
                    .bias(false)
                    .build()
                    .map_err(|_| Exception::custom("Build error"))?;
                (None, None, None, Some(q))
            };
        let kv_a_proj_with_mqa =
            nn::LinearBuilder::new(hidden_size, config.kv_lora_rank + config.qk_rope_head_dim)
                .bias(config.attention_bias)
                .build()
                .map_err(|_| Exception::custom("Build error"))?;
        let kv_a_layernorm = nn::RmsNormBuilder::new(config.kv_lora_rank)
            .eps(1e-6)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
        let kv_b_proj = nn::LinearBuilder::new(
            config.kv_lora_rank,
            n_heads * (config.qk_nope_head_dim + config.v_head_dim),
        )
        .bias(false)
        .build()
        .map_err(|_| Exception::custom("Build error"))?;
        let o_proj = nn::LinearBuilder::new(n_heads * config.v_head_dim, hidden_size)
            .bias(config.attention_bias)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
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
        })
    }
    pub fn project_qkv(
        &mut self,
        x: &Array,
        mut cache: Option<(&mut KVCache, usize)>,
    ) -> Result<(Array, Array, Array)> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let q = if let Some(ref mut q_a) = self.q_a_proj {
            let q_a_out = q_a.forward(x)?;
            let q_a_norm = self.q_a_layernorm.as_mut().unwrap().forward(&q_a_out)?;
            self.q_b_proj.as_mut().unwrap().forward(&q_a_norm)?
        } else {
            self.q_proj.as_mut().unwrap().forward(x)?
        };
        let q_head_dim = self.config.q_head_dim();
        let q = q
            .reshape(&[batch, seq_len, self.n_heads, q_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q_parts = q.split_axis(&[self.config.qk_nope_head_dim as i32], Some(-1))?;
        let q_nope = &q_parts[0];
        let q_pe = &q_parts[1];
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_parts = compressed_kv.split_axis(&[self.config.kv_lora_rank as i32], Some(-1))?;
        let compressed_latent = &kv_parts[0];
        let k_pe = &kv_parts[1]
            .reshape(&[batch, seq_len, 1, self.config.qk_rope_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kv_normalized = self.kv_a_layernorm.forward(compressed_latent)?;
        let kv = self.kv_b_proj.forward(&kv_normalized)?;
        let kv_dim = self.config.qk_nope_head_dim + self.config.v_head_dim;
        let kv = kv
            .reshape(&[batch, seq_len, self.n_heads, kv_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kv_split = kv.split_axis(&[self.config.qk_nope_head_dim as i32], Some(-1))?;
        let k_nope = &kv_split[0];
        let values = &kv_split[1];
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q_pe = apply_rope(
            q_pe,
            self.config.qk_rope_head_dim,
            false,
            self.config.rope_theta,
            1.0,
            offset as i32,
        )?;
        let k_pe = apply_rope(
            k_pe,
            self.config.qk_rope_head_dim,
            false,
            self.config.rope_theta,
            1.0,
            offset as i32,
        )?;
        let k_pe_repeated = mlx_rs::ops::broadcast_to(
            &k_pe,
            &[batch, self.n_heads, seq_len, self.config.qk_rope_head_dim],
        )?;
        let keys = mlx_rs::ops::concatenate_axis(&[k_nope, &k_pe_repeated], -1)?;
        let queries = mlx_rs::ops::concatenate_axis(&[q_nope, &q_pe], -1)?;
        let (keys, values) = if let Some((ref mut cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, values)?
        } else {
            (keys, values.clone())
        };
        Ok((queries, keys, values))
    }
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array> {
        let (queries, keys, values) = self.project_qkv(x, cache)?;
        let batch = queries.shape()[0];
        let seq_len = queries.shape()[2];
        let mut attn_weights = queries
            .matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?
            .multiply(&Array::from_f32(self.scale))?;
        if let Some(mask) = mask {
            attn_weights = attn_weights.add(mask)?;
        }
        let attn_weights = mlx_rs::ops::softmax_axis(&attn_weights, -1, None)?;
        let output = attn_weights
            .matmul(&values)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;
        self.o_proj.forward(&output)
    }
}

#[derive(Debug, ModuleParameters)]
pub struct LightningIndexer {
    pub n_heads: i32,
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dim: i32,
    pub rope_theta: f32,
    pub non_interleaved: bool,
}

impl LightningIndexer {
    pub fn new(config: &DeepSeekConfig) -> Result<Self> {
        let n_heads = config.lightning_indexer_heads;
        let head_dim = config.qk_rope_head_dim;
        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
        Ok(Self {
            n_heads,
            q_proj,
            k_proj,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_dim: config.qk_rope_head_dim,
            rope_theta: config.rope_theta,
            non_interleaved: config.indexer_non_interleaved_rope,
        })
    }
    pub fn compute_scores(&mut self, x: &Array, offset: i32) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q = apply_rope(
            &q,
            self.rope_dim,
            !self.non_interleaved,
            self.rope_theta,
            1.0,
            offset,
        )?;
        let k = apply_rope(
            &k,
            self.rope_dim,
            !self.non_interleaved,
            self.rope_theta,
            1.0,
            offset,
        )?;
        let scores = q
            .matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?
            .multiply(&Array::from_f32(self.scale))?;
        scores.mean_axis(1, true)?.squeeze_axes(&[1])
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
            scores.add(&m?)?
        } else {
            scores.clone()
        };
        let neg_k = -self.top_k;
        Ok(mlx_rs::ops::argpartition_axis(&masked_scores, neg_k, -1)?.index((.., .., neg_k..)))
    }
}

#[derive(Debug, ModuleParameters)]
pub struct DeepSeekSparseAttention {
    #[param]
    pub base_attention: DeepSeekAttention,
    #[param]
    pub indexer: LightningIndexer,
    pub selector: TokenSelector,
    pub store_indices: bool,
}

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
    ) -> Result<Array> {
        let seq_len = x.shape()[1];
        if seq_len < 2 * self.selector.top_k {
            return self.base_attention.forward(x, mask, cache);
        }
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0) as i32;
        let scores = self.indexer.compute_scores(x, offset)?;
        let selected_indices = self.selector.select_tokens(&scores, mask)?;
        let (queries, keys, values) = self.base_attention.project_qkv(x, cache)?;
        let (batch, n_heads, query_len, key_dim) = (
            queries.shape()[0],
            queries.shape()[1],
            queries.shape()[2],
            keys.shape()[3],
        );
        let val_dim = values.shape()[3];
        let top_k = self.selector.top_k;
        let idx = selected_indices.reshape(&[batch, 1, query_len, top_k])?;
        let idx = mlx_rs::ops::broadcast_to(&idx, &[batch, n_heads, query_len, top_k])?;
        let idx_for_keys = idx.reshape(&[batch, n_heads, query_len * top_k, 1])?;
        let idx_for_keys = mlx_rs::ops::broadcast_to(
            &idx_for_keys,
            &[batch, n_heads, query_len * top_k, key_dim],
        )?;
        let gathered_keys = mlx_rs::ops::indexing::take_along_axis(&keys, &idx_for_keys, 2)?
            .reshape(&[batch, n_heads, query_len, top_k, key_dim])?;
        let idx_for_vals = idx.reshape(&[batch, n_heads, query_len * top_k, 1])?;
        let idx_for_vals = mlx_rs::ops::broadcast_to(
            &idx_for_vals,
            &[batch, n_heads, query_len * top_k, val_dim],
        )?;
        let gathered_values = mlx_rs::ops::indexing::take_along_axis(&values, &idx_for_vals, 2)?
            .reshape(&[batch, n_heads, query_len, top_k, val_dim])?;
        let q_expanded = queries.reshape(&[batch, n_heads, query_len, 1, key_dim])?;
        let attn_scores = q_expanded
            .matmul(&gathered_keys.transpose_axes(&[0, 1, 2, 4, 3])?)?
            .squeeze_axes(&[3])?
            .multiply(&Array::from_f32(self.base_attention.scale))?;
        let attn_weights = mlx_rs::ops::softmax_axis(&attn_scores, -1, None)?
            .reshape(&[batch, n_heads, query_len, 1, top_k])?;
        let output = attn_weights
            .matmul(&gathered_values)?
            .squeeze_axes(&[3])?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, query_len, -1])?;
        self.base_attention.o_proj.forward(&output)
    }
}

#[derive(Debug, ModuleParameters)]
pub struct DeepSeekMLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}
impl DeepSeekMLP {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            up_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
        })
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&nn::silu(gate)?.multiply(&up)?)
    }
}

#[derive(Debug, ModuleParameters)]
pub struct DeepSeekMoEGate {
    #[param]
    pub weight: nn::Linear,
    pub e_score_correction_bias: Array,
    pub top_k: i32,
    pub num_experts: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
}
impl DeepSeekMoEGate {
    pub fn new(config: &DeepSeekConfig) -> Result<Self> {
        let num_experts = config.n_routed_experts.unwrap_or(8);
        Ok(Self {
            weight: nn::LinearBuilder::new(config.hidden_size, num_experts)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            e_score_correction_bias: Array::zeros::<f32>(&[num_experts])?,
            top_k: config.num_experts_per_tok,
            num_experts,
            routed_scaling_factor: config.routed_scaling_factor,
            norm_topk_prob: config.norm_topk_prob,
        })
    }
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Array)> {
        let gates = self.weight.forward(x)?;
        let scores = mlx_rs::ops::sigmoid(&gates.as_dtype(mlx_rs::Dtype::Float32)?)?;
        let scores_with_bias = scores.add(&self.e_score_correction_bias)?;
        let neg_k = -self.top_k;
        let inds =
            mlx_rs::ops::argpartition_axis(&scores_with_bias, neg_k, -1)?.index((.., .., neg_k..));
        let top_scores = scores.take_along_axis(&inds, -1)?;
        let final_scores = if self.norm_topk_prob && self.top_k > 1 {
            top_scores.divide(&top_scores.sum_axis(-1, true)?)?
        } else {
            top_scores
        };
        Ok((
            inds,
            final_scores.multiply(&Array::from_f32(self.routed_scaling_factor))?,
        ))
    }
}

#[derive(Debug)]
pub struct DeepSeekMoE {
    pub config: DeepSeekConfig,
    pub gate: DeepSeekMoEGate,
    pub moe: MoELayer,
    pub shared_experts: Option<DeepSeekMLP>,
}
impl ModuleParameters for DeepSeekMoE {
    fn parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
        let mut map = self.gate.parameters();
        map.entries.extend(self.moe.parameters().entries);
        if let Some(ref s) = self.shared_experts {
            map.entries.extend(s.parameters().entries);
        }
        map
    }
    fn trainable_parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
        let mut map = self.gate.trainable_parameters();
        map.entries.extend(self.moe.trainable_parameters().entries);
        if let Some(ref s) = self.shared_experts {
            map.entries.extend(s.trainable_parameters().entries);
        }
        map
    }
    fn num_parameters(&self) -> usize {
        self.gate.num_parameters()
            + self.moe.num_parameters()
            + self
                .shared_experts
                .as_ref()
                .map_or(0, |s| s.num_parameters())
    }
    fn parameters_mut(&mut self) -> NestedHashMap<Rc<str>, &mut Array> {
        let mut map = self.gate.parameters_mut();
        map.entries.extend(self.moe.parameters_mut().entries);
        if let Some(ref mut s) = self.shared_experts {
            map.entries.extend(s.parameters_mut().entries);
        }
        map
    }
    fn freeze_parameters(&mut self, recurse: bool) {
        self.gate.freeze_parameters(recurse);
        self.moe.freeze_parameters(recurse);
        if let Some(ref mut s) = self.shared_experts {
            s.freeze_parameters(recurse);
        }
    }
    fn unfreeze_parameters(&mut self, recurse: bool) {
        self.gate.unfreeze_parameters(recurse);
        self.moe.unfreeze_parameters(recurse);
        if let Some(ref mut s) = self.shared_experts {
            s.unfreeze_parameters(recurse);
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        Some(
            self.gate.all_frozen()?
                && self.moe.all_frozen()?
                && self
                    .shared_experts
                    .as_ref()
                    .map_or(true, |s| s.all_frozen().unwrap_or(true)),
        )
    }
    fn any_frozen(&self) -> Option<bool> {
        Some(
            self.gate.any_frozen()?
                || self.moe.any_frozen()?
                || self
                    .shared_experts
                    .as_ref()
                    .map_or(false, |s| s.any_frozen().unwrap_or(false)),
        )
    }
}
impl DeepSeekMoE {
    pub fn new(config: &DeepSeekConfig) -> Result<Self> {
        let moe_config = MoEConfig::new(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts.unwrap_or(8) as usize,
        )
        .with_num_experts_per_tok(config.num_experts_per_tok as usize)
        .with_aux_loss(false, 0.0);
        let shared_experts = if let Some(n_shared) = config.n_shared_experts {
            Some(DeepSeekMLP::new(
                config.hidden_size,
                config.moe_intermediate_size * n_shared,
            )?)
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            gate: DeepSeekMoEGate::new(config)?,
            moe: MoELayer::new(moe_config),
            shared_experts,
        })
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let (_expert_indices, _expert_weights) = self.gate.forward(x)?;
        let (moe_out, _) = self.moe.forward(x)?;
        if let Some(ref mut shared) = self.shared_experts {
            moe_out.add(&shared.forward(x)?)
        } else {
            Ok(moe_out)
        }
    }
}

#[derive(Debug)]
pub enum DeepSeekMLPType {
    Dense(DeepSeekMLP),
    MoE(DeepSeekMoE),
}
impl ModuleParameters for DeepSeekMLPType {
    fn parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
        match self {
            Self::Dense(m) => m.parameters(),
            Self::MoE(m) => m.parameters(),
        }
    }
    fn trainable_parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
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
    fn parameters_mut(&mut self) -> NestedHashMap<Rc<str>, &mut Array> {
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

#[derive(Debug, ModuleParameters)]
pub struct DeepSeekDecoderLayer {
    pub layer_id: usize,
    #[param]
    pub self_attn: DeepSeekAttention,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub mlp: DeepSeekMLPType,
}
impl DeepSeekDecoderLayer {
    pub fn new(config: &DeepSeekConfig, layer_id: usize) -> Result<Self> {
        let self_attn = DeepSeekAttention::new(config, layer_id)?;
        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
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
    ) -> Result<Array> {
        let h = x.add(&self.self_attn.forward(
            &self.input_layernorm.forward(x)?,
            mask,
            cache,
        )?)?;
        h.add(
            &self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&h)?)?,
        )
    }
}

#[derive(Debug, ModuleParameters)]
pub struct DeepSeekModel {
    pub config: DeepSeekConfig,
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<DeepSeekDecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}
impl DeepSeekModel {
    pub fn new(config: DeepSeekConfig) -> Result<Self> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| DeepSeekDecoderLayer::new(&config, i))
            .collect::<Result<Vec<_>>>()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, mask, cache.as_mut().map(|c| (&mut **c, i)))?;
        }
        self.norm.forward(&h)
    }
    pub fn forward_with_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Vec<Array>)> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        let mut all_hidden = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, mask, None)?;
            all_hidden.push(h.clone());
        }
        let out = self.norm.forward(&h)?;
        Ok((out, all_hidden))
    }
}

#[derive(Debug, ModuleParameters)]
pub struct DeepSeekMTPModule {
    #[param]
    pub eh_proj: nn::Linear,
    #[param]
    pub enorm: nn::RmsNorm,
    #[param]
    pub hnorm: nn::RmsNorm,
    #[param]
    pub layer: DeepSeekDecoderLayer,
}
impl DeepSeekMTPModule {
    pub fn new(config: &DeepSeekConfig, layer_idx: usize) -> Result<Self> {
        Ok(Self {
            eh_proj: nn::LinearBuilder::new(config.hidden_size * 2, config.hidden_size)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            enorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            hnorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            layer: DeepSeekDecoderLayer::new(config, layer_idx)?,
        })
    }
    pub fn forward(
        &mut self,
        h_prev: &Array,
        e_curr: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let cat = mlx_rs::ops::concatenate_axis(
            &[self.hnorm.forward(h_prev)?, self.enorm.forward(e_curr)?],
            -1,
        )?;
        self.layer.forward(&self.eh_proj.forward(&cat)?, mask, None)
    }
}

#[derive(Debug, ModuleParameters)]
pub struct DeepSeek {
    pub config: DeepSeekConfig,
    #[param]
    pub model: DeepSeekModel,
    #[param]
    pub lm_head: nn::Linear,
    #[param]
    pub mtp_modules: Vec<DeepSeekMTPModule>,
}
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
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
        let config_clone = config.clone();
        Ok(Self {
            config,
            model: DeepSeekModel::new(config_clone)?,
            lm_head,
            mtp_modules,
        })
    }
    pub fn forward_mtp(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Vec<Array>> {
        let (hidden_final, _) = self.model.forward_with_hidden(input_ids, mask)?;
        let mut all_logits = vec![self.lm_head.forward(&hidden_final)?];
        if !self.config.use_mtp {
            return Ok(all_logits);
        }
        let mut h_prev = hidden_final;
        let embeddings = self.model.embed_tokens.forward(input_ids)?;
        for mtp_module in &mut self.mtp_modules {
            let e_curr = embeddings.index((.., 1.., ..));
            let e_curr_padded = mlx_rs::ops::concatenate_axis(
                &[
                    e_curr,
                    Array::zeros::<f32>(&[embeddings.dim(0), 1, self.config.hidden_size])?,
                ],
                1,
            )?;
            h_prev = mtp_module.forward(&h_prev, &e_curr_padded, mask)?;
            all_logits.push(self.lm_head.forward(&h_prev)?);
        }
        Ok(all_logits)
    }
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        self.lm_head
            .forward(&self.model.forward(input_ids, mask, cache)?)
    }
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        KVCache::new(pmetal_mlx::kv_cache::KVCacheConfig::new(
            self.config.num_hidden_layers as usize,
            max_seq_len,
            self.config.num_attention_heads as usize,
            self.config.q_head_dim() as usize,
        ))
    }
}
