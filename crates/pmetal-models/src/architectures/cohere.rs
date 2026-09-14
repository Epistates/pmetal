//! Cohere Command R architecture.
//!
//! Command R is a family of multilingual models designed for:
//! - Retrieval-Augmented Generation (RAG)
//! - Tool use and function calling
//! - Long context (128K tokens)
//!
//! Variants:
//! - **Command R**: 35B parameters
//! - **Command R+**: 104B parameters
//! - **Command A**: 111B parameters (2025)
use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParameters, ModuleParametersExt, nn, random,
};
use pmetal_bridge::impl_module_params;

use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopePositions, rope},
};
use pmetal_mlx::kv_cache::KVCache;

use serde::{Deserialize, Serialize};

use crate::checkpointing::checkpointed_layer;
use crate::decoder_layer::{AttentionModule, MlpModule, NormModule};
use crate::traits::ModelConfig;

/// Cohere model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub layer_norm_eps: f32,
    /// Output logit scaling factor. Cohere multiplies the LM-head logits by
    /// this (HF `CohereForCausalLM` / mlx-lm `Model.__call__`:
    /// `out = out * logit_scale`). Command-R family ships `0.0625`.
    #[serde(default = "default_logit_scale")]
    pub logit_scale: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Use sliding window attention for certain layers.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Sliding window size.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// Layers that use global attention (no sliding window).
    /// Pattern: every 4th layer uses global attention.
    #[serde(default)]
    pub global_attention_layers: Option<Vec<i32>>,
}

fn default_sliding_window() -> i32 {
    4096
}

fn default_logit_scale() -> f32 {
    0.0625
}

impl Default for CohereConfig {
    fn default() -> Self {
        // Default for Command R 35B
        Self {
            vocab_size: 256000,
            hidden_size: 8192,
            intermediate_size: 22528,
            num_hidden_layers: 40,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 131072,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            logit_scale: 0.0625,
            tie_word_embeddings: true,
            use_sliding_window: true,
            sliding_window: 4096,
            global_attention_layers: None,
        }
    }
}

impl CohereConfig {
    /// Check if a layer uses global attention.
    pub fn uses_global_attention(&self, layer_idx: i32) -> bool {
        if let Some(ref layers) = self.global_attention_layers {
            layers.contains(&layer_idx)
        } else {
            // Default: every 4th layer uses global attention
            (layer_idx + 1) % 4 == 0
        }
    }

    /// Create config for Command R 35B.
    pub fn command_r() -> Self {
        Self::default()
    }

    /// Create config for Command R+ 104B.
    pub fn command_r_plus() -> Self {
        Self {
            hidden_size: 12288,
            intermediate_size: 33792,
            num_hidden_layers: 64,
            num_attention_heads: 96,
            num_key_value_heads: 12,
            ..Default::default()
        }
    }

    /// Create config for Command A 111B (2025).
    pub fn command_a() -> Self {
        Self {
            hidden_size: 12288,
            intermediate_size: 33792,
            num_hidden_layers: 64,
            num_attention_heads: 96,
            num_key_value_heads: 12,
            max_position_embeddings: 262144, // 256K context
            ..Default::default()
        }
    }
}

impl ModelConfig for CohereConfig {
    fn model_type(&self) -> &str {
        "cohere"
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
        self.intermediate_size
    }
    fn max_position_embeddings(&self) -> i32 {
        self.max_position_embeddings
    }
    fn norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

// =============================================================================
// Model Components
// =============================================================================

/// Cohere MLP layer.
#[derive(Debug)]
pub struct CohereMLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(CohereMLP; gate_proj, up_proj, down_proj);

impl CohereMLP {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let gate = nn::silu(&gate);
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up);
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// Cohere attention with optional sliding window.
#[derive(Debug)]
pub struct CohereAttention {
    pub layer_idx: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
    pub use_sliding_window: bool,
    pub sliding_window: i32,

    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
}
impl_module_params!(CohereAttention; q_proj, k_proj, v_proj, o_proj);

impl CohereAttention {
    pub fn new(config: &CohereConfig, layer_idx: usize) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

        let q_proj = nn::LinearBuilder::new(hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, hidden_size)
            .bias(false)
            .build()?;

        // Check if this layer uses global attention
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

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        _position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None, None)
    }

    /// Forward pass with optional KV cache.
    ///
    /// Mirrors the Llama-style serial cached path: project → reshape →
    /// **transpose first** → `apply_rope` with axis -2 = seq → cache write
    /// → fused SDPA. The previous implementation applied RoPE before the
    /// transpose on `[B, S, H, D]`, which silently misrotated heads as
    /// positions for `offset > 0` (same axis bug fixed in Phi). Sliding
    /// window per-layer masking is not yet plumbed; configs with
    /// `use_sliding_window = true` AND non-global layers must build a
    /// custom mask externally and pass it via `mask`.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let q = Module::forward(&mut self.q_proj, x)?;
        let k = Module::forward(&mut self.k_proj, x)?;
        let v = Module::forward(&mut self.v_proj, x)?;

        // Reshape + transpose to [B, heads, seq, head_dim] BEFORE RoPE so
        // axis -2 is correctly the seq axis.
        let q = q
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let k = k
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let v = v
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        // Cohere uses *traditional* (interleaved) RoPE — HF `CohereRotaryEmbedding`
        // rotates adjacent pairs, and mlx-lm builds `nn.RoPE(..., traditional=True)`.
        // The split-half (`traditional=false`) form rotates the wrong element
        // pairs and silently corrupts every attention score.
        let rope_positions = RopePositions::resolve(
            positions,
            cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0),
        );
        let q = rope(
            &q,
            rope_positions,
            self.head_dim,
            true,
            self.rope_theta,
            1.0,
        )?;
        let k = rope(
            &k,
            rope_positions,
            self.head_dim,
            true,
            self.rope_theta,
            1.0,
        )?;

        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });
        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        Module::forward(&mut self.o_proj, &output)
    }
}

impl AttentionModule for CohereAttention {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        CohereAttention::forward_with_cache(self, x, mask, cache, positions)
    }
}

impl MlpModule for CohereMLP {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        CohereMLP::forward(self, x)
    }
}

/// Cohere decoder layer.
#[derive(Debug)]
pub struct CohereDecoderLayer {
    pub layer_idx: usize,

    pub self_attn: CohereAttention,
    pub mlp: CohereMLP,
    pub input_layernorm: nn::LayerNorm,
}
impl_module_params!(CohereDecoderLayer; self_attn, mlp, input_layernorm);

impl CohereDecoderLayer {
    pub fn new(config: &CohereConfig, layer_idx: usize) -> Result<Self, Exception> {
        let self_attn = CohereAttention::new(config, layer_idx)?;
        let mlp = CohereMLP::new(config.hidden_size, config.intermediate_size)?;

        // Cohere uses LayerNorm (not RMSNorm)
        let input_layernorm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            layer_idx,
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let _ = position_ids;
        self.forward_with_cache(x, mask, None, None)
    }

    /// Cohere parallel decoder block with optional KV cache.
    ///
    /// Layout: `x + attn(norm(x)) + ffn(norm(x))` — both branches consume
    /// the *same* normed input. There is no `post_attention_layernorm`.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self
            .self_attn
            .forward_with_cache(&normed, mask, cache, positions)?;
        let ffn_out = self.mlp.forward(&normed)?;
        Ok(x.add(&attn_out).add(&ffn_out))
    }
}

/// Cohere model.
#[derive(Debug)]
pub struct CohereModel {
    pub config: CohereConfig,

    pub embed_tokens: nn::Embedding,
    pub layers: Vec<CohereDecoderLayer>,
    pub norm: nn::LayerNorm,
    /// Recompute each layer's activations during the backward pass instead of
    /// holding them. Training only; see [`crate::checkpointing`].
    pub grad_checkpoint: bool,
}
impl_module_params!(CohereModel; embed_tokens, layers, norm);

impl CohereModel {
    pub fn new(config: CohereConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| CohereDecoderLayer::new(&config, i as usize))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
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
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, None, position_ids)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, cache, None)
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
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, None, positions)
    }

    /// The layer loop, with every per-call input the layers can take.
    ///
    /// `positions` carries one rotary position per token, `[seq_len]`, which
    /// is what a packed batch needs so the second sequence in a row restarts
    /// at 0 rather than continuing the first.
    fn forward_inner(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Hoisted: the loop below borrows `self.layers` mutably.
        let grad_checkpoint = self.grad_checkpoint;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let c = cache.as_deref_mut().map(|c| (c, idx));
            // A cache means generation, which has no backward pass for the
            // recompute to pay for.
            hidden_states = if grad_checkpoint && c.is_none() {
                checkpointed_layer(
                    layer,
                    &hidden_states,
                    mask,
                    positions,
                    |layer, h, mask, positions| layer.forward_with_cache(h, mask, None, positions),
                )?
            } else {
                layer.forward_with_cache(&hidden_states, mask, c, positions)?
            };
        }

        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Cohere for causal language modeling.
#[derive(Debug)]
pub struct CohereForCausalLM {
    pub config: CohereConfig,

    pub model: CohereModel,
    /// None when `tie_word_embeddings` is true — Cohere ships tied weights,
    /// so the LM head reuses `embed_tokens` transposed. A separate `lm_head`
    /// would never load (no `lm_head.weight` in the checkpoint) and emit
    /// random logits.
    pub lm_head: Option<nn::Linear>,
}
impl_module_params!(CohereForCausalLM; model, lm_head);

impl CohereForCausalLM {
    pub fn new(config: CohereConfig) -> Result<Self, Exception> {
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };

        let model = CohereModel::new(config.clone())?;

        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    /// Apply the LM head (tied or untied) and Cohere's output `logit_scale`.
    fn lm_head_forward(&mut self, h: &Array) -> Result<Array, Exception> {
        let logits = if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, h)?
        } else {
            self.model.embed_tokens.as_linear(h)
        };
        Ok(logits.multiply(&Array::from_f32(self.config.logit_scale)))
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids, mask, position_ids)?;
        self.lm_head_forward(&hidden_states)
    }

    /// Forward pass with optional KV cache for incremental decoding.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;
        self.lm_head_forward(&hidden_states)
    }

    /// Forward pass with one rotary position per token; see
    /// `forward_with_positions` on the inner model.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self
            .model
            .forward_with_positions(input_ids, mask, positions)?;
        self.lm_head_forward(&hidden_states)
    }

    /// Create a fresh KV cache sized for this model.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        use pmetal_mlx::kv_cache::KVCacheConfig;
        KVCache::new(KVCacheConfig::new(
            self.config.num_hidden_layers as usize,
            max_seq_len,
            self.config.num_key_value_heads as usize,
            self.config.head_dim as usize,
        ))
    }

    /// Fused batched-decode forward.
    ///
    /// Cohere's parallel decoder block — `x + attn(norm(x)) + ffn(norm(x))` —
    /// has no `post_attention_layernorm`, so it can't reuse
    /// `batched_prenorm_layer`. Routes through
    /// [`crate::common::batched_parallel_block`] instead.
    ///
    /// Sliding-window Cohere configs (`use_sliding_window = true` with
    /// non-global layers) take the serial fallback — gated by
    /// [`crate::dispatcher::DynamicModel::supports_fused_batched`].
    pub fn forward_batched_impl(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut pmetal_mlx::kv_cache::FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        use crate::common::{BatchedGqaAttnCfg, batched_parallel_block};

        let cfg = &self.config;
        let attn_cfg = BatchedGqaAttnCfg::new(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rope_theta,
            1.0,
        )
        .with_rope_traditional(true);

        let mut hidden = Module::forward(&mut self.model.embed_tokens, input_ids)?;
        for (layer_idx, layer) in self.model.layers.iter_mut().enumerate() {
            hidden = batched_parallel_block(
                &hidden,
                &mut layer.input_layernorm,
                &mut layer.self_attn.q_proj,
                &mut layer.self_attn.k_proj,
                &mut layer.self_attn.v_proj,
                &mut layer.self_attn.o_proj,
                None,
                None,
                &mut layer.mlp,
                &attn_cfg,
                cache,
                active_indices,
                layer_idx,
            )?;
        }
        let hidden = Module::forward(&mut self.model.norm, &hidden)?;
        self.lm_head_forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::ModuleParameters;
    use serial_test::serial;

    #[test]
    fn test_cohere_config() {
        let config = CohereConfig::default();

        // Every 4th layer uses global attention
        assert!(!config.uses_global_attention(0));
        assert!(!config.uses_global_attention(1));
        assert!(!config.uses_global_attention(2));
        assert!(config.uses_global_attention(3)); // 4th layer
        assert!(!config.uses_global_attention(4));
        assert!(!config.uses_global_attention(5));
        assert!(!config.uses_global_attention(6));
        assert!(config.uses_global_attention(7)); // 8th layer
    }

    #[test]
    #[serial]
    fn test_cohere_mlp() {
        let mlp = CohereMLP::new(64, 256).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[1, 10, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let mut mlp = mlp;
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[1, 10, 64]);
    }

    #[test]
    #[serial]
    fn test_cohere_model_instantiation() {
        let mut config = CohereConfig::default();
        config.hidden_size = 64;
        config.intermediate_size = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.vocab_size = 1000;

        let model = CohereForCausalLM::new(config).unwrap();

        let params = model.flatten_params();
        assert!(params.len() > 0);
    }
}
