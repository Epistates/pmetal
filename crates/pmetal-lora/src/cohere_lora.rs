//! LoRA-enabled Cohere model architecture.
//!
//! Supports Cohere, Cohere2, and Command-R variants with LoRA adapters on
//! attention and MLP projections for efficient fine-tuning.
//!
//! Key differences from Llama:
//! - `LayerNorm` (with bias) instead of `RmsNorm`
//! - Parallel residual: attn and FFN both run on the same normed input, outputs summed
//! - Per-layer sliding-window attention gated by `use_sliding_window` and global-attention logic
//! - Optional tied embeddings (lm_head == embed_tokens weight when `tie_word_embeddings`)

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param,
    nn, ops,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::rope::apply_rope;
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::cohere::CohereConfig;

use crate::lora::LoraProjection;
use crate::lora_helpers::{
    LoraDecoderStack, collect_lora_parameters, count_trainable_params, load_lora_weights_impl,
    save_lora_weights_impl, set_lora_parameters as helpers_set_lora_parameters,
    training_attention_mask,
};
use crate::{LoraError, LoraLinear};

// ─── Attention ───────────────────────────────────────────────────────────────

/// LoRA-enabled attention layer for Cohere.
#[derive(Debug)]
pub struct CohereLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// RoPE theta.
    pub rope_theta: f32,
    /// Whether this layer uses sliding window attention.
    pub use_sliding_window: bool,
    /// Sliding window size.
    pub sliding_window: i32,

    /// Query projection with LoRA.
    pub q_proj: LoraLinear,
    /// Key projection with LoRA.
    pub k_proj: LoraLinear,
    /// Value projection with LoRA.
    pub v_proj: LoraLinear,
    /// Output projection with LoRA.
    pub o_proj: LoraLinear,
}

impl CohereLoraAttention {
    pub fn new(
        config: &CohereConfig,
        lora_config: &LoraConfig,
        layer_idx: usize,
    ) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let q_rank = crate::effective_rank(lora_config, "q_proj") as i32;
        let k_rank = crate::effective_rank(lora_config, "k_proj") as i32;
        let v_rank = crate::effective_rank(lora_config, "v_proj") as i32;
        let o_rank = crate::effective_rank(lora_config, "o_proj") as i32;

        let q_proj = LoraLinear::new(
            hidden_size,
            n_heads * head_dim,
            q_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let k_proj = LoraLinear::new(
            hidden_size,
            n_kv_heads * head_dim,
            k_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let v_proj = LoraLinear::new(
            hidden_size,
            n_kv_heads * head_dim,
            v_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let o_proj = LoraLinear::new(
            n_heads * head_dim,
            hidden_size,
            o_rank,
            alpha,
            use_rslora,
            false,
        )?;

        let use_sliding_window =
            config.use_sliding_window && !config.uses_global_attention(layer_idx as i32);

        Ok(Self {
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

    /// Forward pass (no KV cache).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape and transpose to [B, H, S, D] BEFORE RoPE so that axis -2 is
        // the sequence axis. Rotating in [B, S, H, D] makes `apply_rope` treat
        // the head axis as positions — the axis bug `CohereAttention` in
        // pmetal-models documents having fixed, and the same one fixed in Phi.
        let q = q
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let k = k
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let v = v
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        // Cohere uses *traditional* (interleaved) RoPE.
        let q = apply_rope(&q, self.head_dim, true, self.rope_theta, 1.0, 0)?;
        let k = apply_rope(&k, self.head_dim, true, self.rope_theta, 1.0, 0)?;

        // GQA: expand KV heads if needed
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

        let k_t = k.transpose_axes(&[0, 1, 3, 2]);
        let mut scores = q.matmul(&k_t);
        scores = scores.multiply(&Array::from_f32(self.scale));

        if let Some(m) = mask {
            scores = scores.add(m);
        }

        let probs = ops::softmax_axis(&scores, -1);
        let out = probs.matmul(&v);

        let out = out
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        self.o_proj.forward(&out)
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Transpose to [B, H, S, D] BEFORE RoPE so axis -2 is the sequence
        // axis. With a cache offset the old ordering misrotated heads as
        // positions, which is what makes a cached decode disagree with the
        // prefill that seeded it.
        let q = q
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let k = k
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let v = v
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        let offset = cache
            .as_ref()
            .map_or(0, |(c, layer)| c.rope_offset_for(*layer));
        let q = apply_rope(&q, self.head_dim, true, self.rope_theta, 1.0, offset)?;
        let k = apply_rope(&k, self.head_dim, true, self.rope_theta, 1.0, offset)?;

        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache
                .update_and_fetch(layer_idx, &k, &v)
                .map_err(LoraError::Mlx)?
        } else {
            (k, v)
        };

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

        let k_t = k.transpose_axes(&[0, 1, 3, 2]);
        let mut scores = q.matmul(&k_t);
        scores = scores.multiply(&Array::from_f32(self.scale));
        if let Some(m) = mask {
            scores = scores.add(m);
        }
        let probs = ops::softmax_axis(&scores, -1);
        let out = probs.matmul(&v);
        let out = out
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        self.o_proj.forward(&out)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

// Manual ModuleParameters — LoraLinear does not impl ModuleParameters via macro.
impl ModuleParameters for CohereLoraAttention {
    fn num_parameters(&self) -> usize {
        self.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();
        let mut q_p = HashMap::new();
        q_p.insert(Rc::from("lora_a"), NestedValue::Value(&self.q_proj.lora_a));
        q_p.insert(Rc::from("lora_b"), NestedValue::Value(&self.q_proj.lora_b));
        params.insert(Rc::from("q_proj"), NestedValue::Map(q_p));
        let mut k_p = HashMap::new();
        k_p.insert(Rc::from("lora_a"), NestedValue::Value(&self.k_proj.lora_a));
        k_p.insert(Rc::from("lora_b"), NestedValue::Value(&self.k_proj.lora_b));
        params.insert(Rc::from("k_proj"), NestedValue::Map(k_p));
        let mut v_p = HashMap::new();
        v_p.insert(Rc::from("lora_a"), NestedValue::Value(&self.v_proj.lora_a));
        v_p.insert(Rc::from("lora_b"), NestedValue::Value(&self.v_proj.lora_b));
        params.insert(Rc::from("v_proj"), NestedValue::Map(v_p));
        let mut o_p = HashMap::new();
        o_p.insert(Rc::from("lora_a"), NestedValue::Value(&self.o_proj.lora_a));
        o_p.insert(Rc::from("lora_b"), NestedValue::Value(&self.o_proj.lora_b));
        params.insert(Rc::from("o_proj"), NestedValue::Map(o_p));
        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();
        let mut q_p = HashMap::new();
        q_p.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.q_proj.lora_a),
        );
        q_p.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.q_proj.lora_b),
        );
        params.insert(Rc::from("q_proj"), NestedValue::Map(q_p));
        let mut k_p = HashMap::new();
        k_p.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.k_proj.lora_a),
        );
        k_p.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.k_proj.lora_b),
        );
        params.insert(Rc::from("k_proj"), NestedValue::Map(k_p));
        let mut v_p = HashMap::new();
        v_p.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.v_proj.lora_a),
        );
        v_p.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.v_proj.lora_b),
        );
        params.insert(Rc::from("v_proj"), NestedValue::Map(v_p));
        let mut o_p = HashMap::new();
        o_p.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.o_proj.lora_a),
        );
        o_p.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.o_proj.lora_b),
        );
        params.insert(Rc::from("o_proj"), NestedValue::Map(o_p));
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

// ─── MLP ─────────────────────────────────────────────────────────────────────

/// LoRA-enabled MLP layer for Cohere (SwiGLU).
#[derive(Debug)]
pub struct CohereLoraMLP {
    pub gate_proj: LoraLinear,
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl CohereLoraMLP {
    pub fn new(config: &CohereConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let gate_rank = crate::effective_rank(lora_config, "gate_proj") as i32;
        let up_rank = crate::effective_rank(lora_config, "up_proj") as i32;
        let down_rank = crate::effective_rank(lora_config, "down_proj") as i32;

        let gate_proj = LoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            gate_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let up_proj = LoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            up_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let down_proj = LoraLinear::new(
            config.intermediate_size,
            config.hidden_size,
            down_rank,
            alpha,
            use_rslora,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

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
}

// Manual ModuleParameters for CohereLoraMLP.
impl ModuleParameters for CohereLoraMLP {
    fn num_parameters(&self) -> usize {
        self.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();
        let mut gate_p = HashMap::new();
        gate_p.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&self.gate_proj.lora_a),
        );
        gate_p.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&self.gate_proj.lora_b),
        );
        params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_p));
        let mut up_p = HashMap::new();
        up_p.insert(Rc::from("lora_a"), NestedValue::Value(&self.up_proj.lora_a));
        up_p.insert(Rc::from("lora_b"), NestedValue::Value(&self.up_proj.lora_b));
        params.insert(Rc::from("up_proj"), NestedValue::Map(up_p));
        let mut down_p = HashMap::new();
        down_p.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&self.down_proj.lora_a),
        );
        down_p.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&self.down_proj.lora_b),
        );
        params.insert(Rc::from("down_proj"), NestedValue::Map(down_p));
        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();
        let mut gate_p = HashMap::new();
        gate_p.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.gate_proj.lora_a),
        );
        gate_p.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.gate_proj.lora_b),
        );
        params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_p));
        let mut up_p = HashMap::new();
        up_p.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.up_proj.lora_a),
        );
        up_p.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.up_proj.lora_b),
        );
        params.insert(Rc::from("up_proj"), NestedValue::Map(up_p));
        let mut down_p = HashMap::new();
        down_p.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.down_proj.lora_a),
        );
        down_p.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.down_proj.lora_b),
        );
        params.insert(Rc::from("down_proj"), NestedValue::Map(down_p));
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

// ─── Decoder Layer ────────────────────────────────────────────────────────────

/// LoRA-enabled decoder layer for Cohere.
///
/// Uses Cohere's parallel residual pattern: both attn and FFN operate on the
/// same pre-norm hidden state and their outputs are jointly added to the
/// residual (unlike the sequential post-attn-then-FFN pattern in Llama).
#[derive(Debug)]
pub struct CohereLoraDecoderLayer {
    pub self_attn: CohereLoraAttention,
    pub mlp: CohereLoraMLP,
    /// Shared input LayerNorm (weight + bias, frozen).
    pub input_layernorm: nn::LayerNorm,
}

impl CohereLoraDecoderLayer {
    pub fn new(
        config: &CohereConfig,
        lora_config: &LoraConfig,
        layer_idx: usize,
    ) -> Result<Self, LoraError> {
        let self_attn = CohereLoraAttention::new(config, lora_config, layer_idx)?;
        let mlp = CohereLoraMLP::new(config, lora_config)?;
        let input_layernorm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        let ffn_out = self.mlp.forward(&normed)?;
        // Parallel residual: x + attn + ffn
        Ok(x.add(&attn_out).add(&ffn_out))
    }

    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let ffn_out = self.mlp.forward(&normed)?;
        Ok(x.add(&attn_out).add(&ffn_out))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.mlp.num_trainable_params()
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// LoRA-enabled Cohere model (without LM head).
#[derive(Debug)]
pub struct CohereLoraModel {
    pub config: CohereConfig,
    pub lora_config: LoraConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<CohereLoraDecoderLayer>,
    /// Final LayerNorm (weight + bias, frozen).
    pub norm: nn::LayerNorm,
}

impl CohereLoraModel {
    pub fn new(config: CohereConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| CohereLoraDecoderLayer::new(&config, &lora_config, i))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            config,
            lora_config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass.
    ///
    /// Builds the causal mask when the caller supplies none.
    /// `CohereLoraAttention::forward` adds whatever mask it is handed and
    /// nothing more, so without this the training pass attends bidirectionally
    /// while inference does not. Cohere is uniformly causal, with no sliding
    /// window to carry.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let mut h = Module::forward(&mut self.embed_tokens, input_ids)?;

        let owned_mask = match mask {
            Some(_) => None,
            None => Some(training_attention_mask(input_ids.dim(1), None)?),
        };
        let mask = mask.or(owned_mask.as_ref());

        for layer in &mut self.layers {
            h = layer.forward(&h, mask)?;
        }
        Ok(Module::forward(&mut self.norm, &h)?)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let mut h = Module::forward(&mut self.embed_tokens, input_ids)?;
        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    h = layer.forward_with_cache(&h, mask, Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    h = layer.forward(&h, mask)?;
                }
            }
        }
        Ok(Module::forward(&mut self.norm, &h)?)
    }

    pub fn num_trainable_params(&self) -> usize {
        count_trainable_params(self)
    }
}

impl LoraDecoderStack for CohereLoraModel {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn attn_projections(&self, layer: usize) -> Vec<&dyn LoraProjection> {
        let l = &self.layers[layer];
        vec![
            &l.self_attn.q_proj,
            &l.self_attn.k_proj,
            &l.self_attn.v_proj,
            &l.self_attn.o_proj,
        ]
    }

    fn attn_projections_mut(&mut self, layer: usize) -> Vec<&mut dyn LoraProjection> {
        let l = &mut self.layers[layer];
        vec![
            &mut l.self_attn.q_proj,
            &mut l.self_attn.k_proj,
            &mut l.self_attn.v_proj,
            &mut l.self_attn.o_proj,
        ]
    }

    fn mlp_projections(&self, layer: usize) -> Vec<&dyn LoraProjection> {
        let l = &self.layers[layer];
        vec![&l.mlp.gate_proj, &l.mlp.up_proj, &l.mlp.down_proj]
    }

    fn mlp_projections_mut(&mut self, layer: usize) -> Vec<&mut dyn LoraProjection> {
        let l = &mut self.layers[layer];
        vec![
            &mut l.mlp.gate_proj,
            &mut l.mlp.up_proj,
            &mut l.mlp.down_proj,
        ]
    }
}

// ─── ForCausalLM ─────────────────────────────────────────────────────────────

/// LoRA-enabled Cohere model with LM head.
#[derive(Debug)]
pub struct CohereLoraForCausalLM {
    pub model: CohereLoraModel,
    /// Separate LM head when `tie_word_embeddings == false`.
    pub lm_head: Option<nn::Linear>,
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl CohereLoraForCausalLM {
    pub fn new(config: CohereConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let tie_word_embeddings = config.tie_word_embeddings;
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;

        let model = CohereLoraModel::new(config, lora_config)?;

        let lm_head = if !tie_word_embeddings {
            Some(
                nn::LinearBuilder::new(hidden_size, vocab_size)
                    .bias(false)
                    .build()?,
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

    fn compute_logits(&mut self, hidden: &Array) -> Result<Array, LoraError> {
        let logits = if let Some(ref mut head) = self.lm_head {
            Module::forward(head, hidden)?
        } else {
            self.model.embed_tokens.as_linear(hidden)
        };
        // Cohere scales the LM-head logits by `logit_scale` (0.0625 for the
        // Command-R family). Applies to both training and inference.
        Ok(logits.multiply(&Array::from_f32(self.model.config.logit_scale)))
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let hidden = self.model.forward(input_ids, mask)?;
        self.compute_logits(&hidden)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden = self.model.forward_with_cache(input_ids, mask, cache)?;
        self.compute_logits(&hidden)
    }

    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.model.forward(input_ids, mask)
    }

    pub fn forward_hidden_states_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        self.forward_hidden_states(input_ids, mask)
    }

    pub fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        self.forward(input_ids, mask)
    }

    pub fn get_lm_head_weight(&self) -> Option<Array> {
        if let Some(ref head) = self.lm_head {
            Some(head.weight.value.clone())
        } else {
            Some(self.model.embed_tokens.weight.value.clone())
        }
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let cfg = KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_key_value_heads as usize,
            self.model.config.head_dim as usize,
        );
        KVCache::new(cfg)
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

    pub fn num_trainable_params(&self) -> usize {
        count_trainable_params(&self.model)
    }

    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        collect_lora_parameters(&self.model)
    }

    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        helpers_set_lora_parameters(&mut self.model, params);
    }

    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        save_lora_weights_impl(&self.model, path)
    }

    pub fn load_lora_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        load_lora_weights_impl(&mut self.model, path)
    }

    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.merge()?;
            layer.self_attn.k_proj.merge()?;
            layer.self_attn.v_proj.merge()?;
            layer.self_attn.o_proj.merge()?;
            layer.mlp.gate_proj.merge()?;
            layer.mlp.up_proj.merge()?;
            layer.mlp.down_proj.merge()?;
        }
        Ok(())
    }

    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".into(),
        ))
    }

    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.model.embed_tokens.weight.value.eval();

        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.weight.eval();
            layer.self_attn.k_proj.weight.eval();
            layer.self_attn.v_proj.weight.eval();
            layer.self_attn.o_proj.weight.eval();
            layer.self_attn.q_proj.lora_a.eval();
            layer.self_attn.q_proj.lora_b.eval();
            layer.self_attn.k_proj.lora_a.eval();
            layer.self_attn.k_proj.lora_b.eval();
            layer.self_attn.v_proj.lora_a.eval();
            layer.self_attn.v_proj.lora_b.eval();
            layer.self_attn.o_proj.lora_a.eval();
            layer.self_attn.o_proj.lora_b.eval();
            layer.mlp.gate_proj.weight.eval();
            layer.mlp.up_proj.weight.eval();
            layer.mlp.down_proj.weight.eval();
            layer.mlp.gate_proj.lora_a.eval();
            layer.mlp.gate_proj.lora_b.eval();
            layer.mlp.up_proj.lora_a.eval();
            layer.mlp.up_proj.lora_b.eval();
            layer.mlp.down_proj.lora_a.eval();
            layer.mlp.down_proj.lora_b.eval();
            if let Some(ref w) = layer.input_layernorm.weight.value {
                w.eval();
            }
            if let Some(ref b) = layer.input_layernorm.bias.value {
                b.eval();
            }
        }

        if let Some(ref w) = self.model.norm.weight.value {
            w.eval();
        }
        if let Some(ref b) = self.model.norm.bias.value {
            b.eval();
        }

        Ok(())
    }

    /// Load base weights from a directory containing HuggingFace safetensors files.
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

    pub fn load_base_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        // Embeddings
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            macro_rules! load_linear {
                ($field:expr, $key:expr) => {
                    if let Some(w) = weights.get($key) {
                        $field = w.clone();
                    }
                };
            }
            macro_rules! load_param {
                ($field:expr, $key:expr) => {
                    if let Some(w) = weights.get($key) {
                        $field = Param::new(Some(w.clone()));
                    }
                };
            }

            load_linear!(
                layer.self_attn.q_proj.weight,
                &format!("{}.self_attn.q_proj.weight", prefix)
            );
            load_linear!(
                layer.self_attn.k_proj.weight,
                &format!("{}.self_attn.k_proj.weight", prefix)
            );
            load_linear!(
                layer.self_attn.v_proj.weight,
                &format!("{}.self_attn.v_proj.weight", prefix)
            );
            load_linear!(
                layer.self_attn.o_proj.weight,
                &format!("{}.self_attn.o_proj.weight", prefix)
            );
            load_linear!(
                layer.mlp.gate_proj.weight,
                &format!("{}.mlp.gate_proj.weight", prefix)
            );
            load_linear!(
                layer.mlp.up_proj.weight,
                &format!("{}.mlp.up_proj.weight", prefix)
            );
            load_linear!(
                layer.mlp.down_proj.weight,
                &format!("{}.mlp.down_proj.weight", prefix)
            );

            // LayerNorm has both weight and bias
            load_param!(
                layer.input_layernorm.weight,
                &format!("{}.input_layernorm.weight", prefix)
            );
            load_param!(
                layer.input_layernorm.bias,
                &format!("{}.input_layernorm.bias", prefix)
            );
        }

        // Final norm
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(Some(w.clone()));
        }
        if let Some(b) = weights.get("model.norm.bias") {
            self.model.norm.bias = Param::new(Some(b.clone()));
        }

        // LM head (optional, only present when tie_word_embeddings == false)
        if let Some(ref mut head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }
}

// ─── ModuleParameters for ForCausalLM ────────────────────────────────────────

impl ModuleParameters for CohereLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        count_trainable_params(&self.model)
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            let mut attn_params = HashMap::new();
            let mut q_p = HashMap::new();
            q_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.q_proj.lora_a),
            );
            q_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.q_proj.lora_b),
            );
            attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_p));

            let mut k_p = HashMap::new();
            k_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.k_proj.lora_a),
            );
            k_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.k_proj.lora_b),
            );
            attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_p));

            let mut v_p = HashMap::new();
            v_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.v_proj.lora_a),
            );
            v_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.v_proj.lora_b),
            );
            attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_p));

            let mut o_p = HashMap::new();
            o_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.o_proj.lora_a),
            );
            o_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.o_proj.lora_b),
            );
            attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_p));
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params = HashMap::new();
            let mut gate_p = HashMap::new();
            gate_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.mlp.gate_proj.lora_a),
            );
            gate_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.mlp.gate_proj.lora_b),
            );
            mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_p));

            let mut up_p = HashMap::new();
            up_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.mlp.up_proj.lora_a),
            );
            up_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.mlp.up_proj.lora_b),
            );
            mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_p));

            let mut down_p = HashMap::new();
            down_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.mlp.down_proj.lora_a),
            );
            down_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.mlp.down_proj.lora_b),
            );
            mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_p));
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

            let mut attn_params = HashMap::new();
            let mut q_p = HashMap::new();
            q_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.q_proj.lora_a),
            );
            q_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.q_proj.lora_b),
            );
            attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_p));

            let mut k_p = HashMap::new();
            k_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.k_proj.lora_a),
            );
            k_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.k_proj.lora_b),
            );
            attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_p));

            let mut v_p = HashMap::new();
            v_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.v_proj.lora_a),
            );
            v_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.v_proj.lora_b),
            );
            attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_p));

            let mut o_p = HashMap::new();
            o_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.o_proj.lora_a),
            );
            o_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.o_proj.lora_b),
            );
            attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_p));
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params = HashMap::new();
            let mut gate_p = HashMap::new();
            gate_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.mlp.gate_proj.lora_a),
            );
            gate_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.mlp.gate_proj.lora_b),
            );
            mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_p));

            let mut up_p = HashMap::new();
            up_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.mlp.up_proj.lora_a),
            );
            up_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.mlp.up_proj.lora_b),
            );
            mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_p));

            let mut down_p = HashMap::new();
            down_p.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.mlp.down_proj.lora_a),
            );
            down_p.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.mlp.down_proj.lora_b),
            );
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

// Implement TrainableModel via shared macro.
crate::impl_trainable_model!(CohereLoraForCausalLM);

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, Exception> {
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::ModuleParameters;

    fn small_config() -> CohereConfig {
        CohereConfig {
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            max_position_embeddings: 256,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            logit_scale: 0.0625,
            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window: 4096,
            global_attention_layers: None,
        }
    }

    fn small_lora_config() -> LoraConfig {
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
                "gate_proj".to_string(),
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
    fn test_cohere_lora_construction() {
        // Construction with a tiny 2-layer config should not panic.
        let config = small_config();
        let lora_config = small_lora_config();
        let model = CohereLoraForCausalLM::new(config, lora_config).unwrap();
        assert_eq!(model.model.layers.len(), 2);
    }

    #[test]
    fn test_cohere_lora_trainable_params() {
        let config = small_config();
        let lora_config = small_lora_config();
        let model = CohereLoraForCausalLM::new(config, lora_config).unwrap();

        // trainable params > 0 (LoRA A+B matrices)
        assert!(model.num_trainable_params() > 0);

        // ModuleParameters only exposes LoRA params — all base weights are frozen
        let all_params = model.parameters();
        // All flattened keys should be lora_a or lora_b
        let flat = pmetal_bridge::compat::ModuleParametersExt::flatten_params(&model);
        for key in flat.keys() {
            let s = key.as_ref();
            assert!(
                s.ends_with("lora_a") || s.ends_with("lora_b"),
                "unexpected non-LoRA key in trainable params: {}",
                s
            );
        }
        // and the map must not be empty
        assert!(!all_params.is_empty());
    }

    #[test]
    fn test_cohere_lora_forward_shape() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = CohereLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids =
            pmetal_bridge::compat::Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        // [1, 4, vocab_size]
        assert_eq!(logits.shape(), &[1, 4, 512]);
    }

    #[test]
    fn test_cohere_lora_save_load_roundtrip() {
        use std::rc::Rc;
        let config = small_config();
        let lora_config = small_lora_config();
        let model = CohereLoraForCausalLM::new(config.clone(), lora_config.clone()).unwrap();

        // Snapshot original LoRA params
        let original = model.lora_parameters();

        // Save to tempdir
        let tmpdir = std::env::temp_dir().join(format!(
            "cohere_lora_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let weights_path = tmpdir.join("lora_weights.safetensors");
        model.save_lora_weights(&weights_path).unwrap();

        // Load into a fresh model
        let mut model2 = CohereLoraForCausalLM::new(config.clone(), lora_config.clone()).unwrap();
        model2.load_lora_weights(&weights_path).unwrap();
        let loaded = model2.lora_parameters();

        // Each parameter tensor must be identical
        assert_eq!(original.len(), loaded.len());
        for (key, arr_orig) in &original {
            let arr_loaded = loaded.get(key).expect("key missing after load");
            // Materialize and compare
            let orig_data = arr_orig.clone();
            let load_data = arr_loaded.clone();
            orig_data.eval().unwrap();
            load_data.eval().unwrap();
            let diff = orig_data.subtract(&load_data).unwrap();
            let max_diff = diff.abs().unwrap().max(None).unwrap();
            max_diff.eval().unwrap();
            assert!(
                max_diff.item::<f32>() < 1e-6,
                "tensor mismatch for key: {}",
                key.as_ref()
            );
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmpdir);
    }
}
