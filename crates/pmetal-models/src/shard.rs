//! Model sharding for distributed pipeline inference.
//!
//! The [`ShardableModel`] trait decomposes a model's forward pass into
//! discrete stages (embed → layers → normalize → lm_head) so that
//! different pipeline stages can be assigned to different nodes.
//!
//! Which layers a node gets is `pmetal_distributed::layer_assignment`'s
//! decision (the memory-bounded dynamic-programming solver, or the
//! proportional split); a [`PipelineShard`] only runs the range it is given.

use pmetal_cache::kv_cache::KVCache;
use pmetal_runtime::compat::{Array, Exception};
use std::ops::Range;

/// A model whose forward pass can be decomposed into pipeline stages.
///
/// Each architecture stores `layers: Vec<DecoderLayer>` internally.
/// This trait exposes the per-layer forward pass and the surrounding
/// embedding/normalization/lm_head stages.
pub trait ShardableModel {
    /// Embed input token IDs into hidden states.
    ///
    /// `input_ids`: `[batch, seq_len]` → returns `[batch, seq_len, hidden_dim]`
    fn embed(&mut self, input_ids: &Array) -> Result<Array, Exception>;

    /// Run a single decoder layer.
    ///
    /// `x`: `[batch, seq_len, hidden_dim]` → returns same shape
    fn apply_layer(
        &mut self,
        layer_idx: usize,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array, Exception>;

    /// Apply the final RMSNorm/LayerNorm to the hidden states.
    fn normalize(&mut self, x: &Array) -> Result<Array, Exception>;

    /// Project hidden states to vocabulary logits.
    ///
    /// `x`: `[batch, seq_len, hidden_dim]` → `[batch, seq_len, vocab_size]`
    fn lm_head(&mut self, x: &Array) -> Result<Array, Exception>;

    /// Total number of decoder layers in this model.
    fn num_layers(&self) -> usize;

    /// Run a contiguous range of layers.
    ///
    /// Default implementation iterates `apply_layer` over the range.
    fn apply_layer_range(
        &mut self,
        range: Range<usize>,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array, Exception> {
        let mut hidden = x.clone();
        for idx in range {
            hidden = self.apply_layer(idx, &hidden, mask, cache)?;
        }
        Ok(hidden)
    }
}

/// A pipeline shard that owns a subset of a model's layers.
///
/// For distributed inference, each node creates a `PipelineShard` that
/// holds a contiguous range of layers. The first shard also owns the
/// embedding, and the last shard owns norm + lm_head.
pub struct PipelineShard<M: ShardableModel> {
    /// The underlying model (with all weights loaded, or only the assigned layers).
    pub model: M,
    /// Contiguous layer range assigned to this shard.
    pub assigned_layers: Range<usize>,
    /// Whether this shard owns the embedding layer (first shard).
    pub is_first: bool,
    /// Whether this shard owns the norm + lm_head (last shard).
    pub is_last: bool,
    /// KV cache for the assigned layers.
    pub cache: KVCache,
}

impl<M: ShardableModel> PipelineShard<M> {
    /// Create a new pipeline shard.
    pub fn new(
        model: M,
        assigned_layers: Range<usize>,
        is_first: bool,
        is_last: bool,
        cache: KVCache,
    ) -> Self {
        Self {
            model,
            assigned_layers,
            is_first,
            is_last,
            cache,
        }
    }

    /// Run the local portion of the forward pass.
    ///
    /// - If `is_first`: embeds input_ids first
    /// - Applies assigned layer range
    /// - If `is_last`: normalizes and projects to logits
    ///
    /// Input: either `input_ids` (if first) or hidden states from previous shard.
    /// Output: either hidden states (if not last) or logits (if last).
    pub fn forward_local(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut x = if self.is_first {
            self.model.embed(input)?
        } else {
            input.clone()
        };

        x = self.model.apply_layer_range(
            self.assigned_layers.clone(),
            &x,
            mask,
            &mut self.cache,
        )?;

        if self.is_last {
            x = self.model.normalize(&x)?;
            x = self.model.lm_head(&x)?;
        }

        Ok(x)
    }
}

/// Load only the safetensors keys needed for a given layer range + first/last shard.
///
/// Filters keys by `model.layers.{idx}.` prefix, plus embed_tokens if first,
/// norm + lm_head if last.
pub fn filter_weight_keys(
    all_keys: &[String],
    layer_range: &Range<usize>,
    is_first: bool,
    is_last: bool,
) -> Vec<String> {
    let mut selected = Vec::new();

    for key in all_keys {
        // Embedding weights
        if is_first
            && (key.contains("embed_tokens")
                || key.contains("wte")
                || key.contains("word_embeddings"))
        {
            selected.push(key.clone());
            continue;
        }

        // Norm + LM head weights
        if is_last
            && (key.contains("model.norm")
                || key.contains("ln_f")
                || key.contains("final_layernorm")
                || key.contains("lm_head"))
        {
            selected.push(key.clone());
            continue;
        }

        // Layer weights: match `model.layers.N.` or `transformer.h.N.`
        for idx in layer_range.clone() {
            let patterns = [
                format!("model.layers.{idx}."),
                format!("transformer.h.{idx}."),
                format!("model.decoder.layers.{idx}."),
            ];
            if patterns.iter().any(|p| key.starts_with(p)) {
                selected.push(key.clone());
                break;
            }
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_keys_first_shard() {
        let keys = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            "model.layers.2.self_attn.q_proj.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];

        let selected = filter_weight_keys(&keys, &(0..2), true, false);
        assert!(selected.contains(&"model.embed_tokens.weight".to_string()));
        assert!(selected.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(selected.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
        assert!(!selected.contains(&"model.layers.2.self_attn.q_proj.weight".to_string()));
        assert!(!selected.contains(&"model.norm.weight".to_string()));
    }

    #[test]
    fn filter_keys_last_shard() {
        let keys = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.30.self_attn.q_proj.weight".to_string(),
            "model.layers.31.self_attn.q_proj.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];

        let selected = filter_weight_keys(&keys, &(30..32), false, true);
        assert!(!selected.contains(&"model.embed_tokens.weight".to_string()));
        assert!(selected.contains(&"model.layers.30.self_attn.q_proj.weight".to_string()));
        assert!(selected.contains(&"model.layers.31.self_attn.q_proj.weight".to_string()));
        assert!(selected.contains(&"model.norm.weight".to_string()));
        assert!(selected.contains(&"lm_head.weight".to_string()));
    }
}
