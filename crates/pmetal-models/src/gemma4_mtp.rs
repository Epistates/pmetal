//! Gemma 4 multi-token prediction decoding.
//!
//! The decode loop is [`crate::mtp`]'s. This module supplies the assistant as
//! an [`MtpDrafter`] and the target as an [`MtpTarget`]. The assistant has no
//! cache of its own: it attends to the target's KV cache, and every step of a
//! round runs at the pending token's position.

use pmetal_mlx::{kv_cache::KVCache, speculative::SpecCapture};
use pmetal_runtime::compat::{Array, Exception, ops::concatenate_axis};
use pmetal_runtime::decode::prompt_tokens_to_input;

use crate::{
    architectures::{
        Gemma4AssistantForCausalLM, Gemma4AssistantSharedKvStates, Gemma4Config, Gemma4ForCausalLM,
    },
    generation::{GenerationConfig, GenerationOutput},
    mtp::{MtpDrafter, MtpTarget, generate_mtp_streaming},
};

#[derive(Debug, Clone)]
pub struct Gemma4MtpConfig {
    pub num_assistant_tokens: usize,
}

impl Default for Gemma4MtpConfig {
    fn default() -> Self {
        Self {
            num_assistant_tokens: 6,
        }
    }
}

/// The target layers whose KV caches a Gemma 4 assistant attends to: the
/// last layer of each attention type before the target's KV-shared layers.
/// Drafting and training read them the same way.
#[derive(Debug, Clone)]
pub struct Gemma4MtpKvSources {
    full_attention: Option<usize>,
    sliding_attention: Option<usize>,
}

impl Gemma4MtpKvSources {
    /// The sources in `target` for `assistant`'s layers, or an error naming
    /// an assistant layer whose attention type the target has no source for.
    pub fn new(target: &Gemma4Config, assistant: &Gemma4Config) -> Result<Self, Exception> {
        let total = target.num_hidden_layers.max(0) as usize;
        let first_shared = target.first_kv_shared_layer_idx().min(total);
        let mut full_attention = None;
        let mut sliding_attention = None;

        for layer_idx in 0..first_shared {
            if target.is_full_attention(layer_idx) {
                full_attention = Some(layer_idx);
            } else {
                sliding_attention = Some(layer_idx);
            }
        }

        for layer_idx in 0..assistant.num_hidden_layers.max(0) as usize {
            let has_source = if assistant.is_full_attention(layer_idx) {
                full_attention.is_some()
            } else {
                sliding_attention.is_some()
            };
            if !has_source {
                let kind = if assistant.is_full_attention(layer_idx) {
                    "full_attention"
                } else {
                    "sliding_attention"
                };
                return Err(Exception::custom(format!(
                    "Gemma 4 MTP target has no non-shared {kind} KV source for assistant layer {layer_idx}"
                )));
            }
        }

        Ok(Self {
            full_attention,
            sliding_attention,
        })
    }

    /// The source layers' entries in the target's `cache`.
    pub fn shared_states(
        &self,
        cache: &KVCache,
    ) -> Result<Gemma4AssistantSharedKvStates, Exception> {
        let full_attention = self
            .full_attention
            .map(|layer| {
                cache.get(layer).ok_or_else(|| {
                    Exception::custom(format!(
                        "Gemma 4 MTP target cache missing full-attention source layer {layer}"
                    ))
                })
            })
            .transpose()?;
        let sliding_attention = self
            .sliding_attention
            .map(|layer| {
                cache.get(layer).ok_or_else(|| {
                    Exception::custom(format!(
                        "Gemma 4 MTP target cache missing sliding-attention source layer {layer}"
                    ))
                })
            })
            .transpose()?;
        Ok(Gemma4AssistantSharedKvStates {
            full_attention,
            sliding_attention,
        })
    }
}

pub fn validate_gemma4_mtp_pair(
    target: &Gemma4ForCausalLM,
    assistant: &Gemma4AssistantForCausalLM,
) -> Result<(), Exception> {
    if assistant.config.backbone_hidden_size != target.config.hidden_size {
        return Err(Exception::custom(format!(
            "Gemma 4 MTP assistant backbone_hidden_size ({}) must match target hidden_size ({})",
            assistant.config.backbone_hidden_size, target.config.hidden_size
        )));
    }
    if assistant.config.text_config.vocab_size != target.config.vocab_size {
        return Err(Exception::custom(format!(
            "Gemma 4 MTP assistant vocab_size ({}) must match target vocab_size ({})",
            assistant.config.text_config.vocab_size, target.config.vocab_size
        )));
    }
    let _ = Gemma4MtpKvSources::new(&target.config, &assistant.config.text_config)?;
    Ok(())
}

/// MTP decode of a Gemma 4 target with its assistant.
pub fn generate_gemma4_mtp_streaming<F>(
    target: &mut Gemma4ForCausalLM,
    assistant: &mut Gemma4AssistantForCausalLM,
    input_ids: &[u32],
    gen_config: GenerationConfig,
    cache: &mut KVCache,
    mtp_config: Gemma4MtpConfig,
    on_token: F,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(u32) -> bool,
{
    let mut drafter = Gemma4MtpDrafter::new(target, assistant)?;
    let mut target = Gemma4MtpVerifier::new(target, cache);
    generate_mtp_streaming(
        &mut target,
        &mut drafter,
        input_ids,
        &gen_config,
        mtp_config.num_assistant_tokens,
        on_token,
    )
}

/// A Gemma 4 model and its KV cache as an [`MtpTarget`].
pub struct Gemma4MtpVerifier<'a> {
    model: &'a mut Gemma4ForCausalLM,
    cache: &'a mut KVCache,
    verified: usize,
}

impl<'a> Gemma4MtpVerifier<'a> {
    pub fn new(model: &'a mut Gemma4ForCausalLM, cache: &'a mut KVCache) -> Self {
        Self {
            model,
            cache,
            verified: 0,
        }
    }
}

impl MtpTarget for Gemma4MtpVerifier<'_> {
    fn reset(&mut self) -> Result<(), Exception> {
        self.cache.reset();
        Ok(())
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<(Array, Array), Exception> {
        let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
        self.model.forward_hidden_with_capture(
            &prompt_tokens_to_input(tokens),
            None,
            Some(self.cache),
            &mut capture,
        )
    }

    fn verify(&mut self, tokens: &[u32]) -> Result<(Array, Array), Exception> {
        self.verified = tokens.len();
        self.forward(tokens)
    }

    fn keep(&mut self, kept: usize) -> Result<(), Exception> {
        let rejected = self.verified.saturating_sub(kept);
        if rejected > 0 {
            self.cache.rollback(rejected);
        }
        Ok(())
    }
}

/// A Gemma 4 assistant as an [`MtpDrafter`] of a [`Gemma4MtpVerifier`]. A
/// step reads the target's scaled embedding of the newest token beside the
/// hidden state that predicted it, and the target's KV cache, which holds
/// every position before the pending token's.
pub struct Gemma4MtpDrafter<'a> {
    assistant: &'a mut Gemma4AssistantForCausalLM,
    kv_sources: Gemma4MtpKvSources,
}

impl<'a> Gemma4MtpDrafter<'a> {
    pub fn new(
        target: &Gemma4ForCausalLM,
        assistant: &'a mut Gemma4AssistantForCausalLM,
    ) -> Result<Self, Exception> {
        validate_gemma4_mtp_pair(target, assistant)?;
        let kv_sources = Gemma4MtpKvSources::new(&target.config, &assistant.config.text_config)?;
        Ok(Self {
            assistant,
            kv_sources,
        })
    }
}

impl MtpDrafter<Gemma4MtpVerifier<'_>> for Gemma4MtpDrafter<'_> {
    fn prefill(
        &mut self,
        _: &mut Gemma4MtpVerifier<'_>,
        _: &[u32],
        _: &Array,
    ) -> Result<(), Exception> {
        Ok(())
    }

    fn step(
        &mut self,
        target: &mut Gemma4MtpVerifier<'_>,
        token: u32,
        hidden: &Array,
        _step: usize,
        position: usize,
    ) -> Result<(Array, Array), Exception> {
        let shared_states = self.kv_sources.shared_states(target.cache)?;
        let embedding = embed_target_token(target.model, token);
        let inputs_embeds = concatenate_axis(&[&embedding, hidden], -1);
        // `[1, 1, H]` for the next step, `[1, V]` logits.
        self.assistant
            .forward_logits(&inputs_embeds, &shared_states, position as i32)
    }

    fn commit(
        &mut self,
        _: &mut Gemma4MtpVerifier<'_>,
        _: u32,
        _: &Array,
        _: &[u32],
        _: usize,
        _: &Array,
    ) -> Result<(), Exception> {
        Ok(())
    }
}

fn embed_target_token(target: &Gemma4ForCausalLM, token: u32) -> Array {
    let input = prompt_tokens_to_input(&[token]);
    target
        .model
        .embed_tokens
        .forward(&input)
        .multiply(&Array::from_f32(target.model.embed_scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config(layer_types: Vec<&str>, shared_layers: i32) -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4_text".to_string(),
            vocab_size: 128,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: layer_types.len() as i32,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            global_head_dim: Some(8),
            num_global_key_value_heads: Some(1),
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            attention_k_eq_v: false,
            tie_word_embeddings: true,
            sliding_window: 32,
            final_logit_softcapping: None,
            layer_types: layer_types.into_iter().map(str::to_string).collect(),
            rope_parameters: None,
            _raw_rope_parameters: None,
            hidden_size_per_layer_input: None,
            vocab_size_per_layer_input: None,
            hidden_activation: None,
            num_kv_shared_layers: Some(shared_layers),
            use_double_wide_mlp: Some(false),
            enable_moe_block: Some(false),
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
        }
    }

    #[test]
    fn mtp_kv_sources_use_last_non_shared_layer_by_type() {
        let target = base_config(
            vec![
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            2,
        );
        let assistant = base_config(vec!["sliding_attention", "full_attention"], 2);
        let sources = Gemma4MtpKvSources::new(&target, &assistant).unwrap();
        assert_eq!(sources.sliding_attention, Some(2));
        assert_eq!(sources.full_attention, Some(3));
    }
}
