//! Gemma 4 multi-token prediction decoding.

use std::collections::HashMap;

use pmetal_bridge::compat::{
    Array, Exception, Module,
    ops::{concatenate_axis, select_axis},
};
use pmetal_mlx::{kv_cache::KVCache, speculative::SpecCapture};

use crate::{
    architectures::{
        Gemma4AssistantForCausalLM, Gemma4AssistantSharedKvStates, Gemma4Config, Gemma4ForCausalLM,
    },
    generation::{
        GenerationConfig, GenerationOutput, SpeculativeDecodeMetrics, sample_from_log_probs,
        sampling_log_probs_with_counts, token_probability_from_log_probs,
    },
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

#[derive(Debug, Clone)]
struct Gemma4MtpKvSources {
    full_attention: Option<usize>,
    sliding_attention: Option<usize>,
}

impl Gemma4MtpKvSources {
    fn new(target: &Gemma4Config, assistant: &Gemma4Config) -> Result<Self, Exception> {
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

    fn shared_states(&self, cache: &KVCache) -> Result<Gemma4AssistantSharedKvStates, Exception> {
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

pub fn generate_gemma4_mtp_streaming<F>(
    target: &mut Gemma4ForCausalLM,
    assistant: &mut Gemma4AssistantForCausalLM,
    input_ids: &[u32],
    gen_config: GenerationConfig,
    cache: &mut KVCache,
    mtp_config: Gemma4MtpConfig,
    mut on_token: F,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(u32) -> bool,
{
    if input_ids.is_empty() {
        return Err(Exception::custom("Gemma 4 MTP requires a non-empty prompt"));
    }
    if gen_config.max_new_tokens == 0 {
        return Ok(GenerationOutput {
            token_ids: input_ids.to_vec(),
            num_generated: 0,
            stopped_by_token: false,
            stopped_by_length: true,
            decode_metrics: None,
            speculative_metrics: None,
        });
    }
    if gen_config.do_sample {
        return generate_gemma4_mtp_sampling_streaming(
            target, assistant, input_ids, gen_config, cache, mtp_config, on_token,
        );
    }

    validate_gemma4_mtp_pair(target, assistant)?;
    let kv_sources = Gemma4MtpKvSources::new(&target.config, &assistant.config.text_config)?;
    cache.reset();

    let prompt = token_array(input_ids);
    let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
    let (hidden, logits) =
        target.forward_hidden_with_capture(&prompt, None, Some(cache), &mut capture)?;
    let mut last_hidden = select_last_sequence(&hidden);
    let mut prev_logits = select_last_logits(&logits);

    let mut token_ids = input_ids.to_vec();
    let mut num_generated = 0usize;
    let mut stopped_by_token = false;
    let draft_budget = mtp_config.num_assistant_tokens.max(1);
    let mut speculative_metrics = SpeculativeDecodeMetrics::default();

    while num_generated < gen_config.max_new_tokens {
        let remaining = gen_config.max_new_tokens - num_generated;
        let max_draft = draft_budget.min(remaining);
        // Gemma 4 assistants use a static target KV cache; HF documents the
        // assistant position IDs as constant throughout a drafting round.
        let position_id = token_ids.len().saturating_sub(1) as i32;
        let shared_states = kv_sources.shared_states(cache)?;
        let mut draft_tokens = Vec::with_capacity(max_draft);
        let mut draft_hidden = last_hidden.clone();
        let mut draft_input_token = *token_ids
            .last()
            .ok_or_else(|| Exception::custom("Gemma 4 MTP internal empty token state"))?;

        for _ in 0..max_draft {
            let token_embedding = embed_target_token(target, draft_input_token);
            let inputs_embeds = concatenate_axis(&[&token_embedding, &draft_hidden], -1);
            let (draft_token, projected_hidden) =
                assistant.draft_greedy_token(&inputs_embeds, &shared_states, position_id)?;
            draft_tokens.push(draft_token);
            draft_hidden = projected_hidden;
            draft_input_token = draft_token;
            if gen_config.stop_tokens.contains(&draft_token) {
                break;
            }
        }

        if draft_tokens.is_empty() {
            let token = greedy_token(&prev_logits);
            emit_token(
                token,
                &mut token_ids,
                &mut num_generated,
                &mut stopped_by_token,
                &gen_config,
                &mut on_token,
            )?;
            if stopped_by_token || num_generated >= gen_config.max_new_tokens {
                break;
            }
            let (hidden, logits) = target_step(target, token, cache)?;
            last_hidden = hidden;
            prev_logits = logits;
            continue;
        }

        let verify_input = token_array(&draft_tokens);
        let verify_logits = target.forward_with_cache(&verify_input, None, Some(cache))?;
        let mut matched = 0usize;
        while matched < draft_tokens.len() {
            let row = if matched == 0 {
                prev_logits.clone()
            } else {
                select_axis(&verify_logits, (matched - 1) as i32, 1)
            };
            if greedy_token(&row) != draft_tokens[matched] {
                break;
            }
            matched += 1;
        }
        speculative_metrics.record_verify_step(draft_tokens.len(), matched);

        let all_matched = matched == draft_tokens.len();
        let mut planned = Vec::with_capacity(draft_tokens.len() + 1);
        planned.extend_from_slice(&draft_tokens[..matched]);
        let append_after_emit = if all_matched {
            if num_generated + planned.len() < gen_config.max_new_tokens {
                let bonus = greedy_token(&select_axis(
                    &verify_logits,
                    (draft_tokens.len() - 1) as i32,
                    1,
                ));
                planned.push(bonus);
                Some(bonus)
            } else {
                None
            }
        } else {
            let correction_logits = if matched == 0 {
                prev_logits.clone()
            } else {
                select_axis(&verify_logits, (matched - 1) as i32, 1)
            };
            let correction = greedy_token(&correction_logits);
            planned.push(correction);
            Some(correction)
        };

        let stop_pos = planned
            .iter()
            .position(|token| gen_config.stop_tokens.contains(token));
        let planned_len = stop_pos.map(|idx| idx + 1).unwrap_or(planned.len());

        let mut continue_stream = true;
        let mut emitted_planned_count = 0usize;
        for &token in &planned[..planned_len] {
            continue_stream = emit_token(
                token,
                &mut token_ids,
                &mut num_generated,
                &mut stopped_by_token,
                &gen_config,
                &mut on_token,
            )?;
            emitted_planned_count += 1;
            if !continue_stream || stopped_by_token || num_generated >= gen_config.max_new_tokens {
                break;
            }
        }
        let emitted_draft_count = emitted_planned_count.min(matched);
        speculative_metrics.emitted_draft_tokens += emitted_draft_count;
        if emitted_planned_count > emitted_draft_count {
            if all_matched {
                speculative_metrics.bonus_tokens += 1;
            } else {
                speculative_metrics.correction_tokens += 1;
            }
        }
        let rollback = draft_tokens.len().saturating_sub(emitted_draft_count);
        if rollback > 0 {
            cache.rollback(rollback);
        }
        if !continue_stream
            || stopped_by_token
            || num_generated >= gen_config.max_new_tokens
            || stop_pos.is_some()
        {
            break;
        }

        if let Some(token) = append_after_emit {
            let appended_is_draft = all_matched && planned.len() <= draft_tokens.len();
            if !appended_is_draft {
                let (hidden, logits) = target_step(target, token, cache)?;
                last_hidden = hidden;
                prev_logits = logits;
            }
        }
    }

    Ok(GenerationOutput {
        token_ids,
        num_generated,
        stopped_by_token,
        stopped_by_length: num_generated >= gen_config.max_new_tokens && !stopped_by_token,
        decode_metrics: None,
        speculative_metrics: Some(speculative_metrics),
    })
}

fn generate_gemma4_mtp_sampling_streaming<F>(
    target: &mut Gemma4ForCausalLM,
    assistant: &mut Gemma4AssistantForCausalLM,
    input_ids: &[u32],
    gen_config: GenerationConfig,
    cache: &mut KVCache,
    mtp_config: Gemma4MtpConfig,
    mut on_token: F,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(u32) -> bool,
{
    if let Some(seed) = gen_config.seed {
        pmetal_bridge::inline_array::random_seed(seed);
        seed_acceptance_rng(seed);
    }

    validate_gemma4_mtp_pair(target, assistant)?;
    let kv_sources = Gemma4MtpKvSources::new(&target.config, &assistant.config.text_config)?;
    cache.reset();

    let prompt = token_array(input_ids);
    let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
    let (hidden, logits) =
        target.forward_hidden_with_capture(&prompt, None, Some(cache), &mut capture)?;
    let mut last_hidden = select_last_sequence(&hidden);
    let mut prev_logits = select_last_logits(&logits);

    let mut token_ids = input_ids.to_vec();
    let mut generated_counts = HashMap::new();
    let mut num_generated = 0usize;
    let mut stopped_by_token = false;
    let draft_budget = mtp_config.num_assistant_tokens.max(1);
    let mut speculative_metrics = SpeculativeDecodeMetrics::default();

    while num_generated < gen_config.max_new_tokens {
        let remaining = gen_config.max_new_tokens - num_generated;
        let max_draft = draft_budget.min(remaining);
        // Gemma 4 assistants use a static target KV cache; HF documents the
        // assistant position IDs as constant throughout a drafting round.
        let position_id = token_ids.len().saturating_sub(1) as i32;
        let shared_states = kv_sources.shared_states(cache)?;
        let mut draft_tokens = Vec::with_capacity(max_draft);
        let mut draft_log_probs = Vec::with_capacity(max_draft);
        let mut draft_history = token_ids.clone();
        let mut draft_counts = generated_counts.clone();
        let mut draft_hidden = last_hidden.clone();
        let mut draft_input_token = *token_ids
            .last()
            .ok_or_else(|| Exception::custom("Gemma 4 MTP internal empty token state"))?;

        for _ in 0..max_draft {
            let token_embedding = embed_target_token(target, draft_input_token);
            let inputs_embeds = concatenate_axis(&[&token_embedding, &draft_hidden], -1);
            let (projected_hidden, logits) =
                assistant.forward_logits(&inputs_embeds, &shared_states, position_id)?;
            let log_probs = sampling_log_probs_with_counts(
                &logits,
                &draft_history,
                &draft_counts,
                &gen_config,
            )?;
            let draft_token = sample_from_log_probs(&log_probs)?;
            draft_tokens.push(draft_token);
            draft_log_probs.push(log_probs);
            draft_history.push(draft_token);
            increment_count(&mut draft_counts, draft_token);
            draft_hidden = projected_hidden;
            draft_input_token = draft_token;
            if gen_config.stop_tokens.contains(&draft_token) {
                break;
            }
        }

        if draft_tokens.is_empty() {
            let log_probs = sampling_log_probs_with_counts(
                &prev_logits,
                &token_ids,
                &generated_counts,
                &gen_config,
            )?;
            let token = sample_from_log_probs(&log_probs)?;
            let continue_stream = emit_token(
                token,
                &mut token_ids,
                &mut num_generated,
                &mut stopped_by_token,
                &gen_config,
                &mut on_token,
            )?;
            increment_count(&mut generated_counts, token);
            if !continue_stream || stopped_by_token || num_generated >= gen_config.max_new_tokens {
                break;
            }
            let (hidden, logits) = target_step(target, token, cache)?;
            last_hidden = hidden;
            prev_logits = logits;
            continue;
        }

        let verify_input = token_array(&draft_tokens);
        let verify_logits = target.forward_with_cache(&verify_input, None, Some(cache))?;
        let mut accepted = 0usize;
        let mut correction = None;
        let mut verify_history = token_ids.clone();
        let mut verify_counts = generated_counts.clone();

        while accepted < draft_tokens.len() {
            let row = if accepted == 0 {
                prev_logits.clone()
            } else {
                select_axis(&verify_logits, (accepted - 1) as i32, 1)
            };
            let target_log_probs =
                sampling_log_probs_with_counts(&row, &verify_history, &verify_counts, &gen_config)?;
            let draft_log_probs_row = &draft_log_probs[accepted];
            let token = draft_tokens[accepted];
            let p_target = token_probability_from_log_probs(&target_log_probs, token)?;
            let p_draft = token_probability_from_log_probs(draft_log_probs_row, token)?;
            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else {
                0.0
            };

            if rand_uniform() < accept_prob {
                accepted += 1;
                verify_history.push(token);
                increment_count(&mut verify_counts, token);
            } else {
                let correction_log_probs =
                    correction_log_probs(&target_log_probs, draft_log_probs_row)?;
                correction = Some(sample_from_log_probs(&correction_log_probs)?);
                break;
            }
        }
        speculative_metrics.record_verify_step(draft_tokens.len(), accepted);

        let all_accepted = accepted == draft_tokens.len();
        let mut planned = Vec::with_capacity(draft_tokens.len() + 1);
        planned.extend_from_slice(&draft_tokens[..accepted]);
        let append_after_emit = if all_accepted {
            if num_generated + planned.len() < gen_config.max_new_tokens {
                let row = select_axis(&verify_logits, (draft_tokens.len() - 1) as i32, 1);
                let log_probs = sampling_log_probs_with_counts(
                    &row,
                    &verify_history,
                    &verify_counts,
                    &gen_config,
                )?;
                let bonus = sample_from_log_probs(&log_probs)?;
                planned.push(bonus);
                Some(bonus)
            } else {
                None
            }
        } else {
            let correction = correction.ok_or_else(|| {
                Exception::custom("Gemma 4 MTP sampling rejected without correction token")
            })?;
            planned.push(correction);
            Some(correction)
        };

        let stop_pos = planned
            .iter()
            .position(|token| gen_config.stop_tokens.contains(token));
        let planned_len = stop_pos.map(|idx| idx + 1).unwrap_or(planned.len());

        let mut continue_stream = true;
        let mut emitted_planned_count = 0usize;
        for &token in &planned[..planned_len] {
            continue_stream = emit_token(
                token,
                &mut token_ids,
                &mut num_generated,
                &mut stopped_by_token,
                &gen_config,
                &mut on_token,
            )?;
            emitted_planned_count += 1;
            increment_count(&mut generated_counts, token);
            if !continue_stream || stopped_by_token || num_generated >= gen_config.max_new_tokens {
                break;
            }
        }
        let emitted_draft_count = emitted_planned_count.min(accepted);
        speculative_metrics.emitted_draft_tokens += emitted_draft_count;
        if emitted_planned_count > emitted_draft_count {
            if all_accepted {
                speculative_metrics.bonus_tokens += 1;
            } else {
                speculative_metrics.correction_tokens += 1;
            }
        }
        let rollback = draft_tokens.len().saturating_sub(emitted_draft_count);
        if rollback > 0 {
            cache.rollback(rollback);
        }
        if !continue_stream
            || stopped_by_token
            || num_generated >= gen_config.max_new_tokens
            || stop_pos.is_some()
        {
            break;
        }

        if let Some(token) = append_after_emit {
            let (hidden, logits) = target_step(target, token, cache)?;
            last_hidden = hidden;
            prev_logits = logits;
        }
    }

    Ok(GenerationOutput {
        token_ids,
        num_generated,
        stopped_by_token,
        stopped_by_length: num_generated >= gen_config.max_new_tokens && !stopped_by_token,
        decode_metrics: None,
        speculative_metrics: Some(speculative_metrics),
    })
}

fn emit_token<F>(
    token: u32,
    token_ids: &mut Vec<u32>,
    num_generated: &mut usize,
    stopped_by_token: &mut bool,
    gen_config: &GenerationConfig,
    on_token: &mut F,
) -> Result<bool, Exception>
where
    F: FnMut(u32) -> bool,
{
    token_ids.push(token);
    *num_generated += 1;
    if gen_config.stop_tokens.contains(&token) {
        *stopped_by_token = true;
    }
    Ok(on_token(token))
}

fn target_step(
    target: &mut Gemma4ForCausalLM,
    token: u32,
    cache: &mut KVCache,
) -> Result<(Array, Array), Exception> {
    let input = token_array(&[token]);
    let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
    let (hidden, logits) =
        target.forward_hidden_with_capture(&input, None, Some(cache), &mut capture)?;
    Ok((select_last_sequence(&hidden), select_last_logits(&logits)))
}

fn embed_target_token(target: &Gemma4ForCausalLM, token: u32) -> Array {
    let input = token_array(&[token]);
    target
        .model
        .embed_tokens
        .forward(&input)
        .multiply(&Array::from_f32(target.model.embed_scale))
}

fn increment_count(token_counts: &mut HashMap<u32, usize>, token: u32) {
    *token_counts.entry(token).or_insert(0) += 1;
}

fn correction_log_probs(
    target_log_probs: &Array,
    draft_log_probs: &Array,
) -> Result<Array, Exception> {
    let target_probs = target_log_probs.exp();
    let draft_probs = draft_log_probs.exp();
    let diff = target_probs.subtract(&draft_probs);
    let clipped = diff.maximum(&Array::from_f32(0.0));
    let total = clipped.sum_axis(-1, true);
    let total_value = total.item::<f32>();
    if !total_value.is_finite() || total_value <= 1e-20 {
        return Ok(target_log_probs.clone());
    }
    Ok(clipped.divide(&total).log())
}

fn greedy_token(logits: &Array) -> u32 {
    logits.argmax(-1).item::<u32>()
}

fn token_array(tokens: &[u32]) -> Array {
    let data: Vec<i32> = tokens.iter().map(|token| *token as i32).collect();
    Array::from_i32_slice(&data).reshape(&[1, data.len() as i32])
}

fn select_last_sequence(values: &Array) -> Array {
    let seq_len = values.dim(1);
    select_axis(values, seq_len - 1, 1).reshape(&[1, 1, -1])
}

fn select_last_logits(logits: &Array) -> Array {
    let seq_len = logits.dim(1);
    select_axis(logits, seq_len - 1, 1)
}

fn seed_acceptance_rng(seed: u64) {
    ACCEPTANCE_RNG_STATE.with(|state| {
        state.set(if seed == 0 {
            0xdead_beef_cafe_1234
        } else {
            seed
        });
    });
}

fn rand_uniform() -> f32 {
    ACCEPTANCE_RNG_STATE.with(|state| {
        let mut x = state.get();
        if x == 0 {
            x = seed_from_time();
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        (x >> 40) as f32 / (1u64 << 24) as f32
    })
}

fn seed_from_time() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    if seed == 0 {
        0xdead_beef_cafe_1234
    } else {
        seed
    }
}

thread_local! {
    static ACCEPTANCE_RNG_STATE: std::cell::Cell<u64> = std::cell::Cell::new(seed_from_time());
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
