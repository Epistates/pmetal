//! Qwen3Next bundled MTP speculative decoding.

use std::collections::HashMap;

use pmetal_bridge::compat::{Array, Exception, ops::select_axis};
use pmetal_mlx::{
    kv_cache::{GdnVerifyInputs, KVCache, MambaCache, MambaSnapshot},
    speculative::SpecCapture,
};

use crate::{
    architectures::{Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3NextMtpForCausalLM},
    generation::{
        GenerationConfig, GenerationOutput, SpeculativeDecodeMetrics, sample_from_log_probs,
        sampling_log_probs_with_counts, token_probability_from_log_probs,
    },
};

#[derive(Debug, Clone)]
pub struct Qwen3NextMtpConfig {
    pub num_draft_tokens: usize,
}

impl Default for Qwen3NextMtpConfig {
    fn default() -> Self {
        Self {
            // llama.cpp's initial Qwen3.6-MTP tuning uses draft_max=3.
            num_draft_tokens: 3,
        }
    }
}

pub trait Qwen3NextMtpTarget {
    fn qwen3_next_config(&self) -> &Qwen3NextConfig;

    fn forward_hidden_for_mtp(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<(Array, Array), Exception>;
}

impl Qwen3NextMtpTarget for Qwen3NextForCausalLM {
    fn qwen3_next_config(&self) -> &Qwen3NextConfig {
        &self.config
    }

    fn forward_hidden_for_mtp(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<(Array, Array), Exception> {
        let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
        self.forward_hidden_with_capture(input_ids, mask, kv_cache, mamba_cache, &mut capture)
    }
}

pub fn validate_qwen3_next_mtp_pair<T: Qwen3NextMtpTarget + ?Sized>(
    target: &T,
    mtp: &Qwen3NextMtpForCausalLM,
) -> Result<(), Exception> {
    let target_config = target.qwen3_next_config();
    if target_config.hidden_size != mtp.config.hidden_size {
        return Err(Exception::custom(format!(
            "Qwen MTP hidden_size ({}) must match target hidden_size ({})",
            mtp.config.hidden_size, target_config.hidden_size
        )));
    }
    if target_config.vocab_size != mtp.config.vocab_size {
        return Err(Exception::custom(format!(
            "Qwen MTP vocab_size ({}) must match target vocab_size ({})",
            mtp.config.vocab_size, target_config.vocab_size
        )));
    }
    if mtp.config.mtp_num_hidden_layers() == 0 {
        return Err(Exception::custom("Qwen MTP has no predictor layers"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_qwen3_next_mtp_streaming<F>(
    target: &mut Qwen3NextForCausalLM,
    mtp: &mut Qwen3NextMtpForCausalLM,
    input_ids: &[u32],
    gen_config: GenerationConfig,
    target_cache: &mut KVCache,
    target_mamba_cache: &mut MambaCache,
    mtp_cache: &mut KVCache,
    mtp_config: Qwen3NextMtpConfig,
    mut on_token: F,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(u32) -> bool,
{
    if input_ids.is_empty() {
        return Err(Exception::custom("Qwen MTP requires a non-empty prompt"));
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
    if let Some(seed) = gen_config.seed {
        pmetal_bridge::inline_array::random_seed(seed);
        seed_acceptance_rng(seed);
    }

    validate_qwen3_next_mtp_pair(target, mtp)?;
    target_cache.reset();
    target_mamba_cache.reset();
    mtp_cache.reset();

    let prompt = token_array(input_ids);
    let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
    let (target_hidden, target_logits) = target.forward_hidden_with_capture(
        &prompt,
        None,
        Some(target_cache),
        Some(target_mamba_cache),
        &mut capture,
    )?;
    let mut prev_target_logits = select_last_logits(&target_logits);

    let (mtp_hidden, mtp_logits) =
        mtp.forward_logits(&prompt, &target_hidden, None, Some(mtp_cache), 0)?;
    let mut last_mtp_hidden = select_last_sequence(&mtp_hidden);
    let mut prev_mtp_logits = select_last_logits(&mtp_logits);

    let mut token_ids = input_ids.to_vec();
    let mut generated_counts = HashMap::new();
    let mut num_generated = 0usize;
    let mut stopped_by_token = false;
    let draft_budget = mtp_config.num_draft_tokens.max(1);
    let mut speculative_metrics = SpeculativeDecodeMetrics::default();

    while num_generated < gen_config.max_new_tokens {
        let remaining = gen_config.max_new_tokens - num_generated;
        let max_draft = draft_budget.min(remaining);
        let base_mtp_hidden = last_mtp_hidden.clone();
        let base_mtp_logits = prev_mtp_logits.clone();

        let mut draft_tokens = Vec::with_capacity(max_draft);
        let mut draft_log_probs = Vec::with_capacity(max_draft);
        let mut draft_history = token_ids.clone();
        let mut draft_counts = generated_counts.clone();
        let mut draft_hidden = last_mtp_hidden.clone();
        let mut draft_logits = prev_mtp_logits.clone();

        for draft_idx in 0..max_draft {
            let draft_token = if gen_config.do_sample {
                let log_probs = sampling_log_probs_with_counts(
                    &draft_logits,
                    &draft_history,
                    &draft_counts,
                    &gen_config,
                )?;
                let token = sample_from_log_probs(&log_probs)?;
                draft_log_probs.push(log_probs);
                token
            } else {
                greedy_token(&draft_logits)
            };
            draft_tokens.push(draft_token);
            draft_history.push(draft_token);
            increment_count(&mut draft_counts, draft_token);

            let (next_hidden, next_logits) =
                mtp_step(mtp, draft_token, &draft_hidden, mtp_cache, draft_idx + 1)?;
            draft_hidden = next_hidden;
            draft_logits = next_logits;
            if gen_config.stop_tokens.contains(&draft_token) || draft_idx + 1 >= max_draft {
                break;
            }
        }

        let mamba_snapshot = target_mamba_cache.snapshot();
        capture.clear();
        let verify_input = token_array(&draft_tokens);
        let (verify_hidden, verify_logits) = target.forward_hidden_with_capture(
            &verify_input,
            None,
            Some(target_cache),
            Some(target_mamba_cache),
            &mut capture,
        )?;

        let (accepted, correction) = if gen_config.do_sample {
            accept_sampled_draft(
                &draft_tokens,
                &draft_log_probs,
                &prev_target_logits,
                &verify_logits,
                &token_ids,
                &generated_counts,
                &gen_config,
            )?
        } else {
            accept_greedy_draft(&draft_tokens, &prev_target_logits, &verify_logits)
        };
        record_speculative_verify(&mut speculative_metrics, draft_tokens.len(), accepted);

        let all_accepted = accepted == draft_tokens.len();
        let mut planned = Vec::with_capacity(draft_tokens.len() + 1);
        planned.extend_from_slice(&draft_tokens[..accepted]);
        let append_after_emit = if all_accepted {
            if num_generated + planned.len() < gen_config.max_new_tokens {
                let row = select_axis(&verify_logits, (draft_tokens.len() - 1) as i32, 1);
                let bonus = if gen_config.do_sample {
                    let mut verify_history = token_ids.clone();
                    let mut verify_counts = generated_counts.clone();
                    for &token in &draft_tokens {
                        verify_history.push(token);
                        increment_count(&mut verify_counts, token);
                    }
                    let log_probs = sampling_log_probs_with_counts(
                        &row,
                        &verify_history,
                        &verify_counts,
                        &gen_config,
                    )?;
                    sample_from_log_probs(&log_probs)?
                } else {
                    greedy_token(&row)
                };
                planned.push(bonus);
                Some(bonus)
            } else {
                None
            }
        } else {
            let correction = correction.ok_or_else(|| {
                Exception::custom("Qwen MTP rejected a draft without correction token")
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

        rollback_target_to_accepted(
            target_cache,
            target_mamba_cache,
            &mamba_snapshot,
            &mut capture,
            draft_tokens.len(),
            emitted_draft_count,
        )?;

        mtp_cache.rollback(draft_tokens.len());
        let (mut next_mtp_hidden, mut next_mtp_logits) =
            (base_mtp_hidden.clone(), base_mtp_logits.clone());
        if emitted_draft_count > 0 {
            (next_mtp_hidden, next_mtp_logits) = replay_mtp_with_target_hidden(
                mtp,
                mtp_cache,
                &draft_tokens[..emitted_draft_count],
                &verify_hidden,
            )?;
        }
        last_mtp_hidden = next_mtp_hidden;
        prev_mtp_logits = next_mtp_logits;

        if !continue_stream
            || stopped_by_token
            || num_generated >= gen_config.max_new_tokens
            || stop_pos.is_some()
        {
            break;
        }

        if let Some(token) = append_after_emit
            && emitted_draft_count < planned_len
        {
            let (target_hidden, target_logits) =
                target_step(target, token, target_cache, target_mamba_cache)?;
            prev_target_logits = target_logits;
            let (mtp_hidden, mtp_logits) = mtp_step(
                mtp,
                token,
                &target_hidden,
                mtp_cache,
                emitted_draft_count + 1,
            )?;
            last_mtp_hidden = mtp_hidden;
            prev_mtp_logits = mtp_logits;
        } else if emitted_draft_count > 0 {
            let last_idx = emitted_draft_count - 1;
            prev_target_logits = select_axis(&verify_logits, last_idx as i32, 1);
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

#[allow(clippy::too_many_arguments)]
pub fn generate_qwen3_next_mtp_streaming_rebuild<T, F>(
    target: &mut T,
    mtp: &mut Qwen3NextMtpForCausalLM,
    input_ids: &[u32],
    gen_config: GenerationConfig,
    target_cache: &mut KVCache,
    target_mamba_cache: &mut MambaCache,
    mtp_cache: &mut KVCache,
    mtp_config: Qwen3NextMtpConfig,
    mut on_token: F,
) -> Result<GenerationOutput, Exception>
where
    T: Qwen3NextMtpTarget,
    F: FnMut(u32) -> bool,
{
    if input_ids.is_empty() {
        return Err(Exception::custom("Qwen MTP requires a non-empty prompt"));
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
    if let Some(seed) = gen_config.seed {
        pmetal_bridge::inline_array::random_seed(seed);
        seed_acceptance_rng(seed);
    }

    validate_qwen3_next_mtp_pair(target, mtp)?;
    let mut mtp_state = rebuild_qwen_mtp_state(
        target,
        mtp,
        input_ids,
        target_cache,
        target_mamba_cache,
        mtp_cache,
    )?;

    let mut token_ids = input_ids.to_vec();
    let mut generated_counts = HashMap::new();
    let mut num_generated = 0usize;
    let mut stopped_by_token = false;
    let draft_budget = mtp_config.num_draft_tokens.max(1);
    let mut speculative_metrics = SpeculativeDecodeMetrics::default();

    while num_generated < gen_config.max_new_tokens {
        let remaining = gen_config.max_new_tokens - num_generated;
        let max_draft = draft_budget.min(remaining);

        let mut draft_tokens = Vec::with_capacity(max_draft);
        let mut draft_log_probs = Vec::with_capacity(max_draft);
        let mut draft_history = token_ids.clone();
        let mut draft_counts = generated_counts.clone();
        let mut draft_hidden = mtp_state.last_mtp_hidden.clone();
        let mut draft_logits = mtp_state.prev_mtp_logits.clone();

        for draft_idx in 0..max_draft {
            let draft_token = if gen_config.do_sample {
                let log_probs = sampling_log_probs_with_counts(
                    &draft_logits,
                    &draft_history,
                    &draft_counts,
                    &gen_config,
                )?;
                let token = sample_from_log_probs(&log_probs)?;
                draft_log_probs.push(log_probs);
                token
            } else {
                greedy_token(&draft_logits)
            };
            draft_tokens.push(draft_token);
            draft_history.push(draft_token);
            increment_count(&mut draft_counts, draft_token);

            let (next_hidden, next_logits) =
                mtp_step(mtp, draft_token, &draft_hidden, mtp_cache, draft_idx + 1)?;
            draft_hidden = next_hidden;
            draft_logits = next_logits;
            if gen_config.stop_tokens.contains(&draft_token) || draft_idx + 1 >= max_draft {
                break;
            }
        }

        let verify_input = token_array(&draft_tokens);
        let (_verify_hidden, verify_logits) = target.forward_hidden_for_mtp(
            &verify_input,
            None,
            Some(target_cache),
            Some(target_mamba_cache),
        )?;

        let (accepted, correction) = if gen_config.do_sample {
            accept_sampled_draft(
                &draft_tokens,
                &draft_log_probs,
                &mtp_state.prev_target_logits,
                &verify_logits,
                &token_ids,
                &generated_counts,
                &gen_config,
            )?
        } else {
            accept_greedy_draft(&draft_tokens, &mtp_state.prev_target_logits, &verify_logits)
        };
        record_speculative_verify(&mut speculative_metrics, draft_tokens.len(), accepted);

        let all_accepted = accepted == draft_tokens.len();
        let mut planned = Vec::with_capacity(draft_tokens.len() + 1);
        planned.extend_from_slice(&draft_tokens[..accepted]);
        if all_accepted {
            if num_generated + planned.len() < gen_config.max_new_tokens {
                let row = select_axis(&verify_logits, (draft_tokens.len() - 1) as i32, 1);
                let bonus = if gen_config.do_sample {
                    let mut verify_history = token_ids.clone();
                    let mut verify_counts = generated_counts.clone();
                    for &token in &draft_tokens {
                        verify_history.push(token);
                        increment_count(&mut verify_counts, token);
                    }
                    let log_probs = sampling_log_probs_with_counts(
                        &row,
                        &verify_history,
                        &verify_counts,
                        &gen_config,
                    )?;
                    sample_from_log_probs(&log_probs)?
                } else {
                    greedy_token(&row)
                };
                planned.push(bonus);
            }
        } else {
            let correction = correction.ok_or_else(|| {
                Exception::custom("Qwen MTP rejected a draft without correction token")
            })?;
            planned.push(correction);
        }

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

        if !continue_stream
            || stopped_by_token
            || num_generated >= gen_config.max_new_tokens
            || stop_pos.is_some()
        {
            break;
        }

        mtp_state = rebuild_qwen_mtp_state(
            target,
            mtp,
            &token_ids,
            target_cache,
            target_mamba_cache,
            mtp_cache,
        )?;
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

struct QwenMtpState {
    prev_target_logits: Array,
    last_mtp_hidden: Array,
    prev_mtp_logits: Array,
}

fn rebuild_qwen_mtp_state<T: Qwen3NextMtpTarget>(
    target: &mut T,
    mtp: &mut Qwen3NextMtpForCausalLM,
    committed_tokens: &[u32],
    target_cache: &mut KVCache,
    target_mamba_cache: &mut MambaCache,
    mtp_cache: &mut KVCache,
) -> Result<QwenMtpState, Exception> {
    target_cache.reset();
    target_mamba_cache.reset();
    mtp_cache.reset();

    let input = token_array(committed_tokens);
    let (target_hidden, target_logits) = target.forward_hidden_for_mtp(
        &input,
        None,
        Some(target_cache),
        Some(target_mamba_cache),
    )?;
    let prev_target_logits = select_last_logits(&target_logits);

    let (mtp_hidden, mtp_logits) =
        mtp.forward_logits(&input, &target_hidden, None, Some(mtp_cache), 0)?;
    Ok(QwenMtpState {
        prev_target_logits,
        last_mtp_hidden: select_last_sequence(&mtp_hidden),
        prev_mtp_logits: select_last_logits(&mtp_logits),
    })
}

fn accept_greedy_draft(
    draft_tokens: &[u32],
    prev_target_logits: &Array,
    verify_logits: &Array,
) -> (usize, Option<u32>) {
    let mut matched = 0usize;
    while matched < draft_tokens.len() {
        let row = if matched == 0 {
            prev_target_logits.clone()
        } else {
            select_axis(verify_logits, (matched - 1) as i32, 1)
        };
        if greedy_token(&row) != draft_tokens[matched] {
            break;
        }
        matched += 1;
    }

    if matched == draft_tokens.len() {
        (matched, None)
    } else {
        let row = if matched == 0 {
            prev_target_logits.clone()
        } else {
            select_axis(verify_logits, (matched - 1) as i32, 1)
        };
        (matched, Some(greedy_token(&row)))
    }
}

fn record_speculative_verify(
    metrics: &mut SpeculativeDecodeMetrics,
    drafted: usize,
    accepted: usize,
) {
    metrics.record_verify_step(drafted, accepted);
}

#[allow(clippy::too_many_arguments)]
fn accept_sampled_draft(
    draft_tokens: &[u32],
    draft_log_probs: &[Array],
    prev_target_logits: &Array,
    verify_logits: &Array,
    token_ids: &[u32],
    generated_counts: &HashMap<u32, usize>,
    gen_config: &GenerationConfig,
) -> Result<(usize, Option<u32>), Exception> {
    let mut accepted = 0usize;
    let mut verify_history = token_ids.to_vec();
    let mut verify_counts = generated_counts.clone();

    while accepted < draft_tokens.len() {
        let row = if accepted == 0 {
            prev_target_logits.clone()
        } else {
            select_axis(verify_logits, (accepted - 1) as i32, 1)
        };
        let target_log_probs =
            sampling_log_probs_with_counts(&row, &verify_history, &verify_counts, gen_config)?;
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
            return Ok((
                accepted,
                Some(sample_from_log_probs(&correction_log_probs)?),
            ));
        }
    }

    Ok((accepted, None))
}

fn rollback_target_to_accepted(
    target_cache: &mut KVCache,
    target_mamba_cache: &mut MambaCache,
    mamba_snapshot: &[MambaSnapshot],
    capture: &mut SpecCapture,
    draft_len: usize,
    accepted: usize,
) -> Result<(), Exception> {
    let rollback = draft_len.saturating_sub(accepted);
    if rollback > 0 {
        target_cache.rollback(rollback);
        let num_layers = target_mamba_cache.num_layers();
        let mut per_layer: Vec<Option<GdnVerifyInputs>> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            per_layer.push(capture.gdn_inputs.remove(&layer_idx));
        }
        target_mamba_cache.rewind_from_snapshots(mamba_snapshot, &per_layer, accepted)?;
    }
    Ok(())
}

fn replay_mtp_with_target_hidden(
    mtp: &mut Qwen3NextMtpForCausalLM,
    mtp_cache: &mut KVCache,
    tokens: &[u32],
    target_hidden: &Array,
) -> Result<(Array, Array), Exception> {
    let mut last_hidden = None;
    let mut last_logits = None;
    for (idx, &token) in tokens.iter().enumerate() {
        let hidden = select_axis(target_hidden, idx as i32, 1).reshape(&[1, 1, -1]);
        let (h, logits) = mtp_step(mtp, token, &hidden, mtp_cache, idx)?;
        last_hidden = Some(h);
        last_logits = Some(logits);
    }
    Ok((
        last_hidden.ok_or_else(|| Exception::custom("Qwen MTP replay received no tokens"))?,
        last_logits.ok_or_else(|| Exception::custom("Qwen MTP replay received no tokens"))?,
    ))
}

fn mtp_step(
    mtp: &mut Qwen3NextMtpForCausalLM,
    token: u32,
    hidden: &Array,
    mtp_cache: &mut KVCache,
    step_idx: usize,
) -> Result<(Array, Array), Exception> {
    let input = token_array(&[token]);
    let (hidden, logits) = mtp.forward_logits(&input, hidden, None, Some(mtp_cache), step_idx)?;
    Ok((select_last_sequence(&hidden), select_last_logits(&logits)))
}

fn target_step(
    target: &mut Qwen3NextForCausalLM,
    token: u32,
    target_cache: &mut KVCache,
    target_mamba_cache: &mut MambaCache,
) -> Result<(Array, Array), Exception> {
    let input = token_array(&[token]);
    let mut capture = SpecCapture::with_layers_and_embedding(Vec::new(), false);
    let (hidden, logits) = target.forward_hidden_with_capture(
        &input,
        None,
        Some(target_cache),
        Some(target_mamba_cache),
        &mut capture,
    )?;
    Ok((select_last_sequence(&hidden), select_last_logits(&logits)))
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

fn increment_count(token_counts: &mut HashMap<u32, usize>, token: u32) {
    *token_counts.entry(token).or_insert(0) += 1;
}

thread_local! {
    static ACCEPTANCE_RNG_STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
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
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x1234_5678_9abc_def0);
    nanos ^ 0xa5a5_5a5a_dead_beef
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_qwen_mtp_rejects_mismatched_hidden_size() {
        let mut target_config = crate::architectures::Qwen3NextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: Some(1),
            head_dim: Some(4),
            vocab_size: 16,
            num_experts: 0,
            mtp_num_hidden_layers: Some(1),
            ..Default::default()
        };
        let target =
            crate::architectures::Qwen3NextForCausalLM::new(target_config.clone()).unwrap();
        target_config.hidden_size = 8;
        let mtp = crate::architectures::Qwen3NextMtpForCausalLM::new(target_config).unwrap();
        assert!(validate_qwen3_next_mtp_pair(&target, &mtp).is_err());
    }

    #[test]
    fn validate_qwen_mtp_accepts_multiple_predictor_layers() {
        let config = crate::architectures::Qwen3NextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: Some(1),
            head_dim: Some(4),
            vocab_size: 16,
            num_experts: 0,
            mtp_num_hidden_layers: Some(2),
            ..Default::default()
        };
        let target = crate::architectures::Qwen3NextForCausalLM::new(config.clone()).unwrap();
        let mtp = crate::architectures::Qwen3NextMtpForCausalLM::new(config).unwrap();

        validate_qwen3_next_mtp_pair(&target, &mtp).unwrap();
        assert_eq!(mtp.create_cache(16).config().num_layers, 2);
    }
}
