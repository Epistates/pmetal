//! Multi-token-prediction speculative decoding: the one decode loop every MTP
//! head (Qwen3Next's bundled predictor, Gemma 4's assistant) runs against
//! every target engine.
//!
//! An MTP head drafts from the target's own hidden states. A target hidden
//! state `h[i]`, at position `i`, is what predicted token `x[i + 1]`; the head
//! reads it together with the embedding of `x[i + 1]` and predicts
//! `x[i + 2]`. This is how the heads are trained (DeepSeek-V3's
//! `[h[i]; Emb(x[i + 1])]`) and how serving engines run them, so a pairing of
//! `x[i]` with `h[i]` drafts as if the newest token were missing.
//!
//! Each round starts from a *pending* token: the newest one emitted, which the
//! target has not run yet, and the hidden state that predicted it. The head
//! drafts `k` tokens after it, the target runs the pending token and the
//! drafts in one forward, and that forward's logits at position `j` check
//! draft `j`. The accepted drafts are emitted, then one token from the target
//! itself: the correction at the first rejected draft, or the bonus after the
//! last. That token is the next round's pending token. A round therefore
//! costs one target forward (a recurrent layer's rewind replays only its
//! state update over the kept tokens) and emits between 1 and `k + 1` tokens, and the
//! output is the target's own: greedy decoding gives exactly the tokens plain
//! greedy decoding does, and sampling draws from the target's distribution
//! (rejection sampling).
//!
//! The sampler's penalties, bias and constraints read the emitted sequence's
//! [`DecodeSampler`]. Drafting and verifying each follow the round's drafts
//! on a fork of it, so a draft or a verify position is constrained by the
//! drafts before it, and only the tokens the round emits go into the
//! sequence's own.

use pmetal_runtime::compat::ops::{concatenate_axis, select_axis, slice_axis, take_along_axis};
use pmetal_runtime::compat::{Array, Exception};
use pmetal_runtime::decode::{DecodeSampler, greedy_token, last_token_hidden, last_token_logits};

use crate::generation::{GenerationConfig, GenerationOutput};
use crate::speculative::{SpeculativeDecodeMetrics, SpeculativeDraws, correction_log_probs};

/// The model whose tokens an MTP decode emits: it checks the drafts.
pub trait MtpTarget {
    /// Forget the sequence.
    fn reset(&mut self) -> Result<(), Exception>;

    /// Run `tokens` after everything run so far. Returns the final hidden
    /// states an MTP head reads, `[1, T, H]`, and the logits, `[1, T, V]`.
    fn forward(&mut self, tokens: &[u32]) -> Result<(Array, Array), Exception>;

    /// Run a verify block: [`forward`](Self::forward) for tokens that
    /// [`keep`](Self::keep) may cut back.
    fn verify(&mut self, tokens: &[u32]) -> Result<(Array, Array), Exception> {
        self.forward(tokens)
    }

    /// Keep the first `kept` tokens of the last [`verify`](Self::verify), as
    /// if only they had run.
    fn keep(&mut self, kept: usize) -> Result<(), Exception>;
}

/// An MTP head: drafts tokens from a target's hidden states.
pub trait MtpDrafter<T: MtpTarget> {
    /// Start a sequence. The target ran `prompt` and produced `hidden`, so
    /// `hidden[i]` predicted `prompt[i + 1]`.
    fn prefill(&mut self, target: &mut T, prompt: &[u32], hidden: &Array) -> Result<(), Exception>;

    /// One draft step: the newest token and the hidden state that predicted
    /// it (`[1, 1, H]`; at a round's first step the target's, after that the
    /// head's own). Returns the head's hidden state for the next step and the
    /// `[1, V]` logits of the token after `token`.
    ///
    /// `step` counts the round's steps from 0; `position` is the sequence
    /// position of the round's pending token.
    fn step(
        &mut self,
        target: &mut T,
        token: u32,
        hidden: &Array,
        step: usize,
        position: usize,
    ) -> Result<(Array, Array), Exception>;

    /// Bring the head up to date after a round. The round started from
    /// `pending` and `seed` (the target hidden state that predicted it), ran
    /// one [`step`](Self::step) per draft, and the target ran
    /// `[pending, drafts..]` into `verify_hidden` (`[1, k + 1, H]`), of which
    /// the first `accepted` drafts were kept.
    fn commit(
        &mut self,
        target: &mut T,
        pending: u32,
        seed: &Array,
        drafts: &[u32],
        accepted: usize,
        verify_hidden: &Array,
    ) -> Result<(), Exception>;
}

/// Run an MTP speculative decode of `target` with `drafter`, drafting up to
/// `num_draft_tokens` per round, and call `on_token` with every emitted token
/// (`false` stops the decode).
pub fn generate_mtp_streaming<T, D, F>(
    target: &mut T,
    drafter: &mut D,
    input_ids: &[u32],
    gen_config: &GenerationConfig,
    num_draft_tokens: usize,
    mut on_token: F,
) -> Result<GenerationOutput, Exception>
where
    T: MtpTarget,
    D: MtpDrafter<T>,
    F: FnMut(u32) -> bool,
{
    if input_ids.is_empty() {
        return Err(Exception::custom(
            "MTP decoding requires a non-empty prompt",
        ));
    }
    let mut out = Emitter {
        token_ids: input_ids.to_vec(),
        sampler: gen_config.sampler(input_ids),
        num_generated: 0,
        stopped_by_token: false,
        stopped: false,
        gen_config,
    };
    if gen_config.max_new_tokens == 0 {
        return Ok(out.finish(None));
    }
    let mut draws = SpeculativeDraws::new(gen_config.seed);
    let mut metrics = SpeculativeDecodeMetrics::default();
    let budget = num_draft_tokens.max(1);

    target.reset()?;
    let (hidden, logits) = target.forward(input_ids)?;
    drafter.prefill(target, input_ids, &hidden)?;

    // The first token is the target's alone.
    let mut pending = pick(
        &last_token_logits(&logits),
        &mut out.sampler,
        gen_config,
        &mut draws,
    )?;
    let mut seed = last_token_hidden(&hidden);
    out.emit(pending, &mut on_token);

    while !out.stopped {
        let remaining = gen_config.max_new_tokens - out.num_generated;
        // A round emits at most one token past its drafts.
        let max_draft = budget.min(remaining - 1);
        let position = out.token_ids.len() - 1;

        let mut drafts = Vec::with_capacity(max_draft);
        let mut draft_log_probs = Vec::with_capacity(max_draft);
        let mut drafting = out.sampler.fork();
        let mut hidden = seed.clone();
        let mut token = pending;
        for step in 0..max_draft {
            let (next_hidden, logits) = drafter.step(target, token, &hidden, step, position)?;
            token = if gen_config.do_sample {
                let log_probs = drafting.log_probs(&logits);
                let token = draws.sample(&log_probs)?;
                draft_log_probs.push(log_probs);
                token
            } else {
                greedy_choice(&logits, &mut drafting)
            };
            hidden = next_hidden;
            drafts.push(token);
            drafting.record_id(token);
            if gen_config.stop_tokens.contains(&token) {
                break;
            }
        }

        let mut block = Vec::with_capacity(drafts.len() + 1);
        block.push(pending);
        block.extend_from_slice(&drafts);
        let (verify_hidden, verify_logits) = target.verify(&block)?;
        let (accepted, next) = if gen_config.do_sample {
            accept_sampled(
                &drafts,
                &draft_log_probs,
                &verify_logits,
                out.sampler.fork(),
                &mut draws,
            )?
        } else {
            accept_greedy(&drafts, &verify_logits, out.sampler.fork())?
        };
        if !drafts.is_empty() {
            metrics.record_verify_step(drafts.len(), accepted);
        }

        for &draft in &drafts[..accepted] {
            if out.stopped {
                break;
            }
            out.emit(draft, &mut on_token);
            metrics.emitted_draft_tokens += 1;
        }
        if out.stopped {
            break;
        }
        out.emit(next, &mut on_token);
        if accepted == drafts.len() {
            metrics.bonus_tokens += usize::from(!drafts.is_empty());
        } else {
            metrics.correction_tokens += 1;
        }
        if out.stopped {
            break;
        }

        target.keep(accepted + 1)?;
        drafter.commit(target, pending, &seed, &drafts, accepted, &verify_hidden)?;
        seed = slice_axis(&verify_hidden, 1, accepted as i32, accepted as i32 + 1);
        pending = next;
    }

    Ok(out.finish(Some(metrics)))
}

/// The emitted sequence, its sampler, and the stop conditions.
struct Emitter<'a> {
    token_ids: Vec<u32>,
    /// The sampler over the prompt and every emitted token, in order.
    sampler: DecodeSampler,
    num_generated: usize,
    stopped_by_token: bool,
    /// A stop token, the length limit, or the caller ended the decode.
    stopped: bool,
    gen_config: &'a GenerationConfig,
}

impl Emitter<'_> {
    /// Emit `token`, unless it is a stop token: that ends the decode and is
    /// neither in the output nor shown to `on_token`, as in plain generation.
    fn emit(&mut self, token: u32, on_token: &mut impl FnMut(u32) -> bool) {
        if self.gen_config.is_stop_token(token) {
            self.stopped_by_token = true;
            self.stopped = true;
            return;
        }
        self.token_ids.push(token);
        self.sampler.record_id(token);
        self.num_generated += 1;
        let go_on = on_token(token);
        self.stopped = !go_on || self.num_generated >= self.gen_config.max_new_tokens;
    }

    fn finish(self, metrics: Option<SpeculativeDecodeMetrics>) -> GenerationOutput {
        GenerationOutput {
            stopped_by_length: self.num_generated >= self.gen_config.max_new_tokens
                && !self.stopped_by_token,
            token_ids: self.token_ids,
            num_generated: self.num_generated,
            stopped_by_token: self.stopped_by_token,
            decode_metrics: None,
            speculative_metrics: metrics,
        }
    }
}

/// The target's token for `[1, V]` logits under `sampler`, which this reads
/// but does not record into: greedy, or drawn from the sampling
/// distribution.
fn pick(
    logits: &Array,
    sampler: &mut DecodeSampler,
    gen_config: &GenerationConfig,
    draws: &mut SpeculativeDraws,
) -> Result<u32, Exception> {
    if gen_config.do_sample {
        let log_probs = sampler.log_probs(logits);
        draws.sample(&log_probs)
    } else {
        Ok(greedy_choice(logits, sampler))
    }
}

/// The greedy token after the sampler's penalties, bias and constraints, as
/// plain greedy decoding picks it: the argmax of the constrained logits.
fn greedy_choice(logits: &Array, sampler: &mut DecodeSampler) -> u32 {
    greedy_token(&sampler.constrained_logits(logits))
}

/// How many drafts the target's greedy choice agrees with, and its token at
/// the first disagreement (or after the last draft). `verify_logits[j]` is
/// the target's choice for draft `j`, constrained by the emitted sequence
/// (`sampler`) and the drafts before `j`.
///
/// The choices at every position are computed first, each as if the drafts
/// before it were kept, and read back in one transfer: a position after the
/// first disagreement is never looked at.
fn accept_greedy(
    drafts: &[u32],
    verify_logits: &Array,
    mut sampler: DecodeSampler,
) -> Result<(usize, u32), Exception> {
    let mut choices = Vec::with_capacity(drafts.len() + 1);
    for j in 0..=drafts.len() {
        let row = select_axis(verify_logits, j as i32, 1);
        choices.push(sampler.constrained_logits(&row).argmax(-1).reshape(&[1]));
        if let Some(&draft) = drafts.get(j) {
            sampler.record_id(draft);
        }
    }
    let refs: Vec<&Array> = choices.iter().collect();
    let chosen = concatenate_axis(&refs, 0);
    chosen.eval();
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| Exception::custom(format!("speculative verification: {e}")))?;
    let chosen = chosen.as_slice::<u32>();
    for (j, &draft) in drafts.iter().enumerate() {
        if chosen[j] != draft {
            return Ok((j, chosen[j]));
        }
    }
    Ok((drafts.len(), chosen[drafts.len()]))
}

/// Rejection sampling: draft `j` is kept with probability
/// `min(1, p(d) / q(d))`; the first rejected one is replaced by a draw from
/// the residual `max(0, p - q)`, and a fully accepted block gets a bonus
/// token drawn from the target. `p` is constrained by the emitted sequence
/// (`sampler`) and the drafts before `j`, as `q` was when the draft was
/// drawn.
///
/// The target's distribution at every position is built first, each as if
/// the drafts before it were kept, and the probabilities of the drafts are
/// read back in one transfer; the acceptance tests then run on the host in
/// order, each with its own uniform, so a rejection leaves the later
/// positions' work unused and the draws are those of a position-by-position
/// loop.
fn accept_sampled(
    drafts: &[u32],
    draft_log_probs: &[Array],
    verify_logits: &Array,
    mut sampler: DecodeSampler,
    draws: &mut SpeculativeDraws,
) -> Result<(usize, u32), Exception> {
    let mut targets = Vec::with_capacity(drafts.len() + 1);
    for j in 0..=drafts.len() {
        let row = select_axis(verify_logits, j as i32, 1);
        targets.push(sampler.log_probs(&row));
        if let Some(&draft) = drafts.get(j) {
            sampler.record_id(draft);
        }
    }

    // log p(d_j) and log q(d_j) for every draft, in one array.
    let log_probs_of_drafts = |dists: &[&Array]| -> Array {
        let picked: Vec<Array> = dists
            .iter()
            .zip(drafts)
            .map(|(dist, &draft)| {
                take_along_axis(dist, &Array::from_slice(&[draft as i32], &[1, 1]), -1)
            })
            .collect();
        let refs: Vec<&Array> = picked.iter().collect();
        concatenate_axis(&refs, 1).reshape(&[-1])
    };
    let mut readback = None;
    if !drafts.is_empty() {
        let vocab = targets[0].dim(-1);
        if let Some(&bad) = drafts.iter().find(|&&d| d as i32 >= vocab) {
            return Err(Exception::custom(format!(
                "speculative verification: draft token {bad} is outside the vocabulary of {vocab}"
            )));
        }
        let target_refs: Vec<&Array> = targets[..drafts.len()].iter().collect();
        let draft_refs: Vec<&Array> = draft_log_probs.iter().collect();
        let both = concatenate_axis(
            &[
                &log_probs_of_drafts(&target_refs),
                &log_probs_of_drafts(&draft_refs),
            ],
            0,
        )
        .exp();
        both.eval();
        pmetal_runtime::check_unobserved_error()
            .map_err(|e| Exception::custom(format!("speculative verification: {e}")))?;
        readback = Some(both.as_slice::<f32>().to_vec());
    }

    for (j, _) in drafts.iter().enumerate() {
        let probs = readback.as_ref().expect("read back with drafts");
        let (p, q) = (probs[j], probs[drafts.len() + j]);
        let keep = if q > 0.0 { (p / q).min(1.0) } else { 0.0 };
        if draws.uniform() >= keep {
            let residual = correction_log_probs(&targets[j], &draft_log_probs[j])?;
            return Ok((j, draws.sample(&residual)?));
        }
    }
    let bonus = draws.sample(&targets[drafts.len()])?;
    Ok((drafts.len(), bonus))
}
