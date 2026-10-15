//! What every speculative decoder shares: the random draws of its tests, the
//! rejection-sampling correction, and its accounting.
//!
//! A speculative decoder proposes a block of tokens cheaply and has the target
//! check the whole block in one forward. For the output to be the target's own,
//! each draft token is kept with probability `min(1, p / q)` (`p` the target's
//! probability of it, `q` the draft's) and the first one rejected is replaced
//! by a draw from the residual `norm(max(0, p - q))`, which together draw every
//! token from exactly `p` (Leviathan et al., arXiv 2211.17192; Chen et al.,
//! arXiv 2302.01318). Greedy decoding is the special case of point-mass
//! distributions: a draft is kept when it is the target's argmax.
//!
//! A draft that is deterministic (DFlash's per-position argmax, a prompt
//! lookup) is a point mass `q`, for which the test reduces to "keep the draft
//! if the target's own draw is the draft": the decoder draws the target's
//! token at each position of the block, in order, and keeps the draft while
//! they agree. That draws the same tokens, from the same keys, as plain
//! sampling does, so a seeded speculative decode equals the seeded plain one.
//! The loops themselves live in [`crate::mtp`] (a draft head sampling from its
//! own distribution) and [`crate::dflash_decoder`] (deterministic drafts).

use pmetal_runtime::compat::ops::take_along_axis;
use pmetal_runtime::compat::{Array, Exception};

/// Acceptance and accounting counters for speculative decoding.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeculativeDecodeMetrics {
    /// Number of draft/assistant tokens proposed to the target verifier.
    pub drafted_tokens: usize,
    /// Number of drafted tokens accepted by the target verifier.
    pub accepted_draft_tokens: usize,
    /// Number of accepted draft tokens actually emitted to the caller.
    pub emitted_draft_tokens: usize,
    /// Target bonus tokens emitted after fully accepted draft blocks.
    pub bonus_tokens: usize,
    /// Target correction tokens emitted after partial or zero-token accept.
    pub correction_tokens: usize,
    /// Number of target verification passes.
    pub verify_steps: usize,
    /// Verification passes where every drafted token was accepted.
    pub full_accept_steps: usize,
    /// Verification passes where some but not all drafted tokens were accepted.
    pub partial_accept_steps: usize,
    /// Verification passes where no drafted token was accepted.
    pub zero_accept_steps: usize,
    /// Largest draft block attempted in one speculative step.
    pub max_draft_tokens: usize,
}

impl SpeculativeDecodeMetrics {
    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f32 / self.drafted_tokens as f32
        }
    }

    pub fn avg_accepted_per_verify(&self) -> f32 {
        if self.verify_steps == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f32 / self.verify_steps as f32
        }
    }

    pub fn record_verify_step(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        let accepted = accepted.min(drafted);
        self.drafted_tokens += drafted;
        self.accepted_draft_tokens += accepted;
        self.verify_steps += 1;
        self.max_draft_tokens = self.max_draft_tokens.max(drafted);
        if accepted == drafted {
            self.full_accept_steps += 1;
        } else if accepted == 0 {
            self.zero_accept_steps += 1;
        } else {
            self.partial_accept_steps += 1;
        }
    }
}

/// The random draws of one speculative generation: the tokens it samples
/// (drafts, corrections, bonus tokens) and the uniforms its acceptance tests
/// compare against.
///
/// Seeded, the `n`-th draw of either kind comes from
/// [`pmetal_runtime::decode::draw_key`]`(seed, n)` (or its bits, for a
/// uniform), as the plain sampler's tokens do: no random state is shared
/// with the thread or with another generation, so a seeded speculative
/// decode picks the same tokens whichever thread runs it and whatever else
/// draws meanwhile. Unseeded, tokens take the thread's MLX key and uniforms
/// a SplitMix64 stream started from the clock.
#[derive(Debug)]
pub struct SpeculativeDraws {
    seed: Option<u64>,
    /// The seed, or the clock when unseeded: where the uniforms come from.
    uniform_seed: u64,
    next: u64,
}

impl SpeculativeDraws {
    /// The draws of a generation with this seed (`None`: unseeded).
    pub fn new(seed: Option<u64>) -> Self {
        let uniform_seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0x1234_5678_9abc_def0, |d| d.as_nanos() as u64)
        });
        Self {
            seed,
            uniform_seed,
            next: 0,
        }
    }

    fn advance(&mut self) -> u64 {
        let draw = self.next;
        self.next += 1;
        draw
    }

    /// Sample a token from normalized log-probabilities.
    pub fn sample(&mut self, log_probs: &Array) -> Result<u32, Exception> {
        let draw = self.advance();
        let key = self
            .seed
            .map(|seed| pmetal_runtime::decode::draw_key(seed, draw));
        let token = pmetal_runtime::compat::random::categorical(log_probs, -1, key.as_ref());
        token.eval();
        pmetal_runtime::check_unobserved_error()
            .map_err(|e| Exception::custom(format!("speculative sample: {e}")))?;
        Ok(token.item::<u32>())
    }

    /// A uniform in `[0, 1)`, with 24 random bits.
    pub fn uniform(&mut self) -> f32 {
        let draw = self.advance();
        let bits = pmetal_runtime::decode::draw_bits(self.uniform_seed, draw);
        (bits >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The log-probabilities a speculative decoder samples its correction token
/// from after rejecting a draft token: the residual `norm(max(0, p − q))` of
/// the target's probabilities `p` over the draft's `q`. A residual with no
/// mass (the draft agreeing with the target) leaves the target's
/// distribution.
pub fn correction_log_probs(
    target_log_probs: &Array,
    draft_log_probs: &Array,
) -> Result<Array, Exception> {
    let residual = target_log_probs
        .exp()
        .subtract(&draft_log_probs.exp())
        .maximum(&Array::from_f32(0.0));
    let total = residual.sum_axis(-1, true);
    let total_value = total.item::<f32>();
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| Exception::custom(format!("speculative correction distribution: {e}")))?;
    if !total_value.is_finite() || total_value <= 1e-20 {
        return Ok(target_log_probs.clone());
    }
    Ok(residual.divide(&total).log())
}

/// Return `exp(log_probs[token])`, validating the token against the vocabulary.
pub fn token_probability_from_log_probs(log_probs: &Array, token: u32) -> Result<f32, Exception> {
    let vocab_size = log_probs.dim(-1);
    if token as i32 >= vocab_size {
        return Err(Exception::custom(format!(
            "token_probability_from_log_probs: token {token} out of range for vocab {vocab_size}"
        )));
    }
    let row = if log_probs.ndim() == 1 {
        log_probs.reshape(&[1, vocab_size])
    } else {
        log_probs.reshape(&[-1, vocab_size])
    };
    if row.dim(0) != 1 {
        return Err(Exception::custom(format!(
            "token_probability_from_log_probs expects one distribution row, got {}",
            row.dim(0)
        )));
    }
    let token_index = Array::from_slice(&[token as i32], &[1, 1]);
    let log_prob = take_along_axis(&row, &token_index, -1);
    Ok(log_prob.item::<f32>().exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The correction distribution is the clipped, renormalized residual of
    /// the target over the draft, and the target when there is none.
    #[test]
    fn the_correction_samples_the_residual_of_target_over_draft() {
        let log = |p: &[f32]| {
            Array::from_slice(
                &p.iter().map(|p| p.ln()).collect::<Vec<_>>(),
                &[1, p.len() as i32],
            )
        };
        let probs = |a: Array| {
            let a = a.exp();
            a.eval();
            a.as_slice::<f32>().to_vec()
        };
        let target = [0.5f32, 0.3, 0.2];
        // Residual (0.3, 0, 0.1) over 0.4.
        let got = probs(correction_log_probs(&log(&target), &log(&[0.2, 0.6, 0.1])).unwrap());
        for (g, w) in got.iter().zip([0.75f32, 0.0, 0.25]) {
            assert!((g - w).abs() < 1e-6, "{got:?}");
        }
        // A draft equal to the target leaves no residual: the target.
        let got = probs(correction_log_probs(&log(&target), &log(&target)).unwrap());
        for (g, w) in got.iter().zip(target) {
            assert!((g - w).abs() < 1e-6, "{got:?}");
        }
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    }

    #[test]
    fn test_speculative_decode_metrics_accumulate_verify_steps() {
        let mut metrics = SpeculativeDecodeMetrics::default();

        metrics.record_verify_step(3, 3);
        metrics.record_verify_step(3, 1);
        metrics.record_verify_step(2, 0);
        metrics.record_verify_step(0, 0);
        metrics.record_verify_step(2, 5);

        assert_eq!(metrics.drafted_tokens, 10);
        assert_eq!(metrics.accepted_draft_tokens, 6);
        assert_eq!(metrics.verify_steps, 4);
        assert_eq!(metrics.full_accept_steps, 2);
        assert_eq!(metrics.partial_accept_steps, 1);
        assert_eq!(metrics.zero_accept_steps, 1);
        assert_eq!(metrics.max_draft_tokens, 3);
        assert!((metrics.acceptance_rate() - 0.6).abs() < f32::EPSILON);
        assert!((metrics.avg_accepted_per_verify() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn token_probability_reads_one_row_and_rejects_out_of_range_tokens() {
        let log_probs = Array::from_slice(&[0.25f32.ln(), 0.75f32.ln()], &[2]);
        let p = token_probability_from_log_probs(&log_probs, 1).unwrap();
        assert!((p - 0.75).abs() < 1e-6);
        assert!(token_probability_from_log_probs(&log_probs, 2).is_err());
    }
}
