//! Speculative decoding emits the target's own distribution.
//!
//! The target and the draft here are Markov chains over a small vocabulary:
//! the logits after a token are a fixed row of a table, so the target's exact
//! distribution over any run of tokens is a product of rows and a test can
//! compare what the decoder draws against it, instead of against another
//! sampler run. The draft is a different chain from the target (so drafts are
//! rejected, and the correction distribution is exercised), over the same
//! vocabulary.
//!
//! * Greedy decoding is token for token the target's argmax chain, whatever
//!   the draft says, including where a stop token ends the run.
//! * Sampled decoding draws every position from the target's distribution:
//!   the empirical distribution over the first tokens of thousands of seeded
//!   runs matches the exact one, and does not match the draft's.
//! * A seeded run is reproducible.

use pmetal_models::generation::{GenerationConfig, GenerationOutput};
use pmetal_models::mtp::{MtpDrafter, MtpTarget, generate_mtp_streaming};
use pmetal_runtime::compat::{Array, Exception};
use serial_test::serial;

const VOCAB: usize = 5;

/// Row `token` of a Markov table: the logits of the next token. `salt`
/// makes the target's table and the draft's different chains.
fn logits_after(token: u32, salt: u32) -> [f32; VOCAB] {
    let mut row = [0.0; VOCAB];
    for (j, slot) in row.iter_mut().enumerate() {
        let h = (token * 31 + j as u32 * 17 + salt * 7 + token * j as u32 * 5) % 23;
        *slot = h as f32 * 0.18;
    }
    row
}

const TARGET_SALT: u32 = 1;
const DRAFT_SALT: u32 = 9;

fn probabilities(logits: &[f32; VOCAB]) -> [f64; VOCAB] {
    let max = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&l| ((l - max) as f64).exp()).collect();
    let total: f64 = exps.iter().sum();
    let mut p = [0.0; VOCAB];
    for (slot, e) in p.iter_mut().zip(&exps) {
        *slot = e / total;
    }
    p
}

fn greedy_after(token: u32, salt: u32) -> u32 {
    let row = logits_after(token, salt);
    (0..VOCAB).fold(0, |best, j| if row[j] > row[best] { j } else { best }) as u32
}

fn rows(tokens: &[u32], salt: u32) -> Array {
    let data: Vec<f32> = tokens.iter().flat_map(|&t| logits_after(t, salt)).collect();
    Array::from_slice(&data, &[1, tokens.len() as i32, VOCAB as i32])
}

/// The target: logits at each position are the table row of the token there.
struct TableTarget {
    seq: Vec<u32>,
    last_block: usize,
}

impl TableTarget {
    fn new() -> Self {
        Self {
            seq: Vec::new(),
            last_block: 0,
        }
    }
}

impl MtpTarget for TableTarget {
    fn reset(&mut self) -> Result<(), Exception> {
        self.seq.clear();
        Ok(())
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<(Array, Array), Exception> {
        self.seq.extend_from_slice(tokens);
        self.last_block = tokens.len();
        let hidden = Array::zeros_f32(&[1, tokens.len() as i32, 4]);
        Ok((hidden, rows(tokens, TARGET_SALT)))
    }

    fn keep(&mut self, kept: usize) -> Result<(), Exception> {
        let drop = self.last_block - kept;
        self.seq.truncate(self.seq.len() - drop);
        self.last_block = kept;
        Ok(())
    }
}

/// The draft: a different Markov chain, sampling from its own logits.
struct TableDrafter;

impl MtpDrafter<TableTarget> for TableDrafter {
    fn prefill(&mut self, _: &mut TableTarget, _: &[u32], _: &Array) -> Result<(), Exception> {
        Ok(())
    }

    fn step(
        &mut self,
        _: &mut TableTarget,
        token: u32,
        _: &Array,
        _: usize,
        _: usize,
    ) -> Result<(Array, Array), Exception> {
        Ok((
            Array::zeros_f32(&[1, 1, 4]),
            rows(&[token], DRAFT_SALT).reshape(&[1, VOCAB as i32]),
        ))
    }

    fn commit(
        &mut self,
        _: &mut TableTarget,
        _: u32,
        _: &Array,
        _: &[u32],
        _: usize,
        _: &Array,
    ) -> Result<(), Exception> {
        Ok(())
    }
}

const PROMPT: [u32; 3] = [2, 0, 3];

fn decode(config: &GenerationConfig, budget: usize) -> GenerationOutput {
    let out = generate_mtp_streaming(
        &mut TableTarget::new(),
        &mut TableDrafter,
        &PROMPT,
        config,
        budget,
        |_| true,
    )
    .expect("decode");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    out
}

#[test]
#[serial]
fn greedy_speculative_decoding_is_the_targets_argmax_chain() {
    let mut want = PROMPT.to_vec();
    for _ in 0..16 {
        want.push(greedy_after(*want.last().unwrap(), TARGET_SALT));
    }
    // The chains differ, so some drafts are wrong; the draft's own argmax
    // chain is not the target's.
    let mut draft_chain = PROMPT.to_vec();
    for _ in 0..16 {
        draft_chain.push(greedy_after(*draft_chain.last().unwrap(), DRAFT_SALT));
    }
    assert_ne!(want, draft_chain, "the draft must disagree with the target");

    for budget in [1, 2, 4] {
        let out = decode(&GenerationConfig::greedy(16), budget);
        assert_eq!(out.token_ids, want, "budget {budget}");
        assert_eq!(out.num_generated, 16);
    }
}

/// A stop token ends the run and is left out, as it is for plain generation;
/// and no callback sees it.
#[test]
#[serial]
fn a_stop_token_ends_a_speculative_run_and_is_not_emitted() {
    let mut chain = PROMPT.to_vec();
    for _ in 0..16 {
        chain.push(greedy_after(*chain.last().unwrap(), TARGET_SALT));
    }
    let made = &chain[PROMPT.len()..];
    // The first token that has already appeared before it, so the run goes
    // a few tokens before stopping.
    let at = (1..made.len())
        .find(|&i| made[..i].contains(&made[i]) && made[i] != made[i - 1])
        .or_else(|| (3..made.len()).next())
        .unwrap();
    let stop = made[at];
    let first = made.iter().position(|&t| t == stop).unwrap();

    for budget in [1, 3, 5] {
        let mut seen = Vec::new();
        let out = generate_mtp_streaming(
            &mut TableTarget::new(),
            &mut TableDrafter,
            &PROMPT,
            &GenerationConfig::greedy(16).with_stop_tokens(vec![stop]),
            budget,
            |t| {
                seen.push(t);
                true
            },
        )
        .expect("decode");
        assert_eq!(
            out.token_ids,
            chain[..PROMPT.len() + first],
            "budget {budget}"
        );
        assert_eq!(out.num_generated, first, "budget {budget}");
        assert_eq!(seen, made[..first], "budget {budget}: streamed tokens");
        assert!(
            out.stopped_by_token && !out.stopped_by_length,
            "budget {budget}"
        );
    }
}

/// The exact distribution of the first `n` tokens after the prompt under the
/// target chain: `(tokens, probability)` for every outcome.
fn exact_joint(n: usize) -> Vec<(Vec<u32>, f64)> {
    let mut outcomes = vec![(Vec::<u32>::new(), 1.0f64)];
    for _ in 0..n {
        let mut next = Vec::new();
        for (tokens, p) in &outcomes {
            let last = *tokens.last().unwrap_or(PROMPT.last().unwrap());
            let row = probabilities(&logits_after(last, TARGET_SALT));
            for (t, &q) in row.iter().enumerate() {
                let mut extended = tokens.clone();
                extended.push(t as u32);
                next.push((extended, p * q));
            }
        }
        outcomes = next;
    }
    outcomes
}

/// Sampled speculative decoding draws from the target's distribution, not
/// the draft's: the first three tokens of seeded runs are distributed as the
/// target chain says (whole-sequence frequencies against the exact joint).
fn check_distribution(config: &GenerationConfig, budget: usize, runs: u64) {
    let n = 3;
    let want = exact_joint(n);
    let mut counts = vec![0u64; want.len()];
    let mut drafted = 0;
    let mut accepted = 0;
    for seed in 0..runs {
        let out = decode(&config.clone().with_seed(seed), budget);
        let tokens = &out.token_ids[PROMPT.len()..PROMPT.len() + n];
        let cell = tokens.iter().fold(0usize, |a, &t| a * VOCAB + t as usize);
        counts[cell] += 1;
        let metrics = out.speculative_metrics.expect("metrics");
        drafted += metrics.drafted_tokens;
        accepted += metrics.accepted_draft_tokens;
    }
    // Some drafts were rejected and some accepted, so the acceptance test
    // and the correction both ran.
    assert!(
        accepted > 0 && accepted < drafted,
        "{accepted} of {drafted}"
    );

    // Pearson's chi-square over the cells expecting at least 5 counts, the
    // rest pooled. 125 cells: the 99.99th percentile of a chi-square with
    // about 125 degrees of freedom is near 190; a draft's distribution
    // lands far beyond it.
    let (mut chi, mut dof) = (0.0f64, 0usize);
    let (mut pooled_expected, mut pooled_seen) = (0.0f64, 0.0f64);
    for ((_, p), &seen) in want.iter().zip(&counts) {
        let expected = p * runs as f64;
        if expected >= 5.0 {
            chi += (seen as f64 - expected).powi(2) / expected;
            dof += 1;
        } else {
            pooled_expected += expected;
            pooled_seen += seen as f64;
        }
    }
    if pooled_expected >= 5.0 {
        chi += (pooled_seen - pooled_expected).powi(2) / pooled_expected;
        dof += 1;
    }
    let limit = dof as f64 + 5.0 * (2.0 * dof as f64).sqrt();
    assert!(
        chi < limit,
        "budget {budget}: chi-square {chi:.1} over {dof} cells exceeds {limit:.1}"
    );
}

#[test]
#[serial]
fn sampled_speculative_decoding_draws_the_targets_distribution() {
    let config = GenerationConfig::sampling(3, 1.0)
        .with_top_k(0)
        .with_top_p(1.0)
        .with_min_p(0.0)
        .with_repetition_penalty(1.0);
    for budget in [1, 3] {
        check_distribution(&config, budget, 6000);
    }
}

/// The filters shape the target's distribution before the verification, and
/// the draft is held to it: top-k 3 at a temperature of 0.8.
#[test]
#[serial]
fn filtered_speculative_sampling_draws_the_filtered_distribution() {
    let config = GenerationConfig::sampling(3, 0.8)
        .with_top_k(3)
        .with_top_p(1.0)
        .with_min_p(0.0)
        .with_repetition_penalty(1.0);
    let n = 3;
    let runs = 4000u64;
    // The exact filtered chain: top-3 of the temperature-scaled row.
    let filtered = |last: u32| {
        let mut row = logits_after(last, TARGET_SALT);
        for l in row.iter_mut() {
            *l /= 0.8;
        }
        let mut order: Vec<usize> = (0..VOCAB).collect();
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        let kth = row[order[2]];
        let mut kept = row;
        for l in kept.iter_mut() {
            if *l < kth {
                *l = f32::NEG_INFINITY;
            }
        }
        probabilities(&kept)
    };
    let mut joint = vec![(Vec::<u32>::new(), 1.0f64)];
    for _ in 0..n {
        let mut next = Vec::new();
        for (tokens, p) in &joint {
            let last = *tokens.last().unwrap_or(PROMPT.last().unwrap());
            for (t, &q) in filtered(last).iter().enumerate() {
                if q > 0.0 {
                    let mut extended = tokens.clone();
                    extended.push(t as u32);
                    next.push((extended, p * q));
                }
            }
        }
        joint = next;
    }
    let mut counts = std::collections::HashMap::<Vec<u32>, u64>::new();
    for seed in 0..runs {
        let out = decode(&config.clone().with_seed(seed), 3);
        let tokens = out.token_ids[PROMPT.len()..PROMPT.len() + n].to_vec();
        assert!(
            joint.iter().any(|(t, _)| *t == tokens),
            "{tokens:?} is outside the filtered support"
        );
        *counts.entry(tokens).or_default() += 1;
    }
    let (mut chi, mut dof) = (0.0f64, 0usize);
    let (mut pooled_expected, mut pooled_seen) = (0.0f64, 0.0f64);
    for (tokens, p) in &joint {
        let expected = p * runs as f64;
        let seen = *counts.get(tokens).unwrap_or(&0) as f64;
        if expected >= 5.0 {
            chi += (seen - expected).powi(2) / expected;
            dof += 1;
        } else {
            pooled_expected += expected;
            pooled_seen += seen;
        }
    }
    if pooled_expected >= 5.0 {
        chi += (pooled_seen - pooled_expected).powi(2) / pooled_expected;
        dof += 1;
    }
    let limit = dof as f64 + 5.0 * (2.0 * dof as f64).sqrt();
    assert!(
        chi < limit,
        "chi-square {chi:.1} over {dof} cells exceeds {limit:.1}"
    );
}

#[test]
#[serial]
fn a_seeded_speculative_run_repeats() {
    let config = GenerationConfig::sampling(12, 1.0).with_seed(77);
    let first = decode(&config, 3);
    for _ in 0..3 {
        assert_eq!(decode(&config, 3).token_ids, first.token_ids);
    }
    assert_ne!(
        decode(&GenerationConfig::sampling(12, 1.0).with_seed(78), 3).token_ids,
        first.token_ids
    );
}
