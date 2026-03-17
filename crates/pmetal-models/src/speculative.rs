//! Speculative decoding via layer-split draft/verify within a single model.
//!
//! This module implements the "self-speculative" paradigm: instead of loading
//! two separate models, one model is split at a chosen layer index so that
//! early layers act as a cheap draft and the full stack acts as the verifier.
//!
//! ## Algorithm
//!
//! Each `decode_step` performs:
//!
//! 1. **Draft phase** — Run embed + layers `0..split` on the current input.
//!    Project the partial hidden state through `lm_head` (weight-tying means
//!    the same projection applies) to get draft logits. Sample `num_draft_tokens`
//!    tokens greedily.
//!
//! 2. **Verify phase** — Concatenate the original input token with the draft
//!    tokens into a single sequence and run the *full* model through all layers
//!    (embed + `0..num_layers` + norm + lm_head). This produces `N+1` logit
//!    rows in one forward pass.
//!
//! 3. **Accept/reject** — For each position `i` in `1..=N`, compare the draft
//!    token with the verifier's argmax at position `i-1`. Accept all consecutive
//!    matches; on the first mismatch take the verifier's token and discard the
//!    rest. Also always keep the verifier's prediction at the last accepted
//!    position as a "bonus" token (this is the standard speculative-decoding
//!    correction step at temperature=0).
//!
//! 4. **Cache management** — The verify pass uses a *fresh* KV cache every step
//!    (the whole accepted prefix is fed from the caller's accumulated token
//!    history). This avoids the complexity of rolling back a partially-filled
//!    draft cache on rejection.
//!
//! ## Throughput
//!
//! For a well-matched draft (high acceptance rate), each step returns 2–N+1
//! tokens at roughly the cost of one full forward pass (the verify pass dominates).

use mlx_rs::{Array, error::Exception, ops::indexing::IndexOp};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};

use crate::shard::ShardableModel;

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to propose per step.
    ///
    /// Defaults to 2. Higher values increase throughput when the draft
    /// acceptance rate is high but add overhead when it is low.
    pub num_draft_tokens: usize,

    /// Layer index at which to split the model.
    ///
    /// Layers `0..split` are the draft; layers `split..num_layers` + norm +
    /// lm_head form the verifier. When `None` the split is auto-computed as
    /// `max(1, num_layers / 3)`.
    pub draft_layer_split: Option<usize>,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 2,
            draft_layer_split: None,
        }
    }
}

impl SpeculativeConfig {
    /// Create a config with explicit draft token count and auto-computed split.
    pub fn new(num_draft_tokens: usize) -> Self {
        Self {
            num_draft_tokens,
            draft_layer_split: None,
        }
    }

    /// Create a config with explicit split layer.
    pub fn with_split(mut self, split: usize) -> Self {
        self.draft_layer_split = Some(split);
        self
    }

    /// Resolve the actual split layer given the total number of model layers.
    pub fn resolve_split(&self, num_layers: usize) -> usize {
        self.draft_layer_split
            .unwrap_or_else(|| (num_layers / 3).max(1))
            .min(num_layers.saturating_sub(1))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Statistics
// ────────────────────────────────────────────────────────────────────────────

/// Cumulative statistics for a speculative decoding session.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total tokens emitted (accepted draft + correction tokens).
    pub total_tokens: usize,
    /// Total draft tokens proposed across all steps.
    pub total_draft_proposed: usize,
    /// Total draft tokens accepted (before the first mismatch).
    pub total_draft_accepted: usize,
    /// Number of decode steps executed.
    pub num_steps: usize,
}

impl SpeculativeStats {
    /// Fraction of draft tokens accepted over the session.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_proposed == 0 {
            0.0
        } else {
            self.total_draft_accepted as f32 / self.total_draft_proposed as f32
        }
    }

    /// Average number of tokens emitted per decode step.
    pub fn tokens_per_step(&self) -> f32 {
        if self.num_steps == 0 {
            0.0
        } else {
            self.total_tokens as f32 / self.num_steps as f32
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SpeculativeDecoder
// ────────────────────────────────────────────────────────────────────────────

/// Layer-split speculative decoder.
///
/// Type parameter `M` must implement [`ShardableModel`], which exposes the
/// per-layer forward pass, embed, normalize, and lm_head as separate calls.
pub struct SpeculativeDecoder<M: ShardableModel> {
    /// The underlying model.
    model: M,
    /// Speculative configuration.
    config: SpeculativeConfig,
    /// Resolved split layer (computed once in `new`).
    split_layer: usize,
    /// KV cache used for the verify pass (cleared each step).
    verify_cache: KVCache,
    /// Accumulated statistics.
    stats: SpeculativeStats,
}

impl<M: ShardableModel> SpeculativeDecoder<M> {
    /// Create a new speculative decoder.
    ///
    /// `cache_config` must be sized for the full model (`num_layers` ==
    /// the total layer count, `max_seq_len` large enough for the prompt +
    /// `num_draft_tokens`).
    pub fn new(model: M, config: SpeculativeConfig, cache_config: KVCacheConfig) -> Self {
        let split_layer = config.resolve_split(model.num_layers());
        let verify_cache = KVCache::new(cache_config);
        Self {
            model,
            config,
            split_layer,
            verify_cache,
            stats: SpeculativeStats::default(),
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    /// Access accumulated statistics.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset accumulated statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Resolved split layer index.
    pub fn split_layer(&self) -> usize {
        self.split_layer
    }

    // ────────────────────────────────────────────────────────────────────────
    // Public decode step
    // ────────────────────────────────────────────────────────────────────────

    /// Perform one draft+verify cycle and return accepted token IDs.
    ///
    /// `input_ids` is the *full* prefix token sequence (batch=1, shape
    /// `[1, seq_len]`). This is passed to the verify forward pass, so the
    /// caller must maintain the token history.
    ///
    /// Returns 1..=`num_draft_tokens + 1` token IDs per call:
    /// - All consecutive draft tokens that the verifier agrees with, plus
    /// - A correction/bonus token from the verifier at the last accepted
    ///   position (or the verifier's choice at the first mismatch).
    ///
    /// On error the verify cache is reset to avoid stale state.
    pub fn decode_step(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Vec<u32>, Exception> {
        // ── 1. Draft phase ──────────────────────────────────────────────────
        let draft_tokens = self.draft_phase(input_ids, mask)?;

        // ── 2. Build verify input: [input_ids | draft_tokens] ───────────────
        let verify_ids = build_verify_input(input_ids, &draft_tokens)?;

        // ── 3. Verify phase ─────────────────────────────────────────────────
        self.verify_cache.reset();
        let verify_logits = self.verify_phase(&verify_ids, mask);
        if let Err(e) = verify_logits {
            self.verify_cache.reset();
            return Err(e);
        }
        let verify_logits = verify_logits.unwrap();
        // verify_logits: [1, seq_len_verify, vocab_size]
        verify_logits.eval()?;

        // ── 4. Accept/reject ─────────────────────────────────────────────────
        let accepted = accept_reject(&verify_logits, &draft_tokens)?;

        // ── 5. Update statistics ─────────────────────────────────────────────
        self.stats.total_draft_proposed += draft_tokens.len();
        // accepted contains the draft matches + 1 correction, so draft
        // accepted = accepted.len() - 1 (or 0 if len==1 and there were drafts)
        let n_accepted_draft = accepted.len().saturating_sub(1).min(draft_tokens.len());
        self.stats.total_draft_accepted += n_accepted_draft;
        self.stats.total_tokens += accepted.len();
        self.stats.num_steps += 1;

        Ok(accepted)
    }

    // ────────────────────────────────────────────────────────────────────────
    // Internal: draft phase
    // ────────────────────────────────────────────────────────────────────────

    /// Run embed + draft layers and sample `num_draft_tokens` greedily.
    fn draft_phase(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Vec<u32>, Exception> {
        let num_draft = self.config.num_draft_tokens;
        let split = self.split_layer;
        let mut draft_tokens: Vec<u32> = Vec::with_capacity(num_draft);

        // We use a throw-away KV cache for the draft pass — we only care about
        // the sampled token IDs, not the cached activations.
        let config = self.verify_cache.config().clone();
        let mut draft_cache = KVCache::new(config);

        // Initial hidden state from the full input prefix through draft layers.
        let mut hidden = self.model.embed(input_ids)?;
        hidden = self
            .model
            .apply_layer_range(0..split, &hidden, mask, &mut draft_cache)?;

        // Project partial hidden state to logits using lm_head (weight-tied).
        // Take only the last token position: hidden[..., -1:, ...]
        let last_hidden = last_token_hidden(&hidden)?;
        let norm_last = self.model.normalize(&last_hidden)?;
        let draft_logits = self.model.lm_head(&norm_last)?;

        // Sample first draft token.
        draft_logits.eval()?;
        let first_token = argmax_last(&draft_logits)?;
        draft_tokens.push(first_token);

        // Autoregressively sample remaining draft tokens through draft layers.
        for _ in 1..num_draft {
            let next_input = Array::from_slice(&[*draft_tokens.last().unwrap() as i32], &[1, 1]);
            let h = self.model.embed(&next_input)?;
            let h = self
                .model
                .apply_layer_range(0..split, &h, mask, &mut draft_cache)?;
            let h_last = last_token_hidden(&h)?;
            let h_norm = self.model.normalize(&h_last)?;
            let logits = self.model.lm_head(&h_norm)?;
            logits.eval()?;
            draft_tokens.push(argmax_last(&logits)?);
        }

        Ok(draft_tokens)
    }

    // ────────────────────────────────────────────────────────────────────────
    // Internal: verify phase
    // ────────────────────────────────────────────────────────────────────────

    /// Run the full model on the verify sequence and return logits for all
    /// `N+1` positions (the original last token + N draft tokens).
    fn verify_phase(
        &mut self,
        verify_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let num_layers = self.model.num_layers();
        let mut hidden = self.model.embed(verify_ids)?;
        hidden =
            self.model
                .apply_layer_range(0..num_layers, &hidden, mask, &mut self.verify_cache)?;
        let normed = self.model.normalize(&hidden)?;
        self.model.lm_head(&normed)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Accept/reject logic
// ────────────────────────────────────────────────────────────────────────────

/// Compare draft tokens against the verifier's argmax predictions.
///
/// The verify logits cover positions `0..N+1` where position `i` is the
/// verifier's prediction for what follows token `i` in the verify input.
///
/// - Position 0 corresponds to the last token of the original input; its
///   argmax is the verifier's greedy prediction for the *next* token.
/// - Positions `1..=N` correspond to the N draft tokens; their argmaxes are
///   what the verifier predicts after each draft token.
///
/// We accept draft token at position `i` if `verify_argmax[i-1] == draft[i-1]`.
/// On acceptance, we also take `verify_argmax[i]` as the next prediction.
/// On the first rejection at position `j`, we take `verify_argmax[j-1]`
/// (the verifier's corrected token) and stop.
///
/// Returns `Vec<u32>` of length 1..=N+1.
fn accept_reject(verify_logits: &Array, draft_tokens: &[u32]) -> Result<Vec<u32>, Exception> {
    let n = draft_tokens.len();
    // verify_logits: [1, N+1+original_prefix, vocab_size]
    // We only care about the last N+1 positions.
    let seq_len = verify_logits.dim(1) as usize;
    let verify_start = seq_len.saturating_sub(n + 1);

    let mut accepted: Vec<u32> = Vec::with_capacity(n + 1);

    // Verify each draft token against the verifier's argmax at that position.
    for (i, &draft_tok) in draft_tokens.iter().enumerate() {
        let pos = (verify_start + i) as i32;
        let logit_row = verify_logits.index((0i32, pos, ..));
        logit_row.eval()?;
        let verifier_token = argmax_1d(&logit_row)?;

        accepted.push(verifier_token);
        if verifier_token != draft_tok {
            // Mismatch: take the verifier's correction and stop.
            return Ok(accepted);
        }
    }

    // All draft tokens accepted — emit a bonus token from the verifier's
    // prediction at the position after the last draft token.
    let bonus_pos = (verify_start + n) as i32;
    let bonus_row = verify_logits.index((0i32, bonus_pos, ..));
    bonus_row.eval()?;
    accepted.push(argmax_1d(&bonus_row)?);

    Ok(accepted)
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Concatenate `input_ids` with `draft_tokens` along the sequence dimension.
///
/// `input_ids`: `[1, seq_len]`; produces `[1, seq_len + N]`.
fn build_verify_input(input_ids: &Array, draft_tokens: &[u32]) -> Result<Array, Exception> {
    if draft_tokens.is_empty() {
        return Ok(input_ids.clone());
    }
    let draft_i32: Vec<i32> = draft_tokens.iter().map(|&t| t as i32).collect();
    let draft_arr = Array::from_slice(&draft_i32, &[1, draft_tokens.len() as i32]);
    mlx_rs::ops::concatenate_axis(&[input_ids, &draft_arr], 1)
}

/// Extract the last token position from hidden states.
///
/// `hidden`: `[batch, seq_len, hidden_dim]` → `[batch, 1, hidden_dim]`
fn last_token_hidden(hidden: &Array) -> Result<Array, Exception> {
    let seq_len = hidden.dim(1) as i32;
    Ok(hidden.index((.., seq_len - 1..seq_len, ..)))
}

/// Greedy argmax for a logits tensor of shape `[1, 1, vocab_size]` or
/// `[1, seq_len, vocab_size]` — returns the argmax at the *last* position.
fn argmax_last(logits: &Array) -> Result<u32, Exception> {
    // logits: [batch=1, seq_len, vocab_size]
    let seq_len = logits.dim(1) as i32;
    let row = logits.index((0i32, seq_len - 1, ..));
    argmax_1d(&row)
}

/// Greedy argmax for a 1-D logits vector `[vocab_size]`.
fn argmax_1d(row: &Array) -> Result<u32, Exception> {
    use mlx_rs::ops::indexing::argmax;
    let idx = argmax(row, None)?;
    idx.eval()?;
    Ok(idx.item::<u32>())
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_auto_split() {
        let cfg = SpeculativeConfig::new(3);
        // 32 layers → split at 32/3 = 10
        assert_eq!(cfg.resolve_split(32), 10);
        // 3 layers → split at 3/3 = 1, capped at num_layers-1 = 2, so 1
        assert_eq!(cfg.resolve_split(3), 1);
        // With only 1 layer the split must be 0 (can't dedicate any layer as
        // draft while keeping at least one for the verifier).
        assert_eq!(cfg.resolve_split(1), 0);
    }

    #[test]
    fn config_explicit_split() {
        let cfg = SpeculativeConfig::new(3).with_split(8);
        assert_eq!(cfg.resolve_split(32), 8);
    }

    #[test]
    fn config_split_capped_at_num_layers_minus_one() {
        let cfg = SpeculativeConfig::new(2).with_split(100);
        // With 10 layers, split cannot exceed 9.
        assert_eq!(cfg.resolve_split(10), 9);
    }

    #[test]
    fn stats_acceptance_rate_zero() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.acceptance_rate(), 0.0);
    }

    #[test]
    fn stats_tokens_per_step() {
        let mut stats = SpeculativeStats::default();
        stats.total_tokens = 15;
        stats.num_steps = 5;
        assert!((stats.tokens_per_step() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn stats_acceptance_rate_full() {
        let mut stats = SpeculativeStats::default();
        stats.total_draft_proposed = 100;
        stats.total_draft_accepted = 75;
        assert!((stats.acceptance_rate() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn build_verify_input_empty_draft() {
        let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let out = build_verify_input(&ids, &[]).unwrap();
        assert_eq!(out.dim(1), 3);
    }

    #[test]
    fn build_verify_input_with_draft() {
        let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let out = build_verify_input(&ids, &[4u32, 5]).unwrap();
        assert_eq!(out.dim(1), 5);
        out.eval().unwrap();
        let data: &[i32] = out.as_slice();
        assert_eq!(data, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn accept_reject_all_match() {
        // Logits for 3 positions, vocab_size=4.
        // Position 0 → token 1; position 1 → token 2; position 2 → token 3 (bonus).
        #[rustfmt::skip]
        let data = [
            // pos 0: max at index 1
            -1.0f32, 5.0, -1.0, -1.0,
            // pos 1: max at index 2
            -1.0, -1.0, 5.0, -1.0,
            // pos 2: max at index 3 (bonus)
            -1.0, -1.0, -1.0, 5.0,
        ];
        let logits = Array::from_slice(&data, &[1, 3, 4]);
        // Draft proposed [1, 2] and verify agrees.
        let accepted = accept_reject(&logits, &[1u32, 2]).unwrap();
        assert_eq!(accepted, vec![1u32, 2, 3]);
    }

    #[test]
    fn accept_reject_first_mismatch() {
        // Position 0 argmax → 9 (not 1), so first draft token is rejected.
        #[rustfmt::skip]
        let data = [
            // pos 0: max at index 9
            -1.0f32, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 5.0,
            // pos 1 (never reached): max at 0
             5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ];
        let logits = Array::from_slice(&data, &[1, 2, 10]);
        let accepted = accept_reject(&logits, &[1u32, 2]).unwrap();
        // First mismatch at i=0: verifier chose 9, draft was 1.
        assert_eq!(accepted, vec![9u32]);
    }

    #[test]
    fn accept_reject_partial_match() {
        // vocab_size = 4
        // pos 0 argmax = 1 (draft[0]=1 ✓)
        // pos 1 argmax = 9 → but vocab is 4, so use small vocab
        // Use vocab_size = 10 so token 9 is valid.
        #[rustfmt::skip]
        let data = [
            // pos 0: max at 1 → matches draft[0]=1
            -1.0f32, 5.0, -1.0, -1.0,
            // -1 rest
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,

            // pos 1: max at 3 → does NOT match draft[1]=2
            -1.0, -1.0, -1.0, 5.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,

            // pos 2: bonus (never reached since mismatch at pos 1)
            5.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ];
        let logits = Array::from_slice(&data, &[1, 3, 10]);
        let accepted = accept_reject(&logits, &[1u32, 2]).unwrap();
        // accept draft[0]=1, then mismatch → verifier token 3
        assert_eq!(accepted, vec![1u32, 3]);
    }
}
