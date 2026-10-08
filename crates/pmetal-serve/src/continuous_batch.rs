//! Continuous-batching scheduler skeleton.
//!
//! This module provides the *scheduling* half of continuous batching:
//! a slot pool with FIFO queuing, lifecycle tracking, and a step driver
//! that tells the caller which slots to prefill / decode on each tick.
//! It does **not** touch the model or KV cache yet — that plumbing lands
//! in phase D.2 alongside batched `forward_with_hybrid_cache`.
//!
//! # Lifecycle
//!
//! ```text
//!   enqueue(req)
//!         │
//!         ▼
//!  [Pending] ──assign──▶ [Prefilling] ──prefilled──▶ [Decoding]
//!                                                        │
//!                                     stop/EOS/cancel    │
//!                                                        ▼
//!                                                 [Finished|Cancelled]
//! ```
//!
//! A request enters as `Pending` and waits until a slot is free. When a
//! slot opens up the scheduler moves the request into `Prefilling`,
//! processes its prompt, then flips it to `Decoding` so the next batched
//! forward pass samples one token per slot. Finished or cancelled slots
//! are retired and their resources released for the next `Pending`
//! request.
//!
//! # Token-block scheduling
//!
//! Admission is constrained by both slot count and a paged-KV-style
//! token-block budget. Each request reserves
//! `ceil((prompt_tokens + max_new_tokens) / block_size)` blocks while it is
//! active, clamped to the full block budget so an oversized request can still
//! run alone. The scheduler scans the pending queue for the first request that
//! fits the remaining block budget, which avoids head-of-line blocking when a
//! large request is waiting ahead of a smaller one.
//!
//! # Prefill policy (for D.1)
//!
//! At most one slot is in the `Prefilling` state at any time. This
//! is the simplest batching pattern: decodes run
//! batched, prefills run serially. A later phase can relax this to
//! chunked interleaved prefill once the batched forward pass is in
//! place.
//!
//! # Decode path selection
//!
//! Phase 2 introduces a **fused `[N_active, 1]` batched decode** built on
//! `pmetal_mlx::kv_cache::FusedBatchKVCache` and per-arch
//! `forward_batched_impl` methods. The driver routes a request through the
//! fused path when
//! [`pmetal_models::dispatcher::DynamicModel::supports_fused_batched`]
//! returns `true`; otherwise it falls back to the per-slot serial loop
//! shipped in Phase 1.
//!
//! Currently fused-capable: Llama, Mistral (no sliding window), Qwen2/3
//! (no sliding window), Qwen3-MoE, GPT-OSS (interleaved sliding-window
//! supported via per-layer mask overlay), Gemma1, Gemma2 / Gemma3 (4-norm
//! peri-norm + per-layer sliding + attn logit softcap), Phi/Phi4 (partial
//! RoPE; SuRoPE and sliding-window configs deferred), Cohere (parallel
//! decoder block; non-global sliding-window configs deferred), Granite
//! (plain `granite` with a unit `residual_multiplier`, which no release has).
//!
//! Currently on the serial fallback (no `forward_batched_impl` yet, see
//! `dispatcher::supports_fused_batched`):
//!
//! - Granite MoE and hybrid families (`granitemoe`, `granitemoeshared`,
//!   `granitemoehybrid`): routed experts, the shared MLP and Mamba-2 state
//!   have no fused path.
//! - Bespoke attention: Llama4 (MoD per-token skip), DeepSeek (MLA
//!   compressed latents).
//!
//! # Hybrid models
//!
//! Models with recurrent layers (Qwen 3.5 / 3.6 / 3.8 and Qwen4Exp
//! gated-delta-net, NemotronH and Granite 4.0-H Mamba2) take the serial path, each slot
//! with its own recurrent state beside its KV cache
//! ([`crate::continuous_driver::ContinuousEngineState::with_recurrent_state`]),
//! reset when the slot is freed. They keep no prefix cache, since a KV
//! snapshot can't restore recurrent state. The fused `[N, 1]` decode
//! carries no recurrent state and refuses them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Opaque identifier for a request in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(pub u64);

impl SlotId {
    fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        SlotId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Sampling + stopping config attached to a slot.
///
/// Kept decoupled from `pmetal_models::generation::GenerationConfig` so
/// this module doesn't drag in model-side types. The engine translates
/// between the two.
#[derive(Debug, Clone)]
pub struct SlotParams {
    pub max_new_tokens: usize,
    pub stop_tokens: Vec<u32>,
    pub stop_sequences: Vec<String>,
    /// Chunked prefill size (tokens per forward during the Prefilling phase).
    pub prefill_step_size: usize,
    /// When `Some(n)`, the driver emits per-token log-probabilities
    /// alongside the sampled token — `n == 0` means the chosen-token
    /// logprob only; `n > 0` includes the top-`n` alternatives. `None`
    /// (default) skips the logprob computation entirely.
    pub logprobs_top_n: Option<usize>,
}

/// One slot in the batch.
pub struct Slot {
    pub id: SlotId,
    pub state: SlotState,
    pub prompt: Vec<u32>,
    pub generated: Vec<u32>,
    pub params: SlotParams,
    /// Index into `prompt` marking how many prompt tokens have been
    /// prefilled so far. Advances during chunked prefill.
    pub prefilled: usize,
    /// Epoch-style counter for ordering ticks deterministically.
    pub enqueued_at: Instant,
    /// Final reason, populated only after transition to
    /// `SlotState::Finished` or `SlotState::Cancelled`.
    pub finish_reason: Option<FinishReason>,
    /// Reserved token blocks for this request while the slot is active.
    pub reserved_blocks: usize,
}

/// State machine for a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Waiting for a slot to open up.
    Pending,
    /// Prompt is being processed (may span multiple steps).
    Prefilling,
    /// Prompt done; per-step batched decode is producing completion tokens.
    Decoding,
    /// Terminated normally (stop token / EOS / length limit).
    Finished,
    /// Terminated by the caller (client disconnect, shutdown, etc).
    Cancelled,
}

/// Terminal outcome of a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit `max_new_tokens`.
    Length,
    /// Emitted a stop token.
    Stop,
    /// Hit a stop sequence in decoded text.
    StopSequence,
    /// Client cancelled or scheduler retired.
    Cancelled,
}

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum number of concurrently Decoding or Prefilling slots.
    pub max_slots: usize,
    /// Upper bound on the pending queue — new requests are rejected with
    /// [`EnqueueError::Saturated`] when this is exceeded.
    pub max_queue_depth: usize,
    /// Token block size used by the paged-KV admission policy.
    pub block_size: usize,
    /// Maximum active token blocks. `0` disables the block-budget gate.
    pub max_blocks: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        // 8 slots is a conservative default for Apple silicon — KV cache
        // memory scales linearly with batch size, and large models
        // already sit close to the memory ceiling at batch 1.
        Self {
            max_slots: 8,
            max_queue_depth: 256,
            block_size: 32,
            max_blocks: 0,
        }
    }
}

impl BatcherConfig {
    /// Effective block size after clamping invalid input.
    pub fn effective_block_size(&self) -> usize {
        self.block_size.max(1)
    }

    /// Reserve enough blocks for a request's worst-case active context.
    pub fn reserved_blocks_for(&self, tokens: usize) -> usize {
        if self.max_blocks == 0 {
            return 0;
        }
        tokens
            .max(1)
            .div_ceil(self.effective_block_size())
            .min(self.max_blocks)
    }
}

/// What the driver should do in the next step.
#[derive(Debug, Clone)]
pub enum StepInstruction {
    /// No work: all slots idle and no pending requests.
    Idle,
    /// One slot is in `Prefilling` — run prefill chunk on it.
    Prefill {
        slot: SlotId,
        /// The exact token slice to feed to forward() this tick.
        chunk: Vec<u32>,
        /// Whether this is the final chunk (so the scheduler can
        /// transition the slot to `Decoding` after this step).
        final_chunk: bool,
    },
    /// One or more slots ready for a batched decode step — forward once
    /// with `[N, 1]` tokens, one per slot.
    Decode { slots: Vec<SlotId> },
}

#[derive(Debug, thiserror::Error)]
pub enum EnqueueError {
    #[error("queue saturated (max_queue_depth reached)")]
    Saturated,
    #[error("empty prompt")]
    EmptyPrompt,
    /// The request's event channel can't hold every event it may produce.
    /// The scheduler never waits on one slow reader, since that would stall
    /// every other slot, so a full channel would lose tokens.
    #[error("event channel holds {capacity} events; the request can produce {needed}")]
    ChannelTooSmall { capacity: usize, needed: usize },
}

/// Pending request, queued but not yet assigned to a slot.
struct PendingRequest {
    id: SlotId,
    prompt: Vec<u32>,
    params: SlotParams,
    enqueued_at: Instant,
    reserved_blocks: usize,
}

/// The continuous batching scheduler.
///
/// Tick the scheduler in a loop: `next_instruction()` tells you what to
/// do, then you call `advance_prefill` or `advance_decode` back with the
/// sampled tokens to update slot state. Finished slots are moved to an
/// outbox the caller drains via `take_finished()`.
pub struct ContinuousBatcher {
    config: BatcherConfig,
    slots: Vec<Slot>,
    pending: VecDeque<PendingRequest>,
}

impl ContinuousBatcher {
    /// Create a new batcher with the given config.
    pub fn new(config: BatcherConfig) -> Self {
        Self {
            config,
            slots: Vec::new(),
            pending: VecDeque::new(),
        }
    }

    /// Config snapshot.
    pub fn config(&self) -> &BatcherConfig {
        &self.config
    }

    /// Number of currently active slots (Prefilling or Decoding).
    pub fn active_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s.state, SlotState::Prefilling | SlotState::Decoding))
            .count()
    }

    /// Depth of the pending queue.
    pub fn pending_depth(&self) -> usize {
        self.pending.len()
    }

    /// Enqueue a new request. Returns its `SlotId`; the caller uses this
    /// to correlate generated tokens and the final outcome.
    pub fn enqueue(
        &mut self,
        prompt: Vec<u32>,
        params: SlotParams,
    ) -> Result<SlotId, EnqueueError> {
        let reserved_tokens = prompt.len().saturating_add(params.max_new_tokens);
        self.enqueue_with_reserved_tokens(prompt, params, reserved_tokens)
    }

    /// Enqueue with an explicit active-context reservation. Serving uses this
    /// when prefix reuse shortens the prefill suffix but the active KV cache
    /// still contains the full prompt prefix.
    pub fn enqueue_with_reserved_tokens(
        &mut self,
        prompt: Vec<u32>,
        params: SlotParams,
        reserved_context_tokens: usize,
    ) -> Result<SlotId, EnqueueError> {
        if prompt.is_empty() {
            return Err(EnqueueError::EmptyPrompt);
        }
        if self.pending.len() >= self.config.max_queue_depth {
            return Err(EnqueueError::Saturated);
        }
        let id = SlotId::next();
        let request_tokens =
            reserved_context_tokens.max(prompt.len().saturating_add(params.max_new_tokens));
        let reserved_blocks = self.config.reserved_blocks_for(request_tokens);
        self.pending.push_back(PendingRequest {
            id,
            prompt,
            params,
            enqueued_at: Instant::now(),
            reserved_blocks,
        });
        Ok(id)
    }

    /// Count of slots not yet terminated (Pending + Prefilling + Decoding).
    /// This is the "pool size" the scheduler caps on, including
    /// not-yet-prefilling slots to prevent runaway enqueueing when
    /// prefill hasn't kicked in yet.
    fn pool_size(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| {
                matches!(
                    s.state,
                    SlotState::Pending | SlotState::Prefilling | SlotState::Decoding
                )
            })
            .count()
    }

    fn active_reserved_blocks(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| {
                matches!(
                    s.state,
                    SlotState::Pending | SlotState::Prefilling | SlotState::Decoding
                )
            })
            .map(|s| s.reserved_blocks)
            .sum()
    }

    fn next_admissible_pending_position(&self) -> Option<usize> {
        if self.pending.is_empty() {
            return None;
        }
        if self.config.max_blocks == 0 {
            return Some(0);
        }
        let remaining = self
            .config
            .max_blocks
            .saturating_sub(self.active_reserved_blocks());
        self.pending
            .iter()
            .position(|req| req.reserved_blocks <= remaining)
    }

    /// Pull pending requests into free slots until the batch is full.
    ///
    /// Called internally at the start of every `next_instruction`.
    fn fill_slots(&mut self) {
        while self.pool_size() < self.config.max_slots {
            let req_pos = match self.next_admissible_pending_position() {
                Some(pos) => pos,
                None => break,
            };
            let req = match self.pending.remove(req_pos) {
                Some(r) => r,
                None => break,
            };
            self.slots.push(Slot {
                id: req.id,
                state: SlotState::Pending,
                prompt: req.prompt,
                generated: Vec::new(),
                params: req.params,
                prefilled: 0,
                enqueued_at: req.enqueued_at,
                finish_reason: None,
                reserved_blocks: req.reserved_blocks,
            });
        }

        // Transition any Pending slot to Prefilling, unless another
        // slot is already prefilling (D.1 policy: serial prefill).
        let any_prefilling = self.slots.iter().any(|s| s.state == SlotState::Prefilling);
        if !any_prefilling {
            for slot in self.slots.iter_mut() {
                if slot.state == SlotState::Pending {
                    slot.state = SlotState::Prefilling;
                    break;
                }
            }
        }
    }

    /// Decide what to do next.
    ///
    /// Preference order:
    /// 1. Run a prefill chunk if any slot is in `Prefilling`.
    /// 2. Otherwise, batch all `Decoding` slots and step them together.
    /// 3. Otherwise, report `Idle`.
    pub fn next_instruction(&mut self) -> StepInstruction {
        self.fill_slots();

        // Serve prefills first. D.1 policy: one slot prefills at a time,
        // so there's at most one match here.
        if let Some(slot) = self.slots.iter().find(|s| s.state == SlotState::Prefilling) {
            let start = slot.prefilled;
            let step = slot.params.prefill_step_size.max(1);
            let end = (start + step).min(slot.prompt.len());
            let chunk: Vec<u32> = slot.prompt[start..end].to_vec();
            let final_chunk = end == slot.prompt.len();
            return StepInstruction::Prefill {
                slot: slot.id,
                chunk,
                final_chunk,
            };
        }

        // Batched decode over all Decoding slots.
        let decoding: Vec<SlotId> = self
            .slots
            .iter()
            .filter(|s| s.state == SlotState::Decoding)
            .map(|s| s.id)
            .collect();
        if !decoding.is_empty() {
            return StepInstruction::Decode { slots: decoding };
        }

        StepInstruction::Idle
    }

    /// Update slot state after a prefill chunk has been processed.
    ///
    /// `chunk_len` is the number of tokens actually consumed (what the
    /// caller fed to `forward`). When `final_chunk` is true, the slot
    /// transitions to `Decoding` so the next tick will include it in a
    /// batched decode.
    pub fn advance_prefill(&mut self, slot: SlotId, chunk_len: usize, final_chunk: bool) {
        if let Some(s) = self.slots.iter_mut().find(|s| s.id == slot) {
            debug_assert_eq!(s.state, SlotState::Prefilling);
            s.prefilled = s.prefilled.saturating_add(chunk_len).min(s.prompt.len());
            if final_chunk {
                s.state = SlotState::Decoding;
            }
        }
    }

    /// Update slot state after a batched decode step.
    ///
    /// `sampled` is a map from slot id → (sampled token, stop-sequence
    /// match). A slot is retired when:
    /// - the sampled token is in its `stop_tokens`, OR
    /// - `stop_match` is `true` (caller detected a stop sequence in the
    ///   decoded text), OR
    /// - `generated.len()` reaches `max_new_tokens`.
    pub fn advance_decode<I>(&mut self, sampled: I)
    where
        I: IntoIterator<Item = (SlotId, u32, bool)>,
    {
        for (id, token, stop_seq_match) in sampled {
            let Some(s) = self.slots.iter_mut().find(|s| s.id == id) else {
                continue;
            };
            debug_assert_eq!(s.state, SlotState::Decoding);

            // Stop token check fires before recording the token — mirrors
            // the single-request decode loop.
            if s.params.stop_tokens.contains(&token) {
                s.state = SlotState::Finished;
                s.finish_reason = Some(FinishReason::Stop);
                continue;
            }

            s.generated.push(token);

            if stop_seq_match {
                s.state = SlotState::Finished;
                s.finish_reason = Some(FinishReason::StopSequence);
                continue;
            }

            if s.generated.len() >= s.params.max_new_tokens {
                s.state = SlotState::Finished;
                s.finish_reason = Some(FinishReason::Length);
            }
        }
    }

    /// Cancel a slot (client disconnect, shutdown, etc).
    ///
    /// Marks the slot `Cancelled` so the next `drain_retired` sweep
    /// collects it. The scheduler doesn't drop the KV state here — that
    /// happens when the model-side driver sees the state flip and
    /// releases per-slot resources.
    pub fn cancel(&mut self, slot: SlotId) -> bool {
        if let Some(s) = self.slots.iter_mut().find(|s| s.id == slot) {
            if matches!(s.state, SlotState::Finished | SlotState::Cancelled) {
                return false;
            }
            s.state = SlotState::Cancelled;
            s.finish_reason = Some(FinishReason::Cancelled);
            true
        } else if let Some(pos) = self.pending.iter().position(|r| r.id == slot) {
            self.pending.remove(pos);
            true
        } else {
            false
        }
    }

    /// Move every `Finished` / `Cancelled` slot out of the active set
    /// and into an outbox, returning them to the caller. Call this
    /// after each step to drain completed work and free slots for the
    /// pending queue.
    pub fn drain_retired(&mut self) -> Vec<Slot> {
        let mut drained = Vec::new();
        let mut i = 0;
        while i < self.slots.len() {
            if matches!(
                self.slots[i].state,
                SlotState::Finished | SlotState::Cancelled
            ) {
                drained.push(self.slots.swap_remove(i));
            } else {
                i += 1;
            }
        }
        drained
    }

    /// Inspect the live slots without taking ownership.
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(max: usize) -> SlotParams {
        SlotParams {
            max_new_tokens: max,
            stop_tokens: vec![],
            stop_sequences: vec![],
            prefill_step_size: 4,
            logprobs_top_n: None,
        }
    }

    #[test]
    fn enqueue_rejects_empty_prompt() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        let err = b.enqueue(vec![], params(16)).unwrap_err();
        assert!(matches!(err, EnqueueError::EmptyPrompt));
    }

    #[test]
    fn enqueue_respects_queue_depth() {
        let mut b = ContinuousBatcher::new(BatcherConfig {
            max_slots: 1,
            max_queue_depth: 2,
            ..BatcherConfig::default()
        });
        b.enqueue(vec![1], params(1)).unwrap();
        b.enqueue(vec![1], params(1)).unwrap();
        let err = b.enqueue(vec![1], params(1)).unwrap_err();
        assert!(matches!(err, EnqueueError::Saturated));
    }

    #[test]
    fn prefill_emits_chunks_of_configured_size() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        let id = b
            .enqueue(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], params(4))
            .unwrap();

        // First tick: prefill chunk [1..=4].
        match b.next_instruction() {
            StepInstruction::Prefill {
                slot,
                chunk,
                final_chunk,
            } => {
                assert_eq!(slot, id);
                assert_eq!(chunk, vec![1, 2, 3, 4]);
                assert!(!final_chunk);
                b.advance_prefill(id, 4, false);
            }
            other => panic!("expected Prefill, got {other:?}"),
        }

        // Second tick: [5..=8].
        match b.next_instruction() {
            StepInstruction::Prefill { chunk, .. } => {
                assert_eq!(chunk, vec![5, 6, 7, 8]);
                b.advance_prefill(id, 4, false);
            }
            other => panic!("expected Prefill, got {other:?}"),
        }

        // Third tick: [9..=10] — final chunk.
        match b.next_instruction() {
            StepInstruction::Prefill {
                chunk, final_chunk, ..
            } => {
                assert_eq!(chunk, vec![9, 10]);
                assert!(final_chunk);
                b.advance_prefill(id, 2, true);
            }
            other => panic!("expected Prefill, got {other:?}"),
        }

        // Now the slot should be Decoding.
        match b.next_instruction() {
            StepInstruction::Decode { slots } => assert_eq!(slots, vec![id]),
            other => panic!("expected Decode, got {other:?}"),
        }
    }

    #[test]
    fn fifo_ordering_of_pending_requests() {
        let mut b = ContinuousBatcher::new(BatcherConfig {
            max_slots: 1,
            max_queue_depth: 16,
            ..BatcherConfig::default()
        });

        let a = b.enqueue(vec![1, 2], params(1)).unwrap();
        let c = b.enqueue(vec![3, 4], params(1)).unwrap();
        let d = b.enqueue(vec![5, 6], params(1)).unwrap();
        assert_eq!(b.pending_depth(), 3);

        // Only slot 'a' should start prefilling first.
        match b.next_instruction() {
            StepInstruction::Prefill { slot, .. } => assert_eq!(slot, a),
            other => panic!("expected Prefill(a), got {other:?}"),
        }

        // Finish a: prefill (one chunk covers 2 tokens), decode 1, stop-token.
        b.advance_prefill(a, 2, true);
        b.advance_decode([(a, 99u32, false)]);
        // generated.len() == 1 == max_new_tokens → Finished.
        assert_eq!(
            b.slots().iter().find(|s| s.id == a).unwrap().finish_reason,
            Some(FinishReason::Length)
        );
        let retired = b.drain_retired();
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].id, a);

        // Now c should be up next.
        match b.next_instruction() {
            StepInstruction::Prefill { slot, .. } => assert_eq!(slot, c),
            other => panic!("expected Prefill(c), got {other:?}"),
        }

        // And d is still pending.
        assert_eq!(b.pending_depth(), 1);
        assert!(b.pending.iter().any(|r| r.id == d));
    }

    #[test]
    fn batched_decode_includes_all_decoding_slots() {
        let mut b = ContinuousBatcher::new(BatcherConfig {
            max_slots: 3,
            max_queue_depth: 16,
            ..BatcherConfig::default()
        });

        let a = b.enqueue(vec![1], params(8)).unwrap();
        let c = b.enqueue(vec![2], params(8)).unwrap();
        let d = b.enqueue(vec![3], params(8)).unwrap();

        // Prefill all three serially.
        for id in [a, c, d] {
            match b.next_instruction() {
                StepInstruction::Prefill { slot, .. } => assert_eq!(slot, id),
                other => panic!("expected Prefill, got {other:?}"),
            }
            b.advance_prefill(id, 1, true);
        }

        match b.next_instruction() {
            StepInstruction::Decode { slots } => {
                assert_eq!(slots.len(), 3);
                assert!(slots.contains(&a));
                assert!(slots.contains(&c));
                assert!(slots.contains(&d));
            }
            other => panic!("expected Decode, got {other:?}"),
        }
    }

    #[test]
    fn stop_token_retires_slot() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        let params = SlotParams {
            max_new_tokens: 16,
            stop_tokens: vec![7],
            stop_sequences: vec![],
            prefill_step_size: 4,
            logprobs_top_n: None,
        };
        let id = b.enqueue(vec![1, 2, 3], params).unwrap();
        let _ = b.next_instruction(); // Prefill
        b.advance_prefill(id, 3, true);
        let _ = b.next_instruction(); // Decode
        b.advance_decode([(id, 7u32, false)]); // stop token

        let retired = b.drain_retired();
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].finish_reason, Some(FinishReason::Stop));
        // Stop token shouldn't be appended to the output.
        assert!(retired[0].generated.is_empty());
    }

    #[test]
    fn cancel_pending_removes_from_queue() {
        let mut b = ContinuousBatcher::new(BatcherConfig {
            max_slots: 1,
            max_queue_depth: 16,
            ..BatcherConfig::default()
        });
        let _active = b.enqueue(vec![1], params(1)).unwrap();
        let queued = b.enqueue(vec![2], params(1)).unwrap();
        // Step once so `_active` is pulled into the slot pool; `queued`
        // remains in the pending queue.
        let _ = b.next_instruction();
        assert_eq!(b.pending_depth(), 1);

        assert!(b.cancel(queued));
        assert_eq!(b.pending_depth(), 0);
    }

    #[test]
    fn cancel_active_retires_next_drain() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        let id = b.enqueue(vec![1, 2, 3], params(8)).unwrap();
        let _ = b.next_instruction();
        b.advance_prefill(id, 3, true);
        assert!(b.cancel(id));

        let retired = b.drain_retired();
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].finish_reason, Some(FinishReason::Cancelled));
    }

    #[test]
    fn idle_when_no_work() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        assert!(matches!(b.next_instruction(), StepInstruction::Idle));
    }

    #[test]
    fn drain_retired_frees_slots_for_pending() {
        let mut b = ContinuousBatcher::new(BatcherConfig {
            max_slots: 1,
            max_queue_depth: 16,
            ..BatcherConfig::default()
        });
        let a = b.enqueue(vec![1], params(1)).unwrap();
        let c = b.enqueue(vec![2], params(1)).unwrap();
        let _ = b.next_instruction();
        b.advance_prefill(a, 1, true);
        b.advance_decode([(a, 99u32, false)]);
        assert_eq!(b.drain_retired().len(), 1);

        // c should now be eligible.
        match b.next_instruction() {
            StepInstruction::Prefill { slot, .. } => assert_eq!(slot, c),
            other => panic!("expected Prefill(c), got {other:?}"),
        }
    }

    #[test]
    fn block_budget_holds_request_until_blocks_free() {
        let mut b = ContinuousBatcher::new(BatcherConfig {
            max_slots: 4,
            max_queue_depth: 16,
            block_size: 4,
            max_blocks: 3,
        });
        let a = b.enqueue(vec![1, 2, 3, 4], params(1)).unwrap(); // 5 tokens -> 2 blocks
        let c = b.enqueue(vec![5, 6, 7, 8], params(1)).unwrap(); // 5 tokens -> 2 blocks

        match b.next_instruction() {
            StepInstruction::Prefill { slot, .. } => assert_eq!(slot, a),
            other => panic!("expected Prefill(a), got {other:?}"),
        }
        assert_eq!(b.active_reserved_blocks(), 2);
        assert_eq!(b.pending_depth(), 1);
        assert!(b.pending.iter().any(|r| r.id == c));

        b.advance_prefill(a, 4, true);
        b.advance_decode([(a, 99u32, false)]);
        assert_eq!(b.drain_retired().len(), 1);

        match b.next_instruction() {
            StepInstruction::Prefill { slot, .. } => assert_eq!(slot, c),
            other => panic!("expected Prefill(c), got {other:?}"),
        }
    }

    #[test]
    fn block_scheduler_skips_head_of_line_when_later_request_fits() {
        let mut b = ContinuousBatcher::new(BatcherConfig {
            max_slots: 4,
            max_queue_depth: 16,
            block_size: 4,
            max_blocks: 3,
        });
        let active = b.enqueue(vec![1, 2, 3, 4], params(4)).unwrap(); // 2 blocks
        let big = b.enqueue(vec![5, 6, 7, 8], params(4)).unwrap(); // 2 blocks
        let small = b.enqueue(vec![9], params(1)).unwrap(); // 1 block

        match b.next_instruction() {
            StepInstruction::Prefill { slot, .. } => assert_eq!(slot, active),
            other => panic!("expected Prefill(active), got {other:?}"),
        }

        assert!(b.slots().iter().any(|s| s.id == small));
        assert!(b.pending.iter().any(|r| r.id == big));
        assert_eq!(b.pending_depth(), 1);
        assert_eq!(b.active_reserved_blocks(), 3);
    }
}
