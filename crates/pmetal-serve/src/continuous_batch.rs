//! Continuous-batching scheduler.
//!
//! This module is the *scheduling* half of continuous batching: a slot pool
//! with a bounded, fair pending queue, lifecycle tracking, and a planner that
//! tells the caller what to run on each iteration. It does not touch the
//! model or the KV cache; [`crate::continuous_pump`] and
//! [`crate::continuous_driver`] do.
//!
//! The scheduling is iteration-level: the batch is re-formed after every
//! model step, so a request joins as soon as a slot is free and leaves the
//! step it finishes, instead of waiting for the slowest request of a fixed
//! batch (Orca, Yu et al., OSDI 2022).
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
//! # Iterations
//!
//! Each iteration ([`Step`]) decodes one token for every `Decoding` slot and,
//! if a slot is `Prefilling`, runs one chunk of its prompt after it. A long
//! prompt therefore delays the decoding slots by one chunk per token and no
//! more, where running a whole prompt before the next decode would stall
//! every stream for its entire prefill (stall-free chunked prefill, Agrawal
//! et al., Sarathi-Serve, OSDI 2024).
//!
//! A request's chunk boundaries are those of the single-request path
//! ([`prefill_chunk_len`]), whatever else is in flight. The boundaries are
//! part of the numerics of a prefill (the keys and values of a chunk are
//! computed with different kernel shapes, and a quantized cache attends the
//! earlier chunks dequantized), so shrinking a chunk to fit a latency target
//! would make a reply depend on its neighbours. A deployment that wants a
//! shorter stall sets a smaller `prefill_step_size`, which every path honors.
//!
//! At most one slot prefills at a time: prefill is compute-bound and fills
//! the GPU on its own, so running two together finishes neither sooner.
//!
//! # Admission
//!
//! Admission is constrained by slot count and by a token-block budget for
//! the KV cache. Each request reserves
//! `ceil((cached_prefix + prompt_tokens + max_new_tokens) / block_size)`
//! blocks while it is active, clamped to the full budget so an oversized
//! request can still run alone. The reservation is the request's worst case,
//! so an admitted request never runs out of memory part-way and nothing is
//! ever preempted. The scheduler scans the pending queue for the first
//! request that fits, which keeps a large request from blocking small ones;
//! to keep small ones from starving a large one, a request that has been
//! overtaken [`BatcherConfig::max_overtakes`] times becomes a barrier that
//! nothing behind it passes until it is admitted.
//!
//! # Decode path selection
//!
//! The driver runs a decode step as one fused `[N_active, 1]` forward over a
//! shared [`pmetal_cache::kv_cache::FusedBatchKVCache`] when the model
//! supports it ([`pmetal_models::dispatcher::DynamicModel::supports_fused_batched`])
//! and the cache is a plain dense one, and otherwise as one forward per slot
//! on that slot's own cache. See [`crate::continuous_driver`].
//!
//! # Hybrid models
//!
//! Models with recurrent layers (Qwen 3.5 / 3.6 / 3.8 and Qwen4Exp
//! gated-delta-net, NemotronH and Granite 4.0-H Mamba2) take the per-slot
//! path, each slot with its own recurrent state beside its KV cache
//! ([`crate::continuous_driver::ContinuousEngineState::with_recurrent_state`]),
//! reset when the slot is freed. They keep no prefix cache, since a KV
//! snapshot can't restore recurrent state. The fused `[N, 1]` decode
//! carries no recurrent state and refuses them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use pmetal_models::generation::prefill_chunk_len;

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
    /// The prompt tokens to prefill: the whole prompt, or on a prefix-cache
    /// hit only the part after the restored prefix.
    pub prompt: Vec<u32>,
    pub generated: Vec<u32>,
    pub params: SlotParams,
    /// Index into `prompt` marking how many prompt tokens have been
    /// prefilled so far. Advances during chunked prefill.
    pub prefilled: usize,
    /// Tokens of the prompt already in the slot's KV cache before `prompt`
    /// starts (a restored prefix); chunk sizing counts them as depth.
    pub cached_prefix: usize,
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
    /// How many later requests may be admitted ahead of a pending one that
    /// does not fit the block budget yet. Past it the request is a barrier:
    /// nothing behind it is admitted until it is, so a stream of small
    /// requests cannot starve a large one.
    pub max_overtakes: usize,
    /// Decode every request on its own cache, one forward per request, even
    /// for a model that can decode a batch as one forward.
    ///
    /// A request then gets, bit for bit, the tokens the single-request path
    /// gives it. In a fused batch the matmuls run as `[N, hidden]` products
    /// where a lone request's run as vector products, which add in another
    /// order, so the last bit of a bf16 logit can differ and a near-tie in
    /// greedy decoding can fall the other way. The fused batch reads the
    /// weights once for all its requests, several times faster under load.
    pub per_slot_decode: bool,
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
            max_overtakes: 16,
            per_slot_decode: false,
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

/// Fraction of the device's recommended working set the model, its KV
/// caches and the scheduler may fill together. The rest is the headroom for
/// activations, the sampling temporaries and the allocator's cache.
pub const DEVICE_MEMORY_SHARE: f64 = 0.85;

/// How many token blocks of KV cache fit in the device's working set beside
/// what is already resident, so admission can stop before memory runs out.
///
/// `device_limit_bytes` is what the device recommends a process keep wired
/// (ask the device: it is a different fraction of the RAM on every Mac),
/// `resident_bytes` what the process holds now (the weights, above all),
/// `bytes_per_block` what one block of a request's KV cache takes. The
/// result is at most `ceiling_blocks` (what every slot could ever use) and at
/// least one, since a request that does not fit beside the model alone is
/// no better off refused.
pub fn kv_block_budget(
    device_limit_bytes: usize,
    resident_bytes: usize,
    bytes_per_block: usize,
    ceiling_blocks: usize,
) -> usize {
    let usable = (device_limit_bytes as f64 * DEVICE_MEMORY_SHARE) as usize;
    let free = usable.saturating_sub(resident_bytes);
    (free / bytes_per_block.max(1)).clamp(1, ceiling_blocks.max(1))
}

/// One chunk of a slot's prompt to feed to the model.
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    pub slot: SlotId,
    /// The exact token slice to feed to forward() this iteration.
    pub chunk: Vec<u32>,
    /// Whether this is the final chunk (so the scheduler can
    /// transition the slot to `Decoding` after this step).
    pub final_chunk: bool,
}

/// One scheduler iteration: what to run before the batch is re-formed.
#[derive(Debug, Clone)]
pub struct Step {
    /// Slots to decode one token each, as one batched step. Empty when no
    /// slot is decoding.
    pub decode: Vec<SlotId>,
    /// One chunk of the prompt of the slot that is prefilling, to run after
    /// the decode.
    pub prefill: Option<PrefillChunk>,
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
    cached_prefix: usize,
    enqueued_at: Instant,
    reserved_blocks: usize,
    /// How many later requests were admitted ahead of this one.
    overtaken: usize,
}

/// The continuous batching scheduler.
///
/// Tick the scheduler in a loop: `next_step()` tells you what to
/// do, then you call `advance_prefill` or `advance_decode` back with the
/// sampled tokens to update slot state. Finished slots are moved to an
/// outbox the caller drains via `drain_retired()`.
pub struct ContinuousBatcher {
    config: BatcherConfig,
    slots: Vec<Slot>,
    pending: VecDeque<PendingRequest>,
    /// Set when the active requests share one padded KV buffer (see
    /// [`pad_rows`](Self::pad_rows)).
    padded_row_step: Option<usize>,
}

impl ContinuousBatcher {
    /// Create a new batcher with the given config.
    pub fn new(config: BatcherConfig) -> Self {
        Self {
            config,
            slots: Vec::new(),
            pending: VecDeque::new(),
            padded_row_step: None,
        }
    }

    /// Count the block budget as one padded buffer instead of a sum.
    ///
    /// Requests that decode together in a fused batch live in one buffer of
    /// `rows` rows, each as long as the longest request, with the number of
    /// rows rounded up to a multiple of `row_step`. The memory the batch
    /// takes is then `rows * longest`, not the sum of what each request
    /// needs, and admission has to charge it that way.
    pub fn pad_rows(&mut self, row_step: usize) {
        self.padded_row_step = Some(row_step.max(1));
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
        self.enqueue_after_prefix(prompt, params, 0)
    }

    /// Enqueue a request whose first `cached_prefix` tokens are already in
    /// its KV cache (a prefix-cache hit): `prompt` is the rest. The active KV
    /// cache holds the prefix too, so the reservation counts it, and chunk
    /// sizing starts at its depth.
    pub fn enqueue_after_prefix(
        &mut self,
        prompt: Vec<u32>,
        params: SlotParams,
        cached_prefix: usize,
    ) -> Result<SlotId, EnqueueError> {
        if prompt.is_empty() {
            return Err(EnqueueError::EmptyPrompt);
        }
        if self.pending.len() >= self.config.max_queue_depth {
            return Err(EnqueueError::Saturated);
        }
        let id = SlotId::next();
        let request_tokens = cached_prefix
            .saturating_add(prompt.len())
            .saturating_add(params.max_new_tokens);
        let reserved_blocks = self.config.reserved_blocks_for(request_tokens);
        self.pending.push_back(PendingRequest {
            id,
            prompt,
            params,
            cached_prefix,
            enqueued_at: Instant::now(),
            reserved_blocks,
            overtaken: 0,
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

    /// Whether a request reserving `blocks` fits beside the active ones.
    fn fits(&self, blocks: usize) -> bool {
        let live = || {
            self.slots.iter().filter(|s| {
                matches!(
                    s.state,
                    SlotState::Pending | SlotState::Prefilling | SlotState::Decoding
                )
            })
        };
        let Some(row_step) = self.padded_row_step else {
            return self.active_reserved_blocks() + blocks <= self.config.max_blocks;
        };
        let (rows, longest) = live().fold((1usize, blocks), |(rows, longest), s| {
            (rows + 1, longest.max(s.reserved_blocks))
        });
        // Alone, a request always fits (its reservation is clamped to the
        // budget); beside others the whole padded buffer must.
        let padded_rows = (rows.div_ceil(row_step) * row_step).min(self.config.max_slots);
        rows == 1 || padded_rows.max(rows) * longest <= self.config.max_blocks
    }

    /// The pending request to admit next, if any: the first that fits the
    /// remaining block budget, unless a request ahead of it has been
    /// overtaken [`BatcherConfig::max_overtakes`] times, which holds the line.
    fn next_admissible_pending_position(&self) -> Option<usize> {
        if self.pending.is_empty() {
            return None;
        }
        if self.config.max_blocks == 0 {
            return Some(0);
        }
        for (pos, req) in self.pending.iter().enumerate() {
            if self.fits(req.reserved_blocks) {
                return Some(pos);
            }
            if req.overtaken >= self.config.max_overtakes {
                return None;
            }
        }
        None
    }

    /// Pull pending requests into free slots until the batch is full.
    ///
    /// Called internally at the start of every `next_step`.
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
            for passed in self.pending.iter_mut().take(req_pos) {
                passed.overtaken += 1;
            }
            self.slots.push(Slot {
                id: req.id,
                state: SlotState::Pending,
                prompt: req.prompt,
                generated: Vec::new(),
                params: req.params,
                prefilled: 0,
                cached_prefix: req.cached_prefix,
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

    /// Plan the next iteration, or `None` when there is nothing to do.
    ///
    /// Every `Decoding` slot decodes one token, and the slot that is
    /// `Prefilling` (there is at most one) gets one chunk of its prompt after
    /// them, so a long prompt never holds the others up for more than a chunk.
    pub fn next_step(&mut self) -> Option<Step> {
        self.fill_slots();

        let decode: Vec<SlotId> = self
            .slots
            .iter()
            .filter(|s| s.state == SlotState::Decoding)
            .map(|s| s.id)
            .collect();
        let prefill = self
            .slots
            .iter()
            .find(|s| s.state == SlotState::Prefilling)
            .map(|slot| {
                let start = slot.prefilled;
                let remaining = slot.prompt.len() - start;
                let step = slot.params.prefill_step_size;
                // The single-request path's chunking, from the same depth: a
                // step size of 0 prefills the rest in one forward.
                let len = if step == 0 {
                    remaining
                } else {
                    prefill_chunk_len(step, slot.cached_prefix + start).min(remaining)
                };
                PrefillChunk {
                    slot: slot.id,
                    chunk: slot.prompt[start..start + len].to_vec(),
                    final_chunk: len == remaining,
                }
            });
        (!decode.is_empty() || prefill.is_some()).then_some(Step { decode, prefill })
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
            // A slot cancelled while its step ran has no token to take.
            let Some(s) = self
                .slots
                .iter_mut()
                .find(|s| s.id == id && s.state == SlotState::Decoding)
            else {
                continue;
            };

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

    /// The state of a live slot.
    pub fn state(&self, slot: SlotId) -> Option<SlotState> {
        self.slots.iter().find(|s| s.id == slot).map(|s| s.state)
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

    fn prefill_of(step: Option<Step>) -> PrefillChunk {
        step.expect("a step")
            .prefill
            .expect("a prefill chunk in the step")
    }

    fn decode_of(step: Option<Step>) -> Vec<SlotId> {
        step.expect("a step").decode
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

        let first = prefill_of(b.next_step());
        assert_eq!(
            (first.slot, first.chunk, first.final_chunk),
            (id, vec![1, 2, 3, 4], false)
        );
        b.advance_prefill(id, 4, false);

        let second = prefill_of(b.next_step());
        assert_eq!(second.chunk, vec![5, 6, 7, 8]);
        b.advance_prefill(id, 4, false);

        let third = prefill_of(b.next_step());
        assert_eq!(third.chunk, vec![9, 10]);
        assert!(third.final_chunk);
        b.advance_prefill(id, 2, true);

        assert_eq!(decode_of(b.next_step()), vec![id]);
    }

    /// A step size of zero is the single-request path's "whole prompt in one
    /// forward".
    #[test]
    fn a_zero_step_size_prefills_the_prompt_in_one_chunk() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        let prompt: Vec<u32> = (1..=100).collect();
        let slot_params = SlotParams {
            prefill_step_size: 0,
            ..params(4)
        };
        b.enqueue(prompt.clone(), slot_params).unwrap();
        let chunk = prefill_of(b.next_step());
        assert_eq!(chunk.chunk, prompt);
        assert!(chunk.final_chunk);
    }

    /// Chunks shrink with the depth they start at, a restored prefix counted,
    /// exactly as the single-request path's do.
    #[test]
    fn chunks_start_from_the_depth_of_a_restored_prefix() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        let step = 2048;
        let depth = 200_000;
        let slot_params = SlotParams {
            prefill_step_size: step,
            ..params(4)
        };
        b.enqueue_after_prefix((0..4096).collect(), slot_params, depth)
            .unwrap();
        let chunk = prefill_of(b.next_step());
        assert_eq!(chunk.chunk.len(), prefill_chunk_len(step, depth));
        assert!(
            chunk.chunk.len() < step,
            "a deep prefill runs shorter chunks"
        );
    }

    /// The reply a request gets must not depend on its neighbours, so its
    /// chunks do not shrink when other slots are decoding: a long prompt is
    /// interleaved with their decode steps, one chunk per step.
    #[test]
    fn a_long_prefill_interleaves_with_decoding_without_changing_its_chunks() {
        let mut b = ContinuousBatcher::new(BatcherConfig {
            max_slots: 2,
            ..BatcherConfig::default()
        });
        let a = b.enqueue(vec![1], params(100)).unwrap();
        let prompt: Vec<u32> = (10..22).collect();
        let long = b.enqueue(prompt.clone(), params(4)).unwrap();

        let step = b.next_step().unwrap();
        assert!(step.decode.is_empty());
        assert_eq!(step.prefill.as_ref().unwrap().slot, a);
        b.advance_prefill(a, 1, true);

        let mut fed = Vec::new();
        let mut steps_with_decode = 0;
        loop {
            let step = b.next_step().unwrap();
            if step.decode.contains(&a) {
                steps_with_decode += 1;
                b.advance_decode([(a, 7u32, false)]);
            }
            let chunk = step.prefill.expect("the long prompt is still prefilling");
            assert_eq!(chunk.slot, long);
            fed.extend_from_slice(&chunk.chunk);
            b.advance_prefill(long, chunk.chunk.len(), chunk.final_chunk);
            if chunk.final_chunk {
                break;
            }
        }
        assert_eq!(fed, prompt);
        assert_eq!(steps_with_decode, 3, "a decode step ran before each chunk");
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
        assert_eq!(prefill_of(b.next_step()).slot, a);

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
        assert_eq!(prefill_of(b.next_step()).slot, c);

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
            assert_eq!(prefill_of(b.next_step()).slot, id);
            b.advance_prefill(id, 1, true);
        }

        let slots = decode_of(b.next_step());
        assert_eq!(slots.len(), 3);
        assert!(slots.contains(&a) && slots.contains(&c) && slots.contains(&d));
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
        let _ = b.next_step(); // Prefill
        b.advance_prefill(id, 3, true);
        let _ = b.next_step(); // Decode
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
        let _ = b.next_step();
        assert_eq!(b.pending_depth(), 1);

        assert!(b.cancel(queued));
        assert_eq!(b.pending_depth(), 0);
    }

    #[test]
    fn cancel_active_retires_next_drain() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        let id = b.enqueue(vec![1, 2, 3], params(8)).unwrap();
        let _ = b.next_step();
        b.advance_prefill(id, 3, true);
        assert!(b.cancel(id));

        let retired = b.drain_retired();
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].finish_reason, Some(FinishReason::Cancelled));
    }

    /// A slot cancelled while its decode step ran is not given the token that
    /// step sampled, and keeps its `Cancelled` outcome.
    #[test]
    fn a_cancelled_slot_takes_no_token_from_its_step() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        let id = b.enqueue(vec![1], params(1)).unwrap();
        let _ = b.next_step();
        b.advance_prefill(id, 1, true);
        assert!(b.cancel(id));
        b.advance_decode([(id, 5u32, false)]);

        let retired = b.drain_retired();
        assert_eq!(retired[0].finish_reason, Some(FinishReason::Cancelled));
        assert!(retired[0].generated.is_empty());
    }

    #[test]
    fn idle_when_no_work() {
        let mut b = ContinuousBatcher::new(BatcherConfig::default());
        assert!(b.next_step().is_none());
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
        let _ = b.next_step();
        b.advance_prefill(a, 1, true);
        b.advance_decode([(a, 99u32, false)]);
        assert_eq!(b.drain_retired().len(), 1);

        // c should now be eligible.
        assert_eq!(prefill_of(b.next_step()).slot, c);
    }

    fn block_config(max_blocks: usize, max_overtakes: usize) -> BatcherConfig {
        BatcherConfig {
            max_slots: 4,
            max_queue_depth: 64,
            block_size: 4,
            max_blocks,
            max_overtakes,
            per_slot_decode: false,
        }
    }

    #[test]
    fn block_budget_holds_request_until_blocks_free() {
        let mut b = ContinuousBatcher::new(block_config(3, 16));
        let a = b.enqueue(vec![1, 2, 3, 4], params(1)).unwrap(); // 5 tokens -> 2 blocks
        let c = b.enqueue(vec![5, 6, 7, 8], params(1)).unwrap(); // 5 tokens -> 2 blocks

        assert_eq!(prefill_of(b.next_step()).slot, a);
        assert_eq!(b.active_reserved_blocks(), 2);
        assert_eq!(b.pending_depth(), 1);
        assert!(b.pending.iter().any(|r| r.id == c));

        b.advance_prefill(a, 4, true);
        b.advance_decode([(a, 99u32, false)]);
        assert_eq!(b.drain_retired().len(), 1);

        assert_eq!(prefill_of(b.next_step()).slot, c);
    }

    #[test]
    fn block_scheduler_skips_head_of_line_when_later_request_fits() {
        let mut b = ContinuousBatcher::new(block_config(3, 16));
        let active = b.enqueue(vec![1, 2, 3, 4], params(4)).unwrap(); // 2 blocks
        let big = b.enqueue(vec![5, 6, 7, 8], params(4)).unwrap(); // 2 blocks
        let small = b.enqueue(vec![9], params(1)).unwrap(); // 1 block

        assert_eq!(prefill_of(b.next_step()).slot, active);

        assert!(b.slots().iter().any(|s| s.id == small));
        assert!(b.pending.iter().any(|r| r.id == big));
        assert_eq!(b.pending_depth(), 1);
        assert_eq!(b.active_reserved_blocks(), 3);
    }

    /// Requests that decode in one padded buffer cost `rows * longest`, not
    /// the sum of what each needs: beside a long request, every row is as
    /// long as it is.
    #[test]
    fn padded_admission_charges_rows_times_the_longest_request() {
        let mut b = ContinuousBatcher::new(BatcherConfig {
            max_slots: 8,
            ..block_config(16, 16)
        });
        b.pad_rows(4);
        // 4 blocks, then one-block requests.
        let _long = b.enqueue(vec![1, 2, 3, 4, 5, 6, 7, 8], params(8)).unwrap();
        for i in 0..5 {
            b.enqueue(vec![20 + i], params(1)).unwrap();
        }
        let _ = b.next_step();
        // Rows come in fours: four rows of four blocks fill the budget, and a
        // fifth request would need eight such rows, though the blocks it and
        // the others reserve add up to far less.
        assert_eq!(b.slots().len(), 4);
        assert_eq!(b.pending_depth(), 2);
        assert!(b.active_reserved_blocks() < 16);

        // The same requests are all admitted when only their sum counts.
        let mut sum = ContinuousBatcher::new(BatcherConfig {
            max_slots: 8,
            ..block_config(16, 16)
        });
        sum.enqueue(vec![1, 2, 3, 4, 5, 6, 7, 8], params(8))
            .unwrap();
        for i in 0..5 {
            sum.enqueue(vec![20 + i], params(1)).unwrap();
        }
        let _ = sum.next_step();
        assert_eq!(sum.slots().len(), 6);
    }

    /// The KV budget is what the device can hold beside the model, in blocks,
    /// never more than every slot could fill and never none.
    #[test]
    fn the_kv_budget_follows_the_device_and_the_model() {
        const GB: usize = 1 << 30;
        let block = 4 << 20; // 4 MiB of K/V per block
        // A 64 GB Mac recommending 48 GB, 8 GB resident: 85% of 48 is 40.8,
        // 32.8 GB free, in blocks of 4 MiB.
        assert_eq!(kv_block_budget(48 * GB, 8 * GB, block, usize::MAX), 8396);
        // The same model on a 16 GB Mac recommending 11.8 GB: 10 GB less 8.
        let small = kv_block_budget(12 * GB, 8 * GB, block, usize::MAX);
        assert!((1..1000).contains(&small), "{small}");
        // Every slot can never use more than its context.
        assert_eq!(kv_block_budget(48 * GB, 8 * GB, block, 500), 500);
        // A model that fills the device still gets a block, so a request
        // can run alone.
        assert_eq!(kv_block_budget(12 * GB, 20 * GB, block, 500), 1);
    }

    /// Small requests may pass a large one that doesn't fit yet, but only so
    /// many times: after that it holds the line, so a steady stream of small
    /// requests cannot keep it out for good.
    #[test]
    fn a_request_overtaken_too_often_holds_the_line() {
        let mut b = ContinuousBatcher::new(block_config(4, 2));
        // One block each; the first two stay active throughout.
        let first = b.enqueue(vec![1], params(1)).unwrap();
        let second = b.enqueue(vec![2], params(1)).unwrap();
        // Three blocks: fits only once at most one other request is active.
        let big = b
            .enqueue(vec![3, 4, 5, 6, 7, 8, 9, 10, 11], params(2))
            .unwrap();
        let smalls: Vec<SlotId> = (0..6)
            .map(|i| b.enqueue(vec![20 + i], params(1)).unwrap())
            .collect();

        let admitted =
            |b: &ContinuousBatcher| -> Vec<SlotId> { b.slots().iter().map(|s| s.id).collect() };
        let _ = b.next_step();
        assert_eq!(admitted(&b), vec![first, second, smalls[0], smalls[1]]);

        // Retire the two smalls: one block each frees, the big one still
        // needs three and has now been passed twice.
        for id in [smalls[0], smalls[1]] {
            b.cancel(id);
        }
        b.drain_retired();
        let _ = b.next_step();
        assert!(
            !admitted(&b).contains(&smalls[2]),
            "a request overtaken max_overtakes times holds the line"
        );
        assert_eq!(b.pending.front().map(|r| r.id), Some(big));

        // Once the others leave, the big one goes first.
        for id in [first, second] {
            b.cancel(id);
        }
        b.drain_retired();
        let _ = b.next_step();
        assert!(admitted(&b).contains(&big));
    }
}
