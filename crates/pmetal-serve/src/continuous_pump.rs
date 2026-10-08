//! D.3 — request pump for continuous batching.
//!
//! Glues [`ContinuousBatcher`](crate::continuous_batch) (pure-Rust
//! scheduler) and [`ContinuousEngineState`](crate::continuous_driver)
//! (MLX-touching driver) into a single object that the serve layer
//! drives tick-by-tick.
//!
//! # Lifecycle
//!
//! ```text
//!   route   ──enqueue(prompt, params)──▶ ContinuousPump
//!     ▲                                         │
//!     │                                         │  Receiver<TokenEvent>
//!     └──────◀──── per-slot events ◀────────────┘
//!
//!   driver thread:
//!     loop {
//!         match pump.tick(&mut forward)? {
//!             Tick::Ran  => continue,   // we did useful work
//!             Tick::Idle => park,       // sleep until an enqueue wakes us
//!         }
//!     }
//! ```
//!
//! A dedicated driver thread calls [`ContinuousPump::tick`] in a loop.
//! Each tick pulls the next instruction from the batcher, runs one
//! prefill chunk or one batched decode step, advances slot state, and
//! drains any retired slots — delivering `Done`/`Error` events to the
//! request's channel.
//!
//! # Per-slot features (Phase 1)
//!
//! The pump tracks per-slot history and per-slot text state so each
//! request gets the same generation semantics as the single-slot
//! `generate_streaming` path:
//!
//! - Stop-sequence matching — when a pump is built with a
//!   `pmetal_data::Tokenizer`, each emitted token triggers a decode of
//!   the slot's generated stream and a suffix check against the slot's
//!   `stop_sequences`. A match retires the slot with
//!   `FinishReason::StopSequence` and propagates `stripped_tokens`.
//! - Per-slot logprobs — when `SlotParams.logprobs_top_n` is set, the
//!   pump computes `token_logprobs` from the raw last-position logits
//!   and attaches the entry to the emitted `TokenEvent::Token`. Tokens
//!   without a logprob request pay nothing.
//! - Per-slot repetition / frequency / presence penalties — each slot's
//!   `Sampler` is built from its own `GenerationConfig`, and the
//!   driver calls `sample_array_with_penalties(&last, &history)` plus
//!   `update_counts(token)` so penalties apply exactly as they do in
//!   single-slot generation. No extra work when penalties are off.

use crate::continuous_batch::{
    BatcherConfig, ContinuousBatcher, EnqueueError, FinishReason, SlotId, SlotParams,
    StepInstruction,
};
use crate::continuous_driver::{
    ContinuousEngineState, SlotForward, drive_decode_step, drive_prefill_step,
};
use crate::engine::{TokenEvent, TokenLogprobEntry, detect_stop_sequence_suffix};
use crate::prefix_cache::ServePrefixCache;
use pmetal_bridge::compat::{Array, Exception};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::generation::{GenerationConfig, Sampler, token_logprobs};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;

/// Per-slot request record that the pump tracks across ticks.
struct SlotRecord {
    tx: mpsc::Sender<TokenEvent>,
    /// Current decoded token used as the next decode input.
    current_token: Option<u32>,
    /// Absolute start of request — for latency metrics.
    start: Instant,
    /// Prompt token count captured at enqueue (for Done metrics).
    prompt_tokens: usize,
    /// Running count of emitted tokens.
    emitted: usize,
    /// Generation config used to build the per-slot sampler at admit
    /// time. Each request carries its own so temperature / top-p /
    /// penalties aren't shared across concurrent requests.
    gen_config: GenerationConfig,
    /// Full token history (prompt + generated so far). Drives the
    /// sampler's repetition-penalty pass each step.
    all_tokens: Vec<u32>,
    /// Just the generated-so-far tokens. Used for stop-sequence
    /// detection; kept separate from `all_tokens` so we can decode only
    /// the output text without re-decoding the prompt each tick.
    generated: Vec<u32>,
    /// Raw text stop sequences to scan for after each emitted token.
    stop_sequences: Vec<String>,
    /// When set, emit per-token logprobs with top-N alternatives.
    logprobs_top_n: Option<usize>,
    /// Count of trailing tokens matching the most recent stop sequence.
    /// Forwarded to the `Done` event so the client can drop the
    /// matched suffix from its rendered text.
    stripped_tokens: usize,
    /// Time (ms from `start`) at which the first token was emitted.
    /// Populated lazily; reported in the final `RequestMetrics`.
    first_token_ms: Option<f64>,
    /// Forked prefix cache to install into the slot at admit time.
    prefix_seed: Option<KVCache>,
    /// Number of prompt tokens restored from `prefix_seed`.
    prefix_hit_len: usize,
}

/// Outcome of a single [`ContinuousPump::tick`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tick {
    /// Ran a prefill or decode step.
    Ran,
    /// Scheduler reports no work — caller should park until the next
    /// enqueue wakes it.
    Idle,
}

/// Cross-request continuous-batching pump.
pub struct ContinuousPump {
    batcher: ContinuousBatcher,
    state: ContinuousEngineState,
    records: HashMap<SlotId, SlotRecord>,
    /// Cache config used for newly-admitted slots. Stashed at
    /// construction since the batched KV cache was allocated with this
    /// same config.
    cache_config: KVCacheConfig,
    /// Tokenizer used for raw-text stop-sequence detection. When
    /// `None`, per-slot `stop_sequences` are silently ignored (the
    /// stop-token list still works). Tests without a real tokenizer
    /// pass `None`.
    tokenizer: Option<Arc<pmetal_data::Tokenizer>>,
    /// Optional cross-request prefix cache shared with the engine.
    prefix_cache: Option<Arc<Mutex<ServePrefixCache>>>,
}

impl ContinuousPump {
    /// Build a new pump with `config.max_slots` concurrent slots, each
    /// with a KV cache shaped by `cache_config`. `tokenizer` is used
    /// for raw-text stop-sequence detection; pass `None` for pumps that
    /// only need stop-token termination (e.g. unit tests).
    pub fn new(
        config: BatcherConfig,
        cache_config: KVCacheConfig,
        tokenizer: Option<Arc<pmetal_data::Tokenizer>>,
    ) -> Self {
        Self::new_with_prefix_cache(config, cache_config, tokenizer, None)
    }

    /// Build a pump with an optional shared prefix cache.
    pub fn new_with_prefix_cache(
        config: BatcherConfig,
        cache_config: KVCacheConfig,
        tokenizer: Option<Arc<pmetal_data::Tokenizer>>,
        prefix_cache: Option<Arc<Mutex<ServePrefixCache>>>,
    ) -> Self {
        let max_slots = config.max_slots;
        let batcher = ContinuousBatcher::new(config);
        let state = ContinuousEngineState::new(max_slots, cache_config.clone());
        Self {
            batcher,
            state,
            records: HashMap::new(),
            cache_config,
            tokenizer,
            prefix_cache,
        }
    }

    /// Enqueue a new request. Returns its `SlotId` plus a receiver that
    /// will emit one `TokenEvent::Token` per generated token, followed
    /// by exactly one `TokenEvent::Done` (or `TokenEvent::Error`).
    ///
    /// `channel_capacity` bounds the receiver buffer; pick something
    /// small (16-64) to apply back-pressure if the route handler falls
    /// behind.
    pub fn enqueue(
        &mut self,
        prompt: Vec<u32>,
        params: SlotParams,
        gen_config: GenerationConfig,
        channel_capacity: usize,
    ) -> Result<(SlotId, mpsc::Receiver<TokenEvent>), EnqueueError> {
        let (tx, rx) = mpsc::channel(channel_capacity);
        let slot = self.enqueue_with_sender(prompt, params, gen_config, tx)?;
        Ok((slot, rx))
    }

    /// [`enqueue`](Self::enqueue) into a channel the caller made. On error
    /// nothing is sent, so the caller can still use `tx` for the request.
    pub fn enqueue_with_sender(
        &mut self,
        prompt: Vec<u32>,
        params: SlotParams,
        gen_config: GenerationConfig,
        tx: mpsc::Sender<TokenEvent>,
    ) -> Result<SlotId, EnqueueError> {
        let prompt_tokens = prompt.len();
        let stop_sequences = params.stop_sequences.clone();
        let logprobs_top_n = params.logprobs_top_n;
        let reserved_tokens = prompt_tokens.saturating_add(params.max_new_tokens);

        let mut scheduler_prompt = prompt.clone();
        let mut prefix_seed = None;
        let mut prefix_hit_len = 0usize;
        if let Some(prefix_cache) = self.prefix_cache.as_ref() {
            match prefix_cache.lock() {
                Ok(mut guard) => {
                    match guard.find_longest_prefix(&prompt, self.cache_config.clone()) {
                        Ok(Some(hit)) => {
                            prefix_hit_len = hit.prefix_len;
                            scheduler_prompt = prompt[prefix_hit_len..].to_vec();
                            prefix_seed = Some(hit.restored_cache);
                            tracing::debug!(
                                target: "pmetal_serve::continuous_pump",
                                restored = prefix_hit_len,
                                total = prompt_tokens,
                                "continuous-batch prefix cache hit"
                            );
                        }
                        Ok(None) => {}
                        Err(e) => {
                            tracing::warn!(
                                target: "pmetal_serve::continuous_pump",
                                error = %e,
                                "ignoring prefix-cache restore failure"
                            );
                        }
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        target: "pmetal_serve::continuous_pump",
                        "ignoring poisoned prefix-cache lock"
                    );
                }
            }
        }

        // Clone the prompt into `all_tokens` so the sampler sees the
        // full prompt when applying repetition penalty to the first
        // generated token. The scheduler consumes the original prompt
        // or, on a prefix hit, only its uncached suffix.
        let all_tokens = prompt.clone();
        let slot =
            self.batcher
                .enqueue_with_reserved_tokens(scheduler_prompt, params, reserved_tokens)?;
        self.records.insert(
            slot,
            SlotRecord {
                tx,
                current_token: None,
                start: Instant::now(),
                prompt_tokens,
                emitted: 0,
                gen_config,
                all_tokens,
                generated: Vec::new(),
                stop_sequences,
                logprobs_top_n,
                stripped_tokens: 0,
                first_token_ms: None,
                prefix_seed,
                prefix_hit_len,
            },
        );
        Ok(slot)
    }

    /// Cancel an in-flight slot. Pending slots are simply dropped; a
    /// slot that's already prefilling/decoding is marked cancelled and
    /// retired on the next tick.
    pub fn cancel(&mut self, slot: SlotId) -> bool {
        let cancelled = self.batcher.cancel(slot);
        if cancelled {
            // Drop the per-slot record so any in-flight send on the tx
            // half errors out — the receiver has already been
            // abandoned by the caller.
            self.records.remove(&slot);
        }
        cancelled
    }

    /// Number of currently-active slots (Prefilling + Decoding).
    pub fn active_slots(&self) -> usize {
        self.batcher.active_slots()
    }

    /// Depth of the pending queue.
    pub fn pending_depth(&self) -> usize {
        self.batcher.pending_depth()
    }

    /// Cache-config used for all slots.
    pub fn cache_config(&self) -> &KVCacheConfig {
        &self.cache_config
    }

    /// Execute one scheduler step against `forward`.
    ///
    /// Returns `Tick::Idle` if the scheduler has nothing to do (caller
    /// should park until the next enqueue), else `Tick::Ran`.
    pub fn tick<F: SlotForward>(&mut self, forward: &mut F) -> Result<Tick, Exception> {
        let instruction = self.batcher.next_instruction();
        match instruction {
            StepInstruction::Idle => Ok(Tick::Idle),

            StepInstruction::Prefill {
                slot,
                chunk,
                final_chunk,
            } => {
                // Admit the slot the first time we see it — sampler and
                // batch row are created lazily here so we don't
                // allocate anything for requests that get cancelled
                // before reaching the prefill stage.
                self.admit_if_needed(slot)?;
                let chunk_len = chunk.len();
                let maybe_last =
                    drive_prefill_step(forward, &mut self.state, slot, &chunk, final_chunk)?;
                self.batcher.advance_prefill(slot, chunk_len, final_chunk);

                // If we just prefilled the final chunk, sample the
                // first decode token immediately. This avoids a
                // "dead" tick between prefill-complete and the first
                // decode step.
                if let Some(last_logits) = maybe_last {
                    self.save_prefix_if_possible(slot);
                    let (history, top_n) = {
                        let rec = self
                            .records
                            .get(&slot)
                            .ok_or_else(|| Exception::custom(format!("no record for {slot:?}")))?;
                        (rec.all_tokens.clone(), rec.logprobs_top_n)
                    };
                    let sampler = self.state.sampler_for(slot).ok_or_else(|| {
                        Exception::custom(format!("slot {slot:?} has no sampler"))
                    })?;
                    let (y, _filtered) =
                        sampler.sample_array_with_penalties(&last_logits, &history)?;
                    let token = y.item::<u32>();
                    sampler.update_counts(token);
                    let logprob = Self::compute_logprob_entry(&last_logits, token, top_n);
                    self.record_first_token(slot, token, logprob)?;
                }

                self.drain_retired();
                Ok(Tick::Ran)
            }

            StepInstruction::Decode { slots } => {
                // Build (slot, current_token, history) inputs. History
                // clones are small (at most max_seq_len tokens per
                // slot) and happen once per tick, so no hot-path
                // amplification.
                let mut histories: Vec<Vec<u32>> = Vec::with_capacity(slots.len());
                let mut inputs: Vec<(SlotId, u32)> = Vec::with_capacity(slots.len());
                for s in &slots {
                    let rec = self
                        .records
                        .get(s)
                        .ok_or_else(|| Exception::custom(format!("missing record for {s:?}")))?;
                    let tok = rec.current_token.ok_or_else(|| {
                        Exception::custom(format!("slot {s:?} in Decode without a current token"))
                    })?;
                    histories.push(rec.all_tokens.clone());
                    inputs.push((*s, tok));
                }
                let input_refs: Vec<(SlotId, u32, &[u32])> = inputs
                    .iter()
                    .zip(histories.iter())
                    .map(|((s, t), h)| (*s, *t, h.as_slice()))
                    .collect();

                let outs = drive_decode_step(forward, &mut self.state, &input_refs)?;

                // Deliver tokens + update per-slot state + feed the
                // scheduler so it can apply stop-token / length /
                // stop-sequence termination.
                let mut sampled: Vec<(SlotId, u32, bool)> = Vec::with_capacity(outs.len());
                for out in outs {
                    let logprob = Self::compute_logprob_entry(
                        &out.logits,
                        out.token,
                        self.records.get(&out.slot).and_then(|r| r.logprobs_top_n),
                    );
                    let stop_seq_match = self.emit_token(out.slot, out.token, logprob)?;
                    sampled.push((out.slot, out.token, stop_seq_match));
                }
                self.batcher.advance_decode(sampled);
                self.drain_retired();
                Ok(Tick::Ran)
            }
        }
    }

    /// Allocate a batch row + per-slot sampler for `slot` if not
    /// already admitted. The sampler is built from the slot's per-request
    /// `GenerationConfig` captured at `enqueue` time, so temperature /
    /// top-p / penalties are honored per request.
    fn admit_if_needed(&mut self, slot: SlotId) -> Result<(), Exception> {
        if self.state.batch_idx(slot).is_some() {
            return Ok(());
        }
        let cfg = self
            .records
            .get(&slot)
            .map(|r| r.gen_config.clone())
            .unwrap_or_default();
        let sampler = Sampler::new(cfg);
        self.state.admit(slot, sampler)?;
        let prefix_seed = self
            .records
            .get_mut(&slot)
            .and_then(|r| r.prefix_seed.take());
        if let Some(cache) = prefix_seed {
            self.state.replace_cache(slot, cache)?;
        }
        Ok(())
    }

    fn save_prefix_if_possible(&mut self, slot: SlotId) {
        let Some(prefix_cache) = self.prefix_cache.as_ref().cloned() else {
            return;
        };
        let Some((tokens, prefix_hit_len)) = self.records.get(&slot).map(|rec| {
            let end = rec.prompt_tokens.min(rec.all_tokens.len());
            (rec.all_tokens[..end].to_vec(), rec.prefix_hit_len)
        }) else {
            return;
        };
        let Some(cache) = self.state.cache_for(slot) else {
            return;
        };
        match prefix_cache.lock() {
            Ok(mut guard) => {
                guard.insert(&tokens, cache);
                if prefix_hit_len > 0 {
                    tracing::debug!(
                        target: "pmetal_serve::continuous_pump",
                        restored = prefix_hit_len,
                        saved = tokens.len(),
                        "saved extended prefix after continuous-batch hit"
                    );
                }
            }
            Err(_) => {
                tracing::warn!(
                    target: "pmetal_serve::continuous_pump",
                    "skipping prefix-cache save after poisoned lock"
                );
            }
        }
    }

    /// Compute per-token logprob data from `last_logits` when the slot
    /// opted in. Mirrors `run_async_decode` so batched results are
    /// byte-for-byte compatible with the single-slot streaming path.
    fn compute_logprob_entry(
        last_logits: &Array,
        token: u32,
        top_n: Option<usize>,
    ) -> Option<TokenLogprobEntry> {
        let top_n = top_n?;
        match token_logprobs(last_logits, token, top_n + 1) {
            Ok((lp, mut top)) => {
                top.retain(|(tok, _)| *tok != token);
                top.truncate(top_n);
                Some(TokenLogprobEntry {
                    token,
                    logprob: lp,
                    top_logprobs: top,
                })
            }
            Err(_) => None,
        }
    }

    /// Record the first decode token produced at end-of-prefill — sends
    /// it through the slot channel and seeds `current_token` for the
    /// next decode batch.
    fn record_first_token(
        &mut self,
        slot: SlotId,
        token: u32,
        logprob: Option<TokenLogprobEntry>,
    ) -> Result<(), Exception> {
        let stop_seq_match = self.emit_token(slot, token, logprob)?;
        // Feed the same token into the scheduler so it counts toward
        // `max_new_tokens` and triggers stop-token / stop-sequence
        // retirement.
        self.batcher.advance_decode([(slot, token, stop_seq_match)]);
        Ok(())
    }

    /// Send `token` down the slot's channel, updating emitted count and
    /// per-slot histories. Returns `true` if the slot's generated text
    /// now ends in one of its `stop_sequences` — callers must surface
    /// the flag to the scheduler so the slot retires.
    fn emit_token(
        &mut self,
        slot: SlotId,
        token: u32,
        logprob: Option<TokenLogprobEntry>,
    ) -> Result<bool, Exception> {
        // Phase 1: update record and send the token. We isolate the
        // `&mut self.records` borrow so we can drop it before taking
        // `&mut self.batcher` in the close-detection path.
        let (closed, stop_seq_match) = {
            let rec = match self.records.get_mut(&slot) {
                Some(r) => r,
                None => return Ok(false),
            };
            rec.current_token = Some(token);
            rec.emitted = rec.emitted.saturating_add(1);
            rec.all_tokens.push(token);
            rec.generated.push(token);
            if rec.first_token_ms.is_none() {
                rec.first_token_ms = Some(rec.start.elapsed().as_secs_f64() * 1000.0);
            }

            let send_res = rec.tx.try_send(TokenEvent::Token { id: token, logprob });
            let closed = matches!(send_res, Err(mpsc::error::TrySendError::Closed(_)));

            // Stop-sequence detection (skipped when the pump has no
            // tokenizer or the slot has no raw-text stops configured).
            let stop_seq_match = if closed || rec.stop_sequences.is_empty() {
                false
            } else if let Some(tokenizer) = self.tokenizer.as_ref() {
                match detect_stop_sequence_suffix(
                    tokenizer.as_ref(),
                    &rec.generated,
                    &rec.stop_sequences,
                ) {
                    Some(n_strip) => {
                        rec.stripped_tokens = n_strip;
                        true
                    }
                    None => false,
                }
            } else {
                false
            };

            (closed, stop_seq_match)
        };

        if closed {
            self.batcher.cancel(slot);
            return Ok(false);
        }
        Ok(stop_seq_match)
    }

    /// Move every retired slot out of the batcher, free its row, and
    /// send a `Done`/`Error` event to its channel.
    fn drain_retired(&mut self) {
        let retired = self.batcher.drain_retired();
        for slot in retired {
            let finish_reason = match slot.finish_reason {
                Some(FinishReason::Length) => "length",
                Some(FinishReason::Stop) => "stop",
                Some(FinishReason::StopSequence) => "stop",
                Some(FinishReason::Cancelled) => "cancelled",
                None => "length",
            };
            self.state.retire(slot.id);
            if let Some(rec) = self.records.remove(&slot.id) {
                if finish_reason != "cancelled" {
                    let elapsed = rec.start.elapsed();
                    let total_ms = elapsed.as_secs_f64() * 1000.0;
                    // `completion_tokens` reports the visible tail —
                    // tokens matched by a stop sequence are stripped
                    // from the client-facing count.
                    let visible = rec.emitted.saturating_sub(rec.stripped_tokens);
                    let metrics = crate::engine::RequestMetrics {
                        first_token_latency_ms: rec.first_token_ms.unwrap_or(0.0),
                        total_latency_ms: total_ms,
                        tokens_per_second: if elapsed.as_secs_f64() > 0.0 {
                            visible as f64 / elapsed.as_secs_f64()
                        } else {
                            0.0
                        },
                        prompt_tokens: rec.prompt_tokens,
                        completion_tokens: visible,
                    };
                    let _ = rec.tx.try_send(TokenEvent::Done {
                        finish_reason: finish_reason.to_string(),
                        metrics,
                        stripped_tokens: rec.stripped_tokens,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::Array;
    use pmetal_mlx::kv_cache::KVCache;

    fn cache_cfg() -> KVCacheConfig {
        KVCacheConfig::new(2, 32, 4, 64)
    }

    fn cache_with_len(seq_len: usize) -> KVCache {
        let mut cache = KVCache::new(cache_cfg());
        if seq_len > 0 {
            let k = Array::zeros_f32(&[1, 4, seq_len as i32, 64]);
            let v = Array::zeros_f32(&[1, 4, seq_len as i32, 64]);
            cache.update_and_fetch(0, &k, &v).unwrap();
            cache.update_and_fetch(1, &k, &v).unwrap();
        }
        cache
    }

    /// Forward stub identical to the driver's unit tests: extends the
    /// cache by `len(tokens)` and returns logits whose last-position
    /// argmax is determined by `next`.
    struct Stub {
        vocab: i32,
        next: HashMap<u32, u32>,
    }

    impl SlotForward for Stub {
        fn forward(&mut self, tokens: &[u32], cache: &mut KVCache) -> Result<Array, Exception> {
            let (h, d) = {
                let c = cache.config();
                (c.num_kv_heads as i32, c.head_dim as i32)
            };
            let s = tokens.len() as i32;
            let k = Array::zeros_f32(&[1, h, s, d]);
            let v = Array::zeros_f32(&[1, h, s, d]);
            cache.update_and_fetch(0, &k, &v)?;
            cache.update_and_fetch(1, &k, &v)?;

            let next = self.next.get(&tokens[0]).copied().unwrap_or(0);
            let mut data = vec![0.0f32; (s * self.vocab) as usize];
            let last = ((s - 1) * self.vocab) as usize;
            data[last + next as usize] = 10.0;
            Ok(Array::from_f32_slice(&data, &[1, s, self.vocab]))
        }
    }

    fn params(max: usize, stop_tokens: Vec<u32>) -> SlotParams {
        SlotParams {
            max_new_tokens: max,
            stop_tokens,
            stop_sequences: vec![],
            prefill_step_size: 64,
            logprobs_top_n: None,
        }
    }

    #[tokio::test]
    async fn enqueue_tick_emits_done_on_length_limit() {
        let mut pump = ContinuousPump::new(BatcherConfig::default(), cache_cfg(), None);
        let mut next = HashMap::new();
        next.insert(1u32, 42u32); // prompt last tok 1 → sample 42
        next.insert(42u32, 43u32); // 42 → 43 (decode loop)
        next.insert(43u32, 44u32); // 43 → 44
        let mut stub = Stub { vocab: 64, next };

        // max=2 tokens — so we expect 2 tokens then Done(length).
        let (_slot, mut rx) = pump
            .enqueue(
                vec![1, 2, 3],
                params(2, vec![]),
                GenerationConfig::default(),
                16,
            )
            .unwrap();

        // Drive the pump until receiver closes.
        let mut tokens = Vec::new();
        let mut finish: Option<String> = None;
        loop {
            let tick = pump.tick(&mut stub).unwrap();
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    TokenEvent::Token { id, .. } => tokens.push(id),
                    TokenEvent::Done { finish_reason, .. } => finish = Some(finish_reason),
                    TokenEvent::Error(e) => panic!("unexpected error: {e}"),
                }
            }
            if finish.is_some() {
                break;
            }
            if tick == Tick::Idle {
                break;
            }
        }

        assert_eq!(tokens, vec![42, 43]);
        assert_eq!(finish.as_deref(), Some("length"));
    }

    #[tokio::test]
    async fn continuous_batching_reuses_and_extends_prefix_cache() {
        struct RecordingStub {
            calls: Vec<Vec<u32>>,
        }

        impl SlotForward for RecordingStub {
            fn forward(&mut self, tokens: &[u32], cache: &mut KVCache) -> Result<Array, Exception> {
                self.calls.push(tokens.to_vec());
                let (h, d) = {
                    let c = cache.config();
                    (c.num_kv_heads as i32, c.head_dim as i32)
                };
                let s = tokens.len() as i32;
                let k = Array::zeros_f32(&[1, h, s, d]);
                let v = Array::zeros_f32(&[1, h, s, d]);
                cache.update_and_fetch(0, &k, &v)?;
                cache.update_and_fetch(1, &k, &v)?;

                let vocab = 16i32;
                let mut data = vec![0.0f32; (s * vocab) as usize];
                let last = ((s - 1) * vocab) as usize;
                data[last + 7] = 10.0;
                Ok(Array::from_f32_slice(&data, &[1, s, vocab]))
            }
        }

        let prefix_cache = Arc::new(Mutex::new(ServePrefixCache::new(8, 0)));
        prefix_cache
            .lock()
            .unwrap()
            .insert(&[1, 2], &cache_with_len(2));

        let mut pump = ContinuousPump::new_with_prefix_cache(
            BatcherConfig::default(),
            cache_cfg(),
            None,
            Some(Arc::clone(&prefix_cache)),
        );
        let mut stub = RecordingStub { calls: Vec::new() };
        let (_slot, _rx) = pump
            .enqueue(
                vec![1, 2, 3],
                params(1, vec![]),
                GenerationConfig::default(),
                16,
            )
            .unwrap();

        assert_eq!(pump.tick(&mut stub).unwrap(), Tick::Ran);
        assert_eq!(
            stub.calls,
            vec![vec![3]],
            "prefill should run only the suffix"
        );

        let hit = prefix_cache
            .lock()
            .unwrap()
            .find_longest_prefix(&[1, 2, 3, 4], cache_cfg())
            .unwrap()
            .expect("extended prompt should be cached after prefill");
        assert_eq!(hit.prefix_len, 3);
        assert_eq!(hit.restored_cache.seq_len(), 3);
    }

    #[tokio::test]
    async fn stop_token_short_circuits_generation() {
        let mut pump = ContinuousPump::new(BatcherConfig::default(), cache_cfg(), None);
        let mut next = HashMap::new();
        next.insert(1u32, 7u32); // first decode emits 7 (stop)
        let mut stub = Stub { vocab: 16, next };

        let (_s, mut rx) = pump
            .enqueue(
                vec![1, 2, 3],
                params(16, vec![7]),
                GenerationConfig::default(),
                16,
            )
            .unwrap();

        let mut tokens = Vec::new();
        let mut finish: Option<String> = None;
        for _ in 0..20 {
            let tick = pump.tick(&mut stub).unwrap();
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    TokenEvent::Token { id, .. } => tokens.push(id),
                    TokenEvent::Done { finish_reason, .. } => finish = Some(finish_reason),
                    TokenEvent::Error(e) => panic!("err: {e}"),
                }
            }
            if finish.is_some() || tick == Tick::Idle {
                break;
            }
        }
        // Stop token 7 was emitted once (it's also counted toward max,
        // but we retire on Stop first). Actually the first-token path
        // records the token THEN the scheduler checks stop_tokens, so
        // the stop token DOES appear in the output stream exactly
        // once. That matches single-slot behavior where the stop
        // token is returned to the caller.
        assert_eq!(tokens, vec![7], "stop token emitted before retirement");
        assert_eq!(finish.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn two_slots_interleave_through_pump() {
        let mut pump = ContinuousPump::new(
            BatcherConfig {
                max_slots: 2,
                max_queue_depth: 16,
                ..BatcherConfig::default()
            },
            cache_cfg(),
            None,
        );
        let mut next = HashMap::new();
        next.insert(1u32, 10u32);
        next.insert(10u32, 11u32);
        next.insert(2u32, 20u32);
        next.insert(20u32, 21u32);
        let mut stub = Stub { vocab: 64, next };

        let (_a, mut rx_a) = pump
            .enqueue(vec![1], params(2, vec![]), GenerationConfig::default(), 16)
            .unwrap();
        let (_b, mut rx_b) = pump
            .enqueue(vec![2], params(2, vec![]), GenerationConfig::default(), 16)
            .unwrap();

        let mut a_tokens = Vec::new();
        let mut b_tokens = Vec::new();
        let mut a_done = false;
        let mut b_done = false;
        for _ in 0..40 {
            let tick = pump.tick(&mut stub).unwrap();
            while let Ok(ev) = rx_a.try_recv() {
                match ev {
                    TokenEvent::Token { id, .. } => a_tokens.push(id),
                    TokenEvent::Done { .. } => a_done = true,
                    TokenEvent::Error(e) => panic!("a err: {e}"),
                }
            }
            while let Ok(ev) = rx_b.try_recv() {
                match ev {
                    TokenEvent::Token { id, .. } => b_tokens.push(id),
                    TokenEvent::Done { .. } => b_done = true,
                    TokenEvent::Error(e) => panic!("b err: {e}"),
                }
            }
            if a_done && b_done {
                break;
            }
            if tick == Tick::Idle {
                break;
            }
        }

        assert_eq!(a_tokens, vec![10, 11]);
        assert_eq!(b_tokens, vec![20, 21]);
        assert!(a_done && b_done);
    }

    #[tokio::test]
    async fn idle_tick_when_no_work() {
        let mut pump = ContinuousPump::new(BatcherConfig::default(), cache_cfg(), None);
        let mut stub = Stub {
            vocab: 4,
            next: HashMap::new(),
        };
        assert_eq!(pump.tick(&mut stub).unwrap(), Tick::Idle);
    }

    #[tokio::test]
    async fn cancel_before_prefill_drops_request() {
        let mut pump = ContinuousPump::new(
            BatcherConfig {
                max_slots: 1,
                max_queue_depth: 16,
                ..BatcherConfig::default()
            },
            cache_cfg(),
            None,
        );
        let mut stub = Stub {
            vocab: 4,
            next: HashMap::new(),
        };

        let (_a, _rx_a) = pump
            .enqueue(vec![1], params(4, vec![]), GenerationConfig::default(), 16)
            .unwrap();
        let (b, rx_b) = pump
            .enqueue(vec![2], params(4, vec![]), GenerationConfig::default(), 16)
            .unwrap();
        drop(rx_b);
        assert!(pump.cancel(b));

        // The queue should still show `a` as pending or active; `b` is gone.
        assert!(pump.pending_depth() <= 1);
        // A single tick against the empty forward: since b was cancelled
        // and never admitted, there's nothing special to verify beyond
        // not panicking.
        let _ = pump.tick(&mut stub);
    }

    #[tokio::test]
    async fn logprobs_attached_when_requested() {
        let mut pump = ContinuousPump::new(BatcherConfig::default(), cache_cfg(), None);
        let mut next = HashMap::new();
        next.insert(1u32, 5u32);
        next.insert(5u32, 6u32);
        let mut stub = Stub { vocab: 16, next };

        let slot_params = SlotParams {
            max_new_tokens: 2,
            stop_tokens: vec![],
            stop_sequences: vec![],
            prefill_step_size: 64,
            logprobs_top_n: Some(3),
        };
        let (_s, mut rx) = pump
            .enqueue(vec![1], slot_params, GenerationConfig::default(), 16)
            .unwrap();

        let mut entries: Vec<(u32, Option<TokenLogprobEntry>)> = Vec::new();
        let mut done = false;
        for _ in 0..20 {
            let tick = pump.tick(&mut stub).unwrap();
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    TokenEvent::Token { id, logprob } => entries.push((id, logprob)),
                    TokenEvent::Done { .. } => done = true,
                    TokenEvent::Error(e) => panic!("err: {e}"),
                }
            }
            if done || tick == Tick::Idle {
                break;
            }
        }
        assert!(done, "stream must terminate");
        assert_eq!(entries.len(), 2);
        for (token, entry) in &entries {
            let e = entry.as_ref().expect("logprob requested → entry required");
            assert_eq!(e.token, *token);
            assert!(e.logprob <= 0.0, "log-softmax is non-positive");
            assert_eq!(e.top_logprobs.len(), 3);
            for (alt, _) in &e.top_logprobs {
                assert_ne!(*alt, *token, "alternatives exclude chosen token");
            }
        }
    }

    #[tokio::test]
    async fn penalties_suppress_repeated_token() {
        // Construct a stub whose argmax is always token 3 regardless of
        // input. Without a repetition penalty the pump would emit
        // `[3, 3, 3]`. With `repetition_penalty > 1`, the sampler's
        // post-penalty argmax should diverge after the first emit
        // because repeated tokens have been scaled down.
        struct BiasedStub {
            vocab: i32,
        }
        impl SlotForward for BiasedStub {
            fn forward(&mut self, tokens: &[u32], cache: &mut KVCache) -> Result<Array, Exception> {
                let (h, d) = {
                    let c = cache.config();
                    (c.num_kv_heads as i32, c.head_dim as i32)
                };
                let s = tokens.len() as i32;
                let k = Array::zeros_f32(&[1, h, s, d]);
                let v = Array::zeros_f32(&[1, h, s, d]);
                cache.update_and_fetch(0, &k, &v)?;
                cache.update_and_fetch(1, &k, &v)?;
                // Last-position logits: 3 > 2 > 1 > 0, all positive so
                // a big repetition penalty can push repeated 3 below 2.
                let mut data = vec![0.0f32; (s * self.vocab) as usize];
                let last = ((s - 1) * self.vocab) as usize;
                data[last + 3] = 5.0;
                data[last + 2] = 4.0;
                data[last + 1] = 3.0;
                data[last] = 2.0;
                Ok(Array::from_f32_slice(&data, &[1, s, self.vocab]))
            }
        }

        let mut pump = ContinuousPump::new(BatcherConfig::default(), cache_cfg(), None);
        let mut stub = BiasedStub { vocab: 8 };

        let mut cfg = GenerationConfig::greedy(3);
        cfg.repetition_penalty = 5.0;
        let (_s, mut rx) = pump.enqueue(vec![3], params(3, vec![]), cfg, 16).unwrap();

        let mut tokens = Vec::new();
        let mut done = false;
        for _ in 0..20 {
            let tick = pump.tick(&mut stub).unwrap();
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    TokenEvent::Token { id, .. } => tokens.push(id),
                    TokenEvent::Done { .. } => done = true,
                    TokenEvent::Error(e) => panic!("err: {e}"),
                }
            }
            if done || tick == Tick::Idle {
                break;
            }
        }
        assert!(done);
        // Prompt contains `3`, so the first generated token already
        // sees `3` as a repeat and picks `2` instead.
        assert_eq!(
            tokens.first(),
            Some(&2u32),
            "repetition penalty should push the sampler away from 3"
        );
    }
}
