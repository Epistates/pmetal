//! Per-slot caches and the decode step.
//!
//! This module bridges the pure-Rust [`crate::continuous_batch`]
//! scheduler and the MLX model forward pass. It owns:
//!
//! - A [`SlotIdxMap`] that tracks which `SlotId` lives in which batch row.
//! - A [`ContinuousEngineState`] that bundles the KV caches with a per-slot
//!   [`DecodeSampler`] so each slot has its own temperature / top-p settings
//!   and its own penalty history on the GPU.
//! - A [`SlotForward`] trait that decouples the driver from `DynamicModel`
//!   so unit tests can stub the forward pass.
//! - Drivers [`drive_prefill_step`], [`drive_decode_step`] and
//!   [`drive_fused_decode_step`] that the pump calls with the instructions
//!   produced by [`crate::continuous_batch::ContinuousBatcher::next_step`].
//!
//! # Two decode paths
//!
//! A request prefills on its own [`KVCache`], one chunk at a time. What
//! happens after that depends on the model:
//!
//! - **Fused.** For a model with a fused `[N, 1]` forward and a plain dense
//!   cache ([`DynamicModel::supports_fused_batched`]), the finished prefill is
//!   copied into a row of a shared [`FusedBatchKVCache`]
//!   ([`ContinuousEngineState::promote`]) and the request's own cache is
//!   freed. A decode step is then **one** forward over every decoding
//!   request, reading the weights once however many there are
//!   ([`drive_fused_decode_step`]).
//! - **Per slot.** Everything else (hybrid models with recurrent state,
//!   quantized, TurboQuant and sliding-window caches, architectures without a
//!   fused forward) decodes one forward per slot on the slot's own cache
//!   ([`drive_decode_step`]), the forwards queued back to back on the GPU
//!   before anything is read.
//!
//! Both sample the same way: each slot's sampler draws its token from its own
//! row of the logits, and the step reads the sampled tokens back in one go,
//! the only values to leave the GPU.
//!
//! [`DynamicModel::supports_fused_batched`]: pmetal_models::dispatcher::DynamicModel::supports_fused_batched

use crate::continuous_batch::SlotId;
use pmetal_cache::kv_cache::{BatchKVCache, FusedBatchKVCache, KVCache, KVCacheConfig, MambaCache};
use pmetal_models::generation::GenerationConfig;
use pmetal_runtime::compat::{Array, Dtype, Exception, ops};
use pmetal_runtime::decode::DecodeSampler;
use std::collections::HashMap;

/// Trait that the driver uses to invoke the model's forward pass.
///
/// Kept generic so unit tests can inject a stub without instantiating
/// a full `DynamicModel`. Real call sites pass a closure wrapping
/// `DynamicModel::forward_with_hybrid_cache`.
pub trait SlotForward {
    /// Forward `tokens` (shape `[1, tokens.len()]`) through the model,
    /// extending `cache` with the produced K/V pairs and, for a hybrid
    /// model, advancing the slot's `recurrent` (Mamba / gated-delta-net)
    /// state. Returns the *lazy* logits tensor, whose last position is the one
    /// sampled from (`[1, 1, vocab]` from `forward_with_hybrid_cache_last`, so
    /// the head runs on that position alone) — the driver batches
    /// `async_eval` across all slots so the caller doesn't pay a host sync
    /// per slot.
    fn forward(
        &mut self,
        tokens: &[u32],
        cache: &mut KVCache,
        recurrent: Option<&mut MambaCache>,
    ) -> Result<Array, Exception>;

    /// One fused `[N_active, 1]` decode forward over every active slot,
    /// writing into the shared [`FusedBatchKVCache`]. `input_ids` is the
    /// `[N_active, 1]` int32 token ids and `active_indices` their rows in the
    /// cache; the logits are `[N_active, 1, vocab]`.
    ///
    /// Called only for a pump built with
    /// [`enable_fused_decode`](crate::continuous_pump::ContinuousPump::enable_fused_decode),
    /// which the engine does for models that support it.
    fn forward_batched(
        &mut self,
        _input_ids: &Array,
        _active_indices: &[usize],
        _cache: &mut FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        Err(Exception::custom(
            "this forward has no fused batched decode path",
        ))
    }
}

impl<F> SlotForward for F
where
    F: FnMut(&[u32], &mut KVCache, Option<&mut MambaCache>) -> Result<Array, Exception>,
{
    fn forward(
        &mut self,
        tokens: &[u32],
        cache: &mut KVCache,
        recurrent: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        self(tokens, cache, recurrent)
    }
}

/// Bidirectional `SlotId ↔ batch_idx` map.
///
/// The batched KV cache indexes slots by compact integer positions
/// (`batch_idx ∈ [0, max_slots)`), while the scheduler identifies them
/// by opaque `SlotId`s. This map owns the allocation of batch rows,
/// re-using freed rows for new slots to keep the cache compact.
#[derive(Debug, Default)]
pub struct SlotIdxMap {
    by_slot: HashMap<SlotId, usize>,
    by_idx: Vec<Option<SlotId>>,
}

impl SlotIdxMap {
    /// Create a map sized for `capacity` batch rows, all initially free.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            by_slot: HashMap::with_capacity(capacity),
            by_idx: vec![None; capacity],
        }
    }

    /// Return `Some(existing_idx)` if `slot` is already mapped, else
    /// allocate the lowest-numbered free row and return its index.
    /// Returns `None` when every row is occupied.
    pub fn allocate(&mut self, slot: SlotId) -> Option<usize> {
        if let Some(&idx) = self.by_slot.get(&slot) {
            return Some(idx);
        }
        let free = self.by_idx.iter().position(|s| s.is_none())?;
        self.by_idx[free] = Some(slot);
        self.by_slot.insert(slot, free);
        Some(free)
    }

    /// Look up the batch row for a slot, if allocated.
    pub fn get(&self, slot: SlotId) -> Option<usize> {
        self.by_slot.get(&slot).copied()
    }

    /// Free the row held by `slot`, returning its index so the caller
    /// can reset the underlying cache entry.
    pub fn release(&mut self, slot: SlotId) -> Option<usize> {
        let idx = self.by_slot.remove(&slot)?;
        self.by_idx[idx] = None;
        Some(idx)
    }

    /// Number of allocated rows.
    pub fn len(&self) -> usize {
        self.by_slot.len()
    }

    /// Whether any rows are allocated.
    pub fn is_empty(&self) -> bool {
        self.by_slot.is_empty()
    }

    /// Iterate `(SlotId, batch_idx)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (SlotId, usize)> + '_ {
        self.by_slot.iter().map(|(&s, &i)| (s, i))
    }
}

/// Continuous-batch engine state.
///
/// Holds the per-slot KV caches, per-slot samplers and penalties, and
/// slot→idx map. Samplers are stored as `Option` so rows can sit
/// "allocated but empty" briefly between release and the next allocate.
pub struct ContinuousEngineState {
    caches: BatchKVCache,
    /// Per-row recurrent (Mamba / gated-delta-net) state for a hybrid
    /// model, `None` for a pure-attention one. Reset when a row is freed,
    /// like its KV cache, so the next request starts from nothing.
    recurrent: Option<Vec<MambaCache>>,
    /// The shared KV cache the decoding slots live in when the model
    /// decodes fused. A slot prefills on its own cache in `caches` and moves
    /// here when its prefill ends; `None` when every slot decodes on its own.
    fused_cache: Option<FusedBatchKVCache>,
    /// Each slot's sampler: its penalties hold the slot's prompt and every
    /// token sampled for it on the GPU, so a slot's history lives with its
    /// row from admission to retirement and no other slot reads it.
    samplers: Vec<Option<DecodeSampler>>,
    map: SlotIdxMap,
}

impl ContinuousEngineState {
    /// Create engine state for `max_slots` concurrent slots, each with
    /// an empty KV cache configured by `cache_config`.
    pub fn new(max_slots: usize, cache_config: KVCacheConfig) -> Self {
        let caches = BatchKVCache::new(max_slots, cache_config);
        let samplers = (0..max_slots).map(|_| None).collect();
        let map = SlotIdxMap::with_capacity(max_slots);
        Self {
            caches,
            recurrent: None,
            fused_cache: None,
            samplers,
            map,
        }
    }

    /// Give every row its own recurrent state for a hybrid model, each a
    /// copy of the empty `template` (the model's `create_mamba_cache()`).
    pub fn with_recurrent_state(mut self, template: &MambaCache) -> Self {
        self.recurrent = Some((0..self.capacity()).map(|_| template.clone()).collect());
        self
    }

    /// Whether rows carry recurrent state (the model is hybrid).
    pub fn has_recurrent_state(&self) -> bool {
        self.recurrent.is_some()
    }

    /// Decode the slots that have finished prefilling together, as one fused
    /// forward over a shared [`FusedBatchKVCache`] (see the module docs).
    ///
    /// # Errors
    /// For a hybrid model: the fused forward carries no recurrent state.
    pub fn enable_fused_cache(&mut self, fused: FusedBatchKVCache) -> Result<(), Exception> {
        if self.recurrent.is_some() {
            return Err(Exception::custom(
                "fused decode carries no recurrent state; hybrid models decode per slot",
            ));
        }
        self.fused_cache = Some(fused);
        Ok(())
    }

    /// Whether the decoding slots share a fused KV cache.
    pub fn has_fused_cache(&self) -> bool {
        self.fused_cache.is_some()
    }

    /// The fused cache, if there is one.
    pub fn fused_cache(&self) -> Option<&FusedBatchKVCache> {
        self.fused_cache.as_ref()
    }

    /// Total number of rows (active + free).
    pub fn capacity(&self) -> usize {
        self.caches.batch_size()
    }

    /// Number of active slots.
    pub fn active(&self) -> usize {
        self.map.len()
    }

    /// Assign a slot to a free batch row, attaching the sampler of a
    /// generation under `config` from `prompt` (the whole prompt, including
    /// any part restored from a prefix cache).
    ///
    /// Returns the allocated `batch_idx`, or an error if all rows are
    /// in use. Callers must check [`Self::active`] or
    /// [`crate::continuous_batch::BatcherConfig::max_slots`] before
    /// enqueueing to the scheduler; hitting this error indicates a
    /// bookkeeping bug.
    pub fn admit(
        &mut self,
        slot: SlotId,
        config: &GenerationConfig,
        prompt: &[u32],
    ) -> Result<usize, Exception> {
        self.admit_sampler(slot, config.sampler(prompt))
    }

    /// [`admit`](Self::admit) with the slot's sampler already built, for a
    /// request whose sampler carries more than its configuration makes (a
    /// grammar's logits processor).
    pub fn admit_sampler(
        &mut self,
        slot: SlotId,
        sampler: DecodeSampler,
    ) -> Result<usize, Exception> {
        let idx = self.map.allocate(slot).ok_or_else(|| {
            Exception::custom(format!(
                "ContinuousEngineState::admit: no free batch row for {:?} (capacity {})",
                slot,
                self.capacity()
            ))
        })?;
        self.samplers[idx] = Some(sampler);
        if let Some(fused) = self.fused_cache.as_mut() {
            fused.release(idx);
        }
        Ok(idx)
    }

    /// Move a slot whose prefill has ended into the fused cache, where it
    /// decodes with the others, and free the cache it prefilled on. Does
    /// nothing when decode is per slot.
    pub fn promote(&mut self, slot: SlotId) -> Result<(), Exception> {
        let Some(fused) = self.fused_cache.as_mut() else {
            return Ok(());
        };
        let idx = self
            .map
            .get(slot)
            .ok_or_else(|| Exception::custom(format!("promote: slot {slot:?} not admitted")))?;
        let own = self
            .caches
            .get_mut(idx)
            .ok_or_else(|| Exception::custom(format!("promote: no cache for slot {slot:?}")))?;
        fused.load_slot(idx, own)?;
        self.caches.reset_indices(&[idx]);
        Ok(())
    }

    /// Release a slot's row, resetting the cache entry so the next
    /// admit starts with a clean state.
    pub fn retire(&mut self, slot: SlotId) {
        if let Some(idx) = self.map.release(slot) {
            self.samplers[idx] = None;
            self.caches.reset_indices(&[idx]);
            if let Some(recurrent) = self.recurrent.as_mut() {
                recurrent[idx].reset();
            }
            if let Some(fused) = self.fused_cache.as_mut() {
                fused.release(idx);
                // Nothing in flight: hold no K/V.
                if self.map.is_empty() {
                    fused.release_if_idle();
                }
            }
        }
    }

    /// Look up the batch row for a slot.
    pub fn batch_idx(&self, slot: SlotId) -> Option<usize> {
        self.map.get(slot)
    }

    /// Borrow the per-slot KV cache mutably.
    pub fn cache_for(&mut self, slot: SlotId) -> Option<&mut KVCache> {
        let idx = self.map.get(slot)?;
        self.caches.get_mut(idx)
    }

    /// Borrow the slot's KV cache and, for a hybrid model, its recurrent
    /// state: everything a forward for that slot reads and advances.
    pub fn slot_caches(&mut self, slot: SlotId) -> Option<(&mut KVCache, Option<&mut MambaCache>)> {
        let idx = self.map.get(slot)?;
        let cache = self.caches.get_mut(idx)?;
        let recurrent = match self.recurrent.as_mut() {
            Some(rows) => Some(rows.get_mut(idx)?),
            None => None,
        };
        Some((cache, recurrent))
    }

    /// The slot's recurrent state, for a hybrid model.
    pub fn recurrent_for(&self, slot: SlotId) -> Option<&MambaCache> {
        let idx = self.map.get(slot)?;
        self.recurrent.as_ref()?.get(idx)
    }

    /// Replace a slot's recurrent state with the state a prefix left.
    pub fn replace_recurrent(&mut self, slot: SlotId, state: MambaCache) -> Result<(), Exception> {
        let idx = self.map.get(slot).ok_or_else(|| {
            Exception::custom(format!(
                "replace recurrent state: slot {slot:?} not admitted"
            ))
        })?;
        let rows = self
            .recurrent
            .as_mut()
            .ok_or_else(|| Exception::custom("replace recurrent state: the model carries none"))?;
        rows[idx] = state;
        Ok(())
    }

    /// Replace a slot's KV cache with a forked prefix cache.
    pub fn replace_cache(&mut self, slot: SlotId, cache: KVCache) -> Result<(), Exception> {
        let idx = self.map.get(slot).ok_or_else(|| {
            Exception::custom(format!("replace cache: slot {slot:?} not admitted"))
        })?;
        self.caches.replace(idx, cache)
    }

    /// Sample `slot`'s next token from its `[1, vocab]` last-position
    /// logits with the slot's sampler, after its repetition / frequency /
    /// presence penalties, and take the token into those penalties. The
    /// token stays lazy; the slot's next sample is penalized for it whether
    /// or not the host has read it.
    pub fn sample(&mut self, slot: SlotId, logits: &Array) -> Result<Array, Exception> {
        let sampler = self
            .map
            .get(slot)
            .and_then(|idx| self.samplers.get_mut(idx))
            .and_then(Option::as_mut)
            .ok_or_else(|| Exception::custom(format!("sample: slot {slot:?} has no sampler")))?;
        Ok(sampler.sample(logits))
    }

    /// Sample one token for each of `slots` from the matching row of
    /// `logits` (`[N, vocab]`): `[N]` uint32, lazy.
    ///
    /// Each slot's token is what its own sampler draws from its own row, so a
    /// request's tokens do not depend on who shares its step. Slots that
    /// decode by plain argmax (temperature 0, no penalty, bias or processor)
    /// take it together in one op over the whole block; the others go
    /// through their samplers a row at a time.
    pub fn sample_rows(&mut self, slots: &[SlotId], logits: &Array) -> Result<Array, Exception> {
        let rows = slots.len();
        let vocab = logits.dim(-1);
        let mut indices = Vec::with_capacity(rows);
        for &slot in slots {
            indices.push(self.map.get(slot).ok_or_else(|| {
                Exception::custom(format!("sample: slot {slot:?} has no sampler"))
            })?);
        }
        let plain: Vec<bool> = indices
            .iter()
            .map(|&i| {
                self.samplers[i]
                    .as_ref()
                    .is_some_and(DecodeSampler::is_plain_argmax)
            })
            .collect();
        let uint32 = Dtype::Uint32.as_i32();
        if plain.iter().all(|&p| p) {
            return Ok(logits.argmax(-1).as_dtype(uint32));
        }
        let argmax = plain
            .iter()
            .any(|&p| p)
            .then(|| logits.argmax(-1).as_dtype(uint32));
        let mut drawn = Vec::with_capacity(rows);
        for (row, &idx) in indices.iter().enumerate() {
            let row_i = row as i32;
            drawn.push(match (&argmax, plain[row]) {
                (Some(argmax), true) => argmax.slice(&[row_i], &[row_i + 1]),
                _ => {
                    let sampler = self.samplers[idx].as_mut().ok_or_else(|| {
                        Exception::custom(format!("sample: slot {:?} has no sampler", slots[row]))
                    })?;
                    let one = logits.slice(&[row_i, 0], &[row_i + 1, vocab]);
                    sampler.sample(&one).reshape(&[1]).as_dtype(uint32)
                }
            });
        }
        let refs: Vec<&Array> = drawn.iter().collect();
        Ok(ops::concatenate_axis(&refs, 0))
    }
}

/// Outcome of a prefill step: `Some(last_position_logits)` on the final
/// chunk, `None` while there are more chunks to process.
///
/// Returned logits are already evaluated (not lazy), so the caller can
/// hand them straight to the slot's sampler.
pub fn drive_prefill_step<F: SlotForward>(
    forward: &mut F,
    state: &mut ContinuousEngineState,
    slot: SlotId,
    chunk: &[u32],
    final_chunk: bool,
) -> Result<Option<Array>, Exception> {
    let (cache, recurrent) = state
        .slot_caches(slot)
        .ok_or_else(|| Exception::custom(format!("prefill: slot {slot:?} not admitted")))?;
    let logits = forward.forward(chunk, cache, recurrent)?;

    if final_chunk {
        let last = extract_last_logits(&logits)?;
        Ok(Some(last))
    } else {
        // Non-final chunk: we don't need the logits at all, but we
        // must evaluate the forward to materialize the cache update.
        logits
            .try_eval()
            .map_err(|e| Exception::custom(e.to_string()))?;
        Ok(None)
    }
}

/// Per-slot decode step output.
#[derive(Debug, Clone)]
pub struct SlotStepOutput {
    pub slot: SlotId,
    pub token: u32,
    /// The raw last-position logits the token was drawn from, `[1, vocab]`,
    /// lazy: read only by a slot that asked for log-probabilities.
    pub logits: Array,
}

/// Sample every slot's token from `logits` (`[N, vocab]`, row `i` for
/// `slots[i]`), evaluate the tokens in one go and read them back in one.
fn sample_and_read(
    state: &mut ContinuousEngineState,
    slots: &[SlotId],
    logits: &Array,
) -> Result<Vec<SlotStepOutput>, Exception> {
    let vocab = logits.dim(-1);
    let tokens = state.sample_rows(slots, logits)?;
    // The step's one eval, and the only values to leave the GPU.
    tokens
        .try_eval()
        .map_err(|e| Exception::custom(e.to_string()))?;
    let drawn = tokens.as_slice::<u32>();
    Ok(slots
        .iter()
        .enumerate()
        .map(|(row, &slot)| SlotStepOutput {
            slot,
            token: drawn[row],
            logits: logits.slice(&[row as i32, 0], &[row as i32 + 1, vocab]),
        })
        .collect())
}

/// Run a decode step across the given `(slot, current_token)` pairs, one
/// forward per slot on the slot's own cache.
///
/// Each slot's repetition / frequency / presence penalties read the
/// history the slot keeps on the GPU since [admission](ContinuousEngineState::admit),
/// so nothing about it is uploaded per step, and a slot whose
/// `GenerationConfig` has no penalties pays nothing.
///
/// The driver:
/// 1. Schedules `forward` for each slot (one token per slot) without
///    evaluating — the returned logits are lazy tensors.
/// 2. Samples the next token for each slot via its own sampler and
///    penalties ([`ContinuousEngineState::sample_rows`]), which take the
///    token into the slot's history lazily.
/// 3. Evaluates the sampled tokens once and reads them back once.
///
/// All forwards run on the caller's current default stream.
pub fn drive_decode_step<F: SlotForward>(
    forward: &mut F,
    state: &mut ContinuousEngineState,
    slots: &[(SlotId, u32)],
) -> Result<Vec<SlotStepOutput>, Exception> {
    if slots.is_empty() {
        return Ok(Vec::new());
    }
    // Schedule the forwards, collecting each slot's lazy last-position
    // logits as one row.
    let mut rows: Vec<Array> = Vec::with_capacity(slots.len());
    for &(slot, token) in slots {
        let (cache, recurrent) = state
            .slot_caches(slot)
            .ok_or_else(|| Exception::custom(format!("decode: slot {slot:?} not admitted")))?;
        let logits = forward.forward(&[token], cache, recurrent)?;
        rows.push(extract_last_logits(&logits)?);
    }
    let refs: Vec<&Array> = rows.iter().collect();
    let logits = if refs.len() == 1 {
        rows.pop().expect("one row")
    } else {
        ops::concatenate_axis(&refs, 0)
    };
    let ids: Vec<SlotId> = slots.iter().map(|&(slot, _)| slot).collect();
    sample_and_read(state, &ids, &logits)
}

/// Fused variant of [`drive_decode_step`].
///
/// Runs **one** `[N_active, 1]` forward across every active slot against
/// the shared [`FusedBatchKVCache`] held by `state`, then samples per
/// slot (penalty-aware, logprob-capable via the raw last-position logits
/// returned alongside each sampled token).
///
/// Prerequisites:
/// - `state.has_fused_cache()` must be true, and each slot must have been
///   [promoted](ContinuousEngineState::promote) into it.
/// - The model powering `forward` must implement
///   [`SlotForward::forward_batched`].
///
/// The `slots` pair format mirrors [`drive_decode_step`] so the pump
/// can hand the same inputs to either path.
pub fn drive_fused_decode_step<F: SlotForward>(
    forward: &mut F,
    state: &mut ContinuousEngineState,
    slots: &[(SlotId, u32)],
) -> Result<Vec<SlotStepOutput>, Exception> {
    if slots.is_empty() {
        return Ok(Vec::new());
    }
    if state.has_recurrent_state() {
        return Err(Exception::custom(
            "fused decode carries no recurrent state; hybrid models decode per slot",
        ));
    }

    // Resolve batch indices in the same order the caller passed.
    let mut active_indices: Vec<usize> = Vec::with_capacity(slots.len());
    let mut input_tokens: Vec<i32> = Vec::with_capacity(slots.len());
    for &(slot, token) in slots {
        let idx = state.batch_idx(slot).ok_or_else(|| {
            Exception::custom(format!("fused decode: slot {slot:?} not admitted"))
        })?;
        active_indices.push(idx);
        input_tokens.push(token as i32);
    }

    let n = slots.len() as i32;
    let input_ids = Array::from_i32_slice(&input_tokens).reshape(&[n, 1]);
    let fused = state.fused_cache.as_mut().ok_or_else(|| {
        Exception::custom("fused decode: ContinuousEngineState has no fused cache")
    })?;

    // [N, 1, vocab] -> one last-position row per slot, [N, vocab].
    let logits = forward
        .forward_batched(&input_ids, &active_indices, fused)?
        .squeeze_axes(&[1]);
    let ids: Vec<SlotId> = slots.iter().map(|&(slot, _)| slot).collect();
    sample_and_read(state, &ids, &logits)
}

/// Extract the last-position logits from a `[B, S, V]` tensor as a
/// `[B, V]` slice.
fn extract_last_logits(logits: &Array) -> Result<Array, Exception> {
    let shape = logits.shape();
    if shape.len() < 2 {
        return Err(Exception::custom(format!(
            "extract_last_logits: expected rank >= 2, got shape {shape:?}"
        )));
    }
    let seq_axis = shape.len() - 2;
    let seq_len = shape[seq_axis];
    if seq_len <= 0 {
        return Err(Exception::custom(format!(
            "extract_last_logits: empty sequence axis in {shape:?}"
        )));
    }
    let idx = Array::from_i32_slice(&[seq_len - 1]);
    Ok(logits
        .take_axis(&idx, seq_axis as i32)
        .squeeze_axes(&[seq_axis as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::Array;

    fn make_config() -> KVCacheConfig {
        KVCacheConfig::new(2, 32, 4, 64)
    }

    #[test]
    fn slot_idx_map_allocates_lowest_free_row() {
        let mut m = SlotIdxMap::with_capacity(3);
        assert_eq!(m.allocate(SlotId(1)), Some(0));
        assert_eq!(m.allocate(SlotId(2)), Some(1));
        assert_eq!(m.allocate(SlotId(3)), Some(2));
        // Full.
        assert_eq!(m.allocate(SlotId(4)), None);

        // Free middle row.
        assert_eq!(m.release(SlotId(2)), Some(1));
        assert_eq!(m.allocate(SlotId(4)), Some(1));
    }

    #[test]
    fn slot_idx_map_idempotent_allocate() {
        let mut m = SlotIdxMap::with_capacity(2);
        assert_eq!(m.allocate(SlotId(1)), Some(0));
        // Same slot — same row, no new allocation.
        assert_eq!(m.allocate(SlotId(1)), Some(0));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn continuous_state_admit_retire_roundtrip() {
        let mut state = ContinuousEngineState::new(2, make_config());
        let idx = state
            .admit(
                SlotId(1),
                &pmetal_models::generation::GenerationConfig::default(),
                &[],
            )
            .unwrap();
        assert_eq!(idx, 0);
        assert_eq!(state.active(), 1);
        assert_eq!(state.batch_idx(SlotId(1)), Some(0));

        state.retire(SlotId(1));
        assert_eq!(state.active(), 0);
        assert_eq!(state.batch_idx(SlotId(1)), None);
    }

    /// Stub forward that mirrors the real surface: takes tokens,
    /// updates the cache with a dummy K/V of the right shape, returns
    /// a `[1, S, V]` logits tensor whose last-position argmax lets us
    /// verify per-slot routing.
    struct StubForward {
        vocab: i32,
        /// Per-call logits to return, keyed by the first input token.
        /// The caller precomputes which token each slot should sample
        /// next and stashes it here.
        next_token_per_input: HashMap<u32, u32>,
        calls: usize,
    }

    impl SlotForward for StubForward {
        fn forward(
            &mut self,
            tokens: &[u32],
            cache: &mut KVCache,
            _recurrent: Option<&mut MambaCache>,
        ) -> Result<Array, Exception> {
            self.calls += 1;
            // Extend the cache a tiny bit so seq_len advances — mirrors
            // the real forward's side effect.
            let (num_heads, head_dim) = {
                let cfg = cache.config();
                (cfg.num_kv_heads as i32, cfg.head_dim as i32)
            };
            let s = tokens.len() as i32;
            let k = Array::zeros_f32(&[1, num_heads, s, head_dim]);
            let v = Array::zeros_f32(&[1, num_heads, s, head_dim]);
            cache.update_and_fetch(0, &k, &v)?;
            cache.update_and_fetch(1, &k, &v)?;

            // Build logits [1, S, V] where the last position scores
            // `next_token_per_input[tokens[0]]` at +10 and 0 elsewhere.
            let mut data = vec![0.0f32; (s * self.vocab) as usize];
            let next = self
                .next_token_per_input
                .get(&tokens[0])
                .copied()
                .unwrap_or(0);
            let last_row_start = ((s - 1) * self.vocab) as usize;
            data[last_row_start + next as usize] = 10.0;
            Ok(Array::from_f32_slice(&data, &[1, s, self.vocab]))
        }
    }

    /// Uniform logits over a 1000-token vocabulary, so every token is a
    /// random draw.
    struct UniformForward;

    impl SlotForward for UniformForward {
        fn forward(
            &mut self,
            tokens: &[u32],
            cache: &mut KVCache,
            _recurrent: Option<&mut MambaCache>,
        ) -> Result<Array, Exception> {
            let (num_heads, head_dim) = {
                let cfg = cache.config();
                (cfg.num_kv_heads as i32, cfg.head_dim as i32)
            };
            let s = tokens.len() as i32;
            let kv = Array::zeros_f32(&[1, num_heads, s, head_dim]);
            cache.update_and_fetch(0, &kv, &kv)?;
            cache.update_and_fetch(1, &kv, &kv)?;
            Ok(Array::zeros_f32(&[1, s, 1000]))
        }
    }

    fn sampling(seed: Option<u64>) -> GenerationConfig {
        let mut config = GenerationConfig::sampling(64, 1.0);
        config.seed = seed;
        config
    }

    /// Eight decode steps of `seeded` (seed 5), with `others` admitted
    /// beside it; returns the seeded slot's tokens.
    fn seeded_slot_tokens(others: &[Option<u64>]) -> Vec<u32> {
        let mut state = ContinuousEngineState::new(1 + others.len(), make_config());
        let seeded = SlotId(0);
        state.admit(seeded, &sampling(Some(5)), &[]).unwrap();
        let other_ids: Vec<SlotId> = (1..=others.len() as u64).map(SlotId).collect();
        for (&id, &seed) in other_ids.iter().zip(others) {
            state.admit(id, &sampling(seed), &[]).unwrap();
        }
        let mut tokens = Vec::new();
        for _ in 0..8 {
            let mut slots = vec![(seeded, 1u32)];
            slots.extend(other_ids.iter().map(|&id| (id, 1u32)));
            let outs = drive_decode_step(&mut UniformForward, &mut state, &slots).unwrap();
            tokens.push(outs.iter().find(|o| o.slot == seeded).unwrap().token);
        }
        tokens
    }

    #[test]
    fn a_seeded_slot_draws_the_same_tokens_whatever_shares_the_batch() {
        // Seeding used to reset the model thread's one MLX key at admission,
        // so every other slot's draw shifted the seeded slot's stream.
        let alone = seeded_slot_tokens(&[]);
        assert_eq!(
            alone,
            seeded_slot_tokens(&[]),
            "the same seed drew twice differently"
        );
        assert_eq!(
            alone,
            seeded_slot_tokens(&[None, Some(9)]),
            "an unseeded and a differently seeded slot moved the seeded slot's tokens"
        );
        pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
    }

    #[test]
    fn decode_step_routes_per_slot_logits_correctly() {
        let mut state = ContinuousEngineState::new(3, make_config());
        let s1 = SlotId(1);
        let s2 = SlotId(2);
        state
            .admit(
                s1,
                &pmetal_models::generation::GenerationConfig::default(),
                &[],
            )
            .unwrap();
        state
            .admit(
                s2,
                &pmetal_models::generation::GenerationConfig::default(),
                &[],
            )
            .unwrap();

        let mut next = HashMap::new();
        next.insert(100u32, 7u32); // slot 1 fed 100 → samples 7
        next.insert(200u32, 13u32); // slot 2 fed 200 → samples 13
        let mut stub = StubForward {
            vocab: 32,
            next_token_per_input: next,
            calls: 0,
        };

        let outs = drive_decode_step(&mut stub, &mut state, &[(s1, 100), (s2, 200)]).unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(stub.calls, 2, "one forward per slot");

        // Each slot must receive its own sampled token.
        let a = outs.iter().find(|o| o.slot == s1).unwrap().token;
        let b = outs.iter().find(|o| o.slot == s2).unwrap().token;
        assert_eq!(a, 7);
        assert_eq!(b, 13);
    }

    #[test]
    fn prefill_step_non_final_returns_none() {
        let mut state = ContinuousEngineState::new(2, make_config());
        let s = SlotId(10);
        state
            .admit(
                s,
                &pmetal_models::generation::GenerationConfig::default(),
                &[1, 2, 3],
            )
            .unwrap();

        let mut next = HashMap::new();
        next.insert(1u32, 0u32);
        let mut stub = StubForward {
            vocab: 8,
            next_token_per_input: next,
            calls: 0,
        };

        let out = drive_prefill_step(&mut stub, &mut state, s, &[1, 2, 3], false).unwrap();
        assert!(out.is_none());
        assert_eq!(stub.calls, 1);
    }

    #[test]
    fn prefill_step_final_returns_last_position_logits() {
        let mut state = ContinuousEngineState::new(2, make_config());
        let s = SlotId(20);
        state
            .admit(
                s,
                &pmetal_models::generation::GenerationConfig::default(),
                &[1, 2, 3],
            )
            .unwrap();

        let mut next = HashMap::new();
        next.insert(1u32, 0u32);
        let mut stub = StubForward {
            vocab: 8,
            next_token_per_input: next,
            calls: 0,
        };

        let out = drive_prefill_step(&mut stub, &mut state, s, &[1, 2, 3], true).unwrap();
        let logits = out.expect("final chunk must return logits");
        // Shape should be [1, V] after squeezing the 1-length seq axis.
        assert_eq!(logits.shape(), &[1, 8]);
    }

    #[test]
    fn decode_step_errors_for_unadmitted_slot() {
        let mut state = ContinuousEngineState::new(2, make_config());
        let mut stub = StubForward {
            vocab: 8,
            next_token_per_input: HashMap::new(),
            calls: 0,
        };
        let err = drive_decode_step(&mut stub, &mut state, &[(SlotId(99), 1)]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("not admitted"), "unexpected error: {msg}");
    }

    /// Records, per forward, the first input token, whether the recurrent
    /// state it was handed was empty, and which sequence that state was;
    /// then marks the state used, as a hybrid model's forward would.
    struct RecurrentStub {
        seen: Vec<(u32, bool, pmetal_cache::kv_cache::SequenceKey)>,
    }

    impl SlotForward for RecurrentStub {
        fn forward(
            &mut self,
            tokens: &[u32],
            _cache: &mut KVCache,
            recurrent: Option<&mut MambaCache>,
        ) -> Result<Array, Exception> {
            let recurrent = recurrent.expect("a hybrid slot gets recurrent state");
            self.seen
                .push((tokens[0], recurrent.is_empty(), recurrent.sequence_key()));
            recurrent.get_mut(0).unwrap().ssm_state = Some(Array::from_f32(1.0));
            let s = tokens.len() as i32;
            Ok(Array::from_f32_slice(
                &vec![0.0; (s * 4) as usize],
                &[1, s, 4],
            ))
        }
    }

    /// Each slot of a hybrid model advances its own recurrent state, and a
    /// row a finished request frees starts the next one from empty.
    #[test]
    fn hybrid_slots_each_get_their_own_recurrent_state() {
        let sampler = || pmetal_models::generation::GenerationConfig::default();
        let mut state =
            ContinuousEngineState::new(2, make_config()).with_recurrent_state(&MambaCache::new(1));
        let (a, b, c) = (SlotId(1), SlotId(2), SlotId(3));
        state.admit(a, &sampler(), &[10, 11]).unwrap();
        state.admit(b, &sampler(), &[20, 21]).unwrap();
        let mut stub = RecurrentStub { seen: Vec::new() };

        drive_prefill_step(&mut stub, &mut state, a, &[10, 11], true).unwrap();
        drive_prefill_step(&mut stub, &mut state, b, &[20, 21], true).unwrap();
        drive_decode_step(&mut stub, &mut state, &[(a, 12), (b, 22)]).unwrap();

        let first_four = stub.seen.clone();
        let [pa, pb, da, db] = &first_four[..] else {
            panic!("expected four forwards, got {}", stub.seen.len());
        };
        assert!(pa.1 && pb.1, "both prefills start from empty state");
        assert!(!da.1 && !db.1, "decodes continue from their prefill");
        assert!(pa.2 == da.2 && pb.2 == db.2);
        assert!(pa.2 != pb.2, "slots share no state");

        // `a` finishes; `c` takes its row and must not inherit its state.
        state.retire(a);
        assert_eq!(state.admit(c, &sampler(), &[30]).unwrap(), 0);
        drive_prefill_step(&mut stub, &mut state, c, &[30], true).unwrap();
        let pc = stub.seen.last().unwrap();
        assert!(pc.1, "a reused row starts from empty state");
        assert!(pc.2 != pa.2);
    }

    #[test]
    fn fused_decode_refuses_recurrent_state() {
        let mut state =
            ContinuousEngineState::new(1, make_config()).with_recurrent_state(&MambaCache::new(1));
        let fused = FusedBatchKVCache::new(make_config(), 1).unwrap();
        let err = state.enable_fused_cache(fused).expect_err("refused");
        assert!(format!("{err:?}").contains("recurrent"));
    }

    /// A forward with a fused path: prefill extends the slot's cache and
    /// decode extends the shared one, with logits whose argmax is
    /// `next[token]` for each row.
    struct FusedStub {
        vocab: i32,
        next: HashMap<u32, u32>,
        batched_calls: usize,
    }

    impl SlotForward for FusedStub {
        fn forward(
            &mut self,
            tokens: &[u32],
            cache: &mut KVCache,
            _recurrent: Option<&mut MambaCache>,
        ) -> Result<Array, Exception> {
            let (h, d) = {
                let c = cache.config();
                (c.num_kv_heads as i32, c.head_dim as i32)
            };
            let s = tokens.len() as i32;
            let kv = Array::zeros_f32(&[1, h, s, d]);
            cache.update_and_fetch(0, &kv, &kv)?;
            cache.update_and_fetch(1, &kv, &kv)?;
            Ok(Array::zeros_f32(&[1, s, self.vocab]))
        }

        fn forward_batched(
            &mut self,
            input_ids: &Array,
            active_indices: &[usize],
            cache: &mut FusedBatchKVCache,
        ) -> Result<Array, Exception> {
            self.batched_calls += 1;
            let n = active_indices.len() as i32;
            let (h, d) = (
                cache.config().num_kv_heads as i32,
                cache.config().head_dim as i32,
            );
            let kv = Array::zeros_f32(&[n, h, 1, d]);
            for layer in 0..cache.config().num_layers {
                cache.update_and_fetch_batched(layer, active_indices, &kv, &kv)?;
            }
            let tokens: Vec<i32> = input_ids.as_slice::<i32>().to_vec();
            let mut data = vec![0.0f32; n as usize * self.vocab as usize];
            for (row, token) in tokens.iter().enumerate() {
                let next = self.next.get(&(*token as u32)).copied().unwrap_or(0);
                data[row * self.vocab as usize + next as usize] = 10.0;
            }
            Ok(Array::from_f32_slice(&data, &[n, 1, self.vocab]))
        }
    }

    fn fused_state(max_slots: usize) -> ContinuousEngineState {
        let mut state = ContinuousEngineState::new(max_slots, make_config());
        state
            .enable_fused_cache(FusedBatchKVCache::new(make_config(), max_slots).unwrap())
            .unwrap();
        state
    }

    /// A slot that has prefilled moves into the fused cache with its tokens
    /// and its own cache is freed; one fused step then decodes every slot
    /// with a single forward and routes each row's logits to its own slot.
    #[test]
    fn a_fused_step_is_one_forward_and_each_slot_gets_its_own_row() {
        let mut state = fused_state(3);
        let greedy = GenerationConfig::default();
        let (a, b) = (SlotId(1), SlotId(2));
        state.admit(a, &greedy, &[1, 2, 3]).unwrap();
        state.admit(b, &greedy, &[4, 5]).unwrap();
        let mut stub = FusedStub {
            vocab: 32,
            next: HashMap::from([(100, 7), (200, 13)]),
            batched_calls: 0,
        };
        drive_prefill_step(&mut stub, &mut state, a, &[1, 2, 3], true).unwrap();
        drive_prefill_step(&mut stub, &mut state, b, &[4, 5], true).unwrap();
        state.promote(a).unwrap();
        state.promote(b).unwrap();
        let fused = state.fused_cache().unwrap();
        assert_eq!(
            (
                fused.offset(state.batch_idx(a).unwrap()),
                fused.offset(state.batch_idx(b).unwrap())
            ),
            (3, 2)
        );
        assert_eq!(
            state.cache_for(a).unwrap().seq_len(),
            0,
            "the prefill cache is freed"
        );

        let outs = drive_fused_decode_step(&mut stub, &mut state, &[(a, 100), (b, 200)]).unwrap();
        assert_eq!(stub.batched_calls, 1, "one forward for both slots");
        assert_eq!(outs.iter().find(|o| o.slot == a).unwrap().token, 7);
        assert_eq!(outs.iter().find(|o| o.slot == b).unwrap().token, 13);
        let fused = state.fused_cache().unwrap();
        assert_eq!(fused.offset(state.batch_idx(a).unwrap()), 4);

        // The last slot out gives the fused buffers back.
        state.retire(a);
        assert!(state.fused_cache().unwrap().memory_usage() > 0);
        state.retire(b);
        assert_eq!(state.fused_cache().unwrap().memory_usage(), 0);
        pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
    }

    /// A grammar-style logits processor that follows the tokens its slot
    /// takes: the next must be one more than the last.
    struct CountUp {
        next: u32,
        vocab: usize,
    }

    impl pmetal_runtime::logits_processor::LogitsProcessor for CountUp {
        fn process(&mut self, logits: &Array) -> Array {
            let mut bias = vec![-1e9f32; self.vocab];
            bias[self.next as usize] = 0.0;
            logits
                .as_dtype(Dtype::Float32.as_i32())
                .add(&Array::from_f32_slice(&bias, &[1, self.vocab as i32]))
        }
        fn accept(&mut self, token: u32) {
            self.next = token + 1;
        }
        fn needs_host_tokens(&self) -> bool {
            true
        }
        fn fork(&self) -> Box<dyn pmetal_runtime::logits_processor::LogitsProcessor> {
            Box::new(CountUp {
                next: self.next,
                vocab: self.vocab,
            })
        }
    }

    /// A slot whose sampler carries a processor that follows its tokens is
    /// sampled in the block like any other: its row is masked by its own
    /// state, step after step, and the plain-argmax slot beside it takes the
    /// argmax of its row.
    #[test]
    fn a_constrained_slot_is_sampled_in_the_block_beside_a_plain_one() {
        let vocab = 20usize;
        let mut state = ContinuousEngineState::new(2, make_config());
        let (constrained, plain) = (SlotId(1), SlotId(2));
        let greedy = GenerationConfig::greedy(8);
        state
            .admit_sampler(
                constrained,
                greedy
                    .sampler(&[1])
                    .with_processor(Box::new(CountUp { next: 5, vocab })),
            )
            .unwrap();
        state.admit(plain, &greedy, &[1]).unwrap();

        let mut rng = 9u64;
        let mut taken = Vec::new();
        for step in 0..6 {
            let data: Vec<f32> = (0..2 * vocab)
                .map(|_| {
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (rng >> 40) as f32 / (1u64 << 24) as f32
                })
                .collect();
            let logits = Array::from_f32_slice(&data, &[2, vocab as i32]);
            let block = state.sample_rows(&[constrained, plain], &logits).unwrap();
            let block = block.as_slice::<u32>().to_vec();
            taken.push(block[0]);
            let best = (0..vocab)
                .max_by(|&a, &b| data[vocab + a].total_cmp(&data[vocab + b]))
                .unwrap() as u32;
            assert_eq!(block[1], best, "step {step}: the plain slot's argmax");
        }
        assert_eq!(taken, vec![5, 6, 7, 8, 9, 10], "the grammar's own sequence");
        pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
    }

    /// Sampling a block of rows draws, for each slot, what that slot's own
    /// sampler draws from its own row alone: plain-argmax slots together, the
    /// seeded and penalized ones one by one, in any mix.
    #[test]
    fn sampling_a_block_of_rows_matches_sampling_each_alone() {
        let vocab = 50usize;
        let configs = [
            GenerationConfig::greedy(8),
            sampling(Some(11)),
            {
                let mut c = GenerationConfig::greedy(8);
                c.repetition_penalty = 1.5;
                c
            },
            GenerationConfig::greedy(8),
            sampling(Some(12)),
        ];
        let prompts: [&[u32]; 5] = [&[1], &[2, 3], &[4, 4, 5], &[6], &[7, 8]];
        let mut state = ContinuousEngineState::new(configs.len(), make_config());
        let ids: Vec<SlotId> = (0..configs.len() as u64).map(SlotId).collect();
        let mut alone: Vec<DecodeSampler> = Vec::new();
        for ((&id, config), prompt) in ids.iter().zip(&configs).zip(prompts) {
            state.admit(id, config, prompt).unwrap();
            alone.push(config.sampler(prompt));
        }
        let mut rng = 5u64;
        for step in 0..6 {
            let data: Vec<f32> = (0..configs.len() * vocab)
                .map(|_| {
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((rng >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
                })
                .collect();
            let logits = Array::from_f32_slice(&data, &[configs.len() as i32, vocab as i32]);
            let block = state.sample_rows(&ids, &logits).unwrap();
            let block = block.as_slice::<u32>().to_vec();
            for (row, sampler) in alone.iter_mut().enumerate() {
                let one = logits.slice(&[row as i32, 0], &[row as i32 + 1, vocab as i32]);
                assert_eq!(
                    block[row],
                    sampler.sample(&one).item::<u32>(),
                    "step {step}, row {row}"
                );
            }
        }
        pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
    }
}
