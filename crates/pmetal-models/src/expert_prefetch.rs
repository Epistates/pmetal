//! Pre-gated expert prediction for SSD-offloaded MoE inference.
//!
//! Uses layer N's pre-attention hidden states to predict layer N+1's experts,
//! dispatching background pread() calls while the GPU computes the current layer.
//!
//! 84-93% hit rate per 2025 papers (expert selection is highly predictable
//! from pre-attention representations).
//!
//! # How it works
//!
//! ```text
//! Layer N:  hidden_states ──┬── [GPU] attention + MoE ──► output
//!                           │
//!                           └── [CPU] predict_and_prefetch(N+1)
//!                                  │  (returns immediately)
//!                                  └── [background thread] pread experts
//!                                                    │
//! Layer N+1: try_get(actual experts) ◄───────────────┘
//!            hit  → reuse prefetched raw bytes
//!            miss → synchronous pread fallback
//! ```
//!
//! `ExpertPrefetcher` is wired into the Qwen3Next offloaded MoE path. Prefetch
//! hits now preserve the aligned Metal expert path by copying the raw expert
//! bytes into aligned GPU-visible buffers before dispatch.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread::{self, JoinHandle};

use pmetal_bridge::compat::{Array, Dtype};
use pmetal_metal::expert_buffer::{AlignedBuffer, ExpertBufferPool};

use crate::expert_io::ExpertOffloadContext;

/// Pre-gated expert prediction engine.
///
/// Maintains gate weight matrices for each MoE layer and a cache of
/// prefetched expert buffers. Thread-safe for concurrent predict/consume.
pub struct ExpertPrefetcher {
    /// Gate weight matrices for each MoE layer, indexed by layer_idx.
    /// Shape: `[num_experts, hidden_dim]` flattened row-major.
    gate_weights: HashMap<usize, Vec<f32>>,
    /// Number of experts per layer.
    num_experts: usize,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Top-k experts to prefetch.
    top_k: usize,
    /// Prefetch results: layer_idx → PrefetchResult.
    /// Background threads write here; try_get reads and removes.
    pending: Arc<Mutex<HashMap<usize, PrefetchResult>>>,
    /// Layers with a currently in-flight prefetch request, scoped by generation.
    inflight_layers: Arc<Mutex<HashMap<usize, u64>>>,
    /// Generation counter used to invalidate stale prefetch results across phases.
    generation: Arc<AtomicU64>,
    /// Persistent background worker pool for prefetch I/O.
    worker_pool: PrefetchIoWorkerPool,
    /// Shared aligned buffer pool for zero-copy prefetched expert hits.
    aligned_pool: Option<Arc<ExpertBufferPool>>,
    /// Hit/miss statistics.
    stats: Mutex<PrefetchStats>,
}

struct PrefetchRequest {
    layer_idx: usize,
    predicted_indices: Vec<usize>,
    offload_ctx: Arc<ExpertOffloadContext>,
    aligned_pool: Option<Arc<ExpertBufferPool>>,
    generation: u64,
}

struct PrefetchIoWorkerPool {
    request_tx: Mutex<Option<mpsc::Sender<PrefetchRequest>>>,
    joins: Mutex<Vec<JoinHandle<()>>>,
    num_workers: usize,
}

/// Cached prefetch result for a layer.
#[derive(Debug)]
struct PrefetchResult {
    /// Expert indices that were predicted and prefetched.
    predicted_indices: Vec<usize>,
    /// Raw byte buffers, one per predicted expert.
    /// Ownership is transferred out on hit (Vec::swap_remove), not cloned.
    buffers: Vec<Option<PrefetchedExpert>>,
}

/// Prefetched expert payload ready for a future sparse-layer lookup.
pub enum PrefetchedExpert {
    /// Raw byte payload. Used when the aligned pool is unavailable.
    Raw(Vec<u8>),
    /// Already resident in a pooled aligned Metal-shared buffer.
    Aligned(PrefetchedAlignedBuffer),
}

impl std::fmt::Debug for PrefetchedExpert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raw(bytes) => f
                .debug_struct("PrefetchedExpert::Raw")
                .field("len", &bytes.len())
                .finish(),
            Self::Aligned(buf) => f
                .debug_tuple("PrefetchedExpert::Aligned")
                .field(buf)
                .finish(),
        }
    }
}

/// A prefetched aligned expert buffer that returns itself to the pool on drop.
pub struct PrefetchedAlignedBuffer {
    pool: Arc<ExpertBufferPool>,
    buffer: Option<AlignedBuffer>,
}

impl PrefetchedAlignedBuffer {
    fn new(pool: Arc<ExpertBufferPool>, buffer: AlignedBuffer) -> Self {
        Self {
            pool,
            buffer: Some(buffer),
        }
    }

    pub fn into_inner(mut self) -> AlignedBuffer {
        self.buffer
            .take()
            .expect("prefetched aligned buffer was already taken")
    }

    pub fn to_vec(&self, len: usize) -> Vec<u8> {
        match self.buffer.as_ref() {
            Some(buffer) => {
                let bytes = buffer.as_bytes();
                bytes[..len.min(bytes.len())].to_vec()
            }
            None => Vec::new(),
        }
    }
}

impl Drop for PrefetchedAlignedBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

impl std::fmt::Debug for PrefetchedAlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefetchedAlignedBuffer")
            .field(
                "size",
                &self
                    .buffer
                    .as_ref()
                    .map(|buffer| buffer.size())
                    .unwrap_or(0),
            )
            .finish_non_exhaustive()
    }
}

/// Prefetch hit/miss statistics.
#[derive(Debug, Default, Clone)]
pub struct PrefetchStats {
    /// Number of experts that were correctly predicted and prefetched.
    pub hits: usize,
    /// Number of experts that needed synchronous fallback.
    pub misses: usize,
    /// Total prefetch attempts.
    pub total: usize,
}

impl PrefetchStats {
    /// Hit rate as a fraction [0, 1].
    pub fn hit_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.hits as f64 / self.total as f64
        }
    }
}

fn complete_prefetch(
    pending: &Arc<Mutex<HashMap<usize, PrefetchResult>>>,
    generation: &Arc<AtomicU64>,
    layer_idx: usize,
    predicted_indices: Vec<usize>,
    buffers: Vec<PrefetchedExpert>,
    request_generation: u64,
) {
    if generation.load(AtomicOrdering::Relaxed) != request_generation {
        return;
    }
    let mut pending = lock_unpoisoned(pending);
    pending.insert(
        layer_idx,
        PrefetchResult {
            predicted_indices,
            buffers: buffers.into_iter().map(Some).collect(),
        },
    );
}

fn try_prefetch_aligned(
    layer_idx: usize,
    predicted_indices: &[usize],
    offload_ctx: &Arc<ExpertOffloadContext>,
    pool: &Arc<ExpertBufferPool>,
) -> Option<Vec<PrefetchedExpert>> {
    let mut aligned = Vec::with_capacity(predicted_indices.len());
    for _ in 0..predicted_indices.len() {
        let buf = match pool.try_acquire() {
            Some(buf) => buf,
            None => {
                for buf in aligned.drain(..) {
                    pool.release(buf);
                }
                return None;
            }
        };
        aligned.push(buf);
    }

    if offload_ctx
        .read_experts_aligned(layer_idx, predicted_indices, &mut aligned)
        .is_err()
    {
        for buf in aligned.drain(..) {
            pool.release(buf);
        }
        return None;
    }

    Some(
        aligned
            .into_iter()
            .map(|buf| PrefetchedExpert::Aligned(PrefetchedAlignedBuffer::new(pool.clone(), buf)))
            .collect(),
    )
}

fn try_mark_inflight(
    inflight: &Mutex<HashMap<usize, u64>>,
    layer_idx: usize,
    generation: u64,
) -> bool {
    let mut inflight = lock_unpoisoned(inflight);
    if inflight.get(&layer_idx).copied() == Some(generation) {
        return false;
    }
    inflight.insert(layer_idx, generation);
    true
}

fn clear_inflight(inflight: &Mutex<HashMap<usize, u64>>, layer_idx: usize, generation: u64) {
    let mut inflight = lock_unpoisoned(inflight);
    if inflight.get(&layer_idx).copied() == Some(generation) {
        inflight.remove(&layer_idx);
    }
}

impl PrefetchIoWorkerPool {
    fn new(
        pending: Arc<Mutex<HashMap<usize, PrefetchResult>>>,
        inflight_layers: Arc<Mutex<HashMap<usize, u64>>>,
        generation: Arc<AtomicU64>,
        num_workers: usize,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<PrefetchRequest>();
        let request_rx = Arc::new(Mutex::new(request_rx));
        let mut joins = Vec::with_capacity(num_workers);

        for worker_idx in 0..num_workers {
            let request_rx = Arc::clone(&request_rx);
            let pending = Arc::clone(&pending);
            let inflight_layers = Arc::clone(&inflight_layers);
            let generation = Arc::clone(&generation);
            let join = thread::Builder::new()
                .name(format!("prefetch-io-{worker_idx}"))
                .spawn(move || {
                    loop {
                        let request = {
                            let rx = lock_unpoisoned(&request_rx);
                            rx.recv()
                        };
                        let Ok(request) = request else {
                            break;
                        };

                        let buffers = if let Some(pool) = request.aligned_pool.as_ref() {
                            if let Some(buffers) = try_prefetch_aligned(
                                request.layer_idx,
                                &request.predicted_indices,
                                &request.offload_ctx,
                                pool,
                            ) {
                                buffers
                            } else {
                                match request
                                    .offload_ctx
                                    .read_experts(request.layer_idx, &request.predicted_indices)
                                {
                                    Ok(bufs) => {
                                        bufs.into_iter().map(PrefetchedExpert::Raw).collect()
                                    }
                                    Err(_) => {
                                        clear_inflight(
                                            &inflight_layers,
                                            request.layer_idx,
                                            request.generation,
                                        );
                                        continue;
                                    }
                                }
                            }
                        } else {
                            match request
                                .offload_ctx
                                .read_experts(request.layer_idx, &request.predicted_indices)
                            {
                                Ok(bufs) => bufs.into_iter().map(PrefetchedExpert::Raw).collect(),
                                Err(_) => {
                                    clear_inflight(
                                        &inflight_layers,
                                        request.layer_idx,
                                        request.generation,
                                    );
                                    continue;
                                }
                            }
                        };
                        complete_prefetch(
                            &pending,
                            &generation,
                            request.layer_idx,
                            request.predicted_indices,
                            buffers,
                            request.generation,
                        );
                        clear_inflight(&inflight_layers, request.layer_idx, request.generation);
                    }
                });
            match join {
                Ok(join) => joins.push(join),
                Err(err) => {
                    tracing::warn!("failed to spawn prefetch-io worker {}: {}", worker_idx, err)
                }
            }
        }

        Self {
            request_tx: Mutex::new(Some(request_tx)),
            joins: Mutex::new(joins),
            num_workers,
        }
    }

    fn enqueue(&self, request: PrefetchRequest) -> bool {
        let tx = lock_unpoisoned(&self.request_tx);
        if let Some(tx) = tx.as_ref() {
            tx.send(request).is_ok()
        } else {
            false
        }
    }
}

impl Drop for PrefetchIoWorkerPool {
    fn drop(&mut self) {
        let _ = lock_unpoisoned(&self.request_tx).take();
        for join in lock_unpoisoned(&self.joins).drain(..) {
            let _ = join.join();
        }
    }
}

impl std::fmt::Debug for PrefetchIoWorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefetchIoWorkerPool")
            .field("num_workers", &self.num_workers)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ExpertPrefetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertPrefetcher")
            .field("num_layers", &self.gate_weights.len())
            .field("num_experts", &self.num_experts)
            .field("hidden_dim", &self.hidden_dim)
            .field("top_k", &self.top_k)
            .field("aligned_pool", &self.aligned_pool.is_some())
            .field(
                "inflight_layers",
                &lock_unpoisoned(&self.inflight_layers).len(),
            )
            .field("worker_pool", &self.worker_pool)
            .finish()
    }
}

impl ExpertPrefetcher {
    /// Create a new prefetcher.
    ///
    /// `gate_weights` maps layer_idx to the exact gate weight matrix
    /// (shape `[num_experts, hidden_dim]`) flattened row-major.
    pub fn new(
        gate_weights: HashMap<usize, Vec<f32>>,
        num_experts: usize,
        hidden_dim: usize,
        top_k: usize,
        aligned_pool: Option<Arc<ExpertBufferPool>>,
    ) -> Self {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let inflight_layers = Arc::new(Mutex::new(HashMap::new()));
        let generation = Arc::new(AtomicU64::new(0));
        Self {
            gate_weights,
            num_experts,
            hidden_dim,
            top_k,
            worker_pool: PrefetchIoWorkerPool::new(
                Arc::clone(&pending),
                Arc::clone(&inflight_layers),
                Arc::clone(&generation),
                prefetch_worker_count(),
            ),
            aligned_pool,
            pending,
            inflight_layers,
            generation,
            stats: Mutex::new(PrefetchStats::default()),
        }
    }

    fn enqueue_prefetch_indices(
        &self,
        layer_idx: usize,
        predicted_indices: Vec<usize>,
        offload_ctx: &Arc<ExpertOffloadContext>,
    ) {
        if predicted_indices.is_empty() {
            return;
        }
        let generation = self.generation.load(AtomicOrdering::Relaxed);

        // Multiple call sites can target the same layer. Keep only one
        // outstanding prefetch per target layer at a time.
        if !try_mark_inflight(&self.inflight_layers, layer_idx, generation) {
            return;
        }

        let request = PrefetchRequest {
            layer_idx,
            predicted_indices,
            offload_ctx: offload_ctx.clone(),
            aligned_pool: self.aligned_pool.clone(),
            generation,
        };
        if !self.worker_pool.enqueue(request) {
            clear_inflight(&self.inflight_layers, layer_idx, generation);
        }
    }

    /// Predict next-layer experts and dispatch background pread (non-blocking).
    ///
    /// Returns immediately after enqueueing work on the persistent IO thread.
    /// The prefetched
    /// buffers will be available via `try_get` once the IO completes.
    ///
    /// For T=1 decode, `hidden` is `[1, D]` — the exact gate projection is tiny.
    pub fn predict_and_prefetch(
        &self,
        next_layer_idx: usize,
        hidden: &Array,
        offload_ctx: &Arc<ExpertOffloadContext>,
    ) {
        let Some(gate_w) = self.gate_weights.get(&next_layer_idx) else {
            return;
        };

        let predicted = match self.predict_topk(hidden, gate_w) {
            Ok(indices) => indices,
            Err(_) => return,
        };

        self.enqueue_prefetch_indices(next_layer_idx, predicted, offload_ctx);
    }

    /// Check if prefetch hit for the given layer and expert indices.
    ///
    /// Returns buffers for experts that were correctly predicted (ownership
    /// transferred, not cloned), and `None` for experts needing sync fallback.
    ///
    /// The returned Vec has the same length and order as `expert_indices`.
    pub fn try_get(
        &self,
        layer_idx: usize,
        expert_indices: &[usize],
    ) -> Vec<Option<PrefetchedExpert>> {
        let mut pending = lock_unpoisoned(&self.pending);
        let prefetch = pending.remove(&layer_idx);

        let mut stats = lock_unpoisoned(&self.stats);

        match prefetch {
            Some(mut result) => {
                // Build index map: predicted_expert_idx → buffer_idx
                let mut idx_map: HashMap<usize, usize> = HashMap::new();
                for (i, &eidx) in result.predicted_indices.iter().enumerate() {
                    idx_map.insert(eidx, i);
                }

                let mut out = Vec::with_capacity(expert_indices.len());
                for &eidx in expert_indices {
                    stats.total += 1;
                    if let Some(&buf_idx) = idx_map.get(&eidx) {
                        // Take ownership — no clone
                        if let Some(buf) = result.buffers[buf_idx].take() {
                            stats.hits += 1;
                            out.push(Some(buf));
                        } else {
                            // Already taken (duplicate expert index)
                            stats.misses += 1;
                            out.push(None);
                        }
                    } else {
                        stats.misses += 1;
                        out.push(None);
                    }
                }
                out
            }
            None => {
                // No prefetch was done for this layer (or IO still in flight)
                stats.total += expert_indices.len();
                stats.misses += expert_indices.len();
                expert_indices.iter().map(|_| None).collect()
            }
        }
    }

    /// Get current prefetch statistics.
    pub fn stats(&self) -> PrefetchStats {
        lock_unpoisoned(&self.stats).clone()
    }

    /// Reset statistics counters.
    pub fn reset_stats(&self) {
        *lock_unpoisoned(&self.stats) = PrefetchStats::default();
    }

    /// Drop any cached / in-flight prefetch state from a previous phase.
    pub fn reset_pending(&self) {
        self.generation.fetch_add(1, AtomicOrdering::Relaxed);
        lock_unpoisoned(&self.pending).clear();
        lock_unpoisoned(&self.inflight_layers).clear();
    }

    /// CPU-side top-k prediction over logically extracted gate rows.
    fn predict_topk(
        &self,
        hidden: &Array,
        gate_w: &[f32],
    ) -> Result<Vec<usize>, pmetal_bridge::compat::Exception> {
        let d = self.hidden_dim as i32;
        let k = self.top_k;

        let hidden_rows = hidden.reshape(&[-1, d]);
        let last_row_idx = hidden_rows.dim(0) - 1;
        let hidden_1d =
            pmetal_bridge::compat::ops::slice_axis(&hidden_rows, 0, last_row_idx, last_row_idx + 1)
                .squeeze_axes(&[0]);
        let hidden_1d = if hidden_1d.dtype() != Dtype::Float32 {
            hidden_1d.as_type::<f32>()
        } else {
            hidden_1d
        };
        hidden_1d.eval();
        let hidden_values: &[f32] = hidden_1d.as_slice();

        let mut scored: Vec<(f32, usize)> = gate_w
            .chunks_exact(self.hidden_dim)
            .enumerate()
            .map(|(expert_idx, row)| {
                let score = row
                    .iter()
                    .zip(hidden_values.iter())
                    .fold(0.0f32, |acc, (w, h)| acc + (*w * *h));
                (score, expert_idx)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        Ok(scored
            .into_iter()
            .map(|(_, expert_idx)| expert_idx)
            .collect())
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("recovering from poisoned ExpertPrefetcher mutex");
        poisoned.into_inner()
    })
}

fn prefetch_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().clamp(2, 3))
        .unwrap_or(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use pmetal_bridge::compat::nn;
    use serial_test::serial;

    use crate::architectures::qwen3_next::Qwen3NextConfig;

    fn raw_prefetched(bytes: &[u8]) -> Option<PrefetchedExpert> {
        Some(PrefetchedExpert::Raw(bytes.to_vec()))
    }

    fn first_byte(prefetched: &PrefetchedExpert) -> u8 {
        match prefetched {
            PrefetchedExpert::Raw(bytes) => bytes[0],
            PrefetchedExpert::Aligned(bytes) => bytes.to_vec(1)[0],
        }
    }

    #[test]
    #[serial]
    fn test_predict_topk_basic() {
        let hidden_dim = 16;
        let num_experts = 4;
        let top_k = 2;

        let gate_w: Vec<f32> = (0..num_experts * hidden_dim)
            .map(|i| ((i * 7 + 3) % 97) as f32 / 97.0 - 0.5)
            .collect();

        let mut gate_weights = HashMap::new();
        gate_weights.insert(0, gate_w);

        let prefetcher = ExpertPrefetcher::new(gate_weights, num_experts, hidden_dim, top_k, None);

        let hidden = Array::from_slice(&vec![1.0f32; hidden_dim], &[hidden_dim as i32]);

        let gate_w = prefetcher.gate_weights.get(&0).unwrap();
        let predicted = prefetcher.predict_topk(&hidden, gate_w).unwrap();

        assert_eq!(predicted.len(), top_k);
        for &idx in &predicted {
            assert!(idx < num_experts, "Expert index {} out of range", idx);
        }
        let mut sorted = predicted.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), top_k, "Predicted indices should be unique");
    }

    #[test]
    #[serial]
    fn test_predict_topk_matches_qwen3next_gate_projection() {
        let config = Qwen3NextConfig {
            hidden_size: 8,
            num_experts: 6,
            num_experts_per_tok: 2,
            ..Default::default()
        };

        let mut gate = nn::LinearBuilder::new(config.hidden_size, config.num_experts)
            .bias(false)
            .build()
            .unwrap();
        let gate_weight = Array::from_slice(
            &[
                0.10f32, 0.20, -0.30, 0.40, 0.50, -0.60, 0.70, -0.80, //
                -0.20, 0.10, 0.30, -0.40, 0.60, 0.20, -0.50, 0.90, //
                0.80, -0.70, 0.20, 0.10, -0.30, 0.40, 0.50, 0.60, //
                -0.90, 0.30, 0.20, 0.70, -0.40, 0.50, -0.10, 0.20, //
                0.40, 0.60, -0.80, 0.10, 0.20, -0.30, 0.90, 0.50, //
                -0.50, -0.10, 0.40, 0.80, -0.20, 0.70, 0.30, -0.60, //
            ],
            &[config.num_experts, config.hidden_size],
        );
        *gate.weight = gate_weight;

        let hidden = Array::from_slice(
            &[
                0.25f32, -0.50, 0.75, 0.10, -0.20, 0.30, 0.40, -0.60, //
                -0.15, 0.35, 0.55, -0.45, 0.65, -0.25, 0.85, 0.05, //
            ],
            &[2, config.hidden_size],
        );

        let raw_weight = gate.weight.as_ref().as_type::<f32>().unwrap();
        raw_weight.eval().unwrap();
        let mut gate_weights = HashMap::new();
        gate_weights.insert(0, raw_weight.as_slice().to_vec());
        let prefetcher = ExpertPrefetcher::new(
            gate_weights,
            config.num_experts as usize,
            config.hidden_size as usize,
            config.num_experts_per_tok as usize,
            None,
        );

        let predicted = prefetcher
            .predict_topk(&hidden, prefetcher.gate_weights.get(&0).unwrap())
            .unwrap();
        let gate_logits = gate.forward(&hidden).unwrap();
        let last_row =
            pmetal_bridge::compat::ops::select_axis(&gate_logits, gate_logits.dim(0) - 1, 0);
        let last_row = last_row.as_type::<f32>().unwrap();
        let mut actual_scores: Vec<(f32, usize)> = (0..config.num_experts as usize)
            .map(|expert_idx| (last_row.index(expert_idx as i32).item::<f32>(), expert_idx))
            .collect();
        actual_scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        actual_scores.truncate(config.num_experts_per_tok as usize);
        let actual: Vec<usize> = actual_scores.iter().map(|(_, idx)| *idx).collect();

        assert_eq!(
            predicted.iter().copied().collect::<BTreeSet<_>>(),
            actual.iter().copied().collect::<BTreeSet<_>>(),
            "predicted={predicted:?} actual={actual:?}"
        );
    }

    #[test]
    #[serial]
    fn test_try_get_hit_miss() {
        let prefetcher = ExpertPrefetcher::new(HashMap::new(), 4, 16, 2, None);

        // Manually insert a prefetch result
        {
            let mut pending = prefetcher.pending.lock().unwrap();
            pending.insert(
                5,
                PrefetchResult {
                    predicted_indices: vec![2, 7],
                    buffers: vec![raw_prefetched(&[0xAA; 100]), raw_prefetched(&[0xBB; 100])],
                },
            );
        }

        // Query with partial overlap
        let results = prefetcher.try_get(5, &[2, 3, 7]);

        assert_eq!(results.len(), 3);
        assert!(results[0].is_some()); // expert 2 was prefetched
        assert_eq!(first_byte(results[0].as_ref().unwrap()), 0xAA);
        assert!(results[1].is_none()); // expert 3 was NOT prefetched
        assert!(results[2].is_some()); // expert 7 was prefetched
        assert_eq!(first_byte(results[2].as_ref().unwrap()), 0xBB);

        let stats = prefetcher.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total, 3);
    }

    #[test]
    #[serial]
    fn test_try_get_no_prefetch() {
        let prefetcher = ExpertPrefetcher::new(HashMap::new(), 4, 16, 2, None);

        let results = prefetcher.try_get(3, &[0, 1]);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_none());
        assert!(results[1].is_none());

        let stats = prefetcher.stats();
        assert_eq!(stats.misses, 2);
    }

    #[test]
    #[serial]
    fn test_try_get_ownership_transfer() {
        let prefetcher = ExpertPrefetcher::new(HashMap::new(), 4, 16, 2, None);

        {
            let mut pending = prefetcher.pending.lock().unwrap();
            pending.insert(
                1,
                PrefetchResult {
                    predicted_indices: vec![0],
                    buffers: vec![raw_prefetched(&[0xFF; 50])],
                },
            );
        }

        // First call takes ownership
        let results = prefetcher.try_get(1, &[0]);
        assert!(results[0].is_some());

        // Second call finds nothing (already consumed)
        let results2 = prefetcher.try_get(1, &[0]);
        assert!(results2[0].is_none());
    }

    #[test]
    #[serial]
    fn test_complete_prefetch_replaces_prior_layer_result() {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let generation = Arc::new(AtomicU64::new(0));

        complete_prefetch(
            &pending,
            &generation,
            7,
            vec![1],
            vec![PrefetchedExpert::Raw(vec![0x11; 4])],
            0,
        );
        complete_prefetch(
            &pending,
            &generation,
            7,
            vec![3],
            vec![PrefetchedExpert::Raw(vec![0x33; 4])],
            0,
        );

        let mut guard = pending.lock().unwrap();
        let result = guard.remove(&7).unwrap();
        assert_eq!(result.predicted_indices, vec![3]);
        assert_eq!(result.buffers.len(), 1);
        assert_eq!(first_byte(result.buffers[0].as_ref().unwrap()), 0x33);
    }

    #[test]
    #[serial]
    fn test_complete_prefetch_ignores_stale_generation() {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let generation = Arc::new(AtomicU64::new(1));

        complete_prefetch(
            &pending,
            &generation,
            7,
            vec![1],
            vec![PrefetchedExpert::Raw(vec![0x11; 4])],
            0,
        );

        assert!(pending.lock().unwrap().is_empty());
    }

    #[test]
    #[serial]
    fn test_reset_stats_clears_counters() {
        let prefetcher = ExpertPrefetcher::new(HashMap::new(), 4, 16, 2, None);

        {
            let mut pending = prefetcher.pending.lock().unwrap();
            pending.insert(
                2,
                PrefetchResult {
                    predicted_indices: vec![0],
                    buffers: vec![raw_prefetched(&[0xAB; 8])],
                },
            );
        }

        let _ = prefetcher.try_get(2, &[0, 1]);
        let before = prefetcher.stats();
        assert_eq!(before.hits, 1);
        assert_eq!(before.misses, 1);

        prefetcher.reset_stats();
        let after = prefetcher.stats();
        assert_eq!(after.hits, 0);
        assert_eq!(after.misses, 0);
        assert_eq!(after.total, 0);
    }

    #[test]
    #[serial]
    fn test_prefetcher_debug_includes_shape_metadata() {
        let prefetcher = ExpertPrefetcher::new(HashMap::new(), 8, 64, 2, None);
        let debug = format!("{prefetcher:?}");
        assert!(debug.contains("num_experts"));
        assert!(debug.contains("hidden_dim"));
        assert!(debug.contains("top_k"));
        assert!(debug.contains("inflight_layers"));
    }

    #[test]
    #[serial]
    fn test_try_mark_inflight_deduplicates_until_cleared() {
        let inflight = Mutex::new(HashMap::new());

        assert!(try_mark_inflight(&inflight, 11, 0));
        assert!(!try_mark_inflight(&inflight, 11, 0));
        assert!(try_mark_inflight(&inflight, 11, 1));

        clear_inflight(&inflight, 11, 0);
        assert!(!try_mark_inflight(&inflight, 11, 1));
        clear_inflight(&inflight, 11, 1);
        assert!(try_mark_inflight(&inflight, 11, 1));
    }

    #[test]
    #[serial]
    fn test_reset_pending_clears_pending_and_advances_generation() {
        let prefetcher = ExpertPrefetcher::new(HashMap::new(), 4, 16, 2, None);
        {
            let mut pending = prefetcher.pending.lock().unwrap();
            pending.insert(
                2,
                PrefetchResult {
                    predicted_indices: vec![0],
                    buffers: vec![raw_prefetched(&[0xAB; 8])],
                },
            );
        }
        {
            let mut inflight = prefetcher.inflight_layers.lock().unwrap();
            inflight.insert(2, 0);
        }

        let before = prefetcher.generation.load(AtomicOrdering::Relaxed);
        prefetcher.reset_pending();
        let after = prefetcher.generation.load(AtomicOrdering::Relaxed);

        assert_eq!(after, before + 1);
        assert!(prefetcher.pending.lock().unwrap().is_empty());
        assert!(prefetcher.inflight_layers.lock().unwrap().is_empty());
    }
}
