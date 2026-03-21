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
//!            hit  → take prefetched buffers (zero-copy, ownership transfer)
//!            miss → synchronous pread fallback
//! ```
//!
//! # Status: Not yet integrated
//!
//! `ExpertPrefetcher` needs to be wired into the MoE inference loop alongside
//! `ExpertOffloadContext`. Background pread prefetching is fully implemented,
//! but the prediction→consumption pipeline is not yet hooked into any
//! architecture's forward pass.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use crate::expert_io::ExpertOffloadContext;

/// Pre-gated expert prediction engine.
///
/// Maintains gate weight matrices for each MoE layer and a cache of
/// prefetched expert buffers. Thread-safe for concurrent predict/consume.
#[derive(Debug)]
pub struct ExpertPrefetcher {
    /// Gate weight matrices for each MoE layer, indexed by layer_idx.
    /// Shape: `[num_experts, hidden_dim]` (flattened row-major).
    gate_weights: HashMap<usize, Arc<Vec<f32>>>,
    /// Number of experts per layer.
    num_experts: usize,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Top-k experts to prefetch.
    top_k: usize,
    /// Prefetch results: layer_idx → PrefetchResult.
    /// Background threads write here; try_get reads and removes.
    pending: Arc<Mutex<HashMap<usize, PrefetchResult>>>,
    /// Hit/miss statistics.
    stats: Mutex<PrefetchStats>,
}

/// Cached prefetch result for a layer.
#[derive(Debug)]
struct PrefetchResult {
    /// Expert indices that were predicted and prefetched.
    predicted_indices: Vec<usize>,
    /// Raw byte buffers, one per predicted expert.
    /// Ownership is transferred out on hit (Vec::swap_remove), not cloned.
    buffers: Vec<Option<Vec<u8>>>,
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

impl ExpertPrefetcher {
    /// Create a new prefetcher.
    ///
    /// `gate_weights` maps layer_idx to the gate weight matrix (flattened,
    /// row-major, shape `[num_experts, hidden_dim]`). Uses `Arc<Vec<f32>>`
    /// to avoid duplicating the model's gate weights.
    pub fn new(
        gate_weights: HashMap<usize, Arc<Vec<f32>>>,
        num_experts: usize,
        hidden_dim: usize,
        top_k: usize,
    ) -> Self {
        Self {
            gate_weights,
            num_experts,
            hidden_dim,
            top_k,
            pending: Arc::new(Mutex::new(HashMap::new())),
            stats: Mutex::new(PrefetchStats::default()),
        }
    }

    /// Predict next-layer experts and dispatch background pread (non-blocking).
    ///
    /// Returns immediately after spawning the IO thread. The prefetched
    /// buffers will be available via `try_get` once the IO completes.
    ///
    /// For T=1 decode, `hidden` is `[1, D]` — the CPU-side matmul is trivial.
    pub fn predict_and_prefetch(
        &self,
        next_layer_idx: usize,
        hidden: &Array,
        offload_ctx: &Arc<ExpertOffloadContext>,
    ) {
        let Some(gate_w) = self.gate_weights.get(&next_layer_idx) else {
            return;
        };

        // CPU-side prediction (fast: D*E FLOPs, ~2M for Qwen3.5)
        let predicted = match self.predict_topk(hidden, gate_w) {
            Ok(indices) => indices,
            Err(_) => return,
        };

        // Spawn background IO thread — returns immediately
        let pending = self.pending.clone();
        let ctx = offload_ctx.clone();
        let predicted_clone = predicted.clone();

        thread::Builder::new()
            .name("prefetch-io".into())
            .spawn(move || {
                let buffers = match ctx.read_experts(next_layer_idx, &predicted_clone) {
                    Ok(bufs) => bufs.into_iter().map(Some).collect(),
                    Err(_) => return, // IO failed, skip
                };

                let mut pending = pending.lock().unwrap();
                pending.insert(
                    next_layer_idx,
                    PrefetchResult {
                        predicted_indices: predicted_clone,
                        buffers,
                    },
                );
            })
            .ok(); // If spawn fails, skip prefetch silently
    }

    /// Check if prefetch hit for the given layer and expert indices.
    ///
    /// Returns buffers for experts that were correctly predicted (ownership
    /// transferred, not cloned), and `None` for experts needing sync fallback.
    ///
    /// The returned Vec has the same length and order as `expert_indices`.
    pub fn try_get(&self, layer_idx: usize, expert_indices: &[usize]) -> Vec<Option<Vec<u8>>> {
        let mut pending = self.pending.lock().unwrap();
        let prefetch = pending.remove(&layer_idx);

        let mut stats = self.stats.lock().unwrap();

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
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics counters.
    pub fn reset_stats(&self) {
        *self.stats.lock().unwrap() = PrefetchStats::default();
    }

    /// CPU-side top-k prediction via matmul.
    fn predict_topk(
        &self,
        hidden: &Array,
        gate_w: &[f32],
    ) -> Result<Vec<usize>, mlx_rs::error::Exception> {
        let d = self.hidden_dim as i32;
        let e = self.num_experts as i32;
        let k = self.top_k;

        let hidden_1d = if hidden.ndim() > 1 {
            hidden.reshape(&[d])?
        } else {
            hidden.clone()
        };

        let gate_arr = Array::from_slice(gate_w, &[e, d]);
        let logits = mlx_rs::ops::matmul(&hidden_1d.reshape(&[1, d])?, &gate_arr.t())?;
        let logits_flat = logits.reshape(&[e])?;

        let probs = mlx_rs::ops::softmax_axis(&logits_flat, -1, None)?;
        let neg_probs = probs.negative()?;
        let neg_k = -(k as i32);
        let part = mlx_rs::ops::argpartition_axis(&neg_probs, neg_k, -1)?;
        let top_indices = part.index(neg_k..);
        let top_indices_i32 = top_indices.as_type::<i32>()?;

        top_indices_i32.eval()?;
        let indices: Vec<i32> = top_indices_i32.as_slice().to_vec();
        Ok(indices.iter().map(|&i| i as usize).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

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
        gate_weights.insert(0, Arc::new(gate_w));

        let prefetcher = ExpertPrefetcher::new(gate_weights, num_experts, hidden_dim, top_k);

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
    fn test_try_get_hit_miss() {
        let prefetcher = ExpertPrefetcher::new(HashMap::new(), 4, 16, 2);

        // Manually insert a prefetch result
        {
            let mut pending = prefetcher.pending.lock().unwrap();
            pending.insert(
                5,
                PrefetchResult {
                    predicted_indices: vec![2, 7],
                    buffers: vec![Some(vec![0xAA; 100]), Some(vec![0xBB; 100])],
                },
            );
        }

        // Query with partial overlap
        let results = prefetcher.try_get(5, &[2, 3, 7]);

        assert_eq!(results.len(), 3);
        assert!(results[0].is_some()); // expert 2 was prefetched
        assert_eq!(results[0].as_ref().unwrap()[0], 0xAA);
        assert!(results[1].is_none()); // expert 3 was NOT prefetched
        assert!(results[2].is_some()); // expert 7 was prefetched
        assert_eq!(results[2].as_ref().unwrap()[0], 0xBB);

        let stats = prefetcher.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total, 3);
    }

    #[test]
    #[serial]
    fn test_try_get_no_prefetch() {
        let prefetcher = ExpertPrefetcher::new(HashMap::new(), 4, 16, 2);

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
        let prefetcher = ExpertPrefetcher::new(HashMap::new(), 4, 16, 2);

        {
            let mut pending = prefetcher.pending.lock().unwrap();
            pending.insert(
                1,
                PrefetchResult {
                    predicted_indices: vec![0],
                    buffers: vec![Some(vec![0xFF; 50])],
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
}
