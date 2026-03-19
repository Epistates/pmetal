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
//!                                  │
//!                                  ├── gate_weights[N+1].T @ hidden → topk
//!                                  └── io_pool.parallel_read(predicted experts)
//!                                                    │
//! Layer N+1: try_get(actual experts) ◄───────────────┘
//!            hit  → use prefetched buffers (zero I/O latency)
//!            miss → synchronous pread fallback
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use crate::expert_io::ExpertOffloadContext;

/// Pre-gated expert prediction engine.
///
/// Maintains gate weight matrices for each MoE layer and a cache of
/// prefetched expert buffers. Thread-safe for concurrent predict/consume.
pub struct ExpertPrefetcher {
    /// Gate weight matrices for each MoE layer, indexed by layer_idx.
    /// Shape: `[hidden_dim, num_experts]` — transposed for fast matmul.
    gate_weights: HashMap<usize, Vec<f32>>,
    /// Number of experts per layer.
    num_experts: usize,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Top-k experts to prefetch.
    top_k: usize,
    /// Prefetch results: layer_idx → (predicted_expert_indices, raw_buffers).
    pending: Mutex<HashMap<usize, PrefetchResult>>,
    /// Hit/miss statistics.
    stats: Mutex<PrefetchStats>,
}

/// Cached prefetch result for a layer.
struct PrefetchResult {
    /// Expert indices that were predicted and prefetched.
    predicted_indices: Vec<usize>,
    /// Raw byte buffers, one per predicted expert (in same order as predicted_indices).
    buffers: Vec<Vec<u8>>,
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
    /// Gate weights should be extracted from the model's MoE gate Linear layers
    /// during model construction. Each entry maps layer_idx to the gate weight
    /// matrix (flattened, row-major, shape `[num_experts, hidden_dim]`).
    pub fn new(
        gate_weights: HashMap<usize, Vec<f32>>,
        num_experts: usize,
        hidden_dim: usize,
        top_k: usize,
    ) -> Self {
        Self {
            gate_weights,
            num_experts,
            hidden_dim,
            top_k,
            pending: Mutex::new(HashMap::new()),
            stats: Mutex::new(PrefetchStats::default()),
        }
    }

    /// Predict next-layer experts and dispatch background pread.
    ///
    /// Uses the current layer's hidden states to predict which experts
    /// will be selected at `next_layer_idx`. The prediction is done via
    /// CPU-side matmul (using Accelerate BLAS via MLX eval) while the GPU
    /// computes the current layer.
    ///
    /// For T=1 decode, `hidden` is `[1, D]` — the matmul is trivial.
    pub fn predict_and_prefetch(
        &self,
        next_layer_idx: usize,
        hidden: &Array,
        offload_ctx: &ExpertOffloadContext,
    ) {
        // Get gate weights for the next layer
        let Some(gate_w) = self.gate_weights.get(&next_layer_idx) else {
            return;
        };

        // CPU-side top-k prediction: hidden @ gate_weights.T → softmax → topk
        let predicted = match self.predict_topk(hidden, gate_w) {
            Ok(indices) => indices,
            Err(_) => return, // Prediction failed, skip prefetch
        };

        // Dispatch parallel pread for predicted experts
        let buffers = match offload_ctx.read_experts(next_layer_idx, &predicted) {
            Ok(bufs) => bufs,
            Err(_) => return, // IO failed, skip — will fall back to sync read
        };

        // Cache the prefetch result
        let mut pending = self.pending.lock().unwrap();
        pending.insert(
            next_layer_idx,
            PrefetchResult {
                predicted_indices: predicted,
                buffers,
            },
        );
    }

    /// Check if prefetch hit for the given layer and expert indices.
    ///
    /// Returns cached buffers for experts that were correctly predicted,
    /// and `None` entries for experts that need synchronous fallback.
    ///
    /// The returned Vec has the same length and order as `expert_indices`.
    pub fn try_get(
        &self,
        layer_idx: usize,
        expert_indices: &[usize],
    ) -> Vec<Option<Vec<u8>>> {
        let mut pending = self.pending.lock().unwrap();
        let prefetch = pending.remove(&layer_idx);

        let mut stats = self.stats.lock().unwrap();

        match prefetch {
            Some(result) => {
                // Build index map: predicted_expert_idx → buffer_idx
                let mut idx_map: HashMap<usize, usize> = HashMap::new();
                for (i, &eidx) in result.predicted_indices.iter().enumerate() {
                    idx_map.insert(eidx, i);
                }

                let mut out = Vec::with_capacity(expert_indices.len());
                for &eidx in expert_indices {
                    stats.total += 1;
                    if let Some(&buf_idx) = idx_map.get(&eidx) {
                        stats.hits += 1;
                        out.push(Some(result.buffers[buf_idx].clone()));
                    } else {
                        stats.misses += 1;
                        out.push(None);
                    }
                }
                out
            }
            None => {
                // No prefetch was done for this layer
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
    ///
    /// Computes `hidden @ gate_weights.T → softmax → argpartition(topk)`
    /// entirely on CPU (via MLX eval). For T=1 decode this is ~D*E FLOPs
    /// (e.g., 4096*512 = 2M FLOPs) — negligible on Apple Silicon.
    fn predict_topk(
        &self,
        hidden: &Array,
        gate_w: &[f32],
    ) -> Result<Vec<usize>, mlx_rs::error::Exception> {
        let d = self.hidden_dim as i32;
        let e = self.num_experts as i32;
        let k = self.top_k;

        // hidden: [1, D] or [D], gate_w: [E, D] (stored row-major)
        let hidden_1d = if hidden.ndim() > 1 {
            // Take first token for prediction (T=1 decode case)
            hidden.reshape(&[d])? // Flatten [1, D] → [D]
        } else {
            hidden.clone()
        };

        // gate_w as Array [E, D]
        let gate_arr = Array::from_slice(gate_w, &[e, d]);

        // logits = hidden @ gate.T → [E]
        let logits = mlx_rs::ops::matmul(&hidden_1d.reshape(&[1, d])?, &gate_arr.t())?;
        let logits_flat = logits.reshape(&[e])?;

        // Softmax → top-k via argpartition
        let probs = mlx_rs::ops::softmax_axis(&logits_flat, -1, None)?;
        let neg_probs = probs.negative()?;
        let neg_k = -(k as i32);
        let part = mlx_rs::ops::argpartition_axis(&neg_probs, neg_k, -1)?;
        let top_indices = part.index(neg_k..);

        top_indices.eval()?;
        let indices: Vec<i32> = top_indices.as_slice().to_vec();
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

        // Create gate weights that strongly favor experts 0 and 2
        let mut gate_w = vec![0.0f32; num_experts * hidden_dim];
        // Expert 0: large positive weights
        for v in gate_w.iter_mut().take(hidden_dim) {
            *v = 1.0;
        }
        // Expert 2: moderate positive weights
        for (i, v) in gate_w[2 * hidden_dim..3 * hidden_dim].iter_mut().enumerate() {
            let _ = i;
            *v = 0.5;
        }

        let mut gate_weights = HashMap::new();
        gate_weights.insert(0, gate_w);

        let prefetcher = ExpertPrefetcher::new(gate_weights, num_experts, hidden_dim, top_k);

        // Create hidden states with all positive values
        let hidden = Array::from_slice(
            &vec![1.0f32; hidden_dim],
            &[hidden_dim as i32],
        );

        let gate_w = prefetcher.gate_weights.get(&0).unwrap();
        let predicted = prefetcher.predict_topk(&hidden, gate_w).unwrap();

        assert_eq!(predicted.len(), top_k);
        // Should predict experts 0 and 2 (highest gate activations)
        assert!(predicted.contains(&0), "Should predict expert 0, got {:?}", predicted);
        assert!(predicted.contains(&2), "Should predict expert 2, got {:?}", predicted);
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
                    buffers: vec![vec![0xAA; 100], vec![0xBB; 100]],
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

        // No prefetch was done for layer 3
        let results = prefetcher.try_get(3, &[0, 1]);

        assert_eq!(results.len(), 2);
        assert!(results[0].is_none());
        assert!(results[1].is_none());

        let stats = prefetcher.stats();
        assert_eq!(stats.misses, 2);
    }
}
