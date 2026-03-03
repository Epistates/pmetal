//! ANE compilation budget tracker.
//!
//! The ANE compiler leaks internal resources, limiting compilations to ~119
//! per process. This module tracks compilation count and provides strategies
//! for when the budget is exhausted.
//!
//! Default budget is 100 (conservative margin below the ~119 hard limit).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Strategy for handling compilation budget exhaustion.
#[derive(Debug, Clone)]
pub enum BudgetExhaustionStrategy {
    /// Save checkpoint and `exec()` restart to reset ANE compiler state.
    ExecRestart {
        /// Path to save checkpoint before restart.
        checkpoint_path: String,
        /// argv[0] for execl.
        argv0: String,
    },
    /// Fall back to GPU (MLX) for remaining training.
    FallbackToGpu,
    /// Return an error and let the caller decide.
    Error,
}

/// Tracks ANE compilation budget across the process lifetime.
///
/// The ANE compiler has a hard limit of ~119 compilations per process.
/// This tracker ensures we don't exceed the budget and provides advance
/// warning to enable checkpoint + restart or fallback.
#[derive(Debug)]
pub struct CompileBudget {
    /// Current compilation count (atomic for thread safety).
    count: Arc<AtomicUsize>,
    /// Maximum allowed compilations (default: 100).
    max: usize,
    /// Number of weight-bearing kernels compiled per batch.
    /// Typically 5 * n_layers (fwdAttn, fwdFFN, ffnBwd, sdpaBwd1, qkvBwd per layer).
    kernels_per_batch: usize,
}

impl CompileBudget {
    /// Create a new compilation budget tracker.
    ///
    /// - `max`: Maximum compilations (default 100, hard limit ~119)
    /// - `kernels_per_batch`: Weight-bearing kernels compiled per accumulation batch
    ///   (typically 5 * n_layers)
    pub fn new(max: usize, kernels_per_batch: usize) -> Self {
        Self {
            count: Arc::new(AtomicUsize::new(0)),
            max,
            kernels_per_batch,
        }
    }

    /// Create with default settings for a given number of layers.
    ///
    /// Uses max=100 and 5 weight-bearing kernels per layer.
    pub fn for_layers(n_layers: usize) -> Self {
        Self::new(100, 5 * n_layers)
    }

    /// Record a single compilation.
    pub fn record_compile(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record multiple compilations at once.
    pub fn record_compiles(&self, n: usize) {
        self.count.fetch_add(n, Ordering::Relaxed);
    }

    /// Get the current compilation count.
    pub fn current(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Get remaining compilations before budget is exhausted.
    pub fn remaining(&self) -> usize {
        self.max.saturating_sub(self.current())
    }

    /// Check if we can compile another full batch of kernels.
    pub fn can_compile_batch(&self) -> bool {
        self.current() + self.kernels_per_batch <= self.max
    }

    /// Maximum number of accumulation steps before budget exhaustion.
    ///
    /// Each accumulation step requires recompiling all weight-bearing kernels.
    pub fn max_accum_steps(&self) -> usize {
        if self.kernels_per_batch == 0 {
            return usize::MAX;
        }
        self.remaining() / self.kernels_per_batch
    }

    /// Check if a restart is needed before the next batch.
    pub fn needs_restart(&self) -> bool {
        !self.can_compile_batch()
    }

    /// Get the maximum compilation limit.
    pub fn max(&self) -> usize {
        self.max
    }

    /// Get the kernels-per-batch count.
    pub fn kernels_per_batch(&self) -> usize {
        self.kernels_per_batch
    }

    /// Clone the atomic counter (shares the same underlying counter).
    pub fn shared(&self) -> Self {
        Self {
            count: Arc::clone(&self.count),
            max: self.max,
            kernels_per_batch: self.kernels_per_batch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_basic() {
        let budget = CompileBudget::for_layers(12);
        assert_eq!(budget.max(), 100);
        assert_eq!(budget.kernels_per_batch(), 60);
        assert_eq!(budget.current(), 0);
        assert_eq!(budget.remaining(), 100);
        assert!(budget.can_compile_batch()); // 60 <= 100
    }

    #[test]
    fn test_budget_record() {
        let budget = CompileBudget::for_layers(12);
        budget.record_compiles(60);
        assert_eq!(budget.current(), 60);
        assert_eq!(budget.remaining(), 40);
        assert!(!budget.can_compile_batch()); // 60 + 60 > 100
    }

    #[test]
    fn test_budget_max_accum_steps() {
        let budget = CompileBudget::for_layers(12);
        // 100 / 60 = 1 full batch
        assert_eq!(budget.max_accum_steps(), 1);

        let small = CompileBudget::new(100, 10);
        // 100 / 10 = 10 batches
        assert_eq!(small.max_accum_steps(), 10);
    }

    #[test]
    fn test_budget_needs_restart() {
        let budget = CompileBudget::for_layers(12);
        assert!(!budget.needs_restart());

        budget.record_compiles(60);
        assert!(budget.needs_restart()); // Can't do another 60
    }

    #[test]
    fn test_budget_shared() {
        let budget = CompileBudget::for_layers(12);
        let shared = budget.shared();

        budget.record_compiles(10);
        assert_eq!(shared.current(), 10); // Same underlying counter
    }

    #[test]
    fn test_exhaustion_strategy_variants() {
        let _restart = BudgetExhaustionStrategy::ExecRestart {
            checkpoint_path: "/tmp/ckpt.bin".to_string(),
            argv0: "pmetal-train".to_string(),
        };
        let _fallback = BudgetExhaustionStrategy::FallbackToGpu;
        let _error = BudgetExhaustionStrategy::Error;
    }
}
