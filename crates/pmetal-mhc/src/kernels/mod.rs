//! Metal kernel implementations for mHC operations.
//!
//! This module provides GPU-accelerated implementations of mHC operations
//! using Apple Metal compute shaders.
//!
//! # Kernels
//!
//! - `compute_mappings`: Fused RMSNorm + projection for H̃^pre, H̃^post, H̃^res
//! - `sinkhorn_knopp`: Iterative doubly stochastic projection
//! - `apply_pre`: Aggregate streams using H^pre weights
//! - `apply_post_res`: Fused post-mapping and residual merge

pub mod metal_impl;
pub mod metal_shaders;

// Re-exports
pub use metal_impl::MhcMetalError;
#[cfg(feature = "metal")]
pub use metal_impl::{create_default_context, MhcMetalContext};

/// Metal shader source code.
pub const MHC_METAL_SHADERS: &str = include_str!("metal_shaders.metal");

/// Kernel configuration for mHC operations.
#[derive(Debug, Clone, Copy)]
pub struct MhcKernelConfig {
    /// Thread group size for compute mappings kernel.
    pub compute_mappings_threads: u32,

    /// Thread group size for Sinkhorn kernel.
    pub sinkhorn_threads: u32,

    /// Thread group size for apply kernels.
    pub apply_threads: u32,

    /// Whether to use mixed precision (BF16/FP32).
    pub use_mixed_precision: bool,
}

impl Default for MhcKernelConfig {
    fn default() -> Self {
        Self {
            compute_mappings_threads: 256,
            sinkhorn_threads: 64, // Smaller for iterative algorithm
            apply_threads: 256,
            use_mixed_precision: true,
        }
    }
}

/// Statistics from kernel execution.
#[derive(Debug, Clone, Default)]
pub struct KernelStats {
    /// Time spent in compute_mappings kernel (microseconds).
    pub compute_mappings_us: u64,

    /// Time spent in Sinkhorn kernel (microseconds).
    pub sinkhorn_us: u64,

    /// Time spent in apply kernels (microseconds).
    pub apply_us: u64,

    /// Total kernel invocations.
    pub invocations: u64,
}

impl KernelStats {
    /// Get total time in microseconds.
    pub fn total_us(&self) -> u64 {
        self.compute_mappings_us + self.sinkhorn_us + self.apply_us
    }

    /// Merge with another stats instance.
    pub fn merge(&mut self, other: &KernelStats) {
        self.compute_mappings_us += other.compute_mappings_us;
        self.sinkhorn_us += other.sinkhorn_us;
        self.apply_us += other.apply_us;
        self.invocations += other.invocations;
    }
}
