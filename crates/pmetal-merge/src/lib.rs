//! Model Merging Toolkit for PMetal
//!
//! This crate provides comprehensive model merging capabilities inspired by MergeKit,
//! optimized for Apple Silicon and memory-efficient operation.
//!
//! # Supported Merge Methods
//!
//! - **Linear**: Simple weighted averaging of parameters
//! - **SLERP**: Spherical linear interpolation for smooth blending
//! - **TIES**: Task arithmetic with sparsification and sign consensus
//! - **DARE**: Random pruning with rescaling
//! - **DELLA**: Adaptive magnitude-based pruning
//! - **Model Stock**: Geometric interpolation based on task vector similarity
//!
//! # Memory Efficiency
//!
//! All operations use lazy tensor loading and streaming to enable merging
//! large models on memory-constrained macOS devices.

// Crate-level lint configuration for ML code patterns
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

//! # Example
//!
//! ```ignore
//! use pmetal_merge::{MergeConfig, MergeMethod, run_merge};
//!
//! let config = MergeConfig {
//!     method: MergeMethod::Slerp { t: 0.5 },
//!     models: vec![
//!         ModelSource::from_path("model_a"),
//!         ModelSource::from_path("model_b"),
//!     ],
//!     output_path: "merged_model".into(),
//!     ..Default::default()
//! };
//!
//! run_merge(&config)?;
//! ```

#![warn(missing_docs)]

pub mod async_merge;
pub mod batched;
mod config;
mod consensus;
mod error;
pub mod fp8_merge;
pub mod gpu_merge;
mod loader;
mod merge;
pub mod methods;
mod sparsify;

pub use async_merge::{AsyncMergeConfig, AsyncMergePipeline, DoubleBufferManager, PipelineStats};
pub use batched::{BatchConfig, BatchResult, BatchedMerger, MergeStats, TensorBatch};
pub use config::*;
pub use consensus::*;
pub use error::*;
pub use fp8_merge::{
    DynamicScale, Fp8Format, Fp8MergeConfig, Fp8Merger, Fp8Tensor, MemorySavingsReport,
};
pub use gpu_merge::{GpuMergeConfig, GpuMerger};
pub use loader::*;
pub use merge::*;
pub use sparsify::*;

/// Re-export merge methods for convenience
pub use methods::{
    DareMerge, LinearMerge, MergeMethod, ModelStockMerge, PassthroughMerge, SlerpMerge,
    TaskArithmeticMerge, TiesMerge,
};
