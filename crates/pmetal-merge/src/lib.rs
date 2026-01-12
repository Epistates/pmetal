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
//!
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

mod config;
mod error;
mod loader;
pub mod methods;
mod merge;
mod sparsify;
mod consensus;

pub use config::*;
pub use error::*;
pub use loader::*;
pub use merge::*;
pub use sparsify::*;
pub use consensus::*;

/// Re-export merge methods for convenience
pub use methods::{
    MergeMethod,
    LinearMerge,
    SlerpMerge,
    TiesMerge,
    DareMerge,
    ModelStockMerge,
    PassthroughMerge,
    TaskArithmeticMerge,
};
