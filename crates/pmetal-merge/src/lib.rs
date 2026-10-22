//! Model Merging Toolkit for PMetal
//!
//! This crate provides comprehensive model merging capabilities
//! optimized for Apple Silicon and memory-efficient operation.
//!
//! # Supported Merge Methods
//!
//! - **Linear** and **Task Arithmetic**: weighted averaging of parameters or
//!   task vectors
//! - **SLERP** and **Multi-SLERP**: spherical interpolation for smooth blending
//! - **TIES**: task arithmetic with sparsification and sign consensus
//! - **DARE**, **DELLA**, **Breadcrumbs**, **Consensus**: sparsified task
//!   vectors
//! - **Model Stock**: geometric interpolation based on task vector similarity
//! - **Fisher**, **RegMean**, **RAM**, **Nearswap**, and **Passthrough**
//!   (frankenmerges with `slices`)
//!
//! Every method is a [`MergeMethod`] over [`Array`](pmetal_runtime::compat::Array)s,
//! and a configuration picks one with [`MergeMethodConfig`].
//!
//! # Memory Efficiency
//!
//! All operations use lazy tensor loading and streaming to enable merging
//! large models on memory-constrained macOS devices.
//!
//! # Example
//!
//! A method merges the same tensor from several models. SLERP halfway between
//! two orthogonal unit vectors lands at 45 degrees:
//!
//! ```
//! use pmetal_merge::{MergeMethod, MergeParameters, ParameterSetting, SlerpMerge};
//! use pmetal_runtime::compat::Array;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let a = Array::from_f32_slice(&[1.0, 0.0], &[2]);
//! let b = Array::from_f32_slice(&[0.0, 1.0], &[2]);
//! let global = MergeParameters {
//!     t: Some(ParameterSetting::Scalar(0.5)),
//!     ..Default::default()
//! };
//!
//! let per_model = [MergeParameters::default(), MergeParameters::default()];
//! let mut merged = SlerpMerge::new().merge(&[a, b], None, &per_model, &global)?;
//!
//! let half = std::f32::consts::FRAC_1_SQRT_2;
//! let values = merged.to_f32_vec(2).unwrap();
//! assert!((values[0] - half).abs() < 1e-4 && (values[1] - half).abs() < 1e-4);
//! # Ok(())
//! # }
//! ```
//!
//! Whole models are merged with [`run_merge`] (or [`MergeBuilder`]); the crate
//! README shows one end to end.

#![warn(missing_docs)]

mod config;
mod consensus;
mod error;
mod loader;
pub mod lora_merge;
mod mapped;
mod merge;
pub mod methods;
pub mod moe_align;
pub mod sanity;
mod seed;
mod sparsify;
#[cfg(test)]
mod test_util;

pub use config::*;
pub use consensus::*;
pub use error::*;
pub use loader::*;
pub use lora_merge::{AccurateMergeConfig, LoraMergeStats, streaming_lora_merge};
pub use merge::*;
pub use sanity::{MergedTensorReport, SanityLevel, check_tensor};
pub use sparsify::*;

/// Re-export merge methods for convenience
pub use methods::{
    BreadcrumbsMerge, ConsensusMerge, DareMerge, DellaMerge, LinearMerge, MergeMethod,
    ModelStockMerge, MultiSlerpMerge, NearswapMerge, PassthroughMerge, RamMerge, SlerpMerge,
    TaskArithmeticMerge, TiesMerge,
};

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
