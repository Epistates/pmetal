//! HuggingFace Hub integration for PMetal.
//!
//! This crate provides:
//! - Model and dataset downloading from the Hugging Face Hub: one commit per
//!   download, only the files the model needs, resumable, verified, with
//!   progress and ETA
//! - Choosing weights: safetensors, or one GGUF quantization out of a
//!   multi-quant repository
//! - Checking that a local model directory is complete
//! - Hub search with "will it fit on this Mac" estimates
//! - Publishing a model directory to the Hub
//! - Local cache management, in the standard Hub cache layout

#![warn(missing_docs)]

mod api;
mod auth;
mod cache;
mod client;
mod dataset;
mod download;
pub mod fit;
mod progress;
mod repo;
pub mod resolve;
pub mod search;
pub mod select;
pub mod tensors;
pub mod torch;
mod upload;
mod verify;

pub use auth::resolve_token;
pub use cache::*;
pub use client::HubOptions;
pub use dataset::*;
pub use download::*;
pub use fit::*;
/// The shard index reader and writer, from `pmetal-core`, for crates that reach
/// the hub but not the core crate directly.
pub use pmetal_core::safetensors_index;
pub use progress::{ProgressCallback, TransferProgress};
pub use resolve::*;
pub use search::*;
pub use select::{GgufVariant, ModelPlan, RepoFile, SelectedWeights, WeightFormat, gguf_variants};
pub use upload::*;
pub use verify::{ModelDirReport, verify_model_dir};

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
