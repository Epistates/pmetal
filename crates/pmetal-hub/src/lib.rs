//! HuggingFace Hub integration for PMetal.
//!
//! This crate provides:
//! - Model downloading from HuggingFace Hub
//! - Model uploading to HuggingFace Hub
//! - Local cache management

#![warn(missing_docs)]

mod cache;
mod download;
pub mod fit;
pub mod resolve;
pub mod search;
mod upload;

pub use cache::*;
pub use download::*;
pub use fit::*;
pub use resolve::*;
pub use search::*;
pub use upload::*;

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
