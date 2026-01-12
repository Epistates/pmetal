//! HuggingFace Hub integration for PMetal.
//!
//! This crate provides:
//! - Model downloading from HuggingFace Hub
//! - Model uploading to HuggingFace Hub
//! - Local cache management

#![warn(missing_docs)]

mod cache;
mod download;
mod upload;

pub use cache::*;
pub use download::*;
pub use upload::*;
