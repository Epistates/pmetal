//! Dataset handling and preprocessing for PMetal.
//!
//! This crate provides:
//! - Dataset loading from various formats (JSONL, Alpaca, ShareGPT)
//! - DataLoader for creating training batches
//! - Sequence packing for efficient training
//! - Data collation and batching
//! - Tokenizer integration
//! - Chat template system with response masking

#![warn(missing_docs)]

pub mod chat_templates;
pub mod collator;
pub mod dataloader;
pub mod dataset;
pub mod gguf_tokenizer;
pub mod image_processing;
pub mod inference_config;
pub mod jinja_chat;
pub mod packing;
pub mod pillow_resample;
pub mod stream_format;
pub mod streaming;
pub mod tokenizer;
pub mod vocab_compact;

pub use collator::*;
pub use dataloader::*;
pub use dataset::*;
pub use image_processing::*;
pub use packing::*;
pub use tokenizer::*;
pub use vocab_compact::VocabCompactor;

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
