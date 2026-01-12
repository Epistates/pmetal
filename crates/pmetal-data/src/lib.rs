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
pub mod image_processing;
pub mod packing;
pub mod tokenizer;

pub use collator::*;
pub use dataloader::*;
pub use dataset::*;
pub use image_processing::*;
pub use packing::*;
pub use tokenizer::*;
