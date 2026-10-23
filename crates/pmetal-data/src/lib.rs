//! Dataset handling and preprocessing for PMetal.
//!
//! This crate provides:
//! - Dataset loading from JSONL, JSON, Parquet, Arrow and CSV, in chat,
//!   instruction, prompt/completion and text layouts ([`rows`], [`dataset`])
//! - DataLoader for creating training batches, sharded and resumable
//!   ([`dataloader`])
//! - Sequence packing for efficient training ([`packing`])
//! - Tokenizer integration, including tokenizers rebuilt from GGUF metadata
//!   ([`tokenizer`], [`gguf_tokenizer`])
//! - Chat template system with response masking, backed by each model's own
//!   Jinja template ([`chat_templates`], [`jinja_chat`])
//! - Streaming pretraining shards ([`streaming`])
//! - Image and video preprocessing for vision-language models
//!   ([`image_processing`], [`qwen_vl_processing`], [`pillow_resample`])

#![warn(missing_docs)]

pub mod calibration;
pub mod chat_templates;
pub mod dataloader;
pub mod dataset;
pub mod gguf_tokenizer;
pub mod image_processing;
pub mod infer_stream;
pub mod inference_config;
pub mod jinja_chat;
pub mod packing;
pub mod pillow_resample;
pub mod qwen_vl_processing;
pub mod rows;
pub mod stream_format;
pub mod streaming;
pub mod tokenizer;

pub use dataloader::*;
pub use dataset::*;
pub use image_processing::*;
pub use packing::*;
pub use tokenizer::*;

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
