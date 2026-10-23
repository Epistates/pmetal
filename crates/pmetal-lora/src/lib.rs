//! LoRA and QLoRA implementations for PMetal.
//!
//! This crate provides:
//! - LoRA (Low-Rank Adaptation), rsLoRA and DoRA adapters on any architecture
//!   the dispatcher loads ([`AdaptedModel`])
//! - QLoRA: the same, over a base packed to NF4, NVFP4 or 8-bit integers
//!   ([`quantize_base`])
//! - Adapters saved and read in the PEFT layout ([`peft`])
//!
//! # Architecture-Agnostic Training
//!
//! Use [`DynamicLoraModel`] to automatically detect and load the correct
//! model architecture for training:
//!
//! ```no_run
//! use pmetal_lora::DynamicLoraModel;
//! use pmetal_core::LoraConfig;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Auto-detects Llama, Qwen2, Qwen3, etc.
//! let model = DynamicLoraModel::from_pretrained("/path/to/model", LoraConfig::default())?;
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

pub mod adapted;
mod dynamic;
mod init;
mod lora;
pub mod peft;
mod qlora;
mod trainable;

pub use adapted::{AdaptedModel, FullFineTune};
pub use dynamic::*;
pub use lora::*;
pub use qlora::*;
pub use trainable::*;

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
