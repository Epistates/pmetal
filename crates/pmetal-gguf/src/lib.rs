//! GGUF file format implementation.
//!
//! GGUF (GGML Universal Format) is the single-file format for storing models
//! for GGML-family inference runtimes.
//!
//! This crate provides:
//! - Types representing the GGUF format (versions 1 to 3)
//! - A reader that validates every count, length and tensor extent against the
//!   file ([`GgufContent`]), and a memory-mapped model that handles split sets
//!   ([`GgufModel`])
//! - A writer for creating GGUF files, in memory ([`GgufBuilder`]) or streamed
//!   atomically to disk ([`GgufFileWriter`])
//! - Dequantization ([`dequant`]) and quantization ([`quantize`]) of the ggml
//!   block formats, with importance-matrix support ([`imatrix`])
//! - Extraction of the tokenizer and `config.json` a file carries
//!   ([`tokenizer`], [`config`])
//!
//! # Example
//!
//! Quantize a tensor, write a GGUF file to bytes, read it back and decode it:
//!
//! ```
//! use std::io::Cursor;
//! use pmetal_gguf::{GgmlType, GgufBuilder, GgufContent, dequant, quantize};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Two rows of 64 weights, quantized to Q8_0 (blocks of 32).
//! let weights: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
//! let packed = quantize::quantize_tensor(&weights, GgmlType::Q8_0, 64, None)?;
//!
//! let mut builder = GgufBuilder::with_model("llama", "tiny");
//! builder.add_raw_tensor("blk.0.ffn_up.weight", vec![2, 64], GgmlType::Q8_0, packed);
//! let mut file = Cursor::new(builder.build_to_bytes()?);
//!
//! let content = GgufContent::read(&mut file)?;
//! assert_eq!(content.architecture(), Some("llama"));
//!
//! let info = content.get_tensor_info("blk.0.ffn_up.weight").unwrap();
//! let data = content.read_tensor_data(&mut file, "blk.0.ffn_up.weight")?;
//! let shape: Vec<i32> = info.dimensions.iter().map(|&d| d as i32).collect();
//! let floats = dequant::dequantize(&data, info.dtype, &shape)?;
//! assert_eq!(floats.len(), 128);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

pub mod config;
pub mod dequant;
pub mod dynamic;
mod fp4;
pub mod imatrix;
mod iq_grids;
pub mod iq_quants;
pub mod k_quants;
mod model;
pub mod quantize;
pub mod reader;
pub mod tokenizer;
mod types;
mod writer;

pub use model::{GgufModel, first_shard_path};
pub use reader::{
    GgufContent, GgufReadError, GgufVersion, MAX_ARRAY_LENGTH, MAX_METADATA_COUNT,
    MAX_STRING_LENGTH, MAX_TENSOR_COUNT, MAX_TENSOR_DIMS,
};
pub use types::*;
pub use writer::*;

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
