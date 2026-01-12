//! GGUF file format implementation.
//!
//! GGUF (GGML Universal Format) is a file format for storing models
//! for inference with GGML-based executors like llama.cpp and Ollama.
//!
//! This crate provides:
//! - Types representing the GGUF format
//! - A reader for loading GGUF files
//! - A writer for creating GGUF files
//! - Dequantization routines for quantized tensors
//!
//! # Example
//!
//! ```ignore
//! use pmetal_gguf::{GgufContent, dequant};
//!
//! // Read GGUF file
//! let content = GgufContent::from_file("model.gguf")?;
//!
//! // Get architecture
//! if let Some(arch) = content.architecture() {
//!     println!("Model architecture: {}", arch);
//! }
//!
//! // Read and dequantize a tensor
//! let mut file = std::fs::File::open("model.gguf")?;
//! let info = content.get_tensor_info("token_embd.weight").unwrap();
//! let data = content.read_tensor_data(&mut file, "token_embd.weight")?;
//! let shape: Vec<i32> = info.dimensions.iter().map(|&d| d as i32).collect();
//! let floats = dequant::dequantize(&data, info.dtype, &shape)?;
//! ```

#![warn(missing_docs)]

pub mod dequant;
pub mod dynamic;
pub mod imatrix;
pub mod iq_quants;
pub mod k_quants;
pub mod quantize;
pub mod reader;
mod types;
pub mod vec_dot;
mod writer;

pub use reader::{
    GgufContent, GgufReadError, GgufVersion, MAX_ARRAY_LENGTH, MAX_METADATA_COUNT,
    MAX_STRING_LENGTH, MAX_TENSOR_COUNT,
};
pub use types::*;
pub use writer::*;
