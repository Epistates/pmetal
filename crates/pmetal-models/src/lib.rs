//! LLM model architectures for PMetal.
//!
//! This crate provides implementations of popular LLM architectures:
//! - Llama (2, 3, 3.1, 3.2, 3.3, 4)
//! - Mistral
//! - Qwen (2, 2.5, 3)
//! - Gemma (2, 3)
//! - Phi (3, 4)
//! - DeepSeek
//!
//! # Architecture Support
//!
//! All architectures implement the [`CausalLMModel`] trait, enabling:
//! - Unified inference interface
//! - Dynamic model dispatch via [`DynamicModel`]
//! - Generic training pipelines
//!
//! [`CausalLMModel`]: traits::CausalLMModel
//! [`DynamicModel`]: dispatcher::DynamicModel

#![warn(missing_docs)]

pub mod architectures;
pub mod dispatcher;
pub mod generation;
pub mod loader;
pub mod moe;
pub mod ollama;
pub mod registry;
pub mod rl_generation;
pub mod sampling;
pub mod traits;
pub mod weight_format;

// Re-exports for convenience
pub use dispatcher::{DispatchError, DynamicModel, ModelArchitecture};
pub use generation::*;
pub use loader::*;
pub use registry::*;
pub use rl_generation::{BatchedGenerationOutput, BatchedRlConfig, BatchedRlGenerator, generate_rl_completions};
pub use traits::{CausalLMModel, LoraCapable, ModelConfig, Quantizable, QuantizationType};
pub use weight_format::{GgufModelConfig, WeightFormat, WeightFormatError, WeightLoader};
