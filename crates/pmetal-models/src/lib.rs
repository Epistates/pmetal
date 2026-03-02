//! LLM model architectures for PMetal.
//!
//! This crate provides implementations of popular LLM architectures:
//!
//! ## Causal LM (dispatched via [`DynamicModel`])
//!
//! - Llama (2, 3, 3.1, 3.2, 3.3), Llama 4
//! - Qwen (2, 2.5, 3, 3-MoE)
//! - DeepSeek (V3)
//! - Mistral / Mixtral
//! - Gemma (2, 3), RecurrentGemma (Griffin)
//! - Phi (3, 4)
//! - Cohere (Command R)
//! - Granite (3.0, 3.1)
//! - NemotronH (Mamba+Attention hybrid)
//! - StarCoder2
//! - Jamba (1.5)
//! - GPT-OSS (architecture only, not dispatched)
//!
//! ## Diffusion (dispatched, experimental)
//!
//! - Flux (1-dev, 1-schnell)
//!
//! ## Vision & Multimodal (architecture only, not yet dispatched)
//!
//! - Pixtral, Qwen2-VL, MLlama, CLIP, T5, VAE, Whisper
//!
//! # Architecture Support
//!
//! CausalLM architectures implement the [`CausalLMModel`] trait, enabling:
//! - Unified inference interface
//! - Dynamic model dispatch via [`DynamicModel`]
//! - Generic training pipelines
//!
//! [`CausalLMModel`]: traits::CausalLMModel
//! [`DynamicModel`]: dispatcher::DynamicModel

// Crate-level lint configuration
#![allow(missing_docs)]
#![allow(dead_code)]
#![allow(unsafe_code)]
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::type_complexity)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::len_zero)]
#![allow(ambiguous_glob_reexports)]

pub mod architectures;
pub mod dispatcher;
pub mod generation;
pub mod loader;
pub mod moe;
pub mod ollama;
pub mod pipelines;
pub mod registry;
pub mod rl_generation;
pub mod sampling;
pub mod traits;
pub mod weight_format;

// Re-exports for convenience
pub use dispatcher::{DynamicModel, ModelArchitecture};
pub use generation::*;
pub use loader::*;
pub use registry::*;
pub use rl_generation::{
    BatchedGenerationOutput, BatchedRlConfig, BatchedRlGenerator, generate_rl_completions,
};
pub use traits::{CausalLMModel, LoraCapable, ModelConfig, Quantizable, QuantizationType};
pub use weight_format::{GgufModelConfig, WeightFormat, WeightFormatError, WeightLoader};
