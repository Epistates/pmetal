//! # PMetal
//!
//! High-performance LLM fine-tuning framework for Apple Silicon.
//!
//! This crate re-exports the PMetal sub-crates behind feature flags for
//! convenient single-dependency usage:
//!
//! ```toml
//! [dependencies]
//! pmetal = "0.3"           # default features: core, gguf, metal, hub, mlx, models, lora, trainer, ane
//! pmetal = { version = "0.3", features = ["full"] }  # everything
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Crate | Default | Notes |
//! |---------|-------|---------|-------|
//! | `core` | [`pmetal-core`] | yes | Foundation types, configs, traits |
//! | `gguf` | [`pmetal-gguf`] | yes | GGUF format support |
//! | `metal` | [`pmetal-metal`] | yes | Metal GPU kernels |
//! | `hub` | [`pmetal-hub`] | yes | HuggingFace Hub integration + model resolution |
//! | `mlx` | [`pmetal-mlx`] | yes | MLX backend |
//! | `models` | [`pmetal-models`] | yes | LLM architectures |
//! | `lora` | [`pmetal-lora`] | yes | LoRA/QLoRA training |
//! | `trainer` | [`pmetal-trainer`] | yes | Training loops (enables `data` + `distill`) |
//! | `ane` | [`pmetal-metal`] | yes | Apple Neural Engine integration |
//! | `data` | [`pmetal-data`] | yes | Dataset loading |
//! | `distill` | [`pmetal-distill`] | yes* | Knowledge distillation (*enabled transitively via `trainer`) |
//! | `lora-metal-fused` | [`pmetal-lora`] | no | ~2x LoRA training speedup via fused Metal kernels |
//! | `merge` | [`pmetal-merge`] | no | Model merging strategies |
//! | `vocoder` | [`pmetal-vocoder`] | no | BigVGAN neural vocoder |
//! | `distributed` | [`pmetal-distributed`] | no | Distributed training |
//! | `mhc` | [`pmetal-mhc`] | no | Manifold-Constrained Hyper-Connections |
//! | `full` | (all) | no | All features |
//!
//! ## Direct SDK Usage
//!
//! Use the sub-crate APIs directly for full control:
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use pmetal::hub::resolve_model_path;
//! use pmetal::data::{Tokenizer, chat_templates::{detect_chat_template, Message}};
//! use pmetal::models::{DynamicModel, GenerationConfig, generate_cached_async};
//!
//! let model_path = resolve_model_path("Qwen/Qwen3-0.6B").await?;
//! let tokenizer = Tokenizer::from_model_dir(&model_path)?;
//! let template = detect_chat_template(&model_path, "Qwen/Qwen3-0.6B");
//! let formatted = template.apply(&[Message::user("What is 2+2?")]).text;
//! let input_ids = tokenizer.encode_with_special_tokens(&formatted)?;
//!
//! let mut model = DynamicModel::load(&model_path)?;
//! let mut cache = model.create_cache(input_ids.len() + 256);
//! let gen_config = GenerationConfig::sampling(256, 0.7);
//! let output = generate_cached_async(
//!     |input, cache| model.forward_with_hybrid_cache(input, None, Some(cache), None),
//!     &input_ids, gen_config, &mut cache,
//! )?;
//! let text = tokenizer.decode(&output.token_ids[input_ids.len()..])?;
//! # Ok(())
//! # }
//! ```

pub mod version;

/// Unified inference pipeline shared by CLI, GUI, and serve.
///
/// Requires features: `hub`, `data`, `models`, `lora`.
#[cfg(all(
    feature = "hub",
    feature = "data",
    feature = "models",
    feature = "lora"
))]
pub mod inference_runner;

// NOTE: `core` below shadows the Rust built-in `core` crate within this file.
// Any code added here that needs `core::fmt`, `core::mem`, etc. must use `::core::`.
#[cfg(feature = "core")]
pub use pmetal_core as core;

#[cfg(feature = "gguf")]
pub use pmetal_gguf as gguf;

#[cfg(feature = "metal")]
pub use pmetal_metal as metal;

#[cfg(feature = "hub")]
pub use pmetal_hub as hub;

#[cfg(feature = "distributed")]
pub use pmetal_distributed as distributed;

#[cfg(feature = "mhc")]
pub use pmetal_mhc as mhc;

#[cfg(feature = "mlx")]
pub use pmetal_mlx as mlx;

#[cfg(feature = "models")]
pub use pmetal_models as models;

#[cfg(feature = "lora")]
pub use pmetal_lora as lora;

#[cfg(feature = "data")]
pub use pmetal_data as data;

#[cfg(feature = "trainer")]
pub use pmetal_trainer as trainer;

#[cfg(feature = "distill")]
pub use pmetal_distill as distill;

#[cfg(feature = "merge")]
pub use pmetal_merge as merge;

#[cfg(feature = "vocoder")]
pub use pmetal_vocoder as vocoder;

/// Convenience re-exports of the most commonly used types.
///
/// ```rust
/// use pmetal::prelude::*;
/// ```
pub mod prelude {
    // Core types take precedence (PMetalError, Result)
    #[cfg(feature = "core")]
    pub use pmetal_core::prelude::*;

    // Metal prelude minus Result (use metal::Result explicitly to avoid ambiguity)
    #[cfg(feature = "metal")]
    pub use pmetal_metal::prelude::{
        BatchedLora, BatchedLoraAdapters, BatchedLoraConfig, BufferUsage, FlashAttention,
        FlashAttentionConfig, FlashAttentionVarlen, FlashAttentionVarlenConfig, FusedCrossEntropy,
        FusedCrossEntropyConfig, FusedLinearCrossEntropy, FusedLinearCrossEntropyConfig, FusedLora,
        FusedLoraConfig, FusedSampler, FusedSamplerConfig, MetalBuffer, MetalContext, MetalError,
        PipelineCache,
    };

    // MLX prelude minus Dtype (use mlx::Dtype explicitly to avoid ambiguity with core::Dtype)
    #[cfg(feature = "mlx")]
    pub use pmetal_mlx::prelude::{Array, Builder, Module, ModuleParameters, Param};

    #[cfg(feature = "distributed")]
    pub use pmetal_distributed::prelude::*;

    #[cfg(feature = "mhc")]
    pub use pmetal_mhc::prelude::*;

    // Trainer types
    #[cfg(feature = "trainer")]
    pub use pmetal_trainer::{
        AdamWGroupsBuilder, CheckpointManager, TrainingLoop, TrainingLoopConfig,
    };

    // Data types
    #[cfg(feature = "data")]
    pub use pmetal_data::{
        DataLoader, DataLoaderConfig, DatasetFormat, Tokenizer, TrainingDataset,
    };

    // Model types
    #[cfg(feature = "models")]
    pub use pmetal_models::{DynamicModel, GenerationConfig, ModelArchitecture};

    // LoRA types
    #[cfg(feature = "lora")]
    pub use pmetal_lora::{DynamicLoraModel, TrainableModel};

    // Hub functions
    #[cfg(feature = "hub")]
    pub use pmetal_hub::{download_file, download_model, resolve_model_path};

    // Callback types (defined in core, but only useful with trainer)
    #[cfg(feature = "core")]
    pub use pmetal_core::TrainingCallback;

    // Trainer callbacks
    #[cfg(feature = "trainer")]
    pub use pmetal_trainer::{LoggingCallback, MetricsJsonCallback, ProgressCallback};
}
