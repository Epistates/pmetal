//! # PMetal
//!
//! High-performance LLM fine-tuning framework for Apple Silicon.
//!
//! This crate re-exports the PMetal sub-crates behind feature flags for
//! convenient single-dependency usage:
//!
//! ```toml
//! [dependencies]
//! pmetal = "0.1"           # default features: core, gguf, metal, hub, mlx, models, lora, trainer
//! pmetal = { version = "0.1", features = ["full"] }  # everything
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Crate | Default |
//! |---------|-------|---------|
//! | `core` | [`pmetal-core`] | yes |
//! | `gguf` | [`pmetal-gguf`] | yes |
//! | `metal` | [`pmetal-metal`] | yes |
//! | `hub` | [`pmetal-hub`] | yes |
//! | `mlx` | [`pmetal-mlx`] | yes |
//! | `models` | [`pmetal-models`] | yes |
//! | `lora` | [`pmetal-lora`] | yes |
//! | `trainer` | [`pmetal-trainer`] | yes |
//! | `data` | [`pmetal-data`] | no |
//! | `distill` | [`pmetal-distill`] | no |
//! | `merge` | [`pmetal-merge`] | no |
//! | `vocoder` | [`pmetal-vocoder`] | no |
//! | `distributed` | [`pmetal-distributed`] | no |
//! | `mhc` | [`pmetal-mhc`] | no |
//! | `easy` | all training/inference crates | yes |
//! | `full` | all of the above | no |
//!
//! ## Easy API
//!
//! The [`easy`] module provides high-level builders for common workflows:
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Fine-tune a model in one call
//! let result = pmetal::easy::finetune("Qwen/Qwen3-0.6B", "data.jsonl")
//!     .lora(16, 32.0)
//!     .epochs(3)
//!     .learning_rate(2e-4)
//!     .output("./output")
//!     .run()
//!     .await?;
//!
//! // Run inference
//! let result = pmetal::easy::infer("Qwen/Qwen3-0.6B")
//!     .lora("./output/lora_weights.safetensors")
//!     .generate("What is 2+2?")
//!     .await?;
//! # Ok(())
//! # }
//! ```

pub mod version;

#[cfg(feature = "easy")]
pub mod easy;

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
    pub use pmetal_hub::{download_file, download_model};

    // Callback types (defined in core, but only useful with trainer)
    #[cfg(feature = "core")]
    pub use pmetal_core::TrainingCallback;

    // Trainer callbacks
    #[cfg(feature = "trainer")]
    pub use pmetal_trainer::{LoggingCallback, MetricsJsonCallback, ProgressCallback};
}
