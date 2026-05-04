//! `pmetal-bridge` — zero-allocation MLX C++ bridge.
//!
//! Provides [`InlineArray`], a stack-allocated wrapper around
//! `mlx::core::array` with no per-op heap allocation.  All C++ calls go
//! directly through `extern "C"` declarations.
//!
//! The `compat` module provides drop-in replacements for mlx-rs types
//! (`Array`, `Dtype`, `Module`, `ModuleParameters`, optimizers, layers, etc.)
//! so that model code can use a familiar API backed by the zero-allocation bridge.

pub mod inline_array;
pub use inline_array::{InlineArray, QuantizedMode};

pub mod error;
pub use error::{
    BridgeError, BridgeResult, check_last_error, clear_last_error, error_log_mode,
    set_error_log_mode,
};

pub mod dtype;

pub mod compile;
pub use compile::CompiledFn;

pub mod scalar;
pub mod try_ops;

pub mod compat;
pub mod decode;

pub mod optimizer;
pub use optimizer::{AdamW, ParamClass, ParamSet};

pub mod mlx_quant;
pub mod training;
pub mod turboquant;
pub mod turboquant_dispatch;

pub mod native_common;
pub mod native_loader;

pub mod deepseek_native;
pub mod gemma4_native;
pub mod gpt_oss_native;
pub mod llama4_native;
pub mod qwen3_native;
