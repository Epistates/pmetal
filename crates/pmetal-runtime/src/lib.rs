//! `pmetal-runtime` — PMetal's tensor runtime for Apple Silicon.
//!
//! [`InlineArray`] is the array every pmetal crate computes with: a
//! stack-allocated handle with no per-op heap allocation, over MLX's lazy
//! graph today, reached through `extern "C"` calls into this crate's C++.
//! On it sit compiled graphs, the native inference engines, the decode
//! sampler, optimizers, the TurboQuant KV cache and pmetal's own Metal
//! kernels. MLX is an implementation detail behind this API, kept where it is
//! the best choice and replaced by our own kernels where they are faster or
//! more correct.
//!
//! The `compat` module builds the model-facing API on top of it (`Array`,
//! `Dtype`, `Module`, `ModuleParameters`, optimizers, layers, etc.), so model
//! code never touches the FFI directly. The `nn` module holds the operations
//! a model's forward and loss are built from (attention, the gated delta rule,
//! the cut cross-entropy loss, rotary embeddings at explicit positions, the
//! mixture-of-experts combine), and `memory` the allocator and system memory
//! figures.

#[macro_use]
mod bridge_extern;

mod callback;

pub mod inline_array;
pub use inline_array::{InlineArray, MetalKernel, QuantizedMode, TemplateArg};

pub mod metallib;

pub mod error;
pub use error::{
    RuntimeError, RuntimeResult, check_last_error, check_unobserved_error, checked,
    clear_last_error, error_log_mode, expect_runtime_errors, set_error_log_mode,
};

pub mod dtype;

pub mod compile;
pub use compile::CompiledFn;

pub mod scalar;
pub mod try_ops;

pub mod compat;
pub mod decode;
pub mod distributed;

pub mod optimizer;
pub use optimizer::{AdamW, ParamClass, ParamSet};

pub mod memory;
pub mod nn;
pub mod rope;

pub mod mlx_quant;
pub mod training;
pub mod turboquant;
pub mod turboquant_dispatch;

pub mod native_checkpoint;
pub mod native_common;
pub mod native_loader;
pub mod native_moe;
pub mod native_weight;
pub use native_weight::{EmbeddingWeight, LayerWeight, QuantParams};

pub mod deepseek_native;
pub mod gemma4_native;
pub mod gpt_oss_native;
pub mod llama4_native;
pub mod qwen3_native;

#[cfg(test)]
mod f64_reference;
