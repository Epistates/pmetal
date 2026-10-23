//! Metal GPU compute kernels and Apple Neural Engine programming for PMetal.
//!
//! This crate holds the Apple-Silicon pieces the tensor runtime
//! (`pmetal-runtime`) has no op for:
//!
//! - **Training kernels** ([`kernels`]): [`DwGemm`](kernels::DwGemm), the f32
//!   weight-gradient GEMM the ANE trainer batches into one
//!   [`BatchedCommandBuffer`](kernels::BatchedCommandBuffer)
//! - **Apple Neural Engine** ([`ane`], feature `ane`, on by default): MIL
//!   program generation, IOSurface transfer, a compile-once training loop and
//!   an inference engine, through the private `AppleNeuralEngine.framework`
//! - **CPU helpers**: [`accelerate`] (vDSP and cblas wrappers for RMSNorm,
//!   softmax, cross-entropy and GEMM) and [`neon_convert`] (bulk
//!   f16 and f32 conversion)
//! - **Device management and buffers**: a thread-safe [`MetalContext`] with a
//!   pipeline cache, type-safe [`MetalBuffer`]s over unified memory, and
//!   zero-copy [views](view) of memory the runtime owns
//!
//! Dense attention, RoPE, SwiGLU, LoRA and GEMM are deliberately absent: the
//! runtime runs them in its lazy graph, with autodiff, faster than the
//! hand-written kernels this crate used to carry for them.
//!
//! The whole crate is `cfg(target_os = "macos")`.
//!
//! # Architecture
//!
//! The crate is designed to work alongside the runtime, sharing unified memory:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Unified Memory                           │
//! │                                                             │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
//! │  │   Runtime   │◄──►│ Metal Buffer│◄──►│ CPU Access  │     │
//! │  │    Array    │    │             │    │             │     │
//! │  └─────────────┘    └─────────────┘    └─────────────┘     │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]
#![cfg(target_os = "macos")]

pub mod accelerate;
#[cfg(feature = "ane")]
pub mod ane;
pub mod buffer;
pub(crate) mod chip;
pub mod context;
pub mod error;
pub mod kernels;
pub(crate) mod msl_layout;
pub mod neon_convert;
pub mod pipeline;
pub mod view;

pub use buffer::{AsMetalBuffer, BufferUsage, MetalBuffer};
pub use context::MetalContext;
pub use error::{MetalError, Result};
pub use pipeline::{FunctionConstant, PipelineCache};
pub use view::{MetalBufferView, MetalBufferViewF16, MetalBufferViewF32, metal_buffer_from_ptr};

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::buffer::{BufferUsage, MetalBuffer};
    pub use crate::context::MetalContext;
    pub use crate::error::{MetalError, Result};
    pub use crate::pipeline::PipelineCache;
}

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
