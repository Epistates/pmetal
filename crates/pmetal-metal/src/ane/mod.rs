//! Apple Neural Engine (ANE) direct programming support.
//!
//! Training and inference on the ANE through the private
//! `AppleNeuralEngine.framework` APIs. All code is feature-gated behind `ane`.
//!
//! # Training (Dynamic Weight Pipeline)
//!
//! ```text
//! ┌────────────────┐    ┌───────────────────┐    ┌──────────┐
//! │  CPU (vDSP)    │    │ IOSurface (fp32)   │    │   ANE    │
//! │  RMSNorm fwd   │───►│ act + W packed     │───►│ kernels  │
//! │  SiLU deriv    │    │ per-ch interleaved  │    │ compiled │
//! │  CrossEntropy  │◄───│ output results      │◄───│ once     │
//! │  Adam          │    └───────────────────┘    └──────────┘
//! │  cblas dW      │
//! └────────────────┘
//! ```
//!
//! Weights travel alongside activations in the IOSurface spatial dimension,
//! so the kernels compile once and a weight update never recompiles:
//!
//! ```text
//! IOSurface [1, IC, 1, SEQ + weight_cols] fp32
//!   sp[0:SEQ]         = activations
//!   sp[SEQ:SEQ+W]     = weight matrix columns
//! ```
//!
//! MIL kernels slice activations and weights, cast fp32→fp16, perform matmul,
//! cast back fp16→fp32. Weight updates are just memcpy into IOSurface.
//!
//! # Modules
//!
//! - [`runtime`]: Private API FFI via dlopen + objc2
//! - [`iosurface`]: IOSurface zero-copy data transfer (fp16 and fp32)
//! - [`mil`]: MIL 1.3 program builder (builder pattern)
//! - [`kernel`]: Weight blob format, int8 quantization and RoPE helpers
//!   shared by the kernel generators
//! - [`dynamic_kernel`]: Dynamic weight kernel generators (compile once)
//! - [`dynamic_trainer`]: Compile-once training loop
//! - [`extend`]: Multi-layer inference kernels over an IOSurface KV cache, one
//!   family for prefill, decode and speculative verification
//! - [`lm`]: Text generation from a Hugging Face checkpoint on the extend
//!   kernels
//! - [`inference_hybrid`]: CPU decode engine for Qwen3.5 hybrid models

pub(crate) mod checkpoint;
pub mod dynamic_kernel;
pub mod dynamic_trainer;
pub mod extend;
pub mod inference_hybrid;
pub mod iosurface;
pub mod kernel;
pub mod lm;
pub mod loss;
pub mod mil;
pub mod runtime;
pub mod scratch;
