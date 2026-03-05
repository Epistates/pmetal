//! Apple Neural Engine (ANE) direct programming support.
//!
//! This module provides a complete ANE training pipeline using private
//! `AppleNeuralEngine.framework` APIs. All code is feature-gated behind `ane`.
//!
//! # Architecture (Dynamic Weight Pipeline)
//!
//! ```text
//! ┌────────────────┐    ┌───────────────────┐    ┌──────────┐
//! │  CPU (vDSP)    │    │ IOSurface (fp32)   │    │   ANE    │
//! │  RMSNorm fwd   │───►│ act + W packed     │───►│ 9 kernels│
//! │  SiLU deriv    │    │ per-ch interleaved  │    │ compiled │
//! │  CrossEntropy  │◄───│ output results      │◄───│ once     │
//! │  Adam          │    └───────────────────┘    └──────────┘
//! │  cblas dW      │
//! └────────────────┘
//! ```
//!
//! Unlike the previous static pipeline (which recompiled ~60 kernels every
//! N steps, consuming ~76% of training time), the dynamic pipeline packs
//! weights alongside activations in the IOSurface spatial dimension:
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
//! - [`kernel`]: Static kernel generators + weight blob format (used by inference)
//! - [`dynamic_kernel`]: Dynamic weight kernel generators (9 kernels, compile once)
//! - [`dynamic_trainer`]: Compile-once training loop (replaces static trainer)
//! - [`inference`]: Forward-only ANE inference engine with autoregressive generation

pub mod dynamic_kernel;
pub mod dynamic_trainer;
pub mod inference;
pub mod inference_hybrid;
pub mod iosurface;
pub mod kernel;
pub mod mil;
pub mod runtime;

// Legacy modules kept for reference but no longer used in training
pub mod budget;
pub mod trainer;
