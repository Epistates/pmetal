//! Apple Neural Engine (ANE) direct programming support.
//!
//! This module provides a complete ANE training pipeline using private
//! `AppleNeuralEngine.framework` APIs. All code is feature-gated behind `ane`.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐    ┌──────────┐    ┌──────────┐
//! │  CPU (vDSP)  │◄──►│ IOSurface│◄──►│   ANE    │
//! │  RMSNorm     │    │ zero-copy│    │  conv/mm │
//! │  Softmax     │    │  fp16    │    │  fwd/bwd │
//! │  CrossEnt    │    └──────────┘    └──────────┘
//! │  Adam        │
//! │  cblas dW    │
//! └─────────────┘
//! ```
//!
//! # Modules
//!
//! - [`runtime`]: Private API FFI via dlopen + objc2
//! - [`iosurface`]: IOSurface zero-copy data transfer
//! - [`mil`]: MIL 1.3 program builder (builder pattern)
//! - [`kernel`]: Transformer kernel generators + weight blob format
//! - [`budget`]: Compilation budget tracker (~100 compiles/process)
//! - [`inference`]: Forward-only ANE inference engine with autoregressive generation
//! - [`trainer`]: Hybrid CPU/ANE training loop

pub mod budget;
pub mod inference;
pub mod iosurface;
pub mod kernel;
pub mod mil;
pub mod runtime;
pub mod trainer;
