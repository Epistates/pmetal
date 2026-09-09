//! Neural vocoder for text-to-speech synthesis.
//!
//! This crate implements BigVGAN, a universal neural vocoder that converts
//! mel spectrograms to high-fidelity audio waveforms. Optimized for Apple
//! Silicon using MLX.
//!
//! # Architecture
//!
//! BigVGAN uses:
//! - Transposed convolutions for upsampling (256× for 24kHz)
//! - Anti-aliased Multi-Periodicity (AMP) blocks with Snake activations
//! - Multi-scale discriminators for adversarial training
//!
//! # Example
//!
//! ```ignore
//! use pmetal_vocoder::{BigVGAN, BigVGANConfig};
//!
//! // Load pretrained model
//! let vocoder = BigVGAN::from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x")?;
//!
//! // Generate audio from mel spectrogram
//! let mel = /* [batch, 100, frames] */;
//! let audio = vocoder.forward(&mel)?;
//! ```

// Crate-level lint configuration for ML/GPU code patterns
#![allow(missing_docs)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::option_map_or_none)]
#![allow(clippy::useless_vec)]

pub mod audio;
pub mod config;
pub mod discriminator;
pub mod error;
pub mod generator;
pub mod loss;
pub mod nn;

pub use config::*;
pub use error::*;
pub use generator::BigVGAN;

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
