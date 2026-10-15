//! Neural vocoder for text-to-speech synthesis.
//!
//! This crate implements BigVGAN, a universal neural vocoder that converts
//! mel spectrograms to high-fidelity audio waveforms (Lee et al., 2023),
//! optimised for Apple Silicon using MLX. Checkpoints released with the
//! reference implementation load unchanged.
//!
//! # Architecture
//!
//! BigVGAN uses:
//! - Transposed convolutions for upsampling (256x for 24 kHz)
//! - Anti-aliased Multi-Periodicity (AMP) blocks with Snake activations
//! - Multi-period and multi-resolution discriminators for adversarial training
//!
//! # Example
//!
//! ```no_run
//! use pmetal_runtime::compat::Array;
//! use pmetal_vocoder::{BigVGAN, Result};
//!
//! # fn run(mel: &Array) -> Result<Array> {
//! // Load a released model (a Hugging Face id or a local directory).
//! let vocoder = BigVGAN::from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x")?;
//!
//! // `mel` is [batch, 100, frames]; the result is [batch, 1, frames * 256].
//! vocoder.forward(mel)
//! # }
//! ```

pub mod audio;
pub mod checkpoint;
pub mod config;
pub mod discriminator;
pub mod error;
pub mod generator;
pub mod loss;
pub mod nn;
pub mod train;

pub use config::*;
pub use error::*;
pub use generator::{BigVGAN, StreamingSynthesizer};

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
