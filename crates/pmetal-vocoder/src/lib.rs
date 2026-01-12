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

#![warn(missing_docs)]

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
