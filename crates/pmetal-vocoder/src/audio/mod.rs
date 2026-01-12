//! Audio processing utilities for vocoder.
//!
//! This module provides MLX-native implementations of:
//! - STFT (Short-Time Fourier Transform)
//! - Mel filterbank
//! - Audio feature extraction

mod mel;
mod stft;

pub use mel::*;
pub use stft::*;
