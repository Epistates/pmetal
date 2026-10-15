//! Audio front end for the vocoder.
//!
//! MLX-native implementations of:
//! - the STFT and its inverse, matching `torch.stft` / `torch.istft`
//! - the librosa-compatible mel filterbank
//! - the BigVGAN mel front end ([`MelFrontEnd`])
//! - windowed-sinc sample-rate conversion ([`Resampler`])
//! - the constant-Q transform ([`Cqt`])

mod cqt;
mod mel;
mod resample;
mod stft;

pub use cqt::*;
pub use mel::*;
pub use resample::*;
pub use stft::*;
