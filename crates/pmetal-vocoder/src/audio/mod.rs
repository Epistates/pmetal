//! Audio front end for the vocoder.
//!
//! MLX-native implementations of:
//! - the STFT and its inverse, matching `torch.stft` / `torch.istft`
//! - the librosa-compatible mel filterbank
//! - the BigVGAN mel front end ([`MelFrontEnd`])

mod mel;
mod stft;

pub use mel::*;
pub use stft::*;
