//! Audio front end for the vocoder.
//!
//! MLX-native implementations of:
//! - the STFT and its inverse, matching `torch.stft` / `torch.istft`
//! - the librosa-compatible mel filterbank
//! - the BigVGAN mel front end ([`MelFrontEnd`])
//! - windowed-sinc sample-rate conversion ([`Resampler`])
//! - the constant-Q transform ([`Cqt`])
//! - RIFF/WAVE reading and writing ([`read_wav`], [`write_wav_pcm16`])

mod cqt;
mod mel;
mod pcm;
mod resample;
mod stft;
mod wav;

pub use cqt::*;
pub use mel::*;
pub use pcm::*;
pub use resample::*;
pub use stft::*;
pub use wav::*;
