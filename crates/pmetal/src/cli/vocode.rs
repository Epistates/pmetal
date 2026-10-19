//! Clap argument struct for `pmetal vocode`.

use clap::Args;
use pmetal_core::jobs::VocodeSpec;

/// Synthesize a waveform from a mel spectrogram with a neural vocoder (BigVGAN).
///
/// The input is a WAV file, which is turned into a log-mel spectrogram by the
/// model's own front end and synthesized back (a round trip that shows what
/// the vocoder does to a recording), or a safetensors file holding the
/// log-mel spectrogram `mel` of a text-to-speech model, `[bins, frames]`.
#[derive(Args, Debug)]
pub struct VocodeArgs {
    /// BigVGAN release (e.g. nvidia/bigvgan_v2_24khz_100band_256x) or a directory
    /// holding its config.json and weights
    #[arg(short, long = "model")]
    pub model: String,

    /// A .wav file, or a safetensors file with a log-mel spectrogram `mel` [bins, frames]
    #[arg(short, long = "input")]
    pub input: String,

    /// The 16-bit mono WAV file to write, at the model's sampling rate
    #[arg(short, long = "output")]
    pub output: String,

    /// Synthesize this many mel frames at a time (same output; memory follows
    /// the chunk); 0 does it in one pass
    #[arg(long = "chunk-frames", default_value_t = 0)]
    pub chunk_frames: usize,
}

impl From<VocodeArgs> for VocodeSpec {
    fn from(a: VocodeArgs) -> Self {
        VocodeSpec {
            model: a.model,
            input: a.input,
            output: a.output,
            chunk_frames: a.chunk_frames,
        }
    }
}
