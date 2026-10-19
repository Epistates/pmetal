//! `pmetal vocode` — synthesize a waveform from a mel spectrogram with a neural vocoder.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Vocode", subcommand = "vocode")]
#[serde(rename_all = "snake_case")]
pub struct VocodeSpec {
    #[job(
        label = "Vocoder",
        group = "Model",
        argv = "--model",
        kind = "model_picker",
        help = "A BigVGAN release (nvidia/bigvgan_v2_24khz_100band_256x) or a directory holding its config.json and weights",
        required
    )]
    #[serde(default)]
    pub model: String,

    #[job(
        label = "Input",
        group = "Input",
        argv = "--input",
        kind = "path",
        help = "A WAV file (re-synthesized through the model's own mel front end) or a safetensors file holding a log-mel spectrogram `mel` [bins, frames]",
        required
    )]
    #[serde(default)]
    pub input: String,

    #[job(
        label = "Output WAV",
        group = "Output",
        argv = "--output",
        kind = "path",
        help = "The 16-bit mono WAV file to write, at the model's sampling rate",
        required
    )]
    #[serde(default)]
    pub output: String,

    #[job(
        label = "Chunk Frames",
        group = "Compute",
        argv = "--chunk-frames",
        min = 0,
        max = 1_000_000,
        default_int = 0,
        help = "Synthesize this many mel frames at a time (same output, memory follows the chunk); 0 does it in one pass"
    )]
    #[serde(default)]
    pub chunk_frames: usize,
}

impl VocodeSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_required_fields_are_named_by_flag() {
        let errors = VocodeSpec::default().validate_descriptors();
        let fields: Vec<_> = errors.iter().map(|e| e.field.as_str()).collect();
        assert_eq!(fields, ["model", "input", "output"]);
        assert_eq!(
            VocodeSpec::argv_flags(),
            ["--model", "--input", "--output", "--chunk-frames"]
        );
    }
}
