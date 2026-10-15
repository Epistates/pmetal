//! Conversion between float waveforms and 16-bit PCM.

use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::{Array, Dtype};

/// Full scale of a 16-bit sample as the reference uses it: `32768 - 1`, so a
/// float sample of exactly `1.0` cannot overflow `i16` (which would pop on
/// playback).
pub const MAX_WAV_VALUE: f32 = 32767.0;

/// Convert a float waveform in `[-1, 1]` to 16-bit PCM samples.
///
/// Values are scaled by [`MAX_WAV_VALUE`] and truncated toward zero, as the
/// reference's `(audio * MAX_WAV_VALUE).astype("int16")` does; out-of-range
/// samples (a vocoder without `tanh` can slightly exceed 1) are clamped
/// rather than wrapped.
pub fn to_pcm16(audio: &Array) -> Result<Vec<i16>> {
    if audio.size() == 0 {
        return Err(VocoderError::Audio("no samples to convert".into()));
    }
    let audio = audio.as_dtype(Dtype::Float32);
    Ok(audio
        .as_slice::<f32>()
        .iter()
        .map(|&v| (v * MAX_WAV_VALUE).clamp(-MAX_WAV_VALUE, MAX_WAV_VALUE) as i16)
        .collect())
}

/// Convert 16-bit PCM samples to a float waveform in `[-1, 1]`, as a
/// `[samples]` array.
pub fn from_pcm16(samples: &[i16]) -> Array {
    let floats: Vec<f32> = samples
        .iter()
        .map(|&s| f32::from(s) / MAX_WAV_VALUE)
        .collect();
    Array::from_f32_slice(&floats, &[samples.len() as i32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_truncates_and_clamps() {
        let audio = Array::from_f32_slice(&[0.0, 1.0, -1.0, 0.5, -0.5, 1.7, -3.0, 0.00001], &[8]);
        assert_eq!(
            to_pcm16(&audio).unwrap(),
            [0, 32767, -32767, 16383, -16383, 32767, -32767, 0]
        );
    }

    #[test]
    fn round_trips_within_one_least_significant_bit() {
        let samples: Vec<i16> = vec![-32767, -12345, -1, 0, 1, 777, 32767];
        let back = to_pcm16(&from_pcm16(&samples)).unwrap();
        for (a, b) in samples.iter().zip(&back) {
            assert!((i32::from(*a) - i32::from(*b)).abs() <= 1, "{a} -> {b}");
        }
    }

    #[test]
    fn rejects_an_empty_waveform() {
        assert!(to_pcm16(&Array::zeros(&[0], Dtype::Float32)).is_err());
    }
}
