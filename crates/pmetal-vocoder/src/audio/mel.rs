//! Mel filterbank and mel spectrogram computation.

use crate::error::Result;
use mlx_rs::Array;

/// Mel filterbank configuration.
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Sampling rate in Hz.
    pub sr: i32,
    /// Number of FFT bins.
    pub n_fft: i32,
    /// Number of mel frequency bins.
    pub n_mels: i32,
    /// Minimum frequency in Hz.
    pub fmin: f32,
    /// Maximum frequency in Hz (defaults to sr/2).
    pub fmax: Option<f32>,
    /// Whether to use HTK formula (vs Slaney).
    pub htk: bool,
    /// Normalization type for filterbank.
    pub norm: MelNorm,
}

/// Mel filterbank normalization.
#[derive(Debug, Clone, Copy, Default)]
pub enum MelNorm {
    /// No normalization.
    None,
    /// Slaney-style normalization (area = 1).
    #[default]
    Slaney,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sr: 24000,
            n_fft: 1024,
            n_mels: 100,
            fmin: 0.0,
            fmax: None,
            htk: false,
            norm: MelNorm::Slaney,
        }
    }
}

/// Convert frequency in Hz to mel scale.
///
/// # Arguments
/// * `freq` - Frequency in Hz
/// * `htk` - Use HTK formula if true, Slaney otherwise
pub fn hz_to_mel(freq: f32, htk: bool) -> f32 {
    if htk {
        // HTK formula: 2595 * log10(1 + f/700)
        2595.0 * (1.0 + freq / 700.0).log10()
    } else {
        // Slaney formula (used by librosa)
        let f_min = 0.0;
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f32).ln() / 27.0; // log(6.4) / 27

        if freq >= min_log_hz {
            min_log_mel + (freq / min_log_hz).ln() / logstep
        } else {
            (freq - f_min) / f_sp
        }
    }
}

/// Convert mel scale to frequency in Hz.
///
/// # Arguments
/// * `mel` - Mel value
/// * `htk` - Use HTK formula if true, Slaney otherwise
pub fn mel_to_hz(mel: f32, htk: bool) -> f32 {
    if htk {
        // HTK formula: 700 * (10^(m/2595) - 1)
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    } else {
        // Slaney formula
        let f_min = 0.0;
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f32).ln() / 27.0;

        if mel >= min_log_mel {
            min_log_hz * ((mel - min_log_mel) * logstep).exp()
        } else {
            f_min + f_sp * mel
        }
    }
}

/// Create mel filterbank matrix.
///
/// # Arguments
/// * `config` - Mel filterbank configuration
///
/// # Returns
/// Mel filterbank matrix [n_mels, n_fft/2+1]
pub fn mel_filterbank(config: &MelConfig) -> Result<Array> {
    let fmax = config.fmax.unwrap_or(config.sr as f32 / 2.0);
    let n_freqs = config.n_fft / 2 + 1;

    // Compute mel points
    let mel_min = hz_to_mel(config.fmin, config.htk);
    let mel_max = hz_to_mel(fmax, config.htk);

    // Linearly spaced mel points
    let n_mels_plus_2 = config.n_mels + 2;
    let mel_points: Vec<f32> = (0..n_mels_plus_2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels_plus_2 - 1) as f32)
        .collect();

    // Convert to Hz
    let hz_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| mel_to_hz(m, config.htk))
        .collect();

    // Convert to FFT bin indices
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&f| (config.n_fft as f32 + 1.0) * f / config.sr as f32)
        .collect();

    // Create filterbank matrix
    let mut filterbank = vec![0.0f32; (config.n_mels * n_freqs) as usize];

    for m in 0..config.n_mels as usize {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        for k in 0..n_freqs as usize {
            let k_f = k as f32;

            let weight = if k_f >= f_left && k_f <= f_center {
                // Rising slope
                (k_f - f_left) / (f_center - f_left + 1e-10)
            } else if k_f >= f_center && k_f <= f_right {
                // Falling slope
                (f_right - k_f) / (f_right - f_center + 1e-10)
            } else {
                0.0
            };

            filterbank[m * n_freqs as usize + k] = weight;
        }

        // Apply Slaney normalization if requested
        if matches!(config.norm, MelNorm::Slaney) {
            let enorm = 2.0 / (hz_points[m + 2] - hz_points[m] + 1e-10);
            for k in 0..n_freqs as usize {
                filterbank[m * n_freqs as usize + k] *= enorm;
            }
        }
    }

    Ok(Array::from_slice(&filterbank, &[config.n_mels, n_freqs]))
}

/// Compute mel spectrogram from audio.
///
/// # Arguments
/// * `audio` - Audio signal [samples] or [batch, samples]
/// * `config` - Mel configuration
/// * `stft_config` - STFT configuration
///
/// # Returns
/// Mel spectrogram [batch, n_mels, frames] or [n_mels, frames]
pub fn mel_spectrogram(
    audio: &Array,
    config: &MelConfig,
    stft_config: &super::StftConfig,
) -> Result<Array> {
    // Compute STFT
    let stft_out = super::stft(audio, stft_config)?;

    // Compute magnitude
    let magnitude = super::stft_magnitude(&stft_out)?;

    // Get mel filterbank
    let mel_fb = mel_filterbank(config)?;

    // Apply filterbank: [n_mels, n_freqs] @ [n_freqs, frames]
    // Need to handle batched case
    let magnitude = if magnitude.ndim() == 2 {
        magnitude
    } else {
        // [batch, freq, frames] -> process each batch
        magnitude
    };

    // Transpose magnitude: [freq, frames] or [batch, freq, frames]
    // mel_fb @ magnitude for [n_mels, freq] @ [freq, frames] -> [n_mels, frames]
    if magnitude.ndim() == 2 {
        Ok(mel_fb.matmul(&magnitude)?)
    } else {
        // Batched: [batch, freq, frames]
        // Need to apply mel_fb to each batch element
        // This can be done with broadcasting or einsum
        let _batch_size = magnitude.dim(0);
        let _frames = magnitude.dim(2);

        // Transpose to [batch, frames, freq], then matmul, then transpose back
        let mag_t = magnitude.transpose_axes(&[0, 2, 1])?; // [batch, frames, freq]
        let mel_fb_t = mel_fb.transpose_axes(&[1, 0])?; // [freq, n_mels]
        let mel_spec = mag_t.matmul(&mel_fb_t)?; // [batch, frames, n_mels]
        Ok(mel_spec.transpose_axes(&[0, 2, 1])?) // [batch, n_mels, frames]
    }
}

/// Apply log compression to mel spectrogram.
///
/// # Arguments
/// * `mel_spec` - Mel spectrogram
/// * `clip_val` - Minimum value for clipping (default 1e-5)
///
/// # Returns
/// Log-compressed mel spectrogram
pub fn log_mel_spectrogram(mel_spec: &Array, clip_val: Option<f32>) -> Result<Array> {
    let clip = Array::from_f32(clip_val.unwrap_or(1e-5));
    let clipped = mlx_rs::ops::maximum(mel_spec, &clip)?;
    Ok(clipped.log()?)
}

/// Dynamic range compression for mel spectrogram.
///
/// # Arguments
/// * `mel_spec` - Mel spectrogram
/// * `c` - Compression factor (default 1.0)
/// * `clip_val` - Minimum value (default 1e-5)
///
/// # Returns
/// Compressed mel spectrogram
pub fn dynamic_range_compression(
    mel_spec: &Array,
    c: Option<f32>,
    clip_val: Option<f32>,
) -> Result<Array> {
    let c = c.unwrap_or(1.0);
    let clip = Array::from_f32(clip_val.unwrap_or(1e-5));
    let c_arr = Array::from_f32(c);

    let clipped = mlx_rs::ops::maximum(mel_spec, &clip)?;
    // log(1 + c * x) / log(1 + c)
    let one = Array::from_f32(1.0);
    let numerator = one.add(&clipped.multiply(&c_arr)?)?.log()?;
    let denominator = (1.0 + c).ln();
    Ok(numerator.divide(Array::from_f32(denominator))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_mel_htk() {
        // HTK formula
        let mel = hz_to_mel(1000.0, true);
        assert!((mel - 1000.0).abs() < 1.0); // ~1000 mels at 1000 Hz
    }

    #[test]
    fn test_hz_to_mel_slaney() {
        // Slaney formula
        let mel = hz_to_mel(1000.0, false);
        // At 1000 Hz, should be ~15 mels (linear region ends at 1000 Hz)
        assert!(mel > 10.0 && mel < 20.0);
    }

    #[test]
    fn test_mel_to_hz_roundtrip() {
        let freq = 2000.0;
        let mel = hz_to_mel(freq, false);
        let freq_back = mel_to_hz(mel, false);
        assert!((freq - freq_back).abs() < 1.0);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let config = MelConfig {
            sr: 24000,
            n_fft: 1024,
            n_mels: 80,
            fmin: 0.0,
            fmax: Some(12000.0),
            htk: false,
            norm: MelNorm::Slaney,
        };

        let fb = mel_filterbank(&config).unwrap();
        fb.eval().unwrap();

        // Should be [n_mels, n_fft/2+1] = [80, 513]
        assert_eq!(fb.shape(), &[80, 513]);
    }

    #[test]
    fn test_mel_filterbank_values() {
        let config = MelConfig {
            sr: 16000,
            n_fft: 512,
            n_mels: 40,
            fmin: 0.0,
            fmax: Some(8000.0),
            htk: false,
            norm: MelNorm::Slaney,
        };

        let fb = mel_filterbank(&config).unwrap();
        fb.eval().unwrap();

        // Sum of each filter should be reasonable (not all zeros)
        let row_sums = fb.sum_axis(1, None).unwrap();
        row_sums.eval().unwrap();

        // Each mel filter should sum to something positive
        // (Slaney normalization makes area = 2)
    }
}
