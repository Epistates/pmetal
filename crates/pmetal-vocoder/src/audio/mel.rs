//! Mel filterbank and mel spectrogram computation.
//!
//! The filterbank follows librosa's `filters.mel` (Slaney mel scale and area
//! normalisation by default), and [`MelFrontEnd`] reproduces the reference
//! BigVGAN front end exactly, so a pretrained generator sees the same mel
//! spectrogram it was trained on.

use super::stft::{PadMode, StftConfig, Window, pad_signal, stft};
use crate::config::BigVGANConfig;
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::{Array, Dtype, ops};

/// Mel filterbank configuration.
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Sampling rate in Hz.
    pub sr: i32,
    /// FFT size.
    pub n_fft: i32,
    /// Number of mel frequency bins.
    pub n_mels: i32,
    /// Minimum frequency in Hz.
    pub fmin: f32,
    /// Maximum frequency in Hz (defaults to `sr / 2`).
    pub fmax: Option<f32>,
    /// Whether to use the HTK mel formula (otherwise Slaney's).
    pub htk: bool,
    /// Normalisation of each filter.
    pub norm: MelNorm,
}

/// Mel filterbank normalisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MelNorm {
    /// Triangles with unit peak.
    None,
    /// Slaney-style: each triangle has unit area (divided by its Hz width).
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

// Slaney's mel scale: linear below 1 kHz, logarithmic above.
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;

fn log_step() -> f64 {
    6.4f64.ln() / 27.0
}

fn hz_to_mel_f64(freq: f64, htk: bool) -> f64 {
    if htk {
        2595.0 * (1.0 + freq / 700.0).log10()
    } else if freq >= MIN_LOG_HZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / log_step()
    } else {
        freq / F_SP
    }
}

fn mel_to_hz_f64(mel: f64, htk: bool) -> f64 {
    if htk {
        700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
    } else if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * log_step()).exp()
    } else {
        F_SP * mel
    }
}

/// Convert frequency in Hz to the mel scale (`htk`: HTK formula, else Slaney).
pub fn hz_to_mel(freq: f32, htk: bool) -> f32 {
    hz_to_mel_f64(f64::from(freq), htk) as f32
}

/// Convert the mel scale to frequency in Hz (`htk`: HTK formula, else Slaney).
pub fn mel_to_hz(mel: f32, htk: bool) -> f32 {
    mel_to_hz_f64(f64::from(mel), htk) as f32
}

/// The triangular mel filterbank as a row-major `[n_mels, n_fft / 2 + 1]`
/// matrix of host floats (librosa's `filters.mel`, computed in double precision).
pub fn mel_filterbank_values(config: &MelConfig) -> Result<Vec<f32>> {
    if config.sr < 1 || config.n_fft < 2 || config.n_mels < 1 {
        return Err(VocoderError::Audio(format!(
            "invalid mel settings (sr {}, n_fft {}, n_mels {})",
            config.sr, config.n_fft, config.n_mels
        )));
    }
    let fmax = f64::from(config.fmax.unwrap_or(config.sr as f32 / 2.0));
    let fmin = f64::from(config.fmin);
    if !(0.0 <= fmin && fmin < fmax && fmax <= f64::from(config.sr) / 2.0 + 1e-3) {
        return Err(VocoderError::Audio(format!(
            "need 0 <= fmin < fmax <= sr / 2, got fmin {fmin}, fmax {fmax}, sr {}",
            config.sr
        )));
    }
    let n_freqs = (config.n_fft / 2 + 1) as usize;
    let n_mels = config.n_mels as usize;

    let mel_min = hz_to_mel_f64(fmin, config.htk);
    let mel_max = hz_to_mel_f64(fmax, config.htk);
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64;
            mel_to_hz_f64(mel, config.htk)
        })
        .collect();
    let bin_hz = |k: usize| k as f64 * f64::from(config.sr) / f64::from(config.n_fft);

    let mut fb = vec![0.0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let (lo, mid, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        let scale = match config.norm {
            MelNorm::Slaney => 2.0 / (hi - lo),
            MelNorm::None => 1.0,
        };
        for k in 0..n_freqs {
            let f = bin_hz(k);
            let rising = (f - lo) / (mid - lo);
            let falling = (hi - f) / (hi - mid);
            let weight = rising.min(falling).max(0.0);
            fb[m * n_freqs + k] = (f64::from(weight as f32) * scale) as f32;
        }
    }
    Ok(fb)
}

/// The mel filterbank as an array `[n_mels, n_fft / 2 + 1]`.
pub fn mel_filterbank(config: &MelConfig) -> Result<Array> {
    let values = mel_filterbank_values(config)?;
    Ok(Array::from_f32_slice(
        &values,
        &[config.n_mels, config.n_fft / 2 + 1],
    ))
}

/// Apply a `[n_mels, freq]` filterbank to a magnitude spectrogram
/// (`[freq, frames]` or `[batch, freq, frames]`).
fn apply_filterbank(filterbank: &Array, magnitude: &Array) -> Array {
    filterbank.matmul(magnitude)
}

/// Mel spectrogram (magnitude, not log) of `audio`.
///
/// * `audio` - `[samples]` or `[batch, samples]`.
///
/// Returns `[n_mels, frames]` or `[batch, n_mels, frames]`.
pub fn mel_spectrogram(
    audio: &Array,
    config: &MelConfig,
    stft_config: &StftConfig,
) -> Result<Array> {
    if stft_config.n_fft != config.n_fft {
        return Err(VocoderError::Audio(format!(
            "the STFT uses n_fft {} but the mel filterbank was built for {}",
            stft_config.n_fft, config.n_fft
        )));
    }
    let magnitude = stft(audio, stft_config)?.abs_val();
    Ok(apply_filterbank(&mel_filterbank(config)?, &magnitude))
}

/// `log(max(x, clip) * c)`: the reference's dynamic range compression.
pub fn dynamic_range_compression(
    x: &Array,
    c: Option<f32>,
    clip_val: Option<f32>,
) -> Result<Array> {
    let clipped = ops::maximum(x, &Array::from_f32(clip_val.unwrap_or(1e-5)));
    Ok(clipped.multiply(&Array::from_f32(c.unwrap_or(1.0))).log())
}

/// `exp(x) / c`: the inverse of [`dynamic_range_compression`] above the clip.
pub fn dynamic_range_decompression(x: &Array, c: Option<f32>) -> Result<Array> {
    Ok(x.exp().divide(&Array::from_f32(c.unwrap_or(1.0))))
}

/// Log-compress a mel spectrogram, clamping at `clip_val` (default `1e-5`).
pub fn log_mel_spectrogram(mel_spec: &Array, clip_val: Option<f32>) -> Result<Array> {
    dynamic_range_compression(mel_spec, None, clip_val)
}

/// The mel front end a BigVGAN model was trained with.
///
/// Reflect-pads the waveform by `(n_fft - hop) / 2` on each side, takes a
/// Hann-windowed STFT without further centring, forms the magnitude as
/// `sqrt(re^2 + im^2 + 1e-9)`, projects onto the Slaney mel filterbank and
/// log-compresses with a floor of `1e-5`. A waveform of `T` samples gives
/// `T / hop_size` frames, so the generator returns exactly `T` samples.
#[derive(Debug)]
pub struct MelFrontEnd {
    stft: StftConfig,
    filterbank: Array,
    pad: i32,
}

/// Added under the square root of the magnitude, as in the reference.
const MAGNITUDE_EPS: f32 = 1e-9;

impl MelFrontEnd {
    /// The front end a model is conditioned on (the configuration's `fmax`).
    pub fn new(config: &BigVGANConfig) -> Result<Self> {
        Self::with_fmax(config, config.fmax)
    }

    /// The front end of the single-scale mel loss, which uses the
    /// configuration's `fmax_for_loss` instead of `fmax`.
    pub fn for_loss(config: &BigVGANConfig) -> Result<Self> {
        Self::with_fmax(config, config.fmax_for_loss)
    }

    fn with_fmax(config: &BigVGANConfig, fmax: Option<i32>) -> Result<Self> {
        let mel = MelConfig {
            sr: config.sampling_rate,
            n_fft: config.n_fft,
            n_mels: config.num_mels,
            fmin: config.fmin as f32,
            fmax: fmax.map(|f| f as f32),
            htk: false,
            norm: MelNorm::Slaney,
        };
        Ok(Self {
            stft: StftConfig {
                n_fft: config.n_fft,
                hop_length: config.hop_size,
                win_length: Some(config.win_size),
                center: false,
                pad_mode: PadMode::Reflect,
                window: Window::Hann,
            },
            filterbank: mel_filterbank(&mel)?,
            pad: (config.n_fft - config.hop_size) / 2,
        })
    }

    /// Log-mel spectrogram `[batch, n_mels, frames]` of `wav`
    /// (`[samples]` or `[batch, samples]`, nominally in `[-1, 1]`).
    pub fn forward(&self, wav: &Array) -> Result<Array> {
        let wav = if wav.ndim() == 1 {
            wav.reshape(&[1, -1])
        } else {
            wav.clone()
        };
        let padded = pad_signal(&wav.as_dtype(Dtype::Float32), self.pad, PadMode::Reflect)?;
        let power = {
            let mag = stft(&padded, &self.stft)?.abs_val();
            mag.multiply(&mag)
        };
        let magnitude = power.add(&Array::from_f32(MAGNITUDE_EPS)).sqrt();
        let mel = apply_filterbank(&self.filterbank, &magnitude);
        log_mel_spectrogram(&mel, None)
    }
}
