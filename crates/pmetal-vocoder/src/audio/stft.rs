//! Short-Time Fourier Transform on MLX arrays.
//!
//! Conventions follow `torch.stft` / `torch.istft` with a Hann window, so
//! spectrograms match the reference vocoder's front end bin for bin.

use crate::error::{Result, VocoderError};
use crate::nn::overlap_add;
use pmetal_runtime::compat::ops::{self, PadMode as MlxPad};
use pmetal_runtime::compat::{Array, Dtype, fft};

/// STFT configuration.
#[derive(Debug, Clone)]
pub struct StftConfig {
    /// FFT size.
    pub n_fft: i32,
    /// Hop size in samples.
    pub hop_length: i32,
    /// Window size (defaults to `n_fft`); the window is centred in the FFT frame.
    pub win_length: Option<i32>,
    /// Whether to pad the signal by `n_fft / 2` each side so frame `t` is
    /// centred on sample `t * hop_length` (`torch.stft`'s `center`).
    pub center: bool,
    /// Padding mode when centring.
    pub pad_mode: PadMode,
    /// Analysis window shape.
    pub window: Window,
}

/// Analysis window shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Window {
    /// Periodic Hann window (`torch.hann_window`).
    #[default]
    Hann,
    /// All ones (`torch.stft` with no window argument).
    Rectangular,
}

/// Padding modes for STFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PadMode {
    /// Reflect padding (mirror about the edge sample).
    #[default]
    Reflect,
    /// Zero padding.
    Zeros,
    /// Replicate edge values.
    Replicate,
}

impl PadMode {
    fn to_mlx(self) -> MlxPad {
        match self {
            Self::Reflect => MlxPad::Reflect,
            Self::Zeros => MlxPad::Constant,
            Self::Replicate => MlxPad::Edge,
        }
    }
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            hop_length: 256,
            win_length: None,
            center: true,
            pad_mode: PadMode::Reflect,
            window: Window::Hann,
        }
    }
}

impl StftConfig {
    fn validate(&self) -> Result<()> {
        let win = self.win_length.unwrap_or(self.n_fft);
        if self.n_fft < 1 || self.hop_length < 1 || win < 1 || win > self.n_fft {
            return Err(VocoderError::Audio(format!(
                "invalid STFT settings (n_fft {}, hop {}, win {win})",
                self.n_fft, self.hop_length
            )));
        }
        Ok(())
    }

    /// Number of frames `stft` produces for a signal of `samples` samples.
    pub fn num_frames(&self, samples: i32) -> i32 {
        let padded = if self.center {
            samples + 2 * (self.n_fft / 2)
        } else {
            samples
        };
        self.num_frames_padded(padded)
    }

    fn num_frames_padded(&self, padded: i32) -> i32 {
        if padded < self.n_fft {
            0
        } else {
            (padded - self.n_fft) / self.hop_length + 1
        }
    }
}

/// A Hann window of `size` samples.
///
/// `periodic` windows (the `torch.hann_window` / `scipy.signal.get_window`
/// default, and the one an STFT needs) divide by `size`; symmetric windows
/// divide by `size - 1`.
pub fn hann_window(size: i32, periodic: bool) -> Result<Array> {
    if size < 1 {
        return Err(VocoderError::Audio("window size must be positive".into()));
    }
    let denom = if periodic {
        f64::from(size)
    } else {
        f64::from((size - 1).max(1))
    };
    let w: Vec<f32> = (0..size)
        .map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * f64::from(n) / denom).cos()) as f32)
        .collect();
    Ok(Array::from_f32_slice(&w, &[size]))
}

/// The analysis window padded (both sides) to `n_fft`.
fn fft_window(config: &StftConfig) -> Result<Array> {
    let win_length = config.win_length.unwrap_or(config.n_fft);
    let window = match config.window {
        Window::Hann => hann_window(win_length, true)?,
        Window::Rectangular => Array::ones(&[win_length], Dtype::Float32),
    };
    if win_length == config.n_fft {
        return Ok(window);
    }
    let left = (config.n_fft - win_length) / 2;
    Ok(ops::pad(
        &window,
        &[(left, config.n_fft - win_length - left)],
        None,
        None,
    ))
}

/// Pad `[batch, time]` by `amount` on both time edges.
pub fn pad_signal(signal: &Array, amount: i32, mode: PadMode) -> Result<Array> {
    if mode == PadMode::Reflect && amount >= signal.dim(1) {
        return Err(VocoderError::Audio(format!(
            "reflect padding of {amount} needs a signal longer than {amount} samples, got {}",
            signal.dim(1)
        )));
    }
    Ok(ops::pad(
        signal,
        &[(0, 0), (amount, amount)],
        Some(mode.to_mlx()),
        None,
    ))
}

/// Pad (if centring) and cut `signal` (`[batch, samples]`) into windowed
/// frames `[batch, frames, n_fft]`.
fn windowed_frames(signal: &Array, config: &StftConfig) -> Result<Array> {
    config.validate()?;
    if signal.ndim() != 2 {
        return Err(VocoderError::Audio(format!(
            "stft expects [samples] or [batch, samples], got shape {:?}",
            signal.shape()
        )));
    }
    let signal = signal.as_dtype(Dtype::Float32);
    let signal = if config.center {
        pad_signal(&signal, config.n_fft / 2, config.pad_mode)?
    } else {
        signal
    };

    let frames = config.num_frames_padded(signal.dim(1));
    if frames < 1 {
        return Err(VocoderError::Audio(format!(
            "a signal of {} samples is shorter than n_fft {}",
            signal.dim(1),
            config.n_fft
        )));
    }

    // Gather every frame at once: index[f, j] = f * hop + j.
    let n = config.n_fft;
    let index: Vec<i32> = (0..frames)
        .flat_map(|f| (0..n).map(move |j| f * config.hop_length + j))
        .collect();
    let index = Array::from_i32_slice_shaped(&index, &[frames, n]);
    let framed = signal.take_axis(&index, 1); // [batch, frames, n_fft]
    Ok(framed.multiply(&fft_window(config)?))
}

/// Compute the Short-Time Fourier Transform.
///
/// * `signal` - `[samples]` or `[batch, samples]`.
///
/// Returns the complex spectrogram `[freq, frames]` or `[batch, freq, frames]`
/// with `freq = n_fft / 2 + 1`.
pub fn stft(signal: &Array, config: &StftConfig) -> Result<Array> {
    let unbatched = signal.ndim() == 1;
    let signal = if unbatched {
        signal.reshape(&[1, -1])
    } else {
        signal.clone()
    };
    let framed = windowed_frames(&signal, config)?;
    let spectrum = fft::rfft(&framed, Some(config.n_fft), -1);
    let spectrum = spectrum.transpose_axes(&[0, 2, 1]); // [batch, freq, frames]
    Ok(if unbatched {
        spectrum.squeeze_axes(&[0])
    } else {
        spectrum
    })
}

/// The STFT as separate real and imaginary parts, each `[batch, frames, freq]`
/// (time-major, for networks that treat the spectrogram as an image).
///
/// Computed as a matrix product with the DFT basis, so both parts stay real,
/// differentiable arrays; `re + i im` equals [`stft`] up to rounding.
pub fn stft_parts(signal: &Array, config: &StftConfig) -> Result<(Array, Array)> {
    let framed = windowed_frames(signal, config)?;
    let n = config.n_fft as usize;
    let freqs = n / 2 + 1;
    let mut cos = Vec::with_capacity(n * freqs);
    let mut sin = Vec::with_capacity(n * freqs);
    for t in 0..n {
        for k in 0..freqs {
            // Reduce the phase index first so large products stay exact.
            let phase = 2.0 * std::f64::consts::PI * ((t * k) % n) as f64 / n as f64;
            cos.push(phase.cos() as f32);
            sin.push(-phase.sin() as f32);
        }
    }
    let shape = [config.n_fft, freqs as i32];
    let re = framed.matmul(&Array::from_f32_slice(&cos, &shape));
    let im = framed.matmul(&Array::from_f32_slice(&sin, &shape));
    Ok((re, im))
}

/// Inverse STFT by windowed overlap-add with squared-window normalisation
/// (`torch.istft`).
///
/// * `stft_matrix` - `[freq, frames]` or `[batch, freq, frames]`.
/// * `length` - trim or zero-pad the result to this many samples.
///
/// Returns `[samples]` or `[batch, samples]`.
pub fn istft(stft_matrix: &Array, config: &StftConfig, length: Option<i32>) -> Result<Array> {
    config.validate()?;
    let unbatched = stft_matrix.ndim() == 2;
    let spec = if unbatched {
        stft_matrix.reshape(&[1, stft_matrix.dim(0), stft_matrix.dim(1)])
    } else {
        stft_matrix.clone()
    };
    let (n, hop) = (config.n_fft, config.hop_length);
    if spec.dim(1) != n / 2 + 1 {
        return Err(VocoderError::ShapeMismatch {
            expected: vec![-1, n / 2 + 1, -1],
            actual: spec.shape().to_vec(),
        });
    }
    let (batch, frames) = (spec.dim(0), spec.dim(2));
    let window = fft_window(config)?;

    let time = fft::irfft(&spec.transpose_axes(&[0, 2, 1]), Some(n), -1); // [batch, frames, n]
    let windowed = time.multiply(&window).reshape(&[batch, frames, n, 1]);
    let signal = overlap_add(&windowed, hop).reshape(&[batch, -1]);

    let window_sq = window.multiply(&window).reshape(&[1, 1, n, 1]);
    let window_sq = ops::broadcast_to(&window_sq, &[1, frames, n, 1]);
    let envelope = overlap_add(&window_sq, hop).reshape(&[1, -1]);

    // Frames at the very edges are covered by a single window tail; leave
    // samples where the envelope vanishes untouched rather than divide by ~0.
    let safe = ops::maximum(&envelope, &Array::from_f32(1e-11));
    let mut out = signal.divide(&safe);

    let total = out.dim(1);
    let start = if config.center { n / 2 } else { 0 };
    let end = match length {
        Some(l) => start + l,
        None if config.center => total - n / 2,
        None => total,
    };
    if end > total {
        out = ops::pad(&out, &[(0, 0), (0, end - total)], None, None);
    }
    out = ops::slice_axis(&out, 1, start, end);
    Ok(if unbatched {
        out.squeeze_axes(&[0])
    } else {
        out
    })
}

/// Magnitude of a complex STFT, `|z|`.
pub fn stft_magnitude(stft_matrix: &Array) -> Result<Array> {
    Ok(stft_matrix.abs_val())
}

/// Power of a complex STFT, `|z|^2`.
pub fn stft_power(stft_matrix: &Array) -> Result<Array> {
    let mag = stft_matrix.abs_val();
    Ok(mag.multiply(&mag))
}
