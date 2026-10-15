//! Short-Time Fourier Transform (STFT) implementation using MLX.

use crate::error::Result;
use pmetal_runtime::compat::{Array, fft, ops};

/// STFT configuration.
#[derive(Debug, Clone)]
pub struct StftConfig {
    /// FFT size.
    pub n_fft: i32,
    /// Hop size in samples.
    pub hop_length: i32,
    /// Window size (defaults to n_fft).
    pub win_length: Option<i32>,
    /// Whether to center the signal with padding.
    pub center: bool,
    /// Padding mode when centering.
    pub pad_mode: PadMode,
}

/// Padding modes for STFT.
#[derive(Debug, Clone, Copy, Default)]
pub enum PadMode {
    /// Reflect padding (mirror).
    #[default]
    Reflect,
    /// Zero padding.
    Zeros,
    /// Replicate edge values.
    Replicate,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            hop_length: 256,
            win_length: None,
            center: true,
            pad_mode: PadMode::Reflect,
        }
    }
}

/// Create a Hann window.
///
/// # Arguments
/// * `size` - Window size
///
/// # Returns
/// Hann window as `[size]` array
pub fn hann_window(size: i32) -> Result<Array> {
    // hann[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
    let n = Array::arange(size, 10); // float32 arange [0..size)
    let pi = std::f32::consts::PI;
    let scale = Array::from_f32(2.0 * pi / (size - 1) as f32);
    let cos_term = n.multiply(&scale).cos();
    let half = Array::from_f32(0.5);
    let one = Array::from_f32(1.0);

    Ok(half.multiply(&one.subtract(&cos_term)))
}

/// Compute Short-Time Fourier Transform.
///
/// # Arguments
/// * `signal` - Input audio signal `[samples]` or `[batch, samples]`
/// * `config` - STFT configuration
///
/// # Returns
/// Complex STFT output [batch, n_fft/2+1, frames] or [n_fft/2+1, frames]
pub fn stft(signal: &Array, config: &StftConfig) -> Result<Array> {
    let win_length = config.win_length.unwrap_or(config.n_fft);

    // Create Hann window
    let window = hann_window(win_length)?;

    // Pad window to n_fft if needed
    let window = if win_length < config.n_fft {
        let pad_left = (config.n_fft - win_length) / 2;
        let pad_right = config.n_fft - win_length - pad_left;
        let zeros_left = Array::zeros(&[pad_left], 10);
        let zeros_right = Array::zeros(&[pad_right], 10);
        ops::concatenate_axis(&[&zeros_left, &window, &zeros_right], 0)
    } else {
        window
    };

    // Handle batched vs unbatched input
    let (signal, was_1d) = if signal.ndim() == 1 {
        (signal.reshape(&[1, -1]), true)
    } else {
        (signal.clone(), false)
    };

    let _batch_size = signal.dim(0);
    let _signal_length = signal.dim(1);

    // Center padding
    let signal = if config.center {
        let pad_amount = config.n_fft / 2;
        pad_signal(&signal, pad_amount, config.pad_mode)?
    } else {
        signal
    };

    let padded_length = signal.dim(1);

    // Calculate number of frames
    let num_frames = (padded_length - config.n_fft) / config.hop_length + 1;

    // Frame the signal using slice-based approach
    let batch = signal.dim(0);
    let mut frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        let start = i * config.hop_length;
        let end = start + config.n_fft;
        let frame = signal.slice(&[0, start], &[batch, end]);
        frames.push(frame);
    }

    // Stack frames: [batch, frames, n_fft]
    let framed = ops::stack_axis(&frames, 1);

    // Apply window: [batch, frames, n_fft] * [n_fft]
    let windowed = framed.multiply(&window);

    // Compute FFT along last axis
    let spectrum = fft::rfft(&windowed, Some(config.n_fft), -1);

    // Transpose to [batch, freq, frames]
    let spectrum = spectrum.transpose_axes(&[0, 2, 1]);

    // Remove batch dim if input was 1D
    if was_1d {
        Ok(spectrum.squeeze_all())
    } else {
        Ok(spectrum)
    }
}

/// Compute inverse STFT using overlap-add reconstruction.
///
/// # Arguments
/// * `stft_matrix` - STFT output [batch, n_fft/2+1, frames] or [n_fft/2+1, frames]
/// * `config` - STFT configuration
///
/// # Returns
/// Reconstructed audio signal `[batch, samples]` or `[samples]`
pub fn istft(stft_matrix: &Array, config: &StftConfig) -> Result<Array> {
    let win_length = config.win_length.unwrap_or(config.n_fft);

    // Create Hann window
    let window = hann_window(win_length)?;

    // Pad window to n_fft if needed
    let window = if win_length < config.n_fft {
        let pad_left = (config.n_fft - win_length) / 2;
        let pad_right = config.n_fft - win_length - pad_left;
        let zeros_left = Array::zeros(&[pad_left], 10);
        let zeros_right = Array::zeros(&[pad_right], 10);
        ops::concatenate_axis(&[&zeros_left, &window, &zeros_right], 0)
    } else {
        window
    };

    // Handle batched input: normalize to [batch, n_fft/2+1, frames]
    let (stft_matrix, was_2d) = if stft_matrix.ndim() == 2 {
        (
            stft_matrix.reshape(&[1, stft_matrix.dim(0), stft_matrix.dim(1)]),
            true,
        )
    } else {
        (stft_matrix.clone(), false)
    };

    let batch_size = stft_matrix.dim(0);
    let num_frames = stft_matrix.dim(2);
    let n_fft = config.n_fft;
    let hop_length = config.hop_length;

    // Transpose to [batch, frames, freq] for irfft along last axis
    let stft_transposed = stft_matrix.transpose_axes(&[0, 2, 1]);

    // Inverse FFT: [batch, frames, n_fft]
    let ifft_frames = fft::irfft(&stft_transposed, Some(n_fft), -1);

    // Apply synthesis window: [batch, frames, n_fft] * [n_fft]
    let windowed_frames = ifft_frames.multiply(&window);

    // Calculate full output length before optional center-trim
    let output_length = n_fft + (num_frames - 1) * hop_length;

    // Precompute window norm denominator using squared window
    let window_sq = window.multiply(&window);

    // Overlap-add via pad-and-sum
    let mut output_sum = Array::zeros(&[batch_size, output_length], 10);
    let mut norm_sum = Array::zeros(&[output_length], 10);

    for i in 0..num_frames {
        let offset = i * hop_length;
        let pad_before = offset;
        let pad_after = output_length - offset - n_fft;

        // Extract frame i for all batches: [batch, n_fft]
        let frame = windowed_frames
            .slice(&[0, i, 0], &[batch_size, i + 1, n_fft])
            .reshape(&[batch_size, n_fft]);

        // Pad frame along axis 1: [batch, output_length]
        let padded_frame = frame.pad_constant(&[0, 0, pad_before, pad_after], 0.0);

        output_sum = output_sum.add(&padded_frame);

        // Pad window_sq (1-D) along axis 0: [output_length]
        let padded_wsq = window_sq.pad_constant(&[pad_before, pad_after], 0.0);
        norm_sum = norm_sum.add(&padded_wsq);
    }

    // Normalize: divide by window norm, guarding against near-zero denominators
    let eps = Array::from_f32(1e-8_f32);
    let norm_safe = ops::maximum(&norm_sum, &eps);
    // Broadcast norm_safe [output_length] across batch dimension for division
    let norm_broadcast = ops::broadcast_to(&norm_safe, &[batch_size, output_length]);
    let output = output_sum.divide(&norm_broadcast);

    // If center=true was used during forward STFT, trim the n_fft/2 padding on each side
    let output = if config.center {
        let trim = n_fft / 2;
        let trimmed_length = output_length - 2 * trim;
        // Slice along axis 1
        output.slice(&[0, trim], &[batch_size, trim + trimmed_length])
    } else {
        output
    };

    if was_2d {
        Ok(output.squeeze_all())
    } else {
        Ok(output)
    }
}

/// Pad signal for STFT.
fn pad_signal(signal: &Array, pad_amount: i32, mode: PadMode) -> Result<Array> {
    let batch_size = signal.dim(0);
    let length = signal.dim(1);

    match mode {
        PadMode::Zeros => {
            let left_pad = Array::zeros(&[batch_size, pad_amount], 10);
            let right_pad = Array::zeros(&[batch_size, pad_amount], 10);
            Ok(ops::concatenate_axis(&[&left_pad, signal, &right_pad], 1))
        }
        PadMode::Reflect => {
            // Reflect padding: mirror the signal at boundaries
            // Left reflection: indices [pad_amount, pad_amount-1, ..., 1]
            let left_indices: Vec<i32> = (1..=pad_amount).rev().collect();
            let left_pad = if !left_indices.is_empty() {
                let indices = Array::from_i32_slice(&left_indices);
                signal.take_axis(&indices, 1)
            } else {
                Array::zeros(&[batch_size, 0], 10)
            };

            // Right reflection: indices [length-2, length-3, ..., length-pad_amount-1]
            let right_indices: Vec<i32> = ((length - pad_amount - 1)..(length - 1)).rev().collect();
            let right_pad = if !right_indices.is_empty() {
                let indices = Array::from_i32_slice(&right_indices);
                signal.take_axis(&indices, 1)
            } else {
                Array::zeros(&[batch_size, 0], 10)
            };

            Ok(ops::concatenate_axis(&[&left_pad, signal, &right_pad], 1))
        }
        PadMode::Replicate => {
            // Replicate edge values
            let left_val = signal.slice(&[0, 0], &[batch_size, 1]);
            let right_start = length - 1;
            let right_val = signal.slice(&[0, right_start], &[batch_size, length]);

            let left_pad = ops::broadcast_to(&left_val, &[batch_size, pad_amount]);
            let right_pad = ops::broadcast_to(&right_val, &[batch_size, pad_amount]);

            Ok(ops::concatenate_axis(&[&left_pad, signal, &right_pad], 1))
        }
    }
}

/// Compute magnitude spectrogram from complex STFT.
pub fn stft_magnitude(stft_matrix: &Array) -> Result<Array> {
    // |z| = sqrt(real² + imag²)
    Ok(stft_matrix.abs_val())
}

/// Compute power spectrogram from complex STFT.
pub fn stft_power(stft_matrix: &Array) -> Result<Array> {
    // |z|² = real² + imag²
    let mag = stft_matrix.abs_val();
    Ok(mag.multiply(&mag))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = hann_window(4).unwrap();
        let w2 = window.clone();
        w2.eval();
        assert_eq!(w2.shape(), &[4]);

        // Hann window should be symmetric and start/end near 0
        // hann(4) = [0, 0.5, 1, 0.5] approximately
    }

    #[test]
    fn test_stft_config() {
        let config = StftConfig::default();
        assert_eq!(config.n_fft, 1024);
        assert_eq!(config.hop_length, 256);
    }

    /// Verify that istft(stft(x)) ≈ x for a simple sine wave.
    ///
    /// The Hann window satisfies the COLA (Constant Overlap-Add) condition for
    /// hop_length = n_fft / 4, so perfect reconstruction is expected up to
    /// floating-point tolerance.
    #[test]
    fn test_stft_istft_roundtrip() {
        // Use small n_fft to keep the test fast; hop = n_fft/4 satisfies COLA.
        let n_fft = 64;
        let hop_length = n_fft / 4;
        let num_samples = 512;

        let config = StftConfig {
            n_fft,
            hop_length,
            win_length: None,
            center: true,
            pad_mode: PadMode::Reflect,
        };

        // Build a 440 Hz sine wave at 16 kHz sample rate (unit amplitude).
        let pi = std::f32::consts::PI;
        let samples: Vec<f32> = (0..num_samples)
            .map(|n| (2.0 * pi * 440.0 * n as f32 / 16000.0).sin())
            .collect();
        let signal = Array::from_f32_slice(&samples, &[num_samples]);

        // Forward STFT.
        let spectrum = stft(&signal, &config).unwrap();

        // Inverse STFT (overlap-add reconstruction).
        let reconstructed = istft(&spectrum, &config).unwrap();
        let r2 = reconstructed.clone();
        r2.eval();

        // The reconstructed signal should have the same length as the original.
        assert_eq!(r2.shape(), &[num_samples]);

        // Verify reconstruction quality: max absolute error should be < 1e-3.
        let diff = r2.subtract(&signal);
        let abs_diff = diff.abs_val();
        // Reduce to scalar maximum.
        let max_err_arr = abs_diff.max(None);
        max_err_arr.eval();
        let max_err: f32 = max_err_arr.item_f32();
        assert!(
            max_err < 1e-3,
            "STFT round-trip error too large: max |error| = {max_err}"
        );
    }

    /// Verify istft produces correct output shape for batched input.
    #[test]
    fn test_istft_batched_shape() {
        let n_fft = 32;
        let hop_length = 8;
        let num_samples = 128;
        let batch_size = 3;

        let config = StftConfig {
            n_fft,
            hop_length,
            win_length: None,
            center: false,
            pad_mode: PadMode::Zeros,
        };

        // Batch of signals: [batch, samples]
        let samples: Vec<f32> = (0..(batch_size * num_samples))
            .map(|i| (i as f32 / num_samples as f32).sin())
            .collect();
        let signal = Array::from_f32_slice(&samples, &[batch_size, num_samples]);

        let spectrum = stft(&signal, &config).unwrap();
        let reconstructed = istft(&spectrum, &config).unwrap();
        let r2 = reconstructed.clone();
        r2.eval();

        // Output batch dimension must be preserved.
        assert_eq!(r2.dim(0), batch_size);
    }
}
