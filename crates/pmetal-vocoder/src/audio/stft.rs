//! Short-Time Fourier Transform (STFT) implementation using MLX.

use crate::error::{Result, VocoderError};
use mlx_rs::Array;

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
/// Hann window as [size] array
pub fn hann_window(size: i32) -> Result<Array> {
    // hann[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
    let n = mlx_rs::ops::arange::<i32, f32>(0, size, None)?;
    let pi = std::f32::consts::PI;
    let scale = Array::from_f32(2.0 * pi / (size - 1) as f32);
    let cos_term = (n.multiply(&scale)?).cos()?;
    let half = Array::from_f32(0.5);
    let one = Array::from_f32(1.0);

    Ok(half.multiply(&one.subtract(&cos_term)?)?)
}

/// Compute Short-Time Fourier Transform.
///
/// # Arguments
/// * `signal` - Input audio signal [samples] or [batch, samples]
/// * `config` - STFT configuration
///
/// # Returns
/// Complex STFT output [batch, n_fft/2+1, frames] or [n_fft/2+1, frames]
pub fn stft(signal: &Array, config: &StftConfig) -> Result<Array> {
    let win_length = config.win_length.unwrap_or(config.n_fft);

    use mlx_rs::ops::indexing::IndexOp;

    // Create Hann window
    let window = hann_window(win_length)?;

    // Pad window to n_fft if needed
    let window = if win_length < config.n_fft {
        let pad_left = (config.n_fft - win_length) / 2;
        let pad_right = config.n_fft - win_length - pad_left;
        let zeros_left = mlx_rs::ops::zeros::<f32>(&[pad_left])?;
        let zeros_right = mlx_rs::ops::zeros::<f32>(&[pad_right])?;
        mlx_rs::ops::concatenate_axis(&[&zeros_left, &window, &zeros_right], 0)?
    } else {
        window
    };

    // Handle batched vs unbatched input
    let (signal, was_1d) = if signal.ndim() == 1 {
        (signal.reshape(&[1, -1])?, true)
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

    // Frame the signal using as_strided or manual indexing
    // For now, use a loop-based approach (can optimize later with as_strided)
    let mut frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        let start = i * config.hop_length;
        let end = start + config.n_fft;
        let frame = signal.index((.., start..end));
        frames.push(frame);
    }

    // Stack frames: [batch, frames, n_fft]
    let frame_refs: Vec<&Array> = frames.iter().collect();
    let framed = mlx_rs::ops::stack_axis(&frame_refs, 1)?;

    // Apply window: [batch, frames, n_fft] * [n_fft]
    let windowed = framed.multiply(&window)?;

    // Compute FFT along last axis
    let spectrum = mlx_rs::fft::rfft(&windowed, Some(config.n_fft), -1)?;

    // Transpose to [batch, freq, frames]
    let spectrum = spectrum.transpose_axes(&[0, 2, 1])?;

    // Remove batch dim if input was 1D
    if was_1d {
        Ok(spectrum.squeeze()?)
    } else {
        Ok(spectrum)
    }
}

/// Compute inverse STFT.
///
/// # Arguments
/// * `stft_matrix` - STFT output [batch, n_fft/2+1, frames]
/// * `config` - STFT configuration
///
/// # Returns
/// Reconstructed audio signal [batch, samples]
pub fn istft(stft_matrix: &Array, config: &StftConfig) -> Result<Array> {
    let win_length = config.win_length.unwrap_or(config.n_fft);

    // Create Hann window
    let window = hann_window(win_length)?;

    // Pad window to n_fft if needed
    let window = if win_length < config.n_fft {
        let pad_left = (config.n_fft - win_length) / 2;
        let pad_right = config.n_fft - win_length - pad_left;
        let zeros_left = mlx_rs::ops::zeros::<f32>(&[pad_left])?;
        let zeros_right = mlx_rs::ops::zeros::<f32>(&[pad_right])?;
        mlx_rs::ops::concatenate_axis(&[&zeros_left, &window, &zeros_right], 0)?
    } else {
        window
    };

    // Handle batched input
    let (stft_matrix, was_2d) = if stft_matrix.ndim() == 2 {
        (stft_matrix.reshape(&[1, stft_matrix.dim(0), stft_matrix.dim(1)])?, true)
    } else {
        (stft_matrix.clone(), false)
    };

    let batch_size = stft_matrix.dim(0);
    let num_frames = stft_matrix.dim(2);

    // Transpose to [batch, frames, freq]
    let stft_matrix = stft_matrix.transpose_axes(&[0, 2, 1])?;

    // Inverse FFT
    let frames = mlx_rs::fft::irfft(&stft_matrix, Some(config.n_fft), -1)?;

    // Apply synthesis window
    let _frames = frames.multiply(&window)?;

    // Calculate output length
    let output_length = config.n_fft + (num_frames - 1) * config.hop_length;

    // Overlap-add - note: full implementation would require scatter_add
    // For now, return zeros as placeholder
    // TODO: Implement proper overlap-add with scatter or loop-based approach
    let output = mlx_rs::ops::zeros::<f32>(&[batch_size, output_length])?;

    if was_2d {
        Ok(output.squeeze()?)
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
            let left_pad = mlx_rs::ops::zeros::<f32>(&[batch_size, pad_amount])?;
            let right_pad = mlx_rs::ops::zeros::<f32>(&[batch_size, pad_amount])?;
            mlx_rs::ops::concatenate_axis(&[&left_pad, signal, &right_pad], 1)
        }
        PadMode::Reflect => {
            // Reflect padding: mirror the signal at boundaries
            // left: signal[pad_amount:0:-1]
            // right: signal[-2:-pad_amount-2:-1]

            // Left reflection
            let left_indices: Vec<i32> = (1..=pad_amount).rev().collect();
            let left_pad = if !left_indices.is_empty() {
                let indices = Array::from_slice(&left_indices, &[pad_amount]);
                signal.take_axis(&indices, 1)?
            } else {
                mlx_rs::ops::zeros::<f32>(&[batch_size, 0])?
            };

            // Right reflection
            let right_indices: Vec<i32> = ((length - pad_amount - 1)..(length - 1))
                .rev()
                .collect();
            let right_pad = if !right_indices.is_empty() {
                let indices = Array::from_slice(&right_indices, &[pad_amount]);
                signal.take_axis(&indices, 1)?
            } else {
                mlx_rs::ops::zeros::<f32>(&[batch_size, 0])?
            };

            mlx_rs::ops::concatenate_axis(&[&left_pad, signal, &right_pad], 1)
        }
        PadMode::Replicate => {
            use mlx_rs::ops::indexing::IndexOp;
            // Replicate edge values
            let left_val = signal.index((.., ..1));
            let right_val = signal.index((.., -1..));

            let left_pad = mlx_rs::ops::broadcast_to(&left_val, &[batch_size, pad_amount])?;
            let right_pad = mlx_rs::ops::broadcast_to(&right_val, &[batch_size, pad_amount])?;

            mlx_rs::ops::concatenate_axis(&[&left_pad, signal, &right_pad], 1)
        }
    }.map_err(VocoderError::from)
}

/// Compute magnitude spectrogram from complex STFT.
pub fn stft_magnitude(stft_matrix: &Array) -> Result<Array> {
    // |z| = sqrt(real² + imag²)
    Ok(stft_matrix.abs()?)
}

/// Compute power spectrogram from complex STFT.
pub fn stft_power(stft_matrix: &Array) -> Result<Array> {
    // |z|² = real² + imag²
    let mag = stft_matrix.abs()?;
    Ok(mag.multiply(&mag)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = hann_window(4).unwrap();
        window.eval().unwrap();
        assert_eq!(window.shape(), &[4]);

        // Hann window should be symmetric and start/end near 0
        // hann(4) = [0, 0.5, 1, 0.5] approximately
    }

    #[test]
    fn test_stft_config() {
        let config = StftConfig::default();
        assert_eq!(config.n_fft, 1024);
        assert_eq!(config.hop_length, 256);
    }
}
