//! Band-limited up- and down-sampling for the anti-aliased activations.
//!
//! Signals are channels-last, `[batch, time, channels]`. Every channel shares
//! one filter, so the resamplers are depthwise convolutions.

use super::filter::kaiser_sinc_filter;
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::Array;
use pmetal_runtime::compat::ops::{self, PadMode};

/// Depthwise filter weight `[channels, taps, 1]` (MLX's grouped-conv layout),
/// the same taps for every channel.
fn depthwise_weight(taps: &[f32], channels: i32) -> Array {
    let mut data = Vec::with_capacity(taps.len() * channels as usize);
    for _ in 0..channels {
        data.extend_from_slice(taps);
    }
    Array::from_f32_slice(&data, &[channels, taps.len() as i32, 1])
}

fn check_dims(ratio: i32, kernel_size: i32, channels: i32) -> Result<()> {
    if ratio < 1 || kernel_size < 1 || channels < 1 {
        return Err(VocoderError::Config(format!(
            "resampler needs ratio, kernel size and channels >= 1 (got {ratio}, {kernel_size}, {channels})"
        )));
    }
    Ok(())
}

/// Replicate-pad the time axis of `[batch, time, channels]`.
fn pad_time(x: &Array, left: i32, right: i32) -> Array {
    ops::pad(
        x,
        &[(0, 0), (left, right), (0, 0)],
        Some(PadMode::Edge),
        None,
    )
}

/// The time range `[start, start + len)` of `x`, zero outside `[0, time)`.
fn window_time(x: &Array, start: i32, len: i32) -> Array {
    let have = x.dim(1);
    let left = (-start).max(0);
    let right = (start + len - have).max(0);
    let x = if left > 0 || right > 0 {
        ops::pad(
            x,
            &[(0, 0), (left, right), (0, 0)],
            Some(PadMode::Constant),
            None,
        )
    } else {
        x.clone()
    };
    let begin = start + left;
    ops::slice_axis(&x, 1, begin, begin + len)
}

/// Upsamples by an integer `ratio` with a Kaiser-windowed sinc interpolator.
///
/// Replicate-pads the edges, zero-stuffs, filters, and trims so the output has
/// exactly `ratio` times the input length. The interpolation gain of `ratio`
/// is folded into the taps.
#[derive(Debug)]
pub struct UpSample1d {
    ratio: i32,
    kernel_size: i32,
    channels: i32,
    pad: i32,
    pad_left: i32,
    taps: Vec<f32>,
    /// The ratio-scaled, time-reversed taps as a depthwise weight.
    weight: Array,
}

impl UpSample1d {
    /// A resampler for `channels` channels with the reference's filter design
    /// (cutoff `0.5 / ratio`, half-width `0.6 / ratio`).
    pub fn new(ratio: i32, kernel_size: i32, channels: i32) -> Result<Self> {
        check_dims(ratio, kernel_size, channels)?;
        let taps = kaiser_sinc_filter(
            0.5 / f64::from(ratio),
            0.6 / f64::from(ratio),
            kernel_size as usize,
        )?;
        Self::with_taps(ratio, channels, taps)
    }

    /// A resampler with explicit filter taps.
    pub fn with_taps(ratio: i32, channels: i32, taps: Vec<f32>) -> Result<Self> {
        let kernel_size = taps.len() as i32;
        check_dims(ratio, kernel_size, channels)?;
        let pad = kernel_size / ratio - 1;
        let pad_left = pad * ratio + (kernel_size - ratio) / 2;
        let weight = Self::build_weight(&taps, ratio, channels);
        Ok(Self {
            ratio,
            kernel_size,
            channels,
            pad,
            pad_left,
            taps,
            weight,
        })
    }

    fn build_weight(taps: &[f32], ratio: i32, channels: i32) -> Array {
        let flipped: Vec<f32> = taps.iter().rev().map(|t| t * ratio as f32).collect();
        depthwise_weight(&flipped, channels)
    }

    /// The interpolation ratio.
    pub fn ratio(&self) -> i32 {
        self.ratio
    }

    /// `(ratio, kernel_size, replicate_pad, trim_left)`.
    pub(crate) fn geometry(&self) -> (i32, i32, i32, i32) {
        (self.ratio, self.kernel_size, self.pad, self.pad_left)
    }

    /// How many input samples on each side an output sample can depend on.
    pub fn input_reach(&self) -> i32 {
        // An output sample reads `ceil(kernel / ratio)` padded input samples;
        // one more covers the half-sample offset of the interpolation grid.
        (self.kernel_size + 2 * self.ratio - 1) / (2 * self.ratio) + 1
    }

    /// The filter taps.
    pub fn taps(&self) -> &[f32] {
        &self.taps
    }

    /// Replace the filter taps (a loaded buffer); the length must not change.
    pub fn set_taps(&mut self, taps: Vec<f32>) -> Result<()> {
        if taps.len() != self.kernel_size as usize {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![self.kernel_size],
                actual: vec![taps.len() as i32],
            });
        }
        self.weight = Self::build_weight(&taps, self.ratio, self.channels);
        self.taps = taps;
        Ok(())
    }

    /// `[batch, time, channels]` -> `[batch, time * ratio, channels]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let (batch, time, channels) = (x.dim(0), x.dim(1), x.dim(2));
        let r = self.ratio;
        let padded = pad_time(x, self.pad, self.pad);
        let padded_len = time + 2 * self.pad;

        // Zero-stuff: [B, T, C] -> [B, T * r, C] with r - 1 zeros after each
        // sample. (Ratio 1 still filters, as in the reference.)
        let stuffed = if r == 1 {
            padded
        } else {
            let zeros = Array::zeros(&[batch, padded_len, r - 1, channels], x.dtype());
            ops::concatenate_axis(&[&padded.expand_dims(2), &zeros], 2).reshape(&[
                batch,
                padded_len * r,
                channels,
            ])
        };

        // out[n] = sum_i f[i] z[n + pad_left - i]; as a correlation with the
        // reversed filter the window starts at pad_left - (k - 1).
        let start = self.pad_left - (self.kernel_size - 1);
        let window = window_time(&stuffed, start, time * r + self.kernel_size - 1);
        let weight = self.weight.as_dtype(x.dtype());
        Ok(window.conv1d(&weight, 1, 0, 1, self.channels))
    }
}

/// Downsamples by an integer `ratio` with a Kaiser-windowed sinc low-pass.
#[derive(Debug)]
pub struct DownSample1d {
    ratio: i32,
    kernel_size: i32,
    channels: i32,
    pad_left: i32,
    pad_right: i32,
    taps: Vec<f32>,
    weight: Array,
}

impl DownSample1d {
    /// A resampler for `channels` channels with the reference's filter design
    /// (cutoff `0.5 / ratio`, half-width `0.6 / ratio`).
    pub fn new(ratio: i32, kernel_size: i32, channels: i32) -> Result<Self> {
        check_dims(ratio, kernel_size, channels)?;
        let taps = kaiser_sinc_filter(
            0.5 / f64::from(ratio),
            0.6 / f64::from(ratio),
            kernel_size as usize,
        )?;
        Self::with_taps(ratio, channels, taps)
    }

    /// A resampler with explicit filter taps.
    pub fn with_taps(ratio: i32, channels: i32, taps: Vec<f32>) -> Result<Self> {
        let kernel_size = taps.len() as i32;
        check_dims(ratio, kernel_size, channels)?;
        let even = i32::from(kernel_size % 2 == 0);
        Ok(Self {
            ratio,
            kernel_size,
            channels,
            pad_left: kernel_size / 2 - even,
            pad_right: kernel_size / 2,
            weight: depthwise_weight(&taps, channels),
            taps,
        })
    }

    /// `(ratio, kernel_size, pad_left, pad_right)`.
    pub(crate) fn geometry(&self) -> (i32, i32, i32, i32) {
        (self.ratio, self.kernel_size, self.pad_left, self.pad_right)
    }

    /// How many input samples on each side an output sample can depend on,
    /// when the input runs at `rate_factor` times the rate the caller counts in
    /// (an activation upsamples by `rate_factor`, so its decimator's input is
    /// `rate_factor` times denser than the activation's).
    pub fn input_reach(&self, rate_factor: i32) -> i32 {
        let widest = self.pad_left.max(self.pad_right);
        (widest + rate_factor - 1) / rate_factor
    }

    /// The filter taps.
    pub fn taps(&self) -> &[f32] {
        &self.taps
    }

    /// Replace the filter taps (a loaded buffer); the length must not change.
    pub fn set_taps(&mut self, taps: Vec<f32>) -> Result<()> {
        if taps.len() != self.kernel_size as usize {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![self.kernel_size],
                actual: vec![taps.len() as i32],
            });
        }
        self.weight = depthwise_weight(&taps, self.channels);
        self.taps = taps;
        Ok(())
    }

    /// `[batch, time, channels]` -> `[batch, ceil(time / ratio), channels]` (for
    /// the reference's kernel sizes, exactly `time / ratio`).
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let padded = pad_time(x, self.pad_left, self.pad_right);
        let weight = self.weight.as_dtype(x.dtype());
        Ok(padded.conv1d(&weight, self.ratio, 0, 1, self.channels))
    }
}
