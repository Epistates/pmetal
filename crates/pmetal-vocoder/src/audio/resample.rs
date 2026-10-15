//! Band-limited sample-rate conversion.
//!
//! Windowed-sinc interpolation (Smith, "Digital Audio Resampling Home Page")
//! with a Hann window, computed the way `torchaudio.functional.resample`
//! does, so converted audio matches it sample for sample. The conversion
//! `orig -> new` reduces to `orig / g -> new / g` with `g = gcd`; each output
//! phase then has its own filter, and the whole bank is applied as one strided
//! convolution.

use crate::error::{Result, VocoderError};
use crate::nn::StateDict;
use pmetal_runtime::compat::{Array, Dtype, ops};

/// Default half-width of the interpolation kernel in zero crossings.
pub const DEFAULT_LOWPASS_FILTER_WIDTH: i32 = 6;
/// Default fraction of the Nyquist band that is kept.
pub const DEFAULT_ROLLOFF: f64 = 0.99;

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A sample-rate converter between two fixed rates.
#[derive(Debug)]
pub struct Resampler {
    /// Input rate over the gcd of both rates.
    orig: i32,
    /// Output rate over the gcd of both rates.
    new: i32,
    /// Zero padding on each side of the input (the kernel's half-width in samples).
    width: i32,
    /// Filter bank `[new, 2 * width + orig, 1]`: one filter per output phase.
    kernel: Array,
}

impl Resampler {
    /// A converter from `orig_freq` to `new_freq` Hz with torchaudio's defaults
    /// (6 zero crossings, 0.99 rolloff, Hann window).
    pub fn new(orig_freq: i32, new_freq: i32) -> Result<Self> {
        Self::with_options(
            orig_freq,
            new_freq,
            DEFAULT_LOWPASS_FILTER_WIDTH,
            DEFAULT_ROLLOFF,
        )
    }

    /// A converter with an explicit kernel width and rolloff.
    pub fn with_options(
        orig_freq: i32,
        new_freq: i32,
        lowpass_filter_width: i32,
        rolloff: f64,
    ) -> Result<Self> {
        if orig_freq < 1
            || new_freq < 1
            || lowpass_filter_width < 1
            || !(rolloff > 0.0 && rolloff <= 1.0)
        {
            return Err(VocoderError::Audio(format!(
                "invalid resampler settings ({orig_freq} -> {new_freq} Hz, width {lowpass_filter_width}, rolloff {rolloff})"
            )));
        }
        let g = gcd(orig_freq, new_freq);
        let (orig, new) = (orig_freq / g, new_freq / g);

        // Anti-aliasing cutoff, relative to the lower of the two rates.
        let base_freq = f64::from(orig.min(new)) * rolloff;
        let width = (f64::from(lowpass_filter_width) * f64::from(orig) / base_freq).ceil() as i32;
        let taps = (2 * width + orig) as usize;
        let lw = f64::from(lowpass_filter_width);
        let scale = base_freq / f64::from(orig);

        let mut kernel = Vec::with_capacity(new as usize * taps);
        for phase in 0..new {
            // The output phase offset is rounded to float32 first, as torch
            // computes `arange(0, -new, -1) / new` in its default dtype.
            let offset = f64::from(-(phase as f32) / new as f32);
            for k in 0..taps {
                let idx = (k as f64 - f64::from(width)) / f64::from(orig);
                let t = ((offset + idx) * base_freq).clamp(-lw, lw);
                let window = (t * std::f64::consts::PI / lw / 2.0).cos().powi(2);
                let t = t * std::f64::consts::PI;
                let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                kernel.push((sinc * window * scale) as f32);
            }
        }
        Ok(Self {
            orig,
            new,
            width,
            kernel: Array::from_f32_slice(&kernel, &[new, taps as i32, 1]),
        })
    }

    /// Output length for `samples` input samples, `ceil(samples * new / orig)`.
    pub fn output_len(&self, samples: i32) -> i32 {
        (i64::from(samples) * i64::from(self.new) + i64::from(self.orig) - 1)
            .div_euclid(i64::from(self.orig)) as i32
    }

    /// The filter bank as float32 values, row-major `[new, 2 * width + orig]`.
    pub fn kernel_values(&self) -> Vec<f32> {
        self.kernel.as_slice::<f32>().to_vec()
    }

    /// Check a checkpoint's `kernel` buffer against the filter bank this
    /// converter computed (taken from `sd` under `key` if present). A mismatch
    /// means the checkpoint was made for different sampling rates.
    pub fn check_buffer(&self, sd: &mut StateDict, key: &str) -> Result<()> {
        let Some(buffer) = sd.take_opt(key) else {
            return Ok(());
        };
        let got = buffer.as_dtype(Dtype::Float32);
        let got = got.as_slice::<f32>();
        let ours = self.kernel.as_slice::<f32>();
        let scale = ours.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        if got.len() != ours.len()
            || got
                .iter()
                .zip(ours)
                .any(|(a, b)| (a - b).abs() > 1e-4 * scale)
        {
            return Err(VocoderError::WeightLoad(format!(
                "`{key}` does not match the resampler this configuration builds; the checkpoint was made for different sampling rates"
            )));
        }
        Ok(())
    }

    /// Convert `[batch, samples]` (or `[samples]`) to `[batch, output_len]`.
    pub fn forward(&self, wave: &Array) -> Result<Array> {
        let unbatched = wave.ndim() == 1;
        let wave = if unbatched {
            wave.reshape(&[1, -1])
        } else {
            wave.clone()
        };
        if wave.ndim() != 2 {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, -1],
                actual: wave.shape().to_vec(),
            });
        }
        let (batch, samples) = (wave.dim(0), wave.dim(1));
        let x = wave.as_dtype(Dtype::Float32).reshape(&[batch, samples, 1]);
        let x = ops::pad(
            &x,
            &[(0, 0), (self.width, self.width + self.orig), (0, 0)],
            None,
            None,
        );
        // [B, T', 1] * [new, K, 1] at stride `orig` -> [B, frames, new]: the
        // phases of consecutive frames interleave into the output sequence.
        let y = x.conv1d(&self.kernel, self.orig, 0, 1, 1);
        let y = y.reshape(&[batch, -1]);
        let y = ops::slice_axis(&y, 1, 0, self.output_len(samples));
        Ok(if unbatched { y.squeeze_axes(&[0]) } else { y })
    }
}
