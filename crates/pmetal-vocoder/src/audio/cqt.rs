//! Constant-Q transform by recursive downsampling (Schorkhuber and Klapuri,
//! "Constant-Q transform toolbox for music processing", 2010).
//!
//! Instead of one enormous kernel bank covering every octave, a small bank
//! covering only the top octave is convolved with the signal, which is then
//! low-pass filtered and halved in rate; the same bank applied to the
//! halved signal is the next octave down. This follows the construction of
//! `nnAudio`'s `CQT2010v2` (constant zero padding, complex output), which the
//! BigVGAN-v2 CQT discriminator was trained against, so its spectrograms match
//! to floating-point rounding.

use crate::error::{Result, VocoderError};
use crate::nn::filter::cqt_lowpass_filter;
use crate::nn::{StateDict, join_key};
use pmetal_runtime::compat::{Array, Dtype, ops};

/// Settings of a [`Cqt`]. Defaults follow `nnAudio` (C0 as the lowest bin).
#[derive(Debug, Clone, PartialEq)]
pub struct CqtConfig {
    /// Sampling rate of the input signal.
    pub sample_rate: i32,
    /// Hop between frames, in samples of the input signal.
    pub hop_length: i32,
    /// Frequency of the lowest bin in Hz.
    pub fmin: f64,
    /// Total number of bins.
    pub n_bins: i32,
    /// Bins per octave.
    pub bins_per_octave: i32,
    /// Scales the kernels' Q factor (1 is the standard constant-Q).
    pub filter_scale: f64,
    /// Whether to downsample up front when the highest bin allows it.
    pub early_downsample: bool,
}

impl CqtConfig {
    /// `n_bins` bins of `bins_per_octave` per octave from C0, for a signal at
    /// `sample_rate` Hz hopping by `hop_length` samples.
    pub fn new(sample_rate: i32, hop_length: i32, n_bins: i32, bins_per_octave: i32) -> Self {
        Self {
            sample_rate,
            hop_length,
            fmin: 32.70,
            n_bins,
            bins_per_octave,
            filter_scale: 1.0,
            early_downsample: true,
        }
    }
}

/// The lowpass taps used between octaves and ahead of an early downsample.
const FILTER_TAPS: usize = 256;

/// A constant-Q transform producing complex spectrograms.
#[derive(Debug)]
pub struct Cqt {
    config: CqtConfig,
    n_octaves: i32,
    n_filters: i32,
    /// Window of the kernels, a power of two.
    fft_len: i32,
    /// Kernels as one convolution weight `[2 * n_filters, fft_len, 1]`: the
    /// real parts, then the negated imaginary parts.
    kernels: Array,
    /// Per-bin `sqrt(length)` normalisation, `[n_bins]`, times the early
    /// downsampling factor.
    scale: Array,
    /// Octave-to-octave lowpass, `[1, taps, 1]` (MLX weight layout).
    lowpass: Array,
    /// Early downsampling filter, if any.
    early: Option<(Array, i32)>,
    /// Hop after early downsampling.
    hop: i32,
    /// The kernels' window lengths per bin (before the sqrt), for diagnostics.
    lengths: Vec<f32>,
}

fn next_pow2_exponent(a: f64) -> i32 {
    a.log2().ceil() as i32
}

impl Cqt {
    /// Build the kernel bank and filters.
    pub fn new(config: CqtConfig) -> Result<Self> {
        let bad = |msg: String| Err(VocoderError::Audio(format!("CQT: {msg}")));
        if config.sample_rate < 1
            || config.hop_length < 1
            || config.n_bins < 1
            || config.bins_per_octave < 1
        {
            return bad("sample rate, hop, bins and bins per octave must be positive".into());
        }
        let bpo = config.bins_per_octave;
        let q = config.filter_scale / (2f64.powf(1.0 / f64::from(bpo)) - 1.0);
        let n_filters = bpo.min(config.n_bins);
        let n_octaves = (f64::from(config.n_bins) / f64::from(bpo)).ceil() as i32;

        // Lowest and highest frequency of the top octave's kernels.
        let mut fmin_t = config.fmin * 2f64.powi(n_octaves - 1);
        let remainder = config.n_bins % bpo;
        let fmax_t = if remainder == 0 {
            fmin_t * 2f64.powf(f64::from(bpo - 1) / f64::from(bpo))
        } else {
            fmin_t * 2f64.powf(f64::from(remainder - 1) / f64::from(bpo))
        };
        fmin_t = fmax_t / 2f64.powf(1.0 - 1.0 / f64::from(bpo));
        if fmax_t > f64::from(config.sample_rate) / 2.0 {
            return bad(format!(
                "the top bin ({fmax_t} Hz) is above the Nyquist frequency; use fewer bins"
            ));
        }

        // Early downsampling: drop the rate while the top bin still fits.
        let mut sr = f64::from(config.sample_rate);
        let mut hop = config.hop_length;
        let mut factor = 1;
        let mut early = None;
        if config.early_downsample {
            let window_bandwidth = 1.5; // Hann
            let cutoff = fmax_t * (1.0 + 0.5 * window_bandwidth / q);
            let nyquist = f64::from(config.sample_rate / 2);
            let count1 = (((0.85 * nyquist / cutoff).log2().ceil() as i32 - 1) - 1).max(0);
            let count2 = (next_pow2_exponent(f64::from(config.hop_length)) - n_octaves + 1).max(0);
            let count = count1.min(count2);
            if count > 0 {
                factor = 1 << count;
                hop /= factor;
                sr /= f64::from(factor);
                let taps = cqt_lowpass_filter(1.0 / f64::from(factor), FILTER_TAPS, 0.03)?;
                early = Some((
                    Array::from_f32_slice(&taps, &[1, FILTER_TAPS as i32, 1]),
                    factor,
                ));
            }
        }
        if hop < 1 {
            return bad(format!(
                "the hop ({}) is too small for this many octaves",
                config.hop_length
            ));
        }

        // Kernels of the top octave, in the time domain.
        let freqs_top: Vec<f64> = (0..n_filters)
            .map(|k| fmin_t * 2f64.powf(f64::from(k) / f64::from(bpo)))
            .collect();
        let lengths_top: Vec<f64> = freqs_top.iter().map(|f| (q * sr / f).ceil()).collect();
        let max_len = lengths_top.iter().copied().fold(0.0, f64::max) as i32;
        let fft_len = 1i32 << next_pow2_exponent(f64::from(max_len));
        let width = fft_len as usize;
        let mut real = vec![0.0f32; n_filters as usize * width];
        let mut imag = vec![0.0f32; n_filters as usize * width];
        for (k, (&freq, &len)) in freqs_top.iter().zip(&lengths_top).enumerate() {
            let l = len as i64;
            let start = if l % 2 == 1 {
                (f64::from(fft_len) / 2.0 - len / 2.0).ceil() as i64 - 1
            } else {
                (f64::from(fft_len) / 2.0 - len / 2.0).ceil() as i64
            };
            // sig[n] = hann[n] * exp(2 pi i f (n - l/2) / sr) / l  for n in 0..l
            let first = (-len / 2.0).floor() as i64;
            let taps: Vec<(f64, f64)> = (0..l)
                .map(|n| {
                    let hann = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / len).cos();
                    let phase = (first + n) as f64 * 2.0 * std::f64::consts::PI * freq / sr;
                    (hann * phase.cos() / len, hann * phase.sin() / len)
                })
                .collect();
            let l1: f64 = taps.iter().map(|(re, im)| re.hypot(*im)).sum();
            for (n, (re, im)) in taps.iter().enumerate() {
                let at = k * width + (start as usize + n);
                real[at] = (re / l1) as f32;
                imag[at] = (im / l1) as f32;
            }
        }
        // One convolution gives [real, -imag]: the transform correlates with
        // the conjugate of the kernel.
        let mut weights = real;
        weights.extend(imag.iter().map(|v| -v));
        let kernels = Array::from_f32_slice(&weights, &[2 * n_filters, fft_len, 1]);

        // Normalisation as librosa: sqrt of each bin's window length, with the
        // lengths computed at the (possibly downsampled) rate for all bins.
        let lengths: Vec<f32> = (0..config.n_bins)
            .map(|b| {
                (q * sr / (config.fmin * 2f64.powf(f64::from(b) / f64::from(bpo)))).ceil() as f32
            })
            .collect();
        let scale: Vec<f32> = lengths.iter().map(|l| l.sqrt() * factor as f32).collect();

        let lowpass = Array::from_f32_slice(
            &cqt_lowpass_filter(0.5, FILTER_TAPS, 0.001)?,
            &[1, FILTER_TAPS as i32, 1],
        );
        Ok(Self {
            n_octaves,
            n_filters,
            fft_len,
            kernels,
            scale: Array::from_f32_slice(&scale, &[config.n_bins]),
            lowpass,
            early,
            hop,
            lengths,
            config,
        })
    }

    /// The transform's settings.
    pub fn config(&self) -> &CqtConfig {
        &self.config
    }

    /// The kernels' window lengths in samples, per bin.
    pub fn lengths(&self) -> &[f32] {
        &self.lengths
    }

    /// Check a checkpoint's derived buffers (`prefix.lenghts`,
    /// `prefix.cqt_kernels_real`, `.cqt_kernels_imag`, `.lowpass_filter`,
    /// `.early_downsample_filter`) against the ones this transform computed.
    ///
    /// They are a function of the settings, so loading them is pointless, but
    /// comparing them catches a checkpoint made for a different sampling
    /// rate, hop or bin layout, which would otherwise load into the
    /// convolutions without complaint and produce garbage. Absent buffers are
    /// skipped.
    pub fn check_buffers(&self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        let weights = self.kernels.as_slice::<f32>();
        let half = weights.len() / 2;
        let (real, neg_imag) = weights.split_at(half);
        let imag: Vec<f32> = neg_imag.iter().map(|v| -v).collect();
        let lowpass = self.lowpass.as_slice::<f32>().to_vec();
        let early = self
            .early
            .as_ref()
            .map(|(f, _)| f.as_slice::<f32>().to_vec());
        let ours: [(&str, &[f32], bool); 5] = [
            ("lenghts", &self.lengths, true),
            ("cqt_kernels_real", real, true),
            ("cqt_kernels_imag", &imag, true),
            ("lowpass_filter", &lowpass, true),
            (
                "early_downsample_filter",
                early.as_deref().unwrap_or(&[]),
                early.is_some(),
            ),
        ];
        for (name, expected, present) in ours {
            let key = join_key(prefix, name);
            let Some(buffer) = sd.take_opt(&key) else {
                continue;
            };
            let got = buffer.as_dtype(Dtype::Float32);
            let got = got.as_slice::<f32>();
            let scale = expected
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()))
                .max(1e-6);
            let same = present
                && got.len() == expected.len()
                && got
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() <= 1e-4 * scale);
            if !same {
                return Err(VocoderError::WeightLoad(format!(
                    "`{key}` does not match the CQT this configuration builds; the checkpoint was made for different settings"
                )));
            }
        }
        Ok(())
    }

    /// Downsample by `stride` with an FIR (`[1, taps, 1]`), zero padded to
    /// the filter's centre.
    fn decimate(&self, x: &Array, filter: &Array, stride: i32) -> Array {
        let pad = (filter.dim(1) - 1) / 2;
        x.conv1d(filter, stride, pad, 1, 1)
    }

    /// One octave: zero-pad by half a kernel and correlate with the bank,
    /// giving `[batch, frames, 2 * n_filters]` (real parts then imaginary).
    fn octave(&self, x: &Array, hop: i32) -> Array {
        let pad = self.fft_len / 2;
        let x = ops::pad(x, &[(0, 0), (pad, pad), (0, 0)], None, None);
        x.conv1d(&self.kernels, hop, 0, 1, 1)
    }

    /// The complex CQT of `signal` (`[batch, samples]` or `[samples]`) as
    /// `[batch, frames, n_bins, 2]` with the real and imaginary parts last.
    ///
    /// Frames run along the second axis and bins, lowest first, along the
    /// third: the layout a 2-D convolution wants. The hop halves with every
    /// octave, so it must stay an integer down to the lowest one (a hop
    /// divisible by `2^(octaves - 1)`, after early downsampling); otherwise the
    /// octaves would produce different frame counts, which is an error rather
    /// than a silently ragged result.
    pub fn forward(&self, signal: &Array) -> Result<Array> {
        let wave = if signal.ndim() == 1 {
            signal.reshape(&[1, -1])
        } else {
            signal.clone()
        };
        if wave.ndim() != 2 {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, -1],
                actual: signal.shape().to_vec(),
            });
        }
        let batch = wave.dim(0);
        let mut x = wave
            .as_dtype(Dtype::Float32)
            .reshape(&[batch, wave.dim(1), 1]);
        if let Some((filter, factor)) = &self.early {
            x = self.decimate(&x, filter, *factor);
        }

        let mut hop = self.hop;
        let to_bins = |o: Array| -> Array {
            // [B, frames, 2 * nf] -> [B, frames, nf, 2]
            let frames = o.dim(1);
            o.reshape(&[batch, frames, 2, self.n_filters])
                .transpose_axes(&[0, 1, 3, 2])
        };
        let mut octaves = vec![to_bins(self.octave(&x, hop))];
        for _ in 1..self.n_octaves {
            hop /= 2;
            if hop < 1 {
                return Err(VocoderError::Audio(
                    "CQT: the hop runs out before the lowest octave".into(),
                ));
            }
            x = self.decimate(&x, &self.lowpass, 2);
            octaves.push(to_bins(self.octave(&x, hop)));
        }
        let frames = octaves[0].dim(1);
        if octaves.iter().any(|o| o.dim(1) != frames) {
            return Err(VocoderError::Audio(format!(
                "CQT: octaves produce different frame counts; the hop ({}) must be divisible by 2^(octaves - 1) = {}",
                self.config.hop_length,
                1 << (self.n_octaves - 1)
            )));
        }
        // Lowest octave first, then keep the top `n_bins` bins.
        octaves.reverse();
        let refs: Vec<&Array> = octaves.iter().collect();
        let all = ops::concatenate_axis(&refs, 2);
        let total = all.dim(2);
        let kept = ops::slice_axis(&all, 2, total - self.config.n_bins, total);
        Ok(kept.multiply(&self.scale.reshape(&[1, 1, self.config.n_bins, 1])))
    }
}
