//! Mel-spectrogram reconstruction losses.

use crate::audio::{MelConfig, PadMode, StftConfig, Window, mel_filterbank, stft};
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::{Array, Dtype, ops};

/// Plain L1 distance between two (log-)mel spectrograms: BigVGAN v1's
/// reconstruction loss, computed on the front end's output for the real and
/// the generated waveform.
pub fn mel_l1_loss(real_mel: &Array, fake_mel: &Array) -> Result<Array> {
    if real_mel.shape() != fake_mel.shape() {
        return Err(VocoderError::ShapeMismatch {
            expected: real_mel.shape().to_vec(),
            actual: fake_mel.shape().to_vec(),
        });
    }
    Ok(real_mel.subtract(fake_mel).abs_val().mean_all())
}

/// Settings of the multi-scale mel loss.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiScaleMelLossConfig {
    /// Sampling rate of the audio.
    pub sampling_rate: i32,
    /// Mel bins at each scale.
    pub n_mels: Vec<i32>,
    /// STFT window (and FFT) length at each scale; the hop is a quarter of it.
    pub window_lengths: Vec<i32>,
    /// Lowest mel edge (Hz) at each scale.
    pub fmin: Vec<f32>,
    /// Highest mel edge (Hz) at each scale; `None` is Nyquist.
    pub fmax: Vec<Option<f32>>,
    /// Floor applied to the mel magnitude before the log.
    pub clamp_eps: f32,
    /// Exponent applied to the magnitude before the log.
    pub pow: f32,
    /// Weight of the log-magnitude term.
    pub log_weight: f32,
}

impl MultiScaleMelLossConfig {
    /// BigVGAN v2's configuration: seven scales from 5 mel bins / 32-sample
    /// windows to 320 / 2048.
    pub fn bigvgan_v2(sampling_rate: i32) -> Self {
        Self {
            sampling_rate,
            n_mels: vec![5, 10, 20, 40, 80, 160, 320],
            window_lengths: vec![32, 64, 128, 256, 512, 1024, 2048],
            fmin: vec![0.0; 7],
            fmax: vec![None; 7],
            clamp_eps: 1e-5,
            pow: 1.0,
            log_weight: 1.0,
        }
    }
}

/// The multi-scale mel loss of BigVGAN v2 (after the Descript audio codec):
/// the sum over scales of the L1 distance between log10 mel spectrograms
/// computed directly from the two waveforms.
#[derive(Debug)]
pub struct MultiScaleMelLoss {
    config: MultiScaleMelLossConfig,
    scales: Vec<Scale>,
}

#[derive(Debug)]
struct Scale {
    stft: StftConfig,
    filterbank: Array,
}

impl MultiScaleMelLoss {
    /// Build the loss, precomputing every scale's filterbank.
    pub fn new(config: MultiScaleMelLossConfig) -> Result<Self> {
        let n = config.window_lengths.len();
        if n == 0 || config.n_mels.len() != n || config.fmin.len() != n || config.fmax.len() != n {
            return Err(VocoderError::Config(
                "multi-scale mel loss needs equal-length, non-empty per-scale lists".into(),
            ));
        }
        let scales = (0..n)
            .map(|i| {
                let w = config.window_lengths[i];
                let filterbank = mel_filterbank(&MelConfig {
                    sr: config.sampling_rate,
                    n_fft: w,
                    n_mels: config.n_mels[i],
                    fmin: config.fmin[i],
                    fmax: config.fmax[i],
                    ..MelConfig::default()
                })?;
                Ok(Scale {
                    stft: StftConfig {
                        n_fft: w,
                        hop_length: w / 4,
                        win_length: None,
                        center: true,
                        pad_mode: PadMode::Reflect,
                        window: Window::Hann,
                    },
                    filterbank,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { config, scales })
    }

    /// BigVGAN v2's loss for audio at `sampling_rate`.
    pub fn bigvgan_v2(sampling_rate: i32) -> Result<Self> {
        Self::new(MultiScaleMelLossConfig::bigvgan_v2(sampling_rate))
    }

    /// The loss between an estimate and a reference waveform, each
    /// `[batch, samples]` or `[batch, 1, samples]`. Differentiable in both.
    pub fn forward(&self, estimate: &Array, reference: &Array) -> Result<Array> {
        let estimate = flatten_channel(estimate)?;
        let reference = flatten_channel(reference)?;
        let ln10 = std::f32::consts::LN_10;
        let eps = Array::from_f32(self.config.clamp_eps);
        let to_log_mel = |wave: &Array, scale: &Scale| -> Result<Array> {
            let magnitude = stft(wave, &scale.stft)?.abs_val();
            let mel = scale.filterbank.matmul(&magnitude);
            let clamped = ops::maximum(&mel, &eps);
            Ok(clamped
                .log()
                .multiply(&Array::from_f32(self.config.pow / ln10)))
        };

        let mut total: Option<Array> = None;
        for scale in &self.scales {
            let diff = to_log_mel(&estimate, scale)?.subtract(&to_log_mel(&reference, scale)?);
            let term = diff
                .abs_val()
                .mean_all()
                .multiply(&Array::from_f32(self.config.log_weight));
            total = Some(match total {
                Some(t) => t.add(&term),
                None => term,
            });
        }
        Ok(total.expect("validated: at least one scale"))
    }
}

fn flatten_channel(wave: &Array) -> Result<Array> {
    match wave.ndim() {
        2 => Ok(wave.as_dtype(Dtype::Float32)),
        3 if wave.dim(1) == 1 => Ok(wave
            .reshape(&[wave.dim(0), wave.dim(2)])
            .as_dtype(Dtype::Float32)),
        _ => Err(VocoderError::ShapeMismatch {
            expected: vec![-1, 1, -1],
            actual: wave.shape().to_vec(),
        }),
    }
}
