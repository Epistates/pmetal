//! Multi-Band Discriminator (MBD; BigVGAN v2, after Defossez et al., 2022).
//!
//! Each sub-discriminator takes the complex STFT at one window length,
//! splits it into five frequency bands, runs a separate convolution stack per
//! band, and joins the bands for a final convolution. The input is first
//! DC-removed and peak-normalised so the discriminator judges spectral shape,
//! not level.

use super::{Discriminator, DiscriminatorOutput, conv_act, conv2d, flatten_logits};
use crate::audio::{PadMode, StftConfig, Window, hann_window, stft_parts};
use crate::error::{Result, VocoderError};
use crate::nn::{Conv2d, StateDict, join_key};
use pmetal_runtime::compat::{Array, Dtype, ops};
use pmetal_runtime::impl_module_params;

/// The five frequency bands as fractions of the spectrum.
const BANDS: [(f32, f32); 5] = [
    (0.0, 0.1),
    (0.1, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 1.0),
];

/// A bank of band discriminators, one per window length.
#[derive(Debug)]
pub struct MultiBandDiscriminator {
    /// One discriminator per window length.
    pub discriminators: Vec<BandDiscriminator>,
}

impl MultiBandDiscriminator {
    /// Discriminators at the given window lengths (the reference uses
    /// `[2048, 1024, 512]` with 32 channels).
    pub fn new(window_lengths: &[i32], channels: i32) -> Result<Self> {
        if window_lengths.is_empty() {
            return Err(VocoderError::Config(
                "the MBD needs at least one window length".into(),
            ));
        }
        Ok(Self {
            discriminators: window_lengths
                .iter()
                .map(|&w| BandDiscriminator::new(w, channels))
                .collect::<Result<_>>()?,
        })
    }

    /// Fold weight normalisation into the weights.
    pub fn fuse_weight_norm(&mut self) {
        self.discriminators
            .iter_mut()
            .for_each(BandDiscriminator::fuse);
    }
}

impl Discriminator for MultiBandDiscriminator {
    fn discriminate(&self, audio: &Array) -> Result<Vec<DiscriminatorOutput>> {
        self.discriminators
            .iter()
            .map(|d| d.forward(audio))
            .collect()
    }

    fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        for (i, d) in self.discriminators.iter_mut().enumerate() {
            d.load_state(sd, &join_key(prefix, &format!("discriminators.{i}")))?;
        }
        Ok(())
    }

    fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        for (i, d) in self.discriminators.iter().enumerate() {
            d.save_state(&join_key(prefix, &format!("discriminators.{i}")), out);
        }
    }
}

impl_module_params!(MultiBandDiscriminator; discriminators);

/// Discriminator at one window length.
#[derive(Debug)]
pub struct BandDiscriminator {
    stft: StftConfig,
    /// `[start, end)` frequency bins of each band.
    bands: Vec<(i32, i32)>,
    /// One convolution stack per band.
    pub band_convs: Vec<Vec<Conv2d>>,
    /// Output convolution over the joined bands.
    pub conv_post: Conv2d,
}

impl BandDiscriminator {
    /// A discriminator for window length `window_length` (hop is a quarter of
    /// it) with `channels` channels per band convolution.
    pub fn new(window_length: i32, channels: i32) -> Result<Self> {
        if window_length < 8 || channels < 1 {
            return Err(VocoderError::Config(format!(
                "invalid MBD settings (window {window_length}, channels {channels})"
            )));
        }
        let freqs = window_length / 2 + 1;
        let bands: Vec<(i32, i32)> = BANDS
            .iter()
            .map(|&(lo, hi)| ((lo * freqs as f32) as i32, (hi * freqs as f32) as i32))
            .collect();
        let stack = || -> Result<Vec<Conv2d>> {
            Ok(vec![
                conv2d(2, channels, (3, 9), (1, 1), (1, 4))?,
                conv2d(channels, channels, (3, 9), (1, 2), (1, 4))?,
                conv2d(channels, channels, (3, 9), (1, 2), (1, 4))?,
                conv2d(channels, channels, (3, 9), (1, 2), (1, 4))?,
                conv2d(channels, channels, (3, 3), (1, 1), (1, 1))?,
            ])
        };
        let band_convs = bands.iter().map(|_| stack()).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            stft: StftConfig {
                n_fft: window_length,
                hop_length: window_length / 4,
                win_length: None,
                center: true,
                pad_mode: PadMode::Reflect,
                window: Window::Hann,
            },
            bands,
            band_convs,
            conv_post: conv2d(channels, 1, (3, 3), (1, 1), (1, 1))?,
        })
    }

    /// Remove DC, peak-normalise to 0.8, and return `[batch, frames, freq, 2]`
    /// with the real and imaginary parts as channels.
    fn spectrogram(&self, audio: &Array) -> Result<Array> {
        let wave = audio.reshape(&[audio.dim(0), audio.dim(2)]);
        let centred = wave.subtract(&wave.mean_axis(1, true));
        let peak = centred.abs_val().max_axis(1, true);
        let normalised = centred
            .multiply(&Array::from_f32(0.8))
            .divide(&peak.add(&Array::from_f32(1e-9)));
        let (re, im) = stft_parts(&normalised, &self.stft)?;
        Ok(ops::stack_axis(&[re, im], 3))
    }

    /// Score `audio` (`[batch, 1, samples]`).
    pub fn forward(&self, audio: &Array) -> Result<DiscriminatorOutput> {
        if audio.ndim() != 3 || audio.dim(1) != 1 {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, 1, -1],
                actual: audio.shape().to_vec(),
            });
        }
        let spec = self.spectrogram(audio)?;
        let mut features = Vec::new();
        let mut joined = Vec::with_capacity(self.bands.len());
        for (&(lo, hi), stack) in self.bands.iter().zip(&self.band_convs) {
            let mut band = ops::slice_axis(&spec, 2, lo, hi);
            for (i, layer) in stack.iter().enumerate() {
                band = conv_act(layer, &band)?;
                // The first layer's output is not used for feature matching.
                if i > 0 {
                    features.push(band.clone());
                }
            }
            joined.push(band);
        }
        let refs: Vec<&Array> = joined.iter().collect();
        let x = self.conv_post.forward(&ops::concatenate_axis(&refs, 2))?;
        features.push(x.clone());
        Ok(DiscriminatorOutput {
            logits: flatten_logits(&x),
            features,
        })
    }

    fn fuse(&mut self) {
        self.band_convs.iter_mut().flatten().for_each(Conv2d::fuse);
        self.conv_post.fuse();
    }

    fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        for (b, stack) in self.band_convs.iter_mut().enumerate() {
            for (i, c) in stack.iter_mut().enumerate() {
                c.load_state(sd, &join_key(prefix, &format!("band_convs.{b}.{i}")))?;
            }
        }
        self.conv_post
            .load_state(sd, &join_key(prefix, "conv_post"))?;
        // The reference keeps the STFT window as a buffer. It is the periodic
        // Hann window by construction, so check rather than load it.
        let window_key = join_key(prefix, "spec_fn.window");
        if let Some(window) = sd.take_opt(&window_key) {
            let ours = hann_window(self.stft.n_fft, true)?;
            let same = window.shape() == ours.shape()
                && window
                    .as_dtype(Dtype::Float32)
                    .as_slice::<f32>()
                    .iter()
                    .zip(ours.as_slice::<f32>())
                    .all(|(a, b)| (a - b).abs() <= 1e-6);
            if !same {
                return Err(VocoderError::WeightLoad(format!(
                    "`{window_key}` is not the periodic Hann window of {} samples",
                    self.stft.n_fft
                )));
            }
        }
        Ok(())
    }

    fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        for (b, stack) in self.band_convs.iter().enumerate() {
            for (i, c) in stack.iter().enumerate() {
                c.save_state(&join_key(prefix, &format!("band_convs.{b}.{i}")), out);
            }
        }
        self.conv_post
            .save_state(&join_key(prefix, "conv_post"), out);
    }
}

impl_module_params!(BandDiscriminator; band_convs, conv_post);
