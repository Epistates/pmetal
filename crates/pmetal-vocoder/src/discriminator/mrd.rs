//! Multi-Resolution Discriminator (MRD, UnivNet; Jang et al., 2021).
//!
//! Each sub-discriminator looks at the STFT magnitude of the waveform at one
//! resolution and convolves over the (frequency, time) plane. Using several
//! `(n_fft, hop, win)` triples lets the bank catch artefacts at different
//! time-frequency trade-offs.

use super::{Discriminator, DiscriminatorOutput, conv_act, conv2d, flatten_logits, scaled};
use crate::audio::{PadMode, StftConfig, Window, pad_signal, stft};
use crate::error::{Result, VocoderError};
use crate::nn::{Conv2d, StateDict, join_key};
use pmetal_runtime::compat::Array;
use pmetal_runtime::impl_module_params;

/// The reference resolutions `(n_fft, hop, win)`.
const DEFAULT_RESOLUTIONS: [[i32; 3]; 3] = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]];

/// A bank of resolution discriminators.
#[derive(Debug)]
pub struct MultiResolutionDiscriminator {
    /// One discriminator per resolution.
    pub discriminators: Vec<ResolutionDiscriminator>,
}

impl MultiResolutionDiscriminator {
    /// The reference's three resolutions at full width.
    pub fn new() -> Result<Self> {
        Self::with_resolutions(&DEFAULT_RESOLUTIONS, 1.0)
    }

    /// Custom `[n_fft, hop, win]` resolutions; `channel_mult` scales the width.
    pub fn with_resolutions(resolutions: &[[i32; 3]], channel_mult: f32) -> Result<Self> {
        if resolutions.is_empty() {
            return Err(VocoderError::Config(
                "the MRD needs at least one resolution".into(),
            ));
        }
        Ok(Self {
            discriminators: resolutions
                .iter()
                .map(|&[n_fft, hop, win]| {
                    ResolutionDiscriminator::new(n_fft, hop, win, channel_mult)
                })
                .collect::<Result<_>>()?,
        })
    }

    /// Fold weight normalisation into the weights.
    pub fn fuse_weight_norm(&mut self) {
        self.discriminators
            .iter_mut()
            .for_each(ResolutionDiscriminator::fuse);
    }
}

impl Discriminator for MultiResolutionDiscriminator {
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

impl_module_params!(MultiResolutionDiscriminator; discriminators);

/// Discriminator for one STFT resolution.
#[derive(Debug)]
pub struct ResolutionDiscriminator {
    /// STFT settings: a rectangular window, no centring.
    pub stft_config: StftConfig,
    /// Reflect padding applied to the waveform before the STFT.
    pad: i32,
    /// Five convolutions over the magnitude spectrogram.
    pub convs: Vec<Conv2d>,
    /// Output convolution to a single channel.
    pub conv_post: Conv2d,
}

impl ResolutionDiscriminator {
    /// A discriminator for the STFT `(n_fft, hop_length, win_length)`.
    pub fn new(n_fft: i32, hop_length: i32, win_length: i32, channel_mult: f32) -> Result<Self> {
        if n_fft < 2 || hop_length < 1 || win_length < 1 || win_length > n_fft || hop_length > n_fft
        {
            return Err(VocoderError::Config(format!(
                "invalid MRD resolution (n_fft {n_fft}, hop {hop_length}, win {win_length})"
            )));
        }
        let ch = scaled(32, channel_mult);
        let wide = (3, 9);
        let pad = (1, 4);
        let convs = vec![
            conv2d(1, ch, wide, (1, 1), pad)?,
            conv2d(ch, ch, wide, (1, 2), pad)?,
            conv2d(ch, ch, wide, (1, 2), pad)?,
            conv2d(ch, ch, wide, (1, 2), pad)?,
            conv2d(ch, ch, (3, 3), (1, 1), (1, 1))?,
        ];
        Ok(Self {
            stft_config: StftConfig {
                n_fft,
                hop_length,
                win_length: Some(win_length),
                center: false,
                pad_mode: PadMode::Reflect,
                // The reference passes no window to `torch.stft`, which means
                // a rectangular one.
                window: Window::Rectangular,
            },
            pad: (n_fft - hop_length) / 2,
            convs,
            conv_post: conv2d(ch, 1, (3, 3), (1, 1), (1, 1))?,
        })
    }

    /// The STFT magnitude `[batch, freq, frames, 1]` of `audio` (`[batch, 1, samples]`).
    fn spectrogram(&self, audio: &Array) -> Result<Array> {
        let wave = audio.reshape(&[audio.dim(0), audio.dim(2)]);
        let padded = pad_signal(&wave, self.pad, PadMode::Reflect)?;
        // `abs` of a complex value has a zero subgradient at the origin, so
        // digital silence does not produce NaN gradients.
        let magnitude = stft(&padded, &self.stft_config)?.abs_val();
        Ok(magnitude.expand_dims(3))
    }

    /// Score `audio` (`[batch, 1, samples]`).
    pub fn forward(&self, audio: &Array) -> Result<DiscriminatorOutput> {
        if audio.ndim() != 3 || audio.dim(1) != 1 {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, 1, -1],
                actual: audio.shape().to_vec(),
            });
        }
        let mut x = self.spectrogram(audio)?;
        let mut features = Vec::with_capacity(self.convs.len() + 1);
        for conv in &self.convs {
            x = conv_act(conv, &x)?;
            features.push(x.clone());
        }
        let x = self.conv_post.forward(&x)?;
        features.push(x.clone());
        Ok(DiscriminatorOutput {
            logits: flatten_logits(&x),
            features,
        })
    }

    fn fuse(&mut self) {
        self.convs.iter_mut().for_each(Conv2d::fuse);
        self.conv_post.fuse();
    }

    fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        for (i, c) in self.convs.iter_mut().enumerate() {
            c.load_state(sd, &join_key(prefix, &format!("convs.{i}")))?;
        }
        self.conv_post
            .load_state(sd, &join_key(prefix, "conv_post"))
    }

    fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        for (i, c) in self.convs.iter().enumerate() {
            c.save_state(&join_key(prefix, &format!("convs.{i}")), out);
        }
        self.conv_post
            .save_state(&join_key(prefix, "conv_post"), out);
    }
}

impl_module_params!(ResolutionDiscriminator; convs, conv_post);
