//! Multi-Period Discriminator (MPD, HiFi-GAN; Kong et al., 2020).
//!
//! A period-`p` discriminator folds the waveform into a `[time / p, p]` grid so
//! that samples `p` apart land in the same column, then convolves along time
//! only (kernels of shape `(k, 1)`): each column is one phase of a
//! `p`-periodic signal.

use super::{Discriminator, DiscriminatorOutput, conv_act, conv2d, flatten_logits, scaled};
use crate::error::{Result, VocoderError};
use crate::nn::{Conv2d, StateDict, join_key};
use pmetal_runtime::compat::Array;
use pmetal_runtime::compat::ops::{self, PadMode};
use pmetal_runtime::impl_module_params;

/// Periods of the reference model.
const DEFAULT_PERIODS: [i32; 5] = [2, 3, 5, 7, 11];

/// A bank of period discriminators.
#[derive(Debug)]
pub struct MultiPeriodDiscriminator {
    /// One discriminator per period.
    pub discriminators: Vec<PeriodDiscriminator>,
}

impl MultiPeriodDiscriminator {
    /// The reference's periods `[2, 3, 5, 7, 11]` at full width.
    pub fn new() -> Result<Self> {
        Self::with_periods(&DEFAULT_PERIODS, 1.0)
    }

    /// Custom periods; `channel_mult` scales every layer's width.
    pub fn with_periods(periods: &[i32], channel_mult: f32) -> Result<Self> {
        if periods.is_empty() {
            return Err(VocoderError::Config(
                "the MPD needs at least one period".into(),
            ));
        }
        Ok(Self {
            discriminators: periods
                .iter()
                .map(|&p| PeriodDiscriminator::new(p, channel_mult))
                .collect::<Result<_>>()?,
        })
    }

    /// Fold weight normalisation into the weights.
    pub fn fuse_weight_norm(&mut self) {
        self.discriminators
            .iter_mut()
            .for_each(PeriodDiscriminator::fuse);
    }
}

impl Discriminator for MultiPeriodDiscriminator {
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

impl_module_params!(MultiPeriodDiscriminator; discriminators);

/// Discriminator for one period.
#[derive(Debug)]
pub struct PeriodDiscriminator {
    /// The period the waveform is folded by.
    pub period: i32,
    /// Four strided convolutions and one stride-1 convolution.
    pub convs: Vec<Conv2d>,
    /// Output convolution to a single channel.
    pub conv_post: Conv2d,
}

impl PeriodDiscriminator {
    /// A discriminator for `period` at `channel_mult` times the reference width
    /// (1 -> 32, 128, 512, 1024, 1024 channels).
    pub fn new(period: i32, channel_mult: f32) -> Result<Self> {
        if period < 1 {
            return Err(VocoderError::Config(format!(
                "period must be >= 1, got {period}"
            )));
        }
        let widths = [
            1,
            scaled(32, channel_mult),
            scaled(128, channel_mult),
            scaled(512, channel_mult),
            scaled(1024, channel_mult),
            scaled(1024, channel_mult),
        ];
        let mut convs = Vec::with_capacity(5);
        for i in 0..5 {
            // Kernel (5, 1) over time; the first four layers stride by 3.
            let stride = if i < 4 { 3 } else { 1 };
            convs.push(conv2d(
                widths[i],
                widths[i + 1],
                (5, 1),
                (stride, 1),
                (2, 0),
            )?);
        }
        Ok(Self {
            period,
            convs,
            conv_post: conv2d(widths[5], 1, (3, 1), (1, 1), (1, 0))?,
        })
    }

    /// Score `audio` (`[batch, 1, samples]`).
    pub fn forward(&self, audio: &Array) -> Result<DiscriminatorOutput> {
        if audio.ndim() != 3 || audio.dim(1) != 1 {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, 1, -1],
                actual: audio.shape().to_vec(),
            });
        }
        let (batch, samples) = (audio.dim(0), audio.dim(2));
        // Reflect-pad up to a whole number of periods.
        let remainder = samples % self.period;
        let wave = audio.reshape(&[batch, samples]);
        let wave = if remainder != 0 {
            let pad = self.period - remainder;
            if pad >= samples {
                return Err(VocoderError::Audio(format!(
                    "a {samples}-sample signal is too short to reflect-pad for period {}",
                    self.period
                )));
            }
            ops::pad(&wave, &[(0, 0), (0, pad)], Some(PadMode::Reflect), None)
        } else {
            wave
        };
        // [B, T] -> [B, T / p, p, 1]: rows are time steps of the period grid.
        let mut x = wave.reshape(&[batch, wave.dim(1) / self.period, self.period, 1]);

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

impl_module_params!(PeriodDiscriminator; convs, conv_post);
