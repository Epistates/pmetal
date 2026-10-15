//! Multi-Scale Sub-band CQT Discriminator (MS-SB-CQT; BigVGAN v2).
//!
//! Each sub-discriminator resamples the waveform to twice its rate, takes a
//! constant-Q transform whose frequency axis is logarithmic (so equal musical
//! intervals get equal resolution, unlike the linear STFT bins of the MRD),
//! convolves each octave of the complex spectrogram separately, and joins the
//! octaves for a strided, dilated convolution stack. The idea comes from the
//! multi-scale CQT discriminator of Gu et al. (2023, "MSSB-CQTD" in Amphion).

use super::{Discriminator, DiscriminatorOutput, LEAKY_SLOPE, flatten_logits};
use crate::audio::{Cqt, CqtConfig, Resampler};
use crate::error::{Result, VocoderError};
use crate::nn::{Conv2d, Conv2dConfig, StateDict, join_key};
use pmetal_runtime::compat::nn;
use pmetal_runtime::compat::{Array, ops};
use pmetal_runtime::impl_module_params;

/// Settings of the CQT discriminator bank; the field names follow the
/// reference's `cqtd_*` configuration keys.
#[derive(Debug, Clone, PartialEq)]
pub struct CqtDiscriminatorConfig {
    /// Sampling rate of the audio being judged.
    pub sampling_rate: i32,
    /// Channels of the first convolution.
    pub filters: i32,
    /// Cap on the channel count of any layer.
    pub max_filters: i32,
    /// Factor by which the channel count grows per layer (1 keeps it fixed).
    pub filters_scale: i32,
    /// Dilation along time of each strided convolution.
    pub dilations: Vec<i32>,
    /// Hop of each sub-discriminator's CQT, in samples at twice the audio rate.
    pub hop_lengths: Vec<i32>,
    /// Octaves of each sub-discriminator.
    pub n_octaves: Vec<i32>,
    /// Bins per octave of each sub-discriminator.
    pub bins_per_octaves: Vec<i32>,
    /// Remove DC and peak-normalise the input first.
    pub normalize_volume: bool,
}

impl CqtDiscriminatorConfig {
    /// BigVGAN v2's settings for audio at `sampling_rate`.
    pub fn bigvgan_v2(sampling_rate: i32) -> Self {
        Self {
            sampling_rate,
            filters: 128,
            max_filters: 1024,
            filters_scale: 1,
            dilations: vec![1, 2, 4],
            hop_lengths: vec![512, 256, 256],
            n_octaves: vec![9, 9, 9],
            bins_per_octaves: vec![24, 36, 48],
            normalize_volume: false,
        }
    }
}

/// A bank of CQT discriminators at several time-frequency resolutions.
#[derive(Debug)]
pub struct MultiScaleSubbandCqtDiscriminator {
    /// One discriminator per scale.
    pub discriminators: Vec<CqtDiscriminator>,
}

impl MultiScaleSubbandCqtDiscriminator {
    /// Build the bank for `config`.
    pub fn new(config: &CqtDiscriminatorConfig) -> Result<Self> {
        let scales = config.hop_lengths.len();
        if scales == 0
            || config.n_octaves.len() != scales
            || config.bins_per_octaves.len() != scales
        {
            return Err(VocoderError::Config(
                "cqtd_hop_lengths, cqtd_n_octaves and cqtd_bins_per_octaves must be non-empty and equally long".into(),
            ));
        }
        Ok(Self {
            discriminators: (0..scales)
                .map(|i| {
                    CqtDiscriminator::new(
                        config,
                        config.hop_lengths[i],
                        config.n_octaves[i],
                        config.bins_per_octaves[i],
                    )
                })
                .collect::<Result<_>>()?,
        })
    }

    /// Fold weight normalisation into the weights.
    pub fn fuse_weight_norm(&mut self) {
        self.discriminators
            .iter_mut()
            .for_each(CqtDiscriminator::fuse);
    }
}

impl Discriminator for MultiScaleSubbandCqtDiscriminator {
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
}

impl_module_params!(MultiScaleSubbandCqtDiscriminator; discriminators);

/// One CQT discriminator.
#[derive(Debug)]
pub struct CqtDiscriminator {
    resampler: Resampler,
    cqt: Cqt,
    n_octaves: i32,
    bins_per_octave: i32,
    normalize_volume: bool,
    /// Per-octave 3x9 convolutions on the real and imaginary channels.
    pub conv_pres: Vec<Conv2d>,
    /// The convolution stack: a plain first layer, one strided dilated
    /// layer per dilation, and a closing 3x3 layer.
    pub convs: Vec<Conv2d>,
    /// Output convolution to a single channel.
    pub conv_post: Conv2d,
}

impl CqtDiscriminator {
    /// A discriminator with CQT hop `hop_length`, `n_octaves` octaves of
    /// `bins_per_octave` bins.
    pub fn new(
        config: &CqtDiscriminatorConfig,
        hop_length: i32,
        n_octaves: i32,
        bins_per_octave: i32,
    ) -> Result<Self> {
        if config.sampling_rate < 1 || config.filters < 1 || config.max_filters < config.filters {
            return Err(VocoderError::Config(
                "invalid CQT discriminator widths".into(),
            ));
        }
        if config.filters_scale < 1 {
            return Err(VocoderError::Config(
                "cqtd_filters_scale must be >= 1".into(),
            ));
        }
        let fs = config.sampling_rate;
        let resampler = Resampler::new(fs, 2 * fs)?;
        let cqt = Cqt::new(CqtConfig::new(
            2 * fs,
            hop_length,
            bins_per_octave * n_octaves,
            bins_per_octave,
        ))?;

        // Real and imaginary parts are the two input channels.
        let channels = 2;
        let width = |level: usize| {
            config
                .filters_scale
                .saturating_pow(level as u32)
                .saturating_mul(config.filters)
                .min(config.max_filters)
        };
        let pad = |kernel: (i32, i32), dilation: (i32, i32)| {
            (
                (kernel.0 - 1) * dilation.0 / 2,
                (kernel.1 - 1) * dilation.1 / 2,
            )
        };
        let wide = (3, 9);

        let conv_pres = (0..n_octaves)
            .map(|_| {
                Conv2d::new(
                    Conv2dConfig::new(channels, channels, wide)
                        .padding(pad(wide, (1, 1)))
                        .weight_norm(false),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let mut convs = vec![Conv2d::new(
            Conv2dConfig::new(channels, config.filters, wide)
                .padding(pad(wide, (1, 1)))
                .weight_norm(false),
        )?];
        let mut in_channels = width(1).min(config.max_filters);
        // The reference sizes the first strided layer's input from the scaled
        // width even though the plain layer before it emits `filters`; they
        // agree for scale 1, and a larger scale is rejected below.
        if in_channels != config.filters {
            return Err(VocoderError::Config(
                "cqtd_filters_scale > 1 is not supported: the reference stack feeds its first layer a mismatched width".into(),
            ));
        }
        for (i, &dilation) in config.dilations.iter().enumerate() {
            let out_channels = width(i + 1);
            convs.push(Conv2d::new(
                Conv2dConfig::new(in_channels, out_channels, wide)
                    .stride((1, 2))
                    .dilation((dilation, 1))
                    .padding(pad(wide, (dilation, 1))),
            )?);
            in_channels = out_channels;
        }
        let closing = (3, 3);
        let out_channels = width(config.dilations.len() + 1);
        convs.push(Conv2d::new(
            Conv2dConfig::new(in_channels, out_channels, closing).padding(pad(closing, (1, 1))),
        )?);
        let conv_post =
            Conv2d::new(Conv2dConfig::new(out_channels, 1, closing).padding(pad(closing, (1, 1))))?;

        Ok(Self {
            resampler,
            cqt,
            n_octaves,
            bins_per_octave,
            normalize_volume: config.normalize_volume,
            conv_pres,
            convs,
            conv_post,
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
        let mut wave = audio.reshape(&[audio.dim(0), audio.dim(2)]);
        if self.normalize_volume {
            wave = wave.subtract(&wave.mean_axis(1, true));
            let peak = wave.abs_val().max_axis(1, true);
            wave = wave
                .multiply(&Array::from_f32(0.8))
                .divide(&peak.add(&Array::from_f32(1e-9)));
        }
        let wave = self.resampler.forward(&wave)?;
        // [batch, frames, bins, 2]: real and imaginary parts as channels.
        let z = self.cqt.forward(&wave)?;

        let octaves = (0..self.n_octaves)
            .map(|i| {
                let slice = ops::slice_axis(
                    &z,
                    2,
                    i * self.bins_per_octave,
                    (i + 1) * self.bins_per_octave,
                );
                self.conv_pres[i as usize].forward(&slice)
            })
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Array> = octaves.iter().collect();
        let mut x = ops::concatenate_axis(&refs, 2);

        let mut features = Vec::with_capacity(self.convs.len());
        for conv in &self.convs {
            x = nn::leaky_relu(&conv.forward(&x)?, LEAKY_SLOPE);
            features.push(x.clone());
        }
        let logits = self.conv_post.forward(&x)?;
        Ok(DiscriminatorOutput {
            logits: flatten_logits(&logits),
            features,
        })
    }

    fn fuse(&mut self) {
        self.convs.iter_mut().for_each(Conv2d::fuse);
        self.conv_post.fuse();
    }

    fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        for (i, c) in self.conv_pres.iter_mut().enumerate() {
            c.load_state(sd, &join_key(prefix, &format!("conv_pres.{i}")))?;
        }
        for (i, c) in self.convs.iter_mut().enumerate() {
            c.load_state(sd, &join_key(prefix, &format!("convs.{i}")))?;
        }
        self.conv_post
            .load_state(sd, &join_key(prefix, "conv_post"))?;
        // Derived buffers: verified, not loaded.
        self.cqt
            .check_buffers(sd, &join_key(prefix, "cqt_transform"))?;
        self.resampler
            .check_buffer(sd, &join_key(prefix, "resample.kernel"))
    }
}

impl_module_params!(CqtDiscriminator; conv_pres, convs, conv_post);
