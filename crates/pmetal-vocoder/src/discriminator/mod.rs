//! Discriminators for BigVGAN adversarial training.
//!
//! BigVGAN trains against two families of discriminators:
//! - the Multi-Period Discriminator (MPD, HiFi-GAN): reshapes the waveform
//!   into 2-D grids of several periods to expose periodic structure;
//! - a spectral discriminator: the Multi-Resolution Discriminator (MRD,
//!   BigVGAN v1) on STFT magnitudes, or in v2 the Multi-Band Discriminator
//!   (MBD) on band-split complex spectrograms.
//!
//! All take waveforms `[batch, 1, samples]` and return, per sub-discriminator,
//! logits `[batch, n]` and the intermediate feature maps used by the feature
//! matching loss. Feature maps are channels-last (`[batch, h, w, channels]`).

mod mbd;
mod mpd;
mod mrd;

pub use mbd::{BandDiscriminator, MultiBandDiscriminator};
pub use mpd::{MultiPeriodDiscriminator, PeriodDiscriminator};
pub use mrd::{MultiResolutionDiscriminator, ResolutionDiscriminator};

use crate::error::{Result, VocoderError};
use crate::nn::{Conv2d, Conv2dConfig, StateDict};
use pmetal_runtime::compat::{Array, ModuleParameters, nn};
use serde::{Deserialize, Serialize};

/// Slope of the leaky ReLU every discriminator uses.
pub(crate) const LEAKY_SLOPE: f32 = 0.1;

/// Output from a single sub-discriminator.
#[derive(Debug)]
pub struct DiscriminatorOutput {
    /// Real/fake scores, flattened to `[batch, n]`.
    pub logits: Array,
    /// Intermediate feature maps, channels-last, for the feature matching loss.
    pub features: Vec<Array>,
}

/// A bank of sub-discriminators over waveforms.
pub trait Discriminator: ModuleParameters {
    /// Score `audio` (`[batch, 1, samples]`) with every sub-discriminator.
    fn discriminate(&self, audio: &Array) -> Result<Vec<DiscriminatorOutput>>;

    /// Load PyTorch-layout tensors under `prefix` (the reference's
    /// `mpd` / `mrd` state dicts).
    fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()>;
}

/// Which spectral discriminator accompanies the MPD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SpectralKind {
    /// Multi-resolution STFT-magnitude discriminator (BigVGAN v1).
    #[default]
    Resolution,
    /// Multi-band, multi-window complex-spectrogram discriminator.
    MultiBand,
}

/// Discriminator settings; deserialises from the reference `config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscriminatorConfig {
    /// Periods of the MPD's sub-discriminators.
    #[serde(default = "default_mpd_reshapes")]
    pub mpd_reshapes: Vec<i32>,

    /// STFT resolutions `[n_fft, hop, win]` of the MRD.
    #[serde(default = "default_resolutions")]
    pub resolutions: Vec<[i32; 3]>,

    /// Multiplier on the discriminators' channel widths.
    #[serde(default = "default_channel_mult")]
    pub discriminator_channel_mult: f32,

    /// Spectral norm instead of weight norm (not supported).
    #[serde(default)]
    pub use_spectral_norm: bool,

    /// Use the multi-band discriminator instead of the MRD.
    #[serde(default)]
    pub use_mbd_instead_of_mrd: bool,

    /// FFT sizes of the multi-band discriminator.
    #[serde(default = "default_mbd_fft_sizes")]
    pub mbd_fft_sizes: Vec<i32>,
}

fn default_mpd_reshapes() -> Vec<i32> {
    vec![2, 3, 5, 7, 11]
}
fn default_resolutions() -> Vec<[i32; 3]> {
    vec![[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]
}
fn default_channel_mult() -> f32 {
    1.0
}
fn default_mbd_fft_sizes() -> Vec<i32> {
    vec![2048, 1024, 512]
}

impl Default for DiscriminatorConfig {
    fn default() -> Self {
        Self {
            mpd_reshapes: default_mpd_reshapes(),
            resolutions: default_resolutions(),
            discriminator_channel_mult: 1.0,
            use_spectral_norm: false,
            use_mbd_instead_of_mrd: false,
            mbd_fft_sizes: default_mbd_fft_sizes(),
        }
    }
}

impl DiscriminatorConfig {
    /// Parse the discriminator settings from a reference `config.json`.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| VocoderError::Config(e.to_string()))
    }

    /// The spectral discriminator this configuration selects.
    pub fn spectral_kind(&self) -> SpectralKind {
        if self.use_mbd_instead_of_mrd {
            SpectralKind::MultiBand
        } else {
            SpectralKind::Resolution
        }
    }

    fn check(&self) -> Result<()> {
        if self.use_spectral_norm {
            return Err(VocoderError::Config(
                "use_spectral_norm is not supported; the released models use weight norm".into(),
            ));
        }
        if self.discriminator_channel_mult <= 0.0 {
            return Err(VocoderError::Config(
                "discriminator_channel_mult must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// `int(base * mult)` as the reference computes channel widths, at least 1.
pub(crate) fn scaled(base: i32, mult: f32) -> i32 {
    ((base as f32 * mult) as i32).max(1)
}

/// Apply a convolution and the discriminators' leaky ReLU.
pub(crate) fn conv_act(conv: &Conv2d, x: &Array) -> Result<Array> {
    Ok(nn::leaky_relu(&conv.forward(x)?, LEAKY_SLOPE))
}

/// Flatten `[batch, h, w, c]` scores to `[batch, h * w * c]`.
pub(crate) fn flatten_logits(x: &Array) -> Array {
    x.reshape(&[x.dim(0), -1])
}

/// The generator-side view of a Conv2d stack: build a weight-normalised
/// convolution with the reference's `get_padding`.
pub(crate) fn conv2d(
    in_channels: i32,
    out_channels: i32,
    kernel: (i32, i32),
    stride: (i32, i32),
    padding: (i32, i32),
) -> Result<Conv2d> {
    Conv2d::new(
        Conv2dConfig::new(in_channels, out_channels, kernel)
            .stride(stride)
            .padding(padding),
    )
}

/// The combined discriminator BigVGAN trains against: an MPD plus a spectral
/// discriminator.
#[derive(Debug)]
pub struct BigVGANDiscriminator {
    /// Multi-Period Discriminator.
    pub mpd: MultiPeriodDiscriminator,
    /// Spectral discriminator (the reference calls it `mrd` whatever its kind).
    pub spectral: SpectralDiscriminator,
}

/// The spectral half of [`BigVGANDiscriminator`].
#[derive(Debug)]
pub enum SpectralDiscriminator {
    /// Multi-resolution STFT-magnitude discriminator.
    Resolution(MultiResolutionDiscriminator),
    /// Multi-band complex-spectrogram discriminator.
    MultiBand(MultiBandDiscriminator),
}

impl BigVGANDiscriminator {
    /// Build the discriminators a configuration selects.
    pub fn new(config: &DiscriminatorConfig) -> Result<Self> {
        config.check()?;
        let spectral = match config.spectral_kind() {
            SpectralKind::Resolution => {
                SpectralDiscriminator::Resolution(MultiResolutionDiscriminator::with_resolutions(
                    &config.resolutions,
                    config.discriminator_channel_mult,
                )?)
            }
            SpectralKind::MultiBand => SpectralDiscriminator::MultiBand(
                MultiBandDiscriminator::new(&config.mbd_fft_sizes, 32)?,
            ),
        };
        Ok(Self {
            mpd: MultiPeriodDiscriminator::with_periods(
                &config.mpd_reshapes,
                config.discriminator_channel_mult,
            )?,
            spectral,
        })
    }

    /// Score `audio` with both discriminator banks.
    pub fn forward(
        &self,
        audio: &Array,
    ) -> Result<(Vec<DiscriminatorOutput>, Vec<DiscriminatorOutput>)> {
        Ok((
            self.mpd.discriminate(audio)?,
            self.spectral.discriminate(audio)?,
        ))
    }

    /// Load the reference's `bigvgan_discriminator_optimizer.pt` entries:
    /// `mpd` into the MPD and `mrd` into the spectral discriminator.
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> Result<()> {
        let mut mpd = crate::checkpoint::load_state_dict(path, Some("mpd"))?;
        self.mpd.load_state(&mut mpd, "")?;
        mpd.finish()?;
        let mut mrd = crate::checkpoint::load_state_dict(path, Some("mrd"))?;
        self.spectral.load_state(&mut mrd, "")?;
        mrd.finish()
    }
}

impl SpectralDiscriminator {
    /// Score `audio`.
    pub fn discriminate(&self, audio: &Array) -> Result<Vec<DiscriminatorOutput>> {
        match self {
            Self::Resolution(d) => d.discriminate(audio),
            Self::MultiBand(d) => d.discriminate(audio),
        }
    }

    /// Load PyTorch-layout tensors under `prefix`.
    pub fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        match self {
            Self::Resolution(d) => d.load_state(sd, prefix),
            Self::MultiBand(d) => d.load_state(sd, prefix),
        }
    }
}

pmetal_runtime::impl_module_params!(BigVGANDiscriminator; mpd, spectral);

crate::nn::impl_no_linears!(SpectralDiscriminator);

impl ModuleParameters for SpectralDiscriminator {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Resolution(d) => d.num_parameters(),
            Self::MultiBand(d) => d.num_parameters(),
        }
    }

    fn parameters(&self) -> pmetal_runtime::compat::ModuleParamRef<'_> {
        match self {
            Self::Resolution(d) => d.parameters(),
            Self::MultiBand(d) => d.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> pmetal_runtime::compat::ModuleParamMut<'_> {
        match self {
            Self::Resolution(d) => d.parameters_mut(),
            Self::MultiBand(d) => d.parameters_mut(),
        }
    }
}
