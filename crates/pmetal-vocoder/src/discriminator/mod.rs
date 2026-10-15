//! Discriminators for BigVGAN adversarial training.
//!
//! BigVGAN trains against two families of discriminators:
//! - the Multi-Period Discriminator (MPD, HiFi-GAN): reshapes the waveform
//!   into 2-D grids of several periods to expose periodic structure;
//! - a spectral discriminator: the Multi-Resolution Discriminator (MRD,
//!   BigVGAN v1) on STFT magnitudes, or in v2 either the Multi-Band
//!   Discriminator (MBD) on band-split complex spectrograms or the
//!   Multi-Scale Sub-band CQT Discriminator (the one the released v2 models
//!   were trained with).
//!
//! All take waveforms `[batch, 1, samples]` and return, per sub-discriminator,
//! logits `[batch, n]` and the intermediate feature maps used by the feature
//! matching loss. Feature maps are channels-last (`[batch, h, w, channels]`).

mod cqt;
mod mbd;
mod mpd;
mod mrd;

pub use cqt::{CqtDiscriminator, CqtDiscriminatorConfig, MultiScaleSubbandCqtDiscriminator};
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

    /// Append the learned tensors under `prefix` in the reference's naming and
    /// PyTorch layout, the inverse of [`load_state`](Self::load_state).
    fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>);
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
    /// Multi-scale sub-band constant-Q discriminator (BigVGAN v2).
    Cqt,
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

    /// Multiplier on the discriminators' channel widths (MPD and MRD).
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

    /// Use the CQT discriminator instead of the MRD.
    #[serde(default)]
    pub use_cqtd_instead_of_mrd: bool,

    /// Sampling rate of the audio (the CQT discriminator resamples it).
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: i32,

    /// Channels of the CQT discriminator's first convolution.
    #[serde(default = "default_cqtd_filters")]
    pub cqtd_filters: i32,

    /// Cap on the CQT discriminator's channel count.
    #[serde(default = "default_cqtd_max_filters")]
    pub cqtd_max_filters: i32,

    /// Per-layer growth of the CQT discriminator's channel count.
    #[serde(default = "default_cqtd_filters_scale")]
    pub cqtd_filters_scale: i32,

    /// Time dilations of the CQT discriminator's strided layers.
    #[serde(default = "default_cqtd_dilations")]
    pub cqtd_dilations: Vec<i32>,

    /// CQT hop (at twice the audio rate) of each scale.
    #[serde(default = "default_cqtd_hop_lengths")]
    pub cqtd_hop_lengths: Vec<i32>,

    /// Octaves of each scale.
    #[serde(default = "default_cqtd_n_octaves")]
    pub cqtd_n_octaves: Vec<i32>,

    /// Bins per octave of each scale.
    #[serde(default = "default_cqtd_bins_per_octaves")]
    pub cqtd_bins_per_octaves: Vec<i32>,

    /// Remove DC and peak-normalise the audio before the CQT discriminator.
    #[serde(default)]
    pub cqtd_normalize_volume: bool,
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
fn default_sampling_rate() -> i32 {
    24000
}
fn default_cqtd_filters() -> i32 {
    32
}
fn default_cqtd_max_filters() -> i32 {
    1024
}
fn default_cqtd_filters_scale() -> i32 {
    1
}
fn default_cqtd_dilations() -> Vec<i32> {
    vec![1, 2, 4]
}
fn default_cqtd_hop_lengths() -> Vec<i32> {
    vec![512, 256, 256]
}
fn default_cqtd_n_octaves() -> Vec<i32> {
    vec![9, 9, 9]
}
fn default_cqtd_bins_per_octaves() -> Vec<i32> {
    vec![24, 36, 48]
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
            use_cqtd_instead_of_mrd: false,
            sampling_rate: default_sampling_rate(),
            cqtd_filters: default_cqtd_filters(),
            cqtd_max_filters: default_cqtd_max_filters(),
            cqtd_filters_scale: default_cqtd_filters_scale(),
            cqtd_dilations: default_cqtd_dilations(),
            cqtd_hop_lengths: default_cqtd_hop_lengths(),
            cqtd_n_octaves: default_cqtd_n_octaves(),
            cqtd_bins_per_octaves: default_cqtd_bins_per_octaves(),
            cqtd_normalize_volume: false,
        }
    }
}

impl DiscriminatorConfig {
    /// Parse the discriminator settings from a reference `config.json`.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| VocoderError::Config(e.to_string()))
    }

    /// The spectral discriminator this configuration selects (the reference
    /// checks the multi-band flag first, then the CQT one).
    pub fn spectral_kind(&self) -> SpectralKind {
        if self.use_mbd_instead_of_mrd {
            SpectralKind::MultiBand
        } else if self.use_cqtd_instead_of_mrd {
            SpectralKind::Cqt
        } else {
            SpectralKind::Resolution
        }
    }

    /// The CQT discriminator's settings.
    pub fn cqt_config(&self) -> CqtDiscriminatorConfig {
        CqtDiscriminatorConfig {
            sampling_rate: self.sampling_rate,
            filters: self.cqtd_filters,
            max_filters: self.cqtd_max_filters,
            filters_scale: self.cqtd_filters_scale,
            dilations: self.cqtd_dilations.clone(),
            hop_lengths: self.cqtd_hop_lengths.clone(),
            n_octaves: self.cqtd_n_octaves.clone(),
            bins_per_octaves: self.cqtd_bins_per_octaves.clone(),
            normalize_volume: self.cqtd_normalize_volume,
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

/// A weight-normalised convolution with explicit geometry.
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
    /// Multi-scale sub-band constant-Q discriminator.
    Cqt(MultiScaleSubbandCqtDiscriminator),
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
            SpectralKind::Cqt => SpectralDiscriminator::Cqt(
                MultiScaleSubbandCqtDiscriminator::new(&config.cqt_config())?,
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
            Self::Cqt(d) => d.discriminate(audio),
        }
    }

    /// Load PyTorch-layout tensors under `prefix`.
    pub fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        match self {
            Self::Resolution(d) => d.load_state(sd, prefix),
            Self::MultiBand(d) => d.load_state(sd, prefix),
            Self::Cqt(d) => d.load_state(sd, prefix),
        }
    }

    /// Append the learned tensors under `prefix` in PyTorch layout.
    pub fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        match self {
            Self::Resolution(d) => d.save_state(prefix, out),
            Self::MultiBand(d) => d.save_state(prefix, out),
            Self::Cqt(d) => d.save_state(prefix, out),
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
            Self::Cqt(d) => d.num_parameters(),
        }
    }

    fn parameters(&self) -> pmetal_runtime::compat::ModuleParamRef<'_> {
        match self {
            Self::Resolution(d) => d.parameters(),
            Self::MultiBand(d) => d.parameters(),
            Self::Cqt(d) => d.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> pmetal_runtime::compat::ModuleParamMut<'_> {
        match self {
            Self::Resolution(d) => d.parameters_mut(),
            Self::MultiBand(d) => d.parameters_mut(),
            Self::Cqt(d) => d.parameters_mut(),
        }
    }
}
