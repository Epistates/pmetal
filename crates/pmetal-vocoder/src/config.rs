//! BigVGAN configuration.
//!
//! Field names and defaults follow the reference implementation's
//! `config.json`, so a released model's file deserialises as is (unknown
//! training-only keys are ignored).

use crate::error::{Result, VocoderError};
use serde::{Deserialize, Serialize};

/// Which residual block layout the generator uses.
///
/// Serialised as the reference's `"1"` / `"2"` strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ResBlockKind {
    /// `AMPBlock1`: a dilated and a dilation-1 convolution per stage.
    #[default]
    #[serde(rename = "1")]
    One,
    /// `AMPBlock2`: a single dilated convolution per stage.
    #[serde(rename = "2")]
    Two,
}

/// Activation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ActivationType {
    /// Snake activation: `x + sin^2(alpha x) / alpha`.
    Snake,
    /// SnakeBeta activation: `x + sin^2(alpha x) / beta`.
    #[default]
    SnakeBeta,
}

/// BigVGAN model and audio front-end configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BigVGANConfig {
    /// Residual block layout (`"1"` or `"2"`).
    #[serde(default)]
    pub resblock: ResBlockKind,

    /// Number of mel frequency bins.
    #[serde(default = "default_num_mels")]
    pub num_mels: i32,

    /// Upsampling rate of each transposed-convolution stage; the product is
    /// the hop size.
    #[serde(default = "default_upsample_rates")]
    pub upsample_rates: Vec<i32>,

    /// Kernel size of each upsampling convolution.
    #[serde(default = "default_upsample_kernel_sizes")]
    pub upsample_kernel_sizes: Vec<i32>,

    /// Channels after the input convolution; halved by every stage.
    #[serde(default = "default_upsample_initial_channel")]
    pub upsample_initial_channel: i32,

    /// Kernel size of each parallel residual block.
    #[serde(default = "default_resblock_kernel_sizes")]
    pub resblock_kernel_sizes: Vec<i32>,

    /// Dilations of each parallel residual block.
    #[serde(default = "default_resblock_dilation_sizes")]
    pub resblock_dilation_sizes: Vec<Vec<i32>>,

    /// Whether the output goes through `tanh` (otherwise it is clamped to
    /// `[-1, 1]`). BigVGAN-v2 models set this to `false`.
    #[serde(default = "default_true")]
    pub use_tanh_at_final: bool,

    /// Whether the final convolution has a bias. BigVGAN-v2 models set this to
    /// `false`.
    #[serde(default = "default_true")]
    pub use_bias_at_final: bool,

    /// Periodic activation.
    #[serde(default)]
    pub activation: ActivationType,

    /// Whether the activation parameters are stored in log scale.
    #[serde(default = "default_true")]
    pub snake_logscale: bool,

    /// Audio sampling rate in Hz.
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: i32,

    /// FFT size of the mel front end.
    #[serde(default = "default_n_fft")]
    pub n_fft: i32,

    /// Hop size in samples.
    #[serde(default = "default_hop_size")]
    pub hop_size: i32,

    /// Window size of the mel front end.
    #[serde(default = "default_win_size")]
    pub win_size: i32,

    /// Lowest mel filter edge in Hz.
    #[serde(default)]
    pub fmin: i32,

    /// Highest mel filter edge in Hz; `None` (JSON `null`) means Nyquist.
    #[serde(default)]
    pub fmax: Option<i32>,

    /// Highest mel edge used by the single-scale mel loss; `None` means Nyquist.
    #[serde(default)]
    pub fmax_for_loss: Option<i32>,
}

fn default_num_mels() -> i32 {
    100
}
fn default_upsample_rates() -> Vec<i32> {
    vec![4, 4, 2, 2, 2, 2]
}
fn default_upsample_kernel_sizes() -> Vec<i32> {
    vec![8, 8, 4, 4, 4, 4]
}
fn default_upsample_initial_channel() -> i32 {
    1536
}
fn default_resblock_kernel_sizes() -> Vec<i32> {
    vec![3, 7, 11]
}
fn default_resblock_dilation_sizes() -> Vec<Vec<i32>> {
    vec![vec![1, 3, 5]; 3]
}
fn default_true() -> bool {
    true
}
fn default_sampling_rate() -> i32 {
    24000
}
fn default_n_fft() -> i32 {
    1024
}
fn default_hop_size() -> i32 {
    256
}
fn default_win_size() -> i32 {
    1024
}

impl Default for BigVGANConfig {
    fn default() -> Self {
        Self::v2_24khz_100band()
    }
}

/// The settings every released BigVGAN-v2 model shares.
fn v2_base(
    num_mels: i32,
    sampling_rate: i32,
    rates: &[i32],
    kernels: &[i32],
    n_fft: i32,
    hop_size: i32,
    fmax: Option<i32>,
) -> BigVGANConfig {
    BigVGANConfig {
        resblock: ResBlockKind::One,
        num_mels,
        upsample_rates: rates.to_vec(),
        upsample_kernel_sizes: kernels.to_vec(),
        upsample_initial_channel: 1536,
        resblock_kernel_sizes: vec![3, 7, 11],
        resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
        use_tanh_at_final: false,
        use_bias_at_final: false,
        activation: ActivationType::SnakeBeta,
        snake_logscale: true,
        sampling_rate,
        n_fft,
        hop_size,
        win_size: n_fft,
        fmin: 0,
        fmax,
        fmax_for_loss: None,
    }
}

/// The settings every released BigVGAN-v1 model shares (`tanh` output, final bias).
fn v1_base(
    num_mels: i32,
    sampling_rate: i32,
    rates: &[i32],
    kernels: &[i32],
    initial_channel: i32,
    fmax: i32,
) -> BigVGANConfig {
    BigVGANConfig {
        use_tanh_at_final: true,
        use_bias_at_final: true,
        upsample_initial_channel: initial_channel,
        fmax: Some(fmax),
        ..v2_base(num_mels, sampling_rate, rates, kernels, 1024, 256, None)
    }
}

impl BigVGANConfig {
    /// `nvidia/bigvgan_v2_24khz_100band_256x`.
    pub fn v2_24khz_100band() -> Self {
        v2_base(
            100,
            24000,
            &[4, 4, 2, 2, 2, 2],
            &[8, 8, 4, 4, 4, 4],
            1024,
            256,
            None,
        )
    }

    /// `nvidia/bigvgan_v2_22khz_80band_256x`.
    pub fn v2_22khz_80band() -> Self {
        v2_base(
            80,
            22050,
            &[4, 4, 2, 2, 2, 2],
            &[8, 8, 4, 4, 4, 4],
            1024,
            256,
            None,
        )
    }

    /// `nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`.
    pub fn v2_22khz_80band_fmax8k() -> Self {
        v2_base(
            80,
            22050,
            &[4, 4, 2, 2, 2, 2],
            &[8, 8, 4, 4, 4, 4],
            1024,
            256,
            Some(8000),
        )
    }

    /// `nvidia/bigvgan_v2_44khz_128band_256x`.
    pub fn v2_44khz_128band_256x() -> Self {
        v2_base(
            128,
            44100,
            &[4, 4, 2, 2, 2, 2],
            &[8, 8, 4, 4, 4, 4],
            1024,
            256,
            None,
        )
    }

    /// `nvidia/bigvgan_v2_44khz_128band_512x`.
    pub fn v2_44khz_128band_512x() -> Self {
        v2_base(
            128,
            44100,
            &[8, 4, 2, 2, 2, 2],
            &[16, 8, 4, 4, 4, 4],
            2048,
            512,
            None,
        )
    }

    /// `nvidia/bigvgan_24khz_100band` (BigVGAN v1, large).
    pub fn v1_24khz_100band() -> Self {
        v1_base(
            100,
            24000,
            &[4, 4, 2, 2, 2, 2],
            &[8, 8, 4, 4, 4, 4],
            1536,
            12000,
        )
    }

    /// `nvidia/bigvgan_22khz_80band` (BigVGAN v1, large).
    pub fn v1_22khz_80band() -> Self {
        v1_base(
            80,
            22050,
            &[4, 4, 2, 2, 2, 2],
            &[8, 8, 4, 4, 4, 4],
            1536,
            8000,
        )
    }

    /// `nvidia/bigvgan_base_24khz_100band` (BigVGAN v1, base: about 14M parameters).
    pub fn base_24khz_100band() -> Self {
        v1_base(100, 24000, &[8, 8, 2, 2], &[16, 16, 4, 4], 512, 12000)
    }

    /// `nvidia/bigvgan_base_22khz_80band` (BigVGAN v1, base: about 14M parameters).
    pub fn base_22khz_80band() -> Self {
        v1_base(80, 22050, &[8, 8, 2, 2], &[16, 16, 4, 4], 512, 8000)
    }

    /// The preset behind a released model's Hugging Face id, if there is one
    /// (`nvidia/bigvgan_v2_24khz_100band_256x`, ...).
    pub fn for_model_id(model_id: &str) -> Option<Self> {
        let name = model_id.rsplit('/').next().unwrap_or(model_id);
        Some(match name {
            "bigvgan_v2_24khz_100band_256x" => Self::v2_24khz_100band(),
            "bigvgan_v2_22khz_80band_256x" => Self::v2_22khz_80band(),
            "bigvgan_v2_22khz_80band_fmax8k_256x" => Self::v2_22khz_80band_fmax8k(),
            "bigvgan_v2_44khz_128band_256x" => Self::v2_44khz_128band_256x(),
            "bigvgan_v2_44khz_128band_512x" => Self::v2_44khz_128band_512x(),
            "bigvgan_24khz_100band" => Self::v1_24khz_100band(),
            "bigvgan_22khz_80band" => Self::v1_22khz_80band(),
            "bigvgan_base_24khz_100band" => Self::base_24khz_100band(),
            "bigvgan_base_22khz_80band" => Self::base_22khz_80band(),
            _ => return None,
        })
    }

    /// Parse a reference `config.json`.
    pub fn from_json(json: &str) -> Result<Self> {
        let config: Self =
            serde_json::from_str(json).map_err(|e| VocoderError::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Total upsampling factor: samples produced per mel frame.
    pub fn upsample_factor(&self) -> i32 {
        self.upsample_rates.iter().product()
    }

    /// Number of upsampling stages.
    pub fn num_upsample_stages(&self) -> usize {
        self.upsample_rates.len()
    }

    /// Channel count entering stage `i`'s upsampler (`[0]` is after the input
    /// convolution; the last entry is the channel count before the output
    /// convolution).
    pub fn channel_sizes(&self) -> Vec<i32> {
        (0..=self.num_upsample_stages())
            .map(|i| self.upsample_initial_channel >> i)
            .collect()
    }

    /// The mel filterbank's upper edge in Hz.
    pub fn effective_fmax(&self) -> f32 {
        self.fmax
            .map_or(self.sampling_rate as f32 / 2.0, |f| f as f32)
    }

    /// Check the configuration is internally consistent.
    pub fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(VocoderError::Config(msg));
        let stages = self.upsample_rates.len();
        if stages == 0 {
            return bad("upsample_rates must not be empty".into());
        }
        if self.upsample_kernel_sizes.len() != stages {
            return bad(format!(
                "upsample_kernel_sizes has {} entries for {stages} upsample stages",
                self.upsample_kernel_sizes.len()
            ));
        }
        for (&rate, &kernel) in self.upsample_rates.iter().zip(&self.upsample_kernel_sizes) {
            // The transposed convolution pads (k - u) / 2 per side; the output is
            // exactly `u` times the input only when k - u is even and non-negative.
            if rate < 1 || kernel < rate || (kernel - rate) % 2 != 0 {
                return bad(format!(
                    "upsample stage (rate {rate}, kernel {kernel}) must have kernel >= rate and an even kernel - rate"
                ));
            }
        }
        if self.upsample_initial_channel < 1 << stages {
            return bad(format!(
                "upsample_initial_channel {} cannot be halved {stages} times",
                self.upsample_initial_channel
            ));
        }
        if self.resblock_kernel_sizes.is_empty()
            || self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len()
        {
            return bad(format!(
                "resblock_kernel_sizes ({}) and resblock_dilation_sizes ({}) must pair up",
                self.resblock_kernel_sizes.len(),
                self.resblock_dilation_sizes.len()
            ));
        }
        if self
            .resblock_kernel_sizes
            .iter()
            .any(|&k| k < 1 || k % 2 == 0)
        {
            return bad("resblock kernel sizes must be odd (symmetric same-padding)".into());
        }
        if self
            .resblock_dilation_sizes
            .iter()
            .any(|d| d.is_empty() || d.iter().any(|&x| x < 1))
        {
            return bad("resblock dilations must be non-empty and positive".into());
        }
        if self.upsample_factor() != self.hop_size {
            return bad(format!(
                "the upsample rates multiply to {} but hop_size is {}",
                self.upsample_factor(),
                self.hop_size
            ));
        }
        if self.num_mels < 1 || self.n_fft < 2 || self.win_size < 1 || self.win_size > self.n_fft {
            return bad(format!(
                "invalid mel front end (num_mels {}, n_fft {}, win_size {})",
                self.num_mels, self.n_fft, self.win_size
            ));
        }
        if self.effective_fmax() > self.sampling_rate as f32 / 2.0 {
            return bad(format!(
                "fmax {} exceeds the Nyquist frequency of a {} Hz signal",
                self.effective_fmax(),
                self.sampling_rate
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_valid() {
        for config in [
            BigVGANConfig::v2_24khz_100band(),
            BigVGANConfig::v2_22khz_80band(),
            BigVGANConfig::v2_22khz_80band_fmax8k(),
            BigVGANConfig::v2_44khz_128band_256x(),
            BigVGANConfig::v2_44khz_128band_512x(),
            BigVGANConfig::v1_24khz_100band(),
            BigVGANConfig::v1_22khz_80band(),
            BigVGANConfig::base_24khz_100band(),
            BigVGANConfig::base_22khz_80band(),
        ] {
            config.validate().unwrap();
        }
    }

    #[test]
    fn upsample_factor_is_the_hop_size() {
        assert_eq!(BigVGANConfig::v2_24khz_100band().upsample_factor(), 256);
        assert_eq!(
            BigVGANConfig::v2_44khz_128band_512x().upsample_factor(),
            512
        );
        assert_eq!(BigVGANConfig::base_24khz_100band().upsample_factor(), 256);
    }

    #[test]
    fn channel_sizes_halve_per_stage() {
        let channels = BigVGANConfig::v2_24khz_100band().channel_sizes();
        assert_eq!(channels, [1536, 768, 384, 192, 96, 48, 24]);
    }

    #[test]
    fn rejects_a_hop_size_the_upsampling_cannot_produce() {
        let mut config = BigVGANConfig::v2_24khz_100band();
        config.hop_size = 300;
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_an_odd_stride_kernel_difference() {
        let mut config = BigVGANConfig::base_24khz_100band();
        config.upsample_kernel_sizes[0] = 15;
        assert!(config.validate().is_err());
    }
}
