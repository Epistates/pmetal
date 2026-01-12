//! BigVGAN configuration.

use serde::{Deserialize, Serialize};

/// BigVGAN model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigVGANConfig {
    /// Number of mel frequency bins.
    #[serde(default = "default_num_mels")]
    pub num_mels: i32,

    /// Upsampling rates for each stage.
    /// Product should equal hop_size (e.g., [4,4,2,2,2,2] = 256).
    #[serde(default = "default_upsample_rates")]
    pub upsample_rates: Vec<i32>,

    /// Kernel sizes for upsampling convolutions.
    #[serde(default = "default_upsample_kernel_sizes")]
    pub upsample_kernel_sizes: Vec<i32>,

    /// Initial channel dimension after first convolution.
    #[serde(default = "default_upsample_initial_channel")]
    pub upsample_initial_channel: i32,

    /// Kernel sizes for residual blocks.
    #[serde(default = "default_resblock_kernel_sizes")]
    pub resblock_kernel_sizes: Vec<i32>,

    /// Dilation sizes for each residual block layer.
    #[serde(default = "default_resblock_dilation_sizes")]
    pub resblock_dilation_sizes: Vec<Vec<i32>>,

    /// Activation function type.
    #[serde(default = "default_activation")]
    pub activation: ActivationType,

    /// Whether to use log scale for Snake alpha parameter.
    #[serde(default = "default_snake_logscale")]
    pub snake_logscale: bool,

    /// Audio sampling rate in Hz.
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: i32,

    /// FFT size for mel spectrogram.
    #[serde(default = "default_n_fft")]
    pub n_fft: i32,

    /// Hop size in samples.
    #[serde(default = "default_hop_size")]
    pub hop_size: i32,

    /// Window size for STFT.
    #[serde(default = "default_win_size")]
    pub win_size: i32,

    /// Maximum frequency for mel filterbank.
    #[serde(default = "default_fmax")]
    pub fmax: i32,

    /// Minimum frequency for mel filterbank.
    #[serde(default = "default_fmin")]
    pub fmin: i32,
}

/// Activation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ActivationType {
    /// Snake activation: x + (1/α) * sin²(αx)
    Snake,
    /// SnakeBeta activation: x + (1/β) * sin²(αx)
    #[default]
    SnakeBeta,
}

// Default functions for serde
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
    vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
}
fn default_activation() -> ActivationType {
    ActivationType::SnakeBeta
}
fn default_snake_logscale() -> bool {
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
fn default_fmax() -> i32 {
    12000
}
fn default_fmin() -> i32 {
    0
}

impl Default for BigVGANConfig {
    fn default() -> Self {
        Self::v2_24khz_100band()
    }
}

impl BigVGANConfig {
    /// BigVGAN v2 configuration for 24kHz, 100 mel bands.
    ///
    /// This is the default configuration matching:
    /// `nvidia/bigvgan_v2_24khz_100band_256x`
    pub fn v2_24khz_100band() -> Self {
        Self {
            num_mels: 100,
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            upsample_initial_channel: 1536,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            activation: ActivationType::SnakeBeta,
            snake_logscale: true,
            sampling_rate: 24000,
            n_fft: 1024,
            hop_size: 256,
            win_size: 1024,
            fmax: 12000,
            fmin: 0,
        }
    }

    /// BigVGAN v2 configuration for 44.1kHz, 128 mel bands.
    ///
    /// Higher quality configuration matching:
    /// `nvidia/bigvgan_v2_44khz_128band_512x`
    pub fn v2_44khz_128band() -> Self {
        Self {
            num_mels: 128,
            upsample_rates: vec![4, 4, 2, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4, 4],
            upsample_initial_channel: 1536,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            activation: ActivationType::SnakeBeta,
            snake_logscale: true,
            sampling_rate: 44100,
            n_fft: 2048,
            hop_size: 512,
            win_size: 2048,
            fmax: 22050,
            fmin: 0,
        }
    }

    /// Base (smaller) configuration for 24kHz.
    ///
    /// ~14M parameters, faster inference.
    pub fn base_24khz_100band() -> Self {
        Self {
            num_mels: 100,
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            upsample_initial_channel: 512, // Smaller
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            activation: ActivationType::SnakeBeta,
            snake_logscale: true,
            sampling_rate: 24000,
            n_fft: 1024,
            hop_size: 256,
            win_size: 1024,
            fmax: 12000,
            fmin: 0,
        }
    }

    /// Calculate total upsampling factor.
    pub fn upsample_factor(&self) -> i32 {
        self.upsample_rates.iter().product()
    }

    /// Get the number of upsampling stages.
    pub fn num_upsample_stages(&self) -> usize {
        self.upsample_rates.len()
    }

    /// Calculate channel dimensions for each stage.
    pub fn channel_sizes(&self) -> Vec<i32> {
        let mut channels = vec![self.upsample_initial_channel];
        let mut ch = self.upsample_initial_channel;
        for _ in 0..self.num_upsample_stages() {
            ch /= 2;
            channels.push(ch);
        }
        channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BigVGANConfig::default();
        assert_eq!(config.num_mels, 100);
        assert_eq!(config.sampling_rate, 24000);
        assert_eq!(config.upsample_factor(), 256);
    }

    #[test]
    fn test_upsample_factor() {
        let config = BigVGANConfig::v2_24khz_100band();
        assert_eq!(config.upsample_factor(), 256); // 4*4*2*2*2*2

        let config = BigVGANConfig::v2_44khz_128band();
        assert_eq!(config.upsample_factor(), 512); // 4*4*2*2*2*2*2
    }

    #[test]
    fn test_channel_sizes() {
        let config = BigVGANConfig::v2_24khz_100band();
        let channels = config.channel_sizes();
        assert_eq!(channels[0], 1536);
        assert_eq!(channels[1], 768);
        assert_eq!(channels[2], 384);
        assert_eq!(channels[3], 192);
        assert_eq!(channels[4], 96);
        assert_eq!(channels[5], 48);
        assert_eq!(channels[6], 24);
    }
}
