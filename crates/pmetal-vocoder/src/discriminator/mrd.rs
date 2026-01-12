//! Multi-Resolution Discriminator (MRD) for BigVGAN.
//!
//! The MRD analyzes audio at multiple spectral resolutions to capture
//! both fine-grained and coarse spectral details.

use super::DiscriminatorOutput;
use crate::audio::{stft, StftConfig};
use crate::error::Result;
use crate::nn::WeightNormConv1d;
use mlx_rs::Array;

/// Multi-Resolution Discriminator.
///
/// Applies STFT at multiple resolutions and uses 2D convolutions
/// on the magnitude spectrograms.
#[derive(Debug)]
pub struct MultiResolutionDiscriminator {
    /// Individual resolution discriminators.
    pub discriminators: Vec<ResolutionDiscriminator>,
}

impl MultiResolutionDiscriminator {
    /// Create a new MRD with default resolutions.
    ///
    /// Default: n_fft = [1024, 2048, 512], hop = [120, 240, 50]
    pub fn new() -> Result<Self> {
        let resolutions = vec![
            (1024, 120, 600),  // (n_fft, hop_length, win_length)
            (2048, 240, 1200),
            (512, 50, 240),
        ];

        let discriminators = resolutions
            .into_iter()
            .map(|(n_fft, hop, win)| ResolutionDiscriminator::new(n_fft, hop, win))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { discriminators })
    }

    /// Create MRD with custom resolutions.
    pub fn with_resolutions(resolutions: Vec<(i32, i32, i32)>) -> Result<Self> {
        let discriminators = resolutions
            .into_iter()
            .map(|(n_fft, hop, win)| ResolutionDiscriminator::new(n_fft, hop, win))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { discriminators })
    }

    /// Forward pass through all resolution discriminators.
    pub fn forward(&self, audio: &Array) -> Result<Vec<DiscriminatorOutput>> {
        self.discriminators
            .iter()
            .map(|d| d.forward(audio))
            .collect()
    }
}

impl Default for MultiResolutionDiscriminator {
    fn default() -> Self {
        Self::new().expect("Failed to create MRD")
    }
}

/// Single resolution discriminator.
///
/// Computes STFT at a specific resolution and applies convolutions.
#[derive(Debug)]
pub struct ResolutionDiscriminator {
    /// STFT configuration.
    pub stft_config: StftConfig,
    /// Convolutional layers operating on magnitude spectrogram.
    pub convs: Vec<WeightNormConv1d>,
    /// Final output convolution.
    pub conv_post: WeightNormConv1d,
}

impl ResolutionDiscriminator {
    /// Create a new resolution discriminator.
    ///
    /// # Arguments
    /// * `n_fft` - FFT size
    /// * `hop_length` - Hop size
    /// * `win_length` - Window size
    pub fn new(n_fft: i32, hop_length: i32, win_length: i32) -> Result<Self> {
        let stft_config = StftConfig {
            n_fft,
            hop_length,
            win_length: Some(win_length),
            center: true,
            ..Default::default()
        };

        // Number of frequency bins
        let n_freq = n_fft / 2 + 1;

        // Convolutional layers: process spectrogram as 1D signal over time
        // with channels = frequency bins
        let channels = vec![
            (n_freq, 32),
            (32, 128),
            (128, 512),
            (512, 1024),
            (1024, 1024),
        ];

        let mut convs = Vec::with_capacity(channels.len());
        for (i, (in_ch, out_ch)) in channels.iter().enumerate() {
            let stride = if i < 4 { 2 } else { 1 };
            let conv = WeightNormConv1d::new(
                *in_ch,
                *out_ch,
                3,
                Some(stride),
                Some(1),
                None,
                None,
                Some(true),
            )?;
            convs.push(conv);
        }

        // Final convolution
        let conv_post = WeightNormConv1d::new(1024, 1, 3, Some(1), Some(1), None, None, Some(true))?;

        Ok(Self {
            stft_config,
            convs,
            conv_post,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `audio` - Audio waveform [batch, 1, samples]
    ///
    /// # Returns
    /// Discriminator output with logits and feature maps
    pub fn forward(&self, audio: &Array) -> Result<DiscriminatorOutput> {
        // Remove channel dimension for STFT: [B, 1, T] -> [B, T]
        let audio_2d = audio.squeeze()?;

        // Compute STFT magnitude spectrogram
        let stft_out = stft(&audio_2d, &self.stft_config)?;
        let magnitude = stft_out.abs()?;

        // magnitude shape: [B, freq, frames] or [freq, frames]
        // Ensure batch dimension
        let x = if magnitude.ndim() == 2 {
            magnitude.reshape(&[1, magnitude.dim(0), magnitude.dim(1)])?
        } else {
            magnitude
        };

        // Apply convolutions and collect features
        let mut features = Vec::new();
        let mut x = x;

        for conv in &self.convs {
            x = conv.forward(&x)?;
            x = mlx_rs::nn::leaky_relu(&x, 0.1)?;
            features.push(x.clone());
        }

        // Final convolution
        let logits = self.conv_post.forward(&x)?;

        Ok(DiscriminatorOutput { logits, features })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_discriminator() {
        let disc = ResolutionDiscriminator::new(1024, 256, 1024).unwrap();

        let audio = mlx_rs::random::normal::<f32>(&[1, 1, 4096], None, None, None).unwrap();
        let output = disc.forward(&audio).unwrap();
        output.logits.eval().unwrap();

        assert!(!output.features.is_empty());
    }

    #[test]
    fn test_mrd() {
        let mrd = MultiResolutionDiscriminator::new().unwrap();

        let audio = mlx_rs::random::normal::<f32>(&[1, 1, 8000], None, None, None).unwrap();
        let outputs = mrd.forward(&audio).unwrap();

        assert_eq!(outputs.len(), 3); // 3 resolutions
    }

    #[test]
    fn test_mrd_custom_resolutions() {
        let mrd =
            MultiResolutionDiscriminator::with_resolutions(vec![(512, 128, 512), (1024, 256, 1024)])
                .unwrap();
        assert_eq!(mrd.discriminators.len(), 2);
    }
}
