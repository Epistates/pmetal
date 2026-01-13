//! Multi-Period Discriminator (MPD) for BigVGAN.
//!
//! The MPD captures periodic patterns in audio by reshaping the waveform
//! into 2D representations with different periods and applying 2D convolutions.

use super::DiscriminatorOutput;
use crate::error::Result;
use crate::nn::WeightNormConv1d;
use mlx_rs::Array;

/// Multi-Period Discriminator.
///
/// Consists of multiple period discriminators with different periods
/// (e.g., 2, 3, 5, 7, 11) to capture various periodic structures.
#[derive(Debug)]
pub struct MultiPeriodDiscriminator {
    /// Individual period discriminators.
    pub discriminators: Vec<PeriodDiscriminator>,
}

impl MultiPeriodDiscriminator {
    /// Create a new MPD with default periods [2, 3, 5, 7, 11].
    pub fn new() -> Result<Self> {
        let periods = vec![2, 3, 5, 7, 11];
        let discriminators = periods
            .into_iter()
            .map(PeriodDiscriminator::new)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { discriminators })
    }

    /// Create MPD with custom periods.
    pub fn with_periods(periods: Vec<i32>) -> Result<Self> {
        let discriminators = periods
            .into_iter()
            .map(PeriodDiscriminator::new)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { discriminators })
    }

    /// Forward pass through all period discriminators.
    pub fn forward(&self, audio: &Array) -> Result<Vec<DiscriminatorOutput>> {
        self.discriminators
            .iter()
            .map(|d| d.forward(audio))
            .collect()
    }
}

impl Default for MultiPeriodDiscriminator {
    fn default() -> Self {
        Self::new().expect("Failed to create MPD")
    }
}

/// Single period discriminator.
///
/// Reshapes audio with a specific period and applies 2D convolutions.
#[derive(Debug)]
pub struct PeriodDiscriminator {
    /// Period for reshaping.
    pub period: i32,
    /// Convolutional layers.
    pub convs: Vec<WeightNormConv1d>,
    /// Final output convolution.
    pub conv_post: WeightNormConv1d,
}

impl PeriodDiscriminator {
    /// Create a new period discriminator.
    ///
    /// # Arguments
    /// * `period` - Reshaping period
    pub fn new(period: i32) -> Result<Self> {
        // Channel progression: period -> 32 -> 128 -> 512 -> 1024 -> 1024
        // First layer takes `period` channels since we reshape [B, 1, T] -> [B, period, T/period]
        let channels = vec![
            (period, 32),
            (32, 128),
            (128, 512),
            (512, 1024),
            (1024, 1024),
        ];

        let mut convs = Vec::with_capacity(channels.len());
        for (i, (in_ch, out_ch)) in channels.iter().enumerate() {
            // Stride of 3 for first 4 layers, 1 for last
            let stride = if i < 4 { 3 } else { 1 };
            let kernel = 5;
            let padding = 2;

            // Note: In actual BigVGAN, these are 2D convs applied after reshape
            // We approximate with 1D conv treating period as channels
            let conv = WeightNormConv1d::new(
                *in_ch,
                *out_ch,
                kernel,
                Some(stride),
                Some(padding),
                None,
                None,
                Some(true),
            )?;
            convs.push(conv);
        }

        // Final 1x1 conv to single channel
        let conv_post =
            WeightNormConv1d::new(1024, 1, 3, Some(1), Some(1), None, None, Some(true))?;

        Ok(Self {
            period,
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
        let batch = audio.dim(0);
        let samples = audio.dim(2);

        // Pad to make divisible by period
        let remainder = samples % self.period;
        let x = if remainder != 0 {
            let pad_size = self.period - remainder;
            let padding = mlx_rs::ops::zeros::<f32>(&[batch, 1, pad_size])?;
            mlx_rs::ops::concatenate_axis(&[audio, &padding], -1)?
        } else {
            audio.clone()
        };

        // Reshape: [B, 1, T] -> [B, period, T/period]
        // Then treat period as channels for 1D conv
        let new_length = x.dim(2);
        let x = x.reshape(&[batch, self.period, new_length / self.period])?;

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
    fn test_period_discriminator() {
        let disc = PeriodDiscriminator::new(2).unwrap();

        let audio = mlx_rs::random::normal::<f32>(&[1, 1, 1024], None, None, None).unwrap();
        let output = disc.forward(&audio).unwrap();
        output.logits.eval().unwrap();

        assert!(!output.features.is_empty());
    }

    #[test]
    fn test_mpd() {
        let mpd = MultiPeriodDiscriminator::new().unwrap();

        let audio = mlx_rs::random::normal::<f32>(&[2, 1, 2048], None, None, None).unwrap();
        let outputs = mpd.forward(&audio).unwrap();

        assert_eq!(outputs.len(), 5); // 5 periods
    }

    #[test]
    fn test_mpd_different_periods() {
        let mpd = MultiPeriodDiscriminator::with_periods(vec![2, 5, 11]).unwrap();
        assert_eq!(mpd.discriminators.len(), 3);
    }
}
