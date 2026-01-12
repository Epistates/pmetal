//! Loss functions for BigVGAN training.
//!
//! BigVGAN uses a combination of losses:
//! - Adversarial loss (generator and discriminator)
//! - Feature matching loss
//! - Mel spectrogram reconstruction loss

use crate::audio::{mel_spectrogram, MelConfig, StftConfig};
use crate::discriminator::DiscriminatorOutput;
use crate::error::Result;
use mlx_rs::Array;

/// Generator adversarial loss.
///
/// L_adv(G) = E[(1 - D(G(z)))²]
///
/// Encourages generator to produce outputs that fool discriminators.
pub fn generator_adversarial_loss(disc_outputs: &[DiscriminatorOutput]) -> Result<Array> {
    let mut total_loss = Array::from_f32(0.0);

    for output in disc_outputs {
        // MSE loss: (1 - logits)²
        let one = Array::from_f32(1.0);
        let diff = one.subtract(&output.logits)?;
        let squared = diff.multiply(&diff)?;
        let loss = squared.mean(None)?;
        total_loss = total_loss.add(&loss)?;
    }

    Ok(total_loss)
}

/// Discriminator adversarial loss.
///
/// L_adv(D) = E[(1 - D(x))² + D(G(z))²]
///
/// Trains discriminator to distinguish real from fake.
pub fn discriminator_adversarial_loss(
    real_outputs: &[DiscriminatorOutput],
    fake_outputs: &[DiscriminatorOutput],
) -> Result<(Array, Array)> {
    let mut real_loss = Array::from_f32(0.0);
    let mut fake_loss = Array::from_f32(0.0);

    for (real, fake) in real_outputs.iter().zip(fake_outputs.iter()) {
        // Real loss: (1 - D(x))²
        let one = Array::from_f32(1.0);
        let real_diff = one.subtract(&real.logits)?;
        let real_sq = real_diff.multiply(&real_diff)?;
        real_loss = real_loss.add(&real_sq.mean(None)?)?;

        // Fake loss: D(G(z))²
        let fake_sq = fake.logits.multiply(&fake.logits)?;
        fake_loss = fake_loss.add(&fake_sq.mean(None)?)?;
    }

    Ok((real_loss, fake_loss))
}

/// Feature matching loss.
///
/// L_fm = E[|D_i(x) - D_i(G(z))|]
///
/// Matches intermediate feature representations between real and fake.
pub fn feature_matching_loss(
    real_outputs: &[DiscriminatorOutput],
    fake_outputs: &[DiscriminatorOutput],
) -> Result<Array> {
    let mut total_loss = Array::from_f32(0.0);
    let mut num_features = 0;

    for (real, fake) in real_outputs.iter().zip(fake_outputs.iter()) {
        for (real_feat, fake_feat) in real.features.iter().zip(fake.features.iter()) {
            // L1 loss on features
            let diff = real_feat.subtract(fake_feat)?;
            let abs_diff = diff.abs()?;
            let loss = abs_diff.mean(None)?;
            total_loss = total_loss.add(&loss)?;
            num_features += 1;
        }
    }

    // Average over all features
    if num_features > 0 {
        let num = Array::from_int(num_features);
        total_loss = total_loss.divide(&num)?;
    }

    Ok(total_loss)
}

/// Mel spectrogram reconstruction loss.
///
/// L_mel = |M(x) - M(G(z))|
///
/// Ensures generated audio has similar spectral content to target.
pub fn mel_reconstruction_loss(
    real_audio: &Array,
    fake_audio: &Array,
    mel_config: &MelConfig,
    stft_config: &StftConfig,
) -> Result<Array> {
    // Remove channel dimension if present: [B, 1, T] -> [B, T]
    let real = if real_audio.ndim() == 3 {
        real_audio.squeeze()?
    } else {
        real_audio.clone()
    };

    let fake = if fake_audio.ndim() == 3 {
        fake_audio.squeeze()?
    } else {
        fake_audio.clone()
    };

    // Compute mel spectrograms
    let real_mel = mel_spectrogram(&real, mel_config, stft_config)?;
    let fake_mel = mel_spectrogram(&fake, mel_config, stft_config)?;

    // L1 loss
    let diff = real_mel.subtract(&fake_mel)?;
    let abs_diff = diff.abs()?;

    Ok(abs_diff.mean(None)?)
}

/// Multi-scale mel spectrogram loss.
///
/// Computes mel loss at multiple STFT resolutions for better frequency coverage.
pub fn multi_scale_mel_loss(
    real_audio: &Array,
    fake_audio: &Array,
    mel_config: &MelConfig,
) -> Result<Array> {
    let scales = vec![
        (512, 128),   // (n_fft, hop_length)
        (1024, 256),
        (2048, 512),
    ];

    let mut total_loss = Array::from_f32(0.0);

    for (n_fft, hop_length) in scales {
        let stft_config = StftConfig {
            n_fft,
            hop_length,
            win_length: Some(n_fft),
            center: true,
            ..Default::default()
        };

        let loss = mel_reconstruction_loss(real_audio, fake_audio, mel_config, &stft_config)?;
        total_loss = total_loss.add(&loss)?;
    }

    // Average over scales
    let num_scales = Array::from_f32(3.0);
    Ok(total_loss.divide(&num_scales)?)
}

/// Combined generator loss.
///
/// L_G = λ_adv * L_adv + λ_fm * L_fm + λ_mel * L_mel
#[derive(Debug, Clone)]
pub struct GeneratorLossConfig {
    /// Weight for adversarial loss.
    pub lambda_adv: f32,
    /// Weight for feature matching loss.
    pub lambda_fm: f32,
    /// Weight for mel loss.
    pub lambda_mel: f32,
}

impl Default for GeneratorLossConfig {
    fn default() -> Self {
        Self {
            lambda_adv: 1.0,
            lambda_fm: 2.0,
            lambda_mel: 45.0,
        }
    }
}

/// Compute combined generator loss.
pub fn generator_loss(
    real_audio: &Array,
    fake_audio: &Array,
    real_outputs_mpd: &[DiscriminatorOutput],
    fake_outputs_mpd: &[DiscriminatorOutput],
    real_outputs_mrd: &[DiscriminatorOutput],
    fake_outputs_mrd: &[DiscriminatorOutput],
    mel_config: &MelConfig,
    config: &GeneratorLossConfig,
) -> Result<GeneratorLossOutput> {
    // Adversarial losses
    let adv_loss_mpd = generator_adversarial_loss(fake_outputs_mpd)?;
    let adv_loss_mrd = generator_adversarial_loss(fake_outputs_mrd)?;
    let adv_loss = adv_loss_mpd.add(&adv_loss_mrd)?;

    // Feature matching losses
    let fm_loss_mpd = feature_matching_loss(real_outputs_mpd, fake_outputs_mpd)?;
    let fm_loss_mrd = feature_matching_loss(real_outputs_mrd, fake_outputs_mrd)?;
    let fm_loss = fm_loss_mpd.add(&fm_loss_mrd)?;

    // Mel loss
    let mel_loss = multi_scale_mel_loss(real_audio, fake_audio, mel_config)?;

    // Combine with weights
    let total = adv_loss
        .multiply(&Array::from_f32(config.lambda_adv))?
        .add(&fm_loss.multiply(&Array::from_f32(config.lambda_fm))?)?
        .add(&mel_loss.multiply(&Array::from_f32(config.lambda_mel))?)?;

    Ok(GeneratorLossOutput {
        total,
        adversarial: adv_loss,
        feature_matching: fm_loss,
        mel: mel_loss,
    })
}

/// Output from generator loss computation.
#[derive(Debug)]
pub struct GeneratorLossOutput {
    /// Total weighted loss.
    pub total: Array,
    /// Adversarial loss component.
    pub adversarial: Array,
    /// Feature matching loss component.
    pub feature_matching: Array,
    /// Mel reconstruction loss component.
    pub mel: Array,
}

/// Compute discriminator loss.
pub fn discriminator_loss(
    real_outputs: &[DiscriminatorOutput],
    fake_outputs: &[DiscriminatorOutput],
) -> Result<DiscriminatorLossOutput> {
    let (real_loss, fake_loss) = discriminator_adversarial_loss(real_outputs, fake_outputs)?;
    let total = real_loss.add(&fake_loss)?;

    Ok(DiscriminatorLossOutput {
        total,
        real: real_loss,
        fake: fake_loss,
    })
}

/// Output from discriminator loss computation.
#[derive(Debug)]
pub struct DiscriminatorLossOutput {
    /// Total loss.
    pub total: Array,
    /// Loss on real samples.
    pub real: Array,
    /// Loss on fake samples.
    pub fake: Array,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_outputs(batch: i32) -> Vec<DiscriminatorOutput> {
        vec![DiscriminatorOutput {
            logits: mlx_rs::random::normal::<f32>(&[batch, 1, 10], None, None, None).unwrap(),
            features: vec![
                mlx_rs::random::normal::<f32>(&[batch, 32, 64], None, None, None).unwrap(),
                mlx_rs::random::normal::<f32>(&[batch, 64, 32], None, None, None).unwrap(),
            ],
        }]
    }

    #[test]
    fn test_generator_adversarial_loss() {
        let outputs = create_dummy_outputs(2);
        let loss = generator_adversarial_loss(&outputs).unwrap();
        loss.eval().unwrap();

        assert_eq!(loss.ndim(), 0); // scalar
    }

    #[test]
    fn test_discriminator_adversarial_loss() {
        let real = create_dummy_outputs(2);
        let fake = create_dummy_outputs(2);
        let (real_loss, fake_loss) = discriminator_adversarial_loss(&real, &fake).unwrap();
        real_loss.eval().unwrap();
        fake_loss.eval().unwrap();

        assert_eq!(real_loss.ndim(), 0);
        assert_eq!(fake_loss.ndim(), 0);
    }

    #[test]
    fn test_feature_matching_loss() {
        let real = create_dummy_outputs(2);
        let fake = create_dummy_outputs(2);
        let loss = feature_matching_loss(&real, &fake).unwrap();
        loss.eval().unwrap();

        assert_eq!(loss.ndim(), 0);
    }

    #[test]
    fn test_mel_reconstruction_loss() {
        let mel_config = MelConfig::default();
        let stft_config = StftConfig::default();

        let real = mlx_rs::random::normal::<f32>(&[1, 8000], None, None, None).unwrap();
        let fake = mlx_rs::random::normal::<f32>(&[1, 8000], None, None, None).unwrap();

        let loss = mel_reconstruction_loss(&real, &fake, &mel_config, &stft_config).unwrap();
        loss.eval().unwrap();

        assert_eq!(loss.ndim(), 0);
    }

    #[test]
    fn test_loss_config_default() {
        let config = GeneratorLossConfig::default();
        assert_eq!(config.lambda_adv, 1.0);
        assert_eq!(config.lambda_fm, 2.0);
        assert_eq!(config.lambda_mel, 45.0);
    }
}
