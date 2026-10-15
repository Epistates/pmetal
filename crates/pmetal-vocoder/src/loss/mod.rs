//! Loss functions for BigVGAN training.
//!
//! BigVGAN is trained with
//! - least-squares GAN losses (Mao et al., 2017) over every sub-discriminator,
//! - a feature matching loss (Kumar et al., 2019): the L1 distance between the
//!   discriminators' intermediate activations on real and generated audio,
//! - a mel-spectrogram L1 reconstruction loss: a single mel scale in v1, and
//!   in v2 a multi-scale one computed straight from the waveforms.
//!
//! The reductions follow the reference training script: per-discriminator
//! terms are *summed* (not averaged) over discriminators and layers.

mod mel;

pub use mel::{MultiScaleMelLoss, MultiScaleMelLossConfig, mel_l1_loss};

use crate::discriminator::DiscriminatorOutput;
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::Array;

/// Generator adversarial loss, `sum_d mean((1 - D_d(G(z)))^2)`: pushes the
/// generator toward outputs the discriminators score as real.
pub fn generator_adversarial_loss(disc_outputs: &[DiscriminatorOutput]) -> Result<Array> {
    let one = Array::from_f32(1.0);
    sum_arrays(disc_outputs.iter().map(|o| {
        let diff = one.subtract(&o.logits);
        diff.multiply(&diff).mean_all()
    }))
}

/// Discriminator adversarial loss, returned as `(real, fake)` where
/// `real = sum_d mean((1 - D_d(x))^2)` and `fake = sum_d mean(D_d(G(z))^2)`.
pub fn discriminator_adversarial_loss(
    real_outputs: &[DiscriminatorOutput],
    fake_outputs: &[DiscriminatorOutput],
) -> Result<(Array, Array)> {
    check_pairs(real_outputs.len(), fake_outputs.len())?;
    let one = Array::from_f32(1.0);
    let real = sum_arrays(real_outputs.iter().map(|o| {
        let diff = one.subtract(&o.logits);
        diff.multiply(&diff).mean_all()
    }))?;
    let fake = sum_arrays(
        fake_outputs
            .iter()
            .map(|o| o.logits.multiply(&o.logits).mean_all()),
    )?;
    Ok((real, fake))
}

/// Feature matching loss, `sum_d sum_l mean(|f_dl(x) - f_dl(G(z))|)`.
///
/// This is the unweighted sum; the reference multiplies it by 2, which is
/// [`GeneratorLossConfig::lambda_fm`]'s default.
pub fn feature_matching_loss(
    real_outputs: &[DiscriminatorOutput],
    fake_outputs: &[DiscriminatorOutput],
) -> Result<Array> {
    check_pairs(real_outputs.len(), fake_outputs.len())?;
    let mut terms = Vec::new();
    for (real, fake) in real_outputs.iter().zip(fake_outputs) {
        if real.features.len() != fake.features.len() {
            return Err(VocoderError::Config(format!(
                "real and generated outputs have {} and {} feature maps",
                real.features.len(),
                fake.features.len()
            )));
        }
        for (r, f) in real.features.iter().zip(&fake.features) {
            terms.push(r.subtract(f).abs_val().mean_all());
        }
    }
    sum_arrays(terms.into_iter())
}

fn check_pairs(real: usize, fake: usize) -> Result<()> {
    if real != fake || real == 0 {
        return Err(VocoderError::Config(format!(
            "need matching, non-empty discriminator outputs (got {real} real, {fake} generated)"
        )));
    }
    Ok(())
}

fn sum_arrays(terms: impl Iterator<Item = Array>) -> Result<Array> {
    terms
        .reduce(|a, b| a.add(&b))
        .ok_or_else(|| VocoderError::Config("no discriminator outputs to reduce".into()))
}

/// Weights of the combined generator loss,
/// `L_G = lambda_adv L_adv + lambda_fm L_fm + lambda_mel L_mel`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorLossConfig {
    /// Weight for the adversarial loss.
    pub lambda_adv: f32,
    /// Weight for the feature matching loss.
    pub lambda_fm: f32,
    /// Weight for the mel loss.
    pub lambda_mel: f32,
}

impl GeneratorLossConfig {
    /// BigVGAN v1: single-scale mel loss weighted 45.
    pub fn v1() -> Self {
        Self {
            lambda_adv: 1.0,
            lambda_fm: 2.0,
            lambda_mel: 45.0,
        }
    }

    /// BigVGAN v2: multi-scale mel loss weighted 15.
    pub fn v2() -> Self {
        Self {
            lambda_mel: 15.0,
            ..Self::v1()
        }
    }
}

impl Default for GeneratorLossConfig {
    fn default() -> Self {
        Self::v2()
    }
}

/// The components of the combined generator loss.
#[derive(Debug)]
pub struct GeneratorLossOutput {
    /// Weighted total, the quantity to differentiate.
    pub total: Array,
    /// Adversarial loss, summed over every discriminator bank (unweighted).
    pub adversarial: Array,
    /// Feature matching loss, summed over every bank (unweighted).
    pub feature_matching: Array,
    /// The mel loss that was passed in (unweighted).
    pub mel: Array,
}

/// The combined generator loss over any number of discriminator banks (the
/// MPD and the spectral discriminator, for BigVGAN).
///
/// * `real_banks` / `fake_banks` - each bank's outputs on real and generated audio.
/// * `mel_loss` - the unweighted mel reconstruction loss
///   ([`MultiScaleMelLoss::forward`] or [`mel_l1_loss`]).
pub fn generator_loss(
    real_banks: &[&[DiscriminatorOutput]],
    fake_banks: &[&[DiscriminatorOutput]],
    mel_loss: Array,
    config: &GeneratorLossConfig,
) -> Result<GeneratorLossOutput> {
    check_pairs(real_banks.len(), fake_banks.len())?;
    let adversarial = sum_arrays(
        fake_banks
            .iter()
            .map(|b| generator_adversarial_loss(b))
            .collect::<Result<Vec<_>>>()?
            .into_iter(),
    )?;
    let feature_matching = sum_arrays(
        real_banks
            .iter()
            .zip(fake_banks)
            .map(|(r, f)| feature_matching_loss(r, f))
            .collect::<Result<Vec<_>>>()?
            .into_iter(),
    )?;
    let total = adversarial
        .multiply(&Array::from_f32(config.lambda_adv))
        .add(&feature_matching.multiply(&Array::from_f32(config.lambda_fm)))
        .add(&mel_loss.multiply(&Array::from_f32(config.lambda_mel)));
    Ok(GeneratorLossOutput {
        total,
        adversarial,
        feature_matching,
        mel: mel_loss,
    })
}

/// The components of the discriminator loss.
#[derive(Debug)]
pub struct DiscriminatorLossOutput {
    /// Total loss, `real + fake`.
    pub total: Array,
    /// Loss on real samples.
    pub real: Array,
    /// Loss on generated samples.
    pub fake: Array,
}

/// The discriminator loss for one bank of sub-discriminators.
pub fn discriminator_loss(
    real_outputs: &[DiscriminatorOutput],
    fake_outputs: &[DiscriminatorOutput],
) -> Result<DiscriminatorLossOutput> {
    let (real, fake) = discriminator_adversarial_loss(real_outputs, fake_outputs)?;
    Ok(DiscriminatorLossOutput {
        total: real.add(&fake),
        real,
        fake,
    })
}
