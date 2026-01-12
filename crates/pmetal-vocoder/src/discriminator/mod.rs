//! Discriminators for BigVGAN adversarial training.
//!
//! BigVGAN uses multiple discriminators to capture different aspects of audio:
//! - Multi-Period Discriminator (MPD): Captures periodic patterns
//! - Multi-Resolution Discriminator (MRD): Multi-scale spectral analysis
//! - Multi-Band Discriminator (MBD): Frequency band analysis

mod mpd;
mod mrd;

pub use mpd::{MultiPeriodDiscriminator, PeriodDiscriminator};
pub use mrd::{MultiResolutionDiscriminator, ResolutionDiscriminator};

use crate::error::Result;
use mlx_rs::Array;

/// Combined discriminator for BigVGAN training.
///
/// Uses both MPD and MRD for comprehensive audio quality assessment.
#[derive(Debug)]
pub struct BigVGANDiscriminator {
    /// Multi-Period Discriminator.
    pub mpd: MultiPeriodDiscriminator,
    /// Multi-Resolution Discriminator.
    pub mrd: MultiResolutionDiscriminator,
}

impl BigVGANDiscriminator {
    /// Create a new combined discriminator.
    pub fn new() -> Result<Self> {
        Ok(Self {
            mpd: MultiPeriodDiscriminator::new()?,
            mrd: MultiResolutionDiscriminator::new()?,
        })
    }

    /// Forward pass returning all discriminator outputs.
    ///
    /// # Arguments
    /// * `audio` - Audio waveform [batch, 1, samples]
    ///
    /// # Returns
    /// Tuple of (mpd_outputs, mrd_outputs) where each is a list of
    /// (logits, feature_maps) for each sub-discriminator
    pub fn forward(
        &self,
        audio: &Array,
    ) -> Result<(Vec<DiscriminatorOutput>, Vec<DiscriminatorOutput>)> {
        let mpd_outputs = self.mpd.forward(audio)?;
        let mrd_outputs = self.mrd.forward(audio)?;
        Ok((mpd_outputs, mrd_outputs))
    }
}

impl Default for BigVGANDiscriminator {
    fn default() -> Self {
        Self::new().expect("Failed to create discriminator")
    }
}

/// Output from a single discriminator.
#[derive(Debug)]
pub struct DiscriminatorOutput {
    /// Final logits (real/fake prediction).
    pub logits: Array,
    /// Intermediate feature maps for feature matching loss.
    pub features: Vec<Array>,
}
