//! Adversarial training of BigVGAN.
//!
//! [`GanTrainer`] runs the recipe of the reference training script: per step
//! the discriminators take a least-squares GAN step on real audio and the
//! generator's detached output, then the generator takes a step on a mel
//! reconstruction loss plus (after `freeze_step` warm-up steps) the
//! adversarial and feature matching losses scored by the *updated*
//! discriminators. Both networks use AdamW with an exponential learning-rate
//! decay applied per step, and gradients are clipped per network (the
//! discriminator banks separately).
//!
//! A step takes a mel spectrogram `[batch, n_mels, frames]` and the audio it
//! came from `[batch, 1, frames * hop_size]`.

use crate::audio::MelFrontEnd;
use crate::config::BigVGANConfig;
use crate::discriminator::{BigVGANDiscriminator, Discriminator};
use crate::error::{Result, VocoderError};
use crate::generator::BigVGAN;
use crate::loss::{
    GeneratorLossConfig, MultiScaleMelLoss, discriminator_loss, generator_loss, mel_l1_loss,
};
use pmetal_runtime::compat::optimizers::{AdamW, AdamWBuilder, Optimizer};
use pmetal_runtime::compat::{Array, Exception, FlattenedModuleParam, nn, ops};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

/// Training hyper-parameters; the field names are the reference
/// `config.json`'s, so a release's file deserialises as is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Initial learning rate of both networks.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,

    /// Adam's beta1.
    #[serde(default = "default_adam_b1")]
    pub adam_b1: f32,

    /// Adam's beta2.
    #[serde(default = "default_adam_b2")]
    pub adam_b2: f32,

    /// Decoupled weight decay. The reference leaves AdamW's default (0.01);
    /// biases are exempt here, as everywhere in this workspace.
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,

    /// The learning rate is multiplied by this after every step.
    #[serde(default = "default_lr_decay")]
    pub lr_decay: f64,

    /// Gradient clipping threshold (global norm per network); the reference
    /// defaults to 1000 when the config does not set it.
    #[serde(default = "default_clip_grad_norm")]
    pub clip_grad_norm: f32,

    /// Whether to use BigVGAN-v2's multi-scale mel loss on waveforms (otherwise
    /// the single-scale L1 mel loss of v1).
    #[serde(default)]
    pub use_multiscale_melloss: bool,

    /// Weight of the mel loss; the reference's default is 45.
    #[serde(default = "default_lambda_melloss")]
    pub lambda_melloss: f32,

    /// Steps during which the discriminators are frozen and the generator
    /// trains on the mel loss alone (a regression warm-up).
    #[serde(default)]
    pub freeze_step: u64,
}

fn default_learning_rate() -> f32 {
    1e-4
}
fn default_adam_b1() -> f32 {
    0.8
}
fn default_adam_b2() -> f32 {
    0.99
}
fn default_weight_decay() -> f32 {
    0.01
}
fn default_lr_decay() -> f64 {
    0.999_999_6
}
fn default_clip_grad_norm() -> f32 {
    1000.0
}
fn default_lambda_melloss() -> f32 {
    45.0
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            adam_b1: default_adam_b1(),
            adam_b2: default_adam_b2(),
            weight_decay: default_weight_decay(),
            lr_decay: default_lr_decay(),
            clip_grad_norm: default_clip_grad_norm(),
            use_multiscale_melloss: false,
            lambda_melloss: default_lambda_melloss(),
            freeze_step: 0,
        }
    }
}

impl TrainingConfig {
    /// Parse the training settings from a reference `config.json`.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| VocoderError::Config(e.to_string()))
    }

    /// The learning rate used at step `step` (0-based): the initial rate
    /// decayed once per completed step.
    pub fn learning_rate_at(&self, step: u64) -> f32 {
        (f64::from(self.learning_rate) * self.lr_decay.powf(step as f64)) as f32
    }
}

/// The mel reconstruction objective.
#[derive(Debug)]
enum MelObjective {
    /// v2: multi-scale L1 on log mels computed from the waveforms.
    MultiScale(MultiScaleMelLoss),
    /// v1: L1 between the log mels of real and generated audio.
    SingleScale(MelFrontEnd),
}

impl MelObjective {
    fn loss(&self, generated: &Array, real: &Array) -> Result<Array> {
        match self {
            Self::MultiScale(m) => m.forward(generated, real),
            Self::SingleScale(front_end) => {
                let flat = |a: &Array| a.reshape(&[a.dim(0), a.dim(2)]);
                mel_l1_loss(
                    &front_end.forward(&flat(real))?,
                    &front_end.forward(&flat(generated))?,
                )
            }
        }
    }
}

/// What one training step measured.
#[derive(Debug, Clone, PartialEq)]
pub struct StepMetrics {
    /// The 0-based index of the step that just ran.
    pub step: u64,
    /// Weighted generator loss (the quantity that was differentiated).
    pub generator_loss: f32,
    /// The unweighted mel reconstruction loss of the generator's output
    /// before this step's update.
    pub mel_error: f32,
    /// Discriminator loss, or `None` during the frozen warm-up.
    pub discriminator_loss: Option<f32>,
    /// Gradient norm of the generator before clipping.
    pub grad_norm_generator: f32,
    /// Gradient norms of the period and spectral discriminator banks before
    /// clipping, or `None` during the frozen warm-up.
    pub grad_norm_discriminator: Option<(f32, f32)>,
    /// The learning rate this step used.
    pub learning_rate: f32,
}

/// Convert a vocoder error into the autograd callback's error type.
fn callback_error(e: VocoderError) -> Exception {
    Exception::custom(e.to_string())
}

/// Scale the gradients whose names pass `select` so their global L2 norm is at
/// most `max_norm` (`torch.nn.utils.clip_grad_norm_`); returns the norm before
/// clipping.
fn clip_grad_norm(
    grads: &mut FlattenedModuleParam,
    max_norm: f32,
    select: impl Fn(&str) -> bool,
) -> Result<f32> {
    let keys: Vec<Rc<str>> = grads.keys().filter(|k| select(k)).cloned().collect();
    if keys.is_empty() {
        return Ok(0.0);
    }
    let total_sq = keys
        .iter()
        .map(|k| {
            let g = &grads[k];
            g.multiply(g).sum_all()
        })
        .reduce(|a, b| a.add(&b))
        .expect("keys is not empty");
    let norm = total_sq.sqrt().item_f32();
    if !norm.is_finite() {
        return Err(VocoderError::Mlx(Exception::custom(format!(
            "the gradient norm is {norm}; a non-finite gradient would corrupt the weights"
        ))));
    }
    let coefficient = (max_norm / (norm + 1e-6)).min(1.0);
    if coefficient < 1.0 {
        let c = Array::from_f32(coefficient);
        for k in keys {
            let scaled = grads[&k].multiply(&c);
            grads.insert(k, scaled);
        }
    }
    Ok(norm)
}

/// The adversarial training loop for a generator and its discriminators.
#[derive(Debug)]
pub struct GanTrainer {
    /// The generator being trained.
    pub generator: BigVGAN,
    /// The discriminators it trains against.
    pub discriminator: BigVGANDiscriminator,
    config: TrainingConfig,
    loss_weights: GeneratorLossConfig,
    mel: MelObjective,
    opt_g: AdamW,
    opt_d: AdamW,
    steps: u64,
}

impl GanTrainer {
    /// A trainer for `generator` against `discriminator`.
    ///
    /// The generator is switched to its trainable form (weight normalisation
    /// re-imposed, activations on the differentiable op path); a model
    /// prepared for inference would otherwise not train.
    pub fn new(
        mut generator: BigVGAN,
        discriminator: BigVGANDiscriminator,
        config: TrainingConfig,
    ) -> Result<Self> {
        generator.prepare_for_training();
        let model: &BigVGANConfig = &generator.config;
        let mel = if config.use_multiscale_melloss {
            MelObjective::MultiScale(MultiScaleMelLoss::bigvgan_v2(model.sampling_rate)?)
        } else {
            MelObjective::SingleScale(MelFrontEnd::for_loss(model)?)
        };
        let optimizer = |lr: f32| {
            AdamWBuilder::new(lr)
                .weight_decay(config.weight_decay)
                .betas((config.adam_b1, config.adam_b2))
                .build()
                .map_err(|e| VocoderError::Config(e.to_string()))
        };
        let loss_weights = GeneratorLossConfig {
            lambda_mel: config.lambda_melloss,
            ..GeneratorLossConfig::v1()
        };
        Ok(Self {
            opt_g: optimizer(config.learning_rate)?,
            opt_d: optimizer(config.learning_rate)?,
            generator,
            discriminator,
            config,
            loss_weights,
            mel,
            steps: 0,
        })
    }

    /// Steps completed so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The training settings.
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Run one training step on `mel` (`[batch, n_mels, frames]`) and the
    /// matching audio `wav` (`[batch, 1, frames * hop_size]`).
    pub fn step(&mut self, mel: &Array, wav: &Array) -> Result<StepMetrics> {
        let hop = self.generator.config.hop_size;
        if mel.ndim() != 3 || wav.ndim() != 3 || wav.dim(1) != 1 || wav.dim(2) != mel.dim(2) * hop {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![mel.dim(0), 1, mel.dim(2) * hop],
                actual: wav.shape().to_vec(),
            });
        }
        let step = self.steps;
        let frozen = step < self.config.freeze_step;
        let lr = self.config.learning_rate_at(step);
        self.opt_g.set_lr(lr);
        self.opt_d.set_lr(lr);

        // The generator's current output, detached: the discriminator step's
        // fake input, and what the mel error is read from.
        let generated = ops::stop_gradient(&self.generator.forward(mel)?);
        let mel_error = self.mel.loss(&generated, wav)?.item_f32();

        // --- discriminator step -------------------------------------------
        let mut discriminator_loss_value = None;
        let mut grad_norm_discriminator = None;
        if !frozen {
            let mut value_and_grad = nn::value_and_grad(
                |d: &mut BigVGANDiscriminator, (real, fake): (&Array, &Array)| {
                    let (real_mpd, real_spectral) = d.forward(real).map_err(callback_error)?;
                    let (fake_mpd, fake_spectral) = d.forward(fake).map_err(callback_error)?;
                    let mpd = discriminator_loss(&real_mpd, &fake_mpd).map_err(callback_error)?;
                    let spectral = discriminator_loss(&real_spectral, &fake_spectral)
                        .map_err(callback_error)?;
                    Ok(mpd.total.add(&spectral.total))
                },
            );
            let (loss, mut grads) = value_and_grad(&mut self.discriminator, (wav, &generated))?;
            let clip = self.config.clip_grad_norm;
            let mpd_norm = clip_grad_norm(&mut grads, clip, |k| k.starts_with("mpd."))?;
            let spectral_norm = clip_grad_norm(&mut grads, clip, |k| k.starts_with("spectral."))?;
            self.opt_d.update(&mut self.discriminator, grads)?;
            discriminator_loss_value = Some(loss.item_f32());
            grad_norm_discriminator = Some((mpd_norm, spectral_norm));
        }

        // --- generator step ------------------------------------------------
        // The real audio's scores come from the discriminators as just updated.
        let real_outputs = if frozen {
            None
        } else {
            let (mpd, spectral) = self.discriminator.forward(wav)?;
            for out in mpd.iter().chain(&spectral) {
                out.logits.eval();
                out.features.iter().for_each(|f| {
                    f.eval();
                });
            }
            Some((mpd, spectral))
        };
        let discriminator = &self.discriminator;
        let mel_objective = &self.mel;
        let weights = &self.loss_weights;
        let real_ref = real_outputs.as_ref();
        let mut value_and_grad =
            nn::value_and_grad(|g: &mut BigVGAN, (mel, wav): (&Array, &Array)| {
                let fake_audio = g.forward(mel).map_err(callback_error)?;
                let mel_loss = mel_objective
                    .loss(&fake_audio, wav)
                    .map_err(callback_error)?;
                let Some((real_mpd, real_spectral)) = real_ref else {
                    return Ok(mel_loss.multiply(&Array::from_f32(weights.lambda_mel)));
                };
                let (fake_mpd, fake_spectral) =
                    discriminator.forward(&fake_audio).map_err(callback_error)?;
                generator_loss(
                    &[real_mpd.as_slice(), real_spectral.as_slice()],
                    &[fake_mpd.as_slice(), fake_spectral.as_slice()],
                    mel_loss,
                    weights,
                )
                .map(|l| l.total)
                .map_err(callback_error)
            });
        let (loss, mut grads) = value_and_grad(&mut self.generator, (mel, wav))?;
        let grad_norm_generator = clip_grad_norm(&mut grads, self.config.clip_grad_norm, |_| true)?;
        self.opt_g.update(&mut self.generator, grads)?;
        let generator_loss_value = loss.item_f32();

        pmetal_runtime::check_unobserved_error()
            .map_err(|e| VocoderError::Mlx(Exception::custom(e.to_string())))?;
        self.steps += 1;
        Ok(StepMetrics {
            step,
            generator_loss: generator_loss_value,
            mel_error,
            discriminator_loss: discriminator_loss_value,
            grad_norm_generator,
            grad_norm_discriminator,
            learning_rate: lr,
        })
    }

    /// Save everything needed to resume: the generator and discriminators in
    /// the reference's tensor names and layout (`generator.*`, `mpd.*`,
    /// `mrd.*`), both optimizers' moments and the step count, as one
    /// safetensors file.
    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        let mut entries: Vec<(String, Array)> = Vec::new();
        for (name, tensor) in self.generator.state_dict() {
            entries.push((format!("generator.{name}"), tensor));
        }
        self.discriminator.mpd.save_state("mpd", &mut entries);
        self.discriminator.spectral.save_state("mrd", &mut entries);
        for (tag, optimizer) in [("optim_g", &self.opt_g), ("optim_d", &self.opt_d)] {
            for (key, (m, v)) in &optimizer.state {
                entries.push((format!("{tag}.m.{key}"), m.clone()));
                entries.push((format!("{tag}.v.{key}"), v.clone()));
            }
            entries.push((format!("{tag}.step"), counter_array(optimizer.step_count())));
        }
        entries.push(("steps".to_owned(), counter_array(self.steps)));

        let borrowed: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let path = path
            .to_str()
            .ok_or_else(|| VocoderError::WeightLoad("non-UTF-8 path".into()))?;
        Array::save_safetensors(path, &borrowed)
            .map_err(|e| VocoderError::WeightLoad(e.to_string()))
    }

    /// Restore a checkpoint written by [`save_checkpoint`](Self::save_checkpoint):
    /// the networks, the optimizers' state and the step count, so training
    /// continues exactly where it stopped.
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let mut sd = crate::checkpoint::load_state_dict(path, None)?;
        let steps = counter_value(&sd.take("steps")?);

        self.generator
            .load_state_dict(sd.split_prefix("generator"))?;
        let mut mpd = sd.split_prefix("mpd");
        self.discriminator.mpd.load_state(&mut mpd, "")?;
        mpd.finish()?;
        let mut mrd = sd.split_prefix("mrd");
        self.discriminator.spectral.load_state(&mut mrd, "")?;
        mrd.finish()?;

        for (tag, optimizer) in [("optim_g", &mut self.opt_g), ("optim_d", &mut self.opt_d)] {
            let step_count = counter_value(&sd.take(&format!("{tag}.step"))?);
            let moments_m = sd.split_prefix(&format!("{tag}.m"));
            let mut moments_v = sd.split_prefix(&format!("{tag}.v"));
            let mut state: HashMap<Rc<str>, (Array, Array)> = HashMap::new();
            for key in moments_m.keys().cloned().collect::<Vec<_>>() {
                let m = moments_m
                    .get(&key)
                    .cloned()
                    .expect("the key was just listed");
                let v = moments_v.take(&key)?;
                state.insert(Rc::from(key.as_str()), (m, v));
            }
            optimizer.restore_state(step_count, state);
        }
        // Everything must have found a home.
        sd.finish()?;
        self.steps = steps;
        // The weights changed: refresh the activations' view of them.
        self.generator.prepare_for_training();
        Ok(())
    }
}

/// A counter as an array of three 24-bit limbs held in float32 (each exactly
/// representable), the one element type checkpoints here store.
fn counter_array(v: u64) -> Array {
    const MASK: u64 = (1 << 24) - 1;
    let limbs = [(v >> 48) & MASK, (v >> 24) & MASK, v & MASK].map(|l| l as f32);
    Array::from_f32_slice(&limbs, &[3])
}

fn counter_value(a: &Array) -> u64 {
    a.as_slice::<f32>()
        .iter()
        .fold(0u64, |acc, &limb| (acc << 24) | limb as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grads(values: &[(&str, &[f32])]) -> FlattenedModuleParam {
        values
            .iter()
            .map(|(k, v)| (Rc::from(*k), Array::from_f32_slice(v, &[v.len() as i32])))
            .collect()
    }

    fn values(g: &FlattenedModuleParam, key: &str) -> Vec<f32> {
        g[key].as_slice::<f32>().to_vec()
    }

    #[test]
    fn clipping_scales_to_the_threshold_and_reports_the_original_norm() {
        // The global norm of (3, 4) is 5.
        let mut g = grads(&[("a.x", &[3.0]), ("a.y", &[4.0])]);
        let norm = clip_grad_norm(&mut g, 1.0, |_| true).unwrap();
        assert!((norm - 5.0).abs() < 1e-5);
        assert!((values(&g, "a.x")[0] - 0.6).abs() < 1e-4);
        assert!((values(&g, "a.y")[0] - 0.8).abs() < 1e-4);
    }

    #[test]
    fn a_gradient_under_the_threshold_is_left_alone() {
        let mut g = grads(&[("a.x", &[0.3, 0.4])]);
        let norm = clip_grad_norm(&mut g, 10.0, |_| true).unwrap();
        assert!((norm - 0.5).abs() < 1e-6);
        assert_eq!(values(&g, "a.x"), [0.3, 0.4]);
    }

    #[test]
    fn only_the_selected_gradients_are_measured_and_scaled() {
        let mut g = grads(&[("mpd.w", &[6.0, 8.0]), ("spectral.w", &[30.0, 40.0])]);
        let norm = clip_grad_norm(&mut g, 1.0, |k| k.starts_with("mpd.")).unwrap();
        assert!((norm - 10.0).abs() < 1e-4);
        assert!((values(&g, "mpd.w")[0] - 0.6).abs() < 1e-4);
        assert_eq!(values(&g, "spectral.w"), [30.0, 40.0]);
        assert_eq!(
            clip_grad_norm(&mut g, 1.0, |k| k.starts_with("none.")).unwrap(),
            0.0
        );
    }

    #[test]
    fn a_non_finite_gradient_is_an_error_not_a_corrupted_weight() {
        let mut g = grads(&[("a.x", &[1.0, f32::NAN])]);
        assert!(clip_grad_norm(&mut g, 1.0, |_| true).is_err());
        let mut g = grads(&[("a.x", &[f32::INFINITY])]);
        assert!(clip_grad_norm(&mut g, 1.0, |_| true).is_err());
    }

    #[test]
    fn step_counters_round_trip_through_float_limbs() {
        for v in [
            0,
            1,
            1_000_000,
            (1 << 24) + 1,
            3_000_000_000,
            u64::MAX >> 16,
        ] {
            assert_eq!(counter_value(&counter_array(v)), v);
        }
    }
}
