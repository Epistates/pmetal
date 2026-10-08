//! Kahneman-Tversky Optimization (Ethayarajh et al. 2024, arXiv:2402.01306).
//!
//! KTO learns from single completions labelled desirable or undesirable, so
//! it needs no pairs. With `r = log π(y|x) − log π_ref(y|x)` (summed over the
//! completion) and `z` a detached estimate of `KL(π ‖ π_ref)`:
//!
//! ```text
//! desirable:    λ_D · (1 − σ(β·(r − z)))
//! undesirable:  λ_U · (1 − σ(β·(z − r)))
//! ```
//!
//! averaged over the batch. As in the paper and TRL's `KTOTrainer`, `z` is
//! the batch mean of `log π − log π_ref` over *mismatched* completions (each
//! prompt paired with another example's completion), clamped at zero and kept
//! out of the gradient. It is the reference point the prospect-theory value
//! is measured from: without it, KTO only pushes rewards up or down.

use pmetal_bridge::compat::{Array, Exception, nn, ops};
use pmetal_core::{TrainingCallback, TrainingConfig};
use pmetal_lora::TrainableModel;

use super::{
    MicroBatch, PreferenceError, PreferenceResult, PreferenceStepMetrics, Sequence,
    batch_log_probs, f32_array, optimize, sequence_log_probs,
};

/// KTO hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KtoConfig {
    /// β: how sharply the value function saturates. Default 0.1.
    pub beta: f32,
    /// λ_D, the weight of desirable examples. Default 1.0.
    pub desirable_weight: f32,
    /// λ_U, the weight of undesirable examples. Default 1.0.
    pub undesirable_weight: f32,
}

impl Default for KtoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            desirable_weight: 1.0,
            undesirable_weight: 1.0,
        }
    }
}

impl KtoConfig {
    fn validate(&self) -> Result<(), String> {
        if !(self.beta.is_finite() && self.beta > 0.0) {
            return Err(format!("kto: beta must be positive, got {}", self.beta));
        }
        if !(self.desirable_weight >= 0.0 && self.undesirable_weight >= 0.0) {
            return Err("kto: example weights must not be negative".into());
        }
        Ok(())
    }

    /// The KTO loss for a batch.
    ///
    /// `policy` / `reference` are summed log-probs `[B]` of the labelled
    /// completions, `kl` the detached KL estimate (scalar), `desirable` the
    /// labels. Returns the mean loss and the rewards `β·r` `[B]`.
    pub fn loss(
        &self,
        policy: &Array,
        reference: &Array,
        kl: &Array,
        desirable: &[bool],
    ) -> (Array, Array) {
        let mask: Vec<f32> = desirable
            .iter()
            .map(|&d| if d { 1.0 } else { 0.0 })
            .collect();
        let mask = f32_array(&mask);
        let undesirable = Array::from_f32(1.0).subtract(&mask);
        let beta = Array::from_f32(self.beta);
        let ratio = policy.subtract(reference);
        let one = Array::from_f32(1.0);

        let desirable_loss = one
            .subtract(&nn::sigmoid(&ratio.subtract(kl).multiply(&beta)))
            .multiply(&Array::from_f32(self.desirable_weight))
            .multiply(&mask);
        let undesirable_loss = one
            .subtract(&nn::sigmoid(&kl.subtract(&ratio).multiply(&beta)))
            .multiply(&Array::from_f32(self.undesirable_weight))
            .multiply(&undesirable);
        (
            desirable_loss.add(&undesirable_loss).mean(None),
            ratio.multiply(&beta),
        )
    }
}

/// One labelled completion, and the mismatched completion KTO's KL estimate
/// reads for the same prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct KtoSample {
    /// Prompt + completion.
    pub sequence: Sequence,
    /// Prompt + another example's completion.
    pub mismatched: Sequence,
    /// Whether the completion is desirable.
    pub desirable: bool,
}

/// Offline KTO trainer.
pub struct KtoTrainer {
    kto: KtoConfig,
    config: TrainingConfig,
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl KtoTrainer {
    /// Create a trainer; see [`super::PreferenceTrainer::new`] for what
    /// `config` controls.
    pub fn new(kto: KtoConfig, config: TrainingConfig) -> PreferenceResult<Self> {
        kto.validate().map_err(PreferenceError::Config)?;
        Ok(Self {
            kto,
            config,
            callbacks: Vec::new(),
        })
    }

    /// Add a training callback.
    pub fn add_callback(&mut self, callback: Box<dyn TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Train `model` on `samples`. As with the paired trainer, `model` must
    /// still be the reference when this is called.
    pub fn train<M, O>(
        &mut self,
        model: &mut M,
        samples: &[KtoSample],
        optimizer: &mut O,
        set_lr: impl FnMut(&mut O, f32),
    ) -> PreferenceResult<Vec<PreferenceStepMetrics>>
    where
        M: TrainableModel,
        O: pmetal_bridge::compat::optimizers::Optimizer,
    {
        if samples.is_empty() {
            return Err(PreferenceError::Data("no KTO examples".into()));
        }
        let desirable = samples.iter().filter(|s| s.desirable).count();
        let undesirable = samples.len() - desirable;
        if desirable == 0 || undesirable == 0 {
            tracing::warn!(
                "KTO data has {desirable} desirable and {undesirable} undesirable examples; \
                 it learns most from a mix of both"
            );
        }
        // The paper's guidance: λ_D·n_D / (λ_U·n_U) between 1 and 4/3.
        let balance = (self.kto.desirable_weight * desirable as f32)
            / (self.kto.undesirable_weight * undesirable.max(1) as f32);
        if undesirable > 0 && !(1.0..=4.0 / 3.0).contains(&balance) {
            tracing::warn!(
                "KTO: desirable_weight·n_desirable / (undesirable_weight·n_undesirable) is {balance:.2}; \
                 the paper recommends 1 to 4/3"
            );
        }

        let batch = self.config.batch_size.max(1);
        if batch < 4 {
            tracing::warn!(
                "KTO estimates its KL reference point from each micro-batch; with a batch size \
                 of {batch} that estimate is noisy. 4 or more is recommended"
            );
        }
        tracing::info!(
            "Scoring {} completions with the reference model before training",
            2 * samples.len()
        );
        let seqs: Vec<&Sequence> = samples
            .iter()
            .flat_map(|s| [&s.sequence, &s.mismatched])
            .collect();
        let reference = sequence_log_probs(model, &seqs, 2 * batch, false)?;
        let reference: Vec<(f32, f32)> = reference.chunks(2).map(|c| (c[0], c[1])).collect();

        let tokens: Vec<usize> = samples
            .iter()
            .map(|s| s.sequence.ids.len() + s.mismatched.ids.len())
            .collect();
        let kto = self.kto;

        optimize(
            model,
            optimizer,
            set_lr,
            &self.config,
            &mut self.callbacks,
            &tokens,
            "kto",
            |model, idx| {
                let seqs: Vec<&Sequence> = idx.iter().map(|&i| &samples[i].sequence).collect();
                let policy = batch_log_probs(model, &seqs, false)?;

                // KL estimate from the mismatched completions, out of the gradient.
                let mismatched: Vec<&Sequence> =
                    idx.iter().map(|&i| &samples[i].mismatched).collect();
                let policy_kl = ops::stop_gradient(&batch_log_probs(model, &mismatched, false)?);
                let ref_kl: Vec<f32> = idx.iter().map(|&i| reference[i].1).collect();
                let kl = ops::stop_gradient(&ops::maximum(
                    &policy_kl.subtract(&f32_array(&ref_kl)).mean(None),
                    &Array::from_f32(0.0),
                ));

                let ref_logps: Vec<f32> = idx.iter().map(|&i| reference[i].0).collect();
                let labels: Vec<bool> = idx.iter().map(|&i| samples[i].desirable).collect();
                let (loss, rewards) = kto.loss(&policy, &f32_array(&ref_logps), &kl, &labels);

                let (good, bad): (Vec<usize>, Vec<usize>) =
                    (0..idx.len()).partition(|&j| labels[j]);
                Ok(MicroBatch {
                    loss,
                    chosen_rewards: take(&rewards, &good),
                    rejected_rewards: take(&rewards, &bad),
                    paired: false,
                    kl: Some(kl),
                })
            },
        )
    }
}

/// `values[indices]`, or an empty array.
fn take(values: &Array, indices: &[usize]) -> Array {
    if indices.is_empty() {
        return Array::from_slice::<f32>(&[], &[0]);
    }
    let idx: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    ops::take_axis(values, &Array::from_slice(&idx, &[idx.len() as i32]), 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn loss_matches_hand_computation() {
        // r = policy − ref = [2, −1, 0.5, −3]; desirable = [T, T, F, F]; z = 0.4
        let policy = f32_array(&[-10.0, -21.0, -7.5, -33.0]);
        let reference = f32_array(&[-12.0, -20.0, -8.0, -30.0]);
        let kl = Array::from_f32(0.4);
        let cfg = KtoConfig {
            beta: 0.5,
            desirable_weight: 1.0,
            undesirable_weight: 2.0,
        };
        let (loss, rewards) = cfg.loss(&policy, &reference, &kl, &[true, true, false, false]);
        loss.eval();
        rewards.eval();
        pmetal_bridge::check_last_error().unwrap();

        let r = [2.0f64, -1.0, 0.5, -3.0];
        let z = 0.4;
        let expected = ((1.0 - sigmoid(0.5 * (r[0] - z)))
            + (1.0 - sigmoid(0.5 * (r[1] - z)))
            + 2.0 * (1.0 - sigmoid(0.5 * (z - r[2])))
            + 2.0 * (1.0 - sigmoid(0.5 * (z - r[3]))))
            / 4.0;
        assert!((loss.item_f32() as f64 - expected).abs() < 1e-6);
        assert_eq!(rewards.as_slice::<f32>(), &[1.0, -0.5, 0.25, -1.5]);
    }

    #[test]
    fn take_selects_and_handles_empty() {
        let v = f32_array(&[1.0, 2.0, 3.0]);
        let t = take(&v, &[2, 0]);
        t.eval();
        assert_eq!(t.as_slice::<f32>(), &[3.0, 1.0]);
        let e = take(&v, &[]);
        e.eval();
        assert_eq!(e.shape(), &[0]);
    }

    #[test]
    fn config_validation() {
        assert!(KtoConfig::default().validate().is_ok());
        assert!(
            KtoConfig {
                beta: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
