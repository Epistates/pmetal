//! Shared offline trainer for **paired preference** losses
//! (DPO / ORPO / OnlineDPO and friends).
//!
//! Every paired-preference trainer in `pmetal-trainer` walks the same
//! gradient-step shape: forward chosen + rejected, reduce per-token
//! log-probs into per-sample sums, resolve reference log-probs (one of
//! three strategies: `stop_gradient(policy)` / zeros / external reference
//! model), call a loss-specific kernel, then `value_and_grad` +
//! `optimizer.update`. This module pulls all of that out into one
//! generic loop ([`PairedPreferenceTrainer`]) parameterised by a
//! [`PreferenceLoss`] impl that owns the loss-specific bits.
//!
//! ## Why a trait, not a closure
//!
//! `nn::value_and_grad` requires a closure with a fairly rigid
//! `FnMut(&mut M, ()) -> Result<Array, Exception>` signature. The
//! `PreferenceLoss` impl is captured by reference inside that closure —
//! the trait gives us a concrete, named type to thread through the
//! generic loop without dragging closure-lifetime gymnastics into every
//! call site.
//!
//! ## What it doesn't cover
//!
//! KTO is unpaired (single-sample sigmoid logits) — it'll get its own
//! `UnpairedPreferenceTrainer` follow-on. GRPO is group-relative + PPO
//! clipping — different shape entirely, stays standalone.

use pmetal_bridge::compat::{Array, Exception, nn, ops, optimizers::Optimizer};
use pmetal_core::{StepMetrics, TrainingCallback, TrainingConfig};
use pmetal_lora::TrainableModel;
use std::cell::RefCell;
use std::time::Instant;

/// How a paired-preference loss obtains the reference log-probs that
/// land in `compute_loss`'s last two arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceStrategy {
    /// Inside the closure, derive `ref_logps = stop_gradient(policy_logps)`.
    /// Memory-cheap (no second model forward) but the reference is
    /// degenerate — the loss reduces to a self-anchored margin.
    StopGradient,
    /// Inside the closure, treat `ref_logps` as identically zero.
    /// Used by reference-free losses (SimPO, ORPO).
    Zero,
    /// Outside the closure, run a frozen reference model forward and
    /// pass the precomputed (chosen, rejected) log-prob arrays through
    /// to `compute_loss` unchanged.
    Precomputed,
}

/// Per-loss kernel for the paired-preference family.
///
/// Implementors carry their own config (beta, label smoothing, etc.) —
/// the trainer never inspects the loss's internals. `RawMetrics` is what
/// gets stashed inside the gradient closure (typically MLX `Array`s);
/// `StepOutput` is the materialised per-step metric pushed to the
/// trainer's history (typically a plain struct of `f32`s).
pub trait PreferenceLoss: Clone + Send + Sync {
    /// Opaque per-step arrays the loss kernel produces alongside the
    /// loss tensor (e.g. `(chosen_rewards, rejected_rewards)`). Stored
    /// pre-eval inside the gradient closure; finalised after eval.
    type RawMetrics;

    /// Per-step metrics surfaced through the trainer's `Vec` history
    /// and the `TrainingCallback` events.
    type StepOutput: Clone + Send + Sync;

    /// Reduce per-token log-probs from logits + labels.
    ///
    /// The default uses a plain sum (`compute_log_probs`). SimPO-style
    /// losses override to length-normalised mean.
    fn reduce_log_probs(&self, logits: &Array, labels: &Array) -> Result<Array, Exception> {
        crate::logprob_utils::compute_log_probs(logits, labels)
    }

    /// Compute the loss tensor + raw per-step metrics. Called inside
    /// the gradient closure — must not call `.eval()`.
    fn compute_loss(
        &self,
        policy_chosen: &Array,
        policy_rejected: &Array,
        ref_chosen: &Array,
        ref_rejected: &Array,
    ) -> Result<(Array, Self::RawMetrics), Exception>;

    /// Convert the loss's evaluated scalar + raw metrics arrays into
    /// the wire-final `StepOutput` shape.
    fn finalize(&self, loss_value: f32, raw: Self::RawMetrics) -> Self::StepOutput;

    /// Where reference log-probs come from for this loss.
    fn reference_strategy(&self) -> ReferenceStrategy;
}

/// Generic offline trainer that drives any [`PreferenceLoss`] impl over a
/// dataset of (prompt, chosen, rejected) triples.
///
/// Owns the training/optimizer config, the callback list, and the step
/// counter. The loss kernel `L` decides what each gradient step actually
/// computes.
pub struct PairedPreferenceTrainer<L: PreferenceLoss> {
    pub loss: L,
    pub training_config: TrainingConfig,
    pub callbacks: Vec<Box<dyn TrainingCallback>>,
    step: usize,
}

impl<L: PreferenceLoss> PairedPreferenceTrainer<L> {
    /// Create a new trainer with the given loss kernel and training config.
    pub fn new(loss: L, training_config: TrainingConfig) -> Self {
        Self {
            loss,
            training_config,
            callbacks: Vec::new(),
            step: 0,
        }
    }

    /// Add a training callback. Called in registration order on each
    /// `on_*` event.
    pub fn add_callback(&mut self, callback: Box<dyn TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Current step counter — useful for resuming / for callbacks that
    /// want to query state outside the loop.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Run one gradient step over a single (chosen, rejected) batch and
    /// return the materialised per-step metrics.
    ///
    /// Public so per-loss trainers can keep their own train loop shape
    /// (data loaders, scheduling, distillation glue) while still routing
    /// the gradient/optimizer step through the shared body.
    ///
    /// `precomputed_ref` MUST be `Some` when the loss's
    /// [`PreferenceLoss::reference_strategy`] is `Precomputed`, and
    /// `None` for the other two strategies — the function asserts this
    /// to fail loudly on miswiring.
    pub fn step<M, O>(
        &self,
        policy_model: &mut M,
        optimizer: &mut O,
        chosen_inputs: &Array,
        chosen_labels: &Array,
        rejected_inputs: &Array,
        rejected_labels: &Array,
        precomputed_ref: Option<(Array, Array)>,
    ) -> Result<(f32, L::StepOutput), Exception>
    where
        M: TrainableModel,
        O: Optimizer,
    {
        match (self.loss.reference_strategy(), precomputed_ref.is_some()) {
            (ReferenceStrategy::Precomputed, false) => {
                return Err(Exception::custom(
                    "PairedPreferenceTrainer: Precomputed reference strategy \
                     requires Some(precomputed_ref)",
                ));
            }
            (ReferenceStrategy::StopGradient | ReferenceStrategy::Zero, true) => {
                return Err(Exception::custom(
                    "PairedPreferenceTrainer: stop-gradient/zero strategies \
                     do not accept precomputed_ref — pass None",
                ));
            }
            _ => {}
        }

        let strategy = self.loss.reference_strategy();
        let loss_kernel = self.loss.clone();
        let raw_cell: RefCell<Option<L::RawMetrics>> = RefCell::new(None);

        let loss_fn = |model: &mut M, _: ()| -> Result<Array, Exception> {
            let chosen_logits = model
                .forward(chosen_inputs, None)
                .map_err(|e| Exception::custom(e.to_string()))?;
            let rejected_logits = model
                .forward(rejected_inputs, None)
                .map_err(|e| Exception::custom(e.to_string()))?;

            let chosen_policy = loss_kernel.reduce_log_probs(&chosen_logits, chosen_labels)?;
            let rejected_policy =
                loss_kernel.reduce_log_probs(&rejected_logits, rejected_labels)?;

            let (chosen_ref, rejected_ref) = match strategy {
                ReferenceStrategy::StopGradient => (
                    ops::stop_gradient(&chosen_policy),
                    ops::stop_gradient(&rejected_policy),
                ),
                ReferenceStrategy::Zero => (Array::from_f32(0.0), Array::from_f32(0.0)),
                ReferenceStrategy::Precomputed => {
                    let (c, r) = precomputed_ref
                        .as_ref()
                        .expect("checked above before entering closure");
                    (c.clone(), r.clone())
                }
            };

            let (loss, raw) = loss_kernel.compute_loss(
                &chosen_policy,
                &rejected_policy,
                &chosen_ref,
                &rejected_ref,
            )?;
            *raw_cell.borrow_mut() = Some(raw);
            Ok(loss)
        };

        let (mut loss, grads) = {
            let mut value_and_grad = nn::value_and_grad(loss_fn);
            value_and_grad(policy_model, ())?
        };
        optimizer.update(policy_model, grads)?;
        loss.eval();
        let loss_value = loss.item_f32();

        let raw = raw_cell
            .into_inner()
            .expect("loss_fn always populates raw_cell");
        Ok((loss_value, self.loss.finalize(loss_value, raw)))
    }

    /// Emit a `StepMetrics` event to all callbacks. Public so per-loss
    /// trainers can integrate it into their own loop shape after a
    /// successful [`step`].
    pub fn record_step(
        &mut self,
        loss_value: f32,
        elapsed_secs: f64,
        tokens: usize,
        epoch: usize,
        total_epochs: usize,
        total_steps: usize,
    ) {
        self.step += 1;
        let lr = self.training_config.learning_rate;
        let metrics = StepMetrics {
            step: self.step,
            epoch,
            total_epochs,
            total_steps,
            loss: loss_value as f64,
            lr,
            tok_sec: if elapsed_secs > 0.0 {
                tokens as f64 / elapsed_secs
            } else {
                0.0
            },
            total_ms: elapsed_secs * 1000.0,
            tokens,
            ..Default::default()
        };
        for cb in &mut self.callbacks {
            cb.on_step_end_with_metrics(&metrics);
        }
    }

    /// Returns true if any callback has requested early stop. Per-loss
    /// trainers should check this between steps and bail with a
    /// trainer-specific cancellation error.
    pub fn should_stop(&self) -> bool {
        self.callbacks.iter().any(|cb| cb.should_stop())
    }

    /// Time-keeper helper — returns an `Instant` for the start of the
    /// next step. Kept here so per-loss trainers don't reimplement.
    pub fn step_start(&self) -> Instant {
        Instant::now()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Built-in PreferenceLoss impls — re-exported under the trainer's namespace
// so callers can pick them up without an extra `use`.
// ────────────────────────────────────────────────────────────────────────────

/// [`PreferenceLoss`] kernel for DPO-family losses (Sigmoid, IPO, Hinge,
/// Robust, SimPO).
///
/// Wraps a [`crate::dpo::DpoConfig`] and delegates to the same static
/// loss math the standalone [`crate::dpo::DpoTrainer`] uses, so the
/// numerical behaviour is identical when DPO eventually migrates onto
/// [`PairedPreferenceTrainer`]. Today this struct is the recommended
/// entry point for new callers; existing DPO consumers can keep using
/// `DpoTrainer::train` until P3b lands the migration.
#[derive(Clone)]
pub struct DpoLoss {
    pub config: crate::dpo::DpoConfig,
}

impl DpoLoss {
    pub fn new(config: crate::dpo::DpoConfig) -> Self {
        Self { config }
    }
}

impl PreferenceLoss for DpoLoss {
    type RawMetrics = (Array, Array);
    type StepOutput = crate::dpo::DpoMetrics;

    fn reduce_log_probs(&self, logits: &Array, labels: &Array) -> Result<Array, Exception> {
        // SimPO is length-normalised; every other DPO variant uses the
        // plain summed log-probs.
        if matches!(self.config.loss_type, crate::dpo::DpoLossType::SimPo) {
            let (_sum, avg) = crate::logprob_utils::compute_log_probs_with_avg(logits, labels)?;
            Ok(avg)
        } else {
            crate::logprob_utils::compute_log_probs(logits, labels)
        }
    }

    fn compute_loss(
        &self,
        policy_chosen: &Array,
        policy_rejected: &Array,
        ref_chosen: &Array,
        ref_rejected: &Array,
    ) -> Result<(Array, (Array, Array)), Exception> {
        let (loss, chosen_rewards, rejected_rewards) =
            crate::dpo::DpoTrainer::compute_dpo_loss_for(
                &self.config,
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
            )?;
        Ok((loss, (chosen_rewards, rejected_rewards)))
    }

    fn finalize(&self, loss_value: f32, raw: (Array, Array)) -> crate::dpo::DpoMetrics {
        let (mut chosen_rewards, mut rejected_rewards) = raw;
        chosen_rewards.eval();
        rejected_rewards.eval();
        let chosen_vec = chosen_rewards.as_slice::<f32>().to_vec();
        let rejected_vec = rejected_rewards.as_slice::<f32>().to_vec();
        crate::dpo::DpoMetrics::compute(loss_value, &chosen_vec, &rejected_vec)
    }

    fn reference_strategy(&self) -> ReferenceStrategy {
        // Order matches DpoTrainer::train precedence: stop-gradient takes
        // priority over reference-free, which takes priority over the
        // external-reference path.
        if self.config.use_stop_gradient_reference {
            ReferenceStrategy::StopGradient
        } else if self.config.reference_free
            || matches!(self.config.loss_type, crate::dpo::DpoLossType::SimPo)
        {
            ReferenceStrategy::Zero
        } else {
            ReferenceStrategy::Precomputed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny PreferenceLoss impl that doesn't touch the model — just lets us
    /// verify the trait wires together correctly without spinning up a real
    /// architecture.
    #[derive(Clone)]
    struct ZeroLoss;

    impl PreferenceLoss for ZeroLoss {
        type RawMetrics = ();
        type StepOutput = f32;
        fn compute_loss(
            &self,
            _: &Array,
            _: &Array,
            _: &Array,
            _: &Array,
        ) -> Result<(Array, ()), Exception> {
            Ok((Array::from_f32(0.0), ()))
        }
        fn finalize(&self, loss: f32, _: ()) -> f32 {
            loss
        }
        fn reference_strategy(&self) -> ReferenceStrategy {
            ReferenceStrategy::Zero
        }
    }

    #[test]
    fn reference_strategy_round_trip() {
        let loss = ZeroLoss;
        assert_eq!(loss.reference_strategy(), ReferenceStrategy::Zero);
    }

    #[test]
    fn finalize_round_trip() {
        let loss = ZeroLoss;
        let v = loss.finalize(0.42, ());
        assert!((v - 0.42).abs() < 1e-6);
    }

    /// DpoLoss uses the same kernel the DpoTrainer does internally —
    /// verify they produce identical loss + reward arrays for the same
    /// inputs. Guards against accidental drift between the two surfaces.
    #[test]
    fn dpo_loss_matches_dpo_trainer_kernel() {
        use crate::dpo::{DpoConfig, DpoLossType, DpoTrainer};

        let cfg = DpoConfig {
            beta: 0.1,
            loss_type: DpoLossType::Sigmoid,
            ..Default::default()
        };
        // Synthetic 4-sample logps — chosen consistently exceeds rejected.
        let policy_chosen = Array::from_slice(&[-1.0_f32, -1.5, -0.5, -2.0], &[4]);
        let policy_rejected = Array::from_slice(&[-2.0_f32, -2.5, -1.5, -3.0], &[4]);
        let ref_chosen = Array::from_slice(&[-1.2_f32, -1.7, -0.8, -2.2], &[4]);
        let ref_rejected = Array::from_slice(&[-1.8_f32, -2.3, -1.3, -2.8], &[4]);

        // Reference path: DpoTrainer's static kernel.
        let (mut ref_loss, mut ref_chosen_rew, mut ref_rejected_rew) =
            DpoTrainer::compute_dpo_loss_for(
                &cfg,
                &policy_chosen,
                &policy_rejected,
                &ref_chosen,
                &ref_rejected,
            )
            .unwrap();

        // PreferenceLoss path: DpoLoss::compute_loss.
        let loss_kernel = DpoLoss::new(cfg);
        let (mut new_loss, raw) = loss_kernel
            .compute_loss(&policy_chosen, &policy_rejected, &ref_chosen, &ref_rejected)
            .unwrap();
        let (mut new_chosen_rew, mut new_rejected_rew) = raw;

        ref_loss.eval();
        new_loss.eval();
        ref_chosen_rew.eval();
        new_chosen_rew.eval();
        ref_rejected_rew.eval();
        new_rejected_rew.eval();

        assert!(
            (ref_loss.item_f32() - new_loss.item_f32()).abs() < 1e-5,
            "loss mismatch: trainer={} preference_loss={}",
            ref_loss.item_f32(),
            new_loss.item_f32()
        );
        let r0: &[f32] = ref_chosen_rew.as_slice();
        let n0: &[f32] = new_chosen_rew.as_slice();
        for (a, b) in r0.iter().zip(n0.iter()) {
            assert!((a - b).abs() < 1e-5, "chosen reward mismatch");
        }
        let r1: &[f32] = ref_rejected_rew.as_slice();
        let n1: &[f32] = new_rejected_rew.as_slice();
        for (a, b) in r1.iter().zip(n1.iter()) {
            assert!((a - b).abs() < 1e-5, "rejected reward mismatch");
        }
    }

    /// Reference strategy is config-driven — verify each DPO config
    /// shape resolves to the expected `ReferenceStrategy`.
    #[test]
    fn dpo_loss_reference_strategy_follows_config() {
        use crate::dpo::{DpoConfig, DpoLossType};

        let stop_grad = DpoLoss::new(DpoConfig {
            use_stop_gradient_reference: true,
            ..Default::default()
        });
        assert_eq!(
            stop_grad.reference_strategy(),
            ReferenceStrategy::StopGradient
        );

        let ref_free = DpoLoss::new(DpoConfig {
            reference_free: true,
            ..Default::default()
        });
        assert_eq!(ref_free.reference_strategy(), ReferenceStrategy::Zero);

        let simpo = DpoLoss::new(DpoConfig {
            loss_type: DpoLossType::SimPo,
            ..Default::default()
        });
        // SimPO is reference-free even when the flag isn't set.
        assert_eq!(simpo.reference_strategy(), ReferenceStrategy::Zero);

        let standard = DpoLoss::new(DpoConfig::default());
        assert_eq!(
            standard.reference_strategy(),
            ReferenceStrategy::Precomputed
        );
    }
}
