//! Offline preference optimization: DPO, IPO, hinge, SimPO, ORPO and KTO.
//!
//! [`PreferenceTrainer`] trains on (prompt, chosen, rejected) pairs with any
//! [`PreferenceLoss`]; [`KtoTrainer`] trains on single completions labelled
//! desirable or undesirable. Both share one optimization loop: shuffled
//! micro-batches, gradient accumulation, clipping, the learning-rate schedule
//! from [`TrainingConfig`], and the usual [`TrainingCallback`] events.
//!
//! # The reference model costs no memory
//!
//! DPO, IPO, hinge and KTO compare the policy with a frozen reference, which
//! is the model before training. With LoRA the adapters start at zero (`B` is
//! initialised to zeros), so the policy *is* the reference until the first
//! update. Both trainers therefore score every example with the policy once,
//! before training, and keep those log-probs: one forward pass per sequence,
//! a few bytes per example, and no second copy of the model.

mod data;
mod kto;
mod loss;

pub use data::{load_kto_samples, load_preference_pairs, read_rows};
pub use kto::{KtoConfig, KtoSample, KtoTrainer};
pub use loss::{PairLoss, PreferenceLoss};

use std::cell::RefCell;
use std::time::Instant;

use pmetal_bridge::compat::{
    Array, Exception, FlattenedModuleParam, nn, optimizers::Optimizer, transforms,
};
use pmetal_core::{
    EvalMetrics, LearningRateScheduler, StepMetrics, TrainingCallback, TrainingConfig,
};
use pmetal_lora::TrainableModel;
use rand::SeedableRng;
use rand::seq::SliceRandom;

use crate::logprob_utils::{compute_log_probs, compute_log_probs_with_avg};

/// Error from preference training.
#[derive(Debug, thiserror::Error)]
pub enum PreferenceError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Invalid configuration.
    #[error("Configuration error: {0}")]
    Config(String),
    /// Unusable training data.
    #[error("Data error: {0}")]
    Data(String),
    /// A callback asked training to stop.
    #[error("Training cancelled")]
    Cancelled,
}

/// Result type for preference training.
pub type PreferenceResult<T> = std::result::Result<T, PreferenceError>;

/// A tokenized sequence: input ids and labels, with `-100` on every position
/// that is not part of the completion (the prompt and any padding).
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    /// Prompt and completion tokens.
    pub ids: Vec<u32>,
    /// `ids` with the prompt masked to `-100`.
    pub labels: Vec<i64>,
}

impl Sequence {
    /// Build a sequence from a prompt and a completion, masking the prompt.
    pub fn new(prompt: &[u32], completion: &[u32]) -> Self {
        let mut ids = Vec::with_capacity(prompt.len() + completion.len());
        ids.extend_from_slice(prompt);
        ids.extend_from_slice(completion);
        let mut labels = vec![-100i64; prompt.len()];
        labels.extend(completion.iter().map(|&t| t as i64));
        Self { ids, labels }
    }

    /// Number of completion tokens the loss reads. The first position is
    /// never predicted, so a label there doesn't count.
    pub fn completion_len(&self) -> usize {
        self.labels.iter().skip(1).filter(|&&l| l != -100).count()
    }
}

/// One preference pair: the same prompt with a chosen and a rejected completion.
#[derive(Debug, Clone, PartialEq)]
pub struct PreferencePair {
    /// Prompt + chosen completion.
    pub chosen: Sequence,
    /// Prompt + rejected completion.
    pub rejected: Sequence,
}

impl PreferencePair {
    /// Build a pair from token ids.
    pub fn new(prompt: &[u32], chosen: &[u32], rejected: &[u32]) -> Self {
        Self {
            chosen: Sequence::new(prompt, chosen),
            rejected: Sequence::new(prompt, rejected),
        }
    }
}

/// Metrics of one optimizer step, averaged over its micro-batches.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PreferenceStepMetrics {
    /// Optimizer step, from 1.
    pub step: usize,
    /// Loss.
    pub loss: f32,
    /// Learning rate used for the step.
    pub learning_rate: f32,
    /// Gradient norm before clipping (0 when clipping is off).
    pub grad_norm: f32,
    /// Mean implicit reward of chosen (KTO: desirable) completions.
    pub chosen_reward: f32,
    /// Mean implicit reward of rejected (KTO: undesirable) completions.
    pub rejected_reward: f32,
    /// Fraction of pairs whose chosen reward beats the rejected one. Paired
    /// objectives only.
    pub accuracy: Option<f32>,
    /// The KL estimate KTO subtracted this step (KTO only).
    pub kl: Option<f32>,
}

impl PreferenceStepMetrics {
    /// Chosen minus rejected reward.
    pub fn margin(&self) -> f32 {
        self.chosen_reward - self.rejected_reward
    }
}

/// Offline trainer for paired preference objectives.
pub struct PreferenceTrainer {
    loss: PreferenceLoss,
    config: TrainingConfig,
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl PreferenceTrainer {
    /// Create a trainer. `config` supplies the learning rate and its schedule,
    /// batch size, gradient accumulation, clipping, epochs or `max_steps`,
    /// `logging_steps` and the shuffling seed.
    pub fn new(loss: PreferenceLoss, config: TrainingConfig) -> PreferenceResult<Self> {
        loss.validate().map_err(PreferenceError::Config)?;
        Ok(Self {
            loss,
            config,
            callbacks: Vec::new(),
        })
    }

    /// Add a training callback.
    pub fn add_callback(&mut self, callback: Box<dyn TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// The objective being trained.
    pub fn loss(&self) -> PreferenceLoss {
        self.loss
    }

    /// Train `model` on `pairs`.
    ///
    /// For objectives with a reference, `model` must still be the reference:
    /// call this before any update (a fresh LoRA model is), and the reference
    /// log-probs are taken from it before the first step. `set_lr` applies
    /// the scheduled learning rate to `optimizer`.
    pub fn train<M, O>(
        &mut self,
        model: &mut M,
        pairs: &[PreferencePair],
        optimizer: &mut O,
        set_lr: impl FnMut(&mut O, f32),
    ) -> PreferenceResult<Vec<PreferenceStepMetrics>>
    where
        M: TrainableModel,
        O: Optimizer,
    {
        if pairs.is_empty() {
            return Err(PreferenceError::Data("no preference pairs".into()));
        }
        let loss = self.loss;
        let normalized = loss.length_normalized();
        let batch = self.config.batch_size.max(1);

        let reference: Option<Vec<(f32, f32)>> = if loss.uses_reference() {
            tracing::info!(
                "Scoring {} pairs with the reference model before training",
                pairs.len()
            );
            let seqs: Vec<&Sequence> = pairs
                .iter()
                .flat_map(|p| [&p.chosen, &p.rejected])
                .collect();
            let logps = sequence_log_probs(model, &seqs, 2 * batch, normalized)?;
            Some(logps.chunks(2).map(|c| (c[0], c[1])).collect())
        } else {
            None
        };

        let tokens: Vec<usize> = pairs
            .iter()
            .map(|p| p.chosen.ids.len() + p.rejected.ids.len())
            .collect();

        optimize(
            model,
            optimizer,
            set_lr,
            &self.config,
            &mut self.callbacks,
            &tokens,
            loss.name(),
            |model, idx| {
                let seqs: Vec<&Sequence> = idx
                    .iter()
                    .map(|&i| &pairs[i].chosen)
                    .chain(idx.iter().map(|&i| &pairs[i].rejected))
                    .collect();
                let logps = batch_log_probs(model, &seqs, normalized)?;
                let mut halves = pmetal_bridge::compat::ops::split(&logps, 2, 0).into_iter();
                let (chosen, rejected) = (halves.next().unwrap(), halves.next().unwrap());
                let reference = reference.as_ref().map(|r| {
                    let c: Vec<f32> = idx.iter().map(|&i| r[i].0).collect();
                    let l: Vec<f32> = idx.iter().map(|&i| r[i].1).collect();
                    (f32_array(&c), f32_array(&l))
                });
                let out =
                    loss.compute(&chosen, &rejected, reference.as_ref().map(|(c, l)| (c, l)))?;
                Ok(MicroBatch {
                    loss: out.loss,
                    chosen_rewards: out.chosen_rewards,
                    rejected_rewards: out.rejected_rewards,
                    paired: true,
                    kl: None,
                })
            },
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared machinery
// ─────────────────────────────────────────────────────────────────────────────

/// What a micro-batch's loss closure hands back besides the loss.
pub(crate) struct MicroBatch {
    pub loss: Array,
    pub chosen_rewards: Array,
    pub rejected_rewards: Array,
    /// Whether `chosen_rewards[i]` and `rejected_rewards[i]` belong to one pair.
    pub paired: bool,
    pub kl: Option<Array>,
}

fn f32_array(values: &[f32]) -> Array {
    Array::from_slice(values, &[values.len() as i32])
}

/// Right-pad sequences into `[N, T]` ids (pad 0) and labels (pad -100).
fn pad(seqs: &[&Sequence]) -> (Array, Array) {
    let len = seqs.iter().map(|s| s.ids.len()).max().unwrap_or(1).max(2);
    let mut ids = Vec::with_capacity(seqs.len() * len);
    let mut labels = Vec::with_capacity(seqs.len() * len);
    for s in seqs {
        ids.extend(s.ids.iter().map(|&t| t as i32));
        ids.extend(std::iter::repeat_n(0i32, len - s.ids.len()));
        labels.extend_from_slice(&s.labels);
        labels.extend(std::iter::repeat_n(-100i64, len - s.labels.len()));
    }
    let shape = [seqs.len() as i32, len as i32];
    (
        Array::from_slice(&ids, &shape),
        Array::from_slice(&labels, &shape),
    )
}

/// Sequence log-probs `[N]` of the model, summed over completion tokens or
/// averaged over them.
///
/// Right padding leaves earlier positions untouched under the causal mask,
/// and padded positions carry label `-100`, so they add nothing.
pub(crate) fn batch_log_probs<M: TrainableModel>(
    model: &mut M,
    seqs: &[&Sequence],
    length_normalized: bool,
) -> Result<Array, Exception> {
    let (ids, labels) = pad(seqs);
    let logits = model
        .forward(&ids, None)
        .map_err(|e| Exception::custom(e.to_string()))?;
    if length_normalized {
        Ok(compute_log_probs_with_avg(&logits, &labels)?.1)
    } else {
        compute_log_probs(&logits, &labels)
    }
}

/// [`batch_log_probs`] over many sequences in chunks of `chunk`, evaluated:
/// one value per sequence, in order.
pub(crate) fn sequence_log_probs<M: TrainableModel>(
    model: &mut M,
    seqs: &[&Sequence],
    chunk: usize,
    length_normalized: bool,
) -> Result<Vec<f32>, Exception> {
    let mut out = Vec::with_capacity(seqs.len());
    for part in seqs.chunks(chunk.max(1)) {
        let logps = pmetal_bridge::compat::ops::stop_gradient(&batch_log_probs(
            model,
            part,
            length_normalized,
        )?);
        logps.eval();
        pmetal_bridge::check_last_error().map_err(|e| Exception::custom(e.to_string()))?;
        out.extend_from_slice(logps.as_slice::<f32>());
    }
    Ok(out)
}

/// An evaluated f32 array's values; empty for a zero-length array, whose
/// data pointer MLX leaves null.
fn floats(a: &Array) -> &[f32] {
    if a.size() == 0 {
        &[]
    } else {
        a.as_slice::<f32>()
    }
}

fn mean_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Add `grads` into `acc`.
fn accumulate(acc: &mut Option<FlattenedModuleParam>, grads: FlattenedModuleParam) {
    match acc {
        None => *acc = Some(grads),
        Some(acc) => {
            for (key, grad) in grads {
                match acc.get_mut(&key) {
                    Some(sum) => *sum = sum.add(&grad),
                    None => {
                        acc.insert(key, grad);
                    }
                }
            }
        }
    }
}

/// The optimization loop both preference trainers run.
///
/// Each epoch shuffles the examples with `config.seed`, cuts them into
/// micro-batches of `batch_size`, and takes an optimizer step every
/// `gradient_accumulation_steps` micro-batches (and at the end of the epoch)
/// on the mean of their gradients, clipped to `max_grad_norm`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize<M, O>(
    model: &mut M,
    optimizer: &mut O,
    mut set_lr: impl FnMut(&mut O, f32),
    config: &TrainingConfig,
    callbacks: &mut [Box<dyn TrainingCallback>],
    example_tokens: &[usize],
    objective: &str,
    mut micro_loss: impl FnMut(&mut M, &[usize]) -> Result<MicroBatch, Exception>,
) -> PreferenceResult<Vec<PreferenceStepMetrics>>
where
    M: TrainableModel,
    O: Optimizer,
{
    let n = example_tokens.len();
    let batch = config.batch_size.max(1);
    let accum = config.gradient_accumulation_steps.max(1);
    let epochs = config.num_epochs.max(1);
    // Each epoch ends on a step, partial accumulation or not.
    let total_steps = pmetal_core::total_training_steps(config, n.div_ceil(batch), accum, true);
    let warmup = pmetal_core::warmup_steps_for(config, total_steps);
    let scheduler = LearningRateScheduler::for_training(config, total_steps);
    let log_every = config.logging_steps.max(1);

    tracing::info!(
        "{objective}: {n} examples, {epochs} epoch(s), {total_steps} steps \
         (batch {batch} x {accum} accumulation), {warmup} warmup steps, {:?} schedule, \
         {} trainable parameters",
        config.lr_scheduler,
        model.num_trainable_params()
    );

    for cb in callbacks.iter_mut() {
        cb.on_train_start();
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let mut history = Vec::with_capacity(total_steps);
    let mut step = 0usize;
    let mut last_epoch = 0usize;

    'epochs: for epoch in 0..epochs {
        last_epoch = epoch;
        for cb in callbacks.iter_mut() {
            cb.on_epoch_start(epoch);
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        let micro_batches: Vec<&[usize]> = order.chunks(batch).collect();

        for group in micro_batches.chunks(accum) {
            if step >= total_steps {
                break 'epochs;
            }
            let started = Instant::now();
            for cb in callbacks.iter_mut() {
                cb.on_step_start(step + 1);
            }

            let mut grads: Option<FlattenedModuleParam> = None;
            let mut losses = Vec::with_capacity(group.len());
            let mut chosen = Vec::new();
            let mut rejected = Vec::new();
            let mut wins = 0usize;
            let mut pairs = 0usize;
            let mut kls = Vec::new();
            let mut tokens = 0usize;

            for idx in group {
                let stash: RefCell<Option<MicroBatch>> = RefCell::new(None);
                let (loss, micro_grads) = {
                    let loss_fn = |m: &mut M, _: ()| -> Result<Array, Exception> {
                        let out = micro_loss(m, idx)?;
                        let loss = out.loss.clone();
                        *stash.borrow_mut() = Some(out);
                        Ok(loss)
                    };
                    let mut value_and_grad = nn::value_and_grad(loss_fn);
                    value_and_grad(model, ())?
                };
                accumulate(&mut grads, micro_grads);
                let out = stash
                    .into_inner()
                    .expect("value_and_grad calls the loss closure once");
                let mut to_eval = vec![&loss, &out.chosen_rewards, &out.rejected_rewards];
                if let Some(kl) = &out.kl {
                    to_eval.push(kl);
                }
                transforms::eval(to_eval)?;
                pmetal_bridge::check_last_error().map_err(|e| Exception::custom(e.to_string()))?;

                losses.push(loss.item_f32());
                let c = floats(&out.chosen_rewards);
                let r = floats(&out.rejected_rewards);
                if out.paired {
                    pairs += c.len();
                    wins += c.iter().zip(r).filter(|(c, r)| c > r).count();
                }
                chosen.extend_from_slice(c);
                rejected.extend_from_slice(r);
                if let Some(kl) = &out.kl {
                    kls.push(kl.item_f32());
                }
                tokens += idx.iter().map(|&i| example_tokens[i]).sum::<usize>();
            }

            let mut grads = grads.expect("a step has at least one micro-batch");
            if group.len() > 1 {
                let scale = Array::from_f32(1.0 / group.len() as f32);
                for g in grads.values_mut() {
                    *g = g.multiply(&scale);
                }
            }
            let grad_norm = pmetal_bridge::training::clip_grad_norm_map(
                &mut grads,
                config.max_grad_norm as f32,
            );
            let lr = scheduler.get_lr(step) as f32;
            set_lr(optimizer, lr);
            optimizer.update(model, grads)?;
            grad_norm.eval();
            let params = model.lora_parameters();
            transforms::eval(params.values())?;
            pmetal_bridge::check_last_error().map_err(|e| Exception::custom(e.to_string()))?;
            step += 1;

            let metrics = PreferenceStepMetrics {
                step,
                loss: mean_of(&losses),
                learning_rate: lr,
                grad_norm: if config.max_grad_norm > 0.0 {
                    grad_norm.item_f32()
                } else {
                    0.0
                },
                chosen_reward: mean_of(&chosen),
                rejected_reward: mean_of(&rejected),
                accuracy: (pairs > 0).then(|| wins as f32 / pairs as f32),
                kl: (!kls.is_empty()).then(|| mean_of(&kls)),
            };
            if !metrics.loss.is_finite() {
                return Err(PreferenceError::Data(format!(
                    "{objective}: loss is {} at step {step}",
                    metrics.loss
                )));
            }

            let elapsed = started.elapsed().as_secs_f64();
            let step_metrics = StepMetrics {
                step,
                epoch,
                total_epochs: epochs,
                total_steps,
                loss: metrics.loss as f64,
                lr: lr as f64,
                tok_sec: if elapsed > 0.0 {
                    tokens as f64 / elapsed
                } else {
                    0.0
                },
                total_ms: elapsed * 1000.0,
                tokens,
                grad_norm: (config.max_grad_norm > 0.0).then_some(metrics.grad_norm as f64),
                ..Default::default()
            };
            for cb in callbacks.iter_mut() {
                cb.on_step_end_with_metrics(&step_metrics);
            }
            if step == 1 || step % log_every == 0 || step == total_steps {
                let accuracy = metrics
                    .accuracy
                    .map(|a| format!(" acc {:.2}", a))
                    .unwrap_or_default();
                let kl = metrics
                    .kl
                    .map(|k| format!(" kl {:.4}", k))
                    .unwrap_or_default();
                tracing::info!(
                    "{objective} step {step}/{total_steps}: loss {:.4} | rewards chosen {:.4} \
                     rejected {:.4} margin {:.4}{accuracy}{kl} | lr {:.2e} | {:.0} tok/s",
                    metrics.loss,
                    metrics.chosen_reward,
                    metrics.rejected_reward,
                    metrics.margin(),
                    lr,
                    step_metrics.tok_sec,
                );
            }
            history.push(metrics);
            if callbacks.iter().any(|cb| cb.should_stop()) {
                return Err(PreferenceError::Cancelled);
            }
        }
    }

    let eval = EvalMetrics {
        loss: history.last().map(|m| m.loss as f64).unwrap_or(0.0),
        perplexity: 0.0,
        accuracy: history.last().and_then(|m| m.accuracy.map(f64::from)),
        custom: std::collections::HashMap::new(),
    };
    for cb in callbacks.iter_mut() {
        cb.on_epoch_end(last_epoch, &eval);
        cb.on_train_end();
    }
    Ok(history)
}

#[cfg(test)]
mod tests;
