//! The training loop: data, model and optimizer connected, with gradient
//! accumulation, learning-rate scheduling, adaptive control, evaluation,
//! checkpointing and resume.
//!
//! There is one loop, [`TrainingLoop::run`]. It reads micro-batches from a
//! source (padded batches, or rows of packed sequences), turns
//! each into a loss and gradients, adds them up until an optimizer
//! step's worth has been seen, and takes the step. Packing, cut cross-entropy,
//! NEFTune, vision inputs and data-parallel training are properties of the
//! source, the loss and the gradient exchange, not separate loops.
//!
//! # One evaluation per micro-batch
//!
//! MLX is lazy: a step is a graph until something evaluates it. The loop
//! evaluates once per micro-batch, and everything the micro-batch needs goes
//! into that one call: the loss, and the accumulated gradients when more
//! micro-batches follow; the loss, the gradient norm, every trainable
//! parameter and the optimizer's state when the optimizer has stepped. The
//! loss and norm are read back after it, so reading them waits for nothing.
//! Evaluating each step also frees its forward and backward graph, which
//! otherwise stays alive and grows to many times the model.
//!
//! # Gradient accumulation
//!
//! Each micro-batch contributes the sum of its tokens' losses and the number
//! of tokens that count; an optimizer step's gradient is the gradient of the
//! mean over all the tokens of its micro-batches. A step is
//! `gradient_accumulation_steps` micro-batches, however they are padded. The
//! learning-rate schedule, `max_steps`, logging, evaluation and checkpoints
//! count optimizer steps; [`TrainingLoop::current_step`] counts micro-batches.
//!
//! # Resume
//!
//! A checkpoint holds the weights, the optimizer's state (moments, step
//! counters and the f32 master copies of half-precision weights) and the data
//! loader's position. [`TrainingLoop::resume_from`] loads the latest one, and
//! the run continues with the batch the interrupted one would have drawn next.

use pmetal_core::{EvalMetrics, TrainingConfig};
use pmetal_data::{DataLoader, DataLoaderConfig, TrainingDataset};
use pmetal_lora::TrainableModel;
use pmetal_runtime::compat::FlattenedModuleParam;
use pmetal_runtime::compat::{Array, Exception, indexing::argmax_axis};

use crate::checkpoint::TrainState;
use crate::{CheckpointManager, CheckpointMetadata, Result, SftError};

mod run;
mod source;
pub(crate) mod step;
#[cfg(test)]
mod tests;

pub(crate) use step::LossSpec;

/// Training loop configuration.
#[derive(Debug, Clone)]
pub struct TrainingLoopConfig {
    /// Training hyperparameters.
    pub training: TrainingConfig,
    /// DataLoader configuration.
    pub dataloader: DataLoaderConfig,
    /// Log every N optimizer steps.
    pub log_every: usize,
    /// Checkpoint every N optimizer steps (0 to disable).
    pub checkpoint_every: usize,
    /// Evaluate every N optimizer steps (0 to disable).
    pub eval_every: usize,
    /// Pack short sequences into shared rows, with block-diagonal attention
    /// masks keeping them apart. A model whose recurrent layers cannot reset at
    /// a packed boundary, or data that does not pack, trains unpacked.
    pub use_sequence_packing: bool,
    /// Enable gradient checkpointing to reduce memory usage.
    /// Trades compute for memory by recomputing activations during backward pass.
    /// Allows ~2x larger batch sizes with ~30% slowdown.
    pub gradient_checkpointing: bool,
    /// Number of layers per checkpoint block.
    /// Lower = more memory savings but slower. Recommended: 4 for most models.
    pub gradient_checkpointing_layers: usize,
    /// Separate learning rate for embedding parameters.
    /// Recommended default is 5e-5 for embeddings vs 2e-4 for LoRA params.
    /// None means use the same learning rate as other parameters.
    pub embedding_lr: Option<f32>,
    /// LoRA+ B/A learning rate ratio (Hayou et al., ICML 2024).
    ///
    /// When set, LoRA B matrices receive `base_lr * ratio` while A matrices use `base_lr`.
    /// Breaking this symmetry accelerates convergence because B starts at zero and directly
    /// controls the output magnitude.
    ///
    /// Recommended value: 16.0.  `None` disables LoRA+ (default).
    pub loraplus_lr_ratio: Option<f32>,

    /// Use Cut Cross-Entropy for memory-efficient loss computation.
    ///
    /// When enabled, the training loop computes the cross-entropy loss directly from
    /// hidden states without materializing the full logits tensor.  This reduces memory
    /// by up to 37x for large vocabularies (e.g., 150K tokens in Qwen3.5).
    ///
    /// Covers packed and unpacked text batches. When it can't apply (image
    /// batches, a model with no usable LM head) the run warns once and
    /// computes the loss from the full logits; see
    /// `TrainingLoop::cut_cross_entropy_applies`.
    pub use_cut_cross_entropy: bool,

    /// Override the maximum sequence length used for sequence packing.
    ///
    /// When `None` (default), packing auto-computes the length from the p99 of
    /// sample lengths in the training dataset, rounded up to the next power of
    /// two and capped at `dataloader.max_seq_len`.
    ///
    /// When `Some(n)`, the explicit value `n` is used directly, bypassing the
    /// adaptive computation.  Useful for reproducibility or when the dataset p99
    /// heuristic overshoots (e.g., a single outlier sample inflates the bucket).
    pub pack_max_seq_len: Option<usize>,

    /// Distributed training configuration.
    /// When set, gradients are synchronized across multiple nodes after each
    /// accumulation cycle, enabling data-parallel training over a home cluster.
    #[cfg(feature = "distributed")]
    pub distributed: Option<pmetal_core::DistributedTrainingConfig>,
}

impl Default for TrainingLoopConfig {
    fn default() -> Self {
        Self {
            training: TrainingConfig::default(),
            dataloader: DataLoaderConfig::default(),
            log_every: 10,
            checkpoint_every: 500,
            eval_every: 100,
            use_sequence_packing: false,
            gradient_checkpointing: false,
            gradient_checkpointing_layers: 4,
            embedding_lr: None,
            loraplus_lr_ratio: None,
            use_cut_cross_entropy: false,
            pack_max_seq_len: None,
            #[cfg(feature = "distributed")]
            distributed: None,
        }
    }
}

impl TrainingLoopConfig {
    /// Check the settings the loop acts on, before it does.
    ///
    /// # Errors
    /// [`SftError::Config`] naming every setting that cannot work: the
    /// training section's own checks ([`pmetal_core::ConfigValidator`]), a
    /// zero batch size, a rank outside its world, and the rates.
    pub fn validate(&self) -> Result<()> {
        use pmetal_core::ConfigValidator;
        let mut problems = Vec::new();
        if let Err(e) = self.training.validate() {
            problems.push(e.to_string());
        }
        if self.dataloader.batch_size == 0 {
            problems.push("dataloader.batch_size must be at least 1".to_string());
        }
        if self.dataloader.world_size == 0
            || self.dataloader.rank >= self.dataloader.world_size.max(1)
        {
            problems.push(format!(
                "rank {} is not in a world of {}",
                self.dataloader.rank, self.dataloader.world_size
            ));
        }
        if let Some(lr) = self.embedding_lr
            && !(lr.is_finite() && lr >= 0.0)
        {
            problems.push(format!(
                "embedding_lr must be finite and non-negative, got {lr}"
            ));
        }
        if let Some(ratio) = self.loraplus_lr_ratio
            && !(ratio.is_finite() && ratio > 0.0)
        {
            problems.push(format!(
                "loraplus_lr_ratio must be finite and positive, got {ratio}"
            ));
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(SftError::Config(problems.join("; ")))
        }
    }
}

/// Statistics for a single training step.
#[derive(Debug, Clone)]
pub struct StepStats {
    /// Step number.
    pub step: usize,
    /// Loss value.
    pub loss: f32,
    /// Learning rate.
    pub learning_rate: f32,
    /// Tokens processed in this step.
    pub tokens: usize,
    /// Gradient norm (if computed).
    pub grad_norm: Option<f32>,
    /// Time taken for this step (ms).
    pub step_time_ms: u64,
}

/// Action signalled by the adaptive LR controller after processing a training step.
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveAction {
    /// No intervention needed.
    Continue,
    /// Restore weights from the best in-memory snapshot, reset optimizer momentum,
    /// and continue training with reduced LR.
    Rollback,
    /// Too many rollbacks — stop training and use the best checkpoint.
    EarlyStop,
    /// External request to save a checkpoint (training continues).
    SaveCheckpoint,
    /// External request: restore best weights, save checkpoint, exit.
    GracefulStop,
}

/// Training loop that connects all components.
pub struct TrainingLoop {
    /// Configuration.
    pub(crate) config: TrainingLoopConfig,
    /// Current step: micro-batches taken so far.
    pub(crate) step: usize,
    /// Current epoch.
    pub(crate) epoch: usize,
    /// Running loss (EMA).
    pub(crate) running_loss: f64,
    /// Total tokens processed.
    pub(crate) total_tokens: usize,
    /// The micro-batches of the optimizer step in progress.
    pub(crate) accumulator: step::GradAccumulator,
    /// Tokens accumulated since last log (for throughput calculation).
    pub(crate) tokens_since_log: usize,
    /// Time of last log (for throughput calculation).
    pub(crate) last_log_time: Option<std::time::Instant>,
    /// Step number at the last log boundary.
    pub(crate) last_log_step: Option<usize>,
    /// Background thread handle for the most recent async checkpoint write.
    ///
    /// Checkpoint I/O (safetensors serialization + JSON metadata) is offloaded to a
    /// background thread so the training loop is not stalled by disk latency.
    /// The handle is polled before each new checkpoint spawn; errors are logged as
    /// warnings rather than terminating training.
    pub(crate) pending_checkpoint: Option<std::thread::JoinHandle<std::result::Result<(), String>>>,
    /// Training callbacks for metrics logging, progress reporting, and dashboard.
    pub(crate) callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>>,
    /// Optional adaptive LR controller for reactive scheduling.
    pub(crate) adaptive_lr: Option<crate::adaptive_lr::AdaptiveLrController>,
    /// Adaptive-adjusted LR for the next step (set by `apply_adaptive_lr`).
    pub(crate) adaptive_lr_override: Option<f32>,
    /// In-memory snapshot of the best LoRA weights for rollback.
    /// LoRA params are typically a few MB, so this is cheap to hold in memory.
    pub(crate) best_lora_snapshot: Option<FlattenedModuleParam>,
    /// When set, each in-memory snapshot is also persisted to this directory
    /// as `best_snapshot.safetensors` so it survives process interruptions.
    pub(crate) snapshot_persist_dir: Option<std::path::PathBuf>,
    /// Distributed gradient synchronization bridge.
    /// When present, gradients are all-reduced across nodes after each accumulation cycle.
    #[cfg(feature = "distributed")]
    pub(crate) distributed: Option<crate::distributed_bridge::DistributedGradientSync>,
    /// Optimizer steps the LR schedule spans, set by [`Self::plan_schedule`]
    /// when a run starts.
    pub(crate) schedule_total_steps: Option<usize>,
    /// Micro-batches (calls that advance `step`) per optimizer step.
    pub(crate) micro_steps_per_update: usize,
    /// Set once the run has said why cut cross-entropy is not in effect, so
    /// the reason is given once rather than every step.
    pub(crate) cce_unavailable_reported: std::cell::Cell<bool>,
    /// What [`Self::resume_from`] loaded, for [`Self::run`] to continue from.
    pub(crate) pending_resume: Option<TrainState>,
}

impl TrainingLoop {
    /// Create a training loop.
    ///
    /// # Errors
    /// [`SftError::Config`] if the configuration cannot work
    /// ([`TrainingLoopConfig::validate`]): a loop is never built on settings
    /// that would fail or silently do nothing partway through a run.
    pub fn new(mut config: TrainingLoopConfig) -> Result<Self> {
        config.validate()?;
        config.log_every = config.log_every.max(1);

        Ok(Self {
            config,
            step: 0,
            epoch: 0,
            running_loss: 0.0,
            total_tokens: 0,
            accumulator: step::GradAccumulator::default(),
            tokens_since_log: 0,
            last_log_time: None,
            last_log_step: None,
            pending_checkpoint: None,
            callbacks: Vec::new(),
            adaptive_lr: None,
            adaptive_lr_override: None,
            best_lora_snapshot: None,
            snapshot_persist_dir: None,
            #[cfg(feature = "distributed")]
            distributed: None,
            schedule_total_steps: None,
            micro_steps_per_update: 1,
            cce_unavailable_reported: std::cell::Cell::new(false),
            pending_resume: None,
        })
    }

    /// Add a training callback for metrics logging or dashboard integration.
    pub fn add_callback(&mut self, callback: Box<dyn pmetal_core::TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Apply direct training-loop gradient checkpointing configuration.
    ///
    /// The orchestrator already configures checkpointing on loaded models, but
    /// callers that use `TrainingLoop` directly rely on the loop entry points to
    /// keep dispatch modes consistent.
    pub(crate) fn apply_gradient_checkpointing<M>(&self, model: &mut M, mode: &str)
    where
        M: TrainableModel,
    {
        if !self.config.gradient_checkpointing {
            return;
        }

        if !model.supports_gradient_checkpointing() {
            if mode.is_empty() {
                tracing::warn!(
                    "Gradient checkpointing requested but not supported by this model; \
                     continuing without it."
                );
            } else {
                tracing::warn!(
                    "{}: gradient checkpointing requested but not supported by this model; \
                     continuing without it.",
                    mode
                );
            }
            return;
        }

        // One decoder layer is the checkpoint unit, matching what PyTorch
        // does. `--gradient-checkpointing-layers` predates that and
        // has never selected anything; say so rather than echo it back as if
        // it had.
        let layers = self.config.gradient_checkpointing_layers.max(1);
        if layers != 1 {
            tracing::warn!(
                layers_per_block = layers,
                "--gradient-checkpointing-layers has no effect: one decoder layer is the \
                 checkpoint unit, as in PyTorch. The flag is accepted for \
                 compatibility and will be removed."
            );
        }
        model.enable_gradient_checkpointing(layers);

        if mode.is_empty() {
            tracing::info!("Gradient checkpointing enabled, one decoder layer per block");
        } else {
            tracing::info!("{mode}: gradient checkpointing enabled, one decoder layer per block");
        }
    }

    /// The parameter groups every run trains: the configured weight decay,
    /// and the embedding and LoRA+ rates when they are set.
    ///
    /// `output_projection` is where the model's output head sits
    /// ([`pmetal_lora::TrainableModel::output_projection_path`]), so the head trains at
    /// the embedding rate whatever it is called.
    pub fn parameter_groups(
        &self,
        output_projection: Option<String>,
    ) -> crate::ParameterGroupConfig {
        let training = &self.config.training;
        let base_lr = training.learning_rate as f32;
        let mut groups = crate::ParameterGroupConfig::new(training.weight_decay as f32)
            .with_output_projection(output_projection);
        if let Some(emb_lr) = self.config.embedding_lr {
            groups = groups.with_embedding_lr(emb_lr, base_lr);
            tracing::info!("Using separate embedding LR: {emb_lr:.2e} (base: {base_lr:.2e})");
        }
        if let Some(ratio) = self.config.loraplus_lr_ratio {
            groups = groups.with_loraplus_lr_ratio(ratio);
        }
        groups
    }

    /// The optimizer the loop trains with: the kind `training.optimizer`
    /// names, at the configured learning rate, over [`Self::parameter_groups`].
    pub fn build_optimizer(&self, output_projection: Option<String>) -> crate::ParamGroupOptimizer {
        let training = &self.config.training;
        let base_lr = training.learning_rate as f32;
        tracing::info!(
            "Optimizer: {:?}, lr={base_lr:.2e}, weight_decay={}",
            training.optimizer,
            training.weight_decay
        );
        crate::ParamGroupOptimizer::new(
            training.optimizer,
            base_lr,
            self.parameter_groups(output_projection),
        )
    }

    /// Start a new logging interval from the current training step.
    pub(crate) fn reset_log_interval(&mut self) {
        self.last_log_time = Some(std::time::Instant::now());
        self.last_log_step = Some(self.step);
        self.tokens_since_log = 0;
    }

    /// Consume the current logging interval and return averaged throughput metrics.
    pub(crate) fn take_log_interval_metrics(
        &mut self,
        now: std::time::Instant,
    ) -> LogIntervalMetrics {
        let total_ms = self
            .last_log_time
            .map(|last| now.duration_since(last).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let tokens = self.tokens_since_log;
        let steps = self
            .last_log_step
            .map(|last| self.step.saturating_sub(last))
            .unwrap_or(1)
            .max(1);
        let tok_sec = if total_ms > 0.0 {
            tokens as f64 / (total_ms / 1000.0)
        } else {
            0.0
        };

        self.last_log_time = Some(now);
        self.last_log_step = Some(self.step);
        self.tokens_since_log = 0;

        LogIntervalMetrics {
            steps,
            tokens,
            total_ms,
            tok_sec,
        }
    }

    /// Enable adaptive LR control with the given config.
    pub fn enable_adaptive_lr(&mut self, config: crate::adaptive_lr::AdaptiveLrConfig) {
        self.adaptive_lr = Some(crate::adaptive_lr::AdaptiveLrController::new(config));
    }

    /// Set the distributed gradient sync bridge for multi-node training.
    #[cfg(feature = "distributed")]
    pub fn set_distributed(&mut self, sync: crate::distributed_bridge::DistributedGradientSync) {
        // Each rank trains its own share of the data.
        self.config.dataloader.rank = sync.rank();
        self.config.dataloader.world_size = sync.world_size();
        self.distributed = Some(sync);
    }

    /// Agree on a step's loss across nodes, so every node feeds the adaptive
    /// LR controller the same value. Only `loss` changes: `train_step` has
    /// already folded the local loss into the running average.
    #[cfg(feature = "distributed")]
    pub(crate) fn sync_step_loss(&self, stats: StepStats) -> Result<StepStats> {
        let Some(dist) = &self.distributed else {
            return Ok(stats);
        };
        let loss = crate::distributed_bridge::block_on(dist.sync_loss(stats.loss))?;
        Ok(StepStats { loss, ..stats })
    }

    /// A single machine's loss is already agreed on.
    #[cfg(not(feature = "distributed"))]
    pub(crate) fn sync_step_loss(&self, stats: StepStats) -> Result<StepStats> {
        Ok(stats)
    }

    /// Set the directory where best-loss snapshots are persisted to disk.
    ///
    /// When set, every call to `snapshot_best_weights` also writes
    /// `best_snapshot.safetensors` to this directory so the snapshot survives
    /// process interruptions and is available on resume.
    pub fn set_snapshot_persist_dir(&mut self, dir: std::path::PathBuf) {
        self.snapshot_persist_dir = Some(dir);
    }

    /// Enable adaptive LR with control file for TUI communication.
    pub fn enable_adaptive_lr_with_control(
        &mut self,
        config: crate::adaptive_lr::AdaptiveLrConfig,
        control_file: std::path::PathBuf,
    ) {
        self.adaptive_lr = Some(
            crate::adaptive_lr::AdaptiveLrController::new(config).with_control_file(control_file),
        );
    }

    /// Take all callbacks out of the training loop (transfers ownership back to caller).
    pub fn take_callbacks(&mut self) -> Vec<Box<dyn pmetal_core::TrainingCallback>> {
        std::mem::take(&mut self.callbacks)
    }

    /// Returns true if any callback has requested training to stop.
    pub(crate) fn check_cancelled(&self) -> bool {
        self.callbacks.iter().any(|cb| cb.should_stop())
    }

    /// Get current learning rate based on scheduler.
    ///
    /// Delegates to the canonical `pmetal_core::LearningRateScheduler` so all
    /// trainers share a single, consistent LR computation path.
    pub fn get_learning_rate(&self) -> f32 {
        // If adaptive controller has set an override, use it
        if let Some(lr) = self.adaptive_lr_override {
            return lr;
        }

        self.get_scheduled_lr()
    }

    /// Get the base scheduled LR (ignoring adaptive adjustments).
    ///
    /// The schedule runs over optimizer steps, not micro-batches: with
    /// gradient accumulation the rate holds across the micro-batches of one
    /// step. Before [`Self::plan_schedule`] has run, the schedule spans
    /// `max_steps` if set, and otherwise holds the peak rate after warmup.
    pub(crate) fn get_scheduled_lr(&self) -> f32 {
        let cfg = &self.config.training;
        let total_steps = self.schedule_total_steps.or(cfg.max_steps).unwrap_or(0);
        pmetal_core::LearningRateScheduler::for_training(cfg, total_steps)
            .get_lr(self.optimizer_steps_taken()) as f32
    }

    /// Optimizer steps taken so far.
    pub fn optimizer_steps_taken(&self) -> usize {
        self.step / self.micro_steps_per_update.max(1)
    }

    /// Whether `max_steps` optimizer steps have been taken.
    pub(crate) fn reached_max_steps(&self) -> bool {
        self.config
            .training
            .max_steps
            .is_some_and(|max| self.optimizer_steps_taken() >= max)
    }

    /// Micro-batches the dataloader yields from `dataset` in one epoch.
    pub(crate) fn micro_batches_per_epoch(&self, dataset: &TrainingDataset) -> usize {
        DataLoader::new(dataset.clone(), self.config.dataloader.clone(), None).num_batches()
    }

    /// Size the learning-rate schedule for a run of `micro_batches_per_epoch`
    /// micro-batches an epoch, `micro_steps_per_update` of them per optimizer
    /// step, as the reference trainers do: `max_steps` when set, else every
    /// epoch's micro-batches over the accumulation. Returns the run's length
    /// in micro-batches, the unit `step` and the callbacks count in.
    pub(crate) fn plan_schedule(
        &mut self,
        micro_batches_per_epoch: usize,
        micro_steps_per_update: usize,
    ) -> usize {
        let per_update = micro_steps_per_update.max(1);
        let cfg = &self.config.training;
        // A partial accumulation carries into the next epoch.
        let total =
            pmetal_core::total_training_steps(cfg, micro_batches_per_epoch, per_update, false);
        let warmup = pmetal_core::warmup_steps_for(cfg, total);
        tracing::info!(
            "LR schedule: {:?} over {total} optimizer steps ({} epoch(s) of {micro_batches_per_epoch} \
             micro-batches, {per_update} per step{}), {warmup} warmup steps, peak lr {:.2e}",
            cfg.lr_scheduler,
            cfg.num_epochs.max(1),
            if cfg.max_steps.is_some() {
                ", capped by max_steps"
            } else {
                ""
            },
            cfg.learning_rate,
        );
        self.schedule_total_steps = Some(total);
        self.micro_steps_per_update = per_update;
        if let Some(ref mut ctrl) = self.adaptive_lr {
            // The controller is fed once per micro-batch.
            ctrl.set_total_steps(total * per_update);
            ctrl.set_warmup_steps(warmup * per_update);
        }
        total * per_update
    }

    /// Feed loss to the adaptive LR controller and update the override.
    ///
    /// Returns an `AdaptiveAction` indicating whether the training loop should
    /// continue normally, roll back to the best checkpoint, or stop early.
    ///
    /// Call this after each training step.
    pub fn apply_adaptive_lr(&mut self, loss: f64) -> AdaptiveAction {
        let scheduled = self.get_scheduled_lr() as f64;
        let step = self.step;
        if let Some(ref mut ctrl) = self.adaptive_lr {
            let (adjusted, event) = ctrl.step(step, loss, scheduled);
            self.adaptive_lr_override = Some(adjusted as f32);

            // Determine action from event
            let action = match &event {
                crate::adaptive_lr::LrEvent::RollbackTriggered { .. } => AdaptiveAction::Rollback,
                crate::adaptive_lr::LrEvent::EarlyStop { .. } => AdaptiveAction::EarlyStop,
                crate::adaptive_lr::LrEvent::ControlCheckpoint => AdaptiveAction::SaveCheckpoint,
                crate::adaptive_lr::LrEvent::ControlStop => AdaptiveAction::GracefulStop,
                _ => AdaptiveAction::Continue,
            };

            // Log non-scheduled events
            if !matches!(event, crate::adaptive_lr::LrEvent::Scheduled) {
                for cb in &mut self.callbacks {
                    cb.on_lr_event(&format!("{event}"));
                }
            }

            action
        } else {
            AdaptiveAction::Continue
        }
    }

    /// Take a snapshot of the model's LoRA weights as the best checkpoint.
    ///
    /// Called when the adaptive LR controller indicates loss has improved.
    /// The snapshot is held in memory for fast rollback (LoRA params are small).
    /// When `snapshot_persist_dir` is configured, the snapshot is also written to
    /// disk as `best_snapshot.safetensors` for durability across restarts.
    pub fn snapshot_best_weights<M: TrainableModel>(&mut self, model: &M) {
        let params = model.lora_parameters();
        tracing::debug!(
            "Snapshot: saved best LoRA weights at step {} ({} params, ~{:.1} MB)",
            self.step,
            params.len(),
            params.values().map(|a| a.nbytes()).sum::<usize>() as f64 / 1_048_576.0,
        );

        // Persist to disk if a snapshot directory is configured.
        if let Some(ref dir) = self.snapshot_persist_dir {
            if let Err(e) = crate::checkpoint::save_best_snapshot(dir, &params) {
                tracing::warn!("Failed to persist best snapshot to disk: {e}");
            }
        }

        self.best_lora_snapshot = Some(params);
    }

    /// Restore model weights from the best in-memory snapshot.
    ///
    /// Returns `true` if weights were successfully restored.
    pub fn restore_best_weights<M: TrainableModel>(&mut self, model: &mut M) -> bool {
        if let Some(ref snapshot) = self.best_lora_snapshot {
            model.set_lora_parameters(snapshot);

            // Notify the controller that rollback is complete
            if let Some(ref mut ctrl) = self.adaptive_lr {
                ctrl.on_rollback_complete();
            }

            // Reset running loss to approximate the best loss
            if let Some(ref ctrl) = self.adaptive_lr {
                self.running_loss = ctrl.best_ema_loss();
            }

            true
        } else {
            tracing::warn!("Rollback requested but no best snapshot available");
            false
        }
    }

    /// Check if the adaptive LR controller recommends snapshotting (new best EMA loss).
    pub fn should_snapshot_best(&self) -> bool {
        if let Some(ref ctrl) = self.adaptive_lr {
            // The controller's should_snapshot_best was already called in apply_adaptive_lr,
            // so we check if the current EMA is the best we've seen
            ctrl.best_ema_step() == self.step
        } else {
            false
        }
    }

    /// Whether this rank should perform checkpoint writes.
    ///
    /// Non-distributed builds always write; distributed builds only write
    /// on the master rank. Shared across every training loop variant so
    /// the `cfg(feature = "distributed")` guard lives in one place.
    #[inline]
    pub fn is_checkpoint_master(&self) -> bool {
        #[cfg(feature = "distributed")]
        {
            self.distributed.as_ref().is_none_or(|d| d.is_master())
        }
        #[cfg(not(feature = "distributed"))]
        {
            true
        }
    }

    /// Whether a step's loss goes through cut cross-entropy: asked for, and
    /// possible for this model and configuration.
    ///
    /// When it was asked for and is not possible, the run says why, once, and
    /// computes the loss from the full logits; the run is not refused, since
    /// CCE changes memory and not the loss. Every path decides here, so none
    /// of them falls back without saying so. `images` is for a batch with a
    /// vision input, which goes through the vision forward, which returns
    /// logits.
    pub(crate) fn cut_cross_entropy_applies<M: TrainableModel>(
        &self,
        model: &M,
        images: bool,
    ) -> bool {
        if !self.config.use_cut_cross_entropy {
            return false;
        }
        let reason = if images {
            "image batches go through the vision forward, which returns logits"
        } else if model.lm_head().is_none() {
            "the model has no LM head in the form it computes from (or an adapter on the \
             head applies dropout, which no single weight reproduces)"
        } else {
            return true;
        };
        self.cut_cross_entropy_unavailable(reason);
        false
    }

    /// Say, once per run, that `--cut-cross-entropy` is not in effect and why.
    pub(crate) fn cut_cross_entropy_unavailable(&self, reason: &str) {
        if !self.cce_unavailable_reported.replace(true) {
            tracing::warn!(
                "Cut cross-entropy was requested, but {reason}; this run computes the loss \
                 from the full logits"
            );
        }
    }

    /// The loss's shape for a step on `model`: the config's smoothing, z-loss
    /// and NEFTune noise, and whether cut cross-entropy applies.
    pub(crate) fn loss_spec<M: TrainableModel>(&self, model: &M, images: bool) -> LossSpec {
        let training = &self.config.training;
        LossSpec {
            label_smoothing: training.label_smoothing as f32,
            z_loss: training.z_loss as f32,
            neftune_alpha: training.neftune_noise_alpha.map(|a| a as f32),
            cut_cross_entropy: self.cut_cross_entropy_applies(model, images),
        }
    }

    /// Evaluate the model on a dataset.
    ///
    /// Returns comprehensive evaluation metrics including:
    /// - Average loss across all batches
    /// - Perplexity (exp(loss))
    /// - Token-level accuracy (correct next-token predictions)
    ///
    /// The loss is the plain next-token cross-entropy, whatever smoothing or
    /// z-loss the training loss adds: evaluation measures the model, not the
    /// regularizer.
    pub fn evaluate<M>(&self, model: &mut M, dataset: &TrainingDataset) -> Result<EvalMetrics>
    where
        M: TrainableModel,
    {
        let mut dataloader = self.eval_loader(std::sync::Arc::new(dataset.clone()));
        self.evaluate_with(model, &mut dataloader)
    }

    /// [`evaluate`](Self::evaluate) over a loader built once and reused, so a
    /// run that evaluates every few steps does not copy the dataset each time.
    pub(crate) fn evaluate_with<M>(
        &self,
        model: &mut M,
        dataloader: &mut DataLoader,
    ) -> Result<EvalMetrics>
    where
        M: TrainableModel,
    {
        dataloader.reset(None);

        let mut total_loss = 0.0;
        let mut total_correct = 0_u64;
        let mut total_tokens = 0_u64;
        let mut num_batches = 0;

        while let Some(batch) = dataloader
            .try_next_batch()
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
        {
            let logits = model
                .forward(&batch.input_ids, None)
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

            let loss = pmetal_runtime::training::causal_lm_loss(
                &logits,
                &batch.labels,
                step::IGNORE_INDEX,
            );
            loss.eval();
            total_loss += loss.item::<f32>() as f64;

            // Compute token-level accuracy
            let (correct, tokens) = Self::compute_accuracy(&logits, &batch.labels)?;
            total_correct += correct;
            total_tokens += tokens;

            num_batches += 1;
        }

        let avg_loss = if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        };

        // Perplexity = exp(loss)
        // Clamp to avoid overflow for very high losses
        let perplexity = if avg_loss < 100.0 {
            avg_loss.exp()
        } else {
            f64::MAX
        };

        // Accuracy as percentage
        let accuracy = if total_tokens > 0 {
            Some((total_correct as f64 / total_tokens as f64) * 100.0)
        } else {
            None
        };

        Ok(EvalMetrics {
            loss: avg_loss,
            perplexity,
            accuracy,
            custom: std::collections::HashMap::new(),
        })
    }

    /// Compute token-level accuracy.
    ///
    /// Counts how many next-token predictions match the labels,
    /// ignoring padding tokens (label = -100).
    ///
    /// Returns (correct_count, total_valid_tokens).
    fn compute_accuracy(logits: &Array, labels: &Array) -> Result<(u64, u64)> {
        let seq_len = logits.dim(1);
        let vocab_size = logits.dim(2);

        // Shift: logits[:-1] predicts labels[1:]
        let shift_logits = logits.index((.., ..seq_len - 1, ..));
        let shift_labels = labels.index((.., 1..));

        // Get predictions (argmax over vocab dimension)
        let flat_logits = shift_logits.reshape(&[-1, vocab_size]);
        let predictions = argmax_axis(&flat_logits, -1, false);

        let flat_labels = shift_labels.reshape(&[-1]);

        // Valid tokens: label != -100.
        let valid_mask = flat_labels.ne(&Array::from_int(step::IGNORE_INDEX));
        let valid_mask_f32 = valid_mask.as_type::<f32>();
        let correct = predictions.eq(&flat_labels).as_type::<f32>();
        let correct_sum = correct.multiply(&valid_mask_f32).sum(None);
        let total_valid = valid_mask_f32.sum(None);
        // One evaluation for both counts.
        pmetal_runtime::compat::transforms::eval([&total_valid, &correct_sum])?;

        Ok((correct_sum.item_f32() as u64, total_valid.item_f32() as u64))
    }

    /// Get current training step: micro-batches taken.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Get current epoch.
    pub fn current_epoch(&self) -> usize {
        self.epoch
    }

    /// Get running loss.
    pub fn current_loss(&self) -> f64 {
        self.running_loss
    }

    /// Get total tokens processed.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Set step (for testing and checkpoint restore).
    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }

    /// Set epoch (for testing and checkpoint restore).
    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }

    /// Save a checkpoint of the model's weights, offloading the I/O to a
    /// background thread. A run's own checkpoints also hold the optimizer's
    /// state and the data position; this is the weights alone.
    ///
    /// The LoRA parameter arrays are evaluated (materialized) on the calling thread
    /// before the background thread is spawned. This ensures the GPU computation
    /// graph is resolved here, and only the file-write work crosses the thread boundary.
    ///
    /// Errors in the background write are logged as warnings. The returned `PathBuf`
    /// is the expected step directory path (computed eagerly); the actual write may
    /// still be in flight.
    pub fn save_checkpoint<M>(
        &mut self,
        model: &M,
        manager: &CheckpointManager,
        is_best: bool,
        eval_loss: Option<f64>,
    ) -> Result<std::path::PathBuf>
    where
        M: TrainableModel,
    {
        self.save_checkpoint_with_state(model, manager, is_best, eval_loss, None)
    }

    /// [`save_checkpoint`](Self::save_checkpoint) with the state a resume
    /// continues from.
    pub(crate) fn save_checkpoint_with_state<M>(
        &mut self,
        model: &M,
        manager: &CheckpointManager,
        is_best: bool,
        eval_loss: Option<f64>,
        state: Option<TrainState>,
    ) -> Result<std::path::PathBuf>
    where
        M: TrainableModel,
    {
        let lora_params = model.lora_parameters();

        let mut metadata = CheckpointMetadata::new(
            self.step,
            self.epoch,
            self.running_loss,
            self.get_learning_rate() as f64,
        );

        if let Some(loss) = eval_loss {
            metadata = metadata.with_best_val_loss(loss);
        }

        // Compute the expected output path so callers can log it immediately.
        let step_dir = manager
            .checkpoint_dir()
            .join(format!("step_{}", metadata.step));

        tracing::info!(
            "Queuing async checkpoint write for step {} to {:?}",
            self.step,
            step_dir
        );

        // Offload the blocking I/O to a background thread.
        self.spawn_async_checkpoint(lora_params, metadata, manager, is_best, state)?;

        Ok(step_dir)
    }

    /// Poll the pending checkpoint background thread.
    ///
    /// If the previous checkpoint I/O has completed, this checks for errors and
    /// logs a warning if the write failed. Must be called before spawning a new
    /// checkpoint thread to avoid unbounded handle accumulation.
    fn poll_pending_checkpoint(&mut self) {
        if let Some(handle) = self.pending_checkpoint.take() {
            if handle.is_finished() {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        tracing::warn!("Async checkpoint write failed: {}", e);
                    }
                    Err(_) => {
                        tracing::warn!("Async checkpoint thread panicked");
                    }
                }
            } else {
                // Still running — put it back so we don't lose the handle
                self.pending_checkpoint = Some(handle);
            }
        }
    }

    /// Spawn checkpoint I/O on a background thread so the training loop is not stalled
    /// by safetensors serialization or metadata JSON writes.
    ///
    /// The method:
    /// 1. Forces evaluation of every array the checkpoint holds (weights and
    ///    optimizer state) so the background thread only performs I/O.
    /// 2. Converts the `Rc<str>`-keyed map to a `String`-keyed map that is `Send`.
    /// 3. Clones the metadata and extracts the manager state needed on the background thread.
    /// 4. Polls the previous handle for completion/errors before spawning the new one.
    ///
    /// Failures are logged as warnings rather than propagated, so a transient I/O error
    /// does not abort training.
    fn spawn_async_checkpoint(
        &mut self,
        lora_params: FlattenedModuleParam,
        metadata: CheckpointMetadata,
        manager: &CheckpointManager,
        is_best: bool,
        state: Option<TrainState>,
    ) -> Result<()> {
        // Poll the previous handle so we surface any errors and free the handle.
        self.poll_pending_checkpoint();

        {
            let mut arrays: Vec<&Array> = lora_params.values().collect();
            if let Some(opt) = state.as_ref().and_then(|s| s.optimizer.as_ref()) {
                arrays.extend(opt.arrays());
            }
            if !arrays.is_empty() {
                pmetal_runtime::compat::transforms::eval(arrays)?;
            }
        }

        // Convert Rc<str> keys → String keys so the map is Send.
        let params_send: std::collections::HashMap<String, Array> = lora_params
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        // Clone the manager data we need on the background thread.
        let checkpoint_dir = manager.checkpoint_dir().to_path_buf();
        let max_checkpoints = manager.max_checkpoints_limit();
        let save_best_flag = manager.save_best_flag();

        let step_for_log = metadata.step;

        let handle = std::thread::spawn(move || -> std::result::Result<(), String> {
            // Reconstruct a local CheckpointManager on the background thread
            // using the cloned configuration data.
            let bg_manager =
                CheckpointManager::from_parts(checkpoint_dir, max_checkpoints, save_best_flag);

            bg_manager
                .save_checkpoint_owned(params_send, &metadata, is_best, state.as_ref())
                .map_err(|e| e.to_string())?;

            tracing::info!("Async checkpoint write complete (step {})", step_for_log);
            Ok(())
        });

        self.pending_checkpoint = Some(handle);
        Ok(())
    }

    /// Wait for the checkpoint being written, if one is, and report how it went.
    pub(crate) fn finish_pending_checkpoint(&mut self) {
        if let Some(handle) = self.pending_checkpoint.take() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!("Checkpoint write failed: {e}"),
                Err(_) => tracing::warn!("Checkpoint thread panicked"),
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LogIntervalMetrics {
    pub(crate) steps: usize,
    pub(crate) tokens: usize,
    pub(crate) total_ms: f64,
    pub(crate) tok_sec: f64,
}

impl Drop for TrainingLoop {
    fn drop(&mut self) {
        if let Some(handle) = self.pending_checkpoint.take() {
            // Wait for the final checkpoint write to complete so we don't
            // exit the process with a half-written safetensors file.
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!("Final async checkpoint failed: {e}"),
                Err(_) => tracing::warn!("Final async checkpoint thread panicked"),
            }
        }
    }
}
