//! The loop itself: epochs of micro-batches, an optimizer step every
//! `gradient_accumulation_steps` of them, and what follows each step.

use std::sync::Arc;
use std::time::Instant;

use pmetal_data::{DataLoader, TrainingDataset};
use pmetal_lora::TrainableModel;
use pmetal_runtime::compat::{
    Array, ModuleParametersExt,
    optimizers::{Optimizer, StatefulOptimizer, Updatable},
    transforms,
};

use super::source::{BatchSource, BatchView};
use super::step::loss_and_grads;
use super::{AdaptiveAction, StepStats, TrainingLoop};
use crate::checkpoint::TrainState;
use crate::step_check::{check_grad_norm, check_step};
use crate::{CheckpointManager, ParamGroupOptimizer, Result, SftError};

/// What a micro-batch did.
pub(crate) struct StepOutcome {
    pub(crate) stats: StepStats,
    /// Whether it completed an optimizer step.
    pub(crate) updated: bool,
}

/// Whether the loop goes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Continue,
    Stop,
}

/// What a run holds besides the loop's own counters.
struct Run<'a, M: TrainableModel> {
    model: &'a mut M,
    optimizer: ParamGroupOptimizer,
    /// Where the model's output head sits, for rebuilding the optimizer.
    output_projection: Option<String>,
    source: BatchSource,
    checkpoints: Option<&'a CheckpointManager>,
    eval: Option<DataLoader>,
    best_eval_loss: f64,
}

impl TrainingLoop {
    /// Run training: `num_epochs` over `train_dataset` (or `max_steps`
    /// optimizer steps), evaluating on `eval_dataset` and checkpointing on the
    /// configured intervals.
    ///
    /// The batches are padded, or packed when `use_sequence_packing` is set
    /// and the model and data allow it; see the module documentation for the
    /// rest of what a step is. When the optimizer keeps separate evaluation
    /// weights (schedule-free AdamW), the model holds them when this returns.
    pub fn run<M>(
        &mut self,
        model: &mut M,
        train_dataset: TrainingDataset,
        eval_dataset: Option<TrainingDataset>,
        checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<()>
    where
        M: TrainableModel,
    {
        let accum = self.config.training.gradient_accumulation_steps.max(1);
        let num_epochs = self.config.training.num_epochs;
        let base_seed = self.config.dataloader.seed;

        // Gradient checkpointing must be on before any forward is traced.
        self.apply_gradient_checkpointing(model, "");

        let output_projection = model.output_projection_path();
        let optimizer = self.build_optimizer(output_projection.clone());
        let source = self.build_source(model, Arc::new(train_dataset))?;
        let total_micro_batches = self.plan_schedule(source.num_batches(), accum);

        tracing::info!(
            "Starting training: {} trainable params, batch_size={}, grad_accum={}, {} micro-batches per epoch",
            model.num_trainable_params(),
            self.config.training.batch_size,
            accum,
            source.num_batches(),
        );

        let eval = eval_dataset.map(|d| self.eval_loader(Arc::new(d)));
        let mut run = Run {
            model,
            optimizer,
            output_projection,
            source,
            checkpoints: checkpoint_manager,
            eval,
            best_eval_loss: f64::MAX,
        };

        // Continue from a checkpoint [`Self::resume_from`] loaded.
        let mut mid_epoch = false;
        if let Some(state) = self.pending_resume.take() {
            if let Some(optimizer_state) = &state.optimizer {
                run.optimizer
                    .import_state(optimizer_state)
                    .map_err(SftError::Mlx)?;
                // Schedule-free AdamW's checkpoints hold the averaged weights.
                run.optimizer.train_weights(run.model);
            } else {
                tracing::warn!(
                    "The checkpoint holds no optimizer state (it predates it): the optimizer \
                     starts from zero moments at step {}",
                    self.step
                );
            }
            if let Some(loader) = state.loader {
                run.source.restore(loader)?;
                mid_epoch = true;
            }
        }

        self.reset_log_interval();

        #[cfg(feature = "distributed")]
        if let Some(ref dist) = self.distributed {
            tracing::info!(
                "Distributed training: rank={}/{}, waiting at start barrier...",
                dist.rank(),
                dist.world_size()
            );
            crate::distributed_bridge::block_on(dist.barrier())?;
        }

        let first_epoch = self.epoch;
        'epochs: for epoch in first_epoch..num_epochs {
            self.epoch = epoch;
            tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);

            #[cfg(feature = "distributed")]
            if epoch > first_epoch {
                if let Some(ref dist) = self.distributed {
                    crate::distributed_bridge::block_on(dist.barrier())?;
                }
            }

            // A loader's order is a function of its seed, so each epoch needs
            // its own: the configured seed would serve the same order every
            // epoch. A resumed epoch keeps the order and position it was
            // restored to.
            if mid_epoch {
                mid_epoch = false;
            } else {
                run.source.reset(base_seed + epoch as u64);
            }

            while let Some(micro) = run.source.next()? {
                run.optimizer.set_learning_rate(self.get_learning_rate());
                let outcome = self.micro_step(run.model, micro.view(), &mut run.optimizer)?;
                if outcome.updated
                    && self.after_update(&mut run, outcome.stats, total_micro_batches)?
                        == Flow::Stop
                {
                    break 'epochs;
                }
            }
        }

        // Micro-batches left over from the last epoch's accumulation are one
        // more, smaller, optimizer step rather than lost.
        if self.accumulator.micro_batches() > 0 && !self.reached_max_steps() {
            tracing::info!(
                "Taking the last optimizer step on the {} micro-batch(es) left over",
                self.accumulator.micro_batches()
            );
            run.optimizer.set_learning_rate(self.get_learning_rate());
            let stats = self.flush_step(run.model, &mut run.optimizer)?;
            self.after_update(&mut run, stats, total_micro_batches)?;
        }

        // Hand back the weights to evaluate and save.
        run.optimizer.eval_weights(run.model);
        let trainable = run.model.flatten_trainable_params();
        transforms::eval(trainable.values())?;
        self.finish_pending_checkpoint();

        tracing::info!(
            "Training complete: {} steps, {:.4} final loss",
            self.step,
            self.running_loss
        );
        Ok(())
    }

    /// The source `dataset` is trained from: packed rows when asked for and
    /// possible, else padded batches.
    fn build_source<M: TrainableModel>(
        &self,
        model: &M,
        dataset: Arc<TrainingDataset>,
    ) -> Result<BatchSource> {
        let padded =
            |dataset: Arc<TrainingDataset>| BatchSource::padded(dataset, &self.config.dataloader);
        if !self.config.use_sequence_packing {
            return Ok(padded(dataset));
        }
        // A gated delta net carries its state from one packed sequence into
        // the next: the block-diagonal mask only reaches attention, so the
        // second sequence in a row would train on the first one's state.
        // (A Mamba-2 layer resets at the position ids' restarts and does not
        // count: `has_recurrent_layers` says false for it.)
        if model.has_recurrent_layers() {
            tracing::info!(
                "Sequence packing is off for this model: its recurrent layers would carry \
                 state across packed sequence boundaries, which the attention mask cannot \
                 stop and they cannot reset. Training on unpacked batches."
            );
            return Ok(padded(dataset));
        }
        let (source, stats) = BatchSource::packed(
            &dataset,
            &self.config.dataloader,
            self.config.pack_max_seq_len,
        )?;
        tracing::info!(
            "Packing: {} sequences → {} rows, {:.1}% efficiency, avg {:.1} seqs/row",
            stats.num_sequences,
            stats.num_batches,
            stats.efficiency * 100.0,
            stats.avg_sequences_per_batch
        );
        if stats.max_sequences_per_batch <= 1 {
            tracing::info!(
                "Sequence packing produced only single-sequence rows; training on unpacked \
                 batches"
            );
            return Ok(padded(dataset));
        }
        // The block-diagonal mask keeps one packed sequence from attending to
        // another; a model that drops the position IDs additionally runs the
        // second sequence in a row at positions continuing from wherever the
        // first ended. Plain RoPE absorbs that — it rotates by the difference
        // between positions, and inside a sealed-off block a uniform shift
        // cancels — but anything reading the absolute position does not.
        if !model.supports_packed_positions() {
            tracing::warn!(
                "Sequence packing is on, but this model ignores the position IDs, so packed \
                 sequences run at positions continuing from the previous one rather than \
                 restarting at zero. Attention is still blocked across sequences by the \
                 mask, and relative RoPE is unaffected by the shift; what it does reach is \
                 absolute-position terms (Llama 4 attention temperature tuning, Phi-3 \
                 LongRoPE table selection) and positions running past the trained window. \
                 Pass --no-sequence-packing to train on unpacked batches instead."
            );
        }
        Ok(source)
    }

    /// Continue from the latest checkpoint of `manager`, if it has one: load
    /// its weights into `model`, and its optimizer state, data position and
    /// counters into the loop, for [`Self::run`] to pick up. Returns whether
    /// there was a checkpoint.
    ///
    /// The checkpoint must have been taken by a run of the same configuration
    /// on the same data: the data position is an offset into the order the
    /// seed makes.
    pub fn resume_from<M: TrainableModel>(
        &mut self,
        model: &mut M,
        manager: &CheckpointManager,
    ) -> Result<bool> {
        let Some(dir) = manager.latest_dir()? else {
            return Ok(false);
        };
        let (params, metadata) = CheckpointManager::load_checkpoint(&dir)?;
        tracing::info!("Resuming from checkpoint at step {}", metadata.step);
        model.set_lora_parameters(&params);
        self.step = metadata.step;
        self.epoch = metadata.epoch;
        self.running_loss = metadata.running_loss;
        self.pending_resume = TrainState::load(&dir)?;

        // Restore the persisted best-loss snapshot so rollback works after resume.
        if let Some(snapshot) = crate::checkpoint::load_best_snapshot(manager.checkpoint_dir()) {
            self.best_lora_snapshot = Some(snapshot);
            if let Some(ctrl) = self.adaptive_lr.as_mut() {
                ctrl.set_has_snapshot(metadata.running_loss, metadata.step);
            }
            tracing::info!("Restored best snapshot from disk for rollback-on-resume");
        }
        Ok(true)
    }

    /// One micro-batch: its loss and gradients, folded into the optimizer
    /// step in progress, which it completes when it is the last of its
    /// `gradient_accumulation_steps`.
    ///
    /// Evaluates once; see the module documentation.
    pub(crate) fn micro_step<M, O>(
        &mut self,
        model: &mut M,
        batch: BatchView<'_>,
        optimizer: &mut O,
    ) -> Result<StepOutcome>
    where
        M: TrainableModel,
        O: Optimizer + Updatable,
    {
        let start = Instant::now();
        let tokens = batch.tokens();
        let images = matches!(batch, BatchView::Padded(b) if b.pixel_values.is_some());
        let spec = self.loss_spec(model, images);

        if self.step <= 1 {
            tracing::debug!(
                step = self.step,
                tokens,
                "micro_step: computing loss and gradients"
            );
        }
        let out = loss_and_grads(model, batch, &spec)?;
        let micro_loss = out
            .loss_sum
            .divide(&out.count.maximum(&Array::from_f32(1.0)));
        self.accumulator.add(out);

        let accum = self.config.training.gradient_accumulation_steps.max(1);
        let updated = self.accumulator.micro_batches() >= accum;
        let (loss, grad_norm) = if updated {
            self.apply_step(model, optimizer)?
        } else {
            // More micro-batches follow: evaluate what is held, which frees
            // this micro-batch's graph, and read its loss.
            self.accumulator.eval_held(&[&micro_loss])?;
            (check_step(self.step + 1, micro_loss.item_f32())?, None)
        };

        self.step += 1;
        self.total_tokens += tokens;
        self.tokens_since_log += tokens;
        self.running_loss = if self.step == 1 {
            f64::from(loss)
        } else {
            0.99 * self.running_loss + 0.01 * f64::from(loss)
        };

        Ok(StepOutcome {
            stats: StepStats {
                step: self.step,
                loss,
                learning_rate: self.get_learning_rate(),
                tokens,
                grad_norm,
                step_time_ms: start.elapsed().as_millis() as u64,
            },
            updated,
        })
    }

    /// Finish the optimizer step in progress: clip, exchange with the other
    /// ranks, update, and evaluate everything in one call. Returns the
    /// step's mean loss and, when clipping is on, its gradient norm.
    fn apply_step<M, O>(&mut self, model: &mut M, optimizer: &mut O) -> Result<(f32, Option<f32>)>
    where
        M: TrainableModel,
        O: Optimizer + Updatable,
    {
        let max_norm = self.config.training.max_grad_norm as f32;
        let Some(finished) = self.accumulator.finish(max_norm) else {
            return Err(SftError::Mlx(pmetal_runtime::compat::Exception::custom(
                "no gradients to apply",
            )));
        };
        let mut grads = finished.grads;

        // Data parallelism: every rank steps on the mean gradient.
        #[cfg(feature = "distributed")]
        if let Some(ref mut dist) = self.distributed {
            crate::distributed_bridge::block_on(dist.sync_gradients(&mut grads))?;
        }

        optimizer
            .update(model, std::mem::take(&mut grads))
            .map_err(|e| {
                SftError::Mlx(pmetal_runtime::compat::Exception::custom(format!(
                    "training step {} failed: {e}",
                    self.step + 1
                )))
            })?;

        // The one evaluation: loss and norm, the parameters the optimizer
        // changed, and its state.
        let trainable = model.flatten_trainable_params();
        let mut arrays: Vec<&Array> = vec![&finished.loss];
        arrays.extend(finished.grad_norm.iter());
        arrays.extend(trainable.values());
        arrays.extend(optimizer.updatable_states());
        transforms::eval(arrays)?;

        let loss = check_step(self.step + 1, finished.loss.item_f32())?;
        let norm = finished
            .grad_norm
            .map(|n| check_grad_norm(self.step + 1, n.item_f32()))
            .transpose()?;
        Ok((loss, norm))
    }

    /// The step [`Self::apply_step`] takes, as a [`StepStats`], for the
    /// leftover micro-batches at the end of a run.
    fn flush_step<M, O>(&mut self, model: &mut M, optimizer: &mut O) -> Result<StepStats>
    where
        M: TrainableModel,
        O: Optimizer + Updatable,
    {
        let start = Instant::now();
        let (loss, grad_norm) = self.apply_step(model, optimizer)?;
        Ok(StepStats {
            step: self.step,
            loss,
            learning_rate: self.get_learning_rate(),
            tokens: 0,
            grad_norm,
            step_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// One micro-batch of a padded run, outside the loop: the primitive
    /// [`Self::run`] is made of, for callers with their own data or their own
    /// optimizer. `step` counts micro-batches; the optimizer steps on the
    /// last of each `gradient_accumulation_steps`, and the stats then carry
    /// the gradient norm (when clipping is on).
    pub fn train_step<M, O>(
        &mut self,
        model: &mut M,
        batch: &pmetal_data::TrainingBatch,
        optimizer: &mut O,
    ) -> Result<StepStats>
    where
        M: TrainableModel,
        O: Optimizer + Updatable,
    {
        Ok(self
            .micro_step(model, BatchView::Padded(batch), optimizer)?
            .stats)
    }

    /// What follows an optimizer step: adaptive control, logging and
    /// callbacks, evaluation, checkpointing, and the stop conditions.
    fn after_update<M: TrainableModel>(
        &mut self,
        run: &mut Run<'_, M>,
        stats: StepStats,
        total_micro_batches: usize,
    ) -> Result<Flow> {
        let stats = self.sync_step_loss(stats)?;

        if self.post_step_adaptive_with_checkpoint(stats.loss as f64, run)? {
            return Ok(Flow::Stop);
        }

        let optimizer_step = self.optimizer_steps_taken();
        if optimizer_step % self.config.log_every == 0 || optimizer_step == 1 {
            let now = Instant::now();
            let interval = self.take_log_interval_metrics(now);

            tracing::info!(
                "Step {}: loss={:.4}, lr={:.2e}, tokens/s={:.0}{}",
                stats.step,
                self.running_loss,
                stats.learning_rate,
                interval.tok_sec,
                stats
                    .grad_norm
                    .map(|n| format!(", grad_norm={n:.2}"))
                    .unwrap_or_default()
            );

            if !self.callbacks.is_empty() {
                let step_metrics = pmetal_core::StepMetrics {
                    step: self.step,
                    epoch: self.epoch,
                    total_epochs: self.config.training.num_epochs,
                    total_steps: total_micro_batches,
                    loss: self.running_loss,
                    lr: stats.learning_rate as f64,
                    tok_sec: interval.tok_sec,
                    total_ms: interval.total_ms / interval.steps as f64,
                    tokens: interval.tokens,
                    grad_norm: stats.grad_norm.map(|n| n as f64),
                    ..Default::default()
                };
                for cb in &mut self.callbacks {
                    cb.on_step_end_with_metrics(&step_metrics);
                }
                if self.check_cancelled() {
                    tracing::info!("Training cancelled by callback at step {}", self.step);
                    return Err(SftError::Cancelled);
                }
            }
        }

        self.maybe_evaluate(run)?;
        self.maybe_save_regular_checkpoint(run)?;

        if self.reached_max_steps() {
            tracing::info!(
                "Reached max_steps={:?} optimizer steps, stopping",
                self.config.training.max_steps
            );
            return Ok(Flow::Stop);
        }
        Ok(Flow::Continue)
    }

    /// Feed the loss to the adaptive LR controller and act on what it says:
    /// snapshot the best weights, roll back (restoring the weights and
    /// starting the optimizer's moments afresh), or stop with a final
    /// checkpoint. Returns whether the run should stop.
    fn post_step_adaptive_with_checkpoint<M: TrainableModel>(
        &mut self,
        loss: f64,
        run: &mut Run<'_, M>,
    ) -> Result<bool> {
        match self.apply_adaptive_lr(loss) {
            AdaptiveAction::Continue => {
                if self.should_snapshot_best() {
                    self.snapshot_best_weights(run.model);
                }
                Ok(false)
            }
            AdaptiveAction::Rollback => {
                tracing::warn!(
                    step = self.step,
                    "Divergence detected — rolling back to best snapshot"
                );
                if self.restore_best_weights(run.model) {
                    // The moments the diverged trajectory built point away
                    // from the restored weights.
                    run.optimizer = self.build_optimizer(run.output_projection.clone());
                    run.optimizer.set_learning_rate(self.get_learning_rate());
                }
                Ok(false)
            }
            AdaptiveAction::EarlyStop => {
                tracing::warn!(step = self.step, "Max rollbacks exhausted — early stopping");
                self.stop_with_checkpoint(run)?;
                Ok(true)
            }
            AdaptiveAction::SaveCheckpoint => {
                tracing::info!(step = self.step, "External checkpoint save requested");
                if self.is_checkpoint_master() && run.checkpoints.is_some() {
                    self.save_run_checkpoint(run, false, Some(self.running_loss))?;
                }
                Ok(false)
            }
            AdaptiveAction::GracefulStop => {
                tracing::info!(step = self.step, "External graceful stop requested");
                self.restore_best_weights(run.model);
                self.stop_with_checkpoint(run)?;
                Ok(true)
            }
        }
    }

    /// Write the final "best" checkpoint when the controller ends the run.
    fn stop_with_checkpoint<M: TrainableModel>(&mut self, run: &mut Run<'_, M>) -> Result<()> {
        if self.is_checkpoint_master() && run.checkpoints.is_some() {
            self.save_run_checkpoint(run, true, Some(self.running_loss))?;
        }
        Ok(())
    }

    /// Run evaluation if the schedule calls for it and write a "best"
    /// checkpoint on improvement. The model is evaluated at the weights it
    /// would be saved at.
    fn maybe_evaluate<M: TrainableModel>(&mut self, run: &mut Run<'_, M>) -> Result<()> {
        let every = self.config.eval_every;
        if every == 0 || self.optimizer_steps_taken() % every != 0 {
            return Ok(());
        }
        let Some(eval) = run.eval.as_mut() else {
            return Ok(());
        };
        run.optimizer.eval_weights(run.model);
        let metrics = self.evaluate_with(run.model, eval);
        run.optimizer.train_weights(run.model);
        let metrics = metrics?;

        let acc_str = metrics
            .accuracy
            .map(|a| format!(", acc={a:.2}%"))
            .unwrap_or_default();
        tracing::info!(
            "Step {}: eval_loss={:.4}, ppl={:.2}{}",
            self.step,
            metrics.loss,
            metrics.perplexity,
            acc_str
        );
        if metrics.loss < run.best_eval_loss {
            run.best_eval_loss = metrics.loss;
            if self.is_checkpoint_master() && run.checkpoints.is_some() {
                self.save_run_checkpoint(run, true, Some(metrics.loss))?;
            }
        }
        Ok(())
    }

    /// Save a regular checkpoint if the schedule calls for it and this rank
    /// is the checkpoint master.
    fn maybe_save_regular_checkpoint<M: TrainableModel>(
        &mut self,
        run: &mut Run<'_, M>,
    ) -> Result<()> {
        let every = self.config.checkpoint_every;
        if every > 0
            && self.optimizer_steps_taken() % every == 0
            && self.is_checkpoint_master()
            && run.checkpoints.is_some()
        {
            self.save_run_checkpoint(run, false, None)?;
        }
        Ok(())
    }

    /// Write a checkpoint a resume can continue from: the weights to
    /// evaluate, the optimizer's state, and the data position.
    fn save_run_checkpoint<M: TrainableModel>(
        &mut self,
        run: &mut Run<'_, M>,
        is_best: bool,
        eval_loss: Option<f64>,
    ) -> Result<()> {
        let Some(manager) = run.checkpoints else {
            return Ok(());
        };
        // Schedule-free AdamW saves its averaged weights, and says so in its
        // state, which is taken in the same mode.
        run.optimizer.eval_weights(run.model);
        let state = TrainState {
            optimizer: Some(run.optimizer.export_state()),
            loader: Some(run.source.state()),
        };
        let saved =
            self.save_checkpoint_with_state(run.model, manager, is_best, eval_loss, Some(state));
        run.optimizer.train_weights(run.model);
        saved.map(|_| ())
    }
}

impl TrainingLoop {
    /// A loader over `dataset` for evaluation: in order, nothing dropped.
    pub(crate) fn eval_loader(&self, dataset: Arc<TrainingDataset>) -> DataLoader {
        let mut config = self.config.dataloader.clone();
        config.shuffle = false;
        config.drop_last = false;
        DataLoader::new(dataset, config, None)
    }
}
