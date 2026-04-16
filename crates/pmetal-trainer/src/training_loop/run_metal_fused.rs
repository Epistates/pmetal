use super::*;
use crate::mlx_metal_optimizer::MlxMetalOptimizer;

impl TrainingLoop {
    /// Run training with Metal fused optimizer for maximum throughput.
    ///
    /// This method uses custom Metal kernels that process all parameters in a single
    /// GPU dispatch, eliminating per-parameter synchronization overhead.
    ///
    /// Expected performance gain: ~40% compared to standard mlx-rs optimizer.
    ///
    /// **Requirements:**
    /// - Apple Silicon with Metal support
    /// - Model parameters must be float32
    pub fn run_metal_fused<M>(
        &mut self,
        model: &mut M,
        train_dataset: TrainingDataset,
        eval_dataset: Option<TrainingDataset>,
        checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<()>
    where
        M: TrainableModel,
    {
        // Check Metal availability
        if !is_mlx_metal_optimizer_available() {
            return Err(SftError::Mlx(Exception::custom(
                "Metal fused optimizer not available on this system",
            )));
        }

        // Initialize Metal optimizer
        let base_lr = self.config.training.learning_rate as f32;
        let weight_decay = self.config.training.weight_decay as f32;

        let mut metal_optimizer = MlxMetalOptimizerBuilder::new(base_lr)
            .weight_decay(weight_decay)
            .build()
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

        // Initialize optimizer with model parameters
        metal_optimizer
            .initialize(model)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

        let max_steps = self.config.training.max_steps;
        let num_epochs = self.config.training.num_epochs;

        // Apply gradient checkpointing if configured (matches run() behavior)
        self.apply_gradient_checkpointing(model, "Metal-fused");

        tracing::info!(
            "Starting Metal-fused training: {} trainable params, batch_size={}, grad_accum={}",
            model.num_trainable_params(),
            self.config.training.batch_size,
            self.config.training.gradient_accumulation_steps
        );
        tracing::info!(
            "Metal fused optimizer: {} total elements",
            metal_optimizer.total_elements()
        );

        // Initialize timing for throughput measurement
        self.last_log_time = Some(std::time::Instant::now());
        self.tokens_since_log = 0;

        let mut best_eval_loss = f64::MAX;

        // Compute total steps: max_steps takes priority, otherwise estimate from dataset
        let steps_per_epoch_est = train_dataset
            .len()
            .div_ceil(self.config.training.batch_size);
        let computed_total_steps = max_steps.unwrap_or(num_epochs * steps_per_epoch_est);
        if let Some(ref mut ctrl) = self.adaptive_lr {
            ctrl.set_total_steps(computed_total_steps);
        }

        // Distributed training: barrier at start of metal fused training
        #[cfg(feature = "distributed")]
        if let Some(ref dist) = self.distributed {
            tracing::info!(
                "Distributed metal-fused training: rank={}/{}, waiting at start barrier...",
                dist.rank(),
                dist.world_size()
            );
            tokio::runtime::Handle::current().block_on(async { dist.barrier().await })?;
        }

        for epoch in 0..num_epochs {
            self.epoch = epoch;
            tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);

            // Distributed: barrier at epoch boundary
            #[cfg(feature = "distributed")]
            if epoch > 0 {
                if let Some(ref dist) = self.distributed {
                    tokio::runtime::Handle::current().block_on(async { dist.barrier().await })?;
                }
            }

            // Create dataloader for this epoch
            let mut dataloader =
                DataLoader::new(train_dataset.clone(), self.config.dataloader.clone(), None);

            // Double-buffered batch prefetch: overlap CPU data prep with GPU compute.
            let mut prefetched_batch = dataloader
                .try_next_batch()
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
            while let Some(batch) = prefetched_batch {
                // Kick off the next batch fetch before submitting the current one.
                prefetched_batch = dataloader
                    .try_next_batch()
                    .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

                // Apply learning rate schedule before each step.
                // This must happen outside train_step_metal because the inner
                // set_learning_rate call is guarded by should_apply_gradients(),
                // which skips non-accumulation steps — causing the LR to be stale
                // for the first step of each new accumulation cycle.
                metal_optimizer.set_learning_rate(self.get_scheduled_lr());

                // Training step with Metal optimizer
                let mut stats = self.train_step_metal(model, &batch, &mut metal_optimizer)?;

                // Distributed: sync loss across nodes so all agree on LR decisions
                #[cfg(feature = "distributed")]
                if let Some(ref dist) = self.distributed {
                    let synced = tokio::runtime::Handle::current()
                        .block_on(async { dist.sync_loss(stats.loss).await })?;
                    stats.loss = synced;
                }

                // Adaptive LR + snapshot/rollback/checkpoint-on-stop.
                // See `TrainingLoop::post_step_adaptive_with_checkpoint` in mod.rs
                // for the shared handling of every AdaptiveAction variant.
                if self.post_step_adaptive_with_checkpoint(
                    stats.loss as f64,
                    model,
                    checkpoint_manager,
                )? {
                    return Ok(());
                }

                // Logging + callback dispatch
                let mut tokens_per_sec = 0.0f64;
                if self.step % self.config.log_every == 0 {
                    let now = std::time::Instant::now();
                    tokens_per_sec = match self.last_log_time {
                        Some(last) => {
                            let elapsed_secs = now.duration_since(last).as_secs_f64();
                            if elapsed_secs > 0.0 {
                                self.tokens_since_log as f64 / elapsed_secs
                            } else {
                                0.0
                            }
                        }
                        None => 0.0,
                    };

                    self.last_log_time = Some(now);
                    self.tokens_since_log = 0;

                    tracing::info!(
                        "Step {}: loss={:.4}, lr={:.2e}, tokens/s={:.0}{}",
                        stats.step,
                        self.running_loss,
                        stats.learning_rate,
                        tokens_per_sec,
                        stats
                            .grad_norm
                            .map(|n| format!(", grad_norm={:.2}", n))
                            .unwrap_or_default()
                    );
                }

                // Dispatch to callbacks
                if !self.callbacks.is_empty() {
                    let step_metrics = pmetal_core::StepMetrics {
                        step: stats.step,
                        epoch,
                        total_epochs: num_epochs,
                        total_steps: computed_total_steps,
                        loss: stats.loss as f64,
                        lr: stats.learning_rate as f64,
                        tok_sec: tokens_per_sec,
                        total_ms: stats.step_time_ms as f64,
                        tokens: stats.tokens,
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

                // Scheduled evaluation + best-checkpoint-on-improvement.
                best_eval_loss = self.maybe_evaluate(
                    model,
                    eval_dataset.as_ref(),
                    checkpoint_manager,
                    best_eval_loss,
                )?;

                // Scheduled regular checkpointing (rank-0 only in distributed mode).
                self.maybe_save_regular_checkpoint(model, checkpoint_manager)?;

                // Check max steps
                if let Some(max) = max_steps {
                    if self.step >= max {
                        tracing::info!("Reached max_steps={}, stopping", max);
                        return Ok(());
                    }
                }
            }

            dataloader.reset(Some(self.config.dataloader.seed + epoch as u64 + 1));
        }

        tracing::info!(
            "Metal-fused training complete: {} steps, {:.4} final loss",
            self.step,
            self.running_loss
        );

        Ok(())
    }

    /// Single training step using Metal fused optimizer.
    pub(super) fn train_step_metal<M>(
        &mut self,
        model: &mut M,
        batch: &TrainingBatch,
        metal_optimizer: &mut MlxMetalOptimizer,
    ) -> Result<StepStats>
    where
        M: TrainableModel,
    {
        let start_time = std::time::Instant::now();

        let batch_tokens = batch
            .batch_size
            .checked_mul(batch.seq_len)
            .unwrap_or(usize::MAX);

        // PROFILING: Track time for each phase
        let t0 = std::time::Instant::now();

        // Compute loss and gradients (forward + backward pass) - stays lazy
        let (loss, grads) = self.compute_text_loss_and_grads(model, batch)?;

        let t1 = std::time::Instant::now();
        let fwd_bwd_us = t1.duration_since(t0).as_micros();

        // Accumulate gradients (no eval yet - stays lazy)
        self.accumulate_gradients(grads)?;

        let t2 = std::time::Instant::now();
        let accum_us = t2.duration_since(t1).as_micros();

        // Apply gradients if accumulation is complete
        let (loss_val, grad_norm, clip_us, opt_us) = if self.should_apply_gradients() {
            if let Some(mut accumulated) = self.take_accumulated_gradients() {
                // Clip gradients (lazy computation)
                let lazy_norm = self.clip_gradients_gpu(&mut accumulated)?;

                // Distributed gradient sync: all-reduce gradients across nodes
                #[cfg(feature = "distributed")]
                if let Some(ref mut dist) = self.distributed {
                    tokio::runtime::Handle::current()
                        .block_on(async { dist.sync_gradients(&mut accumulated).await })?;
                }

                // Update learning rate in Metal optimizer
                metal_optimizer.set_learning_rate(self.get_learning_rate());

                let t3 = std::time::Instant::now();
                let clip_elapsed = t3.duration_since(t2).as_micros();

                // Apply with Metal fused optimizer
                // This does the actual computation and batched eval of params/state
                metal_optimizer
                    .update_model_fused(model, &accumulated)
                    .map_err(|e| SftError::Mlx(e))?;

                let t4 = std::time::Instant::now();
                let opt_elapsed = t4.duration_since(t3).as_micros();

                // Now evaluate loss and grad_norm together (single eval call for both)
                let mut to_eval: Vec<&Array> = vec![&loss];
                if let Some(ref norm) = lazy_norm {
                    to_eval.push(norm);
                }
                let _ = pmetal_bridge::compat::transforms::eval(to_eval);

                let mut loss = loss;
                let loss_val = loss.item_f32();
                let norm = lazy_norm.map(|mut n| n.item_f32());

                (loss_val, norm, clip_elapsed, opt_elapsed)
            } else {
                // No gradients accumulated - just eval loss
                let mut loss = loss;
                loss.eval();
                (loss.item::<f32>(), None, 0, 0)
            }
        } else {
            // Gradient accumulation not complete - just eval loss
            let mut loss = loss;
            loss.eval();
            (loss.item::<f32>(), None, 0, 0)
        };

        // Update stats
        self.step += 1;
        self.total_tokens += batch_tokens;
        self.tokens_since_log += batch_tokens;

        self.running_loss = if self.step == 1 {
            loss_val as f64
        } else {
            0.99 * self.running_loss + 0.01 * loss_val as f64
        };

        let step_time_ms = start_time.elapsed().as_millis() as u64;

        // Log profiling info on first few steps and periodically
        static PROFILE_COUNT: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let profile_idx = PROFILE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if profile_idx < 5 || profile_idx % 50 == 0 {
            let total_us = fwd_bwd_us + accum_us + clip_us + opt_us;
            tracing::info!(
                "PROFILE step={}: fwd+bwd={}us ({:.1}%), accum={}us ({:.1}%), clip={}us ({:.1}%), opt={}us ({:.1}%), total={}us",
                self.step,
                fwd_bwd_us,
                100.0 * fwd_bwd_us as f64 / total_us as f64,
                accum_us,
                100.0 * accum_us as f64 / total_us as f64,
                clip_us,
                100.0 * clip_us as f64 / total_us as f64,
                opt_us,
                100.0 * opt_us as f64 / total_us as f64,
                total_us
            );
        }

        Ok(StepStats {
            step: self.step,
            loss: loss_val,
            learning_rate: self.get_learning_rate(),
            tokens: batch_tokens,
            grad_norm,
            step_time_ms,
        })
    }
}
