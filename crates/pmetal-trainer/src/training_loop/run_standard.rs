use super::*;

impl TrainingLoop {
    /// Run the full training loop.
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
        // Initialize optimizer with optional embedding learning rate
        let base_lr = self.config.training.learning_rate as f32;
        let weight_decay = self.config.training.weight_decay as f32;

        // Use AdamWGroups when embedding_lr is specified
        let mut optimizer = crate::AdamWGroupsBuilder::new(base_lr).with_weight_decay(weight_decay);

        if let Some(emb_lr) = self.config.embedding_lr {
            optimizer = optimizer.with_embedding_lr(emb_lr);
            tracing::info!(
                "Using separate embedding LR: {:.2e} (base: {:.2e})",
                emb_lr,
                base_lr
            );
        }

        // Wire LoRA+ differential learning rates for B vs A matrices
        if let Some(ratio) = self.config.loraplus_lr_ratio {
            optimizer = optimizer.with_loraplus_lr_ratio(ratio);
        }

        let mut optimizer = optimizer
            .build()
            .map_err(|_| SftError::Mlx(Exception::custom("Failed to build optimizer")))?;

        let max_steps = self.config.training.max_steps;
        let num_epochs = self.config.training.num_epochs;

        tracing::info!(
            "Starting training: {} trainable params, batch_size={}, grad_accum={}",
            model.num_trainable_params(),
            self.config.training.batch_size,
            self.config.training.gradient_accumulation_steps
        );

        // Apply gradient checkpointing when requested.
        // This must be done before the training loop so that all forward passes
        // use checkpointed activations.
        self.apply_gradient_checkpointing(model, "");

        // Initialize timing for throughput measurement
        self.reset_log_interval();

        let mut best_eval_loss = f64::MAX;

        // Compute total steps: max_steps takes priority, otherwise estimate from dataset
        let steps_per_epoch_est = train_dataset
            .len()
            .div_ceil(self.config.training.batch_size);
        let computed_total_steps = max_steps.unwrap_or(num_epochs * steps_per_epoch_est);
        if let Some(ref mut ctrl) = self.adaptive_lr {
            ctrl.set_total_steps(computed_total_steps);
        }

        // Distributed training: barrier at start
        #[cfg(feature = "distributed")]
        if let Some(ref dist) = self.distributed {
            tracing::info!(
                "Distributed training: rank={}/{}, waiting at start barrier...",
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
            let mut dataloader = DataLoader::new(
                train_dataset.clone(),
                self.config.dataloader.clone(),
                None, // No image processor for text-only training
            );

            // Double-buffered batch prefetch: fetch the next batch while the GPU
            // processes the current one, overlapping CPU data prep with GPU compute.
            let mut prefetched_batch = dataloader
                .try_next_batch()
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
            while let Some(batch) = prefetched_batch {
                // Prefetch the next batch before submitting the current one to the GPU.
                // DataLoader::next_batch() is CPU-bound (tokenization + array construction),
                // so starting it now lets it run while MLX evaluates the training step.
                prefetched_batch = dataloader
                    .try_next_batch()
                    .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

                // Apply learning rate schedule (warmup, cosine decay, etc.)
                let scheduled_lr = self.get_learning_rate();
                optimizer.set_learning_rate(scheduled_lr);

                // Training step
                let mut stats = self.train_step(model, &batch, &mut optimizer)?;

                // Distributed: sync loss across nodes so all agree on LR decisions
                #[cfg(feature = "distributed")]
                if let Some(ref dist) = self.distributed {
                    let synced = tokio::runtime::Handle::current()
                        .block_on(async { dist.sync_loss(stats.loss).await })?;
                    stats.loss = synced;
                    // Note: do NOT update running_loss here — train_step already
                    // applied EMA. We only override stats.loss so that all nodes
                    // agree on the value fed to the adaptive LR controller.
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

                // Logging — always log step 1 for immediate GUI feedback,
                // then at regular intervals.
                if self.step % self.config.log_every == 0 || self.step == 1 {
                    // Calculate throughput over the entire logging interval
                    let now = std::time::Instant::now();
                    let interval = self.take_log_interval_metrics(now);

                    tracing::info!(
                        "Step {}: loss={:.4}, lr={:.2e}, tokens/s={:.0}{}",
                        stats.step,
                        self.running_loss,
                        stats.learning_rate,
                        interval.tok_sec,
                        stats
                            .grad_norm
                            .map(|n| format!(", grad_norm={:.2}", n))
                            .unwrap_or_default()
                    );

                    // Dispatch to callbacks
                    if !self.callbacks.is_empty() {
                        let step_metrics = pmetal_core::StepMetrics {
                            step: self.step,
                            epoch,
                            total_epochs: num_epochs,
                            total_steps: computed_total_steps,
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

            // Reset dataloader with new seed for next epoch
            dataloader.reset(Some(self.config.dataloader.seed + epoch as u64 + 1));
        }

        tracing::info!(
            "Training complete: {} steps, {:.4} final loss",
            self.step,
            self.running_loss
        );

        Ok(())
    }
}
