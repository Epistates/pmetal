use super::*;
use pmetal_bridge::compat::module::ModuleParameters;
use pmetal_bridge::compat::optimizers::Updatable;

impl TrainingLoop {
    /// Run optimized training with fused forward/backward/optimizer step.
    ///
    /// This method uses a fused training step that combines forward pass, backward pass,
    /// and optimizer update into a single function, allowing MLX's lazy evaluation to
    /// optimize the computation graph.
    ///
    /// ## Implementation Note
    ///
    /// The step is not traced with MLX's `compile`, which has known limitations with
    /// models + optimizers whose state count changes during execution. It still
    /// benefits from MLX's lazy evaluation and graph fusion.
    ///
    /// ## Warmup Pattern
    ///
    /// This method implements a warmup step to initialize optimizer states:
    /// - AdamW lazily creates momentum/velocity buffers on first update
    /// - Warmup step ensures all optimizer states are initialized
    /// - This is required for correct training with any approach
    ///
    /// **Requirements:**
    /// - gradient_accumulation_steps must be 1 (accumulation not yet supported)
    /// - Model must implement ModuleParameters
    ///
    /// **Note:** Takes ownership of the model and returns it after training.
    pub fn run_compiled<M>(
        &mut self,
        model: M,
        train_dataset: TrainingDataset,
        eval_dataset: Option<TrainingDataset>,
        checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<M>
    where
        M: TrainableModel + ModuleParameters + 'static,
    {
        // Validate configuration
        if self.config.training.gradient_accumulation_steps != 1 {
            return Err(SftError::Mlx(Exception::custom(
                "JIT compilation requires gradient_accumulation_steps=1",
            )));
        }

        // The Rust bridge currently has no working compile-with-state path.
        // Keep the config flag for compatibility, but fall back explicitly
        // instead of routing through a fake compiled path.
        if self.config.use_jit_compilation {
            tracing::warn!(
                "JIT compilation was requested, but compile-with-state is currently unavailable \
                 in the Rust bridge; falling back to fused eager execution"
            );
        }

        let mut model = model;
        self.apply_gradient_checkpointing(&mut model, "Compiled");

        let optimizer = self.build_optimizer();

        let max_steps = self.config.training.max_steps;
        let num_epochs = self.config.training.num_epochs;

        tracing::info!(
            "Starting optimized training: {} trainable params, batch_size={}",
            model.num_trainable_params(),
            self.config.training.batch_size,
        );

        let use_cce = self.cut_cross_entropy_applies(&model, false);

        // Create state tuple that owns both model and optimizer
        // This allows the step function to mutate both in a single function
        let mut state = (model, optimizer);

        // One optimizer step per batch (this path requires no accumulation).
        let micro_batches = self.micro_batches_per_epoch(&train_dataset);
        let computed_total_steps = self.plan_schedule(micro_batches, 1);

        // =========================================================================
        // PHASE 1: WARMUP - Initialize optimizer states with one step
        // =========================================================================
        //
        // AdamW and similar optimizers lazily create internal state (momentum, velocity)
        // on first update(). We run one warmup step to initialize these states before
        // the main training loop.

        let mut dataloader = DataLoader::new(
            train_dataset.clone(),
            self.config.dataloader.clone(),
            None, // No image processor for text-only training
        );

        // Get first batch for warmup
        let warmup_batch = dataloader
            .try_next_batch()
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
            .ok_or_else(|| SftError::Mlx(Exception::custom("Dataset is empty, cannot warmup")))?;

        // Record state count BEFORE warmup (optimizer states not yet initialized)
        let state_count_before = state.updatable_states_len();

        tracing::info!(
            "Warmup: Running uncompiled step to initialize optimizer states (state_count={})",
            state_count_before
        );

        // The warmup step is the run's first step, on the schedule's first rate.
        state.1.set_learning_rate(self.get_learning_rate());

        // Run ONE uncompiled training step
        let warmup_loss = if use_cce {
            jit_training_step_cce(&mut state, (&warmup_batch.input_ids, &warmup_batch.labels))?
        } else {
            jit_training_step_inner(
                &mut state,
                (&warmup_batch.input_ids, &warmup_batch.labels),
                self.config.neftune_noise_alpha,
            )?
        };
        warmup_loss.eval();
        let warmup_loss_val = check_step(1, warmup_loss.item_f32())?;

        // Record state count AFTER warmup (optimizer states now initialized)
        let state_count_after = state.updatable_states_len();

        tracing::info!(
            "Warmup complete: loss={:.4}, state_count {} -> {} (delta={})",
            warmup_loss_val,
            state_count_before,
            state_count_after,
            state_count_after as i64 - state_count_before as i64
        );

        // Update stats for warmup step - use checked arithmetic
        let warmup_tokens = warmup_batch.batch_size.saturating_mul(warmup_batch.seq_len);
        self.step = 1;
        self.total_tokens = warmup_tokens;
        self.running_loss = warmup_loss_val as f64;

        // =========================================================================
        // PHASE 2: State verification and main loop setup
        // =========================================================================
        // After warmup, optimizer state is stable (no more lazy initialization).
        //
        // NOTE: the step is not compiled (see the `training_loop` module note).
        // Instead, defer evaluation (batch evals at logging boundaries).
        // This achieves ~80% of full JIT performance by minimizing GPU-CPU syncs.

        tracing::info!(
            "State initialized (count={}), starting main training loop",
            state_count_after
        );

        tracing::info!("Fused mode enabled - using deferred evaluation for optimized throughput");

        // Initialize timing for throughput measurement
        self.reset_log_interval();

        // =========================================================================
        // PHASE 3: MAIN LOOP - Deferred Evaluation Pattern for 2x throughput
        // =========================================================================
        // Key optimization: Instead of calling eval() every step (forces GPU-CPU sync),
        // we accumulate lazy loss Arrays and only evaluate at logging boundaries.
        // This achieves ~1500-2000 tok/s by minimizing GPU-CPU synchronization overhead.

        // Pre-allocate vector for accumulated losses
        let mut accumulated_losses: Vec<Array> = Vec::with_capacity(self.config.log_every);
        let mut best_eval_loss = f64::MAX;

        for epoch in 0..num_epochs {
            self.epoch = epoch;

            // First epoch continues from where warmup left off
            // Subsequent epochs need fresh dataloader
            if epoch > 0 {
                dataloader = DataLoader::new(
                    train_dataset.clone(),
                    self.config.dataloader.clone(),
                    None, // No image processor for text-only training
                );
                dataloader.reset(Some(self.config.dataloader.seed + epoch as u64));
            }

            if epoch == 0 {
                tracing::info!(
                    "Epoch {}/{} (continuing after warmup)",
                    epoch + 1,
                    num_epochs
                );
            } else {
                tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);
            }

            // Double-buffered batch prefetch: overlap CPU data prep with GPU compute.
            // Fetching the next batch before the current training step completes allows
            // tokenization and array construction to run while MLX builds its lazy graph.
            let mut prefetched_batch = dataloader
                .try_next_batch()
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
            while let Some(batch) = prefetched_batch {
                // Prefetch next batch before the GPU executes the current step.
                prefetched_batch = dataloader
                    .try_next_batch()
                    .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

                let batch_tokens = batch.batch_size.saturating_mul(batch.seq_len);

                // Apply learning rate schedule before each step
                let scheduled_lr = self.get_learning_rate();
                state.1.set_learning_rate(scheduled_lr);

                // Execute fused training step (forward + backward + optimizer update)
                // DEFERRED EVAL: Loss remains a lazy Array, no GPU-CPU sync here
                // MLX's lazy evaluation automatically fuses operations when not evaluated
                let max_grad_norm = self.config.training.max_grad_norm as f32;
                let loss = if use_cce {
                    jit_training_step_cce_clipped(
                        &mut state,
                        (&batch.input_ids, &batch.labels),
                        max_grad_norm,
                    )?
                } else {
                    jit_training_step_inner_clipped(
                        &mut state,
                        (&batch.input_ids, &batch.labels),
                        self.config.neftune_noise_alpha,
                        max_grad_norm,
                    )?
                };
                // Evaluate each step immediately to prevent computation graph
                // accumulation. Without mx.compile, each step builds a new graph
                // (~10 GB for a 0.6B model). Deferring across steps causes OOM.
                loss.eval();
                check_step(self.step + 1, loss.item_f32())?;
                eval_training_state(&[], &state)?;

                accumulated_losses.push(loss);

                // Update step counters (these are just integers, no GPU involvement)
                self.step += 1;
                self.total_tokens += batch_tokens;
                self.tokens_since_log += batch_tokens;

                // Safety valve kept for GDN models with sequential recurrence.
                const MAX_DEFERRED_STEPS: usize = 5;
                if accumulated_losses.len() >= MAX_DEFERRED_STEPS
                    && self.step % self.config.log_every != 0
                {
                    // Already evaluated above, just process the losses
                    for loss in &mut accumulated_losses {
                        let loss_val = loss.item_f32();
                        self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;
                        let action = self.apply_adaptive_lr(loss_val as f64);
                        if action == AdaptiveAction::Continue && self.should_snapshot_best() {
                            self.snapshot_best_weights(&state.0);
                        }
                        if action == AdaptiveAction::Rollback {
                            self.restore_best_weights(&mut state.0);
                        }
                        if action == AdaptiveAction::EarlyStop
                            || action == AdaptiveAction::GracefulStop
                        {
                            if action == AdaptiveAction::EarlyStop {
                                self.restore_best_weights(&mut state.0);
                            }
                            if let Some(manager) = checkpoint_manager {
                                self.save_checkpoint(
                                    &state.0,
                                    manager,
                                    true,
                                    Some(self.running_loss),
                                )?;
                            }
                            return Ok(state.0);
                        }
                        if action == AdaptiveAction::SaveCheckpoint {
                            if let Some(manager) = checkpoint_manager {
                                self.save_checkpoint(
                                    &state.0,
                                    manager,
                                    false,
                                    Some(self.running_loss),
                                )?;
                            }
                        }
                    }
                    accumulated_losses.clear();
                }

                // Logging boundary: NOW we evaluate accumulated losses
                if self.step % self.config.log_every == 0 {
                    // Batch evaluate all accumulated losses, model params, and
                    // optimizer states together to prevent graph growth
                    eval_training_state(&accumulated_losses, &state)?;

                    // Now extract values and compute running loss
                    let mut adaptive_action = AdaptiveAction::Continue;
                    for loss in &mut accumulated_losses {
                        let loss_val = loss.item_f32();
                        self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;
                        let action = self.apply_adaptive_lr(loss_val as f64);
                        match action {
                            AdaptiveAction::EarlyStop | AdaptiveAction::GracefulStop => {
                                adaptive_action = action;
                                break;
                            }
                            AdaptiveAction::SaveCheckpoint
                                if adaptive_action == AdaptiveAction::Continue =>
                            {
                                adaptive_action = AdaptiveAction::SaveCheckpoint;
                            }
                            AdaptiveAction::Rollback
                                if adaptive_action == AdaptiveAction::Continue
                                    || adaptive_action == AdaptiveAction::SaveCheckpoint =>
                            {
                                adaptive_action = AdaptiveAction::Rollback;
                            }
                            _ => {}
                        }
                    }
                    accumulated_losses.clear();

                    if adaptive_action == AdaptiveAction::Continue && self.should_snapshot_best() {
                        self.snapshot_best_weights(&state.0);
                    }
                    if adaptive_action == AdaptiveAction::Rollback {
                        self.restore_best_weights(&mut state.0);
                    }
                    if adaptive_action == AdaptiveAction::EarlyStop
                        || adaptive_action == AdaptiveAction::GracefulStop
                    {
                        if adaptive_action == AdaptiveAction::EarlyStop {
                            self.restore_best_weights(&mut state.0);
                        }
                        if let Some(manager) = checkpoint_manager {
                            self.save_checkpoint(&state.0, manager, true, Some(self.running_loss))?;
                        }
                        return Ok(state.0);
                    }
                    // Handle external checkpoint save request
                    if adaptive_action == AdaptiveAction::SaveCheckpoint {
                        if let Some(manager) = checkpoint_manager {
                            self.save_checkpoint(
                                &state.0,
                                manager,
                                false,
                                Some(self.running_loss),
                            )?;
                        }
                    }

                    // Calculate throughput
                    let now = std::time::Instant::now();
                    let interval = self.take_log_interval_metrics(now);

                    tracing::info!(
                        "Step {}: loss={:.4}, lr={:.2e}, tokens/s={:.0}",
                        self.step,
                        self.running_loss,
                        self.get_learning_rate(),
                        interval.tok_sec,
                    );

                    // Dispatch to callbacks
                    if !self.callbacks.is_empty() {
                        let step_metrics = pmetal_core::StepMetrics {
                            step: self.step,
                            epoch,
                            total_epochs: num_epochs,
                            total_steps: computed_total_steps,
                            loss: self.running_loss,
                            lr: self.get_learning_rate() as f64,
                            tok_sec: interval.tok_sec,
                            total_ms: interval.total_ms / interval.steps as f64,
                            tokens: interval.tokens,
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
                    &mut state.0,
                    eval_dataset.as_ref(),
                    checkpoint_manager,
                    best_eval_loss,
                )?;

                // Scheduled regular checkpointing (rank-0 only in distributed mode).
                self.maybe_save_regular_checkpoint(&mut state.0, checkpoint_manager)?;

                // Check max steps
                if let Some(max) = max_steps {
                    if self.step >= max {
                        // Eval any remaining losses before returning
                        if !accumulated_losses.is_empty() {
                            eval_training_state(&accumulated_losses, &state)?;
                            for loss in &mut accumulated_losses {
                                let loss_val = loss.item_f32();
                                self.running_loss =
                                    0.99 * self.running_loss + 0.01 * loss_val as f64;
                            }
                        }
                        tracing::info!("Reached max_steps={}, stopping", max);
                        // Return the model from the state tuple
                        return Ok(state.0);
                    }
                }
            }
        }

        // Eval any remaining accumulated losses at end of training
        if !accumulated_losses.is_empty() {
            eval_training_state(&accumulated_losses, &state)?;
            for loss in &mut accumulated_losses {
                let loss_val = loss.item_f32();
                self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;
            }
        }

        tracing::info!(
            "Training complete: {} steps, {:.4} final loss",
            self.step,
            self.running_loss
        );

        // Return the trained model
        Ok(state.0)
    }

    /// Attempt JIT-compiled training using `compile_with_state`.
    ///
    /// This currently returns an explicit error because the Rust bridge does
    /// not yet expose a working compile-with-state path for training.
    ///
    /// **Requirements:**
    /// - gradient_accumulation_steps must be 1
    /// - Model must implement ModuleParameters + TrainableModel
    pub fn run_jit_compiled<M>(
        &mut self,
        _model: M,
        _train_dataset: TrainingDataset,
        _eval_dataset: Option<TrainingDataset>,
        _checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<M>
    where
        M: TrainableModel + ModuleParameters + 'static,
    {
        Err(SftError::Mlx(Exception::custom(
            "JIT-compiled training is currently unavailable in the Rust bridge; \
             use the fused training path instead",
        )))
    }
}
