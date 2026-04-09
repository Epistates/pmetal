use super::*;
use pmetal_bridge::compat::module::ModuleParameters;
use pmetal_bridge::compat::optimizers::Updatable;

impl TrainingLoop {
    /// Run training with sequence packing for 2-5x throughput improvement.
    ///
    /// This method packs multiple shorter sequences into single batches,
    /// eliminating padding waste and dramatically improving GPU utilization.
    ///
    /// Uses block-diagonal attention masks to prevent cross-sequence attention.
    pub fn run_packed<M>(
        &mut self,
        model: M,
        train_dataset: TrainingDataset,
        eval_dataset: Option<TrainingDataset>,
        checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<M>
    where
        M: TrainableModel + ModuleParameters + 'static,
    {
        let num_epochs = self.config.training.num_epochs;
        let max_steps = self.config.training.max_steps;

        // Initialize optimizer with optional embedding learning rate
        let base_lr = self.config.training.learning_rate as f32;
        let weight_decay = self.config.training.weight_decay as f32;

        let mut optimizer_builder =
            crate::AdamWGroupsBuilder::new(base_lr).with_weight_decay(weight_decay);

        if let Some(emb_lr) = self.config.embedding_lr {
            optimizer_builder = optimizer_builder.with_embedding_lr(emb_lr);
            tracing::info!(
                "Using separate embedding LR: {:.2e} (base: {:.2e})",
                emb_lr,
                base_lr
            );
        }

        // Wire LoRA+ differential learning rates for B vs A matrices
        if let Some(ratio) = self.config.loraplus_lr_ratio {
            optimizer_builder = optimizer_builder.with_loraplus_lr_ratio(ratio);
        }

        let optimizer = optimizer_builder
            .build()
            .map_err(|_| SftError::Mlx(Exception::custom("Failed to build optimizer")))?;

        tracing::info!("Starting packed training with sequence packing enabled");

        // Get samples from dataset - need to access the samples directly
        let samples: Vec<_> = (0..train_dataset.len())
            .filter_map(|i| train_dataset.get(i).cloned())
            .collect();

        // Create PackedDataLoader from dataset samples
        // CRITICAL: Set max_seq_length to truncate long sequences instead of skipping them!
        //
        // Use adaptive packing sequence length: p99 of actual sample lengths rounded
        // up to the next power of 2, capped at the configured max. This avoids the
        // O(n²) attention cost when max_seq_len auto-detects to the model architectural
        // maximum (e.g. 8192) for datasets with short sequences (e.g. 50–400 tokens).
        //
        // If the caller provided an explicit override via `--pack-max-seq-len`, use
        // that value directly instead of the adaptive computation.
        let max_seq_len = if let Some(explicit) = self.config.pack_max_seq_len {
            tracing::info!(
                "Using explicit pack_max_seq_len={} (overrides adaptive p99 heuristic)",
                explicit
            );
            explicit
        } else {
            let sample_lengths: Vec<usize> = samples.iter().map(|s| s.input_ids.len()).collect();
            compute_pack_seq_len(&sample_lengths, self.config.dataloader.max_seq_len)
        };
        let packer_config = PackerConfig::with_max_length(max_seq_len)
            .with_max_seq_length(max_seq_len) // Truncate sequences to adaptive max
            .mask_boundaries(true);

        let mut packed_dataloader = PackedDataLoader::new(
            &samples,
            packer_config,
            self.config.dataloader.shuffle,
            self.config.dataloader.seed,
        )
        .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

        // Log packing statistics
        let stats = packed_dataloader.stats();
        tracing::info!(
            "Packing: {} sequences → {} batches, {:.1}% efficiency, avg {:.1} seqs/batch",
            stats.num_sequences,
            stats.num_batches,
            stats.efficiency * 100.0,
            stats.avg_sequences_per_batch
        );

        if stats.max_sequences_per_batch <= 1 {
            tracing::info!(
                "Sequence packing produced only single-sequence batches; \
                 falling back to the standard training loop"
            );
            return self.run_standard_owned(model, train_dataset, eval_dataset, checkpoint_manager);
        }

        let mut model = model;
        self.apply_gradient_checkpointing(&mut model, "Packed");

        // Compute actual total steps: max_steps takes priority, otherwise epochs * batches_per_epoch
        let computed_total_steps = max_steps.unwrap_or(num_epochs * stats.num_batches);
        if let Some(ref mut ctrl) = self.adaptive_lr {
            ctrl.set_total_steps(computed_total_steps);
            // Extend grace period to cover the full LR warmup so the adaptive
            // controller doesn't intervene while the LR is still ramping.
            let warmup_steps = if let Some(ratio) = self.config.training.warmup_ratio {
                (computed_total_steps as f64 * ratio) as usize
            } else {
                self.config.training.warmup_steps
            };
            ctrl.set_warmup_steps(warmup_steps);
        }

        // Create state tuple for training
        let mut state = (model, optimizer);

        // =========================================================================
        // PHASE 1: Warmup step for optimizer state initialization
        // =========================================================================
        // AdamW lazily creates momentum/velocity buffers on first update().
        // Running one step before the main loop ensures optimizer state is
        // properly initialized, avoiding first-batch anomalies.

        let warmup_batch = packed_dataloader
            .next_batch()
            .ok_or_else(|| {
                SftError::Mlx(Exception::custom("No packed batches available for warmup"))
            })?
            .map_err(|e| SftError::Mlx(e))?;

        let warmup_tokens = warmup_batch.total_tokens;

        // Record optimizer state count BEFORE warmup (states not yet initialized)
        let state_count_before = state.updatable_states_len();

        tracing::info!(
            "Warmup: Running step to initialize optimizer states (state_count={})",
            state_count_before
        );

        // Execute warmup step - this initializes optimizer momentum/velocity buffers
        let max_grad_norm = self.config.training.max_grad_norm as f32;
        let mut warmup_loss = if self.config.use_cut_cross_entropy {
            jit_training_step_packed_cce(&mut state, &warmup_batch, max_grad_norm)?
        } else {
            jit_training_step_packed(&mut state, &warmup_batch, max_grad_norm)?
        };
        warmup_loss.eval();
        let warmup_loss_val: f32 = warmup_loss.item();

        // Record optimizer state count AFTER warmup (states now initialized)
        let state_count_after = state.updatable_states_len();

        tracing::info!(
            "Warmup complete: loss={:.4}, state_count {} -> {} (delta={})",
            warmup_loss_val,
            state_count_before,
            state_count_after,
            state_count_after as i64 - state_count_before as i64
        );

        // Initialize step tracking with warmup step
        self.step = 1;
        self.total_tokens = warmup_tokens;
        self.running_loss = warmup_loss_val as f64;

        // =========================================================================
        // PHASE 2: Main training loop
        // =========================================================================

        // Initialize timing for throughput measurement
        self.reset_log_interval();

        // Pre-allocate vector for accumulated losses (deferred evaluation)
        let mut accumulated_losses: Vec<Array> = Vec::with_capacity(self.config.log_every);

        for epoch in 0..num_epochs {
            self.epoch = epoch;

            if epoch > 0 {
                packed_dataloader.reset(Some(self.config.dataloader.seed + epoch as u64));
            }

            tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);

            // Double-buffered batch prefetch for packed sequences.
            // PackedDataLoader::next_batch() packs and tokenizes sequences on the CPU,
            // so prefetching overlaps that work with the GPU training step.
            let mut prefetched_batch_result = packed_dataloader.next_batch();
            while let Some(batch_result) = prefetched_batch_result {
                // Prefetch next packed batch before executing the current training step.
                prefetched_batch_result = packed_dataloader.next_batch();

                let packed_batch = batch_result.map_err(|e| SftError::Mlx(e))?;

                let batch_tokens = packed_batch.total_tokens;

                // Apply learning rate schedule before each step
                let scheduled_lr = self.get_learning_rate();
                state.1.set_learning_rate(scheduled_lr);

                // Execute packed training step (forward + backward + optimizer update)
                let mut loss = if self.config.use_cut_cross_entropy {
                    jit_training_step_packed_cce(&mut state, &packed_batch, max_grad_norm)?
                } else {
                    jit_training_step_packed(&mut state, &packed_batch, max_grad_norm)?
                };

                // Evaluate each step immediately to prevent computation graph
                // accumulation. Without mx.compile (unavailable in mlx-rs), each
                // step builds a NEW graph (~10 GB for a 0.6B model). Deferring
                // evaluation across 10 steps means 10 graphs (~100 GB) in memory.
                loss.eval();
                eval_training_state(&[], &state)?;

                accumulated_losses.push(loss);

                // Update step counters
                self.step += 1;
                self.total_tokens += batch_tokens;
                self.tokens_since_log += batch_tokens;

                // Logging boundary: NOW we evaluate accumulated losses
                if self.step % self.config.log_every == 0 || self.step == 1 {
                    // Batch evaluate all accumulated losses, model params, and
                    // optimizer states together to prevent graph growth
                    eval_training_state(&accumulated_losses, &state)?;

                    // Extract values and compute running loss via EMA.
                    // Track adaptive action across all accumulated losses.
                    // Each accumulated loss was computed at a specific training step.
                    // Retroactively set self.step so the adaptive LR controller and
                    // scheduler see the correct per-step context (not the batch-end step).
                    let saved_step = self.step;
                    let batch_size = accumulated_losses.len();
                    let mut adaptive_action = AdaptiveAction::Continue;
                    for (i, loss) in accumulated_losses.iter_mut().enumerate() {
                        let loss_val = loss.item_f32();
                        self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;
                        self.step = saved_step - batch_size + i;
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
                    self.step = saved_step;
                    accumulated_losses.clear();

                    // Snapshot best weights when adaptive controller detects improvement
                    if adaptive_action == AdaptiveAction::Continue && self.should_snapshot_best() {
                        self.snapshot_best_weights(&state.0);
                    }

                    // Handle rollback: restore best weights + let optimizer adapt
                    if adaptive_action == AdaptiveAction::Rollback
                        && self.restore_best_weights(&mut state.0)
                    {
                        tracing::info!(
                            "Weights restored from best snapshot. \
                             Continuing training with LR {:.2e}.",
                            self.get_learning_rate(),
                        );
                    }

                    // Handle early stop / graceful stop
                    if adaptive_action == AdaptiveAction::EarlyStop
                        || adaptive_action == AdaptiveAction::GracefulStop
                    {
                        if adaptive_action == AdaptiveAction::EarlyStop {
                            tracing::info!(
                                "Early stopping triggered. Restoring best weights and exiting."
                            );
                            self.restore_best_weights(&mut state.0);
                        } else {
                            tracing::info!(
                                "Graceful stop requested. Saving checkpoint and exiting."
                            );
                        }
                        // Save the best checkpoint before exiting
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

                // Regular checkpointing
                if self.config.checkpoint_every > 0 && self.step % self.config.checkpoint_every == 0
                {
                    // Eval any pending losses before checkpointing
                    if !accumulated_losses.is_empty() {
                        eval_training_state(&accumulated_losses, &state)?;
                        for loss in &mut accumulated_losses {
                            let loss_val = loss.item_f32();
                            self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;
                        }
                        accumulated_losses.clear();
                    }

                    if let Some(manager) = checkpoint_manager {
                        self.save_checkpoint(&state.0, manager, false, None)?;
                    }
                }

                // Check max steps
                if let Some(max) = max_steps {
                    if self.step >= max {
                        // Eval any remaining losses before returning
                        if !accumulated_losses.is_empty() {
                            eval_training_state(&accumulated_losses, &state)?;
                        }
                        tracing::info!("Reached max_steps={}, stopping", max);
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
            "Packed training complete: {} steps, {:.4} final loss",
            self.step,
            self.running_loss
        );

        // Return the trained model
        Ok(state.0)
    }
}
