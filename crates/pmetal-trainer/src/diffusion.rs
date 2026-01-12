//! LLaDA-style masked diffusion training for language models.
//!
//! This module implements discrete diffusion training following the LLaDA/MDLM approach:
//!
//! - **Forward Process**: Randomly mask tokens with probability t ~ U(0,1)
//! - **Training**: Predict masked tokens with ELBO-weighted cross-entropy loss
//! - **Sampling**: Iterative unmasking with confidence-based remasking
//!
//! # References
//!
//! - LLaDA: Large Language Diffusion Models (arXiv:2502.09992)
//! - MDLM: Simple and Effective Masked Diffusion Language Models (NeurIPS 2024)
//! - Gemini Diffusion (Google DeepMind, 2025)

use std::collections::HashMap;

use pmetal_core::{EvalMetrics, LrSchedulerType, TrainingConfig};
use pmetal_data::{DataLoader, DataLoaderConfig, TrainingBatch, TrainingDataset};
use pmetal_lora::TrainableModel;
use pmetal_mlx::kernels::cross_entropy::cross_entropy_loss;
use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::{FlattenedModuleParam, ModuleParameters},
    nn,
    optimizers::{AdamWBuilder, Optimizer},
    transforms::eval_params,
    Array, Dtype,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{CheckpointManager, CheckpointMetadata, Result, SftError};

/// Noise schedule for the forward diffusion process.
#[derive(Debug, Clone, Copy, Default)]
pub enum NoiseSchedule {
    /// Linear schedule: α_t = 1 - t (recommended, most stable)
    #[default]
    Linear,

    /// Polynomial schedule: α_t = (1 - t)^2
    Polynomial2,

    /// Cosine schedule: α_t = cos(πt/2)
    Cosine,
}

impl NoiseSchedule {
    /// Compute α_t (probability of keeping original token) at time t.
    pub fn alpha(&self, t: f32) -> f32 {
        match self {
            NoiseSchedule::Linear => 1.0 - t,
            NoiseSchedule::Polynomial2 => (1.0 - t).powi(2),
            NoiseSchedule::Cosine => (std::f32::consts::FRAC_PI_2 * t).cos(),
        }
    }
}

/// Remasking strategy for inference.
#[derive(Debug, Clone, Copy, Default)]
pub enum RemaskingStrategy {
    /// Remask tokens with lowest prediction confidence.
    #[default]
    LowConfidence,

    /// Random remasking.
    Random,

    /// Semi-autoregressive: unmask left-to-right in blocks.
    SemiAutoregressive {
        /// Block size for semi-autoregressive generation.
        block_size: usize,
    },
}

/// Configuration for diffusion training.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Mask token ID (from tokenizer).
    pub mask_token_id: i64,

    /// Noise schedule type.
    pub noise_schedule: NoiseSchedule,

    /// Number of diffusion steps for inference.
    pub num_inference_steps: usize,

    /// Remasking strategy for inference.
    pub remasking_strategy: RemaskingStrategy,

    /// Use principled ELBO loss (true) or simpler MaskGIT loss (false).
    pub use_elbo_loss: bool,

    /// SFT mode: only mask response tokens, not prompt tokens.
    pub sft_mode: bool,

    /// Minimum noise level to sample during training.
    pub min_noise_level: f32,

    /// Temperature for sampling during inference.
    pub sampling_temperature: f32,

    /// Training hyperparameters.
    pub training: TrainingConfig,

    /// DataLoader configuration.
    pub dataloader: DataLoaderConfig,

    /// Log every N steps.
    pub log_every: usize,

    /// Checkpoint every N steps (0 to disable).
    pub checkpoint_every: usize,

    /// Evaluate every N steps (0 to disable).
    pub eval_every: usize,

    /// Random seed.
    pub seed: u64,
}

impl DiffusionConfig {
    /// Create a new diffusion config with default values.
    pub fn new(mask_token_id: i64) -> Self {
        Self {
            mask_token_id,
            noise_schedule: NoiseSchedule::Linear,
            num_inference_steps: 64,
            remasking_strategy: RemaskingStrategy::LowConfidence,
            use_elbo_loss: true,
            sft_mode: false,
            min_noise_level: 0.0001,
            sampling_temperature: 1.0,
            training: TrainingConfig::default(),
            dataloader: DataLoaderConfig::default(),
            log_every: 10,
            checkpoint_every: 500,
            eval_every: 100,
            seed: 42,
        }
    }

    /// Enable SFT mode (only mask responses).
    pub fn with_sft_mode(mut self) -> Self {
        self.sft_mode = true;
        self
    }

    /// Set noise schedule.
    pub fn with_noise_schedule(mut self, schedule: NoiseSchedule) -> Self {
        self.noise_schedule = schedule;
        self
    }

    /// Set number of inference steps.
    pub fn with_inference_steps(mut self, steps: usize) -> Self {
        self.num_inference_steps = steps;
        self
    }
}

/// Statistics for a single diffusion training step.
#[derive(Debug, Clone)]
pub struct DiffusionStepStats {
    /// Step number.
    pub step: usize,
    /// Loss value.
    pub loss: f32,
    /// Noise level t used in this step.
    pub noise_level: f32,
    /// Fraction of tokens masked.
    pub mask_ratio: f32,
    /// Learning rate.
    pub learning_rate: f32,
    /// Tokens processed in this step.
    pub tokens: usize,
    /// Gradient norm (if computed).
    pub grad_norm: Option<f32>,
    /// Time taken for this step (ms).
    pub step_time_ms: u64,
}

/// Apply the forward masking process (GPU-native).
///
/// Given clean tokens x_0 and noise level t, produce masked x_t.
/// Returns the masked tokens and a boolean mask array (both on GPU).
pub fn forward_process_gpu(
    x_0: &Array,
    t: f32,
    mask_token_id: i64,
    seed: Option<u64>,
) -> std::result::Result<(Array, Array), Exception> {
    let shape = x_0.shape();

    // Generate random values on GPU: shape = x_0.shape(), values in [0, 1)
    let random_vals = if let Some(s) = seed {
        let key = mlx_rs::random::key(s)?;
        mlx_rs::random::uniform::<_, f32>(0.0_f32, 1.0_f32, shape, Some(&key))?
    } else {
        mlx_rs::random::uniform::<_, f32>(0.0_f32, 1.0_f32, shape, None::<&Array>)?
    };

    // Create mask: positions where random < t should be masked
    let t_arr = Array::from_f32(t);
    let mask = random_vals.lt(&t_arr)?;

    // Create masked tokens: where(mask, mask_token_id, x_0)
    let mask_value = Array::from_int(mask_token_id as i32);
    let mask_tokens = Array::full::<i32>(shape, &mask_value)?;
    let x_0_i32 = x_0.as_dtype(Dtype::Int32)?;
    let x_t = mlx_rs::ops::r#where(&mask, &mask_tokens, &x_0_i32)?;

    Ok((x_t, mask))
}

/// Apply the forward masking process (CPU fallback for compatibility).
///
/// Given clean tokens x_0 and noise level t, produce masked x_t.
pub fn forward_process(
    x_0: &Array,
    t: f32,
    mask_token_id: i64,
    rng: &mut StdRng,
) -> std::result::Result<(Array, Vec<bool>), Exception> {
    x_0.eval()?;
    let shape = x_0.shape();
    let total_elements = shape.iter().product::<i32>() as usize;

    // Get original tokens as slice
    let x_0_i32 = x_0.as_dtype(Dtype::Int32)?;
    x_0_i32.eval()?;
    let original: Vec<i32> = x_0_i32.as_slice().to_vec();

    // Generate mask: each position independently masked with probability t
    let mut masked_tokens = Vec::with_capacity(total_elements);
    let mut mask_flags = Vec::with_capacity(total_elements);

    for &token in &original {
        if rng.gen::<f32>() < t {
            masked_tokens.push(mask_token_id as i32);
            mask_flags.push(true);
        } else {
            masked_tokens.push(token);
            mask_flags.push(false);
        }
    }

    let x_t = Array::from_slice(&masked_tokens, shape);

    Ok((x_t, mask_flags))
}

/// Compute the diffusion loss (GPU-native).
///
/// L(θ) = -E[1/t · Σ_i 1[x_t^i=M] log p_θ(x_0^i|x_t)]
///
/// Uses GPU mask array directly without CPU readback.
pub fn diffusion_loss_gpu(
    logits: &Array,
    targets: &Array,
    mask: &Array,
    t: f32,
    use_elbo_weighting: bool,
    ignore_index: i64,
) -> std::result::Result<Array, Exception> {
    let vocab_size = logits.dim(2);

    // Reshape for cross-entropy computation
    let flat_logits = logits.reshape(&[-1, vocab_size])?;
    let flat_targets = targets.reshape(&[-1])?;
    let flat_mask = mask.reshape(&[-1])?.as_dtype(Dtype::Float32)?;

    // Compute per-token cross-entropy loss using the fused kernel
    let per_token_loss = cross_entropy_loss(&flat_logits, &flat_targets, Some(ignore_index), 0.0)?;

    // Apply mask: only count loss for masked positions
    let masked_loss = per_token_loss.multiply(&flat_mask)?;

    // Sum and normalize by number of masked tokens
    let num_masked = mlx_rs::ops::maximum(&flat_mask.sum(None)?, &Array::from_f32(1.0))?;
    let mean_loss = masked_loss.sum(None)?.divide(&num_masked)?;

    // Apply ELBO weighting: multiply by 1/t
    if use_elbo_weighting && t > 0.0001 {
        let weight = Array::from_f32(1.0 / t);
        mean_loss.multiply(&weight)
    } else {
        Ok(mean_loss)
    }
}

/// Compute the diffusion loss (CPU fallback for compatibility).
///
/// L(θ) = -E[1/t · Σ_i 1[x_t^i=M] log p_θ(x_0^i|x_t)]
pub fn diffusion_loss(
    logits: &Array,
    targets: &Array,
    mask_flags: &[bool],
    t: f32,
    use_elbo_weighting: bool,
    ignore_index: i64,
) -> std::result::Result<Array, Exception> {
    let vocab_size = logits.dim(2) as usize;

    // Reshape for cross-entropy
    let flat_logits = logits.reshape(&[-1, vocab_size as i32])?;
    let flat_targets = targets.reshape(&[-1])?;

    // Compute log softmax
    let log_probs = nn::log_softmax(&flat_logits, -1)?;
    log_probs.eval()?;

    // Get targets as i64
    let flat_targets_i64 = flat_targets.as_dtype(Dtype::Int64)?;
    flat_targets_i64.eval()?;
    let target_vec: Vec<i64> = flat_targets_i64.as_slice().to_vec();

    // Get log probs as slice
    let log_probs_data: Vec<f32> = log_probs.as_slice().to_vec();

    // Compute masked loss manually
    let mut total_loss = 0.0_f32;
    let mut num_masked = 0_usize;

    for (i, (&target, &is_masked)) in target_vec.iter().zip(mask_flags.iter()).enumerate() {
        if is_masked && target != ignore_index && target >= 0 && (target as usize) < vocab_size {
            let log_prob = log_probs_data[i * vocab_size + target as usize];
            total_loss -= log_prob; // Cross-entropy = -log p
            num_masked += 1;
        }
    }

    // Apply ELBO weighting: 1/t
    if use_elbo_weighting && t > 0.0001 {
        total_loss /= t;
    }

    // Mean over masked positions
    let mean_loss = if num_masked > 0 {
        total_loss / num_masked as f32
    } else {
        0.0
    };

    Ok(Array::from_f32(mean_loss))
}

/// Diffusion training loop.
pub struct DiffusionTrainingLoop {
    /// Configuration.
    config: DiffusionConfig,
    /// Current step.
    step: usize,
    /// Current epoch.
    epoch: usize,
    /// Running loss (EMA).
    running_loss: f64,
    /// Total tokens processed.
    total_tokens: usize,
    /// Accumulated gradients.
    accumulated_grads: Option<FlattenedModuleParam>,
    /// Accumulation step counter.
    accumulation_step: usize,
    /// Random number generator.
    rng: StdRng,
}

impl DiffusionTrainingLoop {
    /// Create a new diffusion training loop.
    pub fn new(config: DiffusionConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);

        Self {
            config,
            step: 0,
            epoch: 0,
            running_loss: 0.0,
            total_tokens: 0,
            accumulated_grads: None,
            accumulation_step: 0,
            rng,
        }
    }

    /// Get current learning rate based on scheduler.
    pub fn get_learning_rate(&self) -> f32 {
        let warmup = self.config.training.warmup_steps;
        let total_steps = self.config.training.max_steps.unwrap_or(10000);
        let base_lr = self.config.training.learning_rate as f32;

        if self.step < warmup {
            base_lr * (self.step as f32 / warmup.max(1) as f32)
        } else {
            match self.config.training.lr_scheduler {
                LrSchedulerType::Constant => base_lr,
                LrSchedulerType::Linear => {
                    let decay_steps = (total_steps - warmup) as f32;
                    let current = (self.step - warmup) as f32;
                    base_lr * (1.0 - current / decay_steps).max(0.0)
                }
                LrSchedulerType::Cosine => {
                    let decay_steps = (total_steps - warmup) as f32;
                    let current = (self.step - warmup) as f32;
                    let progress = (current / decay_steps).min(1.0);
                    base_lr * 0.5 * (1.0 + (std::f64::consts::PI as f32 * progress).cos())
                }
                _ => base_lr,
            }
        }
    }

    /// Clip gradients by global norm.
    fn clip_gradients(&self, grads: &mut FlattenedModuleParam) -> Result<Option<f32>> {
        let max_norm = self.config.training.max_grad_norm as f32;
        if max_norm <= 0.0 {
            return Ok(None);
        }

        let mut total_norm_sq = 0.0_f32;
        for (_, grad) in grads.iter() {
            grad.eval()?;
            let norm_sq = grad.multiply(grad)?.sum(None)?;
            norm_sq.eval()?;
            total_norm_sq += norm_sq.item::<f32>();
        }
        let total_norm = total_norm_sq.sqrt();

        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            for (_, grad) in grads.iter_mut() {
                *grad = grad.multiply(&Array::from_f32(scale))?;
            }
        }

        Ok(Some(total_norm))
    }

    /// Accumulate gradients.
    fn accumulate_gradients(&mut self, new_grads: FlattenedModuleParam) -> Result<()> {
        let accum_steps = self.config.training.gradient_accumulation_steps;

        match &mut self.accumulated_grads {
            None => {
                let scale = 1.0 / accum_steps as f32;
                let scaled: FlattenedModuleParam = new_grads
                    .into_iter()
                    .map(|(k, v)| {
                        let scaled = v.multiply(&Array::from_f32(scale)).unwrap();
                        (k, scaled)
                    })
                    .collect();
                self.accumulated_grads = Some(scaled);
            }
            Some(acc) => {
                let scale = 1.0 / accum_steps as f32;
                for (key, new_grad) in new_grads {
                    if let Some(existing) = acc.get_mut(&key) {
                        let scaled = new_grad.multiply(&Array::from_f32(scale))?;
                        *existing = existing.add(&scaled)?;
                    }
                }
            }
        }

        self.accumulation_step += 1;
        Ok(())
    }

    /// Check if we should apply accumulated gradients.
    fn should_apply_gradients(&self) -> bool {
        self.accumulation_step >= self.config.training.gradient_accumulation_steps
    }

    /// Take accumulated gradients and reset counter.
    fn take_accumulated_gradients(&mut self) -> Option<FlattenedModuleParam> {
        self.accumulation_step = 0;
        self.accumulated_grads.take()
    }

    /// Perform a single diffusion training step (GPU-native, high performance).
    ///
    /// This version stays entirely on GPU, avoiding CPU readbacks for masking.
    pub fn train_step<M, O>(
        &mut self,
        model: &mut M,
        batch: &TrainingBatch,
        optimizer: &mut O,
    ) -> Result<DiffusionStepStats>
    where
        M: TrainableModel,
        O: Optimizer,
    {
        let start_time = std::time::Instant::now();
        let batch_tokens = batch.batch_size.checked_mul(batch.seq_len).unwrap_or(usize::MAX);

        // 1. Sample noise level t ~ U(min_noise, 1]
        let t: f32 = self.rng.gen_range(self.config.min_noise_level..=1.0);

        // 2. Apply GPU-native forward masking process
        // Use step as seed for reproducibility while staying on GPU
        let seed = self.step as u64 + self.config.seed * 1000;
        let (x_t, mask) = forward_process_gpu(
            &batch.input_ids,
            t,
            self.config.mask_token_id,
            Some(seed),
        )?;

        // IMPORTANT: Eval x_t and mask before any further operations to avoid
        // Metal command buffer conflicts (AGXG16X tryCoalescingPreviousComputeCommandEncoder)
        x_t.eval()?;
        mask.eval()?;

        // Compute mask ratio on GPU and read single scalar
        let mask_f32 = mask.as_dtype(Dtype::Float32)?;
        let mask_sum = mask_f32.sum(None)?;
        mask_sum.eval()?;
        let mask_ratio = mask_sum.item::<f32>() / batch_tokens as f32;

        // 3. Prepare targets on GPU: original tokens for masked positions, -100 for unmasked
        // targets = where(mask, original_tokens, -100)
        let ignore_value = Array::from_int(-100);
        let ignore_tokens = Array::full::<i32>(batch.input_ids.shape(), &ignore_value)?;
        let original_i32 = batch.input_ids.as_dtype(Dtype::Int32)?;
        let targets = mlx_rs::ops::r#where(&mask, &original_i32, &ignore_tokens)?;

        // Sync all inputs before loss computation
        targets.eval()?;

        // 4. Define loss function for autodiff
        let loss_fn = |model: &mut M,
                       (input, targets): (&Array, &Array)|
         -> std::result::Result<Array, Exception> {
            let logits = model
                .forward(input, None)
                .map_err(|e| Exception::custom(e.to_string()))?;

            // Use standard cross-entropy with ignore_index for non-masked tokens
            let vocab_size = logits.dim(2);
            let flat_logits = logits.reshape(&[-1, vocab_size])?;
            let flat_targets = targets.reshape(&[-1])?;

            cross_entropy_loss(&flat_logits, &flat_targets, Some(-100), 0.0)?
                .mean(None)
                .map_err(|e| e.into())
        };

        // Create value_and_grad function
        let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);

        // 5. Compute loss and gradients
        let (loss, grads) = loss_and_grad_fn(model, (&x_t, &targets))?;

        loss.eval()?;
        let mut loss_val = loss.item::<f32>();

        // Apply ELBO weighting
        if self.config.use_elbo_loss && t > 0.0001 {
            loss_val /= t;
        }

        // 6. Accumulate gradients
        self.accumulate_gradients(grads)?;

        // 7. Apply gradients if accumulation is complete
        let grad_norm = if self.should_apply_gradients() {
            if let Some(mut accumulated) = self.take_accumulated_gradients() {
                let grad_norm = self.clip_gradients(&mut accumulated)?;
                optimizer.update(model, accumulated)?;
                eval_params(model.parameters())?;
                grad_norm
            } else {
                None
            }
        } else {
            None
        };

        // Update stats
        self.step += 1;
        self.total_tokens += batch_tokens;
        self.running_loss = if self.step == 1 {
            loss_val as f64
        } else {
            0.99 * self.running_loss + 0.01 * loss_val as f64
        };

        let step_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(DiffusionStepStats {
            step: self.step,
            loss: loss_val,
            noise_level: t,
            mask_ratio,
            learning_rate: self.get_learning_rate(),
            tokens: batch_tokens,
            grad_norm,
            step_time_ms,
        })
    }

    /// Run the full diffusion training loop.
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
        let mut optimizer = AdamWBuilder::new(self.config.training.learning_rate as f32)
            .weight_decay(self.config.training.weight_decay as f32)
            .build()
            .map_err(|_| SftError::Mlx(Exception::custom("Failed to build optimizer")))?;

        let max_steps = self.config.training.max_steps;
        let num_epochs = self.config.training.num_epochs;

        tracing::info!(
            "Starting diffusion training: {} trainable params, batch_size={}, grad_accum={}",
            model.num_trainable_params(),
            self.config.training.batch_size,
            self.config.training.gradient_accumulation_steps
        );

        let mut best_eval_loss = f64::MAX;

        for epoch in 0..num_epochs {
            self.epoch = epoch;
            tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);

            let mut dataloader =
                DataLoader::new(train_dataset.clone(), self.config.dataloader.clone(), None);

            while let Some(batch) = dataloader.next_batch() {
                let stats = self.train_step(model, &batch, &mut optimizer)?;

                if self.step % self.config.log_every == 0 {
                    let tokens_per_sec = if stats.step_time_ms > 0 {
                        (stats.tokens as f64 / stats.step_time_ms as f64) * 1000.0
                    } else {
                        0.0
                    };

                    tracing::info!(
                        "Step {}: loss={:.4}, t={:.3}, mask={:.1}%, lr={:.2e}, tok/s={:.0}{}",
                        stats.step,
                        self.running_loss,
                        stats.noise_level,
                        stats.mask_ratio * 100.0,
                        stats.learning_rate,
                        tokens_per_sec,
                        stats
                            .grad_norm
                            .map(|n| format!(", grad_norm={:.2}", n))
                            .unwrap_or_default()
                    );
                }

                if self.config.eval_every > 0
                    && self.step % self.config.eval_every == 0
                    && eval_dataset.is_some()
                {
                    let metrics = self.evaluate(model, eval_dataset.as_ref().unwrap())?;

                    tracing::info!(
                        "Step {}: eval_loss={:.4}, ppl={:.2}",
                        self.step,
                        metrics.loss,
                        metrics.perplexity,
                    );

                    if metrics.loss < best_eval_loss {
                        best_eval_loss = metrics.loss;

                        if let Some(manager) = checkpoint_manager {
                            self.save_checkpoint(model, manager, true, Some(metrics.loss))?;
                        }
                    }
                }

                if self.config.checkpoint_every > 0 && self.step % self.config.checkpoint_every == 0
                {
                    if let Some(manager) = checkpoint_manager {
                        self.save_checkpoint(model, manager, false, None)?;
                    }
                }

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
            "Diffusion training complete: {} steps, {:.4} final loss",
            self.step,
            self.running_loss
        );

        Ok(())
    }

    /// Evaluate the model on a dataset.
    pub fn evaluate<M>(&mut self, model: &mut M, dataset: &TrainingDataset) -> Result<EvalMetrics>
    where
        M: TrainableModel,
    {
        let mut eval_config = self.config.dataloader.clone();
        eval_config.shuffle = false;
        eval_config.drop_last = false;

        let dataloader = DataLoader::new(dataset.clone(), eval_config, None);

        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let eval_t = 0.5_f32;

        for batch in dataloader {
            let (x_t, mask_flags) =
                forward_process(&batch.input_ids, eval_t, self.config.mask_token_id, &mut self.rng)?;

            let logits = model
                .forward(&x_t, None)
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

            let targets = batch.input_ids.as_dtype(Dtype::Int64)?;
            let loss = diffusion_loss(&logits, &targets, &mask_flags, eval_t, false, -100)?;
            loss.eval()?;
            total_loss += loss.item::<f32>() as f64;
            num_batches += 1;
        }

        let avg_loss = if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        };

        let perplexity = if avg_loss < 100.0 {
            avg_loss.exp()
        } else {
            f64::MAX
        };

        Ok(EvalMetrics {
            loss: avg_loss,
            perplexity,
            accuracy: None,
            custom: HashMap::new(),
        })
    }

    /// Save a checkpoint.
    pub fn save_checkpoint<M>(
        &self,
        model: &M,
        manager: &CheckpointManager,
        is_best: bool,
        eval_loss: Option<f64>,
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

        let path = manager.save_checkpoint(&lora_params, &metadata, is_best)?;
        tracing::info!("Saved checkpoint to {:?}", path);

        Ok(path)
    }

    /// Get current training step.
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
}

/// Diffusion sampler for inference.
pub struct DiffusionSampler {
    /// Mask token ID.
    mask_token_id: i64,
    /// Number of diffusion steps.
    num_steps: usize,
    /// Sampling temperature.
    temperature: f32,
    /// Remasking strategy.
    remasking: RemaskingStrategy,
    /// Random number generator.
    rng: StdRng,
}

impl DiffusionSampler {
    /// Create a new diffusion sampler.
    pub fn new(config: &DiffusionConfig) -> Self {
        Self {
            mask_token_id: config.mask_token_id,
            num_steps: config.num_inference_steps,
            temperature: config.sampling_temperature,
            remasking: config.remasking_strategy,
            rng: StdRng::seed_from_u64(config.seed),
        }
    }

    /// Sample from the model.
    pub fn sample<M>(
        &mut self,
        model: &mut M,
        prompt: &Array,
        max_new_tokens: usize,
    ) -> std::result::Result<Array, Exception>
    where
        M: TrainableModel,
    {
        prompt.eval()?;
        let prompt_len = prompt.dim(1) as usize;
        let total_len = prompt_len + max_new_tokens;

        // Get prompt tokens
        let prompt_i32 = prompt.as_dtype(Dtype::Int32)?;
        prompt_i32.eval()?;
        let prompt_vec: Vec<i32> = prompt_i32.as_slice().to_vec();

        // Initialize: prompt + masks
        let mut tokens: Vec<i32> = prompt_vec.clone();
        tokens.extend(vec![self.mask_token_id as i32; max_new_tokens]);

        // Track which positions are still masked
        let mut is_masked: Vec<bool> = vec![false; prompt_len];
        is_masked.extend(vec![true; max_new_tokens]);

        // Diffusion steps
        for step in 0..self.num_steps {
            let t = 1.0 - (step as f32 / self.num_steps as f32);
            let s = 1.0 - ((step + 1) as f32 / self.num_steps as f32);

            // Create input array
            let x_t = Array::from_slice(&tokens, &[1, total_len as i32]);

            // Forward pass
            let logits = model
                .forward(&x_t, None)
                .map_err(|e| Exception::custom(e.to_string()))?;

            // Get predictions and confidences
            // softmax = exp(log_softmax)
            let log_probs = nn::log_softmax(&logits, -1)?;
            let probs = log_probs.exp()?;
            probs.eval()?;

            let probs_data: Vec<f32> = probs.as_slice().to_vec();
            let vocab_size = logits.dim(2) as usize;

            // Predict masked positions
            let mut confidences = vec![1.0_f32; total_len];
            for i in prompt_len..total_len {
                if is_masked[i] {
                    // Find argmax
                    let offset = i * vocab_size;
                    let mut best_idx = 0;
                    let mut best_prob = 0.0_f32;
                    for j in 0..vocab_size {
                        let prob = probs_data[offset + j];
                        if prob > best_prob {
                            best_prob = prob;
                            best_idx = j;
                        }
                    }
                    tokens[i] = best_idx as i32;
                    confidences[i] = best_prob;
                    is_masked[i] = false;
                }
            }

            // Remask if not final step
            if step < self.num_steps - 1 {
                let remask_ratio = s / t;
                let num_to_remask =
                    ((max_new_tokens as f32) * remask_ratio).ceil() as usize;

                match self.remasking {
                    RemaskingStrategy::Random => {
                        // Random remasking
                        let mut indices: Vec<usize> =
                            (prompt_len..total_len).collect();
                        for i in 0..num_to_remask.min(indices.len()) {
                            let j = self.rng.gen_range(i..indices.len());
                            indices.swap(i, j);
                            let idx = indices[i];
                            tokens[idx] = self.mask_token_id as i32;
                            is_masked[idx] = true;
                        }
                    }
                    RemaskingStrategy::LowConfidence => {
                        // Sort by confidence (ascending) and remask lowest
                        let mut indexed: Vec<(usize, f32)> = (prompt_len..total_len)
                            .map(|i| (i, confidences[i]))
                            .collect();
                        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                        for i in 0..num_to_remask.min(indexed.len()) {
                            let idx = indexed[i].0;
                            tokens[idx] = self.mask_token_id as i32;
                            is_masked[idx] = true;
                        }
                    }
                    RemaskingStrategy::SemiAutoregressive { block_size } => {
                        // Remask tokens beyond current block
                        let current_block = (step + 1) * block_size;
                        for i in (prompt_len + current_block)..total_len {
                            tokens[i] = self.mask_token_id as i32;
                            is_masked[i] = true;
                        }
                    }
                }
            }
        }

        Ok(Array::from_slice(&tokens, &[1, total_len as i32]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_noise_schedule_linear() {
        let schedule = NoiseSchedule::Linear;

        assert!((schedule.alpha(0.0) - 1.0).abs() < 1e-6);
        assert!((schedule.alpha(0.5) - 0.5).abs() < 1e-6);
        assert!((schedule.alpha(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    #[serial]
    fn test_noise_schedule_polynomial() {
        let schedule = NoiseSchedule::Polynomial2;

        assert!((schedule.alpha(0.0) - 1.0).abs() < 1e-6);
        assert!((schedule.alpha(0.5) - 0.25).abs() < 1e-6);
        assert!((schedule.alpha(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    #[serial]
    fn test_diffusion_config() {
        let config = DiffusionConfig::new(32000)
            .with_noise_schedule(NoiseSchedule::Cosine)
            .with_inference_steps(128)
            .with_sft_mode();

        assert_eq!(config.mask_token_id, 32000);
        assert_eq!(config.num_inference_steps, 128);
        assert!(config.sft_mode);
    }

    #[test]
    #[serial]
    fn test_forward_process() {
        let x_0 = Array::from_slice(&[1_i32, 2, 3, 4, 5], &[1, 5]);
        let mut rng = StdRng::seed_from_u64(42);

        // With t=1.0, all tokens should be masked
        let (x_t, mask_flags) = forward_process(&x_0, 1.0, 0, &mut rng).unwrap();
        x_t.eval().unwrap();

        let x_t_slice: Vec<i32> = x_t.as_slice().to_vec();
        assert!(x_t_slice.iter().all(|&x| x == 0));
        assert!(mask_flags.iter().all(|&m| m));

        // With t=0, no tokens should be masked
        let (x_t, mask_flags) = forward_process(&x_0, 0.0, 0, &mut rng).unwrap();
        x_t.eval().unwrap();

        let x_t_slice: Vec<i32> = x_t.as_slice().to_vec();
        assert_eq!(x_t_slice, vec![1, 2, 3, 4, 5]);
        assert!(mask_flags.iter().all(|&m| !m));
    }

    #[test]
    #[serial]
    fn test_training_loop_creation() {
        let config = DiffusionConfig::new(32000);
        let training_loop = DiffusionTrainingLoop::new(config);

        assert_eq!(training_loop.current_step(), 0);
        assert_eq!(training_loop.current_epoch(), 0);
    }

    #[test]
    #[serial]
    fn test_sampler_creation() {
        let config = DiffusionConfig::new(32000).with_inference_steps(64);

        let sampler = DiffusionSampler::new(&config);

        assert_eq!(sampler.num_steps, 64);
        assert_eq!(sampler.mask_token_id, 32000);
    }
}
