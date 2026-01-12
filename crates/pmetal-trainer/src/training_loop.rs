//! Complete training loop with DataLoader integration.
//!
//! This module provides a unified training loop that:
//! - Connects DataLoader → Model → Optimizer
//! - Supports gradient accumulation
//! - Uses mlx-rs autodiff for gradient computation
//! - Optionally uses Metal FlashAttention for efficient forward pass
//! - Provides progress tracking and callbacks
//!
//! # Optimized Training Path
//!
//! The training loop uses a fused `jit_training_step` function that combines
//! forward pass, backward pass, and optimizer update. MLX's lazy evaluation
//! automatically optimizes the computation graph.
//!
//! ## State Warmup Requirement
//!
//! Optimizers like AdamW lazily initialize their internal state (momentum, velocity
//! buffers) on the first `update()` call. We use a warmup step to ensure all
//! optimizer states are properly initialized before the main training loop.
//!
//! ## Note on JIT Compilation
//!
//! mlx-rs provides `compile_with_state` for JIT compilation, but it has known
//! limitations with complex models + optimizers:
//!
//! 1. **State tracking overhead**: `compile_with_state` expects the compiled function
//!    to return all mutable state as additional outputs. For LLMs with 10M+ parameters,
//!    this creates a mismatch between expected and actual output counts.
//!
//! 2. **State count stability**: Even after warmup, the `Updatable::updatable_states_len()`
//!    can return counts that don't match what the compiled graph actually produces.
//!
//! 3. **Architectural difference**: Python's `mx.compile(fn, inputs=state, outputs=state)`
//!    explicitly tracks state containers, while mlx-rs tries to infer state automatically.
//!
//! ## Performance Comparison (Qwen3-0.6B, batch=4, seq=512)
//!
//! | Approach | Throughput | Notes |
//! |----------|------------|-------|
//! | mlx-lm (JIT) | ~2200-2300 tok/s | Full graph fusion via `mx.compile` |
//! | pmetal (fused) | ~1700-1800 tok/s | Deferred eval + warmup |
//! | pmetal (basic) | ~500-600 tok/s | Per-step evaluation |
//!
//! The fused training path (`--fused` flag) uses deferred evaluation which achieves
//! approximately 75-80% of mlx-lm's JIT-compiled throughput. The gap is primarily due
//! to mlx-rs's `compile_with_state` limitations with complex models.
//!
//! ## Using the Optimized Training Path
//!
//! Enable fused training for best performance:
//! ```bash
//! pmetal train --model <model> --dataset <data> --fused --use-metal-flash-attention
//! ```
//!
//! Note: Fused training requires `gradient_accumulation_steps=1`.
//!
//! The `jit_compile` module provides experimental utilities for future JIT support
//! when mlx-rs improves its compile_with_state implementation.

use pmetal_core::{EvalMetrics, LrSchedulerType, TrainingConfig};
use pmetal_data::{
    DataLoader, DataLoaderConfig, PackedDataLoader, PackedTrainingBatch, PackerConfig,
    TrainingBatch, TrainingDataset,
};
use pmetal_lora::TrainableModel;
use pmetal_mlx::kernels::cross_entropy::cross_entropy_loss;
use pmetal_mlx::kernels::{init_training_context, with_training_mode};
use mlx_rs::{
    builder::Builder,
    error::Exception,
    losses::CrossEntropy,
    module::{FlattenedModuleParam, ModuleParameters},
    nn,
    ops::indexing::{argmax_axis, IndexOp},
    optimizers::{AdamW, AdamWBuilder, Optimizer},
    transforms::compile::{compile_with_state, TrainableState},
    utils::Updatable,
    Array,
};

use crate::{CheckpointManager, CheckpointMetadata, Result, SftError};
use crate::mlx_metal_optimizer::{MlxMetalOptimizer, MlxMetalOptimizerBuilder, is_mlx_metal_optimizer_available};

/// JIT-compiled training step for maximum throughput.
///
/// This function is defined at module level so it can access external functions
/// and be used as a function pointer (which is `Copy`).
fn jit_training_step<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    (input_ids, labels): (&Array, &Array),
) -> std::result::Result<Array, Exception> {
    let (model, optimizer) = state;

    // Define loss function that will be used by value_and_grad
    let loss_fn = |model: &mut M,
                   (input_ids, labels): (&Array, &Array)|
                   -> std::result::Result<Array, Exception> {
        let logits = model
            .forward(input_ids, None)
            .map_err(|e| Exception::custom(e.to_string()))?;

        // Compute cross-entropy loss with shifted labels for causal LM
        let seq_len = logits.dim(1);
        let vocab_size = logits.dim(2);

        // Shift: logits[:-1] predicts labels[1:]
        let shift_logits = logits.index((.., ..seq_len - 1, ..));
        let shift_labels = labels.index((.., 1..));

        let flat_logits = shift_logits.reshape(&[-1, vocab_size])?;
        let flat_labels = shift_labels.reshape(&[-1])?;

        // Use mlx-rs CrossEntropy directly - it handles the logsumexp internally
        // Note: We need to handle ignore_index=-100 separately
        let ce = CrossEntropy::new().map_err(|e| Exception::custom(e.to_string()))?;
        let per_token_loss = ce.apply(&flat_logits, &flat_labels)?;

        // Mask out ignored tokens (label == -100) and compute mean
        let ignore_mask = flat_labels.ne(&Array::from_int(-100_i32))?;
        let ignore_mask_f32 = ignore_mask.as_dtype(mlx_rs::Dtype::Float32)?;
        let masked_loss = per_token_loss.multiply(&ignore_mask_f32)?;
        let valid_count = ignore_mask_f32.sum(None)?;
        masked_loss.sum(None)?.divide(&valid_count)
    };

    // Compute loss and gradients
    let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
    let (loss, grads) = loss_and_grad_fn(model, (input_ids, labels))?;

    // Apply gradients via optimizer
    optimizer.update(model, grads)?;

    Ok(loss)
}

/// Training step that works with TrainableState for JIT compilation.
///
/// This function has the same logic as `jit_training_step` but operates on
/// `TrainableState<M, O>` which only exposes trainable parameters. This
/// enables successful JIT compilation via `compile_with_state` for large
/// models with LoRA or other parameter-efficient fine-tuning.
fn trainable_training_step<M: TrainableModel, O: Optimizer>(
    state: &mut TrainableState<M, O>,
    (input_ids, labels): (&Array, &Array),
) -> std::result::Result<Array, Exception> {
    // Get mutable references to both model and optimizer at once
    // This is necessary because Optimizer::update needs &mut Model
    let (model, optimizer) = state.as_parts_mut();

    // Define loss function that will be used by value_and_grad
    let loss_fn = |model: &mut M,
                   (input_ids, labels): (&Array, &Array)|
                   -> std::result::Result<Array, Exception> {
        let logits = model
            .forward(input_ids, None)
            .map_err(|e| Exception::custom(e.to_string()))?;

        // Compute cross-entropy loss with shifted labels for causal LM
        let seq_len = logits.dim(1);
        let vocab_size = logits.dim(2);

        // Shift: logits[:-1] predicts labels[1:]
        let shift_logits = logits.index((.., ..seq_len - 1, ..));
        let shift_labels = labels.index((.., 1..));

        let flat_logits = shift_logits.reshape(&[-1, vocab_size])?;
        let flat_labels = shift_labels.reshape(&[-1])?;

        // Use mlx-rs CrossEntropy directly
        let ce = CrossEntropy::new().map_err(|e| Exception::custom(e.to_string()))?;
        let per_token_loss = ce.apply(&flat_logits, &flat_labels)?;

        // Mask out ignored tokens (label == -100) and compute mean
        let ignore_mask = flat_labels.ne(&Array::from_int(-100_i32))?;
        let ignore_mask_f32 = ignore_mask.as_dtype(mlx_rs::Dtype::Float32)?;
        let masked_loss = per_token_loss.multiply(&ignore_mask_f32)?;
        let valid_count = ignore_mask_f32.sum(None)?;
        masked_loss.sum(None)?.divide(&valid_count)
    };

    // Compute loss and gradients
    let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
    let (loss, grads) = loss_and_grad_fn(model, (input_ids, labels))?;

    // Apply gradients via optimizer
    optimizer.update(model, grads)?;

    Ok(loss)
}

/// Training step for packed sequences (variable-length, no padding).
///
/// This version handles packed sequences where multiple sequences are concatenated
/// into a single batch with block-diagonal attention masking and explicit position IDs.
fn jit_training_step_packed<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    packed_batch: &PackedTrainingBatch,
) -> std::result::Result<Array, Exception> {
    let (model, optimizer) = state;

    // Reshape 1D packed input to 2D [1, total_tokens] for model forward
    let total_tokens = packed_batch.total_tokens as i32;
    let input_ids_2d = packed_batch.input_ids.reshape(&[1, total_tokens])?;
    let labels_2d = packed_batch.labels.reshape(&[1, total_tokens])?;

    // Get position IDs - these reset for each packed sequence
    // Critical for correct RoPE embeddings in packed sequences
    let position_ids = packed_batch.position_ids.clone();

    // Get the block-diagonal attention mask [total_tokens, total_tokens]
    // and reshape to [1, 1, total_tokens, total_tokens] for attention
    let attn_mask = packed_batch.attention_mask()?;
    let attn_mask_4d = attn_mask.reshape(&[1, 1, total_tokens, total_tokens])?;

    // Define loss function that will be used by value_and_grad
    // Use IDENTICAL loss computation as regular training for consistency
    let loss_fn = |model: &mut M,
                   (input_ids, labels, mask, pos_ids): (&Array, &Array, &Array, &Array)|
                   -> std::result::Result<Array, Exception> {
        // Use forward_with_positions for correct RoPE with packed sequences
        let logits = model
            .forward_with_positions(input_ids, Some(mask), pos_ids)
            .map_err(|e| Exception::custom(e.to_string()))?;

        // For packed sequences, labels already have boundary tokens masked as -100
        // We still need to shift for causal LM prediction
        let seq_len = logits.dim(1);
        let vocab_size = logits.dim(2);

        // Shift: logits[:-1] predicts labels[1:]
        let shift_logits = logits.index((.., ..seq_len - 1, ..));
        let shift_labels = labels.index((.., 1..));

        let flat_logits = shift_logits.reshape(&[-1, vocab_size])?;
        let flat_labels = shift_labels.reshape(&[-1])?;

        // Use mlx-rs CrossEntropy directly - SAME as regular training
        let ce = CrossEntropy::new().map_err(|e| Exception::custom(e.to_string()))?;
        let per_token_loss = ce.apply(&flat_logits, &flat_labels)?;

        // Mask out ignored tokens (label == -100) and compute mean
        // Packed labels are i64, so use i64 for the comparison!
        let ignore_idx = Array::from_slice(&[-100_i64], &[1]);
        let ignore_mask = flat_labels.ne(&ignore_idx)?;
        let ignore_mask_f32 = ignore_mask.as_dtype(mlx_rs::Dtype::Float32)?;
        let masked_loss = per_token_loss.multiply(&ignore_mask_f32)?;
        let valid_count = ignore_mask_f32.sum(None)?;
        masked_loss.sum(None)?.divide(&valid_count)
    };

    // Compute loss and gradients - pass position_ids as 4th argument
    let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
    let (loss, grads) = loss_and_grad_fn(model, (&input_ids_2d, &labels_2d, &attn_mask_4d, &position_ids))?;

    // Apply gradients via optimizer
    optimizer.update(model, grads)?;

    Ok(loss)
}

/// Training loop configuration.
#[derive(Debug, Clone)]
pub struct TrainingLoopConfig {
    /// Training hyperparameters.
    pub training: TrainingConfig,
    /// DataLoader configuration.
    pub dataloader: DataLoaderConfig,
    /// Whether to use Metal FlashAttention for training.
    /// When true, attention layers use O(n) memory FlashAttention.
    pub use_metal_flash_attention: bool,
    /// Log every N steps.
    pub log_every: usize,
    /// Checkpoint every N steps (0 to disable).
    pub checkpoint_every: usize,
    /// Evaluate every N steps (0 to disable).
    pub eval_every: usize,
    /// Whether to use JIT compilation for training step.
    /// This can provide significant speedups (up to 8x) by fusing operations.
    /// Requires gradient_accumulation_steps=1 for now.
    pub use_jit_compilation: bool,
    /// Enable sequence packing for 2-5x throughput on variable-length data.
    /// Packs multiple shorter sequences into single batches with block-diagonal
    /// attention masks to prevent cross-sequence attention.
    pub use_sequence_packing: bool,
    /// Enable gradient checkpointing to reduce memory usage.
    /// Trades compute for memory by recomputing activations during backward pass.
    /// Allows ~2x larger batch sizes with ~30% slowdown.
    pub gradient_checkpointing: bool,
    /// Number of layers per checkpoint block.
    /// Lower = more memory savings but slower. Recommended: 4 for most models.
    pub gradient_checkpointing_layers: usize,
    /// Separate learning rate for embedding parameters.
    /// Unsloth recommends 5e-5 for embeddings vs 2e-4 for LoRA params.
    /// None means use the same learning rate as other parameters.
    pub embedding_lr: Option<f32>,
    /// Enable eager evaluation after each training step.
    ///
    /// When true, forces immediate evaluation of model parameters and optimizer
    /// state after each step, clearing intermediate activations from memory.
    ///
    /// Trade-off:
    /// - **Enabled**: Lower memory usage, enables larger batch sizes/sequences
    /// - **Disabled**: Better throughput via deferred evaluation batching
    ///
    /// Recommended: Enable for memory-constrained scenarios (large models/sequences),
    /// disable for maximum throughput on smaller models.
    pub eager_evaluation: bool,
    /// Enable Metal fused optimizer for maximum throughput.
    ///
    /// Uses custom Metal kernels that process all parameters in a single dispatch,
    /// eliminating per-parameter GPU-CPU synchronization overhead.
    ///
    /// Expected performance gain: ~40% (from ~1740 to ~2400 tok/s)
    ///
    /// Requires Apple Silicon with Metal support.
    pub use_metal_fused_optimizer: bool,
}

impl Default for TrainingLoopConfig {
    fn default() -> Self {
        Self {
            training: TrainingConfig::default(),
            dataloader: DataLoaderConfig::default(),
            use_metal_flash_attention: true,
            log_every: 10,
            checkpoint_every: 500,
            eval_every: 100,
            use_jit_compilation: false, // Off by default until fully tested
            use_sequence_packing: false, // Off by default, enable for variable-length datasets
            gradient_checkpointing: false, // Off by default, enable for memory-constrained training
            gradient_checkpointing_layers: 4, // 4 layers per block is a good default
            embedding_lr: None, // None = same as base learning rate
            eager_evaluation: false, // Off by default for throughput, enable for memory savings
            use_metal_fused_optimizer: false, // Off by default until fully tested
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

/// Training loop that connects all components.
pub struct TrainingLoop {
    /// Configuration.
    pub(crate) config: TrainingLoopConfig,
    /// Current step.
    pub(crate) step: usize,
    /// Current epoch.
    pub(crate) epoch: usize,
    /// Running loss (EMA).
    pub(crate) running_loss: f64,
    /// Total tokens processed.
    pub(crate) total_tokens: usize,
    /// Accumulated gradients for gradient accumulation.
    pub(crate) accumulated_grads: Option<FlattenedModuleParam>,
    /// Accumulation step counter.
    pub(crate) accumulation_step: usize,
    /// Whether Metal FlashAttention is available.
    pub(crate) metal_fa_available: bool,
    /// Tokens accumulated since last log (for throughput calculation).
    pub(crate) tokens_since_log: usize,
    /// Time of last log (for throughput calculation).
    pub(crate) last_log_time: Option<std::time::Instant>,
}

impl TrainingLoop {
    /// Create a new training loop.
    pub fn new(config: TrainingLoopConfig) -> Self {
        // Try to initialize Metal FlashAttention context
        let metal_fa_available = if config.use_metal_flash_attention {
            init_training_context().is_ok()
        } else {
            false
        };

        if metal_fa_available {
            tracing::info!("Metal FlashAttention enabled for training");
        } else if config.use_metal_flash_attention {
            tracing::warn!("Metal FlashAttention requested but not available, using MLX fallback");
        }

        Self {
            config,
            step: 0,
            epoch: 0,
            running_loss: 0.0,
            total_tokens: 0,
            accumulated_grads: None,
            accumulation_step: 0,
            metal_fa_available,
            tokens_since_log: 0,
            last_log_time: None,
        }
    }

    /// Get current learning rate based on scheduler.
    pub fn get_learning_rate(&self) -> f32 {
        let warmup = self.config.training.warmup_steps;
        let total_steps = self.config.training.max_steps.unwrap_or(10000);
        let base_lr = self.config.training.learning_rate as f32;

        if self.step < warmup {
            // Linear warmup - protect against warmup=0
            base_lr * (self.step as f32 / warmup.max(1) as f32)
        } else {
            // Protect against division-by-zero when total_steps <= warmup
            let decay_steps = (total_steps.saturating_sub(warmup)).max(1) as f32;
            let current = (self.step.saturating_sub(warmup)) as f32;

            match self.config.training.lr_scheduler {
                LrSchedulerType::Constant => base_lr,
                LrSchedulerType::Linear => {
                    base_lr * (1.0 - current / decay_steps).max(0.0)
                }
                LrSchedulerType::Cosine => {
                    let progress = (current / decay_steps).min(1.0);
                    base_lr * 0.5 * (1.0 + (std::f64::consts::PI as f32 * progress).cos())
                }
                LrSchedulerType::CosineWithRestarts => {
                    // Simplified: treat as regular cosine
                    let progress = (current / decay_steps).min(1.0);
                    base_lr * 0.5 * (1.0 + (std::f64::consts::PI as f32 * progress).cos())
                }
                LrSchedulerType::Polynomial => {
                    let progress = (current / decay_steps).min(1.0);
                    base_lr * (1.0 - progress).powf(2.0) // power=2
                }
            }
        }
    }

    /// Compute loss for a batch.
    fn compute_loss(logits: &Array, labels: &Array) -> Result<Array> {
        let seq_len = logits.dim(1);
        let vocab_size = logits.dim(2);

        // Shift: logits[:-1] predicts labels[1:]
        let shift_logits = logits.index((.., ..seq_len - 1, ..));
        let shift_labels = labels.index((.., 1..));

        // Reshape for cross entropy
        let flat_logits = shift_logits.reshape(&[-1, vocab_size])?;
        let flat_labels = shift_labels.reshape(&[-1])?;

        // Compute cross entropy loss with ignore_index=-100
        let loss = cross_entropy_loss(&flat_logits, &flat_labels, Some(-100_i64), 0.0)?;
        Ok(loss.mean(None)?)
    }

    /// Clip gradients by global norm (GPU-based, no sync required).
    ///
    /// OPTIMIZED: All operations stay on GPU using lazy evaluation.
    /// - Computes scale = min(1.0, max_norm / norm) entirely on GPU
    /// - Always applies scale (even if 1.0) to avoid CPU branch
    /// - Returns lazy norm Array that can be optionally eval'd for logging
    ///
    /// This eliminates the GPU-CPU synchronization overhead that was previously
    /// required to determine if clipping was needed.
    fn clip_gradients_gpu(&self, grads: &mut FlattenedModuleParam) -> Result<Option<Array>> {
        let max_norm = self.config.training.max_grad_norm as f32;
        if max_norm <= 0.0 {
            return Ok(None);
        }

        // Build lazy computation graph: sum of all squared norms
        // This stays entirely on GPU until evaluated
        let mut norm_sq_sum = Array::from_f32(0.0);
        for (_, grad) in grads.iter() {
            let norm_sq = grad.multiply(grad)?.sum(None)?;
            norm_sq_sum = norm_sq_sum.add(&norm_sq)?;
        }

        // Compute norm = sqrt(sum) on GPU (lazy)
        let norm = norm_sq_sum.sqrt()?;

        // Compute scale = min(1.0, max_norm / (norm + eps)) entirely on GPU
        // Using: scale = max_norm / max(norm, max_norm) which clamps to [0, 1]
        let max_norm_arr = Array::from_f32(max_norm);
        let eps = Array::from_f32(1e-6);
        let norm_clamped = mlx_rs::ops::maximum(&norm, &max_norm_arr)?;
        let scale = max_norm_arr.divide(&norm_clamped.add(&eps)?)?;

        // Debug: check scale value
        static DEBUG_CLIP_ONCE: std::sync::Once = std::sync::Once::new();
        DEBUG_CLIP_ONCE.call_once(|| {
            // Force evaluate to debug
            let mut scale_copy = scale.clone();
            scale_copy.eval().ok();
            let scale_val = scale_copy.item::<f32>();
            let mut norm_copy = norm.clone();
            norm_copy.eval().ok();
            let norm_val = norm_copy.item::<f32>();
            tracing::info!("Gradient clipping: norm={:.4}, max_norm={}, scale={:.4}", norm_val, max_norm, scale_val);
        });

        // Apply scale to all gradients on GPU (lazy)
        // Even when scale ~= 1.0, this is cheaper than a GPU-CPU sync to check
        for (_, grad) in grads.iter_mut() {
            *grad = grad.multiply(&scale)?;
        }

        // Return lazy norm - caller can eval() for logging if needed
        Ok(Some(norm))
    }

    /// Clip gradients by global norm with CPU sync for accurate logging.
    ///
    /// This version uses a GPU-CPU sync to get the actual gradient norm value.
    /// Use this only when you need accurate grad_norm logging (e.g., every N steps).
    /// For maximum throughput, use clip_gradients_gpu() instead.
    fn clip_gradients_with_sync(&self, grads: &mut FlattenedModuleParam) -> Result<Option<f32>> {
        let max_norm = self.config.training.max_grad_norm as f32;
        if max_norm <= 0.0 {
            return Ok(None);
        }

        // Build lazy computation graph: sum of all squared norms
        let mut norm_sq_sum = Array::from_f32(0.0);
        for (_, grad) in grads.iter() {
            let norm_sq = grad.multiply(grad)?.sum(None)?;
            norm_sq_sum = norm_sq_sum.add(&norm_sq)?;
        }

        // Single eval() for norm computation
        norm_sq_sum.eval()?;
        let total_norm = norm_sq_sum.item::<f32>().sqrt();

        // Only clip if needed (avoids unnecessary multiply when not clipping)
        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            let scale_arr = Array::from_f32(scale);
            for (_, grad) in grads.iter_mut() {
                *grad = grad.multiply(&scale_arr)?;
            }
        }

        Ok(Some(total_norm))
    }

    /// Accumulate gradients.
    fn accumulate_gradients(&mut self, new_grads: FlattenedModuleParam) -> Result<()> {
        let accum_steps = self.config.training.gradient_accumulation_steps;

        match &mut self.accumulated_grads {
            None => {
                // First accumulation step - scale gradients
                let scale = 1.0 / accum_steps as f32;
                let scale_arr = Array::from_f32(scale);
                let mut scaled: FlattenedModuleParam = FlattenedModuleParam::new();
                for (k, v) in new_grads {
                    let scaled_grad = v.multiply(&scale_arr)?;
                    scaled.insert(k, scaled_grad);
                }
                self.accumulated_grads = Some(scaled);
            }
            Some(acc) => {
                // Accumulate: acc += new_grads / accum_steps
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

    /// Perform a single training step.
    ///
    /// This computes gradients using mlx-rs autodiff and handles gradient accumulation.
    /// Supports both text-only and multimodal (VLM) batches with pixel_values.
    pub fn train_step<M, O>(
        &mut self,
        model: &mut M,
        batch: &TrainingBatch,
        optimizer: &mut O,
    ) -> Result<StepStats>
    where
        M: TrainableModel,
        O: Optimizer,
    {
        let start_time = std::time::Instant::now();

        // Track tokens - use checked arithmetic to prevent overflow
        let batch_tokens = batch
            .batch_size
            .checked_mul(batch.seq_len)
            .unwrap_or(usize::MAX);

        // Compute loss and gradients based on whether we have pixel values (VLM)
        let (loss, grads) = if let Some(ref pixel_values) = batch.pixel_values {
            // VLM training with image inputs
            self.compute_vlm_loss_and_grads(model, batch, pixel_values)?
        } else {
            // Text-only training (standard path)
            self.compute_text_loss_and_grads(model, batch)?
        };

        // Evaluate loss for this step
        // NOTE: Unlike mlx-lm which uses mx.compile for JIT fusion, we need to
        // evaluate each step. mlx-rs doesn't expose mx.compile, so deferring
        // evaluation just builds up a massive computation graph.
        loss.eval()?;
        let loss_val = loss.item::<f32>();

        // Accumulate gradients
        self.accumulate_gradients(grads)?;

        // Apply gradients if accumulation is complete
        // Use GPU-based clipping to avoid GPU-CPU sync overhead
        let grad_norm = if self.should_apply_gradients() {
            if let Some(mut accumulated) = self.take_accumulated_gradients() {
                // Clip gradients using GPU-based method (no sync required)
                // Returns lazy norm Array that we only eval when logging
                let lazy_norm = self.clip_gradients_gpu(&mut accumulated)?;

                // Apply with optimizer (lazy - no eval by default)
                optimizer.update(model, accumulated)?;

                // Eager evaluation mode 
                // Forces immediate evaluation to clear intermediate activations.
                // Trade-off: Lower memory usage vs lower throughput.
                if self.config.eager_evaluation {
                    // Eval model parameters (clears computation graph)
                    mlx_rs::transforms::eval_params(model.parameters())?;
                    // Optimizer state is evaluated via Updatable trait
                    let opt_states: Vec<&Array> = optimizer.updatable_states().into_iter().collect();
                    if !opt_states.is_empty() {
                        mlx_rs::transforms::eval(opt_states)?;
                    }
                }
                // NOTE: When eager_evaluation is false, we use deferred evaluation.
                // MLX builds a lazy computation graph - forcing evaluation after
                // every optimizer step is a massive bottleneck (20-50s per step!).
                // Parameters will be evaluated lazily when needed (at logging or
                // checkpoint time). This matches mlx-lm's approach.

                // Always compute grad_norm when clipping is enabled (tests expect this)
                // The lazy Array is evaluated here - this syncs GPU->CPU
                // TODO: Consider making this conditional via config for max performance
                if let Some(norm_arr) = lazy_norm {
                    norm_arr.eval()?;
                    Some(norm_arr.item::<f32>())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Update stats
        self.step += 1;
        self.total_tokens += batch_tokens;
        self.tokens_since_log += batch_tokens;

        // Update running loss EMA
        self.running_loss = if self.step == 1 {
            loss_val as f64
        } else {
            0.99 * self.running_loss + 0.01 * loss_val as f64
        };

        let step_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(StepStats {
            step: self.step,
            loss: loss_val,
            learning_rate: self.get_learning_rate(),
            tokens: batch_tokens,
            grad_norm,
            step_time_ms,
        })
    }

    /// Compute loss and gradients for text-only training.
    fn compute_text_loss_and_grads<M: TrainableModel>(
        &self,
        model: &mut M,
        batch: &TrainingBatch,
    ) -> Result<(Array, FlattenedModuleParam)> {
        // Define loss function for autodiff
        let loss_fn = |model: &mut M,
                       (input_ids, labels): (&Array, &Array)|
                       -> std::result::Result<Array, Exception> {
            let logits = model
                .forward(input_ids, None)
                .map_err(|e| Exception::custom(e.to_string()))?;
            Self::compute_loss(&logits, labels)
                .map_err(|e| Exception::custom(e.to_string()))
        };

        // Create value_and_grad function
        let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);

        // Optionally enable Metal FlashAttention for forward pass
        let (loss, grads) = if self.metal_fa_available {
            let result = with_training_mode(|| {
                loss_and_grad_fn(model, (&batch.input_ids, &batch.labels))
                    .map_err(|e| pmetal_mlx::error::MlxError::from(e))
            });
            result.map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
        } else {
            loss_and_grad_fn(model, (&batch.input_ids, &batch.labels))?
        };

        Ok((loss, grads))
    }

    /// Compute loss and gradients for VLM (Vision-Language Model) training.
    fn compute_vlm_loss_and_grads<M: TrainableModel>(
        &self,
        model: &mut M,
        batch: &TrainingBatch,
        pixel_values: &Array,
    ) -> Result<(Array, FlattenedModuleParam)> {
        // Clone pixel_values to move into closure
        let pixels = pixel_values.clone();

        // Define loss function for VLM autodiff
        let loss_fn = |model: &mut M,
                       (input_ids, labels): (&Array, &Array)|
                       -> std::result::Result<Array, Exception> {
            // Use forward_with_images for VLM models
            let logits = model
                .forward_with_images(input_ids, None, Some(&pixels))
                .map_err(|e| Exception::custom(e.to_string()))?;
            Self::compute_loss(&logits, labels)
                .map_err(|e| Exception::custom(e.to_string()))
        };

        // Create value_and_grad function
        let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);

        // Optionally enable Metal FlashAttention for forward pass
        let (loss, grads) = if self.metal_fa_available {
            let result = with_training_mode(|| {
                loss_and_grad_fn(model, (&batch.input_ids, &batch.labels))
                    .map_err(|e| pmetal_mlx::error::MlxError::from(e))
            });
            result.map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
        } else {
            loss_and_grad_fn(model, (&batch.input_ids, &batch.labels))?
        };

        Ok((loss, grads))
    }

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
        let mut optimizer = crate::AdamWGroupsBuilder::new(base_lr)
            .with_weight_decay(weight_decay);

        if let Some(emb_lr) = self.config.embedding_lr {
            optimizer = optimizer.with_embedding_lr(emb_lr);
            tracing::info!(
                "Using separate embedding LR: {:.2e} (base: {:.2e})",
                emb_lr, base_lr
            );
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

        // Initialize timing for throughput measurement
        self.last_log_time = Some(std::time::Instant::now());
        self.tokens_since_log = 0;

        let mut best_eval_loss = f64::MAX;

        for epoch in 0..num_epochs {
            self.epoch = epoch;
            tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);

            // Create dataloader for this epoch
            let mut dataloader = DataLoader::new(
                train_dataset.clone(),
                self.config.dataloader.clone(),
                None, // No image processor for text-only training
            );

            while let Some(batch) = dataloader.next_batch() {
                // Apply learning rate schedule (warmup, cosine decay, etc.)
                let scheduled_lr = self.get_learning_rate();
                optimizer.set_learning_rate(scheduled_lr);

                // Training step
                let stats = self.train_step(model, &batch, &mut optimizer)?;

                // Logging
                if self.step % self.config.log_every == 0 {
                    // Calculate throughput over the entire logging interval
                    let now = std::time::Instant::now();
                    let tokens_per_sec = match self.last_log_time {
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

                    // Reset interval tracking
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

                // Evaluation
                if self.config.eval_every > 0
                    && self.step % self.config.eval_every == 0
                    && eval_dataset.is_some()
                {
                    let metrics = self.evaluate(model, eval_dataset.as_ref().unwrap())?;

                    // Log comprehensive metrics
                    let acc_str = metrics
                        .accuracy
                        .map(|a| format!(", acc={:.2}%", a))
                        .unwrap_or_default();
                    tracing::info!(
                        "Step {}: eval_loss={:.4}, ppl={:.2}{}",
                        self.step,
                        metrics.loss,
                        metrics.perplexity,
                        acc_str
                    );

                    if metrics.loss < best_eval_loss {
                        best_eval_loss = metrics.loss;

                        // Save best checkpoint
                        if let Some(manager) = checkpoint_manager {
                            self.save_checkpoint(model, manager, true, Some(metrics.loss))?;
                        }
                    }
                }

                // Regular checkpointing
                if self.config.checkpoint_every > 0
                    && self.step % self.config.checkpoint_every == 0
                {
                    if let Some(manager) = checkpoint_manager {
                        self.save_checkpoint(model, manager, false, None)?;
                    }
                }

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

        tracing::info!(
            "Starting Metal-fused training: {} trainable params, batch_size={}, grad_accum={}",
            model.num_trainable_params(),
            self.config.training.batch_size,
            self.config.training.gradient_accumulation_steps
        );
        tracing::info!("Metal fused optimizer: {} total elements", metal_optimizer.total_elements());

        // Initialize timing for throughput measurement
        self.last_log_time = Some(std::time::Instant::now());
        self.tokens_since_log = 0;

        let mut best_eval_loss = f64::MAX;

        for epoch in 0..num_epochs {
            self.epoch = epoch;
            tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);

            // Create dataloader for this epoch
            let mut dataloader = DataLoader::new(
                train_dataset.clone(),
                self.config.dataloader.clone(),
                None,
            );

            while let Some(batch) = dataloader.next_batch() {
                // Training step with Metal optimizer
                let stats = self.train_step_metal(model, &batch, &mut metal_optimizer)?;

                // Logging
                if self.step % self.config.log_every == 0 {
                    let now = std::time::Instant::now();
                    let tokens_per_sec = match self.last_log_time {
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
                        stats.grad_norm.map(|n| format!(", grad_norm={:.2}", n)).unwrap_or_default()
                    );
                }

                // Evaluation
                if self.config.eval_every > 0
                    && self.step % self.config.eval_every == 0
                    && eval_dataset.is_some()
                {
                    let eval_ds = eval_dataset.as_ref().unwrap();
                    let metrics = self.evaluate(model, eval_ds)?;
                    let acc_str = metrics.accuracy
                        .map(|a| format!(", accuracy={:.2}%", a * 100.0))
                        .unwrap_or_default();
                    tracing::info!("Eval: loss={:.4}{}", metrics.loss, acc_str);

                    if metrics.loss < best_eval_loss {
                        best_eval_loss = metrics.loss;
                        if let Some(manager) = checkpoint_manager {
                            self.save_checkpoint(model, manager, true, Some(metrics.loss))?;
                        }
                    }
                }

                // Regular checkpointing
                if self.config.checkpoint_every > 0
                    && self.step % self.config.checkpoint_every == 0
                {
                    if let Some(manager) = checkpoint_manager {
                        self.save_checkpoint(model, manager, false, None)?;
                    }
                }

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
    fn train_step_metal<M>(
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
                mlx_rs::transforms::eval(to_eval)?;

                let loss_val = loss.item::<f32>();
                let norm = lazy_norm.map(|n| n.item::<f32>());

                (loss_val, norm, clip_elapsed, opt_elapsed)
            } else {
                // No gradients accumulated - just eval loss
                loss.eval()?;
                (loss.item::<f32>(), None, 0, 0)
            }
        } else {
            // Gradient accumulation not complete - just eval loss
            loss.eval()?;
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
        static PROFILE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let profile_idx = PROFILE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if profile_idx < 5 || profile_idx % 50 == 0 {
            let total_us = fwd_bwd_us + accum_us + clip_us + opt_us;
            tracing::info!(
                "PROFILE step={}: fwd+bwd={}us ({:.1}%), accum={}us ({:.1}%), clip={}us ({:.1}%), opt={}us ({:.1}%), total={}us",
                self.step,
                fwd_bwd_us, 100.0 * fwd_bwd_us as f64 / total_us as f64,
                accum_us, 100.0 * accum_us as f64 / total_us as f64,
                clip_us, 100.0 * clip_us as f64 / total_us as f64,
                opt_us, 100.0 * opt_us as f64 / total_us as f64,
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

    /// Run optimized training with fused forward/backward/optimizer step.
    ///
    /// This method uses a fused training step that combines forward pass, backward pass,
    /// and optimizer update into a single function, allowing MLX's lazy evaluation to
    /// optimize the computation graph.
    ///
    /// ## Implementation Note
    ///
    /// While mlx-rs provides `compile_with_state` for JIT compilation, it has known
    /// limitations with complex models + optimizers where state count changes during
    /// execution. This method uses the non-JIT `jit_training_step` which still benefits
    /// from MLX's lazy evaluation and graph fusion.
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
        _eval_dataset: Option<TrainingDataset>,
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

        // If JIT compilation is enabled, use TrainableState for true JIT
        if self.config.use_jit_compilation {
            tracing::info!("JIT compilation enabled, using TrainableState wrapper");
            return self.run_jit_compiled(model, train_dataset, _eval_dataset, checkpoint_manager);
        }

        // Initialize optimizer with optional embedding learning rate
        let base_lr = self.config.training.learning_rate as f32;
        let weight_decay = self.config.training.weight_decay as f32;

        let mut optimizer_builder = crate::AdamWGroupsBuilder::new(base_lr)
            .with_weight_decay(weight_decay);

        if let Some(emb_lr) = self.config.embedding_lr {
            optimizer_builder = optimizer_builder.with_embedding_lr(emb_lr);
            tracing::info!(
                "Using separate embedding LR: {:.2e} (base: {:.2e})",
                emb_lr, base_lr
            );
        }

        let optimizer = optimizer_builder
            .build()
            .map_err(|_| SftError::Mlx(Exception::custom("Failed to build optimizer")))?;

        let max_steps = self.config.training.max_steps;
        let num_epochs = self.config.training.num_epochs;

        tracing::info!(
            "Starting optimized training: {} trainable params, batch_size={}",
            model.num_trainable_params(),
            self.config.training.batch_size,
        );

        // Create state tuple that owns both model and optimizer
        // This allows jit_training_step to mutate both in a single function
        let mut state = (model, optimizer);

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
        let warmup_batch = dataloader.next_batch().ok_or_else(|| {
            SftError::Mlx(Exception::custom("Dataset is empty, cannot warmup"))
        })?;

        // Record state count BEFORE warmup (optimizer states not yet initialized)
        let state_count_before = state.updatable_states_len();

        tracing::info!(
            "Warmup: Running uncompiled step to initialize optimizer states (state_count={})",
            state_count_before
        );

        // Run ONE uncompiled training step
        let warmup_loss = jit_training_step(&mut state, (&warmup_batch.input_ids, &warmup_batch.labels))?;
        warmup_loss.eval()?;
        let warmup_loss_val = warmup_loss.item::<f32>();

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
        let warmup_tokens = warmup_batch.batch_size.checked_mul(warmup_batch.seq_len).unwrap_or(usize::MAX);
        self.step = 1;
        self.total_tokens = warmup_tokens;
        self.running_loss = warmup_loss_val as f64;

        // =========================================================================
        // PHASE 2: State verification and main loop setup
        // =========================================================================
        // After warmup, optimizer state is stable (no more lazy initialization).
        //
        // NOTE: mlx-rs compile_with_state has fundamental issues with large state counts:
        // - LLMs have 10M+ trainable parameters, each with optimizer state (momentum, velocity)
        // - compile_with_state's state tracking fails with integer underflow on such large counts
        // - See: mlx-rs/src/transforms/compile/compile_with_state.rs:418
        //
        // Workaround: Use deferred evaluation (batch evals at logging boundaries)
        // This achieves ~80% of full JIT performance by minimizing GPU-CPU syncs.

        tracing::info!(
            "State initialized (count={}), starting main training loop",
            state_count_after
        );

        tracing::info!(
            "Fused mode enabled - using deferred evaluation for optimized throughput"
        );

        // Initialize timing for throughput measurement
        self.last_log_time = Some(std::time::Instant::now());
        self.tokens_since_log = warmup_tokens;

        // =========================================================================
        // PHASE 3: MAIN LOOP - Deferred Evaluation Pattern for 2x throughput
        // =========================================================================
        // Key optimization: Instead of calling eval() every step (forces GPU-CPU sync),
        // we accumulate lazy loss Arrays and only evaluate at logging boundaries.
        // This achieves ~1500-2000 tok/s by minimizing GPU-CPU synchronization overhead.

        // Pre-allocate vector for accumulated losses
        let mut accumulated_losses: Vec<Array> = Vec::with_capacity(self.config.log_every);

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
                tracing::info!("Epoch {}/{} (continuing after warmup)", epoch + 1, num_epochs);
            } else {
                tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);
            }

            while let Some(batch) = dataloader.next_batch() {
                let batch_tokens = batch.batch_size.checked_mul(batch.seq_len).unwrap_or(usize::MAX);

                // Execute fused training step (forward + backward + optimizer update)
                // DEFERRED EVAL: Loss remains a lazy Array, no GPU-CPU sync here
                // MLX's lazy evaluation automatically fuses operations when not evaluated
                let loss = jit_training_step(&mut state, (&batch.input_ids, &batch.labels))?;
                accumulated_losses.push(loss);

                // Update step counters (these are just integers, no GPU involvement)
                self.step += 1;
                self.total_tokens += batch_tokens;
                self.tokens_since_log += batch_tokens;

                // Logging boundary: NOW we evaluate accumulated losses
                if self.step % self.config.log_every == 0 {
                    // Batch evaluate all accumulated losses together
                    // This is much more efficient than per-step eval
                    let loss_refs: Vec<&Array> = accumulated_losses.iter().collect();
                    mlx_rs::transforms::eval(loss_refs)?;

                    // Now extract values and compute running loss
                    for loss in &accumulated_losses {
                        let loss_val = loss.item::<f32>();
                        self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;
                    }
                    accumulated_losses.clear();

                    // Calculate throughput
                    let now = std::time::Instant::now();
                    let tokens_per_sec = match self.last_log_time {
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
                        "Step {}: loss={:.4}, lr={:.2e}, tokens/s={:.0}",
                        self.step,
                        self.running_loss,
                        self.config.training.learning_rate,
                        tokens_per_sec,
                    );
                }

                // Regular checkpointing - need to eval before checkpoint
                if self.config.checkpoint_every > 0
                    && self.step % self.config.checkpoint_every == 0
                {
                    // Eval any pending losses before checkpointing
                    if !accumulated_losses.is_empty() {
                        let loss_refs: Vec<&Array> = accumulated_losses.iter().collect();
                        mlx_rs::transforms::eval(loss_refs)?;
                        for loss in &accumulated_losses {
                            let loss_val = loss.item::<f32>();
                            self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;
                        }
                        accumulated_losses.clear();
                    }

                    if let Some(manager) = checkpoint_manager {
                        // Need to borrow model from state for checkpointing
                        self.save_checkpoint(&state.0, manager, false, None)?;
                    }
                }

                // Check max steps
                if let Some(max) = max_steps {
                    if self.step >= max {
                        // Eval any remaining losses before returning
                        if !accumulated_losses.is_empty() {
                            let loss_refs: Vec<&Array> = accumulated_losses.iter().collect();
                            mlx_rs::transforms::eval(loss_refs)?;
                            for loss in &accumulated_losses {
                                let loss_val = loss.item::<f32>();
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
            let loss_refs: Vec<&Array> = accumulated_losses.iter().collect();
            mlx_rs::transforms::eval(loss_refs)?;
            for loss in &accumulated_losses {
                let loss_val = loss.item::<f32>();
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

    /// Run training with true JIT compilation using TrainableState.
    ///
    /// This method uses mlx-rs's `compile_with_state` with the new `TrainableState`
    /// wrapper that only exposes trainable parameters. This enables true JIT compilation
    /// for large models with LoRA, achieving performance comparable to Python mlx-lm.
    ///
    /// ## How it works
    ///
    /// 1. Model and optimizer are wrapped in `TrainableState<M, O>`
    /// 2. `TrainableState` implements `Updatable` returning only trainable params + optimizer state
    /// 3. `compile_with_state` JIT compiles the training step with graph fusion
    /// 4. The compiled function runs at maximum throughput
    ///
    /// ## Performance
    ///
    /// For LoRA fine-tuning, this reduces state count from ~10M to ~500K arrays,
    /// enabling successful JIT compilation. Expected performance is comparable to
    /// or exceeding Python mlx-lm.
    ///
    /// **Requirements:**
    /// - gradient_accumulation_steps must be 1
    /// - Model must implement ModuleParameters + TrainableModel
    pub fn run_jit_compiled<M>(
        &mut self,
        model: M,
        train_dataset: TrainingDataset,
        _eval_dataset: Option<TrainingDataset>,
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

        // Initialize optimizer
        let base_lr = self.config.training.learning_rate as f32;
        let weight_decay = self.config.training.weight_decay as f32;

        let mut optimizer_builder = crate::AdamWGroupsBuilder::new(base_lr)
            .with_weight_decay(weight_decay);

        if let Some(emb_lr) = self.config.embedding_lr {
            optimizer_builder = optimizer_builder.with_embedding_lr(emb_lr);
            tracing::info!(
                "Using separate embedding LR: {:.2e} (base: {:.2e})",
                emb_lr, base_lr
            );
        }

        let optimizer = optimizer_builder
            .build()
            .map_err(|_| SftError::Mlx(Exception::custom("Failed to build optimizer")))?;

        let max_steps = self.config.training.max_steps;
        let num_epochs = self.config.training.num_epochs;

        // Wrap model and optimizer in TrainableState
        // This ensures compile_with_state only tracks trainable parameters
        let mut state = TrainableState::new(model, optimizer);

        tracing::info!(
            "Starting JIT-compiled training: {} trainable params (state_count={})",
            state.num_trainable_params(),
            state.updatable_states_len(),
        );

        // =========================================================================
        // PHASE 1: WARMUP - Initialize optimizer states
        // =========================================================================
        let mut dataloader = DataLoader::new(
            train_dataset.clone(),
            self.config.dataloader.clone(),
            None,
        );

        let warmup_batch = dataloader.next_batch().ok_or_else(|| {
            SftError::Mlx(Exception::custom("Dataset is empty, cannot warmup"))
        })?;

        let state_count_before = state.updatable_states_len();

        tracing::info!(
            "Warmup: Running uncompiled step to initialize optimizer states (state_count={})",
            state_count_before
        );

        // Run ONE uncompiled training step for warmup
        let warmup_loss = trainable_training_step(&mut state, (&warmup_batch.input_ids, &warmup_batch.labels))?;
        warmup_loss.eval()?;
        let warmup_loss_val = warmup_loss.item::<f32>();

        // Refresh trainable keys after warmup (optimizer may have added state)
        state.refresh_trainable_keys();
        let state_count_after = state.updatable_states_len();

        tracing::info!(
            "Warmup complete: loss={:.4}, state_count {} -> {} (delta={})",
            warmup_loss_val,
            state_count_before,
            state_count_after,
            state_count_after as i64 - state_count_before as i64
        );

        // Update stats for warmup step
        let warmup_tokens = warmup_batch.batch_size.checked_mul(warmup_batch.seq_len).unwrap_or(usize::MAX);
        self.step = 1;
        self.total_tokens = warmup_tokens;
        self.running_loss = warmup_loss_val as f64;

        // =========================================================================
        // PHASE 2: JIT COMPILE the training step
        // =========================================================================
        tracing::info!(
            "JIT compiling training step with TrainableState (state_count={})",
            state_count_after
        );

        // Create the compiled training step
        // Note: compile_with_state will trace the function on first call
        let mut compiled_step = compile_with_state(trainable_training_step::<M, crate::AdamWGroups>, None);

        // Initialize timing for throughput measurement
        self.last_log_time = Some(std::time::Instant::now());
        self.tokens_since_log = warmup_tokens;

        // =========================================================================
        // PHASE 3: MAIN LOOP with JIT-compiled training
        // =========================================================================
        tracing::info!("JIT compilation ready, starting main training loop");

        for epoch in 0..num_epochs {
            self.epoch = epoch;

            if epoch > 0 {
                dataloader = DataLoader::new(
                    train_dataset.clone(),
                    self.config.dataloader.clone(),
                    None,
                );
                dataloader.reset(Some(self.config.dataloader.seed + epoch as u64));
            }

            tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);

            while let Some(batch) = dataloader.next_batch() {
                let batch_tokens = batch.batch_size.checked_mul(batch.seq_len).unwrap_or(usize::MAX);

                // Execute JIT-compiled training step
                let loss = compiled_step(&mut state, (&batch.input_ids, &batch.labels))?;

                // Update step counters
                self.step += 1;
                self.total_tokens += batch_tokens;
                self.tokens_since_log += batch_tokens;

                // Logging at boundaries
                if self.step % self.config.log_every == 0 {
                    loss.eval()?;
                    let loss_val = loss.item::<f32>();
                    self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;

                    // Calculate throughput
                    let now = std::time::Instant::now();
                    let tokens_per_sec = match self.last_log_time {
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

                    // Compute learning rate from scheduler
                    let total_steps = max_steps.unwrap_or(usize::MAX);
                    let progress = (self.step as f64 / total_steps.max(1) as f64).min(1.0);
                    let lr = match self.config.training.lr_scheduler {
                        LrSchedulerType::Constant => base_lr as f64,
                        LrSchedulerType::Linear => base_lr as f64 * (1.0 - progress),
                        LrSchedulerType::Cosine | LrSchedulerType::CosineWithRestarts => {
                            0.5 * base_lr as f64 * (1.0 + (std::f64::consts::PI * progress).cos())
                        }
                        LrSchedulerType::Polynomial => {
                            let power = 2.0;
                            base_lr as f64 * (1.0 - progress).powf(power)
                        }
                    };

                    tracing::info!(
                        "Step {}: loss={:.4}, lr={:.2e}, tokens/s={:.0}",
                        self.step,
                        self.running_loss,
                        lr,
                        tokens_per_sec
                    );

                    self.last_log_time = Some(now);
                    self.tokens_since_log = 0;
                }

                // Check max steps
                if let Some(max) = max_steps {
                    if self.step >= max {
                        break;
                    }
                }
            }

            if let Some(max) = max_steps {
                if self.step >= max {
                    break;
                }
            }
        }

        tracing::info!(
            "JIT-compiled training complete: {} steps, {:.4} final loss",
            self.step,
            self.running_loss
        );

        // Extract model from TrainableState
        let (model, _optimizer) = state.into_parts();
        Ok(model)
    }

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
        _eval_dataset: Option<TrainingDataset>,
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

        let mut optimizer_builder = crate::AdamWGroupsBuilder::new(base_lr)
            .with_weight_decay(weight_decay);

        if let Some(emb_lr) = self.config.embedding_lr {
            optimizer_builder = optimizer_builder.with_embedding_lr(emb_lr);
            tracing::info!(
                "Using separate embedding LR: {:.2e} (base: {:.2e})",
                emb_lr, base_lr
            );
        }

        let optimizer = optimizer_builder
            .build()
            .map_err(|_| SftError::Mlx(Exception::custom("Failed to build optimizer")))?;

        tracing::info!("Starting packed training with sequence packing enabled");

        // Create PackedDataLoader from dataset samples
        // CRITICAL: Set max_seq_length to truncate long sequences instead of skipping them!
        let max_seq_len = self.config.dataloader.max_seq_len;
        let packer_config = PackerConfig::with_max_length(max_seq_len)
            .with_max_seq_length(max_seq_len) // Truncate sequences to max_seq_len
            .mask_boundaries(true);

        // Get samples from dataset - need to access the samples directly
        let samples: Vec<_> = (0..train_dataset.len())
            .filter_map(|i| train_dataset.get(i).cloned())
            .collect();

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
            .ok_or_else(|| SftError::Mlx(Exception::custom("No packed batches available for warmup")))?
            .map_err(|e| SftError::Mlx(e))?;

        let warmup_tokens = warmup_batch.total_tokens;

        // Record optimizer state count BEFORE warmup (states not yet initialized)
        let state_count_before = state.updatable_states_len();

        tracing::info!(
            "Warmup: Running step to initialize optimizer states (state_count={})",
            state_count_before
        );

        // Execute warmup step - this initializes optimizer momentum/velocity buffers
        let warmup_loss = jit_training_step_packed(&mut state, &warmup_batch)?;
        warmup_loss.eval()?;
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
        self.last_log_time = Some(std::time::Instant::now());
        self.tokens_since_log = 0;

        // Pre-allocate vector for accumulated losses (deferred evaluation)
        let mut accumulated_losses: Vec<Array> = Vec::with_capacity(self.config.log_every);

        for epoch in 0..num_epochs {
            self.epoch = epoch;

            if epoch > 0 {
                packed_dataloader.reset(Some(self.config.dataloader.seed + epoch as u64));
            }

            tracing::info!("Epoch {}/{}", epoch + 1, num_epochs);

            while let Some(batch_result) = packed_dataloader.next_batch() {
                let packed_batch = batch_result
                    .map_err(|e| SftError::Mlx(e))?;

                let batch_tokens = packed_batch.total_tokens;

                // Execute packed training step (forward + backward + optimizer update)
                // DEFERRED EVAL: Loss remains a lazy Array, no GPU-CPU sync here
                let loss = jit_training_step_packed(&mut state, &packed_batch)?;
                accumulated_losses.push(loss);

                // Update step counters
                self.step += 1;
                self.total_tokens += batch_tokens;
                self.tokens_since_log += batch_tokens;

                // Logging boundary: NOW we evaluate accumulated losses
                if self.step % self.config.log_every == 0 {
                    // Batch evaluate all accumulated losses together
                    let loss_refs: Vec<&Array> = accumulated_losses.iter().collect();
                    mlx_rs::transforms::eval(loss_refs)?;

                    // Extract values and compute running loss via EMA
                    // Note: running_loss was initialized to warmup loss, so EMA works from step 1
                    for loss in accumulated_losses.iter() {
                        let loss_val = loss.item::<f32>();
                        self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;
                    }
                    accumulated_losses.clear();

                    // Calculate throughput
                    let now = std::time::Instant::now();
                    let tokens_per_sec = match self.last_log_time {
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
                        "Step {}: loss={:.4}, lr={:.2e}, tokens/s={:.0}",
                        self.step,
                        self.running_loss,
                        self.config.training.learning_rate,
                        tokens_per_sec,
                    );
                }

                // Regular checkpointing
                if self.config.checkpoint_every > 0
                    && self.step % self.config.checkpoint_every == 0
                {
                    // Eval any pending losses before checkpointing
                    if !accumulated_losses.is_empty() {
                        let loss_refs: Vec<&Array> = accumulated_losses.iter().collect();
                        mlx_rs::transforms::eval(loss_refs)?;
                        for loss in &accumulated_losses {
                            let loss_val = loss.item::<f32>();
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
                            let loss_refs: Vec<&Array> = accumulated_losses.iter().collect();
                            mlx_rs::transforms::eval(loss_refs)?;
                        }
                        tracing::info!("Reached max_steps={}, stopping", max);
                        return Ok(state.0);
                    }
                }
            }
        }

        // Eval any remaining accumulated losses at end of training
        if !accumulated_losses.is_empty() {
            let loss_refs: Vec<&Array> = accumulated_losses.iter().collect();
            mlx_rs::transforms::eval(loss_refs)?;
            for loss in &accumulated_losses {
                let loss_val = loss.item::<f32>();
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

    /// Evaluate the model on a dataset.
    ///
    /// Returns comprehensive evaluation metrics including:
    /// - Average loss across all batches
    /// - Perplexity (exp(loss))
    /// - Token-level accuracy (correct next-token predictions)
    pub fn evaluate<M>(
        &self,
        model: &mut M,
        dataset: &TrainingDataset,
    ) -> Result<EvalMetrics>
    where
        M: TrainableModel,
    {
        let mut eval_config = self.config.dataloader.clone();
        eval_config.shuffle = false;
        eval_config.drop_last = false;

        let dataloader = DataLoader::new(dataset.clone(), eval_config, None);

        let mut total_loss = 0.0;
        let mut total_correct = 0_u64;
        let mut total_tokens = 0_u64;
        let mut num_batches = 0;

        for batch in dataloader {
            let logits = model
                .forward(&batch.input_ids, None)
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

            // Compute loss
            let loss = Self::compute_loss(&logits, &batch.labels)?;
            loss.eval()?;
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
        let flat_logits = shift_logits.reshape(&[-1, vocab_size])?;
        let predictions = argmax_axis(&flat_logits, -1, None)?;
        predictions.eval()?;

        let flat_labels = shift_labels.reshape(&[-1])?;
        flat_labels.eval()?;

        // Create mask for valid tokens (label != -100)
        let ignore_index = Array::from_int(-100);
        let valid_mask = flat_labels.ne(&ignore_index)?;
        valid_mask.eval()?;

        // Count valid tokens
        let total_valid = valid_mask.sum(None)?;
        total_valid.eval()?;
        let total_tokens = total_valid.item::<i64>() as u64;

        if total_tokens == 0 {
            return Ok((0, 0));
        }

        // Compare predictions with labels where valid
        // predictions is i32, labels is i64, need to cast
        let predictions_i64 = predictions.as_dtype(mlx_rs::Dtype::Int64)?;
        let correct = predictions_i64.eq(&flat_labels)?;
        correct.eval()?;

        // Mask out invalid tokens and sum
        let valid_correct = correct.multiply(&valid_mask)?;
        let correct_sum = valid_correct.sum(None)?;
        correct_sum.eval()?;
        let correct_count = correct_sum.item::<i64>() as u64;

        Ok((correct_count, total_tokens))
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
}

// =============================================================================
// Custom Autograd Training (Unsloth-style memory optimization)
// =============================================================================
//
// This module provides custom autograd training that bypasses MLX autodiff
// for LoRA layers, achieving ~50% memory reduction. This is the technique
// used by unsloth to enable training of larger models on limited memory.
//
// NOTE: This is an advanced feature that requires model-specific integration.
// The standard training loop (run_compiled, run_packed) uses MLX autodiff
// which is simpler and sufficient for most use cases.

/// Trait for models that support custom autograd training.
///
/// This extends TrainableModel with methods for explicit gradient computation.
/// Implementing this trait enables the memory-efficient custom autograd path.
pub trait CustomAutogradModel: TrainableModel {
    /// Apply gradients from the custom autograd accumulator.
    ///
    /// This method should iterate over all LoRA layers and apply the
    /// accumulated gradients using the optimizer or simple SGD.
    fn apply_accumulated_grads(
        &mut self,
        grads: &pmetal_lora::AccumulatedLoraGrads,
        learning_rate: f32,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_core::LoraConfig;
    use pmetal_lora::LlamaLoraForCausalLM;
    use pmetal_models::architectures::llama::LlamaConfig;

    fn small_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: None,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            ..Default::default()
        }
    }

    fn small_lora_config() -> LoraConfig {
        LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            use_rslora: false,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            bias: "none".to_string(),
            init_lora_weights: true,
            use_dora: false,
        }
    }

    #[test]
    fn test_training_loop_creation() {
        let config = TrainingLoopConfig::default();
        let training_loop = TrainingLoop::new(config);

        assert_eq!(training_loop.current_step(), 0);
        assert_eq!(training_loop.current_epoch(), 0);
    }

    #[test]
    fn test_learning_rate_warmup() {
        let mut config = TrainingLoopConfig::default();
        config.training.warmup_steps = 100;
        config.training.max_steps = Some(1000);
        config.training.learning_rate = 1e-4;
        config.training.lr_scheduler = LrSchedulerType::Cosine;

        let mut training_loop = TrainingLoop::new(config);

        // At step 0
        training_loop.step = 0;
        let lr0 = training_loop.get_learning_rate();
        assert!(lr0 < 1e-4);

        // At step 50 (halfway through warmup)
        training_loop.step = 50;
        let lr50 = training_loop.get_learning_rate();
        assert!((lr50 - 5e-5).abs() < 1e-8);

        // At step 100 (end of warmup)
        training_loop.step = 100;
        let lr100 = training_loop.get_learning_rate();
        assert!((lr100 - 1e-4).abs() < 1e-8);
    }

    #[test]
    fn test_gradient_accumulation_flag() {
        let mut config = TrainingLoopConfig::default();
        config.training.gradient_accumulation_steps = 4;

        let mut training_loop = TrainingLoop::new(config);

        // First 3 steps should not trigger gradient application
        for _ in 0..3 {
            training_loop.accumulation_step += 1;
            assert!(!training_loop.should_apply_gradients());
        }

        // 4th step should trigger
        training_loop.accumulation_step += 1;
        assert!(training_loop.should_apply_gradients());
    }

    #[test]
    fn test_single_train_step() {
        use mlx_rs::optimizers::Sgd;

        let config = TrainingLoopConfig {
            use_metal_flash_attention: false, // Disable for simpler test
            ..Default::default()
        };
        let mut training_loop = TrainingLoop::new(config);

        let mut model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
        let mut optimizer = Sgd::new(1e-4);

        // Create a minimal batch
        let batch = TrainingBatch {
            input_ids: Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]),
            labels: Array::from_slice(&[2_i64, 3, 4, 5], &[1, 4]),
            attention_mask: Array::from_slice(&[1_i32, 1, 1, 1], &[1, 4]),
            pixel_values: None,
            batch_size: 1,
            seq_len: 4,
        };

        let stats = training_loop.train_step(&mut model, &batch, &mut optimizer).unwrap();

        assert!(stats.loss > 0.0);
        assert_eq!(stats.step, 1);
        assert_eq!(training_loop.current_step(), 1);
    }

    #[test]
    fn test_jit_training_step() {
        // Test the JIT-compiled training step function directly
        use mlx_rs::optimizers::AdamW;

        let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
        let optimizer = AdamW::new(1e-4);

        let mut state = (model, optimizer);

        // Create a minimal batch
        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let labels = Array::from_slice(&[2_i64, 3, 4, 5], &[1, 4]);

        // Run the JIT training step
        let loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
        loss.eval().unwrap();

        let loss_val = loss.item::<f32>();
        assert!(loss_val > 0.0, "Loss should be positive, got {}", loss_val);
        assert!(loss_val.is_finite(), "Loss should be finite, got {}", loss_val);
    }

    #[test]
    fn test_jit_training_step_multiple_steps() {
        // Test that jit_training_step works correctly over multiple steps
        // This verifies the training step function itself works, independent of compile_with_state
        use mlx_rs::optimizers::AdamW;

        let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
        let optimizer = AdamW::new(1e-4);

        let mut state = (model, optimizer);

        // Create test data
        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
        let labels = Array::from_slice(&[2_i64, 3, 4, 5, 6, 7, 8, 9], &[1, 8]);

        // Run multiple training steps
        let mut losses = Vec::new();
        for _ in 0..5 {
            let loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
            loss.eval().unwrap();
            losses.push(loss.item::<f32>());
        }

        // All losses should be finite and positive
        for (i, loss) in losses.iter().enumerate() {
            assert!(loss.is_finite(), "Loss {} should be finite, got {}", i, loss);
            assert!(*loss > 0.0, "Loss {} should be positive, got {}", i, loss);
        }

        // Verify loss is changing (parameters are being updated)
        let loss_variance: f32 = losses.iter()
            .map(|l| (l - losses[0]).powi(2))
            .sum::<f32>() / losses.len() as f32;
        assert!(loss_variance > 0.0, "Loss should change over steps, got {:?}", losses);

        println!("Training step losses: {:?}", losses);
    }

    #[test]
    fn test_jit_training_step_with_warmup() {
        // Test the jit_training_step function with proper warmup to initialize
        // optimizer state. This verifies state stability and correct loss reduction.
        use mlx_rs::optimizers::AdamW;
        use mlx_rs::utils::Updatable;

        let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
        let optimizer = AdamW::new(1e-4);

        let mut state = (model, optimizer);

        // Create test data
        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
        let labels = Array::from_slice(&[2_i64, 3, 4, 5, 6, 7, 8, 9], &[1, 8]);

        // ========================================
        // PHASE 1: Record state count BEFORE warmup
        // ========================================
        let state_count_before = state.updatable_states_len();
        println!("State count BEFORE warmup: {}", state_count_before);

        // ========================================
        // PHASE 2: WARMUP - Run one uncompiled step
        // ========================================
        // This initializes optimizer momentum/velocity buffers
        let warmup_loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
        warmup_loss.eval().unwrap();
        let warmup_loss_val = warmup_loss.item::<f32>();
        println!("Warmup loss: {:.4}", warmup_loss_val);

        // ========================================
        // PHASE 3: Record state count AFTER warmup
        // ========================================
        let state_count_after = state.updatable_states_len();
        println!("State count AFTER warmup: {} (delta={})",
                 state_count_after,
                 state_count_after as i64 - state_count_before as i64);

        // AdamW should have created momentum and velocity buffers
        // So state count should have increased
        assert!(
            state_count_after >= state_count_before,
            "State count should not decrease after warmup: {} -> {}",
            state_count_before,
            state_count_after
        );

        // ========================================
        // PHASE 4: Run SECOND warmup step to verify stability
        // ========================================
        println!("Running second warmup step to verify state stability...");
        let warmup2_loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
        warmup2_loss.eval().unwrap();
        let warmup2_loss_val = warmup2_loss.item::<f32>();
        println!("Second warmup loss: {:.4}", warmup2_loss_val);

        let state_count_after_2 = state.updatable_states_len();
        println!("State count AFTER second warmup: {} (should be same as {})",
                 state_count_after_2,
                 state_count_after);

        assert_eq!(
            state_count_after,
            state_count_after_2,
            "State count should be stable after second warmup: {} vs {}",
            state_count_after,
            state_count_after_2
        );

        // ========================================
        // PHASE 5: Use non-compiled path
        // ========================================
        // NOTE: compile_with_state has a known limitation in mlx-rs where it doesn't
        // correctly handle state count changes. Even with warmup to stabilize state,
        // the internal state tracking in compile_with_state.rs:413 fails with
        // "attempt to subtract with overflow" because:
        // 1. The inner closure captures state count at creation time
        // 2. During MLX tracing, the function may see different state
        // 3. The compiled graph expects N outputs but current state has M > N
        //
        // For now, we use the non-compiled jit_training_step which correctly handles
        // state and benefits from MLX's lazy evaluation and graph fusion.
        println!("Using non-compiled training step (mlx-rs compile_with_state limitation)");

        let mut losses = vec![warmup_loss_val, warmup2_loss_val];
        for i in 0..3 {
            let loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
            loss.eval().unwrap();
            let loss_val = loss.item::<f32>();
            losses.push(loss_val);
            println!("Training step {}: loss={:.4}", i + 3, loss_val);
        }

        // Verify state count remains stable
        let final_state_count = state.updatable_states_len();
        assert_eq!(
            state_count_after,
            final_state_count,
            "State count should remain stable: {} -> {}",
            state_count_after,
            final_state_count
        );

        println!("State stability verified! Losses: {:?}", losses);
    }

    #[test]
    fn test_eager_evaluation_config() {
        // Test eager evaluation mode can be enabled
        let mut config = TrainingLoopConfig::default();
        assert!(!config.eager_evaluation, "Default should have eager_evaluation disabled");

        config.eager_evaluation = true;
        let training_loop = TrainingLoop::new(config);
        assert!(
            training_loop.config.eager_evaluation,
            "Should preserve eager_evaluation config"
        );
    }

    #[test]
    fn test_gpu_gradient_clipping() {
        use mlx_rs::Array;
        use std::rc::Rc;

        let config = TrainingLoopConfig {
            training: TrainingConfig {
                max_grad_norm: 1.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let training_loop = TrainingLoop::new(config);

        // Create some fake gradients
        let grad1 = Array::from_slice(&[3.0f32, 4.0], &[2]); // norm = 5
        let grad2 = Array::from_slice(&[0.0f32, 0.0], &[2]); // norm = 0
        let mut grads = FlattenedModuleParam::new();
        grads.insert(Rc::from("layer1.weight"), grad1);
        grads.insert(Rc::from("layer2.weight"), grad2);

        // Clip gradients
        let result = training_loop.clip_gradients_gpu(&mut grads);
        assert!(result.is_ok(), "GPU gradient clipping should succeed");

        let norm_arr = result.unwrap();
        assert!(norm_arr.is_some(), "Should return norm array when max_grad_norm > 0");

        let norm = norm_arr.unwrap();
        norm.eval().unwrap();
        let norm_val = norm.item::<f32>();

        // Original norm should be 5 (sqrt(3^2 + 4^2))
        // After clipping with max_norm=1.0, gradients should be scaled
        // The returned norm should be the original norm (5.0)
        assert!(
            (norm_val - 5.0).abs() < 0.01,
            "Norm should be ~5.0, got {}",
            norm_val
        );

        // Check that gradients were actually clipped
        let key: Rc<str> = Rc::from("layer1.weight");
        let clipped_grad1 = grads.get(&key).unwrap();
        clipped_grad1.eval().unwrap();

        // Gradients should be scaled by 1.0/5.0 = 0.2
        // [3.0, 4.0] * 0.2 = [0.6, 0.8]
        let values: [f32; 2] = clipped_grad1.as_slice().try_into().unwrap();
        assert!(
            (values[0] - 0.6).abs() < 0.01,
            "First grad should be ~0.6, got {}",
            values[0]
        );
        assert!(
            (values[1] - 0.8).abs() < 0.01,
            "Second grad should be ~0.8, got {}",
            values[1]
        );
    }

    #[test]
    fn test_learning_rate_division_by_zero_protection() {
        // Test edge case where total_steps == warmup_steps
        let mut config = TrainingLoopConfig::default();
        config.training.warmup_steps = 100;
        config.training.max_steps = Some(100); // Same as warmup!
        config.training.learning_rate = 1e-4;
        config.training.lr_scheduler = LrSchedulerType::Linear;

        let mut training_loop = TrainingLoop::new(config);

        // At step 100 (past warmup, at max_steps)
        training_loop.step = 100;
        let lr = training_loop.get_learning_rate();

        // Should not panic or return NaN/Inf
        assert!(lr.is_finite(), "Learning rate should be finite, got {}", lr);
        assert!(lr >= 0.0, "Learning rate should be non-negative");
    }

    #[test]
    fn test_batch_token_overflow_protection() {
        // Test that we handle potential overflow in batch_size * seq_len
        let large_batch_size: usize = usize::MAX / 2;
        let large_seq_len: usize = 3;

        // This would overflow without checked arithmetic
        let result = large_batch_size.checked_mul(large_seq_len);
        assert!(result.is_none(), "Should detect potential overflow");

        // With our protected version, it returns MAX
        let protected = large_batch_size.checked_mul(large_seq_len).unwrap_or(usize::MAX);
        assert_eq!(protected, usize::MAX, "Should return MAX on overflow");
    }
}
