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

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    losses::CrossEntropy,
    module::{FlattenedModuleParam, ModuleParameters},
    nn,
    ops::indexing::{IndexOp, argmax_axis},
    optimizers::{AdamW, AdamWBuilder, Optimizer},
    transforms::compile::compile_with_state,
    utils::Updatable,
};
use pmetal_core::{EvalMetrics, LrSchedulerType, TrainingConfig};
use pmetal_data::{
    DataLoader, DataLoaderConfig, PackedDataLoader, PackedTrainingBatch, PackerConfig,
    TrainingBatch, TrainingDataset,
};
use pmetal_lora::TrainableModel;
use pmetal_mlx::kernels::cross_entropy::cross_entropy_loss;
use pmetal_mlx::kernels::{init_training_context, with_training_mode};

use crate::mlx_metal_optimizer::{
    MlxMetalOptimizer, MlxMetalOptimizerBuilder, is_mlx_metal_optimizer_available,
};
use crate::{CheckpointManager, CheckpointMetadata, Result, SftError};

mod run_compiled;
mod run_metal_fused;
mod run_packed;
mod run_standard;
mod step_functions;
#[cfg(test)]
mod tests;

// Re-export step functions so run_*.rs submodules can access them via `use super::*`
pub(crate) use step_functions::{
    compute_cce_loss, eval_training_state, jit_training_step, jit_training_step_cce,
    jit_training_step_inner, jit_training_step_packed, jit_training_step_packed_cce,
};

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
    /// Recommended default is 5e-5 for embeddings vs 2e-4 for LoRA params.
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

    /// NEFTune noise alpha for embedding fine-tuning (Jain et al., 2023).
    ///
    /// When set, uniform noise U(-mag, mag) is added to token embeddings during
    /// each training forward pass, where `mag = alpha / sqrt(seq_len * embed_dim)`.
    /// This regularisation improves instruction-following by 15-20% with no extra
    /// compute cost and no change to inference.
    ///
    /// Recommended values: 5.0 for small models, 15.0 for larger models.
    /// `None` disables NEFTune (default).
    pub neftune_noise_alpha: Option<f32>,

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
    /// Requires the model to implement `forward_hidden()` and `lm_head_weight()`.
    /// Falls back to standard cross-entropy silently when the model does not support it.
    pub use_cut_cross_entropy: bool,

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
            use_metal_flash_attention: true,
            log_every: 10,
            checkpoint_every: 500,
            eval_every: 100,
            use_jit_compilation: false,
            use_sequence_packing: false,
            gradient_checkpointing: false,
            gradient_checkpointing_layers: 4,
            embedding_lr: None,
            eager_evaluation: false,
            use_metal_fused_optimizer: false,
            neftune_noise_alpha: None,
            loraplus_lr_ratio: None,
            use_cut_cross_entropy: false,
            #[cfg(feature = "distributed")]
            distributed: None,
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
    /// Accumulated loss across micro-batches for gradient accumulation.
    pub(crate) accumulated_loss: f64,
    /// Number of micro-batches accumulated (for averaging).
    pub(crate) loss_accumulation_count: usize,
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
    pub(crate) best_lora_snapshot: Option<std::collections::HashMap<std::rc::Rc<str>, Array>>,
    /// When set, each in-memory snapshot is also persisted to this directory
    /// as `best_snapshot.safetensors` so it survives process interruptions.
    pub(crate) snapshot_persist_dir: Option<std::path::PathBuf>,
    /// Distributed gradient synchronization bridge.
    /// When present, gradients are all-reduced across nodes after each accumulation cycle.
    #[cfg(feature = "distributed")]
    pub(crate) distributed: Option<crate::distributed_bridge::DistributedGradientSync>,
}

impl TrainingLoop {
    /// Create a new training loop.
    pub fn new(mut config: TrainingLoopConfig) -> Self {
        config.log_every = config.log_every.max(1);

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
            accumulated_loss: 0.0,
            loss_accumulation_count: 0,
            pending_checkpoint: None,
            callbacks: Vec::new(),
            adaptive_lr: None,
            adaptive_lr_override: None,
            best_lora_snapshot: None,
            snapshot_persist_dir: None,
            #[cfg(feature = "distributed")]
            distributed: None,
        }
    }

    /// Add a training callback for metrics logging or dashboard integration.
    pub fn add_callback(&mut self, callback: Box<dyn pmetal_core::TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Enable adaptive LR control with the given config.
    pub fn enable_adaptive_lr(&mut self, config: crate::adaptive_lr::AdaptiveLrConfig) {
        self.adaptive_lr = Some(crate::adaptive_lr::AdaptiveLrController::new(config));
    }

    /// Set the distributed gradient sync bridge for multi-node training.
    #[cfg(feature = "distributed")]
    pub fn set_distributed(&mut self, sync: crate::distributed_bridge::DistributedGradientSync) {
        // Also update the DataLoader config for data sharding
        self.config.dataloader.rank = sync.rank();
        self.config.dataloader.world_size = sync.world_size();
        self.distributed = Some(sync);
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
    fn check_cancelled(&self) -> bool {
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
    pub(crate) fn get_scheduled_lr(&self) -> f32 {
        use pmetal_core::LearningRateScheduler;

        let cfg = &self.config.training;
        let total_steps = cfg.max_steps.unwrap_or(10000);

        let scheduler = LearningRateScheduler::new(
            cfg.learning_rate,
            total_steps,
            cfg.warmup_steps,
            cfg.lr_scheduler,
        );

        scheduler.get_lr(self.step) as f32
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
                crate::adaptive_lr::LrEvent::Scheduled => {
                    // Check if we should snapshot the current weights as best
                    if ctrl.should_snapshot_best(step) {
                        // Signal that the training loop should take a snapshot
                        // (actual snapshotting happens in the training method,
                        //  since we don't have access to the model here)
                        AdaptiveAction::Continue
                    } else {
                        AdaptiveAction::Continue
                    }
                }
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

    /// Unified post-step logic: adaptive LR + snapshot + rollback handling.
    ///
    /// Consolidates the pattern duplicated across all training loop variants:
    ///   1. Feed loss to adaptive LR controller
    ///   2. Snapshot best weights if loss improved
    ///   3. Restore from snapshot on rollback
    ///   4. Signal early stop when max rollbacks exhausted
    ///
    /// Returns `AdaptiveAction` indicating what the caller should do next.
    pub fn post_step_adaptive<M: TrainableModel>(
        &mut self,
        loss: f64,
        model: &mut M,
    ) -> AdaptiveAction {
        let action = self.apply_adaptive_lr(loss);
        match action {
            AdaptiveAction::Continue => {
                if self.should_snapshot_best() {
                    self.snapshot_best_weights(model);
                }
                AdaptiveAction::Continue
            }
            AdaptiveAction::Rollback => {
                tracing::warn!(step = self.step, "Divergence detected — rolling back to best snapshot");
                if self.restore_best_weights(model) {
                    AdaptiveAction::Rollback
                } else {
                    // No snapshot available — treat as continue
                    AdaptiveAction::Continue
                }
            }
            AdaptiveAction::EarlyStop => {
                tracing::warn!(step = self.step, "Max rollbacks exhausted — early stopping");
                AdaptiveAction::EarlyStop
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

        // Compute masked mean: only average over non-ignored (-100) tokens.
        // loss.mean(None) would incorrectly divide by all positions including
        // ignored ones, producing an underestimated loss.
        let mask = flat_labels.ne(&Array::from_slice(&[-100_i64], &[1]))?;
        let valid_count_raw = mask.as_dtype(mlx_rs::Dtype::Float32)?.sum(None)?;
        let valid_count = mlx_rs::ops::maximum(&valid_count_raw, &Array::from_f32(1.0))?;
        let masked_loss = loss.multiply(&mask.as_dtype(loss.dtype())?)?;
        Ok(masked_loss.sum(None)?.divide(&valid_count)?)
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
    pub(crate) fn clip_gradients_gpu(
        &self,
        grads: &mut FlattenedModuleParam,
    ) -> Result<Option<Array>> {
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

        // Compute scale = min(1.0, max_norm / norm) entirely on GPU.
        // `maximum(norm, max_norm)` clamps the denominator to [max_norm, ∞),
        // so scale ∈ (0, 1].  No epsilon is needed: when norm < max_norm the
        // clamped denominator equals max_norm (not zero), avoiding division by zero.
        let max_norm_arr = Array::from_f32(max_norm);
        let norm_clamped = mlx_rs::ops::maximum(&norm, &max_norm_arr)?;
        let scale = max_norm_arr.divide(&norm_clamped)?;

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
            tracing::info!(
                "Gradient clipping: norm={:.4}, max_norm={}, scale={:.4}",
                norm_val,
                max_norm,
                scale_val
            );
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
    /// Uses a GPU-CPU sync to get the actual gradient norm value.
    /// For maximum throughput, use clip_gradients_gpu() instead.
    #[allow(dead_code)] // Available for callers that need precise grad_norm values
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

        // Only clip if norm exceeds max and is finite (NaN/Inf gradients should not be scaled)
        if total_norm > max_norm && total_norm.is_finite() {
            let scale = max_norm / (total_norm + 1e-6);
            let scale_arr = Array::from_f32(scale);
            for (_, grad) in grads.iter_mut() {
                *grad = grad.multiply(&scale_arr)?;
            }
        }

        Ok(Some(total_norm))
    }

    /// Accumulate gradients.
    pub(crate) fn accumulate_gradients(&mut self, new_grads: FlattenedModuleParam) -> Result<()> {
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
                let scale_arr = Array::from_f32(scale);
                for (key, new_grad) in new_grads {
                    if let Some(existing) = acc.get_mut(&key) {
                        let scaled = new_grad.multiply(&scale_arr)?;
                        *existing = existing.add(&scaled)?;
                    } else {
                        let scaled = new_grad.multiply(&scale_arr)?;
                        acc.insert(key, scaled);
                    }
                }
            }
        }

        self.accumulation_step += 1;

        // Evaluate accumulated gradients to free the backward computation graph.
        // Without this, each micro-batch's gradient arrays keep the entire
        // forward+backward graph alive (~14 GB per step for a 0.6B model).
        // With grad_accum=4, that's 56 GB before any evaluation happens.
        if let Some(ref acc) = self.accumulated_grads {
            let grad_arrays: Vec<&Array> = acc.values().collect();
            if !grad_arrays.is_empty() {
                mlx_rs::transforms::eval(grad_arrays)?;
            }
        }
        Ok(())
    }

    /// Check if we should apply accumulated gradients.
    pub(crate) fn should_apply_gradients(&self) -> bool {
        self.accumulation_step >= self.config.training.gradient_accumulation_steps
    }

    /// Take accumulated gradients and reset counter.
    pub(crate) fn take_accumulated_gradients(&mut self) -> Option<FlattenedModuleParam> {
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

        if self.step <= 1 {
            tracing::debug!(
                step = self.step,
                batch_size = batch.batch_size,
                seq_len = batch.seq_len,
                "train_step: computing loss and gradients"
            );
        }

        // Compute loss and gradients based on whether we have pixel values (VLM)
        let (loss, grads) = if let Some(ref pixel_values) = batch.pixel_values {
            // VLM training with image inputs
            self.compute_vlm_loss_and_grads(model, batch, pixel_values)?
        } else {
            // Text-only training (standard path)
            self.compute_text_loss_and_grads(model, batch)?
        };

        // Evaluate loss for this micro-batch
        // NOTE: Unlike mlx-lm which uses mx.compile for JIT fusion, we need to
        // evaluate each step. mlx-rs doesn't expose mx.compile, so deferring
        // evaluation just builds up a massive computation graph.
        if self.step <= 1 {
            tracing::debug!(step = self.step, "train_step: evaluating loss...");
        }
        loss.eval()?;
        let micro_batch_loss = loss.item::<f32>();
        if self.step <= 3 {
            let active = pmetal_mlx::memory::get_active_memory();
            let cache = pmetal_mlx::memory::get_cache_memory();
            tracing::info!(
                step = self.step,
                loss = micro_batch_loss,
                active_mb = active / (1024 * 1024),
                cache_mb = cache / (1024 * 1024),
                "train_step: loss evaluated"
            );
        }

        // Accumulate loss across micro-batches for accurate reporting
        self.accumulated_loss += micro_batch_loss as f64;
        self.loss_accumulation_count += 1;

        // Accumulate gradients
        self.accumulate_gradients(grads)?;

        // Apply gradients if accumulation is complete
        // Use GPU-based clipping to avoid GPU-CPU sync overhead
        let grad_norm = if self.should_apply_gradients() {
            if let Some(mut accumulated) = self.take_accumulated_gradients() {
                // Clip gradients using GPU-based method (no sync required)
                // Returns lazy norm Array that we only eval when logging
                let lazy_norm = self.clip_gradients_gpu(&mut accumulated)?;

                // Distributed gradient sync: all-reduce gradients across nodes
                #[cfg(feature = "distributed")]
                if let Some(ref mut dist) = self.distributed {
                    tokio::runtime::Handle::current()
                        .block_on(async { dist.sync_gradients(&mut accumulated).await })?;
                }

                // Apply with optimizer (lazy - no eval by default)
                optimizer.update(model, accumulated)?;

                // Eager evaluation mode
                // Forces immediate evaluation to clear intermediate activations.
                // Trade-off: Lower memory usage vs lower throughput.
                if self.config.eager_evaluation {
                    // Eval only trainable (LoRA) parameters — NOT the frozen base
                    // model. Evaluating all 600M+ params of a frozen model every step
                    // is wasteful and can spike memory via unnecessary GPU->CPU sync.
                    // This matches mlx-lm's approach: mx.eval(state, losses, ...).
                    mlx_rs::transforms::eval_params(model.trainable_parameters())?;
                    // Optimizer state (momentum/variance for Adam)
                    let opt_states: Vec<&Array> =
                        optimizer.updatable_states().into_iter().collect();
                    if !opt_states.is_empty() {
                        mlx_rs::transforms::eval(opt_states)?;
                    }
                }
                // NOTE: Do NOT clear the buffer cache here. MLX's cache holds
                // freed buffers for reuse — clearing it forces fresh allocations
                // that inflate RSS. Without mx.compile (unavailable in mlx-rs),
                // every step allocates new buffers; the cache is what prevents
                // unbounded RSS growth by recycling them. MLX's built-in
                // backpressure (memory_limit) handles OOM prevention.

                // NOTE: When eager_evaluation is false, we use deferred evaluation.
                // MLX builds a lazy computation graph - forcing evaluation after
                // every optimizer step is a massive bottleneck (20-50s per step!).
                // Parameters will be evaluated lazily when needed (at logging or
                // checkpoint time). This matches mlx-lm's approach.

                // Always compute grad_norm when clipping is enabled (tests expect this).
                // The lazy Array is evaluated here — syncs GPU->CPU.
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

        // Compute the reported loss: average across micro-batches when gradient
        // accumulation completes, otherwise report the current micro-batch loss.
        let loss_val = if grad_norm.is_some() {
            // Gradients were applied — report averaged loss across all micro-batches
            let avg = self.accumulated_loss / self.loss_accumulation_count.max(1) as f64;
            self.accumulated_loss = 0.0;
            self.loss_accumulation_count = 0;
            avg as f32
        } else {
            micro_batch_loss
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
    ///
    /// When `use_cut_cross_entropy` is set and the model supports it, computes the
    /// loss directly from hidden states (CCE path) without materialising the full
    /// [batch, seq, vocab] logits tensor.  Falls back to standard cross-entropy
    /// when NEFTune is active or the model does not implement the CCE methods.
    pub(crate) fn compute_text_loss_and_grads<M: TrainableModel>(
        &self,
        model: &mut M,
        batch: &TrainingBatch,
    ) -> Result<(Array, FlattenedModuleParam)> {
        let neftune_alpha = self.config.neftune_noise_alpha;
        // Fetch the LM head weight once — serves as capability probe and avoids
        // a second call to lm_head_weight() inside the gradient closure.
        let lm_weight_cached = if self.config.use_cut_cross_entropy && neftune_alpha.is_none() {
            model.lm_head_weight()
        } else {
            None
        };
        let use_cce = lm_weight_cached.is_some();

        if use_cce {
            let cached_weight = lm_weight_cached.expect("checked above");
            // CCE path: forward hidden states, then compute loss without logits.
            let loss_fn = |model: &mut M,
                           (input_ids, labels): (&Array, &Array)|
             -> std::result::Result<Array, Exception> {
                let hidden_opt = model.forward_hidden(input_ids, None);
                match hidden_opt {
                    Some(Ok(hidden_states)) => {
                        let seq_len = hidden_states.dim(1);
                        let shift_hidden = hidden_states.index((.., ..seq_len - 1, ..));
                        let shift_labels = labels.index((.., 1..));
                        let flat_labels = shift_labels.reshape(&[-1])?;
                        compute_cce_loss(&shift_hidden, &cached_weight, &flat_labels)
                    }
                    _ => {
                        // CCE unavailable at runtime — fall back to standard CE.
                        let logits = model
                            .forward(input_ids, None)
                            .map_err(|e| Exception::custom(e.to_string()))?;
                        Self::compute_loss(&logits, labels)
                            .map_err(|e| Exception::custom(e.to_string()))
                    }
                }
            };

            let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
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
        } else {
            // Standard path: forward through lm_head, compute cross-entropy.
            let loss_fn = |model: &mut M,
                           (input_ids, labels): (&Array, &Array)|
             -> std::result::Result<Array, Exception> {
                let logits = if let Some(alpha) = neftune_alpha {
                    model
                        .forward_noised(input_ids, None, alpha)
                        .map_err(|e| Exception::custom(e.to_string()))?
                } else {
                    model
                        .forward(input_ids, None)
                        .map_err(|e| Exception::custom(e.to_string()))?
                };
                Self::compute_loss(&logits, labels).map_err(|e| Exception::custom(e.to_string()))
            };

            let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
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
            Self::compute_loss(&logits, labels).map_err(|e| Exception::custom(e.to_string()))
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

    /// Evaluate the model on a dataset.
    ///
    /// Returns comprehensive evaluation metrics including:
    /// - Average loss across all batches
    /// - Perplexity (exp(loss))
    /// - Token-level accuracy (correct next-token predictions)
    pub fn evaluate<M>(&self, model: &mut M, dataset: &TrainingDataset) -> Result<EvalMetrics>
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

    /// Save a checkpoint, offloading the I/O to a background thread.
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
        self.spawn_async_checkpoint(lora_params, metadata, manager, is_best)?;

        Ok(step_dir)
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
    /// 1. Forces evaluation of all LoRA parameter arrays (materializes GPU tensors on CPU).
    /// 2. Converts the `Rc<str>`-keyed map to a `String`-keyed map that is `Send`.
    /// 3. Clones the metadata and extracts the manager state needed on the background thread.
    /// 4. Polls the previous handle for completion/errors before spawning the new one.
    ///
    /// Failures are logged as warnings rather than propagated, so a transient I/O error
    /// does not abort training.
    fn spawn_async_checkpoint(
        &mut self,
        lora_params: std::collections::HashMap<std::rc::Rc<str>, Array>,
        metadata: CheckpointMetadata,
        manager: &CheckpointManager,
        is_best: bool,
    ) -> Result<()> {
        // Poll the previous handle so we surface any errors and free the handle.
        self.poll_pending_checkpoint();

        // Force evaluation of all arrays before crossing the thread boundary.
        // This materializes the MLX lazy computation graph so the background thread
        // only performs the I/O work (no GPU/CPU work happens on the other side).
        {
            let arrays: Vec<&Array> = lora_params.values().collect();
            if !arrays.is_empty() {
                mlx_rs::transforms::eval(arrays)?;
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
                .save_checkpoint_owned(params_send, &metadata, is_best)
                .map_err(|e| e.to_string())?;

            tracing::info!("Async checkpoint write complete (step {})", step_for_log);
            Ok(())
        });

        self.pending_checkpoint = Some(handle);
        Ok(())
    }
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

// =============================================================================
// Custom Autograd Training (memory-efficient LoRA optimization)
// =============================================================================
//
// This module provides custom autograd training that bypasses MLX autodiff
// for LoRA layers, achieving ~50% memory reduction by avoiding materialization
// of full intermediate activations, enabling training of larger models on limited memory.
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
