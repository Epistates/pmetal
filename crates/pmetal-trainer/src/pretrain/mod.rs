//! Full-parameter pretraining.
//!
//! This module parallels the LoRA training loop rather than sharing it —
//! pretraining needs a different checkpoint layout, full optimizer-state
//! persistence, gradient-checkpointing hooks, and depth-aware initialization
//! that would clutter the adapter pipeline.

pub mod checkpoint;
pub mod factory;
pub mod init;
pub mod loss;
pub mod train_loop;

pub use checkpoint::{CheckpointMeta, load_checkpoint, save_checkpoint};
pub use factory::{PretrainModel, create_model, n_layers};
pub use init::{apply_depth_scaled_init, zero_biases};
pub use loss::causal_lm_loss;
pub use train_loop::{pretrain_step, run_pretrain};

use pmetal_bridge::compat::{Array, Exception, module::ModuleParameters};
use pmetal_core::LrSchedulerType;
use pmetal_models::architectures;

/// Runtime configuration for a pretraining run.
#[derive(Debug, Clone)]
pub struct PretrainConfig {
    pub num_steps: usize,
    pub learning_rate: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub lr_schedule: LrSchedulerType,
    pub weight_decay: f32,
    pub betas: (f32, f32),
    pub eps: f32,
    pub max_grad_norm: Option<f32>,
    pub ignore_index: Option<i32>,
    pub z_loss_coef: Option<f32>,
    pub n_layers: usize,
    pub apply_init: bool,
    pub checkpoint_every: Option<usize>,
    pub checkpoint_dir: Option<std::path::PathBuf>,
    /// Number of micro-batches to accumulate before one optimizer step.
    /// Effective batch = `batch_size * gradient_accumulation_steps`.
    pub gradient_accumulation_steps: usize,
    /// Print step/loss/LR/tok-per-sec every N steps. 0 disables.
    pub log_every: usize,
    /// Run eval on held-out batches every N steps. 0 disables.
    pub eval_every: usize,
    /// Number of eval batches per evaluation round.
    pub eval_batches: usize,
}

impl Default for PretrainConfig {
    fn default() -> Self {
        Self {
            num_steps: 100,
            learning_rate: 3e-4,
            min_lr: 1e-5,
            warmup_steps: 100,
            lr_schedule: LrSchedulerType::Cosine,
            weight_decay: 0.1,
            betas: (0.9, 0.95),
            eps: 1e-8,
            max_grad_norm: Some(1.0),
            ignore_index: None,
            z_loss_coef: None,
            n_layers: 0,
            apply_init: false,
            checkpoint_every: None,
            checkpoint_dir: None,
            gradient_accumulation_steps: 1,
            log_every: 10,
            eval_every: 0,
            eval_batches: 10,
        }
    }
}

/// Causal LM interface required by the pretraining loop.
pub trait CausalLm: ModuleParameters {
    fn forward_logits(&mut self, input_ids: &Array) -> Result<Array, Exception>;
    fn vocab_size(&self) -> i32;
}

// Macro for architectures where forward takes (input_ids, mask) and config()
// exposes vocab_size.
macro_rules! impl_causal_lm {
    ($ty:ty) => {
        impl CausalLm for $ty {
            fn forward_logits(&mut self, input_ids: &Array) -> Result<Array, Exception> {
                self.forward(input_ids, None)
            }
            fn vocab_size(&self) -> i32 {
                self.config().vocab_size
            }
        }
    };
}

// GptOss has a different forward signature: (input_ids, mask, cache)
impl CausalLm for architectures::GptOssForCausalLM {
    fn forward_logits(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.forward(input_ids, None, None)
    }
    fn vocab_size(&self) -> i32 {
        architectures::GptOssForCausalLM::vocab_size(self)
    }
}

impl_causal_lm!(architectures::LlamaForCausalLM);
impl_causal_lm!(architectures::Qwen2ForCausalLM);
impl_causal_lm!(architectures::MistralForCausalLM);
impl_causal_lm!(architectures::GemmaForCausalLM);
impl_causal_lm!(architectures::PhiForCausalLM);

// Qwen3 and Qwen3.5 store config on the outer struct, not behind config().
impl CausalLm for architectures::Qwen3ForCausalLM {
    fn forward_logits(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.forward(input_ids, None)
    }
    fn vocab_size(&self) -> i32 {
        self.config.vocab_size
    }
}

impl CausalLm for architectures::Qwen3NextForCausalLM {
    fn forward_logits(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.forward(input_ids, None)
    }
    fn vocab_size(&self) -> i32 {
        self.config.vocab_size
    }
}

impl CausalLm for architectures::Gemma4ForCausalLM {
    fn forward_logits(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.forward(input_ids, None)
    }
    fn vocab_size(&self) -> i32 {
        self.config.vocab_size
    }
}

// Qwen3 MoE has 3-arg forward like GptOss (input_ids, mask, cache)
impl CausalLm for architectures::Qwen3MoE {
    fn forward_logits(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.forward(input_ids, None, None)
    }
    fn vocab_size(&self) -> i32 {
        self.config.vocab_size
    }
}

/// Errors returned by the pretraining loop.
#[derive(Debug, thiserror::Error)]
pub enum PretrainError {
    #[error("pretrain autograd failed: {0}")]
    Autograd(String),
    #[error("pretrain optimizer update failed: {0}")]
    Optimizer(String),
    #[error("pretrain batch iterator exhausted at step {step}")]
    BatchIteratorExhausted { step: usize },
    #[error("pretrain checkpoint: {0}")]
    Checkpoint(String),
}
