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
pub use train_loop::{pretrain_step, pretrain_step_batches, run_pretrain, run_pretrain_with_state};

use pmetal_core::LrSchedulerType;
use pmetal_models::architectures;
use pmetal_runtime::compat::{Array, Exception, module::ModuleParameters};

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

impl PretrainConfig {
    /// Check the settings the loop acts on.
    ///
    /// # Errors
    /// [`PretrainError::Config`] naming every setting that cannot work: a
    /// run's length, rates, betas, and a learning-rate floor above the peak,
    /// all of which the schedule would otherwise turn into a nonsense curve
    /// without complaint.
    pub fn validate(&self) -> Result<(), PretrainError> {
        let mut problems = Vec::new();
        let finite_positive = |name: &str, v: f32, problems: &mut Vec<String>| {
            if !(v.is_finite() && v > 0.0) {
                problems.push(format!("{name} must be a finite number > 0, got {v}"));
            }
        };
        if self.num_steps == 0 {
            problems.push("num_steps must be at least 1".to_string());
        }
        finite_positive("learning_rate", self.learning_rate, &mut problems);
        finite_positive("eps", self.eps, &mut problems);
        if !(self.min_lr.is_finite() && self.min_lr >= 0.0) {
            problems.push(format!(
                "min_lr must be a finite number >= 0, got {}",
                self.min_lr
            ));
        } else if self.min_lr > self.learning_rate {
            problems.push(format!(
                "min_lr ({}) is above learning_rate ({}); the schedule would climb",
                self.min_lr, self.learning_rate
            ));
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            problems.push(format!(
                "weight_decay must be a finite number >= 0, got {}",
                self.weight_decay
            ));
        }
        for (name, beta) in [("betas.0", self.betas.0), ("betas.1", self.betas.1)] {
            if !(0.0..1.0).contains(&beta) {
                problems.push(format!("{name} must be in [0, 1), got {beta}"));
            }
        }
        if let Some(norm) = self.max_grad_norm
            && !(norm.is_finite() && norm > 0.0)
        {
            problems.push(format!(
                "max_grad_norm must be a finite number > 0 when set, got {norm}"
            ));
        }
        if let Some(z) = self.z_loss_coef
            && !(z.is_finite() && z >= 0.0)
        {
            problems.push(format!("z_loss_coef must be a finite number >= 0, got {z}"));
        }
        if self.gradient_accumulation_steps == 0 {
            problems.push("gradient_accumulation_steps must be at least 1".to_string());
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(PretrainError::Config(problems.join("; ")))
        }
    }
}

/// Causal LM interface required by the pretraining loop.
pub trait CausalLm: ModuleParameters {
    fn forward_logits(&mut self, input_ids: &Array) -> Result<Array, Exception>;

    /// [`forward_logits`](Self::forward_logits) with an additive attention
    /// mask (`0` where a token may attend, `-inf` where not), as a
    /// block-diagonal document mask is. An architecture that cannot take one
    /// says so instead of running unmasked, which would let a packed row's
    /// documents attend to each other.
    fn forward_logits_masked(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        match mask {
            None => self.forward_logits(input_ids),
            Some(_) => Err(Exception::custom(
                "this architecture does not take a document attention mask",
            )),
        }
    }

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
            fn forward_logits_masked(
                &mut self,
                input_ids: &Array,
                mask: Option<&Array>,
            ) -> Result<Array, Exception> {
                self.forward(input_ids, mask)
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
    fn forward_logits_masked(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward(input_ids, mask, None)
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
    fn forward_logits_masked(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward(input_ids, mask)
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
    fn forward_logits_masked(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward(input_ids, mask)
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
    fn forward_logits_masked(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward(input_ids, mask, None)
    }
    fn vocab_size(&self) -> i32 {
        self.config.vocab_size
    }
}

/// One micro-batch of a pretraining run.
pub struct PretrainBatch {
    /// `[batch, seq_len]` token ids.
    pub tokens: Array,
    /// Where the stream is after it, written into checkpoints so a resume
    /// restarts at the exact corpus boundary.
    pub position: Option<pmetal_data::streaming::StreamPosition>,
    /// The documents in each row of `tokens` (each length counts its own
    /// separator). When present, the loop masks attention to the block
    /// diagonal so a document attends only to itself; when absent, a row is
    /// one sequence.
    pub doc_lengths: Option<Vec<Vec<usize>>>,
}

impl PretrainBatch {
    /// A batch of one sequence per row.
    pub fn new(tokens: Array) -> Self {
        Self {
            tokens,
            position: None,
            doc_lengths: None,
        }
    }
}

/// The additive block-diagonal causal mask of rows made of documents:
/// `[batch, 1, seq_len, seq_len]`, `0` where a token may attend (an earlier
/// or equal position of its own document) and `-inf` elsewhere.
///
/// `doc_lengths[row]` lists each document's length in the row, in order, and
/// must sum to `seq_len`.
pub fn document_mask(doc_lengths: &[Vec<usize>], seq_len: usize) -> Result<Array, PretrainError> {
    let mut block_ids = Vec::with_capacity(doc_lengths.len() * seq_len);
    for (row, docs) in doc_lengths.iter().enumerate() {
        let total: usize = docs.iter().sum();
        if total != seq_len {
            return Err(PretrainError::Config(format!(
                "row {row}'s documents hold {total} tokens, not the row's {seq_len}"
            )));
        }
        for (id, &len) in docs.iter().enumerate() {
            block_ids.extend(std::iter::repeat_n(id as i32, len));
        }
    }
    let (b, t) = (doc_lengths.len() as i32, seq_len as i32);
    let block = Array::from_i32_slice_shaped(&block_ids, &[b, t]);
    let same = block.reshape(&[b, t, 1]).eq(&block.reshape(&[b, 1, t]));
    let position = pmetal_runtime::compat::ops::arange_n(t);
    let causal = pmetal_runtime::compat::ops::greater_equal(
        &position.reshape(&[t, 1]),
        &position.reshape(&[1, t]),
    );
    let allowed = pmetal_runtime::compat::ops::logical_and(&same, &causal.reshape(&[1, t, t]));
    Ok(allowed
        .where_cond(&Array::from_f32(0.0), &Array::from_f32(f32::NEG_INFINITY))
        .reshape(&[b, 1, t, t]))
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
    #[error("invalid pretrain configuration: {0}")]
    Config(String),
    #[error("pretrain data: {0}")]
    Data(String),
}
