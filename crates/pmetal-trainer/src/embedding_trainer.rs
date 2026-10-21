//! Embedding / sentence-transformer training loop.
//!
//! Trains encoder models (BERT-family) for producing high-quality sentence
//! embeddings using contrastive learning objectives, either the whole encoder
//! or LoRA adapters on it ([`AdaptedModel`]).
//!
//! ## Supported losses
//!
//! | Loss | Data format | Best for |
//! |------|-------------|----------|
//! | `InfoNce` / `Mnrl` | pairs | Large batches, no explicit labels |
//! | `Triplet` | triplets | Hard negative mining workflows |
//! | `CoSent` | pairs with labels | Ranks pairs by their labels |
//! | `CosineSimilarity` | pairs with labels | Direct similarity regression |
//!
//! In-batch negatives are the other examples of the batch, so the batch size
//! is the number of negatives: it is not split across gradient accumulation.
//!
//! ## Usage
//!
//! ```ignore
//! use pmetal_trainer::embedding_trainer::{EmbeddingTrainer, EmbeddingTrainerConfig};
//! use pmetal_data::EmbeddingDataset;
//!
//! let dataset = EmbeddingDataset::from_jsonl("data/pairs.jsonl")?;
//! let mut trainer = EmbeddingTrainer::new(EmbeddingTrainerConfig::default())?;
//!
//! // encoder: EncoderModel (or an AdaptedModel over one), tokenizer: Tokenizer
//! trainer.run(&mut encoder, &tokenizer, &dataset)?;
//! ```

use pmetal_core::{ConfigValidator, EvalMetrics, TrainingCallback, TrainingConfig};
use pmetal_data::{EmbeddingDataset, Tokenizer};
use pmetal_lora::AdaptedModel;
use pmetal_models::architectures::encoder::EncoderModel;
use pmetal_models::pooling::{PoolingMode, normalize_embeddings, pool};
use pmetal_runtime::compat::{
    Array, Dtype, Exception, ModuleParameters, eval_params, nn, optimizers::Optimizer,
};

use crate::contrastive_loss;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Embedding loss function selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum EmbeddingLossType {
    /// InfoNCE with in-batch negatives. Equivalent to MNRL.
    /// Best for large batch sizes (≥ 32) with no explicit negative labels.
    #[default]
    InfoNce,
    /// Triplet margin loss. Requires triplet data (anchor, positive, negative).
    Triplet,
    /// CoSENT: every pair labelled less similar than another should score a
    /// lower cosine. Binary or graded labels.
    CoSent,
    /// Multiple Negatives Ranking Loss — alias for InfoNCE.
    Mnrl,
    /// Cosine similarity MSE regression. Requires float labels in [0, 1].
    CosineSimilarity,
}

impl std::fmt::Display for EmbeddingLossType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InfoNce => write!(f, "info_nce"),
            Self::Triplet => write!(f, "triplet"),
            Self::CoSent => write!(f, "cosent"),
            Self::Mnrl => write!(f, "mnrl"),
            Self::CosineSimilarity => write!(f, "cosine_similarity"),
        }
    }
}

/// Configuration for embedding training.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingTrainerConfig {
    /// Core training hyperparameters: learning rate and its schedule, weight
    /// decay, gradient clipping, batch size, epochs.
    #[serde(default)]
    pub training: TrainingConfig,
    /// Loss function.
    #[serde(default)]
    pub loss_type: EmbeddingLossType,
    /// Temperature for InfoNCE, and 1/scale for CoSENT (0.05 is CoSENT's
    /// reference scale of 20).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Margin for triplet loss.
    #[serde(default = "default_margin")]
    pub margin: f32,
    /// Pooling strategy for extracting sentence embeddings from hidden states.
    #[serde(default)]
    pub pooling_mode: PoolingMode,
    /// L2-normalise the embeddings before the loss, which makes the losses
    /// cosine-based (the usual choice; required by CoSENT). `false` trains on
    /// raw dot products.
    #[serde(default = "default_true")]
    pub normalize: bool,
    /// Maximum input sequence length, special tokens included.
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    /// Log training progress every N steps.
    #[serde(default = "default_log_every")]
    pub log_every: usize,
    /// Shuffle the dataset at the start of each epoch.
    #[serde(default = "default_true")]
    pub shuffle: bool,
    /// Random seed for shuffling.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_temperature() -> f32 {
    0.05
}
fn default_margin() -> f32 {
    0.3
}
fn default_true() -> bool {
    true
}
fn default_max_seq_len() -> usize {
    512
}
fn default_log_every() -> usize {
    10
}
fn default_seed() -> u64 {
    42
}

impl Default for EmbeddingTrainerConfig {
    fn default() -> Self {
        Self {
            training: TrainingConfig::default(),
            loss_type: EmbeddingLossType::InfoNce,
            temperature: default_temperature(),
            margin: default_margin(),
            pooling_mode: PoolingMode::Mean,
            normalize: true,
            max_seq_len: default_max_seq_len(),
            log_every: default_log_every(),
            shuffle: true,
            seed: default_seed(),
        }
    }
}

impl EmbeddingTrainerConfig {
    /// Check the configuration, naming what cannot work.
    pub fn validate(&self) -> EmbeddingResult<()> {
        self.training
            .validate()
            .map_err(|e| EmbeddingTrainerError::Config(e.to_string()))?;
        let mut problems = Vec::new();
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            problems.push(format!(
                "temperature must be positive, got {}",
                self.temperature
            ));
        }
        if !(self.margin.is_finite() && self.margin >= 0.0) {
            problems.push(format!("margin must not be negative, got {}", self.margin));
        }
        if self.max_seq_len < 2 {
            problems.push("max_seq_len must hold at least the two special tokens".into());
        }
        if self.training.batch_size == 0 {
            problems.push("batch_size must be at least 1".into());
        }
        let in_batch = !matches!(
            self.loss_type,
            EmbeddingLossType::Triplet | EmbeddingLossType::CosineSimilarity
        );
        if in_batch && self.training.batch_size < 2 {
            problems.push(format!(
                "{} takes its negatives from the rest of the batch, so batch_size must be at least 2",
                self.loss_type
            ));
        }
        if self.loss_type == EmbeddingLossType::CoSent && !self.normalize {
            problems.push("cosent scores cosines: it needs normalize".into());
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(EmbeddingTrainerError::Config(problems.join("; ")))
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the embedding trainer.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingTrainerError {
    /// An MLX operation failed.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// A text could not be tokenized.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    /// The configuration or the data cannot be trained on.
    #[error("Configuration error: {0}")]
    Config(String),
    /// An I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type of the embedding trainer.
pub type EmbeddingResult<T> = Result<T, EmbeddingTrainerError>;

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// A model whose token states the trainer pools into sentence embeddings.
///
/// Implemented for the whole encoder ([`EncoderModel`], every parameter
/// trains) and for LoRA adapters over one ([`AdaptedModel`], only the
/// adapters train).
pub trait EmbeddingModel: ModuleParameters {
    /// Hidden states `[batch, seq, hidden]` of `input_ids` under
    /// `attention_mask` (1 for real tokens, 0 for padding).
    fn token_states(
        &mut self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<Array, Exception>;
}

impl EmbeddingModel for EncoderModel {
    fn token_states(
        &mut self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<Array, Exception> {
        self.forward(input_ids, Some(attention_mask))
    }
}

impl EmbeddingModel for AdaptedModel {
    fn token_states(
        &mut self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<Array, Exception> {
        self.forward(input_ids, Some(attention_mask))
            .map_err(|e| Exception::custom(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

/// One text as the encoder was trained to read it: the tokenizer's own
/// special tokens around it (`[CLS] ... [SEP]`, `<s> ... </s>`), cut to
/// `max_len` tokens *inside* them, so a long text keeps its closing token.
///
/// A tokenizer failure is an error, not an empty row.
pub fn encode_text(tokenizer: &Tokenizer, text: &str, max_len: usize) -> EmbeddingResult<Vec<i32>> {
    let fail = |e: pmetal_core::PMetalError| {
        EmbeddingTrainerError::Tokenizer(format!("Failed to tokenize {text:?}: {e}"))
    };
    let plain = tokenizer.encode(text).map_err(fail)?;
    let framed = tokenizer.encode_with_special_tokens(text).map_err(fail)?;
    // The post-processor adds a fixed prefix and suffix around the text's own
    // tokens; find where those sit.
    let start = if plain.is_empty() {
        0
    } else {
        framed
            .windows(plain.len())
            .position(|window| window == plain.as_slice())
            .unwrap_or(0)
    };
    let end = (start + plain.len()).min(framed.len());
    let (prefix, suffix) = (&framed[..start], &framed[end..]);
    let room = max_len.saturating_sub(prefix.len() + suffix.len());
    Ok(prefix
        .iter()
        .chain(plain.iter().take(room))
        .chain(suffix)
        .map(|&id| id as i32)
        .collect())
}

/// Right-padded `(input_ids, attention_mask)` for `texts`.
fn tokenize_batch(
    tokenizer: &Tokenizer,
    texts: &[&str],
    max_len: usize,
) -> EmbeddingResult<(Array, Array)> {
    let pad_id = tokenizer.pad_token_id().unwrap_or(0) as i32;
    let rows = texts
        .iter()
        .map(|text| encode_text(tokenizer, text, max_len))
        .collect::<EmbeddingResult<Vec<_>>>()?;
    let seq = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut ids = vec![pad_id; rows.len() * seq];
    let mut mask = vec![0i32; rows.len() * seq];
    for (i, row) in rows.iter().enumerate() {
        for (j, &id) in row.iter().enumerate() {
            ids[i * seq + j] = id;
            mask[i * seq + j] = 1;
        }
    }
    let shape = [rows.len() as i32, seq as i32];
    Ok((
        Array::from_slice(&ids, &shape),
        Array::from_slice(&mask, &shape),
    ))
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Embedding / sentence-transformer trainer.
pub struct EmbeddingTrainer {
    /// Configuration.
    pub config: EmbeddingTrainerConfig,
    /// Current global step count.
    pub step: usize,
    /// Training callbacks.
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

/// The texts of one training example, in the order the loss reads them.
struct Example<'a> {
    texts: Vec<&'a str>,
    label: f32,
}

impl EmbeddingTrainer {
    /// Create a new trainer with the given configuration, validated.
    pub fn new(config: EmbeddingTrainerConfig) -> EmbeddingResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            step: 0,
            callbacks: Vec::new(),
        })
    }

    /// Register a training callback.
    pub fn add_callback(&mut self, cb: Box<dyn TrainingCallback>) {
        self.callbacks.push(cb);
    }

    /// Train `model` on `dataset`; returns each step's loss.
    ///
    /// The optimizer is the configuration's ([`crate::TrainOptimizer`]), on its
    /// learning-rate schedule (warmup, decay) with the gradient clipped to
    /// `max_grad_norm`. Pairs train with InfoNCE, CoSENT or cosine regression
    /// and triplets with the triplet margin loss; a dataset the loss cannot
    /// use is an error.
    pub fn run<M: EmbeddingModel>(
        &mut self,
        model: &mut M,
        tokenizer: &Tokenizer,
        dataset: &EmbeddingDataset,
    ) -> EmbeddingResult<Vec<f32>> {
        let examples: Vec<Example<'_>> = match dataset {
            EmbeddingDataset::Pairs(pairs) => {
                if self.config.loss_type == EmbeddingLossType::Triplet {
                    return Err(EmbeddingTrainerError::Config(
                        "triplet loss requires triplet data with anchor/positive/negative fields"
                            .into(),
                    ));
                }
                pairs
                    .iter()
                    .map(|p| Example {
                        texts: vec![p.text_a.as_str(), p.text_b.as_str()],
                        label: p.label.unwrap_or(1.0),
                    })
                    .collect()
            }
            EmbeddingDataset::Triplets(triplets) => {
                if self.config.loss_type != EmbeddingLossType::Triplet {
                    tracing::warn!(
                        "triplet data trains with the triplet margin loss, not {}",
                        self.config.loss_type
                    );
                }
                triplets
                    .iter()
                    .map(|t| Example {
                        texts: vec![t.anchor.as_str(), t.positive.as_str(), t.negative.as_str()],
                        label: 1.0,
                    })
                    .collect()
            }
        };
        if examples.is_empty() {
            return Err(EmbeddingTrainerError::Config(
                "the dataset has no examples".into(),
            ));
        }
        let triplets = matches!(dataset, EmbeddingDataset::Triplets(_));

        let batch_size = self.config.training.batch_size;
        let epochs = self.config.training.num_epochs.max(1);
        let batches = examples.len().div_ceil(batch_size);
        let total_steps =
            pmetal_core::total_training_steps(&self.config.training, batches, 1, true);
        let scheduler =
            pmetal_core::LearningRateScheduler::for_training(&self.config.training, total_steps);
        let max_grad_norm = self.config.training.max_grad_norm as f32;
        let mut optimizer = crate::TrainOptimizer::from_config(&self.config.training);

        self.fire_train_start();
        tracing::info!(
            "Embedding training ({}): {} examples, batch={}, epochs={epochs}, {total_steps} steps, loss={}, {:?} schedule",
            if triplets { "triplets" } else { "pairs" },
            examples.len(),
            batch_size,
            self.config.loss_type,
            self.config.training.lr_scheduler,
        );

        let mut losses = Vec::with_capacity(total_steps);
        let mut order: Vec<usize> = (0..examples.len()).collect();
        'epochs: for epoch in 0..epochs {
            if self.config.shuffle {
                use rand::SeedableRng;
                use rand::seq::SliceRandom;
                let mut rng =
                    rand::rngs::StdRng::seed_from_u64(self.config.seed.wrapping_add(epoch as u64));
                order.shuffle(&mut rng);
            }
            self.fire_epoch_start(epoch);
            let mut epoch_loss = 0.0f64;
            let mut epoch_steps = 0usize;

            for chunk in order.chunks(batch_size) {
                if self.step >= total_steps {
                    break 'epochs;
                }
                let started = std::time::Instant::now();
                let batch: Vec<&Example<'_>> = chunk.iter().map(|&i| &examples[i]).collect();
                // One padded tensor pair per text of the example.
                let groups = (0..batch[0].texts.len())
                    .map(|k| {
                        let texts: Vec<&str> = batch.iter().map(|e| e.texts[k]).collect();
                        tokenize_batch(tokenizer, &texts, self.config.max_seq_len)
                    })
                    .collect::<EmbeddingResult<Vec<_>>>()?;
                let labels = Array::from_slice(
                    &batch.iter().map(|e| e.label).collect::<Vec<_>>(),
                    &[batch.len() as i32],
                );

                let (loss_type, temperature, margin) = (
                    self.config.loss_type,
                    self.config.temperature,
                    self.config.margin,
                );
                let (pooling_mode, normalize) = (self.config.pooling_mode, self.config.normalize);
                let loss_fn = |model: &mut M,
                               (groups, labels): (&Vec<(Array, Array)>, &Array)|
                 -> Result<Array, Exception> {
                    let embeddings = groups
                        .iter()
                        .map(|(ids, mask)| encode_inner(model, ids, mask, pooling_mode, normalize))
                        .collect::<Result<Vec<_>, _>>()?;
                    compute_loss(&embeddings, labels, loss_type, temperature, margin)
                };
                let mut loss_and_grad = nn::value_and_grad(loss_fn);
                let (loss, mut grads) = loss_and_grad(model, (&groups, &labels))?;

                let grad_norm = (max_grad_norm > 0.0).then(|| {
                    pmetal_runtime::training::clip_grad_norm_map(&mut grads, max_grad_norm)
                });
                let lr = scheduler.get_lr(self.step) as f32;
                optimizer.set_lr(lr);
                optimizer.update(model, grads)?;
                eval_params(model.trainable_parameters())?;

                let loss_val = crate::step_check::check_step(self.step + 1, loss.item_f32())?;
                let grad_norm = match grad_norm {
                    Some(norm) => {
                        norm.eval();
                        Some(crate::step_check::check_grad_norm(
                            self.step + 1,
                            norm.item_f32(),
                        )?)
                    }
                    None => None,
                };
                losses.push(loss_val);
                epoch_loss += loss_val as f64;
                epoch_steps += 1;
                self.step += 1;

                if self.step % self.config.log_every.max(1) == 0 {
                    tracing::info!(
                        "step={} loss={:.4} lr={lr:.2e} epoch={}/{epochs}",
                        self.step,
                        loss_val,
                        epoch + 1,
                    );
                }
                let elapsed = started.elapsed().as_secs_f64();
                let metrics = pmetal_core::StepMetrics {
                    step: self.step,
                    epoch,
                    total_epochs: epochs,
                    total_steps,
                    loss: loss_val as f64,
                    lr: lr as f64,
                    total_ms: elapsed * 1000.0,
                    grad_norm: grad_norm.map(f64::from),
                    ..Default::default()
                };
                for cb in &mut self.callbacks {
                    cb.on_step_end_with_metrics(&metrics);
                }
                // A callback's request to stop ends the run after this step;
                // what is trained so far is the caller's to save.
                if self.callbacks.iter().any(|cb| cb.should_stop()) {
                    tracing::info!("stop requested: ending after step {}", self.step);
                    break 'epochs;
                }
            }

            let avg_loss = epoch_loss / epoch_steps.max(1) as f64;
            tracing::info!("epoch={}/{epochs} avg_loss={avg_loss:.4}", epoch + 1);
            self.fire_epoch_end(epoch, avg_loss as f32);
        }

        self.fire_train_end();
        Ok(losses)
    }

    // -----------------------------------------------------------------------
    // Inference / encoding (no gradient)
    // -----------------------------------------------------------------------

    /// Encode a batch of texts into sentence embeddings (f32, no gradient
    /// tracking), framed and pooled as in training.
    ///
    /// Useful for evaluation and similarity search after training.
    pub fn encode<M: EmbeddingModel>(
        &self,
        model: &mut M,
        tokenizer: &Tokenizer,
        texts: &[&str],
    ) -> EmbeddingResult<Array> {
        let (ids, mask) = tokenize_batch(tokenizer, texts, self.config.max_seq_len)?;
        Ok(encode_inner(
            model,
            &ids,
            &mask,
            self.config.pooling_mode,
            self.config.normalize,
        )?)
    }

    // -----------------------------------------------------------------------
    // Callback helpers
    // -----------------------------------------------------------------------

    fn fire_train_start(&mut self) {
        for cb in &mut self.callbacks {
            cb.on_train_start();
        }
    }

    fn fire_train_end(&mut self) {
        for cb in &mut self.callbacks {
            cb.on_train_end();
        }
    }

    fn fire_epoch_start(&mut self, epoch: usize) {
        for cb in &mut self.callbacks {
            cb.on_epoch_start(epoch);
        }
    }

    fn fire_epoch_end(&mut self, epoch: usize, avg_loss: f32) {
        let metrics = EvalMetrics {
            loss: avg_loss as f64,
            ..Default::default()
        };
        for cb in &mut self.callbacks {
            cb.on_epoch_end(epoch, &metrics);
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions used inside autograd closures (must be stateless)
// ---------------------------------------------------------------------------

/// Forward pass → pool → f32 → optional L2-normalise.
///
/// The embeddings are f32 whatever the model's dtype: the contrastive losses
/// divide similarities by a small temperature, which magnifies half-precision
/// rounding.
fn encode_inner<M: EmbeddingModel>(
    model: &mut M,
    input_ids: &Array,
    attention_mask: &Array,
    pooling_mode: PoolingMode,
    normalize: bool,
) -> Result<Array, Exception> {
    let hidden = model.token_states(input_ids, attention_mask)?;
    let emb = pool(&hidden, attention_mask, pooling_mode)?.as_dtype(Dtype::Float32.as_i32());
    if normalize {
        normalize_embeddings(&emb)
    } else {
        Ok(emb)
    }
}

/// The contrastive loss of a batch's embeddings: one `[B, dim]` matrix per
/// text of the example (anchor and positive, or anchor, positive, negative).
fn compute_loss(
    embeddings: &[Array],
    labels: &Array,
    loss_type: EmbeddingLossType,
    temperature: f32,
    margin: f32,
) -> Result<Array, Exception> {
    match (loss_type, embeddings) {
        (_, [anchor, positive, negative]) => {
            contrastive_loss::triplet_loss(anchor, positive, negative, margin)
        }
        (EmbeddingLossType::InfoNce | EmbeddingLossType::Mnrl, [a, b]) => {
            contrastive_loss::info_nce_loss(a, b, temperature)
        }
        (EmbeddingLossType::CoSent, [a, b]) => {
            contrastive_loss::cosent_loss(a, b, labels, temperature)
        }
        (EmbeddingLossType::CosineSimilarity, [a, b]) => {
            contrastive_loss::cosine_similarity_loss(a, b, labels)
        }
        (EmbeddingLossType::Triplet, _) => Err(Exception::custom(
            "the triplet loss needs anchor, positive and negative texts",
        )),
        _ => Err(Exception::custom("unexpected number of texts per example")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_data::EmbeddingPair;

    /// A WordPiece tokenizer whose post-processor frames text as
    /// `[CLS] text [SEP]`, as a BERT encoder's does.
    fn bert_tokenizer() -> Tokenizer {
        let json = r###"{
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "[PAD]", "special": true, "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false},
                {"id": 1, "content": "[UNK]", "special": true, "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false},
                {"id": 2, "content": "[CLS]", "special": true, "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false},
                {"id": 3, "content": "[SEP]", "special": true, "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false}
            ],
            "normalizer": {"type": "Lowercase"},
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
                ],
                "pair": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
                ],
                "special_tokens": {
                    "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                    "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
                }
            },
            "model": {
                "type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                          "one": 4, "two": 5, "three": 6, "four": 7,
                          "red": 8, "blue": 9, "green": 10, "cat": 11, "dog": 12, "bird": 13}
            }
        }"###;
        Tokenizer::from_bytes(json.as_bytes()).expect("tokenizer parses")
    }

    /// An encoder reads `[CLS] ... [SEP]`; rows built without them train a
    /// model on text it is never shown at inference. A long text is cut inside
    /// the frame and keeps its `[SEP]`.
    #[test]
    fn rows_are_framed_by_the_special_tokens_and_cut_inside_them() {
        let tokenizer = bert_tokenizer();
        assert_eq!(
            encode_text(&tokenizer, "one two", 16).unwrap(),
            vec![2, 4, 5, 3]
        );
        assert_eq!(
            encode_text(&tokenizer, "one two three four", 5).unwrap(),
            vec![2, 4, 5, 6, 3]
        );
        assert_eq!(encode_text(&tokenizer, "", 8).unwrap(), vec![2, 3]);
    }

    const BERT: &str = r#"{
        "model_type": "bert", "vocab_size": 16, "hidden_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64,
        "max_position_embeddings": 32, "type_vocab_size": 2, "layer_norm_eps": 1e-12,
        "hidden_act": "gelu", "pad_token_id": 0
    }"#;

    /// Pairs whose two sides are the same words: a trained encoder should put
    /// each pair together and the other pairs apart.
    fn pairs() -> EmbeddingDataset {
        let words = [
            "red cat",
            "blue dog",
            "green bird",
            "one two",
            "three four",
            "red dog",
            "blue bird",
            "green cat",
        ];
        EmbeddingDataset::Pairs(
            words
                .iter()
                .map(|w| EmbeddingPair {
                    text_a: w.to_string(),
                    text_b: w.to_string(),
                    label: None,
                })
                .collect(),
        )
    }

    fn config(epochs: usize) -> EmbeddingTrainerConfig {
        EmbeddingTrainerConfig {
            training: TrainingConfig {
                learning_rate: 3e-3,
                num_epochs: epochs,
                batch_size: 4,
                warmup_steps: 0,
                lr_scheduler: pmetal_core::LrSchedulerType::Constant,
                weight_decay: 0.0,
                ..Default::default()
            },
            max_seq_len: 8,
            ..Default::default()
        }
    }

    /// Training lowers the InfoNCE loss of both backbones: the whole encoder
    /// and LoRA adapters on it. Each starts from the same weights, so what moves is
    /// what trains.
    #[test]
    fn the_loss_falls_for_the_encoder_and_for_lora_adapters() {
        let tokenizer = bert_tokenizer();
        let dataset = pairs();

        // The whole encoder.
        pmetal_runtime::compat::random::seed(31);
        let mut encoder = EncoderModel::from_config_str(BERT).unwrap();
        let mut trainer = EmbeddingTrainer::new(config(40)).unwrap();
        let losses = trainer.run(&mut encoder, &tokenizer, &dataset).unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        assert_eq!(losses.len(), 80);
        let (first, last) = (mean(&losses[..4]), mean(&losses[losses.len() - 4..]));
        assert!(last < 0.5 * first, "encoder: {first} -> {last}");

        // LoRA adapters on the same architecture.
        pmetal_runtime::compat::random::seed(31);
        let mut adapted = AdaptedModel::attach(
            pmetal_models::DynamicModel::from_config(BERT).unwrap(),
            pmetal_core::LoraConfig {
                r: 8,
                alpha: 16.0,
                dropout: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        let before = {
            let mut params: Vec<_> = adapted.lora_parameters().into_iter().collect();
            params.sort_by(|a, b| a.0.cmp(&b.0));
            params
                .into_iter()
                .map(|(_, p)| p.as_dtype(Dtype::Float32.as_i32()))
                .collect::<Vec<_>>()
        };
        let mut trainer = EmbeddingTrainer::new(EmbeddingTrainerConfig {
            training: TrainingConfig {
                learning_rate: 1e-2,
                ..config(40).training
            },
            ..config(40)
        })
        .unwrap();
        let losses = trainer.run(&mut adapted, &tokenizer, &dataset).unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        let (first, last) = (mean(&losses[..4]), mean(&losses[losses.len() - 4..]));
        assert!(last < 0.7 * first, "adapters: {first} -> {last}");
        let after = {
            let mut params: Vec<_> = adapted.lora_parameters().into_iter().collect();
            params.sort_by(|a, b| a.0.cmp(&b.0));
            params
                .into_iter()
                .map(|(_, p)| p.as_dtype(Dtype::Float32.as_i32()))
                .collect::<Vec<_>>()
        };
        let moved = before.iter().zip(&after).any(|(b, a)| {
            let d = a.subtract(b).abs().max(None);
            d.eval();
            d.item_f32() > 1e-6
        });
        assert!(moved, "no adapter moved");
    }

    fn mean(v: &[f32]) -> f32 {
        v.iter().sum::<f32>() / v.len() as f32
    }

    /// The learning rate follows the schedule (warmup then decay) rather than
    /// the constant peak, and the gradient norm is reported.
    #[test]
    fn the_schedule_and_clipping_are_applied() {
        type Seen = std::sync::Arc<std::sync::Mutex<Vec<(f64, Option<f64>)>>>;
        struct Collect(Seen);
        impl TrainingCallback for Collect {
            fn on_step_end_with_metrics(&mut self, m: &pmetal_core::StepMetrics) {
                self.0.lock().unwrap().push((m.lr, m.grad_norm));
            }
        }
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut cfg = config(2);
        cfg.training.lr_scheduler = pmetal_core::LrSchedulerType::Linear;
        cfg.training.warmup_steps = 2;
        let mut trainer = EmbeddingTrainer::new(cfg).unwrap();
        trainer.add_callback(Box::new(Collect(seen.clone())));
        pmetal_runtime::compat::random::seed(31);
        let mut encoder = EncoderModel::from_config_str(BERT).unwrap();
        trainer
            .run(&mut encoder, &bert_tokenizer(), &pairs())
            .unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 4);
        let lrs: Vec<f64> = seen.iter().map(|s| s.0).collect();
        assert!(lrs[0] < lrs[1], "warmup: {lrs:?}");
        assert!(lrs[3] < lrs[2], "decay: {lrs:?}");
        assert!(seen.iter().all(|s| s.1.is_some_and(f64::is_finite)));
    }

    /// A callback's request to stop ends the run after the step in flight,
    /// cleanly, so the command saves what was trained.
    #[test]
    fn a_stop_request_ends_the_run_after_its_step() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct StopAfter(usize, std::sync::Arc<AtomicUsize>);
        impl TrainingCallback for StopAfter {
            fn on_step_end_with_metrics(&mut self, _: &pmetal_core::StepMetrics) {
                self.1.fetch_add(1, Ordering::SeqCst);
            }
            fn should_stop(&self) -> bool {
                self.1.load(Ordering::SeqCst) >= self.0
            }
        }
        let steps = std::sync::Arc::new(AtomicUsize::new(0));
        let mut trainer = EmbeddingTrainer::new(config(3)).unwrap();
        trainer.add_callback(Box::new(StopAfter(2, steps.clone())));
        pmetal_runtime::compat::random::seed(31);
        let mut encoder = EncoderModel::from_config_str(BERT).unwrap();
        let losses = trainer
            .run(&mut encoder, &bert_tokenizer(), &pairs())
            .unwrap();
        assert_eq!(losses.len(), 2, "stopped after the second step of 3 epochs");
        assert_eq!(steps.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn the_configuration_is_validated_at_construction() {
        let with = |f: fn(&mut EmbeddingTrainerConfig)| {
            let mut c = config(1);
            f(&mut c);
            EmbeddingTrainer::new(c)
        };
        assert!(with(|_| {}).is_ok());
        assert!(with(|c| c.temperature = 0.0).is_err());
        assert!(with(|c| c.training.batch_size = 1).is_err());
        assert!(with(|c| c.max_seq_len = 1).is_err());
        assert!(
            with(|c| {
                c.loss_type = EmbeddingLossType::CoSent;
                c.normalize = false;
            })
            .is_err()
        );
        // A triplet loss on pair data is refused when it runs.
        let mut c = config(1);
        c.loss_type = EmbeddingLossType::Triplet;
        let mut trainer = EmbeddingTrainer::new(c).unwrap();
        let mut encoder = EncoderModel::from_config_str(BERT).unwrap();
        assert!(
            trainer
                .run(&mut encoder, &bert_tokenizer(), &pairs())
                .is_err()
        );
    }
}
