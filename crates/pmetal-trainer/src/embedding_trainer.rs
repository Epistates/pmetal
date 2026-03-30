//! Embedding / sentence-transformer training loop.
//!
//! Trains encoder models (BERT-family and decoder-only models used as embedders)
//! for producing high-quality sentence embeddings using contrastive learning
//! objectives.
//!
//! ## Supported losses
//!
//! | Loss | Data format | Best for |
//! |------|-------------|----------|
//! | `InfoNce` / `Mnrl` | pairs | Large batches, no explicit labels |
//! | `Triplet` | triplets | Hard negative mining workflows |
//! | `CoSent` | pairs with labels | Similarity regression + ranking |
//! | `CosineSimilarity` | pairs with labels | Direct similarity regression |
//!
//! ## Usage
//!
//! ```ignore
//! use pmetal_trainer::embedding_trainer::{EmbeddingTrainer, EmbeddingTrainerConfig};
//! use pmetal_data::EmbeddingDataset;
//!
//! let dataset = EmbeddingDataset::from_jsonl("data/pairs.jsonl")?;
//! let config = EmbeddingTrainerConfig::default();
//! let mut trainer = EmbeddingTrainer::new(config);
//!
//! // bert_model: BertForEmbedding, tokenizer: Tokenizer, optimizer: AdamW
//! trainer.run(&mut bert_model, &tokenizer, &dataset, &mut optimizer)?;
//! ```

use std::collections::HashMap;

use pmetal_bridge::compat::{
    Array, Exception, eval_params, module::ModuleParameters, nn, optimizers::Optimizer,
};
use pmetal_core::{EvalMetrics, TrainingCallback, TrainingConfig};
use pmetal_data::{EmbeddingDataset, EmbeddingPair, EmbeddingTriplet, Tokenizer};
use pmetal_lora::TrainableModel;
use pmetal_models::pooling::{PoolingMode, normalize_embeddings, pool};

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
    /// CoSENT (circle loss variant). Works well with binary or continuous labels.
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
    /// Core training hyperparameters (learning rate, batch size, epochs, etc.).
    #[serde(default)]
    pub training: TrainingConfig,
    /// Loss function.
    #[serde(default)]
    pub loss_type: EmbeddingLossType,
    /// Temperature for InfoNCE / CoSENT losses.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Margin for triplet loss.
    #[serde(default = "default_margin")]
    pub margin: f32,
    /// Pooling strategy for extracting sentence embeddings from hidden states.
    #[serde(default)]
    pub pooling_mode: PoolingMode,
    /// Apply L2 normalisation to embeddings before computing loss.
    /// Strongly recommended for InfoNCE / CoSENT (required for cosine similarity).
    #[serde(default = "default_true")]
    pub normalize: bool,
    /// Maximum input sequence length. Sequences are truncated to this length.
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

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the embedding trainer.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingTrainerError {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type EmbeddingResult<T> = Result<T, EmbeddingTrainerError>;

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Embedding / sentence-transformer trainer.
///
/// Works with any model that implements [`TrainableModel`] — most commonly
/// [`pmetal_models::architectures::bert::BertForEmbedding`], but also
/// decoder-only models used as asymmetric embedders.
///
/// ## Causal LM Models
///
/// When using a causal (decoder-only) model as the backbone, the `forward` pass
/// returns logits `[batch, seq_len, vocab_size]` rather than hidden states
/// `[batch, seq_len, hidden_size]`.  You must wrap the model so that its
/// `TrainableModel::forward` returns the encoder hidden states, not logits —
/// for example by hooking into a penultimate layer.  Failing to do so will pass
/// logits of shape `[batch, seq, vocab]` into the pooling layer, which will pool
/// over the vocabulary dimension instead of the hidden dimension.  The resulting
/// embeddings will be numerically valid but semantically meaningless.
pub struct EmbeddingTrainer {
    /// Configuration.
    pub config: EmbeddingTrainerConfig,
    /// Current global step count.
    pub step: usize,
    /// Training callbacks.
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl EmbeddingTrainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: EmbeddingTrainerConfig) -> Self {
        Self {
            config,
            step: 0,
            callbacks: Vec::new(),
        }
    }

    /// Register a training callback.
    pub fn add_callback(&mut self, cb: Box<dyn TrainingCallback>) {
        self.callbacks.push(cb);
    }

    // -----------------------------------------------------------------------
    // Public entry points
    // -----------------------------------------------------------------------

    /// Run embedding training on a dataset.
    ///
    /// Dispatches to `run_pairs` or `run_triplets` based on the dataset variant.
    pub fn run<M, O>(
        &mut self,
        model: &mut M,
        tokenizer: &Tokenizer,
        dataset: &EmbeddingDataset,
        optimizer: &mut O,
    ) -> EmbeddingResult<()>
    where
        M: TrainableModel,
        O: Optimizer,
    {
        match dataset {
            EmbeddingDataset::Pairs(pairs) => self.run_pairs(model, tokenizer, pairs, optimizer),
            EmbeddingDataset::Triplets(triplets) => {
                self.run_triplets(model, tokenizer, triplets, optimizer)
            }
        }
    }

    /// Run pair-based embedding training (InfoNCE / CoSENT / cosine-similarity).
    pub fn run_pairs<M, O>(
        &mut self,
        model: &mut M,
        tokenizer: &Tokenizer,
        pairs: &[EmbeddingPair],
        optimizer: &mut O,
    ) -> EmbeddingResult<()>
    where
        M: TrainableModel,
        O: Optimizer,
    {
        if pairs.is_empty() {
            return Ok(());
        }

        let batch_size = self.config.training.batch_size;
        let n_epochs = self.config.training.num_epochs;
        let max_len = self.config.max_seq_len;

        self.fire_train_start();
        tracing::info!(
            "Embedding training (pairs): {} examples, batch={}, epochs={}, loss={}",
            pairs.len(),
            batch_size,
            n_epochs,
            self.config.loss_type,
        );

        // Mutable shuffle buffer
        let mut indices: Vec<usize> = (0..pairs.len()).collect();

        for epoch in 0..n_epochs {
            // Shuffle order for this epoch
            if self.config.shuffle {
                use rand::SeedableRng;
                use rand::seq::SliceRandom;
                let mut rng =
                    rand::rngs::StdRng::seed_from_u64(self.config.seed.wrapping_add(epoch as u64));
                indices.shuffle(&mut rng);
            }

            self.fire_epoch_start(epoch);

            let n_batches = pairs.len().div_ceil(batch_size);
            let mut epoch_loss = 0.0f64;

            for batch_idx in 0..n_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(pairs.len());
                let batch_indices = &indices[start..end];
                let batch: Vec<&EmbeddingPair> = batch_indices.iter().map(|&i| &pairs[i]).collect();

                let texts_a: Vec<&str> = batch.iter().map(|p| p.text_a.as_str()).collect();
                let texts_b: Vec<&str> = batch.iter().map(|p| p.text_b.as_str()).collect();

                // Tokenize both sides (outside the autograd closure)
                let (ids_a, mask_a) = self.tokenize_batch(tokenizer, &texts_a, max_len)?;
                let (ids_b, mask_b) = self.tokenize_batch(tokenizer, &texts_b, max_len)?;

                // Optional label array for CoSENT / cosine-similarity losses
                let label_data: Vec<f32> = batch.iter().map(|p| p.label.unwrap_or(1.0)).collect();
                let labels = Array::from_slice(&label_data, &[batch.len() as i32]);

                let loss_type = self.config.loss_type;
                let temperature = self.config.temperature;
                let pooling_mode = self.config.pooling_mode;
                let do_normalize = self.config.normalize;

                let loss_fn = |model: &mut M,
                               (ids_a, mask_a, ids_b, mask_b, labels): (
                    &Array,
                    &Array,
                    &Array,
                    &Array,
                    &Array,
                )|
                 -> Result<Array, Exception> {
                    let emb_a = encode_inner(model, ids_a, mask_a, pooling_mode, do_normalize)?;
                    let emb_b = encode_inner(model, ids_b, mask_b, pooling_mode, do_normalize)?;
                    compute_pair_loss(&emb_a, &emb_b, labels, loss_type, temperature)
                };

                let mut loss_and_grad = nn::value_and_grad(loss_fn);
                let (mut loss, grads) =
                    loss_and_grad(model, (&ids_a, &mask_a, &ids_b, &mask_b, &labels))
                        .map_err(EmbeddingTrainerError::Mlx)?;

                optimizer
                    .update(model, grads)
                    .map_err(EmbeddingTrainerError::Mlx)?;
                eval_params(model.trainable_parameters()).map_err(EmbeddingTrainerError::Mlx)?;

                let loss_val: f32 = loss.item();
                epoch_loss += loss_val as f64;
                self.step += 1;

                if self.step % self.config.log_every == 0 {
                    tracing::info!(
                        "step={} loss={:.4} epoch={}/{}",
                        self.step,
                        loss_val,
                        epoch + 1,
                        n_epochs
                    );
                }

                self.fire_step_end(self.step, loss_val as f64);
            }

            let avg_loss = epoch_loss / n_batches as f64;
            tracing::info!("epoch={}/{} avg_loss={:.4}", epoch + 1, n_epochs, avg_loss);
            self.fire_epoch_end(epoch, avg_loss as f32);
        }

        self.fire_train_end();
        Ok(())
    }

    /// Run triplet-based embedding training.
    pub fn run_triplets<M, O>(
        &mut self,
        model: &mut M,
        tokenizer: &Tokenizer,
        triplets: &[EmbeddingTriplet],
        optimizer: &mut O,
    ) -> EmbeddingResult<()>
    where
        M: TrainableModel,
        O: Optimizer,
    {
        if triplets.is_empty() {
            return Ok(());
        }

        let batch_size = self.config.training.batch_size;
        let n_epochs = self.config.training.num_epochs;
        let max_len = self.config.max_seq_len;
        let margin = self.config.margin;
        let pooling_mode = self.config.pooling_mode;
        let do_normalize = self.config.normalize;

        self.fire_train_start();
        tracing::info!(
            "Embedding training (triplets): {} examples, batch={}, epochs={}, margin={}",
            triplets.len(),
            batch_size,
            n_epochs,
            margin,
        );

        let mut indices: Vec<usize> = (0..triplets.len()).collect();

        for epoch in 0..n_epochs {
            if self.config.shuffle {
                use rand::SeedableRng;
                use rand::seq::SliceRandom;
                let mut rng =
                    rand::rngs::StdRng::seed_from_u64(self.config.seed.wrapping_add(epoch as u64));
                indices.shuffle(&mut rng);
            }

            self.fire_epoch_start(epoch);

            let n_batches = triplets.len().div_ceil(batch_size);
            let mut epoch_loss = 0.0f64;

            for batch_idx in 0..n_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(triplets.len());
                let batch_indices = &indices[start..end];
                let batch: Vec<&EmbeddingTriplet> =
                    batch_indices.iter().map(|&i| &triplets[i]).collect();

                let anchors: Vec<&str> = batch.iter().map(|t| t.anchor.as_str()).collect();
                let positives: Vec<&str> = batch.iter().map(|t| t.positive.as_str()).collect();
                let negatives: Vec<&str> = batch.iter().map(|t| t.negative.as_str()).collect();

                let (ids_a, mask_a) = self.tokenize_batch(tokenizer, &anchors, max_len)?;
                let (ids_p, mask_p) = self.tokenize_batch(tokenizer, &positives, max_len)?;
                let (ids_n, mask_n) = self.tokenize_batch(tokenizer, &negatives, max_len)?;

                let loss_fn = |model: &mut M,
                               (ids_a, mask_a, ids_p, mask_p, ids_n, mask_n): (
                    &Array,
                    &Array,
                    &Array,
                    &Array,
                    &Array,
                    &Array,
                )|
                 -> Result<Array, Exception> {
                    let emb_a = encode_inner(model, ids_a, mask_a, pooling_mode, do_normalize)?;
                    let emb_p = encode_inner(model, ids_p, mask_p, pooling_mode, do_normalize)?;
                    let emb_n = encode_inner(model, ids_n, mask_n, pooling_mode, do_normalize)?;
                    contrastive_loss::triplet_loss(&emb_a, &emb_p, &emb_n, margin)
                };

                let mut loss_and_grad = nn::value_and_grad(loss_fn);
                let (mut loss, grads) =
                    loss_and_grad(model, (&ids_a, &mask_a, &ids_p, &mask_p, &ids_n, &mask_n))
                        .map_err(EmbeddingTrainerError::Mlx)?;

                optimizer
                    .update(model, grads)
                    .map_err(EmbeddingTrainerError::Mlx)?;
                eval_params(model.trainable_parameters()).map_err(EmbeddingTrainerError::Mlx)?;

                let loss_val: f32 = loss.item();
                epoch_loss += loss_val as f64;
                self.step += 1;

                if self.step % self.config.log_every == 0 {
                    tracing::info!(
                        "step={} loss={:.4} epoch={}/{}",
                        self.step,
                        loss_val,
                        epoch + 1,
                        n_epochs
                    );
                }

                self.fire_step_end(self.step, loss_val as f64);
            }

            let avg_loss = epoch_loss / n_batches as f64;
            tracing::info!("epoch={}/{} avg_loss={:.4}", epoch + 1, n_epochs, avg_loss);
            self.fire_epoch_end(epoch, avg_loss as f32);
        }

        self.fire_train_end();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Inference / encoding (no gradient)
    // -----------------------------------------------------------------------

    /// Encode a batch of texts into sentence embeddings (no gradient tracking).
    ///
    /// Useful for evaluation and similarity search after training.
    ///
    /// ## Warning: Causal LM output shape
    ///
    /// This method calls `model.forward(input_ids, attention_mask)` and feeds the
    /// result directly into the pooling layer.  If the model is a causal LM whose
    /// `forward` returns logits `[batch, seq, vocab]` instead of hidden states
    /// `[batch, seq, hidden]`, pooling will operate over the vocabulary dimension,
    /// producing embeddings of size `vocab_size` instead of `hidden_size`.
    /// Wrap causal models to return hidden states before using this method.
    pub fn encode<M: TrainableModel>(
        &self,
        model: &mut M,
        tokenizer: &Tokenizer,
        texts: &[&str],
    ) -> EmbeddingResult<Array> {
        let max_len = self.config.max_seq_len;
        let (ids, mask) = self.tokenize_batch(tokenizer, texts, max_len)?;
        let hidden = model
            .forward(&ids, Some(&mask))
            .map_err(|e| EmbeddingTrainerError::Mlx(Exception::custom(e.to_string())))?;
        let emb =
            pool(&hidden, &mask, self.config.pooling_mode).map_err(EmbeddingTrainerError::Mlx)?;
        if self.config.normalize {
            Ok(normalize_embeddings(&emb).map_err(EmbeddingTrainerError::Mlx)?)
        } else {
            Ok(emb)
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Tokenize a batch of text strings and return padded `(input_ids, attention_mask)`.
    fn tokenize_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        max_len: usize,
    ) -> EmbeddingResult<(Array, Array)> {
        let pad_id = tokenizer.pad_token_id().unwrap_or(0) as i32;
        let mut all_ids: Vec<Vec<i32>> = Vec::with_capacity(texts.len());
        let mut actual_max_len = 0usize;

        for text in texts {
            let ids = tokenizer
                .encode(text)
                .map_err(|e| EmbeddingTrainerError::Tokenizer(e.to_string()))?;
            let len = ids.len().min(max_len);
            actual_max_len = actual_max_len.max(len);
            all_ids.push(ids[..len].iter().map(|&x| x as i32).collect());
        }

        let batch = texts.len();
        let seq = actual_max_len;
        let mut flat_ids = vec![pad_id; batch * seq];
        let mut flat_mask = vec![0i32; batch * seq];

        for (i, ids) in all_ids.iter().enumerate() {
            for (j, &id) in ids.iter().enumerate() {
                flat_ids[i * seq + j] = id;
                flat_mask[i * seq + j] = 1;
            }
        }

        let input_ids = Array::from_slice(&flat_ids, &[batch as i32, seq as i32]);
        let attention_mask = Array::from_slice(&flat_mask, &[batch as i32, seq as i32]);
        Ok((input_ids, attention_mask))
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

    fn fire_step_end(&mut self, step: usize, loss: f64) {
        for cb in &mut self.callbacks {
            cb.on_step_end(step, loss);
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions used inside autograd closures (must be stateless)
// ---------------------------------------------------------------------------

/// Forward pass → pool → optional L2-normalise.
///
/// This function is designed to be called **inside** a `value_and_grad` closure.
/// It is generic over `M: TrainableModel` and does not capture `self`.
fn encode_inner<M: TrainableModel>(
    model: &mut M,
    input_ids: &Array,
    attention_mask: &Array,
    pooling_mode: PoolingMode,
    normalize: bool,
) -> Result<Array, Exception> {
    let hidden = model
        .forward(input_ids, Some(attention_mask))
        .map_err(|e| Exception::custom(e.to_string()))?;
    let emb = pool(&hidden, attention_mask, pooling_mode)?;
    if normalize {
        normalize_embeddings(&emb)
    } else {
        Ok(emb)
    }
}

/// Compute the contrastive loss for a batch of pairs.
fn compute_pair_loss(
    emb_a: &Array,
    emb_b: &Array,
    labels: &Array,
    loss_type: EmbeddingLossType,
    temperature: f32,
) -> Result<Array, Exception> {
    match loss_type {
        EmbeddingLossType::InfoNce | EmbeddingLossType::Mnrl => {
            contrastive_loss::info_nce_loss(emb_a, emb_b, temperature)
        }
        EmbeddingLossType::CoSent => {
            contrastive_loss::cosent_loss(emb_a, emb_b, labels, temperature)
        }
        EmbeddingLossType::CosineSimilarity => {
            contrastive_loss::cosine_similarity_loss(emb_a, emb_b, labels)
        }
        EmbeddingLossType::Triplet => {
            // Pair data has no negatives — fall back to InfoNCE which uses
            // in-batch negatives from other positive pairs.
            contrastive_loss::info_nce_loss(emb_a, emb_b, temperature)
        }
    }
}
