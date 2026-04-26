//! Clap argument struct for `pmetal embed-train`.

use clap::Args;

/// Thin clap argument struct for `pmetal embed-train`.
#[derive(Args, Debug)]
pub struct EmbedTrainArgs {
    /// Path to the BERT / encoder model directory.
    #[arg(short, long = "model")]
    pub model: String,

    /// Path to the training dataset (JSONL pairs or triplets).
    #[arg(short, long = "dataset")]
    pub dataset: String,

    /// Output directory for trained model weights.
    #[arg(short, long = "output", default_value = "./output-embed")]
    pub output: String,

    /// Contrastive loss function.
    #[arg(long = "loss", default_value = "info_nce")]
    pub loss: String,

    /// Pooling strategy for sentence embeddings.
    #[arg(long = "pooling", default_value = "mean")]
    pub pooling: String,

    /// Temperature for InfoNCE / CoSENT losses.
    #[arg(long = "temperature", default_value = "0.05")]
    pub temperature: f32,

    /// Margin for triplet loss.
    #[arg(long = "margin", default_value = "0.3")]
    pub margin: f32,

    /// Learning rate.
    #[arg(long = "learning-rate", default_value = "2e-5")]
    pub learning_rate: f64,

    /// Training batch size.
    #[arg(long = "batch-size", default_value = "32")]
    pub batch_size: usize,

    /// Number of training epochs.
    #[arg(long = "epochs", default_value = "3")]
    pub epochs: usize,

    /// Maximum input sequence length.
    #[arg(long = "max-seq-len", default_value = "512")]
    pub max_seq_len: usize,

    /// AdamW weight decay.
    #[arg(long = "weight-decay", default_value = "0.01")]
    pub weight_decay: f64,

    /// Disable L2 normalisation of embeddings before loss.
    #[arg(long = "no-normalize")]
    pub no_normalize: bool,

    /// Log training progress every N steps.
    #[arg(long = "log-every", default_value = "10")]
    pub log_every: usize,

    /// Random seed for dataset shuffling.
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,
}
