//! Clap types for `pmetal dataset`.

use clap::{Subcommand, ValueEnum};

/// Dataset subcommands for data preparation.
#[derive(Subcommand, Debug)]
pub enum DatasetAction {
    /// Analyze a dataset and show statistics
    Analyze {
        /// Path to dataset file (JSONL)
        #[arg(short, long)]
        path: String,

        /// Model ID for tokenization (required for accurate token counts)
        #[arg(short, long)]
        model: Option<String>,

        /// Show detailed per-sample statistics
        #[arg(long)]
        detailed: bool,
    },

    /// Download a dataset from HuggingFace Hub
    Download {
        /// Dataset ID (e.g., "tatsu-lab/alpaca", "TeichAI/gemini-3-pro-preview-high-reasoning-1000x")
        dataset_id: String,

        /// Dataset split to download (e.g., "train", "test")
        #[arg(long, default_value = "train")]
        split: String,

        /// Output path for converted JSONL file
        #[arg(short, long)]
        output: Option<String>,

        /// Specific revision/branch
        #[arg(long)]
        revision: Option<String>,
    },

    /// Convert a dataset to pmetal-compatible JSONL format
    Convert {
        /// Input file path (Parquet, JSON, JSONL, CSV)
        #[arg(short, long)]
        input: String,

        /// Output file path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Input format (auto-detected if not specified)
        #[arg(long, value_enum)]
        format: Option<InputFormat>,

        /// Column mapping for non-standard formats (e.g., "text=content,prompt=instruction")
        #[arg(long)]
        columns: Option<String>,

        /// Shuffle the output data
        #[arg(long)]
        shuffle: bool,

        /// Random seed for shuffling
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Validate a dataset for training
    Validate {
        /// Path to dataset file (JSONL)
        #[arg(short, long)]
        path: String,

        /// Model ID for tokenization validation
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum sequence length to check against
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,
    },

    /// Preview first N samples from a HuggingFace dataset
    Preview {
        /// Dataset ID (e.g., "tatsu-lab/alpaca")
        dataset_id: String,

        /// Dataset split (e.g., "train", "test")
        #[arg(long, default_value = "train")]
        split: String,

        /// Number of samples to preview
        #[arg(short, long, default_value = "5")]
        num: usize,
    },

    /// Filter a dataset by various criteria
    Filter {
        /// Input dataset path (JSONL)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Model ID for token-based filtering
        #[arg(short, long)]
        model: Option<String>,

        /// Minimum token count (requires --model)
        #[arg(long)]
        min_tokens: Option<usize>,

        /// Maximum token count (requires --model)
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Remove duplicate samples (exact match)
        #[arg(long)]
        dedup: bool,

        /// Regex pattern to match (keeps matching samples)
        #[arg(long)]
        pattern: Option<String>,

        /// Invert pattern matching (keeps non-matching samples)
        #[arg(long)]
        invert: bool,

        /// Require all conversation turns (filters incomplete conversations)
        #[arg(long)]
        complete_only: bool,
    },

    /// Split a dataset into train/validation/test sets
    Split {
        /// Input dataset path (JSONL)
        #[arg(short, long)]
        input: String,

        /// Output directory for split files
        #[arg(short, long)]
        output_dir: String,

        /// Validation set ratio (0.0 to 1.0)
        #[arg(long, default_value = "0.1")]
        val_ratio: f64,

        /// Test set ratio (0.0 to 1.0)
        #[arg(long, default_value = "0.0")]
        test_ratio: f64,

        /// Random seed for splitting
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Stratify by a field (e.g., "category", "difficulty")
        #[arg(long)]
        stratify: Option<String>,
    },

    /// Merge multiple datasets into one
    Merge {
        /// Input dataset paths (JSONL)
        #[arg(short, long, num_args = 1..)]
        inputs: Vec<String>,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Shuffle after merging
        #[arg(long)]
        shuffle: bool,

        /// Random seed for shuffling
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Interleave samples from each dataset (alternating)
        #[arg(long)]
        interleave: bool,

        /// Weights for each input dataset (comma-separated, e.g., "1.0,2.0,0.5")
        #[arg(long)]
        weights: Option<String>,
    },

    /// Take a random sample from a dataset
    Sample {
        /// Input dataset path (JSONL)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Number of samples to take
        #[arg(short, long)]
        num: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Apply a chat template to format conversations
    Template {
        /// Input dataset path (JSONL with conversations)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL with formatted text)
        #[arg(short, long)]
        output: String,

        /// Chat template to use
        #[arg(long, value_enum, default_value = "chatml")]
        template: ChatTemplate,

        /// Custom system message (overrides default)
        #[arg(long)]
        system: Option<String>,

        /// Model ID for tokenizer-based template (uses tokenizer's chat_template)
        #[arg(short, long)]
        model: Option<String>,

        /// Add generation prompt marker at end
        #[arg(long)]
        add_generation_prompt: bool,

        /// Only mask prompt tokens in labels (for SFT)
        #[arg(long)]
        mask_prompt: bool,
    },

    /// Prepare a dataset for training (full pipeline)
    Prepare {
        /// HuggingFace dataset ID or local path
        dataset: String,

        /// Output directory
        #[arg(short, long)]
        output_dir: String,

        /// Model ID for tokenization
        #[arg(short, long)]
        model: String,

        /// Chat template to apply
        #[arg(long, value_enum, default_value = "chatml")]
        template: ChatTemplate,

        /// Maximum sequence length (filters longer samples)
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,

        /// Validation split ratio
        #[arg(long, default_value = "0.05")]
        val_ratio: f64,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Skip deduplication
        #[arg(long)]
        no_dedup: bool,

        /// Column mapping: remap source columns to standard names.
        /// Format: "target=source,target=source" (e.g., "instruction=problem,output=solution")
        #[arg(long)]
        columns: Option<String>,
    },

    /// Show supported formats and templates
    Formats,
}

/// Chat template formats for conversation formatting.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ChatTemplate {
    /// ChatML format: <|im_start|>role\ncontent<|im_end|>
    Chatml,
    /// Llama 3 format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
    Llama3,
    /// Llama 2 format: [INST] user [/INST] assistant </s>
    Llama2,
    /// Mistral/Zephyr format: <s>[INST] user [/INST] assistant</s>
    Mistral,
    /// Qwen format: <|im_start|>role\ncontent<|im_end|>
    Qwen,
    /// Phi format: <|user|>\ncontent<|end|>\n<|assistant|>\ncontent<|end|>
    Phi,
    /// Gemma format: <start_of_turn>role\ncontent<end_of_turn>
    Gemma,
    /// Raw text (no template, just concatenate)
    Raw,
    /// Use tokenizer's built-in chat template
    Auto,
}

/// Input format for dataset conversion.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum InputFormat {
    /// Parquet format
    Parquet,
    /// JSON array format
    Json,
    /// JSONL (JSON Lines) format
    Jsonl,
    /// CSV format
    Csv,
    /// ShareGPT conversation format
    ShareGpt,
    /// Alpaca instruction format
    Alpaca,
}
