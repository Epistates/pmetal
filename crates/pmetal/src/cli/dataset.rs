//! Clap types for `pmetal dataset`.

use clap::{Subcommand, ValueEnum};

/// Dataset subcommands for data preparation.
///
/// Every command that reads a dataset reads JSONL, JSON, Parquet, Arrow, CSV
/// or TSV, a file or a directory (its train split), and a row of any layout
/// `pmetal dataset formats` lists.
#[derive(Subcommand, Debug)]
pub enum DatasetAction {
    /// Analyze a dataset and show statistics
    Analyze {
        /// Dataset file or directory
        #[arg(short, long)]
        path: String,

        /// Model ID or path whose tokenizer counts tokens and whose chat
        /// template renders conversations, so the lengths are what training
        /// sees
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

    /// Convert a dataset to JSONL
    Convert {
        /// Input file or directory (Parquet, JSON, JSONL, Arrow, CSV, TSV)
        #[arg(short, long)]
        input: String,

        /// Output file path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Rename columns: "target=source,target=source" (e.g., "text=content,prompt=instruction"
        /// takes `text` from the column `content`)
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
        /// Dataset file or directory
        #[arg(short, long)]
        path: String,

        /// Model ID or path whose tokenizer (and chat template) measures the
        /// samples
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum sequence length to check against
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,
    },

    /// Preview the first N samples of a dataset
    Preview {
        /// HuggingFace dataset ID (e.g., "tatsu-lab/alpaca") or a local path
        dataset_id: String,

        /// Dataset split (e.g., "train", "test"); for a HuggingFace dataset
        #[arg(long, default_value = "train")]
        split: String,

        /// Number of samples to preview
        #[arg(short, long, default_value = "5")]
        num: usize,
    },

    /// Filter a dataset by various criteria
    Filter {
        /// Input dataset (file or directory)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Model ID or path for token-based filtering
        #[arg(short, long)]
        model: Option<String>,

        /// Minimum token count (requires --model)
        #[arg(long)]
        min_tokens: Option<usize>,

        /// Maximum token count (requires --model)
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Remove duplicate samples (exact match of their text)
        #[arg(long)]
        dedup: bool,

        /// Regex pattern to match against the sample's text (keeps matching
        /// samples; a conversation reads as `Role: content` blocks)
        #[arg(long)]
        pattern: Option<String>,

        /// Invert pattern matching (keeps non-matching samples)
        #[arg(long)]
        invert: bool,

        /// Keep only conversations that have both a user and an assistant turn
        #[arg(long)]
        complete_only: bool,
    },

    /// Split a dataset into train/validation/test sets
    Split {
        /// Input dataset (file or directory)
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

        /// Keep the proportions of a column's values in every split (e.g.,
        /// "category", "difficulty")
        #[arg(long)]
        stratify: Option<String>,
    },

    /// Merge multiple datasets into one
    Merge {
        /// Input datasets (files or directories)
        #[arg(short, long, num_args = 1..)]
        inputs: Vec<String>,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Shuffle after merging
        #[arg(long)]
        shuffle: bool,

        /// Random seed for shuffling and weighted sampling
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Interleave samples from each dataset (alternating)
        #[arg(long)]
        interleave: bool,

        /// Weights for each input dataset (comma-separated, one per input,
        /// e.g., "1.0,2.0,0.5"): 2.0 repeats a dataset, 0.5 keeps half of it
        #[arg(long)]
        weights: Option<String>,
    },

    /// Take a random sample from a dataset
    Sample {
        /// Input dataset (file or directory)
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
        /// Input dataset (file or directory)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL with formatted text)
        #[arg(short, long)]
        output: String,

        /// A built-in chat template, in place of the model's own. Without it
        /// the template of --model is used, and chatml without a model
        #[arg(long, value_enum)]
        template: Option<TemplateName>,

        /// System message for the conversations that have none
        #[arg(long)]
        system: Option<String>,

        /// Model ID or path whose chat template formats the conversations
        #[arg(short, long)]
        model: Option<String>,

        /// Write each conversation only up to the generation prompt of its
        /// last assistant turn, as a model is prompted to answer it
        #[arg(long)]
        add_generation_prompt: bool,

        /// Write each sample as a prompt and a completion (the last assistant
        /// turn), so a trainer that reads such rows trains on the completion
        /// only
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

        /// Model ID or path whose tokenizer measures the samples and whose
        /// chat template formats them
        #[arg(short, long)]
        model: String,

        /// A built-in chat template, in place of the model's own
        #[arg(long, value_enum)]
        template: Option<TemplateName>,

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

/// A built-in chat template, for conversation formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TemplateName {
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
    /// No chat template: `Role: content` blocks
    Raw,
    /// The model's own chat template (needs --model)
    Auto,
}
