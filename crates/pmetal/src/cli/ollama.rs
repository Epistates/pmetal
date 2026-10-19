//! Clap types for `pmetal ollama`.

use clap::{Subcommand, ValueEnum};

/// Modelfile export helpers for external Ollama registration.
#[derive(Subcommand, Debug)]
pub enum OllamaAction {
    /// Generate a Modelfile for a trained model
    Modelfile {
        /// Base model (GGUF path or Ollama model name)
        #[arg(short, long)]
        base: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Output Modelfile path
        #[arg(short, long, default_value = "Modelfile")]
        output: String,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long)]
        temperature: Option<f32>,

        /// Context window size
        #[arg(long)]
        num_ctx: Option<i32>,

        /// Top-k sampling
        #[arg(long)]
        top_k: Option<i32>,

        /// Top-p nucleus sampling
        #[arg(long)]
        top_p: Option<f32>,

        /// Model template (auto-detected from architecture if not specified)
        #[arg(long, value_enum)]
        template: Option<OllamaTemplate>,

        /// License text for the model
        #[arg(long)]
        license: Option<String>,
    },

    /// Generate a Modelfile for later registration with Ollama
    Create {
        /// Model name for Ollama (e.g., "my-finetuned-model")
        #[arg(short, long)]
        name: String,

        /// Base model (GGUF path or Ollama model name)
        #[arg(short, long)]
        base: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long)]
        temperature: Option<f32>,

        /// Context window size
        #[arg(long)]
        num_ctx: Option<i32>,

        /// Output Modelfile path
        #[arg(short, long, default_value = "Modelfile")]
        output: String,

        /// Model template (auto-detected from architecture if not specified)
        #[arg(long, value_enum)]
        template: Option<OllamaTemplate>,
    },

    /// List available templates
    Templates,
}

/// Ollama template presets.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OllamaTemplate {
    /// Llama 3 chat format
    Llama3,
    /// Qwen3/ChatML format
    Qwen3,
    /// Gemma instruct format
    Gemma,
    /// Mistral instruct format
    Mistral,
    /// Phi-3 instruct format
    Phi3,
    /// DeepSeek chat format
    DeepSeek,
}
