//! Clap argument struct for `pmetal dflash`.

use clap::Args;

/// Thin clap argument struct for `pmetal dflash`.
#[derive(Args, Debug)]
pub struct DflashArgs {
    /// Target Qwen3 model (HF id or local path).
    #[arg(long = "target")]
    pub target: String,

    /// DFlash draft model (HF id or local path, e.g. `z-lab/Qwen3-4B-DFlash-b16`).
    #[arg(long = "draft")]
    pub draft: String,

    /// Input prompt.
    #[arg(short, long = "prompt")]
    pub prompt: String,

    /// Maximum tokens to generate after the prompt.
    #[arg(long = "max-new-tokens", default_value = "128")]
    pub max_new_tokens: usize,

    /// Sampling temperature (0 = greedy, bit-identical to baseline).
    #[arg(long = "temperature", default_value = "0.0")]
    pub temperature: f32,

    /// Override the draft block size (defaults to the draft checkpoint's `block_size`).
    #[arg(long = "speculative-tokens")]
    pub speculative_tokens: Option<usize>,

    /// Quantize the draft's Linear weights to FP8 (E4M3) on load.
    #[arg(long = "draft-fp8")]
    pub draft_fp8: bool,

    /// Emit a JSON report of tokens + metrics instead of plain text.
    #[arg(long = "json")]
    pub json: bool,

    /// Skip the model's chat template and tokenize the prompt verbatim.
    #[arg(long = "no-chat")]
    pub no_chat: bool,

    /// Tree-verify budget.
    #[arg(long = "tree-budget", default_value = "0")]
    pub tree_budget: usize,
}
