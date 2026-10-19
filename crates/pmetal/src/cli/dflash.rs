//! Clap argument struct for `pmetal dflash`.

use clap::Args;
use pmetal_core::jobs::DflashSpec;

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

    /// Sampling temperature (0 = greedy). The draft only changes the speed:
    /// the tokens are the target's own, greedy or drawn at this temperature.
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

    /// Also decode the prompt greedily, one token per forward with the same
    /// target, and report its speed and whether DFlash's tokens match it.
    #[arg(long = "compare-greedy")]
    pub compare_greedy: bool,
}

impl From<DflashArgs> for DflashSpec {
    fn from(a: DflashArgs) -> Self {
        DflashSpec {
            target: a.target,
            draft: a.draft,
            prompt: a.prompt,
            max_new_tokens: a.max_new_tokens,
            temperature: a.temperature,
            speculative_tokens: a.speculative_tokens,
            draft_fp8: a.draft_fp8,
            json: a.json,
            no_chat: a.no_chat,
            tree_budget: a.tree_budget,
            compare_greedy: a.compare_greedy,
        }
    }
}
