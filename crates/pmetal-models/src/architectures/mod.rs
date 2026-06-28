//! Model architecture implementations.

pub mod utils;
pub use utils::create_causal_mask;

pub mod bert;
pub mod clip;
pub mod cohere;
pub mod deepseek;
pub mod dflash_draft;
pub mod flux;
pub mod gemma;
pub mod gemma4;
pub mod gpt_oss;
pub mod granite;
pub mod llama;
pub mod llama4;
pub mod mistral;
pub mod mllama;
pub mod nemotron_h;
pub mod phi;
pub mod pixtral;
pub mod qwen2;
pub mod qwen2_vl;
pub mod qwen3;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod qwen3_next_inline;
pub mod t5;
pub mod vae;
pub mod whisper;

pub use bert::*;
pub use clip::*;
pub use cohere::*;
pub use deepseek::*;
pub use dflash_draft::*;
pub use flux::*;
pub use gemma::*;
pub use gemma4::*;
pub use gpt_oss::*;
pub use granite::*;
pub use llama::*;
pub use llama4::*;
pub use mistral::*;
pub use mllama::*;
pub use nemotron_h::*;
pub use phi::*;
pub use pixtral::*;
pub use qwen2::*;
pub use qwen2_vl::*;
pub use qwen3::*;
pub use qwen3_moe::*;
pub use qwen3_next::*;
pub use t5::*;
pub use vae::*;
pub use whisper::*;

/// Summary of a [`<model>::load_weights`] call.
#[derive(Debug, Default, Clone)]
pub struct LoadReport {
    /// Number of weights successfully assigned.
    pub loaded: usize,
    /// Names of weights that did not match a known parameter.
    pub skipped: Vec<String>,
}

/// Allow combining multiple [`LoadReport`]s into one, e.g. when loading multiple shards using `+=`.
impl std::ops::AddAssign for LoadReport {
    fn add_assign(&mut self, rhs: Self) {
        self.loaded += rhs.loaded;
        self.skipped.extend(rhs.skipped);
    }
}
