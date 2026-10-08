//! Tab definitions and implementations for the PMetal TUI.

mod bench;
pub mod dashboard;
mod datasets;
mod device;
mod dflash;
mod distillation;
mod embed_train;
mod eval;
mod grpo;
mod inference;
mod jobs;
mod merge;
mod models;
mod ollama;
mod preference;
mod pretrain;
mod quantize;
mod rlkd;
mod serve;
mod tokenize;
mod training;

pub use bench::BenchTab;
pub use dashboard::DashboardTab;
pub use datasets::DatasetsTab;
pub use device::DeviceTab;
pub use dflash::DflashTab;
pub use distillation::DistillationTab;
pub use embed_train::EmbedTrainTab;
pub use eval::EvalTab;
pub use grpo::GrpoTab;
pub use inference::{InferenceFocus, InferenceTab};
pub use jobs::JobsTab;
pub use merge::MergeTab;
pub use models::{ModelSource, ModelsTab, write_training_info};
pub use ollama::OllamaTab;
pub use preference::PreferenceTab;
pub use pretrain::PretrainTab;
pub use quantize::QuantizeTab;
pub use rlkd::RlkdTab;
pub use serve::ServeTab;
pub use tokenize::TokenizeTab;
pub use training::{TrainingStatus, TrainingTab};

/// Extract a short model name from a model ID.
/// e.g. "Qwen/Qwen3-0.6B" → "Qwen3-0.6B", "trained/foo" → "foo"
pub fn model_short_name(model_id: &str) -> String {
    model_id.rsplit('/').next().unwrap_or(model_id).to_string()
}

/// All available tabs in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Device,
    Models,
    Datasets,
    Tokenize,
    Training,
    EmbedTrain,
    Pretrain,
    Distillation,
    Rlkd,
    Grpo,
    Preference,
    Inference,
    Dflash,
    Serve,
    Quantize,
    Merge,
    Bench,
    Eval,
    Ollama,
    Jobs,
}

impl Tab {
    /// All tabs in display order.
    ///
    /// Ordering rationale:
    /// - System/data tabs first: Device, Models, Datasets, Tokenize
    /// - Training family: Training, EmbedTrain, Pretrain, Distillation, Rlkd, Grpo, Preference
    /// - Dashboard (metrics monitor) after training group
    /// - Inference/serving: Inference, DFlash, Serve
    /// - Post-processing: Quantize, Merge, Bench, Eval
    /// - External exports: Modelfile
    /// - Jobs log last
    pub const ALL: &[Tab] = &[
        Tab::Device,
        Tab::Models,
        Tab::Datasets,
        Tab::Tokenize,
        Tab::Training,
        Tab::EmbedTrain,
        Tab::Pretrain,
        Tab::Distillation,
        Tab::Rlkd,
        Tab::Grpo,
        Tab::Preference,
        Tab::Dashboard,
        Tab::Inference,
        Tab::Dflash,
        Tab::Serve,
        Tab::Quantize,
        Tab::Merge,
        Tab::Bench,
        Tab::Eval,
        Tab::Ollama,
        Tab::Jobs,
    ];

    /// Icon for the tab (using ASCII-safe characters for wide terminal compat).
    pub fn icon(self) -> &'static str {
        match self {
            Tab::Dashboard => "#",
            Tab::Device => "@",
            Tab::Models => "~",
            Tab::Datasets => "&",
            Tab::Tokenize => "|",
            Tab::Training => ">",
            Tab::EmbedTrain => "e",
            Tab::Pretrain => "]",
            Tab::Distillation => "^",
            Tab::Rlkd => "r",
            Tab::Grpo => "!",
            Tab::Preference => "<",
            Tab::Inference => "$",
            Tab::Dflash => "d",
            Tab::Serve => "*",
            Tab::Quantize => "=",
            Tab::Merge => "x",
            Tab::Bench => "+",
            Tab::Eval => "?",
            Tab::Ollama => "o",
            Tab::Jobs => "%",
        }
    }

    /// Next tab (wraps around).
    pub fn next(self) -> Tab {
        let all = Tab::ALL;
        let idx = all.iter().position(|t| *t == self).unwrap_or(0);
        all[(idx + 1) % all.len()]
    }

    /// Previous tab (wraps around).
    pub fn prev(self) -> Tab {
        let all = Tab::ALL;
        let idx = all.iter().position(|t| *t == self).unwrap_or(0);
        all[(idx + all.len() - 1) % all.len()]
    }
}

impl std::fmt::Display for Tab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tab::Dashboard => write!(f, "Monitor"),
            Tab::Device => write!(f, "System"),
            Tab::Models => write!(f, "Models"),
            Tab::Datasets => write!(f, "Datasets"),
            Tab::Tokenize => write!(f, "Tokenize"),
            Tab::Training => write!(f, "Training"),
            Tab::EmbedTrain => write!(f, "EmbedTrain"),
            Tab::Pretrain => write!(f, "Pretrain"),
            Tab::Distillation => write!(f, "Distill"),
            Tab::Rlkd => write!(f, "RLKD"),
            Tab::Grpo => write!(f, "GRPO"),
            Tab::Preference => write!(f, "Preference"),
            Tab::Inference => write!(f, "Inference"),
            Tab::Dflash => write!(f, "DFlash"),
            Tab::Serve => write!(f, "Serve"),
            Tab::Quantize => write!(f, "Quantize"),
            Tab::Merge => write!(f, "Merge"),
            Tab::Bench => write!(f, "Bench"),
            Tab::Eval => write!(f, "Eval"),
            Tab::Ollama => write!(f, "Modelfile"),
            Tab::Jobs => write!(f, "Jobs"),
        }
    }
}
