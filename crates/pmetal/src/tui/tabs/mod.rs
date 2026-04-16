//! Tab definitions and implementations for the PMetal TUI.

mod bench;
pub mod dashboard;
mod datasets;
mod device;
mod distillation;
mod eval;
mod grpo;
mod inference;
mod jobs;
mod merge;
mod models;
mod pretrain;
mod quantize;
mod serve;
mod training;

pub use bench::BenchTab;
pub use dashboard::DashboardTab;
pub use datasets::DatasetsTab;
pub use device::DeviceTab;
pub use distillation::DistillationTab;
pub use eval::EvalTab;
pub use grpo::GrpoTab;
pub use inference::{InferenceFocus, InferenceTab};
pub use jobs::JobsTab;
pub use merge::MergeTab;
pub use models::{ModelSource, ModelsTab, write_training_info};
pub use pretrain::PretrainTab;
pub use quantize::QuantizeTab;
pub use serve::ServeTab;
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
    Training,
    Pretrain,
    Distillation,
    Grpo,
    Inference,
    Serve,
    Quantize,
    Merge,
    Bench,
    Eval,
    Jobs,
}

impl Tab {
    /// All tabs in display order.
    pub const ALL: &[Tab] = &[
        Tab::Device,
        Tab::Models,
        Tab::Datasets,
        Tab::Training,
        Tab::Pretrain,
        Tab::Distillation,
        Tab::Grpo,
        Tab::Dashboard,
        Tab::Inference,
        Tab::Serve,
        Tab::Quantize,
        Tab::Merge,
        Tab::Bench,
        Tab::Eval,
        Tab::Jobs,
    ];

    /// Icon for the tab (using ASCII-safe characters for wide terminal compat).
    pub fn icon(self) -> &'static str {
        match self {
            Tab::Dashboard => "#",
            Tab::Device => "@",
            Tab::Models => "~",
            Tab::Datasets => "&",
            Tab::Training => ">",
            Tab::Pretrain => "]",
            Tab::Distillation => "^",
            Tab::Grpo => "!",
            Tab::Inference => "$",
            Tab::Serve => "*",
            Tab::Quantize => "=",
            Tab::Merge => "x",
            Tab::Bench => "+",
            Tab::Eval => "?",
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
            Tab::Training => write!(f, "Training"),
            Tab::Pretrain => write!(f, "Pretrain"),
            Tab::Distillation => write!(f, "Distill"),
            Tab::Grpo => write!(f, "GRPO"),
            Tab::Inference => write!(f, "Inference"),
            Tab::Serve => write!(f, "Serve"),
            Tab::Quantize => write!(f, "Quantize"),
            Tab::Merge => write!(f, "Merge"),
            Tab::Bench => write!(f, "Bench"),
            Tab::Eval => write!(f, "Eval"),
            Tab::Jobs => write!(f, "Jobs"),
        }
    }
}
