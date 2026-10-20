//! Tab definitions and implementations for the PMetal TUI.

mod bench;
mod custom_dirs;
pub mod dashboard;
mod datasets;
mod device;
mod infer_stream;
mod inference;
mod jobs;
mod models;
mod ollama;
pub mod panel;
mod serve;
mod status_panel;
mod tools;
mod trainers;

pub use bench::BenchTab;
pub use dashboard::DashboardTab;
pub use datasets::scan as scan_datasets;
pub use datasets::{DatasetEntry, DatasetPeek, DatasetsTab};
#[cfg(test)]
pub use datasets::{DatasetSource, Readiness as DatasetReadiness};
pub use device::{DeviceInfo, DeviceTab, read_memory_in_use};
pub use inference::{InferenceFocus, InferenceTab};
pub use jobs::{JobEntry, JobsTab, Selection as JobSelection, history_roots, scan_history};
pub use models::{
    ModelEntry, ModelSource, ModelsTab, fit_style, is_lora_adapter, query_device_spec,
    scan as scan_models, write_training_info,
};
pub use ollama::OllamaTab;
pub use panel::{Applied, FormTab, Pick, RunState};
pub use serve::ServeTab;
pub use tools::ToolTab;
pub use trainers::TrainerTab;

/// Extract a short model name from a model ID.
/// e.g. "Qwen/Qwen3-0.6B" → "Qwen3-0.6B", "trained/foo" → "foo"
pub fn model_short_name(model_id: &str) -> String {
    model_id.rsplit('/').next().unwrap_or(model_id).to_string()
}

/// All available tabs in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Whether the tab runs a job from a form.
    pub fn is_form(self) -> bool {
        !matches!(
            self,
            Tab::Dashboard | Tab::Device | Tab::Models | Tab::Datasets | Tab::Inference | Tab::Jobs
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tab_is_listed_once_and_cycling_visits_them_all() {
        let mut seen = std::collections::HashSet::new();
        let mut tab = Tab::ALL[0];
        for _ in 0..Tab::ALL.len() {
            assert!(seen.insert(tab), "{tab} twice");
            tab = tab.next();
        }
        assert_eq!(tab, Tab::ALL[0]);
        assert_eq!(seen.len(), Tab::ALL.len());
        assert_eq!(Tab::ALL[0].prev(), *Tab::ALL.last().unwrap());
    }

    #[test]
    fn icons_and_names_tell_the_tabs_apart() {
        let mut icons = std::collections::HashSet::new();
        let mut names = std::collections::HashSet::new();
        for &tab in Tab::ALL {
            assert!(icons.insert(tab.icon()), "icon of {tab}");
            assert!(names.insert(tab.to_string()), "name of {tab}");
        }
    }
}
