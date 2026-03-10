//! Tab definitions and implementations for the PMetal TUI.

mod dashboard;
mod datasets;
mod device;
mod inference;
mod jobs;
mod models;
mod training;

pub use dashboard::DashboardTab;
pub use datasets::DatasetsTab;
pub use device::DeviceTab;
pub use inference::InferenceTab;
pub use jobs::JobsTab;
pub use models::ModelsTab;
pub use training::TrainingTab;

/// All available tabs in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Device,
    Models,
    Datasets,
    Training,
    Inference,
    Jobs,
}

impl Tab {
    /// All tabs in display order.
    pub const ALL: &[Tab] = &[
        Tab::Dashboard,
        Tab::Device,
        Tab::Models,
        Tab::Datasets,
        Tab::Training,
        Tab::Inference,
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
            Tab::Inference => "$",
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
            Tab::Dashboard => write!(f, "Dashboard"),
            Tab::Device => write!(f, "Device"),
            Tab::Models => write!(f, "Models"),
            Tab::Datasets => write!(f, "Datasets"),
            Tab::Training => write!(f, "Training"),
            Tab::Inference => write!(f, "Inference"),
            Tab::Jobs => write!(f, "Jobs"),
        }
    }
}
