//! PMetal TUI: a terminal control center for LLM fine-tuning on Apple Silicon.
//!
//! A multi-tab interface for monitoring training, inspecting hardware, managing
//! models and datasets, configuring and launching jobs, and chatting with a
//! model.
//!
//! Every job tab is a form built from the job's spec in `pmetal-core`, the
//! same spec the CLI, the MCP server and the desktop app build their jobs
//! from: its fields, defaults, ranges and the command it launches are the
//! spec's. Jobs run as child processes under `pmetal-supervisor`.
//!
//! # Usage
//!
//! ```bash
//! pmetal tui
//! pmetal tui --metrics-file training_metrics.jsonl
//! pmetal dashboard --metrics-file training_metrics.jsonl
//! ```

mod app;
mod event;
mod jobs;
mod keys;
mod modal;
pub mod tabs;
pub mod terminal;
pub mod theme;
pub mod widgets;

#[cfg(test)]
mod cli_check;
#[cfg(test)]
mod forms_cli;

pub use app::App;

use crate::tui::tabs::Tab;

/// Run the TUI application. Must be called from within a tokio runtime
/// (the CLI entry point is `#[tokio::main]`).
pub async fn run(metrics_file: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    App::new(metrics_file, Tab::Device).run().await
}

/// Run the TUI on the training monitor, following `metrics_file`.
pub async fn run_dashboard(metrics_file: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    App::new(metrics_file, Tab::Dashboard).run().await
}
