//! PMetal TUI — comprehensive terminal interface for LLM fine-tuning on Apple Silicon.
//!
//! Provides a multi-tab interface for monitoring training, inspecting hardware,
//! managing models and datasets, configuring and launching training runs,
//! distillation, GRPO, and interactive inference.
//!
//! # Usage
//!
//! ```bash
//! pmetal tui
//! pmetal tui --metrics-file training_metrics.jsonl
//! ```

mod app;
mod command_runner;
mod event;
mod modal;
pub mod tabs;
pub mod theme;
pub mod widgets;

pub use app::App;

/// Run the TUI application. Must be called from within a tokio runtime
/// (the CLI entry point is `#[tokio::main]`).
pub async fn run(metrics_file: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    let mut app = App::new(metrics_file);
    app.run().await
}
