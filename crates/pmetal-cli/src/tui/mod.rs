//! PMetal TUI — comprehensive terminal interface for LLM fine-tuning on Apple Silicon.
//!
//! Provides a multi-tab interface for monitoring training, inspecting hardware,
//! managing models and datasets, configuring training runs, and interactive inference.
//!
//! # Usage
//!
//! ```bash
//! pmetal tui
//! pmetal tui --metrics-file training_metrics.jsonl
//! ```

mod app;
mod event;
pub mod tabs;
pub mod theme;
pub mod widgets;

pub use app::App;

/// Run the TUI application.
pub fn run(metrics_file: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    let mut app = App::new(metrics_file);
    app.run()?;
    Ok(())
}
