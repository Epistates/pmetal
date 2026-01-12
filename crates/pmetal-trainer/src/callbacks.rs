//! Training callbacks for monitoring and logging.
//!
//! This module provides various callbacks for training visualization and logging:
//!
//! - [`ProgressCallback`] - Progress bar for training visualization
//! - [`LoggingCallback`] - Basic logging with tracing
//! - [`CheckpointCallback`] - Checkpoint event logging
//! - [`MetricsJsonCallback`] - JSONL metrics file (Wandb-compatible import)
//! - [`TensorBoardCallback`] - TensorBoard logging (requires `tensorboard` feature)
//!
//! # Wandb Integration
//!
//! For Wandb integration, use [`MetricsJsonCallback`] to write JSONL metrics files,
//! then import them using Wandb's offline sync:
//!
//! ```bash
//! wandb sync --include-offline --include-synced path/to/metrics.jsonl
//! ```
//!
//! # TensorBoard Integration
//!
//! Enable the `tensorboard` feature to use [`TensorBoardCallback`]:
//!
//! ```toml
//! pmetal-trainer = { version = "0.1", features = ["tensorboard"] }
//! ```

use pmetal_core::{EvalMetrics, TrainingCallback};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Progress bar callback for training visualization.
pub struct ProgressCallback {
    progress: indicatif::ProgressBar,
}

impl ProgressCallback {
    /// Create a new progress callback.
    pub fn new(total_steps: usize) -> Self {
        let progress = indicatif::ProgressBar::new(total_steps as u64);
        progress.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        Self { progress }
    }
}

impl TrainingCallback for ProgressCallback {
    fn on_train_start(&mut self) {
        self.progress.reset();
    }

    fn on_train_end(&mut self) {
        self.progress.finish_with_message("Training complete!");
    }

    fn on_step_end(&mut self, step: usize, loss: f64) {
        self.progress.set_position(step as u64);
        self.progress.set_message(format!("loss: {:.4}", loss));
    }
}

/// Logging callback for training metrics.
pub struct LoggingCallback {
    log_every: usize,
}

impl LoggingCallback {
    /// Create a new logging callback.
    pub fn new(log_every: usize) -> Self {
        Self { log_every }
    }
}

impl TrainingCallback for LoggingCallback {
    fn on_step_end(&mut self, step: usize, loss: f64) {
        if step % self.log_every == 0 {
            tracing::info!(step = step, loss = loss, "Training step");
        }
    }

    fn on_epoch_end(&mut self, epoch: usize, metrics: &EvalMetrics) {
        tracing::info!(
            epoch = epoch,
            loss = metrics.loss,
            perplexity = metrics.perplexity,
            "Epoch complete"
        );
    }
}

/// Checkpoint callback for saving training state.
///
/// **Note**: This callback provides logging only. Actual checkpoint saving
/// is handled by [`CheckpointManager`] in the training loop. This callback
/// can be used to add custom behavior around checkpoint events.
pub struct CheckpointCallback {
    save_every: usize,
    output_dir: String,
}

impl CheckpointCallback {
    /// Create a new checkpoint callback.
    ///
    /// This callback logs checkpoint events. Actual checkpoint persistence
    /// is managed by the training loop's `CheckpointManager`.
    pub fn new(save_every: usize, output_dir: &str) -> Self {
        Self {
            save_every,
            output_dir: output_dir.to_string(),
        }
    }
}

impl TrainingCallback for CheckpointCallback {
    fn on_step_end(&mut self, step: usize, _loss: f64) {
        if step % self.save_every == 0 && step > 0 {
            let path = format!("{}/checkpoint-{}", self.output_dir, step);
            tracing::info!(path = path, "Checkpoint milestone reached");
            // Note: Actual checkpoint saving is handled by CheckpointManager
            // in the training loop. This callback is for logging/custom hooks.
        }
    }
}

/// JSONL metrics callback for Wandb-compatible logging.
///
/// Writes training metrics to a JSONL file that can be imported into Wandb
/// using `wandb sync` or imported into other visualization tools.
///
/// # Example
///
/// ```ignore
/// use pmetal_trainer::callbacks::MetricsJsonCallback;
///
/// let callback = MetricsJsonCallback::new("./output/metrics.jsonl")?;
/// trainer.add_callback(Box::new(callback));
/// ```
///
/// The output format is JSONL with one JSON object per line:
///
/// ```json
/// {"step": 0, "loss": 2.5, "epoch": 0, "timestamp": "2024-12-31T12:00:00Z"}
/// {"step": 1, "loss": 2.3, "epoch": 0, "timestamp": "2024-12-31T12:00:01Z"}
/// ```
pub struct MetricsJsonCallback {
    writer: BufWriter<File>,
    path: PathBuf,
    current_epoch: usize,
    run_name: Option<String>,
    config: Option<serde_json::Value>,
}

impl MetricsJsonCallback {
    /// Create a new JSONL metrics callback.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output JSONL file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created.
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        Ok(Self {
            writer: BufWriter::new(file),
            path,
            current_epoch: 0,
            run_name: None,
            config: None,
        })
    }

    /// Set an optional run name for identification.
    pub fn with_run_name(mut self, name: impl Into<String>) -> Self {
        self.run_name = Some(name.into());
        self
    }

    /// Set training configuration to log at start.
    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.config = Some(config);
        self
    }

    /// Get the path to the metrics file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn write_entry(&mut self, entry: serde_json::Value) {
        if let Ok(line) = serde_json::to_string(&entry) {
            let _ = writeln!(self.writer, "{}", line);
        }
    }
}

impl TrainingCallback for MetricsJsonCallback {
    fn on_train_start(&mut self) {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let mut entry = serde_json::json!({
            "event": "train_start",
            "timestamp": timestamp,
        });

        if let Some(ref name) = self.run_name {
            entry["run_name"] = serde_json::json!(name);
        }

        if let Some(ref config) = self.config {
            entry["config"] = config.clone();
        }

        self.write_entry(entry);
        let _ = self.writer.flush();
    }

    fn on_train_end(&mut self) {
        let timestamp = chrono::Utc::now().to_rfc3339();
        self.write_entry(serde_json::json!({
            "event": "train_end",
            "timestamp": timestamp,
        }));
        let _ = self.writer.flush();
    }

    fn on_epoch_start(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn on_epoch_end(&mut self, epoch: usize, metrics: &EvalMetrics) {
        let timestamp = chrono::Utc::now().to_rfc3339();
        self.write_entry(serde_json::json!({
            "event": "epoch_end",
            "epoch": epoch,
            "loss": metrics.loss,
            "perplexity": metrics.perplexity,
            "timestamp": timestamp,
        }));
        let _ = self.writer.flush();
    }

    fn on_step_end(&mut self, step: usize, loss: f64) {
        let timestamp = chrono::Utc::now().to_rfc3339();
        self.write_entry(serde_json::json!({
            "step": step,
            "epoch": self.current_epoch,
            "loss": loss,
            "timestamp": timestamp,
        }));

        // Flush every 10 steps to balance I/O and data safety
        if step % 10 == 0 {
            let _ = self.writer.flush();
        }
    }

    fn on_save(&mut self, path: &Path) {
        let timestamp = chrono::Utc::now().to_rfc3339();
        self.write_entry(serde_json::json!({
            "event": "checkpoint_saved",
            "path": path.display().to_string(),
            "timestamp": timestamp,
        }));
        let _ = self.writer.flush();
    }
}

/// TensorBoard callback for real-time training visualization.
///
/// Writes training metrics to TensorBoard event files that can be viewed
/// using `tensorboard --logdir <path>`.
///
/// # Feature Flag
///
/// Requires the `tensorboard` feature:
///
/// ```toml
/// pmetal-trainer = { version = "0.1", features = ["tensorboard"] }
/// ```
///
/// # Example
///
/// ```ignore
/// use pmetal_trainer::callbacks::TensorBoardCallback;
///
/// let callback = TensorBoardCallback::new("./output/tensorboard")?;
/// trainer.add_callback(Box::new(callback));
/// ```
#[cfg(feature = "tensorboard")]
pub struct TensorBoardCallback {
    writer: tensorboard_rs::summary_writer::SummaryWriter,
    log_dir: PathBuf,
    current_epoch: usize,
}

#[cfg(feature = "tensorboard")]
impl TensorBoardCallback {
    /// Create a new TensorBoard callback.
    ///
    /// # Arguments
    ///
    /// * `log_dir` - Directory to write TensorBoard event files
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn new(log_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&log_dir)?;

        let writer =
            tensorboard_rs::summary_writer::SummaryWriter::new(&log_dir.display().to_string());

        Ok(Self {
            writer,
            log_dir,
            current_epoch: 0,
        })
    }

    /// Get the log directory path.
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }
}

#[cfg(feature = "tensorboard")]
impl TrainingCallback for TensorBoardCallback {
    fn on_train_start(&mut self) {
        tracing::info!(
            log_dir = %self.log_dir.display(),
            "TensorBoard logging started"
        );
    }

    fn on_train_end(&mut self) {
        self.writer.flush();
        tracing::info!("TensorBoard logging complete");
    }

    fn on_epoch_start(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn on_epoch_end(&mut self, epoch: usize, metrics: &EvalMetrics) {
        use std::collections::HashMap;

        let mut scalars = HashMap::new();
        scalars.insert("epoch_loss".to_string(), metrics.loss as f32);
        scalars.insert("epoch_perplexity".to_string(), metrics.perplexity as f32);

        self.writer.add_scalars("epoch", &scalars, epoch);
        self.writer.flush();
    }

    fn on_step_end(&mut self, step: usize, loss: f64) {
        use std::collections::HashMap;

        let mut scalars = HashMap::new();
        scalars.insert("loss".to_string(), loss as f32);
        scalars.insert("epoch".to_string(), self.current_epoch as f32);

        self.writer.add_scalars("train", &scalars, step);

        // Flush every 50 steps to reduce I/O overhead
        if step % 50 == 0 {
            self.writer.flush();
        }
    }
}

/// Composite callback that forwards events to multiple callbacks.
///
/// Useful for combining multiple logging backends (e.g., progress bar + JSONL + TensorBoard).
///
/// # Example
///
/// ```ignore
/// use pmetal_trainer::callbacks::{CompositeCallback, ProgressCallback, MetricsJsonCallback};
///
/// let mut composite = CompositeCallback::new();
/// composite.add(Box::new(ProgressCallback::new(1000)));
/// composite.add(Box::new(MetricsJsonCallback::new("metrics.jsonl")?));
/// trainer.add_callback(Box::new(composite));
/// ```
pub struct CompositeCallback {
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl CompositeCallback {
    /// Create a new empty composite callback.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback to the composite.
    pub fn add(&mut self, callback: Box<dyn TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Get the number of callbacks.
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Check if there are no callbacks.
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }
}

impl Default for CompositeCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingCallback for CompositeCallback {
    fn on_train_start(&mut self) {
        for cb in &mut self.callbacks {
            cb.on_train_start();
        }
    }

    fn on_train_end(&mut self) {
        for cb in &mut self.callbacks {
            cb.on_train_end();
        }
    }

    fn on_epoch_start(&mut self, epoch: usize) {
        for cb in &mut self.callbacks {
            cb.on_epoch_start(epoch);
        }
    }

    fn on_epoch_end(&mut self, epoch: usize, metrics: &EvalMetrics) {
        for cb in &mut self.callbacks {
            cb.on_epoch_end(epoch, metrics);
        }
    }

    fn on_step_start(&mut self, step: usize) {
        for cb in &mut self.callbacks {
            cb.on_step_start(step);
        }
    }

    fn on_step_end(&mut self, step: usize, loss: f64) {
        for cb in &mut self.callbacks {
            cb.on_step_end(step, loss);
        }
    }

    fn on_save(&mut self, path: &Path) {
        for cb in &mut self.callbacks {
            cb.on_save(path);
        }
    }
}
