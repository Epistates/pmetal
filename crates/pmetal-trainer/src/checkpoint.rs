//! Training checkpoint save/load functionality.
//!
//! This module provides utilities for saving and loading training state,
//! including model weights, optimizer state, and training progress.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pmetal_data::LoaderState;
use pmetal_runtime::compat::optimizers::OptimizerState;
use pmetal_runtime::compat::{Array, Exception};
use pmetal_runtime::inline_array;
use serde::{Deserialize, Serialize};

use crate::{Result, SftError};

// ---------------------------------------------------------------------------
// Best-snapshot persistence (for rollback-on-resume)
// ---------------------------------------------------------------------------

/// Persist the best in-memory LoRA snapshot to disk as a safetensors file.
///
/// Writes `best_snapshot.safetensors` inside `checkpoint_dir`. Intended to be
/// called whenever the training loop takes a new in-memory best snapshot so
/// that the weights are recoverable if the process is interrupted and resumed.
///
/// The file is written atomically (beside the file, then renamed) to avoid leaving a
/// partially-written snapshot that could corrupt a resume.
pub fn save_best_snapshot(checkpoint_dir: &Path, weights: &HashMap<Rc<str>, Array>) -> Result<()> {
    fs::create_dir_all(checkpoint_dir).map_err(|e| {
        SftError::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to create checkpoint directory: {e}"),
        ))
    })?;

    // The save is atomic (written beside the file and renamed over it).
    let final_path = checkpoint_dir.join("best_snapshot.safetensors");
    let entries: Vec<(&str, &Array)> = weights.iter().map(|(k, v)| (k.as_ref(), v)).collect();
    Array::save_safetensors(path_as_str(&final_path)?, &entries).map_err(Exception::from)?;

    tracing::debug!("Persisted best snapshot to {:?}", final_path);
    Ok(())
}

/// Load the persisted best LoRA snapshot from `checkpoint_dir`, if present.
///
/// Returns `None` if `best_snapshot.safetensors` does not exist.
/// Errors in the file I/O are logged as warnings and `None` is returned so
/// that a missing or corrupt snapshot degrades gracefully to a fresh start.
pub fn load_best_snapshot(checkpoint_dir: &Path) -> Option<HashMap<Rc<str>, Array>> {
    let path = checkpoint_dir.join("best_snapshot.safetensors");
    if !path.exists() {
        return None;
    }

    match inline_array::load_safetensors_shard(path_as_str(&path).ok()?) {
        Some(params) => {
            let weights: HashMap<Rc<str>, Array> =
                params.into_iter().map(|(k, v)| (Rc::from(k), v)).collect();
            tracing::info!(
                "Loaded best snapshot from {:?} ({} params)",
                path,
                weights.len()
            );
            Some(weights)
        }
        None => {
            tracing::warn!("Failed to load best snapshot from {:?}", path);
            None
        }
    }
}

/// A loaded checkpoint: its parameters by name, and its metadata.
pub type LoadedCheckpoint = (HashMap<Rc<str>, Array>, CheckpointMetadata);

/// What a resumed run needs beyond the weights to take the steps the
/// uninterrupted run would have: the optimizer's moments and counters (and the
/// f32 master copies of half-precision weights), and where the data loader
/// was in its epoch.
#[derive(Debug, Clone, Default)]
pub struct TrainState {
    /// The optimizer's state.
    pub optimizer: Option<OptimizerState>,
    /// The data loader's position, for this rank.
    pub loader: Option<LoaderState>,
}

/// The optimizer's tensors, beside the weights.
const OPTIMIZER_FILE: &str = "optimizer.safetensors";
/// Everything else a resume needs.
const TRAIN_STATE_FILE: &str = "train_state.json";

/// The JSON half of [`TrainState`].
#[derive(Debug, Serialize, Deserialize)]
struct TrainStateJson {
    /// The kind of optimizer that wrote the state.
    optimizer_kind: Option<String>,
    /// Its counters.
    #[serde(default)]
    optimizer_scalars: std::collections::BTreeMap<String, f64>,
    /// The data loader's position.
    loader: Option<LoaderState>,
}

impl TrainState {
    /// Write the state into `dir`. Every array must be evaluated, as it is
    /// when this runs on a background thread.
    fn write(&self, dir: &Path) -> Result<()> {
        let mut json = TrainStateJson {
            optimizer_kind: None,
            optimizer_scalars: Default::default(),
            loader: self.loader,
        };
        if let Some(state) = &self.optimizer {
            json.optimizer_kind = Some(state.kind.clone());
            json.optimizer_scalars = state.scalars.clone();
            let entries: Vec<(&str, &Array)> = state
                .tensors
                .iter()
                .map(|(name, array)| (name.as_str(), array))
                .collect();
            Array::save_safetensors(path_as_str(&dir.join(OPTIMIZER_FILE))?, &entries)
                .map_err(Exception::from)?;
        }
        let text = serde_json::to_string_pretty(&json).map_err(|e| {
            SftError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to serialize the training state: {e}"),
            ))
        })?;
        fs::write(dir.join(TRAIN_STATE_FILE), text).map_err(|e| {
            SftError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to write the training state: {e}"),
            ))
        })
    }

    /// Read the state the checkpoint at `dir` holds, if it holds one.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let json_path = dir.join(TRAIN_STATE_FILE);
        if !json_path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&json_path)?;
        let json: TrainStateJson = serde_json::from_str(&text).map_err(|e| {
            SftError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse {}: {e}", json_path.display()),
            ))
        })?;
        let optimizer = match json.optimizer_kind {
            Some(kind) => {
                let tensors_path = dir.join(OPTIMIZER_FILE);
                let tensors = inline_array::load_safetensors_shard(path_as_str(&tensors_path)?)
                    .ok_or_else(|| {
                        SftError::Io(std::io::Error::other(format!(
                            "Failed to load the optimizer state from {}",
                            tensors_path.display()
                        )))
                    })?;
                Some(OptimizerState {
                    kind,
                    scalars: json.optimizer_scalars,
                    tensors: tensors.into_iter().collect(),
                })
            }
            None => None,
        };
        Ok(Some(Self {
            optimizer,
            loader: json.loader,
        }))
    }
}

/// Training state metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Current training step.
    pub step: usize,
    /// Current epoch.
    pub epoch: usize,
    /// Running loss average.
    pub running_loss: f64,
    /// Best validation loss seen.
    pub best_val_loss: Option<f64>,
    /// Learning rate at checkpoint.
    pub learning_rate: f64,
    /// Model configuration (as JSON string for flexibility).
    pub model_config: Option<String>,
    /// Training configuration (as JSON string).
    pub training_config: Option<String>,
    /// Random seed used.
    pub seed: u64,
    /// Timestamp (ISO 8601).
    pub timestamp: String,
}

impl CheckpointMetadata {
    /// Create new metadata for the current training state.
    pub fn new(step: usize, epoch: usize, running_loss: f64, learning_rate: f64) -> Self {
        Self {
            step,
            epoch,
            running_loss,
            best_val_loss: None,
            learning_rate,
            model_config: None,
            training_config: None,
            seed: 42,
            timestamp: chrono_timestamp(),
        }
    }

    /// Set the best validation loss.
    pub fn with_best_val_loss(mut self, loss: f64) -> Self {
        self.best_val_loss = Some(loss);
        self
    }

    /// Set the model configuration.
    pub fn with_model_config(mut self, config: &str) -> Self {
        self.model_config = Some(config.to_string());
        self
    }

    /// Set the training configuration.
    pub fn with_training_config(mut self, config: &str) -> Self {
        self.training_config = Some(config.to_string());
        self
    }

    /// Set the seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Get current timestamp in ISO 8601 format.
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// Write checkpoint files (safetensors + metadata JSON) to a directory.
///
/// Creates the directory if it doesn't exist, writes `lora_weights.safetensors`
/// and `metadata.json`. Used by both `save_checkpoint` and `save_checkpoint_owned`
/// to eliminate duplicated file I/O logic.
fn write_checkpoint_to_dir<'a>(
    dir: &Path,
    params: impl IntoIterator<Item = (&'a str, &'a Array)>,
    metadata_json: &str,
    state: Option<&TrainState>,
) -> Result<()> {
    fs::create_dir_all(dir).map_err(|e| {
        SftError::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to create directory: {e}"),
        ))
    })?;

    let weights_path = dir.join("lora_weights.safetensors");
    let entries: Vec<(&str, &Array)> = params.into_iter().collect();
    Array::save_safetensors(path_as_str(&weights_path)?, &entries).map_err(Exception::from)?;

    let metadata_path = dir.join("metadata.json");
    let mut file = File::create(&metadata_path).map_err(|e| {
        SftError::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to create metadata file: {e}"),
        ))
    })?;
    file.write_all(metadata_json.as_bytes()).map_err(|e| {
        SftError::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to write metadata: {e}"),
        ))
    })?;

    if let Some(state) = state {
        state.write(dir)?;
    }

    Ok(())
}

/// Checkpoint manager for saving and loading training state.
pub struct CheckpointManager {
    /// Base directory for checkpoints.
    checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep (None = unlimited).
    max_checkpoints: Option<usize>,
    /// Save best model separately.
    save_best: bool,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();
        fs::create_dir_all(&checkpoint_dir).map_err(|e| {
            SftError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to create checkpoint directory: {}", e),
            ))
        })?;

        Ok(Self {
            checkpoint_dir,
            max_checkpoints: Some(5),
            save_best: true,
        })
    }

    /// Set maximum number of checkpoints to keep.
    pub fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = Some(max);
        self
    }

    /// Set whether to save best model separately.
    pub fn with_save_best(mut self, save_best: bool) -> Self {
        self.save_best = save_best;
        self
    }

    /// Save a training checkpoint.
    ///
    /// # Arguments
    /// * `lora_params` - LoRA parameters to save
    /// * `metadata` - Training state metadata
    /// * `is_best` - Whether this is the best checkpoint so far
    pub fn save_checkpoint(
        &self,
        lora_params: &HashMap<Rc<str>, Array>,
        metadata: &CheckpointMetadata,
        is_best: bool,
    ) -> Result<PathBuf> {
        let entries = lora_params.iter().map(|(k, v)| (k.as_ref(), v)).collect();
        self.save_entries(entries, metadata, is_best, None)
    }

    /// [`save_checkpoint`](Self::save_checkpoint) with the optimizer's state
    /// and the data loader's position, which a resume continues from.
    pub fn save_checkpoint_with_state(
        &self,
        lora_params: &HashMap<Rc<str>, Array>,
        metadata: &CheckpointMetadata,
        is_best: bool,
        state: &TrainState,
    ) -> Result<PathBuf> {
        let entries = lora_params.iter().map(|(k, v)| (k.as_ref(), v)).collect();
        self.save_entries(entries, metadata, is_best, Some(state))
    }

    /// Write one checkpoint (and the best one, when this is it) and trim the
    /// old ones.
    fn save_entries(
        &self,
        entries: Vec<(&str, &Array)>,
        metadata: &CheckpointMetadata,
        is_best: bool,
        state: Option<&TrainState>,
    ) -> Result<PathBuf> {
        let metadata_json = serde_json::to_string_pretty(metadata).map_err(|e| {
            SftError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to serialize metadata: {e}"),
            ))
        })?;

        let step_dir = self.checkpoint_dir.join(format!("step_{}", metadata.step));
        write_checkpoint_to_dir(&step_dir, entries.iter().copied(), &metadata_json, state)?;

        self.update_latest_link(metadata.step)?;

        if is_best && self.save_best {
            let best_dir = self.checkpoint_dir.join("best");
            write_checkpoint_to_dir(&best_dir, entries.iter().copied(), &metadata_json, state)?;
            tracing::info!("Saved best checkpoint at step {}", metadata.step);
        }

        self.cleanup_old_checkpoints()?;

        tracing::info!(
            "Saved checkpoint at step {} to {:?}",
            metadata.step,
            step_dir
        );

        Ok(step_dir)
    }

    /// Update the "latest" symlink/marker.
    fn update_latest_link(&self, step: usize) -> Result<()> {
        let latest_path = self.checkpoint_dir.join("latest");
        let step_name = format!("step_{}", step);

        // Write step name to file (cross-platform alternative to symlinks)
        let mut file = File::create(&latest_path).map_err(|e| {
            SftError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to create latest marker: {}", e),
            ))
        })?;
        file.write_all(step_name.as_bytes()).map_err(|e| {
            SftError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to write latest marker: {}", e),
            ))
        })?;

        Ok(())
    }

    /// Clean up old checkpoints, keeping only the most recent ones.
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let Some(max) = self.max_checkpoints else {
            return Ok(());
        };

        // List step directories
        let mut step_dirs: Vec<(usize, PathBuf)> = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| {
                SftError::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to read checkpoint directory: {}", e),
                ))
            })?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("step_") {
                    let step = name.strip_prefix("step_")?.parse::<usize>().ok()?;
                    Some((step, entry.path()))
                } else {
                    None
                }
            })
            .collect();

        // Sort by step (oldest first)
        step_dirs.sort_by_key(|(step, _)| *step);

        // Remove old checkpoints
        while step_dirs.len() > max {
            let (step, path) = step_dirs.remove(0);
            if let Err(e) = fs::remove_dir_all(&path) {
                tracing::warn!("Failed to remove old checkpoint {}: {}", step, e);
            } else {
                tracing::debug!("Removed old checkpoint at step {}", step);
            }
        }

        Ok(())
    }

    /// Load a checkpoint from a directory.
    pub fn load_checkpoint<P: AsRef<Path>>(checkpoint_path: P) -> Result<LoadedCheckpoint> {
        let checkpoint_path = checkpoint_path.as_ref();

        // Load weights
        let weights_path = checkpoint_path.join("lora_weights.safetensors");
        let params =
            inline_array::load_safetensors_shard(path_as_str(&weights_path)?).ok_or_else(|| {
                SftError::Io(std::io::Error::other(format!(
                    "Failed to load weights from {}",
                    weights_path.display()
                )))
            })?;

        // Convert HashMap<String, Array> to HashMap<Rc<str>, Array>
        let params: HashMap<Rc<str>, Array> =
            params.into_iter().map(|(k, v)| (Rc::from(k), v)).collect();

        // Load metadata
        let metadata_path = checkpoint_path.join("metadata.json");
        let mut file = File::open(&metadata_path).map_err(|e| {
            SftError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to open metadata file: {}", e),
            ))
        })?;
        let mut metadata_json = String::new();
        file.read_to_string(&mut metadata_json).map_err(|e| {
            SftError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read metadata: {}", e),
            ))
        })?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json).map_err(|e| {
            SftError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse metadata: {}", e),
            ))
        })?;

        tracing::info!(
            "Loaded checkpoint from step {} ({:?})",
            metadata.step,
            checkpoint_path
        );

        Ok((params, metadata))
    }

    /// Load the latest checkpoint.
    pub fn load_latest(&self) -> Result<Option<LoadedCheckpoint>> {
        match self.latest_dir()? {
            Some(dir) => Self::load_checkpoint(&dir).map(Some),
            None => Ok(None),
        }
    }

    /// The directory of the latest checkpoint, if there is one.
    pub fn latest_dir(&self) -> Result<Option<PathBuf>> {
        let latest_path = self.checkpoint_dir.join("latest");
        if !latest_path.exists() {
            return Ok(None);
        }

        // Read latest marker
        let mut file = File::open(&latest_path).map_err(|e| {
            SftError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to open latest marker: {}", e),
            ))
        })?;
        let mut step_name = String::new();
        file.read_to_string(&mut step_name).map_err(|e| {
            SftError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read latest marker: {}", e),
            ))
        })?;

        let checkpoint_path = self.checkpoint_dir.join(step_name.trim());
        Ok(checkpoint_path.exists().then_some(checkpoint_path))
    }

    /// Load the best checkpoint.
    pub fn load_best(&self) -> Result<Option<LoadedCheckpoint>> {
        let best_path = self.checkpoint_dir.join("best");
        if best_path.exists() {
            Self::load_checkpoint(&best_path).map(Some)
        } else {
            Ok(None)
        }
    }

    /// List all available checkpoints.
    pub fn list_checkpoints(&self) -> Result<Vec<(usize, PathBuf)>> {
        let mut checkpoints: Vec<(usize, PathBuf)> = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| {
                SftError::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to read checkpoint directory: {}", e),
                ))
            })?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("step_") {
                    let step = name.strip_prefix("step_")?.parse::<usize>().ok()?;
                    Some((step, entry.path()))
                } else {
                    None
                }
            })
            .collect();

        checkpoints.sort_by_key(|(step, _)| *step);
        Ok(checkpoints)
    }

    /// Get the checkpoint directory.
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }

    /// Return the configured maximum number of checkpoints to retain.
    pub fn max_checkpoints_limit(&self) -> Option<usize> {
        self.max_checkpoints
    }

    /// Return whether the best-checkpoint-separately flag is set.
    pub fn save_best_flag(&self) -> bool {
        self.save_best
    }

    /// Construct a `CheckpointManager` from raw parts without creating directories.
    ///
    /// Used by background checkpoint threads that operate on directories that have
    /// already been created by the foreground trainer thread.
    pub(crate) fn from_parts(
        checkpoint_dir: PathBuf,
        max_checkpoints: Option<usize>,
        save_best: bool,
    ) -> Self {
        Self {
            checkpoint_dir,
            max_checkpoints,
            save_best,
        }
    }

    /// Save a checkpoint using `String`-keyed (i.e. `Send`) parameter maps.
    ///
    /// This is the background-thread-compatible counterpart of `save_checkpoint`.
    /// It accepts `HashMap<String, Array>` (where both `String` and `Array` are `Send`)
    /// rather than the `Rc<str>`-keyed variant used in the public API.
    pub(crate) fn save_checkpoint_owned(
        &self,
        lora_params: HashMap<String, Array>,
        metadata: &CheckpointMetadata,
        is_best: bool,
        state: Option<&TrainState>,
    ) -> Result<PathBuf> {
        let entries = lora_params.iter().map(|(k, v)| (k.as_str(), v)).collect();
        self.save_entries(entries, metadata, is_best, state)
    }
}

fn path_as_str(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        SftError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Non-UTF8 path: {}", path.display()),
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_metadata() {
        let meta = CheckpointMetadata::new(100, 2, 0.5, 1e-4)
            .with_best_val_loss(0.4)
            .with_seed(123);

        assert_eq!(meta.step, 100);
        assert_eq!(meta.epoch, 2);
        assert!((meta.running_loss - 0.5).abs() < 1e-6);
        assert_eq!(meta.best_val_loss, Some(0.4));
        assert_eq!(meta.seed, 123);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        // Create dummy parameters
        let mut params: HashMap<Rc<str>, Array> = HashMap::new();
        params.insert(
            Rc::from("test_param"),
            Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]),
        );

        // Create metadata
        let metadata = CheckpointMetadata::new(50, 1, 0.3, 1e-4);

        // Save checkpoint
        let saved_path = manager.save_checkpoint(&params, &metadata, false).unwrap();
        assert!(saved_path.exists());

        // Load checkpoint
        let (loaded_params, loaded_meta) = CheckpointManager::load_checkpoint(&saved_path).unwrap();

        assert!(loaded_params.contains_key("test_param"));
        assert_eq!(loaded_meta.step, 50);
        assert_eq!(loaded_meta.epoch, 1);
    }

    #[test]
    fn test_checkpoint_latest() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        // Save multiple checkpoints
        let mut params: HashMap<Rc<str>, Array> = HashMap::new();
        params.insert(Rc::from("p1"), Array::from_f32(1.0));

        for step in [10, 20, 30] {
            let meta = CheckpointMetadata::new(step, 0, 0.5, 1e-4);
            manager.save_checkpoint(&params, &meta, false).unwrap();
        }

        // Load latest
        let (_, meta) = manager.load_latest().unwrap().unwrap();
        assert_eq!(meta.step, 30);
    }

    #[test]
    fn test_checkpoint_best() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let mut params: HashMap<Rc<str>, Array> = HashMap::new();
        params.insert(Rc::from("p1"), Array::from_f32(1.0));

        let meta = CheckpointMetadata::new(100, 2, 0.3, 1e-4).with_best_val_loss(0.25);

        // Save as best
        manager.save_checkpoint(&params, &meta, true).unwrap();

        // Load best
        let (_, loaded_meta) = manager.load_best().unwrap().unwrap();
        assert_eq!(loaded_meta.step, 100);
        assert_eq!(loaded_meta.best_val_loss, Some(0.25));
    }

    #[test]
    fn test_checkpoint_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path())
            .unwrap()
            .with_max_checkpoints(3);

        let mut params: HashMap<Rc<str>, Array> = HashMap::new();
        params.insert(Rc::from("p1"), Array::from_f32(1.0));

        // Save 5 checkpoints
        for step in [10, 20, 30, 40, 50] {
            let meta = CheckpointMetadata::new(step, 0, 0.5, 1e-4);
            manager.save_checkpoint(&params, &meta, false).unwrap();
        }

        // Should only have 3 checkpoints
        let checkpoints = manager.list_checkpoints().unwrap();
        assert_eq!(checkpoints.len(), 3);

        // Should have kept the most recent ones
        let steps: Vec<usize> = checkpoints.iter().map(|(s, _)| *s).collect();
        assert_eq!(steps, vec![30, 40, 50]);
    }
}
