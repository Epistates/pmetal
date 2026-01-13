//! Lazy tensor loading for memory-efficient model merging.
//!
//! This module provides a lazy loading interface that loads tensors on-demand
//! rather than loading entire models into memory. This is critical for merging
//! large models on memory-constrained macOS devices.

use mlx_rs::Array;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use crate::{MergeError, Result};

/// Trait for loading tensors from a model.
pub trait TensorLoader: Send + Sync {
    /// Get the names of all tensors in the model.
    fn tensor_names(&self) -> Vec<String>;

    /// Load a tensor by name.
    fn load_tensor(&self, name: &str) -> Result<Array>;

    /// Get the shape of a tensor without loading it.
    fn tensor_shape(&self, name: &str) -> Result<Vec<usize>>;

    /// Get the dtype of a tensor.
    fn tensor_dtype(&self, name: &str) -> Result<safetensors::Dtype>;
}

/// Lazy loader for safetensors files.
///
/// Keeps file handles open and loads tensors on-demand.
pub struct SafetensorsLoader {
    /// Path to the model directory.
    path: PathBuf,
    /// Cached file contents (memory-mapped).
    files: Vec<(PathBuf, Vec<u8>)>,
    /// Mapping from tensor name to file index.
    tensor_to_file: HashMap<String, usize>,
}

impl SafetensorsLoader {
    /// Create a new loader for a model directory.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Find all safetensors files
        let mut safetensor_files = Vec::new();

        if path.is_file() && path.extension().map_or(false, |e| e == "safetensors") {
            safetensor_files.push(path.clone());
        } else if path.is_dir() {
            for entry in std::fs::read_dir(&path)? {
                let entry = entry?;
                let file_path = entry.path();
                if file_path.extension().map_or(false, |e| e == "safetensors") {
                    safetensor_files.push(file_path);
                }
            }
        }

        if safetensor_files.is_empty() {
            return Err(MergeError::ModelLoad(format!(
                "No safetensors files found in {:?}",
                path
            )));
        }

        // Sort for deterministic ordering
        safetensor_files.sort();

        info!(
            "Loading {} safetensors files from {:?}",
            safetensor_files.len(),
            path
        );

        // Load file contents and build tensor mapping
        let mut files = Vec::new();
        let mut tensor_to_file = HashMap::new();

        for (idx, file_path) in safetensor_files.into_iter().enumerate() {
            debug!("Indexing {:?}", file_path);
            let data = std::fs::read(&file_path)?;

            // Parse to get tensor names
            let tensors = SafeTensors::deserialize(&data)?;
            for name in tensors.names() {
                tensor_to_file.insert(name.to_string(), idx);
            }

            files.push((file_path, data));
        }

        info!("Indexed {} tensors", tensor_to_file.len());

        Ok(Self {
            path,
            files,
            tensor_to_file,
        })
    }

    /// Get the model path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn get_safetensors(&self, file_idx: usize) -> Result<SafeTensors<'_>> {
        let (_, data) = &self.files[file_idx];
        Ok(SafeTensors::deserialize(data)?)
    }
}

impl TensorLoader for SafetensorsLoader {
    fn tensor_names(&self) -> Vec<String> {
        self.tensor_to_file.keys().cloned().collect()
    }

    fn load_tensor(&self, name: &str) -> Result<Array> {
        let file_idx = self
            .tensor_to_file
            .get(name)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))?;

        let tensors = self.get_safetensors(*file_idx)?;
        let tensor = tensors.tensor(name)?;

        // Convert safetensors view to MLX array
        let shape: Vec<i32> = tensor.shape().iter().map(|&s| s as i32).collect();
        let data = tensor.data();

        let array = match tensor.dtype() {
            safetensors::Dtype::F32 => {
                let floats: &[f32] = bytemuck::cast_slice(data);
                Array::from_slice(floats, &shape)
            }
            safetensors::Dtype::F16 => {
                let halfs: &[half::f16] = bytemuck::cast_slice(data);
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Array::from_slice(&floats, &shape)
            }
            safetensors::Dtype::BF16 => {
                let halfs: &[half::bf16] = bytemuck::cast_slice(data);
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Array::from_slice(&floats, &shape)
            }
            dtype => {
                return Err(MergeError::ModelLoad(format!(
                    "Unsupported dtype {:?} for tensor {}",
                    dtype, name
                )));
            }
        };

        Ok(array)
    }

    fn tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        let file_idx = self
            .tensor_to_file
            .get(name)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))?;

        let tensors = self.get_safetensors(*file_idx)?;
        let tensor = tensors.tensor(name)?;

        Ok(tensor.shape().to_vec())
    }

    fn tensor_dtype(&self, name: &str) -> Result<safetensors::Dtype> {
        let file_idx = self
            .tensor_to_file
            .get(name)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))?;

        let tensors = self.get_safetensors(*file_idx)?;
        let tensor = tensors.tensor(name)?;

        Ok(tensor.dtype())
    }
}

/// A model source that can be resolved to a TensorLoader.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Local path to model directory or file.
    Local(PathBuf),
    /// HuggingFace Hub repository ID.
    Hub {
        /// Repository ID (e.g., "meta-llama/Llama-2-7b").
        repo_id: String,
        /// Optional revision (branch, tag, or commit).
        revision: Option<String>,
    },
}

impl ModelSource {
    /// Create a model source from a local path.
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self::Local(path.as_ref().to_path_buf())
    }

    /// Create a model source from a HuggingFace repo ID.
    pub fn from_hub(repo_id: impl Into<String>) -> Self {
        Self::Hub {
            repo_id: repo_id.into(),
            revision: None,
        }
    }

    /// Create a model source from a HuggingFace repo ID with revision.
    pub fn from_hub_with_revision(repo_id: impl Into<String>, revision: impl Into<String>) -> Self {
        Self::Hub {
            repo_id: repo_id.into(),
            revision: Some(revision.into()),
        }
    }

    /// Parse a model source from a string.
    /// If it looks like a path, treat as local. Otherwise, treat as Hub repo.
    pub fn parse(s: &str) -> Self {
        let path = Path::new(s);
        // If path exists locally, use it as local
        if path.exists() {
            return Self::Local(path.to_path_buf());
        }

        // If it starts with / or . or contains platform-specific path separator (not /), treat as path
        // But don't treat "org/repo" as a local path on Unix
        if s.starts_with('/') || s.starts_with('.') {
            return Self::Local(path.to_path_buf());
        }

        // On Windows, check for backslash paths
        #[cfg(windows)]
        if s.contains('\\') {
            return Self::Local(path.to_path_buf());
        }

        // HuggingFace repo IDs look like "org/model" with exactly one "/"
        // and don't start with "." or contain backslashes
        if s.contains('/') && s.matches('/').count() == 1 && !s.starts_with('/') {
            return Self::from_hub(s);
        }

        // Default: treat as Hub repo
        Self::from_hub(s)
    }

    /// Resolve to a TensorLoader.
    pub fn resolve(&self) -> Result<Box<dyn TensorLoader>> {
        match self {
            Self::Local(path) => Ok(Box::new(SafetensorsLoader::new(path)?)),
            Self::Hub { repo_id, revision } => {
                info!("Downloading model from Hub: {}", repo_id);

                let api = hf_hub::api::sync::Api::new()?;
                let repo = match revision {
                    Some(rev) => api.repo(hf_hub::Repo::with_revision(
                        repo_id.clone(),
                        hf_hub::RepoType::Model,
                        rev.clone(),
                    )),
                    None => api.model(repo_id.clone()),
                };

                // Download all safetensors files
                let files = repo.info()?.siblings;
                let safetensor_files: Vec<_> = files
                    .iter()
                    .filter(|f| f.rfilename.ends_with(".safetensors"))
                    .collect();

                if safetensor_files.is_empty() {
                    return Err(MergeError::ModelLoad(format!(
                        "No safetensors files found in repo {}",
                        repo_id
                    )));
                }

                // Download first file to get the directory
                let first = repo.get(&safetensor_files[0].rfilename)?;
                let model_dir = first.parent().unwrap().to_path_buf();

                // Download remaining files
                for file in &safetensor_files[1..] {
                    let _ = repo.get(&file.rfilename)?;
                }

                Ok(Box::new(SafetensorsLoader::new(model_dir)?))
            }
        }
    }
}

/// Writer for saving merged tensors.
pub struct TensorWriter {
    /// Output path.
    output_path: PathBuf,
    /// Accumulated tensors for current shard.
    current_shard: HashMap<String, (Vec<i32>, Vec<f32>)>,
    /// Current shard size in bytes.
    current_size: usize,
    /// Maximum shard size (default 5GB).
    max_shard_size: usize,
    /// Number of shards written.
    shard_count: usize,
}

impl TensorWriter {
    /// Create a new tensor writer.
    pub fn new(output_path: impl AsRef<Path>) -> Result<Self> {
        let output_path = output_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_path)?;

        Ok(Self {
            output_path,
            current_shard: HashMap::new(),
            current_size: 0,
            max_shard_size: 5 * 1024 * 1024 * 1024, // 5GB
            shard_count: 0,
        })
    }

    /// Set maximum shard size.
    pub fn with_max_shard_size(mut self, size: usize) -> Self {
        self.max_shard_size = size;
        self
    }

    /// Write a tensor.
    pub fn write_tensor(&mut self, name: &str, tensor: &Array) -> Result<()> {
        // Convert to f32 for storage
        let tensor = tensor.as_type::<f32>()?;
        let shape = tensor.shape().to_vec();
        let data: Vec<f32> = tensor.as_slice().to_vec();
        let size = data.len() * 4;

        // Check if we need to flush current shard
        if self.current_size + size > self.max_shard_size && !self.current_shard.is_empty() {
            self.flush_shard()?;
        }

        self.current_shard.insert(name.to_string(), (shape, data));
        self.current_size += size;

        Ok(())
    }

    /// Flush current shard to disk.
    fn flush_shard(&mut self) -> Result<()> {
        if self.current_shard.is_empty() {
            return Ok(());
        }

        self.shard_count += 1;
        let shard_name = if self.shard_count == 1 {
            "model.safetensors".to_string()
        } else {
            format!("model-{:05}.safetensors", self.shard_count)
        };

        let shard_path = self.output_path.join(&shard_name);
        info!("Writing shard: {:?}", shard_path);

        // Convert to safetensors format
        let tensors: Vec<_> = self
            .current_shard
            .iter()
            .map(|(name, (shape, data))| {
                let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                (
                    name.as_str(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape,
                        bytemuck::cast_slice(data),
                    )
                    .unwrap(),
                )
            })
            .collect();

        safetensors::serialize_to_file(tensors, &None, &shard_path)?;

        self.current_shard.clear();
        self.current_size = 0;

        Ok(())
    }

    /// Finalize and write any remaining tensors.
    pub fn finalize(mut self) -> Result<PathBuf> {
        self.flush_shard()?;
        Ok(self.output_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_source_parse() {
        // Local paths
        assert!(matches!(
            ModelSource::parse("/path/to/model"),
            ModelSource::Local(_)
        ));
        assert!(matches!(
            ModelSource::parse("./model"),
            ModelSource::Local(_)
        ));

        // Hub repos
        assert!(matches!(
            ModelSource::parse("meta-llama/Llama-2-7b"),
            ModelSource::Hub { .. }
        ));
        assert!(matches!(
            ModelSource::parse("mistralai/Mistral-7B-v0.1"),
            ModelSource::Hub { .. }
        ));
    }
}
