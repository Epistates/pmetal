//! Lazy tensor loading for memory-efficient model merging.
//!
//! This module provides a lazy loading interface that loads tensors on-demand
//! rather than loading entire models into memory. This is critical for merging
//! large models on memory-constrained macOS devices.
//!
//! # Zero-Copy Loading
//!
//! For GPU-accelerated merging, the module supports zero-copy tensor access
//! via [`ZeroCopyLoader`]. This avoids intermediate copies by:
//!
//! 1. Memory-mapping safetensors files (via mmap)
//! 2. Providing raw pointers for Metal buffer creation
//! 3. Keeping data in GPU-accessible unified memory
//!
//! ```ignore
//! let loader = ZeroCopyLoader::new("model/")?;
//! let ptr = loader.tensor_ptr("model.layers.0.weight")?;
//! let view = unsafe { metal_buffer_from_ptr(&ctx, ptr, len)? };
//! ```

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

        if path.is_file() && path.extension().is_some_and(|e| e == "safetensors") {
            safetensor_files.push(path.clone());
        } else if path.is_dir() {
            for entry in std::fs::read_dir(&path)? {
                let entry = entry?;
                let file_path = entry.path();
                if file_path.extension().is_some_and(|e| e == "safetensors") {
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

// =============================================================================
// Zero-Copy Loading
// =============================================================================

/// Trait extension for zero-copy tensor access.
///
/// Loaders implementing this trait can provide raw pointers to tensor data
/// without intermediate copies, enabling direct Metal buffer creation.
pub trait ZeroCopyTensorLoader: TensorLoader {
    /// Get raw pointer to tensor data.
    ///
    /// Returns (pointer, length in bytes, dtype) for the tensor.
    ///
    /// # Safety
    /// The returned pointer is only valid while the loader is alive.
    /// The caller must ensure proper synchronization for GPU access.
    fn tensor_ptr(&self, name: &str) -> Result<TensorPtr>;

    /// Check if a tensor can be accessed zero-copy.
    ///
    /// Returns false if the tensor requires conversion (e.g., bf16 to f32).
    fn supports_zero_copy(&self, name: &str) -> Result<bool>;
}

/// Raw pointer to tensor data for zero-copy access.
#[derive(Debug)]
pub struct TensorPtr {
    /// Pointer to the raw data.
    pub ptr: *const u8,
    /// Length in bytes.
    pub len: usize,
    /// Data type.
    pub dtype: safetensors::Dtype,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
}

impl TensorPtr {
    /// Get number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get pointer as f32 slice (unsafe, must check dtype first).
    ///
    /// # Safety
    /// Caller must ensure dtype is F32.
    #[allow(unsafe_code)]
    pub unsafe fn as_f32_ptr(&self) -> *const f32 {
        self.ptr as *const f32
    }

    /// Get pointer as f16 slice (unsafe, must check dtype first).
    ///
    /// # Safety
    /// Caller must ensure dtype is F16.
    #[allow(unsafe_code)]
    pub unsafe fn as_f16_ptr(&self) -> *const half::f16 {
        self.ptr as *const half::f16
    }
}

// SAFETY: TensorPtr is just a view into data owned by the loader.
// It can be sent between threads as long as the loader is alive.
#[allow(unsafe_code)]
unsafe impl Send for TensorPtr {}
#[allow(unsafe_code)]
unsafe impl Sync for TensorPtr {}

/// Memory-mapped loader for zero-copy tensor access.
///
/// Uses mmap to map safetensors files directly into memory, enabling
/// zero-copy Metal buffer creation on Apple Silicon unified memory.
#[derive(Debug)]
pub struct ZeroCopyLoader {
    /// Path to the model directory.
    path: PathBuf,
    /// Memory-mapped files.
    mmaps: Vec<(PathBuf, memmap2::Mmap)>,
    /// Mapping from tensor name to (file index, offset, length, dtype, shape).
    tensor_info: HashMap<String, TensorLocation>,
}

/// Location of a tensor within a memory-mapped file.
#[derive(Debug, Clone)]
struct TensorLocation {
    /// Index into mmaps array.
    file_idx: usize,
    /// Byte offset within the file.
    offset: usize,
    /// Length in bytes.
    len: usize,
    /// Data type.
    dtype: safetensors::Dtype,
    /// Shape.
    shape: Vec<usize>,
}

impl ZeroCopyLoader {
    /// Create a new zero-copy loader for a model directory.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Find all safetensors files
        let mut safetensor_files = Vec::new();

        if path.is_file() && path.extension().is_some_and(|e| e == "safetensors") {
            safetensor_files.push(path.clone());
        } else if path.is_dir() {
            for entry in std::fs::read_dir(&path)? {
                let entry = entry?;
                let file_path = entry.path();
                if file_path.extension().is_some_and(|e| e == "safetensors") {
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

        safetensor_files.sort();

        info!(
            "Memory-mapping {} safetensors files from {:?}",
            safetensor_files.len(),
            path
        );

        // Memory-map files and build tensor index
        let mut mmaps = Vec::new();
        let mut tensor_info = HashMap::new();

        for (idx, file_path) in safetensor_files.into_iter().enumerate() {
            debug!("Memory-mapping {:?}", file_path);

            let file = std::fs::File::open(&file_path)?;
            // SAFETY: The file is opened read-only and we maintain the mmap for
            // the lifetime of this loader.
            #[allow(unsafe_code)]
            let mmap = unsafe { memmap2::Mmap::map(&file)? };

            // Parse safetensors header to get tensor locations
            let tensors = SafeTensors::deserialize(&mmap)?;
            for name in tensors.names() {
                let tensor = tensors.tensor(name)?;
                let data = tensor.data();

                // Calculate offset from mmap base
                let base_ptr = mmap.as_ptr() as usize;
                let data_ptr = data.as_ptr() as usize;
                let offset = data_ptr - base_ptr;

                tensor_info.insert(
                    name.to_string(),
                    TensorLocation {
                        file_idx: idx,
                        offset,
                        len: data.len(),
                        dtype: tensor.dtype(),
                        shape: tensor.shape().to_vec(),
                    },
                );
            }

            mmaps.push((file_path, mmap));
        }

        info!("Indexed {} tensors for zero-copy access", tensor_info.len());

        Ok(Self {
            path,
            mmaps,
            tensor_info,
        })
    }

    /// Get the model path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check if all tensors are in a format suitable for direct GPU access.
    ///
    /// Returns true if all tensors are f32 or f16 (no bf16 conversion needed).
    pub fn all_gpu_compatible(&self) -> bool {
        self.tensor_info
            .values()
            .all(|loc| matches!(loc.dtype, safetensors::Dtype::F32 | safetensors::Dtype::F16))
    }
}

impl TensorLoader for ZeroCopyLoader {
    fn tensor_names(&self) -> Vec<String> {
        self.tensor_info.keys().cloned().collect()
    }

    fn load_tensor(&self, name: &str) -> Result<Array> {
        let loc = self
            .tensor_info
            .get(name)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))?;

        let (_, mmap) = &self.mmaps[loc.file_idx];
        let data = &mmap[loc.offset..loc.offset + loc.len];

        let shape: Vec<i32> = loc.shape.iter().map(|&s| s as i32).collect();

        let array = match loc.dtype {
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
        let loc = self
            .tensor_info
            .get(name)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))?;
        Ok(loc.shape.clone())
    }

    fn tensor_dtype(&self, name: &str) -> Result<safetensors::Dtype> {
        let loc = self
            .tensor_info
            .get(name)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))?;
        Ok(loc.dtype)
    }
}

impl ZeroCopyTensorLoader for ZeroCopyLoader {
    #[allow(unsafe_code)]
    fn tensor_ptr(&self, name: &str) -> Result<TensorPtr> {
        let loc = self
            .tensor_info
            .get(name)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))?;

        let (_, mmap) = &self.mmaps[loc.file_idx];
        let ptr = unsafe { mmap.as_ptr().add(loc.offset) };

        Ok(TensorPtr {
            ptr,
            len: loc.len,
            dtype: loc.dtype,
            shape: loc.shape.clone(),
        })
    }

    fn supports_zero_copy(&self, name: &str) -> Result<bool> {
        let loc = self
            .tensor_info
            .get(name)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))?;

        // Only f32 and f16 can be used zero-copy with Metal
        // bf16 requires conversion on Apple Silicon
        Ok(matches!(
            loc.dtype,
            safetensors::Dtype::F32 | safetensors::Dtype::F16
        ))
    }
}

// =============================================================================
// Model Sources
// =============================================================================

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

    #[test]
    fn test_tensor_ptr_num_elements() {
        let ptr = TensorPtr {
            ptr: std::ptr::null(),
            len: 1024,
            dtype: safetensors::Dtype::F32,
            shape: vec![4, 8, 32],
        };

        assert_eq!(ptr.num_elements(), 4 * 8 * 32);
    }

    #[test]
    fn test_tensor_ptr_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TensorPtr>();
    }

    #[test]
    fn test_zero_copy_loader_missing_files() {
        let result = ZeroCopyLoader::new("/nonexistent/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_copy_tensor_loader_trait() {
        // Verify ZeroCopyLoader implements ZeroCopyTensorLoader
        fn assert_zero_copy<T: ZeroCopyTensorLoader>() {}
        assert_zero_copy::<ZeroCopyLoader>();
    }

    #[test]
    fn test_zero_copy_loader_empty_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = ZeroCopyLoader::new(temp_dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No safetensors files found"));
    }

    #[test]
    fn test_zero_copy_loader_with_safetensors_file() {
        use safetensors::serialize;
        use std::io::Write;

        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.safetensors");

        // Create a minimal safetensors file
        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let metadata = std::collections::HashMap::from([(
            "test_tensor".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &tensor_bytes)
                .unwrap(),
        )]);

        let serialized =
            serialize(metadata.iter().map(|(k, v)| (k.as_str(), v.clone())), &None).unwrap();

        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(&serialized).unwrap();
        drop(file);

        // Now test the loader
        let loader = ZeroCopyLoader::new(&file_path).unwrap();
        assert_eq!(loader.tensor_names().len(), 1);
        assert!(loader.tensor_names().contains(&"test_tensor".to_string()));

        // Test tensor_ptr
        let ptr = loader.tensor_ptr("test_tensor").unwrap();
        assert_eq!(ptr.num_elements(), 4);
        assert!(matches!(ptr.dtype, safetensors::Dtype::F32));

        // Test load_tensor
        let array = loader.load_tensor("test_tensor").unwrap();
        let slice: Vec<f32> = array.as_slice().to_vec();
        assert_eq!(slice, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
