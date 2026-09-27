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

use pmetal_bridge::compat::Array;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};
use zerocopy::FromBytes;

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
                let floats: &[f32] =
                    <[f32]>::ref_from_bytes(data).expect("safetensors data aligned");
                Array::from_f32_slice(floats, &shape)
            }
            safetensors::Dtype::F16 => {
                let halfs: &[half::f16] =
                    <[half::f16]>::ref_from_bytes(data).expect("safetensors data aligned");
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Array::from_f32_slice(&floats, &shape)
            }
            safetensors::Dtype::BF16 => {
                let halfs: &[half::bf16] =
                    <[half::bf16]>::ref_from_bytes(data).expect("safetensors data aligned");
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Array::from_f32_slice(&floats, &shape)
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
    /// The returned [`TensorPtr`] borrows from `self`, ensuring the pointer
    /// remains valid for the lifetime of the borrow.
    fn tensor_ptr(&self, name: &str) -> Result<TensorPtr<'_>>;

    /// Check if a tensor can be accessed zero-copy.
    ///
    /// Returns false if the tensor requires conversion (e.g., bf16 to f32).
    fn supports_zero_copy(&self, name: &str) -> Result<bool>;
}

/// Raw pointer to tensor data for zero-copy access.
///
/// The lifetime `'a` ties this pointer to the [`ZeroCopyLoader`] that created it,
/// ensuring the underlying memory-mapped data remains valid.
#[derive(Debug)]
pub struct TensorPtr<'a> {
    /// Pointer to the raw data.
    pub ptr: *const u8,
    /// Length in bytes.
    pub len: usize,
    /// Data type.
    pub dtype: safetensors::Dtype,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Ties lifetime to the owning loader's memory-mapped data.
    _lifetime: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> TensorPtr<'a> {
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

// SAFETY: TensorPtr borrows from memory-mapped data (&'a [u8]) which is
// Send+Sync. The raw pointer points into this borrowed slice, so it is
// safe to send across threads as long as the borrow is live.
#[allow(unsafe_code)]
unsafe impl Send for TensorPtr<'_> {}
#[allow(unsafe_code)]
unsafe impl Sync for TensorPtr<'_> {}

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

    /// Advise the OS that this mmap will be read sequentially.
    ///
    /// Calling this before a sequential scan improves prefetch behaviour and reduces
    /// page-fault overhead on large models.  Silently ignores errors — the hint is
    /// best-effort and non-critical.
    pub fn advise_sequential(&self) {
        for (_, mmap) in &self.mmaps {
            let _ = mmap.advise(memmap2::Advice::Sequential);
        }
    }

    /// Reset the mmap advice to the OS default (normal).
    ///
    /// Call after completing the sequential scan so cached pages can be evicted.
    pub fn advise_dontneed(&self) {
        for (_, mmap) in &self.mmaps {
            let _ = mmap.advise(memmap2::Advice::Normal);
        }
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
                let floats: &[f32] =
                    <[f32]>::ref_from_bytes(data).expect("safetensors data aligned");
                Array::from_f32_slice(floats, &shape)
            }
            safetensors::Dtype::F16 => {
                let halfs: &[half::f16] =
                    <[half::f16]>::ref_from_bytes(data).expect("safetensors data aligned");
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Array::from_f32_slice(&floats, &shape)
            }
            safetensors::Dtype::BF16 => {
                let halfs: &[half::bf16] =
                    <[half::bf16]>::ref_from_bytes(data).expect("safetensors data aligned");
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Array::from_f32_slice(&floats, &shape)
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
    fn tensor_ptr(&self, name: &str) -> Result<TensorPtr<'_>> {
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
            _lifetime: std::marker::PhantomData,
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

                let client = hf_hub::HFClientSync::new()?;
                let (owner, name) = hf_hub::split_id(repo_id);
                let repo = client.model(owner, name);
                let get = |filename: &str| {
                    repo.download_file()
                        .filename(filename)
                        .maybe_revision(revision.clone())
                        .send()
                };

                // Download all safetensors files
                let files = repo
                    .info()
                    .maybe_revision(revision.clone())
                    .send()?
                    .siblings
                    .unwrap_or_default();
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
                let first = get(&safetensor_files[0].rfilename)?;
                let model_dir = first
                    .parent()
                    .ok_or_else(|| {
                        MergeError::ModelLoad(format!(
                            "Failed to get parent directory of {:?}",
                            first
                        ))
                    })?
                    .to_path_buf();

                // Download remaining files
                for file in &safetensor_files[1..] {
                    let _ = get(&file.rfilename)?;
                }

                Ok(Box::new(SafetensorsLoader::new(model_dir)?))
            }
        }
    }
}

/// Detect a tied lm_head ↔ embed_tokens pair on disk. HuggingFace models
/// signal this in `config.json` via `tie_word_embeddings: true` and
/// (importantly) ship only one of the two tensors in the safetensors —
/// usually `model.embed_tokens.weight`, sometimes `lm_head.weight`. When
/// the merge target is tied, the writer should emit the canonical name
/// once and let the consumer's `from_pretrained()` re-tie at load time.
///
/// Returns `Some((canonical_name, alias_name))` if a tied pair is detected
/// and `alias_name` should be skipped on save; otherwise `None`. The
/// canonical choice favors the embedding name because that's what HF
/// tries first when re-tying.
pub fn detect_tied_embeddings(
    model_dir: &std::path::Path,
    available_names: &std::collections::HashSet<String>,
) -> Option<(String, String)> {
    // Read `tie_word_embeddings` from config.json. Absent / false → no tie.
    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let tied = value
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !tied {
        return None;
    }

    // Common variants. We bias toward the form HF saves under.
    let embed = available_names
        .iter()
        .find(|n| n.ends_with("embed_tokens.weight"))
        .cloned();
    let head = available_names
        .iter()
        .find(|n| n.ends_with("lm_head.weight"))
        .cloned();

    match (embed, head) {
        (Some(e), Some(h)) => {
            // Both present → keep `embed_tokens`, drop the `lm_head` alias.
            Some((e, h))
        }
        // Only one present → already canonical, no work to do.
        _ => None,
    }
}

/// Copy every non-weight, non-index sidecar file from `src_dir` to `dst_dir`.
///
/// Preserves `config.json`, `tokenizer.json`, `tokenizer_config.json`,
/// `special_tokens_map.json`, `generation_config.json`, and any other
/// metadata HF loaders expect to find next to the safetensors weights.
/// Skips:
///   * `*.safetensors` — these are written by [`TensorWriter`].
///   * `model.safetensors.index.json` — same reason.
///
/// Returns an `Io` error on any copy failure so partial directories surface
/// loudly instead of silently shipping a half-merged model.
pub fn copy_side_files(src_dir: &Path, dst_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dst_dir).map_err(|e| {
        MergeError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to create {}: {}", dst_dir.display(), e),
        ))
    })?;

    for entry in std::fs::read_dir(src_dir)? {
        let entry = entry?;
        let src = entry.path();
        if !src.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.ends_with(".safetensors") || name_str == "model.safetensors.index.json" {
            continue;
        }
        let dst = dst_dir.join(&name);
        std::fs::copy(&src, &dst).map_err(|e| {
            MergeError::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "failed to copy {} to {}: {}",
                    src.display(),
                    dst.display(),
                    e
                ),
            ))
        })?;
        debug!("copied side file {}", name_str);
    }

    Ok(())
}

/// Patch the `torch_dtype` field in a `config.json` next to the merged weights
/// so HF loaders apply the right cast on load. Best-effort: silently no-ops if
/// the file is missing, malformed, or doesn't contain the field.
pub fn patch_config_dtype(dst_dir: &Path, dtype: safetensors::Dtype) -> Result<()> {
    let path = dst_dir.join("config.json");
    if !path.exists() {
        return Ok(());
    }
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };
    let mut value: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    let label = match dtype {
        safetensors::Dtype::F32 => "float32",
        safetensors::Dtype::F16 => "float16",
        safetensors::Dtype::BF16 => "bfloat16",
        _ => return Ok(()),
    };
    if let Some(map) = value.as_object_mut() {
        map.insert(
            "torch_dtype".to_string(),
            serde_json::Value::String(label.to_string()),
        );
        let pretty = serde_json::to_string_pretty(&value)?;
        std::fs::write(&path, pretty)?;
    }
    Ok(())
}

/// Parse a dtype string ("float16" / "bfloat16" / "float32" and common aliases)
/// into a [`safetensors::Dtype`]. Case-insensitive.
pub fn parse_dtype(s: &str) -> Result<safetensors::Dtype> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "fp32" | "float32" => Ok(safetensors::Dtype::F32),
        "f16" | "fp16" | "float16" | "half" => Ok(safetensors::Dtype::F16),
        "bf16" | "bfloat16" => Ok(safetensors::Dtype::BF16),
        other => Err(MergeError::UnsupportedDtype(other.to_string())),
    }
}

/// Element size in bytes for the supported writer dtypes.
fn dtype_elem_bytes(dt: safetensors::Dtype) -> usize {
    match dt {
        safetensors::Dtype::F32 => 4,
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
        _ => 4,
    }
}

/// Materialize an f32 slice as packed bytes in the requested target dtype.
/// Used by [`TensorWriter`] on the save path.
fn pack_f32_to_dtype(src: &[f32], dt: safetensors::Dtype) -> Vec<u8> {
    match dt {
        safetensors::Dtype::F32 => {
            // Reinterpret as little-endian bytes.
            src.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        safetensors::Dtype::F16 => src
            .iter()
            .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
            .collect(),
        safetensors::Dtype::BF16 => src
            .iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
            .collect(),
        // Other dtypes are not supported by the writer; caller should reject upstream.
        _ => src.iter().flat_map(|f| f.to_le_bytes()).collect(),
    }
}

/// Writer for saving merged tensors.
pub struct TensorWriter {
    /// Output path.
    output_path: PathBuf,
    /// Accumulated tensors for current shard. Stored as packed bytes already in
    /// the target dtype so `flush_shard` is a thin marshal step.
    current_shard: HashMap<String, (Vec<i32>, Vec<u8>)>,
    /// Current shard size in bytes.
    current_size: usize,
    /// Maximum shard size (default 5GB).
    max_shard_size: usize,
    /// Number of shards written.
    shard_count: usize,
    /// Target on-disk dtype.
    target_dtype: safetensors::Dtype,
}

impl TensorWriter {
    /// Create a new tensor writer with default dtype `F16` (legacy default).
    /// Prefer [`TensorWriter::with_dtype`] for new call-sites.
    pub fn new(output_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_dtype(output_path, safetensors::Dtype::F16)
    }

    /// Create a new tensor writer with an explicit target dtype.
    pub fn with_dtype(
        output_path: impl AsRef<Path>,
        target_dtype: safetensors::Dtype,
    ) -> Result<Self> {
        let output_path = output_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_path)?;

        // Reject unsupported dtypes early so we never accumulate a shard we
        // cannot serialize.
        match target_dtype {
            safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16 => {}
            other => {
                return Err(MergeError::UnsupportedDtype(format!(
                    "TensorWriter does not support {:?}",
                    other
                )));
            }
        }

        Ok(Self {
            output_path,
            current_shard: HashMap::new(),
            current_size: 0,
            max_shard_size: 5 * 1024 * 1024 * 1024, // 5GB
            shard_count: 0,
            target_dtype,
        })
    }

    /// Set maximum shard size.
    pub fn with_max_shard_size(mut self, size: usize) -> Self {
        self.max_shard_size = size;
        self
    }

    /// Target on-disk dtype for this writer.
    pub fn target_dtype(&self) -> safetensors::Dtype {
        self.target_dtype
    }

    /// Write a tensor. The array is materialized via f32 (the canonical
    /// in-memory dtype during merging) and then packed into bytes at the
    /// writer's target dtype.
    pub fn write_tensor(&mut self, name: &str, tensor: &Array) -> Result<()> {
        let mut tensor = tensor.as_dtype(pmetal_bridge::compat::Dtype::Float32.as_i32());
        let shape = tensor.shape().to_vec();
        let n = shape.iter().map(|&s| s as usize).product();
        let data: Vec<f32> = tensor.to_f32_vec(n).unwrap_or_default();
        let bytes = pack_f32_to_dtype(&data, self.target_dtype);
        let size = bytes.len();

        // Check if we need to flush current shard
        if self.current_size + size > self.max_shard_size && !self.current_shard.is_empty() {
            self.flush_shard()?;
        }

        self.current_shard.insert(name.to_string(), (shape, bytes));
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
        info!(
            "Writing shard: {:?} (dtype={:?})",
            shard_path, self.target_dtype
        );

        // Sanity: every payload's length must be a multiple of the element size.
        let elem = dtype_elem_bytes(self.target_dtype);
        let tensors: Vec<_> = self
            .current_shard
            .iter()
            .map(|(name, (shape, bytes))| {
                debug_assert!(
                    bytes.len() % elem == 0,
                    "tensor {} byte length {} not aligned to element size {}",
                    name,
                    bytes.len(),
                    elem
                );
                let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                let tensor_view = safetensors::tensor::TensorView::new(
                    self.target_dtype,
                    shape,
                    bytes.as_slice(),
                )
                .map_err(|e| {
                    MergeError::ModelLoad(format!("Failed to create TensorView: {}", e))
                })?;
                Ok((name.as_str(), tensor_view))
            })
            .collect::<Result<Vec<_>>>()?;

        safetensors::serialize_to_file(tensors, None, &shard_path)?;

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
            _lifetime: std::marker::PhantomData,
        };

        assert_eq!(ptr.num_elements(), 4 * 8 * 32);
    }

    #[test]
    fn test_tensor_ptr_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TensorPtr<'static>>();
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
    fn parse_dtype_aliases() {
        use safetensors::Dtype;
        assert_eq!(parse_dtype("float16").unwrap(), Dtype::F16);
        assert_eq!(parse_dtype("FP16").unwrap(), Dtype::F16);
        assert_eq!(parse_dtype("half").unwrap(), Dtype::F16);
        assert_eq!(parse_dtype("bfloat16").unwrap(), Dtype::BF16);
        assert_eq!(parse_dtype("BF16").unwrap(), Dtype::BF16);
        assert_eq!(parse_dtype("float32").unwrap(), Dtype::F32);
        assert_eq!(parse_dtype("fp32").unwrap(), Dtype::F32);
        assert!(parse_dtype("float8").is_err());
        assert!(parse_dtype("").is_err());
    }

    /// Round-trip a tensor through every supported writer dtype and verify the
    /// on-disk representation reports the right safetensors::Dtype and decodes
    /// back to the original f32 values within each dtype's precision.
    #[test]
    fn dtype_round_trip_f32_f16_bf16() {
        use pmetal_bridge::compat::Array;
        use safetensors::Dtype;

        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.125 - 2.0).collect();
        let shape: Vec<i32> = vec![4, 8];

        for (label, dtype, tol) in [
            ("float32", Dtype::F32, 0.0_f32),
            ("float16", Dtype::F16, 1e-3_f32),
            ("bfloat16", Dtype::BF16, 1e-2_f32),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut writer = TensorWriter::with_dtype(dir.path(), dtype).unwrap();
            let tensor = Array::from_f32_slice(&values, &shape);
            writer.write_tensor("t", &tensor).unwrap();
            let out_dir = writer.finalize().unwrap();

            // Reload via SafetensorsLoader and confirm dtype + values.
            let loader = SafetensorsLoader::new(&out_dir).unwrap();
            assert_eq!(
                loader.tensor_dtype("t").unwrap(),
                dtype,
                "wrong on-disk dtype for {}",
                label
            );

            let mut arr = loader.load_tensor("t").unwrap();
            let got = arr.to_f32_vec(values.len()).unwrap();
            for (i, (a, b)) in values.iter().zip(got.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol,
                    "{}: idx {} mismatch {} vs {} (tol {})",
                    label,
                    i,
                    a,
                    b,
                    tol
                );
            }
        }
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
            serialize(metadata.iter().map(|(k, v)| (k.as_str(), v.clone())), None).unwrap();

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
        let mut array = loader.load_tensor("test_tensor").unwrap();
        let slice = array.to_f32_vec(4).unwrap();
        assert_eq!(slice, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
