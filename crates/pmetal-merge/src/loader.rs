//! Lazy tensor loading and sharded writing for model merging.
//!
//! [`SafetensorsLoader`] memory-maps a checkpoint (one file, a directory of
//! shards, or a directory with a `model.safetensors.index.json`) and decodes
//! one tensor to f32 on request, so a merge holds a few tensors at a time
//! however large the models are. [`TensorWriter`] is the other half: it
//! streams merged tensors to disk, rolls shards over at a size limit, writes
//! every file atomically and finishes with the shard index Hugging Face
//! loaders expect.

use pmetal_runtime::compat::Array;
use safetensors::Dtype;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use crate::mapped::MappedFile;
use crate::{MergeError, Result};

/// Trait for loading tensors from a model.
pub trait TensorLoader: Send + Sync {
    /// Get the names of all tensors in the model.
    fn tensor_names(&self) -> Vec<String>;

    /// Whether the model holds a tensor called `name`.
    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_names().iter().any(|n| n == name)
    }

    /// Load a tensor by name, decoded to f32.
    fn load_tensor(&self, name: &str) -> Result<Array>;

    /// Get the shape of a tensor without loading it.
    fn tensor_shape(&self, name: &str) -> Result<Vec<usize>>;

    /// Get the dtype of a tensor.
    fn tensor_dtype(&self, name: &str) -> Result<Dtype>;
}

/// Decode little-endian tensor bytes of `dtype` to f32.
///
/// Reads the bytes without assuming any alignment: a safetensors header may
/// have any length, so tensor data can start at any byte offset.
pub(crate) fn decode_f32(bytes: &[u8], dtype: Dtype, name: &str) -> Result<Vec<f32>> {
    let width = match dtype {
        Dtype::F16 | Dtype::BF16 => 2,
        Dtype::F32 => 4,
        Dtype::F64 => 8,
        other => {
            return Err(MergeError::ModelLoad(format!(
                "Unsupported dtype {other:?} for tensor {name}"
            )));
        }
    };
    if bytes.len() % width != 0 {
        return Err(MergeError::ModelLoad(format!(
            "tensor {name}: {} bytes is not a whole number of {dtype:?} elements",
            bytes.len()
        )));
    }
    Ok(match dtype {
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
            .collect(),
        Dtype::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        _ => unreachable!("dtype checked above"),
    })
}

/// Lazy, memory-mapped loader for a safetensors checkpoint.
///
/// Opening a model maps its files and reads only the headers. A tensor is
/// decoded to an f32 array when [`load_tensor`](TensorLoader::load_tensor) is
/// called, and its pages are released by the operating system as the merge
/// moves on.
pub struct SafetensorsLoader {
    /// Path to the model directory or file.
    path: PathBuf,
    files: Vec<MappedFile>,
    /// Tensor name to the file holding it.
    tensor_to_file: HashMap<String, usize>,
    /// FP8 weight name to the tensor holding its block-wise inverse scales.
    fp8_scales: HashMap<String, String>,
}

/// Weight files named by a `model.safetensors.index.json`, in the index's
/// own spelling, deduplicated.
fn indexed_shards(index_path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(index_path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let map = value
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| {
            MergeError::ModelLoad(format!("{} has no weight_map object", index_path.display()))
        })?;
    let shards: BTreeSet<String> = map
        .values()
        .filter_map(|v| v.as_str().map(str::to_owned))
        .collect();
    Ok(shards.into_iter().collect())
}

impl SafetensorsLoader {
    /// Open a model: a `.safetensors` file, or a directory holding either a
    /// `model.safetensors.index.json` (its shards, and only those) or plain
    /// `.safetensors` files.
    ///
    /// # Errors
    /// No weights found, an index naming a missing shard, an unreadable
    /// header, or one tensor name present in two files.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut weight_files: Vec<PathBuf> = Vec::new();
        if path.is_file() && path.extension().is_some_and(|e| e == "safetensors") {
            weight_files.push(path.clone());
        } else if path.is_dir() {
            let index_path = path.join("model.safetensors.index.json");
            if index_path.is_file() {
                for shard in indexed_shards(&index_path)? {
                    let shard_path = path.join(&shard);
                    if !shard_path.is_file() {
                        return Err(MergeError::ModelLoad(format!(
                            "{} lists shard {shard}, which is missing from {}",
                            index_path.display(),
                            path.display()
                        )));
                    }
                    weight_files.push(shard_path);
                }
            } else {
                for entry in std::fs::read_dir(&path)? {
                    let entry = entry?;
                    // Hidden files are never weights; macOS litters archives and
                    // non-native volumes with `._model.safetensors` metadata stubs.
                    if entry.file_name().to_string_lossy().starts_with('.') {
                        continue;
                    }
                    let file_path = entry.path();
                    if file_path.extension().is_some_and(|e| e == "safetensors") {
                        weight_files.push(file_path);
                    }
                }
            }
        }

        if weight_files.is_empty() {
            return Err(MergeError::ModelLoad(format!(
                "No safetensors files found in {path:?}"
            )));
        }
        weight_files.sort();

        info!(
            "Mapping {} safetensors files from {:?}",
            weight_files.len(),
            path
        );

        let mut files = Vec::with_capacity(weight_files.len());
        let mut tensor_to_file: HashMap<String, usize> = HashMap::new();
        for (idx, file_path) in weight_files.iter().enumerate() {
            debug!("Indexing {:?}", file_path);
            let mapped = MappedFile::open(file_path)?;
            for (name, _) in &mapped.tensors {
                if let Some(&other) = tensor_to_file.get(name) {
                    return Err(MergeError::ModelLoad(format!(
                        "tensor {name} is in both {} and {}; remove one of the files or add a \
                         model.safetensors.index.json that names the shards to use",
                        weight_files[other].display(),
                        file_path.display()
                    )));
                }
                tensor_to_file.insert(name.clone(), idx);
            }
            files.push(mapped);
        }

        // FP8 (E4M3) weights come with `<weight>_scale_inv` tensors: one inverse
        // scale per block. They are part of the weight, so they are consumed
        // when it loads and are not tensors of their own.
        let mut fp8_scales = HashMap::new();
        for (name, &file_idx) in &tensor_to_file {
            let is_fp8 = files[file_idx]
                .tensor_meta(name)
                .is_ok_and(|m| m.dtype == Dtype::F8_E4M3);
            let scale = format!("{name}_scale_inv");
            if is_fp8 && tensor_to_file.contains_key(&scale) {
                fp8_scales.insert(name.clone(), scale);
            }
        }

        info!(
            "Indexed {} tensors ({} FP8 weights with block scales)",
            tensor_to_file.len() - fp8_scales.len(),
            fp8_scales.len()
        );

        Ok(Self {
            path,
            files,
            tensor_to_file,
            fp8_scales,
        })
    }

    /// Get the model path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether `name` is a scale tensor consumed by an FP8 weight.
    fn is_fp8_scale(&self, name: &str) -> bool {
        name.strip_suffix("_scale_inv")
            .is_some_and(|weight| self.fp8_scales.get(weight).is_some_and(|s| s == name))
    }

    /// An FP8 (E4M3) weight dequantized to f32 with its block-wise inverse
    /// scales, `weight = fp8 * scale_inv[block]`; without scales the payload
    /// is just widened.
    fn load_fp8(&self, name: &str, file: &MappedFile, shape: &[i32]) -> Result<Array> {
        let f32_code = pmetal_runtime::compat::Dtype::Float32.as_i32();
        let payload = Array::from_u8_slice(file.tensor_bytes(name)?, shape);
        let dequantized = match self.fp8_scales.get(name) {
            Some(scale_name) => {
                let scale_file = self.locate(scale_name)?;
                let scale_meta = scale_file.tensor_meta(scale_name)?;
                let scale_shape: Vec<i32> = scale_meta.shape.iter().map(|&d| d as i32).collect();
                let scale_values = decode_f32(
                    scale_file.tensor_bytes(scale_name)?,
                    scale_meta.dtype,
                    scale_name,
                )?;
                let scale = Array::from_f32_slice(&scale_values, &scale_shape);
                pmetal_runtime::native_loader::dequantize_fp8_e4m3_scaled_weight(
                    &payload, &scale, f32_code,
                )
                .map_err(|e| MergeError::ModelLoad(format!("tensor {name}: {e}")))?
            }
            None => payload
                .try_from_fp8(f32_code)
                .map_err(|e| MergeError::ModelLoad(format!("tensor {name}: {e}")))?,
        };
        pmetal_runtime::check_unobserved_error()?;
        Ok(dequantized)
    }

    fn locate(&self, name: &str) -> Result<&MappedFile> {
        self.tensor_to_file
            .get(name)
            .map(|&i| &self.files[i])
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))
    }
}

impl TensorLoader for SafetensorsLoader {
    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .tensor_to_file
            .keys()
            .filter(|n| !self.is_fp8_scale(n))
            .cloned()
            .collect();
        names.sort();
        names
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_to_file.contains_key(name) && !self.is_fp8_scale(name)
    }

    fn load_tensor(&self, name: &str) -> Result<Array> {
        let file = self.locate(name)?;
        let meta = file.tensor_meta(name)?;
        let shape = meta
            .shape
            .iter()
            .map(|&d| {
                i32::try_from(d).map_err(|_| {
                    MergeError::ModelLoad(format!("tensor {name}: dimension {d} exceeds i32"))
                })
            })
            .collect::<Result<Vec<i32>>>()?;
        if meta.dtype == Dtype::F8_E4M3 {
            return self.load_fp8(name, file, &shape);
        }
        let values = decode_f32(file.tensor_bytes(name)?, meta.dtype, name)?;
        Ok(Array::from_f32_slice(&values, &shape))
    }

    fn tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.locate(name)?.tensor_meta(name)?.shape.clone())
    }

    fn tensor_dtype(&self, name: &str) -> Result<Dtype> {
        Ok(self.locate(name)?.tensor_meta(name)?.dtype)
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

/// Whether `name` is a weight file in some format, the shard index, or a
/// hidden file: everything a sidecar copy must leave behind, because the
/// merged weights replace the source's and a stale copy next to them would be
/// loaded in their place.
fn is_weight_or_hidden_file(name: &str) -> bool {
    const WEIGHT_SUFFIXES: [&str; 9] = [
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".gguf",
        ".h5",
        ".msgpack",
        ".onnx",
    ];
    name.starts_with('.')
        || name == "model.safetensors.index.json"
        || WEIGHT_SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// Copy every non-weight, non-index sidecar file from `src_dir` to `dst_dir`.
///
/// Preserves `config.json`, `tokenizer.json`, `tokenizer_config.json`,
/// `special_tokens_map.json`, `generation_config.json`, and any other
/// metadata HF loaders expect to find next to the safetensors weights.
/// Skips:
///   * weight files in any format (`*.safetensors`, `*.bin`, `*.gguf`, ...) —
///     the merged weights are written by [`TensorWriter`].
///   * `model.safetensors.index.json` — same reason.
///   * hidden files (`.gitattributes`, ...).
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
        if is_weight_or_hidden_file(&name_str) {
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
        // `torch_dtype` is the long-standing key; newer releases of
        // transformers write `dtype` and read both, so keep an existing one
        // in step.
        let label = serde_json::Value::String(label.to_string());
        if map.contains_key("dtype") {
            map.insert("dtype".to_string(), label.clone());
        }
        map.insert("torch_dtype".to_string(), label);
        let pretty = serde_json::to_string_pretty(&value)?;
        write_atomic(&path, pretty.as_bytes())?;
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

/// The `"dtype"` tag of the safetensors header for the supported targets.
fn dtype_tag(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "F32",
        Dtype::F16 => "F16",
        _ => "BF16",
    }
}

/// Pack `src` as little-endian bytes of `dtype`. Returns the bytes and the
/// number of finite values that overflowed to infinity in the target type.
fn pack_f32_to_dtype(src: &[f32], dtype: Dtype) -> (Vec<u8>, usize) {
    match dtype {
        Dtype::F16 => {
            let mut overflowed = 0;
            let bytes = src
                .iter()
                .flat_map(|&f| {
                    let h = half::f16::from_f32(f);
                    if h.is_infinite() && f.is_finite() {
                        overflowed += 1;
                    }
                    h.to_le_bytes()
                })
                .collect();
            (bytes, overflowed)
        }
        Dtype::BF16 => (
            src.iter()
                .flat_map(|&f| half::bf16::from_f32(f).to_le_bytes())
                .collect(),
            0,
        ),
        _ => (src.iter().flat_map(|f| f.to_le_bytes()).collect(), 0),
    }
}

/// Write `bytes` to `path` all or nothing: to a `.partial` sibling, forced to
/// disk, then renamed into place.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut partial = path.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    let result = (|| -> std::io::Result<()> {
        let mut file = File::create(&partial)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&partial, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    Ok(result?)
}

/// One tensor already appended to the open shard's data file.
struct ShardEntry {
    name: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

/// Bytes reserved at the start of every shard for the length prefix and the
/// header. Payloads stream in behind it, and the header (space-padded to the
/// reserved size, which the format allows) is written over it when the shard
/// closes; a shard whose header outgrows the reserve is copied once instead.
const HEADER_RESERVE: u64 = 1024 * 1024;

/// The shard being written: payloads stream into the partial file, behind the
/// reserved header region, as tensors arrive.
struct OpenShard {
    partial_path: PathBuf,
    data: BufWriter<File>,
    entries: Vec<ShardEntry>,
    size: u64,
}

/// A closed shard, written under a temporary name until [`TensorWriter::finalize`].
struct ClosedShard {
    partial_path: PathBuf,
    names: Vec<String>,
    size: u64,
}

/// Writer for saving merged tensors.
///
/// Tensors are converted to the target dtype and streamed to a scratch file,
/// so memory use is one tensor however large the shard. A shard closes when
/// the next tensor would take it past the size limit; `finalize` renames the
/// shards to `model-00001-of-0000N.safetensors` (or `model.safetensors` for
/// a single shard) and writes `model.safetensors.index.json`. Nothing under a
/// final name exists until `finalize`, and every file is renamed into place,
/// so an interrupted merge leaves no torn model behind.
pub struct TensorWriter {
    output_path: PathBuf,
    open: Option<OpenShard>,
    closed: Vec<ClosedShard>,
    written: HashSet<String>,
    /// Maximum shard size (default 5GB).
    max_shard_size: usize,
    /// Target on-disk dtype.
    target_dtype: Dtype,
    finished: bool,
}

impl TensorWriter {
    /// Create a new tensor writer with default dtype `F16` (legacy default).
    /// Prefer [`TensorWriter::with_dtype`] for new call-sites.
    pub fn new(output_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_dtype(output_path, Dtype::F16)
    }

    /// Create a new tensor writer with an explicit target dtype.
    pub fn with_dtype(output_path: impl AsRef<Path>, target_dtype: Dtype) -> Result<Self> {
        let output_path = output_path.as_ref().to_path_buf();

        // Reject unsupported dtypes early so we never accumulate a shard we
        // cannot serialize.
        match target_dtype {
            Dtype::F32 | Dtype::F16 | Dtype::BF16 => {}
            other => {
                return Err(MergeError::UnsupportedDtype(format!(
                    "TensorWriter does not support {other:?}"
                )));
            }
        }
        std::fs::create_dir_all(&output_path)?;

        Ok(Self {
            output_path,
            open: None,
            closed: Vec::new(),
            written: HashSet::new(),
            max_shard_size: 5 * 1024 * 1024 * 1024,
            target_dtype,
            finished: false,
        })
    }

    /// Set maximum shard size.
    pub fn with_max_shard_size(mut self, size: usize) -> Self {
        self.max_shard_size = size;
        self
    }

    /// Target on-disk dtype for this writer.
    pub fn target_dtype(&self) -> Dtype {
        self.target_dtype
    }

    /// Write a tensor. The array is evaluated as f32 (the canonical dtype
    /// during merging) and packed at the writer's target dtype.
    ///
    /// # Errors
    /// A name written before, a runtime error behind the array, or a finite
    /// value that overflows the target dtype (float16 holds up to 65504).
    pub fn write_tensor(&mut self, name: &str, tensor: &Array) -> Result<()> {
        if !self.written.insert(name.to_string()) {
            return Err(MergeError::InvalidConfig(format!(
                "tensor {name} written twice"
            )));
        }
        let mut tensor = tensor.as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
        let shape: Vec<usize> = tensor.shape().iter().map(|&s| s as usize).collect();
        let n: usize = shape.iter().product();
        let data = tensor.to_f32_vec(n);
        pmetal_runtime::check_unobserved_error()?;
        let data = data.ok_or_else(|| {
            MergeError::ModelLoad(format!("tensor {name}: failed to read {n} values"))
        })?;
        let (bytes, overflowed) = pack_f32_to_dtype(&data, self.target_dtype);
        if overflowed > 0 {
            return Err(MergeError::InvalidConfig(format!(
                "tensor {name}: {overflowed} values overflow {:?}; merge in a wider dtype",
                self.target_dtype
            )));
        }

        let size = bytes.len() as u64;
        let over_limit = self
            .open
            .as_ref()
            .is_some_and(|s| !s.entries.is_empty() && s.size + size > self.max_shard_size as u64);
        if over_limit {
            self.close_shard()?;
        }
        if self.open.is_none() {
            let index = self.closed.len() + 1;
            let partial_path = self
                .output_path
                .join(format!(".merge-shard-{index:05}.safetensors.partial"));
            // Read access too, for the copy when the header outgrows the reserve.
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&partial_path)?;
            let mut data = BufWriter::with_capacity(8 * 1024 * 1024, file);
            data.seek(SeekFrom::Start(HEADER_RESERVE))?;
            self.open = Some(OpenShard {
                partial_path,
                data,
                entries: Vec::new(),
                size: 0,
            });
        }
        let shard = self.open.as_mut().expect("shard opened above");
        shard.data.write_all(&bytes)?;
        shard.entries.push(ShardEntry {
            name: name.to_string(),
            shape,
            start: shard.size,
            end: shard.size + size,
        });
        shard.size += size;
        Ok(())
    }

    /// Close the open shard: assemble header and payloads into a partial
    /// file next to the output.
    fn close_shard(&mut self) -> Result<()> {
        let Some(shard) = self.open.take() else {
            return Ok(());
        };
        let index = self.closed.len() + 1;
        let partial_path = shard.partial_path.clone();

        let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        header.insert(
            "__metadata__".to_string(),
            serde_json::json!({ "format": "pt" }),
        );
        for e in &shard.entries {
            header.insert(
                e.name.clone(),
                serde_json::json!({
                    "dtype": dtype_tag(self.target_dtype),
                    "shape": e.shape,
                    "data_offsets": [e.start, e.end],
                }),
            );
        }
        let mut header_bytes = serde_json::to_vec(&header)?;
        let fits = 8 + header_bytes.len() as u64 <= HEADER_RESERVE;
        if fits {
            header_bytes.resize((HEADER_RESERVE - 8) as usize, b' ');
        } else {
            // Pad to a multiple of 8 so the data section stays 8-byte aligned.
            while header_bytes.len() % 8 != 0 {
                header_bytes.push(b' ');
            }
        }

        let finish = |shard_data: BufWriter<File>| -> std::io::Result<Option<PathBuf>> {
            let mut file = shard_data.into_inner().map_err(|e| e.into_error())?;
            if fits {
                file.seek(SeekFrom::Start(0))?;
                file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
                file.write_all(&header_bytes)?;
                file.sync_all()?;
                return Ok(None);
            }
            // The header outgrew the reserve: copy the payloads behind it.
            let wide = partial_path.with_extension("partial.wide");
            let mut out = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&wide)?);
            out.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
            out.write_all(&header_bytes)?;
            file.seek(SeekFrom::Start(HEADER_RESERVE))?;
            std::io::copy(&mut file, &mut out)?;
            out.flush()?;
            out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
            Ok(Some(wide))
        };
        let (data, size, names) = (
            shard.data,
            shard.size,
            shard
                .entries
                .into_iter()
                .map(|e| e.name)
                .collect::<Vec<_>>(),
        );
        let partial_path = match finish(data) {
            Ok(None) => partial_path,
            Ok(Some(wide)) => {
                let _ = std::fs::remove_file(&partial_path);
                std::fs::rename(&wide, &partial_path)?;
                partial_path
            }
            Err(e) => {
                let _ = std::fs::remove_file(&partial_path);
                return Err(e.into());
            }
        };

        info!(
            "Wrote shard {index}: {} tensors, {size} bytes (dtype={:?})",
            names.len(),
            self.target_dtype
        );
        self.closed.push(ClosedShard {
            partial_path,
            names,
            size,
        });
        Ok(())
    }

    /// Finish the model: close the last shard, replace any weights already
    /// in the output directory, and move the shards and index into place.
    ///
    /// # Errors
    /// Nothing was written, or the filesystem refused a rename.
    pub fn finalize(mut self) -> Result<PathBuf> {
        self.close_shard()?;
        if self.closed.is_empty() {
            return Err(MergeError::InvalidConfig(
                "no tensors were written".to_string(),
            ));
        }
        self.finished = true;

        let total = self.closed.len();
        let final_names: Vec<String> = if total == 1 {
            vec!["model.safetensors".to_string()]
        } else {
            (1..=total)
                .map(|i| format!("model-{i:05}-of-{total:05}.safetensors"))
                .collect()
        };

        // Weights from an earlier run would shadow or contradict the new ones.
        for entry in std::fs::read_dir(&self.output_path)? {
            let entry = entry?;
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            let stale = file_name == "model.safetensors"
                || file_name == "model.safetensors.index.json"
                || (file_name.starts_with("model-") && file_name.ends_with(".safetensors"));
            if stale && entry.path().is_file() {
                std::fs::remove_file(entry.path())?;
            }
        }

        let mut weight_map = serde_json::Map::new();
        let mut total_size = 0u64;
        for (shard, final_name) in self.closed.iter().zip(&final_names) {
            std::fs::rename(&shard.partial_path, self.output_path.join(final_name))?;
            for name in &shard.names {
                weight_map.insert(name.clone(), serde_json::Value::String(final_name.clone()));
            }
            total_size += shard.size;
        }
        if total > 1 {
            let index = serde_json::json!({
                "metadata": { "total_size": total_size },
                "weight_map": weight_map,
            });
            write_atomic(
                &self.output_path.join("model.safetensors.index.json"),
                serde_json::to_string_pretty(&index)?.as_bytes(),
            )?;
        }
        Ok(self.output_path.clone())
    }
}

impl Drop for TensorWriter {
    /// A writer dropped without `finalize` leaves nothing behind but the
    /// weights already in the directory.
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if let Some(shard) = self.open.take() {
            drop(shard.data);
            let _ = std::fs::remove_file(shard.partial_path);
        }
        for shard in &self.closed {
            let _ = std::fs::remove_file(&shard.partial_path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{arr, assert_close, values};
    use safetensors::tensor::TensorView;

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
    fn parse_dtype_aliases() {
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

    fn write_safetensors(path: &Path, tensors: &[(&str, &[f32], &[usize])]) {
        let bytes: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
            .iter()
            .map(|(n, v, s)| {
                (
                    n.to_string(),
                    v.iter().flat_map(|f| f.to_le_bytes()).collect(),
                    s.to_vec(),
                )
            })
            .collect();
        let views: HashMap<&str, TensorView<'_>> = bytes
            .iter()
            .map(|(n, b, s)| {
                (
                    n.as_str(),
                    TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
                )
            })
            .collect();
        std::fs::write(path, safetensors::serialize(views, None).unwrap()).unwrap();
    }

    /// Round-trip a tensor through every supported writer dtype and verify the
    /// on-disk representation reports the right safetensors::Dtype and decodes
    /// back to the original f32 values within each dtype's precision.
    #[test]
    fn dtype_round_trip_f32_f16_bf16() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.125 - 2.0).collect();
        for (label, dtype, tol) in [
            ("float32", Dtype::F32, 0.0_f32),
            ("float16", Dtype::F16, 1e-3_f32),
            ("bfloat16", Dtype::BF16, 1e-2_f32),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut writer = TensorWriter::with_dtype(dir.path(), dtype).unwrap();
            writer.write_tensor("t", &arr(&data, &[4, 8])).unwrap();
            let out_dir = writer.finalize().unwrap();

            let loader = SafetensorsLoader::new(&out_dir).unwrap();
            assert_eq!(loader.tensor_dtype("t").unwrap(), dtype, "{label}");
            assert_eq!(loader.tensor_shape("t").unwrap(), vec![4, 8], "{label}");
            assert_close(&values(&loader.load_tensor("t").unwrap()), &data, tol);
        }
    }

    /// A safetensors header can have any length, so a tensor's data can start
    /// at an offset that is not a multiple of its element size.
    #[test]
    fn loads_tensors_whose_data_is_not_aligned() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = [1.5_f32, -2.0, 3.25]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let mut header = format!(
            r#"{{"w":{{"dtype":"F32","shape":[3],"data_offsets":[0,{}]}}}}"#,
            data.len()
        )
        .into_bytes();
        // Pad so that 8 + header length leaves the data 1 byte off a 4-byte boundary.
        while (8 + header.len()) % 4 != 1 {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&data);
        std::fs::write(dir.path().join("model.safetensors"), &file).unwrap();

        let loader = SafetensorsLoader::new(dir.path()).unwrap();
        assert_close(
            &values(&loader.load_tensor("w").unwrap()),
            &[1.5, -2.0, 3.25],
            0.0,
        );
    }

    #[test]
    fn an_index_selects_the_shards_and_ignores_other_files() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(&dir.path().join("a.safetensors"), &[("x", &[1.0], &[1])]);
        write_safetensors(&dir.path().join("b.safetensors"), &[("y", &[2.0], &[1])]);
        // Same tensor name as in `a`: a stray file must not collide with the index.
        write_safetensors(
            &dir.path().join("stray.safetensors"),
            &[("x", &[9.0], &[1])],
        );
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"metadata": {}, "weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}"#,
        )
        .unwrap();

        let loader = SafetensorsLoader::new(dir.path()).unwrap();
        assert_eq!(loader.tensor_names(), vec!["x", "y"]);
        assert_close(&values(&loader.load_tensor("x").unwrap()), &[1.0], 0.0);
    }

    #[test]
    fn an_index_naming_a_missing_shard_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(&dir.path().join("a.safetensors"), &[("x", &[1.0], &[1])]);
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map": {"x": "a.safetensors", "y": "gone.safetensors"}}"#,
        )
        .unwrap();
        let err = load_error(dir.path());
        assert!(err.to_string().contains("gone.safetensors"), "{err}");
    }

    #[test]
    fn a_tensor_in_two_files_without_an_index_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(&dir.path().join("a.safetensors"), &[("x", &[1.0], &[1])]);
        write_safetensors(&dir.path().join("b.safetensors"), &[("x", &[2.0], &[1])]);
        let err = load_error(dir.path());
        assert!(err.to_string().contains("tensor x is in both"), "{err}");
    }

    #[test]
    fn hidden_files_next_to_the_weights_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[("x", &[1.0], &[1])],
        );
        // The AppleDouble stub macOS leaves on non-native volumes.
        std::fs::write(
            dir.path().join("._model.safetensors"),
            b"\x00\x05\x16\x07stub",
        )
        .unwrap();
        let loader = SafetensorsLoader::new(dir.path()).unwrap();
        assert_eq!(loader.tensor_names(), vec!["x"]);
    }

    #[test]
    fn an_empty_directory_has_no_weights() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_error(dir.path());
        assert!(err.to_string().contains("No safetensors files found"));
        assert!(SafetensorsLoader::new("/nonexistent/path").is_err());
    }

    #[test]
    fn has_tensor_and_missing_tensors() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(&dir.path().join("m.safetensors"), &[("x", &[1.0], &[1])]);
        let loader = SafetensorsLoader::new(dir.path().join("m.safetensors")).unwrap();
        assert!(loader.has_tensor("x"));
        assert!(!loader.has_tensor("y"));
        assert!(matches!(
            loader.load_tensor("y"),
            Err(MergeError::TensorNotFound(_))
        ));
    }

    /// The message of the error opening `path` must produce.
    fn load_error(path: impl AsRef<Path>) -> String {
        match SafetensorsLoader::new(path) {
            Ok(_) => panic!("opening the model should have failed"),
            Err(e) => e.to_string(),
        }
    }

    fn list(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn one_shard_is_model_safetensors_with_hf_metadata_and_no_scratch_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = TensorWriter::with_dtype(dir.path(), Dtype::F32).unwrap();
        writer.write_tensor("a", &arr(&[1.0, 2.0], &[2])).unwrap();
        writer.write_tensor("b", &arr(&[3.0], &[1])).unwrap();
        writer.finalize().unwrap();

        assert_eq!(list(dir.path()), vec!["model.safetensors"]);
        // transformers refuses an archive without `format` metadata.
        let file = MappedFile::open(&dir.path().join("model.safetensors")).unwrap();
        assert_eq!(
            file.metadata
                .as_ref()
                .and_then(|m| m.get("format"))
                .map(String::as_str),
            Some("pt")
        );
    }

    #[test]
    fn shards_roll_over_with_hf_names_and_an_index() {
        let dir = tempfile::tempdir().unwrap();
        // 8 bytes per tensor, 16-byte shards: two tensors per shard.
        let mut writer = TensorWriter::with_dtype(dir.path(), Dtype::F32)
            .unwrap()
            .with_max_shard_size(16);
        for i in 0..5 {
            let name = format!("t{i}");
            writer
                .write_tensor(&name, &arr(&[i as f32, 0.5], &[2]))
                .unwrap();
        }
        writer.finalize().unwrap();

        assert_eq!(
            list(dir.path()),
            vec![
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
                "model.safetensors.index.json",
            ]
        );
        let index: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(index["metadata"]["total_size"], 40);
        assert_eq!(
            index["weight_map"]["t0"],
            "model-00001-of-00003.safetensors"
        );
        assert_eq!(
            index["weight_map"]["t3"],
            "model-00002-of-00003.safetensors"
        );
        assert_eq!(
            index["weight_map"]["t4"],
            "model-00003-of-00003.safetensors"
        );

        // A loader follows the index, and every tensor reads back.
        let loader = SafetensorsLoader::new(dir.path()).unwrap();
        assert_eq!(loader.tensor_names().len(), 5);
        assert_close(
            &values(&loader.load_tensor("t4").unwrap()),
            &[4.0, 0.5],
            0.0,
        );
    }

    /// A header too large for the reserved region (a shard of tens of
    /// thousands of tensors) takes the copying path and still reads back.
    #[test]
    fn a_shard_whose_header_outgrows_the_reserve_still_loads() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = TensorWriter::with_dtype(dir.path(), Dtype::F32).unwrap();
        let count = 20_000;
        for i in 0..count {
            writer
                .write_tensor(&format!("t{i:05}"), &arr(&[i as f32], &[1]))
                .unwrap();
        }
        writer.finalize().unwrap();
        let file = dir.path().join("model.safetensors");
        let len = std::fs::metadata(&file).unwrap().len();
        assert!(
            len > HEADER_RESERVE,
            "header should exceed the reserve: {len}"
        );
        let loader = SafetensorsLoader::new(dir.path()).unwrap();
        assert_eq!(loader.tensor_names().len(), count);
        assert_close(
            &values(&loader.load_tensor("t19999").unwrap()),
            &[19999.0],
            0.0,
        );
        assert_close(&values(&loader.load_tensor("t00007").unwrap()), &[7.0], 0.0);
        assert_eq!(list(dir.path()), vec!["model.safetensors"]);
    }

    #[test]
    fn a_tensor_larger_than_the_limit_gets_its_own_shard() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = TensorWriter::with_dtype(dir.path(), Dtype::F32)
            .unwrap()
            .with_max_shard_size(4);
        writer
            .write_tensor("big", &arr(&[1.0, 2.0, 3.0], &[3]))
            .unwrap();
        writer.write_tensor("small", &arr(&[4.0], &[1])).unwrap();
        writer.finalize().unwrap();
        assert_eq!(list(dir.path()).len(), 3); // two shards and the index
    }

    #[test]
    fn dropping_a_writer_without_finalize_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut writer = TensorWriter::with_dtype(dir.path(), Dtype::F32)
                .unwrap()
                .with_max_shard_size(8);
            writer.write_tensor("a", &arr(&[1.0, 2.0], &[2])).unwrap();
            writer.write_tensor("b", &arr(&[3.0, 4.0], &[2])).unwrap();
        }
        assert!(list(dir.path()).is_empty(), "{:?}", list(dir.path()));
    }

    #[test]
    fn finalize_replaces_weights_from_an_earlier_run() {
        let dir = tempfile::tempdir().unwrap();
        for old in [
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
        ] {
            std::fs::write(dir.path().join(old), b"stale").unwrap();
        }
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let mut writer = TensorWriter::with_dtype(dir.path(), Dtype::F32).unwrap();
        writer.write_tensor("a", &arr(&[1.0], &[1])).unwrap();
        writer.finalize().unwrap();
        assert_eq!(list(dir.path()), vec!["config.json", "model.safetensors"]);
    }

    #[test]
    fn writing_a_name_twice_or_nothing_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = TensorWriter::with_dtype(dir.path(), Dtype::F32).unwrap();
        writer.write_tensor("a", &arr(&[1.0], &[1])).unwrap();
        assert!(writer.write_tensor("a", &arr(&[2.0], &[1])).is_err());
        drop(writer);

        let writer = TensorWriter::with_dtype(dir.path(), Dtype::F32).unwrap();
        assert!(writer.finalize().is_err());
    }

    #[test]
    fn a_value_that_overflows_float16_is_an_error_not_infinity() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = TensorWriter::with_dtype(dir.path(), Dtype::F16).unwrap();
        let err = writer
            .write_tensor("big", &arr(&[1.0, 70000.0], &[2]))
            .expect_err("must fail");
        assert!(err.to_string().contains("overflow"), "{err}");
        // bfloat16 has the range for it.
        let mut writer = TensorWriter::with_dtype(dir.path(), Dtype::BF16).unwrap();
        writer
            .write_tensor("big", &arr(&[1.0, 70000.0], &[2]))
            .unwrap();
    }

    #[test]
    fn unsupported_writer_dtypes_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        assert!(TensorWriter::with_dtype(dir.path(), Dtype::I8).is_err());
    }

    #[test]
    fn copy_side_files_skips_weights_and_index() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        for name in [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model-00001-of-00002.bin",
            "model.gguf",
            ".gitattributes",
        ] {
            std::fs::write(src.path().join(name), b"x").unwrap();
        }
        copy_side_files(src.path(), dst.path()).unwrap();
        assert_eq!(list(dst.path()), vec!["config.json", "tokenizer.json"]);
    }
}
