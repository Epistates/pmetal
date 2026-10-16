//! Finding a checkpoint's weight files and reading their tensors.
//!
//! A checkpoint directory holds its weights as safetensors (one file, or shards
//! named by `model.safetensors.index.json`) or, for models only ever published
//! that way, as PyTorch pickles (`pytorch_model.bin`, or shards named by
//! `pytorch_model.bin.index.json`). [`Checkpoint`] resolves which, validates
//! where each shard lives, and reads tensors without caring which it was.
//! Everything above it (quantization unpacking, assignment) sees one
//! `name -> Array` map.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use pmetal_hub::torch::{TorchArchive, TorchDtype, TorchTensor};
use pmetal_runtime::compat::Array;

use super::LoadError;

/// Weight index for sharded models.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightIndex {
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

/// How a checkpoint's tensors are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileFormat {
    Safetensors,
    PyTorch,
}

/// Where a checkpoint's tensors are.
#[derive(Debug)]
enum Layout {
    /// All of them in one file.
    Single(PathBuf),
    /// Spread over the shards `weight_map` names, relative to the directory.
    Sharded(WeightIndex),
}

/// The weight files of a checkpoint directory.
#[derive(Debug)]
pub(crate) struct Checkpoint {
    dir: PathBuf,
    format: FileFormat,
    layout: Layout,
}

/// The files that name a checkpoint, in the order they are preferred.
const CANDIDATES: [(&str, &str, FileFormat); 2] = [
    (
        "model.safetensors",
        "model.safetensors.index.json",
        FileFormat::Safetensors,
    ),
    (
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        FileFormat::PyTorch,
    ),
];

fn no_weights(dir: &Path) -> LoadError {
    LoadError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "No weights found in {}: expected model.safetensors, \
             model.safetensors.index.json, pytorch_model.bin or \
             pytorch_model.bin.index.json",
            dir.display()
        ),
    ))
}

impl Checkpoint {
    /// Find the weights in `model_dir`: safetensors if there are any, else a
    /// PyTorch checkpoint.
    pub(crate) fn open(model_dir: &Path) -> Result<Self, LoadError> {
        for (single, index, format) in CANDIDATES {
            let single_path = model_dir.join(single);
            if single_path.exists() {
                return Ok(Self {
                    dir: model_dir.to_path_buf(),
                    format,
                    layout: Layout::Single(single_path),
                });
            }
            let index_path = model_dir.join(index);
            if index_path.exists() {
                let text = std::fs::read_to_string(&index_path)?;
                let index: WeightIndex = serde_json::from_str(&text)?;
                return Ok(Self {
                    dir: model_dir.to_path_buf(),
                    format,
                    layout: Layout::Sharded(index),
                });
            }
        }
        Err(no_weights(model_dir))
    }

    /// Read the tensors `keep` accepts. A sharded checkpoint opens only the
    /// shards that hold one.
    pub(crate) fn read(
        &self,
        mut keep: impl FnMut(&str) -> bool,
    ) -> Result<HashMap<String, Array>, LoadError> {
        match &self.layout {
            Layout::Single(path) => Ok(read_file(path, self.format)?
                .into_iter()
                .filter(|(key, _)| keep(key))
                .collect()),
            Layout::Sharded(index) => {
                let wanted: HashSet<&str> = index
                    .weight_map
                    .keys()
                    .map(String::as_str)
                    .filter(|key| keep(key))
                    .collect();
                let shards: HashSet<&String> = index
                    .weight_map
                    .iter()
                    .filter(|(key, _)| wanted.contains(key.as_str()))
                    .map(|(_, shard)| shard)
                    .collect();

                let mut tensors = HashMap::new();
                for shard in shards {
                    let path = validate_shard_path(&self.dir, shard)?;
                    tensors.extend(
                        read_file(&path, self.format)?
                            .into_iter()
                            .filter(|(key, _)| wanted.contains(key.as_str())),
                    );
                }
                Ok(tensors)
            }
        }
    }
}

fn read_file(path: &Path, format: FileFormat) -> Result<HashMap<String, Array>, LoadError> {
    match format {
        FileFormat::Safetensors => load_shard(path),
        FileFormat::PyTorch => read_torch(path),
    }
}

/// Read one safetensors file into a name → array map, as stored (no
/// dequantization or renaming).
pub fn load_safetensors_file(
    path: impl AsRef<std::path::Path>,
) -> Result<HashMap<String, Array>, LoadError> {
    load_shard(path.as_ref())
}

/// Load a safetensors shard file into a `HashMap<String, Array>`.
fn load_shard(path: &Path) -> Result<HashMap<String, Array>, LoadError> {
    let path_str = path.to_str().ok_or_else(|| {
        LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Non-UTF8 path: {path:?}"),
        ))
    })?;
    pmetal_runtime::inline_array::try_load_safetensors_shard(path_str)
        .map(|pairs| pairs.into_iter().collect())
        .map_err(|e| LoadError::SafeTensors(e.to_string()))
}

fn open_torch(path: &Path) -> Result<TorchArchive, LoadError> {
    TorchArchive::open(path).map_err(torch_error)
}

fn torch_error(e: pmetal_core::PMetalError) -> LoadError {
    LoadError::SafeTensors(format!("PyTorch checkpoint: {e}"))
}

/// Read every tensor of a PyTorch checkpoint file.
///
/// `pmetal_hub::torch` maps the file and runs the pickle through a restricted
/// interpreter that refuses anything outside a short allow-list, so opening an
/// untrusted file executes nothing. The tensors keep their stored dtype.
fn read_torch(path: &Path) -> Result<HashMap<String, Array>, LoadError> {
    let archive = open_torch(path)?;
    archive
        .tensors(None)
        .map_err(torch_error)?
        .into_iter()
        .map(|tensor| {
            let array = torch_tensor_to_array(&tensor, path)?;
            Ok((tensor.name, array))
        })
        .collect()
}

/// A torch tensor's bytes as an array of the same dtype and shape.
fn torch_tensor_to_array(tensor: &TorchTensor<'_>, path: &Path) -> Result<Array, LoadError> {
    let unsupported = || {
        LoadError::SafeTensors(format!(
            "{}: tensor {} has dtype {:?}, which pmetal does not load",
            path.display(),
            tensor.name,
            tensor.dtype
        ))
    };
    let shape = tensor
        .shape
        .iter()
        .map(|&d| i32::try_from(d))
        .collect::<Result<Vec<i32>, _>>()
        .map_err(|_| {
            LoadError::SafeTensors(format!(
                "{}: tensor {} has a dimension beyond i32",
                path.display(),
                tensor.name
            ))
        })?;
    let data = tensor.data;
    let u16s = || -> Vec<u16> {
        data.as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect()
    };
    Ok(match tensor.dtype {
        TorchDtype::F32 => {
            let values: Vec<f32> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            Array::from_f32_slice(&values, &shape)
        }
        TorchDtype::F64 => {
            let values: Vec<f32> = data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b) as f32)
                .collect();
            Array::from_f32_slice(&values, &shape)
        }
        TorchDtype::F16 => Array::from_u16_bits_slice(&u16s(), &shape, pmetal_runtime::dtype::F16),
        TorchDtype::Bf16 => {
            Array::from_u16_bits_slice(&u16s(), &shape, pmetal_runtime::dtype::BF16)
        }
        // MLX holds an FP8 payload in a uint8 array, as the safetensors reader does.
        TorchDtype::F8E4M3 | TorchDtype::U8 => Array::from_u8_slice(data, &shape),
        TorchDtype::I32 => {
            let values: Vec<i32> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect();
            Array::from_i32_slice_shaped(&values, &shape)
        }
        TorchDtype::I64 => {
            let values = data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .map(|v| i32::try_from(v).map_err(|_| unsupported()))
                .collect::<Result<Vec<i32>, _>>()?;
            Array::from_i32_slice_shaped(&values, &shape)
        }
        TorchDtype::F8E5M2 | TorchDtype::I16 | TorchDtype::I8 | TorchDtype::Bool => {
            return Err(unsupported());
        }
    })
}

/// Validate that a shard path does not escape the model directory (path traversal protection).
///
/// HuggingFace cache uses symlinks: snapshot files point to `../../blobs/`.
/// We validate that the canonical shard path stays within the repository's
/// cache folder or the cache-wide blob store (see
/// [`pmetal_hub::CacheLocation::stores`]), not just the snapshot directory.
pub(crate) fn validate_shard_path(
    model_dir: &Path,
    shard_file: &str,
) -> Result<PathBuf, LoadError> {
    // Reject path traversal in the shard filename itself
    if shard_file.contains("..") || shard_file.starts_with('/') {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("Shard filename contains path traversal: {shard_file}"),
        )));
    }
    let shard_path = model_dir.join(shard_file);
    let canonical_dir = model_dir.canonicalize().map_err(LoadError::Io)?;
    let canonical_shard = shard_path.canonicalize().map_err(|e| {
        LoadError::Io(std::io::Error::new(
            e.kind(),
            format!(
                "Shard file not found: {} (in {})",
                shard_file,
                model_dir.display()
            ),
        ))
    })?;
    // Return the original path, not the canonical one, so the file extension
    // is kept.
    if canonical_shard.starts_with(&canonical_dir) {
        return Ok(shard_path);
    }
    if pmetal_hub::locate_in_cache(&canonical_dir)
        .is_some_and(|cache| cache.stores(&canonical_shard))
    {
        return Ok(shard_path);
    }
    Err(LoadError::Io(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!("Shard path escapes model directory: {shard_path:?}"),
    )))
}
