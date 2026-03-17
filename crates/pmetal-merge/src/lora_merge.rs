//! Streaming f64-accurate LoRA merge for safetensors models.
//!
//! # Why f64?
//!
//! LoRA delta matrices are computed as `B @ A` where both `B` and `A` are
//! typically stored in f16 or bf16.  Accumulating this matmul in f32 introduces
//! rounding errors that compound across the rank dimension.  Using f64
//! throughout the matmul — and for the `base + scaling * delta` addition —
//! gives bit-accurate results before the final downcast back to the storage
//! dtype.
//!
//! # Memory model
//!
//! - **Normal path**: materialises the full `[out, in]` delta in f64, then
//!   performs a fused row-by-row base+delta+downcast pass.  Peak memory =
//!   `rows * cols * 8` bytes for the delta.
//!
//! - **Low-memory path** (`--low-memory` / `tile_size`): tiles the B matrix in
//!   512-row chunks.  At any given moment only `tile_size * rank * 8` bytes of
//!   f64 state are live.  Suitable for very large tensors on memory-constrained
//!   machines.
//!
//! Non-adapted tensors are copied byte-for-byte from the mmap, keeping the
//! pass through zero-allocation.
//!
//! # Adapter name conventions
//!
//! PEFT adapters use the `base_model.model.` prefix convention:
//!
//! ```text
//! base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
//! base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
//! ```
//!
//! These are resolved back to the base tensor name
//! `model.layers.0.self_attn.q_proj.weight` by stripping the prefix and the
//! `.lora_{A,B}.weight` suffix.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use memmap2::Mmap;
use ndarray::{Array2, s};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::{MergeError, Result};

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for the streaming f64-accurate LoRA merge.
#[derive(Debug, Clone)]
pub struct AccurateMergeConfig {
    /// Path to the base model directory or single `.safetensors` file.
    pub base_model_path: PathBuf,

    /// Path to the adapter directory (must contain `adapter_config.json` and
    /// `adapter_model.safetensors`).
    pub adapter_path: PathBuf,

    /// Output path.  If the base model is a single file, the output is a
    /// single file at this path.  If the base model is sharded (contains
    /// `model.safetensors.index.json`) the output is written as a sharded
    /// directory at this path.
    pub output_path: PathBuf,

    /// Override the scaling factor (`lora_alpha / lora_rank`).
    ///
    /// When `None` the value is read from `adapter_config.json`.
    pub scaling: Option<f64>,

    /// Use the tiled low-memory path instead of the full-matrix path.
    pub low_memory: bool,

    /// Number of rows per tile when `low_memory` is `true`.
    ///
    /// Defaults to 512.
    pub tile_size: usize,
}

impl AccurateMergeConfig {
    /// Create a config with sensible defaults.
    pub fn new(
        base_model_path: impl Into<PathBuf>,
        adapter_path: impl Into<PathBuf>,
        output_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            base_model_path: base_model_path.into(),
            adapter_path: adapter_path.into(),
            output_path: output_path.into(),
            scaling: None,
            low_memory: false,
            tile_size: 512,
        }
    }

    /// Enable the low-memory tiled path with the given tile size.
    pub fn with_low_memory(mut self, tile_size: usize) -> Self {
        self.low_memory = true;
        self.tile_size = tile_size;
        self
    }

    /// Override the LoRA scaling factor.
    pub fn with_scaling(mut self, scaling: f64) -> Self {
        self.scaling = Some(scaling);
        self
    }
}

/// Statistics returned after a successful merge.
#[derive(Debug, Clone, Default)]
pub struct LoraMergeStats {
    /// Number of tensors merged (base + LoRA delta).
    pub tensors_merged: usize,

    /// Number of bias tensors merged (base bias + adapter bias).
    pub biases_merged: usize,

    /// Number of tensors passed through unchanged.
    pub tensors_copied: usize,

    /// Total bytes written to the output file(s).
    pub bytes_written: u64,

    /// Wall-clock duration of the merge in milliseconds.
    pub elapsed_ms: f64,
}

// ---------------------------------------------------------------------------
// Adapter config parsing
// ---------------------------------------------------------------------------

/// Parsed fields from `adapter_config.json` that are needed for the merge.
#[derive(Debug, Clone, Deserialize)]
struct AdapterConfigJson {
    r: u32,
    lora_alpha: f64,
    #[serde(default)]
    target_modules: Vec<String>,
    #[serde(default)]
    fan_in_fan_out: bool,
    /// Accept any string for peft_type; we only support "LORA".
    #[serde(default)]
    peft_type: String,
}

impl AdapterConfigJson {
    fn from_path(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path).map_err(|e| {
            MergeError::InvalidConfig(format!(
                "failed to read adapter_config.json at {}: {}",
                path.display(),
                e
            ))
        })?;
        let cfg: AdapterConfigJson = serde_json::from_str(&contents).map_err(|e| {
            MergeError::InvalidConfig(format!("failed to parse adapter_config.json: {}", e))
        })?;
        if cfg.r == 0 {
            return Err(MergeError::InvalidConfig(
                "adapter rank (r) must be greater than zero".to_string(),
            ));
        }
        if !cfg.lora_alpha.is_finite() {
            return Err(MergeError::InvalidConfig(format!(
                "lora_alpha must be finite, got {}",
                cfg.lora_alpha
            )));
        }
        Ok(cfg)
    }

    /// `lora_alpha / r`
    fn scaling(&self) -> f64 {
        self.lora_alpha / f64::from(self.r)
    }
}

// ---------------------------------------------------------------------------
// Internal low-level tensor helpers
// ---------------------------------------------------------------------------

/// Decode raw safetensors bytes into an f64 `Array2`.
///
/// Handles `F16`, `BF16`, and `F32` storage dtypes.  Requires the tensor to be
/// exactly 2-dimensional.
fn bytes_to_f64_2d(bytes: &[u8], dtype: Dtype, shape: &[usize], name: &str) -> Result<Array2<f64>> {
    if shape.len() != 2 {
        return Err(MergeError::ShapeMismatch {
            name: format!("{name} (expected 2D, got {}D)", shape.len()),
            expected: vec![2],
            actual: shape.iter().map(|&s| s as i32).collect(),
        });
    }
    let (rows, cols) = (shape[0], shape[1]);
    let n = rows * cols;

    let values: Vec<f64> = match dtype {
        Dtype::F16 => {
            if bytes.len() != n * 2 {
                return Err(byte_len_mismatch(name, n * 2, bytes.len()));
            }
            bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f64())
                .collect()
        }
        Dtype::BF16 => {
            if bytes.len() != n * 2 {
                return Err(byte_len_mismatch(name, n * 2, bytes.len()));
            }
            bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f64())
                .collect()
        }
        Dtype::F32 => {
            if bytes.len() != n * 4 {
                return Err(byte_len_mismatch(name, n * 4, bytes.len()));
            }
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect()
        }
        Dtype::F64 => {
            if bytes.len() != n * 8 {
                return Err(byte_len_mismatch(name, n * 8, bytes.len()));
            }
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        }
        other => {
            return Err(MergeError::ModelLoad(format!(
                "unsupported dtype {other:?} for tensor '{name}'"
            )));
        }
    };

    Array2::from_shape_vec((rows, cols), values)
        .map_err(|e| MergeError::ModelLoad(format!("failed to reshape tensor '{name}': {e}")))
}

fn byte_len_mismatch(name: &str, expected: usize, actual: usize) -> MergeError {
    MergeError::ShapeMismatch {
        name: format!("{name} (byte length)"),
        expected: vec![expected as i32],
        actual: vec![actual as i32],
    }
}

/// Decode raw safetensors bytes for a 1-D bias tensor into a `Vec<f64>`.
///
/// Accepts F16, BF16, F32, and F64 storage dtypes.  The `shape` slice must
/// describe a 1-D tensor (length 1).
fn bytes_to_f64_1d(bytes: &[u8], dtype: Dtype, shape: &[usize], name: &str) -> Result<Vec<f64>> {
    if shape.len() != 1 {
        return Err(MergeError::ShapeMismatch {
            name: format!("{name} (expected 1D bias, got {}D)", shape.len()),
            expected: vec![1],
            actual: shape.iter().map(|&s| s as i32).collect(),
        });
    }
    let n = shape[0];

    let values: Vec<f64> = match dtype {
        Dtype::F16 => {
            if bytes.len() != n * 2 {
                return Err(byte_len_mismatch(name, n * 2, bytes.len()));
            }
            bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f64())
                .collect()
        }
        Dtype::BF16 => {
            if bytes.len() != n * 2 {
                return Err(byte_len_mismatch(name, n * 2, bytes.len()));
            }
            bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f64())
                .collect()
        }
        Dtype::F32 => {
            if bytes.len() != n * 4 {
                return Err(byte_len_mismatch(name, n * 4, bytes.len()));
            }
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect()
        }
        Dtype::F64 => {
            if bytes.len() != n * 8 {
                return Err(byte_len_mismatch(name, n * 8, bytes.len()));
            }
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        }
        other => {
            return Err(MergeError::ModelLoad(format!(
                "unsupported dtype {other:?} for bias tensor '{name}'"
            )));
        }
    };

    Ok(values)
}

/// Add adapter bias to base bias and write the result in the original storage dtype.
///
/// Both tensors must be 1-D and have the same length.  The addition is done in
/// f64 before downcasting back to the base dtype.
fn merge_and_write_bias(
    base_bytes: &[u8],
    base_shape: &[usize],
    base_dtype: Dtype,
    adapter_bytes: &[u8],
    adapter_shape: &[usize],
    adapter_dtype: Dtype,
    base_name: &str,
    adapter_name: &str,
    writer: &mut impl Write,
) -> Result<()> {
    let base_vals = bytes_to_f64_1d(base_bytes, base_dtype, base_shape, base_name)?;
    let adapter_vals = bytes_to_f64_1d(adapter_bytes, adapter_dtype, adapter_shape, adapter_name)?;

    if base_vals.len() != adapter_vals.len() {
        return Err(MergeError::ShapeMismatch {
            name: format!("bias merge size mismatch for '{base_name}'"),
            expected: vec![base_vals.len() as i32],
            actual: vec![adapter_vals.len() as i32],
        });
    }

    // Perform the addition in f64 then downcast to base storage dtype.
    match base_dtype {
        Dtype::F16 => {
            for (&b, &a) in base_vals.iter().zip(adapter_vals.iter()) {
                let merged = ((b + a) as f32).clamp(f16::MIN.to_f32(), f16::MAX.to_f32());
                writer.write_all(&f16::from_f32(merged).to_le_bytes())?;
            }
        }
        Dtype::BF16 => {
            for (&b, &a) in base_vals.iter().zip(adapter_vals.iter()) {
                let merged = ((b + a) as f32).clamp(bf16::MIN.to_f32(), bf16::MAX.to_f32());
                writer.write_all(&bf16::from_f32(merged).to_le_bytes())?;
            }
        }
        Dtype::F32 => {
            for (&b, &a) in base_vals.iter().zip(adapter_vals.iter()) {
                let merged = (b + a) as f32;
                writer.write_all(&merged.to_le_bytes())?;
            }
        }
        Dtype::F64 => {
            for (&b, &a) in base_vals.iter().zip(adapter_vals.iter()) {
                let merged = b + a;
                writer.write_all(&merged.to_le_bytes())?;
            }
        }
        other => {
            return Err(MergeError::ModelLoad(format!(
                "unsupported output dtype {other:?} for bias merge of '{base_name}'"
            )));
        }
    }

    Ok(())
}

/// Element size in bytes for supported dtypes.
fn dtype_elem_size(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::F16 | Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        Dtype::F64 => Ok(8),
        Dtype::I32 | Dtype::U32 => Ok(4),
        Dtype::I64 | Dtype::U64 => Ok(8),
        Dtype::I16 | Dtype::U16 => Ok(2),
        Dtype::I8 | Dtype::U8 | Dtype::BOOL => Ok(1),
        other => Err(MergeError::ModelLoad(format!(
            "unsupported dtype {other:?} for element size computation"
        ))),
    }
}

/// dtype tag string for the safetensors JSON header.
fn dtype_tag(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::F32 => "F32",
        Dtype::F64 => "F64",
        Dtype::I8 => "I8",
        Dtype::U8 => "U8",
        Dtype::I16 => "I16",
        Dtype::U16 => "U16",
        Dtype::I32 => "I32",
        Dtype::U32 => "U32",
        Dtype::I64 => "I64",
        Dtype::U64 => "U64",
        Dtype::BOOL => "BOOL",
        _ => "F32",
    }
}

// ---------------------------------------------------------------------------
// Memory-mapped safetensors file handle
// ---------------------------------------------------------------------------

/// Parsed tensor metadata entry from a safetensors file.
#[derive(Debug, Clone)]
struct TensorMeta {
    dtype: Dtype,
    shape: Vec<usize>,
    /// Absolute byte offset within the mmap (past the header).
    data_start: usize,
    data_end: usize,
}

/// A memory-mapped safetensors file with pre-indexed tensor locations.
///
/// Tensors are stored in file order for sequential access patterns.
struct MappedFile {
    mmap: Mmap,
    /// Ordered by file offset.
    tensors: Vec<(String, TensorMeta)>,
    index: HashMap<String, usize>,
    metadata: Option<HashMap<String, String>>,
}

impl MappedFile {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            MergeError::ModelLoad(format!("failed to open {}: {}", path.display(), e))
        })?;

        // SAFETY: The file is opened read-only; we hold the Mmap for the
        // duration of the merge and do not mutate the file.
        #[allow(unsafe_code)]
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            MergeError::ModelLoad(format!("failed to mmap {}: {}", path.display(), e))
        })?;

        let (header_len, metadata_view) = SafeTensors::read_metadata(&mmap).map_err(|e| {
            MergeError::ModelLoad(format!(
                "failed to parse safetensors header in {}: {}",
                path.display(),
                e
            ))
        })?;

        // Safetensors format: 8-byte LE u64 header length, JSON header, then data.
        let data_offset = 8 + header_len;

        let mut tensors: Vec<(String, TensorMeta)> = Vec::new();
        for (name, info) in metadata_view.tensors() {
            let (start, end) = info.data_offsets;
            tensors.push((
                name,
                TensorMeta {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    data_start: data_offset + start,
                    data_end: data_offset + end,
                },
            ));
        }

        // Sort by file offset to maximise sequential read locality.
        tensors.sort_by_key(|(_, m)| m.data_start);

        let index: HashMap<String, usize> = tensors
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();

        let metadata = metadata_view.metadata().clone();

        Ok(Self {
            mmap,
            tensors,
            index,
            metadata,
        })
    }

    fn tensor_meta(&self, name: &str) -> Result<&TensorMeta> {
        self.index
            .get(name)
            .map(|&i| &self.tensors[i].1)
            .ok_or_else(|| MergeError::TensorNotFound(name.to_string()))
    }

    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let meta = self.tensor_meta(name)?;
        Ok(&self.mmap[meta.data_start..meta.data_end])
    }

    fn advise_sequential(&self) {
        let _ = self.mmap.advise(memmap2::Advice::Sequential);
    }

    fn advise_dontneed(&self) {
        let _ = self.mmap.advise(memmap2::Advice::Normal);
    }
}

// ---------------------------------------------------------------------------
// Safetensors header builder
// ---------------------------------------------------------------------------

/// Builds a valid safetensors binary header for a set of tensors and returns
/// the raw bytes.
///
/// The header is a JSON object with an optional `__metadata__` key followed by
/// one entry per tensor.  Tensor entries record dtype, shape, and the
/// `[start, end]` byte offsets within the data section.
fn build_header(
    tensors: &[(String, TensorMeta)],
    metadata: Option<&HashMap<String, String>>,
) -> Result<Vec<u8>> {
    let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut offset: usize = 0;

    if let Some(meta) = metadata {
        header.insert(
            "__metadata__".to_string(),
            serde_json::to_value(meta).map_err(MergeError::Serde)?,
        );
    }

    for (name, meta) in tensors {
        let elem_size = dtype_elem_size(meta.dtype)?;
        let n: usize = meta.shape.iter().product();
        let byte_size = n * elem_size;

        let entry = serde_json::json!({
            "dtype": dtype_tag(meta.dtype),
            "shape": meta.shape,
            "data_offsets": [offset, offset + byte_size],
        });
        header.insert(name.clone(), entry);
        offset += byte_size;
    }

    serde_json::to_vec(&header).map_err(MergeError::Serde)
}

// ---------------------------------------------------------------------------
// Name-mapping helpers (same convention as safetensors-surgery)
// ---------------------------------------------------------------------------

/// The PEFT adapter prefix that all adapter tensor names begin with.
const ADAPTER_PREFIX: &str = "base_model.model.";

/// Build a mapping from base tensor name → `(lora_A_adapter_name, lora_B_adapter_name)`.
///
/// Adapter tensor names follow the pattern:
/// `base_model.model.<base_name_without_weight>.lora_{A,B}.weight`
fn build_lora_pairs(
    base_names: &[&str],
    adapter_names: &[&str],
) -> HashMap<String, (String, String)> {
    let base_set: std::collections::HashSet<&str> = base_names.iter().copied().collect();

    let mut lora_a: HashMap<String, String> = HashMap::new();
    let mut lora_b: HashMap<String, String> = HashMap::new();

    for &adapter_name in adapter_names {
        let Some(stripped) = adapter_name.strip_prefix(ADAPTER_PREFIX) else {
            continue;
        };

        if let Some(base_part) = stripped.strip_suffix(".lora_A.weight") {
            let base_weight = format!("{base_part}.weight");
            if base_set.contains(base_weight.as_str()) {
                lora_a.insert(base_weight, adapter_name.to_string());
            }
        } else if let Some(base_part) = stripped.strip_suffix(".lora_B.weight") {
            let base_weight = format!("{base_part}.weight");
            if base_set.contains(base_weight.as_str()) {
                lora_b.insert(base_weight, adapter_name.to_string());
            }
        }
    }

    let mut pairs: HashMap<String, (String, String)> = HashMap::new();
    for (base_name, a_name) in &lora_a {
        if let Some(b_name) = lora_b.get(base_name) {
            pairs.insert(base_name.clone(), (a_name.clone(), b_name.clone()));
        } else {
            warn!(
                "adapter has lora_A for '{}' but no matching lora_B — skipping",
                base_name
            );
        }
    }
    for base_name in lora_b.keys() {
        if !lora_a.contains_key(base_name) {
            warn!(
                "adapter has lora_B for '{}' but no matching lora_A — skipping",
                base_name
            );
        }
    }

    pairs
}

/// Build a mapping from base bias tensor name → adapter bias tensor name.
///
/// Adapter bias tensors follow the pattern:
/// `base_model.model.<base_name_without_bias>.bias`
///
/// These are tensors in the adapter that end with `.bias` but are **not**
/// LoRA weight tensors (i.e. they don't contain `.lora_A.` or `.lora_B.`).
/// When present they represent the trained bias term that should be added to
/// the corresponding base bias (if the base has one) or written as-is if the
/// base has no bias for that projection.
fn build_bias_map(base_names: &[&str], adapter_names: &[&str]) -> HashMap<String, String> {
    let base_set: std::collections::HashSet<&str> = base_names.iter().copied().collect();
    let mut bias_map: HashMap<String, String> = HashMap::new();

    for &adapter_name in adapter_names {
        let Some(stripped) = adapter_name.strip_prefix(ADAPTER_PREFIX) else {
            continue;
        };

        // Skip LoRA A/B weight tensors — those are handled by build_lora_pairs.
        if stripped.contains(".lora_A.") || stripped.contains(".lora_B.") {
            continue;
        }

        // Match tensors whose base name ends with `.bias`.
        if !stripped.ends_with(".bias") {
            continue;
        }

        let base_name = stripped.to_string();

        // Only include if the base model has a matching bias tensor.
        if base_set.contains(base_name.as_str()) {
            bias_map.insert(base_name, adapter_name.to_string());
        } else {
            warn!(
                "adapter has bias '{}' but base model has no matching tensor — skipping",
                adapter_name
            );
        }
    }

    bias_map
}

// ---------------------------------------------------------------------------
// Core merge math
// ---------------------------------------------------------------------------

/// Compute `delta = B_f64 @ A_f64` (or its transpose when `fan_in_fan_out`),
/// then write `base + scaling * delta` in the original storage dtype to
/// `writer`, row by row.
///
/// When `low_memory` is false the full delta matrix is materialised in f64 and
/// the fused base+delta pass is done in a single O(rows*cols) loop.  When
/// `low_memory` is true the matmul is tiled: at any moment only `tile_size`
/// rows of B × A are live in f64.
#[allow(clippy::too_many_arguments)]
fn merge_and_write_tensor(
    base_bytes: &[u8],
    lora_a: &Array2<f64>,
    lora_b: &Array2<f64>,
    scaling: f64,
    fan_in_fan_out: bool,
    base_shape: &[usize],
    dtype: Dtype,
    low_memory: bool,
    tile_size: usize,
    writer: &mut impl Write,
) -> Result<()> {
    if base_shape.len() != 2 {
        return Err(MergeError::InvalidConfig(format!(
            "LoRA merge only supports 2D weight tensors, got {}D (shape: {:?})",
            base_shape.len(),
            base_shape
        )));
    }
    let rows = base_shape[0];
    let cols = base_shape[1];

    let elem_size = match dtype {
        Dtype::F16 | Dtype::BF16 => 2usize,
        Dtype::F32 => 4,
        other => {
            return Err(MergeError::ModelLoad(format!(
                "unsupported output dtype {other:?} for LoRA merge"
            )));
        }
    };
    let row_bytes = cols * elem_size;

    if base_bytes.len() != rows * row_bytes {
        return Err(MergeError::ShapeMismatch {
            name: "base tensor bytes".to_string(),
            expected: vec![(rows * row_bytes) as i32],
            actual: vec![base_bytes.len() as i32],
        });
    }

    // Validate LoRA dimensions before the matmul so we get a clean error
    // rather than a panic inside ndarray.
    //
    // Without fan_in_fan_out: delta = B @ A  →  [rows, cols]
    // With    fan_in_fan_out: delta = (B @ A).T  →  [cols, rows] (base stored transposed)
    let (exp_b_rows, exp_a_cols) = if fan_in_fan_out {
        (cols, rows)
    } else {
        (rows, cols)
    };
    if lora_b.nrows() != exp_b_rows {
        return Err(MergeError::ShapeMismatch {
            name: "lora_B rows".to_string(),
            expected: vec![exp_b_rows as i32],
            actual: vec![lora_b.nrows() as i32],
        });
    }
    if lora_a.ncols() != exp_a_cols {
        return Err(MergeError::ShapeMismatch {
            name: "lora_A cols".to_string(),
            expected: vec![exp_a_cols as i32],
            actual: vec![lora_a.ncols() as i32],
        });
    }
    if lora_b.ncols() != lora_a.nrows() {
        return Err(MergeError::ShapeMismatch {
            name: "LoRA rank (lora_B.ncols vs lora_A.nrows)".to_string(),
            expected: vec![lora_b.ncols() as i32],
            actual: vec![lora_a.nrows() as i32],
        });
    }

    if low_memory {
        // Tiled path — only `tile_size` rows of B (or A columns) live at once.
        let mut start = 0usize;
        while start < rows {
            let end = (start + tile_size).min(rows);
            let tile_rows = end - start;

            let delta_tile: Array2<f64> = if fan_in_fan_out {
                // rows [start..end] of (B @ A).T  =  cols [start..end] of B @ A
                //  = B @ A[:, start..end], then transpose
                let a_slice = lora_a.slice(s![.., start..end]);
                (lora_b.dot(&a_slice) * scaling).t().to_owned()
            } else {
                let b_slice = lora_b.slice(s![start..end, ..]).to_owned();
                b_slice.dot(lora_a) * scaling
            };

            let base_tile = &base_bytes[start * row_bytes..end * row_bytes];
            write_merged_rows(base_tile, &delta_tile, tile_rows, cols, dtype, writer)?;

            start = end;
        }
    } else {
        // Full-matrix path.
        let delta_f64: Array2<f64> = if fan_in_fan_out {
            (lora_b.dot(lora_a) * scaling)
                .t()
                .as_standard_layout()
                .into_owned()
        } else {
            lora_b.dot(lora_a) * scaling
        };

        if delta_f64.nrows() != rows || delta_f64.ncols() != cols {
            return Err(MergeError::ShapeMismatch {
                name: "LoRA delta shape".to_string(),
                expected: vec![rows as i32, cols as i32],
                actual: vec![delta_f64.nrows() as i32, delta_f64.ncols() as i32],
            });
        }

        write_merged_rows(base_bytes, &delta_f64, rows, cols, dtype, writer)?;
    }

    Ok(())
}

/// Fused row-by-row base + delta downcast pass.
///
/// Reads `base_bytes` and `delta` (which must have `rows` rows × `cols` cols)
/// element-by-element.  Every addition is done in f64 before the final
/// downcast to `dtype`.  The result is written directly to `writer` without
/// any intermediate allocation beyond a single row-sized output buffer.
fn write_merged_rows(
    base_bytes: &[u8],
    delta: &Array2<f64>,
    rows: usize,
    cols: usize,
    dtype: Dtype,
    writer: &mut impl Write,
) -> Result<()> {
    let elem_size: usize = match dtype {
        Dtype::F16 | Dtype::BF16 => 2,
        Dtype::F32 => 4,
        other => {
            return Err(MergeError::ModelLoad(format!(
                "unsupported dtype {other:?} in write_merged_rows"
            )));
        }
    };
    let row_bytes = cols * elem_size;

    let delta_flat = delta.as_slice().ok_or_else(|| {
        MergeError::InvalidConfig("delta array is not contiguous in memory".to_string())
    })?;

    let mut out_row: Vec<u8> = vec![0u8; row_bytes];

    for i in 0..rows {
        let base_row = &base_bytes[i * row_bytes..(i + 1) * row_bytes];
        let delta_row = &delta_flat[i * cols..(i + 1) * cols];

        match dtype {
            Dtype::F16 => {
                for ((base_chunk, &d), out_chunk) in base_row
                    .chunks_exact(2)
                    .zip(delta_row)
                    .zip(out_row.chunks_exact_mut(2))
                {
                    let base_val = f16::from_le_bytes([base_chunk[0], base_chunk[1]]).to_f64();
                    let merged =
                        ((base_val + d) as f32).clamp(f16::MIN.to_f32(), f16::MAX.to_f32());
                    out_chunk.copy_from_slice(&f16::from_f32(merged).to_le_bytes());
                }
            }
            Dtype::BF16 => {
                for ((base_chunk, &d), out_chunk) in base_row
                    .chunks_exact(2)
                    .zip(delta_row)
                    .zip(out_row.chunks_exact_mut(2))
                {
                    let base_val = bf16::from_le_bytes([base_chunk[0], base_chunk[1]]).to_f64();
                    let merged =
                        ((base_val + d) as f32).clamp(bf16::MIN.to_f32(), bf16::MAX.to_f32());
                    out_chunk.copy_from_slice(&bf16::from_f32(merged).to_le_bytes());
                }
            }
            Dtype::F32 => {
                for ((base_chunk, &d), out_chunk) in base_row
                    .chunks_exact(4)
                    .zip(delta_row)
                    .zip(out_row.chunks_exact_mut(4))
                {
                    let base_val = f32::from_le_bytes([
                        base_chunk[0],
                        base_chunk[1],
                        base_chunk[2],
                        base_chunk[3],
                    ]) as f64;
                    let merged = (base_val + d) as f32;
                    out_chunk.copy_from_slice(&merged.to_le_bytes());
                }
            }
            _ => unreachable!("dtype already validated above"),
        }

        writer.write_all(&out_row)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single-file merge
// ---------------------------------------------------------------------------

/// Merge a single-file base model with the adapter and write the output.
fn merge_single_file(
    base: &MappedFile,
    adapter: &MappedFile,
    lora_pairs: &HashMap<String, (String, String)>,
    bias_map: &HashMap<String, String>,
    scaling: f64,
    fan_in_fan_out: bool,
    low_memory: bool,
    tile_size: usize,
    output_path: &Path,
    stats: &mut LoraMergeStats,
) -> Result<()> {
    // Build the output header from the base file's tensor order and dtypes.
    let output_tensors: Vec<(String, TensorMeta)> = base
        .tensors
        .iter()
        .map(|(name, meta)| (name.clone(), meta.clone()))
        .collect();

    let header_bytes = build_header(&output_tensors, base.metadata.as_ref())?;

    // Create parent dirs if needed.
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                MergeError::ModelLoad(format!(
                    "failed to create output directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }
    }

    let out_file = File::create(output_path).map_err(|e| {
        MergeError::ModelLoad(format!(
            "failed to create output file {}: {}",
            output_path.display(),
            e
        ))
    })?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, out_file);

    // Safetensors binary format: 8-byte LE u64 header length, then header JSON, then data.
    let header_len = header_bytes.len() as u64;
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header_bytes)?;
    stats.bytes_written += 8 + header_bytes.len() as u64;

    base.advise_sequential();

    for (name, meta) in &base.tensors {
        let base_data = &base.mmap[meta.data_start..meta.data_end];

        if let Some((a_name, b_name)) = lora_pairs.get(name) {
            debug!("merging LoRA for '{name}'");

            let a_meta = adapter.tensor_meta(a_name)?;
            let b_meta = adapter.tensor_meta(b_name)?;
            let a_bytes = adapter.tensor_bytes(a_name)?;
            let b_bytes = adapter.tensor_bytes(b_name)?;

            let lora_a = bytes_to_f64_2d(a_bytes, a_meta.dtype, &a_meta.shape, a_name)?;
            let lora_b = bytes_to_f64_2d(b_bytes, b_meta.dtype, &b_meta.shape, b_name)?;

            merge_and_write_tensor(
                base_data,
                &lora_a,
                &lora_b,
                scaling,
                fan_in_fan_out,
                &meta.shape,
                meta.dtype,
                low_memory,
                tile_size,
                &mut writer,
            )?;

            stats.tensors_merged += 1;
            stats.bytes_written += base_data.len() as u64;
        } else if let Some(adapter_bias_name) = bias_map.get(name) {
            debug!("merging bias for '{name}' from adapter tensor '{adapter_bias_name}'");

            let adapter_meta = adapter.tensor_meta(adapter_bias_name)?;
            let adapter_bytes = adapter.tensor_bytes(adapter_bias_name)?;

            merge_and_write_bias(
                base_data,
                &meta.shape,
                meta.dtype,
                adapter_bytes,
                &adapter_meta.shape,
                adapter_meta.dtype,
                name,
                adapter_bias_name,
                &mut writer,
            )?;

            stats.biases_merged += 1;
            stats.bytes_written += base_data.len() as u64;
        } else {
            // Pass-through: write bytes unchanged.
            writer.write_all(base_data)?;
            stats.tensors_copied += 1;
            stats.bytes_written += base_data.len() as u64;
        }
    }

    writer.flush()?;
    base.advise_dontneed();

    Ok(())
}

// ---------------------------------------------------------------------------
// Sharded index helpers
// ---------------------------------------------------------------------------

/// Parsed `model.safetensors.index.json`.
#[derive(Debug, Deserialize)]
struct ShardIndex {
    #[serde(default)]
    metadata: serde_json::Value,
    weight_map: BTreeMap<String, String>,
}

impl ShardIndex {
    fn from_path(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path).map_err(|e| {
            MergeError::ModelLoad(format!("failed to read {}: {}", path.display(), e))
        })?;
        let idx: ShardIndex = serde_json::from_str(&contents)
            .map_err(|e| MergeError::ModelLoad(format!("failed to parse shard index: {}", e)))?;
        if idx.weight_map.is_empty() {
            return Err(MergeError::InvalidConfig(
                "model.safetensors.index.json: weight_map is empty".to_string(),
            ));
        }
        Ok(idx)
    }

    fn shard_filenames(&self) -> Vec<String> {
        let mut names: Vec<String> = self.weight_map.values().cloned().collect();
        names.sort();
        names.dedup();
        names
    }
}

// ---------------------------------------------------------------------------
// Sharded merge
// ---------------------------------------------------------------------------

fn merge_sharded(
    base_dir: &Path,
    index: &ShardIndex,
    adapter: &MappedFile,
    lora_pairs: &HashMap<String, (String, String)>,
    bias_map: &HashMap<String, String>,
    scaling: f64,
    fan_in_fan_out: bool,
    low_memory: bool,
    tile_size: usize,
    output_dir: &Path,
    stats: &mut LoraMergeStats,
) -> Result<()> {
    fs::create_dir_all(output_dir).map_err(|e| {
        MergeError::ModelLoad(format!(
            "failed to create output directory {}: {}",
            output_dir.display(),
            e
        ))
    })?;

    for shard_filename in index.shard_filenames() {
        // Guard against path traversal via a crafted index.json.
        let component = Path::new(&shard_filename);
        if component.is_absolute()
            || component
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(MergeError::InvalidConfig(format!(
                "shard filename '{shard_filename}' contains path traversal"
            )));
        }

        let shard_path = base_dir.join(&shard_filename);
        let shard = MappedFile::open(&shard_path)?;
        shard.advise_sequential();

        let output_shard_path = output_dir.join(&shard_filename);
        let shard_tensors: Vec<(String, TensorMeta)> = shard
            .tensors
            .iter()
            .map(|(n, m)| (n.clone(), m.clone()))
            .collect();

        let header_bytes = build_header(&shard_tensors, shard.metadata.as_ref())?;

        let out_file = File::create(&output_shard_path).map_err(|e| {
            MergeError::ModelLoad(format!(
                "failed to create {}: {}",
                output_shard_path.display(),
                e
            ))
        })?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, out_file);

        let header_len = header_bytes.len() as u64;
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(&header_bytes)?;
        stats.bytes_written += 8 + header_bytes.len() as u64;

        for (name, meta) in &shard.tensors {
            let base_data = &shard.mmap[meta.data_start..meta.data_end];

            if let Some((a_name, b_name)) = lora_pairs.get(name) {
                debug!("merging LoRA for '{name}' in shard '{shard_filename}'");

                let a_meta = adapter.tensor_meta(a_name)?;
                let b_meta = adapter.tensor_meta(b_name)?;
                let a_bytes = adapter.tensor_bytes(a_name)?;
                let b_bytes = adapter.tensor_bytes(b_name)?;

                let lora_a = bytes_to_f64_2d(a_bytes, a_meta.dtype, &a_meta.shape, a_name)?;
                let lora_b = bytes_to_f64_2d(b_bytes, b_meta.dtype, &b_meta.shape, b_name)?;

                merge_and_write_tensor(
                    base_data,
                    &lora_a,
                    &lora_b,
                    scaling,
                    fan_in_fan_out,
                    &meta.shape,
                    meta.dtype,
                    low_memory,
                    tile_size,
                    &mut writer,
                )?;

                stats.tensors_merged += 1;
                stats.bytes_written += base_data.len() as u64;
            } else if let Some(adapter_bias_name) = bias_map.get(name) {
                debug!(
                    "merging bias for '{name}' from adapter tensor '{adapter_bias_name}' \
                     in shard '{shard_filename}'"
                );

                let adapter_meta = adapter.tensor_meta(adapter_bias_name)?;
                let adapter_bytes = adapter.tensor_bytes(adapter_bias_name)?;

                merge_and_write_bias(
                    base_data,
                    &meta.shape,
                    meta.dtype,
                    adapter_bytes,
                    &adapter_meta.shape,
                    adapter_meta.dtype,
                    name,
                    adapter_bias_name,
                    &mut writer,
                )?;

                stats.biases_merged += 1;
                stats.bytes_written += base_data.len() as u64;
            } else {
                writer.write_all(base_data)?;
                stats.tensors_copied += 1;
                stats.bytes_written += base_data.len() as u64;
            }
        }

        writer.flush()?;
        shard.advise_dontneed();
    }

    // Copy the index.json to the output, preserving the weight_map.
    let output_index = serde_json::json!({
        "metadata": index.metadata,
        "weight_map": index.weight_map,
    });
    let json = serde_json::to_string_pretty(&output_index).map_err(MergeError::Serde)?;
    fs::write(output_dir.join("model.safetensors.index.json"), json)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Perform a streaming f64-accurate LoRA merge.
///
/// Reads `adapter_config.json` from `config.adapter_path`, resolves the base
/// model (single file or sharded directory), maps adapter tensors to base
/// tensors, and writes the merged model to `config.output_path`.
///
/// Non-adapted tensors are copied byte-for-byte (zero-allocation path).
///
/// # Errors
///
/// Returns [`MergeError`] if any of the following occur:
///
/// - `adapter_config.json` is missing or malformed
/// - `adapter_model.safetensors` is missing
/// - Base model path does not exist or contains no safetensors files
/// - A LoRA tensor has an unsupported dtype or unexpected shape
/// - I/O or serialization errors during output writing
pub fn streaming_lora_merge(config: &AccurateMergeConfig) -> Result<LoraMergeStats> {
    let start = std::time::Instant::now();

    // ---- Adapter config -------------------------------------------------
    let config_path = config.adapter_path.join("adapter_config.json");
    let adapter_cfg = AdapterConfigJson::from_path(&config_path)?;
    let scaling = config.scaling.unwrap_or_else(|| adapter_cfg.scaling());

    info!(
        "LoRA merge: rank={}, alpha={}, scaling={:.4}, fan_in_fan_out={}",
        adapter_cfg.r, adapter_cfg.lora_alpha, scaling, adapter_cfg.fan_in_fan_out
    );

    // ---- Adapter weights ------------------------------------------------
    let adapter_weights_path = config.adapter_path.join("adapter_model.safetensors");
    if !adapter_weights_path.exists() {
        return Err(MergeError::ModelLoad(format!(
            "adapter_model.safetensors not found in {}",
            config.adapter_path.display()
        )));
    }
    let adapter = MappedFile::open(&adapter_weights_path)?;
    info!(
        "Loaded adapter with {} tensors from {}",
        adapter.tensors.len(),
        adapter_weights_path.display()
    );

    let mut stats = LoraMergeStats::default();

    // ---- Route single vs sharded ----------------------------------------
    let base_path = &config.base_model_path;

    if base_path.is_file() {
        // Single-file base model.
        let base = MappedFile::open(base_path)?;
        info!(
            "Single-file base model: {} tensors in {}",
            base.tensors.len(),
            base_path.display()
        );

        let base_names: Vec<&str> = base.tensors.iter().map(|(n, _)| n.as_str()).collect();
        let adapter_names: Vec<&str> = adapter.tensors.iter().map(|(n, _)| n.as_str()).collect();
        let lora_pairs = build_lora_pairs(&base_names, &adapter_names);
        let bias_map = build_bias_map(&base_names, &adapter_names);

        info!(
            "Identified {} LoRA-adapted tensors, {} adapter biases",
            lora_pairs.len(),
            bias_map.len()
        );

        merge_single_file(
            &base,
            &adapter,
            &lora_pairs,
            &bias_map,
            scaling,
            adapter_cfg.fan_in_fan_out,
            config.low_memory,
            config.tile_size,
            &config.output_path,
            &mut stats,
        )?;
    } else if base_path.is_dir() {
        let single = base_path.join("model.safetensors");
        let index_path = base_path.join("model.safetensors.index.json");

        if single.is_file() {
            // Directory containing a single model.safetensors.
            let base = MappedFile::open(&single)?;
            info!(
                "Single-file base model (in directory): {} tensors",
                base.tensors.len()
            );

            let base_names: Vec<&str> = base.tensors.iter().map(|(n, _)| n.as_str()).collect();
            let adapter_names: Vec<&str> =
                adapter.tensors.iter().map(|(n, _)| n.as_str()).collect();
            let lora_pairs = build_lora_pairs(&base_names, &adapter_names);
            let bias_map = build_bias_map(&base_names, &adapter_names);

            info!(
                "Identified {} LoRA-adapted tensors, {} adapter biases",
                lora_pairs.len(),
                bias_map.len()
            );

            merge_single_file(
                &base,
                &adapter,
                &lora_pairs,
                &bias_map,
                scaling,
                adapter_cfg.fan_in_fan_out,
                config.low_memory,
                config.tile_size,
                &config.output_path.join("model.safetensors"),
                &mut stats,
            )?;

            // Copy non-safetensors files (config.json, tokenizer, etc.).
            copy_side_files(base_path, &config.output_path)?;
        } else if index_path.is_file() {
            // Sharded base model.
            let index = ShardIndex::from_path(&index_path)?;
            info!(
                "Sharded base model: {} shards",
                index.shard_filenames().len()
            );

            // Collect all base tensor names from all shards for name mapping.
            let all_base_names: Vec<String> = index.weight_map.keys().cloned().collect();
            let base_name_refs: Vec<&str> = all_base_names.iter().map(|s| s.as_str()).collect();
            let adapter_names: Vec<&str> =
                adapter.tensors.iter().map(|(n, _)| n.as_str()).collect();
            let lora_pairs = build_lora_pairs(&base_name_refs, &adapter_names);
            let bias_map = build_bias_map(&base_name_refs, &adapter_names);

            info!(
                "Identified {} LoRA-adapted tensors, {} adapter biases across shards",
                lora_pairs.len(),
                bias_map.len()
            );

            merge_sharded(
                base_path,
                &index,
                &adapter,
                &lora_pairs,
                &bias_map,
                scaling,
                adapter_cfg.fan_in_fan_out,
                config.low_memory,
                config.tile_size,
                &config.output_path,
                &mut stats,
            )?;

            // Copy non-safetensors files.
            copy_side_files(base_path, &config.output_path)?;
        } else {
            return Err(MergeError::ModelLoad(format!(
                "base model directory '{}' contains neither model.safetensors \
                 nor model.safetensors.index.json",
                base_path.display()
            )));
        }
    } else {
        return Err(MergeError::ModelLoad(format!(
            "base model path '{}' does not exist",
            base_path.display()
        )));
    }

    stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    info!(
        "Merge complete: {} weight tensors merged, {} biases merged, {} copied, \
         {:.1} MB written, {:.1}ms",
        stats.tensors_merged,
        stats.biases_merged,
        stats.tensors_copied,
        stats.bytes_written as f64 / (1024.0 * 1024.0),
        stats.elapsed_ms,
    );

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Side-file copy helpers
// ---------------------------------------------------------------------------

/// Copy all non-safetensors and non-index files from `src_dir` to `dst_dir`.
///
/// This preserves `config.json`, `tokenizer.json`, `tokenizer_config.json`,
/// `special_tokens_map.json`, and similar metadata files that are needed for
/// a complete model directory.
fn copy_side_files(src_dir: &Path, dst_dir: &Path) -> Result<()> {
    fs::create_dir_all(dst_dir).map_err(|e| {
        MergeError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to create {}: {}", dst_dir.display(), e),
        ))
    })?;

    for entry in fs::read_dir(src_dir)? {
        let entry = entry?;
        let src = entry.path();
        if !src.is_file() {
            continue;
        }

        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip files that we write ourselves.
        if name_str.ends_with(".safetensors") || name_str == "model.safetensors.index.json" {
            continue;
        }

        let dst = dst_dir.join(&name);
        fs::copy(&src, &dst).map_err(|e| {
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
        debug!("copied {}", name_str);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{TensorView, serialize};

    // ---- helpers -----------------------------------------------------------

    fn write_safetensors_f16(path: &Path, tensors: Vec<(&str, Vec<f32>, Vec<usize>)>) {
        let bytes_map: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
            .into_iter()
            .map(|(name, vals, shape)| {
                let bytes: Vec<u8> = vals
                    .iter()
                    .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                    .collect();
                (name.to_string(), bytes, shape)
            })
            .collect();

        let views: Vec<_> = bytes_map
            .iter()
            .map(|(name, bytes, shape)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();

        let serialized = serialize(views, None).unwrap();
        fs::write(path, serialized).unwrap();
    }

    fn write_adapter_config(path: &Path, r: u32, alpha: f32) {
        let json = serde_json::json!({
            "r": r,
            "lora_alpha": alpha,
            "target_modules": ["q_proj", "v_proj"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        });
        fs::write(path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }

    fn read_f16_tensor(path: &Path, tensor_name: &str) -> Vec<f32> {
        let file = MappedFile::open(path).unwrap();
        let meta = file.tensor_meta(tensor_name).unwrap();
        let bytes = file.tensor_bytes(tensor_name).unwrap();
        bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect::<Vec<_>>()[..(meta.shape.iter().product::<usize>())]
            .to_vec()
    }

    // ---- bytes_to_f64_2d ---------------------------------------------------

    #[test]
    fn bytes_to_f64_roundtrip_f16() {
        let vals = [1.0_f32, -2.0, 0.5, 3.0];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        let arr = bytes_to_f64_2d(&bytes, Dtype::F16, &[2, 2], "test").unwrap();
        assert!((arr[[0, 0]] - 1.0).abs() < 0.01);
        assert!((arr[[0, 1]] - (-2.0)).abs() < 0.01);
        assert!((arr[[1, 0]] - 0.5).abs() < 0.01);
        assert!((arr[[1, 1]] - 3.0).abs() < 0.01);
    }

    #[test]
    fn bytes_to_f64_rejects_1d_shape() {
        let bytes = vec![0u8; 8];
        let err = bytes_to_f64_2d(&bytes, Dtype::F32, &[2], "t").unwrap_err();
        assert!(err.to_string().contains("expected 2D"));
    }

    #[test]
    fn bytes_to_f64_rejects_wrong_byte_len() {
        let bytes = vec![0u8; 4]; // only 1 f32, not 2x2=4 f32s
        let err = bytes_to_f64_2d(&bytes, Dtype::F32, &[2, 2], "t").unwrap_err();
        assert!(err.to_string().contains("byte length") || err.to_string().contains("shape"));
    }

    // ---- merge math --------------------------------------------------------

    #[test]
    fn merge_identity_gives_base_unchanged() {
        // B = [[0],[0]], A = [[0,0]] — zero delta, output = base
        let base_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let base_bytes: Vec<u8> = base_vals
            .iter()
            .flat_map(|&v| f32::to_le_bytes(v))
            .collect();

        let lora_a = Array2::from_shape_vec((1, 2), vec![0.0_f64, 0.0]).unwrap();
        let lora_b = Array2::from_shape_vec((2, 1), vec![0.0_f64, 0.0]).unwrap();

        let mut out = Vec::new();
        merge_and_write_tensor(
            &base_bytes,
            &lora_a,
            &lora_b,
            1.0,
            false,
            &[2, 2],
            Dtype::F32,
            false,
            512,
            &mut out,
        )
        .unwrap();

        let result: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(result, base_vals);
    }

    #[test]
    fn merge_adds_correct_delta() {
        // base = I2, B = [[1],[1]], A = [[1,1]], scaling = 2.0
        // delta = B @ A = [[1,1],[1,1]], scaled = [[2,2],[2,2]]
        // result = [[3,2],[2,3]]
        let base_vals: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let base_bytes: Vec<u8> = base_vals
            .iter()
            .flat_map(|&v| f32::to_le_bytes(v))
            .collect();

        let lora_a = Array2::from_shape_vec((1, 2), vec![1.0_f64, 1.0]).unwrap();
        let lora_b = Array2::from_shape_vec((2, 1), vec![1.0_f64, 1.0]).unwrap();

        let mut out = Vec::new();
        merge_and_write_tensor(
            &base_bytes,
            &lora_a,
            &lora_b,
            2.0,
            false,
            &[2, 2],
            Dtype::F32,
            false,
            512,
            &mut out,
        )
        .unwrap();

        let result: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 2.0).abs() < 1e-6);
        assert!((result[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn tiled_path_matches_full_path() {
        let base_vals: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let base_bytes: Vec<u8> = base_vals
            .iter()
            .flat_map(|&v| f32::to_le_bytes(v))
            .collect();

        let lora_a = Array2::from_shape_vec((2, 4), vec![0.1_f64; 8]).unwrap();
        let lora_b = Array2::from_shape_vec((4, 2), vec![0.2_f64; 8]).unwrap();

        let mut out_full = Vec::new();
        merge_and_write_tensor(
            &base_bytes,
            &lora_a,
            &lora_b,
            0.5,
            false,
            &[4, 4],
            Dtype::F32,
            false,
            512,
            &mut out_full,
        )
        .unwrap();

        let mut out_tiled = Vec::new();
        merge_and_write_tensor(
            &base_bytes,
            &lora_a,
            &lora_b,
            0.5,
            false,
            &[4, 4],
            Dtype::F32,
            true,
            2, // tile_size = 2 rows
            &mut out_tiled,
        )
        .unwrap();

        let full: Vec<f32> = out_full
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let tiled: Vec<f32> = out_tiled
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for (a, b) in full.iter().zip(tiled.iter()) {
            assert!((a - b).abs() < 1e-6, "full={a} tiled={b}");
        }
    }

    #[test]
    fn fan_in_fan_out_transposes_delta() {
        // base = I2 (stored transposed), B = [[3],[4]], A = [[1,2]]
        // delta (no fif) = [[3,6],[4,8]]
        // delta (fif)    = (B @ A).T = [[3,4],[6,8]]
        // result (fif)   = I2 + [[3,4],[6,8]] = [[4,4],[6,9]]
        let base_vals: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let base_bytes: Vec<u8> = base_vals
            .iter()
            .flat_map(|&v| f32::to_le_bytes(v))
            .collect();

        let lora_a = Array2::from_shape_vec((1, 2), vec![1.0_f64, 2.0]).unwrap();
        let lora_b = Array2::from_shape_vec((2, 1), vec![3.0_f64, 4.0]).unwrap();

        let mut out = Vec::new();
        merge_and_write_tensor(
            &base_bytes,
            &lora_a,
            &lora_b,
            1.0,
            true, // fan_in_fan_out
            &[2, 2],
            Dtype::F32,
            false,
            512,
            &mut out,
        )
        .unwrap();

        let result: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!((result[0] - 4.0).abs() < 1e-6, "result[0]={}", result[0]);
        assert!((result[1] - 4.0).abs() < 1e-6, "result[1]={}", result[1]);
        assert!((result[2] - 6.0).abs() < 1e-6, "result[2]={}", result[2]);
        assert!((result[3] - 9.0).abs() < 1e-6, "result[3]={}", result[3]);
    }

    // ---- end-to-end --------------------------------------------------------

    #[test]
    fn end_to_end_single_file() {
        let dir = tempfile::tempdir().unwrap();

        // Base model: q_proj weight (identity) + embed (pass-through)
        let base_path = dir.path().join("model.safetensors");
        write_safetensors_f16(
            &base_path,
            vec![
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    vec![1.0, 0.0, 0.0, 1.0],
                    vec![2, 2],
                ),
                (
                    "model.embed_tokens.weight",
                    vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    vec![3, 2],
                ),
            ],
        );

        // Adapter
        let adapter_dir = dir.path().join("adapter");
        fs::create_dir_all(&adapter_dir).unwrap();
        write_adapter_config(&adapter_dir.join("adapter_config.json"), 1, 1.0);
        write_safetensors_f16(
            &adapter_dir.join("adapter_model.safetensors"),
            vec![
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    vec![1.0, 1.0],
                    vec![1, 2],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![1.0, 1.0],
                    vec![2, 1],
                ),
            ],
        );

        let output_path = dir.path().join("merged.safetensors");
        let config = AccurateMergeConfig::new(&base_path, &adapter_dir, &output_path);
        let stats = streaming_lora_merge(&config).unwrap();

        assert_eq!(stats.tensors_merged, 1);
        assert_eq!(stats.tensors_copied, 1);

        // Verify: base I2 + 1.0*(B@A) = I2 + [[1,1],[1,1]] = [[2,1],[1,2]]
        let result = read_f16_tensor(&output_path, "model.layers.0.self_attn.q_proj.weight");
        assert!((result[0] - 2.0).abs() < 0.02, "result[0]={}", result[0]);
        assert!((result[1] - 1.0).abs() < 0.02, "result[1]={}", result[1]);
        assert!((result[2] - 1.0).abs() < 0.02, "result[2]={}", result[2]);
        assert!((result[3] - 2.0).abs() < 0.02, "result[3]={}", result[3]);

        // Embed should be unchanged
        let embed = read_f16_tensor(&output_path, "model.embed_tokens.weight");
        for v in &embed {
            assert!((v - 0.5).abs() < 0.02, "embed element: {v}");
        }
    }

    #[test]
    fn end_to_end_low_memory_path() {
        let dir = tempfile::tempdir().unwrap();

        let base_path = dir.path().join("model.safetensors");
        write_safetensors_f16(
            &base_path,
            vec![(
                "model.layers.0.self_attn.q_proj.weight",
                vec![1.0, 0.0, 0.0, 1.0],
                vec![2, 2],
            )],
        );

        let adapter_dir = dir.path().join("adapter");
        fs::create_dir_all(&adapter_dir).unwrap();
        write_adapter_config(&adapter_dir.join("adapter_config.json"), 1, 1.0);
        write_safetensors_f16(
            &adapter_dir.join("adapter_model.safetensors"),
            vec![
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    vec![1.0, 1.0],
                    vec![1, 2],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![1.0, 1.0],
                    vec![2, 1],
                ),
            ],
        );

        let output_path = dir.path().join("merged_low_mem.safetensors");
        let config =
            AccurateMergeConfig::new(&base_path, &adapter_dir, &output_path).with_low_memory(1);
        let stats = streaming_lora_merge(&config).unwrap();

        assert_eq!(stats.tensors_merged, 1);

        let result = read_f16_tensor(&output_path, "model.layers.0.self_attn.q_proj.weight");
        assert!((result[0] - 2.0).abs() < 0.02);
        assert!((result[3] - 2.0).abs() < 0.02);
    }

    #[test]
    fn missing_adapter_config_errors() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("model.safetensors");
        fs::write(&base_path, b"dummy").unwrap();

        let adapter_dir = dir.path().join("adapter");
        fs::create_dir_all(&adapter_dir).unwrap();
        // No adapter_config.json — should error.

        let config = AccurateMergeConfig::new(&base_path, &adapter_dir, dir.path().join("out.st"));
        let err = streaming_lora_merge(&config).unwrap_err();
        assert!(err.to_string().contains("adapter_config.json"));
    }

    #[test]
    fn missing_base_model_errors() {
        let dir = tempfile::tempdir().unwrap();
        let adapter_dir = dir.path().join("adapter");
        fs::create_dir_all(&adapter_dir).unwrap();
        write_adapter_config(&adapter_dir.join("adapter_config.json"), 4, 8.0);
        // Write a minimal adapter weights file so we get past that check.
        write_safetensors_f16(
            &adapter_dir.join("adapter_model.safetensors"),
            vec![("base_model.model.some.weight", vec![0.0], vec![1, 1])],
        );

        let config = AccurateMergeConfig::new(
            dir.path().join("nonexistent"),
            &adapter_dir,
            dir.path().join("out"),
        );
        let err = streaming_lora_merge(&config).unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn scaling_override_is_respected() {
        let dir = tempfile::tempdir().unwrap();

        let base_path = dir.path().join("model.safetensors");
        // base = zero matrix so result = scaling * (B@A)
        write_safetensors_f16(
            &base_path,
            vec![(
                "model.layers.0.self_attn.q_proj.weight",
                vec![0.0, 0.0, 0.0, 0.0],
                vec![2, 2],
            )],
        );

        let adapter_dir = dir.path().join("adapter");
        fs::create_dir_all(&adapter_dir).unwrap();
        // Config says scaling = alpha/r = 1/1 = 1.0, but we override to 3.0.
        write_adapter_config(&adapter_dir.join("adapter_config.json"), 1, 1.0);
        write_safetensors_f16(
            &adapter_dir.join("adapter_model.safetensors"),
            vec![
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    vec![1.0, 0.0],
                    vec![1, 2],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![1.0, 0.0],
                    vec![2, 1],
                ),
            ],
        );

        let output_path = dir.path().join("out.safetensors");
        let config =
            AccurateMergeConfig::new(&base_path, &adapter_dir, &output_path).with_scaling(3.0);
        streaming_lora_merge(&config).unwrap();

        // delta = B @ A = [[1,0],[0,0]], scaled by 3.0 → [[3,0],[0,0]]
        let result = read_f16_tensor(&output_path, "model.layers.0.self_attn.q_proj.weight");
        assert!((result[0] - 3.0).abs() < 0.1, "result[0]={}", result[0]);
        assert!(result[1].abs() < 0.05);
        assert!(result[2].abs() < 0.05);
        assert!(result[3].abs() < 0.05);
    }

    // ---- bias merge --------------------------------------------------------

    #[test]
    fn bias_merge_adds_correctly_f32() {
        let base_vals: Vec<f32> = vec![1.0, 2.0, 3.0];
        let base_bytes: Vec<u8> = base_vals
            .iter()
            .flat_map(|&v| f32::to_le_bytes(v))
            .collect();

        let adapter_vals: Vec<f32> = vec![0.1, 0.2, 0.3];
        let adapter_bytes: Vec<u8> = adapter_vals
            .iter()
            .flat_map(|&v| f32::to_le_bytes(v))
            .collect();

        let mut out = Vec::new();
        merge_and_write_bias(
            &base_bytes,
            &[3],
            Dtype::F32,
            &adapter_bytes,
            &[3],
            Dtype::F32,
            "test.bias",
            "adapter.test.bias",
            &mut out,
        )
        .unwrap();

        let result: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!((result[0] - 1.1).abs() < 1e-5, "got {}", result[0]);
        assert!((result[1] - 2.2).abs() < 1e-5, "got {}", result[1]);
        assert!((result[2] - 3.3).abs() < 1e-5, "got {}", result[2]);
    }

    #[test]
    fn bias_merge_size_mismatch_errors() {
        let base_bytes: Vec<u8> = vec![0u8; 8]; // 2 x f32
        let adapter_bytes: Vec<u8> = vec![0u8; 12]; // 3 x f32

        let err = merge_and_write_bias(
            &base_bytes,
            &[2],
            Dtype::F32,
            &adapter_bytes,
            &[3],
            Dtype::F32,
            "a.bias",
            "b.bias",
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("mismatch") || err.to_string().contains("size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn build_bias_map_detects_adapter_biases() {
        let base_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.v_proj.weight",
        ];
        let adapter_names = [
            // LoRA weights — must NOT appear in bias_map
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            // Bias tensor — must appear
            "base_model.model.model.layers.0.self_attn.q_proj.bias",
            // Bias with no base counterpart — must NOT appear
            "base_model.model.model.layers.0.self_attn.v_proj.bias",
        ];

        let map = build_bias_map(&base_names, &adapter_names);

        assert!(
            map.contains_key("model.layers.0.self_attn.q_proj.bias"),
            "q_proj.bias should be in bias map"
        );
        assert!(
            !map.contains_key("model.layers.0.self_attn.q_proj.weight"),
            "weight tensor must not be in bias map"
        );
        assert!(
            !map.contains_key("model.layers.0.self_attn.v_proj.bias"),
            "bias without base counterpart must not be in map"
        );
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn end_to_end_bias_merge() {
        let dir = tempfile::tempdir().unwrap();

        // Base model: q_proj weight + q_proj bias
        let base_path = dir.path().join("model.safetensors");
        write_safetensors_f16(
            &base_path,
            vec![
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    vec![1.0, 0.0, 0.0, 1.0],
                    vec![2, 2],
                ),
                (
                    "model.layers.0.self_attn.q_proj.bias",
                    vec![0.5, 0.5],
                    vec![2],
                ),
            ],
        );

        let adapter_dir = dir.path().join("adapter");
        fs::create_dir_all(&adapter_dir).unwrap();
        write_adapter_config(&adapter_dir.join("adapter_config.json"), 1, 1.0);
        // Adapter has both lora_A/B and a bias term.
        write_safetensors_f16(
            &adapter_dir.join("adapter_model.safetensors"),
            vec![
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    vec![1.0, 1.0],
                    vec![1, 2],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![1.0, 1.0],
                    vec![2, 1],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.bias",
                    vec![0.25, 0.25],
                    vec![2],
                ),
            ],
        );

        let output_path = dir.path().join("merged.safetensors");
        let config = AccurateMergeConfig::new(&base_path, &adapter_dir, &output_path);
        let stats = streaming_lora_merge(&config).unwrap();

        assert_eq!(stats.tensors_merged, 1, "one weight tensor merged");
        assert_eq!(stats.biases_merged, 1, "one bias tensor merged");

        // bias result = 0.5 + 0.25 = 0.75 for each element
        let bias = read_f16_tensor(&output_path, "model.layers.0.self_attn.q_proj.bias");
        assert_eq!(bias.len(), 2);
        assert!((bias[0] - 0.75).abs() < 0.02, "bias[0]={}", bias[0]);
        assert!((bias[1] - 0.75).abs() < 0.02, "bias[1]={}", bias[1]);
    }
}
