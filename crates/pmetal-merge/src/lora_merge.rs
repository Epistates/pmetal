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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use ndarray::{Array2, s};
use pmetal_runtime::compat::Array;
use safetensors::Dtype;
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::loader::{copy_side_files, decode_f32, write_atomic};
use crate::mapped::{MappedFile, TensorMeta};
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
///
/// Both layouts are read: PEFT's (`lora_alpha`, `use_rslora`, `use_dora`,
/// `alpha_pattern`, `fan_in_fan_out`) and the one `pmetal train` writes
/// (`alpha`, `use_rslora`).
#[derive(Debug, Clone, Deserialize)]
struct AdapterConfigJson {
    r: u32,
    #[serde(alias = "alpha")]
    lora_alpha: f64,
    #[serde(default)]
    fan_in_fan_out: bool,
    /// rsLoRA (Kalajdzievski 2023): scale by `alpha / sqrt(r)`.
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    use_dora: bool,
    /// Per-module `alpha` overrides, keyed by module-name regex (PEFT).
    #[serde(default)]
    alpha_pattern: HashMap<String, f64>,
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

    /// The `alpha` of `module`: the first `alpha_pattern` entry whose key
    /// matches the module name (PEFT matches `(.*\.)?<key>$`), else `lora_alpha`.
    fn alpha_for(&self, module: &str) -> Result<f64> {
        let mut keys: Vec<&String> = self.alpha_pattern.keys().collect();
        keys.sort();
        for key in keys {
            let re = regex::Regex::new(&format!(r"^(.*\.)?{key}$")).map_err(|e| {
                MergeError::InvalidConfig(format!("alpha_pattern key {key:?} is not a regex: {e}"))
            })?;
            if re.is_match(module) {
                return Ok(self.alpha_pattern[key]);
            }
        }
        Ok(self.lora_alpha)
    }

    /// The scale applied to `B A` for a module whose adapter has `rank`:
    /// `alpha / rank`, or `alpha / sqrt(rank)` under rsLoRA.
    fn scaling_for(&self, module: &str, rank: usize) -> Result<f64> {
        let alpha = self.alpha_for(module)?;
        let rank = rank as f64;
        Ok(if self.use_rslora {
            alpha / rank.sqrt()
        } else {
            alpha / rank
        })
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
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_le_bytes(*c).to_f64())
                .collect()
        }
        Dtype::BF16 => {
            if bytes.len() != n * 2 {
                return Err(byte_len_mismatch(name, n * 2, bytes.len()));
            }
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_le_bytes(*c).to_f64())
                .collect()
        }
        Dtype::F32 => {
            if bytes.len() != n * 4 {
                return Err(byte_len_mismatch(name, n * 4, bytes.len()));
            }
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c) as f64)
                .collect()
        }
        Dtype::F64 => {
            if bytes.len() != n * 8 {
                return Err(byte_len_mismatch(name, n * 8, bytes.len()));
            }
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
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
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_le_bytes(*c).to_f64())
                .collect()
        }
        Dtype::BF16 => {
            if bytes.len() != n * 2 {
                return Err(byte_len_mismatch(name, n * 2, bytes.len()));
            }
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_le_bytes(*c).to_f64())
                .collect()
        }
        Dtype::F32 => {
            if bytes.len() != n * 4 {
                return Err(byte_len_mismatch(name, n * 4, bytes.len()));
            }
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c) as f64)
                .collect()
        }
        Dtype::F64 => {
            if bytes.len() != n * 8 {
                return Err(byte_len_mismatch(name, n * 8, bytes.len()));
            }
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
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
#[expect(
    clippy::too_many_arguments,
    reason = "bytes, shape, dtype and name of each operand"
)]
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

/// What a LoRA adapter tensor is, from its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdapterPart {
    /// Down-projection `A`, `[r, in]`.
    A,
    /// Up-projection `B`, `[out, r]`.
    B,
    /// DoRA magnitude vector, one entry per output row.
    Magnitude,
    /// A stacked expert projection's down-projections, `[experts, r, in]`.
    ExpertA,
    /// A stacked expert projection's up-projections, `[experts, out, r]`.
    ExpertB,
}

/// Split an adapter tensor name into its module path and what it holds.
///
/// PEFT: `base_model.model.<module>.lora_A.weight` (also `lora_B`, and
/// `lora_magnitude_vector[.weight]`); a routed experts' stacked projection
/// (`<layer>.switch_mlp.<proj>`) has `lora_expert_A` / `lora_expert_B`, one
/// factor pair per expert. Native (`pmetal train`): `<module>.lora_a`,
/// `.lora_b`, `.lora_magnitude`.
fn classify_adapter_tensor(name: &str) -> Option<(&str, AdapterPart)> {
    let name = name.strip_prefix(ADAPTER_PREFIX).unwrap_or(name);
    const SUFFIXES: [(&str, AdapterPart); 9] = [
        (".lora_expert_A", AdapterPart::ExpertA),
        (".lora_expert_B", AdapterPart::ExpertB),
        (".lora_A.weight", AdapterPart::A),
        (".lora_B.weight", AdapterPart::B),
        (".lora_magnitude_vector.weight", AdapterPart::Magnitude),
        (".lora_magnitude_vector", AdapterPart::Magnitude),
        (".lora_a", AdapterPart::A),
        (".lora_b", AdapterPart::B),
        (".lora_magnitude", AdapterPart::Magnitude),
    ];
    SUFFIXES
        .iter()
        .find_map(|(suffix, part)| name.strip_suffix(suffix).map(|module| (module, *part)))
}

/// One adapted weight: where its factors live in the adapter file and the
/// scale to fold them with.
#[derive(Debug, Clone)]
struct LoraTarget {
    a: String,
    b: String,
    /// DoRA magnitude vector, if the adapter has one for this module.
    magnitude: Option<String>,
    scaling: f64,
}

/// One expert's share of a base tensor: `W[expert][rows..] += scaling · B_e A_e`.
#[derive(Debug, Clone)]
struct ExpertPatch {
    /// The adapter's `[experts, r, in]` and `[experts, out, r]` tensors.
    a: String,
    b: String,
    scaling: f64,
    /// Which expert of the adapter.
    adapter_expert: usize,
    /// The expert's slot in a stacked base tensor `[experts, rows, cols]`;
    /// `None` when the base tensor is that one expert's `[rows, cols]`.
    base_expert: Option<usize>,
    /// First row of the expert's `[out, in]` block the patch covers (a fused
    /// gate and up projection share a block, the up half below the gate's).
    row_offset: usize,
}

/// Everything an adapter folds into: dense weights and the experts' tensors.
struct Targets {
    dense: HashMap<String, LoraTarget>,
    /// By base tensor name.
    experts: HashMap<String, Vec<ExpertPatch>>,
}

impl Targets {
    fn len(&self) -> usize {
        self.dense.len() + self.experts.len()
    }
}

/// The base tensors holding the experts of an adapted stacked projection
/// `module` (`<layer>.switch_mlp.<proj>`), in the layout the checkpoint uses:
///
/// * stacked, `<module>.weight` `[experts, out, in]` (an MLX conversion);
/// * one tensor per expert, `<layer>.experts.{e}.<proj>.weight` (the released
///   layout);
/// * fused, `<layer>.experts.gate_up_proj` (the gate projection above the up
///   projection) and `<layer>.experts.down_proj` (transformers 5).
///
/// `fc1` / `fc2` of a gate-less expert (Nemotron-H) are `up_proj` / `down_proj`
/// in the released layout.
fn expert_patches(
    module: &str,
    experts: usize,
    out_rows: usize,
    base_names: &HashSet<&str>,
    make: impl Fn(Option<usize>, usize, usize) -> ExpertPatch,
) -> Result<Option<Vec<(String, ExpertPatch)>>> {
    let exists = |name: &str| base_names.contains(name);
    // The adapter names modules as the Hugging Face model does, and a native
    // base may drop the leading `model.`.
    let prefixes = ["", "model."];

    for prefix in prefixes {
        let stacked = format!("{prefix}{module}.weight");
        if exists(&stacked) {
            return Ok(Some(
                (0..experts)
                    .map(|e| (stacked.clone(), make(Some(e), e, 0)))
                    .collect(),
            ));
        }
    }

    let Some((layer, projection)) = module.split_once(".switch_mlp.") else {
        return Ok(None);
    };
    let released = match projection {
        "fc1" => "up_proj",
        "fc2" => "down_proj",
        other => other,
    };
    for prefix in prefixes {
        let layer = format!("{prefix}{layer}");
        let names: Vec<String> = (0..experts)
            .map(|e| format!("{layer}.experts.{e}.{released}.weight"))
            .collect();
        let present = names.iter().filter(|n| exists(n)).count();
        if present == experts {
            return Ok(Some(
                names
                    .into_iter()
                    .enumerate()
                    .map(|(e, name)| (name, make(None, e, 0)))
                    .collect(),
            ));
        }
        if present > 0 {
            let missing = names.iter().find(|n| !exists(n)).expect("one is missing");
            return Err(MergeError::InvalidConfig(format!(
                "the base model holds {present} of the {experts} experts of '{module}' \
                 ({missing} is missing)"
            )));
        }
        let (fused, row_offset) = match released {
            "gate_proj" => (format!("{layer}.experts.gate_up_proj"), 0),
            "up_proj" => (format!("{layer}.experts.gate_up_proj"), out_rows),
            "down_proj" => (format!("{layer}.experts.down_proj"), 0),
            _ => continue,
        };
        if exists(&fused) {
            return Ok(Some(
                (0..experts)
                    .map(|e| (fused.clone(), make(Some(e), e, row_offset)))
                    .collect(),
            ));
        }
    }
    Ok(None)
}

/// Map each adapted base weight to its [`LoraTarget`].
///
/// The base weight of module `m` is `m.weight`, or `model.m.weight` (the
/// native layout drops the leading `model.`). Everything that looks like an
/// adapter tensor but cannot be applied is an error, because a merge that
/// quietly skips layers hands back a model that is not the fine-tune.
fn discover_targets(
    base_names: &HashSet<&str>,
    adapter: &MappedFile,
    cfg: &AdapterConfigJson,
    scaling_override: Option<f64>,
) -> Result<Targets> {
    struct Parts {
        a: Option<String>,
        b: Option<String>,
        magnitude: Option<String>,
        expert_a: Option<String>,
        expert_b: Option<String>,
    }
    let mut modules: BTreeMap<&str, Parts> = BTreeMap::new();
    let mut unsupported: Vec<&str> = Vec::new();
    for (name, _) in &adapter.tensors {
        match classify_adapter_tensor(name) {
            Some((module, part)) => {
                let parts = modules.entry(module).or_insert(Parts {
                    a: None,
                    b: None,
                    magnitude: None,
                    expert_a: None,
                    expert_b: None,
                });
                let slot = match part {
                    AdapterPart::A => &mut parts.a,
                    AdapterPart::B => &mut parts.b,
                    AdapterPart::Magnitude => &mut parts.magnitude,
                    AdapterPart::ExpertA => &mut parts.expert_a,
                    AdapterPart::ExpertB => &mut parts.expert_b,
                };
                *slot = Some(name.clone());
            }
            None if name.contains("lora_") || name.contains(".modules_to_save.") => {
                unsupported.push(name);
            }
            None => {}
        }
    }
    if !unsupported.is_empty() {
        return Err(MergeError::InvalidConfig(format!(
            "adapter tensors of a kind this merge cannot fold ({} of them, e.g. {}): embedding \
             LoRA and modules_to_save are not supported",
            unsupported.len(),
            unsupported[..unsupported.len().min(3)].join(", ")
        )));
    }

    let mut targets = HashMap::new();
    let mut expert_targets: HashMap<String, Vec<ExpertPatch>> = HashMap::new();
    let mut unmatched: Vec<&str> = Vec::new();
    for (module, parts) in &modules {
        if parts.expert_a.is_some() || parts.expert_b.is_some() {
            let (Some(a), Some(b)) = (&parts.expert_a, &parts.expert_b) else {
                return Err(MergeError::InvalidConfig(format!(
                    "adapter has {} for '{module}' but not its partner",
                    if parts.expert_a.is_some() {
                        "lora_expert_A"
                    } else {
                        "lora_expert_B"
                    }
                )));
            };
            if parts.a.is_some() || parts.b.is_some() || parts.magnitude.is_some() {
                return Err(MergeError::InvalidConfig(format!(
                    "adapter holds both per-expert and dense factors for '{module}'"
                )));
            }
            let (a_meta, b_meta) = (adapter.tensor_meta(a)?, adapter.tensor_meta(b)?);
            let (&[experts, rank, _], &[b_experts, out_rows, b_rank]) =
                (a_meta.shape.as_slice(), b_meta.shape.as_slice())
            else {
                return Err(MergeError::InvalidConfig(format!(
                    "the expert factors of '{module}' are {:?} and {:?}, not [experts, r, in] and \
                     [experts, out, r]",
                    a_meta.shape, b_meta.shape
                )));
            };
            if (b_experts, b_rank) != (experts, rank) {
                return Err(MergeError::ShapeMismatch {
                    name: format!("expert factors of '{module}'"),
                    expected: vec![experts as i32, rank as i32],
                    actual: vec![b_experts as i32, b_rank as i32],
                });
            }
            let scaling = match scaling_override {
                Some(s) => s,
                None => cfg.scaling_for(module, rank)?,
            };
            let make = |base_expert: Option<usize>, adapter_expert: usize, row_offset: usize| {
                ExpertPatch {
                    a: a.clone(),
                    b: b.clone(),
                    scaling,
                    adapter_expert,
                    base_expert,
                    row_offset,
                }
            };
            match expert_patches(module, experts, out_rows, base_names, make)? {
                Some(patches) => {
                    for (name, patch) in patches {
                        expert_targets.entry(name).or_default().push(patch);
                    }
                }
                None => unmatched.push(module),
            }
            continue;
        }
        let (Some(a), Some(b)) = (&parts.a, &parts.b) else {
            return Err(MergeError::InvalidConfig(format!(
                "adapter has {} for '{module}' but not its partner",
                if parts.a.is_some() {
                    "lora_A"
                } else {
                    "lora_B"
                }
            )));
        };
        let candidates = [format!("{module}.weight"), format!("model.{module}.weight")];
        let Some(base_name) = candidates.iter().find(|c| base_names.contains(c.as_str())) else {
            unmatched.push(module);
            continue;
        };
        let a_meta = adapter.tensor_meta(a)?;
        let rank = *a_meta.shape.first().ok_or_else(|| {
            MergeError::InvalidConfig(format!("lora_A for '{module}' is a scalar"))
        })?;
        let scaling = match scaling_override {
            Some(s) => s,
            None => cfg.scaling_for(module, rank)?,
        };
        if cfg.use_dora && parts.magnitude.is_none() {
            return Err(MergeError::InvalidConfig(format!(
                "adapter_config says use_dora but '{module}' has no magnitude vector"
            )));
        }
        targets.insert(
            base_name.clone(),
            LoraTarget {
                a: a.clone(),
                b: b.clone(),
                magnitude: parts.magnitude.clone(),
                scaling,
            },
        );
    }
    if !unmatched.is_empty() {
        return Err(MergeError::InvalidConfig(format!(
            "{} adapted module(s) have no weight in the base model (e.g. {}); is this adapter \
             for a different model?",
            unmatched.len(),
            unmatched[..unmatched.len().min(3)].join(", ")
        )));
    }
    if targets.is_empty() && expert_targets.is_empty() {
        return Err(MergeError::InvalidConfig(
            "the adapter holds no LoRA tensors".to_string(),
        ));
    }
    Ok(Targets {
        dense: targets,
        experts: expert_targets,
    })
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
    magnitude: Option<&[f64]>,
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

    if let Some(m) = magnitude {
        if m.len() != rows {
            return Err(MergeError::ShapeMismatch {
                name: "DoRA magnitude length".to_string(),
                expected: vec![rows as i32],
                actual: vec![m.len() as i32],
            });
        }
    }

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
            write_merged_rows(
                base_tile,
                &delta_tile,
                tile_rows,
                cols,
                dtype,
                magnitude.map(|m| &m[start..end]),
                writer,
            )?;

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

        write_merged_rows(base_bytes, &delta_f64, rows, cols, dtype, magnitude, writer)?;
    }

    Ok(())
}

/// Fold per-expert adapters into one base tensor of experts.
///
/// The tensor is `[rows, cols]` (one expert) or `[experts, rows, cols]`
/// (stacked, or fused with the gate above the up projection). Expert by
/// expert, each patch adds `scaling · B_e A_e` (in f64) to its rows of the
/// expert's block, so at any moment one expert's delta is live, not the stack's.
fn merge_and_write_experts(
    ctx: &MergeContext<'_>,
    name: &str,
    base_bytes: &[u8],
    shape: &[usize],
    dtype: Dtype,
    patches: &[ExpertPatch],
    writer: &mut impl Write,
) -> Result<()> {
    let (blocks, rows, cols) = match *shape {
        [rows, cols] => (1, rows, cols),
        [experts, rows, cols] => (experts, rows, cols),
        _ => {
            return Err(MergeError::InvalidConfig(format!(
                "expert tensor '{name}' is {shape:?}, not [rows, cols] or [experts, rows, cols]"
            )));
        }
    };
    let elem = match dtype {
        Dtype::F16 | Dtype::BF16 => 2usize,
        Dtype::F32 => 4,
        other => {
            return Err(MergeError::ModelLoad(format!(
                "expert tensor '{name}' is {other:?}: an expert adapter folds into floating-point \
                 expert weights, not quantized ones"
            )));
        }
    };
    let block_bytes = rows * cols * elem;
    if base_bytes.len() != blocks * block_bytes {
        return Err(MergeError::ShapeMismatch {
            name: format!("bytes of '{name}'"),
            expected: vec![(blocks * block_bytes) as i32],
            actual: vec![base_bytes.len() as i32],
        });
    }

    // One expert's `[dim0, dim1]` factor, decoded from the adapter's stack.
    let factor = |tensor: &str, expert: usize| -> Result<Array2<f64>> {
        let meta = ctx.adapter.tensor_meta(tensor)?;
        let [experts, dim0, dim1] = meta.shape[..] else {
            return Err(MergeError::InvalidConfig(format!(
                "'{tensor}' is {:?}, not a stack of factors",
                meta.shape
            )));
        };
        if expert >= experts {
            return Err(MergeError::InvalidConfig(format!(
                "'{tensor}' holds {experts} experts, and '{name}' asks for expert {expert}"
            )));
        }
        let size = dtype_elem_size(meta.dtype)?;
        let at = expert * dim0 * dim1 * size;
        let bytes = ctx.adapter.tensor_bytes(tensor)?;
        bytes_to_f64_2d(
            &bytes[at..at + dim0 * dim1 * size],
            meta.dtype,
            &[dim0, dim1],
            tensor,
        )
    };

    for block in 0..blocks {
        let base_block = &base_bytes[block * block_bytes..(block + 1) * block_bytes];
        let mut delta = Array2::<f64>::zeros((rows, cols));
        let mut patched = false;
        for patch in patches
            .iter()
            .filter(|p| p.base_expert.is_none_or(|e| e == block))
        {
            let (a, b) = (
                factor(&patch.a, patch.adapter_expert)?,
                factor(&patch.b, patch.adapter_expert)?,
            );
            let (out_rows, rank) = b.dim();
            if a.ncols() != cols || a.nrows() != rank || patch.row_offset + out_rows > rows {
                return Err(MergeError::ShapeMismatch {
                    name: format!(
                        "expert {} of '{}' against '{name}'",
                        patch.adapter_expert, patch.b
                    ),
                    expected: vec![rows as i32, cols as i32],
                    actual: vec![(patch.row_offset + out_rows) as i32, a.ncols() as i32],
                });
            }
            let mut part = b.dot(&a);
            part *= patch.scaling;
            let mut window = delta.slice_mut(s![patch.row_offset..patch.row_offset + out_rows, ..]);
            window += &part;
            patched = true;
        }
        if patched {
            write_merged_rows(base_block, &delta, rows, cols, dtype, None, writer)?;
        } else {
            writer.write_all(base_block)?;
        }
    }
    Ok(())
}

/// Decode one row of `dtype` bytes into f64.
fn decode_row(bytes: &[u8], dtype: Dtype, out: &mut [f64]) {
    match dtype {
        Dtype::F16 => {
            for (c, o) in bytes.as_chunks::<2>().0.iter().zip(out) {
                *o = f16::from_le_bytes(*c).to_f64();
            }
        }
        Dtype::BF16 => {
            for (c, o) in bytes.as_chunks::<2>().0.iter().zip(out) {
                *o = bf16::from_le_bytes(*c).to_f64();
            }
        }
        _ => {
            for (c, o) in bytes.as_chunks::<4>().0.iter().zip(out) {
                *o = f64::from(f32::from_le_bytes(*c));
            }
        }
    }
}

/// Encode a row of f64 as `dtype` bytes. Values beyond the range of the
/// narrow dtypes saturate instead of becoming infinity.
fn encode_row(values: &[f64], dtype: Dtype, out: &mut [u8]) {
    match dtype {
        Dtype::F16 => {
            for (v, c) in values.iter().zip(out.as_chunks_mut::<2>().0) {
                let v = (*v as f32).clamp(f16::MIN.to_f32(), f16::MAX.to_f32());
                c.copy_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        Dtype::BF16 => {
            for (v, c) in values.iter().zip(out.as_chunks_mut::<2>().0) {
                let v = (*v as f32).clamp(bf16::MIN.to_f32(), bf16::MAX.to_f32());
                c.copy_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
        }
        _ => {
            for (v, c) in values.iter().zip(out.as_chunks_mut::<4>().0) {
                c.copy_from_slice(&(*v as f32).to_le_bytes());
            }
        }
    }
}

/// Fused row-by-row base + delta downcast pass.
///
/// Reads `base_bytes` and `delta` (which must have `rows` rows × `cols` cols)
/// row by row. Every addition is done in f64 before the final downcast to
/// `dtype`. With a DoRA `magnitude` (one entry per row) each combined row is
/// renormalised to that magnitude, `m_i * row / ||row||`, before the
/// downcast. Memory beyond the inputs is one row of f64 and one of bytes.
fn write_merged_rows(
    base_bytes: &[u8],
    delta: &Array2<f64>,
    rows: usize,
    cols: usize,
    dtype: Dtype,
    magnitude: Option<&[f64]>,
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

    let mut row = vec![0.0_f64; cols];
    let mut out_row: Vec<u8> = vec![0u8; row_bytes];

    for i in 0..rows {
        decode_row(
            &base_bytes[i * row_bytes..(i + 1) * row_bytes],
            dtype,
            &mut row,
        );
        for (v, d) in row.iter_mut().zip(&delta_flat[i * cols..(i + 1) * cols]) {
            *v += d;
        }
        if let Some(magnitude) = magnitude {
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                let scale = magnitude[i] / norm;
                for v in &mut row {
                    *v *= scale;
                }
            }
        }
        encode_row(&row, dtype, &mut out_row);
        writer.write_all(&out_row)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shard merge
// ---------------------------------------------------------------------------

/// Everything a shard merge needs besides the shard itself.
struct MergeContext<'a> {
    adapter: &'a MappedFile,
    targets: &'a HashMap<String, LoraTarget>,
    /// Per-expert adapters by the base tensor they fold into.
    experts: &'a HashMap<String, Vec<ExpertPatch>>,
    bias_map: &'a HashMap<String, String>,
    fan_in_fan_out: bool,
    low_memory: bool,
    tile_size: usize,
    /// The base model's `quantization` config, for MLX-quantized weights.
    quantization: Option<&'a serde_json::Value>,
}

/// Write a file all or nothing: build it in a `.partial` sibling, force it to
/// disk and rename it into place; the partial is removed on failure.
fn write_file_atomically(
    path: &Path,
    build: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
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
    let mut partial = path.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    let result = (|| -> Result<()> {
        let file = File::create(&partial).map_err(|e| {
            MergeError::ModelLoad(format!(
                "failed to create output file {}: {}",
                partial.display(),
                e
            ))
        })?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        build(&mut writer)?;
        writer.flush()?;
        writer
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        fs::rename(&partial, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

/// The A and B factors (f64) and DoRA magnitude of a target.
struct LoraFactors {
    a: Array2<f64>,
    b: Array2<f64>,
    magnitude: Option<Vec<f64>>,
}

fn load_factors(ctx: &MergeContext<'_>, target: &LoraTarget) -> Result<LoraFactors> {
    let a_meta = ctx.adapter.tensor_meta(&target.a)?;
    let b_meta = ctx.adapter.tensor_meta(&target.b)?;
    let lora_a = bytes_to_f64_2d(
        ctx.adapter.tensor_bytes(&target.a)?,
        a_meta.dtype,
        &a_meta.shape,
        &target.a,
    )?;
    let lora_b = bytes_to_f64_2d(
        ctx.adapter.tensor_bytes(&target.b)?,
        b_meta.dtype,
        &b_meta.shape,
        &target.b,
    )?;
    let magnitude = target
        .magnitude
        .as_ref()
        .map(|name| {
            let meta = ctx.adapter.tensor_meta(name)?;
            let n: usize = meta.shape.iter().product();
            bytes_to_f64_1d(ctx.adapter.tensor_bytes(name)?, meta.dtype, &[n], name)
        })
        .transpose()?;
    Ok(LoraFactors {
        a: lora_a,
        b: lora_b,
        magnitude,
    })
}

/// Bytes of the three tensors an MLX-quantized layer is stored as, produced
/// together when the layer is folded and written as each comes up in the
/// shard's order.
#[derive(Default)]
struct QuantizedLayer {
    weight: Option<Vec<u8>>,
    scales: Option<Vec<u8>>,
    biases: Option<Vec<u8>>,
}

/// `(group_size, bits)` for `module` from the base config's `quantization`
/// block: a per-module entry wins over the top-level setting.
fn quant_params(quantization: &serde_json::Value, module: &str) -> Option<(i32, i32)> {
    let read = |v: &serde_json::Value| {
        Some((
            i32::try_from(v.get("group_size")?.as_i64()?).ok()?,
            i32::try_from(v.get("bits")?.as_i64()?).ok()?,
        ))
    };
    quantization
        .get(module)
        .and_then(read)
        .or_else(|| read(quantization))
}

/// Encode f32 values as `dtype` bytes (F16, BF16 or F32).
fn encode_f32(values: &[f32], dtype: Dtype) -> Vec<u8> {
    let wide: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    let elem = if dtype == Dtype::F32 { 4 } else { 2 };
    let mut out = vec![0u8; values.len() * elem];
    encode_row(&wide, dtype, &mut out);
    out
}

/// Fold an adapter into an MLX affine-quantized weight: dequantize,
/// `W + s·B·A` (DoRA: renormalise the rows), quantize again with the layer's
/// group size and bit width. Requantizing is lossy by construction; the
/// result is within half a quantization step of the exact sum.
fn fold_quantized(
    shard: &MappedFile,
    weight_name: &str,
    ctx: &MergeContext<'_>,
) -> Result<QuantizedLayer> {
    let target = &ctx.targets[weight_name];
    let module = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let (group_size, bits) = ctx
        .quantization
        .and_then(|q| quant_params(q, module))
        .ok_or_else(|| {
            MergeError::ModelLoad(format!(
                "{weight_name} is quantized but the base config.json has no `quantization` \
                 block with group_size and bits"
            ))
        })?;
    if ctx.fan_in_fan_out {
        return Err(MergeError::InvalidConfig(
            "fan_in_fan_out adapters cannot be folded into quantized weights".to_string(),
        ));
    }

    let scales_name = format!("{module}.scales");
    let biases_name = format!("{module}.biases");
    for sibling in [&scales_name, &biases_name] {
        if shard.tensor_meta(sibling).is_err() {
            return Err(MergeError::ModelLoad(format!(
                "quantized layer {module}: {sibling} is not in the same shard as its weight"
            )));
        }
    }
    let weight_meta = shard.tensor_meta(weight_name)?;
    let scales_meta = shard.tensor_meta(&scales_name)?;
    let biases_meta = shard.tensor_meta(&biases_name)?;
    if weight_meta.shape.len() != 2 {
        return Err(MergeError::ShapeMismatch {
            name: format!("{weight_name} (quantized, expected 2D)"),
            expected: vec![2],
            actual: weight_meta.shape.iter().map(|&s| s as i32).collect(),
        });
    }
    let (out_features, packed) = (weight_meta.shape[0], weight_meta.shape[1]);
    let in_features = packed * 32 / bits as usize;

    let packed_words: Vec<u32> = shard
        .tensor_bytes(weight_name)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let weight = Array::from_u32_slice(&packed_words, &[out_features as i32, packed as i32]);
    let scales_values = decode_f32(
        shard.tensor_bytes(&scales_name)?,
        scales_meta.dtype,
        &scales_name,
    )?;
    let biases_values = decode_f32(
        shard.tensor_bytes(&biases_name)?,
        biases_meta.dtype,
        &biases_name,
    )?;
    let group_shape: Vec<i32> = scales_meta.shape.iter().map(|&s| s as i32).collect();
    let scales = Array::from_f32_slice(&scales_values, &group_shape);
    let biases = Array::from_f32_slice(&biases_values, &group_shape);
    let dense = weight.dequantize(&scales, &biases, group_size, bits);

    let LoraFactors {
        a: lora_a,
        b: lora_b,
        magnitude,
    } = load_factors(ctx, target)?;
    if lora_b.nrows() != out_features || lora_a.ncols() != in_features {
        return Err(MergeError::ShapeMismatch {
            name: format!(
                "{weight_name} (adapter [{}x{}] · [{}x{}])",
                lora_b.nrows(),
                lora_b.ncols(),
                lora_a.nrows(),
                lora_a.ncols()
            ),
            expected: vec![out_features as i32, in_features as i32],
            actual: vec![lora_b.nrows() as i32, lora_a.ncols() as i32],
        });
    }
    let delta = lora_b.dot(&lora_a) * target.scaling;
    let delta_f32: Vec<f32> = delta.iter().map(|&v| v as f32).collect();
    let mut combined = dense.add(&Array::from_f32_slice(
        &delta_f32,
        &[out_features as i32, in_features as i32],
    ));
    if let Some(magnitude) = magnitude {
        let magnitude: Vec<f32> = magnitude.iter().map(|&v| v as f32).collect();
        let norm = combined.square().sum_axis(1, true).sqrt();
        let factor = Array::from_f32_slice(&magnitude, &[out_features as i32, 1]).divide(&norm);
        combined = combined.multiply(&factor);
    }

    let (new_weight, new_scales, new_biases) = combined.quantize_weights(group_size, bits);
    let weight_bytes: Vec<u8> = new_weight
        .as_slice::<u32>()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let read_f32 = |mut a: Array| a.to_f32_vec(a.size());
    let new_scales = read_f32(new_scales);
    let new_biases = read_f32(new_biases);
    pmetal_runtime::check_unobserved_error()?;
    let (Some(new_scales), Some(new_biases)) = (new_scales, new_biases) else {
        return Err(MergeError::ModelLoad(format!(
            "{weight_name}: reading the requantized layer failed"
        )));
    };
    Ok(QuantizedLayer {
        weight: Some(weight_bytes),
        scales: Some(encode_f32(&new_scales, scales_meta.dtype)),
        biases: Some(encode_f32(&new_biases, biases_meta.dtype)),
    })
}

/// Merge one base shard (or single-file model) with the adapter and write the
/// result atomically to `output_path`.
///
/// Dense adapted weights are folded row by row in f64, quantized ones through
/// [`fold_quantized`], adapter biases are added, and every other tensor is
/// copied byte for byte.
fn merge_shard(
    shard: &MappedFile,
    ctx: &MergeContext<'_>,
    output_path: &Path,
    stats: &mut LoraMergeStats,
) -> Result<()> {
    // Quantized layers: the `.weight` is a u32 tensor whose `.scales` and
    // `.biases` change along with it.
    let mut quantized_sibling: HashMap<String, String> = HashMap::new();
    for (name, meta) in &shard.tensors {
        if meta.dtype == Dtype::U32 && ctx.targets.contains_key(name) {
            let module = name.strip_suffix(".weight").unwrap_or(name);
            quantized_sibling.insert(format!("{module}.scales"), name.clone());
            quantized_sibling.insert(format!("{module}.biases"), name.clone());
        }
    }

    write_file_atomically(output_path, |writer| {
        let output_tensors: Vec<(String, TensorMeta)> = shard
            .tensors
            .iter()
            .map(|(name, meta)| (name.clone(), meta.clone()))
            .collect();
        let header_bytes = build_header(&output_tensors, shard.metadata.as_ref())?;
        writer.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(&header_bytes)?;
        stats.bytes_written += 8 + header_bytes.len() as u64;

        shard.advise_sequential();
        let mut pending: HashMap<String, QuantizedLayer> = HashMap::new();

        for (name, meta) in &shard.tensors {
            let base_data = shard.slice(meta);
            let is_quant_weight = meta.dtype == Dtype::U32 && ctx.targets.contains_key(name);

            if is_quant_weight || quantized_sibling.contains_key(name) {
                let weight_name = if is_quant_weight {
                    name
                } else {
                    &quantized_sibling[name]
                };
                if !pending.contains_key(weight_name) {
                    debug!("merging LoRA into quantized '{weight_name}'");
                    pending.insert(
                        weight_name.clone(),
                        fold_quantized(shard, weight_name, ctx)?,
                    );
                    stats.tensors_merged += 1;
                }
                let layer = pending.get_mut(weight_name).expect("inserted above");
                let piece = if name == weight_name {
                    layer.weight.take()
                } else if name.ends_with(".scales") {
                    layer.scales.take()
                } else {
                    layer.biases.take()
                };
                let piece = piece
                    .filter(|p| p.len() == base_data.len())
                    .ok_or_else(|| {
                        MergeError::ModelLoad(format!(
                            "requantized '{name}' does not have the size of the base tensor"
                        ))
                    })?;
                writer.write_all(&piece)?;
                if layer.weight.is_none() && layer.scales.is_none() && layer.biases.is_none() {
                    pending.remove(weight_name);
                }
            } else if let Some(patches) = ctx.experts.get(name) {
                debug!("merging {} expert adapter(s) for '{name}'", patches.len());
                merge_and_write_experts(
                    ctx,
                    name,
                    base_data,
                    &meta.shape,
                    meta.dtype,
                    patches,
                    writer,
                )?;
                stats.tensors_merged += 1;
            } else if let Some(target) = ctx.targets.get(name) {
                debug!("merging LoRA for '{name}'");
                let LoraFactors {
                    a: lora_a,
                    b: lora_b,
                    magnitude,
                } = load_factors(ctx, target)?;
                if magnitude.is_some() && ctx.fan_in_fan_out {
                    return Err(MergeError::InvalidConfig(
                        "DoRA adapters with fan_in_fan_out are not supported".to_string(),
                    ));
                }
                merge_and_write_tensor(
                    base_data,
                    &lora_a,
                    &lora_b,
                    target.scaling,
                    ctx.fan_in_fan_out,
                    &meta.shape,
                    meta.dtype,
                    ctx.low_memory,
                    ctx.tile_size,
                    magnitude.as_deref(),
                    writer,
                )?;
                stats.tensors_merged += 1;
            } else if let Some(adapter_bias_name) = ctx.bias_map.get(name) {
                debug!("merging bias for '{name}' from adapter tensor '{adapter_bias_name}'");
                let adapter_meta = ctx.adapter.tensor_meta(adapter_bias_name)?;
                merge_and_write_bias(
                    base_data,
                    &meta.shape,
                    meta.dtype,
                    ctx.adapter.tensor_bytes(adapter_bias_name)?,
                    &adapter_meta.shape,
                    adapter_meta.dtype,
                    name,
                    adapter_bias_name,
                    writer,
                )?;
                stats.biases_merged += 1;
            } else {
                // Pass-through: write bytes unchanged.
                writer.write_all(base_data)?;
                stats.tensors_copied += 1;
            }
            stats.bytes_written += base_data.len() as u64;
        }
        if !pending.is_empty() {
            return Err(MergeError::ModelLoad(format!(
                "quantized layer(s) {:?} were folded but their scales or biases are not in this \
                 shard",
                pending.keys().collect::<Vec<_>>()
            )));
        }
        shard.advise_dontneed();
        Ok(())
    })
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
    ctx: &MergeContext<'_>,
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
        let shard = MappedFile::open(&base_dir.join(&shard_filename))?;
        merge_shard(&shard, ctx, &output_dir.join(&shard_filename), stats)?;
    }

    // The tensors keep their shards, so the index carries over unchanged.
    let output_index = serde_json::json!({
        "metadata": index.metadata,
        "weight_map": index.weight_map,
    });
    let json = serde_json::to_string_pretty(&output_index).map_err(MergeError::Serde)?;
    write_atomic(
        &output_dir.join("model.safetensors.index.json"),
        json.as_bytes(),
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// The adapter's weight file and `adapter_config.json`: PEFT's
/// `adapter_model.safetensors` or `pmetal train`'s `lora_weights.safetensors`
/// in a directory, or a weight file directly.
fn locate_adapter(adapter_path: &Path) -> Result<(PathBuf, PathBuf)> {
    let (dir, file) = if adapter_path.is_file() {
        (
            adapter_path
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
            Some(adapter_path.to_path_buf()),
        )
    } else {
        (adapter_path.to_path_buf(), None)
    };
    let weights = match file {
        Some(file) => file,
        None => ["adapter_model.safetensors", "lora_weights.safetensors"]
            .iter()
            .map(|n| dir.join(n))
            .find(|p| p.is_file())
            .ok_or_else(|| {
                MergeError::ModelLoad(format!(
                    "neither adapter_model.safetensors nor lora_weights.safetensors found in {}",
                    dir.display()
                ))
            })?,
    };
    Ok((weights, dir.join("adapter_config.json")))
}

/// Perform a streaming f64-accurate LoRA merge.
///
/// Reads `adapter_config.json` next to the adapter weights, resolves the base
/// model (single file or sharded directory), maps adapter tensors to base
/// tensors, and writes the merged model to `config.output_path`.
///
/// Adapters in PEFT's layout (`adapter_model.safetensors`, keys
/// `base_model.model.<module>.lora_{A,B}.weight`) and in `pmetal train`'s
/// (`lora_weights.safetensors`, keys `<module>.lora_{a,b}`) are both read.
/// Each adapted weight becomes `W + s·B·A` with `s = alpha / r` (`alpha /
/// sqrt(r)` under rsLoRA, `alpha_pattern` overriding `alpha` per module), and
/// DoRA adapters renormalise each output row to their magnitude vector.
/// MLX-quantized weights are dequantized, folded and quantized again. A
/// mixture-of-experts adapter (`lora_expert_A` / `lora_expert_B`, one factor
/// pair per expert) is folded expert by expert into the experts' weights,
/// whether the base stores them one tensor per expert, fused into
/// `gate_up_proj` and `down_proj`, or stacked as `switch_mlp`; quantized
/// experts are refused.
///
/// Non-adapted tensors are copied byte-for-byte (zero-allocation path), and
/// every output file is written atomically.
///
/// # Errors
///
/// Returns [`MergeError`] if any of the following occur:
///
/// - `adapter_config.json` is missing or malformed
/// - the adapter weights are missing
/// - an adapted module has no weight in the base model (an expert projection
///   in none of the three layouts, or in a run with an expert missing), an
///   adapter tensor is of a kind that cannot be folded, or a factor pair is
///   incomplete
/// - Base model path does not exist or contains no safetensors files
/// - A LoRA tensor has an unsupported dtype or unexpected shape
/// - I/O or serialization errors during output writing
pub fn streaming_lora_merge(config: &AccurateMergeConfig) -> Result<LoraMergeStats> {
    let start = std::time::Instant::now();

    // ---- Adapter ----------------------------------------------------------
    let config_dir = if config.adapter_path.is_file() {
        config.adapter_path.parent().unwrap_or(Path::new("."))
    } else {
        &config.adapter_path
    };
    let adapter_cfg = AdapterConfigJson::from_path(&config_dir.join("adapter_config.json"))?;
    let (adapter_weights_path, _) = locate_adapter(&config.adapter_path)?;
    let adapter = MappedFile::open(&adapter_weights_path)?;
    info!(
        "LoRA merge: rank={}, alpha={}, rslora={}, dora={}, fan_in_fan_out={}, {} adapter \
         tensors from {}",
        adapter_cfg.r,
        adapter_cfg.lora_alpha,
        adapter_cfg.use_rslora,
        adapter_cfg.use_dora,
        adapter_cfg.fan_in_fan_out,
        adapter.tensors.len(),
        adapter_weights_path.display()
    );

    let mut stats = LoraMergeStats::default();

    // ---- Base model: single file or sharded ----------------------------------
    let base_path = &config.base_model_path;
    let base_dir = if base_path.is_file() {
        base_path.parent().unwrap_or(Path::new(".")).to_path_buf()
    } else {
        base_path.clone()
    };
    let quantization_config: Option<serde_json::Value> =
        fs::read_to_string(base_dir.join("config.json"))
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|cfg| {
                cfg.get("quantization")
                    .or_else(|| cfg.get("quantization_config"))
                    .cloned()
            });

    enum BaseLayout {
        Single(PathBuf),
        Sharded(ShardIndex),
    }
    let layout = if base_path.is_file() {
        BaseLayout::Single(base_path.clone())
    } else if base_path.is_dir() {
        let single = base_path.join("model.safetensors");
        let index_path = base_path.join("model.safetensors.index.json");
        if single.is_file() {
            BaseLayout::Single(single)
        } else if index_path.is_file() {
            BaseLayout::Sharded(ShardIndex::from_path(&index_path)?)
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
    };

    // Base tensor names (and, for a single file, the mapped file itself).
    let single_file = match &layout {
        BaseLayout::Single(path) => Some(MappedFile::open(path)?),
        BaseLayout::Sharded(_) => None,
    };
    let base_name_list: Vec<&str> = match (&single_file, &layout) {
        (Some(file), _) => file.tensors.iter().map(|(n, _)| n.as_str()).collect(),
        (None, BaseLayout::Sharded(index)) => index.weight_map.keys().map(String::as_str).collect(),
        (None, BaseLayout::Single(_)) => unreachable!("single layouts are mapped above"),
    };
    let base_names: HashSet<&str> = base_name_list.iter().copied().collect();
    let adapter_names: Vec<&str> = adapter.tensors.iter().map(|(n, _)| n.as_str()).collect();

    let targets = discover_targets(&base_names, &adapter, &adapter_cfg, config.scaling)?;
    let bias_map = build_bias_map(&base_name_list, &adapter_names);
    info!(
        "Identified {} LoRA-adapted tensors ({} of experts), {} adapter biases",
        targets.len(),
        targets.experts.len(),
        bias_map.len()
    );

    let ctx = MergeContext {
        adapter: &adapter,
        targets: &targets.dense,
        experts: &targets.experts,
        bias_map: &bias_map,
        fan_in_fan_out: adapter_cfg.fan_in_fan_out,
        low_memory: config.low_memory,
        tile_size: config.tile_size,
        quantization: quantization_config.as_ref(),
    };

    match (&layout, &single_file) {
        (BaseLayout::Single(_), Some(file)) => {
            // A directory base gets its model.safetensors and sidecars; a file
            // base is written to the output path itself.
            let output = if base_path.is_file() {
                config.output_path.clone()
            } else {
                config.output_path.join("model.safetensors")
            };
            merge_shard(file, &ctx, &output, &mut stats)?;
            if base_path.is_dir() {
                copy_side_files(base_path, &config.output_path)?;
            }
        }
        (BaseLayout::Sharded(index), _) => {
            merge_sharded(base_path, index, &ctx, &config.output_path, &mut stats)?;
            copy_side_files(base_path, &config.output_path)?;
        }
        (BaseLayout::Single(_), None) => unreachable!("single layouts are mapped above"),
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

// `copy_side_files` lives in `loader.rs` (factored Phase 2.2 — now shared with
// the full-model merge path). It is imported via the `use` block at the top
// of this file; `lora_merge.rs` is a pure consumer.

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
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
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
            None,
            &mut out,
        )
        .unwrap();

        let result: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
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
            None,
            &mut out,
        )
        .unwrap();

        let result: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
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
            None,
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
            None,
            &mut out_tiled,
        )
        .unwrap();

        let full: Vec<f32> = out_full
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let tiled: Vec<f32> = out_tiled
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
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
            None,
            &mut out,
        )
        .unwrap();

        let result: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert!((result[0] - 4.0).abs() < 1e-6, "result[0]={}", result[0]);
        assert!((result[1] - 4.0).abs() < 1e-6, "result[1]={}", result[1]);
        assert!((result[2] - 6.0).abs() < 1e-6, "result[2]={}", result[2]);
        assert!((result[3] - 9.0).abs() < 1e-6, "result[3]={}", result[3]);
    }

    // ---- end-to-end --------------------------------------------------------

    /// An adapter on a 2-expert projection with 2x2 experts, rank 1: expert 0
    /// adds `[[1,1],[1,1]]` (B = [1,1], A = [1,1]) and expert 1 adds
    /// `[[2,0],[2,0]]` (B = [1,1], A = [2,0]), both at scaling alpha / r = 1.
    fn write_expert_adapter(dir: &Path, module: &str) {
        fs::create_dir_all(dir).unwrap();
        write_adapter_config(&dir.join("adapter_config.json"), 1, 1.0);
        write_safetensors_f16(
            &dir.join("adapter_model.safetensors"),
            vec![
                (
                    &format!("base_model.model.{module}.lora_expert_A"),
                    vec![1.0, 1.0, 2.0, 0.0],
                    vec![2, 1, 2],
                ),
                (
                    &format!("base_model.model.{module}.lora_expert_B"),
                    vec![1.0, 1.0, 1.0, 1.0],
                    vec![2, 2, 1],
                ),
            ],
        );
    }

    fn merged(dir: &Path, base: Vec<(&str, Vec<f32>, Vec<usize>)>) -> Result<PathBuf> {
        let base_path = dir.join("model.safetensors");
        write_safetensors_f16(&base_path, base);
        let out = dir.join("merged.safetensors");
        streaming_lora_merge(&AccurateMergeConfig::new(
            &base_path,
            dir.join("adapter"),
            &out,
        ))
        .map(|_| out)
    }

    const EXPERT_ZERO_PLUS: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    const EXPERT_ONE_PLUS: [f32; 4] = [2.0, 0.0, 2.0, 0.0];

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 0.02, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn expert_adapters_fold_into_a_stacked_tensor_expert_by_expert() {
        let dir = tempfile::tempdir().unwrap();
        let module = "model.layers.0.mlp.switch_mlp.gate_proj";
        write_expert_adapter(&dir.path().join("adapter"), module);
        // The base experts are 10 and 20 everywhere, so a fold into the wrong
        // expert or row shows.
        let mut base = vec![10.0; 4];
        base.extend([20.0; 4]);
        let out = merged(
            dir.path(),
            vec![(&format!("{module}.weight"), base, vec![2, 2, 2])],
        )
        .unwrap();
        let got = read_f16_tensor(&out, &format!("{module}.weight"));
        let want: Vec<f32> = EXPERT_ZERO_PLUS
            .iter()
            .map(|d| 10.0 + d)
            .chain(EXPERT_ONE_PLUS.iter().map(|d| 20.0 + d))
            .collect();
        close(&got, &want);
    }

    #[test]
    fn expert_adapters_fold_into_one_tensor_per_expert() {
        let dir = tempfile::tempdir().unwrap();
        write_expert_adapter(
            &dir.path().join("adapter"),
            "model.layers.0.mlp.switch_mlp.up_proj",
        );
        let out = merged(
            dir.path(),
            vec![
                (
                    "model.layers.0.mlp.experts.0.up_proj.weight",
                    vec![10.0; 4],
                    vec![2, 2],
                ),
                (
                    "model.layers.0.mlp.experts.1.up_proj.weight",
                    vec![20.0; 4],
                    vec![2, 2],
                ),
                (
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    vec![5.0; 4],
                    vec![2, 2],
                ),
            ],
        )
        .unwrap();
        let tensor = |n: &str| read_f16_tensor(&out, n);
        close(
            &tensor("model.layers.0.mlp.experts.0.up_proj.weight"),
            &EXPERT_ZERO_PLUS.map(|d| 10.0 + d),
        );
        close(
            &tensor("model.layers.0.mlp.experts.1.up_proj.weight"),
            &EXPERT_ONE_PLUS.map(|d| 20.0 + d),
        );
        // Another projection of the same expert is not the adapter's.
        close(
            &tensor("model.layers.0.mlp.experts.0.down_proj.weight"),
            &[5.0; 4],
        );
    }

    /// The gate projection folds into the first half of a fused
    /// `gate_up_proj`'s rows and the up projection into the second.
    #[test]
    fn expert_adapters_fold_into_the_right_half_of_a_fused_gate_up() {
        let dir = tempfile::tempdir().unwrap();
        // The up projection is adapted; its rows are the last 2 of 4.
        write_expert_adapter(
            &dir.path().join("adapter"),
            "model.layers.0.mlp.switch_mlp.up_proj",
        );
        let out = merged(
            dir.path(),
            vec![(
                "model.layers.0.mlp.experts.gate_up_proj",
                vec![10.0; 16],
                vec![2, 4, 2],
            )],
        )
        .unwrap();
        let got = read_f16_tensor(&out, "model.layers.0.mlp.experts.gate_up_proj");
        let mut want = Vec::new();
        for delta in [EXPERT_ZERO_PLUS, EXPERT_ONE_PLUS] {
            want.extend([10.0; 4]); // gate rows
            want.extend(delta.map(|d| 10.0 + d)); // up rows
        }
        close(&got, &want);
    }

    #[test]
    fn an_expert_adapter_with_no_experts_in_the_base_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let module = "model.layers.0.mlp.switch_mlp.gate_proj";
        write_expert_adapter(&dir.path().join("adapter"), module);
        let err = merged(
            dir.path(),
            vec![("model.embed_tokens.weight", vec![1.0; 4], vec![2, 2])],
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains(module) && err.contains("no weight in the base"),
            "{err}"
        );

        // A run of experts with one missing is named, not folded partly.
        let err = merged(
            dir.path(),
            vec![(
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                vec![1.0; 4],
                vec![2, 2],
            )],
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("experts.1.gate_proj.weight") && err.contains("missing"),
            "{err}"
        );
    }

    #[test]
    fn expert_factors_with_the_wrong_experts_or_rank_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let adapter = dir.path().join("adapter");
        fs::create_dir_all(&adapter).unwrap();
        write_adapter_config(&adapter.join("adapter_config.json"), 1, 1.0);
        let module = "model.layers.0.mlp.switch_mlp.gate_proj";
        write_safetensors_f16(
            &adapter.join("adapter_model.safetensors"),
            vec![
                (
                    &format!("base_model.model.{module}.lora_expert_A"),
                    vec![1.0; 4],
                    vec![2, 1, 2],
                ),
                (
                    &format!("base_model.model.{module}.lora_expert_B"),
                    vec![1.0; 6],
                    vec![3, 2, 1],
                ),
            ],
        );
        let err = merged(
            dir.path(),
            vec![(&format!("{module}.weight"), vec![1.0; 8], vec![2, 2, 2])],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expert factors"), "{err}");
    }

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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
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
