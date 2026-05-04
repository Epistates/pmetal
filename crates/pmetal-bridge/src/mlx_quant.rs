//! MLX-format weight quantization with per-tensor sensitivity analysis.
//!
//! Produces MLX-compatible safetensors (packed weights + scales + biases) with
//! per-tensor bit allocation driven by reconstruction quality rather than
//! static per-class rules.
//!
//! # Pipeline
//!
//! 1. **Evaluate quality** — for each candidate bit-width, quantize → dequantize
//!    via the MLX bridge, then measure NRMSE + (1 − cosine_similarity) on the
//!    reconstructed tensor. This is the per-tensor sensitivity score.
//! 2. **Allocate bits** — given a target bits-per-weight (BPW), use a greedy
//!    quality-aware scheduler: start all non-critical tensors at minimum bits,
//!    then upgrade the tensor with the worst reconstruction score first until
//!    the BPW budget is consumed.
//! 3. **Quantize and save** — write packed MLX safetensors shards plus a
//!    `config.json` with embedded quantization metadata.
//!
//! # Critical tensors
//!
//! Norms, embeddings, MoE router gates, and GDN-specific parameters
//! (`A_log`, `dt_bias`, `conv1d`) are always kept at 8-bit or full precision
//! because errors in these tensors have outsized downstream effects.

use std::collections::HashMap;
use std::path::Path;

use crate::InlineArray;

/// Per-tensor sensitivity scores: `(name, num_elements, [(bits, score)])`.
/// The inner list is sorted ascending by bits as returned by
/// [`evaluate_tensor_quality`]. An empty inner list signals passthrough.
type TensorScores = Vec<(String, usize, Vec<(i32, f64)>)>;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default MLX quantization group size.
pub const DEFAULT_GROUP_SIZE: i32 = 64;

/// Candidate bit-widths for MLX quantization (2 is too lossy for production).
pub const DEFAULT_BITS_CANDIDATES: &[i32] = &[3, 4, 5, 6, 8];

/// Shard size cap in bytes (~5 GiB, matching HF/MLX convention).
const SHARD_SIZE_BYTES: u64 = 5 * 1024 * 1024 * 1024;

/// Bits value that signals "keep in original precision — do not quantize".
///
/// Critical tensors and any tensor with fewer than 2 dimensions receive this
/// assignment and are saved verbatim.
pub const BITS_PASSTHROUGH: i32 = 0;

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-tensor quantization assignment produced by [`allocate_bits_for_bpw`].
#[derive(Debug, Clone)]
pub struct TensorQuant {
    /// Safetensors key (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Bits assigned (3, 4, 5, 6, or 8).
    ///
    /// [`BITS_PASSTHROUGH`] (0) means the tensor is saved verbatim without
    /// quantization (used for critical tensors and non-matrix parameters).
    pub bits: i32,
    /// Group size that was used for quality evaluation and will be used for
    /// quantization.
    pub group_size: i32,
    /// Number of scalar elements in the weight tensor.
    pub param_count: usize,
    /// Quality score at the assigned bit-width. Lower = better reconstruction.
    ///
    /// `0.0` for passthrough tensors.
    pub quality_score: f64,
}

// ── Critical tensor classification ───────────────────────────────────────────

/// Returns `true` if this tensor must be kept at high precision.
///
/// Matched patterns:
/// - `model.norm.weight` — final layer norm
/// - `model.embed_tokens.weight` — token embeddings
/// - `lm_head.weight` — LM head
/// - `*layernorm*`, `*ln_*`, `*.norm.*`, `*.norm` — any normalization layer
/// - `*.gate.weight` — MoE router gate (routing precision is critical)
/// - `*a_log*`, `*A_log*`, `*dt_bias*`, `*conv1d*` — GDN / Mamba parameters
pub fn is_critical_tensor(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();

    // Exact high-value keys.
    if lower == "model.norm.weight"
        || lower == "model.embed_tokens.weight"
        || lower == "lm_head.weight"
    {
        return true;
    }

    // Any normalization layer in any variant.
    // Covers: layernorm, ln_1, .norm., q_norm.weight, post_attention_layernorm, etc.
    if lower.contains("layernorm") || lower.contains("ln_") || lower.contains(".norm.") {
        return true;
    }
    // Plain ".norm" at the end of a key or followed by ".weight"/".bias".
    if lower.ends_with(".norm") || lower.ends_with(".norm.weight") || lower.ends_with(".norm.bias")
    {
        return true;
    }
    // Any path segment that ends in "norm" followed by ".weight" or ".bias",
    // e.g. "q_norm.weight", "k_norm.weight", "post_norm.weight".
    if lower.ends_with("norm.weight") || lower.ends_with("norm.bias") {
        return true;
    }

    // MoE router gates — small tensors, but routing errors compound across
    // all layers.
    if lower.ends_with(".gate.weight") {
        return true;
    }

    // Per-expert tensors are stacked for native MoE inference. Keeping all
    // experts at the same high bit-width avoids mixed packed shapes within one
    // stacked projection while preserving affine-quantized execution.
    if lower.contains(".experts.") {
        return true;
    }

    // GDN / Mamba-family parameters — quantization degrades recurrent stability.
    if lower.contains("a_log") || lower.contains("dt_bias") || lower.contains("conv1d") {
        return true;
    }

    false
}

// ── Quality evaluation ────────────────────────────────────────────────────────

/// Evaluate reconstruction quality for a single weight tensor at each candidate
/// bit-width using the MLX quantize/dequantize bridge.
///
/// Returns `(bits, quality_score)` pairs sorted ascending by `bits`.  A
/// **lower** quality score means better reconstruction fidelity.
///
/// Score = NRMSE(original, reconstructed) + (1 − cosine_similarity).
///
/// Both the original and reconstructed arrays are `.eval()`-ed before any
/// arithmetic to honour MLX lazy evaluation semantics.
///
/// # Requirements
///
/// `weight` must be a ≥ 2-D matrix. Pass 1-D or scalar tensors directly to the
/// allocator with an empty `scores` list and they will be marked passthrough.
pub fn evaluate_tensor_quality(
    weight: &InlineArray,
    group_size: i32,
    bits_candidates: &[i32],
) -> Vec<(i32, f64)> {
    assert!(
        !bits_candidates.is_empty(),
        "bits_candidates must not be empty"
    );
    assert!(
        weight.ndim() >= 2,
        "evaluate_tensor_quality requires a matrix (ndim >= 2)"
    );

    // Force the original tensor to be materialised in f32 for metric math.
    weight.eval();
    let orig_f32 = weight.as_dtype(crate::compat::Dtype::Float32.as_i32());
    // Flatten to 1-D so NRMSE and cosine work over the full parameter vector.
    let flat_orig = orig_f32.flatten(0, -1);
    flat_orig.eval();

    let mut results: Vec<(i32, f64)> = Vec::with_capacity(bits_candidates.len());

    for &bits in bits_candidates {
        let (packed, scales, biases) = weight.quantize_weights(group_size, bits);
        let recon = packed.dequantize(&scales, &biases, group_size, bits);
        let recon_f32 = recon.as_dtype(crate::compat::Dtype::Float32.as_i32());
        let flat_recon = recon_f32.flatten(0, -1);

        flat_orig.eval();
        flat_recon.eval();

        let score = quality_score(&flat_orig, &flat_recon);
        results.push((bits, score));
    }

    // Guarantee ascending-bits order for the allocator.
    results.sort_by_key(|(b, _)| *b);
    results
}

/// Compute NRMSE + (1 − cosine_similarity) for two 1-D f32 arrays.
///
/// Both arrays must already be evaluated.
fn quality_score(orig: &InlineArray, recon: &InlineArray) -> f64 {
    let diff = orig.subtract(recon);

    // NRMSE: sqrt(mean(diff²)) / sqrt(mean(orig²))
    let nrmse_num = diff.square().mean_all().sqrt();
    let nrmse_den = orig.square().mean_all().sqrt();

    // Cosine similarity: (a · b) / (‖a‖ × ‖b‖)
    let dot = orig.multiply(recon).sum_all();
    let norm_orig = orig.norm_l2(0, false);
    let norm_recon = recon.norm_l2(0, false);
    let norms_product = norm_orig.multiply(&norm_recon);

    // Batch-eval all scalars in a single GPU submission.
    nrmse_num.eval();
    nrmse_den.eval();
    dot.eval();
    norms_product.eval();

    let nn = nrmse_num.item_f32() as f64;
    let nd = nrmse_den.item_f32() as f64;
    let d = dot.item_f32() as f64;
    let np = norms_product.item_f32() as f64;

    let nrmse = if nd > 1e-12 { nn / nd } else { 0.0 };
    let cosine = if np > 1e-12 {
        (d / np).clamp(-1.0, 1.0)
    } else {
        1.0 // zero-norm tensor: treat as perfectly reconstructed
    };

    nrmse + (1.0 - cosine)
}

// ── Bit allocation ────────────────────────────────────────────────────────────

/// Assign per-tensor bit-widths to meet a target average bits-per-weight.
///
/// # Algorithm
///
/// 1. Critical tensors (matched by [`is_critical_tensor`] or `critical_tensors`
///    fragments) are pinned to the highest candidate bit-width (usually 8-bit).
/// 2. All other tensors start at the lowest candidate bit-width.
/// 3. Current BPW = Σ(param_count × bits) / Σ(param_count).
/// 4. While BPW < `target_bpw`: upgrade the tensor with the worst quality
///    score at its current bit-width to the next higher candidate.  Ties are
///    broken by parameter count (larger tensor upgraded first).
/// 5. Stop when the budget is met or all tensors are at maximum bits.
///
/// # Parameters
///
/// - `tensor_scores`: `(name, param_count, [(bits, score)])` tuples.  The inner
///   bits/score list must be sorted ascending by bits (as returned by
///   [`evaluate_tensor_quality`]).  An empty inner list signals passthrough
///   (no quantization, e.g. 1-D vectors).
/// - `target_bpw`: desired average bits per weight, e.g. `4.0`.
/// - `critical_tensors`: additional name fragments to force to the highest
///   available bit-width, supplementing [`is_critical_tensor`].
pub fn allocate_bits_for_bpw(
    tensor_scores: &TensorScores,
    target_bpw: f32,
    critical_tensors: &[&str],
) -> Vec<TensorQuant> {
    if tensor_scores.is_empty() {
        return Vec::new();
    }

    let n = tensor_scores.len();
    // Each element is the index into the tensor's scores list for its current
    // bit-width assignment.
    let mut bits_step: Vec<usize> = vec![0; n];

    // --- Initial pass: pin critical tensors to highest available bits. ---
    for (i, (name, _, scores)) in tensor_scores.iter().enumerate() {
        if scores.is_empty() {
            // Passthrough tensor (non-matrix or explicit skip).
            continue;
        }
        let forced =
            is_critical_tensor(name) || critical_tensors.iter().any(|pat| name.contains(pat));
        if forced {
            // Find the step for bits >= 8, or clamp to the last step.
            let high_step = scores
                .iter()
                .position(|(b, _)| *b >= 8)
                .unwrap_or(scores.len() - 1);
            bits_step[i] = high_step;
        }
    }

    // --- Greedy upgrade loop ---
    loop {
        let bpw = effective_bpw(tensor_scores, &bits_step);
        if bpw >= target_bpw as f64 {
            break;
        }

        // Find the non-critical, upgradeable tensor with the worst quality
        // score at its current assignment.
        let candidate = tensor_scores
            .iter()
            .enumerate()
            .filter(|(i, (name, _, scores))| {
                // Skip passthrough tensors.
                if scores.is_empty() {
                    return false;
                }
                // Skip critical tensors (already pinned to high bits).
                let forced = is_critical_tensor(name)
                    || critical_tensors.iter().any(|pat| name.contains(pat));
                if forced {
                    return false;
                }
                // Skip tensors already at the highest candidate.
                bits_step[*i] + 1 < scores.len()
            })
            .map(|(i, (_, param_count, scores))| {
                let current_score = scores[bits_step[i]].1;
                (i, current_score, *param_count)
            })
            .max_by(|(_, score_a, count_a), (_, score_b, count_b)| {
                // Primary sort: highest (worst) quality score first.
                // Tie-break: largest tensor first.
                score_a
                    .partial_cmp(score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| count_a.cmp(count_b))
            });

        match candidate {
            Some((idx, _, _)) => bits_step[idx] += 1,
            None => break, // all tensors are at max bits
        }
    }

    // --- Build output ---
    tensor_scores
        .iter()
        .enumerate()
        .map(|(i, (name, param_count, scores))| {
            let (bits, quality_score) = if scores.is_empty() {
                (BITS_PASSTHROUGH, 0.0)
            } else {
                scores[bits_step[i]]
            };
            TensorQuant {
                name: name.clone(),
                bits,
                group_size: DEFAULT_GROUP_SIZE,
                param_count: *param_count,
                quality_score,
            }
        })
        .collect()
}

/// Compute the effective average bits-per-weight across all tensors.
///
/// Passthrough tensors (empty scores / `bits == 0`) count as 16 bits (bf16).
fn effective_bpw(tensor_scores: &TensorScores, bits_step: &[usize]) -> f64 {
    let mut total_weighted: f64 = 0.0;
    let mut total_params: f64 = 0.0;
    for (i, (_, param_count, scores)) in tensor_scores.iter().enumerate() {
        let pc = *param_count as f64;
        let bits = if scores.is_empty() {
            16.0
        } else {
            scores[bits_step[i]].0 as f64
        };
        total_weighted += pc * bits;
        total_params += pc;
    }
    if total_params > 0.0 {
        total_weighted / total_params
    } else {
        0.0
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// A single entry scheduled for a safetensors shard.
struct ShardRecord<'a> {
    /// Index into the originating `assignments` slice.
    idx: usize,
    /// Borrowed reference to the source weight array.
    array: &'a InlineArray,
    /// Conservative byte-size estimate used for shard budget tracking.
    byte_estimate: u64,
}

// ── Quantize and save ─────────────────────────────────────────────────────────

/// Quantize a model and write it as MLX-compatible safetensors shards.
///
/// # Output layout
///
/// Quantized weights produce three safetensors keys:
/// - `{name}` — packed integer weights
/// - `{name}.scales` — per-group f16 scales
/// - `{name}.biases` — per-group f16 zero-points
///
/// This layout matches `mlx.core.quantize` output and is loadable by
/// `mlx.core.dequantize` and `mlx.nn.QuantizedLinear`.
///
/// Passthrough tensors (`bits == BITS_PASSTHROUGH`) are saved verbatim.
///
/// Shards are named `model-00001-of-NNNNN.safetensors` and are capped at
/// 5 GiB each.  A `model.safetensors.index.json` weight-map is written
/// alongside the shards.
///
/// # Errors
///
/// Returns `Err(String)` if any filesystem operation fails or if an assignment
/// references a weight name absent from `weights`.
pub fn quantize_and_save_mlx(
    weights: &HashMap<String, InlineArray>,
    assignments: &[TensorQuant],
    output_dir: &Path,
    _group_size: i32,
) -> Result<(), String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("failed to create output dir '{}: {e}", output_dir.display()))?;

    // --- Estimate per-assignment byte sizes for shard partitioning ---
    let mut records: Vec<ShardRecord<'_>> = Vec::with_capacity(assignments.len());

    for (idx, assignment) in assignments.iter().enumerate() {
        let array = weights.get(&assignment.name).ok_or_else(|| {
            format!(
                "weight '{}' in assignment not found in weights map",
                assignment.name
            )
        })?;

        let byte_estimate = if assignment.bits == BITS_PASSTHROUGH {
            array.size() as u64 * dtype_element_bytes(array.dtype_raw())
        } else {
            let n = array.size() as u64;
            let packed_bytes = (n * assignment.bits as u64).div_ceil(8);
            let n_groups = n.div_ceil(assignment.group_size as u64);
            // scales + biases: f16 each, one per group.
            let sb_bytes = n_groups * 2 * 2;
            packed_bytes + sb_bytes
        };

        records.push(ShardRecord {
            idx,
            array,
            byte_estimate,
        });
    }

    // --- Partition into shards ---
    let shards = partition_into_shards(&records, SHARD_SIZE_BYTES);
    let n_shards = shards.len();

    // weight_map: tensor key → shard filename
    let mut weight_map: HashMap<String, String> = HashMap::new();

    for (shard_num, shard_record_indices) in shards.iter().enumerate() {
        let shard_filename = format!("model-{:05}-of-{:05}.safetensors", shard_num + 1, n_shards);
        let shard_path = output_dir.join(&shard_filename);
        let shard_path_str = shard_path
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 path: {}", shard_path.display()))?;

        // Materialise quantized arrays for this shard.
        // We need owned storage for packed/scales/biases so the references
        // passed to save_safetensors stay valid.
        let mut owned_arrays: Vec<InlineArray> = Vec::new();
        // (key_string, index into owned_arrays)
        let mut shard_key_indices: Vec<(String, usize)> = Vec::new();

        for &ri in shard_record_indices {
            let rec = &records[ri];
            let assignment = &assignments[rec.idx];

            if assignment.bits == BITS_PASSTHROUGH {
                owned_arrays.push(rec.array.clone());
                let own_idx = owned_arrays.len() - 1;
                shard_key_indices.push((assignment.name.clone(), own_idx));
                weight_map.insert(assignment.name.clone(), shard_filename.clone());
            } else {
                let (packed, scales, biases) = rec
                    .array
                    .quantize_weights(assignment.group_size, assignment.bits);
                // Force evaluation before passing to save.
                packed.eval();
                scales.eval();
                biases.eval();

                owned_arrays.push(packed);
                let packed_idx = owned_arrays.len() - 1;
                owned_arrays.push(scales);
                let scales_idx = owned_arrays.len() - 1;
                owned_arrays.push(biases);
                let biases_idx = owned_arrays.len() - 1;

                shard_key_indices.push((assignment.name.clone(), packed_idx));
                shard_key_indices.push((format!("{}.scales", assignment.name), scales_idx));
                shard_key_indices.push((format!("{}.biases", assignment.name), biases_idx));

                weight_map.insert(assignment.name.clone(), shard_filename.clone());
                weight_map.insert(
                    format!("{}.scales", assignment.name),
                    shard_filename.clone(),
                );
                weight_map.insert(
                    format!("{}.biases", assignment.name),
                    shard_filename.clone(),
                );
            }
        }

        let save_entries: Vec<(&str, &InlineArray)> = shard_key_indices
            .iter()
            .map(|(k, idx)| (k.as_str(), &owned_arrays[*idx]))
            .collect();

        InlineArray::save_safetensors(shard_path_str, &save_entries);
    }

    // --- Write weight-map index ---
    write_weight_map_index(output_dir, &weight_map)?;

    Ok(())
}

/// Write `model.safetensors.index.json`.
fn write_weight_map_index(
    output_dir: &Path,
    weight_map: &HashMap<String, String>,
) -> Result<(), String> {
    use std::io::Write as _;

    let index = serde_json::json!({
        "metadata": {
            "format": "pt",
        },
        "weight_map": weight_map,
    });

    let index_path = output_dir.join("model.safetensors.index.json");
    let pretty = serde_json::to_string_pretty(&index)
        .map_err(|e| format!("failed to serialise safetensors index: {e}"))?;

    let mut f = std::fs::File::create(&index_path)
        .map_err(|e| format!("failed to create index file: {e}"))?;
    f.write_all(pretty.as_bytes())
        .map_err(|e| format!("failed to write index file: {e}"))?;

    Ok(())
}

/// Partition shard records into groups not exceeding `max_bytes` each.
///
/// Respects insertion order; never splits a single record across shards.
fn partition_into_shards<'a>(records: &'a [ShardRecord<'a>], max_bytes: u64) -> Vec<Vec<usize>> {
    let mut shards: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_bytes: u64 = 0;

    for (i, rec) in records.iter().enumerate() {
        // Start a new shard if adding this record would overflow the current
        // one (unless the shard is still empty — a single oversized record
        // must go into its own shard).
        if !current.is_empty() && current_bytes + rec.byte_estimate > max_bytes {
            shards.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current.push(i);
        current_bytes += rec.byte_estimate;
    }

    if !current.is_empty() {
        shards.push(current);
    }
    // Always produce at least one shard so the caller's loop body runs.
    if shards.is_empty() {
        shards.push(Vec::new());
    }
    shards
}

/// Byte-width of one element for a given MLX dtype code.
fn dtype_element_bytes(dtype_raw: i32) -> u64 {
    use crate::compat::Dtype;
    match Dtype::from_raw(dtype_raw) {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Uint64 | Dtype::Int64 | Dtype::Complex64 => 8,
    }
}

// ── Config writing ────────────────────────────────────────────────────────────

/// Copy `source_config_path` to `output_dir/config.json`, injecting a
/// `"quantization"` block with `group_size`, `bits`, and any per-tensor
/// overrides that differ from the default.
///
/// The `"quantization"` key is written at the top level, matching MLX's
/// convention (used by `mlx_lm` and compatible loaders).
///
/// # Errors
///
/// Returns `Err(String)` if the source config cannot be read, parsed, or if
/// writing the output fails.
pub fn write_quantization_config(
    source_config_path: &Path,
    output_dir: &Path,
    default_bits: i32,
    group_size: i32,
    per_tensor_overrides: &HashMap<String, i32>,
) -> Result<(), String> {
    let raw = std::fs::read_to_string(source_config_path)
        .map_err(|e| format!("failed to read '{}': {e}", source_config_path.display()))?;

    let mut json: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("failed to parse config.json: {e}"))?;

    // Build the quantization block.
    let mut quant = serde_json::json!({
        "group_size": group_size,
        "bits": default_bits,
    });

    // Only record overrides that differ from the default to keep the file tidy.
    let non_default: serde_json::Map<String, serde_json::Value> = per_tensor_overrides
        .iter()
        .filter(|(_, bits)| **bits != default_bits)
        .map(|(k, v)| (k.clone(), serde_json::Value::Number((*v).into())))
        .collect();

    if !non_default.is_empty() {
        quant["per_tensor_overrides"] = serde_json::Value::Object(non_default);
    }

    match &mut json {
        serde_json::Value::Object(map) => {
            map.insert("quantization".to_owned(), quant);
        }
        _ => return Err("config.json root is not a JSON object".to_owned()),
    }

    std::fs::create_dir_all(output_dir).map_err(|e| format!("failed to create output dir: {e}"))?;

    let out_path = output_dir.join("config.json");
    let serialised = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("failed to serialise config: {e}"))?;
    std::fs::write(&out_path, &serialised)
        .map_err(|e| format!("failed to write '{}': {e}", out_path.display()))?;

    Ok(())
}

// ── Full pipeline ─────────────────────────────────────────────────────────────

/// Run the end-to-end sensitivity-analysis quantization pipeline.
///
/// Combines quality evaluation, bit allocation, safetensors sharding, and
/// config writing in a single call.
///
/// Non-matrix tensors (1-D or scalar) skip quality evaluation and are always
/// passed through verbatim, because `mlx.core.quantize` requires ≥ 2 dims.
/// Critical tensors are pinned to their highest available candidate (≥ 8-bit).
///
/// Returns the per-tensor assignments that were written to disk.
pub fn quantize_model(
    weights: &HashMap<String, InlineArray>,
    source_config_path: &Path,
    output_dir: &Path,
    target_bpw: f32,
    group_size: i32,
    bits_candidates: &[i32],
    extra_critical: &[&str],
) -> Result<Vec<TensorQuant>, String> {
    // Sort deterministically so output is reproducible.
    let mut sorted_names: Vec<&String> = weights.keys().collect();
    sorted_names.sort();

    // Phase 1: evaluate quality for each quantizable weight.
    let mut tensor_scores: TensorScores = Vec::with_capacity(weights.len());

    for name in &sorted_names {
        let weight = &weights[*name];
        let param_count = weight.size();
        let is_matrix = weight.ndim() >= 2;

        // MLX quantize requires last dim divisible by group_size and ndim >= 2.
        let last_dim = if is_matrix {
            weight.dim(weight.ndim() - 1)
        } else {
            0
        };
        let quantizable = is_matrix && last_dim >= group_size && last_dim % group_size == 0;

        let scores: Vec<(i32, f64)> = if !quantizable {
            // Non-matrix or incompatible shape: passthrough.
            Vec::new()
        } else if is_critical_tensor(name) || extra_critical.iter().any(|p| name.contains(p)) {
            // Critical: fixed to highest candidate, skip slow evaluation.
            let high_bits = bits_candidates.iter().max().copied().unwrap_or(8);
            vec![(high_bits, 0.0)]
        } else {
            evaluate_tensor_quality(weight, group_size, bits_candidates)
        };

        tensor_scores.push(((*name).clone(), param_count, scores));
    }

    // Phase 2: allocate bits.
    let mut assignments = allocate_bits_for_bpw(&tensor_scores, target_bpw, extra_critical);
    for assignment in &mut assignments {
        assignment.group_size = group_size;
    }

    // Phase 3: quantize and save.
    quantize_and_save_mlx(weights, &assignments, output_dir, group_size)?;

    // Phase 4: write config.
    //
    // Compute the modal bit-width as the default for the config (keeps the
    // override table small for homogeneous models).
    let default_bits = {
        let mut freq: HashMap<i32, usize> = HashMap::new();
        for a in &assignments {
            if a.bits != BITS_PASSTHROUGH {
                *freq.entry(a.bits).or_insert(0) += 1;
            }
        }
        freq.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(bits, _)| bits)
            .unwrap_or(4)
    };

    let per_tensor_overrides: HashMap<String, i32> = assignments
        .iter()
        .filter(|a| a.bits != BITS_PASSTHROUGH && a.bits != default_bits)
        .map(|a| (a.name.clone(), a.bits))
        .collect();

    write_quantization_config(
        source_config_path,
        output_dir,
        default_bits,
        group_size,
        &per_tensor_overrides,
    )?;

    Ok(assignments)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_critical_tensor ────────────────────────────────────────────────────

    #[test]
    fn critical_tensor_exact_keys() {
        assert!(is_critical_tensor("model.norm.weight"));
        assert!(is_critical_tensor("model.embed_tokens.weight"));
        assert!(is_critical_tensor("lm_head.weight"));
    }

    #[test]
    fn critical_tensor_norm_variants() {
        assert!(is_critical_tensor("model.layers.0.input_layernorm.weight"));
        assert!(is_critical_tensor(
            "model.layers.0.post_attention_layernorm.weight"
        ));
        assert!(is_critical_tensor("model.layers.0.ln_1.weight"));
        assert!(is_critical_tensor("transformer.h.0.ln_2.bias"));
        // .norm. variant
        assert!(is_critical_tensor("model.layers.0.self_attn.q_norm.weight"));
    }

    #[test]
    fn critical_tensor_gate_and_gdn() {
        assert!(is_critical_tensor("model.layers.0.mlp.gate.weight"));
        assert!(is_critical_tensor(
            "model.layers.0.mlp.experts.12.gate_proj.weight"
        ));
        assert!(is_critical_tensor("model.layers.0.linear_attn.a_log"));
        assert!(is_critical_tensor("model.layers.0.linear_attn.A_log"));
        assert!(is_critical_tensor("model.layers.0.linear_attn.dt_bias"));
        assert!(is_critical_tensor(
            "model.layers.0.linear_attn.conv1d.weight"
        ));
    }

    #[test]
    fn non_critical_tensor_examples() {
        assert!(!is_critical_tensor(
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!is_critical_tensor("model.layers.0.mlp.down_proj.weight"));
        assert!(!is_critical_tensor("model.layers.0.mlp.up_proj.weight"));
        assert!(!is_critical_tensor("model.layers.0.mlp.gate_proj.weight"));
    }

    // ── effective_bpw ─────────────────────────────────────────────────────────

    #[test]
    fn effective_bpw_uniform_4bit() {
        let scores: TensorScores = vec![
            ("a".into(), 100, vec![(4, 0.01)]),
            ("b".into(), 200, vec![(4, 0.02)]),
        ];
        let steps = vec![0, 0];
        let bpw = effective_bpw(&scores, &steps);
        assert!((bpw - 4.0).abs() < 1e-9, "expected 4.0, got {bpw}");
    }

    #[test]
    fn effective_bpw_mixed() {
        // 100-param tensor at 4-bit, 100-param tensor at 8-bit → 6.0 BPW.
        let scores: TensorScores = vec![
            ("a".into(), 100, vec![(4, 0.05), (8, 0.0)]),
            ("b".into(), 100, vec![(4, 0.05), (8, 0.0)]),
        ];
        let steps = vec![0, 1]; // a at step 0 (4-bit), b at step 1 (8-bit)
        let bpw = effective_bpw(&scores, &steps);
        assert!((bpw - 6.0).abs() < 1e-9, "expected 6.0, got {bpw}");
    }

    #[test]
    fn effective_bpw_counts_passthrough_as_bf16() {
        let scores: TensorScores = vec![
            ("a".into(), 100, vec![]),
            ("b".into(), 100, vec![(4, 0.02)]),
        ];
        let steps = vec![0, 0];
        let bpw = effective_bpw(&scores, &steps);
        assert!((bpw - 10.0).abs() < 1e-9, "expected 10.0, got {bpw}");
    }

    // ── allocate_bits_for_bpw ─────────────────────────────────────────────────

    #[test]
    fn allocate_critical_always_high() {
        // Critical tensor must end up at 8-bit even with a very low target BPW.
        let scores: TensorScores = vec![
            ("model.norm.weight".into(), 1024, vec![(4, 0.1), (8, 0.0)]),
            (
                "model.layers.0.mlp.down_proj.weight".into(),
                4096,
                vec![(4, 0.05), (8, 0.01)],
            ),
        ];
        let result = allocate_bits_for_bpw(&scores, 2.0, &[]);
        let norm = result
            .iter()
            .find(|t| t.name == "model.norm.weight")
            .unwrap();
        assert_eq!(norm.bits, 8, "critical tensor must be pinned to 8-bit");
    }

    #[test]
    fn allocate_upgrades_worst_quality_first() {
        // Two equal-size non-critical tensors.  down_proj has much worse
        // quality at 4-bit (score 0.50 vs 0.01) and should be upgraded first.
        //
        // At BPW=6.0 with two equal-size tensors and only [4, 8] as candidates,
        // exactly one upgrade is needed: (4+8)/2 = 6.0. The greedy allocator
        // should choose down_proj (higher score → worse quality).
        let scores: TensorScores = vec![
            (
                "model.layers.0.mlp.gate_proj.weight".into(),
                1024,
                vec![(4, 0.01), (8, 0.0)],
            ),
            (
                "model.layers.0.mlp.down_proj.weight".into(),
                1024,
                vec![(4, 0.50), (8, 0.0)],
            ),
        ];
        // Target 6 BPW — one upgrade needed; down_proj should receive it.
        let result = allocate_bits_for_bpw(&scores, 6.0, &[]);
        let gate_proj = result
            .iter()
            .find(|t| t.name.contains("gate_proj"))
            .unwrap();
        let down_proj = result
            .iter()
            .find(|t| t.name.contains("down_proj"))
            .unwrap();
        assert_eq!(
            down_proj.bits, 8,
            "worst-quality tensor should be upgraded first"
        );
        assert_eq!(
            gate_proj.bits, 4,
            "better-quality tensor should remain at lower bits"
        );

        // At BPW=8 both tensors must be at 8-bit.
        let result_high = allocate_bits_for_bpw(&scores, 8.0, &[]);
        for t in &result_high {
            assert_eq!(t.bits, 8, "at BPW=8 all tensors should reach 8-bit");
        }
    }

    #[test]
    fn allocate_passthrough_for_empty_scores() {
        // A tensor with an empty scores list should become BITS_PASSTHROUGH.
        let scores: TensorScores = vec![
            ("model.embed_tokens.weight".into(), 512, vec![]),
            (
                "model.layers.0.mlp.down_proj.weight".into(),
                4096,
                vec![(4, 0.02), (8, 0.0)],
            ),
        ];
        let result = allocate_bits_for_bpw(&scores, 4.0, &[]);
        let embed = result
            .iter()
            .find(|t| t.name == "model.embed_tokens.weight")
            .unwrap();
        assert_eq!(embed.bits, BITS_PASSTHROUGH);
    }

    // ── dtype_element_bytes ───────────────────────────────────────────────────

    #[test]
    fn dtype_bytes_spot_checks() {
        assert_eq!(
            dtype_element_bytes(crate::compat::Dtype::Float32.as_i32()),
            4
        );
        assert_eq!(
            dtype_element_bytes(crate::compat::Dtype::Bfloat16.as_i32()),
            2
        );
        assert_eq!(dtype_element_bytes(crate::compat::Dtype::Uint8.as_i32()), 1);
        assert_eq!(
            dtype_element_bytes(crate::compat::Dtype::Float16.as_i32()),
            2
        );
    }

    // ── partition_into_shards ─────────────────────────────────────────────────

    #[test]
    fn partition_splits_at_cap() {
        let dummy = InlineArray::from_f32(0.0);
        // Build two records that together exceed SHARD_SIZE_BYTES.
        let records = [
            ShardRecord {
                idx: 0,
                array: &dummy,
                byte_estimate: 3 * 1024 * 1024 * 1024,
            },
            ShardRecord {
                idx: 1,
                array: &dummy,
                byte_estimate: 3 * 1024 * 1024 * 1024,
            },
        ];
        let shards = partition_into_shards(&records, SHARD_SIZE_BYTES);
        assert_eq!(shards.len(), 2, "expected two shards");
        assert_eq!(shards[0], vec![0]);
        assert_eq!(shards[1], vec![1]);
    }

    #[test]
    fn partition_fits_in_one_shard() {
        let dummy = InlineArray::from_f32(0.0);
        let records = [
            ShardRecord {
                idx: 0,
                array: &dummy,
                byte_estimate: 1024,
            },
            ShardRecord {
                idx: 1,
                array: &dummy,
                byte_estimate: 1024,
            },
        ];
        let shards = partition_into_shards(&records, SHARD_SIZE_BYTES);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], vec![0, 1]);
    }

    #[test]
    fn partition_empty_input_returns_one_empty_shard() {
        let empty: &[ShardRecord<'_>] = &[];
        let shards = partition_into_shards(empty, SHARD_SIZE_BYTES);
        assert_eq!(shards.len(), 1);
        assert!(shards[0].is_empty());
    }
}
