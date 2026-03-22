//! Expert weight packing tool for SSD-offloaded MoE inference.
//!
//! Reads safetensors model files and writes per-layer packed expert binary files
//! for use with the expert offloading system.  Each layer produces a single
//! `layer_XX.bin` file where expert `E` starts at byte `E * expert_size`, and a
//! `layout.json` describes the binary layout for the reader.
//!
//! Supports both pre-quantized models (safetensors already contains `.scales`
//! and `.biases` alongside `.weight`) and full-precision models where
//! affine per-group quantization is performed on the fly.
//!
//! # Usage
//!
//! ```text
//! pmetal pack-experts --model <path> --output <dir> [--bits 4]
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{Seek, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use pmetal_models::expert_layout::{ExpertComponent, ExpertPackLayout, PackedBits};

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Pack expert weights from a safetensors model directory into per-layer binary
/// files suitable for fast parallel `pread()` access during SSD-offloaded
/// MoE inference.
///
/// # Arguments
///
/// * `model_dir` – HuggingFace model directory containing `config.json` and
///   one or more `.safetensors` files (or an index JSON).
/// * `output_dir` – Destination directory; created if it does not exist.
/// * `bits` – Target bit width for weight quantization (`2` or `4`).
///   Defaults to `4` when `None`. Full-precision models are
///   quantized on-the-fly using affine per-group quantization.
///
/// # Errors
///
/// Returns an error if the model directory cannot be read, if required config
/// keys are missing, if any tensor size mismatches the expected layout, or if
/// the final file-size verification fails.
pub fn pack_experts(model_dir: &Path, output_dir: &Path, bits: Option<u8>) -> Result<()> {
    let bits = match bits.unwrap_or(4) {
        4 => PackedBits::Four,
        2 => PackedBits::Two,
        b => bail!("unsupported bit width: {b} (must be 2 or 4)"),
    };

    // ── 1. Read config.json ───────────────────────────────────────────────────
    let config_text =
        fs::read_to_string(model_dir.join("config.json")).context("Failed to read config.json")?;
    let config: serde_json::Value =
        serde_json::from_str(&config_text).context("Failed to parse config.json")?;

    // Some VLM checkpoints wrap model config under `text_config`.
    let mc = config.get("text_config").unwrap_or(&config);

    let num_layers = field_usize(mc, "num_hidden_layers")?;
    let hidden_dim = field_usize(mc, "hidden_size")?;
    let num_experts = field_usize(mc, "num_experts")?;
    let intermediate = field_usize(mc, "moe_intermediate_size")?;

    // `decoder_sparse_step`: every Nth layer is a MoE layer (default 1 = all).
    let step = mc
        .get("decoder_sparse_step")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;

    // `mlp_only_layers`: indices that are dense MLP, not MoE.
    let mlp_only: Vec<usize> = mc
        .get("mlp_only_layers")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|i| i as usize))
                .collect()
        })
        .unwrap_or_default();

    // Group size for quantization — falls back to 64 if not in config.
    let group_size = mc
        .get("quantization_config")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(64) as usize;

    // Determine which layers are MoE layers.
    let moe_layers: Vec<usize> = (0..num_layers)
        .filter(|&l| {
            let is_moe_slot = step == 1 || (l + 1) % step == 0;
            is_moe_slot && !mlp_only.contains(&l)
        })
        .collect();

    if moe_layers.is_empty() {
        bail!(
            "No MoE layers found — check config.json fields \
             (num_experts, decoder_sparse_step, mlp_only_layers)"
        );
    }

    let model_name = model_dir
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let layout = ExpertPackLayout::new(
        model_name,
        num_layers,
        num_experts,
        hidden_dim,
        intermediate,
        group_size,
        bits,
        moe_layers.clone(),
    );

    eprintln!("Expert pack layout:");
    eprintln!("  Model:           {}", layout.model_name);
    eprintln!("  MoE layers:      {} / {}", moe_layers.len(), num_layers);
    eprintln!("  Experts/layer:   {}", num_experts);
    eprintln!("  Group size:      {}", group_size);
    eprintln!(
        "  Bits:            {}",
        if bits == PackedBits::Four { 4 } else { 2 }
    );
    eprintln!(
        "  Expert size:     {:.2} MB",
        layout.expert_size as f64 / 1e6
    );
    eprintln!(
        "  Total/layer:     {:.2} GB",
        (layout.expert_size * num_experts) as f64 / 1e9
    );

    fs::create_dir_all(output_dir).context("Failed to create output directory")?;

    // ── 2. Build tensor→shard map from index.json ─────────────────────────────
    //
    // For single-file models the map remains empty and we fall back to
    // resolving against "model.safetensors".
    let index_path = model_dir.join("model.safetensors.index.json");
    let shard_map: HashMap<String, String> = if index_path.exists() {
        let idx_text = fs::read_to_string(&index_path).context("Failed to read index.json")?;
        let idx: serde_json::Value =
            serde_json::from_str(&idx_text).context("Failed to parse index.json")?;
        idx["weight_map"]
            .as_object()
            .context("index.json missing weight_map")?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
            .collect()
    } else {
        HashMap::new()
    };

    // ── 3. Detect whether the model is pre-quantized ──────────────────────────
    //
    // A pre-quantized model has `.scales` and `.biases` tensors alongside
    // `.weight`.  We detect this by looking for any expert scales key in the
    // index, or by peeking inside the single safetensors file.
    let is_prequantized = detect_prequantized(&shard_map, model_dir)?;
    eprintln!("  Pre-quantized:   {}", is_prequantized);

    // ── 4. Shard byte cache keyed by shard filename ───────────────────────────
    //
    // Shard bytes are loaded on demand and retained for the lifetime of the
    // packing run so that later layers sharing a shard file do not re-read it.
    // The parsed `SafeTensors` view is zero-copy into these bytes.
    let mut shard_byte_cache: HashMap<String, Vec<u8>> = HashMap::new();

    // ── 5. Pack each MoE layer ────────────────────────────────────────────────

    let record = &layout.record;

    // Component list in binary layout order (matches ExpertRecord::compute).
    let components: &[(&str, &ExpertComponent)] = &[
        ("gate_proj.weight", &record.gate_weight),
        ("gate_proj.scales", &record.gate_scales),
        ("gate_proj.biases", &record.gate_biases),
        ("up_proj.weight", &record.up_weight),
        ("up_proj.scales", &record.up_scales),
        ("up_proj.biases", &record.up_biases),
        ("down_proj.weight", &record.down_weight),
        ("down_proj.scales", &record.down_scales),
        ("down_proj.biases", &record.down_biases),
    ];

    let total = moe_layers.len();

    for (progress, &layer_idx) in moe_layers.iter().enumerate() {
        eprint!(
            "\r  Packing layer {}/{} (layer_idx={})...",
            progress + 1,
            total,
            layer_idx
        );

        // Create and pre-allocate the layer file.
        let layer_path = layout.layer_file_path(output_dir, layer_idx);
        let mut layer_file = fs::File::create(&layer_path)
            .with_context(|| format!("Cannot create {}", layer_path.display()))?;
        let total_layer_bytes = layout.expert_size * num_experts;
        layer_file
            .set_len(total_layer_bytes as u64)
            .with_context(|| format!("set_len failed for {}", layer_path.display()))?;

        // Pre-load all shards needed for this layer (deduplicated).
        let needed_shards = collect_needed_shards(
            layer_idx,
            num_experts,
            is_prequantized,
            &shard_map,
            model_dir,
        );
        for shard_name in &needed_shards {
            if !shard_byte_cache.contains_key(shard_name) {
                let shard_path = model_dir.join(shard_name);
                let data = fs::read(&shard_path)
                    .with_context(|| format!("Failed to read {}", shard_path.display()))?;
                shard_byte_cache.insert(shard_name.clone(), data);
            }
        }

        // Write each expert.  The inner write_component call receives the raw
        // shard bytes; it parses the safetensors header once per call (O(header)),
        // which is cheap compared to the disk read we already paid.
        for expert_idx in 0..num_experts {
            let expert_base = layout.expert_offset(expert_idx);

            for (suffix, component) in components {
                let key = format!("model.layers.{layer_idx}.mlp.experts.{expert_idx}.{suffix}");
                let file_offset = (expert_base + component.offset) as u64;

                let shard_name = resolve_shard(&key, &shard_map, model_dir);

                match shard_name {
                    None => {
                        write_zeros(&mut layer_file, file_offset, component.size)?;
                    }
                    Some(ref sn) => {
                        if let Some(bytes) = shard_byte_cache.get(sn) {
                            write_component(
                                &mut layer_file,
                                bytes,
                                &key,
                                suffix,
                                component,
                                file_offset,
                                is_prequantized,
                                bits,
                                group_size,
                            )?;
                        } else {
                            // Shard expected but not in cache — write zeros.
                            write_zeros(&mut layer_file, file_offset, component.size)?;
                        }
                    }
                }
            }
        }

        layer_file
            .flush()
            .with_context(|| format!("Flush failed for {}", layer_path.display()))?;
    }

    eprintln!(
        "\r  Packed {} layers.                                      ",
        total
    );

    // ── 6. Save layout.json ───────────────────────────────────────────────────
    layout
        .save(output_dir)
        .context("Failed to save layout.json")?;
    eprintln!("  Layout written → {}/layout.json", output_dir.display());

    // ── 7. Verification pass ──────────────────────────────────────────────────
    verify_output(&layout, output_dir)?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Component write helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Write a single expert component into the open layer file.
///
/// `shard_bytes` is the raw safetensors file contents; this function
/// deserializes the header (O(header_size)) to locate the tensor.
///
/// For pre-quantized models the raw tensor bytes are copied verbatim.
/// For full-precision models the weight pass also writes scales and biases
/// for the projection so that the shard is not re-parsed for those suffixes.
#[allow(clippy::too_many_arguments)]
fn write_component(
    file: &mut fs::File,
    shard_bytes: &[u8],
    key: &str,
    suffix: &str,
    component: &ExpertComponent,
    file_offset: u64,
    is_prequantized: bool,
    bits: PackedBits,
    group_size: usize,
) -> Result<()> {
    let tensors = safetensors::SafeTensors::deserialize(shard_bytes)
        .map_err(|e| anyhow::anyhow!("safetensors parse error for `{key}`: {e}"))?;

    if is_prequantized {
        // Pre-quantized path: copy the raw tensor bytes verbatim.
        match tensors.tensor(key) {
            Ok(tv) => {
                let data = tv.data();
                if data.len() != component.size {
                    bail!(
                        "Size mismatch for `{key}`: layout expects {} bytes, \
                         tensor has {} bytes",
                        component.size,
                        data.len()
                    );
                }
                file.seek(std::io::SeekFrom::Start(file_offset))?;
                file.write_all(data)?;
            }
            Err(_) => {
                // Tensor absent — sparse checkpoint or optional field.
                write_zeros(file, file_offset, component.size)?;
            }
        }
    } else {
        // Full-precision path: quantize weight tensors on the fly.
        // The weight pass also writes scales and biases; the subsequent
        // `.scales` / `.biases` suffix visits are no-ops.
        if suffix.ends_with(".weight") {
            quantize_and_write(
                file,
                &tensors,
                key,
                component,
                file_offset,
                bits,
                group_size,
            )?;
        }
        // For `.scales` / `.biases` on a non-quantized model: bytes were
        // already written by the preceding `.weight` pass — skip.
    }

    Ok(())
}

/// Quantize a full-precision weight tensor (f32/bf16/f16) to affine per-group
/// quantization and write weight, scales, and biases at the correct offsets.
///
/// # Quantization scheme
///
/// Matches the MLX `mx.quantize` / flash-moe affine format:
/// - Weight matrix shape: `[out_dim, in_dim]`
/// - Divided into groups of `group_size` columns
/// - Per group: `scale = (max − min) / (2^bits − 1)`, `bias = min`
/// - Quantized value: `q = round((w − bias) / scale)` clamped to `[0, 2^bits − 1]`
/// - Packed: `bits` values per uint32 (LSB-first, row-major)
///
/// Scales and biases are stored as bf16 at the offsets immediately following the
/// weight in the binary layout (as computed by `ExpertRecord::compute`).
fn quantize_and_write(
    file: &mut fs::File,
    tensors: &safetensors::SafeTensors<'_>,
    weight_key: &str,
    weight_component: &ExpertComponent,
    weight_file_offset: u64,
    bits: PackedBits,
    group_size: usize,
) -> Result<()> {
    use safetensors::Dtype;

    let tv = tensors
        .tensor(weight_key)
        .map_err(|_| anyhow::anyhow!("Tensor not found: `{weight_key}`"))?;

    let shape = tv.shape();
    if shape.len() != 2 {
        bail!("Expected 2-D weight tensor for `{weight_key}`, got shape {shape:?}");
    }
    let out_dim = shape[0];
    let in_dim = shape[1];
    let raw = tv.data();

    // Decode the raw bytes to f32.
    let weights_f32: Vec<f32> = match tv.dtype() {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                // bf16 → f32: shift the 16-bit pattern to the top of a f32.
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        dt => bail!("Unsupported dtype {dt:?} for weight tensor `{weight_key}`"),
    };

    let pf = bits.pack_factor(); // values packed per uint32
    let bit_width = bits as u32;
    let max_q = ((1u32 << bit_width) - 1) as f32;

    // Derived dimensions.
    let n_groups_per_row = in_dim.div_ceil(group_size);
    // packed_cols assumes in_dim is divisible by pf (guaranteed by model design).
    let packed_cols = in_dim / pf;
    let weight_u32_count = out_dim * packed_cols;
    let sb_count = out_dim * n_groups_per_row; // scales / biases count

    let mut packed_weight = vec![0u32; weight_u32_count];
    let mut scales_bf16 = vec![0u16; sb_count];
    let mut biases_bf16 = vec![0u16; sb_count];

    for row in 0..out_dim {
        let row_base = row * in_dim;
        for g in 0..n_groups_per_row {
            let col_start = g * group_size;
            let col_end = (col_start + group_size).min(in_dim);
            let group_slice = &weights_f32[row_base + col_start..row_base + col_end];

            // Compute per-group affine parameters.
            let mut w_min = f32::INFINITY;
            let mut w_max = f32::NEG_INFINITY;
            for &w in group_slice {
                if w < w_min {
                    w_min = w;
                }
                if w > w_max {
                    w_max = w;
                }
            }
            if w_min == w_max {
                // Constant group — use a tiny non-zero range to avoid NaN.
                w_max = w_min + 1e-6;
            }
            let scale = (w_max - w_min) / max_q;
            let bias = w_min;

            let sg = row * n_groups_per_row + g;
            scales_bf16[sg] = f32_to_bf16(scale);
            biases_bf16[sg] = f32_to_bf16(bias);

            // Pack quantized values into uint32 words.
            let packed_col_start = col_start / pf;
            let u32_count_this_group = group_slice.len().div_ceil(pf);

            for u in 0..u32_count_this_group {
                let mut word: u32 = 0;
                let packed_col = packed_col_start + u;
                for b_bit in 0..pf {
                    let col_idx = col_start + u * pf + b_bit;
                    if col_idx >= col_end {
                        break;
                    }
                    let w = weights_f32[row_base + col_idx];
                    let q = ((w - bias) / scale).round().clamp(0.0, max_q) as u32;
                    word |= q << (b_bit as u32 * bit_width);
                }
                if packed_col < packed_cols {
                    packed_weight[row * packed_cols + packed_col] = word;
                }
            }
        }
    }

    // Validate that computed sizes match the layout descriptor.
    let weight_bytes = weight_u32_count * 4;
    if weight_bytes != weight_component.size {
        bail!(
            "Quantized weight byte count {weight_bytes} != layout expects {} \
             for `{weight_key}`",
            weight_component.size
        );
    }
    let scales_bytes = sb_count * 2;

    // Write weight, scales, biases contiguously at their layout offsets.
    file.seek(std::io::SeekFrom::Start(weight_file_offset))?;
    file.write_all(&u32_slice_to_bytes(&packed_weight))?;

    let scales_offset = weight_file_offset + weight_bytes as u64;
    file.seek(std::io::SeekFrom::Start(scales_offset))?;
    file.write_all(&u16_slice_to_bytes(&scales_bf16))?;

    let biases_offset = scales_offset + scales_bytes as u64;
    file.seek(std::io::SeekFrom::Start(biases_offset))?;
    file.write_all(&u16_slice_to_bytes(&biases_bf16))?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Shard resolution helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Collect the unique shard file names needed to pack a given layer.
///
/// Probes expert 0 (and expert 1 for large models where experts may span shard
/// boundaries) for each projection and suffix.
fn collect_needed_shards(
    layer_idx: usize,
    num_experts: usize,
    is_prequantized: bool,
    shard_map: &HashMap<String, String>,
    model_dir: &Path,
) -> Vec<String> {
    let projections = ["gate_proj", "up_proj", "down_proj"];
    let mut suffixes = vec!["weight"];
    if is_prequantized {
        suffixes.push("scales");
        suffixes.push("biases");
    }

    let mut shards: Vec<String> = Vec::new();
    // Always probe expert 0.
    for proj in &projections {
        for sfx in &suffixes {
            let key = format!("model.layers.{layer_idx}.mlp.experts.0.{proj}.{sfx}");
            if let Some(s) = resolve_shard(&key, shard_map, model_dir) {
                if !shards.contains(&s) {
                    shards.push(s);
                }
            }
        }
    }
    // Also probe expert 1 to catch shard splits in very large models.
    if num_experts > 1 {
        for proj in &projections {
            let key = format!("model.layers.{layer_idx}.mlp.experts.1.{proj}.weight");
            if let Some(s) = resolve_shard(&key, shard_map, model_dir) {
                if !shards.contains(&s) {
                    shards.push(s);
                }
            }
        }
    }
    shards
}

/// Resolve the shard filename that contains tensor `key`.
///
/// Returns `None` if the key is absent from the index and no single-file model
/// exists.
fn resolve_shard(
    key: &str,
    shard_map: &HashMap<String, String>,
    model_dir: &Path,
) -> Option<String> {
    if !shard_map.is_empty() {
        shard_map.get(key).cloned()
    } else {
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            Some("model.safetensors".to_string())
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detection helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return `true` if the model checkpoint is already quantized (has `.scales`
/// and `.biases` alongside expert `.weight` tensors).
fn detect_prequantized(shard_map: &HashMap<String, String>, model_dir: &Path) -> Result<bool> {
    // Fast path: scan the index map.
    if !shard_map.is_empty() {
        let found = shard_map
            .keys()
            .any(|k| k.contains(".experts.") && k.ends_with(".scales"));
        return Ok(found);
    }

    // Single-file fallback: peek tensor names.
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        let data = fs::read(&single).context("Failed to read model.safetensors")?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| anyhow::anyhow!("safetensors parse error: {e}"))?;
        let names = tensors.names();
        let mut found = false;
        for n in names {
            if n.contains(".experts.") && n.ends_with(".scales") {
                found = true;
                break;
            }
        }
        return Ok(found);
    }

    Ok(false)
}

// ─────────────────────────────────────────────────────────────────────────────
// Verification
// ─────────────────────────────────────────────────────────────────────────────

/// Verify the output by checking file sizes and read-back sampling.
///
/// Checks:
/// 1. Every expected layer file exists with the correct byte count.
/// 2. The first 64 bytes of the gate_weight region in the first and last layers
///    are not all zero (guards against silent write failures).
/// 3. `layout.json` roundtrips through JSON without data loss.
fn verify_output(layout: &ExpertPackLayout, output_dir: &Path) -> Result<()> {
    eprint!("  Verifying output...");

    let expected_layer_bytes = (layout.expert_size * layout.num_experts) as u64;

    for &layer_idx in &layout.moe_layer_indices {
        let path = layout.layer_file_path(output_dir, layer_idx);
        let actual = fs::metadata(&path)
            .with_context(|| format!("Cannot stat {}", path.display()))?
            .len();
        if actual != expected_layer_bytes {
            bail!(
                "Verification failed: {} has {actual} bytes, expected {expected_layer_bytes}",
                path.display()
            );
        }
    }

    // Read-back sample from first and last MoE layers.
    let verify_layers: Vec<usize> = {
        let mut v = Vec::new();
        if let Some(&first) = layout.moe_layer_indices.first() {
            v.push(first);
        }
        if let Some(&last) = layout.moe_layer_indices.last() {
            if Some(last) != layout.moe_layer_indices.first().copied() {
                v.push(last);
            }
        }
        v
    };

    for layer_idx in verify_layers {
        let path = layout.layer_file_path(output_dir, layer_idx);
        let data =
            fs::read(&path).with_context(|| format!("Readback failed for {}", path.display()))?;

        let sample_offset = layout.record.gate_weight.offset;
        let sample_end = (sample_offset + 64).min(data.len());
        let sample = &data[sample_offset..sample_end];

        if sample.iter().all(|&b| b == 0) {
            bail!(
                "Verification failed: layer {layer_idx} gate_weight bytes are all zero — \
                 this may indicate missing expert tensors in the source checkpoint"
            );
        }
    }

    eprintln!(" OK");

    // Roundtrip layout.json.
    let reloaded = ExpertPackLayout::load(output_dir).context("Failed to reload layout.json")?;
    if reloaded.expert_size != layout.expert_size
        || reloaded.num_experts != layout.num_experts
        || reloaded.moe_layer_indices != layout.moe_layer_indices
    {
        bail!("layout.json roundtrip mismatch — JSON serialization may be corrupted");
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Seek to `offset` and write `size` zero bytes.
fn write_zeros(file: &mut fs::File, offset: u64, size: usize) -> Result<()> {
    file.seek(std::io::SeekFrom::Start(offset))?;
    file.write_all(&vec![0u8; size])?;
    Ok(())
}

/// Extract a required `usize` field from a JSON object.
fn field_usize(obj: &serde_json::Value, key: &str) -> Result<usize> {
    obj.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| anyhow::anyhow!("config.json missing or invalid field `{key}`"))
}

/// Convert f32 to bf16 bit pattern with round-to-nearest-even.
#[inline]
fn f32_to_bf16(v: f32) -> u16 {
    // bf16 = upper 16 bits of f32 (same exponent width, truncated mantissa).
    // Round-to-nearest-even: add 0x8000 (or 0x7FFF + lsb) before shifting.
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let round_bit = bits & 0xFFFF;
    let rounded = bits
        + if round_bit > 0x8000 || (round_bit == 0x8000 && lsb != 0) {
            0x8000
        } else {
            0
        };
    (rounded >> 16) as u16
}

/// Serialize `&[u32]` to little-endian bytes.
fn u32_slice_to_bytes(src: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 4);
    for &v in src {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Serialize `&[u16]` to little-endian bytes.
fn u16_slice_to_bytes(src: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &v in src {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Convert IEEE 754 f16 bit pattern to f32 without external crate dependency.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Signed zero.
            f32::from_bits(sign << 31)
        } else {
            // Subnormal f16 → normalized f32.
            let mut m = mant;
            let mut e = 0i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Infinity or NaN.
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normal f16.
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_round_trip() {
        #[allow(clippy::approx_constant)]
        for &v in &[0.0f32, 1.0, -1.0, 3.14, -0.001, 65504.0] {
            let bf = f32_to_bf16(v);
            let recovered = f32::from_bits((bf as u32) << 16);
            let rel_err = ((v - recovered).abs() / (v.abs().max(1e-10))) as f64;
            assert!(
                rel_err < 0.01 || v.abs() < 1e-4,
                "bf16 round-trip error too large for {v}: got {recovered} (rel_err={rel_err:.6})"
            );
        }
    }

    #[test]
    fn test_f32_to_bf16_special() {
        assert_eq!(f32_to_bf16(0.0), 0x0000u16);
        assert_eq!(f32_to_bf16(-0.0), 0x8000u16);
        // f32(1.0) = 0x3F800000; bf16(1.0) = 0x3F80.
        assert_eq!(f32_to_bf16(1.0), 0x3F80u16);
    }

    #[test]
    fn test_u32_slice_to_bytes_endian() {
        let src = vec![0xDEAD_BEEFu32];
        let bytes = u32_slice_to_bytes(&src);
        // Little-endian: EF BE AD DE.
        assert_eq!(bytes, [0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn test_u16_slice_to_bytes_endian() {
        let src = vec![0x1234u16, 0xABCDu16];
        let bytes = u16_slice_to_bytes(&src);
        assert_eq!(bytes, [0x34, 0x12, 0xCD, 0xAB]);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // f16(1.0) = 0x3C00: sign=0, exp=15, mant=0.
        assert_eq!(f16_to_f32(0x3C00), 1.0f32);
    }

    #[test]
    fn test_f16_to_f32_negative() {
        // f16(-1.0) = 0xBC00.
        assert_eq!(f16_to_f32(0xBC00), -1.0f32);
    }

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
        assert!(f16_to_f32(0x8000).is_sign_negative());
    }

    #[test]
    fn test_quantize_4bit_packs_values() {
        // Verify that pack_factor=8 and 4-bit encoding works correctly.
        // Weights [0, 1, 2, 3, 4, 5, 6, 7] with scale=7/15, bias=0.
        // q_i = round(i / (7/15)) = round(i * 15/7).
        let pf = PackedBits::Four.pack_factor();
        assert_eq!(pf, 8);
        // Ensure scale computation gives non-zero word.
        let scale = 7.0f32 / 15.0;
        let max_q = 15.0f32;
        let mut word: u32 = 0;
        for (i, w) in (0..8u32).map(|i| (i, i as f32)) {
            let q = (w / scale).round().clamp(0.0, max_q) as u32;
            word |= q << (i * 4);
        }
        assert_ne!(word, 0, "packed word should not be all zeros");
        // The last value (w=7) should quantize to 15 = 0xF in the top nibble.
        assert_eq!((word >> 28) & 0xF, 15);
    }

    /// End-to-end smoke test: build a synthetic pre-quantized safetensors model
    /// and verify pack_experts produces correct layer files.
    #[test]
    fn test_pack_experts_smoke_prequantized() {
        use std::collections::HashMap;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let model_dir = tmp.path().join("model");
        let out_dir = tmp.path().join("packed");
        fs::create_dir_all(&model_dir).unwrap();

        // Dimensions must satisfy: hidden % pf == 0 AND intermediate % pf == 0
        // where pf = pack_factor(4-bit) = 8.
        // Use hidden=16, intermediate=8 (multiples of 8), group_size=8.
        let hidden = 16usize;
        let inter = 8usize;
        let gs = 8usize;

        let config = serde_json::json!({
            "num_hidden_layers": 2,
            "hidden_size": hidden,
            "num_experts": 2,
            "moe_intermediate_size": inter,
            "decoder_sparse_step": 1,
            "quantization_config": { "group_size": gs, "bits": 4 },
        });
        fs::write(
            model_dir.join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        // Build synthetic tensor buffers matching the ExpertRecord layout.
        use pmetal_models::expert_layout::ExpertRecord;
        let record = ExpertRecord::compute(hidden, inter, gs, PackedBits::Four);

        let mut tensors: HashMap<String, (Vec<usize>, safetensors::Dtype, Vec<u8>)> =
            HashMap::new();

        let projections: &[(&str, usize, usize)] = &[
            ("gate_proj", inter, hidden),
            ("up_proj", inter, hidden),
            ("down_proj", hidden, inter),
        ];

        let pf = PackedBits::Four.pack_factor(); // 8
        // Sanity: verify packing is possible with chosen dimensions.
        assert!(hidden % pf == 0, "hidden must be divisible by pack_factor");
        assert!(
            inter % pf == 0,
            "intermediate must be divisible by pack_factor"
        );

        for l in 0..2usize {
            for e in 0..2usize {
                for &(proj, out, inp) in projections {
                    let packed_cols = inp / pf;
                    let n_groups = inp.div_ceil(gs);

                    // Weight: [out, packed_cols] uint32 — each element is a packed
                    // group of 8 4-bit values.
                    let w_len = out * packed_cols * 4;
                    let mut w_bytes = vec![0u8; w_len];
                    // Non-zero pattern so the readback check passes.
                    for (i, byte) in w_bytes.iter_mut().enumerate() {
                        *byte = ((l * 100 + e * 10 + i + 1) & 0xFF) as u8;
                    }

                    // Scales: [out, n_groups] bf16 — bf16(1.0) = 0x3F80.
                    let s_len = out * n_groups * 2;
                    let s_bytes: Vec<u8> = (0..s_len)
                        .map(|i| if i % 2 == 0 { 0x80 } else { 0x3F })
                        .collect();
                    // Biases: same layout, bf16(0.0).
                    let b_bytes = vec![0u8; s_len];

                    let base = format!("model.layers.{l}.mlp.experts.{e}.{proj}");
                    tensors.insert(
                        format!("{base}.weight"),
                        (vec![out, packed_cols], safetensors::Dtype::U32, w_bytes),
                    );
                    tensors.insert(
                        format!("{base}.scales"),
                        (vec![out, n_groups], safetensors::Dtype::BF16, s_bytes),
                    );
                    tensors.insert(
                        format!("{base}.biases"),
                        (vec![out, n_groups], safetensors::Dtype::BF16, b_bytes),
                    );

                    let _ = &record; // suppress unused warning
                }
            }
        }

        let st_bytes = build_safetensors_bytes(&tensors);
        fs::write(model_dir.join("model.safetensors"), &st_bytes).unwrap();

        // Run pack_experts.
        pack_experts(&model_dir, &out_dir, Some(4)).expect("pack_experts failed");

        // Verify layout.json.
        assert!(out_dir.join("layout.json").exists(), "layout.json missing");
        let layout = ExpertPackLayout::load(&out_dir).unwrap();

        // Verify layer files.
        for l in 0..2usize {
            let path = out_dir.join(format!("layer_{l:02}.bin"));
            assert!(path.exists(), "layer_{l:02}.bin missing");
            let size = fs::metadata(&path).unwrap().len();
            assert_eq!(
                size,
                (layout.expert_size * 2) as u64,
                "layer {l} size mismatch"
            );
        }

        // Verify roundtrip: gate_weight bytes of expert 0 layer 0 are non-zero.
        let layer0 = fs::read(out_dir.join("layer_00.bin")).unwrap();
        let sample =
            &layer0[layout.record.gate_weight.offset..layout.record.gate_weight.offset + 4];
        assert_ne!(
            sample, [0u8; 4],
            "expert 0 gate_weight bytes should be non-zero"
        );
    }

    /// Build a minimal valid safetensors byte stream.
    fn build_safetensors_bytes(
        tensors: &HashMap<String, (Vec<usize>, safetensors::Dtype, Vec<u8>)>,
    ) -> Vec<u8> {
        // safetensors wire format:
        //   [u64 LE] header_len
        //   [header_len bytes] JSON header
        //   [...] concatenated tensor data
        //
        // JSON: { "name": { "dtype": "...", "shape": [...], "data_offsets": [start, end] }, ... }

        let mut data_buf: Vec<u8> = Vec::new();
        let mut header: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        header.insert("__metadata__".to_string(), serde_json::json!({}));

        type TensorEntry = (Vec<usize>, safetensors::Dtype, Vec<u8>);
        let mut sorted: Vec<(&String, &TensorEntry)> = tensors.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());

        for (name, (shape, dtype, data)) in &sorted {
            let start = data_buf.len();
            data_buf.extend_from_slice(data);
            let end = data_buf.len();

            let dtype_str = match dtype {
                safetensors::Dtype::F32 => "F32",
                safetensors::Dtype::BF16 => "BF16",
                safetensors::Dtype::F16 => "F16",
                safetensors::Dtype::U32 => "U32",
                safetensors::Dtype::U8 => "U8",
                _ => "F32",
            };

            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype_str,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }

        let header_json = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let header_bytes = header_json.as_bytes();

        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(&data_buf);
        out
    }
}
