//! Shared loader primitives for the `*_native.rs` model modules.
//!
//! Each native-architecture file previously open-coded the same ~65 LOC
//! shard-discovery + bulk-load block (look for `model.safetensors`, fall
//! back to `model.safetensors.index.json`, read every shard into one
//! keyed map). Bug fixes to error handling had to land in four places.
//!
//! This module hosts the I/O-only portion of the pipeline: parsing
//! `config.json`, resolving shard paths, and merging shards into a
//! `HashMap<String, InlineArray>`. Model-specific weight sanitization,
//! key normalization, and per-layer slicing still live in each arch's
//! own `load_model`.
//!
//! Error-message shape is preserved byte-for-byte from the original
//! callers so tests asserting on `.to_string()` output keep working.

use crate::InlineArray;
use crate::compat::Dtype;
use crate::error::check_last_error;
use crate::inline_array as bridge;
use crate::native_weight::MlxQuantization;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Read `model_dir/config.json` as text.
///
/// Error format matches what the native loaders produced before
/// extraction: `"failed to read {path}: {io-error}"`.
pub fn read_config_json(model_dir: &Path) -> Result<String, String> {
    let path = model_dir.join("config.json");
    std::fs::read_to_string(&path).map_err(|e| format!("failed to read {}: {e}", path.display()))
}

/// The `quantization` block from a checkpoint's `config.json`, or `None` when
/// the weights are dense.
///
/// ⚠️ `mlx_lm.convert` writes the block **twice**, under `quantization` (MLX's
/// own key, which is where per-module overrides live) and `quantization_config`
/// (the HF key), with identical contents. MLX's key wins. HF's multimodal
/// exports also park it inside `text_config`, which is checked last.
///
/// `normalize` maps the checkpoint's module paths onto the caller's key
/// namespace; see [`MlxQuantization::from_json`].
pub fn load_mlx_quantization(
    model_dir: &Path,
    normalize: impl Fn(&str) -> Option<String>,
) -> Result<Option<MlxQuantization>, String> {
    let text = read_config_json(model_dir)?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("failed to parse config.json: {e}"))?;

    let block = ["quantization", "quantization_config"]
        .iter()
        .find_map(|key| {
            json.get(key)
                .or_else(|| json.get("text_config").and_then(|tc| tc.get(key)))
        });
    let Some(block) = block else {
        return Ok(None);
    };
    MlxQuantization::from_json(block, normalize)
        .map(Some)
        .ok_or_else(|| {
            format!(
                "unsupported quantization block in {}: {block}",
                model_dir.join("config.json").display()
            )
        })
}

/// Resolve the set of `.safetensors` shard paths under `model_dir`,
/// using the same fallback order as every pre-existing native loader:
///
/// 1. `model.safetensors` (single-file checkpoint).
/// 2. `model.safetensors.index.json` — parse `weight_map` and collect
///    unique shard filenames. Rejects entries containing `..` or a
///    leading `/` to prevent path traversal.
/// 3. Otherwise, returns an error that matches the legacy wording so
///    callers' user-facing messages do not regress.
pub fn discover_safetensors_shards(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let single_path = model_dir.join("model.safetensors");
    let index_path = model_dir.join("model.safetensors.index.json");

    if single_path.exists() {
        return Ok(vec![single_path]);
    }
    if !index_path.exists() {
        return Err(format!(
            "no model.safetensors or model.safetensors.index.json in {}",
            model_dir.display()
        ));
    }

    let content = std::fs::read_to_string(&index_path)
        .map_err(|e| format!("failed to read index JSON: {e}"))?;
    let index: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("failed to parse index JSON: {e}"))?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "index JSON missing weight_map".to_string())?;

    let mut seen = std::collections::HashSet::new();
    let mut paths = Vec::new();
    for shard_file in weight_map.values() {
        let name = shard_file
            .as_str()
            .ok_or_else(|| "shard filename is not a string".to_string())?;
        if seen.insert(name.to_string()) {
            if name.contains("..") || name.starts_with('/') {
                return Err(format!("shard filename contains path traversal: {name}"));
            }
            paths.push(model_dir.join(name));
        }
    }
    Ok(paths)
}

/// Load every tensor from each shard in `shard_paths`, merging into one
/// keyed map. On key collisions across shards, later shards win — this
/// matches the behavior of the previous inline loaders, which all used
/// `HashMap::insert` in shard-iteration order.
///
/// `model_dir` is only used in the "no weights loaded" error message.
pub fn load_shards_into_map(
    shard_paths: &[PathBuf],
    model_dir: &Path,
) -> Result<HashMap<String, InlineArray>, String> {
    let mut raw: HashMap<String, InlineArray> = HashMap::new();
    for shard_path in shard_paths {
        let path_str = shard_path
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 shard path: {:?}", shard_path))?;
        let entries = bridge::load_safetensors_shard(path_str)
            .ok_or_else(|| format!("failed to load shard: {path_str}"))?;
        for (key, arr) in entries {
            raw.insert(key, arr);
        }
    }
    if raw.is_empty() {
        return Err(format!("no weights loaded from {}", model_dir.display()));
    }
    Ok(raw)
}

/// Dequantize a safetensors FP8 E4M3 weight with its `*_scale_inv` tensor.
///
/// Most HF FP8 checkpoints store weights as E4M3 and scales as block-wise
/// inverse scales with shape `[ceil(out/128), ceil(in/128)]`. A few exporters
/// use scalar, row, column, or directly-broadcastable scales; those are handled
/// here too so native loaders can normalize all variants into ordinary dense
/// weights before building their hot-path structs.
pub fn dequantize_fp8_e4m3_scaled_weight(
    weight: &InlineArray,
    scale_inv: &InlineArray,
    target_dtype: i32,
) -> Result<InlineArray, String> {
    if weight.ndim() != 2 {
        return Err(format!(
            "FP8 dequant expected a 2D weight, got shape {:?}",
            weight.shape()
        ));
    }

    let w_f = fp8_e4m3_to_dtype(weight, target_dtype)?;
    let m = w_f.dim(0);
    let n = w_f.dim(1);
    let bs = 128i32;
    let block_m = div_ceil_i32(m, bs);
    let block_n = div_ceil_i32(n, bs);

    match scale_inv.ndim() {
        2 if scale_inv.dim(0) == block_m && scale_inv.dim(1) == block_n => {
            dequantize_fp8_e4m3_block_scaled(&w_f, scale_inv, target_dtype, block_m, block_n)
        }
        2 if broadcasts_to_2d(scale_inv.dim(0), scale_inv.dim(1), m, n) => {
            multiply_by_scale_inv(&w_f, scale_inv, None, target_dtype)
        }
        1 if scale_inv.dim(0) == m => {
            multiply_by_scale_inv(&w_f, scale_inv, Some(&[m, 1]), target_dtype)
        }
        1 if scale_inv.dim(0) == n => {
            multiply_by_scale_inv(&w_f, scale_inv, Some(&[1, n]), target_dtype)
        }
        1 if scale_inv.dim(0) == 1 => multiply_by_scale_inv(&w_f, scale_inv, None, target_dtype),
        0 => multiply_by_scale_inv(&w_f, scale_inv, None, target_dtype),
        _ => Err(format!(
            "unsupported FP8 scale_inv shape {:?} for weight shape {:?}",
            scale_inv.shape(),
            weight.shape()
        )),
    }
}

/// One NVIDIA ModelOpt quantized tensor, lifted out of a checkpoint.
///
/// ModelOpt writes two schemes, told apart by their sidecars:
///
/// - nvfp4: `weight` is `uint8` holding two e2m1 values per byte, low nibble
///   first, with an e4m3 `weight_scale` per 16 and an fp32 `weight_scale_2`.
///   Those bytes read as `uint32` are MLX's own nvfp4 layout.
/// - per-tensor fp8: `weight` is e4m3 with an fp32 `weight_scale`.
///
/// In both, the fp32 factor multiplies the whole dequantized tensor. `input_scale`
/// calibrates quantized activations, which pmetal does not do, and is dropped.
/// A checkpoint with `kv_cache_scheme` also ships `k_proj.k_scale` and
/// `v_proj.v_scale`, which calibrate an fp8 KV cache and touch no weight;
/// pmetal quantizes its cache itself, so loaders leave them unmatched.
pub enum ModelOptWeight {
    /// Packed in MLX's nvfp4 layout (`uint32`), its e4m3 scale per 16 values,
    /// and the per-tensor factor as an fp32 scalar.
    Nvfp4 {
        packed: InlineArray,
        scales: InlineArray,
        tensor_scale: InlineArray,
    },
    /// The e4m3 bytes as `uint8`, and the per-tensor factor as an fp32 scalar.
    Fp8 {
        weight: InlineArray,
        tensor_scale: InlineArray,
    },
}

impl ModelOptWeight {
    /// Unpack to a dense `[out, in]` weight in `dtype`.
    ///
    /// Rounds once. An fp4 value carries at most two significant bits and an
    /// e4m3 scale four, so their product is exact in bfloat16, and the only
    /// rounding is the fp32 multiply by the tensor scale.
    pub fn to_dense(&self, dtype: i32) -> Result<InlineArray, String> {
        let f32_dtype = Dtype::Float32.as_i32();
        let dense = match self {
            Self::Nvfp4 {
                packed,
                scales,
                tensor_scale,
            } => packed
                .dequantize_mode(scales, None, 16, 4, crate::QuantizedMode::Nvfp4)
                .as_dtype(f32_dtype)
                .multiply(tensor_scale),
            Self::Fp8 {
                weight,
                tensor_scale,
            } => fp8_e4m3_to_dtype(weight, f32_dtype)?.multiply(tensor_scale),
        };
        let dense = dense.as_dtype(dtype);
        check_last_error().map_err(|e| format!("ModelOpt dequantize: {e}"))?;
        Ok(dense)
    }
}

/// Lift every ModelOpt quantized tensor out of `raw`, keyed by module path.
///
/// Removes each one's `weight`, `weight_scale`, `weight_scale_2` and
/// `input_scale`, so what is left in `raw` is the checkpoint's dense tensors.
/// Shapes are checked here, once, for every loader.
pub fn take_modelopt_weights(
    raw: &mut HashMap<String, InlineArray>,
) -> Result<Vec<(String, ModelOptWeight)>, String> {
    const U8: i32 = 1;
    const U32: i32 = 3;
    let f32_dtype = Dtype::Float32.as_i32();

    let bases: Vec<String> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".weight_scale"))
        .map(ToOwned::to_owned)
        .collect();

    let mut taken = Vec::with_capacity(bases.len());
    for base in bases {
        let weight_key = format!("{base}.weight");
        let scale = raw
            .remove(&format!("{base}.weight_scale"))
            .ok_or_else(|| format!("ModelOpt sidecar disappeared during load: {base}"))?;
        let weight = raw
            .remove(&weight_key)
            .ok_or_else(|| format!("ModelOpt scale {base}.weight_scale has no weight"))?;
        raw.remove(&format!("{base}.input_scale"));

        if weight.dtype_raw() != U8 || weight.ndim() != 2 {
            return Err(format!(
                "{weight_key}: expected a 2D uint8 ModelOpt weight, got dtype {} shape {:?}",
                weight.dtype_raw(),
                weight.shape()
            ));
        }

        let entry = if let Some(scale_2) = raw.remove(&format!("{base}.weight_scale_2")) {
            // Two fp4 values per byte, one e4m3 scale per 16 values.
            let in_dim = weight.dim(1) * 2;
            if scale.dtype_raw() != U8 || scale.dim(1) * 16 != in_dim || in_dim % 8 != 0 {
                return Err(format!(
                    "{weight_key}: nvfp4 scales {:?} do not match weight {:?}",
                    scale.shape(),
                    weight.shape()
                ));
            }
            ModelOptWeight::Nvfp4 {
                packed: weight.view(U32),
                scales: scale,
                tensor_scale: scale_2.as_dtype(f32_dtype).reshape(&[]),
            }
        } else {
            if scale.size() != 1 {
                return Err(format!(
                    "{weight_key}: only per-tensor ModelOpt fp8 is supported, got weight_scale {:?}",
                    scale.shape()
                ));
            }
            ModelOptWeight::Fp8 {
                weight,
                tensor_scale: scale.as_dtype(f32_dtype).reshape(&[]),
            }
        };
        taken.push((base, entry));
    }
    Ok(taken)
}

/// The module a per-tensor quantization sidecar belongs to, or `None` when the
/// key is not one. `"model.layers.0.mlp.down_proj.weight_scale_2"` gives
/// `"model.layers.0.mlp.down_proj"`.
///
/// ⚠️ A loader that finishes with one of these still in hand has read its
/// weight as dense bytes. The model loads, runs, and decodes noise, so a
/// leftover has to be an error rather than an "unmatched key" warning.
pub fn quant_sidecar_base(key: &str) -> Option<&str> {
    [
        ".weight_scale",
        ".weight_scale_2",
        ".weight_scale_inv",
        ".input_scale",
    ]
    .iter()
    .find_map(|suffix| key.strip_suffix(suffix))
}

fn fp8_e4m3_to_dtype(weight: &InlineArray, target_dtype: i32) -> Result<InlineArray, String> {
    if weight.dtype_raw() == Dtype::Uint8.as_i32() {
        return weight
            .try_from_fp8(target_dtype)
            .map_err(|e| format!("failed to dequantize uint8 E4M3 payload: {e}"));
    }

    let out = weight.as_dtype(target_dtype);
    check_last_error().map_err(|e| format!("failed to cast FP8 E4M3 weight: {e}"))?;
    Ok(out)
}

fn dequantize_fp8_e4m3_block_scaled(
    weight: &InlineArray,
    scale_inv: &InlineArray,
    target_dtype: i32,
    block_m: i32,
    block_n: i32,
) -> Result<InlineArray, String> {
    let m = weight.dim(0);
    let n = weight.dim(1);
    let bs = 128i32;
    let m_pad = block_m * bs;
    let n_pad = block_n * bs;

    let w_padded = if m != m_pad || n != n_pad {
        let padded = InlineArray::zeros(&[m_pad, n_pad], target_dtype);
        let out = padded.slice_set(weight, &[0, 0], &[m, n]);
        check_last_error().map_err(|e| format!("failed to pad FP8 weight: {e}"))?;
        out
    } else {
        weight.clone()
    };

    let reshaped = w_padded
        .try_reshape(&[block_m, bs, block_n, bs])
        .map_err(|e| format!("failed to reshape FP8 weight blocks: {e}"))?;
    let scale = scale_inv
        .try_reshape(&[block_m, 1, block_n, 1])
        .map_err(|e| format!("failed to reshape FP8 scale_inv blocks: {e}"))?;
    let scale = cast_scale_inv(&scale, target_dtype)?;
    let scaled = reshaped.multiply(&scale);
    check_last_error().map_err(|e| format!("failed to apply FP8 scale_inv: {e}"))?;

    let back = scaled
        .try_reshape(&[m_pad, n_pad])
        .map_err(|e| format!("failed to restore FP8 dequantized weight shape: {e}"))?;
    if m != m_pad || n != n_pad {
        let out = back.slice(&[0, 0], &[m, n]);
        check_last_error().map_err(|e| format!("failed to unpad FP8 weight: {e}"))?;
        Ok(out)
    } else {
        Ok(back)
    }
}

fn multiply_by_scale_inv(
    weight: &InlineArray,
    scale_inv: &InlineArray,
    reshape: Option<&[i32]>,
    target_dtype: i32,
) -> Result<InlineArray, String> {
    let scale = if let Some(shape) = reshape {
        scale_inv
            .try_reshape(shape)
            .map_err(|e| format!("failed to reshape FP8 scale_inv: {e}"))?
    } else {
        scale_inv.clone()
    };
    let scale = cast_scale_inv(&scale, target_dtype)?;
    let out = weight.multiply(&scale);
    check_last_error().map_err(|e| format!("failed to apply FP8 scale_inv: {e}"))?;
    Ok(out)
}

fn cast_scale_inv(scale_inv: &InlineArray, target_dtype: i32) -> Result<InlineArray, String> {
    let out = scale_inv.as_dtype(target_dtype);
    check_last_error().map_err(|e| format!("failed to cast FP8 scale_inv: {e}"))?;
    Ok(out)
}

fn div_ceil_i32(value: i32, divisor: i32) -> i32 {
    (value + divisor - 1) / divisor
}

fn broadcasts_to_2d(scale_m: i32, scale_n: i32, weight_m: i32, weight_n: i32) -> bool {
    (scale_m == 1 || scale_m == weight_m) && (scale_n == 1 || scale_n == weight_n)
}

#[cfg(test)]
mod tests {
    use super::discover_safetensors_shards;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("pmetal-native-loader-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// macOS writes an AppleDouble sidecar named `._<file>` beside every file
    /// on a volume without native extended attributes, which is every external
    /// or network drive people keep checkpoints on. A directory glob picks
    /// those up and the loader dies on "Invalid json header length"; the index
    /// names only the real shards.
    #[test]
    fn apple_double_sidecars_are_not_mistaken_for_shards() {
        let dir = temp_dir();
        for name in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "._model-00001-of-00002.safetensors",
            "._model-00002-of-00002.safetensors",
        ] {
            fs::write(dir.join(name), b"not really a shard").unwrap();
        }
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors"
            }}"#,
        )
        .unwrap();

        let mut shards = discover_safetensors_shards(&dir).unwrap();
        shards.sort();
        let names: Vec<String> = shards
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ]
        );
        let _ = fs::remove_dir_all(dir);
    }
}
