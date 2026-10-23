//! Unpacking quantized checkpoints to dense tensors.
//!
//! The model crates' dense layers hold dense weights, so a quantized checkpoint
//! is unpacked as it is read. Three families ship, and each stores the scales
//! somewhere different:
//!
//! * **MLX** (`mlx-community` and every checkpoint MLX's converter writes):
//!   a quantized module's parameters are *siblings*, `mlp.up_proj.weight`
//!   (the packed words) beside `mlp.up_proj.scales` and, in affine mode,
//!   `mlp.up_proj.biases`. `config.json` carries the width, group size and mode
//!   and may override them per module; [`MlxQuantization`] is the one parser of
//!   that block, shared with the native engines.
//! * **pmetal's own quantizer**, which suffixes the whole tensor name instead:
//!   `mlp.up_proj.weight.scales`, with per-tensor widths in
//!   `per_tensor_overrides`.
//! * **Per-tensor sidecars** (NVIDIA ModelOpt nvfp4 and fp8, and block-scaled
//!   FP8): see [`dequantize_sidecar_weights`].

use std::collections::HashMap;
use std::path::Path;

use pmetal_runtime::QuantParams;
use pmetal_runtime::compat::Array;
use pmetal_runtime::native_weight::MlxQuantization;

use super::LoadError;

/// A checkpoint's quantization settings, as `config.json` states them.
#[derive(Debug, Clone)]
pub(crate) struct QuantConfig {
    mlx: MlxQuantization,
    /// pmetal's mixed-precision output records a width per *tensor* key.
    per_tensor_bits: HashMap<String, i32>,
}

impl QuantConfig {
    /// Read the quantization block of `model_dir`'s `config.json`, or `None`
    /// for a dense checkpoint.
    ///
    /// Through [`crate::dispatcher::config_value`], not `serde_json::from_str`:
    /// Python's `json.dump` emits bare `Infinity`, which is not JSON, and
    /// `nvidia/Nemotron-H-8B-Base-8K` ships `"time_step_limit": [0.0, Infinity]`.
    /// Reading it strictly here failed the whole load with a parse error far
    /// from anything this function cares about.
    pub(crate) fn read(model_dir: &Path) -> Result<Option<Self>, LoadError> {
        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&config_path)?;
        let json = crate::dispatcher::config_value(&raw)
            .map_err(|e| LoadError::SafeTensors(e.to_string()))?;
        Self::from_config(&json)
    }

    /// [`Self::read`] for a parsed config.
    pub(crate) fn from_config(config: &serde_json::Value) -> Result<Option<Self>, LoadError> {
        let Some(block) = pmetal_runtime::native_loader::quantization_block(config) else {
            return Ok(None);
        };
        let mlx =
            MlxQuantization::from_json(block, |path| Some(path.to_string())).ok_or_else(|| {
                LoadError::SafeTensors(format!(
                    "unsupported quantization block in config.json (an unknown `mode`, or a \
                     malformed per-module entry): {block}"
                ))
            })?;
        let per_tensor_bits = block
            .get("per_tensor_overrides")
            .and_then(|value| value.as_object())
            .map(|object| {
                object
                    .iter()
                    .filter_map(|(key, bits)| {
                        Some((key.clone(), i32::try_from(bits.as_i64()?).ok()?))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Some(Self {
            mlx,
            per_tensor_bits,
        }))
    }

    /// How to unpack the packed tensor `weight_key`, or `None` when the
    /// checkpoint says it was left dense.
    fn params_for(&self, weight_key: &str) -> Option<QuantParams> {
        let module = weight_key.strip_suffix(".weight").unwrap_or(weight_key);
        let from_block = self.mlx.params_for(module);
        match (self.per_tensor_bits.get(weight_key), from_block) {
            (Some(&bits), base) => Some(QuantParams {
                bits,
                ..base.or_else(|| self.mlx.default_params())?
            }),
            (None, params) => params,
        }
    }
}

fn aux_base(key: &str) -> Option<&str> {
    key.strip_suffix(".scales")
        .or_else(|| key.strip_suffix(".biases"))
}

/// Names under which `aux_key` may refer to its packed tensor, in preference
/// order.
///
/// ⚠️ The two layouts differ by one path segment. MLX stores `X.weight` beside
/// `X.scales`; pmetal's quantizer stores `X.weight` beside `X.weight.scales`.
/// Preferring the `.weight` child and falling back to the base name lands on the
/// right tensor either way. A module path is never itself a tensor, so the two
/// cases cannot both apply to one entry.
fn packed_candidates(base: &str) -> [String; 2] {
    [format!("{base}.weight"), base.to_string()]
}

/// Unpack every `X.scales` (and `X.biases`) triple in `weights` to a dense
/// `X.weight`, as `config` describes it, and drop the aux tensors.
///
/// Errors, rather than leaving packed words to be assigned to a dense layer,
/// when scales appear that the config cannot account for: no quantization
/// block, a module the block calls dense, a mode that disagrees with whether
/// biases are present, or scales with no packed tensor beside them. Any of
/// those would otherwise load as a model that decodes noise.
pub(crate) fn unpack_mlx_quantized(
    weights: &mut HashMap<String, Array>,
    config: Option<&QuantConfig>,
) -> Result<(), LoadError> {
    let scale_keys: Vec<String> = weights
        .keys()
        .filter(|key| key.ends_with(".scales"))
        .cloned()
        .collect();
    for scales_key in scale_keys {
        let Some(base) = scales_key.strip_suffix(".scales") else {
            continue;
        };
        let Some(weight_key) = packed_candidates(base)
            .into_iter()
            .find(|candidate| weights.contains_key(candidate))
        else {
            return Err(LoadError::SafeTensors(format!(
                "{scales_key} has no packed weight beside it ({base}.weight)"
            )));
        };
        let Some(config) = config else {
            return Err(LoadError::SafeTensors(format!(
                "{scales_key} is a quantization scale, but config.json has no `quantization` \
                 block saying how {weight_key} is packed"
            )));
        };
        let biases_key = format!("{base}.biases");
        let has_biases = weights.contains_key(&biases_key);
        let params = config
            .params_for(&weight_key)
            .ok_or_else(|| {
                LoadError::SafeTensors(format!(
                    "{weight_key} has scales, but config.json marks it as not quantized"
                ))
            })?
            .for_tensors(&weight_key, has_biases)
            .map_err(LoadError::SafeTensors)?;

        // Both the width *and* the group size can vary per tensor. Unpacking a
        // mixed-precision checkpoint at one model-wide setting produces
        // wrongly-shaped weights.
        let (Some(weight), Some(scales)) = (weights.get(&weight_key), weights.get(&scales_key))
        else {
            continue;
        };
        let dense = weight.dequantize_mode(
            scales,
            weights.get(&biases_key),
            params.group_size,
            params.bits,
            params.mode,
        );
        weights.insert(weight_key, dense);
        weights.remove(&scales_key);
        weights.remove(&biases_key);
    }
    Ok(())
}

/// Unpack every per-tensor-sidecar scheme to dense, as MLX's own format is
/// unpacked above: NVIDIA ModelOpt nvfp4 and fp8, and Qwen's block-scaled FP8.
///
/// ⚠️ Before this, their sidecars fell through as unmatched keys with a
/// warning, and the packed bytes were assigned to dense `Linear` weights. The
/// load reported every weight matched, and anything trained on it trained on
/// noise.
pub(crate) fn dequantize_sidecar_weights(
    weights: &mut HashMap<String, Array>,
) -> Result<(), LoadError> {
    use pmetal_runtime::native_loader as nl;

    if !weights
        .keys()
        .any(|key| nl::quant_sidecar_base(key).is_some())
    {
        return Ok(());
    }
    let dtype = pmetal_runtime::native_weight::detect_model_dtype(|key| {
        weights.get(key).map(|w| w.dtype_raw())
    });

    for (base, weight) in nl::take_modelopt_weights(weights).map_err(LoadError::SafeTensors)? {
        let dense = weight
            .to_dense(dtype)
            .map_err(|e| LoadError::SafeTensors(format!("{base}.weight: {e}")))?;
        weights.insert(format!("{base}.weight"), dense);
    }

    let block_fp8: Vec<String> = weights
        .keys()
        .filter_map(|key| key.strip_suffix(".weight_scale_inv"))
        .map(ToOwned::to_owned)
        .collect();
    for base in block_fp8 {
        let weight_key = format!("{base}.weight");
        let (Some(scale_inv), Some(weight)) = (
            weights.remove(&format!("{base}.weight_scale_inv")),
            weights.remove(&weight_key),
        ) else {
            return Err(LoadError::SafeTensors(format!(
                "FP8 scale {base}.weight_scale_inv has no weight"
            )));
        };
        let dense = nl::dequantize_fp8_e4m3_scaled_weight(&weight, &scale_inv, dtype)
            .map_err(|e| LoadError::SafeTensors(format!("{weight_key}: {e}")))?;
        weights.insert(weight_key, dense);
    }

    // An activation calibration with no weight scale beside it changes no
    // weight, so it is safe to drop. A weight scale is not.
    weights.retain(|key, _| !key.ends_with(".input_scale"));
    let leftover: Vec<&String> = weights
        .keys()
        .filter(|key| nl::quant_sidecar_base(key).is_some())
        .take(10)
        .collect();
    if !leftover.is_empty() {
        return Err(LoadError::SafeTensors(format!(
            "quantization scales this loader cannot apply, so their weights would load as raw bytes: {leftover:?}"
        )));
    }
    Ok(())
}

/// One ModelOpt-quantized `Linear`, lifted out of a checkpoint still packed:
/// its words, block scales and the per-tensor factor that multiplies the
/// product.
#[derive(Debug)]
pub(crate) struct PackedLinear {
    weight: Array,
    scales: Array,
    /// Affine quantization's per-group offsets.
    biases: Option<Array>,
    /// ModelOpt's per-tensor factor.
    tensor_scale: Option<Array>,
    params: QuantParams,
}

/// The quantized `Linear`s of a checkpoint (ModelOpt's, or MLX's), by module
/// path, and the
/// dtype the model computes in.
#[derive(Debug, Default)]
pub(crate) struct PackedLinears {
    modules: HashMap<String, PackedLinear>,
    dtype: Option<pmetal_runtime::compat::Dtype>,
}

impl PackedLinears {
    /// Whether nothing was lifted out.
    pub(crate) fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Take `other`'s modules too (they are of one checkpoint, so their paths
    /// differ); the dtype is `other`'s when this has none.
    pub(crate) fn absorb(&mut self, other: &mut PackedLinears) {
        self.modules.extend(std::mem::take(&mut other.modules));
        if self.dtype.is_none() {
            self.dtype = other.dtype;
        }
    }

    /// The module paths, for a loader to account for their `weight`s.
    pub(crate) fn modules(&self) -> impl Iterator<Item = &str> {
        self.modules.keys().map(String::as_str)
    }

    /// Rename the module paths (a loader that canonicalizes checkpoint keys
    /// has to canonicalize these the same way).
    pub(crate) fn rename(&mut self, f: impl Fn(&str) -> String) {
        self.modules = std::mem::take(&mut self.modules)
            .into_iter()
            .map(|(module, packed)| (f(&module), packed))
            .collect();
    }

    /// Put each packed weight on the `Linear` at its module path: the words in
    /// place of the dense weight, the scales and the per-tensor factor beside
    /// them, so the layer runs MLX's quantized matmul and an adapter on it is
    /// QLoRA. Fails if a module names no projection (or an ambiguous one), or
    /// the shapes disagree.
    pub(crate) fn install<M: pmetal_runtime::compat::VisitLinears + ?Sized>(
        mut self,
        model: &mut M,
    ) -> Result<usize, LoadError> {
        let wanted: Vec<String> = self.modules.keys().cloned().collect();
        let mut installed: HashMap<String, usize> = HashMap::new();
        let mut failure: Option<LoadError> = None;
        model.visit_linears_mut("", &mut |path, linear| {
            // The module a checkpoint names may sit under a namespace the
            // model's tree adds (`model.`) or drops.
            let Some(module) = wanted.iter().find(|m| {
                path == m.as_str()
                    || path.ends_with(&format!(".{m}"))
                    || m.ends_with(&format!(".{path}"))
            }) else {
                return;
            };
            *installed.entry(module.clone()).or_default() += 1;
            let Some(packed) = self.modules.remove(module) else {
                return; // a second match; reported below
            };
            let dtype = self
                .dtype
                .unwrap_or(pmetal_runtime::compat::Dtype::Bfloat16);
            let (out, input) = (
                packed.weight.dim(0),
                packed.weight.dim(1) * 32 / packed.params.bits,
            );
            // A layer built "for loading" is a 1x1 placeholder that takes the
            // shape of whatever the checkpoint puts on it.
            if linear.shape() != (out, input) && linear.shape() != (1, 1) {
                failure = Some(LoadError::ShapeMismatch {
                    key: format!("{path}.weight"),
                    expected: vec![linear.shape().0, linear.shape().1],
                    actual: vec![out, input],
                });
                return;
            }
            linear.weight.value = packed.weight;
            // The packed words are not the weight's dtype; the layer keeps
            // computing in the dtype its model does.
            linear.quant = Some(pmetal_runtime::compat::LinearQuant {
                scales: packed.scales,
                biases: packed.biases,
                params: packed.params,
                dtype,
                tensor_scale: packed.tensor_scale,
                codebook: None,
            });
        });
        if let Some(e) = failure {
            return Err(e);
        }
        // A quantized embedding table stays packed too: its lookups unpack the
        // rows they gather, and a tied head multiplies over the packed words.
        let dtype = self
            .dtype
            .unwrap_or(pmetal_runtime::compat::Dtype::Bfloat16);
        let mut failure: Option<LoadError> = None;
        model.visit_embeddings_mut("", &mut |path, embedding| {
            let Some(module) = wanted.iter().find(|m| names_linear(m, path)) else {
                return;
            };
            *installed.entry(module.clone()).or_default() += 1;
            let Some(packed) = self.modules.remove(module) else {
                return;
            };
            let (rows, dims) = (
                packed.weight.dim(0),
                packed.weight.dim(1) * 32 / packed.params.bits,
            );
            let shape = embedding.weight.value.shape().to_vec();
            if shape != [rows, dims] && shape != [1, 1] {
                failure = Some(LoadError::ShapeMismatch {
                    key: format!("{path}.weight"),
                    expected: shape.clone(),
                    actual: vec![rows, dims],
                });
                return;
            }
            embedding.weight.value = packed.weight;
            embedding.quant = Some(pmetal_runtime::compat::LinearQuant {
                scales: packed.scales,
                biases: packed.biases,
                params: packed.params,
                dtype,
                tensor_scale: packed.tensor_scale,
                codebook: None,
            });
        });
        if let Some(e) = failure {
            return Err(e);
        }
        let mut unplaced: Vec<&String> = wanted
            .iter()
            .filter(|m| installed.get(*m).copied().unwrap_or(0) != 1)
            .collect();
        unplaced.sort();
        if !unplaced.is_empty() {
            return Err(LoadError::SafeTensors(format!(
                "quantized modules that name no single projection of the model: {:?}",
                &unplaced[..unplaced.len().min(10)]
            )));
        }
        Ok(installed.len())
    }
}

/// Lift the ModelOpt-quantized projections out of `weights` and keep them
/// packed, where [`dequantize_sidecar_weights`] would unpack them to a dense
/// copy: an NVFP4 checkpoint of a 27B model is 15 GB packed and 54 GB dense.
///
/// nvfp4 is a bitcast of ModelOpt's bytes onto MLX's kernel, per-tensor fp8
/// runs on its mxfp8 kernel with unit block scales, and the per-tensor factor
/// multiplies the product (see `ModelOptWeight::into_native`). A projection
/// no packed kernel can tile, and the routed experts (stacked elsewhere), go
/// back into `weights` dense, as the shared dequantizer would leave them.
pub(crate) fn take_packed_linears(
    weights: &mut HashMap<String, Array>,
) -> Result<PackedLinears, LoadError> {
    use pmetal_runtime::native_loader::{NativeModelOpt, take_modelopt_weights};

    if !weights
        .keys()
        .any(|key| key.ends_with(".weight_scale") || key.ends_with(".weight_scale_2"))
    {
        return Ok(PackedLinears::default());
    }
    let dtype_code = pmetal_runtime::native_weight::detect_model_dtype(|key| {
        weights.get(key).map(|w| w.dtype_raw())
    });
    let dtype = pmetal_runtime::compat::Dtype::from_raw(dtype_code);
    let mut out = PackedLinears {
        modules: HashMap::new(),
        dtype: Some(dtype),
    };
    // The routed experts are stacked by the loader that owns them: their
    // tensors wait here, untouched, while the projections are lifted out.
    const EXPERT_SUFFIXES: [&str; 4] = [
        ".weight",
        ".weight_scale",
        ".weight_scale_2",
        ".input_scale",
    ];
    let experts: Vec<String> = weights
        .keys()
        .filter(|key| key.contains(".experts.") && EXPERT_SUFFIXES.iter().any(|s| key.ends_with(s)))
        .cloned()
        .collect();
    let held: Vec<(String, Array)> = experts
        .into_iter()
        .filter_map(|key| weights.remove(&key).map(|value| (key, value)))
        .collect();
    let taken = take_modelopt_weights(weights).map_err(LoadError::SafeTensors);
    weights.extend(held);
    for (base, weight) in taken? {
        match weight
            .into_native(dtype_code)
            .map_err(|e| LoadError::SafeTensors(format!("{base}.weight: {e}")))?
        {
            NativeModelOpt::Packed {
                weight,
                scales,
                tensor_scale,
                params,
            } => {
                out.modules.insert(
                    base,
                    PackedLinear {
                        weight,
                        scales,
                        biases: None,
                        tensor_scale: Some(tensor_scale),
                        params,
                    },
                );
            }
            NativeModelOpt::Dense(dense) => {
                weights.insert(format!("{base}.weight"), dense);
            }
        }
    }
    Ok(out)
}

/// Whether the checkpoint module `module` is the `Linear` at model path `path`:
/// the same rule [`PackedLinears::install`] places it by, since the module a
/// checkpoint names may sit under a namespace the model's tree adds or drops.
fn names_linear(module: &str, path: &str) -> bool {
    path == module || path.ends_with(&format!(".{module}")) || module.ends_with(&format!(".{path}"))
}

/// Lift the MLX-quantized projections that land on one of the model's
/// `Linear`s (`linear_paths`) out of `weights`, still packed, where
/// [`unpack_mlx_quantized`] would unpack them to a dense copy.
///
/// A decode step reads every weight once, so a 4-bit model unpacked to bf16
/// reads four times the bytes, and decodes at a quarter of the speed. The
/// words, scales and (affine) biases go on the `Linear` as they are
/// ([`PackedLinears::install`]); MLX's quantized matmul reads them. What is
/// not a `Linear` of the model (an embedding, stacked experts, a module the
/// config calls dense) is left for [`unpack_mlx_quantized`] as before.
pub(crate) fn take_packed_mlx_linears(
    weights: &mut HashMap<String, Array>,
    config: Option<&QuantConfig>,
    linear_paths: &[String],
) -> Result<PackedLinears, LoadError> {
    let mut out = PackedLinears::default();
    let Some(config) = config else {
        return Ok(out);
    };
    let scale_keys: Vec<String> = weights
        .keys()
        .filter(|key| key.ends_with(".scales"))
        .cloned()
        .collect();
    let mut dtype_hint = None;
    for scales_key in scale_keys {
        let Some(module) = scales_key.strip_suffix(".scales") else {
            continue;
        };
        // MLX's layout only: `X.weight` beside `X.scales`.
        let weight_key = format!("{module}.weight");
        if !weights.contains_key(&weight_key)
            || !linear_paths.iter().any(|path| names_linear(module, path))
        {
            continue;
        }
        let biases_key = format!("{module}.biases");
        let has_biases = weights.contains_key(&biases_key);
        let Some(params) = config.params_for(&weight_key) else {
            continue; // the config says it was left dense: the unpacker errors
        };
        let params = params
            .for_tensors(&weight_key, has_biases)
            .map_err(LoadError::SafeTensors)?;
        let (Some(weight), Some(scales)) =
            (weights.remove(&weight_key), weights.remove(&scales_key))
        else {
            continue;
        };
        // Affine scales are in the dtype the model computes in.
        if params.mode == pmetal_runtime::QuantizedMode::Affine {
            dtype_hint.get_or_insert(scales.dtype_raw());
        }
        out.modules.insert(
            module.to_string(),
            PackedLinear {
                weight,
                scales,
                biases: weights.remove(&biases_key),
                tensor_scale: None,
                params,
            },
        );
    }
    if !out.is_empty() {
        let code = dtype_hint.unwrap_or_else(|| {
            pmetal_runtime::native_weight::detect_model_dtype(|key| {
                weights.get(key).map(|w| w.dtype_raw())
            })
        });
        out.dtype = Some(pmetal_runtime::compat::Dtype::from_raw(code));
    }
    Ok(out)
}

/// Whether `key`, or the tensor it belongs to, is one `keep` wants: a packed
/// tensor's scales and biases have to come along with it, or it arrives still
/// packed. The packed tensor may be named either way (see
/// [`packed_candidates`]), so both are asked about.
pub(super) fn wanted_with_aux(key: &str, keep: &mut impl FnMut(&str) -> bool) -> bool {
    keep(key)
        || aux_base(key).is_some_and(|base| {
            packed_candidates(base)
                .iter()
                .any(|candidate| keep(candidate))
        })
        || pmetal_runtime::native_loader::quant_sidecar_base(key)
            .is_some_and(|base| keep(&format!("{base}.weight")))
}
