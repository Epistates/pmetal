//! Projection and embedding weights that may be stored packed.
//!
//! MLX models quantization as a property of the *layer*, not of the
//! architecture: `nn.QuantizedLinear` and `nn.QuantizedEmbedding` are drop-in
//! replacements, and a model calls `embed_tokens.as_linear(x)` without knowing
//! which it holds. That is why mlx-lm's architectures contain no `if
//! quantized` branch anywhere.
//!
//! The native engines are hand-rolled rather than module-based, so the
//! equivalent is these two enums plus their dispatchers. An architecture asks
//! for `matmul_from` or `lookup` and gets the right kernel.
//!
//! Layout note: MLX stores a quantized weight row-major as
//! `[out, in / (32 / bits)]` and expects `transpose = true` to signal that it
//! logically needs transposing. Dense weights are pre-transposed to `[in, out]`
//! at load time instead, so the per-step matmul is contiguous. The two arrive
//! at the same product by different routes, which is why only the dense arm
//! calls `.t()` at load.

use crate::{InlineArray, QuantizedMode};

/// How a packed tensor was quantized.
///
/// Mirrors the trailing arguments of `mlx::core::quantized_matmul`, so a
/// checkpoint's `quantization` block maps onto it field for field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantParams {
    pub group_size: i32,
    pub bits: i32,
    pub mode: QuantizedMode,
}

impl QuantParams {
    /// MLX's defaults for a mode, used when a config names the mode but omits
    /// the widths. Taken from `_defaults_for_mode` in `mlx/nn/layers/quantized.py`.
    pub fn defaults_for(mode: QuantizedMode) -> Self {
        let (group_size, bits) = match mode {
            QuantizedMode::Affine => (64, 4),
            QuantizedMode::Mxfp4 => (32, 4),
            QuantizedMode::Nvfp4 => (16, 4),
            QuantizedMode::Mxfp8 => (32, 8),
        };
        Self {
            group_size,
            bits,
            mode,
        }
    }
}

/// A checkpoint's `quantization` block, read the way mlx-lm reads it.
///
/// Module paths sit as *sibling* keys beside `group_size` / `bits` / `mode`,
/// each holding either an override object or `false`:
///
/// ```json
/// "quantization": {
///   "group_size": 64,
///   "bits": 4,
///   "mode": "affine",
///   "language_model.model.layers.0.mlp.gate_proj": {"group_size": 64, "bits": 8}
/// }
/// ```
///
/// Gemma 4's QAT release ships 144 of those overrides, putting every MLP
/// projection at 8 bits and leaving attention at 4.
#[derive(Clone, Debug)]
pub struct MlxQuantization {
    /// How to read a module the block never names.
    ///
    /// `Some` for MLX's own format, where the file-level `bits`/`group_size`
    /// apply to everything not overridden. `None` for NVIDIA ModelOpt, which
    /// names every quantized module explicitly and leaves the rest dense.
    default: Option<QuantParams>,
    overrides: std::collections::HashMap<String, Option<QuantParams>>,
}

impl MlxQuantization {
    /// Parse the object a checkpoint stores under `quantization`.
    ///
    /// `normalize` maps a checkpoint's own module path onto whatever
    /// namespace the caller's weight keys use, and returns `None` for paths
    /// the caller does not load (vision and audio towers). Pass
    /// `|p| Some(p.to_string())` when the two namespaces already agree.
    ///
    /// Returns `None` when `mode` names a scheme this build cannot decode,
    /// which is worth rejecting outright: reading mxfp4 as affine produces a
    /// model that loads and decodes noise.
    pub fn from_json(
        block: &serde_json::Value,
        normalize: impl Fn(&str) -> Option<String>,
    ) -> Option<Self> {
        let map = block.as_object()?;
        // NVIDIA ModelOpt / compressed-tensors states its own schema.
        if map.contains_key("config_groups") {
            return Self::from_modelopt(map, normalize);
        }
        let default = Some(Self::params_from(map)?);

        let mut overrides = std::collections::HashMap::new();
        for (key, value) in map {
            if matches!(key.as_str(), "group_size" | "bits" | "mode") {
                continue;
            }
            let Some(path) = normalize(key) else {
                continue;
            };
            match value {
                // `to_quantized(**params)`: an override object replaces the
                // defaults outright rather than merging into them.
                serde_json::Value::Object(obj) => {
                    overrides.insert(path, Some(Self::params_from(obj)?));
                }
                // The predicate returned a bare bool, so mlx passes the
                // file-level arguments through unchanged.
                serde_json::Value::Bool(true) => {
                    overrides.insert(path, default);
                }
                serde_json::Value::Bool(false) => {
                    overrides.insert(path, None);
                }
                // Anything else is a key that is not a module override, e.g.
                // `quant_method` on a legacy HF block.
                _ => continue,
            }
        }
        Some(Self { default, overrides })
    }

    /// Parse NVIDIA ModelOpt's `config_groups` schema.
    ///
    /// Unlike MLX's block, this names every quantized module explicitly in a
    /// group's `targets` list and leaves everything else dense, so there is no
    /// file-level default to fall back on:
    ///
    /// ```json
    /// "quantization_config": {
    ///   "quant_method": "modelopt",
    ///   "config_groups": {
    ///     "group_0": {"weights": {"num_bits": 8, "type": "float"},
    ///                 "targets": ["model.layers.0.self_attn.q_proj", ...]},
    ///     "group_1": {"weights": {"num_bits": 4, "type": "float", "group_size": 16},
    ///                 "targets": ["model.layers.0.mlp.down_proj", ...]}
    ///   },
    ///   "ignore": ["mtp*"]
    /// }
    /// ```
    ///
    /// `input_activations` is ignored: those describe a quantized activation,
    /// and reading the weights alone is valid, just not the whole speedup.
    fn from_modelopt(
        map: &serde_json::Map<String, serde_json::Value>,
        normalize: impl Fn(&str) -> Option<String>,
    ) -> Option<Self> {
        let groups = map.get("config_groups")?.as_object()?;
        let mut overrides = std::collections::HashMap::new();

        for group in groups.values() {
            let group = group.as_object()?;
            // `None` here is a scheme with no MLX mode, per-tensor fp8 being
            // the one that ships. Its targets are recorded as dense so they
            // take the fp8 path rather than failing the whole checkpoint.
            let params = Self::params_from_modelopt(group.get("weights")?.as_object()?);
            let targets = group.get("targets").and_then(|t| t.as_array());
            for target in targets.into_iter().flatten() {
                let Some(name) = target.as_str() else {
                    continue;
                };
                if let Some(path) = normalize(name) {
                    overrides.insert(path, params);
                }
            }
        }

        // `ignore` wins over any group that also named the module.
        for pattern in map
            .get("ignore")
            .and_then(|i| i.as_array())
            .into_iter()
            .flatten()
        {
            let Some(pattern) = pattern.as_str() else {
                continue;
            };
            let prefix = pattern.trim_end_matches('*');
            let matched: Vec<String> = overrides
                .keys()
                .filter(|path| path.starts_with(prefix))
                .cloned()
                .collect();
            for path in matched {
                overrides.insert(path, None);
            }
        }

        Some(Self {
            default: None,
            overrides,
        })
    }

    /// One group's weight spec, in ModelOpt's vocabulary.
    ///
    /// ⚠️ A `num_bits: 8, type: float` group with no `group_size` is
    /// per-tensor fp8, which is not one of MLX's block-scaled modes. It has no
    /// `QuantParams` spelling, so it returns `None` here and the loader keeps
    /// those tensors on the dense fp8 path.
    fn params_from_modelopt(
        weights: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<QuantParams> {
        let bits = weights.get("num_bits")?.as_i64()? as i32;
        let kind = weights
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("int");
        let group_size = weights
            .get("group_size")
            .and_then(|g| g.as_i64())
            .map(|g| g as i32);

        let mode = match (kind, bits, group_size) {
            ("float", 4, Some(16)) => QuantizedMode::Nvfp4,
            ("float", 4, Some(32)) => QuantizedMode::Mxfp4,
            ("float", 8, Some(32)) => QuantizedMode::Mxfp8,
            // Per-tensor fp8, or an integer scheme this build cannot express.
            ("float", _, _) => return None,
            _ => QuantizedMode::Affine,
        };
        Some(QuantParams {
            group_size: group_size.unwrap_or(QuantParams::defaults_for(mode).group_size),
            bits,
            mode,
        })
    }

    /// One `{group_size, bits, mode}` triple, defaulting the way
    /// `Linear.to_quantized` does.
    ///
    /// ⚠️ `mode` defaults to affine even inside an override on an mxfp4
    /// checkpoint, because mlx spreads the override dict over a signature
    /// whose own default is affine. `group_size` and `bits` then fall back to
    /// that mode's defaults, not to the file-level ones.
    fn params_from(map: &serde_json::Map<String, serde_json::Value>) -> Option<QuantParams> {
        let mode = match map.get("mode").and_then(|m| m.as_str()) {
            Some(name) => QuantizedMode::from_config_name(name)?,
            None => QuantizedMode::Affine,
        };
        let defaults = QuantParams::defaults_for(mode);
        Some(QuantParams {
            group_size: map
                .get("group_size")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(defaults.group_size),
            bits: map
                .get("bits")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(defaults.bits),
            mode,
        })
    }

    /// The parameters for one module, or `None` when the checkpoint says to
    /// leave it dense.
    ///
    /// A module the block never names still gets the file-level parameters:
    /// upstream decides those cases on whether `{path}.scales` is in the
    /// weights, which is the same question [`LayerWeight::new`] answers from
    /// its `scales` argument. So the real switch is the tensor, and this only
    /// says how to read it.
    pub fn params_for(&self, module_path: &str) -> Option<QuantParams> {
        match self.overrides.get(module_path) {
            Some(entry) => *entry,
            None => self.default,
        }
    }

    /// The file-level parameters, for tensors that are not per-module, such as
    /// pre-stacked MoE expert weights.
    ///
    /// `None` for a block that names every quantized module rather than
    /// declaring a default, which is how NVIDIA ModelOpt writes it.
    pub fn default_params(&self) -> Option<QuantParams> {
        self.default
    }
}

/// A projection weight, dense or packed.
///
/// One `Quantized` variant covers every mode: `biases` is `None` for the
/// floating-point modes (mxfp4, nvfp4, mxfp8), which carry scales only. That
/// matches `mlx::core::quantized_matmul`, where `biases` is
/// `std::optional<array>`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum LayerWeight {
    /// Pre-transposed to `[in, out]`; used with `x.matmul(w)`.
    Dense(InlineArray),
    /// Packed `uint32` as the checkpoint stores it, with `transpose = true`.
    Quantized {
        weight: InlineArray,
        scales: InlineArray,
        biases: Option<InlineArray>,
        params: QuantParams,
    },
}

impl LayerWeight {
    /// Build from the three tensors a checkpoint stores for one module.
    ///
    /// `scales` absent means the module was left dense, which is how a
    /// mixed-precision checkpoint marks the layers it did not quantize.
    pub fn new(
        weight: InlineArray,
        scales: Option<InlineArray>,
        biases: Option<InlineArray>,
        params: QuantParams,
    ) -> Self {
        match scales {
            Some(scales) => Self::Quantized {
                weight,
                scales,
                biases,
                params,
            },
            // Only the dense arm transposes: see the layout note above.
            None => Self::Dense(weight.t()),
        }
    }

    /// The packed or dense payload, for pointer export and buffer refresh.
    pub fn weight_arr(&self) -> &InlineArray {
        match self {
            Self::Dense(w) => w,
            Self::Quantized { weight, .. } => weight,
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// The quantization parameters, or `None` when dense.
    pub fn params(&self) -> Option<QuantParams> {
        match self {
            Self::Dense(_) => None,
            Self::Quantized { params, .. } => Some(*params),
        }
    }

    /// `x @ self`.
    #[inline(always)]
    pub fn matmul_from(&self, x: &InlineArray) -> InlineArray {
        match self {
            Self::Dense(w) => x.matmul(w),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => x.quantized_matmul_mode(
                weight,
                scales,
                biases.as_ref(),
                true,
                params.group_size,
                params.bits,
                params.mode,
            ),
        }
    }

    /// Gather-matmul for MoE expert dispatch.
    ///
    /// Every expert tensor in a layer has to be the same variant; a checkpoint
    /// that quantized some experts and not others is not something MLX's
    /// gather kernels can express.
    #[inline(always)]
    pub fn gather_mm_from(
        &self,
        x: &InlineArray,
        lhs_indices: Option<&InlineArray>,
        rhs_indices: Option<&InlineArray>,
        sorted: bool,
    ) -> InlineArray {
        match self {
            Self::Dense(w) => x.gather_mm(w, lhs_indices, rhs_indices, sorted),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => x.gather_qmm_mode(
                weight,
                scales,
                biases.as_ref(),
                lhs_indices,
                rhs_indices,
                true,
                params.group_size,
                params.bits,
                sorted,
                params.mode,
            ),
        }
    }

    /// The C-ABI view a compiled block takes.
    ///
    /// The result borrows this weight, so it stays valid for exactly as long
    /// as the arrays it points at.
    pub(crate) fn as_raw(&self) -> crate::inline_array::QWeightRaw<'_> {
        use crate::inline_array::QWeightRaw;
        match self {
            Self::Dense(w) => QWeightRaw {
                weight: &w.raw,
                scales: std::ptr::null(),
                biases: std::ptr::null(),
                group_size: 0,
                bits: 0,
                mode: 0,
                _owner: std::marker::PhantomData,
            },
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => QWeightRaw {
                weight: &weight.raw,
                scales: &scales.raw,
                biases: biases.as_ref().map_or(std::ptr::null(), |b| &b.raw),
                group_size: params.group_size,
                bits: params.bits,
                mode: params.mode.as_i32(),
                _owner: std::marker::PhantomData,
            },
        }
    }

    /// Re-materialise every array into a fresh buffer.
    pub fn copy_fresh(&self, zero: &InlineArray) -> Self {
        match self {
            Self::Dense(w) => Self::Dense(copy_fresh_arr(w, zero)),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => Self::Quantized {
                weight: copy_fresh_arr(weight, zero),
                scales: copy_fresh_arr(scales, zero),
                biases: biases.as_ref().map(|b| copy_fresh_arr(b, zero)),
                params: *params,
            },
        }
    }

    /// Queue every array for evaluation without blocking.
    pub fn async_eval_ref(&self) {
        match self {
            Self::Dense(w) => w.async_eval_ref(),
            Self::Quantized {
                weight,
                scales,
                biases,
                ..
            } => {
                weight.async_eval_ref();
                scales.async_eval_ref();
                if let Some(b) = biases {
                    b.async_eval_ref();
                }
            }
        }
    }
}

/// An embedding table, dense or packed.
///
/// The two methods are `mlx.nn.Embedding`'s: a lookup, and `as_linear` for the
/// tied LM head. A quantized lookup gathers the packed rows *and* their scales
/// and biases, then dequantizes only those rows, which is why it cannot go
/// through `LayerWeight`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum EmbeddingWeight {
    /// `[vocab, hidden]`.
    Dense(InlineArray),
    Quantized {
        weight: InlineArray,
        scales: InlineArray,
        biases: Option<InlineArray>,
        params: QuantParams,
    },
}

impl EmbeddingWeight {
    pub fn new(
        weight: InlineArray,
        scales: Option<InlineArray>,
        biases: Option<InlineArray>,
        params: QuantParams,
    ) -> Self {
        match scales {
            Some(scales) => Self::Quantized {
                weight,
                scales,
                biases,
                params,
            },
            // No transpose: an embedding is indexed by row, and `as_linear`
            // takes the transpose at the point of use.
            None => Self::Dense(weight),
        }
    }

    pub fn weight_arr(&self) -> &InlineArray {
        match self {
            Self::Dense(w) => w,
            Self::Quantized { weight, .. } => weight,
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// Row lookup: `[.., T] -> [.., T, hidden]`.
    ///
    /// The quantized arm is `mlx.nn.QuantizedEmbedding.__call__`:
    /// `dequantize(w[x], scales[x], biases[x])`. Gathering first means only the
    /// rows this batch touches are unpacked, rather than the whole table.
    pub fn lookup(&self, ids: &InlineArray) -> InlineArray {
        match self {
            Self::Dense(w) => w.take_axis(ids, 0),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => {
                let rows = weight.take_axis(ids, 0);
                let row_scales = scales.take_axis(ids, 0);
                let row_biases = biases.as_ref().map(|b| b.take_axis(ids, 0));
                rows.dequantize_mode(
                    &row_scales,
                    row_biases.as_ref(),
                    params.group_size,
                    params.bits,
                    params.mode,
                )
            }
        }
    }

    /// The table used as the output projection, for a model with tied weights.
    ///
    /// `mlx.nn.QuantizedEmbedding.as_linear`.
    pub fn as_linear(&self, x: &InlineArray) -> InlineArray {
        match self {
            Self::Dense(w) => x.matmul(&w.t()),
            Self::Quantized {
                weight,
                scales,
                biases,
                params,
            } => x.quantized_matmul_mode(
                weight,
                scales,
                biases.as_ref(),
                true,
                params.group_size,
                params.bits,
                params.mode,
            ),
        }
    }

    pub fn async_eval_ref(&self) {
        match self {
            Self::Dense(w) => w.async_eval_ref(),
            Self::Quantized {
                weight,
                scales,
                biases,
                ..
            } => {
                weight.async_eval_ref();
                scales.async_eval_ref();
                if let Some(b) = biases {
                    b.async_eval_ref();
                }
            }
        }
    }
}

/// The dtype a checkpoint's trunk computes in.
///
/// ⚠️ **Not the embedding's.** On a quantized checkpoint `embed_tokens.weight`
/// is the packed `uint32` payload, so reading its dtype makes every scalar
/// derived from it, and the whole KV cache, an integer type. The model then
/// loads, runs at the right speed, and decodes noise.
///
/// Norm weights are one-dimensional and no quantizer touches them, so they
/// carry the real compute dtype. Scales are the second choice: they are float
/// by construction and present exactly when the weights are packed.
pub fn detect_model_dtype(lookup: impl Fn(&str) -> Option<i32>) -> i32 {
    const CANDIDATES: [&str; 6] = [
        "model.norm.weight",
        "backbone.norm_f.weight",
        "language_model.model.norm.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    ];
    CANDIDATES
        .iter()
        .filter_map(|key| lookup(key))
        .find(|dt| is_float_dtype(*dt))
        .unwrap_or(crate::dtype::BF16)
}

/// Whether a raw dtype code is one the trunk can compute in.
fn is_float_dtype(dt: i32) -> bool {
    use crate::dtype;
    matches!(dt, dtype::F16 | dtype::F32 | dtype::BF16)
}

/// Force an array into a fresh Metal buffer (add zero, eval, detach).
pub fn copy_fresh_arr(w: &InlineArray, _hint_zero: &InlineArray) -> InlineArray {
    // A packed weight is already `uint32` in a buffer of its own, and adding
    // zero to it would be an integer op on the payload rather than a copy.
    const UINT32: i32 = 3;
    let dt = w.dtype_raw();
    if dt == UINT32 {
        let mut fresh = w.clone();
        fresh.eval();
        fresh.detach();
        return fresh;
    }
    let own_zero = InlineArray::zeros(&[1], dt);
    let mut fresh = w.add(&own_zero);
    fresh.eval();
    fresh.detach();
    fresh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;
    use std::collections::HashMap;

    fn lookup<'a>(map: &'a HashMap<&'a str, i32>) -> impl Fn(&str) -> Option<i32> + 'a {
        move |key: &str| map.get(key).copied()
    }

    /// ⚠️ The regression this exists for: a quantized checkpoint's
    /// `embed_tokens.weight` is packed `uint32`, and taking the model dtype
    /// from it made every derived scalar and the KV cache integer. The model
    /// loaded, ran at the right speed, and decoded noise.
    #[test]
    fn a_packed_embedding_does_not_set_the_model_dtype() {
        let map = HashMap::from([
            ("model.embed_tokens.weight", dtype::U32),
            ("model.embed_tokens.scales", dtype::BF16),
            ("model.norm.weight", dtype::BF16),
        ]);
        assert_eq!(detect_model_dtype(lookup(&map)), dtype::BF16);
    }

    /// With no norm weight to read, the scales still carry a float dtype.
    #[test]
    fn scales_answer_when_no_norm_weight_is_present() {
        let map = HashMap::from([
            ("model.embed_tokens.weight", dtype::U32),
            ("model.embed_tokens.scales", dtype::F16),
        ]);
        assert_eq!(detect_model_dtype(lookup(&map)), dtype::F16);
    }

    #[test]
    fn a_dense_checkpoint_reports_its_own_dtype() {
        let map = HashMap::from([
            ("model.norm.weight", dtype::F32),
            ("model.embed_tokens.weight", dtype::F32),
        ]);
        assert_eq!(detect_model_dtype(lookup(&map)), dtype::F32);
    }

    /// A checkpoint whose keys are all unfamiliar still has to produce a dtype
    /// the trunk can compute in, rather than whatever integer came first.
    #[test]
    fn an_unrecognised_layout_falls_back_to_bfloat16() {
        let map = HashMap::from([("something.else.weight", dtype::U32)]);
        assert_eq!(detect_model_dtype(lookup(&map)), dtype::BF16);
    }

    fn identity(path: &str) -> Option<String> {
        Some(path.to_string())
    }

    /// The block `mlx-community/gemma-4-12B-it-qat-4bit` actually ships: 4-bit
    /// affine everywhere, with every MLP projection overridden to 8 bits.
    #[test]
    fn per_module_overrides_sit_beside_the_file_level_keys() {
        let block = serde_json::json!({
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "language_model.model.layers.0.mlp.gate_proj": {"group_size": 64, "bits": 8},
            "language_model.model.layers.0.mlp.down_proj": {"group_size": 64, "bits": 8},
        });
        let q = MlxQuantization::from_json(&block, identity).expect("block parses");

        assert_eq!(
            q.params_for("language_model.model.layers.0.mlp.gate_proj")
                .unwrap()
                .bits,
            8
        );
        // Not named, so it takes the file-level parameters.
        assert_eq!(
            q.params_for("language_model.model.layers.0.self_attn.q_proj")
                .unwrap()
                .bits,
            4
        );
    }

    /// The loader normalises `language_model.model.*` onto `model.*`, so the
    /// override keys have to make the same trip or every lookup misses and the
    /// 8-bit MLP tensors get decoded as 4-bit.
    #[test]
    fn override_keys_are_normalised_with_the_weight_keys() {
        let block = serde_json::json!({
            "group_size": 64,
            "bits": 4,
            "language_model.model.layers.3.mlp.up_proj": {"group_size": 64, "bits": 8},
            "vision_tower.encoder.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 8},
        });
        let q = MlxQuantization::from_json(&block, |p| {
            crate::gemma4_native::normalize_checkpoint_key(p)
        })
        .expect("block parses");

        assert_eq!(q.params_for("model.layers.3.mlp.up_proj").unwrap().bits, 8);
        // The tower entry is dropped rather than carried under its own name.
        assert_eq!(
            q.params_for("vision_tower.encoder.layers.0.self_attn.q_proj")
                .unwrap()
                .bits,
            4
        );
    }

    /// `false` is how a block says "this module stayed dense".
    #[test]
    fn a_false_override_leaves_the_module_dense() {
        let block = serde_json::json!({
            "group_size": 64,
            "bits": 4,
            "model.embed_tokens": false,
        });
        let q = MlxQuantization::from_json(&block, identity).expect("block parses");
        assert_eq!(q.params_for("model.embed_tokens"), None);
    }

    /// ⚠️ `Linear.to_quantized(group_size=None, bits=None, mode="affine")`:
    /// mlx spreads an override dict over a signature whose own default is
    /// affine, so an override that omits `mode` is affine even when the file
    /// is mxfp4, and its widths come from affine's defaults.
    #[test]
    fn an_override_replaces_the_file_level_params_rather_than_merging() {
        let block = serde_json::json!({
            "group_size": 32,
            "bits": 4,
            "mode": "mxfp4",
            "model.layers.0.mlp.down_proj": {"bits": 8},
        });
        let q = MlxQuantization::from_json(&block, identity).expect("block parses");

        assert_eq!(q.default_params().unwrap().mode, QuantizedMode::Mxfp4);
        let overridden = q.params_for("model.layers.0.mlp.down_proj").unwrap();
        assert_eq!(overridden.mode, QuantizedMode::Affine);
        assert_eq!(overridden.bits, 8);
        assert_eq!(
            overridden.group_size, 64,
            "affine's default, not the file's 32"
        );
    }

    /// The shape `nvidia/Qwen3.8-27B-NVFP4` ships: two groups, one per-tensor
    /// fp8 and one nvfp4 at group size 16, every quantized module named
    /// explicitly, and the MTP draft layers left out via `ignore`.
    #[test]
    fn modelopt_groups_name_every_quantized_module() {
        let block = serde_json::json!({
            "quant_method": "modelopt",
            "config_groups": {
                "group_0": {
                    "weights": {"dynamic": false, "num_bits": 8, "type": "float"},
                    "input_activations": {"dynamic": false, "num_bits": 8, "type": "float"},
                    "targets": ["model.layers.0.self_attn.q_proj"]
                },
                "group_1": {
                    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["model.layers.0.mlp.down_proj", "mtp.layers.0.mlp.down_proj"]
                }
            },
            "ignore": ["mtp*"]
        });
        let q = MlxQuantization::from_json(&block, identity).expect("modelopt block parses");

        // group_1 is nvfp4: 4-bit float at group size 16.
        let down = q.params_for("model.layers.0.mlp.down_proj").unwrap();
        assert_eq!(down.mode, QuantizedMode::Nvfp4);
        assert_eq!((down.group_size, down.bits), (16, 4));

        // Per-tensor fp8 has no MLX mode, so it stays off the packed path.
        assert_eq!(q.params_for("model.layers.0.self_attn.q_proj"), None);

        // `ignore` wins over the group that also named the module.
        assert_eq!(q.params_for("mtp.layers.0.mlp.down_proj"), None);

        // Nothing is quantized by default: a module the block never names is
        // dense, unlike MLX's format where the file-level params apply.
        assert_eq!(q.default_params(), None);
        assert_eq!(q.params_for("model.layers.7.self_attn.k_proj"), None);
    }

    /// Decoding mxfp4 as affine yields a model that loads and emits noise, so
    /// an unknown mode has to fail the parse instead.
    #[test]
    fn an_unknown_mode_is_rejected() {
        let block = serde_json::json!({"group_size": 64, "bits": 4, "mode": "int3_pairwise"});
        assert!(MlxQuantization::from_json(&block, identity).is_none());
    }

    #[test]
    fn mode_defaults_match_mlx() {
        // `_defaults_for_mode` in mlx/nn/layers/quantized.py.
        for (mode, group_size, bits) in [
            (QuantizedMode::Affine, 64, 4),
            (QuantizedMode::Mxfp4, 32, 4),
            (QuantizedMode::Nvfp4, 16, 4),
            (QuantizedMode::Mxfp8, 32, 8),
        ] {
            let p = QuantParams::defaults_for(mode);
            assert_eq!((p.group_size, p.bits), (group_size, bits), "{mode:?}");
        }
    }
}
