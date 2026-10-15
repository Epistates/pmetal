//! Reading model checkpoints and assigning them to models.
//!
//! The pipeline, front to back:
//!
//! 1. [`checkpoint`]: find the weight files (safetensors, sharded or not, or a
//!    PyTorch pickle) and read tensors from them, lazily.
//! 2. [`quant`]: unpack a quantized checkpoint (MLX, ModelOpt, block FP8) to
//!    dense tensors.
//! 3. [`assign`]: assign tensors to a model's parameters by name, and
//!    [`audit`]: fail unless every tensor was consumed (or exempted for a stated
//!    reason) and every parameter was filled.
//!
//! [`families`] and [`diffusion`] hold the loaders for the architectures whose
//! checkpoint layout needs more than a name match.

use pmetal_runtime::compat::Array;

use std::collections::HashMap;
use std::path::Path;

mod assign;
mod audit;
mod checkpoint;
mod diffusion;
mod families;
mod quant;

pub use assign::{assign_weights, load_generic_weights, load_generic_weights_with};
pub use audit::{Exemption, LoadPolicy, NEVER_PARAMETERS, ParamSnapshot};
pub use checkpoint::{WeightIndex, load_safetensors_file};
pub use diffusion::*;
pub use families::*;

pub(crate) use checkpoint::validate_shard_path;
pub(crate) use quant::dequantize_sidecar_weights;

use checkpoint::Checkpoint;
use quant::QuantConfig;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    #[error("Shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<i32>,
        actual: Vec<i32>,
    },
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("MLX error: {0}")]
    Mlx(String),
    #[error("MLX IO error: {0}")]
    MlxIo(#[from] pmetal_runtime::compat::IoError),
    /// The checkpoint and the model the config describes do not match.
    #[error("{}", describe_incomplete(.unconsumed, .unloaded, .hint))]
    Incomplete {
        /// Checkpoint tensors no parameter consumed.
        unconsumed: Vec<String>,
        /// Parameters no tensor filled.
        unloaded: Vec<String>,
        /// A likely cause, when the config names one.
        hint: Option<String>,
    },
}

impl From<pmetal_runtime::compat::Exception> for LoadError {
    fn from(e: pmetal_runtime::compat::Exception) -> Self {
        Self::Mlx(e.to_string())
    }
}

fn describe_incomplete(
    unconsumed: &[String],
    unloaded: &[String],
    hint: &Option<String>,
) -> String {
    let mut message = String::from(
        "the checkpoint does not match the model its config describes, so it would run \
         with weights missing or unused",
    );
    if !unconsumed.is_empty() {
        message.push_str(&format!(
            "; {} checkpoint tensor(s) no parameter consumes: [{}]",
            unconsumed.len(),
            audit::sample(unconsumed)
        ));
    }
    if !unloaded.is_empty() {
        message.push_str(&format!(
            "; {} parameter(s) the checkpoint does not fill: [{}]",
            unloaded.len(),
            audit::sample(unloaded)
        ));
    }
    if let Some(hint) = hint {
        message.push_str("; ");
        message.push_str(hint);
    }
    message
}

/// `quant_method` values that name a scheme pmetal has no reader for.
const UNREAD_QUANT_METHODS: &[&str] = &[
    "gptq",
    "awq",
    "bitsandbytes",
    "bnb",
    "hqq",
    "aqlm",
    "eetq",
    "quanto",
    "torchao",
    "auto-round",
    "compressed-tensors",
    "fbgemm_fp8",
];

impl LoadError {
    /// For an [`Incomplete`](Self::Incomplete) error, say so when the
    /// checkpoint's config names a quantization method pmetal cannot read: its
    /// tensors (`qweight`, `qzeros`, `absmax`, ...) are exactly what then goes
    /// unconsumed, and the method is the cause a reader wants named.
    pub(crate) fn with_quantization_hint(mut self, model_dir: &Path) -> Self {
        if let Self::Incomplete { hint, .. } = &mut self {
            let method = std::fs::read_to_string(model_dir.join("config.json"))
                .ok()
                .and_then(|raw| crate::dispatcher::config_value(&raw).ok())
                .and_then(|config| {
                    pmetal_runtime::native_loader::quantization_block(&config)?
                        .get("quant_method")?
                        .as_str()
                        .map(str::to_owned)
                });
            if let Some(method) = method
                && UNREAD_QUANT_METHODS.contains(&method.as_str())
            {
                *hint = Some(format!(
                    "config.json says the weights are quantized with `{method}`, which pmetal \
                     does not read; use the original checkpoint or an MLX-format quantization"
                ));
            }
        }
        self
    }
}

/// The tensors `keep` wants, plus the quantization aux tensors and sidecars
/// of each, read as stored: nothing is unpacked.
///
/// For a loader that keeps some quantized weights packed; everyone else wants
/// [`load_weights_filtered`], which unpacks.
pub(crate) fn load_weights_filtered_raw<F>(
    model_dir: &Path,
    mut keep: F,
) -> Result<HashMap<String, Array>, LoadError>
where
    F: FnMut(&str) -> bool,
{
    Checkpoint::open(model_dir)?.read(|key| quant::wanted_with_aux(key, &mut keep))
}

/// Unpack an MLX-quantized checkpoint's `.scales` / `.biases` triples in place,
/// as `config.json`'s quantization block describes them. A no-op for a
/// checkpoint with no scales.
pub(crate) fn unpack_mlx_quantized_weights(
    model_dir: &Path,
    weights: &mut HashMap<String, Array>,
) -> Result<(), LoadError> {
    quant::unpack_mlx_quantized(weights, QuantConfig::read(model_dir)?.as_ref())
}

pub(crate) fn load_weights_filtered<F>(
    model_dir: impl AsRef<Path>,
    mut keep: F,
) -> Result<HashMap<String, Array>, LoadError>
where
    F: FnMut(&str) -> bool,
{
    let model_dir = model_dir.as_ref();
    let mut all_weights = load_weights_filtered_raw(model_dir, &mut keep)?;
    unpack_mlx_quantized_weights(model_dir, &mut all_weights)?;
    dequantize_sidecar_weights(&mut all_weights)?;

    Ok(all_weights
        .into_iter()
        .filter(|(key, _)| keep(key))
        .collect())
}

/// Load all weights from a model directory, unpacked to dense tensors.
pub fn load_weights(model_dir: impl AsRef<Path>) -> Result<HashMap<String, Array>, LoadError> {
    load_weights_filtered(model_dir, |_| true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let n = a.size();
        let f32_dtype = pmetal_runtime::dtype::F32;
        let mut a = a.as_dtype(f32_dtype);
        let mut b = b.as_dtype(f32_dtype);
        let va = a.to_f32_vec(n).expect("read lhs");
        let vb = b.to_f32_vec(n).expect("read rhs");
        va.iter()
            .zip(vb.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    }

    fn write_safetensors(
        path: &Path,
        weights: &HashMap<String, Array>,
    ) -> Result<(), pmetal_runtime::compat::IoError> {
        let entries: Vec<_> = weights
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect();
        Array::save_safetensors(path.to_string_lossy().as_ref(), &entries)?;
        Ok(())
    }

    /// ⚠️ A ModelOpt checkpoint used to load with its sidecars skipped as
    /// "unmatched" and its packed bytes assigned as dense weights, while
    /// reporting every weight matched. Training and eval both read this path.
    #[test]
    fn modelopt_checkpoints_unpack_to_dense() {
        use pmetal_runtime::QuantizedMode;
        let dir = tempdir().unwrap();
        let (out, in_dim) = (4i32, 64i32);
        let values: Vec<f32> = (0..out * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) / 6.0)
            .collect();
        let w = Array::from_f32_slice(&values, &[out, in_dim]);
        let (packed, scales) = w.quantize_weights_mode(16, 4, QuantizedMode::Nvfp4);
        let fp8 = w.to_fp8();
        let scalar = |v: f32| Array::from_f32_slice(&[v], &[]);

        let down = "model.layers.0.mlp.down_proj";
        let q = "model.layers.0.self_attn.q_proj";
        let mut weights = HashMap::new();
        weights.insert(
            format!("{down}.weight"),
            packed.view(pmetal_runtime::dtype::U8),
        );
        weights.insert(format!("{down}.weight_scale"), scales.clone());
        weights.insert(format!("{down}.weight_scale_2"), scalar(0.37));
        weights.insert(format!("{down}.input_scale"), scalar(0.2));
        weights.insert(format!("{q}.weight"), fp8.clone());
        weights.insert(format!("{q}.weight_scale"), scalar(0.5));
        weights.insert(format!("{q}.input_scale"), scalar(0.2));
        weights.insert(
            "model.norm.weight".to_string(),
            Array::ones(&[in_dim], pmetal_runtime::dtype::F32),
        );
        write_safetensors(&dir.path().join("model.safetensors"), &weights).unwrap();

        let loaded = load_weights(dir.path()).unwrap();

        let sidecars: Vec<&String> = loaded
            .keys()
            .filter(|k| pmetal_runtime::native_loader::quant_sidecar_base(k).is_some())
            .collect();
        assert!(sidecars.is_empty(), "sidecars survived: {sidecars:?}");

        let want_down = packed
            .dequantize_mode(&scales, None, 16, 4, QuantizedMode::Nvfp4)
            .as_dtype(pmetal_runtime::dtype::F32)
            .multiply(&scalar(0.37));
        let diff = max_abs_diff(&loaded[&format!("{down}.weight")], &want_down);
        assert!(diff < 1e-6, "nvfp4 unpacked wrongly: max|diff| = {diff}");

        let want_q = fp8
            .from_fp8(pmetal_runtime::dtype::F32)
            .multiply(&scalar(0.5));
        let diff = max_abs_diff(&loaded[&format!("{q}.weight")], &want_q);
        assert!(diff < 1e-6, "fp8 unpacked wrongly: max|diff| = {diff}");
    }

    /// A weight scale with nothing to apply it to has to stop the load.
    #[test]
    fn an_orphaned_weight_scale_is_an_error() {
        let dir = tempdir().unwrap();
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.mlp.down_proj.weight_scale_inv".to_string(),
            Array::from_f32_slice(&[1.0], &[1, 1]),
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Array::ones(&[4], pmetal_runtime::dtype::F32),
        );
        write_safetensors(&dir.path().join("model.safetensors"), &weights).unwrap();

        let err = load_weights(dir.path()).unwrap_err().to_string();
        assert!(err.contains("weight_scale_inv"), "unexpected error: {err}");
    }

    #[test]
    fn load_weights_filtered_reads_only_needed_shards() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let text_shard = model_dir.join("text.safetensors");
        let broken_shard = model_dir.join("broken.safetensors");

        let mut text_weights = HashMap::new();
        text_weights.insert(
            "model.embed_tokens.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        write_safetensors(&text_shard, &text_weights).unwrap();
        std::fs::write(&broken_shard, b"not safetensors").unwrap();

        let index = serde_json::json!({
            "metadata": {},
            "weight_map": {
                "model.embed_tokens.weight": "text.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "broken.safetensors"
            }
        });
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index).unwrap(),
        )
        .unwrap();

        let loaded =
            load_weights_filtered(model_dir, |key| !key.contains(".mlp.experts.")).unwrap();

        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("model.embed_tokens.weight"));
    }

    /// Two quantized modules, one at each width, in the layout MLX writes.
    fn write_mixed_width_checkpoint(model_dir: &Path, config: &str) {
        let wide = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (wide_packed, wide_scales, wide_biases) = wide.quantize_weights(64, 8);
        let narrow = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (narrow_packed, narrow_scales, narrow_biases) = narrow.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        for (module, (packed, scales, biases)) in [
            (
                "model.layers.0.mlp.gate_proj",
                (wide_packed, wide_scales, wide_biases),
            ),
            (
                "model.layers.0.self_attn.q_proj",
                (narrow_packed, narrow_scales, narrow_biases),
            ),
        ] {
            weights.insert(format!("{module}.weight"), packed);
            weights.insert(format!("{module}.scales"), scales);
            weights.insert(format!("{module}.biases"), biases);
        }
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(model_dir.join("config.json"), config).unwrap();
    }

    #[test]
    fn load_weights_dequantizes_mlx_affine_quantized_tensors() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let weight = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (packed, scales, biases) = weight.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert("linear.weight".to_string(), packed);
        weights.insert("linear.scales".to_string(), scales);
        weights.insert("linear.biases".to_string(), biases);
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(loaded.len(), 1);
        let restored = loaded.get("linear.weight").unwrap();
        assert_eq!(restored.shape(), &[2, 64]);
        assert!(!loaded.contains_key("linear.scales"));
        assert!(!loaded.contains_key("linear.biases"));
    }

    /// pmetal's own quantizer writes the aux tensors under the full weight name
    /// rather than beside it, so `quantize_and_save_mlx` output has to keep
    /// loading too.
    #[test]
    fn load_weights_dequantizes_pmetals_own_aux_key_layout() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let weight = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (packed, scales, biases) = weight.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert("linear.weight".to_string(), packed);
        weights.insert("linear.weight.scales".to_string(), scales);
        weights.insert("linear.weight.biases".to_string(), biases);
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get("linear.weight").unwrap().shape(), &[2, 64]);
    }

    /// A mixed-precision checkpoint, which is what mlx-community QAT releases
    /// ship: most tensors at the model-wide width, some at another.
    ///
    /// Unpacking the 8-bit tensor at the model-wide 4 bits reads the right
    /// number of bytes and produces the wrong shape, so the load "succeeds" and
    /// the failure surfaces later as a norm complaining about its input size.
    #[test]
    fn load_weights_honours_per_module_quantization_widths() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        write_mixed_width_checkpoint(
            model_dir,
            r#"{
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 64}
                }
            }"#,
        );

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(
            loaded
                .get("model.layers.0.mlp.gate_proj.weight")
                .unwrap()
                .shape(),
            &[2, 64],
            "the 8-bit tensor unpacked at the model-wide 4 bits"
        );
        assert_eq!(
            loaded
                .get("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .shape(),
            &[2, 64],
            "the tensor with no override should still take the model-wide width"
        );
    }

    /// ⚠️ pmetal's per-tensor widths (`per_tensor_overrides`, keyed by tensor)
    /// must survive moving the block parser to the runtime's.
    #[test]
    fn per_tensor_overrides_set_one_tensors_width() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        write_mixed_width_checkpoint(
            model_dir,
            r#"{
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "per_tensor_overrides": {"model.layers.0.mlp.gate_proj.weight": 8}
                }
            }"#,
        );

        let loaded = load_weights(model_dir).unwrap();
        for module in [
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.self_attn.q_proj",
        ] {
            assert_eq!(
                loaded[&format!("{module}.weight")].shape(),
                &[2, 64],
                "{module}"
            );
        }
    }

    /// Scales whose packing the config does not describe must stop the load:
    /// the packed words would otherwise be assigned to a dense layer.
    #[test]
    fn scales_without_a_quantization_block_are_an_error() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        write_mixed_width_checkpoint(model_dir, r#"{"hidden_size": 64}"#);

        let err = load_weights(model_dir).unwrap_err().to_string();
        assert!(err.contains("no `quantization` block"), "{err}");
    }

    /// A module the block marks `false` is dense and carries no scales; scales
    /// on it mean the config and the tensors disagree.
    #[test]
    fn scales_on_a_module_the_config_calls_dense_are_an_error() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        write_mixed_width_checkpoint(
            model_dir,
            r#"{"quantization": {"bits": 4, "group_size": 64,
                "model.layers.0.mlp.gate_proj": false}}"#,
        );
        let err = load_weights(model_dir).unwrap_err().to_string();
        assert!(err.contains("not quantized"), "{err}");
    }

    /// The float modes (mxfp4, nvfp4, mxfp8) store scales and no biases, and
    /// `config.json` names the mode. They used to be skipped outright: the
    /// triple needed biases, the scales were then dropped as aux tensors, and
    /// the packed words were assigned to a dense layer.
    #[test]
    fn mxfp4_checkpoints_unpack_without_biases() {
        use pmetal_runtime::QuantizedMode;
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let values: Vec<f32> = (0..2 * 64).map(|i| ((i % 11) as f32 - 5.0) / 5.0).collect();
        let dense = Array::from_f32_slice(&values, &[2, 64]);
        let (packed, scales) = dense.quantize_weights_mode(32, 4, QuantizedMode::Mxfp4);

        let mut weights = HashMap::new();
        weights.insert("linear.weight".to_string(), packed.clone());
        weights.insert("linear.scales".to_string(), scales.clone());
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}}"#,
        )
        .unwrap();

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(
            loaded.len(),
            1,
            "the scales must be consumed: {:?}",
            loaded.keys()
        );
        let want = packed.dequantize_mode(&scales, None, 32, 4, QuantizedMode::Mxfp4);
        let diff = max_abs_diff(&loaded["linear.weight"], &want);
        assert!(diff < 1e-6, "mxfp4 unpacked wrongly: max|diff| = {diff}");
        // ...and it is a real round trip, not merely self-consistent.
        let roundtrip = max_abs_diff(&loaded["linear.weight"], &dense);
        assert!(
            roundtrip < 0.25,
            "mxfp4 is nowhere near the source: {roundtrip}"
        );
    }

    /// Affine scales with no biases are not affine quantization; reading them
    /// as such produces noise.
    #[test]
    fn affine_scales_without_biases_are_an_error() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let dense = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (packed, scales, _) = dense.quantize_weights(64, 4);
        let mut weights = HashMap::new();
        weights.insert("linear.weight".to_string(), packed);
        weights.insert("linear.scales".to_string(), scales);
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();
        let err = load_weights(model_dir).unwrap_err().to_string();
        assert!(err.contains("no biases"), "{err}");
    }

    /// A model published only as PyTorch pickles loads through the same
    /// reader, with the stored dtype kept.
    #[test]
    fn a_pytorch_checkpoint_loads_like_safetensors() {
        use pmetal_hub::torch::testing::{CheckpointLayout, FixtureTensor, checkpoint};
        let temp = tempdir().unwrap();
        let bytes = checkpoint(
            &[
                FixtureTensor::f32("model.norm.weight", &[3], &[1.0, 2.0, 3.0]),
                // 1.0 and -2.0 in bfloat16.
                FixtureTensor::bf16_bits("model.embed_tokens.weight", &[2], &[0x3f80, 0xc000]),
            ],
            &CheckpointLayout::default(),
        );
        std::fs::write(temp.path().join("pytorch_model.bin"), bytes).unwrap();

        let loaded = load_weights(temp.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        let norm = &loaded["model.norm.weight"];
        assert_eq!(norm.dtype_raw(), pmetal_runtime::dtype::F32);
        let mut norm = norm.clone();
        assert_eq!(norm.to_f32_vec(3).unwrap(), vec![1.0, 2.0, 3.0]);
        let embed = &loaded["model.embed_tokens.weight"];
        assert_eq!(embed.dtype_raw(), pmetal_runtime::dtype::BF16);
        let mut embed = embed.as_dtype(pmetal_runtime::dtype::F32);
        assert_eq!(embed.to_f32_vec(2).unwrap(), vec![1.0, -2.0]);
    }

    /// Safetensors wins when both are present: it is the maintained format and
    /// the pickles are usually the stale half of a conversion.
    #[test]
    fn safetensors_is_preferred_over_a_pytorch_checkpoint() {
        use pmetal_hub::torch::testing::{CheckpointLayout, FixtureTensor, checkpoint};
        let temp = tempdir().unwrap();
        let bytes = checkpoint(
            &[FixtureTensor::f32("only_in_pickle", &[1], &[1.0])],
            &CheckpointLayout::default(),
        );
        std::fs::write(temp.path().join("pytorch_model.bin"), bytes).unwrap();
        let mut weights = HashMap::new();
        weights.insert(
            "in_safetensors".to_string(),
            Array::from_f32_slice(&[1.0], &[1]),
        );
        write_safetensors(&temp.path().join("model.safetensors"), &weights).unwrap();

        let loaded = load_weights(temp.path()).unwrap();
        assert!(loaded.contains_key("in_safetensors"));
        assert!(!loaded.contains_key("only_in_pickle"));
    }

    #[test]
    fn a_directory_with_no_weights_says_what_it_looked_for() {
        let temp = tempdir().unwrap();
        let err = load_weights(temp.path()).unwrap_err().to_string();
        assert!(err.contains("pytorch_model.bin"), "{err}");
    }

    /// huggingface_hub's shared blob store puts the bytes at
    /// `<cache>/blobs/<xx>/<hash>`, outside the repo directory, behind two
    /// symlinks. Such a shard is inside the cache and must load; one pointing
    /// anywhere else still must not.
    #[test]
    fn shard_paths_follow_the_shared_blob_store_but_no_further() {
        use std::os::unix::fs::symlink;
        let cache = tempdir().unwrap();
        let root = cache.path();
        let snapshot = root.join("models--org--name/snapshots/abc");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(root.join("models--org--name/blobs")).unwrap();
        std::fs::create_dir_all(root.join("blobs/9f")).unwrap();
        std::fs::write(root.join("blobs/9f/9fce"), b"shard").unwrap();
        symlink(
            "../../blobs/9f/9fce",
            root.join("models--org--name/blobs/etag"),
        )
        .unwrap();
        symlink("../../blobs/etag", snapshot.join("model-1.safetensors")).unwrap();
        assert!(validate_shard_path(&snapshot, "model-1.safetensors").is_ok());

        let outside = tempdir().unwrap();
        std::fs::write(outside.path().join("elsewhere"), b"shard").unwrap();
        symlink(
            outside.path().join("elsewhere"),
            snapshot.join("model-2.safetensors"),
        )
        .unwrap();
        assert!(validate_shard_path(&snapshot, "model-2.safetensors").is_err());
        assert!(validate_shard_path(&snapshot, "../model.safetensors").is_err());
    }

    #[test]
    fn an_incomplete_load_names_a_quantization_method_pmetal_cannot_read() {
        let temp = tempdir().unwrap();
        std::fs::write(
            temp.path().join("config.json"),
            r#"{"quantization_config": {"quant_method": "gptq", "bits": 4}}"#,
        )
        .unwrap();
        let err = LoadError::Incomplete {
            unconsumed: vec!["model.layers.0.mlp.up_proj.qweight".into()],
            unloaded: vec!["model.layers.0.mlp.up_proj.weight".into()],
            hint: None,
        }
        .with_quantization_hint(temp.path())
        .to_string();
        assert!(err.contains("`gptq`"), "{err}");
        assert!(err.contains("qweight"), "{err}");
    }
}
