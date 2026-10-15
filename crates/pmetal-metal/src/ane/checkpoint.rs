//! What the ANE engines can load, and loading it without silent fallbacks.
//!
//! The two engines implement fixed architectures. Training
//! ([`super::dynamic_trainer`]) is Llama-shaped: SiLU-gated FFN, no per-head
//! q/k norm, logits from the embedding table. Inference ([`super::lm`]) is
//! Qwen3-shaped: the same FFN plus per-head q/k RMSNorm, with an output
//! projection of its own when the checkpoint has one. Neither routes experts,
//! applies scaled RoPE (`rope_scaling`, or a `rope_parameters` that isn't
//! `default`), adds attention or MLP biases, or uses a sliding window or any
//! layer other than full attention. A checkpoint outside that shape used to
//! load anyway and compute a different function with no error (#34), as did
//! one whose tensors failed to convert or had the wrong size.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{MetalError, Result};

/// A tensor's bytes as f32. The ANE engines read f32, f16 and bf16; anything
/// else (a packed quantized weight, an integer tensor) is an error.
fn bytes_to_f32(dtype: safetensors::Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    use safetensors::Dtype;
    let width = match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        other => return Err(MetalError::UnsupportedDtype(format!("{other:?}"))),
    };
    if bytes.len() % width != 0 {
        return Err(MetalError::UnsupportedDtype(format!(
            "{dtype:?} data of {} bytes isn't a whole number of elements",
            bytes.len()
        )));
    }
    Ok(match dtype {
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        _ => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
    })
}

/// A safetensors checkpoint (one file, or shards with an index), read one
/// tensor at a time: an engine that builds a layer at a time never holds the
/// whole model as f32.
pub(crate) struct Checkpoint {
    files: Vec<(PathBuf, memmap2::Mmap, usize)>,
    /// Tensor name to (file, metadata).
    index: HashMap<String, (usize, safetensors::tensor::TensorInfo)>,
}

impl Checkpoint {
    /// Open the checkpoint in `dir` (or the single file `dir`).
    pub(crate) fn open(dir: &Path) -> Result<Self> {
        let bad = |what: String| MetalError::InvalidConfig(what);
        let paths = if dir.is_file() {
            vec![dir.to_path_buf()]
        } else if dir.join("model.safetensors.index.json").exists() {
            let text = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
                .map_err(|e| bad(format!("model.safetensors.index.json: {e}")))?;
            let index: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| bad(format!("model.safetensors.index.json: {e}")))?;
            let mut shards: Vec<String> = index["weight_map"]
                .as_object()
                .ok_or_else(|| bad("model.safetensors.index.json has no weight_map".into()))?
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shards.sort();
            shards.dedup();
            shards.into_iter().map(|f| dir.join(f)).collect()
        } else {
            vec![dir.join("model.safetensors")]
        };

        let mut files = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (i, path) in paths.into_iter().enumerate() {
            let file =
                std::fs::File::open(&path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
            #[allow(unsafe_code)]
            // SAFETY: the file is opened read-only, and a checkpoint isn't
            // rewritten while a model loads from it.
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| bad(format!("{}: {e}", path.display())))?;
            let (header_len, metadata) = safetensors::SafeTensors::read_metadata(&mmap)
                .map_err(|e| bad(format!("{}: {e}", path.display())))?;
            for (name, info) in metadata.tensors() {
                index.insert(name, (i, info.clone()));
            }
            files.push((path, mmap, 8 + header_len));
        }
        Ok(Self { files, index })
    }

    /// Whether the checkpoint has `name`.
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Fail, naming them, unless the checkpoint has every tensor in
    /// `required`: before an engine spends minutes compiling.
    pub(crate) fn require(&self, required: impl IntoIterator<Item = String>) -> Result<()> {
        missing_tensors(required.into_iter().filter(|n| !self.contains(n)).collect())
    }

    /// `name` as f32, which must have `expected` elements.
    pub(crate) fn f32(&self, name: &str, expected: usize) -> Result<Vec<f32>> {
        let (file, info) = self
            .index
            .get(name)
            .ok_or_else(|| MetalError::InvalidConfig(format!("checkpoint has no tensor {name}")))?;
        let (path, mmap, data_start) = &self.files[*file];
        let (start, end) = info.data_offsets;
        let bytes = mmap
            .get(data_start + start..data_start + end)
            .ok_or_else(|| {
                MetalError::InvalidConfig(format!("{name} runs past the end of {}", path.display()))
            })?;
        let data = bytes_to_f32(info.dtype, bytes).map_err(|e| {
            MetalError::InvalidConfig(format!(
                "{name}: {e} (the ANE engines read f32, f16 and bf16 weights only)"
            ))
        })?;
        if data.len() != expected {
            return Err(MetalError::InvalidConfig(format!(
                "{name} has {} elements; the ANE engine expects {expected}",
                data.len()
            )));
        }
        Ok(data)
    }
}

/// Config limits both engines share. `engine` names the caller in errors.
pub(crate) fn check_shared_limits(
    config: &serde_json::Value,
    engine: &str,
) -> std::result::Result<(), String> {
    check_architecture_limits(config, engine)?;
    check_tied_embeddings(config, engine)
}

/// The RoPE settings object of a `config.json`: `rope_parameters` in the
/// layout newer Transformers write, else the older `rope_scaling`.
fn rope_settings(
    config: &serde_json::Value,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    ["rope_parameters", "rope_scaling"]
        .iter()
        .find_map(|key| config.get(key).and_then(|v| v.as_object()))
}

/// The RoPE base frequency a `config.json` names: `rope_theta`, or the same
/// key inside the RoPE settings object. `None` when it names none.
pub(crate) fn rope_theta(config: &serde_json::Value) -> Option<f64> {
    config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .or_else(|| rope_settings(config)?.get("rope_theta")?.as_f64())
}

/// What no ANE engine implements: routed experts, scaled RoPE, sliding
/// windows or any layer that isn't full attention, attention or MLP biases.
pub(crate) fn check_architecture_limits(
    config: &serde_json::Value,
    engine: &str,
) -> std::result::Result<(), String> {
    let int = |key: &str| config.get(key).and_then(|v| v.as_i64()).unwrap_or(0);
    let flag = |key: &str| config.get(key).and_then(|v| v.as_bool());
    let set = |key: &str| config.get(key).is_some_and(|v| !v.is_null());

    if ["num_experts", "num_local_experts", "n_routed_experts"]
        .iter()
        .any(|k| int(k) > 0)
    {
        return Err(format!(
            "MoE models with routed experts are not supported by ANE {engine}."
        ));
    }
    // Plain RoPE is `{"rope_type": "default"}` or nothing; any other type
    // (yarn, llama3, linear, ...) rescales frequencies these kernels don't.
    for key in ["rope_parameters", "rope_scaling"] {
        let Some(settings) = config.get(key).and_then(|v| v.as_object()) else {
            continue;
        };
        let kind = ["rope_type", "type"]
            .iter()
            .find_map(|k| settings.get(*k).and_then(|v| v.as_str()));
        let scaled = match kind {
            Some(kind) => kind != "default",
            None => settings.contains_key("factor"),
        };
        if scaled {
            return Err(format!(
                "{key} ({}) is not implemented by the ANE {engine} engine, which applies plain RoPE.",
                kind.unwrap_or("scaled")
            ));
        }
    }
    if set("sliding_window") && flag("use_sliding_window") != Some(false) {
        return Err(format!(
            "Sliding-window attention is not implemented by the ANE {engine} engine."
        ));
    }
    if let Some(other) = config
        .get("layer_types")
        .and_then(|v| v.as_array())
        .and_then(|types| {
            types
                .iter()
                .filter_map(|t| t.as_str())
                .find(|t| *t != "full_attention")
        })
    {
        return Err(format!(
            "A {other} layer is not implemented by the ANE {engine} engine, which runs full attention."
        ));
    }
    for key in ["attention_bias", "mlp_bias"] {
        if flag(key) == Some(true) {
            return Err(format!(
                "{key} is not implemented by the ANE {engine} engine."
            ));
        }
    }
    Ok(())
}

/// The older engines take logits from the embedding matrix.
fn check_tied_embeddings(
    config: &serde_json::Value,
    engine: &str,
) -> std::result::Result<(), String> {
    // Hugging Face defaults tie_word_embeddings to true when absent.
    if config.get("tie_word_embeddings").and_then(|v| v.as_bool()) == Some(false) {
        return Err(format!(
            "The ANE {engine} engine takes logits from the embedding matrix, so a model \
             with a separate lm_head (tie_word_embeddings: false) is not supported."
        ));
    }
    Ok(())
}

/// Reject a `model_type` the engine doesn't implement.
pub(crate) fn check_model_type(
    config: &serde_json::Value,
    supported: &[&str],
    engine: &str,
) -> std::result::Result<(), String> {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if supported.contains(&model_type) {
        Ok(())
    } else {
        Err(format!(
            "Model type '{model_type}' is not implemented by the ANE {engine} engine \
             (supported: {}). Use the GPU.",
            supported.join(", ")
        ))
    }
}

/// `Ok` for no missing tensors, else the error naming the first few.
fn missing_tensors(missing: Vec<String>) -> Result<()> {
    if missing.is_empty() {
        return Ok(());
    }
    let shown = missing
        .iter()
        .take(5)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let more = missing.len().saturating_sub(5);
    Err(MetalError::InvalidConfig(format!(
        "checkpoint is missing {} tensor(s) the ANE engine needs: {shown}{}",
        missing.len(),
        if more > 0 {
            format!(" and {more} more")
        } else {
            String::new()
        }
    )))
}

/// Per-layer tensor names under `model.layers.{i}.`.
pub(crate) fn layer_tensors<'a>(
    n_layers: usize,
    suffixes: &'a [&'a str],
) -> impl Iterator<Item = String> + 'a {
    (0..n_layers).flat_map(move |i| {
        suffixes
            .iter()
            .map(move |suffix| format!("model.layers.{i}.{suffix}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn shared_limits_reject_what_neither_engine_implements() {
        let base = json!({"model_type": "llama"});
        assert!(check_shared_limits(&base, "training").is_ok());

        for (key, value) in [
            ("num_experts", json!(8)),
            (
                "rope_scaling",
                json!({"rope_type": "llama3", "factor": 8.0}),
            ),
            ("sliding_window", json!(4096)),
            ("attention_bias", json!(true)),
            ("tie_word_embeddings", json!(false)),
        ] {
            let mut config = base.clone();
            config[key] = value;
            assert!(
                check_shared_limits(&config, "training").is_err(),
                "{key} should be rejected"
            );
        }

        // Present but null or disabled is fine.
        let config = json!({
            "rope_scaling": null,
            "sliding_window": 4096,
            "use_sliding_window": false,
            "tie_word_embeddings": true,
        });
        assert!(check_shared_limits(&config, "inference").is_ok());
    }

    /// Newer Transformers write the RoPE settings as `rope_parameters`
    /// (`rope_type`, `rope_theta`, ...) in place of `rope_scaling` and
    /// `rope_theta`. A scaled RoPE there ran as plain RoPE, and a plain one
    /// written as `{"rope_type": "default"}` was refused.
    #[test]
    fn rope_settings_are_read_in_either_layout() {
        let plain = |v| check_architecture_limits(&v, "inference").is_ok();
        assert!(plain(
            json!({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}})
        ));
        assert!(plain(json!({"rope_scaling": {"rope_type": "default"}})));
        assert!(plain(json!({"rope_parameters": null})));
        for scaled in ["yarn", "linear", "dynamic", "llama3", "longrope"] {
            assert!(
                !plain(json!({"rope_parameters": {"rope_type": scaled, "factor": 4.0}})),
                "rope_parameters {scaled}"
            );
            assert!(
                !plain(json!({"rope_scaling": {"type": scaled, "factor": 4.0}})),
                "rope_scaling {scaled}"
            );
        }
        // A settings object that names no type but carries a factor is scaling.
        assert!(!plain(json!({"rope_parameters": {"factor": 2.0}})));

        assert_eq!(rope_theta(&json!({"rope_theta": 5e5})), Some(5e5));
        assert_eq!(
            rope_theta(&json!({"rope_parameters": {"rope_theta": 1e7}})),
            Some(1e7)
        );
        assert_eq!(
            rope_theta(&json!({"rope_scaling": {"rope_theta": 2e4}})),
            Some(2e4)
        );
        assert_eq!(rope_theta(&json!({})), None);
    }

    /// Per-layer attention types: any layer that isn't full attention is a
    /// window or a recurrence these kernels don't compute.
    #[test]
    fn layer_types_other_than_full_attention_are_refused() {
        let ok = json!({"layer_types": ["full_attention", "full_attention"]});
        assert!(check_architecture_limits(&ok, "inference").is_ok());
        for other in ["sliding_attention", "linear_attention"] {
            let config = json!({"layer_types": ["full_attention", other]});
            let err = check_architecture_limits(&config, "inference").unwrap_err();
            assert!(err.contains(other), "{err}");
        }
    }

    fn write_tensors(path: &std::path::Path, tensors: &[(&str, safetensors::Dtype, Vec<u8>)]) {
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, dtype, bytes)| {
                let width = if *dtype == safetensors::Dtype::F32 {
                    4
                } else {
                    2
                };
                (
                    name.to_string(),
                    safetensors::tensor::TensorView::new(*dtype, vec![bytes.len() / width], bytes)
                        .unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, None, path).unwrap();
    }

    /// f32, f16 and bf16 tensors read back as f32, from one file or from
    /// shards named in an index; a tensor of another dtype, of another size,
    /// or absent is an error that names it.
    #[test]
    fn a_checkpoint_reads_tensors_across_shards_and_refuses_the_rest() {
        let f32s: Vec<u8> = [1.5f32, -2.0, 0.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f16s: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        let bf16s: Vec<u8> = [3.0f32, -0.5]
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        let dir = tempfile::tempdir().unwrap();
        write_tensors(
            &dir.path().join("a.safetensors"),
            &[
                ("w32", safetensors::Dtype::F32, f32s),
                ("w16", safetensors::Dtype::F16, f16s),
            ],
        );
        write_tensors(
            &dir.path().join("b.safetensors"),
            &[
                ("wbf", safetensors::Dtype::BF16, bf16s),
                ("ints", safetensors::Dtype::U16, vec![1, 0, 2, 0]),
            ],
        );
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            json!({"weight_map": {
                "w32": "a.safetensors", "w16": "a.safetensors",
                "wbf": "b.safetensors", "ints": "b.safetensors",
            }})
            .to_string(),
        )
        .unwrap();

        let ckpt = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(ckpt.f32("w32", 3).unwrap(), [1.5, -2.0, 0.25]);
        assert_eq!(ckpt.f32("w16", 2).unwrap(), [1.5, -2.0]);
        assert_eq!(ckpt.f32("wbf", 2).unwrap(), [3.0, -0.5]);

        let err = ckpt.f32("w32", 4).unwrap_err().to_string();
        assert!(err.contains("w32") && err.contains("3 elements"), "{err}");
        let err = ckpt.f32("ints", 2).unwrap_err().to_string();
        assert!(err.contains("ints"), "{err}");
        let err = ckpt.f32("absent", 1).unwrap_err().to_string();
        assert!(err.contains("absent"), "{err}");

        assert!(ckpt.require(["w32".to_string(), "wbf".to_string()]).is_ok());
        let err = ckpt
            .require([
                "w32".into(),
                "model.norm.weight".into(),
                "lm_head.weight".into(),
            ])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("2 tensor(s)") && err.contains("model.norm.weight"),
            "{err}"
        );
    }
}
