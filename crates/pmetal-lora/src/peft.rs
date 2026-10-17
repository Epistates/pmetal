//! Adapters in the PEFT layout.
//!
//! An adapter directory is `adapter_config.json` and `adapter_model.safetensors`,
//! as `peft` writes and reads them, so an adapter trained here loads with
//! `PeftModel.from_pretrained` and one trained there loads here:
//!
//! - tensors `base_model.model.<module>.lora_A.weight` (`[r, in]`) and
//!   `.lora_B.weight` (`[out, r]`), plus `.lora_magnitude_vector` (`[out]`)
//!   for DoRA, where `<module>` is the module's name in the Hugging Face
//!   model (earlier PEFT releases spelled the last `.lora_magnitude_vector.weight`,
//!   which is read too);
//! - a config with `peft_type: "LORA"`, `r`, `lora_alpha`, `target_modules`,
//!   `use_rslora` and `use_dora`.
//!
//! Adapters written before this layout (`<module>.lora_a` and friends, in a
//! file called `lora_weights.safetensors`) still load.

use std::path::{Path, PathBuf};

use pmetal_core::LoraConfig;
use serde_json::{Value, json};

use crate::LoraError;

/// The weights file PEFT looks for.
pub const ADAPTER_WEIGHTS: &str = "adapter_model.safetensors";

/// The config file PEFT looks for.
pub const ADAPTER_CONFIG: &str = "adapter_config.json";

/// The weights file adapters were saved under before the PEFT layout.
pub const LEGACY_ADAPTER_WEIGHTS: &str = "lora_weights.safetensors";

/// What every tensor name in a PEFT adapter starts with.
pub(crate) const PEFT_PREFIX: &str = "base_model.model.";

/// The adapter weights `path` names: the file itself, or the weights file in
/// the directory (PEFT's name first, then the older one).
pub fn find_adapter_weights(path: impl AsRef<Path>) -> Option<PathBuf> {
    let path = path.as_ref();
    if path.is_file() {
        return Some(path.to_path_buf());
    }
    [ADAPTER_WEIGHTS, LEGACY_ADAPTER_WEIGHTS]
        .iter()
        .map(|name| path.join(name))
        .find(|candidate| candidate.is_file())
}

/// The directory an adapter's config sits in, given its weights file or the
/// directory itself.
pub fn adapter_dir(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    }
}

/// A module's name in the Hugging Face model, from its path in the parameter
/// tree: Gemma holds its layers under `layers.gemma1.N` or `layers.gemma2.N`
/// by layer variant, which the checkpoint (and so PEFT) does not.
pub(crate) fn hf_module_name(path: &str) -> String {
    for variant in [".gemma1.", ".gemma2."] {
        if let Some(at) = path.find(variant) {
            return format!("{}.{}", &path[..at], &path[at + variant.len()..]);
        }
    }
    path.to_string()
}

/// The PEFT tensor name of an adapter tensor, `part` being `lora_A.weight`,
/// `lora_B.weight` or `lora_magnitude_vector.weight`.
pub(crate) fn peft_tensor_name(module_path: &str, part: &str) -> String {
    format!("{PEFT_PREFIX}{}.{part}", hf_module_name(module_path))
}

/// `adapter_config.json` for an adapter trained with `config` on the
/// projections `targets` (module names, as `target_modules` spells them).
pub fn peft_config(config: &LoraConfig, targets: &[String], base_model: Option<&str>) -> Value {
    json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "auto_mapping": null,
        "base_model_name_or_path": base_model,
        "revision": null,
        "inference_mode": true,
        "r": config.r,
        "lora_alpha": config.alpha,
        "lora_dropout": config.dropout,
        "fan_in_fan_out": false,
        "bias": "none",
        "target_modules": targets,
        "use_rslora": config.use_rslora,
        "use_dora": config.use_dora,
        "init_lora_weights": true,
        "modules_to_save": null,
        "layers_to_transform": null,
        "layers_pattern": null,
        "rank_pattern": {},
        "alpha_pattern": {},
    })
}

/// Write `adapter_config.json` into `dir`.
pub fn write_adapter_config(
    dir: impl AsRef<Path>,
    config: &LoraConfig,
    targets: &[String],
    base_model: Option<&str>,
) -> Result<(), LoraError> {
    let path = dir.as_ref().join(ADAPTER_CONFIG);
    let text = serde_json::to_string_pretty(&peft_config(config, targets, base_model))
        .map_err(|e| LoraError::Io(e.to_string()))?;
    std::fs::write(&path, text).map_err(|e| LoraError::Io(format!("{}: {e}", path.display())))
}

/// The base model an `adapter_config.json` names: PEFT's
/// `base_model_name_or_path`, else the `base_model` earlier adapters carried.
pub fn adapter_base_model(config: &Value) -> Option<String> {
    ["base_model_name_or_path", "base_model"]
        .iter()
        .find_map(|key| config[key].as_str().filter(|s| !s.is_empty()))
        .map(String::from)
}

/// The [`LoraConfig`] an `adapter_config.json` describes, so a model can be
/// built to take the adapter's tensors.
///
/// Reads PEFT's keys (`lora_alpha`, `lora_dropout`, `use_rslora`, `use_dora`,
/// `target_modules` as a list or `"all-linear"`) and the ones earlier adapters
/// carried (`alpha`). A config that asks for something a single `LoraConfig`
/// cannot say (per-module ranks or alphas, extra trained modules, a bias, a
/// transposed layout) is refused, because loading it as uniform would serve a
/// different model than the one trained.
pub fn lora_config_from_peft(config: &Value) -> Result<LoraConfig, LoraError> {
    let refuse = |what: &str| {
        Err(LoraError::InvalidState(format!(
            "adapter_config.json: {what} is not supported"
        )))
    };
    if let Some(kind) = config["peft_type"].as_str()
        && !kind.eq_ignore_ascii_case("LORA")
    {
        return refuse(&format!("peft_type {kind:?} (only LORA)"));
    }
    let non_empty =
        |key: &str| config[key].as_object().is_some_and(|m| !m.is_empty());
    if non_empty("rank_pattern") || non_empty("alpha_pattern") {
        return refuse("a per-module rank or alpha (rank_pattern, alpha_pattern)");
    }
    if config["modules_to_save"].as_array().is_some_and(|m| !m.is_empty()) {
        return refuse("modules_to_save");
    }
    if config["fan_in_fan_out"].as_bool() == Some(true) {
        return refuse("fan_in_fan_out");
    }
    if config["bias"].as_str().is_some_and(|b| b != "none") {
        return refuse("a trained bias");
    }
    let r = config["r"]
        .as_u64()
        .filter(|r| *r > 0)
        .ok_or_else(|| LoraError::InvalidState("adapter_config.json: no rank `r`".into()))?;
    let alpha = config["lora_alpha"]
        .as_f64()
        .or_else(|| config["alpha"].as_f64())
        .ok_or_else(|| LoraError::InvalidState("adapter_config.json: no `lora_alpha`".into()))?;
    let target_modules = match &config["target_modules"] {
        Value::Array(names) => names
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        Value::String(name) => vec![name.clone()],
        _ => Vec::new(),
    };
    Ok(LoraConfig {
        r: r as usize,
        alpha: alpha as f32,
        dropout: config["lora_dropout"].as_f64().unwrap_or(0.0) as f32,
        target_modules,
        use_rslora: config["use_rslora"].as_bool().unwrap_or(false),
        use_dora: config["use_dora"].as_bool().unwrap_or(false),
        ..LoraConfig::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_layer_variants_lose_their_infix() {
        assert_eq!(
            hf_module_name("model.layers.gemma1.3.self_attn.q_proj"),
            "model.layers.3.self_attn.q_proj"
        );
        assert_eq!(
            hf_module_name("model.layers.gemma2.0.mlp.down_proj"),
            "model.layers.0.mlp.down_proj"
        );
        assert_eq!(hf_module_name("lm_head"), "lm_head");
    }

    #[test]
    fn tensor_names_are_peft_s() {
        assert_eq!(
            peft_tensor_name("model.layers.0.self_attn.q_proj", "lora_A.weight"),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        );
    }

    #[test]
    fn a_written_config_reads_back() {
        let config = LoraConfig {
            r: 8,
            alpha: 24.0,
            dropout: 0.1,
            use_rslora: true,
            use_dora: true,
            ..Default::default()
        };
        let targets = vec!["q_proj".to_string(), "down_proj".to_string()];
        let json = peft_config(&config, &targets, Some("org/model"));
        assert_eq!(json["peft_type"], "LORA");
        assert_eq!(adapter_base_model(&json).as_deref(), Some("org/model"));
        let back = lora_config_from_peft(&json).unwrap();
        assert_eq!(back.r, 8);
        assert_eq!(back.alpha, 24.0);
        assert!((back.dropout - 0.1).abs() < 1e-7);
        assert!(back.use_rslora && back.use_dora);
        assert_eq!(back.target_modules, targets);
    }

    #[test]
    fn what_one_lora_config_cannot_say_is_refused() {
        let base = |extra: Value| {
            let mut v = json!({"peft_type": "LORA", "r": 8, "lora_alpha": 16});
            v.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            v
        };
        assert!(lora_config_from_peft(&base(json!({}))).is_ok());
        for extra in [
            json!({"rank_pattern": {"q_proj": 4}}),
            json!({"alpha_pattern": {"q_proj": 4}}),
            json!({"modules_to_save": ["lm_head"]}),
            json!({"fan_in_fan_out": true}),
            json!({"bias": "all"}),
            json!({"peft_type": "IA3"}),
        ] {
            assert!(lora_config_from_peft(&base(extra.clone())).is_err(), "{extra}");
        }
        // The older spelling of alpha, and `all-linear` as a bare string.
        let old = json!({"r": 4, "alpha": 8.0, "target_modules": "all-linear"});
        let cfg = lora_config_from_peft(&old).unwrap();
        assert_eq!(cfg.alpha, 8.0);
        assert!(cfg.targets_all_linear());
    }
}
