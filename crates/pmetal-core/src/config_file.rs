//! The training config file: its sections, strict loading, and compatibility
//! with files written by earlier releases.
//!
//! A config file has four sections, `model`, `lora`, `training` and
//! `dataset`. Loading is strict: a key a section does not have is an error
//! that names the section and lists the keys it does have, because a
//! silently-dropped key is a setting the user believes is in force. Keys that
//! earlier releases accepted but never read (and that `pmetal init` used to
//! write out) are the exception, handled by [`RETIRED`]: left at their old
//! default they are dropped with a note, set to anything else they are an
//! error that says the key never did anything.

use crate::{
    ConfigValidator, DatasetConfig, LoraConfig, ModelConfig, PMetalError, Result, TrainingConfig,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::Path;

/// The sections of a training config file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullTrainingConfig {
    /// `model`: which checkpoint to train.
    #[serde(default)]
    pub model: ModelConfig,
    /// `lora`: the adapter.
    #[serde(default)]
    pub lora: LoraConfig,
    /// `training`: the optimisation loop.
    #[serde(default)]
    pub training: TrainingConfig,
    /// `dataset`: what to train on.
    #[serde(default)]
    pub dataset: DatasetConfig,
}

/// A config file read and checked, with what the loader changed on the way.
#[derive(Debug, Clone)]
pub struct LoadedConfig {
    /// The sections, validated.
    pub config: FullTrainingConfig,
    /// Keys from earlier releases that were dropped or moved, one sentence
    /// each. Surface them to the user; they are not errors.
    pub notes: Vec<String>,
}

/// What a retired key may still hold for the file to load.
enum Accepted {
    /// Any value: the key is dropped with a note.
    Anything,
    /// Only these (the old default `pmetal init` wrote): dropped silently.
    /// Any other value is an error, since it asked for something that never
    /// happened.
    Default(&'static [&'static str]),
}

/// A key earlier releases accepted and never read.
struct Retired {
    section: &'static str,
    key: &'static str,
    accepted: Accepted,
    /// Said when the value is refused, or in the note when it is dropped.
    why: &'static str,
}

const RETIRED: &[Retired] = &[
    Retired {
        section: "model",
        key: "dtype",
        accepted: Accepted::Default(&["bfloat16"]),
        why: "PMetal does not read it; the model trains in the dtype it loads in",
    },
    Retired {
        section: "model",
        key: "quantization",
        accepted: Accepted::Default(&["none"]),
        why: "PMetal does not read it; QLoRA is chosen with the `--quantization` option of \
              `pmetal train`",
    },
    Retired {
        section: "model",
        key: "device",
        accepted: Accepted::Default(&["gpu", "ane"]),
        why: "PMetal does not read it; training runs on the GPU, or on the Neural Engine with \
              `pmetal train --ane`",
    },
    Retired {
        section: "model",
        key: "trust_remote_code",
        accepted: Accepted::Default(&["false"]),
        why: "PMetal never runs a checkpoint's own code",
    },
    Retired {
        section: "model",
        key: "revision",
        accepted: Accepted::Default(&["null"]),
        why: "PMetal does not read it; pass a local path to pin a revision",
    },
    Retired {
        section: "model",
        key: "hf_token",
        accepted: Accepted::Default(&["null"]),
        why: "tokens do not belong in config files; set HF_TOKEN in the environment",
    },
    Retired {
        section: "dataset",
        key: "split",
        accepted: Accepted::Default(&["train"]),
        why: "PMetal does not read it",
    },
    Retired {
        section: "dataset",
        key: "text_column",
        accepted: Accepted::Default(&["text"]),
        why: "PMetal does not read it; use `pmetal train --text-column`",
    },
    Retired {
        section: "dataset",
        key: "max_samples",
        accepted: Accepted::Default(&["null"]),
        why: "PMetal does not read it, so the whole dataset would be used",
    },
    Retired {
        section: "dataset",
        key: "seed",
        accepted: Accepted::Default(&["42"]),
        why: "PMetal does not read it; the seed is `training.seed`",
    },
    Retired {
        section: "dataset",
        key: "format",
        accepted: Accepted::Anything,
        why: "the dataset format is detected from the file",
    },
    Retired {
        section: "training",
        key: "gradient_checkpointing",
        accepted: Accepted::Anything,
        why: "gradient checkpointing is per decoder layer and on by default; \
              `pmetal train --no-gradient-checkpointing` turns it off",
    },
];

/// The sections' names, as a file spells them.
const SECTIONS: [&str; 4] = ["model", "lora", "training", "dataset"];

impl FullTrainingConfig {
    /// Read a config file.
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<LoadedConfig> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| PMetalError::Config(format!("cannot read {}: {e}", path.display())))?;
        Self::from_yaml(&text).map_err(|e| PMetalError::Config(format!("{}: {e}", path.display())))
    }

    /// Parse and validate config text.
    ///
    /// Unknown keys, wrongly-typed values and values the trainers cannot run
    /// are errors; see the module documentation for the keys earlier releases
    /// wrote that still load.
    pub fn from_yaml(yaml: &str) -> Result<LoadedConfig> {
        let value: Value = serde_saphyr::from_str(yaml)
            .map_err(|e| PMetalError::Config(format!("not valid YAML: {e}")))?;
        let mut root = match value {
            Value::Null => Map::new(),
            Value::Object(map) => map,
            other => {
                return Err(PMetalError::Config(format!(
                    "a config file is a mapping of sections ({}), got {}",
                    SECTIONS.join(", "),
                    kind_of(&other)
                )));
            }
        };
        if let Some(unknown) = root.keys().find(|k| !SECTIONS.contains(&k.as_str())) {
            return Err(PMetalError::Config(format!(
                "unknown section `{unknown}`; a config file has: {}",
                SECTIONS.join(", ")
            )));
        }

        let mut notes = Vec::new();
        migrate_legacy_keys(&mut root, &mut notes)?;

        let config = FullTrainingConfig {
            model: take_section(&mut root, "model")?,
            lora: take_section(&mut root, "lora")?,
            training: take_section(&mut root, "training")?,
            dataset: take_section(&mut root, "dataset")?,
        };
        config.validate()?;
        Ok(LoadedConfig { config, notes })
    }
}

fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "nothing",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "a mapping",
    }
}

/// Read one section, naming it in any error.
fn take_section<T: DeserializeOwned + Default>(
    root: &mut Map<String, Value>,
    name: &str,
) -> Result<T> {
    match root.remove(name) {
        None | Some(Value::Null) => Ok(T::default()),
        Some(value) => serde_path_to_error::deserialize(value).map_err(|e| {
            let key = e.path().to_string();
            let key = if key == "." {
                name.to_string()
            } else {
                format!("{name}.{key}")
            };
            PMetalError::Config(format!("`{key}`: {}", e.inner()))
        }),
    }
}

/// Drop or move the keys earlier releases wrote, or refuse the ones that
/// asked for something that never happened.
fn migrate_legacy_keys(root: &mut Map<String, Value>, notes: &mut Vec<String>) -> Result<()> {
    for retired in RETIRED {
        let Some(Value::Object(section)) = root.get_mut(retired.section) else {
            continue;
        };
        let Some(value) = section.remove(retired.key) else {
            continue;
        };
        let name = format!("{}.{}", retired.section, retired.key);
        match retired.accepted {
            Accepted::Anything => notes.push(format!("`{name}` is ignored: {}", retired.why)),
            Accepted::Default(defaults) => {
                let shown = match &value {
                    Value::String(s) => s.to_ascii_lowercase(),
                    other => other.to_string(),
                };
                if !defaults.contains(&shown.as_str()) {
                    // A credential is never echoed back into an error.
                    let given = if retired.key.contains("token") {
                        name
                    } else {
                        format!("{name}: {shown}")
                    };
                    return Err(PMetalError::Config(format!(
                        "`{given}` has never taken effect: {}. Remove it.",
                        retired.why
                    )));
                }
            }
        }
    }

    // `model.max_seq_len` was never read; the length that took effect was
    // `training.max_seq_len`. It stays the authority; the model's value
    // stands in only when the training section leaves the length out.
    if let Some(Value::Object(model)) = root.get_mut("model")
        && let Some(length) = model.remove("max_seq_len")
    {
        let training = root
            .entry("training")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(training) = training
            && !training.contains_key("max_seq_len")
        {
            training.insert("max_seq_len".into(), length);
            notes.push("`model.max_seq_len` is read as `training.max_seq_len`".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LrSchedulerType, OptimizerType};

    fn error(yaml: &str) -> String {
        match FullTrainingConfig::from_yaml(yaml) {
            Err(PMetalError::Config(message)) => message,
            Err(other) => panic!("wrong error: {other:?}"),
            Ok(_) => panic!("expected an error for:\n{yaml}"),
        }
    }

    /// The file `pmetal init` wrote before the sections were trimmed to the
    /// keys the trainers read.
    const INIT_FILE: &str = "
model:
  model_id: meta-llama/Llama-3.2-1B
  dtype: bfloat16
  quantization: none
  device: ane
  max_seq_len: 2048
  use_flash_attention: true
  trust_remote_code: false
  revision: null
lora:
  r: 16
  alpha: 32.0
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  use_rslora: false
  use_dora: false
  bias: none
  init_lora_weights: true
training:
  learning_rate: 0.0002
  embedding_learning_rate: null
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  max_steps: null
  warmup_steps: 100
  warmup_ratio: null
  weight_decay: 0.01
  max_grad_norm: 1.0
  lr_scheduler: cosine
  min_lr: null
  wsd_stable_ratio: null
  cosine_num_restarts: null
  polynomial_power: null
  gradient_checkpointing: none
  optimizer: adamw
  seed: 42
  logging_steps: 10
  eval_steps: null
  save_steps: 500
  output_dir: ./output
  use_packing: true
  max_seq_len: 2048
dataset:
  dataset_id: ./data/train.jsonl
  split: train
  text_column: text
  max_samples: null
  shuffle: true
  seed: 42
";

    #[test]
    fn a_file_written_by_pmetal_init_still_loads() {
        let loaded = FullTrainingConfig::from_yaml(INIT_FILE).unwrap();
        let c = loaded.config;
        assert_eq!(c.model.model_id, "meta-llama/Llama-3.2-1B");
        assert_eq!(c.training.batch_size, 4);
        assert_eq!(c.training.warmup_steps, 100);
        assert_eq!(c.training.save_steps, Some(500));
        assert_eq!(c.training.max_seq_len, 2048);
        assert_eq!(c.dataset.dataset_id, "./data/train.jsonl");
        // Only `gradient_checkpointing` had a value worth a note; the rest
        // were left at their defaults.
        assert_eq!(loaded.notes.len(), 1, "{:?}", loaded.notes);
        assert!(loaded.notes[0].contains("gradient_checkpointing"));
    }

    #[test]
    fn the_documented_example_loads_with_its_values_in_force() {
        // The example in docs/configuration/config-file.md. Its `epochs`,
        // `lr_schedule`, `name` and `path` used to be dropped silently.
        let loaded = FullTrainingConfig::from_yaml(
            "
model:
  name: meta-llama/Llama-3.2-1B
  dtype: bfloat16
  device: gpu
  max_seq_len: 2048
lora:
  r: 8
  alpha: 16.0
training:
  learning_rate: 2e-4
  batch_size: 1
  epochs: 7
  lr_schedule: wsd
  warmup_steps: 100
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  weight_decay: 0.01
dataset:
  path: ./train.jsonl
  format: jsonl
",
        )
        .unwrap();
        let c = loaded.config;
        assert_eq!(c.model.model_id, "meta-llama/Llama-3.2-1B");
        assert_eq!(c.training.num_epochs, 7);
        assert_eq!(c.training.lr_scheduler, LrSchedulerType::Wsd);
        assert_eq!(c.training.max_seq_len, 2048, "the model's length stands in");
        assert_eq!(c.dataset.dataset_id, "./train.jsonl");
        assert_eq!(c.lora.r, 8);
        assert!(loaded.notes.iter().any(|n| n.contains("dataset.format")));
    }

    #[test]
    fn the_training_length_wins_over_the_models_stale_one() {
        let loaded = FullTrainingConfig::from_yaml(
            "model: {max_seq_len: 8192}\ntraining: {max_seq_len: 1024}",
        )
        .unwrap();
        assert_eq!(loaded.config.training.max_seq_len, 1024);
    }

    #[test]
    fn a_misspelt_key_is_an_error_naming_its_section() {
        let text = error("training:\n  learning_rat: 1e-4\n");
        assert!(text.contains("`training.learning_rat`"), "{text}");
        assert!(
            text.contains("learning_rate"),
            "lists the real keys: {text}"
        );

        let text = error("trainin:\n  seed: 1\n");
        assert!(
            text.contains("trainin") && text.contains("training"),
            "{text}"
        );
    }

    #[test]
    fn a_retired_key_with_a_real_value_is_an_error() {
        let text = error("dataset:\n  max_samples: 1000\n");
        assert!(text.contains("dataset.max_samples") && text.contains("never taken effect"));
        let text = error("model:\n  device: cpu\n");
        assert!(text.contains("model.device"), "{text}");
        let text = error("model:\n  hf_token: hf_abc\n");
        assert!(
            text.contains("HF_TOKEN") && !text.contains("hf_abc"),
            "{text}"
        );
    }

    #[test]
    fn a_wrongly_typed_value_names_the_section() {
        let text = error("training:\n  batch_size: many\n");
        assert!(text.contains("`training.batch_size`"), "{text}");
    }

    #[test]
    fn values_the_trainers_cannot_run_are_refused_at_load() {
        let text = error("training:\n  learning_rate: -1\n  optimizer: muon\n");
        assert!(text.contains("training.learning_rate"), "{text}");
        assert!(text.contains("training.optimizer"), "{text}");
        let text = error("lora:\n  bias: all\n");
        assert!(text.contains("lora.bias"), "{text}");
    }

    #[test]
    fn an_empty_file_is_the_default_config() {
        let loaded = FullTrainingConfig::from_yaml("").unwrap();
        assert_eq!(loaded.config.training.optimizer, OptimizerType::AdamW);
        assert!(loaded.notes.is_empty());
    }

    #[test]
    fn what_serializes_loads_back() {
        let yaml = serde_saphyr::to_string(&FullTrainingConfig::default()).unwrap();
        let loaded = FullTrainingConfig::from_yaml(&yaml).unwrap();
        assert!(loaded.notes.is_empty(), "{:?}", loaded.notes);
        let again = serde_saphyr::to_string(&loaded.config).unwrap();
        assert_eq!(again, yaml);
    }
}
