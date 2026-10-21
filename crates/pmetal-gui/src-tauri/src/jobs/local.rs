//! Jobs the CLI has and `pmetal-core` has no spec for yet.
//!
//! `pmetal bench-workload` and `pmetal ollama modelfile` take their options
//! from clap alone. Until they are specs like the rest, the app describes
//! them here, as a table of flags, and builds the command from the same kind
//! of form text. This is the only place the app names a CLI flag.

use std::collections::{BTreeMap, HashSet};

use super::form::{FieldInfo, FormInfo, Input};
use super::kinds::{FieldIssue, Kind, Launch, LaunchContext};
use crate::paths;

struct Flag {
    name: &'static str,
    label: &'static str,
    help: Option<&'static str>,
    group: &'static str,
    input: fn() -> Input,
    /// The command-line flag.
    flag: &'static str,
    default: &'static str,
    required: bool,
    /// Left out of the command when empty.
    optional: bool,
    /// A toggle that is a bare flag when on.
    switch: bool,
}

const fn flag(
    name: &'static str,
    label: &'static str,
    group: &'static str,
    flag: &'static str,
) -> Flag {
    Flag {
        name,
        label,
        help: None,
        group,
        input: || Input::Text,
        flag,
        default: "",
        required: false,
        optional: true,
        switch: false,
    }
}

struct TableKind {
    key: &'static str,
    label: &'static str,
    description: &'static str,
    subcommand: &'static str,
    /// Words after the subcommand, before the flags.
    verb: &'static [&'static str],
    flags: Vec<Flag>,
    /// The flag whose value is where the job writes, and how it is placed.
    output: Option<&'static str>,
}

impl Kind for TableKind {
    fn key(&self) -> &str {
        self.key
    }

    fn form(&self) -> FormInfo {
        FormInfo {
            key: self.key.to_string(),
            label: self.label.to_string(),
            description: self.description.to_string(),
            subcommand: self.subcommand.to_string(),
            fields: self
                .flags
                .iter()
                .map(|f| FieldInfo {
                    name: f.name.to_string(),
                    label: f.label.to_string(),
                    help: f.help.map(str::to_string),
                    group: f.group.to_string(),
                    input: (f.input)(),
                    required: f.required,
                    list: false,
                    optional: f.optional,
                    default: f.default.to_string(),
                })
                .collect(),
        }
    }

    fn prepare(
        &self,
        values: &BTreeMap<String, String>,
        ctx: &LaunchContext,
    ) -> Result<Launch, Vec<FieldIssue>> {
        let known: HashSet<&str> = self.flags.iter().map(|f| f.name).collect();
        let mut issues: Vec<FieldIssue> = values
            .keys()
            .filter(|name| !known.contains(name.as_str()))
            .map(|name| {
                FieldIssue::new(
                    name,
                    format!("'{name}' is not an option of `pmetal {}`", self.subcommand),
                )
            })
            .collect();

        let mut argv: Vec<String> = self.verb.iter().map(|w| (*w).to_string()).collect();
        let mut output = None;
        let mut title = String::new();
        for f in &self.flags {
            let text = values
                .get(f.name)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| f.default.to_string());
            if text.is_empty() {
                if f.required {
                    issues.push(FieldIssue::new(f.name, "needs a value"));
                }
                continue;
            }
            let input = (f.input)();
            let text = match &input {
                Input::Integer { min, max } => match text.parse::<i64>() {
                    Ok(n) if min.is_none_or(|m| n >= m) && max.is_none_or(|m| n <= m) => text,
                    Ok(_) => {
                        issues.push(FieldIssue::new(f.name, "is outside the allowed range"));
                        continue;
                    }
                    Err(_) => {
                        issues.push(FieldIssue::new(
                            f.name,
                            format!("'{text}' is not a whole number"),
                        ));
                        continue;
                    }
                },
                Input::Number { min, max } => match text.parse::<f64>() {
                    Ok(n)
                        if n.is_finite()
                            && min.is_none_or(|m| n >= m)
                            && max.is_none_or(|m| n <= m) =>
                    {
                        text
                    }
                    Ok(_) => {
                        issues.push(FieldIssue::new(f.name, "is outside the allowed range"));
                        continue;
                    }
                    Err(_) => {
                        issues.push(FieldIssue::new(f.name, format!("'{text}' is not a number")));
                        continue;
                    }
                },
                Input::Choice { options } => {
                    if !options.contains(&text) {
                        issues.push(FieldIssue::new(
                            f.name,
                            format!("must be one of: {}", options.join(", ")),
                        ));
                        continue;
                    }
                    text
                }
                Input::Toggle => match text.to_ascii_lowercase().as_str() {
                    "true" | "on" | "yes" => "true".to_string(),
                    "false" | "off" | "no" => "false".to_string(),
                    _ => {
                        issues.push(FieldIssue::new(
                            f.name,
                            format!("'{text}' is not on or off"),
                        ));
                        continue;
                    }
                },
                _ => text,
            };
            if f.switch {
                if text == "true" {
                    argv.push(f.flag.to_string());
                }
                continue;
            }
            let text = if self.output == Some(f.flag) {
                match paths::resolve_output(&text, &ctx.output_root) {
                    Ok(path) => {
                        output = Some(path.clone());
                        path.display().to_string()
                    }
                    Err(message) => {
                        issues.push(FieldIssue::new(f.name, message));
                        continue;
                    }
                }
            } else {
                text
            };
            if matches!(input, Input::Model) && title.is_empty() {
                title = text.clone();
            }
            argv.push(f.flag.to_string());
            argv.push(text);
        }
        if !issues.is_empty() {
            return Err(issues);
        }
        Ok(Launch {
            subcommand: self.subcommand.to_string(),
            argv,
            output,
            events_file: None,
            metrics_file: None,
            title,
            exclusive: None,
            record: None,
        })
    }
}

fn count(
    name: &'static str,
    label: &'static str,
    flag_name: &'static str,
    default: &'static str,
    help: &'static str,
) -> Flag {
    Flag {
        input: || Input::Integer {
            min: Some(0),
            max: Some(1_048_576),
        },
        default,
        help: Some(help),
        ..flag(name, label, "Workload", flag_name)
    }
}

/// `pmetal bench-workload`: inference and a short LoRA run on a real dataset.
pub fn bench_workload() -> impl Kind {
    TableKind {
        key: "bench_workload",
        label: "Workload benchmark",
        description: "Inference and a short LoRA training run on a real model and dataset",
        subcommand: "bench-workload",
        verb: &[],
        output: Some("--output"),
        flags: vec![
            Flag {
                input: || Input::Choice {
                    options: [
                        "dense-qwen3",
                        "hybrid-qwen3next",
                        "hybrid-qwen35-steady",
                        "moe-nemotronh",
                    ]
                    .map(String::from)
                    .to_vec(),
                },
                help: Some("A named workload; it overrides the model, dataset and sizes below"),
                ..flag("preset", "Preset", "Workload", "--preset")
            },
            Flag {
                input: || Input::Model,
                default: "Qwen/Qwen3-0.6B",
                required: true,
                optional: false,
                ..flag("model", "Model", "Workload", "--model")
            },
            Flag {
                input: || Input::Dataset,
                default: "TeichAI/gemini-3-pro-preview-high-reasoning-250x",
                required: true,
                optional: false,
                ..flag("dataset", "Dataset", "Workload", "--dataset")
            },
            Flag {
                input: || Input::Path,
                ..flag("experts_dir", "Experts Dir", "Workload", "--experts-dir")
            },
            Flag {
                input: || Input::Choice {
                    options: ["auto", "prompt", "text-prefix"].map(String::from).to_vec(),
                },
                default: "auto",
                ..flag(
                    "inference_context",
                    "Inference Context",
                    "Inference",
                    "--inference-context",
                )
            },
            Flag {
                group: "Inference",
                ..count(
                    "prompt_samples",
                    "Prompt Samples",
                    "--prompt-samples",
                    "8",
                    "Prompts to benchmark",
                )
            },
            Flag {
                group: "Inference",
                ..count(
                    "max_prompt_tokens",
                    "Max Prompt Tokens",
                    "--max-prompt-tokens",
                    "0",
                    "0 picks a size from the sampled prompts, capped for a quick run",
                )
            },
            Flag {
                group: "Inference",
                ..count(
                    "decode_steps",
                    "Decode Steps",
                    "--decode-steps",
                    "32",
                    "Tokens decoded per prompt",
                )
            },
            Flag {
                group: "Inference",
                ..count(
                    "inference_warmup_passes",
                    "Warmup Passes",
                    "--inference-warmup-passes",
                    "2",
                    "Untimed passes per prompt before measuring",
                )
            },
            Flag {
                group: "Inference",
                ..count(
                    "inference_session_repeats",
                    "Session Repeats",
                    "--inference-session-repeats",
                    "1",
                    "Independent warmed sessions to aggregate",
                )
            },
            Flag {
                group: "Inference",
                ..count(
                    "inference_repeats",
                    "Timed Passes",
                    "--inference-repeats",
                    "1",
                    "Timed passes per prompt",
                )
            },
            Flag {
                group: "Training",
                ..count(
                    "train_samples",
                    "Train Samples",
                    "--train-samples",
                    "8",
                    "Dataset rows in the training subset",
                )
            },
            Flag {
                group: "Training",
                ..count(
                    "train_steps",
                    "Train Steps",
                    "--train-steps",
                    "4",
                    "LoRA steps to run",
                )
            },
            Flag {
                group: "Training",
                ..count(
                    "batch_size",
                    "Batch Size",
                    "--batch-size",
                    "1",
                    "Training batch size",
                )
            },
            Flag {
                group: "Training",
                ..count(
                    "max_seq_len",
                    "Max Seq Len",
                    "--max-seq-len",
                    "0",
                    "0 picks a length from the sampled data, capped for a quick run",
                )
            },
            Flag {
                input: || Input::Toggle,
                group: "Report",
                switch: true,
                default: "false",
                optional: false,
                ..flag("json", "JSON Report", "Report", "--json")
            },
            Flag {
                input: || Input::Path,
                help: Some("Where to save the JSON report"),
                group: "Report",
                ..flag("output", "Report File", "Report", "--output")
            },
        ],
    }
}

/// `pmetal ollama modelfile`: write a Modelfile for registering a model with
/// an external Ollama.
pub fn ollama_modelfile() -> impl Kind {
    TableKind {
        key: "ollama_modelfile",
        label: "Ollama Modelfile",
        description: "Write a Modelfile to register a model with Ollama",
        subcommand: "ollama",
        verb: &["modelfile"],
        output: Some("--output"),
        flags: vec![
            Flag {
                input: || Input::Model,
                required: true,
                optional: false,
                help: Some("A GGUF file or an Ollama model name"),
                ..flag("base", "Base Model", "Model", "--base")
            },
            Flag {
                input: || Input::Path,
                ..flag("lora", "LoRA Adapter", "Model", "--lora")
            },
            Flag {
                input: || Input::Path,
                default: "Modelfile",
                required: true,
                optional: false,
                ..flag("output", "Modelfile", "Output", "--output")
            },
            Flag {
                input: || Input::Choice {
                    options: ["llama3", "qwen3", "gemma", "mistral", "phi3", "deep-seek"]
                        .map(String::from)
                        .to_vec(),
                },
                help: Some("Chosen from the model's architecture when empty"),
                ..flag("template", "Template", "Prompt", "--template")
            },
            flag("system", "System Prompt", "Prompt", "--system"),
            Flag {
                input: || Input::Number {
                    min: Some(0.0),
                    max: Some(2.0),
                },
                group: "Sampling",
                ..flag("temperature", "Temperature", "Sampling", "--temperature")
            },
            Flag {
                input: || Input::Integer {
                    min: Some(1),
                    max: Some(4_194_304),
                },
                ..flag("num_ctx", "Context Window", "Sampling", "--num-ctx")
            },
            Flag {
                input: || Input::Integer {
                    min: Some(0),
                    max: Some(1_000_000),
                },
                ..flag("top_k", "Top-k", "Sampling", "--top-k")
            },
            Flag {
                input: || Input::Number {
                    min: Some(0.0),
                    max: Some(1.0),
                },
                ..flag("top_p", "Top-p", "Sampling", "--top-p")
            },
            flag("license", "License Text", "Prompt", "--license"),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(root: &std::path::Path) -> LaunchContext {
        LaunchContext {
            output_root: root.to_path_buf(),
            now: chrono::Local::now(),
        }
    }

    fn values(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn a_modelfile_command_names_its_verb_and_places_its_output() {
        let root = tempfile::tempdir().unwrap();
        let launch = ollama_modelfile()
            .prepare(
                &values(&[
                    ("base", "/models/m.gguf"),
                    ("temperature", "0.7"),
                    ("template", "qwen3"),
                ]),
                &ctx(root.path()),
            )
            .unwrap();
        assert_eq!(launch.subcommand, "ollama");
        assert_eq!(&launch.argv[..3], ["modelfile", "--base", "/models/m.gguf"]);
        assert!(
            launch
                .argv
                .windows(2)
                .any(|w| w == ["--temperature", "0.7"])
        );
        assert!(launch.argv.windows(2).any(|w| w == ["--template", "qwen3"]));
        // The default output is a bare "Modelfile": placed under the output root.
        assert_eq!(launch.output.unwrap(), root.path().join("Modelfile"));
        assert!(!launch.argv.iter().any(|a| a == "--top-k"));
    }

    #[test]
    fn a_workload_benchmark_runs_with_the_cli_defaults() {
        let root = tempfile::tempdir().unwrap();
        let launch = bench_workload()
            .prepare(&values(&[]), &ctx(root.path()))
            .unwrap();
        assert_eq!(launch.subcommand, "bench-workload");
        assert!(
            launch
                .argv
                .windows(2)
                .any(|w| w == ["--decode-steps", "32"])
        );
        assert!(
            launch
                .argv
                .windows(2)
                .any(|w| w == ["--model", "Qwen/Qwen3-0.6B"])
        );
        assert!(!launch.argv.iter().any(|a| a == "--json"));
        assert!(launch.output.is_none());
    }

    #[test]
    fn bad_values_are_named_not_passed_on() {
        let root = tempfile::tempdir().unwrap();
        let issues = bench_workload()
            .prepare(
                &values(&[
                    ("train_steps", "lots"),
                    ("inference_context", "sideways"),
                    ("nope", "1"),
                ]),
                &ctx(root.path()),
            )
            .unwrap_err();
        let fields: Vec<&str> = issues.iter().map(|i| i.field.as_str()).collect();
        assert!(fields.contains(&"train_steps"), "{fields:?}");
        assert!(fields.contains(&"inference_context"), "{fields:?}");
        assert!(fields.contains(&"nope"), "{fields:?}");
    }
}
