//! The jobs the app can start, and how a form's text becomes a command.
//!
//! Each job is a `pmetal-core` spec. A form sends the text entered for the
//! spec's fields; [`spec_from_text`] builds the spec the command line would
//! build from the same values, the spec's own `normalize` checks it, and
//! `to_argv` writes the command. This module adds only what the app decides:
//! where the job writes, whether a path is one it may write to, and which
//! files the supervisor follows for progress.

use std::collections::{BTreeMap, HashSet};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::OnceLock;

use pmetal::core::jobs::{
    BenchSpec, DecideSpec, DflashSpec, DistillSpec, EmbedTrainSpec, EvalSpec, FuseSpec, GrpoSpec,
    ImatrixSpec, MergeSpec, PackExpertsSpec, PreferenceSpec, PretrainSpec, QuantizeSpec, RlkdSpec,
    ServeSpec, Spec, TokenizeSpec, TrainDiffusionSpec, TrainDraftSpec, TrainMtpSpec, TrainSpec,
    VocodeSpec,
};
use pmetal::core::{FieldKind, JobFields, default_text, spec_from_text};
use serde::Serialize;
use serde_json::{Map, Value};

use super::form::{FormInfo, form_of};
use super::local;
use crate::paths;

/// Something wrong with one field of a form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FieldIssue {
    /// The field's name, or empty for a problem with the form as a whole.
    pub field: String,
    pub message: String,
}

impl FieldIssue {
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

/// What the app decides about where and when a job runs.
pub struct LaunchContext {
    /// Where relative and generated output locations go.
    pub output_root: PathBuf,
    /// For naming generated output locations.
    pub now: chrono::DateTime<chrono::Local>,
}

/// A command ready to hand to the supervisor.
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    pub subcommand: String,
    pub argv: Vec<String>,
    /// Where the job writes its results.
    pub output: Option<PathBuf>,
    /// The `--log-events` file the job was told to write.
    pub events_file: Option<PathBuf>,
    /// The `--log-metrics` file the job was told to write, when it reports no
    /// events.
    pub metrics_file: Option<PathBuf>,
    /// What to call the run: the model it works on.
    pub title: String,
    /// Two jobs with the same key would collide (a port both would bind).
    pub exclusive: Option<String>,
    /// What to leave beside the job's results (`training_info.json`), so the
    /// adapter list can say what an adapter was trained from.
    pub record: Option<Value>,
}

/// A kind of job.
pub trait Kind: Send + Sync {
    fn key(&self) -> &str;
    fn form(&self) -> FormInfo;
    /// Check `values` (the text entered for fields, by field name) and build
    /// the command.
    fn prepare(
        &self,
        values: &BTreeMap<String, String>,
        ctx: &LaunchContext,
    ) -> Result<Launch, Vec<FieldIssue>>;
}

/// Commands whose `--log-events` the CLI honours, which is how the app learns
/// what phase they are in and when a step finishes.
const REPORTS_EVENTS: &[&str] = &["train", "distill", "grpo", "rlkd", "preference"];

struct SpecKind<S> {
    key: &'static str,
    label: &'static str,
    description: &'static str,
    /// A spec field whose value two running jobs must not share.
    exclusive: Option<&'static str>,
    _spec: PhantomData<fn() -> S>,
}

fn spec_kind<S: Spec>(
    key: &'static str,
    label: &'static str,
    description: &'static str,
) -> Box<dyn Kind> {
    Box::new(SpecKind::<S> {
        key,
        label,
        description,
        exclusive: None,
        _spec: PhantomData,
    })
}

fn exclusive_kind<S: Spec>(
    key: &'static str,
    label: &'static str,
    description: &'static str,
    field: &'static str,
) -> Box<dyn Kind> {
    Box::new(SpecKind::<S> {
        key,
        label,
        description,
        exclusive: Some(field),
        _spec: PhantomData,
    })
}

/// The name a run goes by: the first model the spec names.
fn title_of<S: JobFields>(json: &Map<String, Value>) -> String {
    S::field_descriptors()
        .iter()
        .filter(|d| matches!(d.kind, FieldKind::ModelPicker))
        .filter_map(|d| json.get(d.name)?.as_str())
        .map(str::trim)
        .find(|v| !v.is_empty())
        .unwrap_or_default()
        .to_string()
}

/// The last part of a model reference, as a folder name.
fn short_name(title: &str) -> String {
    let last = title
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(title);
    let cleaned: String = last
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    cleaned.trim_matches(['-', '.']).to_string()
}

/// A new folder name under the output root: `<model>-<job>-<date>`, with a
/// number added if that is taken.
fn generated_output(ctx: &LaunchContext, key: &str, title: &str) -> PathBuf {
    let stamp = ctx.now.format("%Y%m%d-%H%M%S");
    let short = short_name(title);
    let base = if short.is_empty() {
        format!("{key}-{stamp}")
    } else {
        format!("{short}-{key}-{stamp}")
    };
    let mut candidate = ctx.output_root.join(&base);
    let mut n = 2;
    while candidate.exists() {
        candidate = ctx.output_root.join(format!("{base}-{n}"));
        n += 1;
    }
    candidate
}

impl<S: Spec> Kind for SpecKind<S> {
    fn key(&self) -> &str {
        self.key
    }

    fn form(&self) -> FormInfo {
        let mut form = form_of::<S>(self.label, self.description);
        form.key = self.key.to_string();
        form
    }

    fn prepare(
        &self,
        values: &BTreeMap<String, String>,
        ctx: &LaunchContext,
    ) -> Result<Launch, Vec<FieldIssue>> {
        let descriptors = S::field_descriptors();

        // A name the spec does not have is a control the spec cannot drive.
        // It is refused, not dropped, so a form that is out of date says so.
        let known: HashSet<&str> = descriptors.iter().map(|d| d.name).collect();
        let unknown: Vec<FieldIssue> = values
            .keys()
            .filter(|name| !known.contains(name.as_str()))
            .map(|name| {
                FieldIssue::new(
                    name,
                    format!("'{name}' is not an option of `pmetal {}`", S::subcommand()),
                )
            })
            .collect();
        if !unknown.is_empty() {
            return Err(unknown);
        }

        let spec: S = spec_from_text(|d| values.get(d.name).cloned()).map_err(|errors| {
            errors
                .into_iter()
                .map(|e| FieldIssue::new(e.field, e.message))
                .collect::<Vec<_>>()
        })?;

        let mut json = match serde_json::to_value(&spec) {
            Ok(Value::Object(map)) => map,
            _ => {
                return Err(vec![FieldIssue::new(
                    "",
                    "the job's settings could not be read back (this is a bug in PMetal)",
                )]);
            }
        };
        let title = title_of::<S>(&json);
        let defaults = default_text::<S>();
        let default_of = |name: &str| {
            defaults
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, text)| text.as_str())
                .unwrap_or("")
        };
        let mut issues = Vec::new();

        // Where the job writes. Not typed, or still the command line's
        // placeholder: a new folder of its own. Typed: that place, if it is one
        // the app lets a job write to.
        let mut output: Option<PathBuf> = None;
        if let Some(d) = descriptors.iter().find(|d| d.argv == Some("--output")) {
            let given = json
                .get(d.name)
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let default = default_of(d.name);
            let resolved = if given.is_empty() && default.is_empty() {
                if d.required {
                    Err("needs a location".to_string())
                } else {
                    Ok(None)
                }
            } else if given.is_empty() || given == default {
                Ok(Some(generated_output(ctx, self.key, &title)))
            } else {
                paths::resolve_output(&given, &ctx.output_root).map(Some)
            };
            match resolved {
                Ok(Some(path)) => {
                    json.insert(d.name.into(), Value::String(path.display().to_string()));
                    output = Some(path);
                }
                Ok(None) => {}
                Err(message) => issues.push(FieldIssue::new(d.name, message)),
            }
        }

        // The job must not write over, into or around something it reads.
        if let Some(out) = &output {
            for d in descriptors.iter().filter(|d| d.argv != Some("--output")) {
                if !matches!(
                    d.kind,
                    FieldKind::Path | FieldKind::ModelPicker | FieldKind::DatasetPicker
                ) {
                    continue;
                }
                let Some(text) = json.get(d.name).and_then(Value::as_str) else {
                    continue;
                };
                let input = paths::expand_home(text);
                // Only a full path can be a place on disk; "Qwen/Qwen3-0.6B" is
                // a Hub id, whatever the working directory holds.
                if input.is_absolute() && input.exists() && paths::overlaps(out, &input) {
                    issues.push(FieldIssue::new(
                        "output",
                        format!(
                            "{} is {}. Choose a different output location, so the original is not overwritten.",
                            out.display(),
                            if input.starts_with(out) {
                                format!("a folder that contains the {}", d.label.to_lowercase())
                            } else {
                                format!("the {} or inside it", d.label.to_lowercase())
                            }
                        ),
                    ));
                }
            }
        }

        // The file the job logs its steps to, if it has one, kept beside its
        // results.
        let mut metrics_file = None;
        if let Some(value) = json.get("log_metrics") {
            let given = value.as_str().unwrap_or("").trim().to_string();
            let path = if given.is_empty() {
                output.as_ref().map(|out| out.join("metrics.jsonl"))
            } else {
                match paths::resolve_output(&given, &ctx.output_root) {
                    Ok(path) => Some(path),
                    Err(message) => {
                        issues.push(FieldIssue::new("log_metrics", message));
                        None
                    }
                }
            };
            if let Some(path) = path {
                json.insert(
                    "log_metrics".into(),
                    Value::String(path.display().to_string()),
                );
                metrics_file = Some(path);
            }
        }

        if !issues.is_empty() {
            return Err(issues);
        }

        let mut spec: S = serde_json::from_value(Value::Object(json.clone())).map_err(|e| {
            vec![FieldIssue::new(
                "",
                format!("the job's settings are not valid: {e}"),
            )]
        })?;
        spec.normalize_spec().map_err(|errors| {
            errors
                .into_iter()
                .map(|e| FieldIssue::new(e.field, e.message))
                .collect::<Vec<_>>()
        })?;

        let mut argv = spec.to_argv();
        let mut events_file = None;
        if REPORTS_EVENTS.contains(&S::subcommand())
            && let Some(out) = &output
        {
            let file = out.join("events.jsonl");
            argv.extend(["--log-events".to_string(), file.display().to_string()]);
            events_file = Some(file);
            // The events carry the steps; following the metrics file too would
            // count each step twice.
            metrics_file = None;
        }

        let record = (events_file.is_some()).then(|| {
            serde_json::json!({
                "base_model": title,
                "dataset": json.get("dataset").and_then(Value::as_str),
                "created": ctx.now.to_rfc3339(),
            })
        });

        let exclusive = self.exclusive.and_then(|field| {
            let value = json.get(field)?;
            Some(format!(
                "{field}:{}",
                value
                    .as_str()
                    .map_or_else(|| value.to_string(), str::to_string)
            ))
        });

        Ok(Launch {
            subcommand: S::subcommand().to_string(),
            argv,
            output,
            events_file,
            metrics_file,
            title,
            exclusive,
            record,
        })
    }
}

/// Every job the app can start, in the order its forms are listed.
pub fn all() -> &'static [Box<dyn Kind>] {
    static KINDS: OnceLock<Vec<Box<dyn Kind>>> = OnceLock::new();
    KINDS.get_or_init(|| {
        vec![
            spec_kind::<TrainSpec>(
                "train",
                "Fine-tune",
                "LoRA, QLoRA or full-parameter fine-tuning on the GPU or Neural Engine",
            ),
            spec_kind::<DistillSpec>(
                "distill",
                "Distillation",
                "Train a small student model to match a larger teacher",
            ),
            spec_kind::<GrpoSpec>(
                "grpo",
                "GRPO",
                "Reinforcement learning with group-relative rewards",
            ),
            spec_kind::<RlkdSpec>(
                "rlkd",
                "RLKD",
                "Reinforcement learning with a teacher's distillation signal",
            ),
            spec_kind::<PreferenceSpec>(
                "preference",
                "Preference tuning",
                "DPO, IPO, hinge, SimPO, ORPO and KTO on preference data",
            ),
            spec_kind::<EmbedTrainSpec>(
                "embed_train",
                "Embedding training",
                "Contrastive training of a sentence-embedding model",
            ),
            spec_kind::<PretrainSpec>(
                "pretrain",
                "Pretraining",
                "Train a model from scratch on tokenized shards",
            ),
            spec_kind::<TrainMtpSpec>(
                "train_mtp",
                "MTP predictor",
                "Train a multi-token-prediction head or assistant",
            ),
            spec_kind::<TrainDraftSpec>(
                "train_draft",
                "DFlash draft",
                "Train a block-diffusion draft model for speculative decoding",
            ),
            spec_kind::<TrainDiffusionSpec>(
                "train_diffusion",
                "Diffusion LoRA",
                "Fine-tune a block-diffusion language model with LoRA",
            ),
            spec_kind::<DflashSpec>(
                "dflash",
                "DFlash decoding",
                "Generate with block-diffusion speculative decoding",
            ),
            exclusive_kind::<ServeSpec>(
                "serve",
                "Serve",
                "An OpenAI-compatible HTTP server for a model",
                "port",
            ),
            spec_kind::<BenchSpec>("bench", "Benchmark", "Measure training throughput"),
            Box::new(local::bench_workload()),
            spec_kind::<EvalSpec>("eval", "Evaluate", "Perplexity of a model on a dataset"),
            spec_kind::<MergeSpec>(
                "merge",
                "Merge models",
                "Combine the weights of several models",
            ),
            spec_kind::<FuseSpec>(
                "fuse",
                "Fuse adapter",
                "Fold a LoRA adapter into its base model's weights",
            ),
            spec_kind::<QuantizeSpec>(
                "quantize",
                "Quantize",
                "Convert a model to GGUF or MLX quantized weights",
            ),
            spec_kind::<ImatrixSpec>(
                "imatrix",
                "Importance matrix",
                "Measure which weights a model's activations lean on, to guide quantization",
            ),
            spec_kind::<PackExpertsSpec>(
                "pack_experts",
                "Pack experts",
                "Pack a mixture-of-experts model's routed experts into one file",
            ),
            spec_kind::<TokenizeSpec>("tokenize", "Tokenize", "Tokenize text into training shards"),
            spec_kind::<VocodeSpec>(
                "vocode",
                "Vocode",
                "Synthesize audio from a mel spectrogram",
            ),
            spec_kind::<DecideSpec>("decide", "Decide", "Typed decisions from a decision model"),
            Box::new(local::ollama_modelfile()),
        ]
    })
}

/// The job whose key is `key`.
pub fn find(key: &str) -> Option<&'static dyn Kind> {
    all().iter().find(|k| k.key() == key).map(Box::as_ref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal::core::JobKind;

    fn ctx(root: &std::path::Path) -> LaunchContext {
        LaunchContext {
            output_root: root.to_path_buf(),
            now: chrono::TimeZone::with_ymd_and_hms(&chrono::Local, 2026, 3, 18, 21, 45, 7)
                .unwrap(),
        }
    }

    fn values(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn arg_after<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
        let at = argv.iter().position(|a| a == flag)?;
        argv.get(at + 1).map(String::as_str)
    }

    #[test]
    fn every_spec_has_exactly_one_job_under_its_own_key() {
        let keys: Vec<&str> = all().iter().map(|k| k.key()).collect();
        let unique: HashSet<&str> = keys.iter().copied().collect();
        assert_eq!(keys.len(), unique.len(), "{keys:?}");

        // A spec added to pmetal-core without a job here is a form the app
        // cannot show: say so.
        for kind in [
            JobKind::Train,
            JobKind::TrainMtp,
            JobKind::TrainDraft,
            JobKind::TrainDiffusion,
            JobKind::Distill,
            JobKind::Grpo,
            JobKind::Rlkd,
            JobKind::Preference,
            JobKind::EmbedTrain,
            JobKind::Pretrain,
            JobKind::Serve,
            JobKind::Decide,
            JobKind::Bench,
            JobKind::Eval,
            JobKind::Quantize,
            JobKind::Imatrix,
            JobKind::Merge,
            JobKind::Fuse,
            JobKind::Dflash,
            JobKind::PackExperts,
            JobKind::Tokenize,
            JobKind::Vocode,
        ] {
            let key = serde_json::to_value(kind).unwrap();
            assert!(
                keys.contains(&key.as_str().unwrap()),
                "no job for the {key} spec"
            );
        }
    }

    #[test]
    fn a_form_is_listed_under_the_key_a_job_is_started_by() {
        for kind in all() {
            assert_eq!(kind.form().key, kind.key());
        }
    }

    #[test]
    fn a_training_form_becomes_the_command_the_cli_would_run() {
        let root = tempfile::tempdir().unwrap();
        let launch = find("train")
            .unwrap()
            .prepare(
                &values(&[
                    ("model", "Qwen/Qwen3-0.6B"),
                    ("dataset", "/data/train.jsonl"),
                    ("epochs", "2"),
                    ("lora_r", "8"),
                ]),
                &ctx(root.path()),
            )
            .unwrap();

        assert_eq!(launch.subcommand, "train");
        assert_eq!(arg_after(&launch.argv, "--model"), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(arg_after(&launch.argv, "--epochs"), Some("2"));
        assert_eq!(arg_after(&launch.argv, "--lora-r"), Some("8"));
        // Not entered, so the spec's own default, not a number of the app's.
        let default = TrainSpec::default();
        assert_eq!(
            arg_after(&launch.argv, "--batch-size").map(str::to_string),
            Some(default.batch_size.to_string())
        );
        // A folder of its own, named for the model, the job and the time.
        let out = launch.output.unwrap();
        assert_eq!(out, root.path().join("Qwen3-0.6B-train-20260318-214507"));
        assert_eq!(
            arg_after(&launch.argv, "--output"),
            Some(out.to_str().unwrap())
        );
        // Progress comes from the events file, and the steps are not counted twice.
        let events = launch.events_file.unwrap();
        assert_eq!(events, out.join("events.jsonl"));
        assert_eq!(
            arg_after(&launch.argv, "--log-events"),
            Some(events.to_str().unwrap())
        );
        assert!(launch.metrics_file.is_none());
        assert_eq!(launch.title, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn a_name_the_spec_does_not_have_is_refused_not_dropped() {
        let root = tempfile::tempdir().unwrap();
        let issues = find("train")
            .unwrap()
            .prepare(
                &values(&[("model", "m"), ("dataset", "d"), ("load_in_4bit", "true")]),
                &ctx(root.path()),
            )
            .unwrap_err();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].field, "load_in_4bit");
        assert!(issues[0].message.contains("pmetal train"));
    }

    #[test]
    fn problems_are_reported_against_the_field_that_has_them() {
        let root = tempfile::tempdir().unwrap();
        let issues = find("train")
            .unwrap()
            .prepare(
                &values(&[("model", ""), ("dataset", "d"), ("epochs", "many")]),
                &ctx(root.path()),
            )
            .unwrap_err();
        assert!(issues.iter().any(|i| i.field == "epochs"), "{issues:?}");
    }

    #[test]
    fn an_output_that_climbs_out_or_names_a_system_folder_is_refused() {
        let root = tempfile::tempdir().unwrap();
        for bad in ["../elsewhere", "/System/Library/x", "/"] {
            let issues = find("train")
                .unwrap()
                .prepare(
                    &values(&[("model", "m"), ("dataset", "d"), ("output_dir", bad)]),
                    &ctx(root.path()),
                )
                .unwrap_err();
            assert!(
                issues.iter().any(|i| i.field == "output_dir"),
                "{bad}: {issues:?}"
            );
        }
    }

    #[test]
    fn a_relative_output_is_placed_under_the_output_root() {
        let root = tempfile::tempdir().unwrap();
        let launch = find("train")
            .unwrap()
            .prepare(
                &values(&[("model", "m"), ("dataset", "d"), ("output_dir", "my-run")]),
                &ctx(root.path()),
            )
            .unwrap();
        assert_eq!(launch.output.unwrap(), root.path().join("my-run"));
    }

    #[test]
    fn fusing_into_the_model_folder_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("base");
        std::fs::create_dir_all(&model).unwrap();
        let issues = find("fuse")
            .unwrap()
            .prepare(
                &values(&[
                    ("model", model.to_str().unwrap()),
                    ("lora", "/adapters/a"),
                    ("output", model.to_str().unwrap()),
                ]),
                &ctx(root.path()),
            )
            .unwrap_err();
        assert!(
            issues
                .iter()
                .any(|i| i.field == "output" && i.message.contains("overwritten")),
            "{issues:?}"
        );
        // A folder beside it is fine.
        assert!(
            find("fuse")
                .unwrap()
                .prepare(
                    &values(&[
                        ("model", model.to_str().unwrap()),
                        ("lora", "/adapters/a"),
                        ("output", root.path().join("fused").to_str().unwrap()),
                    ]),
                    &ctx(root.path()),
                )
                .is_ok()
        );
    }

    #[test]
    fn a_job_that_names_no_output_still_needs_one_when_the_spec_requires_it() {
        let root = tempfile::tempdir().unwrap();
        let issues = find("merge")
            .unwrap()
            .prepare(&values(&[]), &ctx(root.path()))
            .unwrap_err();
        assert!(issues.iter().any(|i| i.field == "output"), "{issues:?}");
    }

    #[test]
    fn a_server_is_exclusive_on_its_port() {
        let root = tempfile::tempdir().unwrap();
        let launch = find("serve")
            .unwrap()
            .prepare(
                &values(&[("model", "m"), ("port", "9001")]),
                &ctx(root.path()),
            )
            .unwrap();
        assert_eq!(launch.exclusive.as_deref(), Some("port:9001"));
        assert!(launch.output.is_none());
        assert!(launch.events_file.is_none());
    }

    #[test]
    fn short_names_are_safe_folder_names() {
        assert_eq!(short_name("Qwen/Qwen3-0.6B"), "Qwen3-0.6B");
        assert_eq!(short_name("/models/my model/"), "my-model");
        assert_eq!(short_name("a b/../c d"), "c-d");
        assert_eq!(short_name(""), "");
    }
}
