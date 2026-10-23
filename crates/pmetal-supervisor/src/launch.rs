//! The command a spec stands for, and the files the job is asked to write.
//!
//! Every surface that starts a job (the command line's own tests, the terminal
//! UI, the desktop app, the MCP server, the Python bindings) turns a
//! [`pmetal_core::jobs::Spec`] into a [`LaunchPlan`] here and hands the plan to
//! the [`Supervisor`](crate::Supervisor). What is a fact about the command,
//! not about the surface, lives with the spec: whether it writes an event log
//! (`Spec::REPORTS_EVENTS`), which field is its output (`Spec::OUTPUT_FIELD`),
//! which its metrics file (`Spec::LOG_METRICS_FIELD`). The rules that follow
//! from them are made once, here:
//!
//! - a command that has an output directory and can log its steps does so to
//!   `<output>/metrics.jsonl` unless the spec names another file, and a
//!   relative name is relative to the output directory (as the command itself
//!   resolves it);
//! - a command that writes an event log is asked for `<output>/events.jsonl`
//!   (or `events.jsonl` in a directory the caller names), and then the metrics
//!   file is not also followed, because the events carry the steps and
//!   following both would count each one twice.

use std::path::{Path, PathBuf};

use pmetal_core::jobs::Spec;
use serde_json::Value;

use crate::{Error, JobRequest};

/// File name of the `--log-events` file.
pub const EVENTS_FILE: &str = "events.jsonl";
/// File name of the default `--log-metrics` file.
pub const METRICS_FILE: &str = "metrics.jsonl";

/// Where a job's event log goes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum EventsFile {
    /// In the job's output directory, which is where its other records are.
    #[default]
    InOutput,
    /// In this directory, for a caller that keeps events apart from results (a
    /// scratch directory the caller removes).
    InDir(PathBuf),
    /// Not asked for.
    Off,
}

/// What a surface decides about how a spec is launched.
#[derive(Debug, Clone, Default)]
pub struct LaunchOptions {
    /// Where the event log goes, for a command that writes one.
    pub events: EventsFile,
    /// Publish standard output as it arrives, in pieces (a reply being typed
    /// out), not only as whole lines.
    pub deltas: bool,
}

/// A command ready to hand to the supervisor.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LaunchPlan {
    /// The `pmetal` subcommand.
    pub subcommand: String,
    /// Its arguments, with the files it is asked to write included.
    pub argv: Vec<String>,
    /// Where it writes its results.
    pub output: Option<PathBuf>,
    /// The `--log-events` file the argv asks for, which the supervisor follows.
    pub events_file: Option<PathBuf>,
    /// The `--log-metrics` file the argv asks for, when the steps are not
    /// reported as events.
    pub metrics_file: Option<PathBuf>,
    /// Publish standard output as it arrives.
    pub deltas: bool,
}

impl LaunchPlan {
    /// A plan for a command that is not a spec (a benchmark workload, a
    /// generated Modelfile): its subcommand and arguments, no files.
    pub fn command(subcommand: impl Into<String>, argv: Vec<String>) -> Self {
        Self {
            subcommand: subcommand.into(),
            argv,
            ..Self::default()
        }
    }

    /// The plan for `spec`.
    ///
    /// `spec` is expected to be checked already (`Spec::normalize_spec`) and
    /// its paths settled: the output is taken as it stands.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSpec`] if the spec cannot be read back from its own
    /// serialized form, which is a bug in the spec.
    pub fn for_spec<S: Spec>(spec: &S, options: &LaunchOptions) -> Result<Self, Error> {
        let invalid = |e: serde_json::Error| Error::InvalidSpec(e.to_string());
        let Value::Object(mut fields) = serde_json::to_value(spec).map_err(invalid)? else {
            return Err(Error::InvalidSpec(
                "a spec did not serialize to an object".to_string(),
            ));
        };
        let text_of = |fields: &serde_json::Map<String, Value>, name: &str| {
            fields
                .get(name)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };

        let output = S::OUTPUT_FIELD
            .and_then(|name| text_of(&fields, name))
            .map(PathBuf::from);

        // The steps go to a file beside the results unless the spec names one.
        let mut metrics_file = None;
        if let (Some(name), Some(out)) = (S::LOG_METRICS_FIELD, &output) {
            match text_of(&fields, name) {
                None => {
                    let file = out.join(METRICS_FILE);
                    fields.insert(name.into(), Value::String(file.display().to_string()));
                    metrics_file = Some(file);
                }
                Some(given) => metrics_file = Some(pmetal_core::paths::resolve_in(out, &given)),
            }
        }

        let events_file = match (&options.events, S::REPORTS_EVENTS) {
            (EventsFile::Off, _) | (_, false) => None,
            (EventsFile::InDir(dir), true) => Some(dir.join(EVENTS_FILE)),
            (EventsFile::InOutput, true) => output.as_deref().map(|out| out.join(EVENTS_FILE)),
        };

        let spec: S = serde_json::from_value(Value::Object(fields)).map_err(invalid)?;
        let mut argv = spec.to_argv();
        if let Some(file) = &events_file {
            argv.extend(["--log-events".to_string(), file.display().to_string()]);
            // The events carry the steps; following the metrics file too would
            // count each step twice.
            metrics_file = None;
        }

        Ok(Self {
            subcommand: S::subcommand().to_string(),
            argv,
            output,
            events_file,
            metrics_file,
            deltas: options.deltas,
        })
    }

    /// The command line as a person would type it, for a log's first line.
    /// Credentials are hidden.
    pub fn command_line(&self) -> String {
        let mut words = vec!["pmetal".to_string(), self.subcommand.clone()];
        words.extend(self.argv.iter().cloned());
        crate::redact_argv(&words).join(" ")
    }

    /// The request the supervisor starts, with `env` set for the job.
    pub fn into_request_with_env(self, env: Vec<(String, String)>) -> JobRequest {
        JobRequest {
            env,
            ..JobRequest::from(self)
        }
    }

    /// Where the job writes its results, if it says.
    pub fn output(&self) -> Option<&Path> {
        self.output.as_deref()
    }
}

impl From<LaunchPlan> for JobRequest {
    fn from(plan: LaunchPlan) -> Self {
        JobRequest {
            subcommand: plan.subcommand,
            argv: plan.argv,
            output: plan.output,
            metrics_file: plan.metrics_file,
            events_file: plan.events_file,
            deltas: plan.deltas,
            env: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_core::jobs::{
        BenchSpec, DecideSpec, DflashSpec, DistillSpec, EmbedTrainSpec, EvalSpec, FuseSpec,
        GrpoSpec, ImatrixSpec, InferSpec, MergeSpec, PackExpertsSpec, PreferenceSpec, PretrainSpec,
        QuantizeSpec, RlkdSpec, ServeSpec, TokenizeSpec, TrainDiffusionSpec, TrainDraftSpec,
        TrainMtpSpec, TrainSpec, VocodeSpec,
    };

    fn arg_after<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
        let at = argv.iter().position(|a| a == flag)?;
        argv.get(at + 1).map(String::as_str)
    }

    fn train(out: &str) -> TrainSpec {
        TrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            output_dir: out.into(),
            ..TrainSpec::default()
        }
    }

    #[test]
    fn a_trainer_writes_events_and_metrics_beside_its_results() {
        let plan = LaunchPlan::for_spec(&train("/runs/a"), &LaunchOptions::default()).unwrap();
        assert_eq!(plan.subcommand, "train");
        assert_eq!(plan.output.as_deref(), Some(Path::new("/runs/a")));
        assert_eq!(
            plan.events_file.as_deref(),
            Some(Path::new("/runs/a/events.jsonl"))
        );
        assert_eq!(
            arg_after(&plan.argv, "--log-events"),
            Some("/runs/a/events.jsonl")
        );
        assert_eq!(
            arg_after(&plan.argv, "--log-metrics"),
            Some("/runs/a/metrics.jsonl")
        );
        // The steps come with the events; the metrics file is not read as well.
        assert_eq!(plan.metrics_file, None);
    }

    #[test]
    fn a_command_with_no_event_log_follows_its_metrics_file() {
        // Quantize has an output but writes no event log and takes no
        // `--log-metrics`: nothing to follow.
        let plan = LaunchPlan::for_spec(
            &QuantizeSpec {
                output: "/runs/q".into(),
                ..QuantizeSpec::default()
            },
            &LaunchOptions::default(),
        )
        .unwrap();
        assert_eq!(plan.events_file, None);
        assert_eq!(plan.metrics_file, None);
        assert!(!plan.argv.contains(&"--log-events".to_string()));
        assert!(!plan.argv.contains(&"--log-metrics".to_string()));
    }

    #[test]
    fn events_can_be_kept_apart_from_the_results_or_not_asked_for() {
        let apart = LaunchPlan::for_spec(
            &train("/runs/a"),
            &LaunchOptions {
                events: EventsFile::InDir("/scratch".into()),
                ..LaunchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            apart.events_file.as_deref(),
            Some(Path::new("/scratch/events.jsonl"))
        );
        let off = LaunchPlan::for_spec(
            &train("/runs/a"),
            &LaunchOptions {
                events: EventsFile::Off,
                ..LaunchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(off.events_file, None);
        assert!(!off.argv.contains(&"--log-events".to_string()));
        // Without events the metrics file is the progress record.
        assert_eq!(
            off.metrics_file.as_deref(),
            Some(Path::new("/runs/a/metrics.jsonl"))
        );
    }

    #[test]
    fn a_named_metrics_file_is_relative_to_the_output_directory() {
        let mut spec = train("/runs/a");
        spec.log_metrics = Some("logs/steps.jsonl".into());
        let plan = LaunchPlan::for_spec(
            &spec,
            &LaunchOptions {
                events: EventsFile::Off,
                ..LaunchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            plan.metrics_file.as_deref(),
            Some(Path::new("/runs/a/logs/steps.jsonl"))
        );
        // The command is told what the spec said; it resolves it the same way.
        assert_eq!(
            arg_after(&plan.argv, "--log-metrics"),
            Some("logs/steps.jsonl")
        );

        spec.log_metrics = Some("/elsewhere/steps.jsonl".into());
        let plan = LaunchPlan::for_spec(&spec, &LaunchOptions::default()).unwrap();
        assert!(plan.events_file.is_some());
        let plan = LaunchPlan::for_spec(
            &spec,
            &LaunchOptions {
                events: EventsFile::Off,
                ..LaunchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            plan.metrics_file.as_deref(),
            Some(Path::new("/elsewhere/steps.jsonl"))
        );
    }

    #[test]
    fn the_request_carries_what_the_plan_says() {
        let plan = LaunchPlan {
            deltas: true,
            ..LaunchPlan::for_spec(&train("/runs/a"), &LaunchOptions::default()).unwrap()
        };
        let request = plan
            .clone()
            .into_request_with_env(vec![("K".into(), "v".into())]);
        assert_eq!(request.subcommand, "train");
        assert_eq!(request.argv, plan.argv);
        assert_eq!(request.output, plan.output);
        assert_eq!(request.events_file, plan.events_file);
        assert_eq!(request.metrics_file, plan.metrics_file);
        assert!(request.deltas);
        assert_eq!(request.env, vec![("K".to_string(), "v".to_string())]);
    }

    #[test]
    fn the_command_line_hides_credentials() {
        let plan = LaunchPlan::command(
            "serve",
            vec![
                "--model".into(),
                "m".into(),
                "--api-key".into(),
                "sk-secret".into(),
            ],
        );
        let line = plan.command_line();
        assert!(line.starts_with("pmetal serve --model m"), "{line}");
        assert!(!line.contains("sk-secret"), "{line}");
    }

    /// The facts every surface relies on, for every spec: which commands write
    /// an event log, and that the files a plan names are the ones its argv asks
    /// for.
    #[test]
    fn every_spec_agrees_with_its_plan() {
        fn check<S: Spec>(spec: S, writes_events: bool) {
            assert_eq!(S::REPORTS_EVENTS, writes_events, "{}", S::subcommand());
            let json = serde_json::to_value(&spec).unwrap();
            for field in [S::OUTPUT_FIELD, S::LOG_METRICS_FIELD]
                .into_iter()
                .flatten()
            {
                assert!(
                    json.get(field).is_some(),
                    "{}: no field {field}",
                    S::subcommand()
                );
            }
            let options = LaunchOptions {
                events: EventsFile::InDir("/scratch".into()),
                ..LaunchOptions::default()
            };
            let plan = LaunchPlan::for_spec(&spec, &options).unwrap();
            assert_eq!(plan.subcommand, S::subcommand());
            assert_eq!(
                plan.events_file.is_some(),
                writes_events,
                "{}",
                S::subcommand()
            );
            assert_eq!(
                arg_after(&plan.argv, "--log-events").map(Path::new),
                plan.events_file.as_deref(),
                "{}",
                S::subcommand()
            );
            // The steps are followed in one place only.
            assert!(
                !(plan.events_file.is_some() && plan.metrics_file.is_some()),
                "{}",
                S::subcommand()
            );
        }

        check(TrainSpec::default(), true);
        check(DistillSpec::default(), true);
        check(GrpoSpec::default(), true);
        check(RlkdSpec::default(), true);
        check(PreferenceSpec::default(), true);
        check(BenchSpec::default(), false);
        check(DecideSpec::default(), false);
        check(DflashSpec::default(), false);
        check(EmbedTrainSpec::default(), true);
        check(EvalSpec::default(), false);
        check(FuseSpec::default(), false);
        check(ImatrixSpec::default(), false);
        check(InferSpec::default(), false);
        check(MergeSpec::default(), false);
        check(PackExpertsSpec::default(), false);
        check(PretrainSpec::default(), true);
        check(QuantizeSpec::default(), false);
        check(ServeSpec::default(), false);
        check(TokenizeSpec::default(), false);
        check(TrainDiffusionSpec::default(), true);
        check(TrainDraftSpec::default(), true);
        check(TrainMtpSpec::default(), true);
        check(VocodeSpec::default(), false);
    }
}
