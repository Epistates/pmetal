//! Every surface that starts a job agrees about it, for every spec.
//!
//! Whether a command writes an event log is a fact of its spec
//! (`#[spec(events)]`). The command line refuses `--log-events` for the others,
//! the terminal UI, the MCP server and the Python bindings ask for the file
//! only where it is written, and the desktop app's own test makes the same
//! table. Each row here is one spec; the columns are what each surface says.

use clap::Parser;
use pmetal_core::jobs::{
    BenchSpec, DecideSpec, DflashSpec, DistillSpec, EmbedTrainSpec, EvalSpec, FuseSpec, GrpoSpec,
    InferSpec, MergeSpec, PackExpertsSpec, PreferenceSpec, PretrainSpec, QuantizeSpec, RlkdSpec,
    ServeSpec, Spec, TokenizeSpec, TrainDiffusionSpec, TrainDraftSpec, TrainMtpSpec, TrainSpec,
    VocodeSpec,
};
use pmetal_core::spec_from_text;
use pmetal_mcp::policy::PathPolicy;
use serde_json::{Value, json};

use super::Cli;
use crate::tui::tabs::panel::JobPanel;

/// What every surface says about one spec.
#[derive(Debug, PartialEq, Eq)]
struct Row {
    subcommand: &'static str,
    /// The spec's own statement.
    spec: bool,
    /// Whether the command line accepts `--log-events` for it.
    cli: bool,
    /// Whether the terminal UI's plan asks for the event log.
    tui: bool,
    /// Whether the MCP tool's plan asks for it (`None`: the tool runs the
    /// command to completion and follows nothing).
    mcp: Option<bool>,
}

/// Whether a sample gives `d` a value: every field that must have one, and the
/// one that `infer` and `eval` need though no descriptor marks it required (a
/// prompt or a messages file; a dataset or a task).
fn needs_value<S: Spec>(d: &pmetal_core::FieldDescriptor) -> bool {
    d.required
        || matches!(
            (S::subcommand(), d.name),
            ("infer", "prompt") | ("eval", "dataset")
        )
}

/// The spec with a value in every field that must have one.
fn sample<S: Spec>() -> S {
    spec_from_text::<S>(|d| needs_value::<S>(d).then(|| "owner/name".to_string()))
        .unwrap_or_else(|e| panic!("{}: {e:?}", S::subcommand()))
}

fn row<S: Spec>(tools: &[(&'static str, Option<bool>)]) -> Row {
    let spec = sample::<S>();

    // The command line: the argv the spec spells, with the flag after it.
    let mut argv = vec!["pmetal".to_string(), S::subcommand().to_string()];
    argv.extend(spec.to_argv());
    argv.extend(["--log-events".to_string(), "events.jsonl".to_string()]);
    let cli = Cli::try_parse_from(&argv)
        .unwrap_or_else(|e| panic!("{}: {e}", S::subcommand()))
        .command
        .writes_event_log();

    // The terminal UI: the form's own plan.
    let mut panel = JobPanel::new::<S>("T");
    for d in S::field_descriptors()
        .iter()
        .filter(|d| needs_value::<S>(d))
    {
        panel.form.set_value(d.label, "owner/name");
    }
    let plan = panel
        .plan_for::<S>()
        .unwrap_or_else(|e| panic!("{}: {e:?}", S::subcommand()));
    let tui = plan.events_file.is_some();

    let mcp = tools
        .iter()
        .find(|(subcommand, _)| *subcommand == S::subcommand())
        .and_then(|(_, events)| *events);

    Row {
        subcommand: S::subcommand(),
        spec: S::REPORTS_EVENTS,
        cli,
        tui,
        mcp,
    }
}

/// `(subcommand, whether its plan has an event log)` for each tool that
/// starts a job; a tool that runs to completion is not in the list's `Some`.
fn mcp_tools() -> Vec<(&'static str, Option<bool>)> {
    let root = std::env::temp_dir().canonicalize().unwrap();
    let policy = PathPolicy::new(root.clone(), vec![root]);
    let mut found = Vec::new();
    for tool in pmetal_mcp::spec_tool_list() {
        let definition = serde_json::to_value(tool.definition()).unwrap();
        let mut args: serde_json::Map<String, Value> = definition["inputSchema"]["required"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|name| {
                let name = name.as_str().unwrap();
                let list = definition["inputSchema"]["properties"][name]["type"] == "array";
                let value = if list {
                    json!(["owner/name"])
                } else {
                    json!("owner/name")
                };
                (name.to_string(), value)
            })
            .collect();
        // `infer` takes a prompt or a messages file, so neither is required.
        if tool.name() == "generate" {
            args.insert("prompt".to_string(), json!("hello"));
        }
        // `eval` takes a dataset (perplexity) or a task, so neither is required.
        if tool.name() == "eval_perplexity" {
            args.insert("dataset".to_string(), json!("owner/name"));
        }
        let prepared = tool
            .prepare(Value::Object(args), &policy)
            .unwrap_or_else(|e| panic!("{}: {e}", tool.name()));
        let follows = matches!(tool.mode(), pmetal_mcp::spec_tools::Mode::Job)
            .then(|| prepared.plan.events_file.is_some());
        let subcommand: &'static str = Box::leak(prepared.plan.subcommand.into_boxed_str());
        found.push((subcommand, follows));
    }
    found
}

#[test]
fn every_surface_agrees_which_commands_write_an_event_log() {
    let tools = mcp_tools();
    let rows = vec![
        row::<TrainSpec>(&tools),
        row::<DistillSpec>(&tools),
        row::<GrpoSpec>(&tools),
        row::<RlkdSpec>(&tools),
        row::<PreferenceSpec>(&tools),
        row::<EmbedTrainSpec>(&tools),
        row::<PretrainSpec>(&tools),
        row::<TrainMtpSpec>(&tools),
        row::<TrainDraftSpec>(&tools),
        row::<TrainDiffusionSpec>(&tools),
        row::<DflashSpec>(&tools),
        #[cfg(feature = "serve")]
        row::<ServeSpec>(&tools),
        row::<BenchSpec>(&tools),
        row::<EvalSpec>(&tools),
        row::<MergeSpec>(&tools),
        row::<FuseSpec>(&tools),
        row::<QuantizeSpec>(&tools),
        row::<PackExpertsSpec>(&tools),
        row::<TokenizeSpec>(&tools),
        row::<VocodeSpec>(&tools),
        row::<DecideSpec>(&tools),
        row::<InferSpec>(&tools),
    ];
    for row in &rows {
        assert_eq!(row.cli, row.spec, "the command line: {row:?}");
        assert_eq!(row.tui, row.spec, "the terminal UI: {row:?}");
        if let Some(mcp) = row.mcp {
            assert_eq!(mcp, row.spec, "the MCP server: {row:?}");
        }
    }
    // The ones that do write one are the trainers.
    let writers: Vec<&str> = rows
        .iter()
        .filter(|r| r.spec)
        .map(|r| r.subcommand)
        .collect();
    assert_eq!(
        writers,
        ["train", "distill", "grpo", "rlkd", "preference"],
        "{rows:?}"
    );
}
