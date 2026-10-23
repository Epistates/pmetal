//! Bench tab: the training benchmark and the real-workload benchmark.
//!
//! Two modes:
//!
//! * **basic** is `pmetal bench`, which [`BenchSpec`] describes: a short LoRA
//!   training run of a fixed shape.
//! * **workload** is `pmetal bench-workload`: inference and LoRA training over
//!   a real cached dataset, with named presets and a JSON report. It has no job
//!   spec, so its fields are listed here, but their defaults and the values the
//!   presets and context sources take are read from the command's own
//!   definition in the CLI, never copied.
//!
//! What the benchmark prints is shown as it prints it: the results block of the
//! log, unparsed, so a change to the report cannot leave this tab wrong.

use clap::CommandFactory;
use pmetal_core::jobs::BenchSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};

use crate::cli::Cli;
use crate::tui::jobs::LaunchPlan;
use crate::tui::tabs::dashboard::Telemetry;
use crate::tui::tabs::panel::{FormTab, JobPanel, RunState};
use crate::tui::tabs::status_panel::render_standard;
use crate::tui::theme::THEME;
use crate::tui::widgets::{FieldKind, FormField, FormIssue, FormTabState, FormView};

/// The subcommand the workload mode runs.
const WORKLOAD: &str = "bench-workload";
/// The preset that means "no preset": the model and dataset fields rule.
const CUSTOM: &str = "custom";

/// What the CLI's definition of `bench-workload` says about one option.
struct ClapArg {
    default: String,
    values: Vec<String>,
}

/// The default and the allowed values of `--<id>` of `bench-workload`.
fn clap_arg(id: &str) -> ClapArg {
    let command = Cli::command();
    let arg = command
        .find_subcommand(WORKLOAD)
        .and_then(|sub| sub.get_arguments().find(|a| a.get_long() == Some(id)));
    ClapArg {
        default: arg
            .and_then(|a| a.get_default_values().first())
            .map(|v| v.to_string_lossy().into_owned())
            .unwrap_or_default(),
        values: arg
            .map(|a| {
                a.get_possible_values()
                    .iter()
                    .map(|v| v.get_name().to_string())
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Workload options entered as a whole number: `(clap id, label, min, max)`.
const WORKLOAD_COUNTS: &[(&str, &str, i64, i64)] = &[
    ("prompt-samples", "Prompt Samples", 1, 128),
    ("max-prompt-tokens", "Max Prompt Tokens", 0, 16_384),
    ("decode-steps", "Decode Steps", 1, 4096),
    ("inference-warmup-passes", "Inference Warmup", 0, 32),
    (
        "inference-session-repeats",
        "Inference Session Repeats",
        1,
        64,
    ),
    ("inference-repeats", "Inference Repeats", 1, 64),
    ("train-samples", "Train Samples", 1, 4096),
    ("train-steps", "Train Steps", 1, 4096),
    ("batch-size", "Train Batch Size", 1, 128),
    ("max-seq-len", "Train Max Seq Len", 0, 32_768),
];

fn workload_fields() -> Vec<FormField> {
    let preset = clap_arg("preset");
    let mut preset_options = vec![CUSTOM.to_string()];
    preset_options.extend(preset.values);
    let context = clap_arg("inference-context");

    let mut fields = vec![
        FormField::new(
            "Mode",
            "workload",
            FieldKind::Enum {
                options: vec!["workload".into(), "basic".into()],
            },
            "Mode",
        ),
        FormField::new(
            "Preset",
            CUSTOM,
            FieldKind::Enum {
                options: preset_options,
            },
            "Workload",
        ),
        FormField::new(
            "Dataset",
            clap_arg("dataset").default,
            FieldKind::DatasetPicker,
            "Workload",
        ),
        FormField::new(
            "Inference Context",
            context.default,
            FieldKind::Enum {
                options: context.values,
            },
            "Workload",
        ),
        FormField::new("Experts Dir", "", FieldKind::Path, "Workload"),
    ];
    for &(id, label, min, max) in WORKLOAD_COUNTS {
        fields.push(FormField::new(
            label,
            clap_arg(id).default,
            FieldKind::Integer { min, max },
            "Workload",
        ));
    }
    let mut json = FormField::new("JSON Report", "", FieldKind::Path, "Output");
    json.optional = true;
    fields.push(json);
    fields
}

/// Which fields a mode shows. The model is the workload's only with the
/// custom preset (a preset names its own); the shape knobs are basic's alone.
fn visibility(view: &FormView<'_>, field: &FormField) -> bool {
    let workload = view.value("Mode") != "basic";
    let custom = view.value("Preset") == CUSTOM;
    match (field.section.as_str(), field.label.as_str()) {
        ("Mode", _) => true,
        (_, "Model") => !workload || custom,
        ("Bench", _) => !workload,
        ("Workload", "Preset") | ("Workload", "Experts Dir") => workload,
        ("Workload", "Inference Warmup") | ("Workload", "Inference Session Repeats") => workload,
        ("Workload", _) => workload && custom,
        ("Output", _) => workload,
        _ => true,
    }
}

/// The benchmark tab.
pub struct BenchTab {
    panel: JobPanel,
}

impl BenchTab {
    pub fn new() -> Self {
        let mut fields = FormTabState::for_spec::<BenchSpec>().fields;
        let workload = workload_fields();
        // Mode first, then the spec's model, then the rest of the workload's,
        // then the spec's shape, then the report.
        let mut ordered = Vec::new();
        let mut workload = workload.into_iter();
        ordered.extend(workload.next());
        ordered.extend(fields.drain(..1));
        ordered.extend(workload);
        ordered.extend(fields);
        let form = FormTabState::new(ordered).with_visibility(visibility);
        Self {
            panel: JobPanel::with_form("Bench", form),
        }
    }

    fn is_workload(&self) -> bool {
        self.panel.form.value("Mode") != "basic"
    }

    /// The report file, if one was asked for.
    pub fn report_path(&self) -> Option<String> {
        let path = self.panel.form.value("JSON Report");
        (self.is_workload() && !path.is_empty()).then_some(path)
    }

    fn workload_plan(&self) -> Result<LaunchPlan, Vec<FormIssue>> {
        let form = &self.panel.form;
        let mut issues = Vec::new();
        let value_of = |label: &str| form.value(label);
        let preset = value_of("Preset");
        let mut argv: Vec<String> = Vec::new();
        if preset != CUSTOM {
            argv.extend(["--preset".to_string(), preset]);
        } else {
            let model = value_of("Model");
            if model.is_empty() {
                issues.push(FormIssue {
                    field: "Model".into(),
                    message: "required with the custom preset".into(),
                });
            }
            if value_of("Dataset").is_empty() {
                issues.push(FormIssue {
                    field: "Dataset".into(),
                    message: "required with the custom preset".into(),
                });
            }
            argv.extend(["--model".to_string(), model]);
            argv.extend(["--dataset".to_string(), value_of("Dataset")]);
            argv.extend([
                "--inference-context".to_string(),
                value_of("Inference Context"),
            ]);
            for &(id, label, _, _) in WORKLOAD_COUNTS {
                // The warm-up and repeat counts also apply to a preset.
                if matches!(id, "inference-warmup-passes" | "inference-session-repeats") {
                    continue;
                }
                argv.extend([format!("--{id}"), value_of(label)]);
            }
        }
        for &(id, label, _, _) in WORKLOAD_COUNTS {
            if matches!(id, "inference-warmup-passes" | "inference-session-repeats") {
                argv.extend([format!("--{id}"), value_of(label)]);
            }
        }
        let experts = value_of("Experts Dir");
        if !experts.is_empty() {
            argv.extend(["--experts-dir".to_string(), experts]);
        }
        let report = value_of("JSON Report");
        if !report.is_empty() {
            if !report.ends_with(".json") {
                issues.push(FormIssue {
                    field: "JSON Report".into(),
                    message: "must end in .json".into(),
                });
            }
            argv.extend(["--json".to_string(), "--output".to_string(), report]);
        }
        if !issues.is_empty() {
            return Err(issues);
        }
        Ok(LaunchPlan {
            subcommand: WORKLOAD.to_string(),
            argv,
            ..LaunchPlan::default()
        })
    }

    /// The lines of the log from the benchmark's own results heading on.
    fn results(&self) -> Vec<String> {
        let lines = self.panel.log.tail_text(usize::MAX);
        let start = lines
            .iter()
            .rposition(|l| l.starts_with("Results (") || l.starts_with("Workload Benchmark"));
        match start {
            Some(i) => lines[i..]
                .iter()
                .filter(|l| !l.trim().is_empty())
                .take(24)
                .map(|l| l.to_string())
                .collect(),
            None => Vec::new(),
        }
    }
}

impl Default for BenchTab {
    fn default() -> Self {
        Self::new()
    }
}

impl FormTab for BenchTab {
    fn panel(&self) -> &JobPanel {
        &self.panel
    }

    fn panel_mut(&mut self) -> &mut JobPanel {
        &mut self.panel
    }

    fn plan(&self) -> Result<LaunchPlan, Vec<FormIssue>> {
        if self.is_workload() {
            self.workload_plan()
        } else {
            self.panel.plan_for::<BenchSpec>()
        }
    }

    fn start_title(&self) -> String {
        "Run benchmark?".to_string()
    }

    fn summary(&self) -> Vec<String> {
        let form = &self.panel.form;
        if self.is_workload() {
            let mut lines = vec![format!("Mode      workload ({})", form.value("Preset"))];
            if form.value("Preset") == CUSTOM {
                lines.push(format!("Model     {}", form.value("Model")));
                lines.push(format!("Dataset   {}", form.value("Dataset")));
            }
            lines.push(format!(
                "Samples   {} prompts x {} decode steps",
                form.value("Prompt Samples"),
                form.value("Decode Steps")
            ));
            if let Some(report) = self.report_path() {
                lines.push(format!("Report    {report}"));
            }
            lines
        } else {
            form.summary()
        }
    }

    fn render(&mut self, area: Rect, buf: &mut Buffer, _: &Telemetry<'_>) {
        let results = self.results();
        let mut extras: Vec<Line<'static>> = Vec::new();
        if !results.is_empty() {
            extras.push(Line::from(""));
            extras.push(Line::from(Span::styled("  Results", THEME.kv_key)));
            let finished = matches!(self.panel.run, RunState::Completed { .. });
            for line in results {
                extras.push(Line::from(Span::styled(
                    format!("  {line}"),
                    if finished { THEME.text } else { THEME.text_dim },
                )));
            }
        }
        let height = (8 + extras.len() as u16).min(area.height / 2).max(8);
        render_standard(&mut self.panel, area, buf, height, extras, "Benchmark Log");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::cli_check::assert_parses;
    use crate::tui::tabs::panel::Pick;

    #[test]
    fn the_workload_forms_defaults_are_the_commands_own() {
        let tab = BenchTab::new();
        let form = &tab.panel.form;
        // These are what `pmetal bench-workload --help` says.
        assert_eq!(form.value("Prompt Samples"), "8");
        assert_eq!(form.value("Decode Steps"), "32");
        assert_eq!(form.value("Inference Warmup"), "2");
        assert_eq!(form.value("Train Steps"), "4");
        assert_eq!(form.value("Inference Context"), "auto");
        assert_eq!(
            form.value("Dataset"),
            "TeichAI/gemini-3-pro-preview-high-reasoning-250x"
        );
        // The presets and context sources are the commands' value enums.
        let preset = form.fields.iter().find(|f| f.label == "Preset").unwrap();
        let FieldKind::Enum { options } = &preset.kind else {
            panic!("an enum")
        };
        assert_eq!(options[0], "custom");
        assert!(options.contains(&"dense-qwen3".to_string()), "{options:?}");
        let context = form
            .fields
            .iter()
            .find(|f| f.label == "Inference Context")
            .unwrap();
        let FieldKind::Enum { options } = &context.kind else {
            panic!("an enum")
        };
        assert_eq!(options, &["auto", "prompt", "text-prefix"]);
    }

    #[test]
    fn both_modes_launch_what_the_cli_accepts() {
        let mut tab = BenchTab::new();
        // Workload, custom: every knob is spelled.
        let plan = tab.plan().unwrap();
        assert_eq!(plan.subcommand, "bench-workload");
        assert!(plan.argv.contains(&"--prompt-samples".to_string()));
        assert_parses(&plan);
        // A preset stands alone, with the warm-up counts and a report.
        tab.panel.form.set_value("Preset", "dense-qwen3");
        tab.panel.form.set_value("JSON Report", "/tmp/report.json");
        let plan = tab.plan().unwrap();
        assert!(
            plan.argv
                .starts_with(&["--preset".to_string(), "dense-qwen3".to_string()])
        );
        assert!(!plan.argv.contains(&"--model".to_string()));
        assert!(plan.argv.contains(&"--inference-warmup-passes".to_string()));
        assert!(
            plan.argv
                .windows(3)
                .any(|w| w == ["--json", "--output", "/tmp/report.json"])
        );
        assert_parses(&plan);
        // Basic mode is the training benchmark: it takes no report flags.
        tab.panel.form.set_value("Mode", "basic");
        tab.on_pick(
            "Model",
            Pick::Model {
                id: "Qwen/Qwen3-0.6B",
                architecture: None,
            },
        );
        let plan = tab.plan().unwrap();
        assert_eq!(plan.subcommand, "bench");
        assert!(
            !plan.argv.contains(&"--json".to_string()),
            "{:?}",
            plan.argv
        );
        assert_parses(&plan);
    }

    #[test]
    fn problems_name_the_field() {
        let mut tab = BenchTab::new();
        tab.panel.form.set_value("Model", "");
        tab.panel.form.set_value("JSON Report", "report.txt");
        let issues = tab.plan().unwrap_err();
        let fields: Vec<&str> = issues.iter().map(|i| i.field.as_str()).collect();
        assert_eq!(fields, ["Model", "JSON Report"]);
        tab.panel.form.set_value("Mode", "basic");
        tab.panel.form.set_value("Batch Size", "0");
        let issues = tab.plan().unwrap_err();
        assert_eq!(issues[0].field, "Batch Size");
    }

    #[test]
    fn each_mode_shows_its_own_fields() {
        let mut tab = BenchTab::new();
        let shown = |tab: &mut BenchTab| -> Vec<String> {
            let mut seen = Vec::new();
            tab.panel.form.focus_field("Mode");
            for _ in 0..tab.panel.form.fields.len() {
                tab.panel.form.next_param();
                seen.push(tab.panel.form.current().unwrap().label.clone());
            }
            seen
        };
        let workload = shown(&mut tab);
        assert!(workload.contains(&"Dataset".to_string()));
        assert!(
            !workload.contains(&"Batch Size".to_string()),
            "{workload:?}"
        );
        tab.panel.form.set_value("Preset", "dense-qwen3");
        let preset = shown(&mut tab);
        assert!(
            !preset.contains(&"Dataset".to_string()),
            "a preset names its own: {preset:?}"
        );
        assert!(preset.contains(&"Inference Warmup".to_string()));
        tab.panel.form.set_value("Mode", "basic");
        let basic = shown(&mut tab);
        assert!(
            basic.contains(&"Batch Size".to_string()) && basic.contains(&"Seq Len".to_string())
        );
        assert!(!basic.contains(&"Prompt Samples".to_string()));
    }

    #[test]
    fn the_results_block_is_the_benchmarks_own_report_verbatim() {
        let mut tab = BenchTab::new();
        for line in [
            "Training benchmark",
            "  Model:        Qwen/Qwen3-0.6B",
            "",
            "Training...",
            "",
            "Results (4 measured steps)",
            "  Step time:    median 812.4 ms (min 790.1, max 840.9)",
            "  Throughput:   median 1260 tokens/s (1024 tokens per step)",
            "  Peak memory:  3.21 GB",
            "  Final loss:   2.3456",
        ] {
            tab.panel.log.push(line, false);
        }
        let results = tab.results();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0], "Results (4 measured steps)");
        assert!(results[2].contains("1260 tokens/s"));
        // A workload report starts at its own heading.
        let mut tab = BenchTab::new();
        for line in [
            "noise",
            "Workload Benchmark",
            "  Device:  Apple M4 Max",
            "Inference",
        ] {
            tab.panel.log.push(line, false);
        }
        assert_eq!(tab.results()[0], "Workload Benchmark");
    }
}
