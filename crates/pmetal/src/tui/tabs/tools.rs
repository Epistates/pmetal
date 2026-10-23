//! The one-command tabs: quantize, merge, evaluate, speculative decoding and
//! tokenize.
//!
//! Each is a form of its spec's fields, a job started from it, and a status
//! box with whatever that command says about itself. What a command prints is
//! read against the strings the commands print (the tests below carry them
//! verbatim); a line that matches nothing is simply left in the log.

use pmetal_core::jobs::{DflashSpec, EvalSpec, MergeSpec, QuantizeSpec, TokenizeSpec};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};

use crate::tui::jobs::LaunchPlan;
use crate::tui::tabs::dashboard::Telemetry;
use crate::tui::tabs::model_short_name;
use crate::tui::tabs::panel::{FormTab, JobPanel, Pick, RunState};
use crate::tui::tabs::status_panel::render_standard;
use crate::tui::theme::THEME;
use crate::tui::widgets::{FormField, FormIssue, FormTabState, FormView};

/// Which command a [`ToolTab`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Quantize,
    Merge,
    Eval,
    Dflash,
    Tokenize,
}

/// What a command has said about itself so far.
#[derive(Debug, Clone, Default, PartialEq)]
struct Readout {
    /// What the command is doing, in its own words.
    stage: Option<&'static str>,
    perplexity: Option<f64>,
    avg_nll: Option<f64>,
    samples: Option<u64>,
    tokens: Option<u64>,
    bits_per_weight: Option<f64>,
}

/// One of the one-command tabs.
pub struct ToolTab {
    panel: JobPanel,
    tool: Tool,
    plan: fn(&JobPanel) -> Result<LaunchPlan, Vec<FormIssue>>,
    readout: Readout,
}

/// MLX knobs show only for MLX output, the KL settings only with calibration,
/// and the importance matrix (a GGUF quantizer's weighting) not for MLX output.
fn quantize_visibility(view: &FormView<'_>, field: &FormField) -> bool {
    match field.label.as_str() {
        "MLX Bits" | "MLX Group Size" => view.value("Format") == "mlx",
        "IMatrix" | "Calibration Data" => view.value("Format") != "mlx",
        "Calibration Context" | "Calibration Chunks" => {
            view.value("Format") != "mlx" && !view.value("Calibration Data").is_empty()
        }
        "Target BPW" | "KL Threshold" => view.value("KL Calibration") == "Enabled",
        _ => true,
    }
}

impl ToolTab {
    fn build<S: pmetal_core::jobs::Spec>(
        title: &'static str,
        tool: Tool,
        form: Option<FormTabState>,
    ) -> Self {
        let panel = match form {
            Some(form) => JobPanel::with_form(title, form),
            None => JobPanel::new::<S>(title),
        };
        Self {
            panel,
            tool,
            plan: |panel| panel.plan_for::<S>(|_, _| {}),
            readout: Readout::default(),
        }
    }

    pub fn quantize() -> Self {
        let form = FormTabState::for_spec::<QuantizeSpec>().with_visibility(quantize_visibility);
        Self::build::<QuantizeSpec>("Quantize", Tool::Quantize, Some(form))
    }

    pub fn merge() -> Self {
        Self::build::<MergeSpec>("Merge", Tool::Merge, None)
    }

    pub fn eval() -> Self {
        Self::build::<EvalSpec>("Eval", Tool::Eval, None)
    }

    pub fn dflash() -> Self {
        Self::build::<DflashSpec>("DFlash", Tool::Dflash, None)
    }

    pub fn tokenize() -> Self {
        Self::build::<TokenizeSpec>("Tokenize", Tool::Tokenize, None)
    }

    /// Keep the output path's ending in step with the format: a GGUF is one
    /// file, an MLX conversion is a directory.
    fn sync_quantize_output(&mut self) {
        let mlx = self.panel.form.value("Format") == "mlx";
        let current = self.panel.form.value("Output Path");
        if current.is_empty() {
            return;
        }
        let stem = current.trim_end_matches(".gguf").trim_end_matches("-mlx");
        let next = if mlx {
            format!("{stem}-mlx")
        } else {
            format!("{stem}.gguf")
        };
        self.panel.form.set_value("Output Path", next);
    }

    /// The tab's own lines for the status box.
    fn status_extras(&self) -> Vec<Line<'static>> {
        let r = &self.readout;
        let kv = |key: &str, value: String| {
            Line::from(vec![
                Span::styled(format!("  {key:<12}"), THEME.kv_key),
                Span::styled(value, THEME.kv_value),
            ])
        };
        let mut lines = Vec::new();
        match self.tool {
            Tool::Quantize => {
                if let (Some(stage), true) = (r.stage, self.panel.is_running()) {
                    lines.push(kv("Stage", stage.to_string()));
                }
                if matches!(self.panel.run, RunState::Completed { .. }) {
                    lines.push(kv("Output", self.panel.form.value("Output Path")));
                }
                if let Some(bpw) = r.bits_per_weight {
                    lines.push(kv("Bits/weight", format!("{bpw:.2}")));
                }
            }
            Tool::Eval => {
                if let Some(n) = r.samples {
                    lines.push(kv("Samples", n.to_string()));
                }
                if let Some(n) = r.tokens {
                    lines.push(kv("Tokens", n.to_string()));
                }
                if let Some(nll) = r.avg_nll {
                    lines.push(kv("Average NLL", format!("{nll:.4}")));
                }
                if let Some(ppl) = r.perplexity {
                    lines.push(kv("Perplexity", format!("{ppl:.3}")));
                }
            }
            Tool::Merge | Tool::Tokenize | Tool::Dflash => {
                if matches!(self.panel.run, RunState::Completed { .. }) {
                    // The commands print their result as the last lines.
                    for line in self.panel.log.tail_text(6) {
                        lines.push(Line::from(Span::styled(
                            format!("  {line}"),
                            THEME.text_dim,
                        )));
                    }
                }
            }
        }
        lines
    }
}

impl FormTab for ToolTab {
    fn panel(&self) -> &JobPanel {
        &self.panel
    }

    fn panel_mut(&mut self) -> &mut JobPanel {
        &mut self.panel
    }

    fn plan(&self) -> Result<LaunchPlan, Vec<FormIssue>> {
        (self.plan)(&self.panel)
    }

    fn on_pick(&mut self, field: &str, picked: Pick<'_>) {
        self.panel.form.set_value(field, picked.text());
        if let Pick::Model { id, .. } = picked {
            let short = model_short_name(id);
            match (self.tool, field) {
                (Tool::Quantize, "Model") => {
                    let ext = if self.panel.form.value("Format") == "mlx" {
                        "-mlx"
                    } else {
                        ".gguf"
                    };
                    self.panel
                        .derive("Output Path", format!("./output/{short}-quantized{ext}"));
                }
                (Tool::Merge, "Model A") => {
                    self.panel
                        .derive("Output Dir", format!("./output/{short}--merged"));
                }
                _ => {}
            }
        }
    }

    fn field_changed(&mut self, label: &str) {
        if self.tool == Tool::Quantize && label == "Format" {
            self.sync_quantize_output();
        }
    }

    fn begin(&mut self) {
        self.readout = Readout::default();
    }

    fn on_line(&mut self, line: &str) {
        match self.tool {
            Tool::Quantize => read_quantize_line(line, &mut self.readout),
            Tool::Eval => read_eval_line(line, &mut self.readout),
            _ => {}
        }
    }

    fn render(&mut self, area: Rect, buf: &mut Buffer, _: &Telemetry<'_>) {
        let extras = self.status_extras();
        let height = 8 + extras.len() as u16;
        let log_title = match self.tool {
            Tool::Quantize => "Quantize Log",
            Tool::Merge => "Merge Log",
            Tool::Eval => "Eval Log",
            Tool::Dflash => "DFlash Log",
            Tool::Tokenize => "Tokenize Log",
        };
        render_standard(&mut self.panel, area, buf, height, extras, log_title);
    }
}

/// What `pmetal quantize` prints (commands/quantize.rs) tells the stage.
fn read_quantize_line(line: &str, readout: &mut Readout) {
    let trimmed = line.trim();
    let stage = if trimmed.starts_with("Resolving model") {
        Some("resolving the model")
    } else if trimmed.starts_with("Loading weights") || trimmed.starts_with("Loaded ") {
        Some("loading weights")
    } else if trimmed.starts_with("Running KL calibration") {
        Some("calibrating (KL)")
    } else if trimmed.starts_with("Calibration complete") {
        Some("calibration done")
    } else if trimmed.starts_with("Bit allocation summary") {
        Some("allocating bits")
    } else if trimmed.starts_with("Quantization complete")
        || trimmed.contains("quantization complete")
    {
        Some("done")
    } else {
        None
    };
    if stage.is_some() {
        readout.stage = stage;
    }
    // `Effective BPW: 4.123` (MLX) and `est. BPW=4.10` (GGUF calibration).
    if let Some(bpw) =
        number_after(trimmed, "Effective BPW:").or_else(|| number_after(trimmed, "est. BPW="))
    {
        readout.bits_per_weight = Some(bpw);
    }
}

/// What `pmetal eval` prints (commands/eval.rs): a text report, or with
/// `--json` a pretty-printed object with one key per line.
fn read_eval_line(line: &str, readout: &mut Readout) {
    let trimmed = line.trim();
    if let Some(n) =
        number_after(trimmed, "Samples evaluated:").or_else(|| json_number(trimmed, "num_samples"))
    {
        readout.samples = Some(n as u64);
    }
    if let Some(n) =
        number_after(trimmed, "Total tokens:").or_else(|| json_number(trimmed, "total_tokens"))
    {
        readout.tokens = Some(n as u64);
    }
    if let Some(v) =
        number_after(trimmed, "Average NLL:").or_else(|| json_number(trimmed, "avg_nll"))
    {
        readout.avg_nll = Some(v);
    }
    if let Some(v) =
        number_after(trimmed, "Perplexity:").or_else(|| json_number(trimmed, "perplexity"))
    {
        readout.perplexity = Some(v);
    }
}

/// The number that follows `label` in `text`.
fn number_after(text: &str, label: &str) -> Option<f64> {
    let tail = text[text.find(label)? + label.len()..].trim_start();
    let end = tail
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(tail.len());
    tail[..end].parse().ok()
}

/// The number of a `"key": 1.5,` line of pretty-printed JSON.
fn json_number(line: &str, key: &str) -> Option<f64> {
    let rest = line
        .strip_prefix('"')?
        .strip_prefix(key)?
        .strip_prefix("\":")?;
    rest.trim().trim_end_matches(',').parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::cli_check::assert_parses;

    #[test]
    fn the_quantize_readout_follows_what_the_command_prints() {
        let mut r = Readout::default();
        for line in [
            "Resolving model: Qwen/Qwen3-0.6B",
            "Loading weights...",
            "Loaded 311 tensors",
        ] {
            read_quantize_line(line, &mut r);
        }
        assert_eq!(r.stage, Some("loading weights"));
        read_quantize_line("Bit allocation summary:", &mut r);
        read_quantize_line("  Q4: 280 tensors", &mut r);
        read_quantize_line("Effective BPW: 4.531", &mut r);
        assert_eq!(r.stage, Some("allocating bits"));
        assert_eq!(r.bits_per_weight, Some(4.531));
        read_quantize_line(
            "Calibration complete: 311 tensors, avg KL=0.001000, worst=blk.0 (0.020000), est. BPW=4.10",
            &mut r,
        );
        assert_eq!(r.bits_per_weight, Some(4.10));
        read_quantize_line("Quantization complete!", &mut r);
        assert_eq!(r.stage, Some("done"));
    }

    #[test]
    fn the_eval_readout_reads_the_text_report_and_the_json_one() {
        let mut text = Readout::default();
        for line in [
            "Evaluating 100 samples...",
            "Results",
            "=======",
            "Samples evaluated: 100",
            "Total tokens:      20480",
            "Average NLL:       2.6532",
            "Perplexity:        14.20",
        ] {
            read_eval_line(line, &mut text);
        }
        assert_eq!(
            (text.samples, text.tokens, text.avg_nll, text.perplexity),
            (Some(100), Some(20480), Some(2.6532), Some(14.20))
        );
        let mut json = Readout::default();
        for line in [
            "{",
            "  \"model\": \"m\",",
            "  \"num_samples\": 100,",
            "  \"total_tokens\": 20480,",
            "  \"avg_nll\": 2.6532,",
            "  \"perplexity\": 14.2",
            "}",
        ] {
            read_eval_line(line, &mut json);
        }
        assert_eq!(text, json);
    }

    #[test]
    fn a_number_is_read_only_after_its_label() {
        assert_eq!(
            number_after("loss 3 Perplexity: 7.5x", "Perplexity:"),
            Some(7.5)
        );
        assert_eq!(number_after("nothing here", "Perplexity:"), None);
        assert_eq!(json_number("\"perplexity\": 7.5,", "perplexity"), Some(7.5));
        assert_eq!(json_number("\"perplexity_x\": 7.5,", "perplexity"), None);
    }

    #[test]
    fn the_output_path_follows_the_model_and_the_format() {
        let mut tab = ToolTab::quantize();
        tab.on_pick(
            "Model",
            Pick::Model {
                id: "org/Qwen3-0.6B",
                architecture: None,
            },
        );
        assert_eq!(
            tab.panel.form.value("Output Path"),
            "./output/Qwen3-0.6B-quantized.gguf"
        );
        tab.panel.form.set_value("Format", "mlx");
        tab.field_changed("Format");
        assert_eq!(
            tab.panel.form.value("Output Path"),
            "./output/Qwen3-0.6B-quantized-mlx"
        );
        tab.panel.form.set_value("Format", "gguf");
        tab.field_changed("Format");
        assert_eq!(
            tab.panel.form.value("Output Path"),
            "./output/Qwen3-0.6B-quantized.gguf"
        );
        // The MLX knobs are only there for MLX output.
        tab.panel.form.focus_field("MLX Bits");
        assert_ne!(tab.panel.form.current().unwrap().label, "MLX Bits");

        let mut merge = ToolTab::merge();
        merge.on_pick(
            "Model A",
            Pick::Model {
                id: "org/Base",
                architecture: None,
            },
        );
        merge.on_pick(
            "Model B",
            Pick::Model {
                id: "org/Other",
                architecture: None,
            },
        );
        assert_eq!(
            merge.panel.form.value("Output Dir"),
            "./output/Base--merged"
        );
    }

    #[test]
    fn what_each_tab_launches_is_what_the_cli_accepts() {
        let mut quantize = ToolTab::quantize();
        quantize.on_pick(
            "Model",
            Pick::Model {
                id: "m",
                architecture: None,
            },
        );
        let mut merge = ToolTab::merge();
        merge.on_pick(
            "Model A",
            Pick::Model {
                id: "a",
                architecture: None,
            },
        );
        merge.on_pick(
            "Model B",
            Pick::Model {
                id: "b",
                architecture: None,
            },
        );
        let mut eval = ToolTab::eval();
        eval.on_pick(
            "Model",
            Pick::Model {
                id: "m",
                architecture: None,
            },
        );
        eval.on_pick("Dataset", Pick::Dataset("d.jsonl"));
        let mut dflash = ToolTab::dflash();
        dflash.on_pick(
            "Target Model",
            Pick::Model {
                id: "t",
                architecture: None,
            },
        );
        dflash.on_pick(
            "Draft Model",
            Pick::Model {
                id: "d",
                architecture: None,
            },
        );
        dflash.panel.form.set_value("Prompt", "hello");
        let mut tokenize = ToolTab::tokenize();
        tokenize.panel.form.set_value("Input JSONL", "in.jsonl");
        tokenize.panel.form.set_value("Output Dir", "out");
        tokenize.on_pick(
            "Tokenizer Model",
            Pick::Model {
                id: "t",
                architecture: None,
            },
        );
        for tab in [&quantize, &merge, &eval, &dflash, &tokenize] {
            let plan = tab
                .plan()
                .unwrap_or_else(|e| panic!("{}: {e:?}", tab.panel.title));
            assert_parses(&plan);
        }
    }
}
