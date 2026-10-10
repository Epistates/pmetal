//! Bench tab — inference / workload benchmark configuration and trial display.
//!
//! Exposes the two most user-facing `pmetal` bench subcommands behind a
//! Mode enum:
//!
//! * **Basic** → `pmetal bench` (short model forward timing).
//! * **Workload** → `pmetal bench-workload` (preset-driven, full inference
//!   + LoRA training numbers, JSON export).
//!
//! Trials are parsed out of stdout live into a compact table and an
//! average summary line so the operator gets an "is this fast or slow?"
//! read at a glance.
//!
//! Note: the form retains the tab-local field list because [`BenchSpec`]
//! only covers the basic `pmetal bench` subcommand (3 fields). The
//! workload mode fields are TUI-only until a `BenchWorkloadSpec` is
//! added to pmetal-core.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, Widget};

use crate::tui::theme::THEME;
use crate::tui::widgets::{
    FieldKind, FormAction, FormField, FormTabState, JobLog, StatusTone, status_line,
};

/// Status of the currently running bench job.
#[derive(Debug, Clone, Default)]
pub enum BenchStatus {
    #[default]
    Idle,
    Running {
        mode: String,
    },
    Completed,
    Failed(String),
}

/// One measured trial row parsed from the bench subprocess stdout.
#[derive(Debug, Clone, Default)]
pub struct BenchTrial {
    pub index: usize,
    pub prompt_tps: f64,
    pub generation_tps: f64,
    pub peak_memory_gb: f64,
}

/// Bench tab state.
pub struct BenchTab {
    pub form: FormTabState,
    pub status: BenchStatus,
    pub log: JobLog,
    pub trials: Vec<BenchTrial>,
}

impl BenchTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::new(Self::default_fields()),
            status: BenchStatus::Idle,
            log: JobLog::with_default_cap(),
            trials: Vec::new(),
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            FormField::new(
                "Mode",
                "workload",
                FieldKind::Enum {
                    options: vec!["workload".into(), "basic".into()],
                },
                "Mode",
            ),
            FormField::new("Model", "(not selected)", FieldKind::ModelPicker, "Mode"),
            FormField::new(
                "Preset",
                "dense-qwen3",
                FieldKind::Enum {
                    options: vec![
                        "dense-qwen3".into(),
                        "hybrid-qwen3next".into(),
                        "hybrid-qwen35-steady".into(),
                        "moe-nemotronh".into(),
                        "custom".into(),
                    ],
                },
                "Workload",
            ),
            FormField::new(
                "Inference Context",
                "auto",
                FieldKind::Enum {
                    options: vec!["auto".into(), "prompt".into(), "text-prefix".into()],
                },
                "Workload",
            ),
            FormField::new(
                "Dataset",
                "TeichAI/gemini-3-pro-preview-high-reasoning-250x",
                FieldKind::DatasetPicker,
                "Workload",
            ),
            FormField::new("Experts Dir", "", FieldKind::Path, "Workload"),
            FormField::new(
                "Prompt Samples",
                "8",
                FieldKind::Integer { min: 1, max: 128 },
                "Workload",
            ),
            FormField::new(
                "Max Prompt Tokens",
                "0",
                FieldKind::Integer {
                    min: 0,
                    max: 16_384,
                },
                "Workload",
            ),
            FormField::new(
                "Decode Steps",
                "32",
                FieldKind::Integer { min: 1, max: 4096 },
                "Workload",
            ),
            FormField::new(
                "Inference Warmup",
                "2",
                FieldKind::Integer { min: 0, max: 32 },
                "Workload",
            ),
            FormField::new(
                "Inference Session Repeats",
                "1",
                FieldKind::Integer { min: 1, max: 64 },
                "Workload",
            ),
            FormField::new(
                "Inference Repeats",
                "1",
                FieldKind::Integer { min: 1, max: 64 },
                "Workload",
            ),
            FormField::new(
                "Train Samples",
                "8",
                FieldKind::Integer { min: 1, max: 4096 },
                "Workload",
            ),
            FormField::new(
                "Train Steps",
                "4",
                FieldKind::Integer { min: 1, max: 4096 },
                "Workload",
            ),
            FormField::new(
                "Train Batch Size",
                "1",
                FieldKind::Integer { min: 1, max: 128 },
                "Workload",
            ),
            FormField::new(
                "Train Max Seq Len",
                "0",
                FieldKind::Integer {
                    min: 0,
                    max: 32_768,
                },
                "Workload",
            ),
            FormField::new(
                "Batch Size",
                "1",
                FieldKind::Integer { min: 1, max: 128 },
                "Basic",
            ),
            FormField::new(
                "Seq Len",
                "512",
                FieldKind::Integer {
                    min: 32,
                    max: 32_768,
                },
                "Basic",
            ),
            FormField::new("JSON Output", "", FieldKind::Text, "Output"),
        ]
    }

    // ── Form delegation ─────────────────────────────────────────────────

    pub fn is_editing(&self) -> bool {
        self.form.is_editing()
    }
    pub fn handle_edit_key(&mut self, k: crossterm::event::KeyEvent) {
        self.form.handle_edit_key(k);
    }
    pub fn confirm_edit(&mut self) {
        self.form.confirm_edit();
    }
    pub fn cancel_edit(&mut self) {
        self.form.cancel_edit();
    }
    pub fn handle_enter(&mut self) -> Option<FormAction> {
        self.form.handle_enter()
    }

    pub fn next_param(&mut self) {
        let mode = self.form.value("Mode");
        self.form
            .next_param(move |f| section_visible(&mode, &f.section));
    }

    pub fn prev_param(&mut self) {
        let mode = self.form.value("Mode");
        self.form
            .prev_param(move |f| section_visible(&mode, &f.section));
    }

    pub fn set_model(&mut self, model_id: &str) {
        self.form.set_value("Model", model_id);
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, BenchStatus::Running { .. })
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.trials.clear();
        self.status = BenchStatus::Running {
            mode: self.form.value("Mode"),
        };
    }

    pub fn append_log(&mut self, line: &str) {
        if let Some(trial) = parse_trial_line(line) {
            if let Some(existing) = self.trials.iter_mut().find(|t| t.index == trial.index) {
                *existing = trial;
            } else {
                self.trials.push(trial);
            }
        }
        self.log.push(line);
    }

    pub fn mark_completed(&mut self) {
        self.status = BenchStatus::Completed;
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = BenchStatus::Failed(msg.to_string());
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        let mode = self.form.value("Mode");
        let preset = self.form.value("Preset");
        if (mode == "basic" || preset == "custom") && self.form.value("Model") == "(not selected)" {
            return Err("Model is required.".into());
        }
        if mode == "workload" && preset == "custom" && self.form.value("Dataset").is_empty() {
            return Err("Dataset is required for custom workload benchmarks.".into());
        }
        let json_out = self.form.value("JSON Output");
        if !json_out.is_empty() && !json_out.ends_with(".json") {
            return Err("JSON Output, if set, must end in .json".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        let mode = self.form.value("Mode");
        let mut summary = vec![format!("Mode:     {mode}")];
        if mode == "workload" {
            let preset = self.form.value("Preset");
            summary.push(format!("Preset:   {preset}"));
            if preset == "custom" {
                summary.push(format!("Model:    {}", self.form.value("Model")));
                summary.push(format!("Dataset:  {}", self.form.value("Dataset")));
            }
            summary.push(format!(
                "Samples:  {} prompts x {} decode steps",
                self.form.value("Prompt Samples"),
                self.form.value("Decode Steps"),
            ));
        } else {
            summary.push(format!("Model:    {}", self.form.value("Model")));
            summary.push(format!(
                "Shape:    batch={}, seq_len={}",
                self.form.value("Batch Size"),
                self.form.value("Seq Len"),
            ));
        }
        let json_out = self.form.value("JSON Output");
        if !json_out.is_empty() {
            summary.push(format!("JSON:     {json_out}"));
        }
        summary.push(String::new());
        summary.push("Run benchmark?".into());
        summary
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let mode = self.form.value("Mode");
        let mut args = Vec::new();

        if mode == "workload" {
            args.push("bench-workload".into());
            let preset = self.form.value("Preset");
            if preset != "custom" {
                args.extend(["--preset".into(), preset]);
            } else {
                args.extend(["--model".into(), self.form.value("Model")]);
                args.extend(["--dataset".into(), self.form.value("Dataset")]);
                args.extend([
                    "--inference-context".into(),
                    self.form.value("Inference Context"),
                ]);
                args.extend(["--prompt-samples".into(), self.form.value("Prompt Samples")]);
                args.extend([
                    "--max-prompt-tokens".into(),
                    self.form.value("Max Prompt Tokens"),
                ]);
                args.extend(["--decode-steps".into(), self.form.value("Decode Steps")]);
                args.extend([
                    "--inference-repeats".into(),
                    self.form.value("Inference Repeats"),
                ]);
                args.extend(["--train-samples".into(), self.form.value("Train Samples")]);
                args.extend(["--train-steps".into(), self.form.value("Train Steps")]);
                args.extend(["--batch-size".into(), self.form.value("Train Batch Size")]);
                args.extend(["--max-seq-len".into(), self.form.value("Train Max Seq Len")]);
            }
            let experts_dir = self.form.value("Experts Dir");
            if !experts_dir.is_empty() {
                args.extend(["--experts-dir".into(), experts_dir]);
            }
            args.extend([
                "--inference-warmup-passes".into(),
                self.form.value("Inference Warmup"),
            ]);
            args.extend([
                "--inference-session-repeats".into(),
                self.form.value("Inference Session Repeats"),
            ]);
        } else {
            args.push("bench".into());
            args.extend(["--model".into(), self.form.value("Model")]);
            args.extend(["--batch-size".into(), self.form.value("Batch Size")]);
            args.extend(["--seq-len".into(), self.form.value("Seq Len")]);
        }

        let json_out = self.form.value("JSON Output");
        if !json_out.is_empty() {
            args.push("--json".into());
            args.extend(["--output".into(), json_out]);
        }

        args
    }

    pub fn json_output_path(&self) -> Option<PathBuf> {
        let v = self.form.value("JSON Output");
        if v.is_empty() {
            None
        } else {
            Some(PathBuf::from(v))
        }
    }

    fn avg_prompt_tps(&self) -> Option<f64> {
        if self.trials.is_empty() {
            return None;
        }
        let sum: f64 = self.trials.iter().map(|t| t.prompt_tps).sum();
        Some(sum / self.trials.len() as f64)
    }

    fn avg_generation_tps(&self) -> Option<f64> {
        if self.trials.is_empty() {
            return None;
        }
        let sum: f64 = self.trials.iter().map(|t| t.generation_tps).sum();
        Some(sum / self.trials.len() as f64)
    }

    fn peak_mem_gb(&self) -> Option<f64> {
        self.trials
            .iter()
            .map(|t| t.peak_memory_gb)
            .fold(None, |acc, x| Some(acc.map_or(x, |a: f64| a.max(x))))
    }
}

/// Shared visibility rule for nav helpers and the renderer.
fn section_visible(mode: &str, section: &str) -> bool {
    match section {
        "Workload" => mode == "workload",
        "Basic" => mode == "basic",
        _ => true,
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl BenchTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(area);

        let mode = self.form.value("Mode");
        self.form
            .render_list(config_area, buf, "Bench Configuration", move |f| {
                section_visible(&mode, &f.section)
            });

        let [trials_area, log_area] =
            Layout::vertical([Constraint::Min(10), Constraint::Length(10)]).areas(right_area);
        self.render_trials(trials_area, buf);
        self.log.render(log_area, buf, "Bench Log");
    }

    fn render_trials(&self, area: Rect, buf: &mut Buffer) {
        let (tone, label, detail) = match &self.status {
            BenchStatus::Idle => (StatusTone::Idle, "Idle", None),
            BenchStatus::Running { mode } => (StatusTone::Running, "Running", Some(mode.as_str())),
            BenchStatus::Completed => (StatusTone::Completed, "Completed", None),
            BenchStatus::Failed(msg) => (StatusTone::Failed, "Failed", Some(msg.as_str())),
        };

        let block = Block::default()
            .title(" Trials ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        // Reserve the first line for the status badge.
        let [header_area, rest] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(inner);
        Paragraph::new(status_line(tone, label, detail)).render(header_area, buf);

        if self.trials.is_empty() {
            let hint = match &self.status {
                BenchStatus::Idle => "Press S to start a benchmark.",
                BenchStatus::Running { .. } => "Waiting for first trial...",
                BenchStatus::Completed => "No measured trials.",
                BenchStatus::Failed(_) => "Run failed before any trial completed.",
            };
            Paragraph::new(Line::from(Span::styled(
                format!("  {hint}"),
                THEME.text_muted,
            )))
            .render(rest, buf);
            return;
        }

        let header = Row::new(vec![
            Cell::from("Trial").style(THEME.table_header),
            Cell::from("Prompt tok/s").style(THEME.table_header),
            Cell::from("Decode tok/s").style(THEME.table_header),
            Cell::from("Peak GB").style(THEME.table_header),
        ]);

        let rows: Vec<Row<'_>> = self
            .trials
            .iter()
            .map(|t| {
                Row::new(vec![
                    Cell::from(format!("{}", t.index)),
                    Cell::from(format!("{:.1}", t.prompt_tps)),
                    Cell::from(format!("{:.1}", t.generation_tps)),
                    Cell::from(format!("{:.2}", t.peak_memory_gb)),
                ])
            })
            .collect();

        let widths = [
            Constraint::Length(7),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Length(10),
        ];

        // Reserve the last line for the averages summary.
        let [table_area, summary_area] =
            Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(rest);

        let table = Table::new(rows, widths).header(header).column_spacing(2);
        Widget::render(table, table_area, buf);

        if let (Some(pp), Some(gg)) = (self.avg_prompt_tps(), self.avg_generation_tps()) {
            let mut spans = vec![
                Span::styled("  Avg ", THEME.kv_key),
                Span::styled(format!("prompt {pp:.1} "), THEME.kv_value),
                Span::styled("| ", THEME.text_muted),
                Span::styled(format!("decode {gg:.1} "), THEME.kv_value),
            ];
            if let Some(peak) = self.peak_mem_gb() {
                spans.extend([
                    Span::styled("| ", THEME.text_muted),
                    Span::styled(format!("peak {peak:.2} GB"), THEME.kv_value),
                ]);
            }
            Paragraph::new(Line::from(spans)).render(summary_area, buf);
        }
    }
}

/// Parse a line like
/// `Trial 3:  prompt_tps=512.4, generation_tps=102.1, peak_memory=9.23`
/// into a `BenchTrial`. Returns `None` if the line doesn't match.
fn parse_trial_line(line: &str) -> Option<BenchTrial> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix("Trial ")?;
    let (idx_part, rest) = rest.split_once(':')?;
    let index: usize = idx_part.trim().parse().ok()?;

    let prompt_tps = extract_kv_f64(rest, "prompt_tps")?;
    let generation_tps = extract_kv_f64(rest, "generation_tps")?;
    let peak_memory_gb = extract_kv_f64(rest, "peak_memory")?;

    Some(BenchTrial {
        index,
        prompt_tps,
        generation_tps,
        peak_memory_gb,
    })
}

fn extract_kv_f64(hay: &str, key: &str) -> Option<f64> {
    let pos = hay.find(key)?;
    let after = &hay[pos + key.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix('=')?.trim_start();
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e'))
        .unwrap_or(after.len());
    after[..end].parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_trial_line() {
        let t =
            parse_trial_line("Trial 3:  prompt_tps=512.4, generation_tps=102.1, peak_memory=9.23")
                .unwrap();
        assert_eq!(t.index, 3);
        assert!((t.prompt_tps - 512.4).abs() < 1e-3);
        assert!((t.generation_tps - 102.1).abs() < 1e-3);
        assert!((t.peak_memory_gb - 9.23).abs() < 1e-3);
    }

    #[test]
    fn rejects_unrelated_lines() {
        assert!(parse_trial_line("Running warmup..").is_none());
        assert!(parse_trial_line("Averages: prompt_tps=500.0").is_none());
    }
}
