//! Eval tab — run `pmetal eval` against a dataset and show the result.
//!
//! Mirrors the CLI: model picker, dataset picker, optional LoRA adapter,
//! max sequence length, sample count, and a JSON-toggle to persist the
//! full report. Stdout is parsed for per-sample progress and a final
//! metric summary (`perplexity = X` / `accuracy = X`). The right-hand
//! panel shows a compact status badge, the last-seen metrics, and a
//! tailing log.

use crossterm::event::KeyEvent;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{
    FieldKind, FormAction, FormField, FormTabState, JobLog, StatusTone, status_line,
};

/// Metric snapshot extracted from eval stdout. Each field stays `None`
/// until the matching key is seen in the child process output.
#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    pub samples_done: usize,
    pub samples_total: usize,
    pub perplexity: Option<f64>,
    pub accuracy: Option<f64>,
    pub loss: Option<f64>,
}

/// Runtime status of the `pmetal eval` subprocess.
#[derive(Debug, Clone, Default)]
pub enum EvalStatus {
    #[default]
    Idle,
    Running,
    Completed,
    Failed(String),
}

/// Eval tab state.
pub struct EvalTab {
    pub form: FormTabState,
    pub status: EvalStatus,
    pub metrics: EvalMetrics,
    pub log: JobLog,
}

impl EvalTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::new(Self::default_fields()),
            status: EvalStatus::Idle,
            metrics: EvalMetrics::default(),
            log: JobLog::with_default_cap(),
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            FormField::new("Model", "(not selected)", FieldKind::ModelPicker, "Model"),
            FormField::new("LoRA Adapter", "", FieldKind::Text, "Model"),
            FormField::new(
                "Dataset",
                "(not selected)",
                FieldKind::DatasetPicker,
                "Data",
            ),
            FormField::new(
                "Num Samples",
                "0",
                FieldKind::Integer {
                    min: 0,
                    max: 1_000_000,
                },
                "Data",
            ),
            FormField::new(
                "Max Seq Len",
                "1024",
                FieldKind::Integer {
                    min: 64,
                    max: 131_072,
                },
                "Runtime",
            ),
            FormField::new("JSON Report", "Disabled", FieldKind::Toggle, "Output"),
        ]
    }

    // ── Form delegation ─────────────────────────────────────────────────

    pub fn is_editing(&self) -> bool {
        self.form.is_editing()
    }
    pub fn handle_edit_key(&mut self, k: KeyEvent) {
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
        self.form.next_param(|_| true);
    }
    pub fn prev_param(&mut self) {
        self.form.prev_param(|_| true);
    }

    // ── State transitions ──────────────────────────────────────────────

    pub fn is_running(&self) -> bool {
        matches!(self.status, EvalStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.metrics = EvalMetrics::default();
        self.status = EvalStatus::Running;
    }

    pub fn mark_completed(&mut self) {
        self.status = EvalStatus::Completed;
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = EvalStatus::Failed(msg.to_string());
    }

    pub fn append_log(&mut self, line: &str) {
        // Sample progress: `[42/500] evaluated` or `42/500 samples`.
        if let Some((done, total)) = parse_sample_progress(line) {
            self.metrics.samples_done = done;
            self.metrics.samples_total = total;
        }

        // Final metrics. These are usually keyed like `perplexity = 7.42`
        // or `accuracy: 0.812`; accept both separator styles.
        if let Some(v) = parse_metric(line, "perplexity") {
            self.metrics.perplexity = Some(v);
        }
        if let Some(v) = parse_metric(line, "accuracy") {
            self.metrics.accuracy = Some(v);
        }
        if let Some(v) = parse_metric(line, "loss") {
            self.metrics.loss = Some(v);
        }

        self.log.push(line);
    }

    pub fn set_model(&mut self, model_id: &str) {
        self.form.set_value("Model", model_id);
    }

    pub fn set_dataset(&mut self, path: &str) {
        self.form.set_value("Dataset", path);
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        if self.form.value("Model") == "(not selected)" {
            return Err("Model is required.".into());
        }
        if self.form.value("Dataset") == "(not selected)" {
            return Err("Dataset is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        let num_samples = self.form.value("Num Samples");
        let samples_label = if num_samples == "0" {
            "all".to_string()
        } else {
            num_samples
        };
        vec![
            format!("Model:    {}", self.form.value("Model")),
            format!("Dataset:  {}", self.form.value("Dataset")),
            format!("Samples:  {samples_label}"),
            format!("Max Seq:  {}", self.form.value("Max Seq Len")),
            format!("JSON:     {}", self.form.value("JSON Report")),
            String::new(),
            "Run eval?".into(),
        ]
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let mut args = vec!["eval".to_string()];
        args.extend(["--model".into(), self.form.value("Model")]);
        args.extend(["--dataset".into(), self.form.value("Dataset")]);
        args.extend(["--max-seq-len".into(), self.form.value("Max Seq Len")]);
        args.extend(["--num-samples".into(), self.form.value("Num Samples")]);

        let lora = self.form.value("LoRA Adapter");
        if !lora.is_empty() {
            args.extend(["--lora".into(), lora]);
        }
        if self.form.value("JSON Report") == "Enabled" {
            args.push("--json".into());
        }
        args
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl EvalTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(area);

        self.form
            .render_list(config_area, buf, "Eval Configuration", |_| true);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(10), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Eval Log");
    }

    fn render_status(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Metrics ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines: Vec<Line> = Vec::new();

        let (tone, label, detail): (StatusTone, &str, Option<String>) = match &self.status {
            EvalStatus::Idle => (StatusTone::Idle, "Idle", None),
            EvalStatus::Running => {
                let progress = if self.metrics.samples_total > 0 {
                    Some(format!(
                        "{}/{}",
                        self.metrics.samples_done, self.metrics.samples_total
                    ))
                } else {
                    None
                };
                (StatusTone::Running, "Running", progress)
            }
            EvalStatus::Completed => (StatusTone::Completed, "Completed", None),
            EvalStatus::Failed(msg) => (StatusTone::Failed, "Failed", Some(msg.clone())),
        };
        lines.push(status_line(tone, label, detail.as_deref()));

        if self.metrics.samples_total > 0 {
            lines.push(Line::from(Span::styled(
                format!(
                    "  {}",
                    progress_bar(self.metrics.samples_done, self.metrics.samples_total, 30)
                ),
                THEME.text,
            )));
        }

        if let Some(ppl) = self.metrics.perplexity {
            lines.push(Line::from(vec![
                Span::styled("  Perplexity ", THEME.kv_key),
                Span::styled(format!("{ppl:.3}"), THEME.kv_value),
            ]));
        }
        if let Some(acc) = self.metrics.accuracy {
            lines.push(Line::from(vec![
                Span::styled("  Accuracy   ", THEME.kv_key),
                Span::styled(format!("{:.2}%", acc * 100.0), THEME.kv_value),
            ]));
        }
        if let Some(loss) = self.metrics.loss {
            lines.push(Line::from(vec![
                Span::styled("  Loss       ", THEME.kv_key),
                Span::styled(format!("{loss:.4}"), THEME.kv_value),
            ]));
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

// ── Stdout parsing ─────────────────────────────────────────────────────

fn parse_sample_progress(line: &str) -> Option<(usize, usize)> {
    // `[N/M] ...`
    if let Some(rest) = line.trim_start().strip_prefix('[') {
        let end = rest.find(']')?;
        let slice = &rest[..end];
        let (a, b) = slice.split_once('/')?;
        return Some((a.trim().parse().ok()?, b.trim().parse().ok()?));
    }
    // `... N/M samples ...`
    for (i, ch) in line.char_indices() {
        if ch.is_ascii_digit() {
            let tail = &line[i..];
            let end = tail.find([' ', ',']).unwrap_or(tail.len());
            let slice = &tail[..end];
            if let Some((a, b)) = slice.split_once('/') {
                if let (Ok(done), Ok(total)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    if total > 0 && done <= total {
                        return Some((done, total));
                    }
                }
            }
            break;
        }
    }
    None
}

/// Parse `key = value` or `key: value` from a line, ignoring surrounding
/// punctuation. Returns `None` when the key isn't present.
fn parse_metric(line: &str, key: &str) -> Option<f64> {
    let lower = line.to_lowercase();
    let idx = lower.find(key)?;
    let tail = &line[idx + key.len()..];
    let tail = tail.trim_start_matches([' ', '=', ':']);
    let end = tail
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e'))
        .unwrap_or(tail.len());
    tail[..end].parse::<f64>().ok()
}

fn progress_bar(done: usize, total: usize, width: usize) -> String {
    if total == 0 || width == 0 {
        return String::new();
    }
    let filled = (done * width) / total;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "#".repeat(filled), "-".repeat(empty))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bracket_sample_progress() {
        assert_eq!(parse_sample_progress("[42/500] evaluated"), Some((42, 500)));
    }

    #[test]
    fn parses_narrative_sample_progress() {
        assert_eq!(
            parse_sample_progress("processed 100/200 samples"),
            Some((100, 200))
        );
    }

    #[test]
    fn parses_perplexity_equals() {
        assert!(
            (parse_metric("Final perplexity = 7.42", "perplexity").unwrap() - 7.42).abs() < 1e-3
        );
    }

    #[test]
    fn parses_accuracy_colon() {
        assert!((parse_metric("accuracy: 0.812", "accuracy").unwrap() - 0.812).abs() < 1e-3);
    }

    #[test]
    fn ignores_unrelated_keys() {
        assert!(parse_metric("loss: 1.5", "perplexity").is_none());
    }
}
