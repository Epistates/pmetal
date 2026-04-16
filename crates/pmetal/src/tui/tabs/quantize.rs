//! Quantize tab — GGUF / MLX quantization configuration and control.
//!
//! Mirrors `pmetal quantize`. Source model picker, output path, method
//! enum (K-quants + `dynamic` + `f16`/`f32`), format enum (gguf / mlx),
//! optional imatrix file, KL-calibration toggle, and MLX bit-width knobs
//! (only surfaced when `format=mlx`). Spawns as a one-shot background job
//! with stdout tailed into the status panel.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{
    FieldKind, FormAction, FormField, FormTabState, JobLog, StatusTone, status_line,
};

/// Runtime status of the `pmetal quantize` process.
#[derive(Debug, Clone, Default)]
pub enum QuantizeStatus {
    #[default]
    Idle,
    Running {
        phase: String,
        tensors_done: usize,
        tensors_total: usize,
    },
    Completed {
        output: String,
        bpw: Option<f32>,
    },
    Failed(String),
}

/// Quantize tab state.
pub struct QuantizeTab {
    pub form: FormTabState,
    pub status: QuantizeStatus,
    pub log: JobLog,
}

impl QuantizeTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::new(Self::default_fields()),
            status: QuantizeStatus::Idle,
            log: JobLog::with_default_cap(),
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            FormField::new(
                "Source Model",
                "(not selected)",
                FieldKind::ModelPicker,
                "Source",
            ),
            FormField::new("LoRA Adapter", "", FieldKind::Text, "Source"),
            FormField::new(
                "Output Path",
                "./output/quantized.gguf",
                FieldKind::Text,
                "Output",
            ),
            FormField::new(
                "Format",
                "gguf",
                FieldKind::Enum {
                    options: vec!["gguf".into(), "mlx".into()],
                },
                "Output",
            ),
            FormField::new(
                "Method",
                "dynamic",
                FieldKind::Enum {
                    options: vec![
                        "dynamic".into(),
                        "q8_0".into(),
                        "q6_k".into(),
                        "q5_k_m".into(),
                        "q5_k_s".into(),
                        "q4_k_m".into(),
                        "q4_k_s".into(),
                        "q3_k_m".into(),
                        "q3_k_l".into(),
                        "q3_k_s".into(),
                        "q2_k".into(),
                        "f16".into(),
                        "f32".into(),
                    ],
                },
                "Method",
            ),
            FormField::new("IMatrix Path", "", FieldKind::Text, "Method"),
            FormField::new(
                "KL Calibrate",
                "Disabled",
                FieldKind::Toggle,
                "KL Calibration",
            ),
            FormField::new("Target BPW", "", FieldKind::Text, "KL Calibration"),
            FormField::new(
                "KL Threshold",
                "0.01",
                FieldKind::Number {
                    min: 0.0001,
                    max: 1.0,
                },
                "KL Calibration",
            ),
            FormField::new(
                "MLX Bits",
                "4",
                FieldKind::Integer { min: 2, max: 8 },
                "MLX Format",
            ),
            FormField::new(
                "MLX Group Size",
                "64",
                FieldKind::Integer { min: 32, max: 256 },
                "MLX Format",
            ),
        ]
    }

    /// Shared field-level visibility used by nav and the renderer so the
    /// cursor never lands on a hidden row. MLX knobs hide unless Format
    /// is `mlx`; the KL Calibrate master toggle stays visible, but its
    /// dependent knobs hide until calibration is turned on.
    fn field_visible(is_mlx: bool, kl_on: bool, field: &FormField) -> bool {
        match field.section.as_str() {
            "MLX Format" => is_mlx,
            "KL Calibration" => kl_on || field.label == "KL Calibrate",
            _ => true,
        }
    }

    fn vis_snapshot(&self) -> (bool, bool) {
        (
            self.form.value("Format") == "mlx",
            self.form.value("KL Calibrate") == "Enabled",
        )
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

    pub fn next_param(&mut self) {
        let (is_mlx, kl_on) = self.vis_snapshot();
        self.form
            .next_param(move |f| Self::field_visible(is_mlx, kl_on, f));
    }

    pub fn prev_param(&mut self) {
        let (is_mlx, kl_on) = self.vis_snapshot();
        self.form
            .prev_param(move |f| Self::field_visible(is_mlx, kl_on, f));
    }

    pub fn handle_enter(&mut self) -> Option<FormAction> {
        // Track the label before cycling so we can react to Format toggles.
        let label_before = self
            .form
            .fields
            .get(self.form.field_idx())
            .map(|f| f.label.clone());
        let action = self.form.handle_enter();
        if label_before.as_deref() == Some("Format") {
            self.sync_output_extension();
        }
        action
    }

    // ── Setters / derived state ─────────────────────────────────────────

    pub fn set_model(&mut self, model_id: &str) {
        self.form.set_value("Source Model", model_id);
        // Auto-derive a sensible default output path from the model name.
        let short = super::model_short_name(model_id);
        let ext = if self.form.value("Format") == "mlx" {
            "-mlx"
        } else {
            ".gguf"
        };
        self.form
            .set_value("Output Path", format!("./output/{short}-quantized{ext}"));
    }

    /// GGUF is a single file, MLX emits a directory. Flip the output
    /// extension when the user toggles the Format field.
    fn sync_output_extension(&mut self) {
        let is_mlx = self.form.value("Format") == "mlx";
        let current = self.form.value("Output Path");
        let trimmed = current
            .trim_end_matches(".gguf")
            .trim_end_matches("-mlx")
            .to_string();
        let next = if is_mlx {
            format!("{trimmed}-mlx")
        } else {
            format!("{trimmed}.gguf")
        };
        self.form.set_value("Output Path", next);
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, QuantizeStatus::Running { .. })
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = QuantizeStatus::Running {
            phase: "starting".into(),
            tensors_done: 0,
            tensors_total: 0,
        };
    }

    pub fn append_log(&mut self, line: &str) {
        if let Some((done, total)) = parse_tensor_progress(line) {
            if let QuantizeStatus::Running {
                tensors_done,
                tensors_total,
                phase,
            } = &mut self.status
            {
                *tensors_done = done;
                *tensors_total = total;
                *phase = "quantizing".into();
            }
        } else if let QuantizeStatus::Running { phase, .. } = &mut self.status {
            let lower = line.to_lowercase();
            if lower.contains("loading") {
                *phase = "loading".into();
            } else if lower.contains("writing") || lower.contains("saving") {
                *phase = "writing".into();
            } else if lower.contains("calibrat") {
                *phase = "calibrating".into();
            }
        }

        self.log.push(line);
    }

    pub fn mark_completed(&mut self) {
        let output = self.form.value("Output Path");
        // Scan recent log lines for the final BPW summary.
        let bpw = self
            .log
            .lines()
            .iter()
            .rev()
            .take(30)
            .find_map(|l| parse_bpw(l));
        self.status = QuantizeStatus::Completed { output, bpw };
    }

    pub fn mark_failed(&mut self, message: &str) {
        self.status = QuantizeStatus::Failed(message.to_string());
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        if self.form.value("Source Model") == "(not selected)" {
            return Err("Source Model is required.".into());
        }
        if self.form.value("Output Path").is_empty() {
            return Err("Output Path is required.".into());
        }
        if self.form.value("Format") == "mlx" {
            let bits: i32 = self.form.value("MLX Bits").parse().unwrap_or(4);
            if !matches!(bits, 3 | 4 | 5 | 6 | 8) {
                return Err("MLX Bits must be 3, 4, 5, 6, or 8.".into());
            }
        }
        if self.form.value("KL Calibrate") == "Enabled" {
            let bpw_raw = self.form.value("Target BPW");
            if !bpw_raw.is_empty() && bpw_raw.parse::<f32>().is_err() {
                return Err("Target BPW must be a number (e.g. 4.5).".into());
            }
        }
        Ok(())
    }

    pub fn output_path(&self) -> PathBuf {
        PathBuf::from(self.form.value("Output Path"))
    }

    pub fn config_summary(&self) -> Vec<String> {
        let mut summary = vec![
            format!("Source:  {}", self.form.value("Source Model")),
            format!("Output:  {}", self.form.value("Output Path")),
            format!("Format:  {}", self.form.value("Format")),
            format!("Method:  {}", self.form.value("Method")),
        ];
        if self.form.value("KL Calibrate") == "Enabled" {
            let bpw = self.form.value("Target BPW");
            let threshold = self.form.value("KL Threshold");
            let bpw_part = if bpw.is_empty() {
                "no budget".to_string()
            } else {
                format!("target={bpw}bpw")
            };
            summary.push(format!("KL Cal:  {bpw_part}, threshold={threshold}"));
        }
        if self.form.value("Format") == "mlx" {
            summary.push(format!(
                "MLX:     {}-bit, group_size={}",
                self.form.value("MLX Bits"),
                self.form.value("MLX Group Size"),
            ));
        }
        summary.push(String::new());
        summary.push("Proceed?".into());
        summary
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let mut args = vec!["quantize".to_string()];

        args.extend(["--model".into(), self.form.value("Source Model")]);
        args.extend(["--output".into(), self.form.value("Output Path")]);
        args.extend(["--format".into(), self.form.value("Format")]);
        args.extend(["--method".into(), self.form.value("Method")]);

        let lora = self.form.value("LoRA Adapter");
        if !lora.is_empty() {
            args.extend(["--lora".into(), lora]);
        }

        let imatrix = self.form.value("IMatrix Path");
        if !imatrix.is_empty() {
            args.extend(["--imatrix".into(), imatrix]);
        }

        if self.form.value("KL Calibrate") == "Enabled" {
            args.push("--kl-calibrate".into());
            let bpw = self.form.value("Target BPW");
            if !bpw.is_empty() {
                args.extend(["--target-bpw".into(), bpw]);
            }
            args.extend(["--kl-threshold".into(), self.form.value("KL Threshold")]);
        }

        if self.form.value("Format") == "mlx" {
            args.extend(["--bits".into(), self.form.value("MLX Bits")]);
            args.extend(["--group-size".into(), self.form.value("MLX Group Size")]);
        }

        args
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl QuantizeTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        let (is_mlx, kl_on) = self.vis_snapshot();
        self.form
            .render_list(config_area, buf, "Quantize Configuration", move |f| {
                Self::field_visible(is_mlx, kl_on, f)
            });

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(8), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Quantize Log");
    }

    fn render_status(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Status ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines: Vec<Line> = Vec::new();
        match &self.status {
            QuantizeStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Start  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            QuantizeStatus::Running {
                phase,
                tensors_done,
                tensors_total,
            } => {
                lines.push(status_line(StatusTone::Running, "Running", Some(phase)));
                if *tensors_total > 0 {
                    let pct = (*tensors_done as f32 / *tensors_total as f32) * 100.0;
                    lines.push(Line::from(vec![
                        Span::styled("  Tensors ", THEME.kv_key),
                        Span::styled(
                            format!("{tensors_done}/{tensors_total} ({pct:.0}%)"),
                            THEME.kv_value,
                        ),
                    ]));
                    lines.push(Line::from(Span::styled(
                        format!("  {}", progress_bar(*tensors_done, *tensors_total, 30)),
                        THEME.text,
                    )));
                }
            }
            QuantizeStatus::Completed { output, bpw } => {
                lines.push(status_line(StatusTone::Completed, "Completed", None));
                lines.push(Line::from(vec![
                    Span::styled("  Output  ", THEME.kv_key),
                    Span::styled(output.clone(), THEME.kv_value),
                ]));
                if let Some(bpw) = bpw {
                    lines.push(Line::from(vec![
                        Span::styled("  Avg BPW ", THEME.kv_key),
                        Span::styled(format!("{bpw:.2}"), THEME.kv_value),
                    ]));
                }
            }
            QuantizeStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

/// Parse `[N/M] ...` or `Quantized N/M tensors` progress lines.
fn parse_tensor_progress(line: &str) -> Option<(usize, usize)> {
    if let Some(rest) = line.trim_start().strip_prefix('[') {
        let end = rest.find(']')?;
        let slice = &rest[..end];
        let (a, b) = slice.split_once('/')?;
        return Some((a.trim().parse().ok()?, b.trim().parse().ok()?));
    }
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

/// Parse a `BPW = X.YZ` or `avg_bpw=X.YZ` line.
fn parse_bpw(line: &str) -> Option<f32> {
    let lower = line.to_lowercase();
    let needle = lower
        .find("bpw")
        .or_else(|| lower.find("bits per weight"))?;
    let tail = &line[needle..];
    let digits: String = tail
        .chars()
        .skip_while(|c| !c.is_ascii_digit() && *c != '.' && *c != '-')
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    digits.parse::<f32>().ok()
}

/// Simple ASCII progress bar: `[####------]`.
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
    fn parses_bracket_progress() {
        assert_eq!(
            parse_tensor_progress("[42/287] blk.3.attn_q.weight"),
            Some((42, 287))
        );
    }

    #[test]
    fn parses_narrative_progress() {
        assert_eq!(
            parse_tensor_progress("Quantized 10/20 tensors so far"),
            Some((10, 20))
        );
    }

    #[test]
    fn rejects_unrelated_slashes() {
        assert_eq!(parse_tensor_progress("loading /path/to/model"), None);
    }

    #[test]
    fn parses_bpw_summary() {
        assert!((parse_bpw("avg bpw = 4.58 across 287 tensors").unwrap() - 4.58).abs() < 1e-3);
        assert!((parse_bpw("Total BPW: 3.92").unwrap() - 3.92).abs() < 1e-3);
    }
}
