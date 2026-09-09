//! Serve tab — HTTP inference server configuration and control.
//!
//! Mirrors the `pmetal serve` subcommand. Field navigation, inline edit,
//! and log tailing are delegated to `FormTabState` + `JobLog` so this
//! file only owns the tab-specific shape: the `ServeStatus` state machine
//! and the status-panel rendering. Fields are driven by [`ServeSpec`].

use std::time::{Duration, Instant};

use pmetal_core::JobFields as _;
use pmetal_core::jobs::ServeSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{FormAction, FormTabState, JobLog, StatusTone, status_line};

/// Runtime status of the `pmetal serve` process.
#[derive(Debug, Clone, Default)]
pub enum ServeStatus {
    #[default]
    Idle,
    Starting {
        bind_url: String,
        started_at: Instant,
    },
    Running {
        bind_url: String,
        started_at: Instant,
    },
    Stopped {
        reason: String,
    },
    Failed(String),
}

/// Serve tab state.
pub struct ServeTab {
    pub form: FormTabState,
    pub status: ServeStatus,
    pub log: JobLog,
}

impl ServeTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<ServeSpec>(),
            status: ServeStatus::Idle,
            log: JobLog::with_default_cap(),
        }
    }

    // ── Form delegation (all sections always visible) ──────────────────

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
        self.form.next_param(|_| true);
    }
    pub fn prev_param(&mut self) {
        self.form.prev_param(|_| true);
    }
    pub fn handle_enter(&mut self) -> Option<FormAction> {
        self.form.handle_enter()
    }

    // ── State transitions ──────────────────────────────────────────────

    pub fn is_running(&self) -> bool {
        matches!(
            self.status,
            ServeStatus::Starting { .. } | ServeStatus::Running { .. }
        )
    }

    pub fn mark_starting(&mut self, bind_url: String) {
        self.log.clear();
        self.status = ServeStatus::Starting {
            bind_url,
            started_at: Instant::now(),
        };
    }

    /// Append a stdout/stderr line. Triggers the Starting → Running
    /// transition when we see the server's "listening"/"ready" banner.
    pub fn append_log(&mut self, line: &str) {
        if let ServeStatus::Starting { bind_url, .. } = &self.status {
            let lower = line.to_lowercase();
            if lower.contains("listening") || lower.contains("ready") {
                let bind_url = bind_url.clone();
                self.status = ServeStatus::Running {
                    bind_url,
                    started_at: Instant::now(),
                };
            }
        }
        self.log.push(line);
    }

    pub fn set_failed(&mut self, message: &str) {
        self.status = ServeStatus::Failed(message.to_string());
    }

    pub fn set_stopped(&mut self, reason: &str) {
        self.status = ServeStatus::Stopped {
            reason: reason.to_string(),
        };
    }

    pub fn set_model(&mut self, model_id: &str) {
        self.form.set_value("Model", model_id);
    }

    // ── Config summary / CLI args ──────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        let model = self.form.value("Model");
        if model.is_empty() || model == "(not selected)" {
            return Err("Model is required.".into());
        }
        if self.form.value("Host").is_empty() {
            return Err("Host is required (e.g. 127.0.0.1 or 0.0.0.0).".into());
        }
        Ok(())
    }

    pub fn bind_url(&self) -> String {
        let host = self.form.value("Host");
        let port = self.form.value("Port");
        format!("http://{host}:{port}")
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Model:       {}", self.form.value("Model")),
            format!("Bind:        {}", self.bind_url()),
            format!("Max Seq Len: {}", self.form.value("Max Seq Len")),
            format!("KV Cache Bits: {}", self.form.value("KV Cache Bits")),
            format!("FP8:         {}", self.form.value("FP8 Weights")),
            String::new(),
            "Start server?".into(),
        ]
    }

    /// Build CLI args from the form via [`ServeSpec::to_argv`].
    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["serve".to_string()];
        args.extend(spec.to_argv());
        args
    }

    fn spec_from_form(&self) -> ServeSpec {
        let mut spec = ServeSpec::default();
        spec.model = self.form.value("Model");
        spec.host = {
            let v = self.form.value("Host");
            if v.is_empty() { spec.host } else { v }
        };
        spec.port = self.form.value("Port").parse().unwrap_or(spec.port);
        spec.max_seq_len = self
            .form
            .value("Max Seq Len")
            .parse()
            .unwrap_or(spec.max_seq_len);
        spec.fp8 = self.form.value("FP8 Weights") == "Enabled";
        spec.kv_quant = self.form.value("KV Cache Bits").parse().ok();
        spec.no_kv_quant = self.form.value("Disable KV Quant") == "Enabled";
        spec.kv_group_size = self
            .form
            .value("KV Group Size")
            .parse()
            .unwrap_or(spec.kv_group_size);
        spec.kv_turboquant = self.form.value("TurboQuant KV") == "Enabled";
        let preset = self.form.value("TurboQuant Preset");
        spec.kv_turboquant_preset = if preset.is_empty() {
            None
        } else {
            Some(preset)
        };
        spec.ane = self.form.value("Use ANE") == "Enabled";
        spec.ane_max_seq_len = self
            .form
            .value("ANE Max Seq Len")
            .parse()
            .unwrap_or(spec.ane_max_seq_len);
        spec.ane_real_time = self.form.value("ANE Real-Time") == "Enabled";
        spec.continuous_batch = self.form.value("Continuous Batch") == "Enabled";
        spec.cb_max_slots = self
            .form
            .value("CB Max Slots")
            .parse()
            .unwrap_or(spec.cb_max_slots);
        spec.cb_max_queue_depth = self
            .form
            .value("CB Queue Depth")
            .parse()
            .unwrap_or(spec.cb_max_queue_depth);
        spec.cb_block_size = self
            .form
            .value("CB Block Size")
            .parse()
            .unwrap_or(spec.cb_block_size);
        spec.cb_max_blocks = self
            .form
            .value("CB Max Blocks")
            .parse()
            .unwrap_or(spec.cb_max_blocks);
        let experts = self.form.value("Experts Dir");
        spec.experts_dir = if experts.is_empty() {
            None
        } else {
            Some(experts)
        };
        spec
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl ServeTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.form
            .render_list(config_area, buf, "Serve Configuration", |_| true);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(8), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Server Log");
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
            ServeStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Start  [x] Stop",
                    THEME.text_muted,
                )));
            }
            ServeStatus::Starting { bind_url, .. } => {
                lines.push(status_line(StatusTone::Running, "Starting", Some(bind_url)));
            }
            ServeStatus::Running {
                bind_url,
                started_at,
            } => {
                lines.push(status_line(StatusTone::Running, "Serving", Some(bind_url)));
                lines.push(Line::from(vec![
                    Span::styled("  Uptime  ", THEME.kv_key),
                    Span::styled(format_duration(started_at.elapsed()), THEME.kv_value),
                ]));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled("  [x] Stop", THEME.text_muted)));
            }
            ServeStatus::Stopped { reason } => {
                lines.push(status_line(StatusTone::Idle, "Stopped", Some(reason)));
            }
            ServeStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::ServeTab;

    #[test]
    fn serve_tab_defaults_to_loopback() {
        let tab = ServeTab::new();
        assert_eq!(tab.form.value("Host"), "127.0.0.1");
    }

    #[test]
    /// `pmetal serve` has no `--lora`; the documented workflow is to `pmetal fuse`
    /// the adapter first. Guards against the spec field coming back.
    fn serve_tab_does_not_emit_removed_lora_flag() {
        let mut tab = ServeTab::new();
        tab.form.set_value("Model", "/tmp/model");
        let args = tab.build_cli_args();
        assert!(!args.iter().any(|arg| arg == "--lora"));
    }

    #[test]
    fn serve_tab_emits_model_flag() {
        let mut tab = ServeTab::new();
        tab.form.set_value("Model", "/tmp/model");
        let args = tab.build_cli_args();
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"/tmp/model".to_string()));
    }
}
