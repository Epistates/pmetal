//! Jobs tab — training run history and log viewer.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, HighlightSpacing, Paragraph, Row, Scrollbar, ScrollbarOrientation,
    ScrollbarState, StatefulWidget, Table, TableState, Widget, Wrap,
};

use crate::tui::theme::THEME;

/// Status of a training job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatus::Running => write!(f, "Running"),
            JobStatus::Completed => write!(f, "Done"),
            JobStatus::Failed => write!(f, "Failed"),
            JobStatus::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// A training job entry.
#[derive(Debug, Clone)]
pub struct JobEntry {
    pub id: String,
    pub model: String,
    pub method: String,
    pub status: JobStatus,
    pub started: String,
    pub duration: Option<String>,
    pub final_loss: Option<f64>,
    pub output_dir: String,
    pub log_lines: Vec<String>,
}

/// Jobs tab state.
pub struct JobsTab {
    pub jobs: Vec<JobEntry>,
    pub table_state: TableState,
    pub scrollbar_state: ScrollbarState,
    pub log_scroll: usize,
}

impl JobsTab {
    pub fn new() -> Self {
        let mut tab = Self {
            jobs: Vec::new(),
            table_state: TableState::default(),
            scrollbar_state: ScrollbarState::default(),
            log_scroll: 0,
        };
        tab.scan_jobs();
        tab
    }

    /// Scan output directories for completed training runs.
    pub fn scan_jobs(&mut self) {
        self.jobs.clear();

        // Scan common output directories for checkpoints
        let output_dirs = ["./output", "./output/distilled", "./output/grpo"];

        for dir_path in &output_dirs {
            let dir = std::path::Path::new(dir_path);
            if !dir.exists() {
                continue;
            }

            let entries = match std::fs::read_dir(dir) {
                Ok(e) => e,
                Err(_) => continue,
            };

            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }

                // Check for adapter files (indicates a completed training run)
                let has_adapter = path.join("adapter_config.json").exists()
                    || path.join("training_args.json").exists();
                let has_safetensors = std::fs::read_dir(&path)
                    .ok()
                    .map(|entries| {
                        entries
                            .flatten()
                            .any(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
                    })
                    .unwrap_or(false);

                if !has_adapter && !has_safetensors {
                    continue;
                }

                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                // Try to read training metadata
                let (model, method, final_loss) = read_training_meta(&path);

                let modified = std::fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .ok()
                    .map(|t| {
                        let elapsed = t.elapsed().unwrap_or_default();
                        let hours = elapsed.as_secs() / 3600;
                        if hours < 1 {
                            let mins = elapsed.as_secs() / 60;
                            format!("{}m ago", mins)
                        } else if hours < 24 {
                            format!("{}h ago", hours)
                        } else {
                            format!("{}d ago", hours / 24)
                        }
                    })
                    .unwrap_or_else(|| "unknown".to_string());

                // Read log if exists
                let log_path = path.join("training.log");
                let log_lines = if log_path.exists() {
                    std::fs::read_to_string(&log_path)
                        .unwrap_or_default()
                        .lines()
                        .rev()
                        .take(100)
                        .map(String::from)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect()
                } else {
                    Vec::new()
                };

                // Infer status from artifacts
                let status = if path.join(".running").exists() {
                    JobStatus::Running
                } else if path.join("error.log").exists() {
                    JobStatus::Failed
                } else {
                    JobStatus::Completed
                };

                self.jobs.push(JobEntry {
                    id: name,
                    model: model.unwrap_or_else(|| "-".to_string()),
                    method: method.unwrap_or_else(|| "SFT".to_string()),
                    status,
                    started: modified,
                    duration: None,
                    final_loss,
                    output_dir: path.display().to_string(),
                    log_lines,
                });
            }
        }

        // Sort by most recent first
        self.jobs.reverse();

        if !self.jobs.is_empty() {
            self.table_state.select(Some(0));
        }
        self.scrollbar_state = ScrollbarState::new(self.jobs.len());
    }

    pub fn next_row(&mut self) {
        let count = self.jobs.len();
        if count == 0 {
            return;
        }
        let i = self.table_state.selected().map_or(0, |i| (i + 1) % count);
        self.table_state.select(Some(i));
        self.scrollbar_state = self.scrollbar_state.position(i);
        self.log_scroll = 0;
    }

    pub fn prev_row(&mut self) {
        let count = self.jobs.len();
        if count == 0 {
            return;
        }
        let i = self
            .table_state
            .selected()
            .map_or(0, |i| (i + count - 1) % count);
        self.table_state.select(Some(i));
        self.scrollbar_state = self.scrollbar_state.position(i);
        self.log_scroll = 0;
    }

    pub fn selected_job(&self) -> Option<&JobEntry> {
        self.table_state.selected().and_then(|i| self.jobs.get(i))
    }

    pub fn scroll_log_down(&mut self) {
        if let Some(job) = self.selected_job() {
            if self.log_scroll + 1 < job.log_lines.len() {
                self.log_scroll += 1;
            }
        }
    }

    pub fn scroll_log_up(&mut self) {
        self.log_scroll = self.log_scroll.saturating_sub(1);
    }

    /// Append a line of live output from a running job.
    pub fn append_live_output(&mut self, job_id: &str, line: &str) {
        // Find the job index first to avoid borrow conflicts
        let job_idx = self.jobs.iter().position(|j| j.id == *job_id);
        let Some(idx) = job_idx else { return };

        self.jobs[idx].log_lines.push(line.to_string());

        // Auto-scroll if viewing this job
        if self.table_state.selected() == Some(idx) {
            self.log_scroll = self.jobs[idx].log_lines.len().saturating_sub(1);
        }
    }
}

impl Widget for &mut JobsTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [table_area, detail_area] =
            Layout::vertical([Constraint::Percentage(40), Constraint::Percentage(60)]).areas(area);

        self.render_table(table_area, buf);
        self.render_detail(detail_area, buf);
    }
}

impl JobsTab {
    fn render_table(&mut self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(format!(" Jobs ({}) ", self.jobs.len()))
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        if self.jobs.is_empty() {
            Paragraph::new("\n  No training runs found in ./output/")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }

        let header = Row::new(vec!["Run", "Model", "Method", "Status", "Loss", "When"])
            .style(THEME.table_header)
            .height(1);

        let rows: Vec<Row> = self
            .jobs
            .iter()
            .enumerate()
            .map(|(i, job)| {
                let style = if i % 2 == 0 {
                    THEME.table_row
                } else {
                    THEME.table_row_alt
                };
                let status_style = match job.status {
                    JobStatus::Running => THEME.status_running,
                    JobStatus::Completed => THEME.status_success,
                    JobStatus::Failed => THEME.status_error,
                    JobStatus::Cancelled => THEME.status_idle,
                };
                Row::new(vec![
                    Cell::new(job.id.clone()),
                    Cell::new(job.model.clone()),
                    Cell::new(job.method.clone()),
                    Cell::new(Span::styled(job.status.to_string(), status_style)),
                    Cell::new(
                        job.final_loss
                            .map(|l| format!("{l:.4}"))
                            .unwrap_or_else(|| "-".to_string()),
                    ),
                    Cell::new(job.started.clone()),
                ])
                .style(style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Fill(1),
                Constraint::Length(20),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(8),
                Constraint::Length(10),
            ],
        )
        .header(header)
        .block(block)
        .row_highlight_style(THEME.table_selected)
        .highlight_spacing(HighlightSpacing::Always);

        StatefulWidget::render(table, area, buf, &mut self.table_state);

        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .render(area, buf, &mut self.scrollbar_state);
    }

    fn render_detail(&self, area: Rect, buf: &mut Buffer) {
        let Some(job) = self.selected_job() else {
            let block = Block::default()
                .title(" Details ")
                .title_style(THEME.block_title)
                .borders(Borders::ALL)
                .border_style(THEME.block);
            Paragraph::new("Select a job to view details")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        };

        let [info_area, log_area] =
            Layout::horizontal([Constraint::Percentage(35), Constraint::Percentage(65)])
                .areas(area);

        // Info panel
        let info_block = Block::default()
            .title(" Info ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let info_inner = info_block.inner(info_area);
        info_block.render(info_area, buf);

        let mut info_lines = vec![
            Line::from(vec![
                Span::styled("Run:    ", THEME.kv_key),
                Span::styled(&job.id, THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Model:  ", THEME.kv_key),
                Span::styled(&job.model, THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Method: ", THEME.kv_key),
                Span::styled(&job.method, THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Output: ", THEME.kv_key),
                Span::styled(&job.output_dir, THEME.text_dim),
            ]),
        ];

        if let Some(loss) = job.final_loss {
            info_lines.push(Line::from(vec![
                Span::styled("Loss:   ", THEME.kv_key),
                Span::styled(format!("{loss:.4}"), THEME.text_success),
            ]));
        }

        Paragraph::new(info_lines)
            .wrap(Wrap { trim: false })
            .render(info_inner, buf);

        // Log panel
        let log_block = Block::default()
            .title(" Log ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let log_inner = log_block.inner(log_area);
        log_block.render(log_area, buf);

        if job.log_lines.is_empty() {
            Paragraph::new("No log file found")
                .style(THEME.text_muted)
                .render(log_inner, buf);
        } else {
            let lines: Vec<Line> = job
                .log_lines
                .iter()
                .map(|l| Line::from(Span::styled(l.as_str(), THEME.text_dim)))
                .collect();

            Paragraph::new(lines)
                .scroll((self.log_scroll as u16, 0))
                .render(log_inner, buf);
        }
    }
}

fn read_training_meta(path: &std::path::Path) -> (Option<String>, Option<String>, Option<f64>) {
    let config_path = path.join("adapter_config.json");
    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            let model = json["base_model"].as_str().map(String::from);
            let method = json["method"].as_str().map(String::from);
            return (model, method, None);
        }
    }

    let args_path = path.join("training_args.json");
    if let Ok(content) = std::fs::read_to_string(&args_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            let model = json["model_name_or_path"].as_str().map(String::from);
            let loss = json["final_loss"].as_f64();
            return (model, None, loss);
        }
    }

    (None, None, None)
}
