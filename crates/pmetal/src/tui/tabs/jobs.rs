//! Jobs tab: what is running, what ran this session, and finished runs on disk.
//!
//! The live part is the supervisor's: each job it runs has a row with its
//! status and a log that follows it. The history part is the output directories
//! of earlier runs, read in the background.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use pmetal_supervisor::{EventKind, JobId, JobMetrics, ProcessStatus, Stream};
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, HighlightSpacing, Paragraph, Row, Scrollbar, ScrollbarOrientation,
    ScrollbarState, StatefulWidget, Table, TableState, Widget, Wrap,
};

use crate::tui::theme::THEME;
use crate::tui::widgets::{LogView, format_duration};

/// Finished jobs of this session kept in the list.
const MAX_LIVE_ROWS: usize = 40;

/// A job of this session.
pub struct LiveJob {
    pub id: JobId,
    /// What started it ("Training", "Download").
    pub label: String,
    /// The command line.
    pub command: String,
    pub status: ProcessStatus,
    pub metrics: JobMetrics,
    pub phase: Option<String>,
    pub started: Instant,
    pub took: Option<Duration>,
    pub log: LogView,
}

impl LiveJob {
    fn status_word(&self) -> &'static str {
        match self.status {
            ProcessStatus::Running => "Running",
            ProcessStatus::Stopping => "Stopping",
            ProcessStatus::Completed { .. } => "Done",
            ProcessStatus::Failed { .. } => "Failed",
            ProcessStatus::Stopped => "Stopped",
        }
    }

    fn status_style(&self) -> ratatui::style::Style {
        match self.status {
            ProcessStatus::Running | ProcessStatus::Stopping => THEME.status_running,
            ProcessStatus::Completed { .. } => THEME.status_success,
            ProcessStatus::Failed { .. } => THEME.status_error,
            ProcessStatus::Stopped => THEME.status_idle,
        }
    }
}

/// A finished run found on disk.
#[derive(Debug, Clone)]
pub struct JobEntry {
    pub id: String,
    pub model: String,
    pub output_dir: PathBuf,
    pub final_loss: Option<f64>,
    pub modified: std::time::SystemTime,
}

/// Which row is selected: a job of this session, or a finished run on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection<'a> {
    Live(&'a str),
    History(usize),
}

/// Jobs tab state.
pub struct JobsTab {
    pub live: Vec<LiveJob>,
    pub history: Vec<JobEntry>,
    pub table_state: TableState,
    pub scrollbar_state: ScrollbarState,
    pub scanning: bool,
}

impl JobsTab {
    pub fn new() -> Self {
        Self {
            live: Vec::new(),
            history: Vec::new(),
            table_state: TableState::default(),
            scrollbar_state: ScrollbarState::default(),
            scanning: true,
        }
    }

    fn rows(&self) -> usize {
        self.live.len() + self.history.len()
    }

    // ── This session's jobs ─────────────────────────────────────────────

    /// A job was started.
    pub fn started(&mut self, id: JobId, label: impl Into<String>, command: String) {
        let mut log = LogView::new(crate::tui::widgets::DEFAULT_LOG_CAP);
        log.push(format!("$ {command}"), false);
        self.live.insert(
            0,
            LiveJob {
                id,
                label: label.into(),
                command,
                status: ProcessStatus::Running,
                metrics: JobMetrics::default(),
                phase: None,
                started: Instant::now(),
                took: None,
                log,
            },
        );
        // Make room by forgetting the oldest finished jobs.
        while self.live.len() > MAX_LIVE_ROWS {
            if let Some(pos) = self.live.iter().rposition(|j| !j.status.is_active()) {
                self.live.remove(pos);
            } else {
                break;
            }
        }
        if self.table_state.selected().is_none() {
            self.table_state.select(Some(0));
        } else if let Some(i) = self.table_state.selected() {
            // The new row went in at the top: the selection follows what it was on.
            self.table_state
                .select(Some((i + 1).min(self.rows().saturating_sub(1))));
        }
        self.sync_scrollbar();
    }

    /// Fold an event for one of this session's jobs into its row.
    pub fn apply(&mut self, id: &str, kind: &EventKind) {
        let Some(job) = self.live.iter_mut().find(|j| j.id == id) else {
            return;
        };
        match kind {
            EventKind::Output(line) => job
                .log
                .push(line.text.clone(), line.stream == Stream::Stderr),
            EventKind::Job(event) => {
                job.metrics.apply(event);
                if let pmetal_core::events::JobEvent::Phase { phase, .. } = event {
                    job.phase = Some(phase.message());
                }
            }
            EventKind::Exited(status) => {
                job.status = status.clone();
                job.took = Some(job.started.elapsed());
                job.phase = None;
            }
            EventKind::Started | EventKind::Delta(_) => {}
        }
    }

    // ── Finished runs on disk ───────────────────────────────────────────

    pub fn begin_scan(&mut self) {
        self.scanning = true;
    }

    pub fn set_history(&mut self, history: Vec<JobEntry>) {
        let selected = self.selected_history().map(|h| h.output_dir.clone());
        self.history = history;
        self.scanning = false;
        if let Some(path) = selected {
            if let Some(pos) = self.history.iter().position(|h| h.output_dir == path) {
                self.table_state.select(Some(self.live.len() + pos));
            }
        }
        if self.table_state.selected().is_none() && self.rows() > 0 {
            self.table_state.select(Some(0));
        }
        let max = self.rows().saturating_sub(1);
        if let Some(i) = self.table_state.selected() {
            self.table_state.select(Some(i.min(max)));
        }
        self.sync_scrollbar();
    }

    // ── Selection and scrolling ─────────────────────────────────────────

    fn sync_scrollbar(&mut self) {
        self.scrollbar_state =
            ScrollbarState::new(self.rows()).position(self.table_state.selected().unwrap_or(0));
    }

    pub fn next_row(&mut self) {
        let count = self.rows();
        if count == 0 {
            return;
        }
        let i = self.table_state.selected().map_or(0, |i| (i + 1) % count);
        self.table_state.select(Some(i));
        self.sync_scrollbar();
    }

    pub fn prev_row(&mut self) {
        let count = self.rows();
        if count == 0 {
            return;
        }
        let i = self
            .table_state
            .selected()
            .map_or(0, |i| (i + count - 1) % count);
        self.table_state.select(Some(i));
        self.sync_scrollbar();
    }

    pub fn selection(&self) -> Option<Selection<'_>> {
        let i = self.table_state.selected()?;
        if i < self.live.len() {
            Some(Selection::Live(&self.live[i].id))
        } else if i - self.live.len() < self.history.len() {
            Some(Selection::History(i - self.live.len()))
        } else {
            None
        }
    }

    pub fn selected_live(&self) -> Option<&LiveJob> {
        match self.selection()? {
            Selection::Live(id) => self.live.iter().find(|j| j.id == id),
            Selection::History(_) => None,
        }
    }

    fn selected_live_mut(&mut self) -> Option<&mut LiveJob> {
        let i = self.table_state.selected()?;
        self.live.get_mut(i)
    }

    pub fn selected_history(&self) -> Option<&JobEntry> {
        match self.selection()? {
            Selection::History(i) => self.history.get(i),
            Selection::Live(_) => None,
        }
    }

    pub fn scroll_log_up(&mut self, rows: usize) {
        if let Some(job) = self.selected_live_mut() {
            job.log.scroll_up(rows);
        }
    }

    pub fn scroll_log_down(&mut self, rows: usize) {
        if let Some(job) = self.selected_live_mut() {
            job.log.scroll_down(rows);
        }
    }

    pub fn follow_log(&mut self) {
        if let Some(job) = self.selected_live_mut() {
            job.log.follow();
        }
    }
}

impl Default for JobsTab {
    fn default() -> Self {
        Self::new()
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
            .title(format!(
                " Jobs ({} this session, {} finished runs) ",
                self.live.len(),
                self.history.len()
            ))
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        if self.rows() == 0 {
            let text = if self.scanning {
                "\n  Looking for finished runs..."
            } else {
                "\n  Nothing has run yet. Jobs you start appear here, with their output."
            };
            Paragraph::new(text)
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }

        let header = Row::new(vec!["Job", "What", "Status", "Loss / progress", "When"])
            .style(THEME.table_header)
            .height(1);

        let mut rows: Vec<Row<'_>> = Vec::new();
        for (i, job) in self.live.iter().enumerate() {
            let style = if i % 2 == 0 {
                THEME.table_row
            } else {
                THEME.table_row_alt
            };
            let progress = match (job.metrics.last_step, job.metrics.total_steps) {
                (Some(step), Some(total)) => format!("{step}/{total}"),
                (Some(step), None) => format!("step {step}"),
                _ => job.phase.clone().unwrap_or_else(|| "-".to_string()),
            };
            let progress = match job.metrics.last_loss {
                Some(loss) => format!("{progress}  loss {loss:.3}"),
                None => progress,
            };
            let elapsed = job.took.unwrap_or_else(|| job.started.elapsed());
            rows.push(
                Row::new(vec![
                    Cell::new(job.label.clone()),
                    Cell::new(job.command.clone()),
                    Cell::new(Span::styled(job.status_word(), job.status_style())),
                    Cell::new(progress),
                    Cell::new(format_duration(elapsed)),
                ])
                .style(style),
            );
        }
        for (j, entry) in self.history.iter().enumerate() {
            let style = if (self.live.len() + j) % 2 == 0 {
                THEME.table_row
            } else {
                THEME.table_row_alt
            };
            rows.push(
                Row::new(vec![
                    Cell::new(entry.id.clone()),
                    Cell::new(entry.model.clone()),
                    Cell::new(Span::styled("Finished", THEME.status_idle)),
                    Cell::new(
                        entry
                            .final_loss
                            .map_or("-".to_string(), |l| format!("loss {l:.4}")),
                    ),
                    Cell::new(age(entry.modified)),
                ])
                .style(style),
            );
        }

        let table = Table::new(
            rows,
            [
                Constraint::Length(18),
                Constraint::Fill(1),
                Constraint::Length(9),
                Constraint::Length(24),
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

    fn render_detail(&mut self, area: Rect, buf: &mut Buffer) {
        if let Some(i) = self.table_state.selected() {
            if i < self.live.len() {
                let job = &mut self.live[i];
                let [info_area, log_area] =
                    Layout::horizontal([Constraint::Percentage(35), Constraint::Percentage(65)])
                        .areas(area);
                render_live_info(job, info_area, buf);
                job.log.render(log_area, buf, "Log");
                return;
            }
        }
        let block = Block::default()
            .title(" Details ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);
        match self.selected_history() {
            Some(entry) => {
                let lines = vec![
                    Line::from(vec![
                        Span::styled("Run:    ", THEME.kv_key),
                        Span::styled(entry.id.clone(), THEME.kv_value),
                    ]),
                    Line::from(vec![
                        Span::styled("Model:  ", THEME.kv_key),
                        Span::styled(entry.model.clone(), THEME.kv_value),
                    ]),
                    Line::from(vec![
                        Span::styled("Output: ", THEME.kv_key),
                        Span::styled(entry.output_dir.display().to_string(), THEME.text_dim),
                    ]),
                    Line::from(vec![
                        Span::styled("Loss:   ", THEME.kv_key),
                        Span::styled(
                            entry
                                .final_loss
                                .map_or("not recorded".to_string(), |l| format!("{l:.4}")),
                            THEME.text_success,
                        ),
                    ]),
                    Line::from(""),
                    Line::from(Span::styled(
                        "The output of a run from an earlier session is not kept.",
                        THEME.text_muted,
                    )),
                ];
                Paragraph::new(lines)
                    .wrap(Wrap { trim: false })
                    .render(inner, buf);
            }
            None => {
                Paragraph::new("Select a job to see its output")
                    .style(THEME.text_muted)
                    .render(inner, buf);
            }
        }
    }
}

fn render_live_info(job: &LiveJob, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .title(" Info ")
        .title_style(THEME.block_title)
        .borders(Borders::ALL)
        .border_style(THEME.block);
    let inner = block.inner(area);
    block.render(area, buf);
    let kv = |key: &'static str, value: String, style| {
        Line::from(vec![
            Span::styled(format!("{key:<9}"), THEME.kv_key),
            Span::styled(value, style),
        ])
    };
    let mut lines = vec![
        kv("Job:", job.label.clone(), THEME.kv_value),
        Line::from(vec![
            Span::styled(format!("{:<9}", "Status:"), THEME.kv_key),
            Span::styled(job.status_word(), job.status_style()),
        ]),
        kv(
            "Elapsed:",
            format_duration(job.took.unwrap_or_else(|| job.started.elapsed())),
            THEME.text_dim,
        ),
    ];
    if let ProcessStatus::Failed { error, .. } = &job.status {
        lines.push(kv("Error:", error.clone(), THEME.text_error));
    }
    if let Some(phase) = &job.phase {
        lines.push(kv("Phase:", phase.clone(), THEME.text_warning));
    }
    let m = &job.metrics;
    if let Some(step) = m.last_step {
        let steps = m
            .total_steps
            .map_or(step.to_string(), |t| format!("{step}/{t}"));
        lines.push(kv("Step:", steps, THEME.kv_value));
    }
    if let Some(loss) = m.last_loss {
        lines.push(kv("Loss:", format!("{loss:.4}"), THEME.kv_value));
    }
    if let Some(eval) = m.eval_loss {
        lines.push(kv("Eval:", format!("{eval:.4}"), THEME.kv_value));
    }
    if let Some(tps) = m.tokens_per_second {
        lines.push(kv("Tok/sec:", format!("{tps:.0}"), THEME.kv_value));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        job.command.clone(),
        THEME.text_muted,
    )));
    Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .render(inner, buf);
}

fn age(time: std::time::SystemTime) -> String {
    let secs = time.elapsed().unwrap_or_default().as_secs();
    if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

// ── Finished runs on disk ───────────────────────────────────────────────

/// The directories finished runs are found under: `./output` and, where
/// earlier versions put them, `~/.pmetal/output`.
pub fn history_roots() -> Vec<PathBuf> {
    let mut roots = vec![PathBuf::from("./output")];
    if let Some(home) = dirs::home_dir() {
        roots.push(home.join(".pmetal").join("output"));
    }
    roots
}

/// Find the finished runs under `roots`, newest first. Blocking.
pub fn scan_history(roots: &[PathBuf]) -> Vec<JobEntry> {
    let mut found = Vec::new();
    for root in roots {
        collect_runs(root, &mut found, 0);
    }
    found.sort_by_key(|e| std::cmp::Reverse(e.modified));
    found
}

fn collect_runs(dir: &Path, out: &mut Vec<JobEntry>, depth: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if !path.is_dir() || name.starts_with('.') || name == "checkpoints" {
            continue;
        }
        if looks_like_a_run(&path) {
            let meta = read_run_meta(&path);
            out.push(JobEntry {
                id: name,
                model: meta.0.unwrap_or_else(|| "-".to_string()),
                final_loss: meta.1,
                modified: std::fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::UNIX_EPOCH),
                output_dir: path,
            });
        } else if depth < 2 {
            collect_runs(&path, out, depth + 1);
        }
    }
}

/// A run's output: adapter files, trainer arguments, or weights.
fn looks_like_a_run(path: &Path) -> bool {
    path.join("adapter_config.json").exists()
        || path.join("training_args.json").exists()
        || pmetal_core::TrainingInfo::path_in(path).exists()
        || std::fs::read_dir(path).is_ok_and(|entries| {
            entries
                .flatten()
                .any(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
        })
}

/// The base model and the last loss a run recorded.
fn read_run_meta(path: &Path) -> (Option<String>, Option<f64>) {
    let json = |file: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(path.join(file)).ok()?).ok()
    };
    let model = pmetal_core::TrainingInfo::read(path)
        .and_then(|info| info.base_model)
        .or_else(|| {
            let j = json("adapter_config.json")?;
            j["base_model_name_or_path"]
                .as_str()
                .or_else(|| j["base_model"].as_str())
                .map(String::from)
        })
        .or_else(|| {
            json("training_args.json")?["model_name_or_path"]
                .as_str()
                .map(String::from)
        });
    let loss = last_loss(&path.join("metrics.jsonl"))
        .or_else(|| json("training_args.json").and_then(|j| j["final_loss"].as_f64()));
    (model, loss)
}

/// The loss of the last step in a metrics file, read from its tail.
fn last_loss(file: &Path) -> Option<f64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(file).ok()?;
    let len = f.metadata().ok()?.len();
    let from = len.saturating_sub(16 * 1024);
    f.seek(SeekFrom::Start(from)).ok()?;
    let mut tail = Vec::new();
    f.read_to_end(&mut tail).ok()?;
    let text = String::from_utf8_lossy(&tail);
    text.lines()
        .rev()
        .find_map(|line| match pmetal_supervisor::parse_record(line, "")? {
            pmetal_core::events::JobEvent::Metric {
                payload: pmetal_core::events::MetricPayload::Step(step),
                ..
            } if step.loss.is_finite() => Some(step.loss),
            _ => None,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_supervisor::LogLine;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn screen(tab: &mut JobsTab, w: u16, h: u16) -> String {
        let area = Rect::new(0, 0, w, h);
        let mut buf = Buffer::empty(area);
        tab.render(area, &mut buf);
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn a_job_of_this_session_shows_its_progress_and_its_output() {
        let mut tab = JobsTab::new();
        tab.set_history(Vec::new());
        tab.started("j1".into(), "Training", "pmetal train --model m".into());
        tab.apply(
            "j1",
            &EventKind::Output(LogLine {
                seq: 0,
                stream: Stream::Stdout,
                text: "loading weights".into(),
            }),
        );
        tab.apply(
            "j1",
            &EventKind::Job(pmetal_core::events::JobEvent::Metric {
                job_id: "j1".into(),
                payload: pmetal_core::events::MetricPayload::Step(pmetal_core::StepMetrics {
                    step: 7,
                    total_steps: 70,
                    loss: 1.5,
                    ..Default::default()
                }),
            }),
        );
        let text = screen(&mut tab, 140, 36);
        assert!(text.contains("Training"), "{text}");
        assert!(text.contains("Running"));
        assert!(
            text.contains("7/70") && text.contains("loss 1.500"),
            "{text}"
        );
        assert!(text.contains("loading weights"), "{text}");

        tab.apply(
            "j1",
            &EventKind::Exited(ProcessStatus::Failed {
                exit_code: 2,
                error: "out of memory".into(),
            }),
        );
        let text = screen(&mut tab, 140, 36);
        assert!(
            text.contains("Failed") && text.contains("out of memory"),
            "{text}"
        );
    }

    #[test]
    fn the_newest_job_is_on_top_and_old_finished_ones_make_room() {
        let mut tab = JobsTab::new();
        for n in 0..(MAX_LIVE_ROWS + 5) {
            tab.started(format!("j{n}"), "Eval", format!("pmetal eval {n}"));
            tab.apply(
                &format!("j{n}"),
                &EventKind::Exited(ProcessStatus::Completed { exit_code: 0 }),
            );
        }
        assert_eq!(tab.live.len(), MAX_LIVE_ROWS);
        assert_eq!(tab.live[0].id, format!("j{}", MAX_LIVE_ROWS + 4));
        // A running job is never the one forgotten.
        tab.started("run".into(), "Serve", "pmetal serve".into());
        assert!(tab.live.iter().any(|j| j.id == "run"));
    }

    #[test]
    fn selection_reaches_both_the_session_and_the_history() {
        let mut tab = JobsTab::new();
        tab.set_history(vec![JobEntry {
            id: "old".into(),
            model: "m".into(),
            output_dir: PathBuf::from("/o/old"),
            final_loss: Some(0.25),
            modified: std::time::SystemTime::now(),
        }]);
        assert_eq!(tab.selection(), Some(Selection::History(0)));
        tab.started("j1".into(), "Quantize", "pmetal quantize".into());
        // The selection stayed on the old run when the new job went above it.
        assert_eq!(tab.selection(), Some(Selection::History(0)));
        tab.prev_row();
        assert_eq!(tab.selection(), Some(Selection::Live("j1")));
        assert!(tab.selected_history().is_none());
        tab.next_row();
        assert_eq!(tab.selected_history().unwrap().id, "old");
        let text = screen(&mut tab, 140, 36);
        assert!(text.contains("0.2500"), "{text}");
    }

    #[test]
    fn a_log_scrolls_while_new_lines_arrive() {
        let mut tab = JobsTab::new();
        tab.started("j".into(), "Serve", "pmetal serve".into());
        for n in 0..50 {
            tab.apply(
                "j",
                &EventKind::Output(LogLine {
                    seq: n,
                    stream: Stream::Stdout,
                    text: format!("line {n}"),
                }),
            );
        }
        tab.scroll_log_up(10);
        assert!(!tab.live[0].log.is_following());
        tab.follow_log();
        assert!(tab.live[0].log.is_following());
        // History rows have no log to scroll, and that is not an error.
        tab.set_history(vec![]);
        tab.scroll_log_down(3);
    }

    #[test]
    fn finished_runs_are_found_with_their_base_model_and_last_loss() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        write(
            &r.join("output/a--lora/training_info.json"),
            r#"{"base_model":"org/base"}"#,
        );
        write(
            &r.join("output/a--lora/metrics.jsonl"),
            "{\"step\":1,\"loss\":3.0}\n{\"event\":\"epoch_end\",\"loss\":2.0}\n{\"step\":2,\"loss\":2.5}\n",
        );
        write(
            &r.join("output/nested/b/adapter_config.json"),
            r#"{"base_model_name_or_path":"org/other"}"#,
        );
        write(&r.join("output/not-a-run/readme.txt"), "x");
        write(&r.join("output/.hidden/adapter_config.json"), "{}");
        let runs = scan_history(&[r.join("output")]);
        let mut by_id: Vec<(&str, &str, Option<f64>)> = runs
            .iter()
            .map(|e| (e.id.as_str(), e.model.as_str(), e.final_loss))
            .collect();
        by_id.sort_by_key(|x| x.0);
        assert_eq!(
            by_id,
            [("a--lora", "org/base", Some(2.5)), ("b", "org/other", None)]
        );
    }

    #[test]
    fn the_tab_draws_at_any_size() {
        let mut tab = JobsTab::new();
        tab.started("j".into(), "Train", "pmetal train".into());
        for &(w, h) in &[(0, 0), (1, 1), (30, 8), (80, 24), (200, 60)] {
            screen(&mut tab, w, h);
            screen(&mut JobsTab::new(), w, h);
        }
    }
}
