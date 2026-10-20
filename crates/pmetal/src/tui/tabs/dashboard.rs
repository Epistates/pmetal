//! The Monitor tab: loss curve, learning rate, throughput and timing of the
//! run being followed.
//!
//! Steps come from one of two places: the events of a job the interface
//! started, or a metrics file given on the command line
//! (`pmetal tui --metrics-file`). Either way they are the trainer's own
//! [`StepMetrics`], read by the same parser the job supervisor uses.

use pmetal_core::StepMetrics;
use pmetal_supervisor::JobId;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Sparkline, Widget, Wrap,
};

use crate::tui::theme::{THEME, chart};

/// Steps kept for the stats and the timing panel (the newest).
const MAX_STEPS: usize = 5_000;
/// Points kept for the charts. Past this the series is thinned to every other
/// point, so a run of any length stays in view as a coarser curve.
const MAX_CHART_POINTS: usize = 4_000;
/// Throughput readings kept for the sparkline.
const SPARK_POINTS: usize = 120;

/// What the training tabs need from the monitor to draw their charts.
pub struct Telemetry<'a> {
    /// The newest steps of the run being followed, oldest first.
    pub steps: &'a [StepMetrics],
    /// The loss curve of the run, thinned for drawing.
    pub loss: &'a [(f64, f64)],
    /// The run the monitor is following (its charts belong to that tab).
    pub following: Option<&'a str>,
}

impl Telemetry<'_> {
    /// No run is being followed.
    #[cfg(test)]
    pub fn none() -> Telemetry<'static> {
        Telemetry {
            steps: &[],
            loss: &[],
            following: None,
        }
    }

    /// Whether these charts are the run `job`'s.
    pub fn is_of(&self, job: Option<&str>) -> bool {
        job.is_some() && self.following == job
    }
}

/// Monitor tab state.
pub struct DashboardTab {
    steps: Vec<StepMetrics>,
    loss_data: Vec<(f64, f64)>,
    lr_data: Vec<(f64, f64)>,
    throughput: Vec<u64>,
    /// Steps the run reported with a loss that is not a number.
    diverged: u64,
    pub paused: bool,
    /// What the followed job says it is doing, until its first step.
    pub phase: Option<String>,
    following: Option<JobId>,
    /// Where the steps come from, for the title.
    source: Option<String>,
}

impl Default for DashboardTab {
    fn default() -> Self {
        Self::new()
    }
}

impl DashboardTab {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            loss_data: Vec::new(),
            lr_data: Vec::new(),
            throughput: Vec::new(),
            diverged: 0,
            paused: false,
            phase: None,
            following: None,
            source: None,
        }
    }

    /// Follow a job: clear the charts and take its steps from now on.
    pub fn follow(&mut self, job: JobId, label: impl Into<String>) {
        self.reset();
        self.following = Some(job);
        self.source = Some(label.into());
        self.phase = Some("Starting…".to_string());
    }

    /// Show the steps of a metrics file.
    pub fn follow_file(&mut self, label: impl Into<String>) {
        self.reset();
        self.following = None;
        self.source = Some(label.into());
    }

    /// Stop following the job (it ended). The charts stay for reading.
    pub fn release(&mut self, job: &str) {
        if self.following.as_deref() == Some(job) {
            self.following = None;
            self.phase = None;
        }
    }

    pub fn following(&self) -> Option<&str> {
        self.following.as_deref()
    }

    /// The charts' data for the training tabs.
    pub fn telemetry(&self) -> Telemetry<'_> {
        Telemetry {
            steps: &self.steps,
            loss: &self.loss_data,
            following: self.following.as_deref(),
        }
    }

    #[cfg(test)]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Take one step of the run. Ignored while paused.
    pub fn push_step(&mut self, step: StepMetrics) {
        if self.paused {
            return;
        }
        self.phase = None;
        // A loss that is not a number would wreck the chart's bounds, but
        // dropping it hides the one thing the person watching needs to know.
        if !step.loss.is_finite() {
            self.diverged += 1;
            return;
        }
        let x = step.step as f64;
        self.loss_data.push((x, step.loss));
        if step.lr.is_finite() {
            self.lr_data.push((x, step.lr));
        }
        self.throughput.push(step.tok_sec.max(0.0) as u64);
        if self.throughput.len() > SPARK_POINTS * 2 {
            self.throughput
                .drain(..self.throughput.len() - SPARK_POINTS);
        }
        self.steps.push(step);
        if self.steps.len() > MAX_STEPS * 2 {
            self.steps.drain(..self.steps.len() - MAX_STEPS);
        }
        thin(&mut self.loss_data);
        thin(&mut self.lr_data);
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn reset(&mut self) {
        self.steps.clear();
        self.loss_data.clear();
        self.lr_data.clear();
        self.throughput.clear();
        self.diverged = 0;
        self.phase = None;
    }
}

/// Halve a series that has grown past [`MAX_CHART_POINTS`], keeping its ends.
fn thin(series: &mut Vec<(f64, f64)>) {
    if series.len() <= MAX_CHART_POINTS * 2 {
        return;
    }
    let last = series.last().copied();
    let mut kept: Vec<(f64, f64)> = series.iter().step_by(2).copied().collect();
    if let (Some(last), Some(kept_last)) = (last, kept.last().copied()) {
        if last != kept_last {
            kept.push(last);
        }
    }
    *series = kept;
}

impl DashboardTab {
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let [top, bottom] =
            Layout::vertical([Constraint::Percentage(60), Constraint::Percentage(40)]).areas(area);

        let [chart_area, stats_area] =
            Layout::horizontal([Constraint::Percentage(65), Constraint::Percentage(35)]).areas(top);

        let [throughput_area, timing_area] =
            Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)])
                .areas(bottom);

        self.render_loss_chart(chart_area, buf);
        self.render_stats(stats_area, buf);
        self.render_throughput(throughput_area, buf);
        render_timing(self.steps.last(), &self.steps, timing_area, buf);
    }

    fn render_loss_chart(&self, area: Rect, buf: &mut Buffer) {
        let title = match (&self.source, self.paused) {
            (_, true) => " Loss Curve [PAUSED] ".to_string(),
            (Some(source), false) => format!(" Loss Curve — {source} "),
            (None, false) => " Loss Curve ".to_string(),
        };
        let block = Block::default()
            .title(title)
            .title_style(if self.paused {
                THEME.text_warning
            } else {
                THEME.block_title
            })
            .borders(Borders::ALL)
            .border_style(THEME.block);

        if self.loss_data.is_empty() {
            let lines = if let Some(ref phase) = self.phase {
                vec![
                    Line::from(""),
                    Line::from(Span::styled(format!("  {phase}"), THEME.text_warning)),
                    Line::from(""),
                    Line::from(Span::styled(
                        "  The loss curve appears once training begins.",
                        THEME.text_dim,
                    )),
                ]
            } else if self.diverged > 0 {
                vec![
                    Line::from(""),
                    Line::from(Span::styled(
                        "  The loss is NaN or infinite: the run has diverged.",
                        THEME.text_error,
                    )),
                    Line::from(Span::styled(
                        "  Lower the learning rate or check the data.",
                        THEME.text_dim,
                    )),
                ]
            } else {
                vec![
                    Line::from(""),
                    Line::from(Span::styled("  No run to show.", THEME.text_muted)),
                    Line::from(""),
                    Line::from(Span::styled(
                        "  Start one from the Training, Distill, GRPO, RLKD or Preference tab,",
                        THEME.text_dim,
                    )),
                    Line::from(Span::styled(
                        "  or open this view on a file: pmetal tui --metrics-file <path>.",
                        THEME.text_dim,
                    )),
                ]
            };
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(block)
                .render(area, buf);
            return;
        }

        let (min_loss, max_loss) = self
            .loss_data
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), (_, y)| {
                (lo.min(*y), hi.max(*y))
            });
        let max_step = self.loss_data.last().map_or(1.0, |(x, _)| *x).max(1.0);

        let datasets = vec![
            Dataset::default()
                .name("loss")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(chart(0))
                .data(&self.loss_data),
        ];

        // Avoid a zero-height Y axis when every loss is the same.
        let y_range = (max_loss - min_loss).max(0.001);
        let y_min = min_loss - y_range * 0.05;
        let y_max = max_loss + y_range * 0.05;

        Chart::new(datasets)
            .block(block)
            .x_axis(
                Axis::default()
                    .title("Step")
                    .style(THEME.chart_axis)
                    .bounds([0.0, max_step]),
            )
            .y_axis(
                Axis::default()
                    .title("Loss")
                    .style(THEME.chart_axis)
                    .bounds([y_min, y_max])
                    .labels::<Vec<Line<'_>>>(vec![
                        format!("{y_min:.3}").into(),
                        format!("{:.3}", (y_min + y_max) / 2.0).into(),
                        format!("{y_max:.3}").into(),
                    ]),
            )
            .render(area, buf);
    }

    fn render_stats(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Stats ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let kv = |key: &'static str, value: String, style| {
            Line::from(vec![
                Span::styled(format!("{key:<10}"), THEME.kv_key),
                Span::styled(value, style),
            ])
        };
        let mut lines: Vec<Line<'_>> = Vec::new();
        if let Some(last) = self.steps.last() {
            let trend = loss_trend(&self.steps);
            let epoch = if last.total_epochs > 0 {
                format!("{}/{}", last.epoch + 1, last.total_epochs)
            } else {
                format!("{}", last.epoch + 1)
            };
            let step = if last.total_steps > 0 {
                format!("{}/{}", last.step, last.total_steps)
            } else {
                format!("{}", last.step)
            };
            lines.push(kv("Epoch:", epoch, THEME.kv_value));
            lines.push(kv("Step:", step, THEME.kv_value));
            lines.push(Line::from(vec![
                Span::styled(format!("{:<10}", "Loss:"), THEME.kv_key),
                Span::styled(format!("{:.4}", last.loss), THEME.kv_value),
                Span::styled(trend, THEME.text_dim),
            ]));
            lines.push(kv("LR:", format!("{:.2e}", last.lr), THEME.kv_value));
            lines.push(kv(
                "Tok/sec:",
                format!("{:.0}", last.tok_sec),
                THEME.kv_value,
            ));
            if let Some(norm) = last.grad_norm {
                lines.push(kv("Grad norm:", format!("{norm:.3}"), THEME.kv_value));
            }
            lines.push(Line::from(""));
            let losses: Vec<f64> = self.steps.iter().map(|s| s.loss).collect();
            let min = losses.iter().copied().fold(f64::INFINITY, f64::min);
            let avg = losses.iter().sum::<f64>() / losses.len() as f64;
            lines.push(kv("Min loss:", format!("{min:.4}"), THEME.text_success));
            lines.push(kv("Avg loss:", format!("{avg:.4}"), THEME.text_dim));
            if self.diverged > 0 {
                lines.push(Line::from(Span::styled(
                    format!("{} NaN/inf loss step(s)", self.diverged),
                    THEME.text_error,
                )));
            }
        } else if let Some(ref phase) = self.phase {
            lines.push(Line::from(Span::styled(phase.as_str(), THEME.text_warning)));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Waiting for the first step…",
                THEME.text_muted,
            )));
        } else {
            lines.push(Line::from(Span::styled("No data yet", THEME.text_muted)));
            if self.diverged > 0 {
                lines.push(Line::from(Span::styled(
                    format!("{} NaN/inf loss step(s)", self.diverged),
                    THEME.text_error,
                )));
            }
        }
        Paragraph::new(lines).render(inner, buf);
    }

    fn render_throughput(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Throughput (tok/s) ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        if self.throughput.is_empty() {
            Paragraph::new("\n  No data")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }
        let start = self.throughput.len().saturating_sub(SPARK_POINTS);
        Sparkline::default()
            .block(block)
            .data(&self.throughput[start..])
            .style(chart(3))
            .render(area, buf);
    }
}

/// Whether the loss is going down, from the last ten steps.
fn loss_trend(steps: &[StepMetrics]) -> &'static str {
    if steps.len() < 10 {
        return "";
    }
    let mean = |s: &[StepMetrics]| s.iter().map(|s| s.loss).sum::<f64>() / s.len() as f64;
    let recent = mean(&steps[steps.len() - 5..]);
    let previous = mean(&steps[steps.len() - 10..steps.len() - 5]);
    if recent < previous * 0.99 {
        " (decreasing)"
    } else if recent > previous * 1.01 {
        " (increasing)"
    } else {
        " (plateau)"
    }
}

/// The parts of a step's time the trainer measured, as `(name, ms)`; the ones
/// it did not measure are zero and left out.
pub fn timing_components(step: &StepMetrics) -> Vec<(&'static str, f64)> {
    [
        ("GPU fwd+bwd", step.gpu_fwd_bwd_ms),
        ("ANE fwd", step.ane_fwd_ms),
        ("ANE bwd", step.ane_bwd_ms),
        ("RMSNorm", step.rmsnorm_ms),
        ("cblas dW", step.cblas_ms),
        ("Adam", step.adam_ms),
        ("Optimizer", step.optimizer_ms),
        ("I/O staging", step.io_staging_ms),
        ("Overhead", step.overhead_ms),
    ]
    .into_iter()
    .filter(|(_, ms)| ms.is_finite() && *ms > 0.0)
    .collect()
}

/// Draw the timing breakdown of the newest step in a bordered panel.
pub fn render_timing(
    last: Option<&StepMetrics>,
    steps: &[StepMetrics],
    area: Rect,
    buf: &mut Buffer,
) {
    let block = Block::default()
        .title(" Timing Breakdown ")
        .title_style(THEME.block_title)
        .borders(Borders::ALL)
        .border_style(THEME.block);
    let inner = block.inner(area);
    block.render(area, buf);

    let Some(last) = last else {
        Paragraph::new("No timing data")
            .style(THEME.text_muted)
            .render(inner, buf);
        return;
    };

    let components = timing_components(last);
    let measured: f64 = components.iter().map(|(_, ms)| ms).sum();
    // The trainer's own total when it gave one, else what was measured.
    let total = if last.total_ms > 0.0 {
        last.total_ms
    } else {
        measured
    };
    if components.is_empty() && total <= 0.0 {
        Paragraph::new(vec![
            Line::from(Span::styled("No timing data", THEME.text_muted)),
            Line::from(vec![
                Span::styled("Tok/sec: ", THEME.kv_key),
                Span::styled(format!("{:.0}", last.tok_sec), THEME.kv_value),
            ]),
        ])
        .render(inner, buf);
        return;
    }

    // One row per component, two rows tall each when there is room.
    let row_height = if inner.height as usize >= components.len() * 2 + 2 {
        2
    } else {
        1
    };
    let mut y = inner.y;
    for (i, (name, ms)) in components.iter().enumerate() {
        if y + row_height > inner.bottom().saturating_sub(1) {
            break;
        }
        let ratio = if total > 0.0 {
            (ms / total).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Gauge::default()
            .gauge_style(chart(i))
            .ratio(ratio)
            .label(Span::styled(
                format!("{name:>11}: {ms:7.1}ms ({:.0}%)", ratio * 100.0),
                THEME.text,
            ))
            .render(Rect::new(inner.x, y, inner.width, row_height), buf);
        y += row_height;
    }
    if y < inner.bottom() {
        let per_minute = if total > 1.0 {
            format!("  ({:.0} steps/min)", 60_000.0 / total)
        } else {
            String::new()
        };
        let recent = steps.len().saturating_sub(10);
        let window = &steps[recent..];
        let average = (window.len() >= 5)
            .then(|| window.iter().map(|s| s.total_ms).sum::<f64>() / window.len() as f64)
            .filter(|avg| *avg > 0.0)
            .map(|avg| format!("  avg {avg:.1}ms"))
            .unwrap_or_default();
        Line::from(vec![
            Span::styled("   Step: ", THEME.kv_key),
            Span::styled(format!("{total:.1}ms"), THEME.kv_value),
            Span::styled(format!("{per_minute}{average}"), THEME.text_dim),
        ])
        .render(Rect::new(inner.x, y, inner.width, 1), buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(n: usize, loss: f64) -> StepMetrics {
        StepMetrics {
            step: n,
            total_steps: 100,
            total_epochs: 2,
            loss,
            lr: 1e-4,
            tok_sec: 1000.0 + n as f64,
            total_ms: 200.0,
            gpu_fwd_bwd_ms: 150.0,
            optimizer_ms: 30.0,
            ..StepMetrics::default()
        }
    }

    fn draw(tab: &DashboardTab, w: u16, h: u16) -> String {
        let area = Rect::new(0, 0, w, h);
        let mut buf = Buffer::empty(area);
        tab.render(area, &mut buf);
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn an_empty_monitor_says_how_to_get_something_to_show() {
        let screen = draw(&DashboardTab::new(), 100, 30);
        assert!(screen.contains("No run to show"), "{screen}");
        assert!(screen.contains("--metrics-file"));
        assert!(screen.contains("No timing data"));
    }

    #[test]
    fn a_followed_job_shows_its_phase_until_its_first_step() {
        let mut tab = DashboardTab::new();
        tab.follow("j1".into(), "Training");
        tab.phase = Some("Loading model…".into());
        let screen = draw(&tab, 100, 30);
        assert!(screen.contains("Loading model…"), "{screen}");
        assert!(screen.contains("Waiting for the first step"));
        tab.push_step(step(1, 2.5));
        assert!(tab.phase.is_none());
        let screen = draw(&tab, 100, 30);
        assert!(screen.contains("Loss Curve — Training"), "{screen}");
        assert!(screen.contains("1/100"), "{screen}");
        assert!(screen.contains("2.5000"));
        assert!(screen.contains("1/2"), "epoch 1 of 2: {screen}");
    }

    #[test]
    fn the_timing_panel_shows_what_the_trainer_measured_and_nothing_else() {
        let components = timing_components(&step(1, 1.0));
        assert_eq!(components, [("GPU fwd+bwd", 150.0), ("Optimizer", 30.0)]);
        let mut tab = DashboardTab::new();
        tab.push_step(step(1, 1.0));
        let screen = draw(&tab, 100, 30);
        assert!(screen.contains("GPU fwd+bwd"), "{screen}");
        assert!(screen.contains("Optimizer"));
        assert!(!screen.contains("ANE fwd"));
        assert!(screen.contains("300 steps/min"), "{screen}");
    }

    #[test]
    fn a_diverged_run_is_reported_not_hidden() {
        let mut tab = DashboardTab::new();
        tab.push_step(step(1, f64::NAN));
        tab.push_step(step(2, f64::INFINITY));
        let screen = draw(&tab, 100, 30);
        assert!(screen.contains("has diverged"), "{screen}");
        assert!(screen.contains("2 NaN/inf loss step(s)"), "{screen}");
        // Finite steps after it chart as usual and the warning stays in stats.
        tab.push_step(step(3, 4.0));
        let screen = draw(&tab, 100, 30);
        assert!(screen.contains("2 NaN/inf loss step(s)"), "{screen}");
    }

    #[test]
    fn pausing_ignores_steps_and_reset_clears_them() {
        let mut tab = DashboardTab::new();
        tab.push_step(step(1, 2.0));
        tab.toggle_pause();
        tab.push_step(step(2, 1.0));
        assert_eq!(tab.step_count(), 1);
        assert!(draw(&tab, 100, 30).contains("[PAUSED]"));
        tab.toggle_pause();
        tab.reset();
        assert_eq!(tab.step_count(), 0);
    }

    #[test]
    fn a_very_long_run_stays_bounded() {
        let mut tab = DashboardTab::new();
        for n in 0..40_000 {
            tab.push_step(step(n, 3.0 - (n as f64) * 1e-5));
        }
        assert!(tab.steps.len() <= MAX_STEPS * 2, "{}", tab.steps.len());
        assert!(tab.loss_data.len() <= MAX_CHART_POINTS * 2);
        assert!(tab.throughput.len() <= SPARK_POINTS * 2);
        // The curve still reaches the last step.
        assert_eq!(tab.loss_data.last().unwrap().0, 39_999.0);
        draw(&tab, 100, 30);
    }

    #[test]
    fn releasing_a_job_keeps_the_charts_but_stops_following() {
        let mut tab = DashboardTab::new();
        tab.follow("j1".into(), "Training");
        tab.push_step(step(1, 2.0));
        tab.release("other");
        assert_eq!(tab.following(), Some("j1"));
        tab.release("j1");
        assert_eq!(tab.following(), None);
        assert_eq!(tab.step_count(), 1);
    }

    #[test]
    fn the_loss_trend_reads_the_last_ten_steps() {
        let falling: Vec<StepMetrics> = (0..10).map(|n| step(n, 5.0 - n as f64 * 0.3)).collect();
        assert_eq!(loss_trend(&falling), " (decreasing)");
        let flat: Vec<StepMetrics> = (0..10).map(|n| step(n, 2.0)).collect();
        assert_eq!(loss_trend(&flat), " (plateau)");
        assert_eq!(loss_trend(&falling[..4]), "");
    }

    #[test]
    fn drawing_never_panics_at_any_size() {
        let mut tab = DashboardTab::new();
        for n in 0..30 {
            tab.push_step(step(n, 3.0 - n as f64 * 0.01));
        }
        for (w, h) in [
            (0, 0),
            (1, 1),
            (5, 3),
            (20, 8),
            (40, 12),
            (100, 4),
            (200, 60),
        ] {
            draw(&tab, w, h);
            draw(&DashboardTab::new(), w, h);
        }
    }
}
