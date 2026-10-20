//! The status box and the standard two-column layout of a job tab.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Widget, Wrap};

use crate::tui::tabs::dashboard::{Telemetry, render_timing};
use crate::tui::tabs::panel::{JobPanel, RunState};
use crate::tui::theme::{THEME, chart};
use crate::tui::widgets::{StatusTone, format_duration, status_line};

/// The tone and word for a run state.
fn tone_of(run: &RunState) -> (StatusTone, &'static str) {
    match run {
        RunState::Idle => (StatusTone::Idle, "Idle"),
        RunState::Running { .. } => (StatusTone::Running, "Running"),
        RunState::Completed { .. } => (StatusTone::Completed, "Completed"),
        RunState::Failed(_) => (StatusTone::Failed, "Failed"),
        RunState::Stopped => (StatusTone::Idle, "Stopped"),
    }
}

/// The status line and the lines under it that every job tab shows: how long a
/// run has gone, what it is doing, how it ended.
pub fn status_lines(panel: &JobPanel) -> Vec<Line<'static>> {
    let (tone, word) = tone_of(&panel.run);
    let mut lines = Vec::new();
    match &panel.run {
        RunState::Idle => {
            lines.push(status_line(tone, word, None));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  Enter edits a field, S starts the job, x stops it",
                THEME.text_muted,
            )));
        }
        RunState::Running { since, phase } => {
            let elapsed = format_duration(since.elapsed());
            lines.push(status_line(tone, word, Some(&elapsed)));
            if let Some(phase) = phase {
                lines.push(Line::from(Span::styled(
                    format!("  {phase}"),
                    THEME.text_warning,
                )));
            }
        }
        RunState::Completed { took } => {
            let detail = format!("in {}", format_duration(*took));
            lines.push(status_line(tone, word, Some(&detail)));
        }
        RunState::Failed(message) => {
            lines.push(status_line(tone, word, None));
            for line in message.lines().take(6) {
                lines.push(Line::from(Span::styled(
                    format!("  {line}"),
                    THEME.text_error,
                )));
            }
            lines.push(Line::from(Span::styled(
                "  The log has the job's output",
                THEME.text_muted,
            )));
        }
        RunState::Stopped => {
            lines.push(status_line(tone, word, Some("stopped on request")));
        }
    }
    lines
}

/// Draw `lines` in a bordered "Status" box.
pub fn render_status_box(lines: Vec<Line<'static>>, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .title(" Status ")
        .title_style(THEME.block_title)
        .borders(Borders::ALL)
        .border_style(THEME.block);
    let inner = block.inner(area);
    block.render(area, buf);
    Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .render(inner, buf);
}

/// The standard job tab: the form on the left, and on the right the status
/// box (`status_height` rows, `extra` lines after the common ones) over the
/// job's log.
pub fn render_standard(
    panel: &mut JobPanel,
    area: Rect,
    buf: &mut Buffer,
    status_height: u16,
    extra: Vec<Line<'static>>,
    log_title: &str,
) {
    let [config_area, right_area] =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(area);
    panel
        .form
        .render_list(config_area, buf, &format!("{} Configuration", panel.title));
    let [status_area, log_area] =
        Layout::vertical([Constraint::Length(status_height), Constraint::Min(0)]).areas(right_area);
    let mut lines = status_lines(panel);
    lines.extend(extra);
    render_status_box(lines, status_area, buf);
    panel.log.render(log_area, buf, log_title);
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// The status box of a tab that trains: progress, loss, throughput and the
/// timing of the newest step while it runs, the result when it is done.
pub fn render_training_status(
    panel: &JobPanel,
    telemetry: &Telemetry<'_>,
    area: Rect,
    buf: &mut Buffer,
) {
    let (title, title_style) = match &panel.run {
        RunState::Running { .. } => (" Training Monitor ", THEME.status_running),
        RunState::Completed { .. } => (" Training Complete ", THEME.status_success),
        RunState::Failed(_) => (" Training Failed ", THEME.status_error),
        RunState::Stopped => (" Training Stopped ", THEME.status_idle),
        RunState::Idle => (" Status ", THEME.block_title),
    };
    let block = Block::default()
        .title(title)
        .title_style(title_style)
        .borders(Borders::ALL)
        .border_style(if panel.is_running() {
            THEME.block_focused
        } else {
            THEME.block
        });
    let inner = block.inner(area);
    block.render(area, buf);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    // The charts are the monitor's; they are this tab's only while the monitor
    // follows this tab's job.
    let own = telemetry.is_of(panel.job.as_deref());
    let steps = if own { telemetry.steps } else { &[] };
    let m = &panel.metrics;

    match &panel.run {
        RunState::Idle | RunState::Failed(_) | RunState::Stopped => {
            let lines = status_lines(panel);
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(inner, buf);
            if matches!(panel.run, RunState::Stopped | RunState::Failed(_)) && m.last_step.is_some()
            {
                // What it had reached before it ended.
                let y = inner.y + (status_lines(panel).len() as u16 + 1).min(inner.height - 1);
                if y < inner.bottom() {
                    Line::from(Span::styled(
                        format!(
                            "  Reached step {} (loss {})",
                            m.last_step.unwrap_or(0),
                            m.last_loss.map_or("-".to_string(), |l| format!("{l:.4}"))
                        ),
                        THEME.text_dim,
                    ))
                    .render(Rect::new(inner.x, y, inner.width, 1), buf);
                }
            }
        }
        RunState::Running { since, phase } => {
            let [stats_area, spark_area, timing_area, hint_area] = Layout::vertical([
                Constraint::Length(8),
                Constraint::Length(5),
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .areas(inner);

            let epoch = match (m.epoch, m.total_epochs) {
                (Some(e), Some(t)) => format!("Epoch {e:.0}/{t}"),
                (Some(e), None) => format!("Epoch {e:.0}"),
                _ => String::new(),
            };
            let headline = vec![
                Span::styled("  ", THEME.text),
                Span::styled("● Running", THEME.status_running),
                Span::styled(
                    format!("  {}", format_duration(since.elapsed())),
                    THEME.text_dim,
                ),
                Span::styled(
                    if epoch.is_empty() {
                        String::new()
                    } else {
                        format!("  |  {epoch}")
                    },
                    THEME.status_running,
                ),
            ];
            let mut lines = vec![Line::from(headline), Line::from("")];
            // Progress: a gauge when the total is known.
            let step = m.last_step.unwrap_or(0);
            let progress = match m.total_steps {
                Some(total) if total > 0 => {
                    let ratio = (step as f64 / total as f64).clamp(0.0, 1.0);
                    Some((ratio, format!("{step}/{total} ({:.1}%)", ratio * 100.0)))
                }
                _ => None,
            };
            if let Some((ratio, label)) = &progress {
                Gauge::default()
                    .gauge_style(chart(1))
                    .ratio(*ratio)
                    .label(Span::styled(label.clone(), THEME.text))
                    .render(
                        Rect::new(
                            stats_area.x + 1,
                            stats_area.y + 2,
                            stats_area.width.saturating_sub(2),
                            1,
                        ),
                        buf,
                    );
                lines.push(Line::from(""));
            } else if m.last_step.is_some() {
                lines.push(Line::from(vec![
                    Span::styled("  Step:     ", THEME.kv_key),
                    Span::styled(format!("{step}"), THEME.kv_value),
                ]));
            } else if let Some(phase) = phase {
                lines.push(Line::from(Span::styled(
                    format!("  {phase}"),
                    THEME.text_warning,
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    "  Waiting for the first step…",
                    THEME.text_muted,
                )));
            }
            if let Some(loss) = m.last_loss {
                let trend = trend_of(steps);
                lines.push(Line::from(vec![
                    Span::styled("  Loss:     ", THEME.kv_key),
                    Span::styled(format!("{loss:.4}"), THEME.kv_value),
                    Span::styled(trend, THEME.text_dim),
                ]));
            }
            if let Some(lr) = m.last_lr {
                lines.push(Line::from(vec![
                    Span::styled("  LR:       ", THEME.kv_key),
                    Span::styled(format!("{lr:.2e}"), THEME.kv_value),
                ]));
            }
            if let Some(tps) = m.tokens_per_second {
                lines.push(Line::from(vec![
                    Span::styled("  Tok/sec:  ", THEME.kv_key),
                    Span::styled(format!("{tps:.0}"), THEME.kv_value),
                ]));
            }
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(stats_area, buf);

            render_loss_sparkline(
                telemetry.loss.iter().map(|(_, y)| *y),
                own,
                " Loss Trend ",
                spark_area,
                buf,
            );
            render_timing(steps.last(), steps, timing_area, buf);
            Line::from(Span::styled("  Press x to stop", THEME.text_dim)).render(hint_area, buf);
        }
        RunState::Completed { took } => {
            let [stats_area, spark_area] =
                Layout::vertical([Constraint::Length(8), Constraint::Fill(1)]).areas(inner);
            let value = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.4}"));
            let lines = vec![
                Line::from(Span::styled("  ✓ Completed", THEME.status_success)),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  Steps:      ", THEME.kv_key),
                    Span::styled(
                        m.last_step.map_or("-".to_string(), |s| s.to_string()),
                        THEME.kv_value,
                    ),
                ]),
                Line::from(vec![
                    Span::styled("  Final loss: ", THEME.kv_key),
                    Span::styled(value(m.last_loss), THEME.text_success),
                ]),
                Line::from(vec![
                    Span::styled("  Eval loss:  ", THEME.kv_key),
                    Span::styled(value(m.eval_loss), THEME.text_dim),
                ]),
                Line::from(vec![
                    Span::styled("  Took:       ", THEME.kv_key),
                    Span::styled(format_duration(*took), THEME.text_dim),
                ]),
            ];
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(stats_area, buf);
            render_loss_sparkline(
                telemetry.loss.iter().map(|(_, y)| *y),
                own,
                " Loss Curve ",
                spark_area,
                buf,
            );
        }
    }
}

/// A sparkline of the loss, scaled to its own range, from the last 60 points.
fn render_loss_sparkline(
    loss: impl Iterator<Item = f64>,
    own: bool,
    title: &str,
    area: Rect,
    buf: &mut Buffer,
) {
    let block = Block::default()
        .title(title.to_string())
        .title_style(THEME.block_title)
        .borders(Borders::ALL)
        .border_style(THEME.block);
    let values: Vec<f64> = if own { loss.collect() } else { Vec::new() };
    if values.is_empty() {
        block.render(area, buf);
        return;
    }
    let start = values.len().saturating_sub(60);
    let window = &values[start..];
    let min = window.iter().copied().fold(f64::INFINITY, f64::min);
    let max = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(0.001);
    let scaled: Vec<u64> = window
        .iter()
        .map(|v| (((v - min) / range) * 100.0) as u64)
        .collect();
    Sparkline::default()
        .block(block)
        .data(&scaled)
        .max(100)
        .style(chart(0))
        .render(area, buf);
}

fn trend_of(steps: &[pmetal_core::StepMetrics]) -> &'static str {
    if steps.len() < 10 {
        return "";
    }
    let mean =
        |s: &[pmetal_core::StepMetrics]| s.iter().map(|s| s.loss).sum::<f64>() / s.len() as f64;
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

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_core::jobs::TrainSpec;
    use std::time::{Duration, Instant};

    fn screen(panel: &JobPanel, telemetry: &Telemetry<'_>, w: u16, h: u16) -> String {
        let area = Rect::new(0, 0, w, h);
        let mut buf = Buffer::empty(area);
        render_training_status(panel, telemetry, area, &mut buf);
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn panel() -> JobPanel {
        JobPanel::new::<TrainSpec>("Training")
    }

    #[test]
    fn an_idle_tab_says_how_to_start() {
        let text = screen(&panel(), &Telemetry::none(), 60, 20);
        assert!(text.contains("Idle"), "{text}");
        assert!(text.contains("S starts the job"), "{text}");
    }

    #[test]
    fn a_running_job_shows_its_progress_from_its_own_metrics() {
        let mut p = panel();
        p.job = Some("j1".into());
        p.run = RunState::Running {
            since: Instant::now(),
            phase: None,
        };
        p.metrics.last_step = Some(25);
        p.metrics.total_steps = Some(100);
        p.metrics.last_loss = Some(1.2345);
        p.metrics.epoch = Some(1.0);
        p.metrics.total_epochs = Some(3);
        p.metrics.tokens_per_second = Some(842.0);
        let text = screen(&p, &Telemetry::none(), 70, 30);
        assert!(text.contains("Training Monitor"), "{text}");
        assert!(text.contains("25/100 (25.0%)"), "{text}");
        assert!(text.contains("1.2345"));
        assert!(text.contains("Epoch 1/3"));
        assert!(text.contains("842"));
    }

    #[test]
    fn a_job_that_has_not_stepped_shows_what_it_is_doing() {
        let mut p = panel();
        p.run = RunState::Running {
            since: Instant::now(),
            phase: Some("Loading model…".into()),
        };
        let text = screen(&p, &Telemetry::none(), 70, 30);
        assert!(text.contains("Loading model…"), "{text}");
    }

    #[test]
    fn a_finished_job_shows_its_result_and_a_failed_one_its_reason() {
        let mut p = panel();
        p.run = RunState::Completed {
            took: Duration::from_secs(95),
        };
        p.metrics.last_step = Some(100);
        p.metrics.last_loss = Some(0.5);
        p.metrics.eval_loss = Some(0.75);
        let text = screen(&p, &Telemetry::none(), 70, 30);
        assert!(
            text.contains("Training Complete") && text.contains("0.5000"),
            "{text}"
        );
        assert!(text.contains("0.7500") && text.contains("1m 35s"), "{text}");

        p.run = RunState::Failed("out of memory\nwhile loading shard 3".into());
        p.metrics.last_step = Some(12);
        p.metrics.last_loss = Some(2.0);
        let text = screen(&p, &Telemetry::none(), 70, 30);
        assert!(text.contains("Training Failed"), "{text}");
        assert!(
            text.contains("out of memory") && text.contains("shard 3"),
            "{text}"
        );
        assert!(text.contains("Reached step 12"), "{text}");
    }

    #[test]
    fn charts_belong_to_the_job_the_monitor_follows() {
        let step = pmetal_core::StepMetrics {
            step: 1,
            loss: 2.0,
            total_ms: 100.0,
            gpu_fwd_bwd_ms: 80.0,
            ..Default::default()
        };
        let steps = [step];
        let loss = [(1.0, 2.0)];
        let telemetry = Telemetry {
            steps: &steps,
            loss: &loss,
            following: Some("mine"),
        };
        let mut p = panel();
        p.run = RunState::Running {
            since: Instant::now(),
            phase: None,
        };
        p.job = Some("mine".into());
        assert!(screen(&p, &telemetry, 70, 30).contains("GPU fwd+bwd"));
        // Another tab's job is being followed: this tab shows none of it.
        p.job = Some("someone else".into());
        assert!(!screen(&p, &telemetry, 70, 30).contains("GPU fwd+bwd"));
    }

    #[test]
    fn drawing_never_panics_at_any_size() {
        let mut p = panel();
        p.run = RunState::Running {
            since: Instant::now(),
            phase: None,
        };
        p.metrics.total_steps = Some(10);
        for run in [
            p.run.clone(),
            RunState::Idle,
            RunState::Stopped,
            RunState::Failed("x".into()),
            RunState::Completed {
                took: Duration::ZERO,
            },
        ] {
            p.run = run;
            for (w, h) in [(0, 0), (1, 1), (3, 3), (10, 4), (40, 9), (70, 30)] {
                screen(&p, &Telemetry::none(), w, h);
            }
        }
    }
}
