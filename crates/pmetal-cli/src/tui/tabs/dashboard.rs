//! Enhanced training dashboard tab.
//!
//! Shows loss curves, LR schedule, throughput sparklines, timing breakdown
//! with gauges, and memory utilization.

use std::io::BufRead;
use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, List, ListItem, Paragraph, Sparkline,
    Widget,
};

use crate::tui::theme::{THEME, palette};

/// A single metric sample from the training log.
#[derive(Debug, Clone)]
pub struct MetricSample {
    pub step: usize,
    pub epoch: usize,
    pub total_epochs: usize,
    pub total_steps: usize,
    pub loss: f64,
    pub lr: f64,
    pub tok_sec: f64,
    pub ane_fwd_ms: f64,
    pub ane_bwd_ms: f64,
    pub rmsnorm_ms: f64,
    pub cblas_ms: f64,
    pub adam_ms: f64,
    pub total_ms: f64,
}

/// Dashboard tab state.
pub struct DashboardTab {
    metrics_path: Option<PathBuf>,
    pub samples: Vec<MetricSample>,
    loss_data: Vec<(f64, f64)>,
    lr_data: Vec<(f64, f64)>,
    pub throughput_data: Vec<u64>,
    last_read_pos: u64,
    pub paused: bool,
}

impl DashboardTab {
    pub fn new(metrics_path: Option<PathBuf>) -> Self {
        Self {
            metrics_path,
            samples: Vec::new(),
            loss_data: Vec::new(),
            lr_data: Vec::new(),
            throughput_data: Vec::new(),
            last_read_pos: 0,
            paused: false,
        }
    }

    /// Push a metric sample (from in-process training or file polling).
    pub fn push_sample(&mut self, sample: MetricSample) {
        // Guard against NaN/Inf corrupting chart bounds
        if !sample.loss.is_finite() || !sample.lr.is_finite() {
            return;
        }
        let step = sample.step as f64;
        self.loss_data.push((step, sample.loss));
        self.lr_data.push((step, sample.lr));
        self.throughput_data.push(sample.tok_sec as u64);
        // Keep sparkline to last 60 points
        if self.throughput_data.len() > 60 {
            self.throughput_data.remove(0);
        }
        self.samples.push(sample);

        // Cap data to prevent unbounded growth (downsample if > 10K points)
        const MAX_CHART_POINTS: usize = 10_000;
        if self.loss_data.len() > MAX_CHART_POINTS * 2 {
            // Keep every other point
            self.loss_data = self.loss_data.iter().step_by(2).copied().collect();
            self.lr_data = self.lr_data.iter().step_by(2).copied().collect();
        }
    }

    /// Poll for new data from the metrics JSONL file.
    pub fn poll_metrics(&mut self) {
        if self.paused {
            return;
        }
        let Some(path) = &self.metrics_path else {
            return;
        };
        let Ok(file) = std::fs::File::open(path) else {
            return;
        };

        // Detect log rotation: if file is shorter than our last read position, reset
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        if file_len < self.last_read_pos {
            self.last_read_pos = 0;
        }

        let mut reader = std::io::BufReader::new(file);
        if std::io::Seek::seek(&mut reader, std::io::SeekFrom::Start(self.last_read_pos)).is_err() {
            return;
        }

        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                let sample = MetricSample {
                    step: json["step"].as_u64().unwrap_or(0) as usize,
                    epoch: json["epoch"].as_u64().unwrap_or(0) as usize,
                    total_epochs: json["total_epochs"].as_u64().unwrap_or(0) as usize,
                    total_steps: json["total_steps"].as_u64().unwrap_or(0) as usize,
                    loss: json["loss"].as_f64().unwrap_or(0.0),
                    lr: json["lr"].as_f64().unwrap_or(0.0),
                    tok_sec: json["tok_sec"].as_f64().unwrap_or(0.0),
                    ane_fwd_ms: json["ane_fwd_ms"].as_f64().unwrap_or(0.0),
                    ane_bwd_ms: json["ane_bwd_ms"].as_f64().unwrap_or(0.0),
                    rmsnorm_ms: json["rmsnorm_ms"].as_f64().unwrap_or(0.0),
                    cblas_ms: json["cblas_ms"].as_f64().unwrap_or(0.0),
                    adam_ms: json["adam_ms"].as_f64().unwrap_or(0.0),
                    total_ms: json["total_ms"].as_f64().unwrap_or(0.0),
                };
                self.push_sample(sample);
            }
            line.clear();
        }
        self.last_read_pos =
            std::io::Seek::stream_position(&mut reader).unwrap_or(self.last_read_pos);
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn reset(&mut self) {
        self.samples.clear();
        self.loss_data.clear();
        self.lr_data.clear();
        self.throughput_data.clear();
        self.last_read_pos = 0;
    }

    /// Set (or change) the metrics file path. Always resets state so a new
    /// training run to the same output dir starts with a clean dashboard.
    pub fn set_metrics_path(&mut self, path: Option<PathBuf>) {
        self.metrics_path = path;
        self.reset();
    }
}

impl Widget for &DashboardTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
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
        self.render_timing(timing_area, buf);
    }
}

impl DashboardTab {
    fn render_loss_chart(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(if self.paused {
                " Loss Curve [PAUSED] "
            } else {
                " Loss Curve "
            })
            .title_style(if self.paused {
                THEME.text_warning
            } else {
                THEME.block_title
            })
            .borders(Borders::ALL)
            .border_style(THEME.block);

        if self.loss_data.is_empty() {
            Paragraph::new(vec![
                ratatui::text::Line::from(""),
                ratatui::text::Line::from(Span::styled(
                    "  No active training run.",
                    THEME.text_muted,
                )),
                ratatui::text::Line::from(""),
                ratatui::text::Line::from(Span::styled(
                    "  Start a run from the Training, Distill, or GRPO tab.",
                    THEME.text_dim,
                )),
                ratatui::text::Line::from(Span::styled(
                    "  Loss curve and metrics will appear here automatically.",
                    THEME.text_dim,
                )),
            ])
            .block(block)
            .render(area, buf);
            return;
        }

        let min_loss = self
            .loss_data
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::MAX, f64::min);
        let max_loss = self
            .loss_data
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::MIN, f64::max);
        let max_step = self.loss_data.last().map(|(x, _)| *x).unwrap_or(1.0);

        let datasets = vec![
            Dataset::default()
                .name("loss")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(palette::CHART_1)
                .data(&self.loss_data),
        ];

        // Avoid zero-width Y axis when all loss values are identical
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
                    .labels::<Vec<Line>>(vec![
                        format!("{:.3}", y_min).into(),
                        format!("{:.3}", (y_min + y_max) / 2.0).into(),
                        format!("{:.3}", y_max).into(),
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

        let items: Vec<ListItem> = if let Some(last) = self.samples.last() {
            let loss_trend = if self.samples.len() >= 10 {
                let recent_avg: f64 = self.samples[self.samples.len() - 5..]
                    .iter()
                    .map(|s| s.loss)
                    .sum::<f64>()
                    / 5.0;
                let prev_avg: f64 = self.samples[self.samples.len() - 10..self.samples.len() - 5]
                    .iter()
                    .map(|s| s.loss)
                    .sum::<f64>()
                    / 5.0;
                if recent_avg < prev_avg * 0.99 {
                    " (decreasing)"
                } else if recent_avg > prev_avg * 1.01 {
                    " (increasing)"
                } else {
                    " (plateau)"
                }
            } else {
                ""
            };

            let epoch_display = if last.total_epochs > 0 {
                format!("{}/{}", last.epoch + 1, last.total_epochs)
            } else {
                format!("{}", last.epoch + 1)
            };
            let step_display = if last.total_steps > 0 {
                format!("{}/{}", last.step, last.total_steps)
            } else {
                format!("{}", last.step)
            };
            vec![
                ListItem::new(Line::from(vec![
                    Span::styled("Epoch:   ", THEME.kv_key),
                    Span::styled(epoch_display, THEME.kv_value),
                ])),
                ListItem::new(Line::from(vec![
                    Span::styled("Step:    ", THEME.kv_key),
                    Span::styled(step_display, THEME.kv_value),
                ])),
                ListItem::new(Line::from(vec![
                    Span::styled("Loss:    ", THEME.kv_key),
                    Span::styled(format!("{:.4}", last.loss), THEME.kv_value),
                    Span::styled(loss_trend, THEME.text_dim),
                ])),
                ListItem::new(Line::from(vec![
                    Span::styled("LR:      ", THEME.kv_key),
                    Span::styled(format!("{:.2e}", last.lr), THEME.kv_value),
                ])),
                ListItem::new(Line::from(vec![
                    Span::styled("Tok/sec: ", THEME.kv_key),
                    Span::styled(format!("{:.0}", last.tok_sec), THEME.kv_value),
                ])),
                ListItem::new(""),
                {
                    let min_loss = self
                        .samples
                        .iter()
                        .map(|s| s.loss)
                        .filter(|l| *l > 0.0)
                        .fold(f64::MAX, f64::min);
                    if min_loss < f64::MAX {
                        ListItem::new(Line::from(vec![
                            Span::styled("Min loss:", THEME.kv_key),
                            Span::styled(format!(" {:.4}", min_loss), THEME.text_success),
                        ]))
                    } else {
                        ListItem::new(Line::from(vec![
                            Span::styled("Min loss:", THEME.kv_key),
                            Span::styled(" —", THEME.text_muted),
                        ]))
                    }
                },
                {
                    let valid: Vec<f64> = self
                        .samples
                        .iter()
                        .map(|s| s.loss)
                        .filter(|l| *l > 0.0)
                        .collect();
                    let avg = if valid.is_empty() {
                        0.0
                    } else {
                        valid.iter().sum::<f64>() / valid.len() as f64
                    };
                    ListItem::new(Line::from(vec![
                        Span::styled("Avg loss:", THEME.kv_key),
                        Span::styled(
                            if valid.is_empty() {
                                " —".to_string()
                            } else {
                                format!(" {:.4}", avg)
                            },
                            THEME.text_dim,
                        ),
                    ]))
                },
            ]
        } else {
            vec![ListItem::new(Span::styled("No data yet", THEME.text_muted))]
        };

        List::new(items).render(inner, buf);
    }

    fn render_throughput(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Throughput (tok/s) ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        if self.throughput_data.is_empty() {
            Paragraph::new("\n  No data")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }

        Sparkline::default()
            .block(block)
            .data(&self.throughput_data)
            .style(palette::CHART_4)
            .render(area, buf);
    }

    fn render_timing(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Timing Breakdown ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let Some(last) = self.samples.last() else {
            Paragraph::new("No timing data")
                .style(THEME.text_muted)
                .render(inner, buf);
            return;
        };

        let total = last.total_ms.max(1.0);

        // Detect ANE vs standard training based on whether ANE fields are populated
        let ane_sum =
            last.ane_fwd_ms + last.ane_bwd_ms + last.rmsnorm_ms + last.cblas_ms + last.adam_ms;
        let is_ane = ane_sum > 0.0;

        if is_ane {
            // ANE timing breakdown
            let components = [
                ("ANE fwd", last.ane_fwd_ms, palette::CHART_1),
                ("ANE bwd", last.ane_bwd_ms, palette::CHART_2),
                ("RMSNorm", last.rmsnorm_ms, palette::CHART_3),
                ("cblas dW", last.cblas_ms, palette::CHART_4),
                ("Adam", last.adam_ms, palette::CHART_5),
            ];

            let constraints: Vec<Constraint> = components
                .iter()
                .map(|_| Constraint::Length(2))
                .chain(std::iter::once(Constraint::Fill(1)))
                .collect();
            let rows = Layout::vertical(constraints).split(inner);

            for (i, (name, ms, color)) in components.iter().enumerate() {
                if i >= rows.len() - 1 {
                    break;
                }
                let ratio = (ms / total).clamp(0.0, 1.0);
                let label = format!("{:>8}: {:6.1}ms ({:.0}%)", name, ms, ratio * 100.0);
                Gauge::default()
                    .gauge_style(*color)
                    .ratio(ratio)
                    .label(Span::styled(label, THEME.text))
                    .render(rows[i], buf);
            }

            let total_row = rows[rows.len() - 1];
            Line::from(vec![
                Span::styled("   Total: ", THEME.kv_key),
                Span::styled(format!("{:.1}ms", total), THEME.kv_value),
                Span::styled(
                    format!("  ({:.0} steps/min)", 60000.0 / total),
                    THEME.text_dim,
                ),
            ])
            .render(total_row, buf);
        } else {
            // Standard MLX training — show per-step timing and throughput
            let rows = Layout::vertical([
                Constraint::Length(2),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Fill(1),
            ])
            .split(inner);

            // Step time gauge
            // Scale: assume 1000ms as "slow", show relative gauge
            let ratio = (total / 1000.0).clamp(0.0, 1.0);
            let label = format!("Step time: {:.1}ms", total);
            Gauge::default()
                .gauge_style(palette::CHART_2)
                .ratio(ratio)
                .label(Span::styled(label, THEME.text))
                .render(rows[0], buf);

            Line::from(vec![
                Span::styled("  Steps/min: ", THEME.kv_key),
                Span::styled(format!("{:.0}", 60000.0 / total), THEME.kv_value),
            ])
            .render(rows[2], buf);

            Line::from(vec![
                Span::styled("  Tok/sec:   ", THEME.kv_key),
                Span::styled(format!("{:.0}", last.tok_sec), THEME.kv_value),
            ])
            .render(rows[3], buf);

            // Show avg step time over recent samples
            if self.samples.len() >= 5 {
                let recent: Vec<&MetricSample> = self.samples
                    [self.samples.len().saturating_sub(10)..]
                    .iter()
                    .collect();
                let avg_ms = recent.iter().map(|s| s.total_ms).sum::<f64>() / recent.len() as f64;
                Line::from(vec![
                    Span::styled("  Avg step:  ", THEME.kv_key),
                    Span::styled(format!("{:.1}ms", avg_ms), THEME.text_dim),
                ])
                .render(rows[4], buf);
            }
        }
    }
}
