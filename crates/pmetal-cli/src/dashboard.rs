//! Real-time training dashboard using ratatui TUI.
//!
//! Displays loss curves, learning rate schedule, ANE utilization,
//! and per-component timing breakdown in a terminal-based UI.
//!
//! # Usage
//!
//! ```bash
//! pmetal dashboard --metrics-file training_metrics.jsonl
//! ```

#[cfg(feature = "dashboard")]
pub mod app {
    use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use crossterm::ExecutableCommand;
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    };
    use ratatui::Terminal;
    use ratatui::backend::CrosstermBackend;
    use ratatui::layout::{Constraint, Direction, Layout, Rect};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::symbols;
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{
        Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, Paragraph,
    };

    /// A single metric sample from the training log.
    #[derive(Debug, Clone)]
    pub struct MetricSample {
        pub step: usize,
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

    /// Dashboard application state.
    pub struct DashboardApp {
        metrics_path: Option<PathBuf>,
        samples: Vec<MetricSample>,
        loss_data: Vec<(f64, f64)>,
        lr_data: Vec<(f64, f64)>,
        should_quit: bool,
        last_read_pos: u64,
    }

    impl DashboardApp {
        /// Create a new dashboard app.
        pub fn new(metrics_path: Option<PathBuf>) -> Self {
            Self {
                metrics_path,
                samples: Vec::new(),
                loss_data: Vec::new(),
                lr_data: Vec::new(),
                should_quit: false,
                last_read_pos: 0,
            }
        }

        /// Add a metric sample from in-process training.
        pub fn push_sample(&mut self, sample: MetricSample) {
            let step = sample.step as f64;
            self.loss_data.push((step, sample.loss));
            self.lr_data.push((step, sample.lr));
            self.samples.push(sample);
        }

        /// Poll for new data from the metrics JSONL file.
        fn poll_metrics(&mut self) {
            let Some(path) = &self.metrics_path else {
                return;
            };
            let Ok(file) = std::fs::File::open(path) else {
                return;
            };

            let mut reader = BufReader::new(file);
            if reader.seek(SeekFrom::Start(self.last_read_pos)).is_err() {
                return;
            }

            let mut line = String::new();
            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                    let sample = MetricSample {
                        step: json["step"].as_u64().unwrap_or(0) as usize,
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
            self.last_read_pos = reader.stream_position().unwrap_or(self.last_read_pos);
        }

        /// Run the dashboard TUI event loop.
        pub fn run(&mut self) -> io::Result<()> {
            enable_raw_mode()?;
            io::stdout().execute(EnterAlternateScreen)?;
            let backend = CrosstermBackend::new(io::stdout());
            let mut terminal = Terminal::new(backend)?;

            let tick_rate = Duration::from_millis(250);
            let mut last_tick = Instant::now();

            while !self.should_quit {
                terminal.draw(|f| self.draw(f))?;

                let timeout = tick_rate.saturating_sub(last_tick.elapsed());
                if event::poll(timeout)? {
                    if let Event::Key(key) = event::read()? {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
                            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                self.should_quit = true;
                            }
                            _ => {}
                        }
                    }
                }

                if last_tick.elapsed() >= tick_rate {
                    self.poll_metrics();
                    last_tick = Instant::now();
                }
            }

            disable_raw_mode()?;
            io::stdout().execute(LeaveAlternateScreen)?;
            Ok(())
        }

        fn draw(&self, f: &mut ratatui::Frame) {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Title
                    Constraint::Min(10),   // Main content
                    Constraint::Length(3), // Status bar
                ])
                .split(f.area());

            // Title
            let title = Paragraph::new(Line::from(vec![
                Span::styled(
                    " PMetal ANE Training Dashboard ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" | "),
                Span::styled("q", Style::default().fg(Color::Yellow)),
                Span::raw(" to quit"),
            ]))
            .block(Block::default().borders(Borders::ALL));
            f.render_widget(title, chunks[0]);

            // Main content: loss chart | stats panel
            let main_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
                .split(chunks[1]);

            self.draw_loss_chart(f, main_chunks[0]);
            self.draw_stats_panel(f, main_chunks[1]);

            // Status bar
            let status = if let Some(last) = self.samples.last() {
                format!(
                    " Step {} | Loss: {:.4} | {:.0} tok/s | Compiles: 9 (dynamic) ",
                    last.step, last.loss, last.tok_sec
                )
            } else {
                " Waiting for training data...".to_string()
            };
            let status_widget = Paragraph::new(status)
                .style(Style::default().fg(Color::White).bg(Color::DarkGray))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(status_widget, chunks[2]);
        }

        fn draw_loss_chart(&self, f: &mut ratatui::Frame, area: Rect) {
            if self.loss_data.is_empty() {
                let placeholder = Paragraph::new("No data yet")
                    .block(Block::default().title(" Loss ").borders(Borders::ALL));
                f.render_widget(placeholder, area);
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
                    .style(Style::default().fg(Color::Green))
                    .data(&self.loss_data),
            ];

            let chart = Chart::new(datasets)
                .block(Block::default().title(" Loss Curve ").borders(Borders::ALL))
                .x_axis(
                    Axis::default()
                        .title("Step")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([0.0, max_step]),
                )
                .y_axis(
                    Axis::default()
                        .title("Loss")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([min_loss * 0.95, max_loss * 1.05])
                        .labels::<Vec<Line>>(vec![
                            format!("{:.2}", min_loss).into(),
                            format!("{:.2}", (min_loss + max_loss) / 2.0).into(),
                            format!("{:.2}", max_loss).into(),
                        ]),
                );
            f.render_widget(chart, area);
        }

        fn draw_stats_panel(&self, f: &mut ratatui::Frame, area: Rect) {
            let stats_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area);

            // Current stats
            let stats: Vec<ListItem> = if let Some(last) = self.samples.last() {
                vec![
                    ListItem::new(format!("Step:      {}", last.step)),
                    ListItem::new(format!("Loss:      {:.4}", last.loss)),
                    ListItem::new(format!("LR:        {:.2e}", last.lr)),
                    ListItem::new(format!("Tok/sec:   {:.0}", last.tok_sec)),
                    ListItem::new(format!("Samples:   {}", self.samples.len())),
                    ListItem::new(""),
                    ListItem::new("Pipeline:  Dynamic (9 kernels)".to_string()),
                    ListItem::new("Compiles:  9 (one-time)".to_string()),
                ]
            } else {
                vec![ListItem::new("Waiting...")]
            };

            let stats_list =
                List::new(stats).block(Block::default().title(" Stats ").borders(Borders::ALL));
            f.render_widget(stats_list, stats_chunks[0]);

            // Timing breakdown
            let timing: Vec<ListItem> = if let Some(last) = self.samples.last() {
                let total = last.total_ms.max(0.001);
                vec![
                    ListItem::new(format!(
                        "ANE fwd:   {:6.1}ms ({:.0}%)",
                        last.ane_fwd_ms,
                        last.ane_fwd_ms / total * 100.0
                    )),
                    ListItem::new(format!(
                        "ANE bwd:   {:6.1}ms ({:.0}%)",
                        last.ane_bwd_ms,
                        last.ane_bwd_ms / total * 100.0
                    )),
                    ListItem::new(format!(
                        "RMSNorm:   {:6.1}ms ({:.0}%)",
                        last.rmsnorm_ms,
                        last.rmsnorm_ms / total * 100.0
                    )),
                    ListItem::new(format!(
                        "cblas dW:  {:6.1}ms ({:.0}%)",
                        last.cblas_ms,
                        last.cblas_ms / total * 100.0
                    )),
                    ListItem::new(format!(
                        "Adam:      {:6.1}ms ({:.0}%)",
                        last.adam_ms,
                        last.adam_ms / total * 100.0
                    )),
                    ListItem::new(""),
                    ListItem::new(format!("Total:     {:6.1}ms", total)),
                ]
            } else {
                vec![ListItem::new("No timing data")]
            };

            let timing_list =
                List::new(timing).block(Block::default().title(" Timing ").borders(Borders::ALL));
            f.render_widget(timing_list, stats_chunks[1]);
        }
    }
}

/// Run the dashboard from the CLI.
#[cfg(feature = "dashboard")]
pub fn run_dashboard(metrics_file: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    let mut app = app::DashboardApp::new(metrics_file);
    app.run()?;
    Ok(())
}

/// Stub when dashboard feature is not enabled.
#[cfg(not(feature = "dashboard"))]
pub fn run_dashboard(_metrics_file: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    anyhow::bail!("Dashboard requires the 'dashboard' feature: cargo build --features dashboard")
}
