//! Main application state, event loop, and rendering.

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use crossterm::ExecutableCommand;
use crossterm::event::{KeyCode, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::widgets::Widget;

use crate::tui::event::{Event, EventHandler};
use crate::tui::tabs::*;
use crate::tui::theme::THEME;
use crate::tui::widgets::{Footer, Header};

/// Top-level application state.
pub struct App {
    /// Whether the app should quit.
    should_quit: bool,

    /// Active tab.
    active_tab: Tab,

    /// Per-tab state.
    pub dashboard: DashboardTab,
    pub device: DeviceTab,
    pub models: ModelsTab,
    pub datasets: DatasetsTab,
    pub training: TrainingTab,
    pub inference: InferenceTab,
    pub jobs: JobsTab,

    /// Tick counter for periodic updates.
    tick_count: u64,
}

impl App {
    /// Create a new app with optional metrics file for the dashboard tab.
    pub fn new(metrics_file: Option<PathBuf>) -> Self {
        Self {
            should_quit: false,
            active_tab: Tab::Dashboard,
            dashboard: DashboardTab::new(metrics_file),
            device: DeviceTab::new(),
            models: ModelsTab::new(),
            datasets: DatasetsTab::new(),
            training: TrainingTab::new(),
            inference: InferenceTab::new(),
            jobs: JobsTab::new(),
            tick_count: 0,
        }
    }

    /// Run the TUI event loop.
    pub fn run(&mut self) -> io::Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        io::stdout().execute(EnterAlternateScreen)?;
        crossterm::execute!(io::stdout(), crossterm::event::EnableMouseCapture)?;

        // Install panic hook BEFORE creating the terminal so it is always restored.
        let original_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let _ = crossterm::execute!(
                io::stdout(),
                crossterm::event::DisableMouseCapture,
                LeaveAlternateScreen
            );
            original_hook(info);
        }));

        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;

        // Event handler with 200ms tick rate
        let events = EventHandler::new(Duration::from_millis(200));

        // Main loop
        while !self.should_quit {
            // Draw
            terminal.draw(|frame| {
                self.draw(frame);
            })?;

            // Handle events
            match events.next() {
                Ok(Event::Key(key)) => self.handle_key(key),
                Ok(Event::Tick) => self.on_tick(),
                Ok(Event::Resize(_, _)) => {} // Terminal handles resize
                Ok(Event::Mouse(_)) => {}     // Future: mouse support
                Err(_) => self.should_quit = true,
            }
        }

        // Restore terminal (also restore the original panic hook)
        let _ = std::panic::take_hook();
        crossterm::execute!(io::stdout(), crossterm::event::DisableMouseCapture)?;
        disable_raw_mode()?;
        io::stdout().execute(LeaveAlternateScreen)?;
        Ok(())
    }

    /// Handle a key press event.
    fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        // Global keybindings (always active)
        match key.code {
            KeyCode::Char('q') if !matches!(self.active_tab, Tab::Inference) => {
                self.should_quit = true;
                return;
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
                return;
            }
            KeyCode::Tab => {
                self.active_tab = self.active_tab.next();
                return;
            }
            KeyCode::BackTab => {
                self.active_tab = self.active_tab.prev();
                return;
            }
            KeyCode::Right if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.active_tab = self.active_tab.next();
                return;
            }
            KeyCode::Left if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.active_tab = self.active_tab.prev();
                return;
            }
            // Number keys for direct tab access
            KeyCode::Char('1') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.active_tab = Tab::Dashboard;
                return;
            }
            KeyCode::Char('2') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.active_tab = Tab::Device;
                return;
            }
            KeyCode::Char('3') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.active_tab = Tab::Models;
                return;
            }
            KeyCode::Char('4') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.active_tab = Tab::Datasets;
                return;
            }
            KeyCode::Char('5') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.active_tab = Tab::Training;
                return;
            }
            KeyCode::Char('6') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.active_tab = Tab::Inference;
                return;
            }
            KeyCode::Char('7') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.active_tab = Tab::Jobs;
                return;
            }
            _ => {}
        }

        // Per-tab keybindings
        match self.active_tab {
            Tab::Dashboard => match key.code {
                KeyCode::Char('p') => self.dashboard.toggle_pause(),
                KeyCode::Char('r') => self.dashboard.reset(),
                _ => {}
            },
            Tab::Device => {
                // Device tab is read-only, no specific keys needed
            }
            Tab::Models => match key.code {
                KeyCode::Down | KeyCode::Char('j') => self.models.next_row(),
                KeyCode::Up | KeyCode::Char('k') => self.models.prev_row(),
                KeyCode::Char('/') => {
                    self.models.searching = !self.models.searching;
                    if !self.models.searching {
                        self.models.search_query.clear();
                    }
                }
                KeyCode::Char(c) if self.models.searching => {
                    self.models.search_query.push(c);
                }
                KeyCode::Backspace if self.models.searching => {
                    self.models.search_query.pop();
                }
                KeyCode::Esc if self.models.searching => {
                    self.models.searching = false;
                    self.models.search_query.clear();
                }
                KeyCode::Char('R') => self.models.scan_models(),
                _ => {}
            },
            Tab::Datasets => match key.code {
                KeyCode::Down | KeyCode::Char('j') => self.datasets.next_row(),
                KeyCode::Up | KeyCode::Char('k') => self.datasets.prev_row(),
                KeyCode::Char('R') => self.datasets.scan_datasets(),
                _ => {}
            },
            Tab::Training => match key.code {
                KeyCode::Down | KeyCode::Char('j') => self.training.next_param(),
                KeyCode::Up | KeyCode::Char('k') => self.training.prev_param(),
                _ => {}
            },
            Tab::Inference => match key.code {
                KeyCode::Enter => self.inference.submit_message(),
                KeyCode::Backspace => self.inference.delete_char(),
                KeyCode::Left => self.inference.move_cursor_left(),
                KeyCode::Right => self.inference.move_cursor_right(),
                KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.should_quit = true;
                }
                KeyCode::Esc => {
                    // In inference, Esc could stop generation or switch focus
                    if self.inference.generating {
                        self.inference.generating = false;
                    }
                }
                KeyCode::Char(c) => self.inference.insert_char(c),
                _ => {}
            },
            Tab::Jobs => match key.code {
                KeyCode::Down | KeyCode::Char('j') => self.jobs.next_row(),
                KeyCode::Up | KeyCode::Char('k') => self.jobs.prev_row(),
                KeyCode::Char('R') => self.jobs.scan_jobs(),
                _ => {}
            },
        }
    }

    /// Called on each tick (200ms).
    fn on_tick(&mut self) {
        self.tick_count += 1;

        match self.active_tab {
            Tab::Dashboard => {
                self.dashboard.poll_metrics();
            }
            Tab::Device => {
                // Refresh memory every 5 ticks (1 second)
                if self.tick_count % 5 == 0 {
                    self.device.refresh_memory();
                }
            }
            _ => {}
        }
    }
}

impl App {
    /// Draw the full UI to a frame.
    fn draw(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();
        let buf = frame.buffer_mut();

        // Fill background
        buf.set_style(area, THEME.root);

        // Layout: header (2) | content (fill) | footer (1)
        let [header_area, content_area, footer_area] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(area);

        // Header with tab bar
        Header {
            active_tab: self.active_tab,
            tabs: Tab::ALL,
        }
        .render(header_area, buf);

        // Active tab content
        match self.active_tab {
            Tab::Dashboard => (&self.dashboard).render(content_area, buf),
            Tab::Device => (&self.device).render(content_area, buf),
            Tab::Models => (&mut self.models).render(content_area, buf),
            Tab::Datasets => (&mut self.datasets).render(content_area, buf),
            Tab::Training => (&mut self.training).render(content_area, buf),
            Tab::Inference => (&mut self.inference).render(content_area, buf),
            Tab::Jobs => (&mut self.jobs).render(content_area, buf),
        }

        // Footer with keybindings
        Footer {
            tab: self.active_tab,
        }
        .render(footer_area, buf);
    }
}
