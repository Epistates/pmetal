//! Main application state, async event loop, and rendering.

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use crossterm::ExecutableCommand;
use crossterm::event::{KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::widgets::Widget;

use crate::tui::command_runner::CommandRunner;
use crate::tui::event::{AppMsg, CommandSpec, Event, EventHandler, JobType};
use crate::tui::modal::{Modal, ModalAction, PickerEntry};
use crate::tui::tabs::*;
use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::ModelSource;
use crate::tui::theme::THEME;
use crate::tui::widgets::{Footer, Header};

/// Top-level application state.
pub struct App {
    /// Whether the app should quit.
    should_quit: bool,

    /// Active tab.
    active_tab: Tab,

    /// Whether mouse capture is enabled (can be toggled with F2).
    mouse_captured: bool,

    /// Per-tab state.
    pub dashboard: DashboardTab,
    pub device: DeviceTab,
    pub models: ModelsTab,
    pub datasets: DatasetsTab,
    pub training: TrainingTab,
    pub inference: InferenceTab,
    pub jobs: JobsTab,
    pub distillation: DistillationTab,
    pub grpo: GrpoTab,

    /// Tick counter for periodic updates.
    tick_count: u64,

    /// Modal dialog stack (topmost is rendered last / receives keys first).
    modal_stack: Vec<Modal>,

    /// Pending modal action context: which tab/action triggered the modal.
    pending_modal_context: Option<PendingModalTarget>,

    /// Background command runner.
    runner: Option<CommandRunner>,

    /// Currently active training job ID (if any).
    active_training_job: Option<String>,

    /// Output directory of the currently active training job (for auto-registering on completion).
    active_training_output_dir: Option<PathBuf>,

    /// Currently active inference job ID (if any).
    active_inference_job: Option<String>,

    /// Header area for mouse hit-testing.
    header_area: ratatui::layout::Rect,
}

/// Tracks what action a modal was opened for, so we can route the result.
#[derive(Debug, Clone)]
enum PendingModalTarget {
    TrainingModel,
    TrainingDataset,
    TrainingStart,
    TestInference,
    InferenceModel,
    /// Selecting base model for a LoRA adapter (for inference).
    InferenceBaseModel(String),
    /// Selecting base model for fuse/export.
    FuseBaseModel(String),
    /// Confirm fuse/export operation.
    FuseStart,
    DistillTeacher,
    DistillStudent,
    DistillDataset,
    DistillStart,
    GrpoModel,
    GrpoDataset,
    GrpoStart,
    DownloadModel,
    AddModelDir,
    AddDatasetDir,
    ConvertDataset,
}

impl App {
    /// Create a new app with optional metrics file for the dashboard tab.
    pub fn new(metrics_file: Option<PathBuf>) -> Self {
        Self {
            should_quit: false,
            active_tab: Tab::Dashboard,
            mouse_captured: true,
            dashboard: DashboardTab::new(metrics_file),
            device: DeviceTab::new(),
            models: ModelsTab::new(),
            datasets: DatasetsTab::new(),
            training: TrainingTab::new(),
            inference: InferenceTab::new(),
            jobs: JobsTab::new(),
            distillation: DistillationTab::new(),
            grpo: GrpoTab::new(),
            tick_count: 0,
            modal_stack: Vec::new(),
            pending_modal_context: None,
            runner: None,
            active_training_job: None,
            active_training_output_dir: None,
            active_inference_job: None,
            header_area: ratatui::layout::Rect::default(),
        }
    }

    /// Run the TUI event loop (async).
    pub async fn run(&mut self) -> anyhow::Result<()> {
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

        // Create async event handler
        let mut events = EventHandler::new(Duration::from_millis(200));

        // Create command runner with the app message sender
        self.runner = Some(CommandRunner::new(events.app_tx()));

        // Main loop
        while !self.should_quit {
            // Draw
            terminal.draw(|frame| {
                self.draw(frame);
            })?;

            // Wait for next event
            match events.next().await {
                Some(Event::Key(key)) => self.handle_key(key),
                Some(Event::Mouse(mouse)) => self.handle_mouse(mouse),
                Some(Event::Tick) => self.on_tick(),
                Some(Event::Resize(_, _)) => {} // Terminal handles resize
                Some(Event::App(msg)) => self.handle_app_msg(msg),
                None => self.should_quit = true,
            }
        }

        // Restore terminal
        let _ = std::panic::take_hook();
        crossterm::execute!(io::stdout(), crossterm::event::DisableMouseCapture)?;
        disable_raw_mode()?;
        io::stdout().execute(LeaveAlternateScreen)?;
        Ok(())
    }

    /// Handle a key press event.
    fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        // If a modal is open, route keys there first
        if let Some(modal) = self.modal_stack.last_mut() {
            if let Some(action) = modal.handle_key(key) {
                self.modal_stack.pop();
                self.handle_modal_action(action);
            }
            return;
        }

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
            // F2: toggle mouse capture (allows text selection when off)
            KeyCode::F(2) => {
                self.mouse_captured = !self.mouse_captured;
                if self.mouse_captured {
                    let _ = crossterm::execute!(
                        io::stdout(),
                        crossterm::event::EnableMouseCapture
                    );
                } else {
                    let _ = crossterm::execute!(
                        io::stdout(),
                        crossterm::event::DisableMouseCapture
                    );
                }
                return;
            }
            // Number keys for direct tab access
            KeyCode::Char(c @ '1'..='9') if key.modifiers.contains(KeyModifiers::ALT) => {
                let idx = (c as u8 - b'1') as usize;
                if let Some(&tab) = Tab::ALL.get(idx) {
                    self.active_tab = tab;
                }
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
                // Device tab is read-only
            }
            Tab::Models => match key.code {
                KeyCode::Down | KeyCode::Char('j') if !self.models.searching => {
                    self.models.next_row()
                }
                KeyCode::Up | KeyCode::Char('k') if !self.models.searching => {
                    self.models.prev_row()
                }
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
                KeyCode::Char('d') if !self.models.searching => {
                    // Download a new model
                    self.pending_modal_context = Some(PendingModalTarget::DownloadModel);
                    self.modal_stack
                        .push(Modal::text_input("Download Model", "Model ID (e.g. unsloth/Qwen3-0.6B):"));
                }
                KeyCode::Char('a') if !self.models.searching => {
                    // Add custom model directory
                    self.pending_modal_context = Some(PendingModalTarget::AddModelDir);
                    self.modal_stack
                        .push(Modal::text_input("Add Model Directory", "Path to directory containing models:"));
                }
                KeyCode::Char('t') if !self.models.searching => {
                    // Train with selected model → switch to Training tab with model loaded
                    if let Some(model) = self.models.selected_model().cloned() {
                        self.training.set_model(&model.id);
                        self.training.focus_field("Dataset");
                        self.active_tab = Tab::Training;
                        // Auto-open dataset picker
                        self.pending_modal_context = Some(PendingModalTarget::TrainingDataset);
                        let entries = self.dataset_picker_entries();
                        self.modal_stack.push(Modal::dataset_picker(entries));
                    }
                }
                KeyCode::Char('s') if !self.models.searching => {
                    // Distill with selected model as student → switch to Distillation tab
                    if let Some(model) = self.models.selected_model().cloned() {
                        self.distillation.set_student(&model.id);
                        self.distillation.focus_field("Dataset");
                        self.active_tab = Tab::Distillation;
                        // Auto-open dataset picker
                        self.pending_modal_context = Some(PendingModalTarget::DistillDataset);
                        let entries = self.dataset_picker_entries();
                        self.modal_stack.push(Modal::dataset_picker(entries));
                    }
                }
                KeyCode::Char('i') if !self.models.searching => {
                    // Inference with selected model → switch to Inference tab
                    if let Some(model) = self.models.selected_model().cloned() {
                        self.inference.load_model_settings(&model.path);
                        // For LoRA adapters without base model, prompt for base
                        if is_lora_adapter(&model.path) && model.base_model.is_none() {
                            self.pending_modal_context =
                                Some(PendingModalTarget::InferenceBaseModel(model.id.clone()));
                            let entries: Vec<_> = self
                                .model_picker_entries()
                                .into_iter()
                                .filter(|e| !e.id.starts_with("trained/"))
                                .collect();
                            self.modal_stack.push(Modal::model_picker(entries));
                        } else {
                            self.inference.model_id = Some(model.id.clone());
                        }
                        self.active_tab = Tab::Inference;
                    }
                }
                KeyCode::Char('f') if !self.models.searching => {
                    // Fuse/export selected model (only for LoRA adapters)
                    if let Some(model) = self.models.selected_model().cloned() {
                        if is_lora_adapter(&model.path) {
                            if let Some(ref base) = model.base_model {
                                // We know the base model — go straight to confirm
                                let base_display = base.clone();
                                let output_default = format!(
                                    "./output/{}-fused",
                                    model
                                        .path
                                        .file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy()
                                );
                                self.pending_modal_context =
                                    Some(PendingModalTarget::FuseStart);
                                self.modal_stack.push(Modal::confirm(
                                    "Fuse LoRA Adapter",
                                    vec![
                                        format!("Adapter:  {}", model.id),
                                        format!("Base:     {}", base_display),
                                        format!("Output:   {}", output_default),
                                        String::new(),
                                        "This will merge LoRA weights into the base model".into(),
                                        "and save a complete fused model.".into(),
                                        String::new(),
                                        "Proceed?".into(),
                                    ],
                                ));
                            } else {
                                // Need to pick a base model first
                                self.pending_modal_context =
                                    Some(PendingModalTarget::FuseBaseModel(model.id.clone()));
                                let entries: Vec<_> = self
                                    .model_picker_entries()
                                    .into_iter()
                                    .filter(|e| !e.id.starts_with("trained/"))
                                    .collect();
                                self.modal_stack.push(Modal::model_picker(entries));
                            }
                        } else {
                            self.modal_stack.push(Modal::error(
                                "Not a LoRA Adapter",
                                "Fuse is only available for LoRA adapters.",
                            ));
                        }
                    }
                }
                _ => {}
            },
            Tab::Datasets => match key.code {
                KeyCode::Down | KeyCode::Char('j') => self.datasets.next_row(),
                KeyCode::Up | KeyCode::Char('k') => self.datasets.prev_row(),
                KeyCode::Char('R') => self.datasets.scan_datasets(),
                KeyCode::Char('a') => {
                    // Add custom dataset directory
                    self.pending_modal_context = Some(PendingModalTarget::AddDatasetDir);
                    self.modal_stack
                        .push(Modal::text_input("Add Dataset Directory", "Path to directory containing datasets:"));
                }
                KeyCode::Char('c') => {
                    // Convert selected dataset to pmetal-compatible JSONL
                    if let Some(ds) = self.datasets.selected_dataset() {
                        let input_path = ds.path.display().to_string();
                        let output_name = ds.path
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        let output_path = format!("./data/{output_name}_converted.jsonl");
                        self.pending_modal_context = Some(PendingModalTarget::ConvertDataset);
                        self.modal_stack.push(Modal::text_input(
                            "Convert Dataset",
                            format!(
                                "Input: {input_path}\nOutput path (JSONL):"
                            ),
                        ));
                        // Pre-fill the output path
                        if let Some(Modal::TextInput { value, cursor, .. }) = self.modal_stack.last_mut() {
                            *value = output_path.clone();
                            *cursor = output_path.len();
                        }
                    } else {
                        self.modal_stack.push(Modal::error("No Dataset", "Select a dataset first."));
                    }
                }
                _ => {}
            },
            Tab::Training => self.handle_training_key(key),
            Tab::Inference => self.handle_inference_key(key),
            Tab::Jobs => self.handle_jobs_key(key),
            Tab::Distillation => self.handle_distillation_key(key),
            Tab::Grpo => self.handle_grpo_key(key),
        }
    }

    // --- Training tab key handler ---
    fn handle_training_key(&mut self, key: crossterm::event::KeyEvent) {
        // If a field is being edited inline, route keys there
        if self.training.is_editing() {
            match key.code {
                KeyCode::Enter => {
                    self.training.confirm_edit();
                }
                KeyCode::Esc => {
                    self.training.cancel_edit();
                }
                _ => {
                    self.training.handle_edit_key(key);
                }
            }
            return;
        }

        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.training.next_param(),
            KeyCode::Up | KeyCode::Char('k') => self.training.prev_param(),
            KeyCode::Enter => {
                if let Some(action) = self.training.handle_enter() {
                    match action {
                        TrainingAction::OpenModelPicker => {
                            self.pending_modal_context = Some(PendingModalTarget::TrainingModel);
                            let entries = self.model_picker_entries();
                            self.modal_stack.push(Modal::model_picker(entries));
                        }
                        TrainingAction::OpenDatasetPicker => {
                            self.pending_modal_context =
                                Some(PendingModalTarget::TrainingDataset);
                            let entries = self.dataset_picker_entries();
                            self.modal_stack.push(Modal::dataset_picker(entries));
                        }
                        TrainingAction::StartEdit => {} // Handled internally
                    }
                }
            }
            KeyCode::Char('S') => {
                // Start training
                self.start_training_prompt();
            }
            KeyCode::Char('x') => {
                // Stop running training
                if let Some(ref job_id) = self.active_training_job.clone() {
                    if let Some(runner) = &mut self.runner {
                        runner.cancel(job_id);
                    }
                }
            }
            _ => {}
        }
    }

    // --- Inference tab key handler ---
    fn handle_inference_key(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
            KeyCode::Enter if self.inference.input_focused => {
                self.submit_inference();
            }
            KeyCode::Backspace if self.inference.input_focused => {
                self.inference.delete_char();
            }
            KeyCode::Left if self.inference.input_focused => {
                self.inference.move_cursor_left();
            }
            KeyCode::Right if self.inference.input_focused => {
                self.inference.move_cursor_right();
            }
            KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Open model picker (Ctrl+P)
                self.pending_modal_context = Some(PendingModalTarget::InferenceModel);
                let entries = self.model_picker_entries();
                self.modal_stack.push(Modal::model_picker(entries));
            }
            KeyCode::Esc => {
                if self.inference.generating {
                    // Stop inference
                    if let Some(ref job_id) = self.active_inference_job.clone() {
                        if let Some(runner) = &mut self.runner {
                            runner.cancel(job_id);
                        }
                    }
                    self.inference.generating = false;
                }
            }
            KeyCode::Char('s')
                if key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                // Toggle settings focus
                self.inference.toggle_settings_focus();
            }
            KeyCode::Up if self.inference.settings_focused => {
                self.inference.prev_setting();
            }
            KeyCode::Down if self.inference.settings_focused => {
                self.inference.next_setting();
            }
            KeyCode::Char('+') | KeyCode::Char('=') if self.inference.settings_focused => {
                self.inference.increment_setting();
            }
            KeyCode::Char('-') if self.inference.settings_focused => {
                self.inference.decrement_setting();
            }
            // Scroll chat history
            KeyCode::PageUp => {
                self.inference.scroll_up();
            }
            KeyCode::PageDown => {
                self.inference.scroll_down();
            }
            KeyCode::Up if self.inference.input_focused => {
                self.inference.scroll_up();
            }
            KeyCode::Down if self.inference.input_focused => {
                self.inference.scroll_down();
            }
            KeyCode::Char(c) if self.inference.input_focused => {
                self.inference.insert_char(c);
            }
            _ => {}
        }
    }

    // --- Jobs tab key handler ---
    fn handle_jobs_key(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.jobs.next_row(),
            KeyCode::Up | KeyCode::Char('k') => self.jobs.prev_row(),
            KeyCode::Char('R') => self.jobs.scan_jobs(),
            KeyCode::Char('J') => self.jobs.scroll_log_down(),
            KeyCode::Char('K') => self.jobs.scroll_log_up(),
            KeyCode::PageDown => {
                for _ in 0..10 {
                    self.jobs.scroll_log_down();
                }
            }
            KeyCode::PageUp => {
                for _ in 0..10 {
                    self.jobs.scroll_log_up();
                }
            }
            _ => {}
        }
    }

    // --- Distillation tab key handler ---
    fn handle_distillation_key(&mut self, key: crossterm::event::KeyEvent) {
        if self.distillation.is_editing() {
            match key.code {
                KeyCode::Enter => {
                    self.distillation.confirm_edit();
                }
                KeyCode::Esc => {
                    self.distillation.cancel_edit();
                }
                _ => {
                    self.distillation.handle_edit_key(key);
                }
            }
            return;
        }

        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.distillation.next_param(),
            KeyCode::Up | KeyCode::Char('k') => self.distillation.prev_param(),
            KeyCode::Enter => {
                if let Some(action) = self.distillation.handle_enter() {
                    match action {
                        DistillAction::OpenTeacherPicker => {
                            self.pending_modal_context =
                                Some(PendingModalTarget::DistillTeacher);
                            let entries = self.model_picker_entries();
                            self.modal_stack.push(Modal::model_picker(entries));
                        }
                        DistillAction::OpenStudentPicker => {
                            self.pending_modal_context =
                                Some(PendingModalTarget::DistillStudent);
                            let entries = self.model_picker_entries();
                            self.modal_stack.push(Modal::model_picker(entries));
                        }
                        DistillAction::OpenDatasetPicker => {
                            self.pending_modal_context =
                                Some(PendingModalTarget::DistillDataset);
                            let entries = self.dataset_picker_entries();
                            self.modal_stack.push(Modal::dataset_picker(entries));
                        }
                        DistillAction::StartEdit => {}
                    }
                }
            }
            KeyCode::Char('S') => {
                self.start_distillation_prompt();
            }
            _ => {}
        }
    }

    // --- GRPO tab key handler ---
    fn handle_grpo_key(&mut self, key: crossterm::event::KeyEvent) {
        if self.grpo.is_editing() {
            match key.code {
                KeyCode::Enter => {
                    self.grpo.confirm_edit();
                }
                KeyCode::Esc => {
                    self.grpo.cancel_edit();
                }
                _ => {
                    self.grpo.handle_edit_key(key);
                }
            }
            return;
        }

        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.grpo.next_param(),
            KeyCode::Up | KeyCode::Char('k') => self.grpo.prev_param(),
            KeyCode::Enter => {
                if let Some(action) = self.grpo.handle_enter() {
                    match action {
                        GrpoAction::OpenModelPicker => {
                            self.pending_modal_context = Some(PendingModalTarget::GrpoModel);
                            let entries = self.model_picker_entries();
                            self.modal_stack.push(Modal::model_picker(entries));
                        }
                        GrpoAction::OpenDatasetPicker => {
                            self.pending_modal_context = Some(PendingModalTarget::GrpoDataset);
                            let entries = self.dataset_picker_entries();
                            self.modal_stack.push(Modal::dataset_picker(entries));
                        }
                        GrpoAction::StartEdit => {}
                    }
                }
            }
            KeyCode::Char('S') => {
                self.start_grpo_prompt();
            }
            _ => {}
        }
    }

    /// Handle mouse events.
    fn handle_mouse(&mut self, mouse: MouseEvent) {
        // Don't process mouse if a modal is open
        if !self.modal_stack.is_empty() {
            return;
        }

        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                let x = mouse.column;
                let y = mouse.row;

                // Check if click is in header/tab area
                if y < self.header_area.y + self.header_area.height {
                    // Approximate tab positions: logo takes 20 cols, then each tab ~15 cols
                    if x >= 20 {
                        let tab_idx = ((x - 20) / 15) as usize;
                        if let Some(&tab) = Tab::ALL.get(tab_idx) {
                            self.active_tab = tab;
                        }
                    }
                }
            }
            MouseEventKind::ScrollUp => {
                // Scroll up in the active tab's list
                match self.active_tab {
                    Tab::Models => self.models.prev_row(),
                    Tab::Datasets => self.datasets.prev_row(),
                    Tab::Training => self.training.prev_param(),
                    Tab::Jobs => self.jobs.prev_row(),
                    Tab::Distillation => self.distillation.prev_param(),
                    Tab::Grpo => self.grpo.prev_param(),
                    _ => {}
                }
            }
            MouseEventKind::ScrollDown => {
                match self.active_tab {
                    Tab::Models => self.models.next_row(),
                    Tab::Datasets => self.datasets.next_row(),
                    Tab::Training => self.training.next_param(),
                    Tab::Jobs => self.jobs.next_row(),
                    Tab::Distillation => self.distillation.next_param(),
                    Tab::Grpo => self.grpo.next_param(),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Handle a message from a background process.
    fn handle_app_msg(&mut self, msg: AppMsg) {
        match msg {
            AppMsg::JobStarted { job_id: _, job_type } => match job_type {
                JobType::Train | JobType::Distill | JobType::Grpo => {
                    self.training.set_status_running(0, 0, 0, 0, 0.0);
                }
                _ => {}
            },
            AppMsg::JobMetrics {
                job_id: _,
                step,
                epoch,
                total_epochs,
                total_steps,
                loss,
                lr,
                tok_sec,
                ane_fwd_ms,
                ane_bwd_ms,
                rmsnorm_ms,
                cblas_ms,
                adam_ms,
                total_ms,
            } => {
                // Update dashboard with metrics
                self.dashboard.push_sample(MetricSample {
                    step,
                    epoch,
                    total_epochs,
                    total_steps,
                    loss,
                    lr,
                    tok_sec,
                    ane_fwd_ms,
                    ane_bwd_ms,
                    rmsnorm_ms,
                    cblas_ms,
                    adam_ms,
                    total_ms,
                });

                // Update training status
                self.training
                    .set_status_running(step, epoch, total_epochs, total_steps, loss);
            }
            AppMsg::JobOutput { job_id, line } => {
                // Route to jobs tab live log
                self.jobs.append_live_output(&job_id, &line);
            }
            AppMsg::JobFinished {
                job_id,
                success,
                message,
            } => {
                if self.active_training_job.as_deref() == Some(&job_id) {
                    if success {
                        let loss = self.dashboard.samples.last().map(|s| s.loss).unwrap_or(0.0);
                        let steps = self.dashboard.samples.len();
                        self.training.set_status_completed(loss, steps);
                        // Auto-register the training output directory so it persists across sessions
                        if let Some(ref dir) = self.active_training_output_dir {
                            let canonical = dir.canonicalize().unwrap_or_else(|_| dir.clone());
                            self.models.add_custom_dir(canonical);
                        }
                        // Rescan models to pick up the trained output
                        self.models.scan_models();
                        // Offer to test inference with the trained model
                        self.pending_modal_context = Some(PendingModalTarget::TestInference);
                        self.modal_stack.push(Modal::confirm(
                            "Training Complete",
                            vec![
                                format!("Final loss: {loss:.4} over {steps} steps"),
                                String::new(),
                                "Test the trained model with inference?".into(),
                            ],
                        ));
                    } else {
                        self.training.set_status_failed(&message);
                    }
                    self.active_training_job = None;
                    self.active_training_output_dir = None;
                }
                if self.active_inference_job.as_deref() == Some(&job_id) {
                    self.active_inference_job = None;
                }
                // Pop progress modal if one is showing (for convert/download jobs)
                if matches!(self.modal_stack.last(), Some(Modal::Progress { .. })) {
                    self.modal_stack.pop();
                    if !success {
                        self.modal_stack.push(Modal::error("Job Failed", message.clone()));
                    }
                    // Rescan datasets in case a conversion produced new files
                    self.datasets.scan_datasets();
                }
                if let Some(runner) = &mut self.runner {
                    runner.remove(&job_id);
                }
                // Rescan jobs
                self.jobs.scan_jobs();
            }
            AppMsg::DownloadProgress { model_id, progress } => {
                // Could update a progress modal if one exists
                if let Some(Modal::Progress {
                    progress: p,
                    message,
                    ..
                }) = self.modal_stack.last_mut()
                {
                    *p = progress;
                    *message = format!("Downloading {model_id}... {:.0}%", progress * 100.0);
                }
            }
            AppMsg::DownloadComplete {
                model_id: _,
                success,
                message,
            } => {
                // Pop any progress modal
                if matches!(self.modal_stack.last(), Some(Modal::Progress { .. })) {
                    self.modal_stack.pop();
                }
                if success {
                    self.models.scan_models();
                } else {
                    self.modal_stack
                        .push(Modal::error("Download Failed", message));
                }
            }
            AppMsg::InferenceToken { token } => {
                self.inference.append_token(&token);
            }
            AppMsg::InferenceDone {
                tok_sec,
                total_tokens,
            } => {
                self.inference.finish_generation(tok_sec, total_tokens);
                self.active_inference_job = None;
            }
            AppMsg::InferenceError { message } => {
                self.inference.generating = false;
                self.active_inference_job = None;
                self.modal_stack
                    .push(Modal::error("Inference Error", message));
            }
        }
    }

    /// Handle a modal action after a modal is dismissed.
    fn handle_modal_action(&mut self, action: ModalAction) {
        let target = self.pending_modal_context.take();

        match action {
            ModalAction::None => {} // Cancelled
            ModalAction::Confirmed => {
                match target {
                    Some(PendingModalTarget::TrainingStart) => {
                        self.start_training();
                    }
                    Some(PendingModalTarget::DistillStart) => {
                        self.start_distillation();
                    }
                    Some(PendingModalTarget::GrpoStart) => {
                        self.start_grpo();
                    }
                    Some(PendingModalTarget::FuseStart) => {
                        self.start_fuse();
                    }
                    Some(PendingModalTarget::TestInference) => {
                        // Switch to inference tab and load the latest trained model
                        self.active_tab = Tab::Inference;
                        // Find the most recently trained model
                        if let Some(trained) = self
                            .models
                            .models
                            .iter()
                            .find(|m| matches!(m.source, ModelSource::Trained))
                        {
                            self.inference.load_model_settings(&trained.path);
                            self.inference.model_id = Some(trained.id.clone());
                        }
                    }
                    _ => {}
                }
            }
            ModalAction::SelectModel(id) => match target {
                Some(PendingModalTarget::TrainingModel) => {
                    self.training.set_model(&id);
                }
                Some(PendingModalTarget::InferenceModel) => {
                    let model = self.models.models.iter().find(|m| m.id == id).cloned();
                    if let Some(ref m) = model {
                        self.inference.load_model_settings(&m.path);
                    }

                    // Check if this is a LoRA adapter without base model info
                    let needs_base = model.as_ref().map_or(false, |m| {
                        is_lora_adapter(&m.path) && m.base_model.is_none()
                    });

                    if needs_base {
                        // Store the adapter model id, prompt user to pick base model
                        self.pending_modal_context =
                            Some(PendingModalTarget::InferenceBaseModel(id));
                        let entries: Vec<_> = self
                            .model_picker_entries()
                            .into_iter()
                            .filter(|e| !e.id.starts_with("trained/"))
                            .collect();
                        self.modal_stack.push(Modal::model_picker(entries));
                    } else {
                        self.inference.model_id = Some(id);
                    }
                }
                Some(PendingModalTarget::InferenceBaseModel(adapter_id)) => {
                    // User picked a base model for a LoRA adapter
                    let model_path = self.resolve_model_path(&id);
                    if let Some(entry) = self.models.models.iter_mut().find(|m| m.id == adapter_id)
                    {
                        entry.base_model = Some(id.clone());
                        // Also persist it to training_info.json
                        write_training_info(&entry.path, &id, &model_path);
                    }
                    self.inference.model_id = Some(adapter_id);
                }
                Some(PendingModalTarget::DistillTeacher) => {
                    self.distillation.set_teacher(&id);
                }
                Some(PendingModalTarget::DistillStudent) => {
                    self.distillation.set_student(&id);
                }
                Some(PendingModalTarget::GrpoModel) => {
                    self.grpo.set_model(&id);
                }
                Some(PendingModalTarget::FuseBaseModel(adapter_id)) => {
                    // User picked a base model for fusing — store it and confirm
                    let model_path = self.resolve_model_path(&id);
                    if let Some(entry) =
                        self.models.models.iter_mut().find(|m| m.id == adapter_id)
                    {
                        entry.base_model = Some(id.clone());
                        write_training_info(&entry.path, &id, &model_path);
                    }
                    // Now show fuse confirmation
                    let adapter_name = adapter_id
                        .strip_prefix("trained/")
                        .unwrap_or(&adapter_id);
                    let output_default = format!("./output/{adapter_name}--fused");
                    self.pending_modal_context = Some(PendingModalTarget::FuseStart);
                    self.modal_stack.push(Modal::confirm(
                        "Fuse LoRA Adapter",
                        vec![
                            format!("Adapter:  {adapter_id}"),
                            format!("Base:     {id}"),
                            format!("Output:   {output_default}"),
                            String::new(),
                            "This will merge LoRA weights into the base model".into(),
                            "and save a complete fused model.".into(),
                            String::new(),
                            "Proceed?".into(),
                        ],
                    ));
                }
                _ => {}
            },
            ModalAction::SelectDataset(path) => match target {
                Some(PendingModalTarget::TrainingDataset) => {
                    self.training.set_dataset(&path);
                }
                Some(PendingModalTarget::DistillDataset) => {
                    self.distillation.set_dataset(&path);
                }
                Some(PendingModalTarget::GrpoDataset) => {
                    self.grpo.set_dataset(&path);
                }
                _ => {}
            },
            ModalAction::TextSubmitted(text) => match target {
                Some(PendingModalTarget::DownloadModel) => {
                    self.download_model(&text);
                }
                Some(PendingModalTarget::AddModelDir) => {
                    let path = PathBuf::from(expand_tilde(&text));
                    if path.is_dir() {
                        self.models.add_custom_dir(path);
                    } else {
                        self.modal_stack
                            .push(Modal::error("Invalid Path", format!("'{}' is not a directory.", text)));
                    }
                }
                Some(PendingModalTarget::AddDatasetDir) => {
                    let path = PathBuf::from(expand_tilde(&text));
                    if path.is_dir() {
                        self.datasets.add_custom_dir(path);
                    } else {
                        self.modal_stack
                            .push(Modal::error("Invalid Path", format!("'{}' is not a directory.", text)));
                    }
                }
                Some(PendingModalTarget::ConvertDataset) => {
                    self.convert_dataset(&text);
                }
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

    // --- Action helpers ---

    fn model_picker_entries(&self) -> Vec<PickerEntry> {
        self.models
            .models
            .iter()
            .map(|m| {
                let mut tags = Vec::new();
                if matches!(m.source, ModelSource::Trained) {
                    tags.push("[trained]");
                }
                let base_info = m
                    .base_model
                    .as_deref()
                    .map(|b| format!(" (base: {b})"))
                    .unwrap_or_default();
                let tag_str = if tags.is_empty() {
                    String::new()
                } else {
                    format!(" {}", tags.join(" "))
                };
                PickerEntry {
                    id: m.id.clone(),
                    detail: format!(
                        "{} | {}{}{}",
                        m.size_display(),
                        m.architecture.as_deref().unwrap_or("-"),
                        tag_str,
                        base_info,
                    ),
                    path: m.path.display().to_string(),
                }
            })
            .collect()
    }

    /// Resolve a model ID to its filesystem path by looking up the models list.
    fn resolve_model_path(&self, model_id: &str) -> String {
        self.models
            .models
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| m.path.display().to_string())
            .unwrap_or_else(|| model_id.to_string())
    }

    fn dataset_picker_entries(&self) -> Vec<PickerEntry> {
        self.datasets
            .datasets
            .iter()
            .map(|d| PickerEntry {
                id: d.name.clone(),
                detail: format!(
                    "{} | {}",
                    d.format,
                    d.sample_count
                        .map(|c| format!("{c} samples"))
                        .unwrap_or_else(|| d.size_display())
                ),
                path: d.path.display().to_string(),
            })
            .collect()
    }

    fn start_training_prompt(&mut self) {
        let summary = self.training.config_summary();
        if let Err(msg) = self.training.validate_config() {
            self.modal_stack.push(Modal::error("Invalid Config", msg));
            return;
        }
        self.pending_modal_context = Some(PendingModalTarget::TrainingStart);
        self.modal_stack
            .push(Modal::confirm("Start Training?", summary));
    }

    fn start_training(&mut self) {
        let args = self.training.build_cli_args("train");
        let output_dir = self.training.output_dir();
        let metrics_file = output_dir.join("metrics.jsonl");

        // Write training metadata so we know the base model for inference later
        let model_id = extract_arg_value(&args, "--model").unwrap_or_default();
        let model_path = self.resolve_model_path(&model_id);
        write_training_info(&output_dir, &model_id, &model_path);

        // Truncate old metrics file so dashboard starts fresh
        let _ = std::fs::create_dir_all(&output_dir);
        let _ = std::fs::write(&metrics_file, "");

        let spec = CommandSpec {
            job_type: JobType::Train,
            args,
            metrics_file: Some(metrics_file.clone()),
            output_dir: Some(output_dir.clone()),
        };

        if let Some(runner) = &mut self.runner {
            let job_id = runner.spawn(spec);
            self.active_training_job = Some(job_id);
            self.active_training_output_dir = Some(output_dir);
            // Point dashboard at the (now-empty) metrics file
            self.dashboard.set_metrics_path(Some(metrics_file));
            self.active_tab = Tab::Dashboard;
        }
    }

    fn start_distillation_prompt(&mut self) {
        let summary = self.distillation.config_summary();
        if let Err(msg) = self.distillation.validate_config() {
            self.modal_stack.push(Modal::error("Invalid Config", msg));
            return;
        }
        self.pending_modal_context = Some(PendingModalTarget::DistillStart);
        self.modal_stack
            .push(Modal::confirm("Start Distillation?", summary));
    }

    fn start_distillation(&mut self) {
        let args = self.distillation.build_cli_args();
        let output_dir = self.distillation.output_dir();
        let metrics_file = output_dir.join("metrics.jsonl");

        // Write training metadata — for distillation, the student is the base model
        let model_id = extract_arg_value(&args, "--student").unwrap_or_default();
        let model_path = self.resolve_model_path(&model_id);
        write_training_info(&output_dir, &model_id, &model_path);

        // Truncate old metrics so dashboard starts fresh
        let _ = std::fs::create_dir_all(&output_dir);
        let _ = std::fs::write(&metrics_file, "");

        let spec = CommandSpec {
            job_type: JobType::Distill,
            args,
            metrics_file: Some(metrics_file.clone()),
            output_dir: Some(output_dir.clone()),
        };

        if let Some(runner) = &mut self.runner {
            let job_id = runner.spawn(spec);
            self.active_training_job = Some(job_id);
            self.active_training_output_dir = Some(output_dir);
            self.dashboard.set_metrics_path(Some(metrics_file));
            self.active_tab = Tab::Dashboard;
        }
    }

    fn start_grpo_prompt(&mut self) {
        let summary = self.grpo.config_summary();
        if let Err(msg) = self.grpo.validate_config() {
            self.modal_stack.push(Modal::error("Invalid Config", msg));
            return;
        }
        self.pending_modal_context = Some(PendingModalTarget::GrpoStart);
        self.modal_stack
            .push(Modal::confirm("Start GRPO?", summary));
    }

    fn start_grpo(&mut self) {
        let args = self.grpo.build_cli_args();
        let output_dir = self.grpo.output_dir();
        let metrics_file = output_dir.join("metrics.jsonl");

        let model_id = extract_arg_value(&args, "--model").unwrap_or_default();
        let model_path = self.resolve_model_path(&model_id);
        write_training_info(&output_dir, &model_id, &model_path);

        // Truncate old metrics so dashboard starts fresh
        let _ = std::fs::create_dir_all(&output_dir);
        let _ = std::fs::write(&metrics_file, "");

        let spec = CommandSpec {
            job_type: JobType::Grpo,
            args,
            metrics_file: Some(metrics_file.clone()),
            output_dir: Some(output_dir.clone()),
        };

        if let Some(runner) = &mut self.runner {
            let job_id = runner.spawn(spec);
            self.active_training_job = Some(job_id);
            self.active_training_output_dir = Some(output_dir);
            self.dashboard.set_metrics_path(Some(metrics_file));
            self.active_tab = Tab::Dashboard;
        }
    }

    fn start_fuse(&mut self) {
        // Find the currently selected trained model
        let Some(model) = self.models.selected_model().cloned() else {
            return;
        };
        let Some(ref base_id) = model.base_model else {
            self.modal_stack.push(Modal::error(
                "No Base Model",
                "Select a base model first (the model this adapter was trained from).",
            ));
            return;
        };

        let base_path = self.resolve_model_path(base_id);
        let adapter_name = model
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let output_dir = format!("./output/{adapter_name}--fused");
        let lora_file = if model.path.join("lora_weights.safetensors").exists() {
            model.path.join("lora_weights.safetensors")
        } else {
            model.path.clone()
        };

        let args = vec![
            "fuse".to_string(),
            "--model".to_string(),
            base_path,
            "--lora".to_string(),
            lora_file.display().to_string(),
            "--output".to_string(),
            output_dir,
        ];

        let spec = CommandSpec {
            job_type: JobType::Train, // reuse Train type for progress tracking
            args,
            metrics_file: None,
            output_dir: None,
        };

        if let Some(runner) = &mut self.runner {
            let job_id = runner.spawn(spec);
            // Show progress modal
            self.modal_stack.push(Modal::Progress {
                title: "Fusing Model".to_string(),
                message: format!("Fusing {} into {}...", model.id, base_id),
                progress: 0.0,
            });
            // Track as training job so completion triggers rescan
            self.active_training_job = Some(job_id);
        }
    }

    fn submit_inference(&mut self) {
        if self.inference.generating {
            return;
        }
        let prompt = self.inference.take_input();
        if prompt.is_empty() {
            return;
        }

        let Some(model_id) = self.inference.model_id.clone() else {
            self.modal_stack.push(Modal::error(
                "No Model",
                "Select a model first (Ctrl+P).",
            ));
            // Put the input back
            self.inference.set_input(&prompt);
            return;
        };

        // Resolve model: for LoRA adapters, pass base model + --lora; for others, pass path
        let model_entry = self
            .models
            .models
            .iter()
            .find(|m| m.id == model_id);

        let model_arg = model_entry
            .map(|m| m.path.display().to_string())
            .unwrap_or_else(|| model_id.clone());

        // For LoRA adapters: use base model as --model and adapter dir as --lora
        let (effective_model, lora_path) = if let Some(entry) = model_entry {
            if is_lora_adapter(&entry.path) {
                // Find the actual lora weights file
                let lora_file = if entry.path.join("lora_weights.safetensors").exists() {
                    entry.path.join("lora_weights.safetensors")
                } else {
                    // adapter_config.json style — pass the directory
                    entry.path.clone()
                };

                if let Some(ref base) = entry.base_model {
                    // Resolve base model to its actual path
                    let base_path = self.resolve_model_path(base);
                    (base_path, Some(lora_file.display().to_string()))
                } else {
                    // No base model info — pass the adapter dir as --model, pmetal may handle it
                    (model_arg, None)
                }
            } else {
                (model_arg, None)
            }
        } else {
            (model_arg, None)
        };

        // Add user message
        self.inference.add_user_message(&prompt);
        // Start placeholder assistant message
        self.inference.start_generation();

        // Build inference command
        let mut args = vec![
            "infer".to_string(),
            "--model".to_string(),
            effective_model,
            "--prompt".to_string(),
            prompt,
            "--max-tokens".to_string(),
            self.inference.max_tokens.to_string(),
        ];

        if let Some(lora) = lora_path {
            args.extend(["--lora".to_string(), lora]);
        }

        // Always show thinking content so TUI can display it separately
        args.push("--show-thinking".to_string());

        if self.inference.temperature > 0.0 {
            args.extend(["--temperature".to_string(), self.inference.temperature.to_string()]);
        }
        if self.inference.top_k > 0 {
            args.extend(["--top-k".to_string(), self.inference.top_k.to_string()]);
        }
        if self.inference.top_p < 1.0 {
            args.extend(["--top-p".to_string(), self.inference.top_p.to_string()]);
        }

        let spec = CommandSpec {
            job_type: JobType::Infer,
            args,
            metrics_file: None,
            output_dir: None,
        };

        if let Some(runner) = &mut self.runner {
            let job_id = runner.spawn(spec);
            self.active_inference_job = Some(job_id);
        }
    }

    fn download_model(&mut self, model_id: &str) {
        let args = vec!["download".to_string(), model_id.to_string()];
        let spec = CommandSpec {
            job_type: JobType::Download,
            args,
            metrics_file: None,
            output_dir: None,
        };

        self.modal_stack.push(Modal::progress(
            "Downloading",
            format!("Downloading {model_id}..."),
        ));

        if let Some(runner) = &mut self.runner {
            runner.spawn(spec);
        }
    }

    fn convert_dataset(&mut self, output_path: &str) {
        let Some(ds) = self.datasets.selected_dataset() else {
            return;
        };

        let input_path = ds.path.display().to_string();
        let output = expand_tilde(output_path);

        // Ensure output directory exists
        if let Some(parent) = std::path::Path::new(&output).parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let args = vec![
            "dataset".to_string(),
            "convert".to_string(),
            "--input".to_string(),
            input_path.clone(),
            "--output".to_string(),
            output.clone(),
        ];

        let spec = CommandSpec {
            job_type: JobType::Convert,
            args,
            metrics_file: None,
            output_dir: None,
        };

        self.modal_stack.push(Modal::progress(
            "Converting",
            format!("Converting {input_path} -> {output}..."),
        ));

        if let Some(runner) = &mut self.runner {
            runner.spawn(spec);
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

        // Store header area for mouse hit-testing
        self.header_area = header_area;

        // Header with tab bar
        Header {
            active_tab: self.active_tab,
            tabs: Tab::ALL,
        }
        .render(header_area, buf);

        // Clone dashboard data for split-borrow access in training/distill/grpo tabs.
        // This is cheap: max ~10K samples at 5fps render rate.
        let dash_samples;
        let dash_throughput;
        let needs_metrics = matches!(
            self.active_tab,
            Tab::Training | Tab::Distillation | Tab::Grpo
        );
        if needs_metrics {
            dash_samples = self.dashboard.samples.clone();
            dash_throughput = self.dashboard.throughput_data.clone();
        } else {
            dash_samples = Vec::new();
            dash_throughput = Vec::new();
        }

        // Active tab content
        match self.active_tab {
            Tab::Dashboard => (&self.dashboard).render(content_area, buf),
            Tab::Device => (&self.device).render(content_area, buf),
            Tab::Models => (&mut self.models).render(content_area, buf),
            Tab::Datasets => (&mut self.datasets).render(content_area, buf),
            Tab::Training => {
                self.training
                    .render_with_metrics(content_area, buf, &dash_samples, &dash_throughput);
            }
            Tab::Inference => (&mut self.inference).render(content_area, buf),
            Tab::Jobs => (&mut self.jobs).render(content_area, buf),
            Tab::Distillation => {
                self.distillation
                    .render_with_metrics(content_area, buf, &dash_samples, &dash_throughput);
            }
            Tab::Grpo => {
                self.grpo
                    .render_with_metrics(content_area, buf, &dash_samples, &dash_throughput);
            }
        }

        // Footer with keybindings
        Footer {
            tab: self.active_tab,
        }
        .render(footer_area, buf);

        // Render modal overlay (if any)
        if let Some(modal) = self.modal_stack.last_mut() {
            modal.render(area, buf);
        }
    }
}

/// Check if a model directory is a LoRA adapter (has lora_weights.safetensors or adapter_config.json).
fn is_lora_adapter(path: &std::path::Path) -> bool {
    path.join("lora_weights.safetensors").exists() || path.join("adapter_config.json").exists()
}

/// Extract the value following a flag in a CLI args list (e.g., `--model` → the next arg).
fn extract_arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Expand `~` prefix to the user's home directory.
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).display().to_string();
        }
    } else if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.display().to_string();
        }
    }
    path.to_string()
}
