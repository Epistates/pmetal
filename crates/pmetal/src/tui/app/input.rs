//! Keys, the mouse and pasted text.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

use super::{App, Pending};
use crate::tui::modal::{Modal, ModalAction};
use crate::tui::tabs::{InferenceFocus, JobSelection, Pick, Tab};
use crate::tui::widgets::{FormAction, tab_at};

fn ctrl(key: &KeyEvent, c: char) -> bool {
    key.code == KeyCode::Char(c) && key.modifiers.contains(KeyModifiers::CONTROL)
}

/// Rows a page key moves a log.
const PAGE: usize = 10;

impl App {
    /// Whether the keyboard is typing into something: while it is, the
    /// single-letter shortcuts (`q`, `?`, `p`) are letters.
    pub(crate) fn captures_text(&self) -> bool {
        match self.active_tab {
            Tab::Models => self.models.searching,
            Tab::Inference => self.inference.input_focused(),
            tab => self
                .forms
                .get(tab)
                .is_some_and(|form| form.panel().form.is_editing()),
        }
    }

    /// Move to another tab. A field half-edited on the tab left is put back
    /// as it was, so coming back finds a form and not an open edit.
    pub(crate) fn switch_tab(&mut self, tab: Tab) {
        if tab == self.active_tab {
            return;
        }
        if let Some(form) = self.forms.get_mut(self.active_tab) {
            form.panel_mut().form.cancel_edit();
        }
        self.models.searching = false;
        self.active_tab = tab;
    }

    /// Handle a key press.
    pub(crate) fn handle_key(&mut self, key: KeyEvent) {
        // The interface is on its way out; nothing it is asked now matters.
        if self.stopping {
            return;
        }

        // A dialog gets the key first.
        if let Some(modal) = self.modal_stack.last_mut() {
            // A progress dialog dismissed with Esc cancels what it was showing.
            let is_progress = matches!(modal, Modal::Progress { .. });
            if let Some(action) = modal.handle_key(key) {
                self.modal_stack.pop();
                if is_progress && matches!(action, ModalAction::None) {
                    self.cancel_hf_search();
                }
                self.handle_modal_action(action);
            }
            return;
        }

        // Ctrl+C: end an edit, or the tab's job, or the program.
        if ctrl(&key, 'c') {
            self.interrupt();
            return;
        }

        // Keys that mean the same whether or not a field is being typed into.
        match key.code {
            KeyCode::Tab => {
                let next = self.active_tab.next();
                self.switch_tab(next);
                return;
            }
            KeyCode::BackTab => {
                let prev = self.active_tab.prev();
                self.switch_tab(prev);
                return;
            }
            KeyCode::Right | KeyCode::Left if key.modifiers.contains(KeyModifiers::SHIFT) => {
                let to = if key.code == KeyCode::Right {
                    self.active_tab.next()
                } else {
                    self.active_tab.prev()
                };
                self.switch_tab(to);
                return;
            }
            // F2: toggle mouse capture (off lets the terminal select text)
            KeyCode::F(2) => {
                self.mouse_wanted = !self.mouse_wanted;
                self.notify(if self.mouse_wanted {
                    "Mouse capture on"
                } else {
                    "Mouse capture off: select text with the mouse"
                });
                return;
            }
            // Number keys for direct tab access (Alt and Ctrl both work, as
            // some terminals keep one of them).
            KeyCode::Char(c @ '1'..='9')
                if key
                    .modifiers
                    .intersects(KeyModifiers::ALT | KeyModifiers::CONTROL) =>
            {
                if let Some(&tab) = Tab::ALL.get((c as u8 - b'1') as usize) {
                    self.switch_tab(tab);
                }
                return;
            }
            // Jump-to-tab palette: Alt+1-9 reaches nine of twenty-one tabs.
            KeyCode::Char('p')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.active_tab != Tab::Inference =>
            {
                self.modal_stack.push(Modal::tab_picker(self.active_tab));
                return;
            }
            _ => {}
        }

        // The single-letter globals, unless the keyboard is typing.
        if !self.captures_text() {
            match key.code {
                KeyCode::Char('q') if self.active_tab != Tab::Inference => {
                    self.request_quit();
                    return;
                }
                KeyCode::Char('?') => {
                    self.modal_stack.push(Modal::help(self.active_tab));
                    return;
                }
                _ => {}
            }
        }

        match self.active_tab {
            Tab::Dashboard => self.dashboard_key(key),
            Tab::Device => {}
            Tab::Models => self.models_key(key),
            Tab::Datasets => self.datasets_key(key),
            Tab::Inference => self.inference_key(key),
            Tab::Jobs => self.jobs_key(key),
            tab => self.form_key(tab, key),
        }
    }

    /// Ctrl+C: leave an edit, else stop what this tab is running, else quit.
    pub(crate) fn interrupt(&mut self) {
        let tab = self.active_tab;
        if let Some(form) = self.forms.get_mut(tab) {
            if form.panel().form.is_editing() {
                form.panel_mut().form.cancel_edit();
                return;
            }
            if form.panel().is_running() {
                let title = form.panel().title;
                self.stop_tab_job(tab);
                self.notify(format!("Stopping {title}..."));
                return;
            }
        }
        if tab == Tab::Inference && self.inference.generating {
            self.stop_generation();
            return;
        }
        // The monitor shows one run; Ctrl+C there stops that run.
        if tab == Tab::Dashboard {
            if let Some(job) = self.dashboard.following().map(str::to_string) {
                self.stop_job(&job);
                self.notify("Stopping the run on the monitor...");
                return;
            }
        }
        if tab == Tab::Jobs {
            if let Some(job) = self.jobs.selected_live() {
                if job.status.is_active() {
                    let id = job.id.clone();
                    self.stop_job(&id);
                    self.notify("Stopping the selected job...");
                    return;
                }
            }
        }
        self.request_quit();
    }

    // ── Per-tab keys ────────────────────────────────────────────────────

    fn dashboard_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('p') => self.dashboard.toggle_pause(),
            KeyCode::Char('r') => self.dashboard.reset(),
            _ => {}
        }
    }

    fn models_key(&mut self, key: KeyEvent) {
        if self.models.searching {
            match key.code {
                KeyCode::Esc => {
                    self.models.searching = false;
                    self.models.search_query.clear();
                    self.models.search_changed();
                }
                // Enter keeps the filter and gives the keys back.
                KeyCode::Enter => self.models.searching = false,
                KeyCode::Backspace => {
                    self.models.search_query.pop();
                    self.models.search_changed();
                }
                KeyCode::Down => self.models.next_row(),
                KeyCode::Up => self.models.prev_row(),
                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.models.search_query.push(c);
                    self.models.search_changed();
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.models.next_row(),
            KeyCode::Up | KeyCode::Char('k') => self.models.prev_row(),
            KeyCode::Char('/') => self.models.searching = true,
            KeyCode::Esc if !self.models.search_query.is_empty() => {
                self.models.search_query.clear();
                self.models.search_changed();
            }
            KeyCode::Char('R') => self.rescan_models(),
            KeyCode::Char('d') => {
                self.pending = Some(Pending::DownloadModel);
                self.modal_stack.push(Modal::text_input(
                    "Download Model",
                    "Model ID (e.g. Qwen/Qwen3-0.6B):",
                ));
            }
            KeyCode::Char('a') => {
                self.pending = Some(Pending::AddModelDir);
                self.modal_stack.push(Modal::text_input(
                    "Add Model Directory",
                    "Path to directory containing models:",
                ));
            }
            KeyCode::Char('S') => {
                self.pending = Some(Pending::HfSearch);
                self.modal_stack.push(Modal::text_input(
                    "Search HuggingFace Hub",
                    "Search query (e.g. 'qwen3 0.6B', 'llama 8b'):",
                ));
            }
            KeyCode::Char('t') => self.train_selected_model(),
            KeyCode::Char('s') => self.distill_selected_model(),
            KeyCode::Char('i') => self.chat_with_selected_model(),
            KeyCode::Char('f') => self.fuse_selected_model(),
            _ => {}
        }
    }

    fn datasets_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.datasets.next_row(),
            KeyCode::Up | KeyCode::Char('k') => self.datasets.prev_row(),
            KeyCode::Char('R') => self.rescan_datasets(),
            KeyCode::Char('a') => {
                self.pending = Some(Pending::AddDatasetDir);
                self.modal_stack.push(Modal::text_input(
                    "Add Dataset Directory",
                    "Path to directory containing datasets:",
                ));
            }
            KeyCode::Char('c') => self.convert_selected_dataset(),
            _ => {}
        }
    }

    fn jobs_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.jobs.next_row(),
            KeyCode::Up | KeyCode::Char('k') => self.jobs.prev_row(),
            KeyCode::Char('R') => self.rescan_history(),
            KeyCode::PageDown => self.jobs.scroll_log_down(PAGE),
            KeyCode::PageUp => self.jobs.scroll_log_up(PAGE),
            KeyCode::End => self.jobs.follow_log(),
            KeyCode::Char('x') => {
                if let Some(JobSelection::Live(id)) = self.jobs.selection() {
                    let id = id.to_string();
                    self.stop_job(&id);
                    self.notify("Stopping the selected job...");
                }
            }
            _ => {}
        }
    }

    fn inference_key(&mut self, key: KeyEvent) {
        match key.code {
            // --- Whatever has the focus ---
            KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.request_quit();
            }
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.pending = Some(Pending::InferenceModel);
                let entries = self.model_picker_entries(|_| true);
                self.modal_stack.push(Modal::model_picker(entries));
            }
            KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.inference.toggle_settings_focus();
            }
            KeyCode::Char('h') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.inference.toggle_sidebar();
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if self.inference.focus == InferenceFocus::Browse {
                    self.inference.exit_browse_mode();
                } else {
                    self.inference.enter_browse_mode();
                }
            }
            KeyCode::PageUp => self.inference.scroll_up(),
            KeyCode::PageDown => self.inference.scroll_down(),

            // --- Browse mode ---
            KeyCode::Up | KeyCode::Char('k') if self.inference.focus == InferenceFocus::Browse => {
                self.inference.prev_message();
            }
            KeyCode::Down | KeyCode::Char('j')
                if self.inference.focus == InferenceFocus::Browse =>
            {
                self.inference.next_message();
            }
            KeyCode::Char('y') if self.inference.focus == InferenceFocus::Browse => {
                if let Some(text) = self.inference.yank_selected() {
                    // `pbcopy` is a process; it is not waited for here.
                    tokio::task::spawn_blocking(move || copy_to_clipboard(&text));
                    self.notify("Copied");
                }
            }
            KeyCode::Esc if self.inference.focus == InferenceFocus::Browse => {
                self.inference.exit_browse_mode();
            }

            // --- Settings sidebar ---
            KeyCode::Up if self.inference.settings_focused() => self.inference.prev_setting(),
            KeyCode::Down if self.inference.settings_focused() => self.inference.next_setting(),
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Right
                if self.inference.settings_focused() =>
            {
                self.inference.increment_setting();
            }
            KeyCode::Char('-') | KeyCode::Left if self.inference.settings_focused() => {
                self.inference.decrement_setting();
            }
            KeyCode::Char('d') | KeyCode::Delete if self.inference.settings_focused() => {
                self.inference.reset_setting();
            }
            KeyCode::Esc if self.inference.settings_focused() => {
                self.inference.toggle_settings_focus();
            }

            // --- Typing a message ---
            KeyCode::Esc if self.inference.generating => self.stop_generation(),
            KeyCode::Enter if self.inference.input_focused() => self.submit_inference(),
            KeyCode::Backspace if self.inference.input_focused() => self.inference.delete_char(),
            KeyCode::Left if self.inference.input_focused() => self.inference.move_cursor_left(),
            KeyCode::Right if self.inference.input_focused() => self.inference.move_cursor_right(),
            KeyCode::Up if self.inference.input_focused() => self.inference.scroll_up(),
            KeyCode::Down if self.inference.input_focused() => self.inference.scroll_down(),
            KeyCode::Char(c)
                if self.inference.input_focused()
                    && !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.inference.insert_char(c);
            }
            _ => {}
        }
    }

    // ── Form tabs ───────────────────────────────────────────────────────

    fn form_key(&mut self, tab: Tab, key: KeyEvent) {
        let Some(form) = self.forms.get_mut(tab) else {
            return;
        };

        // A field being edited takes every key.
        if form.panel().form.is_editing() {
            match key.code {
                KeyCode::Enter => {
                    let label = form
                        .panel()
                        .form
                        .current()
                        .map(|f| f.label.clone())
                        .unwrap_or_default();
                    if form.panel_mut().form.confirm_edit() {
                        form.field_changed(&label);
                    }
                    // A refused edit stays open and says why on its row.
                }
                KeyCode::Esc => form.panel_mut().form.cancel_edit(),
                _ => form.panel_mut().form.handle_edit_key(key),
            }
            return;
        }

        match key.code {
            KeyCode::Down | KeyCode::Char('j') => form.panel_mut().form.next_param(),
            KeyCode::Up | KeyCode::Char('k') => form.panel_mut().form.prev_param(),
            KeyCode::Left => {
                let label = form.panel().form.current().map(|f| f.label.clone());
                form.panel_mut().form.cycle_back();
                if let Some(label) = label {
                    form.field_changed(&label);
                }
            }
            KeyCode::Right => {
                // Enter on a toggle or an option list is "next".
                if let Some(field) = form.panel().form.current().filter(|f| f.is_cycleable()) {
                    let label = field.label.clone();
                    form.panel_mut().form.handle_enter();
                    form.field_changed(&label);
                }
            }
            KeyCode::Enter => {
                let label = form
                    .panel()
                    .form
                    .current()
                    .map(|f| f.label.clone())
                    .unwrap_or_default();
                match form.panel_mut().form.handle_enter() {
                    Some(FormAction::OpenModelPicker { field_label }) => {
                        self.open_model_picker(tab, field_label);
                    }
                    Some(FormAction::OpenDatasetPicker { field_label }) => {
                        self.open_dataset_picker(tab, field_label);
                    }
                    Some(FormAction::StartEdit) => {}
                    None => form.field_changed(&label),
                }
            }
            KeyCode::Char('S') => self.start_prompt(tab),
            KeyCode::Char('x') => {
                if form.panel().is_running() {
                    let title = form.panel().title;
                    self.stop_tab_job(tab);
                    self.notify(format!("Stopping {title}..."));
                } else {
                    self.notify("Nothing is running here");
                }
            }
            KeyCode::Char('L')
                if matches!(
                    tab,
                    Tab::Training | Tab::Pretrain | Tab::Distillation | Tab::Grpo
                ) =>
            {
                self.open_learning_rate_modal(tab);
            }
            KeyCode::PageUp => form.panel_mut().log.scroll_up(PAGE),
            KeyCode::PageDown => form.panel_mut().log.scroll_down(PAGE),
            KeyCode::End => form.panel_mut().log.follow(),
            _ => {}
        }
    }

    // ── Dialogs that pick ───────────────────────────────────────────────

    pub(crate) fn open_model_picker(&mut self, tab: Tab, field: String) {
        self.pending = Some(Pending::PickModel { tab, field });
        let entries = self.model_picker_entries(|_| true);
        self.modal_stack.push(Modal::model_picker(entries));
    }

    pub(crate) fn open_dataset_picker(&mut self, tab: Tab, field: String) {
        self.pending = Some(Pending::PickDataset { tab, field });
        let entries = self.dataset_picker_entries();
        self.modal_stack.push(Modal::dataset_picker(entries));
    }

    // ── Shortcuts from the Models tab ───────────────────────────────────

    fn train_selected_model(&mut self) {
        let Some(model) = self.models.selected_model().cloned() else {
            return;
        };
        let training = &mut self.forms.training;
        crate::tui::tabs::FormTab::on_pick(
            training,
            "Model",
            Pick::Model {
                id: &model.id,
                architecture: model.architecture.as_deref(),
            },
        );
        self.switch_tab(Tab::Training);
        self.open_dataset_picker(Tab::Training, "Dataset".to_string());
    }

    fn distill_selected_model(&mut self) {
        let Some(model) = self.models.selected_model().cloned() else {
            return;
        };
        crate::tui::tabs::FormTab::on_pick(
            &mut self.forms.distillation,
            "Student",
            Pick::Model {
                id: &model.id,
                architecture: model.architecture.as_deref(),
            },
        );
        self.switch_tab(Tab::Distillation);
        self.open_dataset_picker(Tab::Distillation, "Dataset".to_string());
    }

    // ── The mouse ───────────────────────────────────────────────────────

    /// Handle mouse events.
    pub(crate) fn handle_mouse(&mut self, mouse: MouseEvent) {
        // A dialog owns the screen.
        if !self.modal_stack.is_empty() || self.stopping {
            return;
        }
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) if mouse.row < 2 => {
                if let Some(tab) =
                    tab_at(mouse.column, self.header_width, Tab::ALL, self.active_tab)
                {
                    self.switch_tab(tab);
                }
            }
            MouseEventKind::ScrollUp => self.scroll_active(false),
            MouseEventKind::ScrollDown => self.scroll_active(true),
            _ => {}
        }
    }

    fn scroll_active(&mut self, down: bool) {
        match self.active_tab {
            Tab::Models if down => self.models.next_row(),
            Tab::Models => self.models.prev_row(),
            Tab::Datasets if down => self.datasets.next_row(),
            Tab::Datasets => self.datasets.prev_row(),
            Tab::Jobs if down => self.jobs.next_row(),
            Tab::Jobs => self.jobs.prev_row(),
            Tab::Inference if down => self.inference.scroll_down(),
            Tab::Inference => self.inference.scroll_up(),
            tab => {
                if let Some(form) = self.forms.get_mut(tab) {
                    if down {
                        form.panel_mut().form.next_param();
                    } else {
                        form.panel_mut().form.prev_param();
                    }
                }
            }
        }
    }

    // ── Pasted text ─────────────────────────────────────────────────────

    pub(crate) fn handle_paste(&mut self, text: &str) {
        if let Some(modal) = self.modal_stack.last_mut() {
            modal.paste(text);
            return;
        }
        match self.active_tab {
            Tab::Models if self.models.searching => {
                self.models
                    .search_query
                    .extend(text.chars().filter(|c| !c.is_control()));
                self.models.search_changed();
            }
            Tab::Inference if self.inference.input_focused() => self.inference.paste(text),
            tab => {
                if let Some(form) = self.forms.get_mut(tab) {
                    form.panel_mut().form.paste(text);
                }
            }
        }
    }
}

/// Copy text to the system clipboard using macOS `pbcopy`.
fn copy_to_clipboard(text: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    if let Ok(mut child) = Command::new("pbcopy").stdin(Stdio::piped()).spawn() {
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(text.as_bytes());
        }
        let _ = child.wait();
    }
}
