//! Modal dialog system for the PMetal TUI.
//!
//! Provides overlay dialogs for confirmations, text input, model/dataset
//! selection, and error display. Modals are rendered on top of the active
//! tab content and capture all key events while open.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Clear, HighlightSpacing, List, ListItem, ListState, Paragraph, Row,
    StatefulWidget, Table, TableState, Widget, Wrap,
};

use crate::tui::tabs::Tab;
use crate::tui::theme::THEME;

/// Actions that a modal can produce when dismissed.
#[derive(Debug, Clone)]
pub enum ModalAction {
    /// No action (cancelled/closed).
    None,
    /// User confirmed (for Confirm dialogs).
    Confirmed,
    /// User selected a model by ID.
    SelectModel(String),
    /// User selected a dataset by path.
    SelectDataset(String),
    /// User submitted text input.
    TextSubmitted(String),
    /// User wants to download a model from HF search results.
    HfDownload(String),
    /// User picked a tab to jump to.
    SelectTab(Tab),
}

/// A modal dialog.
pub enum Modal {
    /// Confirmation dialog with yes/no.
    Confirm {
        title: String,
        lines: Vec<String>,
        selected: bool, // true = Yes, false = No
    },
    /// Text input dialog.
    TextInput {
        title: String,
        prompt: String,
        value: String,
        cursor: usize,
    },
    /// Model picker (list of cached models).
    ModelPicker {
        models: Vec<PickerEntry>,
        table_state: TableState,
        search: String,
        searching: bool,
    },
    /// Dataset picker (list of local datasets) with optional `/` search filter.
    DatasetPicker {
        datasets: Vec<PickerEntry>,
        list_state: ListState,
        /// Current search string (filtered on every key while `searching`).
        search: String,
        /// Whether the user has activated search mode with `/`.
        searching: bool,
    },
    /// Error display.
    Error { title: String, message: String },
    /// Progress indicator for ongoing operations.
    Progress {
        title: String,
        message: String,
        progress: f64,
    },
    /// HuggingFace Hub search results.
    HfSearch {
        entries: Vec<HfSearchEntry>,
        table_state: TableState,
    },
    /// Contextual help overlay showing global + per-tab keybindings.
    Help { tab: Tab },
    /// Jump-to-tab palette.
    ///
    /// `Alt+1-9` only reaches the first nine of twenty tabs, and cycling with
    /// Tab is up to ten presses. This lists every tab and filters as you type.
    TabPicker {
        list_state: ListState,
        search: String,
    },
}

/// An entry in the HF search results modal.
#[derive(Debug, Clone)]
pub struct HfSearchEntry {
    /// Model ID (e.g., "Qwen/Qwen3-0.6B").
    pub model_id: String,
    /// Formatted parameter count (e.g., "0.6B").
    pub params: String,
    /// Formatted download count (e.g., "1.2M").
    pub downloads: String,
    /// Formatted memory requirement (e.g., "1.8 GB").
    pub memory: String,
    /// Fit level on this device.
    pub fit_level: pmetal_hub::FitLevel,
    /// Estimated tokens per second.
    pub estimated_tps: String,
}

/// A simple entry for picker modals.
#[derive(Debug, Clone)]
pub struct PickerEntry {
    pub id: String,
    pub detail: String,
    pub path: String,
}

impl Modal {
    // --- Constructors ---

    pub fn confirm(title: impl Into<String>, lines: Vec<String>) -> Self {
        Modal::Confirm {
            title: title.into(),
            lines,
            selected: true,
        }
    }

    pub fn text_input(title: impl Into<String>, prompt: impl Into<String>) -> Self {
        Modal::TextInput {
            title: title.into(),
            prompt: prompt.into(),
            value: String::new(),
            cursor: 0,
        }
    }

    pub fn model_picker(models: Vec<PickerEntry>) -> Self {
        let mut state = TableState::default();
        if !models.is_empty() {
            state.select(Some(0));
        }
        Modal::ModelPicker {
            models,
            table_state: state,
            search: String::new(),
            searching: false,
        }
    }

    pub fn dataset_picker(datasets: Vec<PickerEntry>) -> Self {
        let mut state = ListState::default();
        if !datasets.is_empty() {
            state.select(Some(0));
        }
        Modal::DatasetPicker {
            datasets,
            list_state: state,
            search: String::new(),
            searching: false,
        }
    }

    pub fn help(tab: Tab) -> Self {
        Modal::Help { tab }
    }

    pub fn tab_picker(current: Tab) -> Self {
        let mut state = ListState::default();
        state.select(Some(
            Tab::ALL.iter().position(|t| *t == current).unwrap_or(0),
        ));
        Modal::TabPicker {
            list_state: state,
            search: String::new(),
        }
    }

    pub fn error(title: impl Into<String>, message: impl Into<String>) -> Self {
        Modal::Error {
            title: title.into(),
            message: message.into(),
        }
    }

    pub fn progress(title: impl Into<String>, message: impl Into<String>) -> Self {
        Modal::Progress {
            title: title.into(),
            message: message.into(),
            progress: 0.0,
        }
    }

    pub fn hf_search(entries: Vec<HfSearchEntry>) -> Self {
        let mut state = TableState::default();
        if !entries.is_empty() {
            state.select(Some(0));
        }
        Modal::HfSearch {
            entries,
            table_state: state,
        }
    }

    // --- Event Handling ---

    /// Handle a key event. Returns `Some(action)` if the modal should be dismissed.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalAction> {
        match self {
            Modal::Confirm { selected, .. } => match key.code {
                KeyCode::Left | KeyCode::Char('h') => {
                    *selected = true;
                    None
                }
                KeyCode::Right | KeyCode::Char('l') => {
                    *selected = false;
                    None
                }
                KeyCode::Char('y') | KeyCode::Char('Y') => Some(ModalAction::Confirmed),
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => Some(ModalAction::None),
                KeyCode::Enter => {
                    if *selected {
                        Some(ModalAction::Confirmed)
                    } else {
                        Some(ModalAction::None)
                    }
                }
                _ => None,
            },

            Modal::TextInput { value, cursor, .. } => match key.code {
                KeyCode::Char(c) => {
                    value.insert(*cursor, c);
                    *cursor += c.len_utf8();
                    None
                }
                KeyCode::Backspace => {
                    if *cursor > 0 {
                        let prev = value[..*cursor]
                            .chars()
                            .last()
                            .map(|c| c.len_utf8())
                            .unwrap_or(0);
                        *cursor -= prev;
                        value.remove(*cursor);
                    }
                    None
                }
                KeyCode::Left => {
                    if *cursor > 0 {
                        let prev = value[..*cursor]
                            .chars()
                            .last()
                            .map(|c| c.len_utf8())
                            .unwrap_or(0);
                        *cursor -= prev;
                    }
                    None
                }
                KeyCode::Right => {
                    if *cursor < value.len() {
                        let next = value[*cursor..]
                            .chars()
                            .next()
                            .map(|c| c.len_utf8())
                            .unwrap_or(0);
                        *cursor += next;
                    }
                    None
                }
                KeyCode::Enter => {
                    if !value.trim().is_empty() {
                        Some(ModalAction::TextSubmitted(value.clone()))
                    } else {
                        None
                    }
                }
                KeyCode::Esc => Some(ModalAction::None),
                _ => None,
            },

            Modal::ModelPicker {
                models,
                table_state,
                search,
                searching,
            } => match key.code {
                KeyCode::Char('/') if !*searching => {
                    *searching = true;
                    None
                }
                KeyCode::Char(c) if *searching => {
                    search.push(c);
                    // Always reset to 0 on new character — the filtered set
                    // may have shrunk and the previous index could be past the end.
                    table_state.select(Some(0));
                    None
                }
                KeyCode::Backspace if *searching => {
                    search.pop();
                    // After removing a character the filtered list may grow;
                    // clamp the selection to the new last item if it overshot.
                    let filtered = filtered_models(models, search);
                    let max_idx = filtered.len().saturating_sub(1);
                    if let Some(i) = table_state.selected() {
                        if i > max_idx {
                            table_state.select(Some(max_idx));
                        }
                    }
                    None
                }
                KeyCode::Esc if *searching => {
                    *searching = false;
                    search.clear();
                    // Restore selection to 0 (the full list is restored).
                    if !models.is_empty() {
                        table_state.select(Some(0));
                    }
                    None
                }
                KeyCode::Esc => Some(ModalAction::None),
                KeyCode::Down | KeyCode::Char('j') => {
                    let filtered = filtered_models(models, search);
                    let count = filtered.len();
                    if count > 0 {
                        let i = table_state.selected().map_or(0, |i| (i + 1) % count);
                        table_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    let filtered = filtered_models(models, search);
                    let count = filtered.len();
                    if count > 0 {
                        let i = table_state
                            .selected()
                            .map_or(0, |i| (i + count - 1) % count);
                        table_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Enter => {
                    let filtered = filtered_models(models, search);
                    if let Some(idx) = table_state.selected() {
                        if let Some(entry) = filtered.get(idx) {
                            return Some(ModalAction::SelectModel(entry.id.clone()));
                        }
                    }
                    None
                }
                _ => None,
            },

            Modal::TabPicker { list_state, search } => match key.code {
                KeyCode::Esc => Some(ModalAction::None),
                KeyCode::Backspace => {
                    search.pop();
                    let max_idx = filtered_tabs(search).len().saturating_sub(1);
                    if let Some(i) = list_state.selected() {
                        if i > max_idx {
                            list_state.select(Some(max_idx));
                        }
                    }
                    None
                }
                KeyCode::Down => {
                    let count = filtered_tabs(search).len();
                    if count > 0 {
                        let i = list_state.selected().map_or(0, |i| (i + 1) % count);
                        list_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Up => {
                    let count = filtered_tabs(search).len();
                    if count > 0 {
                        let i = list_state.selected().map_or(0, |i| (i + count - 1) % count);
                        list_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Enter => {
                    let filtered = filtered_tabs(search);
                    list_state
                        .selected()
                        .and_then(|i| filtered.get(i).copied())
                        .map(ModalAction::SelectTab)
                }
                // Every printable key filters; j/k would otherwise shadow the
                // letters in tab names.
                KeyCode::Char(c) => {
                    search.push(c);
                    list_state.select(Some(0));
                    None
                }
                _ => None,
            },

            Modal::DatasetPicker {
                datasets,
                list_state,
                search,
                searching,
            } => match key.code {
                KeyCode::Char('/') if !*searching => {
                    *searching = true;
                    None
                }
                KeyCode::Char(c) if *searching => {
                    search.push(c);
                    // Filtered list may shrink — reset to first visible item.
                    list_state.select(Some(0));
                    None
                }
                KeyCode::Backspace if *searching => {
                    search.pop();
                    // Filtered list may grow — clamp to last item if overshot.
                    let filtered = filtered_datasets(datasets, search);
                    let max_idx = filtered.len().saturating_sub(1);
                    if let Some(i) = list_state.selected() {
                        if i > max_idx {
                            list_state.select(Some(max_idx));
                        }
                    }
                    None
                }
                KeyCode::Esc if *searching => {
                    *searching = false;
                    search.clear();
                    // Restore selection to 0 (full list restored).
                    if !datasets.is_empty() {
                        list_state.select(Some(0));
                    }
                    None
                }
                KeyCode::Esc => Some(ModalAction::None),
                KeyCode::Down | KeyCode::Char('j') => {
                    let filtered = filtered_datasets(datasets, search);
                    let count = filtered.len();
                    if count > 0 {
                        let i = list_state.selected().map_or(0, |i| (i + 1) % count);
                        list_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    let filtered = filtered_datasets(datasets, search);
                    let count = filtered.len();
                    if count > 0 {
                        let i = list_state.selected().map_or(0, |i| (i + count - 1) % count);
                        list_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Enter => {
                    let filtered = filtered_datasets(datasets, search);
                    if let Some(idx) = list_state.selected() {
                        if let Some(entry) = filtered.get(idx) {
                            return Some(ModalAction::SelectDataset(entry.path.clone()));
                        }
                    }
                    None
                }
                _ => None,
            },

            Modal::Error { .. } => match key.code {
                KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') => Some(ModalAction::None),
                _ => None,
            },

            Modal::Progress { .. } => match key.code {
                KeyCode::Esc => Some(ModalAction::None),
                _ => None,
            },

            Modal::HfSearch {
                entries,
                table_state,
            } => match key.code {
                KeyCode::Esc | KeyCode::Char('q') => Some(ModalAction::None),
                KeyCode::Down | KeyCode::Char('j') => {
                    let count = entries.len();
                    if count > 0 {
                        let i = table_state.selected().map_or(0, |i| (i + 1) % count);
                        table_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    let count = entries.len();
                    if count > 0 {
                        let i = table_state
                            .selected()
                            .map_or(0, |i| (i + count - 1) % count);
                        table_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Enter | KeyCode::Char('d') => {
                    // Download selected model
                    if let Some(idx) = table_state.selected() {
                        if let Some(entry) = entries.get(idx) {
                            return Some(ModalAction::HfDownload(entry.model_id.clone()));
                        }
                    }
                    None
                }
                _ => None,
            },

            Modal::Help { .. } => match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') | KeyCode::Char('?') => {
                    Some(ModalAction::None)
                }
                _ => None,
            },
        }
    }

    // --- Rendering ---

    /// Render the modal as an overlay on the given area.
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let popup_area = centered_rect(area, self.width_pct(), self.height_pct());

        // Clear the area behind the modal
        Clear.render(popup_area, buf);

        match self {
            Modal::Confirm {
                title,
                lines,
                selected,
            } => render_confirm(popup_area, buf, title, lines, *selected),
            Modal::TextInput {
                title,
                prompt,
                value,
                cursor,
            } => render_text_input(popup_area, buf, title, prompt, value, *cursor),
            Modal::ModelPicker {
                models,
                table_state,
                search,
                searching,
            } => render_model_picker(popup_area, buf, models, table_state, search, *searching),
            Modal::DatasetPicker {
                datasets,
                list_state,
                search,
                searching,
            } => render_dataset_picker(popup_area, buf, datasets, list_state, search, *searching),
            Modal::Error { title, message } => render_error(popup_area, buf, title, message),
            Modal::Progress {
                title,
                message,
                progress,
            } => render_progress(popup_area, buf, title, message, *progress),
            Modal::HfSearch {
                entries,
                table_state,
            } => render_hf_search(popup_area, buf, entries, table_state),
            Modal::Help { tab } => render_help(popup_area, buf, *tab),
            Modal::TabPicker { list_state, search } => {
                render_tab_picker(popup_area, buf, list_state, search)
            }
        }
    }

    fn width_pct(&self) -> u16 {
        match self {
            Modal::Confirm { .. } | Modal::Error { .. } | Modal::Progress { .. } => 50,
            Modal::TextInput { .. } => 60,
            Modal::ModelPicker { .. } | Modal::DatasetPicker { .. } => 70,
            Modal::HfSearch { .. } => 85,
            Modal::Help { .. } => 65,
            Modal::TabPicker { .. } => 45,
        }
    }

    fn height_pct(&self) -> u16 {
        match self {
            Modal::Confirm { .. } => 30,
            Modal::TextInput { .. } => 20,
            Modal::Error { .. } | Modal::Progress { .. } => 25,
            Modal::ModelPicker { .. } | Modal::DatasetPicker { .. } => 60,
            Modal::HfSearch { .. } => 70,
            Modal::Help { .. } => 75,
            Modal::TabPicker { .. } => 70,
        }
    }
}

// --- Render helpers ---

fn render_confirm(area: Rect, buf: &mut Buffer, title: &str, lines: &[String], selected: bool) {
    let block = Block::default()
        .title(format!(" {title} "))
        .title_style(THEME.block_title_focused)
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);
    let inner = block.inner(area);
    block.render(area, buf);

    let [content_area, _, button_area] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(inner);

    // Message lines
    let msg_lines: Vec<Line> = lines
        .iter()
        .map(|l| Line::from(Span::styled(l.as_str(), THEME.text)))
        .collect();
    Paragraph::new(msg_lines)
        .wrap(Wrap { trim: false })
        .render(content_area, buf);

    // Buttons
    let yes_style = if selected {
        THEME.table_selected
    } else {
        THEME.text_dim
    };
    let no_style = if !selected {
        THEME.table_selected
    } else {
        THEME.text_dim
    };

    Line::from(vec![
        Span::raw("      "),
        Span::styled("  [Y]es  ", yes_style),
        Span::raw("    "),
        Span::styled("  [N]o  ", no_style),
    ])
    .render(button_area, buf);
}

fn render_text_input(
    area: Rect,
    buf: &mut Buffer,
    title: &str,
    prompt: &str,
    value: &str,
    _cursor: usize,
) {
    let block = Block::default()
        .title(format!(" {title} "))
        .title_style(THEME.block_title_focused)
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);
    let inner = block.inner(area);
    block.render(area, buf);

    let [prompt_area, input_area, help_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Fill(1),
    ])
    .areas(inner);

    Line::from(Span::styled(prompt, THEME.text_dim)).render(prompt_area, buf);

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);
    let display = if value.is_empty() { "..." } else { value };
    let style = if value.is_empty() {
        THEME.text_muted
    } else {
        THEME.text_bright
    };
    Paragraph::new(display)
        .style(style)
        .block(input_block)
        .render(input_area, buf);

    Line::from(Span::styled(
        "Enter to confirm, Esc to cancel",
        THEME.text_muted,
    ))
    .render(help_area, buf);
}

fn render_model_picker(
    area: Rect,
    buf: &mut Buffer,
    models: &[PickerEntry],
    table_state: &mut TableState,
    search: &str,
    searching: bool,
) {
    let title = if searching {
        format!(" Select Model [/{search}] ")
    } else {
        " Select Model ".to_string()
    };
    let block = Block::default()
        .title(title)
        .title_style(THEME.block_title_focused)
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);

    let filtered = filtered_models(models, search);

    if filtered.is_empty() {
        Paragraph::new(if models.is_empty() {
            "\n  No cached models. Download one first."
        } else {
            "\n  No models match search."
        })
        .style(THEME.text_muted)
        .block(block)
        .render(area, buf);
        return;
    }

    let header = Row::new(vec!["Model ID", "Details"]).style(THEME.table_header);
    let rows: Vec<Row> = filtered
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let style = if i % 2 == 0 {
                THEME.table_row
            } else {
                THEME.table_row_alt
            };
            Row::new(vec![
                Cell::new(entry.id.clone()),
                Cell::new(entry.detail.clone()),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        [Constraint::Percentage(60), Constraint::Percentage(40)],
    )
    .header(header)
    .block(block)
    .row_highlight_style(THEME.table_selected)
    .highlight_spacing(HighlightSpacing::Always);

    StatefulWidget::render(table, area, buf, table_state);
}

/// Return the subset of `datasets` whose `id` or `path` contains `search`
/// (case-insensitive).  Returns the full slice when `search` is empty.
/// Tabs whose name contains `search`, case-insensitively.
fn filtered_tabs(search: &str) -> Vec<Tab> {
    if search.is_empty() {
        return Tab::ALL.to_vec();
    }
    let lower = search.to_lowercase();
    Tab::ALL
        .iter()
        .copied()
        .filter(|t| t.to_string().to_lowercase().contains(&lower))
        .collect()
}

fn render_tab_picker(area: Rect, buf: &mut Buffer, list_state: &mut ListState, search: &str) {
    let title = if search.is_empty() {
        " Go to Tab ".to_string()
    } else {
        format!(" Go to Tab [{search}] ")
    };
    let block = Block::default()
        .title(title)
        .title_style(THEME.block_title_focused)
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);

    let filtered = filtered_tabs(search);
    if filtered.is_empty() {
        Paragraph::new("\n  No tab matches.")
            .style(THEME.text_muted)
            .block(block)
            .render(area, buf);
        return;
    }

    let items: Vec<ListItem> = filtered
        .iter()
        .map(|tab| {
            ListItem::new(Line::from(vec![
                Span::styled(format!(" {} ", tab.icon()), THEME.text_dim),
                Span::styled(tab.to_string(), THEME.kv_value),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(THEME.table_selected)
        .highlight_symbol("> ");
    StatefulWidget::render(list, area, buf, list_state);
}

fn filtered_datasets<'a>(datasets: &'a [PickerEntry], search: &str) -> Vec<&'a PickerEntry> {
    if search.is_empty() {
        datasets.iter().collect()
    } else {
        let lower = search.to_lowercase();
        datasets
            .iter()
            .filter(|e| {
                e.id.to_lowercase().contains(&lower) || e.path.to_lowercase().contains(&lower)
            })
            .collect()
    }
}

fn render_dataset_picker(
    area: Rect,
    buf: &mut Buffer,
    datasets: &[PickerEntry],
    list_state: &mut ListState,
    search: &str,
    searching: bool,
) {
    let title = if searching {
        format!(" Select Dataset [/{search}] ")
    } else {
        " Select Dataset ".to_string()
    };
    let block = Block::default()
        .title(title)
        .title_style(THEME.block_title_focused)
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);

    let filtered = filtered_datasets(datasets, search);

    if filtered.is_empty() {
        Paragraph::new(if datasets.is_empty() {
            "\n  No datasets found in ./, ./data/, ./datasets/"
        } else {
            "\n  No datasets match search."
        })
        .style(THEME.text_muted)
        .block(block)
        .render(area, buf);
        return;
    }

    let items: Vec<ListItem> = filtered
        .iter()
        .map(|ds| {
            ListItem::new(Line::from(vec![
                Span::styled(ds.id.as_str(), THEME.kv_value),
                Span::raw("  "),
                Span::styled(ds.detail.as_str(), THEME.text_dim),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(THEME.table_selected);

    StatefulWidget::render(list, area, buf, list_state);
}

fn render_error(area: Rect, buf: &mut Buffer, title: &str, message: &str) {
    let block = Block::default()
        .title(format!(" {title} "))
        .title_style(THEME.text_error)
        .borders(Borders::ALL)
        .border_style(THEME.text_error);
    let inner = block.inner(area);
    block.render(area, buf);

    let [msg_area, _, help_area] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(inner);

    Paragraph::new(message)
        .style(THEME.text)
        .wrap(Wrap { trim: false })
        .render(msg_area, buf);

    Line::from(Span::styled(
        "Press Enter or Esc to close",
        THEME.text_muted,
    ))
    .render(help_area, buf);
}

fn render_progress(area: Rect, buf: &mut Buffer, title: &str, message: &str, progress: f64) {
    let block = Block::default()
        .title(format!(" {title} "))
        .title_style(THEME.block_title_focused)
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);
    let inner = block.inner(area);
    block.render(area, buf);

    let [msg_area, gauge_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);

    Paragraph::new(message)
        .style(THEME.text)
        .wrap(Wrap { trim: false })
        .render(msg_area, buf);

    let pct = (progress * 100.0) as u16;
    ratatui::widgets::Gauge::default()
        .gauge_style(THEME.status_running)
        .ratio(progress.clamp(0.0, 1.0))
        .label(format!("{pct}%"))
        .render(gauge_area, buf);
}

fn render_hf_search(
    area: Rect,
    buf: &mut Buffer,
    entries: &[HfSearchEntry],
    table_state: &mut TableState,
) {
    use ratatui::style::{Color, Style};

    let block = Block::default()
        .title(" HuggingFace Hub Search ")
        .title_style(THEME.block_title_focused)
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);

    if entries.is_empty() {
        Paragraph::new("\n  No results found.")
            .style(THEME.text_muted)
            .block(block)
            .render(area, buf);
        return;
    }

    let inner = block.inner(area);
    block.render(area, buf);

    let [table_area, help_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);

    let header = Row::new(vec![
        "Model",
        "Params",
        "Downloads",
        "Memory",
        "Fit",
        "tok/s",
    ])
    .style(THEME.table_header)
    .height(1);

    let rows: Vec<Row> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let base_style = if i % 2 == 0 {
                THEME.table_row
            } else {
                THEME.table_row_alt
            };
            let fit_style = match entry.fit_level {
                pmetal_hub::FitLevel::Fits => Style::default().fg(Color::Green),
                pmetal_hub::FitLevel::Tight => Style::default().fg(Color::Yellow),
                pmetal_hub::FitLevel::TooLarge => Style::default().fg(Color::Red),
            };
            Row::new(vec![
                Cell::new(entry.model_id.clone()),
                Cell::new(entry.params.clone()),
                Cell::new(entry.downloads.clone()),
                Cell::new(entry.memory.clone()),
                Cell::new(entry.fit_level.label()).style(fit_style),
                Cell::new(entry.estimated_tps.clone()),
            ])
            .style(base_style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Fill(1),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .row_highlight_style(THEME.table_selected)
    .highlight_spacing(HighlightSpacing::Always);

    StatefulWidget::render(table, table_area, buf, table_state);

    Line::from(vec![
        Span::styled("j/k", THEME.footer_key),
        Span::styled(" navigate  ", THEME.text_muted),
        Span::styled("Enter/d", THEME.footer_key),
        Span::styled(" download  ", THEME.text_muted),
        Span::styled("Esc", THEME.footer_key),
        Span::styled(" close", THEME.text_muted),
    ])
    .render(help_area, buf);
}

// --- Helpers ---

fn filtered_models<'a>(models: &'a [PickerEntry], search: &str) -> Vec<&'a PickerEntry> {
    if search.is_empty() {
        models.iter().collect()
    } else {
        let q = search.to_lowercase();
        models
            .iter()
            .filter(|m| m.id.to_lowercase().contains(&q))
            .collect()
    }
}

/// Render the contextual help overlay — shows global keybindings
/// alongside the bindings specific to `tab`.
fn render_help(area: Rect, buf: &mut Buffer, tab: Tab) {
    let block = Block::default()
        .title(format!(" Help — {tab} "))
        .title_style(THEME.block_title_focused)
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);
    let inner = block.inner(area);
    block.render(area, buf);

    let [global_area, tab_area, footer_area] = Layout::vertical([
        Constraint::Length(12),
        Constraint::Min(0),
        Constraint::Length(2),
    ])
    .areas(inner);

    // ── Global bindings (same on every tab) ────────────────────────────
    let global_lines = [
        ("?", "Toggle this help overlay"),
        ("Tab / Shift+Tab", "Switch tabs forward / backward"),
        ("Alt+1..9", "Jump directly to tab 1..9"),
        ("Ctrl+P", "Go to any tab (type to filter)"),
        ("Ctrl+1..9", "Jump directly to tab 1..9 (alternative)"),
        ("q / Ctrl+C", "Quit the TUI"),
        ("F2", "Toggle mouse capture (text select/copy)"),
        ("Esc", "Close modal / cancel edit"),
    ];
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(" Global", THEME.kv_key)));
    for (key, desc) in &global_lines {
        lines.push(Line::from(vec![
            Span::styled(format!("  {key:<18}"), THEME.footer_key),
            Span::styled(*desc, THEME.text),
        ]));
    }
    Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .render(global_area, buf);

    // ── Tab-specific bindings ──────────────────────────────────────────
    let tab_bindings: &[(&str, &str)] = tab_help(tab);
    let mut tab_lines: Vec<Line> = Vec::new();
    tab_lines.push(Line::from(Span::styled(
        format!(" {tab} tab"),
        THEME.kv_key,
    )));
    if tab_bindings.is_empty() {
        tab_lines.push(Line::from(Span::styled(
            "  (no tab-specific bindings)",
            THEME.text_muted,
        )));
    } else {
        for (key, desc) in tab_bindings {
            tab_lines.push(Line::from(vec![
                Span::styled(format!("  {key:<18}"), THEME.footer_key),
                Span::styled(*desc, THEME.text),
            ]));
        }
    }
    Paragraph::new(tab_lines)
        .wrap(Wrap { trim: false })
        .render(tab_area, buf);

    Paragraph::new(Line::from(Span::styled(
        " Press ? or Esc to close",
        THEME.text_muted,
    )))
    .render(footer_area, buf);
}

/// Return the help text for a specific tab. Each row is `(key, description)`.
fn tab_help(tab: Tab) -> &'static [(&'static str, &'static str)] {
    match tab {
        Tab::Dashboard => &[
            ("p", "Pause / resume metric updates"),
            ("r", "Reset metric buffers"),
        ],
        Tab::Device => &[],
        Tab::Models => &[
            ("j / k", "Navigate rows"),
            ("/", "Filter model list"),
            ("R", "Rescan model directories"),
            ("d", "Download a model by HF id"),
            ("a", "Add a custom model directory"),
            ("S", "Search HuggingFace Hub"),
            ("t", "Train with the selected model"),
            ("s", "Distill with the selected model as student"),
            ("i", "Infer with the selected model"),
            ("f", "Fuse a LoRA adapter into its base"),
        ],
        Tab::Datasets => &[
            ("j / k", "Navigate rows"),
            ("c", "Convert to pmetal JSONL"),
            ("a", "Add a custom dataset directory"),
            ("R", "Rescan dataset directories"),
        ],
        Tab::Training | Tab::Pretrain | Tab::Distillation | Tab::Grpo => &[
            ("j / k", "Navigate fields"),
            ("Enter", "Edit / cycle / pick"),
            ("S", "Start the job"),
            ("x", "Stop the running job"),
            ("L", "Override learning rate during training"),
        ],
        Tab::Inference => &[
            ("Enter", "Send the prompt"),
            ("Ctrl+P", "Pick a model"),
            ("Ctrl+S", "Focus the settings sidebar"),
            ("Ctrl+H", "Show / hide the settings sidebar"),
            ("F2", "Enter browse mode to yank text"),
            ("Esc", "Stop generation"),
        ],
        Tab::Jobs => &[
            ("j / k", "Navigate jobs"),
            ("J / K", "Scroll the selected job log"),
            ("R", "Rescan job output directories"),
        ],
        Tab::Serve | Tab::Quantize | Tab::Merge | Tab::Bench | Tab::Eval | Tab::Dflash => &[
            ("j / k", "Navigate fields"),
            ("Enter", "Edit / cycle / pick"),
            ("S", "Start the job"),
            ("x", "Cancel the running job"),
        ],
        Tab::EmbedTrain | Tab::Rlkd | Tab::Tokenize => &[
            ("j / k", "Navigate fields"),
            ("Enter", "Edit / cycle / pick"),
            ("S", "Start the job"),
            ("x", "Cancel the running job"),
        ],
        Tab::Ollama => &[
            ("j / k", "Navigate fields"),
            ("Enter", "Edit field value"),
            ("S", "Export the Modelfile"),
            ("x", "Cancel running export"),
        ],
    }
}

/// Compute a centered rectangle within `area` at the given percentage.
fn centered_rect(area: Rect, width_pct: u16, height_pct: u16) -> Rect {
    let [_, v_center, _] = Layout::vertical([
        Constraint::Percentage((100 - height_pct) / 2),
        Constraint::Percentage(height_pct),
        Constraint::Percentage((100 - height_pct) / 2),
    ])
    .areas(area);
    let [_, h_center, _] = Layout::horizontal([
        Constraint::Percentage((100 - width_pct) / 2),
        Constraint::Percentage(width_pct),
        Constraint::Percentage((100 - width_pct) / 2),
    ])
    .areas(v_center);
    h_center
}
