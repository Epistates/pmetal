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
    /// Dataset picker (list of local datasets).
    DatasetPicker {
        datasets: Vec<PickerEntry>,
        list_state: ListState,
    },
    /// Error display.
    Error {
        title: String,
        message: String,
    },
    /// Progress indicator for ongoing operations.
    Progress {
        title: String,
        message: String,
        progress: f64,
    },
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
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    Some(ModalAction::None)
                }
                KeyCode::Enter => {
                    if *selected {
                        Some(ModalAction::Confirmed)
                    } else {
                        Some(ModalAction::None)
                    }
                }
                _ => None,
            },

            Modal::TextInput {
                value, cursor, ..
            } => match key.code {
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
                    table_state.select(Some(0));
                    None
                }
                KeyCode::Backspace if *searching => {
                    search.pop();
                    None
                }
                KeyCode::Esc if *searching => {
                    *searching = false;
                    search.clear();
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

            Modal::DatasetPicker {
                datasets,
                list_state,
            } => match key.code {
                KeyCode::Esc => Some(ModalAction::None),
                KeyCode::Down | KeyCode::Char('j') => {
                    let count = datasets.len();
                    if count > 0 {
                        let i = list_state.selected().map_or(0, |i| (i + 1) % count);
                        list_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    let count = datasets.len();
                    if count > 0 {
                        let i = list_state
                            .selected()
                            .map_or(0, |i| (i + count - 1) % count);
                        list_state.select(Some(i));
                    }
                    None
                }
                KeyCode::Enter => {
                    if let Some(idx) = list_state.selected() {
                        if let Some(entry) = datasets.get(idx) {
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
            } => render_dataset_picker(popup_area, buf, datasets, list_state),
            Modal::Error { title, message } => render_error(popup_area, buf, title, message),
            Modal::Progress {
                title,
                message,
                progress,
            } => render_progress(popup_area, buf, title, message, *progress),
        }
    }

    fn width_pct(&self) -> u16 {
        match self {
            Modal::Confirm { .. } | Modal::Error { .. } | Modal::Progress { .. } => 50,
            Modal::TextInput { .. } => 60,
            Modal::ModelPicker { .. } | Modal::DatasetPicker { .. } => 70,
        }
    }

    fn height_pct(&self) -> u16 {
        match self {
            Modal::Confirm { .. } => 30,
            Modal::TextInput { .. } => 20,
            Modal::Error { .. } | Modal::Progress { .. } => 25,
            Modal::ModelPicker { .. } | Modal::DatasetPicker { .. } => 60,
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

fn render_dataset_picker(
    area: Rect,
    buf: &mut Buffer,
    datasets: &[PickerEntry],
    list_state: &mut ListState,
) {
    let block = Block::default()
        .title(" Select Dataset ")
        .title_style(THEME.block_title_focused)
        .borders(Borders::ALL)
        .border_style(THEME.block_focused);

    if datasets.is_empty() {
        Paragraph::new("\n  No datasets found in ./, ./data/, ./datasets/")
            .style(THEME.text_muted)
            .block(block)
            .render(area, buf);
        return;
    }

    let items: Vec<ListItem> = datasets
        .iter()
        .map(|ds| {
            ListItem::new(Line::from(vec![
                Span::styled(&ds.id, THEME.kv_value),
                Span::raw("  "),
                Span::styled(&ds.detail, THEME.text_dim),
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

    Line::from(Span::styled("Press Enter or Esc to close", THEME.text_muted))
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
