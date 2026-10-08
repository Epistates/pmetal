//! Shared state and behaviour for form-driven tabs.
//!
//! The Training, Distillation, GRPO, Serve, Quantize, and Bench tabs all
//! share the same skeleton: a vertical list of grouped [`FormField`]s with
//! cursor navigation, inline edit dispatch, picker actions, and optional
//! section hiding. `FormTabState` owns that skeleton so each tab only has
//! to declare its fields + render any tab-specific panels (status, logs,
//! trial tables, chat history, …).
//!
//! See `tabs/serve.rs`, `tabs/quantize.rs`, and `tabs/bench.rs` for
//! reference adopters.

use crossterm::event::KeyEvent;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Widget};

use crate::tui::theme::THEME;
use crate::tui::widgets::{FieldKind, FormField};

/// Result of dispatching `Enter` to the current field.
///
/// Tabs return this from `FormTabState::handle_enter` so their app-level
/// key handlers know whether to open a picker modal, start inline edit,
/// or ignore the key (cycle toggles / enums are handled internally).
///
/// Picker variants carry the label of the triggering field so tabs with
/// multiple pickers (e.g. distillation's Teacher vs Student) can route
/// the selection back to the right setter without inventing per-tab
/// action enums.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormAction {
    /// A `ModelPicker` field was activated — open the model picker modal.
    OpenModelPicker { field_label: String },
    /// A `DatasetPicker` field was activated — open the dataset picker.
    OpenDatasetPicker { field_label: String },
    /// An inline-editable field entered edit mode — the app should
    /// route subsequent keys to `FormTabState::handle_edit_key` until the
    /// user confirms or cancels.
    StartEdit,
}

/// Owned state shared by every form-driven tab.
///
/// Holds the field vector, the ratatui `ListState` (so selection and
/// scroll survive renders), and the cursor into `fields`. All navigation,
/// editing, and rendering helpers route through this struct so individual
/// tabs stay focused on their domain logic.
#[derive(Debug)]
pub struct FormTabState {
    pub fields: Vec<FormField>,
    pub list_state: ListState,
    field_idx: usize,
}

impl FormTabState {
    /// Build a `FormTabState` whose fields are derived from a spec's
    /// [`pmetal_core::JobFields::field_descriptors`] using default values.
    ///
    /// This is the spec-driven constructor: each descriptor becomes a
    /// `FormField` via [`FormField::from_descriptor`], so the form automatically
    /// stays in sync with the spec's field set and defaults.
    ///
    /// Call [`FormTabState::new`] directly when you need hand-crafted fields
    /// (e.g. read-only auto-detected values like "Architecture").
    pub fn from_spec_default<S>() -> Self
    where
        S: pmetal_core::JobFields + Default,
    {
        let fields = S::field_descriptors()
            .iter()
            .map(|d| FormField::from_descriptor(d, None))
            .collect();
        Self::new(fields)
    }

    /// Create a form-tab state from a pre-built field list.
    ///
    /// Selects the second visible row by default (index 1) to put the
    /// cursor on the first editable entry, skipping the leading section
    /// divider that the renderer always emits.
    pub fn new(fields: Vec<FormField>) -> Self {
        Self {
            fields,
            list_state: ListState::default().with_selected(Some(1)),
            field_idx: 0,
        }
    }

    // ── Field lookup / mutation ─────────────────────────────────────────

    /// Lookup a field's current string value by label. Returns `""` when
    /// the label isn't found, matching the historical per-tab helper.
    pub fn value(&self, label: &str) -> String {
        self.fields
            .iter()
            .find(|f| f.label == label)
            .map(|f| f.value.clone())
            .unwrap_or_default()
    }

    /// Overwrite a field's value by label. No-op if the label is missing.
    pub fn set_value(&mut self, label: &str, value: impl Into<String>) {
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == label) {
            f.value = value.into();
        }
    }

    pub fn field_idx(&self) -> usize {
        self.field_idx
    }

    // ── Edit state ──────────────────────────────────────────────────────

    pub fn is_editing(&self) -> bool {
        self.fields.get(self.field_idx).is_some_and(|f| f.editing)
    }

    pub fn handle_edit_key(&mut self, key: KeyEvent) {
        if let Some(field) = self.fields.get_mut(self.field_idx) {
            field.handle_edit_key(key);
        }
    }

    pub fn confirm_edit(&mut self) {
        if let Some(field) = self.fields.get_mut(self.field_idx) {
            field.confirm_edit();
        }
    }

    pub fn cancel_edit(&mut self) {
        if let Some(field) = self.fields.get_mut(self.field_idx) {
            field.cancel_edit();
        }
    }

    /// Dispatch `Enter` on the current field. Returns a `FormAction` when
    /// the caller needs to react (open a modal, start an inline edit).
    /// Cycles Toggle/Enum fields internally and returns `None`.
    pub fn handle_enter(&mut self) -> Option<FormAction> {
        let field = self.fields.get_mut(self.field_idx)?;

        if field.is_picker() {
            let label = field.label.clone();
            return match field.kind {
                FieldKind::ModelPicker => Some(FormAction::OpenModelPicker { field_label: label }),
                FieldKind::DatasetPicker => {
                    Some(FormAction::OpenDatasetPicker { field_label: label })
                }
                _ => None,
            };
        }
        if field.is_cycleable() {
            field.cycle();
            return None;
        }
        if field.is_inline_editable() {
            field.start_edit();
            return Some(FormAction::StartEdit);
        }
        None
    }

    // ── Navigation ──────────────────────────────────────────────────────

    /// Advance the cursor to the next visible field.
    ///
    /// `visible` receives each candidate field and returns `true` when
    /// the cursor is allowed to land on it. Callers that never hide
    /// anything can pass `|_| true`; section-gated tabs typically match
    /// on `field.section`; training skips `ReadOnly` kinds. Section
    /// dividers for invisible sections are not rendered either — nav
    /// and rendering share one predicate so the highlight always tracks.
    pub fn next_param(&mut self, visible: impl Fn(&FormField) -> bool) {
        self.step_param(1, visible);
    }

    pub fn prev_param(&mut self, visible: impl Fn(&FormField) -> bool) {
        self.step_param(-1, visible);
    }

    fn step_param(&mut self, step: i32, visible: impl Fn(&FormField) -> bool) {
        let count = self.fields.len();
        if count == 0 {
            return;
        }
        let mut idx = self.field_idx;
        // Walk up to `count` times to find the next visible field. This
        // guards against infinite loops when every field is hidden.
        for _ in 0..count {
            idx = if step >= 0 {
                (idx + 1) % count
            } else {
                (idx + count - 1) % count
            };
            if visible(&self.fields[idx]) {
                self.field_idx = idx;
                self.sync_list_selection(&visible);
                return;
            }
        }
    }

    fn sync_list_selection(&mut self, visible: &impl Fn(&FormField) -> bool) {
        let flat = self.flat_index_for_field(self.field_idx, visible);
        self.list_state.select(Some(flat));
    }

    /// Compute the rendered-row index (with section-header offsets and
    /// visibility filtering applied) for the field at `field_idx`. Stays
    /// in sync with `render_list` so the highlight tracks the cursor.
    fn flat_index_for_field(
        &self,
        field_idx: usize,
        visible: &impl Fn(&FormField) -> bool,
    ) -> usize {
        let mut flat = 0;
        let mut current_section: Option<&str> = None;
        for (i, field) in self.fields.iter().enumerate() {
            if !visible(field) {
                continue;
            }
            if current_section != Some(&field.section) {
                current_section = Some(&field.section);
                flat += 1;
            }
            if i == field_idx {
                return flat;
            }
            flat += 1;
        }
        flat
    }

    // ── Rendering ───────────────────────────────────────────────────────

    /// Render the form as a bordered list with `--- Section ---` dividers
    /// and the current row highlighted. `visible` is the same field-level
    /// predicate used by `next_param` / `prev_param`.
    pub fn render_list(
        &mut self,
        area: Rect,
        buf: &mut Buffer,
        title: &str,
        visible: impl Fn(&FormField) -> bool,
    ) {
        let block = Block::default()
            .title(format!(" {title} "))
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        let key_width = self
            .fields
            .iter()
            .map(|f| f.label.len())
            .max()
            .unwrap_or(10);

        let mut current_section: Option<&str> = None;
        let mut items: Vec<ListItem> = Vec::new();

        for (i, field) in self.fields.iter().enumerate() {
            if !visible(field) {
                continue;
            }
            if current_section != Some(&field.section) {
                current_section = Some(&field.section);
                items.push(ListItem::new(Line::from(Span::styled(
                    format!("  --- {} ---", field.section),
                    THEME.text_muted,
                ))));
            }
            let selected = i == self.field_idx;
            items.push(ListItem::new(field.render_line(key_width, selected)));
        }

        let list = List::new(items)
            .block(block)
            .highlight_style(THEME.table_selected);

        ratatui::widgets::StatefulWidget::render(list, area, buf, &mut self.list_state);
    }
}

// ════════════════════════════════════════════════════════════════════════
// JobLog — shared ring buffer + renderer for long-running jobs
// ════════════════════════════════════════════════════════════════════════

/// Capped ring buffer of recent stdout/stderr lines from a subprocess.
///
/// Every tab that spawns a `pmetal` child and displays its output uses
/// the same pattern: push each line, cap total retained lines, and
/// render the tail into a bordered block. `JobLog` is that pattern.
#[derive(Debug)]
pub struct JobLog {
    lines: Vec<String>,
    cap: usize,
}

impl JobLog {
    /// Cap the log at `cap` lines. Older lines are dropped FIFO when the
    /// limit is exceeded.
    pub fn new(cap: usize) -> Self {
        Self {
            lines: Vec::new(),
            cap,
        }
    }

    /// 500-line default — matches the historical per-tab hardcoded cap
    /// and is enough for several minutes of `pmetal` output without
    /// letting a runaway log blow up resident memory.
    pub fn with_default_cap() -> Self {
        Self::new(500)
    }

    pub fn push(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
        if self.lines.len() > self.cap {
            let drop = self.lines.len() - self.cap;
            self.lines.drain(..drop);
        }
    }

    pub fn clear(&mut self) {
        self.lines.clear();
    }

    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Render the tail of the buffer that fits into `area`, with a two-
    /// space indent and the given title. Uses the default paragraph wrap
    /// so long lines don't disappear off the right edge.
    pub fn render(&self, area: Rect, buf: &mut Buffer, title: &str) {
        use ratatui::widgets::{Paragraph, Wrap};

        let block = Block::default()
            .title(format!(" {title} "))
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let visible = inner.height as usize;
        let start = self.lines.len().saturating_sub(visible);
        let rendered: Vec<Line> = self.lines[start..]
            .iter()
            .map(|l| Line::from(Span::styled(format!("  {l}"), THEME.text)))
            .collect();

        Paragraph::new(rendered)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

impl Default for JobLog {
    fn default() -> Self {
        Self::with_default_cap()
    }
}

// ════════════════════════════════════════════════════════════════════════
// StatusBadge — consistent idle/running/completed/failed rendering
// ════════════════════════════════════════════════════════════════════════

/// Visual tone for a status line. The badge renderer maps each tone to a
/// distinct glyph + colour so every tab reports status the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusTone {
    Idle,
    Running,
    Completed,
    Failed,
}

impl StatusTone {
    fn glyph(self) -> &'static str {
        match self {
            Self::Idle => "·",
            Self::Running => "●",
            Self::Completed => "✓",
            Self::Failed => "✗",
        }
    }

    fn style(self) -> ratatui::style::Style {
        use ratatui::style::{Color, Modifier, Style};
        match self {
            Self::Idle => Style::default().fg(Color::Rgb(140, 150, 170)),
            Self::Running => Style::default()
                .fg(Color::Rgb(56, 189, 248))
                .add_modifier(Modifier::BOLD),
            Self::Completed => Style::default()
                .fg(Color::Rgb(74, 222, 128))
                .add_modifier(Modifier::BOLD),
            Self::Failed => Style::default()
                .fg(Color::Rgb(248, 113, 113))
                .add_modifier(Modifier::BOLD),
        }
    }
}

/// Build a single-line status badge: `  ● Running — binding to port 8080`.
///
/// The glyph and colour are driven by `tone`; `label` is shown in the
/// same accent, and the optional `detail` trails after an em-dash in
/// muted text. Used as the first line of every tab's status panel.
pub fn status_line<'a>(tone: StatusTone, label: &'a str, detail: Option<&'a str>) -> Line<'a> {
    let style = tone.style();
    let mut spans = vec![
        Span::raw("  "),
        Span::styled(tone.glyph(), style),
        Span::raw(" "),
        Span::styled(label, style),
    ];
    if let Some(detail) = detail {
        spans.push(Span::styled(" — ", THEME.text_muted));
        spans.push(Span::styled(detail, THEME.text_muted));
    }
    Line::from(spans)
}
