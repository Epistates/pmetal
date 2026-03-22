//! Editable form field widget.
//!
//! Supports text, number, integer, enum cycling, and toggle field types.
//! Used by the Training, Distillation, and GRPO tabs for parameter editing.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::text::{Line, Span};

use crate::tui::theme::THEME;

/// The kind of form field, which determines editing behavior.
#[derive(Debug, Clone)]
pub enum FieldKind {
    /// Free-form text input.
    Text,
    /// Floating-point number with optional bounds.
    Number { min: f64, max: f64 },
    /// Integer with optional bounds.
    Integer { min: i64, max: i64 },
    /// Cycle through a fixed set of options.
    Enum { options: Vec<String> },
    /// Toggle between Enabled/Disabled.
    Toggle,
    /// Opens a model picker modal (not inline-editable).
    ModelPicker,
    /// Opens a dataset picker modal (not inline-editable).
    DatasetPicker,
    /// Read-only display field.
    ReadOnly,
}

/// A form field with label, value, editing state, and validation.
#[derive(Debug, Clone)]
pub struct FormField {
    pub label: String,
    pub value: String,
    pub kind: FieldKind,
    pub section: String,
    pub editing: bool,
    edit_buffer: String,
    cursor: usize,
}

impl FormField {
    pub fn new(
        label: impl Into<String>,
        value: impl Into<String>,
        kind: FieldKind,
        section: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            kind,
            section: section.into(),
            editing: false,
            edit_buffer: String::new(),
            cursor: 0,
        }
    }

    /// Whether this field can be edited inline (vs. opening a modal).
    pub fn is_inline_editable(&self) -> bool {
        matches!(
            self.kind,
            FieldKind::Text | FieldKind::Number { .. } | FieldKind::Integer { .. }
        )
    }

    /// Whether this field opens a picker modal.
    pub fn is_picker(&self) -> bool {
        matches!(self.kind, FieldKind::ModelPicker | FieldKind::DatasetPicker)
    }

    /// Whether this field can be toggled/cycled with Enter.
    pub fn is_cycleable(&self) -> bool {
        matches!(self.kind, FieldKind::Toggle | FieldKind::Enum { .. })
    }

    /// Start inline editing.
    pub fn start_edit(&mut self) {
        if self.is_inline_editable() {
            self.editing = true;
            self.edit_buffer = self.value.clone();
            self.cursor = self.edit_buffer.len();
        }
    }

    /// Cancel inline editing, discarding changes.
    pub fn cancel_edit(&mut self) {
        self.editing = false;
        self.edit_buffer.clear();
        self.cursor = 0;
    }

    /// Confirm inline editing, validating and applying the new value.
    /// Returns `true` if the value was accepted.
    pub fn confirm_edit(&mut self) -> bool {
        if !self.editing {
            return false;
        }
        let new_val = self.edit_buffer.trim().to_string();
        if self.validate(&new_val) {
            self.value = new_val;
            self.editing = false;
            self.edit_buffer.clear();
            self.cursor = 0;
            true
        } else {
            false
        }
    }

    /// Cycle to the next value (for Toggle/Enum fields).
    pub fn cycle(&mut self) {
        match &self.kind {
            FieldKind::Toggle => {
                self.value = if self.value == "Enabled" {
                    "Disabled".to_string()
                } else {
                    "Enabled".to_string()
                };
            }
            FieldKind::Enum { options } => {
                if let Some(idx) = options.iter().position(|o| o == &self.value) {
                    let next = (idx + 1) % options.len();
                    self.value = options[next].clone();
                } else if let Some(first) = options.first() {
                    self.value = first.clone();
                }
            }
            _ => {}
        }
    }

    /// Handle a key event while this field is in editing mode.
    pub fn handle_edit_key(&mut self, key: KeyEvent) {
        if !self.editing {
            return;
        }
        match key.code {
            KeyCode::Char(c) => {
                // For numeric fields, only allow numeric characters
                match &self.kind {
                    FieldKind::Number { .. } => {
                        if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' {
                            self.edit_buffer.insert(self.cursor, c);
                            self.cursor += 1;
                        }
                    }
                    FieldKind::Integer { .. } => {
                        if c.is_ascii_digit() || c == '-' {
                            self.edit_buffer.insert(self.cursor, c);
                            self.cursor += 1;
                        }
                    }
                    _ => {
                        self.edit_buffer.insert(self.cursor, c);
                        self.cursor += c.len_utf8();
                    }
                }
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let prev = self.edit_buffer[..self.cursor]
                        .chars()
                        .last()
                        .map(|c| c.len_utf8())
                        .unwrap_or(0);
                    self.cursor -= prev;
                    self.edit_buffer.remove(self.cursor);
                }
            }
            KeyCode::Left => {
                if self.cursor > 0 {
                    let prev = self.edit_buffer[..self.cursor]
                        .chars()
                        .last()
                        .map(|c| c.len_utf8())
                        .unwrap_or(0);
                    self.cursor -= prev;
                }
            }
            KeyCode::Right => {
                if self.cursor < self.edit_buffer.len() {
                    let next = self.edit_buffer[self.cursor..]
                        .chars()
                        .next()
                        .map(|c| c.len_utf8())
                        .unwrap_or(0);
                    self.cursor += next;
                }
            }
            _ => {}
        }
    }

    /// Validate a potential new value against the field's constraints.
    fn validate(&self, val: &str) -> bool {
        match &self.kind {
            FieldKind::Text => !val.is_empty(),
            FieldKind::Number { min, max } => {
                if let Ok(n) = val.parse::<f64>() {
                    n >= *min && n <= *max && n.is_finite()
                } else {
                    false
                }
            }
            FieldKind::Integer { min, max } => {
                if let Ok(n) = val.parse::<i64>() {
                    n >= *min && n <= *max
                } else {
                    false
                }
            }
            FieldKind::Enum { options } => options.contains(&val.to_string()),
            FieldKind::Toggle => val == "Enabled" || val == "Disabled",
            FieldKind::ModelPicker | FieldKind::DatasetPicker | FieldKind::ReadOnly => true,
        }
    }

    /// Render as a single line in a form list.
    pub fn render_line(&self, key_width: usize, selected: bool) -> Line<'_> {
        let padded_key = format!("  {:>width$}: ", self.label, width = key_width);

        let (value_text, value_style) = if self.editing {
            let display = if self.edit_buffer.is_empty() {
                "_".to_string()
            } else {
                // Show cursor position with a block cursor
                let mut s = self.edit_buffer.clone();
                if self.cursor < s.len() {
                    s.insert(self.cursor, '|');
                } else {
                    s.push('|');
                }
                s
            };
            (display, THEME.text_warning) // Amber for editing
        } else if self.value == "(not selected)" || self.value == "(none)" {
            (self.value.clone(), THEME.text_muted)
        } else {
            (self.value.clone(), THEME.kv_value)
        };

        // Add a hint for picker/toggle fields
        let hint = if selected && !self.editing {
            match &self.kind {
                FieldKind::ModelPicker | FieldKind::DatasetPicker => " [Enter: pick]",
                FieldKind::Toggle | FieldKind::Enum { .. } => " [Enter: cycle]",
                FieldKind::Text | FieldKind::Number { .. } | FieldKind::Integer { .. } => {
                    " [Enter: edit]"
                }
                FieldKind::ReadOnly => "",
            }
        } else {
            ""
        };

        Line::from(vec![
            Span::styled(padded_key, THEME.kv_key),
            Span::styled(value_text, value_style),
            Span::styled(hint, THEME.text_muted),
        ])
    }
}
