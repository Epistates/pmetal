//! Editable form field widget.
//!
//! A field is one row of a job's form: a label, a value, and how the value is
//! entered (typed, cycled through options, toggled, picked from a list). Fields
//! are built from a job spec's descriptors, so what a field accepts and what it
//! starts as are the spec's, not a copy of them.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::text::{Line, Span};

use crate::tui::theme::THEME;

/// What the TUI shows for a toggle that is on / off.
pub const ENABLED: &str = "Enabled";
/// See [`ENABLED`].
pub const DISABLED: &str = "Disabled";

/// The kind of form field, which determines editing behavior.
#[derive(Debug, Clone)]
pub enum FieldKind {
    /// Free-form text input.
    Text,
    /// Floating-point number with bounds.
    Number { min: f64, max: f64 },
    /// Integer with bounds.
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
    /// Filesystem path (file or directory). Edited inline like `Text`.
    Path,
}

/// A form field with label, value, editing state, and validation.
#[derive(Debug, Clone)]
pub struct FormField {
    pub label: String,
    pub value: String,
    pub kind: FieldKind,
    pub section: String,
    pub editing: bool,
    /// The spec field this row edits, if it edits one (a read-only readout
    /// does not).
    pub name: Option<&'static str>,
    /// What the spec says about the field.
    pub help: Option<&'static str>,
    /// The field may not be left empty.
    pub required: bool,
    /// An empty entry means "not set" (an optional field or a list).
    pub optional: bool,
    /// Why the last attempt to confirm an edit was refused.
    pub error: Option<String>,
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
            name: None,
            help: None,
            required: false,
            optional: false,
            error: None,
            edit_buffer: String::new(),
            cursor: 0,
        }
    }

    /// Whether this field can be edited inline (vs. opening a modal).
    pub fn is_inline_editable(&self) -> bool {
        matches!(
            self.kind,
            FieldKind::Text
                | FieldKind::Number { .. }
                | FieldKind::Integer { .. }
                | FieldKind::Path
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
            self.error = None;
            self.edit_buffer = self.value.clone();
            self.cursor = self.edit_buffer.len();
        }
    }

    /// Cancel inline editing, discarding changes.
    pub fn cancel_edit(&mut self) {
        self.editing = false;
        self.error = None;
        self.edit_buffer.clear();
        self.cursor = 0;
    }

    /// Confirm inline editing, validating and applying the new value.
    /// Returns `true` if the value was accepted; if not the field stays in
    /// editing mode and `error` says why.
    pub fn confirm_edit(&mut self) -> bool {
        if !self.editing {
            return false;
        }
        let new_val = self.edit_buffer.trim().to_string();
        match self.validate(&new_val) {
            Ok(()) => {
                self.value = new_val;
                self.editing = false;
                self.error = None;
                self.edit_buffer.clear();
                self.cursor = 0;
                true
            }
            Err(message) => {
                self.error = Some(message);
                false
            }
        }
    }

    /// Set the value from outside the field (a picker, a derived default).
    pub fn set(&mut self, value: impl Into<String>) {
        self.value = value.into();
        self.error = None;
    }

    /// Cycle to the next value (for Toggle/Enum fields).
    pub fn cycle(&mut self) {
        self.cycle_by(1);
    }

    /// Cycle to the previous value (for Toggle/Enum fields).
    pub fn cycle_back(&mut self) {
        self.cycle_by(-1);
    }

    fn cycle_by(&mut self, step: isize) {
        match &self.kind {
            FieldKind::Toggle => {
                self.value = if self.value == ENABLED {
                    DISABLED.to_string()
                } else {
                    ENABLED.to_string()
                };
            }
            FieldKind::Enum { options } => {
                let count = options.len() as isize;
                if count == 0 {
                    return;
                }
                self.value = match options.iter().position(|o| o == &self.value) {
                    Some(idx) => options[(idx as isize + step).rem_euclid(count) as usize].clone(),
                    None => options[0].clone(),
                };
            }
            _ => {}
        }
    }

    /// Insert pasted text at the cursor of a field being edited. Newlines
    /// become spaces, and characters a numeric field cannot hold are dropped.
    pub fn paste(&mut self, text: &str) {
        for c in text.chars() {
            self.insert(if c == '\n' || c == '\r' { ' ' } else { c });
        }
    }

    fn insert(&mut self, c: char) {
        let allowed = match &self.kind {
            FieldKind::Number { .. } => {
                c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')
            }
            FieldKind::Integer { .. } => c.is_ascii_digit() || c == '-',
            _ => !c.is_control(),
        };
        if allowed {
            self.edit_buffer.insert(self.cursor, c);
            self.cursor += c.len_utf8();
        }
    }

    /// Handle a key event while this field is in editing mode.
    pub fn handle_edit_key(&mut self, key: KeyEvent) {
        if !self.editing {
            return;
        }
        // Whatever was wrong with the last attempt is stale once the text moves.
        let text_changes = |code: KeyCode| {
            matches!(
                code,
                KeyCode::Char(_) | KeyCode::Backspace | KeyCode::Delete
            )
        };
        if text_changes(key.code) {
            self.error = None;
        }
        match key.code {
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.edit_buffer.clear();
                self.cursor = 0;
            }
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor = 0;
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor = self.edit_buffer.len();
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert(c);
            }
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor = self.prev_boundary();
                self.edit_buffer.remove(self.cursor);
            }
            KeyCode::Delete if self.cursor < self.edit_buffer.len() => {
                self.edit_buffer.remove(self.cursor);
            }
            KeyCode::Left if self.cursor > 0 => self.cursor = self.prev_boundary(),
            KeyCode::Right if self.cursor < self.edit_buffer.len() => {
                self.cursor = self.next_boundary();
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.edit_buffer.len(),
            _ => {}
        }
    }

    fn prev_boundary(&self) -> usize {
        self.edit_buffer[..self.cursor]
            .char_indices()
            .next_back()
            .map_or(0, |(i, _)| i)
    }

    fn next_boundary(&self) -> usize {
        self.edit_buffer[self.cursor..]
            .chars()
            .next()
            .map_or(self.cursor, |c| self.cursor + c.len_utf8())
    }

    /// Whether `val` is a value this field accepts, or why not (naming the
    /// field).
    pub fn validate(&self, val: &str) -> Result<(), String> {
        let label = &self.label;
        if val.is_empty() {
            return match &self.kind {
                FieldKind::Number { .. } | FieldKind::Integer { .. } if !self.optional => {
                    Err(format!("{label} needs a number"))
                }
                _ if self.required => Err(format!("{label} is required")),
                _ => Ok(()),
            };
        }
        match &self.kind {
            FieldKind::Number { min, max } => {
                let n: f64 = val
                    .parse()
                    .map_err(|_| format!("{label}: '{val}' is not a number"))?;
                if !n.is_finite() {
                    return Err(format!("{label} must be a finite number"));
                }
                if n < *min || n > *max {
                    return Err(format!(
                        "{label} must be between {} and {}",
                        format_bound(*min),
                        format_bound(*max)
                    ));
                }
                Ok(())
            }
            FieldKind::Integer { min, max } => {
                let n: i64 = val
                    .parse()
                    .map_err(|_| format!("{label}: '{val}' is not a whole number"))?;
                if n < *min || n > *max {
                    return Err(format!("{label} must be between {min} and {max}"));
                }
                Ok(())
            }
            FieldKind::Enum { options } => {
                if options.iter().any(|o| o == val) {
                    Ok(())
                } else {
                    Err(format!("{label} must be one of: {}", options.join(", ")))
                }
            }
            FieldKind::Toggle => {
                if val == ENABLED || val == DISABLED {
                    Ok(())
                } else {
                    Err(format!("{label} must be {ENABLED} or {DISABLED}"))
                }
            }
            FieldKind::Text
            | FieldKind::Path
            | FieldKind::ModelPicker
            | FieldKind::DatasetPicker
            | FieldKind::ReadOnly => Ok(()),
        }
    }

    /// Build a `FormField` from a [`pmetal_core::FieldDescriptor`], showing
    /// `text` (what the spec's default or the user's last value renders as).
    pub fn from_descriptor(desc: &pmetal_core::FieldDescriptor, text: &str) -> FormField {
        let kind = core_kind_to_tui(&desc.kind);
        // Toggle fields store "Enabled"/"Disabled"; the spec's text is "true"/"false".
        let value = if matches!(desc.kind, pmetal_core::FieldKind::Toggle) {
            match text {
                "true" => ENABLED.to_string(),
                "false" => DISABLED.to_string(),
                other => other.to_string(),
            }
        } else {
            text.to_string()
        };
        let mut field = FormField::new(desc.label, value, kind, desc.group);
        field.name = Some(desc.name);
        field.help = desc.help;
        field.required = desc.required;
        field.optional = desc.shape != pmetal_core::FieldShape::Scalar;
        field
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
        } else if self.value.is_empty() && self.is_picker() {
            ("(not selected)".to_string(), THEME.text_muted)
        } else if self.value.is_empty() && self.required {
            ("(required)".to_string(), THEME.text_warning)
        } else if self.value.is_empty() {
            ("(not set)".to_string(), THEME.text_muted)
        } else {
            (self.value.clone(), THEME.kv_value)
        };

        // A hint for what Enter does, or the reason the edit was refused.
        let (hint, hint_style) = if let Some(error) = &self.error {
            (format!("  {error}"), THEME.text_error)
        } else if self.editing {
            ("  Enter: apply  Esc: cancel".to_string(), THEME.text_muted)
        } else if selected {
            let hint = match &self.kind {
                FieldKind::ModelPicker | FieldKind::DatasetPicker => " [Enter: pick]",
                FieldKind::Toggle | FieldKind::Enum { .. } => " [Enter: cycle]",
                FieldKind::Text
                | FieldKind::Number { .. }
                | FieldKind::Integer { .. }
                | FieldKind::Path => " [Enter: edit]",
                FieldKind::ReadOnly => "",
            };
            (hint.to_string(), THEME.text_muted)
        } else {
            (String::new(), THEME.text_muted)
        };

        Line::from(vec![
            Span::styled(padded_key, THEME.kv_key),
            Span::styled(value_text, value_style),
            Span::styled(hint, hint_style),
        ])
    }
}

/// A bound as a person writes it: `1e-8`, `0.5`, `1000`.
fn format_bound(bound: f64) -> String {
    let abs = bound.abs();
    if abs != 0.0 && !(1e-3..1e6).contains(&abs) {
        format!("{bound:e}")
    } else {
        format!("{bound}")
    }
}

/// Map a `pmetal_core::FieldKind` to its local TUI equivalent.
///
/// The two enums mirror each other 1:1 by design; this function is the
/// only place the mapping lives so any future divergence is caught in one
/// spot.
fn core_kind_to_tui(kind: &pmetal_core::FieldKind) -> FieldKind {
    match kind {
        pmetal_core::FieldKind::Text => FieldKind::Text,
        pmetal_core::FieldKind::Number { min, max } => FieldKind::Number {
            min: *min,
            max: *max,
        },
        pmetal_core::FieldKind::Integer { min, max } => FieldKind::Integer {
            min: *min,
            max: *max,
        },
        pmetal_core::FieldKind::Enum { options } => FieldKind::Enum {
            options: options.iter().map(|s| s.to_string()).collect(),
        },
        pmetal_core::FieldKind::Toggle => FieldKind::Toggle,
        pmetal_core::FieldKind::ModelPicker => FieldKind::ModelPicker,
        pmetal_core::FieldKind::DatasetPicker => FieldKind::DatasetPicker,
        pmetal_core::FieldKind::ReadOnly => FieldKind::ReadOnly,
        pmetal_core::FieldKind::Path => FieldKind::Path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_core::JobFields;
    use pmetal_core::jobs::{InferSpec, TrainSpec};

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn field_of<S: JobFields>(name: &str, text: &str) -> FormField {
        let d = S::field_descriptors()
            .iter()
            .find(|d| d.name == name)
            .unwrap();
        FormField::from_descriptor(d, text)
    }

    fn type_into(field: &mut FormField, text: &str) {
        field.start_edit();
        for _ in 0..field.value.len() {
            field.handle_edit_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        }
        for c in text.chars() {
            field.handle_edit_key(key(c));
        }
    }

    #[test]
    fn a_field_carries_what_its_descriptor_says() {
        let model = field_of::<TrainSpec>("model", "");
        assert!(matches!(model.kind, FieldKind::ModelPicker));
        assert_eq!(
            (model.label.as_str(), model.section.as_str()),
            ("Model", "Model")
        );
        assert_eq!(model.name, Some("model"));
        assert!(model.required);

        let lr = field_of::<TrainSpec>("learning_rate", "0.0002");
        let FieldKind::Number { min, max } = lr.kind else {
            panic!("a number");
        };
        assert!((min - 1e-8).abs() < 1e-12 && (max - 1.0).abs() < 1e-9);

        let batch = field_of::<TrainSpec>("batch_size", "4");
        assert!(matches!(batch.kind, FieldKind::Integer { min: 1, .. }));

        let quant = field_of::<TrainSpec>("quantization", "none");
        let FieldKind::Enum { options } = &quant.kind else {
            panic!("an enum");
        };
        assert!(options.contains(&"none".to_string()));

        let cce = field_of::<TrainSpec>("cut_cross_entropy", "false");
        assert_eq!(cce.value, DISABLED);
        assert_eq!(
            field_of::<TrainSpec>("cut_cross_entropy", "true").value,
            ENABLED
        );
        assert!(matches!(
            field_of::<TrainSpec>("output_dir", "./o").kind,
            FieldKind::Path
        ));
    }

    #[test]
    fn an_optional_field_may_be_cleared_and_a_required_one_may_not() {
        let mut temperature = field_of::<InferSpec>("temperature", "0.7");
        assert!(temperature.optional);
        type_into(&mut temperature, "");
        assert!(temperature.confirm_edit());
        assert_eq!(temperature.value, "");

        let mut model = field_of::<TrainSpec>("output_dir", "./o");
        model.required = true;
        type_into(&mut model, "");
        assert!(!model.confirm_edit());
        assert_eq!(model.error.as_deref(), Some("Output Dir is required"));
        assert!(model.editing, "a refused edit stays open");
    }

    #[test]
    fn a_refused_number_says_which_field_and_why() {
        let mut lr = field_of::<TrainSpec>("learning_rate", "0.0002");
        type_into(&mut lr, "5");
        assert!(!lr.confirm_edit());
        let message = lr.error.clone().unwrap();
        assert!(
            message.starts_with("Learning Rate must be between 1e-8 and 1"),
            "{message}"
        );
        // The text changing clears the stale complaint.
        lr.handle_edit_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(lr.error.is_none());
        type_into(&mut lr, "1e-5");
        assert!(lr.confirm_edit());
        assert_eq!(lr.value, "1e-5");

        let mut batch = field_of::<TrainSpec>("batch_size", "4");
        type_into(&mut batch, "0");
        assert!(!batch.confirm_edit());
        assert!(
            batch
                .error
                .as_deref()
                .unwrap()
                .starts_with("Batch Size must be between 1 and")
        );
        type_into(&mut batch, "");
        assert!(!batch.confirm_edit());
        assert_eq!(batch.error.as_deref(), Some("Batch Size needs a number"));
    }

    #[test]
    fn a_numeric_field_takes_only_numeric_keys() {
        let mut batch = field_of::<TrainSpec>("batch_size", "4");
        type_into(&mut batch, "1x2.5");
        assert!(batch.confirm_edit());
        assert_eq!(batch.value, "125");
        let mut lr = field_of::<TrainSpec>("learning_rate", "0.0002");
        type_into(&mut lr, "1e-5z");
        assert!(lr.confirm_edit());
        assert_eq!(lr.value, "1e-5");
    }

    #[test]
    fn editing_handles_multibyte_text_and_cursor_keys() {
        let mut out = field_of::<TrainSpec>("output_dir", "");
        out.start_edit();
        for c in "dé€x".chars() {
            out.handle_edit_key(key(c));
        }
        out.handle_edit_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        out.handle_edit_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        out.handle_edit_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        out.handle_edit_key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        assert!(out.confirm_edit());
        assert_eq!(out.value, "éx");
        // Ctrl+U clears the line.
        out.start_edit();
        out.handle_edit_key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        out.handle_edit_key(key('z'));
        assert!(out.confirm_edit());
        assert_eq!(out.value, "z");
    }

    #[test]
    fn pasted_text_lands_in_the_field_without_its_newlines() {
        let mut out = field_of::<TrainSpec>("output_dir", "");
        out.start_edit();
        out.paste("/data/my run\n");
        assert!(out.confirm_edit());
        assert_eq!(out.value, "/data/my run");
        let mut batch = field_of::<TrainSpec>("batch_size", "4");
        batch.start_edit();
        batch.paste("12abc");
        assert!(batch.confirm_edit());
        assert_eq!(batch.value, "412");
    }

    #[test]
    fn enums_and_toggles_cycle_both_ways() {
        let mut quant = field_of::<TrainSpec>("quantization", "none");
        let FieldKind::Enum { options } = quant.kind.clone() else {
            panic!("an enum");
        };
        quant.cycle();
        assert_eq!(quant.value, options[1]);
        quant.cycle_back();
        assert_eq!(quant.value, options[0]);
        quant.cycle_back();
        assert_eq!(quant.value, *options.last().unwrap());
        let mut toggle = field_of::<TrainSpec>("cut_cross_entropy", "false");
        toggle.cycle();
        assert_eq!(toggle.value, ENABLED);
        toggle.cycle_back();
        assert_eq!(toggle.value, DISABLED);
    }

    #[test]
    fn a_row_says_what_enter_does_and_why_an_edit_was_refused() {
        let text = |line: Line<'_>| {
            line.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        };
        let mut batch = field_of::<TrainSpec>("batch_size", "4");
        assert!(text(batch.render_line(12, true)).contains("[Enter: edit]"));
        assert!(!text(batch.render_line(12, false)).contains("Enter"));
        type_into(&mut batch, "0");
        batch.confirm_edit();
        assert!(text(batch.render_line(12, true)).contains("Batch Size must be between 1 and"));
        let model = field_of::<TrainSpec>("model", "");
        assert!(text(model.render_line(12, false)).contains("(not selected)"));
    }
}
