//! A job's form: fields built from its spec, and the spec built back from them.
//!
//! Every tab that launches a `pmetal` subcommand shows the same thing: a
//! vertical list of grouped [`FormField`]s with cursor navigation, inline edit,
//! pickers and section hiding. [`FormTabState`] is that, and it is the only
//! place where a form meets its spec:
//!
//! - [`for_spec`](FormTabState::for_spec) makes a field of every descriptor of
//!   the spec, starting from the text of `S::default()`, so a form never holds
//!   a default or a flag name of its own;
//! - [`to_spec`](FormTabState::to_spec) reads the text back through
//!   [`pmetal_core::spec_from_text`] and the spec's own `normalize`, so what a
//!   form launches is what the CLI would run for the same values, and a value
//!   that is wrong is reported by field name instead of being replaced by a
//!   default.

use crossterm::event::KeyEvent;
use pmetal_core::jobs::Spec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, List, ListItem, ListState, Paragraph, StatefulWidget, Widget, Wrap,
};

use crate::tui::theme::THEME;
use crate::tui::widgets::{FieldKind, FormField};

/// Result of dispatching `Enter` to the current field.
///
/// Picker variants carry the label of the triggering field so tabs with
/// multiple pickers (a teacher and a student model) can route the selection
/// back to the right field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormAction {
    /// A `ModelPicker` field was activated: open the model picker modal.
    OpenModelPicker { field_label: String },
    /// A `DatasetPicker` field was activated: open the dataset picker.
    OpenDatasetPicker { field_label: String },
    /// An inline-editable field entered edit mode: route subsequent keys to
    /// [`FormTabState::handle_edit_key`] until the user confirms or cancels.
    StartEdit,
}

/// A problem with a form's values, naming the field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormIssue {
    /// The field's label on the form.
    pub field: String,
    /// What is wrong with it.
    pub message: String,
}

impl std::fmt::Display for FormIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

/// A read-only view of a form's fields, for deciding which are shown.
pub struct FormView<'a> {
    fields: &'a [FormField],
}

impl FormView<'_> {
    /// The current value of the field labelled `label` (empty if there is none).
    pub fn value(&self, label: &str) -> &str {
        self.fields
            .iter()
            .find(|f| f.label == label)
            .map_or("", |f| f.value.as_str())
    }
}

/// Decides whether a field is shown, given the rest of the form.
pub type Visibility = fn(&FormView<'_>, &FormField) -> bool;

fn always_visible(_: &FormView<'_>, _: &FormField) -> bool {
    true
}

/// Owned state shared by every form-driven tab.
///
/// Holds the field vector, the ratatui `ListState` (so selection and scroll
/// survive renders), and the cursor into `fields`.
pub struct FormTabState {
    pub fields: Vec<FormField>,
    pub list_state: ListState,
    field_idx: usize,
    /// What each field held when the form was built, to tell what was changed.
    initial: Vec<String>,
    visible: Visibility,
}

impl std::fmt::Debug for FormTabState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FormTabState")
            .field("fields", &self.fields.len())
            .field("field_idx", &self.field_idx)
            .finish_non_exhaustive()
    }
}

impl FormTabState {
    /// A form with a field for every descriptor of `S`, holding `S::default()`.
    pub fn for_spec<S: Spec>() -> Self {
        let defaults = pmetal_core::default_text::<S>();
        let fields = S::field_descriptors()
            .iter()
            .map(|d| {
                let text = defaults
                    .iter()
                    .find(|(name, _)| *name == d.name)
                    .map_or("", |(_, text)| text.as_str());
                FormField::from_descriptor(d, text)
            })
            .collect();
        Self::new(fields)
    }

    /// Create a form-tab state from a pre-built field list, for the few forms
    /// that are not one spec's (a TUI-only readout, a command with no spec).
    pub fn new(fields: Vec<FormField>) -> Self {
        let initial = fields.iter().map(|f| f.value.clone()).collect();
        Self {
            fields,
            list_state: ListState::default().with_selected(Some(1)),
            field_idx: 0,
            initial,
            visible: always_visible,
        }
    }

    /// Show only the fields `visible` accepts. Navigation and drawing both
    /// follow it, so the cursor never lands on a hidden row.
    pub fn with_visibility(mut self, visible: Visibility) -> Self {
        self.visible = visible;
        self
    }

    /// Put a read-only field right after the one labelled `after`.
    pub fn insert_after(&mut self, after: &str, field: FormField) {
        if let Some(pos) = self.fields.iter().position(|f| f.label == after) {
            self.initial.insert(pos + 1, field.value.clone());
            self.fields.insert(pos + 1, field);
        }
    }

    // ── Field lookup / mutation ─────────────────────────────────────────

    /// A field's current value by label; empty if the label isn't found.
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
            f.set(value);
        }
    }

    /// What the field labelled `label` held when the form was built.
    pub fn initial_of(&self, label: &str) -> Option<&str> {
        let idx = self.fields.iter().position(|f| f.label == label)?;
        self.initial.get(idx).map(String::as_str)
    }

    /// The field the cursor is on.
    pub fn current(&self) -> Option<&FormField> {
        self.fields.get(self.field_idx)
    }

    /// A view for visibility rules.
    pub fn view(&self) -> FormView<'_> {
        FormView {
            fields: &self.fields,
        }
    }

    // ── The spec ────────────────────────────────────────────────────────

    /// The spec these values stand for, or what is wrong with them.
    ///
    /// Built the way the CLI builds it: text is read by the field's type
    /// (never replaced by a default when it does not parse), then the spec's
    /// own `normalize` runs its range and cross-field rules.
    pub fn to_spec<S: Spec>(&self) -> Result<S, Vec<FormIssue>> {
        let label_of = |name: &str| {
            S::field_descriptors()
                .iter()
                .find(|d| d.name == name)
                .map_or_else(|| name.to_string(), |d| d.label.to_string())
        };
        let issues = |errors: Vec<pmetal_core::FieldError>| {
            errors
                .into_iter()
                .map(|e| FormIssue {
                    field: label_of(&e.field),
                    message: e.message,
                })
                .collect::<Vec<_>>()
        };
        let mut spec: S = pmetal_core::spec_from_text(|d| {
            self.fields
                .iter()
                .find(|f| f.name == Some(d.name))
                .map(|f| f.value.clone())
        })
        .map_err(issues)?;
        spec.normalize_spec().map_err(issues)?;
        Ok(spec)
    }

    /// The fields that matter for a "start this job?" prompt: the required
    /// ones first, then every other field that is not what it started as.
    pub fn summary(&self) -> Vec<String> {
        let changed = |i: usize| {
            self.initial
                .get(i)
                .is_some_and(|v| v != &self.fields[i].value)
        };
        let shown: Vec<usize> = (0..self.fields.len())
            .filter(|&i| self.fields[i].required || changed(i))
            .collect();
        let width = shown
            .iter()
            .map(|&i| self.fields[i].label.chars().count())
            .max()
            .unwrap_or(0);
        let row = |i: usize| {
            let f = &self.fields[i];
            let value = if f.value.is_empty() {
                "(not set)"
            } else {
                f.value.as_str()
            };
            format!("{:<width$}  {value}", f.label)
        };
        let required = shown.iter().copied().filter(|&i| self.fields[i].required);
        let others = shown.iter().copied().filter(|&i| !self.fields[i].required);
        required.chain(others).map(row).collect()
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

    pub fn paste(&mut self, text: &str) {
        if let Some(field) = self.fields.get_mut(self.field_idx) {
            if field.editing {
                field.paste(text);
            }
        }
    }

    /// Apply the edit in progress. Returns whether it was accepted; if not the
    /// field is still being edited and carries the reason.
    pub fn confirm_edit(&mut self) -> bool {
        self.fields
            .get_mut(self.field_idx)
            .is_some_and(FormField::confirm_edit)
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

    /// Cycle the current Toggle/Enum field backwards.
    pub fn cycle_back(&mut self) {
        if let Some(field) = self.fields.get_mut(self.field_idx) {
            field.cycle_back();
        }
    }

    // ── Navigation ──────────────────────────────────────────────────────

    /// Which fields are shown now.
    fn mask(&self) -> Vec<bool> {
        let view = self.view();
        self.fields
            .iter()
            .map(|f| !matches!(f.kind, FieldKind::ReadOnly) && (self.visible)(&view, f))
            .collect()
    }

    /// Advance the cursor to the next shown field.
    pub fn next_param(&mut self) {
        self.step_param(1);
    }

    pub fn prev_param(&mut self) {
        self.step_param(-1);
    }

    fn step_param(&mut self, step: i32) {
        let count = self.fields.len();
        if count == 0 {
            return;
        }
        let mask = self.mask();
        let mut idx = self.field_idx;
        // Walk at most `count` times: every field hidden must not loop forever.
        for _ in 0..count {
            idx = if step >= 0 {
                (idx + 1) % count
            } else {
                (idx + count - 1) % count
            };
            if mask[idx] {
                self.field_idx = idx;
                self.sync_list_selection();
                return;
            }
        }
    }

    /// Move the cursor to the field labelled `label`, if it is shown.
    pub fn focus_field(&mut self, label: &str) {
        let mask = self.mask();
        if let Some(idx) = self
            .fields
            .iter()
            .position(|f| f.label == label)
            .filter(|&i| mask[i])
        {
            self.field_idx = idx;
            self.sync_list_selection();
        }
    }

    /// Put the cursor on the first shown field if it is on a hidden one.
    pub fn ensure_cursor_visible(&mut self) {
        let mask = self.mask();
        if !mask.get(self.field_idx).copied().unwrap_or(false) {
            if let Some(idx) = mask.iter().position(|&shown| shown) {
                self.field_idx = idx;
            }
        }
        self.sync_list_selection();
    }

    fn sync_list_selection(&mut self) {
        let flat = self.flat_index_for_field(self.field_idx);
        self.list_state.select(Some(flat));
    }

    /// The rendered-row index (with section-header offsets and hidden fields
    /// removed) of the field at `field_idx`. Mirrors [`render_list`](Self::render_list).
    fn flat_index_for_field(&self, field_idx: usize) -> usize {
        let mask = self.mask();
        let mut flat = 0;
        let mut current_section: Option<&str> = None;
        for (i, field) in self.fields.iter().enumerate() {
            if !mask[i] {
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

    /// Render the form as a bordered list with `--- Section ---` dividers, the
    /// current row highlighted, and the spec's help text for it below.
    pub fn render_list(&mut self, area: Rect, buf: &mut Buffer, title: &str) {
        self.ensure_cursor_visible();
        let block = Block::default()
            .title(format!(" {title} "))
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        // The help for the field under the cursor takes the bottom of the box
        // when there is room for it and something to say.
        let help = self
            .current()
            .and_then(|f| f.help)
            .filter(|_| inner.height >= 12);
        let [list_area, help_area] = if help.is_some() {
            Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(inner)
        } else {
            [inner, Rect::new(inner.x, inner.bottom(), inner.width, 0)]
        };

        let mask = self.mask();
        let key_width = self
            .fields
            .iter()
            .zip(&mask)
            .filter(|(_, shown)| **shown)
            .map(|(f, _)| f.label.chars().count())
            .max()
            .unwrap_or(10);

        let mut current_section: Option<&str> = None;
        let mut items: Vec<ListItem<'_>> = Vec::new();
        for (i, field) in self.fields.iter().enumerate() {
            if !mask[i] {
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

        let list = List::new(items).highlight_style(THEME.table_selected);
        StatefulWidget::render(list, list_area, buf, &mut self.list_state);

        if let Some(help) = help {
            Paragraph::new(Line::from(Span::styled(help, THEME.text_dim)))
                .wrap(Wrap { trim: true })
                .block(
                    Block::default()
                        .borders(Borders::TOP)
                        .border_style(THEME.block),
                )
                .render(help_area, buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};
    use pmetal_core::JobFields;
    use pmetal_core::jobs::{GrpoSpec, InferSpec, QuantizeSpec, TrainSpec};

    fn type_text(form: &mut FormTabState, text: &str) {
        for c in text.chars() {
            form.handle_edit_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
    }

    /// Set a field the way a person does: move to it, Enter, type, Enter.
    fn enter_value(form: &mut FormTabState, label: &str, text: &str) {
        form.focus_field(label);
        form.handle_enter();
        if form.is_editing() {
            let len = form.current().unwrap().value.len();
            for _ in 0..len {
                form.handle_edit_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
            }
            type_text(form, text);
            assert!(
                form.confirm_edit(),
                "{label}: {:?}",
                form.current().unwrap().error
            );
        } else {
            form.set_value(label, text);
        }
    }

    #[test]
    fn a_form_has_a_field_for_every_descriptor_and_starts_from_the_default_spec() {
        let form = FormTabState::for_spec::<TrainSpec>();
        assert_eq!(form.fields.len(), TrainSpec::field_descriptors().len());
        let spec = form.to_spec::<TrainSpec>();
        // A fresh form is missing only what is required.
        let issues = spec.unwrap_err();
        let names: Vec<&str> = issues.iter().map(|i| i.field.as_str()).collect();
        assert_eq!(names, ["Model", "Dataset"], "{issues:?}");
        assert!(issues.iter().all(|i| i.message == "required"));
    }

    #[test]
    fn what_a_form_launches_is_what_the_cli_would_run_for_the_same_values() {
        let mut form = FormTabState::for_spec::<GrpoSpec>();
        enter_value(&mut form, "Model", "m");
        enter_value(&mut form, "Dataset", "d.jsonl");
        // Fields the old hand-copied mapping never read.
        enter_value(&mut form, "Loss Type", "gspo");
        enter_value(&mut form, "Updates per Batch", "2");
        enter_value(&mut form, "Checkpoint Every", "10");
        let spec = form.to_spec::<GrpoSpec>().unwrap();
        assert_eq!(spec.loss_type, "gspo");
        assert_eq!(spec.num_iterations, 2);
        assert_eq!(spec.checkpoint_every, 10);
        let argv = spec.to_argv();
        for expected in [
            "--loss-type",
            "gspo",
            "--num-iterations",
            "2",
            "--checkpoint-every",
        ] {
            assert!(argv.iter().any(|a| a == expected), "{expected} in {argv:?}");
        }
    }

    #[test]
    fn a_value_outside_the_specs_bounds_is_refused_by_field_name() {
        let mut form = FormTabState::for_spec::<TrainSpec>();
        enter_value(&mut form, "Model", "m");
        enter_value(&mut form, "Dataset", "d");
        // The inline editor already refuses an out-of-range number...
        form.focus_field("LoRA r");
        form.handle_enter();
        form.handle_edit_key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        type_text(&mut form, "0");
        assert!(!form.confirm_edit());
        form.cancel_edit();
        // ...and a value that got in some other way is refused at launch.
        form.set_value("LoRA r", "0");
        let issues = form.to_spec::<TrainSpec>().unwrap_err();
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].field, "LoRA r");
        assert!(issues[0].message.contains('≥'), "{}", issues[0].message);
        // Text that is not a number is named too, not silently defaulted.
        form.set_value("LoRA r", "sixteen");
        let issues = form.to_spec::<TrainSpec>().unwrap_err();
        assert_eq!(issues[0].field, "LoRA r");
        assert!(
            issues[0].message.contains("'sixteen'"),
            "{}",
            issues[0].message
        );
    }

    #[test]
    fn a_cleared_optional_field_goes_back_to_unset() {
        let mut form = FormTabState::for_spec::<InferSpec>();
        enter_value(&mut form, "Model", "m");
        enter_value(&mut form, "Prompt", "hi");
        enter_value(&mut form, "Temperature", "0.3");
        assert_eq!(form.to_spec::<InferSpec>().unwrap().temperature, Some(0.3));
        enter_value(&mut form, "Temperature", "");
        assert_eq!(form.to_spec::<InferSpec>().unwrap().temperature, None);
    }

    #[test]
    fn hidden_fields_are_skipped_by_the_cursor_and_the_list() {
        fn only_mlx_knobs_for_mlx(view: &FormView<'_>, field: &FormField) -> bool {
            !matches!(field.label.as_str(), "MLX Bits" | "MLX Group Size")
                || view.value("Format") == "mlx"
        }
        let mut form =
            FormTabState::for_spec::<QuantizeSpec>().with_visibility(only_mlx_knobs_for_mlx);
        let mut seen = Vec::new();
        for _ in 0..form.fields.len() {
            form.next_param();
            seen.push(form.current().unwrap().label.clone());
        }
        assert!(!seen.iter().any(|l| l == "MLX Bits"), "{seen:?}");
        form.set_value("Format", "mlx");
        let mut seen = Vec::new();
        for _ in 0..form.fields.len() {
            form.next_param();
            seen.push(form.current().unwrap().label.clone());
        }
        assert!(seen.iter().any(|l| l == "MLX Bits"), "{seen:?}");
    }

    #[test]
    fn the_summary_lists_what_is_required_and_what_was_changed() {
        let mut form = FormTabState::for_spec::<TrainSpec>();
        enter_value(&mut form, "Model", "Qwen/Qwen3-0.6B");
        enter_value(&mut form, "Dataset", "data.jsonl");
        enter_value(&mut form, "Epochs", "3");
        let summary = form.summary();
        assert!(summary[0].starts_with("Model") && summary[0].ends_with("Qwen/Qwen3-0.6B"));
        assert!(summary[1].starts_with("Dataset"));
        assert!(
            summary
                .iter()
                .any(|l| l.starts_with("Epochs") && l.ends_with('3'))
        );
        // Nothing else was touched, so nothing else is listed.
        assert_eq!(summary.len(), 3, "{summary:?}");
    }

    #[test]
    fn rendering_a_form_with_the_cursor_on_a_hidden_row_moves_it() {
        fn hide_model(_: &FormView<'_>, field: &FormField) -> bool {
            field.label != "Model"
        }
        let mut form = FormTabState::for_spec::<TrainSpec>().with_visibility(hide_model);
        assert_eq!(form.current().unwrap().label, "Model");
        let mut buf = Buffer::empty(Rect::new(0, 0, 60, 20));
        form.render_list(buf.area, &mut buf, "Train");
        assert_ne!(form.current().unwrap().label, "Model");
    }
}
