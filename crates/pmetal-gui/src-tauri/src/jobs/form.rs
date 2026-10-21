//! What the frontend needs to draw a job's form: the spec's own field
//! descriptors, in a form that serializes.
//!
//! The form has no field list, default, range or option of its own. It asks for
//! these and renders them, so a field added to or removed from a spec appears
//! in or leaves the form with no change to the app, and the values a form starts
//! from are the ones the command line would run with.

use pmetal::core::{FieldDescriptor, FieldKind, FieldShape, JobFields, default_text};
use serde::Serialize;

/// How a field is entered.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Input {
    Text,
    Number {
        min: Option<f64>,
        max: Option<f64>,
    },
    Integer {
        min: Option<i64>,
        max: Option<i64>,
    },
    Choice {
        options: Vec<String>,
    },
    Toggle,
    /// A model: a Hub id, or a folder on disk.
    Model,
    /// A dataset: a Hub id, or a file or folder on disk.
    Dataset,
    /// A file or folder on disk.
    Path,
    Readonly,
}

/// One field of a form.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldInfo {
    pub name: String,
    pub label: String,
    pub help: Option<String>,
    pub group: String,
    pub input: Input,
    pub required: bool,
    /// Entered as comma-separated values.
    pub list: bool,
    /// May be left empty.
    pub optional: bool,
    /// The text the form starts with.
    pub default: String,
}

/// A job's form.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FormInfo {
    /// The job's key (`train`, `embed_train`, ...).
    pub key: String,
    pub label: String,
    pub description: String,
    pub subcommand: String,
    pub fields: Vec<FieldInfo>,
}

/// A number the JSON can carry, or none when the bound is "unbounded".
fn finite(n: f64) -> Option<f64> {
    (n.is_finite() && n.abs() < 1e15).then_some(n)
}

fn bound(n: i64) -> Option<i64> {
    // The descriptors use the type's own extremes for "no bound", and a
    // JavaScript number cannot hold those exactly.
    (n.unsigned_abs() < (1 << 53)).then_some(n)
}

impl FieldInfo {
    fn of(d: &FieldDescriptor, default: String) -> Self {
        let input = match d.kind {
            FieldKind::Text => Input::Text,
            FieldKind::Number { min, max } => Input::Number {
                min: finite(min),
                max: finite(max),
            },
            FieldKind::Integer { min, max } => Input::Integer {
                min: bound(min),
                max: bound(max),
            },
            FieldKind::Enum { options } => Input::Choice {
                options: options.iter().map(|o| (*o).to_string()).collect(),
            },
            FieldKind::Toggle => Input::Toggle,
            FieldKind::ModelPicker => Input::Model,
            FieldKind::DatasetPicker => Input::Dataset,
            FieldKind::ReadOnly => Input::Readonly,
            FieldKind::Path => Input::Path,
        };
        Self {
            name: d.name.to_string(),
            label: d.label.to_string(),
            help: d.help.map(str::to_string),
            group: d.group.to_string(),
            input,
            required: d.required,
            list: d.shape == FieldShape::List,
            optional: d.shape == FieldShape::Optional,
            default,
        }
    }
}

/// The form of spec `S`, starting from `S::default()`.
pub fn form_of<S>(label: &str, description: &str) -> FormInfo
where
    S: JobFields + Default + Serialize,
{
    let defaults = default_text::<S>();
    FormInfo {
        key: serde_json::to_value(S::job_kind())
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default(),
        label: label.to_string(),
        description: description.to_string(),
        subcommand: S::subcommand().to_string(),
        fields: S::field_descriptors()
            .iter()
            .map(|d| {
                let default = defaults
                    .iter()
                    .find(|(name, _)| *name == d.name)
                    .map(|(_, text)| text.clone())
                    .unwrap_or_default();
                FieldInfo::of(d, default)
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal::core::jobs::{MergeSpec, TrainSpec};

    #[test]
    fn a_form_starts_from_the_text_the_command_line_would_run_with() {
        let form = form_of::<TrainSpec>("Training", "");
        let default = TrainSpec::default();
        let field = |name: &str| form.fields.iter().find(|f| f.name == name).unwrap();
        assert_eq!(field("epochs").default, default.epochs.to_string());
        assert_eq!(field("batch_size").default, default.batch_size.to_string());
        assert_eq!(field("optimizer").default, default.optimizer);
        assert_eq!(form.subcommand, "train");
        assert_eq!(form.key, "train");
    }

    #[test]
    fn every_descriptor_becomes_a_field_with_its_options_and_bounds() {
        let form = form_of::<MergeSpec>("Merge", "");
        let method = form.fields.iter().find(|f| f.name == "method").unwrap();
        let Input::Choice { options } = &method.input else {
            panic!("method is a choice, got {:?}", method.input);
        };
        assert!(options.iter().any(|o| o == "slerp"));
        assert_eq!(form.fields.len(), MergeSpec::field_descriptors().len());
        assert!(form.fields.iter().any(|f| f.required));
    }

    #[test]
    fn a_form_serializes_without_unbounded_numbers() {
        // JSON cannot carry infinities, and the descriptors use them (or the
        // integer extremes) for "no bound".
        for form in [form_of::<TrainSpec>("t", ""), form_of::<MergeSpec>("m", "")] {
            let text = serde_json::to_string(&form).unwrap();
            assert!(!text.contains("9223372036854775807"), "{text}");
            assert!(!text.contains("e308"), "{text}");
        }
    }
}
