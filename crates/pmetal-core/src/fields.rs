//! Field descriptors — one source of truth for spec-driven form rendering,
//! argv reconstruction, and validation across CLI / TUI / GUI / MCP.
//!
//! Specs in [`crate::jobs`] derive [`JobFields`] via the `#[derive(JobSpec)]`
//! macro in `pmetal-core-derive`, exposing per-field metadata that:
//!
//! - The TUI uses to materialise `crate::FormField`-equivalent widgets.
//! - The CLI uses for help text and `default_value` reconciliation.
//! - The GUI uses to auto-build (or hand-build against) Svelte forms.
//! - The MCP server uses to construct argv for subprocess fallback.
//! - All four use to validate user input the same way.

use crate::JobKind;

/// One user-facing input field on a [`JobFields`] type.
#[derive(Debug, Clone, Copy)]
pub struct FieldDescriptor {
    /// Rust field identifier (snake_case).
    pub name: &'static str,
    /// Human-readable label for forms.
    pub label: &'static str,
    /// Optional extended help text.
    pub help: Option<&'static str>,
    /// Section this field belongs to (e.g. `"Training"`, `"Optimization"`, `"Model"`).
    pub group: &'static str,
    /// Render kind (drives the widget choice).
    pub kind: FieldKind,
    /// Default value for renderers that show one before user input.
    pub default: DefaultValue,
    /// Whether the field is required (non-empty / non-default).
    pub required: bool,
    /// CLI flag emitted by `to_argv` when set, or `None` if the field is
    /// form-only.
    pub argv: Option<&'static str>,
    /// True if this field's argv emission is suppressed when the value equals
    /// the default (e.g. `Option::None`, `bool false` for `flag` fields).
    pub argv_optional: bool,
    /// How many values the field holds: one, one that may be absent, or a list.
    pub shape: FieldShape,
}

/// How many values a field holds. A form needs it to know what an empty entry
/// means: unset for an optional field or an empty list, a missing value for a
/// scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldShape {
    /// Exactly one value (`String`, `usize`, `bool`, …).
    Scalar,
    /// One value or none (`Option<T>`); an empty entry is `None`.
    Optional,
    /// A list of strings (`Vec<String>`), entered comma-separated.
    List,
}

/// Field render kind. Mirrors the TUI's `FieldKind` 1:1 so adapters can map
/// across without translation.
#[derive(Debug, Clone, Copy)]
pub enum FieldKind {
    /// Free-form text.
    Text,
    /// Floating-point number with bounds.
    Number {
        /// Inclusive minimum.
        min: f64,
        /// Inclusive maximum.
        max: f64,
    },
    /// Integer with bounds.
    Integer {
        /// Inclusive minimum.
        min: i64,
        /// Inclusive maximum.
        max: i64,
    },
    /// One of a fixed set of choices.
    Enum {
        /// Allowed string values (in CLI / serde representation).
        options: &'static [&'static str],
    },
    /// Boolean toggle.
    Toggle,
    /// Pick a model from the local cache.
    ModelPicker,
    /// Pick a dataset from the local cache.
    DatasetPicker,
    /// Read-only display.
    ReadOnly,
    /// Filesystem path (file or directory).
    Path,
}

/// Typed default value carried by a [`FieldDescriptor`].
#[derive(Debug, Clone, Copy)]
pub enum DefaultValue {
    /// String default.
    Str(&'static str),
    /// Signed-integer default (covers `i8..=i64` and `usize/u32/u64` after cast).
    Int(i64),
    /// Floating-point default.
    Float(f64),
    /// Boolean default.
    Bool(bool),
    /// No default — field is required or has runtime-derived default.
    None,
}

impl DefaultValue {
    /// Render the default as the string the form should pre-fill.
    pub fn display(&self) -> String {
        match self {
            Self::Str(s) => (*s).to_string(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => format_float(*f),
            Self::Bool(b) => b.to_string(),
            Self::None => String::new(),
        }
    }
}

fn format_float(f: f64) -> String {
    // Print scientific for tiny / huge magnitudes, decimal otherwise.
    let abs = f.abs();
    if abs != 0.0 && !(1e-3..1e6).contains(&abs) {
        format!("{f:e}")
    } else {
        format!("{f}")
    }
}

/// Validation error for one field.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FieldError {
    /// Rust field identifier the error applies to.
    pub field: String,
    /// Human-readable error message.
    pub message: String,
}

impl FieldError {
    /// Build a new error.
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

/// The trait derived by `#[derive(JobSpec)]` — the contract every spec
/// satisfies so all four surfaces can consume it uniformly.
pub trait JobFields: Sized {
    /// Whether the command writes the `--log-events` file: a stream of
    /// [`JobEvent`](crate::JobEvent) lines a supervisor follows for phases and
    /// steps (`#[spec(events)]`). The one statement of it: the CLI refuses the
    /// flag for every other command, and each surface that launches jobs
    /// asks for the file only where the command writes it.
    const REPORTS_EVENTS: bool;

    /// The field that `--output` names: where the job writes its results.
    /// `None` for a command with no output location.
    const OUTPUT_FIELD: Option<&'static str>;

    /// The field that `--log-metrics` names, for a command that can write its
    /// steps to a JSONL file.
    const LOG_METRICS_FIELD: Option<&'static str>;

    /// Per-field descriptors (form metadata + argv binding + defaults).
    fn field_descriptors() -> &'static [FieldDescriptor];

    /// Build the `argv` that, when passed to `pmetal <subcommand> ...`,
    /// reproduces this spec. Used by:
    /// - MCP tools to spawn the CLI subprocess for the same job.
    /// - TUI subprocess fallback when the in-process trainer cannot be used.
    /// - CLI's own round-trip tests (parse → spec → argv → parse → equals).
    fn to_argv(&self) -> Vec<String>;

    /// Every command-line flag [`to_argv`](Self::to_argv) can emit, including
    /// the CLI-only fields that have no descriptor. A CLI that accepts a flag
    /// outside this list has a control the spec cannot drive; a flag in it that
    /// the CLI rejects is a spec that cannot start its job.
    fn argv_flags() -> &'static [&'static str];

    /// Run descriptor-driven validation: range checks, required-field checks,
    /// enum-option checks. Spec-specific cross-field rules live in each
    /// spec's hand-written `normalize`.
    fn validate_descriptors(&self) -> Vec<FieldError>;

    /// CLI subcommand string (e.g. `"train"`, `"distill"`).
    fn subcommand() -> &'static str;

    /// Job classification — used by the surface to pick the right callback /
    /// router on the receiving side.
    fn job_kind() -> JobKind;
}

// ---------------------------------------------------------------------------
// Forms: a typed spec from the text a person entered
// ---------------------------------------------------------------------------

/// The placeholder texts a picker shows for "nothing chosen".
const UNSET_PLACEHOLDERS: &[&str] = &["(not selected)", "(none)", "(auto-detect)"];

/// A boolean as a form shows it.
fn parse_toggle(text: &str) -> Option<bool> {
    match text.to_ascii_lowercase().as_str() {
        "true" | "enabled" | "on" | "yes" => Some(true),
        "false" | "disabled" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// The JSON a descriptor's field takes from the text entered for it, or why the
/// text is not a value of that field. `None` is "leave it to the spec's own
/// default".
fn field_value(d: &FieldDescriptor, raw: &str) -> Result<Option<serde_json::Value>, FieldError> {
    use serde_json::Value;

    let text = raw.trim();
    let text = if UNSET_PLACEHOLDERS.contains(&text) {
        ""
    } else {
        text
    };
    let err = |message: String| FieldError::new(d.name, message);

    if text.is_empty() {
        return match (d.shape, d.kind) {
            (FieldShape::Optional | FieldShape::List, _) => Ok(None),
            (_, FieldKind::Number { .. } | FieldKind::Integer { .. } | FieldKind::Toggle) => {
                Err(err("needs a value".to_string()))
            }
            _ => Ok(None),
        };
    }
    let one = |text: &str| -> Result<Value, FieldError> {
        match d.kind {
            FieldKind::Number { .. } => {
                let n: f64 = text
                    .parse()
                    .map_err(|_| err(format!("'{text}' is not a number")))?;
                serde_json::Number::from_f64(n)
                    .map(Value::Number)
                    .ok_or_else(|| err("must be a finite number".to_string()))
            }
            FieldKind::Integer { .. } => text
                .parse::<i64>()
                .map(|n| Value::Number(n.into()))
                .map_err(|_| err(format!("'{text}' is not a whole number"))),
            FieldKind::Toggle => parse_toggle(text)
                .map(Value::Bool)
                .ok_or_else(|| err(format!("'{text}' is not on or off"))),
            _ => Ok(Value::String(text.to_string())),
        }
    };
    match d.shape {
        FieldShape::List => {
            let items = text
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(|item| Value::String(item.to_string()))
                .collect::<Vec<_>>();
            Ok(Some(Value::Array(items)))
        }
        FieldShape::Scalar | FieldShape::Optional => one(text).map(Some),
    }
}

/// The text a form shows for each field of `S::default()`, as `(field name,
/// text)` pairs in descriptor order.
///
/// It is read from the default spec itself, not from the descriptors'
/// `default` attributes, so a form starts from exactly what the CLI would run
/// with no flags. A list is joined with commas, an absent optional is empty.
pub fn default_text<S>() -> Vec<(&'static str, String)>
where
    S: JobFields + Default + serde::Serialize,
{
    // Through text, not `to_value`: a `f32` field would otherwise show its
    // widened `f64` ("0.00019999999494757503" for `2e-4`).
    let value: serde_json::Value = serde_json::to_string(&S::default())
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(serde_json::Value::Null);
    S::field_descriptors()
        .iter()
        .map(|d| {
            let text = match value.get(d.name) {
                None | Some(serde_json::Value::Null) => String::new(),
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Array(items)) => items
                    .iter()
                    .map(|item| {
                        item.as_str()
                            .map_or_else(|| item.to_string(), str::to_string)
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
                Some(other) => other.to_string(),
            };
            (d.name, text)
        })
        .collect()
}

/// Build a spec from the text a person entered into its form.
///
/// `text_of` gives the text entered for a field, or `None` for a field the form
/// does not show (it keeps the spec's own default). The result is the spec the
/// CLI would build from the same values, so a form never carries defaults or
/// flag names of its own. Every problem is reported, each naming the field by
/// its Rust name ([`FieldError::field`]); a value outside its bounds is caught
/// by the spec's own validation (`normalize`), not here.
pub fn spec_from_text<S>(
    mut text_of: impl FnMut(&FieldDescriptor) -> Option<String>,
) -> Result<S, Vec<FieldError>>
where
    S: JobFields + serde::de::DeserializeOwned,
{
    let mut object = serde_json::Map::new();
    let mut errors = Vec::new();
    for d in S::field_descriptors() {
        let Some(raw) = text_of(d) else { continue };
        match field_value(d, &raw) {
            Ok(Some(value)) => {
                object.insert(d.name.to_string(), value);
            }
            Ok(None) => {}
            Err(e) => errors.push(e),
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    serde_path_to_error::deserialize(serde_json::Value::Object(object)).map_err(|e| {
        let field = e.path().to_string();
        vec![FieldError::new(field, e.inner().to_string())]
    })
}
