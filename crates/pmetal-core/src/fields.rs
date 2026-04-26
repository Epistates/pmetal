//! Field descriptors — one source of truth for spec-driven form rendering,
//! argv reconstruction, and validation across CLI / TUI / GUI / MCP.
//!
//! Specs in [`crate::jobs`] derive [`JobFields`] via the `#[derive(JobSpec)]`
//! macro in `pmetal-core-derive`, exposing per-field metadata that:
//!
//! - The TUI uses to materialise [`crate::FormField`]-equivalent widgets.
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
    /// Per-field descriptors (form metadata + argv binding + defaults).
    fn field_descriptors() -> &'static [FieldDescriptor];

    /// Build the `argv` that, when passed to `pmetal <subcommand> ...`,
    /// reproduces this spec. Used by:
    /// - MCP tools to spawn the CLI subprocess for the same job.
    /// - TUI subprocess fallback when the in-process trainer cannot be used.
    /// - CLI's own round-trip tests (parse → spec → argv → parse → equals).
    fn to_argv(&self) -> Vec<String>;

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
