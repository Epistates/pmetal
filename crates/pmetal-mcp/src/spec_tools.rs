//! Tools built from the job specs in `pmetal-core`.
//!
//! One `pmetal` subcommand has one spec, and the spec already says everything a
//! caller needs: its fields, their types, ranges, options, defaults and which
//! are required. A tool is generated from it rather than written beside it, so
//! the schema a client sees, what the server accepts and the command line it
//! runs cannot drift from the CLI, the TUI and the GUI, which read the same
//! spec. A field added to or removed from a spec appears in or leaves the tool
//! with no change here.
//!
//! The arguments of a call go through four steps: the JSON is checked against
//! the generated schema (so an error names the parameter and what was wrong),
//! deserialized into the spec (every default is the spec's own), its paths are
//! checked against the [`PathPolicy`], and the spec's `normalize` runs its
//! cross-field rules. What comes out is the argv to run.

use std::marker::PhantomData;

use pmetal_core::{FieldDescriptor, FieldError, FieldKind};
use pmetal_supervisor::{EventsFile, LaunchOptions, LaunchPlan};
use serde::Serialize;
use serde_json::{Map, Value, json};
use turbomcp::ToolOutputSchema;
use turbomcp::prelude::*;

use crate::policy::PathPolicy;
use crate::schema_check;
use crate::tooldef::Hints;

/// A job spec the server can build a tool from: every spec in `pmetal-core`.
pub use pmetal_core::jobs::Spec;

/// How a tool runs its command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Start a background job and return its ID.
    Job,
    /// Run to completion and return the output.
    Blocking,
}

/// A command ready to run.
#[derive(Debug, Clone)]
pub struct Prepared {
    /// The command and the files it is asked to write.
    pub plan: LaunchPlan,
    /// The spec as the command will see it.
    pub spec: Value,
}

/// A tool generated from a spec, as the server's registry holds it.
pub trait SpecTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn mode(&self) -> Mode;
    /// The MCP definition: name, description, input schema, annotations.
    fn definition(&self) -> Tool;
    /// Validate `args` and build the command.
    fn prepare(&self, args: Value, policy: &PathPolicy) -> McpResult<Prepared>;
    /// The spec's parameters, for tests and documentation.
    fn parameters(&self) -> Vec<String>;
    /// The spec as the CLI would default it, as JSON.
    fn defaults(&self) -> Value;
    /// The spec `args` describe once their shape is checked and unknown
    /// parameters refused, as JSON. Paths are not yet checked and cross-field
    /// rules have not run.
    fn parsed(&self, args: Value) -> McpResult<Value>;
}

/// A check on the spec after its paths are settled, for rules about a field's
/// value that the spec itself cannot know (what a server may expose, say).
pub type SpecCheck = fn(&Map<String, Value>) -> Result<(), String>;

/// Docs for the spec fields that several specs have and no descriptor shows.
fn shared_doc(field: &str) -> Option<&'static str> {
    match field {
        "resume" => Some("Resume from the latest checkpoint in the output directory"),
        "log_metrics" => Some(
            "Write per-step metrics as JSONL to this file (default: metrics.jsonl in the output \
             directory)",
        ),
        _ => None,
    }
}

/// What distinguishes one generated tool from another.
pub struct SpecToolConfig {
    pub name: &'static str,
    pub description: &'static str,
    pub mode: Mode,
    /// Hints for a client deciding whether to ask before calling. Whether the
    /// tool reaches the Hub is read off the spec's fields.
    pub hints: Hints,
    /// Docs for spec fields that have no descriptor, `(field, description)`.
    pub extra_docs: &'static [(&'static str, &'static str)],
    /// Spec fields that hold paths without a path descriptor: strings, or lists
    /// of strings, with comma-separated entries.
    pub extra_path_fields: &'static [&'static str],
    pub check: Option<SpecCheck>,
}

impl SpecToolConfig {
    pub const fn new(
        name: &'static str,
        description: &'static str,
        mode: Mode,
        hints: Hints,
    ) -> Self {
        Self {
            name,
            description,
            mode,
            hints,
            extra_docs: &[],
            extra_path_fields: &[],
            check: None,
        }
    }

    #[must_use]
    pub const fn docs(mut self, docs: &'static [(&'static str, &'static str)]) -> Self {
        self.extra_docs = docs;
        self
    }

    #[must_use]
    pub const fn paths(mut self, fields: &'static [&'static str]) -> Self {
        self.extra_path_fields = fields;
        self
    }

    #[must_use]
    pub const fn check(mut self, check: SpecCheck) -> Self {
        self.check = Some(check);
        self
    }
}

/// The tool for spec `S`.
pub struct SpecEntry<S: Spec> {
    config: SpecToolConfig,
    _spec: PhantomData<fn() -> S>,
}

impl<S: Spec> SpecEntry<S> {
    pub const fn new(config: SpecToolConfig) -> Self {
        Self {
            config,
            _spec: PhantomData,
        }
    }

    fn spec_defaults() -> Map<String, Value> {
        spec_json(&S::default())
            .as_object()
            .cloned()
            .unwrap_or_default()
    }

    /// Whether a field names a model, a dataset or a path.
    fn path_role(&self, name: &str) -> Option<PathRole> {
        // The metrics file a training spec may name.
        if S::LOG_METRICS_FIELD == Some(name) {
            return Some(PathRole::Path);
        }
        if let Some(d) = S::field_descriptors().iter().find(|d| d.name == name) {
            match d.kind {
                FieldKind::Path => return Some(PathRole::Path),
                FieldKind::ModelPicker | FieldKind::DatasetPicker => {
                    return Some(PathRole::Reference);
                }
                _ => {}
            }
        }
        self.config
            .extra_path_fields
            .contains(&name)
            .then_some(PathRole::PathList)
    }

    /// The spec built from `args`, before `normalize` and the paths: the shape
    /// of the arguments is right and unknown ones are refused.
    pub fn parse(&self, args: Value) -> McpResult<S> {
        // Whether a required parameter is present is the spec's to say (a model
        // must be non-empty, not merely there), so it is checked with the rest of
        // the spec's rules and the schema checked here leaves `required` out.
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": self.properties(),
        });
        schema_check::check_arguments(self.config.name, &schema, &args)
            .map_err(McpError::invalid_params)?;
        // A null is an argument left out.
        let args: Map<String, Value> = args
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(_, v)| !v.is_null())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        serde_json::from_value(Value::Object(args))
            .map_err(|e| McpError::invalid_params(format!("invalid arguments: {e}")))
    }

    /// `parse`, then everything else.
    fn finish(&self, spec: S, policy: &PathPolicy) -> McpResult<Prepared> {
        let mut value = spec_json(&spec);
        let Some(fields) = value.as_object_mut() else {
            return Err(McpError::internal("a spec did not serialize to an object"));
        };

        // A relative metrics file is relative to the output directory, as the
        // command itself reads it, not to this server's working directory.
        if let (Some(metrics), Some(output)) = (S::LOG_METRICS_FIELD, S::OUTPUT_FIELD) {
            let text = |name: &str| {
                fields
                    .get(name)
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
            };
            if let (Some(given), Some(out)) = (text(metrics), text(output))
                && !std::path::Path::new(&given).is_absolute()
                && let Ok(out) = policy.check_path(output, &out)
            {
                let file = pmetal_core::paths::resolve_in(&out, &given);
                fields.insert(metrics.into(), Value::String(file.display().to_string()));
            }
        }

        // Paths: every one the caller gave, and every default, must be allowed.
        let mut problems = Vec::new();
        for (name, field) in fields.iter_mut() {
            let Some(role) = self.path_role(name) else {
                continue;
            };
            if let Err(e) = apply_policy(name, field, role, policy) {
                problems.push(e);
            }
        }
        if !problems.is_empty() {
            return Err(McpError::invalid_params(problems.join("; ")));
        }

        if let Some(check) = self.config.check {
            check(fields).map_err(McpError::invalid_params)?;
        }

        let mut spec: S = serde_json::from_value(value.clone())
            .map_err(|e| McpError::internal(format!("a spec did not round-trip: {e}")))?;
        spec.normalize_spec().map_err(field_errors)?;
        // A command that runs to completion is read by its output, not followed.
        let options = LaunchOptions {
            events: match self.config.mode {
                Mode::Job => EventsFile::InOutput,
                Mode::Blocking => EventsFile::Off,
            },
            ..LaunchOptions::default()
        };
        let plan =
            LaunchPlan::for_spec(&spec, &options).map_err(|e| McpError::internal(e.to_string()))?;
        Ok(Prepared { plan, spec: value })
    }

    /// The `properties` of the input schema.
    fn properties(&self) -> Map<String, Value> {
        let defaults = Self::spec_defaults();
        let mut properties = Map::new();
        for d in S::field_descriptors() {
            properties.insert(
                d.name.to_string(),
                descriptor_schema(d, defaults.get(d.name)),
            );
        }
        // Fields the spec has and no descriptor shows.
        for (name, default) in &defaults {
            if properties.contains_key(name) {
                continue;
            }
            let description = self
                .config
                .extra_docs
                .iter()
                .find(|(field, _)| field == name)
                .map(|(_, doc)| *doc)
                .or_else(|| shared_doc(name));
            let mut schema = match default {
                Value::Bool(_) => json!({"type": "boolean"}),
                Value::Number(_) => json!({"type": "number"}),
                Value::Array(_) => json!({"type": "array", "items": {"type": "string"}}),
                _ => json!({"type": "string"}),
            };
            if let Some(text) = description {
                schema["description"] = json!(text);
            }
            if !default.is_null() {
                schema["default"] = default.clone();
            }
            properties.insert(name.clone(), schema);
        }
        properties
    }
}

impl<S: Spec> SpecTool for SpecEntry<S> {
    fn name(&self) -> &'static str {
        self.config.name
    }

    fn mode(&self) -> Mode {
        self.config.mode
    }

    fn definition(&self) -> Tool {
        let defaults = Self::spec_defaults();
        let properties = self.properties();
        let required: Vec<String> = S::field_descriptors()
            .iter()
            .filter(|d| d.required && is_unset(defaults.get(d.name)))
            .map(|d| d.name.to_string())
            .collect();
        let mut schema = turbomcp::ToolInputSchema::with_required_properties(
            properties.clone().into_iter().collect(),
            required,
        );
        if schema.required.as_ref().is_some_and(Vec::is_empty) {
            schema.required = None;
        }
        // Whether the tool can reach out to the Hub: it takes a model or dataset
        // by name.
        let open_world = S::field_descriptors()
            .iter()
            .any(|d| matches!(d.kind, FieldKind::ModelPicker | FieldKind::DatasetPicker));
        let mut hints = self.config.hints;
        hints.open_world = open_world;
        let mut tool = Tool::new(self.config.name, self.config.description)
            .with_schema(schema)
            .with_annotations(hints.annotations());
        if self.config.mode == Mode::Job {
            tool = tool.with_output_schema(ToolOutputSchema::from_value(
                crate::results::job_started_schema(),
            ));
        }
        tool
    }

    fn prepare(&self, args: Value, policy: &PathPolicy) -> McpResult<Prepared> {
        let spec = self.parse(args)?;
        self.finish(spec, policy)
    }

    fn parameters(&self) -> Vec<String> {
        self.properties().keys().cloned().collect()
    }

    fn defaults(&self) -> Value {
        Value::Object(Self::spec_defaults())
    }

    fn parsed(&self, args: Value) -> McpResult<Value> {
        self.parse(args).map(|spec| spec_json(&spec))
    }
}

#[derive(Clone, Copy)]
enum PathRole {
    /// A filesystem path.
    Path,
    /// A Hub ID or a path.
    Reference,
    /// A string or list of strings of paths, comma-separated.
    PathList,
}

fn apply_policy(
    name: &str,
    field: &mut Value,
    role: PathRole,
    policy: &PathPolicy,
) -> Result<(), String> {
    match field {
        Value::Null => Ok(()),
        Value::String(s) if s.is_empty() => Ok(()),
        Value::String(s) => {
            *s = match role {
                PathRole::Path => policy
                    .check_path(name, s)
                    .map(|p| p.display().to_string())?,
                PathRole::Reference => policy.check_reference(name, s)?,
                PathRole::PathList => check_list(name, s, policy)?,
            };
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                if let Value::String(s) = item {
                    *s = check_list(name, s, policy)?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Each comma-separated entry of `list` is a path that must be allowed.
fn check_list(name: &str, list: &str, policy: &PathPolicy) -> Result<String, String> {
    let mut out = Vec::new();
    for entry in list.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        out.push(policy.check_path(name, entry)?.display().to_string());
    }
    Ok(out.join(","))
}

fn field_errors(errors: Vec<FieldError>) -> McpError {
    McpError::invalid_params(format!(
        "validation failed: {}",
        errors
            .iter()
            .map(|e| format!("{}: {}", e.field, e.message))
            .collect::<Vec<_>>()
            .join("; ")
    ))
}

/// A spec as JSON the way it prints: an `f32` field shows as `0.05`, not as the
/// `f64` that is nearest to it.
fn spec_json<S: Serialize>(spec: &S) -> Value {
    serde_json::to_string(spec)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null)
}

fn is_unset(default: Option<&Value>) -> bool {
    match default {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        Some(Value::Array(a)) => a.is_empty(),
        _ => false,
    }
}

/// The JSON Schema of one spec field.
fn descriptor_schema(d: &FieldDescriptor, default: Option<&Value>) -> Value {
    let mut schema = match d.kind {
        FieldKind::Text | FieldKind::ReadOnly => {
            if matches!(default, Some(Value::Array(_))) {
                json!({"type": "array", "items": {"type": "string"}})
            } else {
                json!({"type": "string"})
            }
        }
        FieldKind::Path | FieldKind::ModelPicker | FieldKind::DatasetPicker => {
            json!({"type": "string"})
        }
        FieldKind::Toggle => json!({"type": "boolean"}),
        FieldKind::Enum { options } => json!({"type": "string", "enum": options}),
        FieldKind::Number { min, max } => {
            let mut s = json!({"type": "number"});
            // A bound at the edge of the numeric type says nothing.
            if min.is_finite() && min.abs() < 1e30 {
                s["minimum"] = json!(min);
            }
            if max.is_finite() && max.abs() < 1e30 {
                s["maximum"] = json!(max);
            }
            s
        }
        FieldKind::Integer { min, max } => {
            let mut s = json!({"type": "integer"});
            if min != i64::MIN {
                s["minimum"] = json!(min);
            }
            if max != i64::MAX {
                s["maximum"] = json!(max);
            }
            s
        }
    };
    schema["description"] = json!(match d.help {
        Some(help) => format!("{}. {help}", d.label),
        None => d.label.to_string(),
    });
    if let Some(default) = default {
        if !default.is_null() && !(d.required && is_unset(Some(default))) {
            schema["default"] = default.clone();
        }
    }
    schema
}
