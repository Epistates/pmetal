//! What tools hand back.
//!
//! A tool that has data returns it as structured content, with the same JSON
//! as text for clients that do not read structured content. A tool that fails
//! returns a result marked `isError`, so the model can read what went wrong and
//! retry; only an unknown tool is a protocol error.

use schemars::JsonSchema;
use serde::Serialize;
use serde_json::Value;
use turbomcp::prelude::*;

/// The result of starting a background job.
#[derive(Debug, Serialize, JsonSchema)]
pub struct JobStarted {
    /// Pass to `job_status`, `job_logs` and `stop_job`.
    pub job_id: String,
    /// The `pmetal` subcommand the job runs.
    pub command: String,
    pub status: String,
    /// Where the job writes its results, when it has an output location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    pub message: String,
}

impl JobStarted {
    pub fn new(job_id: &str, command: &str, output: Option<String>) -> Self {
        Self {
            job_id: job_id.to_string(),
            command: command.to_string(),
            status: "running".to_string(),
            output,
            message: format!(
                "Job started. Use job_status with job_id '{job_id}' to monitor progress."
            ),
        }
    }
}

/// The JSON Schema of `T`, as the value an `outputSchema` holds.
pub fn schema_of<T: JsonSchema>() -> Value {
    // Inlined, so a schema is one self-contained object with no `$ref` to chase.
    let settings = schemars::generate::SchemaSettings::draft2020_12().with(|s| {
        s.inline_subschemas = true;
    });
    let mut schema = serde_json::to_value(settings.into_generator().into_root_schema_for::<T>())
        .unwrap_or(Value::Null);
    // The MCP schema dialect is declared by the SDK; the generator's own `$schema`
    // and `title` add nothing.
    if let Some(object) = schema.as_object_mut() {
        object.remove("$schema");
        object.remove("title");
    }
    schema
}

pub fn job_started_schema() -> Value {
    schema_of::<JobStarted>()
}

/// A result carrying `value` as structured content and as text.
///
/// Structured content must be a JSON object, so a list is wrapped by the caller
/// (`{"models": [...]}`) before it gets here.
pub fn structured<T: Serialize>(value: &T) -> McpResult<ToolResult> {
    let json = serde_json::to_value(value).map_err(|e| McpError::internal(e.to_string()))?;
    if !json.is_object() {
        return Err(McpError::internal(
            "structured content must be a JSON object",
        ));
    }
    ToolResult::json(&json).map_err(|e| McpError::internal(e.to_string()))
}

/// Plain text output.
pub fn text(output: impl Into<String>) -> ToolResult {
    ToolResult::text(output)
}

/// A tool failure for the model to read, with the error's kind in `_meta`.
pub fn failure(error: &McpError) -> ToolResult {
    error.to_tool_result()
}
