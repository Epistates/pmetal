//! OpenAI Responses API: `POST /v1/responses`.
//!
//! A response is a list of output items, not a single message: the model's
//! reasoning, its answer as a `message` item of `output_text`, and each tool
//! call as a `function_call` item. The request side is items too: a string, or
//! a list of messages, earlier calls (`function_call`) and their results
//! (`function_call_output`), and reasoning to carry over. All of it maps onto
//! the same chat template, assembler, tool parsers and grammars as
//! `/v1/chat/completions`; only the wire differs, including the streaming
//! events (`response.created`, `response.output_item.added`,
//! `response.output_text.delta`, ..., `response.completed`).
//!
//! Scope: text and image input, function tools, `tool_choice`, `text.format`
//! (`json_object`, `json_schema`), `reasoning.effort`, `max_output_tokens`,
//! sampling. A response is not stored: `previous_response_id`, `conversation`,
//! `background`, built-in tools and the other server-side features are refused
//! by name rather than ignored.

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Json, Response};
use chrono::Utc;
use pmetal_data::chat_templates::{FunctionDefinition, ToolDefinition};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::assemble::{Assembler, AssemblerConfig, ChoiceDone, ChoiceEvent, Failure, Piece};
use crate::error::ServeError;
use crate::extract::ApiJson;
use crate::generate;
use crate::request::{ResolvedStops, SamplingRequest, reject_unsupported};
use crate::routes::{AppState, StreamTotals, merged_sse, new_call_id, sse_response};
use crate::tools::{ToolChoice, ToolPlan};
use crate::types::{ChatMessage, JsonSchemaSpec, ResponseFormat};

/// `input`: a string (one user message) or a list of items.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<Value>),
}

/// The `text` field: the output's format.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TextConfig {
    #[serde(default)]
    pub format: Option<Value>,
}

/// The `reasoning` field.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ReasoningConfig {
    #[serde(default)]
    pub effort: Option<String>,
}

/// `POST /v1/responses` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub text: Option<TextConfig>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub include: Option<Vec<String>>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

/// Fields of the Responses API that change what is generated or kept, and
/// that this server does not implement.
const UNSUPPORTED: &[(&str, &str)] = &[
    (
        "previous_response_id",
        "responses are not stored, so there is nothing to continue; send the whole conversation as `input`",
    ),
    (
        "conversation",
        "conversations are not stored; send the whole conversation as `input`",
    ),
    (
        "background",
        "responses run in the request, not in the background",
    ),
    ("prompt", "stored prompts are not implemented"),
    (
        "context_management",
        "context compaction is not implemented",
    ),
    (
        "top_logprobs",
        "log-probabilities are not reported on this endpoint",
    ),
];

fn bad_item(index: usize, why: impl std::fmt::Display) -> ServeError {
    ServeError::invalid(format!("input.{index}"), why.to_string())
}

/// One `content` of an input message as the chat message parser takes it: a
/// string, or parts of `text` and `image_url`.
fn content_value(index: usize, content: &Value) -> Result<Value, ServeError> {
    let parts = match content {
        Value::String(_) | Value::Null => return Ok(content.clone()),
        Value::Array(parts) => parts,
        _ => return Err(bad_item(index, "`content` is a string or a list of parts")),
    };
    let converted = parts
        .iter()
        .enumerate()
        .map(|(j, part)| match part["type"].as_str() {
            Some("input_text" | "output_text" | "text") => Ok(json!({
                "type": "text",
                "text": part["text"].as_str().unwrap_or_default(),
            })),
            Some("input_image") => Ok(part.clone()),
            Some("refusal") => Ok(json!({
                "type": "text",
                "text": part["refusal"].as_str().unwrap_or_default(),
            })),
            Some(other) => Err(ServeError::unsupported(
                format!("input.{index}.content.{j}.type"),
                format!("`{other}` content is not supported"),
            )),
            None => Err(bad_item(index, "a content part needs a `type`")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value::Array(converted))
}

fn message_from(value: Value) -> Result<ChatMessage, ServeError> {
    serde_json::from_value(value).map_err(|e| ServeError::BadRequest(e.to_string()))
}

/// The request's `instructions` and `input` as chat messages.
fn input_messages(
    instructions: Option<&str>,
    input: &ResponsesInput,
) -> Result<Vec<ChatMessage>, ServeError> {
    let mut messages = Vec::new();
    if let Some(instructions) = instructions.filter(|i| !i.is_empty()) {
        messages.push(ChatMessage::text("system", instructions));
    }
    let items = match input {
        ResponsesInput::Text(text) => {
            messages.push(ChatMessage::text("user", text.clone()));
            return Ok(messages);
        }
        ResponsesInput::Items(items) => items,
    };
    // Reasoning items ride on the assistant turn that follows them.
    let mut pending_reasoning = String::new();
    for (i, item) in items.iter().enumerate() {
        let kind = item["type"].as_str().unwrap_or("message");
        match kind {
            "message" => {
                let role = match item["role"].as_str() {
                    Some("developer") => "system",
                    Some(role @ ("user" | "assistant" | "system")) => role,
                    other => return Err(bad_item(i, format!("a message's role is not {other:?}"))),
                };
                let mut message = message_from(json!({
                    "role": role,
                    "content": content_value(i, &item["content"])?,
                }))?;
                if role == "assistant" && !pending_reasoning.is_empty() {
                    message.reasoning_content = Some(std::mem::take(&mut pending_reasoning));
                }
                messages.push(message);
            }
            "function_call" => {
                let call = json!({
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                });
                // Calls made together are one assistant turn.
                let joins = messages.last().is_some_and(|m| {
                    m.role == "assistant" && m.tool_calls.is_some() && m.content.is_empty()
                });
                let call = serde_json::from_value(call)
                    .map_err(|e| bad_item(i, format!("a function_call item: {e}")))?;
                if joins {
                    if let Some(calls) = messages.last_mut().and_then(|m| m.tool_calls.as_mut()) {
                        calls.push(call);
                    }
                } else {
                    let mut message = ChatMessage::text("assistant", "");
                    message.tool_calls = Some(vec![call]);
                    if !pending_reasoning.is_empty() {
                        message.reasoning_content = Some(std::mem::take(&mut pending_reasoning));
                    }
                    messages.push(message);
                }
            }
            "function_call_output" => {
                let output = match &item["output"] {
                    Value::String(text) => text.clone(),
                    Value::Array(parts) => parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n"),
                    other => other.to_string(),
                };
                let mut message = ChatMessage::text("tool", output);
                message.tool_call_id = item["call_id"].as_str().map(str::to_owned);
                messages.push(message);
            }
            "reasoning" => {
                let texts = item["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .chain(item["summary"].as_array().into_iter().flatten())
                    .filter_map(|p| p["text"].as_str());
                for text in texts {
                    pending_reasoning.push_str(text);
                }
            }
            other => {
                return Err(ServeError::unsupported(
                    format!("input.{i}.type"),
                    format!("`{other}` items are not supported"),
                ));
            }
        }
    }
    if messages.is_empty() {
        return Err(ServeError::invalid("input", "`input` holds no messages"));
    }
    Ok(messages)
}

/// The function tools of a request (the Responses shape is flat: `name`,
/// `description`, `parameters` beside `type`).
fn function_tools(tools: &[Value]) -> Result<Vec<ToolDefinition>, ServeError> {
    tools
        .iter()
        .enumerate()
        .map(|(i, tool)| {
            if tool["type"] != "function" {
                return Err(ServeError::unsupported(
                    format!("tools.{i}.type"),
                    format!(
                        "`{}` tools are not implemented; only `function`",
                        tool["type"].as_str().unwrap_or("?")
                    ),
                ));
            }
            let name = tool["name"].as_str().ok_or_else(|| {
                ServeError::invalid(format!("tools.{i}.name"), "a function tool needs a `name`")
            })?;
            Ok(ToolDefinition {
                tool_type: "function".into(),
                function: FunctionDefinition {
                    name: name.to_owned(),
                    description: tool["description"].as_str().map(str::to_owned),
                    parameters: tool.get("parameters").filter(|p| !p.is_null()).cloned(),
                },
            })
        })
        .collect()
}

/// `text.format` as a chat completions `response_format`.
fn response_format(text: Option<&TextConfig>) -> Result<Option<ResponseFormat>, ServeError> {
    let Some(format) = text.and_then(|t| t.format.as_ref()) else {
        return Ok(None);
    };
    match format["type"].as_str() {
        Some("text") | None => Ok(None),
        Some("json_object") => Ok(Some(ResponseFormat::JsonObject)),
        Some("json_schema") => Ok(Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: format["name"].as_str().unwrap_or("response").to_owned(),
                description: format["description"].as_str().map(str::to_owned),
                schema: format.get("schema").cloned(),
                strict: format["strict"].as_bool(),
            },
        })),
        Some(other) => Err(ServeError::invalid_with_code(
            "text.format.type",
            "invalid_value",
            format!("`text.format.type` is text, json_object or json_schema, not `{other}`"),
        )),
    }
}

fn tool_plan(
    req: &ResponsesRequest,
    tools: Vec<ToolDefinition>,
    format: crate::tools::ToolFormat,
) -> Result<ToolPlan, ServeError> {
    let bad = |why: String| ServeError::invalid_with_code("tool_choice", "invalid_value", why);
    let choice = match &req.tool_choice {
        None => ToolChoice::Auto,
        Some(Value::String(mode)) => match mode.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => {
                return Err(bad(format!(
                    "`tool_choice` is auto, none, required or a function, not `{other}`"
                )));
            }
        },
        Some(object) => match object["type"].as_str() {
            Some("function") => ToolChoice::Named(
                object["name"]
                    .as_str()
                    .ok_or_else(|| bad("a named `tool_choice` needs `name`".into()))?
                    .to_owned(),
            ),
            other => {
                return Err(bad(format!(
                    "`tool_choice.type` must be `function`, not {other:?}"
                )));
            }
        },
    };
    ToolPlan::new(
        tools,
        choice,
        req.parallel_tool_calls == Some(false),
        format,
    )
    .map_err(|why| ServeError::invalid("tool_choice", why))
}

// ────────────────────────────────────────────────────────────────────────────
// The response and its events
// ────────────────────────────────────────────────────────────────────────────

/// What the response says back about the request.
#[derive(Debug, Clone)]
struct Echo {
    model: String,
    instructions: Option<String>,
    max_output_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    parallel_tool_calls: bool,
    tool_choice: Value,
    tools: Value,
    text: Value,
    reasoning_effort: Option<String>,
}

enum Open {
    Reasoning {
        id: String,
        text: String,
        index: usize,
    },
    Message {
        id: String,
        text: String,
        index: usize,
    },
}

/// Builds the response and its streaming events from a choice's pieces; the
/// non-streaming reply is the same run, kept as its last state.
struct ResponseBuilder {
    id: String,
    created_at: i64,
    echo: Echo,
    output: Vec<Value>,
    open: Option<Open>,
    sequence: u64,
}

fn item_id(prefix: &str) -> String {
    format!(
        "{prefix}_{}",
        &uuid::Uuid::new_v4().simple().to_string()[..24]
    )
}

impl ResponseBuilder {
    fn new(echo: Echo) -> Self {
        Self {
            id: item_id("resp"),
            created_at: Utc::now().timestamp(),
            echo,
            output: Vec::new(),
            open: None,
            sequence: 0,
        }
    }

    /// The event `name` with `fields`, numbered.
    fn event(&mut self, name: &str, fields: Value) -> (String, Value) {
        let mut body = json!({"type": name, "sequence_number": self.sequence});
        self.sequence += 1;
        if let (Value::Object(into), Value::Object(from)) = (&mut body, fields) {
            into.extend(from);
        }
        (name.to_string(), body)
    }

    /// The response object in `status`.
    fn object(
        &self,
        status: &str,
        usage: Option<Value>,
        incomplete: Option<&str>,
        error: Option<Value>,
    ) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "error": error,
            "incomplete_details": incomplete.map(|reason| json!({"reason": reason})),
            "instructions": self.echo.instructions,
            "max_output_tokens": self.echo.max_output_tokens,
            "model": self.echo.model,
            "output": self.output,
            "parallel_tool_calls": self.echo.parallel_tool_calls,
            "previous_response_id": null,
            "reasoning": {"effort": self.echo.reasoning_effort, "summary": null},
            "store": false,
            "temperature": self.echo.temperature,
            "text": self.echo.text,
            "tool_choice": self.echo.tool_choice,
            "tools": self.echo.tools,
            "top_p": self.echo.top_p,
            "truncation": "disabled",
            "usage": usage,
            "metadata": {},
        })
    }

    fn start(&mut self) -> Vec<(String, Value)> {
        let response = self.object("in_progress", None, None, None);
        vec![
            self.event("response.created", json!({"response": response.clone()})),
            self.event("response.in_progress", json!({"response": response})),
        ]
    }

    /// Close the open reasoning or message item.
    fn close(&mut self) -> Vec<(String, Value)> {
        let mut events = Vec::new();
        match self.open.take() {
            Some(Open::Reasoning { id, text, index }) => {
                events.push(self.event(
                    "response.reasoning_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                ));
                let item = json!({
                    "id": id, "type": "reasoning", "summary": [],
                    "content": [{"type": "reasoning_text", "text": text}],
                });
                self.output.push(item.clone());
                events.push(self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                ));
            }
            Some(Open::Message { id, text, index }) => {
                let part =
                    json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []});
                events.push(self.event(
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                           "text": text, "logprobs": []}),
                ));
                events.push(self.event(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": part.clone()}),
                ));
                let item = json!({
                    "id": id, "type": "message", "status": "completed", "role": "assistant",
                    "content": [part],
                });
                self.output.push(item.clone());
                events.push(self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                ));
            }
            None => {}
        }
        events
    }

    /// Open the answer's `message` item unless it is open.
    fn open_message(&mut self) -> Vec<(String, Value)> {
        if matches!(self.open, Some(Open::Message { .. })) {
            return Vec::new();
        }
        let mut events = self.close();
        let (id, index) = (item_id("msg"), self.output.len());
        events.push(self.event(
            "response.output_item.added",
            json!({"output_index": index, "item": {
                "id": id, "type": "message", "status": "in_progress",
                "role": "assistant", "content": []}}),
        ));
        events.push(self.event(
            "response.content_part.added",
            json!({"item_id": id, "output_index": index, "content_index": 0,
                   "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
        ));
        self.open = Some(Open::Message {
            id,
            text: String::new(),
            index,
        });
        events
    }

    fn piece(&mut self, piece: Piece) -> Vec<(String, Value)> {
        let mut events = Vec::new();
        match piece {
            Piece::Reasoning(delta) => {
                if !matches!(self.open, Some(Open::Reasoning { .. })) {
                    events.extend(self.close());
                    let (id, index) = (item_id("rs"), self.output.len());
                    events.push(self.event(
                        "response.output_item.added",
                        json!({"output_index": index,
                               "item": {"id": id, "type": "reasoning", "summary": [], "content": []}}),
                    ));
                    self.open = Some(Open::Reasoning {
                        id,
                        text: String::new(),
                        index,
                    });
                }
                if let Some(Open::Reasoning { id, text, index }) = &mut self.open {
                    text.push_str(&delta);
                    let (id, index) = (id.clone(), *index);
                    events.push(self.event(
                        "response.reasoning_text.delta",
                        json!({"item_id": id, "output_index": index, "content_index": 0, "delta": delta}),
                    ));
                }
            }
            Piece::Content { text: delta, .. } => {
                events.extend(self.open_message());
                if let Some(Open::Message { id, text, index }) = &mut self.open {
                    text.push_str(&delta);
                    let (id, index) = (id.clone(), *index);
                    if !delta.is_empty() {
                        events.push(self.event(
                            "response.output_text.delta",
                            json!({"item_id": id, "output_index": index, "content_index": 0,
                                   "delta": delta, "logprobs": []}),
                        ));
                    }
                }
            }
            Piece::ToolCall(call) => {
                events.extend(self.close());
                let (id, index) = (item_id("fc"), self.output.len());
                let call_id = new_call_id();
                events.push(self.event(
                    "response.output_item.added",
                    json!({"output_index": index, "item": {
                        "id": id, "type": "function_call", "status": "in_progress",
                        "call_id": call_id, "name": call.name, "arguments": ""}}),
                ));
                events.push(self.event(
                    "response.function_call_arguments.delta",
                    json!({"item_id": id, "output_index": index, "delta": call.arguments}),
                ));
                events.push(self.event(
                    "response.function_call_arguments.done",
                    json!({"item_id": id, "output_index": index, "name": call.name,
                           "arguments": call.arguments}),
                ));
                let item = json!({
                    "id": id, "type": "function_call", "status": "completed",
                    "call_id": call_id, "name": call.name, "arguments": call.arguments,
                });
                self.output.push(item.clone());
                events.push(self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                ));
            }
        }
        events
    }

    /// The choice ended: close its last item and say how the response ended.
    fn finish(&mut self, done: &ChoiceDone, prompt_tokens: usize) -> Vec<(String, Value)> {
        // A reply with nothing in it (a budget spent on text that decodes to
        // nothing) is still a message, as it is in OpenAI's API.
        let mut events = if self.output.is_empty() && self.open.is_none() {
            self.open_message()
        } else {
            Vec::new()
        };
        events.extend(self.close());
        let usage = json!({
            "input_tokens": prompt_tokens,
            "input_tokens_details": {
                "cached_tokens": done.metrics.as_ref().map_or(0, |m| m.cached_prompt_tokens)
            },
            "output_tokens": done.finish.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": done.finish.reasoning_tokens},
            "total_tokens": prompt_tokens + done.finish.completion_tokens,
        });
        let (name, status, incomplete) = if done.finish.reason == "length" {
            (
                "response.incomplete",
                "incomplete",
                Some("max_output_tokens"),
            )
        } else {
            ("response.completed", "completed", None)
        };
        let response = self.object(status, Some(usage), incomplete, None);
        events.push(self.event(name, json!({"response": response})));
        events
    }

    fn fail(&mut self, failure: &Failure) -> Vec<(String, Value)> {
        let error = match failure {
            Failure::Timeout(secs) => ServeError::Timeout(*secs),
            Failure::Engine(message) => {
                tracing::error!("Responses generation error: {message}");
                ServeError::Internal(message.clone())
            }
        };
        let info = error.describe();
        let response = self.object(
            "failed",
            None,
            None,
            Some(json!({"code": info.kind, "message": info.message})),
        );
        vec![self.event("response.failed", json!({"response": response}))]
    }
}

fn sse_event((name, body): (String, Value)) -> Event {
    Event::default().event(name).data(body.to_string())
}

// ────────────────────────────────────────────────────────────────────────────
// Handler
// ────────────────────────────────────────────────────────────────────────────

/// `POST /v1/responses`.
pub async fn responses(
    State(state): State<Arc<AppState>>,
    ApiJson(req): ApiJson<ResponsesRequest>,
) -> Result<Response, ServeError> {
    state.engine.check_model(&req.model)?;
    reject_unsupported(&req.other, UNSUPPORTED)?;
    if req.include.as_ref().is_some_and(|i| !i.is_empty()) {
        return Err(ServeError::unsupported(
            "include",
            "no extra output (logprobs, encrypted reasoning, search results) can be included",
        ));
    }
    let messages = input_messages(req.instructions.as_deref(), &req.input)?;
    let tools = function_tools(req.tools.as_deref().unwrap_or_default())?;
    let plan = tool_plan(&req, tools, state.engine.tool_format())?;
    let format = response_format(req.text.as_ref())?;
    let stream = req.stream.unwrap_or(false);
    let permit = state.try_acquire_request_permit()?;

    let mut kwargs = pmetal_data::chat_templates::ChatTemplateKwargs::new();
    if let Some(effort) = req.reasoning.as_ref().and_then(|r| r.effort.as_deref()) {
        kwargs.set_default(
            pmetal_data::chat_templates::ChatTemplateKwargs::REASONING_EFFORT,
            effort,
        );
    }
    let prepared = generate::prepare(
        &state,
        &messages,
        &plan,
        &kwargs,
        &SamplingRequest {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            max_tokens: req.max_output_tokens,
            ..SamplingRequest::default()
        },
        &ResolvedStops::default(),
        format.as_ref(),
    )
    .await?;
    let prompt_tokens = prepared.prompt.input_ids.len();

    let echo = Echo {
        model: state.engine.model_id().to_string(),
        instructions: req.instructions.clone(),
        max_output_tokens: req.max_output_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        parallel_tool_calls: req.parallel_tool_calls != Some(false),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| json!("auto")),
        tools: Value::Array(req.tools.clone().unwrap_or_default()),
        text: json!({"format": req.text.as_ref().and_then(|t| t.format.clone()).unwrap_or_else(|| json!({"type": "text"}))}),
        reasoning_effort: req.reasoning.as_ref().and_then(|r| r.effort.clone()),
    };
    let mut builder = ResponseBuilder::new(echo);
    let make_assembler = |_: usize| {
        Assembler::new(AssemblerConfig {
            tokenizer: state.engine.tokenizer_arc(),
            prompt: prepared.prompt.input_ids.clone(),
            stops: Vec::new(),
            tools: plan.parse.clone(),
            single_tool_call: plan.single_call,
        })
    };
    let mut choices = generate::start_choices(
        &state,
        &prepared.prompt,
        &prepared.params,
        1,
        make_assembler,
    );
    let choice = choices.pop().expect("one choice");

    if stream {
        let opening: Vec<Event> = builder.start().into_iter().map(sse_event).collect();
        let convert = move |_: usize, event: ChoiceEvent, _: &mut StreamTotals| -> Vec<Event> {
            let events = match event {
                ChoiceEvent::Piece(piece) => builder.piece(piece),
                ChoiceEvent::PromptLogprobs(_) => Vec::new(),
                ChoiceEvent::Done(done) => builder.finish(&done, prompt_tokens),
                ChoiceEvent::Failed(failure) => builder.fail(&failure),
            };
            events.into_iter().map(sse_event).collect()
        };
        // The Responses stream has no `[DONE]`: `response.completed` ends it.
        let body = merged_sse(permit, opening, vec![choice], convert, |_| Vec::new());
        return Ok(sse_response(body));
    }

    let mut pieces = Vec::new();
    let mut done = None;
    {
        use futures::StreamExt as _;
        futures::pin_mut!(choice);
        while let Some(event) = choice.next().await {
            match event {
                ChoiceEvent::Piece(piece) => pieces.push(piece),
                ChoiceEvent::PromptLogprobs(_) => {}
                ChoiceEvent::Done(d) => done = Some(d),
                ChoiceEvent::Failed(failure) => return Err(failure.into()),
            }
        }
    }
    let done =
        done.ok_or_else(|| ServeError::Internal("generation ended without a result".into()))?;
    let _ = builder.start();
    for piece in pieces {
        let _ = builder.piece(piece);
    }
    let events = builder.finish(&done, prompt_tokens);
    let response = events
        .into_iter()
        .last()
        .map(|(_, body)| body["response"].clone())
        .unwrap_or(Value::Null);
    Ok(Json(response).into_response())
}
