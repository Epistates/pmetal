//! Anthropic-compatible `/v1/messages` endpoint.
//!
//! Accepts the Anthropic Messages API request shape (string or block
//! content, a `system` prompt as text or blocks, `tools` with `tool_choice`,
//! `thinking`) and runs it through the same engine, assembler and tool-call
//! parsers as `/v1/chat/completions`. Only the wire differs: content blocks
//! (`thinking`, `text`, `tool_use`), `stop_reason`, and the streaming events
//! `message_start`, `content_block_start`, `content_block_delta`,
//! `content_block_stop`, `message_delta`, `message_stop`, `ping`, `error`.
//!
//! Scope: text, images (base64 `image` blocks, for a model that reads them;
//! see [`crate::media`]), tool use and tool results, thinking. Server-side
//! tools (web search, code execution), documents, MCP connectors and batches
//! are out of scope and refused by name.

use crate::assemble::{Assembler, AssemblerConfig, ChoiceEvent, Failure, Piece};
use crate::engine::InferenceEngine;
use crate::error::ServeError;
use crate::extract::ApiJson;
use crate::generate;
use crate::request::{SamplingRequest, resolve_stop_sequences, sampling_params};
use crate::routes::{AppState, StreamTotals, merged_sse};
use crate::tools::{ParsedCall, ToolChoice, ToolPlan};
use crate::types::ChatMessage;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use pmetal_data::chat_templates::{
    ChatTemplateKwargs, FunctionDefinition, ToolCall, ToolDefinition,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::Arc;

// ────────────────────────────────────────────────────────────────────────────
// Request types
// ────────────────────────────────────────────────────────────────────────────

/// `system`: a string, or a list of text blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<Value>),
}

impl SystemPrompt {
    fn text(&self) -> Result<String, ServeError> {
        match self {
            Self::Text(text) => Ok(text.clone()),
            Self::Blocks(blocks) => blocks
                .iter()
                .enumerate()
                .map(
                    |(i, block)| match (block["type"].as_str(), block["text"].as_str()) {
                        (Some("text"), Some(text)) => Ok(text.to_owned()),
                        _ => Err(ServeError::invalid(
                            format!("system.{i}"),
                            "system blocks must be {\"type\": \"text\", \"text\": ...}",
                        )),
                    },
                )
                .collect::<Result<Vec<_>, _>>()
                .map(|texts| texts.join("\n\n")),
        }
    }
}

/// A message's `content`: a string, or a list of blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    String(String),
    Blocks(Vec<Value>),
}

/// Anthropic message.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

/// A tool as the Messages API defines it.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Value>,
    /// `custom` (or absent) for a client tool; anything else is a server tool.
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
}

/// `tool_choice`.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicToolChoice {
    /// `auto`, `any`, `tool` or `none`.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub disable_parallel_tool_use: Option<bool>,
}

/// The `thinking` field of a messages request.
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkingConfig {
    /// `"enabled"` or `"disabled"`.
    #[serde(rename = "type")]
    pub kind: String,
}

/// Anthropic `/v1/messages` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: usize,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Anthropic uses `stop_sequences` (plural) where OpenAI uses `stop`.
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    /// `{"type": "enabled", "budget_tokens": N}` or `{"type": "disabled"}`:
    /// the chat template's `enable_thinking`. Absent leaves the template's
    /// default.
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl MessagesRequest {
    /// The chat template kwargs this request asks for.
    fn template_kwargs(&self) -> Result<ChatTemplateKwargs, ServeError> {
        let mut kwargs = ChatTemplateKwargs::new();
        match self.thinking.as_ref().map(|t| t.kind.as_str()) {
            None => {}
            Some("enabled") => kwargs.set(ChatTemplateKwargs::ENABLE_THINKING, true),
            Some("disabled") => kwargs.set(ChatTemplateKwargs::ENABLE_THINKING, false),
            Some(other) => {
                return Err(ServeError::invalid(
                    "thinking.type",
                    format!("thinking.type must be \"enabled\" or \"disabled\", not {other:?}"),
                ));
            }
        }
        Ok(kwargs)
    }
}

/// Fields of the Messages API that change what is generated and that this
/// server does not implement.
const UNSUPPORTED: &[(&str, &str)] = &[
    ("mcp_servers", "MCP connectors are not implemented"),
    ("container", "code execution containers are not implemented"),
];

// ────────────────────────────────────────────────────────────────────────────
// Messages into the chat template's shape
// ────────────────────────────────────────────────────────────────────────────

/// The image URL an `image` block's `source` stands for: a `data:` URI for a
/// base64 source, the URL itself for a `url` source (which
/// [`crate::media`] then refuses unless the server fetches).
fn image_source_url(source: &Value) -> Result<String, String> {
    let field = |key: &str| source.get(key).and_then(Value::as_str);
    match field("type") {
        Some("base64") => {
            let media_type = field("media_type").ok_or("a base64 image needs a media_type")?;
            let data = field("data").ok_or("a base64 image needs its data")?;
            Ok(format!("data:{media_type};base64,{data}"))
        }
        Some("url") => Ok(field("url").ok_or("a url image needs its url")?.to_owned()),
        Some(other) => Err(format!(
            "image sources of type {other:?} are not supported; send the image as base64"
        )),
        None => Err("an image block needs a source with a type".into()),
    }
}

/// The text of a `tool_result`'s `content`: a string, or text blocks.
fn tool_result_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

impl AnthropicMessage {
    /// The message as chat messages. Text and images stay in one message in
    /// order (images as content parts, each a `data:` URI); an assistant
    /// turn's `tool_use` blocks become its `tool_calls`, its `thinking` its
    /// `reasoning_content`; each `tool_result` of a user turn becomes a
    /// `tool` message ahead of the rest of the turn, as the API requires
    /// them first. `index` names the message in errors.
    fn to_chat_messages(&self, index: usize) -> Result<Vec<ChatMessage>, ServeError> {
        let blocks = match &self.content {
            MessageContent::String(text) => {
                return Ok(vec![ChatMessage::text(self.role.clone(), text.clone())]);
            }
            MessageContent::Blocks(blocks) => blocks,
        };
        let mut tool_messages = Vec::new();
        let mut parts: Vec<Value> = Vec::new();
        let mut text = String::new();
        let mut has_media = false;
        let mut reasoning = String::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        for (j, block) in blocks.iter().enumerate() {
            let at = |why: &str| {
                ServeError::invalid(format!("messages.{index}.content.{j}"), why.to_string())
            };
            match block["type"].as_str() {
                Some("text") => {
                    let t = block["text"]
                        .as_str()
                        .ok_or_else(|| at("a text block needs `text`"))?;
                    parts.push(json!({"type": "text", "text": t}));
                    text.push_str(t);
                }
                Some("image") => {
                    let url = image_source_url(&block["source"]).map_err(|why| at(&why))?;
                    parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                    has_media = true;
                }
                Some("thinking") => {
                    reasoning.push_str(block["thinking"].as_str().unwrap_or_default());
                }
                // Redacted thinking is opaque to every model but the one that
                // wrote it.
                Some("redacted_thinking") => {}
                Some("tool_use") => {
                    let name = block["name"]
                        .as_str()
                        .ok_or_else(|| at("a tool_use block needs `name`"))?;
                    calls.push(ToolCall {
                        id: block["id"].as_str().map(str::to_owned),
                        tool_type: "function".into(),
                        function: pmetal_data::chat_templates::FunctionCall {
                            name: name.to_owned(),
                            arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
                        },
                    });
                }
                Some("tool_result") => {
                    let id = block["tool_use_id"]
                        .as_str()
                        .ok_or_else(|| at("a tool_result block needs `tool_use_id`"))?;
                    let mut message =
                        ChatMessage::text("tool", tool_result_text(&block["content"]));
                    message.tool_call_id = Some(id.to_owned());
                    tool_messages.push(message);
                }
                Some(other) => {
                    return Err(ServeError::unsupported(
                        format!("messages.{index}.content.{j}.type"),
                        format!("`{other}` blocks are not supported"),
                    ));
                }
                None => return Err(at("a content block needs a string `type`")),
            }
        }
        let mut out = tool_messages;
        if !text.is_empty()
            || has_media
            || !calls.is_empty()
            || !reasoning.is_empty()
            || out.is_empty()
        {
            let mut message = ChatMessage::text(self.role.clone(), text);
            if has_media {
                message.parts = Some(parts);
            }
            if !calls.is_empty() {
                message.tool_calls = Some(calls);
            }
            if !reasoning.is_empty() {
                message.reasoning_content = Some(reasoning);
            }
            out.push(message);
        }
        Ok(out)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Response types
// ────────────────────────────────────────────────────────────────────────────

/// Block inside an assistant response — `thinking`, `text` or `tool_use`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    /// The model's reasoning, ahead of its answer. A local model's thinking
    /// is not signed, so `signature` is empty.
    Thinking {
        thinking: String,
        signature: String,
    },
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

/// Token counts for an Anthropic response.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cache_creation_input_tokens: usize,
    pub cache_read_input_tokens: usize,
}

impl AnthropicUsage {
    fn new(input_tokens: usize, output_tokens: usize) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }
    }

    /// `cached` of the input tokens were read from the cache (and so are not
    /// counted in `input_tokens`, as Anthropic counts them).
    fn with_cache_read(mut self, cached: usize) -> Self {
        let cached = cached.min(self.input_tokens);
        self.input_tokens -= cached;
        self.cache_read_input_tokens = cached;
        self
    }
}

/// Anthropic `/v1/messages` non-streaming response.
#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: &'static str,
    pub role: &'static str,
    pub content: Vec<ResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<&'static str>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// Map a choice's finish reason to Anthropic's `stop_reason`.
fn to_stop_reason(finish: &str, stop_sequence: &Option<String>) -> &'static str {
    match finish {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ if stop_sequence.is_some() => "stop_sequence",
        _ => "end_turn",
    }
}

/// A call's `input`: the arguments, as the object they are.
fn call_input(call: &ParsedCall) -> Value {
    serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}))
}

fn tool_use_id() -> String {
    format!("toolu_{}", uuid::Uuid::new_v4().simple())
}

// ────────────────────────────────────────────────────────────────────────────
// Streaming event types
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MessageEvent {
    MessageStart {
        message: MessagesResponse,
    },
    ContentBlockStart {
        index: usize,
        content_block: ResponseContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: DeltaBlock,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDeltaPayload,
        usage: MessageDeltaUsage,
    },
    MessageStop,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
enum DeltaBlock {
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Debug, Clone, Serialize)]
struct MessageDeltaPayload {
    stop_reason: Option<&'static str>,
    stop_sequence: Option<String>,
}

/// `message_delta`'s usage: cumulative output tokens.
#[derive(Debug, Clone, Serialize)]
struct MessageDeltaUsage {
    output_tokens: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Handler
// ────────────────────────────────────────────────────────────────────────────

/// The tools and choice of a request as the plan the chat endpoint uses.
fn plan_tools(req: &MessagesRequest, engine: &InferenceEngine) -> Result<ToolPlan, ServeError> {
    let defs: Vec<ToolDefinition> = req
        .tools
        .iter()
        .flatten()
        .enumerate()
        .map(|(i, tool)| {
            if tool.kind.as_deref().is_some_and(|k| k != "custom") {
                return Err(ServeError::unsupported(
                    format!("tools.{i}.type"),
                    format!(
                        "server tools (`{}`) are not implemented",
                        tool.kind.as_deref().unwrap_or("")
                    ),
                ));
            }
            Ok(ToolDefinition {
                tool_type: "function".into(),
                function: FunctionDefinition {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.input_schema.clone(),
                },
            })
        })
        .collect::<Result<_, _>>()?;
    let choice = match req.tool_choice.as_ref() {
        None => ToolChoice::Auto,
        Some(choice) => match choice.kind.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "any" => ToolChoice::Required,
            "tool" => ToolChoice::Named(choice.name.clone().ok_or_else(|| {
                ServeError::invalid(
                    "tool_choice.name",
                    "`tool_choice` of type `tool` needs `name`",
                )
            })?),
            other => {
                return Err(ServeError::invalid(
                    "tool_choice.type",
                    format!("`tool_choice.type` must be auto, any, tool or none, not `{other}`"),
                ));
            }
        },
    };
    let single = req
        .tool_choice
        .as_ref()
        .and_then(|c| c.disable_parallel_tool_use)
        .unwrap_or(false);
    let plan = ToolPlan::new(defs, choice, single, engine.tool_format())
        .map_err(|why| ServeError::invalid("tool_choice", why))?;
    if matches!(plan.choice, ToolChoice::Required | ToolChoice::Named(_)) {
        return Err(ServeError::unsupported(
            "tool_choice",
            "`any` and `tool` need constrained decoding",
        ));
    }
    Ok(plan)
}

/// `POST /v1/messages` — Anthropic-compatible message generation.
pub async fn messages(
    State(state): State<Arc<AppState>>,
    ApiJson(req): ApiJson<MessagesRequest>,
) -> Result<Response, ServeError> {
    state.engine.check_model(&req.model)?;
    // An empty list of connectors connects nothing.
    let mut requested = req.other.clone();
    requested.retain(|_, value| value.as_array().is_none_or(|list| !list.is_empty()));
    crate::request::reject_unsupported(&requested, UNSUPPORTED)?;
    if req.temperature.is_some_and(|t| !(0.0..=1.0).contains(&t)) {
        return Err(ServeError::invalid(
            "temperature",
            "`temperature` must be between 0 and 1",
        ));
    }
    let tool_plan = plan_tools(&req, &state.engine)?;
    let permit = state.try_acquire_request_permit()?;

    // Assemble the internal chat-message list: optional system prompt first,
    // then each Anthropic message.
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(req.messages.len() + 1);
    if let Some(system) = req.system.as_ref() {
        let system = system.text()?;
        if !system.is_empty() {
            messages.push(ChatMessage::text("system", system));
        }
    }
    for (i, m) in req.messages.iter().enumerate() {
        if m.role != "user" && m.role != "assistant" {
            return Err(ServeError::invalid(
                format!("messages.{i}.role"),
                "a message's role is `user` or `assistant`",
            ));
        }
        messages.extend(m.to_chat_messages(i)?);
    }
    if req.messages.is_empty() {
        return Err(ServeError::invalid(
            "messages",
            "`messages` must hold at least one message",
        ));
    }

    let template_kwargs = req.template_kwargs()?;
    let prompt = state
        .engine
        .prepare_chat(&messages, tool_plan.prompt.as_deref(), &template_kwargs)
        .await?;
    let prompt_tokens = prompt.input_ids.len();
    let thinking = state.engine.thinks_with(&template_kwargs);
    let stops = resolve_stop_sequences(req.stop_sequences.as_deref(), None, &state.engine)?;
    let params = sampling_params(
        &state.engine,
        &SamplingRequest {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: Some(req.max_tokens),
            ..SamplingRequest::default()
        },
        &stops,
        thinking,
        prompt_tokens,
    )?;
    state.engine.validate_sampling_params(&params)?;

    let request_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model_id = state.engine.model_id().to_string();
    let make_assembler = |_: usize| {
        Assembler::new(AssemblerConfig {
            tokenizer: state.engine.tokenizer_arc(),
            prompt: prompt.input_ids.clone(),
            stops: stops.sequences.clone(),
            tools: tool_plan.parse.clone(),
            single_tool_call: tool_plan.single_call,
        })
    };
    let mut choices = generate::start_choices(&state, &prompt, &params, 1, make_assembler);
    let choice = choices.pop().expect("one choice");

    if req.stream.unwrap_or(false) {
        let start = MessageEvent::MessageStart {
            message: MessagesResponse {
                id: request_id,
                message_type: "message",
                role: "assistant",
                content: Vec::new(),
                model: model_id,
                stop_reason: None,
                stop_sequence: None,
                usage: AnthropicUsage::new(prompt_tokens, 1),
            },
        };
        let mut blocks = BlockStream::default();
        let convert = move |_: usize, event: ChoiceEvent, _: &mut StreamTotals| -> Vec<Event> {
            match event {
                ChoiceEvent::Piece(piece) => {
                    blocks.events(piece).iter().map(encode_event).collect()
                }
                ChoiceEvent::Done(done) => {
                    let mut events = blocks.closing();
                    events.push(MessageEvent::MessageDelta {
                        delta: MessageDeltaPayload {
                            stop_reason: Some(to_stop_reason(
                                done.finish.reason,
                                &done.finish.stop_sequence,
                            )),
                            stop_sequence: done.finish.stop_sequence.clone(),
                        },
                        usage: MessageDeltaUsage {
                            output_tokens: done.finish.completion_tokens,
                        },
                    });
                    events.push(MessageEvent::MessageStop);
                    events.iter().map(encode_event).collect()
                }
                ChoiceEvent::Failed(failure) => vec![error_event(&failure)],
            }
        };
        let body = merged_sse(
            permit,
            vec![encode_event(&start)],
            vec![choice],
            convert,
            |_| Vec::new(),
        );
        return Ok(Sse::new(body)
            .keep_alive(
                KeepAlive::new().event(Event::default().event("ping").data(r#"{"type": "ping"}"#)),
            )
            .into_response());
    }

    let reply = generate::collect(choice).await?;
    let done = reply.done.clone().expect("a finished choice");
    let mut content = Vec::new();
    if !reply.reasoning.is_empty() {
        content.push(ResponseContentBlock::Thinking {
            thinking: reply.reasoning.clone(),
            signature: String::new(),
        });
    }
    if !reply.content.trim().is_empty() || reply.tool_calls.is_empty() {
        content.push(ResponseContentBlock::Text {
            text: reply.content.clone(),
        });
    }
    content.extend(
        reply
            .tool_calls
            .iter()
            .map(|call| ResponseContentBlock::ToolUse {
                id: tool_use_id(),
                name: call.name.clone(),
                input: call_input(call),
            }),
    );
    Ok(Json(MessagesResponse {
        id: request_id,
        message_type: "message",
        role: "assistant",
        content,
        model: model_id,
        stop_reason: Some(to_stop_reason(
            done.finish.reason,
            &done.finish.stop_sequence,
        )),
        stop_sequence: done.finish.stop_sequence,
        usage: AnthropicUsage::new(prompt_tokens, done.finish.completion_tokens)
            .with_cache_read(reply.cached_tokens()),
    })
    .into_response())
}

// ────────────────────────────────────────────────────────────────────────────
// Streaming
// ────────────────────────────────────────────────────────────────────────────

/// The content block a stream has open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum OpenBlock {
    #[default]
    None,
    Thinking,
    Text,
    ToolUse,
}

/// Opens, fills and closes a stream's content blocks: a `thinking` block
/// while the model reasons, a `text` block for its answer, a `tool_use`
/// block for each call.
#[derive(Debug, Default)]
struct BlockStream {
    open: OpenBlock,
    /// The index the next block takes.
    next: usize,
    opened_any: bool,
}

impl BlockStream {
    fn events(&mut self, piece: Piece) -> Vec<MessageEvent> {
        let mut events = Vec::new();
        match piece {
            Piece::Reasoning(text) if !text.is_empty() => {
                self.switch(OpenBlock::Thinking, None, &mut events);
                events.push(MessageEvent::ContentBlockDelta {
                    index: self.index(),
                    delta: DeltaBlock::Thinking { thinking: text },
                });
            }
            Piece::Content { text, .. } if !text.is_empty() => {
                self.switch(OpenBlock::Text, None, &mut events);
                events.push(MessageEvent::ContentBlockDelta {
                    index: self.index(),
                    delta: DeltaBlock::Text { text },
                });
            }
            Piece::ToolCall(call) => {
                // A call is a block of its own, however many came before.
                self.switch(OpenBlock::None, None, &mut events);
                self.switch(
                    OpenBlock::ToolUse,
                    Some(ResponseContentBlock::ToolUse {
                        id: tool_use_id(),
                        name: call.name.clone(),
                        input: json!({}),
                    }),
                    &mut events,
                );
                events.push(MessageEvent::ContentBlockDelta {
                    index: self.index(),
                    delta: DeltaBlock::InputJson {
                        partial_json: call.arguments,
                    },
                });
            }
            _ => {}
        }
        events
    }

    /// Close the open block. A reply with no block at all gets an empty text
    /// block first, so every reply carries one.
    fn closing(&mut self) -> Vec<MessageEvent> {
        let mut events = Vec::new();
        if !self.opened_any {
            self.switch(OpenBlock::Text, None, &mut events);
        }
        self.switch(OpenBlock::None, None, &mut events);
        events
    }

    /// The index of the open block.
    fn index(&self) -> usize {
        self.next - 1
    }

    fn switch(
        &mut self,
        to: OpenBlock,
        start: Option<ResponseContentBlock>,
        events: &mut Vec<MessageEvent>,
    ) {
        if self.open == to && to != OpenBlock::ToolUse {
            return;
        }
        if self.open != OpenBlock::None {
            if self.open == OpenBlock::Thinking {
                // A thinking block ends with the signature that lets the API
                // verify it; a local model's is empty.
                events.push(MessageEvent::ContentBlockDelta {
                    index: self.index(),
                    delta: DeltaBlock::Signature {
                        signature: String::new(),
                    },
                });
            }
            events.push(MessageEvent::ContentBlockStop {
                index: self.index(),
            });
            self.open = OpenBlock::None;
        }
        if to == OpenBlock::None {
            return;
        }
        let content_block = start.unwrap_or_else(|| match to {
            OpenBlock::Thinking => ResponseContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            _ => ResponseContentBlock::Text {
                text: String::new(),
            },
        });
        events.push(MessageEvent::ContentBlockStart {
            index: self.next,
            content_block,
        });
        self.next += 1;
        self.open = to;
        self.opened_any = true;
    }
}

fn encode_event(ev: &MessageEvent) -> Event {
    // Name the SSE event per Anthropic's spec: the event line carries the
    // event type while the data line carries the JSON payload.
    let ty = match ev {
        MessageEvent::MessageStart { .. } => "message_start",
        MessageEvent::ContentBlockStart { .. } => "content_block_start",
        MessageEvent::ContentBlockDelta { .. } => "content_block_delta",
        MessageEvent::ContentBlockStop { .. } => "content_block_stop",
        MessageEvent::MessageDelta { .. } => "message_delta",
        MessageEvent::MessageStop => "message_stop",
    };
    Event::default()
        .event(ty)
        .data(serde_json::to_string(ev).unwrap_or_default())
}

/// Anthropic reports a failure after the stream began as an `error` event.
fn error_event(failure: &Failure) -> Event {
    let error: ServeError = match failure {
        Failure::Timeout(secs) => ServeError::Timeout(*secs),
        Failure::Engine(message) => {
            tracing::error!("Anthropic streaming generation error: {message}");
            ServeError::Internal(message.clone())
        }
    };
    Event::default()
        .event("error")
        .data(error.describe().anthropic_body().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> MessagesRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn message_content_string_deserializes() {
        let req =
            parse(r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#);
        let messages = req.messages[0].to_chat_messages(0).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "hi");
    }

    #[test]
    fn text_blocks_join_and_images_become_parts() {
        let req = parse(
            r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[
                {"type":"text","text":"hello"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},
                {"type":"text","text":"world"}]}]}"#,
        );
        let message = &req.messages[0].to_chat_messages(0).unwrap()[0];
        assert_eq!(message.content, "helloworld");
        let parts = message.parts.as_ref().unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn tool_use_and_tool_result_round_trip_into_chat_messages() {
        let req = parse(
            r#"{"model":"m","max_tokens":10,"messages":[
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"I should call it","signature":"x"},
                    {"type":"text","text":"Checking."},
                    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"SF"}}]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"72F"}]},
                    {"type":"text","text":"and tomorrow?"}]}]}"#,
        );
        let assistant = &req.messages[0].to_chat_messages(0).unwrap()[0];
        assert_eq!(assistant.content, "Checking.");
        assert_eq!(
            assistant.reasoning_content.as_deref(),
            Some("I should call it")
        );
        let call = &assistant.tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.id.as_deref(), Some("toolu_1"));
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(call.function.arguments["city"], "SF");

        let user = req.messages[1].to_chat_messages(1).unwrap();
        assert_eq!(user.len(), 2);
        assert_eq!(user[0].role, "tool");
        assert_eq!(user[0].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(user[0].content, "72F");
        assert_eq!(user[1].role, "user");
        assert_eq!(user[1].content, "and tomorrow?");
    }

    #[test]
    fn unknown_block_types_are_refused_by_name() {
        let req = parse(
            r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"document","source":{}}]}]}"#,
        );
        let err = req.messages[0].to_chat_messages(0).unwrap_err();
        assert!(
            matches!(err, ServeError::Invalid { param: Some(p), .. } if p == "messages.0.content.0.type")
        );
    }

    #[test]
    fn system_is_a_string_or_text_blocks() {
        let text = |json: &str| parse(json).system.unwrap().text().unwrap();
        assert_eq!(
            text(r#"{"model":"m","max_tokens":1,"messages":[],"system":"be brief"}"#),
            "be brief"
        );
        assert_eq!(
            text(
                r#"{"model":"m","max_tokens":1,"messages":[],"system":[{"type":"text","text":"a"},{"type":"text","text":"b","cache_control":{"type":"ephemeral"}}]}"#
            ),
            "a\n\nb"
        );
    }

    #[test]
    fn thinking_maps_to_enable_thinking() {
        let kwargs = |thinking: &str| {
            parse(&format!(
                r#"{{"model":"m","max_tokens":10,"messages":[]{thinking}}}"#
            ))
            .template_kwargs()
            .map(|k| k.enable_thinking())
        };
        assert_eq!(kwargs("").unwrap(), None);
        assert_eq!(
            kwargs(r#","thinking":{"type":"disabled"}"#).unwrap(),
            Some(false)
        );
        assert_eq!(
            kwargs(r#","thinking":{"type":"enabled","budget_tokens":1024}"#).unwrap(),
            Some(true)
        );
        assert!(kwargs(r#","thinking":{"type":"sometimes"}"#).is_err());
    }

    #[test]
    fn stop_reasons() {
        assert_eq!(to_stop_reason("stop", &None), "end_turn");
        assert_eq!(to_stop_reason("stop", &Some("END".into())), "stop_sequence");
        assert_eq!(to_stop_reason("length", &None), "max_tokens");
        assert_eq!(to_stop_reason("tool_calls", &None), "tool_use");
    }

    fn shape(events: &[MessageEvent]) -> Vec<(String, u64)> {
        events
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .map(|e| {
                let kind = e["content_block"]["type"]
                    .as_str()
                    .or(e["delta"]["type"].as_str())
                    .unwrap_or(e["type"].as_str().unwrap())
                    .to_string();
                (kind, e["index"].as_u64().unwrap_or(u64::MAX))
            })
            .collect()
    }

    fn content(text: &str) -> Piece {
        Piece::Content {
            text: text.into(),
            logprobs: Vec::new(),
        }
    }

    #[test]
    fn reasoning_then_text_then_a_call_stream_as_indexed_blocks() {
        let mut blocks = BlockStream::default();
        let mut events = Vec::new();
        for piece in [
            Piece::Reasoning("plan".into()),
            Piece::Reasoning(" more".into()),
            content("Paris"),
            Piece::ToolCall(ParsedCall {
                name: "f".into(),
                arguments: r#"{"a":1}"#.into(),
            }),
        ] {
            events.extend(blocks.events(piece));
        }
        events.extend(blocks.closing());
        let want = |s: &[(&str, u64)]| -> Vec<(String, u64)> {
            s.iter().map(|(k, i)| (k.to_string(), *i)).collect()
        };
        assert_eq!(
            shape(&events),
            want(&[
                ("thinking", 0),
                ("thinking_delta", 0),
                ("thinking_delta", 0),
                ("signature_delta", 0),
                ("content_block_stop", 0),
                ("text", 1),
                ("text_delta", 1),
                ("content_block_stop", 1),
                ("tool_use", 2),
                ("input_json_delta", 2),
                ("content_block_stop", 2),
            ])
        );
    }

    #[test]
    fn two_calls_are_two_blocks_and_no_text_is_added() {
        let mut blocks = BlockStream::default();
        let call = |n: &str| {
            Piece::ToolCall(ParsedCall {
                name: n.into(),
                arguments: "{}".into(),
            })
        };
        let mut events = blocks.events(call("a"));
        events.extend(blocks.events(call("b")));
        events.extend(blocks.closing());
        let kinds: Vec<_> = shape(&events).into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            kinds,
            [
                "tool_use",
                "input_json_delta",
                "content_block_stop",
                "tool_use",
                "input_json_delta",
                "content_block_stop"
            ]
        );
    }

    #[test]
    fn an_empty_reply_is_one_empty_text_block() {
        let events = BlockStream::default().closing();
        let kinds: Vec<_> = shape(&events).into_iter().map(|(k, _)| k).collect();
        assert_eq!(kinds, ["text", "content_block_stop"]);
    }
}
