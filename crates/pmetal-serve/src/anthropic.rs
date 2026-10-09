//! Anthropic-compatible `/v1/messages` endpoint.
//!
//! Accepts the Anthropic Messages API request shape (string or block
//! content, optional `system` prompt, optional `tools`) and delegates to the
//! same `InferenceEngine::generate` / `generate_streaming` path as
//! `/v1/chat/completions`. Response shapes differ — see [`MessagesResponse`]
//! and the streaming event enum below — but the underlying generation is
//! identical.
//!
//! Scope: text, images (base64 `image` blocks, for a model that reads them;
//! see [`crate::media`]) and tool calling. Structured output and batches are
//! out of scope.

use crate::error::ServeError;
use crate::routes::{AppState, resolve_stop_sequences, split_reply};
use crate::types::try_parse_tool_calls;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use futures::stream::{self, StreamExt};
use pmetal_data::chat_templates::{ChatTemplateKwargs, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::{SamplingParams, TokenEvent};
use crate::sse::{ChannelDecoder, ChannelDelta};
use crate::types::ChatMessage;

// ────────────────────────────────────────────────────────────────────────────
// Request types
// ────────────────────────────────────────────────────────────────────────────

/// Message content — either a plain string or an array of typed blocks.
///
/// Anthropic's spec allows `content` to be either `"text"` or `[{type, ...}]`.
/// We accept both. Text blocks are joined; image blocks go to the model as
/// images (see [`AnthropicMessage::to_chat_message`]); tool_use and
/// tool_result blocks are accepted and contribute nothing.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    String(String),
    Blocks(Vec<ContentBlock>),
}

/// One block inside a message's content array.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// `{"type": "image", "source": {"type": "base64", "media_type":
    /// "image/png", "data": "..."}}`. Other source types are refused when the
    /// message is read.
    Image {
        source: serde_json::Value,
    },
    /// Tool-use / tool-result blocks are accepted but ignored by the text
    /// extractor — prevents 400s for valid Anthropic payloads.
    #[serde(other)]
    Other,
}

/// Anthropic message.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

impl AnthropicMessage {
    /// The message as a chat-completions message. Text-only content is
    /// flattened by [`text`](Self::text); content with images keeps its
    /// blocks in order as content parts, each image a `data:` URI.
    /// `index` names the message in errors.
    fn to_chat_message(&self, index: usize) -> Result<ChatMessage, ServeError> {
        let MessageContent::Blocks(blocks) = &self.content else {
            return Ok(ChatMessage::text(self.role.clone(), self.text()));
        };
        if !blocks
            .iter()
            .any(|block| matches!(block, ContentBlock::Image { .. }))
        {
            return Ok(ChatMessage::text(self.role.clone(), self.text()));
        }
        let mut parts = Vec::with_capacity(blocks.len());
        for (j, block) in blocks.iter().enumerate() {
            match block {
                ContentBlock::Text { text } => {
                    parts.push(serde_json::json!({"type": "text", "text": text}));
                }
                ContentBlock::Image { source } => {
                    let url = image_source_url(source).map_err(|why| {
                        ServeError::BadRequest(format!("messages[{index}].content[{j}]: {why}"))
                    })?;
                    parts.push(serde_json::json!({"type": "image_url", "image_url": {"url": url}}));
                }
                ContentBlock::Other => {}
            }
        }
        Ok(ChatMessage {
            parts: Some(parts),
            ..ChatMessage::text(self.role.clone(), self.text())
        })
    }

    /// Flatten content to a plain string. Non-text blocks contribute nothing.
    fn text(&self) -> String {
        match &self.content {
            MessageContent::String(s) => s.clone(),
            MessageContent::Blocks(blocks) => {
                let mut out = String::new();
                for b in blocks {
                    if let ContentBlock::Text { text } = b {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(text);
                    }
                }
                out
            }
        }
    }
}

/// The image URL an `image` block's `source` stands for: a `data:` URI for a
/// base64 source, the URL itself for a `url` source (which
/// [`crate::media`] then refuses, since the server fetches nothing).
fn image_source_url(source: &serde_json::Value) -> Result<String, String> {
    let field = |key: &str| source.get(key).and_then(serde_json::Value::as_str);
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

/// Anthropic `/v1/messages` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: usize,
    pub messages: Vec<AnthropicMessage>,
    /// System prompt — prepended as a `system`-role message when present.
    #[serde(default)]
    pub system: Option<String>,
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
    pub tools: Option<Vec<ToolDefinition>>,
    /// `{"type": "enabled", "budget_tokens": N}` or `{"type": "disabled"}`:
    /// the chat template's `enable_thinking`. Absent leaves the template's
    /// default.
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
}

/// The `thinking` field of a messages request.
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkingConfig {
    /// `"enabled"` or `"disabled"`.
    #[serde(rename = "type")]
    pub kind: String,
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
                return Err(ServeError::BadRequest(format!(
                    "thinking.type must be \"enabled\" or \"disabled\", not {other:?}"
                )));
            }
        }
        Ok(kwargs)
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
        input: serde_json::Value,
    },
}

/// Token counts for an Anthropic response.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
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
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// Map OpenAI-style finish reason → Anthropic `stop_reason`.
fn to_stop_reason(openai_finish: &str) -> String {
    match openai_finish {
        "stop" | "eos" => "end_turn".to_string(),
        "length" | "max_tokens" => "max_tokens".to_string(),
        "stop_sequence" => "stop_sequence".to_string(),
        "tool_calls" => "tool_use".to_string(),
        other => other.to_string(),
    }
}

/// Translate a single ToolCall into an Anthropic tool_use block with a
/// generated id when the caller didn't supply one.
fn tool_call_to_block(idx: usize, tc: ToolCall) -> ResponseContentBlock {
    ResponseContentBlock::ToolUse {
        id: tc
            .id
            .unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4())),
        name: tc.function.name,
        input: match tc.function.arguments {
            // Anthropic's `input` is an object; if the upstream ToolCall carried
            // a string-encoded JSON (some trainers emit this), try to parse it
            // back; otherwise pass through.
            serde_json::Value::String(s) => serde_json::from_str::<serde_json::Value>(&s)
                .unwrap_or(serde_json::Value::String(s)),
            other => other,
        },
    }
    .pin_index(idx)
}

impl ResponseContentBlock {
    // Index is carried in streaming events but not in the non-streaming
    // response. This no-op method exists so the helper above can be chained
    // symmetrically with the streaming path later.
    fn pin_index(self, _index: usize) -> Self {
        self
    }
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
        usage: AnthropicUsage,
    },
    MessageStop,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DeltaBlock {
    ThinkingDelta { thinking: String },
    TextDelta { text: String },
}

#[derive(Debug, Clone, Serialize)]
struct MessageDeltaPayload {
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
}

// ────────────────────────────────────────────────────────────────────────────
// Handler
// ────────────────────────────────────────────────────────────────────────────

/// `POST /v1/messages` — Anthropic-compatible message generation.
pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MessagesRequest>,
) -> Result<axum::response::Response, ServeError> {
    let permit = state.try_acquire_request_permit()?;

    // Assemble the internal chat-message list: optional system prompt first,
    // then each Anthropic message with content flattened to plain text.
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(req.messages.len() + 1);
    if let Some(sys) = req.system.as_ref().filter(|s| !s.is_empty()) {
        messages.push(ChatMessage::text("system", sys.clone()));
    }
    for (i, m) in req.messages.iter().enumerate() {
        messages.push(m.to_chat_message(i)?);
    }

    let template_kwargs = req.template_kwargs()?;
    let prompt = state
        .engine
        .prepare_chat(&messages, req.tools.as_deref(), &template_kwargs)
        .await?;
    let prompt_tokens = prompt.input_ids.len();
    let tools_requested = req.tools.is_some();

    let resolved_stops = resolve_stop_sequences(&req.stop_sequences, &state.engine);
    // Sampling the request leaves out follows the model maker's
    // recommendation for the mode it runs in, as `pmetal infer` does.
    let defaults = state
        .engine
        .sampling_defaults(state.engine.thinks_with(&template_kwargs));
    let temperature = req.temperature.unwrap_or(defaults.temperature);
    let request_id = format!("msg_{}", uuid::Uuid::new_v4());
    let model_id = state.engine.model_id().to_string();

    let params = SamplingParams {
        max_tokens: req.max_tokens,
        temperature,
        top_k: req.top_k.or(Some(defaults.top_k)),
        top_p: req.top_p.or(Some(defaults.top_p)),
        min_p: Some(defaults.min_p),
        // An explicitly greedy request decodes plain argmax: no default
        // penalties shift it.
        repetition_penalty: (temperature > 0.0).then_some(defaults.repetition_penalty),
        frequency_penalty: (temperature > 0.0).then_some(defaults.frequency_penalty),
        presence_penalty: (temperature > 0.0).then_some(defaults.presence_penalty),
        seed: None,
        extra_stop_token_ids: resolved_stops.token_ids.clone(),
        stop_sequences: resolved_stops.sequences.clone(),
        // Anthropic /v1/messages does not expose OpenAI-style logprobs.
        logprobs_top_n: None,
    };
    state.engine.validate_sampling_params(&params)?;

    if req.stream.unwrap_or(false) {
        let rx = crate::routes::stream_tokens(&state.engine, &prompt, params);
        let tokenizer = state.engine.tokenizer_arc();
        let metrics_handle = Arc::clone(&state);
        let sse = anthropic_sse_stream(
            rx,
            tokenizer,
            &prompt.input_ids,
            request_id,
            model_id,
            tools_requested,
            metrics_handle,
            permit,
            resolved_stops.holdback_tokens(),
        );
        return Ok(Sse::new(sse)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response());
    }

    // Non-streaming path — ignore OpenAI-style logprobs slot.
    let (tokens, _logprobs, finish_reason, metrics) =
        state.engine.generate_prompt(&prompt, params).await?;
    state.metrics.record(&metrics);
    let reply = split_reply(&state.engine, &prompt.input_ids, &tokens, tools_requested)?;
    let text = reply.content;
    let output_tokens = tokens.len();

    let (content, stop_reason): (Vec<ResponseContentBlock>, String) = if tools_requested {
        match try_parse_tool_calls(&text) {
            Some(calls) => {
                let blocks = calls
                    .into_iter()
                    .enumerate()
                    .map(|(i, tc)| tool_call_to_block(i, tc))
                    .collect();
                (blocks, "tool_use".to_string())
            }
            None => (
                vec![ResponseContentBlock::Text { text }],
                to_stop_reason(&finish_reason),
            ),
        }
    } else {
        (
            vec![ResponseContentBlock::Text { text }],
            to_stop_reason(&finish_reason),
        )
    };
    // The reasoning, when there was any, comes first, as a thinking block.
    let content = reply
        .reasoning
        .map(|thinking| ResponseContentBlock::Thinking {
            thinking,
            signature: String::new(),
        })
        .into_iter()
        .chain(content)
        .collect();

    Ok(Json(MessagesResponse {
        id: request_id,
        message_type: "message",
        role: "assistant",
        content,
        model: model_id,
        stop_reason: Some(stop_reason),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: prompt_tokens,
            output_tokens,
        },
    })
    .into_response())
}

// ────────────────────────────────────────────────────────────────────────────
// Streaming
// ────────────────────────────────────────────────────────────────────────────

/// Assemble the Anthropic streaming SSE event sequence.
///
/// Events emitted in order:
///   1. `message_start` with an empty-content skeleton message.
///   2. For a model that reasons first, a `thinking` block:
///      `content_block_start`, one `thinking_delta` per newly decoded
///      UTF-8 prefix of the reasoning, `content_block_stop`.
///   3. A `text` block the same way, with `text_delta`s of the answer (an
///      empty one when the model gave none).
///   4. `message_delta` carrying the final stop_reason + output_tokens.
///   5. `message_stop`.
///
/// Tool-call detection runs on the full accumulated answer at Done — when a
/// tool call parses, the stop_reason becomes `tool_use`. For Phase 1 we do
/// not stream tool_use blocks incrementally; the text deltas already
/// carry the raw JSON, and tool-aware clients can parse it from the final
/// message_delta metadata when they see `stop_reason == "tool_use"`.
#[allow(clippy::too_many_arguments)]
fn anthropic_sse_stream(
    rx: tokio::sync::mpsc::Receiver<TokenEvent>,
    tokenizer: Arc<pmetal_data::Tokenizer>,
    prompt: &[u32],
    request_id: String,
    model_id: String,
    tools_requested: bool,
    state: Arc<AppState>,
    _permit: OwnedSemaphorePermit,
    holdback_tokens: usize,
) -> impl futures::Stream<Item = Result<Event, Infallible>> + Send + 'static {
    let prompt_tokens = prompt.len();
    // Opening event — pre-built so the first token arrival doesn't pay the
    // cost of serialising it.
    let opening_message_start = MessageEvent::MessageStart {
        message: MessagesResponse {
            id: request_id.clone(),
            message_type: "message",
            role: "assistant",
            content: Vec::new(),
            model: model_id,
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: prompt_tokens,
                output_tokens: 0,
            },
        },
    };
    let openings = stream::iter(vec![Ok::<Event, Infallible>(encode_event(
        &opening_message_start,
    ))]);

    // Shared UTF-8 boundary buffering, one decoder per channel — see
    // crate::sse::ChannelDecoder. Anthropic stream doesn't surface OpenAI
    // logprobs, aux is `()`.
    let mut decoder: ChannelDecoder<()> = ChannelDecoder::new(tokenizer, prompt);
    let mut blocks = BlockStream::default();
    let mut pending_tokens: VecDeque<u32> = VecDeque::new();

    let mapped = ReceiverStream::new(rx).flat_map(move |event| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        match event {
            TokenEvent::Token {
                id: tok,
                logprob: _,
            } => {
                pending_tokens.push_back(tok);
                while pending_tokens.len() > holdback_tokens {
                    let Some(next_tok) = pending_tokens.pop_front() else {
                        break;
                    };
                    blocks.emit(decoder.push(next_tok), &mut events);
                }
            }
            TokenEvent::Done {
                finish_reason,
                metrics,
                stripped_tokens,
            } => {
                if stripped_tokens > pending_tokens.len() {
                    tracing::warn!(
                        stripped_tokens,
                        buffered_tokens = pending_tokens.len(),
                        "stop-sequence suffix exceeded the streaming holdback window"
                    );
                    pending_tokens.clear();
                } else {
                    for _ in 0..stripped_tokens {
                        pending_tokens.pop_back();
                    }
                }

                while let Some(next_tok) = pending_tokens.pop_front() {
                    blocks.emit(decoder.push(next_tok), &mut events);
                }
                blocks.emit(decoder.flush_aux(), &mut events);
                state.metrics.record(&metrics);

                let output_tokens = decoder.token_count();
                let stop_reason = if tools_requested {
                    if try_parse_tool_calls(&decoder.answer_with_special_tokens()).is_some() {
                        "tool_use".to_string()
                    } else {
                        to_stop_reason(&finish_reason)
                    }
                } else {
                    to_stop_reason(&finish_reason)
                };

                blocks.finish(&mut events);
                events.push(Ok(encode_event(&MessageEvent::MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some(stop_reason),
                        stop_sequence: None,
                    },
                    usage: AnthropicUsage {
                        input_tokens: prompt_tokens,
                        output_tokens,
                    },
                })));
                events.push(Ok(encode_event(&MessageEvent::MessageStop)));
            }
            TokenEvent::Error(msg) => {
                tracing::error!("Anthropic streaming generation error: {msg}");
                // Anthropic surfaces errors as a dedicated `error` event frame;
                // we keep the shape minimal — clients treat any non-message event
                // type as fatal and close the stream.
                let err = serde_json::json!({
                    "type": "error",
                    "error": {"type": "server_error", "message": "internal model error"}
                });
                events.push(Ok(Event::default().data(err.to_string())));
            }
        }
        stream::iter(events)
    });

    openings.chain(mapped)
}

/// The content block a stream has open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum OpenBlock {
    #[default]
    None,
    Thinking,
    Text,
}

/// Opens, fills and closes a stream's content blocks: a `thinking` block
/// while the model reasons, then a `text` block for its answer.
#[derive(Debug, Default)]
struct BlockStream {
    open: OpenBlock,
    index: usize,
}

impl BlockStream {
    /// Emit `delta`'s reasoning and answer into their blocks.
    fn emit(&mut self, delta: ChannelDelta<()>, events: &mut Vec<Result<Event, Infallible>>) {
        events.extend(self.events(delta).iter().map(|e| Ok(encode_event(e))));
    }

    /// Close the open block, opening an empty text block first when the
    /// stream produced no answer, so every reply carries one.
    fn finish(&mut self, events: &mut Vec<Result<Event, Infallible>>) {
        events.extend(self.closing().iter().map(|e| Ok(encode_event(e))));
    }

    fn events(&mut self, delta: ChannelDelta<()>) -> Vec<MessageEvent> {
        let mut events = Vec::new();
        if !delta.reasoning.is_empty() {
            self.switch(OpenBlock::Thinking, &mut events);
            events.push(MessageEvent::ContentBlockDelta {
                index: self.index,
                delta: DeltaBlock::ThinkingDelta {
                    thinking: delta.reasoning,
                },
            });
        }
        if !delta.content.is_empty() {
            self.switch(OpenBlock::Text, &mut events);
            events.push(MessageEvent::ContentBlockDelta {
                index: self.index,
                delta: DeltaBlock::TextDelta {
                    text: delta.content,
                },
            });
        }
        events
    }

    fn closing(&mut self) -> Vec<MessageEvent> {
        let mut events = Vec::new();
        self.switch(OpenBlock::Text, &mut events);
        events.push(MessageEvent::ContentBlockStop { index: self.index });
        events
    }

    fn switch(&mut self, to: OpenBlock, events: &mut Vec<MessageEvent>) {
        if self.open == to {
            return;
        }
        if self.open != OpenBlock::None {
            events.push(MessageEvent::ContentBlockStop { index: self.index });
            self.index += 1;
        }
        let content_block = match to {
            OpenBlock::Thinking => ResponseContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            _ => ResponseContentBlock::Text {
                text: String::new(),
            },
        };
        events.push(MessageEvent::ContentBlockStart {
            index: self.index,
            content_block,
        });
        self.open = to;
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

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_content_string_deserializes() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(req.messages[0].text(), "hi");
    }

    #[test]
    fn message_content_blocks_flatten_to_text() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"hello"},{"type":"image","source":{}},{"type":"text","text":"world"}]}]}"#,
        )
        .unwrap();
        assert_eq!(req.messages[0].text(), "hello\nworld");
    }

    #[test]
    fn thinking_maps_to_enable_thinking() {
        let parse = |thinking: &str| -> MessagesRequest {
            serde_json::from_str(&format!(
                r#"{{"model":"m","max_tokens":10,"messages":[]{thinking}}}"#
            ))
            .unwrap()
        };
        let kwargs = |req: MessagesRequest| req.template_kwargs().map(|k| k.enable_thinking());
        assert_eq!(kwargs(parse("")).unwrap(), None);
        assert_eq!(
            kwargs(parse(r#","thinking":{"type":"disabled"}"#)).unwrap(),
            Some(false)
        );
        assert_eq!(
            kwargs(parse(
                r#","thinking":{"type":"enabled","budget_tokens":1024}"#
            ))
            .unwrap(),
            Some(true)
        );
        assert!(kwargs(parse(r#","thinking":{"type":"sometimes"}"#)).is_err());
    }

    #[test]
    fn system_prompt_is_optional() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert!(req.system.is_none());
    }

    #[test]
    fn finish_reason_mapping_covers_expected_cases() {
        assert_eq!(to_stop_reason("stop"), "end_turn");
        assert_eq!(to_stop_reason("eos"), "end_turn");
        assert_eq!(to_stop_reason("length"), "max_tokens");
        assert_eq!(to_stop_reason("max_tokens"), "max_tokens");
        assert_eq!(to_stop_reason("stop_sequence"), "stop_sequence");
        assert_eq!(to_stop_reason("tool_calls"), "tool_use");
        // Unknown reasons pass through so the wire format never drops data.
        assert_eq!(to_stop_reason("something_new"), "something_new");
    }

    #[test]
    fn response_content_block_text_serializes_with_type_tag() {
        let block = ResponseContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"text""#));
        assert!(json.contains(r#""text":"hello""#));
    }

    #[test]
    fn reasoning_streams_as_a_thinking_block_ahead_of_the_text() {
        let delta = |reasoning: &str, content: &str| ChannelDelta {
            reasoning: reasoning.into(),
            content: content.into(),
            aux: Vec::new(),
        };
        let mut blocks = BlockStream::default();
        let mut events = Vec::new();
        for d in [delta("plan", ""), delta(" more", ""), delta("", "Paris")] {
            events.extend(blocks.events(d));
        }
        events.extend(blocks.closing());
        let events: Vec<serde_json::Value> = events
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .collect();
        let shape: Vec<(String, u64)> = events
            .iter()
            .map(|e| {
                let kind = e["content_block"]["type"]
                    .as_str()
                    .or(e["delta"]["type"].as_str())
                    .unwrap_or(e["type"].as_str().unwrap());
                (kind.to_string(), e["index"].as_u64().unwrap())
            })
            .collect();
        let expect = |s: &[(&str, u64)]| -> Vec<(String, u64)> {
            s.iter().map(|(k, i)| (k.to_string(), *i)).collect()
        };
        assert_eq!(
            shape,
            expect(&[
                ("thinking", 0),
                ("thinking_delta", 0),
                ("thinking_delta", 0),
                ("content_block_stop", 0),
                ("text", 1),
                ("text_delta", 1),
                ("content_block_stop", 1),
            ])
        );
        assert_eq!(events[1]["delta"]["thinking"], "plan");
        assert_eq!(events[5]["delta"]["text"], "Paris");

        // No reasoning and no answer: one empty text block, as before.
        let shape: Vec<serde_json::Value> = BlockStream::default()
            .closing()
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .collect();
        assert_eq!(shape[0]["content_block"]["type"], "text");
        assert_eq!(shape[1]["type"], "content_block_stop");
    }
}
