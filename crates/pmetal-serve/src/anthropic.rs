//! Anthropic-compatible `/v1/messages` endpoint.
//!
//! Accepts the Anthropic Messages API request shape (string or text-block
//! content, optional `system` prompt, optional `tools`) and delegates to the
//! same `InferenceEngine::generate` / `generate_streaming` path as
//! `/v1/chat/completions`. Response shapes differ — see [`MessagesResponse`]
//! and the streaming event enum below — but the underlying generation is
//! identical.
//!
//! Scope: text + tool calling. Vision / structured output / batch are out of
//! scope for the first phase and live in the plan's deferred list.

use crate::error::ServeError;
use crate::routes::AppState;
use crate::types::try_parse_tool_calls;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use futures::stream::{self, StreamExt};
use pmetal_data::chat_templates::{ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::{SamplingParams, TokenEvent};
use crate::types::ChatMessage;

// ────────────────────────────────────────────────────────────────────────────
// Request types
// ────────────────────────────────────────────────────────────────────────────

/// Message content — either a plain string or an array of typed blocks.
///
/// Anthropic's spec allows `content` to be either `"text"` or `[{type, ...}]`.
/// We accept both; non-text blocks (images, tool_use, tool_result) are
/// flattened to their text portions (or empty strings for now). A follow-on
/// phase can expand block handling.
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
    /// Vision / tool-use / tool-result blocks are accepted but ignored by
    /// the text extractor — prevents 400s for valid Anthropic payloads.
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
}

// ────────────────────────────────────────────────────────────────────────────
// Response types
// ────────────────────────────────────────────────────────────────────────────

/// Block inside an assistant response — `text` or `tool_use`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
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
    // Assemble the internal chat-message list: optional system prompt first,
    // then each Anthropic message with content flattened to plain text.
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(req.messages.len() + 1);
    if let Some(sys) = req.system.as_ref().filter(|s| !s.is_empty()) {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: sys.clone(),
            tool_calls: None,
        });
    }
    for m in &req.messages {
        messages.push(ChatMessage {
            role: m.role.clone(),
            content: m.text(),
            tool_calls: None,
        });
    }

    let prompt = state
        .engine
        .format_chat_with_tools(&messages, req.tools.as_deref());
    let input_ids = state.engine.tokenize(&prompt)?;
    let prompt_tokens = input_ids.len();
    let tools_requested = req.tools.is_some();

    let extra_stop_ids = resolve_stop_ids(&req.stop_sequences, state.as_ref());
    let temperature = req.temperature.unwrap_or(0.0);
    let request_id = format!("msg_{}", uuid::Uuid::new_v4());
    let model_id = state.engine.model_id().to_string();

    let params = SamplingParams {
        max_tokens: req.max_tokens,
        temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        min_p: None,
        repetition_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        extra_stop_token_ids: extra_stop_ids,
    };

    if req.stream.unwrap_or(false) {
        let rx = state.engine.generate_streaming(&input_ids, params);
        let tokenizer = state.engine.tokenizer_arc();
        let metrics_handle = Arc::clone(&state);
        let sse = anthropic_sse_stream(
            rx,
            tokenizer,
            request_id,
            model_id,
            prompt_tokens,
            tools_requested,
            metrics_handle,
        );
        return Ok(Sse::new(sse)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response());
    }

    // Non-streaming path.
    let (tokens, finish_reason, metrics) = state.engine.generate(&input_ids, params).await?;
    state.metrics.record(&metrics);
    let text = state.engine.decode(&tokens)?;
    let output_tokens = tokens.len();

    let (content, stop_reason) = if tools_requested {
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
///   2. `content_block_start` (text block at index 0).
///   3. One `content_block_delta` per newly decoded UTF-8 text prefix.
///   4. `content_block_stop`.
///   5. `message_delta` carrying the final stop_reason + output_tokens.
///   6. `message_stop`.
///
/// Tool-call detection runs on the full accumulated text at Done — when a
/// tool call parses, the stop_reason becomes `tool_use`. For Phase 1 we do
/// not stream tool_use blocks incrementally; the text deltas already
/// carry the raw JSON, and tool-aware clients can parse it from the final
/// message_delta metadata when they see `stop_reason == "tool_use"`.
#[allow(clippy::too_many_arguments)]
fn anthropic_sse_stream(
    rx: tokio::sync::mpsc::Receiver<TokenEvent>,
    tokenizer: Arc<pmetal_data::Tokenizer>,
    request_id: String,
    model_id: String,
    prompt_tokens: usize,
    tools_requested: bool,
    state: Arc<AppState>,
) -> impl futures::Stream<Item = Result<Event, Infallible>> + Send + 'static {
    // Opening events — pre-built so the first token arrival doesn't pay the
    // cost of serialising three SSE frames in a row.
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
    let opening_block_start = MessageEvent::ContentBlockStart {
        index: 0,
        content_block: ResponseContentBlock::Text {
            text: String::new(),
        },
    };

    let openings = stream::iter(vec![
        Ok::<Event, Infallible>(encode_event(&opening_message_start)),
        Ok(encode_event(&opening_block_start)),
    ]);

    // UTF-8 boundary buffering — mirrors chat_sse_stream's approach so
    // multi-byte codepoints at BPE boundaries don't leak partial bytes.
    let mut token_buffer: Vec<u32> = Vec::new();
    let mut emitted_text_len: usize = 0;

    let mapped = ReceiverStream::new(rx).flat_map(move |event| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        match event {
            TokenEvent::Token(tok) => {
                token_buffer.push(tok);
                let decoded = tokenizer.decode(&token_buffer).unwrap_or_default();
                if decoded.len() > emitted_text_len {
                    let new_text = decoded[emitted_text_len..].to_owned();
                    emitted_text_len = decoded.len();
                    let delta = MessageEvent::ContentBlockDelta {
                        index: 0,
                        delta: DeltaBlock::TextDelta { text: new_text },
                    };
                    events.push(Ok(encode_event(&delta)));
                }
            }
            TokenEvent::Done(finish_reason, metrics) => {
                // Flush any residual buffered text.
                let decoded = tokenizer.decode(&token_buffer).unwrap_or_default();
                if decoded.len() > emitted_text_len {
                    let remaining = decoded[emitted_text_len..].to_owned();
                    let delta = MessageEvent::ContentBlockDelta {
                        index: 0,
                        delta: DeltaBlock::TextDelta { text: remaining },
                    };
                    events.push(Ok(encode_event(&delta)));
                }
                state.metrics.record(&metrics);

                let output_tokens = token_buffer.len();
                let stop_reason = if tools_requested {
                    let full = tokenizer.decode(&token_buffer).unwrap_or_default();
                    if try_parse_tool_calls(&full).is_some() {
                        "tool_use".to_string()
                    } else {
                        to_stop_reason(&finish_reason)
                    }
                } else {
                    to_stop_reason(&finish_reason)
                };

                events.push(Ok(encode_event(&MessageEvent::ContentBlockStop {
                    index: 0,
                })));
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
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn resolve_stop_ids(stop: &Option<Vec<String>>, state: &AppState) -> Vec<u32> {
    stop.as_deref()
        .unwrap_or(&[])
        .iter()
        .filter_map(|s| match state.engine.tokenize(s) {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        })
        .collect()
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
}
