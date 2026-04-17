//! HTTP route handlers for the OpenAI-compatible API.

use crate::engine::{InferenceEngine, RequestMetrics, SamplingParams, TokenEvent};
use crate::error::ServeError;
use crate::sse::IncrementalDecoder;
use crate::types::*;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use chrono::Utc;
use futures::stream::{self, StreamExt};
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio_stream::wrappers::ReceiverStream;

// ────────────────────────────────────────────────────────────────────────────
// Serving metrics state
// ────────────────────────────────────────────────────────────────────────────

/// Atomic metrics counters accumulated across all requests.
///
/// All values use fixed-point representation where applicable:
/// - Latency sums: microseconds (integer) for precision without f64 atomics.
/// - Token counts: plain integers.
#[derive(Debug, Default)]
pub struct ServingMetrics {
    /// Total completed requests.
    pub total_requests: AtomicU64,
    /// Sum of first-token latencies in microseconds.
    pub sum_first_token_us: AtomicU64,
    /// Sum of total request latencies in microseconds.
    pub sum_total_latency_us: AtomicU64,
    /// Sum of completion token counts.
    pub sum_completion_tokens: AtomicU64,
    /// Sum of prompt token counts.
    pub sum_prompt_tokens: AtomicU64,
}

impl ServingMetrics {
    /// Record metrics from a completed request.
    pub fn record(&self, metrics: &RequestMetrics) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.sum_first_token_us.fetch_add(
            (metrics.first_token_latency_ms * 1000.0) as u64,
            Ordering::Relaxed,
        );
        self.sum_total_latency_us.fetch_add(
            (metrics.total_latency_ms * 1000.0) as u64,
            Ordering::Relaxed,
        );
        self.sum_completion_tokens
            .fetch_add(metrics.completion_tokens as u64, Ordering::Relaxed);
        self.sum_prompt_tokens
            .fetch_add(metrics.prompt_tokens as u64, Ordering::Relaxed);
    }

    /// Compute rolling averages and return as a JSON value.
    pub fn snapshot(&self) -> serde_json::Value {
        let total = self.total_requests.load(Ordering::Relaxed);
        let sum_first = self.sum_first_token_us.load(Ordering::Relaxed);
        let sum_total = self.sum_total_latency_us.load(Ordering::Relaxed);
        let sum_compl = self.sum_completion_tokens.load(Ordering::Relaxed);
        let sum_prompt = self.sum_prompt_tokens.load(Ordering::Relaxed);

        let avg_first_ms = if total > 0 {
            (sum_first as f64 / total as f64) / 1000.0
        } else {
            0.0
        };
        let avg_total_ms = if total > 0 {
            (sum_total as f64 / total as f64) / 1000.0
        } else {
            0.0
        };
        // tokens/s: total completion tokens / total wall time across all reqs
        let rolling_tps = if sum_total > 0 {
            sum_compl as f64 / (sum_total as f64 / 1_000_000.0)
        } else {
            0.0
        };

        json!({
            "total_requests": total,
            "avg_first_token_latency_ms": avg_first_ms,
            "avg_total_latency_ms": avg_total_ms,
            "rolling_tokens_per_second": rolling_tps,
            "total_prompt_tokens": sum_prompt,
            "total_completion_tokens": sum_compl,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Application state
// ────────────────────────────────────────────────────────────────────────────

/// Shared application state.
pub struct AppState {
    pub engine: InferenceEngine,
    pub metrics: ServingMetrics,
}

// ────────────────────────────────────────────────────────────────────────────
// Route handlers
// ────────────────────────────────────────────────────────────────────────────

/// GET /health
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// GET /v1/models
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let model = ModelInfo {
        id: state.engine.model_id().to_string(),
        object: "model".to_string(),
        created: state.engine.created_at(),
        owned_by: "pmetal".to_string(),
    };

    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![model],
    })
}

/// GET /v1/metrics
///
/// Returns rolling aggregate serving metrics as JSON.
pub async fn serving_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.metrics.snapshot())
}

/// POST /v1/chat/completions
///
/// Supports both non-streaming (default) and streaming (`stream: true`) modes.
///
/// Non-streaming: generates all tokens, returns a single JSON response.
///
/// Streaming: returns `text/event-stream` SSE. Each generated token is sent
/// to the client individually as it is produced — no batching or buffering.
/// Wire format follows the OpenAI streaming protocol.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, ServeError> {
    // Format messages using chat template, optionally including tool definitions.
    let prompt = state
        .engine
        .format_chat_with_tools(&req.messages, req.tools.as_deref());
    let input_ids = state.engine.tokenize(&prompt)?;
    let prompt_tokens = input_ids.len();
    let tools_requested = req.tools.is_some();

    // Resolve stop strings to token IDs.
    let extra_stop_ids = resolve_stop_ids(&req.stop, &state.engine);

    // temperature == None or 0.0 → greedy decoding.
    let temperature = req.temperature.unwrap_or(0.0);
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.engine.model_id().to_string();
    // Use request-time timestamp per OpenAI spec — not model creation time.
    let created = Utc::now().timestamp();

    // Chat completions: honour the request's logprobs/top_logprobs fields
    // for both streaming and non-streaming. Streaming attaches logprobs to
    // each ChatDelta as deltas cross codepoint boundaries.
    let logprobs_top_n = if req.logprobs.unwrap_or(false) {
        Some(req.top_logprobs.unwrap_or(0) as usize)
    } else {
        None
    };

    let params = SamplingParams {
        max_tokens: req.max_tokens,
        temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        seed: req.seed,
        extra_stop_token_ids: extra_stop_ids,
        logprobs_top_n,
    };

    if req.stream.unwrap_or(false) {
        // ── True token-by-token streaming ────────────────────────────────────
        //
        // generate_streaming spawns a blocking thread immediately and returns
        // an mpsc Receiver. We wrap it in ReceiverStream and flat_map each
        // TokenEvent to one or more SSE events.
        //
        // Token decoding happens on the async side using a cloned Arc to the
        // tokenizer — pmetal_data::Tokenizer is Send + Sync, so this is safe.
        let rx = state.engine.generate_streaming(&input_ids, params);
        let tokenizer = state.engine.tokenizer_arc();
        let metrics_handle = Arc::clone(&state);

        let sse_stream = chat_sse_stream(
            rx,
            tokenizer,
            request_id,
            model_id,
            created,
            metrics_handle,
            tools_requested,
        );

        return Ok(Sse::new(sse_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response());
    }

    // ── Non-streaming path ───────────────────────────────────────────────────
    let (tokens, logprob_entries, finish_reason, metrics) =
        state.engine.generate(&input_ids, params).await?;

    state.metrics.record(&metrics);

    let completion_tokens = tokens.len();
    let text = state.engine.decode(&tokens)?;

    // Best-effort tool-call detection: only attempted when the caller declared
    // `tools` in the request. Falls back to plain content otherwise.
    let (content, tool_calls, reason) = if tools_requested {
        match try_parse_tool_calls(&text) {
            Some(calls) => (String::new(), Some(calls), "tool_calls".to_string()),
            None => (text, None, finish_reason),
        }
    } else {
        (text, None, finish_reason)
    };

    // Convert engine logprob entries to the wire shape. Per-token byte
    // decode uses the tokenizer so clients can reconstruct exact bytes
    // even when a token lands mid-codepoint.
    let logprobs = logprob_entries.map(|entries| {
        let tokenizer = state.engine.tokenizer_arc();
        let content = entries
            .into_iter()
            .map(|e| {
                let token_str = tokenizer.decode(&[e.token]).unwrap_or_default();
                let top = e
                    .top_logprobs
                    .into_iter()
                    .map(|(tok, lp)| TopLogprob {
                        token: tokenizer.decode(&[tok]).unwrap_or_default(),
                        logprob: lp,
                        bytes: None,
                    })
                    .collect();
                TokenLogprobContent {
                    token: token_str,
                    logprob: e.logprob,
                    bytes: None,
                    top_logprobs: top,
                }
            })
            .collect();
        ChatLogprobs { content }
    });

    Ok(Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
                tool_calls,
            },
            finish_reason: Some(reason),
            logprobs,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        system_fingerprint: None,
    })
    .into_response())
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, ServeError> {
    let input_ids = state.engine.tokenize(&req.prompt)?;
    let prompt_tokens = input_ids.len();

    let extra_stop_ids = resolve_stop_ids(&req.stop, &state.engine);
    let temperature = req.temperature.unwrap_or(0.0);
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.engine.model_id().to_string();
    // Use request-time timestamp per OpenAI spec — not model creation time.
    let created = Utc::now().timestamp();

    // /v1/completions: logprobs is a numeric field (number of top
    // alternatives). Honoured on both the streaming and non-streaming
    // paths — streaming deltas emit per-token 4-parallel-array chunks.
    let logprobs_top_n = req.logprobs.map(|n| n as usize);

    let params = SamplingParams {
        max_tokens: req.max_tokens,
        temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        seed: req.seed,
        extra_stop_token_ids: extra_stop_ids,
        logprobs_top_n,
    };

    if req.stream.unwrap_or(false) {
        // ── Streaming text completions ───────────────────────────────────────
        let rx = state.engine.generate_streaming(&input_ids, params);
        let tokenizer = state.engine.tokenizer_arc();
        let metrics_handle = Arc::clone(&state);

        let logprobs_enabled = logprobs_top_n.is_some();
        let sse_stream = completion_sse_stream(
            rx,
            tokenizer,
            request_id,
            model_id,
            created,
            metrics_handle,
            logprobs_enabled,
        );

        return Ok(Sse::new(sse_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response());
    }

    // ── Non-streaming completions ────────────────────────────────────────────
    let (tokens, logprob_entries, finish_reason, metrics) =
        state.engine.generate(&input_ids, params).await?;

    state.metrics.record(&metrics);

    let completion_tokens = tokens.len();
    let text = state.engine.decode(&tokens)?;

    // Build OpenAI's 4-parallel-array logprobs object when the caller
    // opted in. text_offset gets recomputed by decoding incrementally so
    // each token's start position aligns with the returned `text`.
    let logprobs = logprob_entries.map(|entries| {
        let tokenizer = state.engine.tokenizer_arc();
        let n = entries.len();
        let mut out_tokens = Vec::with_capacity(n);
        let mut token_logprobs = Vec::with_capacity(n);
        let mut top_logprobs = Vec::with_capacity(n);
        let mut text_offset = Vec::with_capacity(n);

        // Re-emit text incrementally to track byte offsets — single-token
        // decode can produce different boundaries than batched decode for
        // BPE tokenisers, so this matches exactly what `text` contains
        // when the offsets are summed.
        let mut running = 0usize;
        for entry in entries {
            let tok_str = tokenizer.decode(&[entry.token]).unwrap_or_default();
            let mut top = std::collections::HashMap::with_capacity(entry.top_logprobs.len());
            for (alt_id, lp) in entry.top_logprobs {
                let alt_str = tokenizer.decode(&[alt_id]).unwrap_or_default();
                top.insert(alt_str, lp);
            }
            text_offset.push(running);
            running += tok_str.len();
            out_tokens.push(tok_str);
            token_logprobs.push(entry.logprob);
            top_logprobs.push(top);
        }

        CompletionLogprobs {
            tokens: out_tokens,
            token_logprobs,
            top_logprobs,
            text_offset,
        }
    });

    Ok(Json(CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created,
        model: model_id,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: Some(finish_reason),
            logprobs,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        system_fingerprint: None,
    })
    .into_response())
}

// ────────────────────────────────────────────────────────────────────────────
// SSE stream builders
// ────────────────────────────────────────────────────────────────────────────

/// Build the per-delta `ChatLogprobs` payload from a drained aux batch.
///
/// Returns `None` when none of the drained tokens carried a logprob (e.g.
/// the request didn't set `logprobs: true`) — that lets the
/// `skip_serializing_if = Option::is_none` annotation on `ChatDelta.logprobs`
/// keep the wire shape unchanged for the default streaming path.
fn delta_logprobs_from_aux(
    tokenizer: &Arc<pmetal_data::Tokenizer>,
    aux: Vec<Option<crate::engine::TokenLogprobEntry>>,
) -> Option<ChatLogprobs> {
    let any_present = aux.iter().any(Option::is_some);
    if !any_present {
        return None;
    }
    let content = aux
        .into_iter()
        .flatten()
        .map(|entry| {
            let token_str = tokenizer.decode(&[entry.token]).unwrap_or_default();
            let top = entry
                .top_logprobs
                .into_iter()
                .map(|(tok, lp)| TopLogprob {
                    token: tokenizer.decode(&[tok]).unwrap_or_default(),
                    logprob: lp,
                    bytes: None,
                })
                .collect();
            TokenLogprobContent {
                token: token_str,
                logprob: entry.logprob,
                bytes: None,
                top_logprobs: top,
            }
        })
        .collect();
    Some(ChatLogprobs { content })
}

/// Convert an mpsc token stream into an SSE event stream for chat completions.
///
/// Emits:
/// 1. An opening event with `role: "assistant"` and no content.
/// 2. One event per token (decoded to text, with UTF-8 boundary buffering).
/// 3. A closing event with `finish_reason` and empty delta.
/// 4. A `[DONE]` sentinel (only on success — not emitted after errors).
///
/// Metrics are recorded to `state.metrics` when the `Done` event arrives.
fn chat_sse_stream(
    rx: tokio::sync::mpsc::Receiver<TokenEvent>,
    tokenizer: Arc<pmetal_data::Tokenizer>,
    request_id: String,
    model_id: String,
    created: i64,
    state: Arc<AppState>,
    tools_requested: bool,
) -> impl futures::Stream<Item = Result<Event, Infallible>> + Send + 'static {
    // Pre-build the opening event once.
    let opening = {
        let chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    tool_calls: None,
                    logprobs: None,
                },
                finish_reason: None,
            }],
        };
        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
    };

    // BPE boundaries can split multi-byte codepoints across tokens — the
    // decoder buffers the full sequence and exposes only the confirmed
    // prefix, so clients never see half-codepoint byte sequences.
    //
    // The aux payload is the per-token logprob (None when caller didn't
    // opt in). On boundary-flush, drained aux entries align 1:1 with the
    // tokens that contributed to the new text — we attach them to the
    // outgoing ChatDelta as `logprobs.content`.
    let mut decoder: IncrementalDecoder<Option<crate::engine::TokenLogprobEntry>> =
        IncrementalDecoder::new(Arc::clone(&tokenizer));

    // Prepend the opening event to the token-event stream.
    let token_stream = ReceiverStream::new(rx);
    let tokenizer_for_aux = Arc::clone(&tokenizer);

    // Map each TokenEvent to a Vec of SSE events (flat_map expands the vec).
    let mapped = token_stream.flat_map(move |event| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();

        match event {
            TokenEvent::Token { id: token_id, logprob } => {
                let (new_text, drained_aux) = decoder.push_with_aux(token_id, logprob);
                if !new_text.is_empty() {
                    // Build the per-delta logprobs payload from any aux that
                    // crossed the codepoint boundary in this emit.
                    let logprobs =
                        delta_logprobs_from_aux(&tokenizer_for_aux, drained_aux);
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: Some(new_text),
                                tool_calls: None,
                                logprobs,
                            },
                            finish_reason: None,
                        }],
                    };
                    events.push(Ok(Event::default()
                        .data(serde_json::to_string(&chunk).unwrap_or_default())));
                }
            }
            TokenEvent::Done(finish_reason, metrics) => {
                let (remaining, _drained_aux) = decoder.flush_aux();
                if !remaining.is_empty() {
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: Some(remaining),
                                tool_calls: None,
                                logprobs: None,
                            },
                            finish_reason: None,
                        }],
                    };
                    events.push(Ok(Event::default()
                        .data(serde_json::to_string(&chunk).unwrap_or_default())));
                }

                // Record request metrics now that generation is complete.
                state.metrics.record(&metrics);

                // Best-effort tool-call detection on the accumulated response.
                let (tool_calls, reason) = if tools_requested {
                    match try_parse_tool_calls(&decoder.decoded_text()) {
                        Some(calls) => (Some(calls), "tool_calls".to_string()),
                        None => (None, finish_reason),
                    }
                } else {
                    (None, finish_reason)
                };

                // Closing chunk: empty delta, finish_reason set. When a tool
                // call was detected, attach structured tool_calls so tool-aware
                // clients can use the parsed form without re-parsing content.
                let closing = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: None,
                            tool_calls,
                            logprobs: None,
                        },
                        finish_reason: Some(reason),
                    }],
                };
                events.push(Ok(Event::default()
                    .data(serde_json::to_string(&closing).unwrap_or_default())));
                // OpenAI streaming sentinel — only on successful completion.
                events.push(Ok(Event::default().data("[DONE]")));
            }
            TokenEvent::Error(msg) => {
                tracing::error!("Streaming generation error: {msg}");
                let err = json!({
                    "error": { "message": "internal model error", "type": "server_error", "code": 500 }
                });
                events.push(Ok(Event::default().data(err.to_string())));
                // Do NOT emit [DONE] after an error — signals abnormal termination.
            }
        }

        stream::iter(events)
    });

    // Prepend the role announcement before the first token.
    stream::once(async move { Ok::<Event, Infallible>(opening) }).chain(mapped)
}

/// Convert an mpsc token stream into an SSE event stream for text completions.
///
/// Uses the same UTF-8 boundary buffering as `chat_sse_stream`. When
/// `logprobs_enabled` is true, each delta carries a per-chunk
/// `CompletionLogprobs` payload with 4-parallel-array shape aligned to
/// the substring boundaries of the current delta. [DONE] is only emitted
/// on successful completion — not after errors.
fn completion_sse_stream(
    rx: tokio::sync::mpsc::Receiver<TokenEvent>,
    tokenizer: Arc<pmetal_data::Tokenizer>,
    request_id: String,
    model_id: String,
    created: i64,
    state: Arc<AppState>,
    logprobs_enabled: bool,
) -> impl futures::Stream<Item = Result<Event, Infallible>> + Send + 'static {
    let token_stream = ReceiverStream::new(rx);

    // BPE boundaries can split multi-byte codepoints across tokens — see
    // crate::sse::IncrementalDecoder for the buffering rationale. When
    // logprobs are off, aux is always `None`; `push_with_aux` still drives
    // the decoder but drained aux is ignored.
    let mut decoder: IncrementalDecoder<Option<crate::engine::TokenLogprobEntry>> =
        IncrementalDecoder::new(Arc::clone(&tokenizer));
    // Running byte offset into the concatenated streamed text — aligns
    // `text_offset` across delta boundaries per OpenAI's shape.
    let mut running_offset: usize = 0;
    let tokenizer_for_aux = Arc::clone(&tokenizer);

    token_stream.flat_map(move |event| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();

        match event {
            TokenEvent::Token { id: token_id, logprob } => {
                let (new_text, drained_aux) = decoder.push_with_aux(token_id, logprob);
                if !new_text.is_empty() {
                    let logprobs_payload = if logprobs_enabled {
                        Some(build_completion_logprobs(
                            &tokenizer_for_aux,
                            drained_aux,
                            &mut running_offset,
                        ))
                    } else {
                        None
                    };
                    let mut choice = json!({
                        "index": 0,
                        "text": new_text,
                        "finish_reason": null,
                    });
                    if let Some(lp) = logprobs_payload {
                        choice["logprobs"] = serde_json::to_value(lp).unwrap_or(json!(null));
                    }
                    let chunk = json!({
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [choice],
                    });
                    events.push(Ok(Event::default().data(chunk.to_string())));
                }
            }
            TokenEvent::Done(finish_reason, metrics) => {
                let (remaining, _drained_aux) = decoder.flush_aux();
                if !remaining.is_empty() {
                    let flush = json!({
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "text": remaining,
                            "finish_reason": null,
                        }]
                    });
                    events.push(Ok(Event::default().data(flush.to_string())));
                }

                state.metrics.record(&metrics);
                let closing = json!({
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "text": "",
                        "finish_reason": finish_reason,
                    }]
                });
                events.push(Ok(Event::default().data(closing.to_string())));
                // OpenAI streaming sentinel — only on successful completion.
                events.push(Ok(Event::default().data("[DONE]")));
            }
            TokenEvent::Error(msg) => {
                tracing::error!("Streaming generation error: {msg}");
                let err = json!({
                    "error": { "message": "internal model error", "type": "server_error", "code": 500 }
                });
                events.push(Ok(Event::default().data(err.to_string())));
                // Do NOT emit [DONE] after an error — signals abnormal termination.
            }
        }

        stream::iter(events)
    })
}

/// Build a per-delta `CompletionLogprobs` payload from drained aux.
///
/// `running_offset` is the caller-owned byte cursor into the concatenated
/// streamed text — it advances by each drained token's decoded length so
/// `text_offset` entries align across chunk boundaries per OpenAI's shape.
/// When aux contains no logprob entries (e.g. an accelerated path that
/// can't produce them) the returned payload has empty arrays rather than
/// being `None` — callers opt in via `logprobs_enabled` upstream.
fn build_completion_logprobs(
    tokenizer: &Arc<pmetal_data::Tokenizer>,
    aux: Vec<Option<crate::engine::TokenLogprobEntry>>,
    running_offset: &mut usize,
) -> CompletionLogprobs {
    let entries: Vec<crate::engine::TokenLogprobEntry> = aux.into_iter().flatten().collect();
    let n = entries.len();
    let mut tokens = Vec::with_capacity(n);
    let mut token_logprobs = Vec::with_capacity(n);
    let mut top_logprobs = Vec::with_capacity(n);
    let mut text_offset = Vec::with_capacity(n);
    for entry in entries {
        let tok_str = tokenizer.decode(&[entry.token]).unwrap_or_default();
        let mut top = std::collections::HashMap::with_capacity(entry.top_logprobs.len());
        for (alt_id, lp) in entry.top_logprobs {
            let alt_str = tokenizer.decode(&[alt_id]).unwrap_or_default();
            top.insert(alt_str, lp);
        }
        text_offset.push(*running_offset);
        *running_offset += tok_str.len();
        tokens.push(tok_str);
        token_logprobs.push(entry.logprob);
        top_logprobs.push(top);
    }
    CompletionLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }
}

/// `POST /v1/embeddings` — OpenAI-compatible sentence embeddings.
///
/// Accepts a single string or an array via [`EmbeddingsInput`]; returns one
/// embedding entry per input. Uses mean pooling by default. The model's
/// pre-lm-head trunk is pulled through [`InferenceEngine::embed`], which
/// errors cleanly for architectures that don't expose hidden states
/// (Flux, Qwen3MoE, hybrid attn+mamba variants — see
/// `DynamicModel::forward_hidden` for the supported set).
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ServeError> {
    let mode = parse_pooling_mode(req.pooling.as_deref())
        .ok_or_else(|| ServeError::BadRequest("unknown pooling mode".into()))?;
    let inputs = req.input.into_batch();
    if inputs.is_empty() {
        return Err(ServeError::BadRequest("input must be non-empty".into()));
    }

    // Count prompt tokens once (tokenisation is fast + deterministic).
    let prompt_tokens: usize = inputs
        .iter()
        .map(|s| state.engine.tokenize(s).map(|ids| ids.len()))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum();

    let vectors = state.engine.embed(&inputs, mode).await?;

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            index,
            object: "embedding",
            embedding,
        })
        .collect();

    Ok(Json(EmbeddingsResponse {
        object: "list",
        data,
        model: state.engine.model_id().to_string(),
        usage: EmbeddingsUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Resolve a list of stop strings to token IDs.
///
/// Only stop strings that encode to exactly one token are accepted — multi-token
/// stop sequences require a separate detokenization buffer not yet implemented.
/// A warning is logged for multi-token entries so callers know they were dropped.
fn resolve_stop_ids(stop: &Option<Vec<String>>, engine: &InferenceEngine) -> Vec<u32> {
    stop.as_deref()
        .unwrap_or(&[])
        .iter()
        .filter_map(|s| match engine.tokenize(s) {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            Ok(ids) => {
                tracing::warn!(
                    "stop string {:?} encodes to {} tokens — \
                         only single-token stop strings are supported, ignoring",
                    s,
                    ids.len()
                );
                None
            }
            Err(_) => None,
        })
        .collect()
}
