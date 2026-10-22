//! HTTP route handlers for the OpenAI-compatible API.

use crate::assemble::{Assembler, AssemblerConfig, ChoiceEvent, Failure, Piece};
use crate::engine::{InferenceEngine, PreparedPrompt, RequestMetrics, SamplingParams, TokenEvent};
use crate::error::ServeError;
use crate::extract::ApiJson;
use crate::generate::{self, Collected};
use crate::request::{
    MAX_STOP_SEQUENCES, SamplingRequest, check_choices, reject_unsupported, resolve_stop_sequences,
    sampling_params,
};
use crate::structured::{Constraint, ConstraintSpec, ToolSchema, tool_call_grammar};
use crate::tools::{ParsedCall, ToolChoice as ToolSelection, ToolFormat, ToolPlan};
use crate::types::*;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use chrono::Utc;
use futures::stream::{self, Stream, StreamExt};
use pmetal_data::chat_templates::ToolDefinition;
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

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
    /// Tokens the draft model or MTP head proposed, over every speculative
    /// generation.
    pub sum_drafted_tokens: AtomicU64,
    /// Drafted tokens the target model accepted.
    pub sum_accepted_tokens: AtomicU64,
}

/// Pick the right streaming receiver: the continuous-batching pump
/// when enabled, else the single-request path. Falls through to the
/// single-request path if the pump rejects (e.g. queue saturated) so
/// saturation degrades to higher latency rather than 5xx errors.
///
/// A prompt with images or videos always takes the single-request path: the
/// pump drives each slot by token ids alone, so it has no way to feed the
/// model vision features.
pub(crate) fn stream_tokens(
    engine: &InferenceEngine,
    prompt: &PreparedPrompt,
    params: SamplingParams,
) -> tokio::sync::mpsc::Receiver<TokenEvent> {
    // A request that scores its prompt needs every prompt position's logits,
    // which a batch's prefill does not keep.
    if !prompt.has_media()
        && params.prompt_logprobs_top_n.is_none()
        && engine.continuous_batching_enabled()
    {
        match engine.generate_batched(&prompt.input_ids, params.clone()) {
            Ok(rx) => return rx,
            Err(e) => {
                tracing::warn!(
                    "continuous-batching enqueue failed ({e}); falling back to single-request path"
                );
            }
        }
    }
    engine.generate_streaming_prompt(prompt, params)
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
        if let Some(spec) = &metrics.speculative {
            self.sum_drafted_tokens
                .fetch_add(spec.drafted_tokens as u64, Ordering::Relaxed);
            self.sum_accepted_tokens
                .fetch_add(spec.accepted_tokens as u64, Ordering::Relaxed);
        }
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

        let drafted = self.sum_drafted_tokens.load(Ordering::Relaxed);
        let accepted = self.sum_accepted_tokens.load(Ordering::Relaxed);
        let acceptance = if drafted > 0 {
            accepted as f64 / drafted as f64
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
            "speculative_drafted_tokens": drafted,
            "speculative_accepted_tokens": accepted,
            "speculative_acceptance_rate": acceptance,
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
    /// The Prometheus registry behind `GET /metrics`.
    pub telemetry: crate::telemetry::Telemetry,
    pub request_permits: Arc<Semaphore>,
    /// How many permits `request_permits` started with.
    pub configured_permits: usize,
    /// How long one request may run; see [`crate::server::ServeConfig`].
    pub request_timeout: Option<std::time::Duration>,
    /// The most choices (`n`) a request may ask for.
    pub max_choices: usize,
}

impl AppState {
    pub(crate) fn try_acquire_request_permit(
        self: &Arc<Self>,
    ) -> Result<OwnedSemaphorePermit, ServeError> {
        Arc::clone(&self.request_permits)
            .try_acquire_owned()
            .map_err(|_| ServeError::Busy)
    }

    /// Count a finished generation in the JSON and the Prometheus metrics.
    pub(crate) fn record_request(&self, metrics: &RequestMetrics) {
        self.metrics.record(metrics);
        self.telemetry.record_request(metrics);
    }

    /// When a request that starts now must be done, and the limit in seconds.
    pub(crate) fn deadline(&self) -> Option<(std::time::Instant, u64)> {
        generate::deadline_after(self.request_timeout)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Route handlers
// ────────────────────────────────────────────────────────────────────────────

/// GET /health
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

fn served_model(state: &AppState) -> ModelInfo {
    ModelInfo {
        id: state.engine.model_id().to_string(),
        object: "model".to_string(),
        created: state.engine.created_at(),
        owned_by: "pmetal".to_string(),
    }
}

/// GET /v1/models
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![served_model(&state)],
    })
}

/// GET /v1/models/{id}
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<ModelInfo>, ServeError> {
    state.engine.check_model(&id)?;
    Ok(Json(served_model(&state)))
}

/// GET /v1/metrics
///
/// Returns rolling aggregate serving metrics as JSON.
pub async fn serving_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.metrics.snapshot())
}

// ────────────────────────────────────────────────────────────────────────────
// Streaming plumbing shared by the generation endpoints
// ────────────────────────────────────────────────────────────────────────────

/// What a stream has seen so far, across its choices.
#[derive(Debug, Default)]
pub(crate) struct StreamTotals {
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
    /// Prompt tokens the prefix cache supplied to the first choice.
    pub cached_tokens: usize,
    pub finished: usize,
    pub failed: bool,
}

pub(crate) type SseEvent = Result<Event, Infallible>;

/// Merge the choices' events into one SSE body: `opening` first, then each
/// choice's events turned into frames by `convert` as they arrive, then
/// whatever `closing` makes of the totals. `permit` is held until the
/// client has the last byte or goes away, which also cancels the generation.
pub(crate) fn merged_sse<S, C, T>(
    permit: OwnedSemaphorePermit,
    opening: Vec<Event>,
    streams: Vec<S>,
    mut convert: C,
    closing: T,
) -> impl Stream<Item = SseEvent> + Send + 'static
where
    S: Stream<Item = ChoiceEvent> + Send + 'static,
    C: FnMut(usize, ChoiceEvent, &mut StreamTotals) -> Vec<Event> + Send + 'static,
    T: FnOnce(&StreamTotals) -> Vec<Event> + Send + 'static,
{
    let totals = Arc::new(Mutex::new(StreamTotals::default()));
    let body_totals = Arc::clone(&totals);
    let tagged: Vec<_> = streams
        .into_iter()
        .enumerate()
        .map(|(i, s)| s.map(move |event| (i, event)).boxed())
        .collect();
    let body = stream::select_all(tagged).flat_map(move |(index, event)| {
        let mut totals = body_totals.lock().unwrap_or_else(|e| e.into_inner());
        if let ChoiceEvent::Done(done) = &event {
            totals.completion_tokens += done.finish.completion_tokens;
            totals.reasoning_tokens += done.finish.reasoning_tokens;
            if index == 0 {
                totals.cached_tokens = done.metrics.as_ref().map_or(0, |m| m.cached_prompt_tokens);
            }
            totals.finished += 1;
        }
        if matches!(event, ChoiceEvent::Failed(_)) {
            totals.failed = true;
        }
        let frames = convert(index, event, &mut totals);
        stream::iter(frames.into_iter().map(Ok))
    });
    let tail = stream::once(async move {
        let totals = totals.lock().unwrap_or_else(|e| e.into_inner());
        stream::iter(closing(&totals).into_iter().map(Ok))
    })
    .flatten();
    stream::iter(opening.into_iter().map(Ok))
        .chain(body)
        .chain(tail)
        // Held until the stream is dropped.
        .inspect(move |_| {
            let _ = &permit;
        })
}

/// An SSE response over `body`, with keep-alive comments.
pub(crate) fn sse_response(body: impl Stream<Item = SseEvent> + Send + 'static) -> Response {
    Sse::new(body)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// The frame OpenAI sends for a failure after the stream has begun; no
/// `[DONE]` follows it.
pub(crate) fn openai_error_event(failure: &Failure) -> Event {
    let error: ServeError = match failure {
        Failure::Timeout(secs) => ServeError::Timeout(*secs),
        Failure::Engine(message) => {
            tracing::error!("Streaming generation error: {message}");
            ServeError::Internal(message.clone())
        }
    };
    Event::default().data(error.describe().openai_body().to_string())
}

fn data_event(value: &impl serde::Serialize) -> Event {
    Event::default().data(serde_json::to_string(value).unwrap_or_default())
}

// ────────────────────────────────────────────────────────────────────────────
// Logprobs on the wire
// ────────────────────────────────────────────────────────────────────────────

/// OpenAI's entry for a token: its text, bytes and log-probability.
/// The text and bytes a token adds to a reply.
fn token_piece(
    engine: &InferenceEngine,
    vocab: Option<&crate::vocab::ByteVocab>,
    id: u32,
) -> (String, Vec<u8>) {
    match vocab {
        Some(vocab) => (vocab.text(id), vocab.bytes(id).to_vec()),
        None => {
            let text = engine.decode(&[id]).unwrap_or_default();
            let bytes = text.clone().into_bytes();
            (text, bytes)
        }
    }
}

fn wire_logprob(
    engine: &InferenceEngine,
    entry: crate::engine::TokenLogprobEntry,
) -> TokenLogprobContent {
    let vocab = engine.byte_vocab();
    let text_and_bytes = |id: u32| token_piece(engine, vocab.as_deref(), id);
    let (token, bytes) = text_and_bytes(entry.token);
    TokenLogprobContent {
        token,
        logprob: entry.logprob,
        bytes,
        top_logprobs: entry
            .top_logprobs
            .into_iter()
            .map(|(id, logprob)| {
                let (token, bytes) = text_and_bytes(id);
                TopLogprob {
                    token,
                    logprob,
                    bytes,
                }
            })
            .collect(),
    }
}

fn wire_logprobs(
    engine: &InferenceEngine,
    entries: Vec<crate::engine::TokenLogprobEntry>,
) -> ChatLogprobs {
    ChatLogprobs::new(
        entries
            .into_iter()
            .map(|entry| wire_logprob(engine, entry))
            .collect(),
    )
}

/// A fresh tool-call id, in OpenAI's form.
pub(crate) fn new_call_id() -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    format!("call_{}", &id[..24])
}

// ────────────────────────────────────────────────────────────────────────────
// POST /v1/chat/completions
// ────────────────────────────────────────────────────────────────────────────

/// OpenAI chat-completion fields that change what is generated and that this
/// server does not implement.
const CHAT_UNSUPPORTED: &[(&str, &str)] = &[
    ("audio", "this server generates text only"),
    ("prediction", "predicted outputs are not implemented"),
    ("web_search_options", "this server has no web search"),
    (
        "verbosity",
        "the model's answer length is set by its prompt and max_completion_tokens",
    ),
];

/// `tools` and `tool_choice` (or their deprecated `functions` and
/// `function_call`) as a [`ToolPlan`].
fn plan_tools(
    req: &ChatCompletionRequest,
    engine: &InferenceEngine,
) -> Result<ToolPlan, ServeError> {
    let mut tools = req.tools.clone();
    if tools.is_none() {
        tools = req.functions.as_ref().map(|functions| {
            functions
                .iter()
                .map(|function| ToolDefinition {
                    tool_type: "function".into(),
                    function: function.clone(),
                })
                .collect()
        });
    }
    if let Some((i, tool)) = tools
        .iter()
        .flatten()
        .enumerate()
        .find(|(_, t)| t.tool_type != "function")
    {
        return Err(ServeError::invalid(
            format!("tools.{i}.type"),
            format!(
                "tools of type `{}` are not supported; only `function`",
                tool.tool_type
            ),
        ));
    }
    let bad_choice =
        |why: String| ServeError::invalid_with_code("tool_choice", "invalid_value", why);
    let named = |name: Option<&str>| -> Result<ToolSelection, ServeError> {
        name.map(|name| ToolSelection::Named(name.to_string()))
            .ok_or_else(|| bad_choice("a named `tool_choice` needs `function.name`".into()))
    };
    let choice = match (&req.tool_choice, &req.function_call) {
        (Some(choice), _) => Some(choice.clone()),
        (None, Some(Value::String(mode))) => Some(ToolChoiceParam::Mode(mode.clone())),
        (None, Some(Value::Object(object))) => Some(ToolChoiceParam::Object(
            json!({"type": "function", "function": {"name": object.get("name")}})
                .as_object()
                .cloned()
                .unwrap_or_default(),
        )),
        _ => None,
    };
    let mut tools = tools.unwrap_or_default();
    let choice = match &choice {
        None => ToolSelection::Auto,
        Some(ToolChoiceParam::Mode(mode)) => match mode.as_str() {
            "none" => ToolSelection::None,
            "auto" => ToolSelection::Auto,
            "required" => ToolSelection::Required,
            other => {
                return Err(bad_choice(format!(
                    "`tool_choice` must be `none`, `auto`, `required` or a named function, not `{other}`"
                )));
            }
        },
        Some(ToolChoiceParam::Object(object)) => match object.get("type").and_then(Value::as_str) {
            Some("function") => named(object["function"]["name"].as_str())?,
            // The set of tools the model may call this turn, and whether it
            // must call one.
            Some("allowed_tools") => {
                let allowed = &object["allowed_tools"];
                let names: Vec<&str> = allowed["tools"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|t| t["function"]["name"].as_str())
                    .collect();
                tools.retain(|t| names.contains(&t.function.name.as_str()));
                match allowed["mode"].as_str() {
                    Some("required") => ToolSelection::Required,
                    _ => ToolSelection::Auto,
                }
            }
            other => {
                return Err(bad_choice(format!(
                    "`tool_choice.type` must be `function` or `allowed_tools`, not {other:?}"
                )));
            }
        },
    };
    let plan = ToolPlan::new(
        tools,
        choice,
        req.parallel_tool_calls == Some(false),
        engine.tool_format(),
    )
    .map_err(|why| ServeError::invalid("tool_choice", why))?;
    Ok(plan)
}

/// The grammar a chat request's output must follow: the calls a required or
/// named `tool_choice` demands, or the `response_format`.
pub(crate) async fn chat_constraint(
    engine: &InferenceEngine,
    response_format: Option<&ResponseFormat>,
    plan: &ToolPlan,
    prompt: &[u32],
    thinks: bool,
) -> Result<Option<Constraint>, ServeError> {
    let forced = matches!(
        plan.choice,
        ToolSelection::Required | ToolSelection::Named(_)
    );
    let structured = !matches!(response_format, None | Some(ResponseFormat::Text));
    if structured && (forced || plan.prompt.is_some()) {
        return Err(ServeError::unsupported(
            "response_format",
            "it cannot be combined with `tools`; the reply is either a structured answer or tool calls",
        ));
    }
    let spec = if forced {
        let named = match &plan.choice {
            ToolSelection::Named(name) => Some(name.as_str()),
            _ => None,
        };
        let schemas: Vec<ToolSchema> = plan
            .prompt
            .iter()
            .flatten()
            .filter(|tool| named.is_none_or(|name| tool.function.name == name))
            .map(|tool| ToolSchema {
                name: tool.function.name.clone(),
                parameters: tool.function.parameters.clone(),
            })
            .collect();
        let format = plan.parse.as_ref().map_or(ToolFormat::Hermes, |(f, _)| *f);
        let lark = tool_call_grammar(format, &schemas, !plan.single_call, &|text| {
            engine.token_id(text)
        })
        .ok_or_else(|| {
            ServeError::unsupported(
                "tool_choice",
                format!(
                    "this model writes its tool calls in the {format:?} syntax, which cannot be \
                     enforced; use `auto`"
                ),
            )
        })?;
        ConstraintSpec::Lark(lark)
    } else {
        match response_format {
            None | Some(ResponseFormat::Text) => return Ok(None),
            Some(ResponseFormat::JsonObject) => ConstraintSpec::JsonObject,
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                let schema = json_schema.schema.clone().ok_or_else(|| {
                    ServeError::invalid(
                        "response_format.json_schema.schema",
                        "a `json_schema` response format needs its `schema`",
                    )
                })?;
                ConstraintSpec::JsonSchema(schema)
            }
        }
    };
    engine
        .compile_constraint(spec, prompt, thinks)
        .await
        .map(Some)
}

/// The roles a chat message may have, with `developer` read as `system`
/// (it is OpenAI's name for it) and `function` as `tool`.
fn normalize_messages(messages: &[ChatMessage]) -> Result<Vec<ChatMessage>, ServeError> {
    if messages.is_empty() {
        return Err(ServeError::invalid(
            "messages",
            "`messages` must hold at least one message",
        ));
    }
    messages
        .iter()
        .enumerate()
        .map(|(i, message)| {
            let role = match message.role.as_str() {
                "developer" => "system",
                "function" => "tool",
                role @ ("system" | "user" | "assistant" | "tool") => role,
                other => {
                    return Err(ServeError::invalid_with_code(
                        format!("messages.{i}.role"),
                        "invalid_value",
                        format!(
                            "`{other}` is not a role; use system, developer, user, assistant or tool"
                        ),
                    ));
                }
            };
            Ok(ChatMessage {
                role: role.to_string(),
                ..message.clone()
            })
        })
        .collect()
}

/// POST /v1/chat/completions
///
/// Supports both non-streaming (default) and streaming (`stream: true`) modes.
///
/// Non-streaming: generates all tokens, returns a single JSON response.
///
/// Streaming: returns `text/event-stream` SSE. Each generated token is sent
/// to the client individually as it is produced — no batching or buffering
/// beyond what UTF-8 and stop sequences need. Wire format follows the OpenAI
/// streaming protocol.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    ApiJson(req): ApiJson<ChatCompletionRequest>,
) -> Result<Response, ServeError> {
    state.engine.check_model(&req.model)?;
    reject_unsupported(&req.other, CHAT_UNSUPPORTED)?;
    if let Some(modalities) = req.other.get("modalities") {
        if modalities != &json!(["text"]) {
            return Err(ServeError::unsupported(
                "modalities",
                "this server generates text only",
            ));
        }
    }
    if req
        .mm_processor_kwargs
        .as_ref()
        .is_some_and(|kwargs| !kwargs.is_null() && kwargs != &json!({}))
    {
        return Err(ServeError::unsupported(
            "mm_processor_kwargs",
            "images and videos are preprocessed with the checkpoint's own processor settings",
        ));
    }
    let n = check_choices(req.n, state.max_choices)?;
    let stream = req.stream.unwrap_or(false);
    if req.stream_options.is_some() && !stream {
        return Err(ServeError::invalid(
            "stream_options",
            "`stream_options` can only be set when `stream` is true",
        ));
    }
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);
    let logprobs_top_n = match (req.logprobs.unwrap_or(false), req.top_logprobs) {
        (true, top) => Some(top.unwrap_or(0) as usize),
        (false, Some(_)) => {
            return Err(ServeError::invalid(
                "top_logprobs",
                "`top_logprobs` can only be set when `logprobs` is true",
            ));
        }
        (false, None) => None,
    };
    let messages = normalize_messages(&req.messages)?;
    let tool_plan = plan_tools(&req, &state.engine)?;
    let permit = state.try_acquire_request_permit()?;

    // The chat template renders the messages (and the tools the request
    // offers), media are preprocessed, a grammar is compiled, and the
    // sampling is settled against the context that is left.
    let template_kwargs = req.template_kwargs();
    let stops =
        resolve_stop_sequences(req.stop.as_deref(), Some(MAX_STOP_SEQUENCES), &state.engine)?;
    let generate::Prepared { prompt, params } = generate::prepare(
        &state,
        &messages,
        &tool_plan,
        &template_kwargs,
        &SamplingRequest {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            max_tokens: req.max_completion_tokens.or(req.max_tokens),
            repetition_penalty: req.repetition_penalty,
            frequency_penalty: req.frequency_penalty,
            presence_penalty: req.presence_penalty,
            seed: req.seed,
            logit_bias: req.logit_bias.as_ref(),
            logprobs_top_n,
            ..SamplingRequest::default()
        },
        &stops,
        req.response_format.as_ref(),
    )
    .await?;
    let prompt_tokens = prompt.input_ids.len();

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.engine.model_id().to_string();
    // Use request-time timestamp per OpenAI spec — not model creation time.
    let created = Utc::now().timestamp();

    let make_assembler = |_: usize| {
        Assembler::new(AssemblerConfig {
            tokenizer: state.engine.tokenizer_arc(),
            prompt: prompt.input_ids.clone(),
            stops: stops.sequences.clone(),
            tools: tool_plan.parse.clone(),
            single_tool_call: tool_plan.single_call,
        })
    };
    let choices = generate::start_choices(&state, &prompt, &params, n, make_assembler);
    let wants_logprobs = logprobs_top_n.is_some();

    if stream {
        let engine_state = Arc::clone(&state);
        let chunk = {
            let (id, model) = (request_id.clone(), model_id.clone());
            move |index: usize,
                  delta: ChatDelta,
                  logprobs: Option<ChatLogprobs>,
                  finish_reason: Option<String>,
                  usage: Option<Value>|
                  -> Event {
                data_event(&ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: if index == usize::MAX {
                        Vec::new()
                    } else {
                        vec![ChatChunkChoice {
                            index,
                            delta,
                            finish_reason,
                            logprobs,
                        }]
                    },
                    usage,
                })
            }
        };
        let with_usage = |index, delta, logprobs, finish| {
            chunk(
                index,
                delta,
                logprobs,
                finish,
                include_usage.then_some(Value::Null),
            )
        };
        let opening = (0..n)
            .map(|i| {
                with_usage(
                    i,
                    ChatDelta {
                        role: Some("assistant"),
                        content: Some(String::new()),
                        ..ChatDelta::default()
                    },
                    None,
                    None,
                )
            })
            .collect();
        let mut opened_calls = vec![0usize; n];
        let convert = {
            let chunk = chunk.clone();
            move |index: usize, event: ChoiceEvent, _totals: &mut StreamTotals| -> Vec<Event> {
                let frame = |delta, logprobs, finish: Option<String>| {
                    chunk(
                        index,
                        delta,
                        logprobs,
                        finish,
                        include_usage.then_some(Value::Null),
                    )
                };
                match event {
                    ChoiceEvent::Piece(Piece::Reasoning(text)) => vec![frame(
                        ChatDelta {
                            reasoning_content: Some(text),
                            ..ChatDelta::default()
                        },
                        None,
                        None,
                    )],
                    ChoiceEvent::Piece(Piece::Content { text, logprobs }) => {
                        let logprobs =
                            wants_logprobs.then(|| wire_logprobs(&engine_state.engine, logprobs));
                        vec![frame(
                            ChatDelta {
                                content: Some(text),
                                ..ChatDelta::default()
                            },
                            logprobs,
                            None,
                        )]
                    }
                    // Only completions score their prompt.
                    ChoiceEvent::PromptLogprobs(_) => Vec::new(),
                    ChoiceEvent::Piece(Piece::ToolCall(call)) => {
                        let slot = opened_calls[index];
                        opened_calls[index] += 1;
                        tool_call_frames(slot, call)
                            .into_iter()
                            .map(|delta| frame(delta, None, None))
                            .collect()
                    }
                    ChoiceEvent::Done(done) => vec![frame(
                        ChatDelta::default(),
                        None,
                        Some(done.finish.reason.to_string()),
                    )],
                    ChoiceEvent::Failed(failure) => vec![openai_error_event(&failure)],
                }
            }
        };
        let closing = {
            let chunk = chunk.clone();
            move |totals: &StreamTotals| -> Vec<Event> {
                if totals.failed {
                    // No usage and no [DONE] after an error: the stream ended
                    // abnormally.
                    return Vec::new();
                }
                let mut frames = Vec::new();
                if include_usage {
                    let usage = generate::usage_of(
                        prompt_tokens,
                        totals.cached_tokens,
                        [(totals.completion_tokens, totals.reasoning_tokens)],
                    );
                    frames.push(chunk(
                        usize::MAX,
                        ChatDelta::default(),
                        None,
                        None,
                        serde_json::to_value(usage).ok(),
                    ));
                }
                frames.push(Event::default().data("[DONE]"));
                frames
            }
        };
        let body = merged_sse(permit, opening, choices, convert, closing);
        return Ok(sse_response(body));
    }

    // ── Non-streaming path ───────────────────────────────────────────────────
    let replies = generate::collect_all(choices).await?;
    let usage = generate::usage_of(
        prompt_tokens,
        replies.first().map_or(0, Collected::cached_tokens),
        replies
            .iter()
            .map(|r| (r.completion_tokens(), r.reasoning_tokens())),
    );
    let choices = replies
        .into_iter()
        .enumerate()
        .map(|(index, reply)| chat_choice(&state.engine, index, reply, wants_logprobs))
        .collect();
    Ok(Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: model_id,
        choices,
        usage,
        system_fingerprint: None,
    })
    .into_response())
}

/// The stream deltas that open a tool call and give its arguments.
fn tool_call_frames(slot: usize, call: ParsedCall) -> Vec<ChatDelta> {
    let open = ToolCallDelta {
        index: slot,
        id: Some(new_call_id()),
        kind: Some("function"),
        function: FunctionDelta {
            name: Some(call.name),
            arguments: Some(String::new()),
        },
    };
    let arguments = ToolCallDelta {
        index: slot,
        id: None,
        kind: None,
        function: FunctionDelta {
            name: None,
            arguments: Some(call.arguments),
        },
    };
    [open, arguments]
        .into_iter()
        .map(|call| ChatDelta {
            tool_calls: Some(vec![call]),
            ..ChatDelta::default()
        })
        .collect()
}

fn chat_choice(
    engine: &InferenceEngine,
    index: usize,
    reply: Collected,
    wants_logprobs: bool,
) -> ChatChoice {
    let finish_reason = reply.finish_reason().to_string();
    let tool_calls: Vec<WireToolCall> = reply
        .tool_calls
        .into_iter()
        .map(|call| WireToolCall::new(new_call_id(), call.name, call.arguments))
        .collect();
    let has_calls = !tool_calls.is_empty();
    ChatChoice {
        index,
        message: ResponseMessage {
            role: "assistant",
            // OpenAI: `content` is null when the reply is tool calls alone.
            content: (!has_calls || !reply.content.trim().is_empty()).then_some(reply.content),
            refusal: None,
            tool_calls: has_calls.then_some(tool_calls),
            reasoning_content: (!reply.reasoning.is_empty()).then_some(reply.reasoning),
        },
        finish_reason: Some(finish_reason),
        logprobs: wants_logprobs.then(|| wire_logprobs(engine, reply.logprobs)),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// POST /v1/completions
// ────────────────────────────────────────────────────────────────────────────

/// Fields of the completions API that change what is generated and that this
/// server does not implement.
const COMPLETION_UNSUPPORTED: &[(&str, &str)] = &[];

/// The prompts of a completions request, as token ids, with the text each
/// stands for (for `echo`).
fn completion_prompts(
    engine: &InferenceEngine,
    prompt: &CompletionPrompt,
) -> Result<Vec<(Vec<u32>, String)>, ServeError> {
    let text = |ids: &[u32]| engine.decode(ids).unwrap_or_default();
    let one_text = |s: &String| -> Result<(Vec<u32>, String), ServeError> {
        Ok((engine.tokenize(s)?, s.clone()))
    };
    let prompts = match prompt {
        CompletionPrompt::Text(s) => vec![one_text(s)?],
        CompletionPrompt::Texts(list) => list.iter().map(one_text).collect::<Result<_, _>>()?,
        CompletionPrompt::Tokens(ids) => vec![(ids.clone(), text(ids))],
        CompletionPrompt::TokenLists(lists) => {
            lists.iter().map(|ids| (ids.clone(), text(ids))).collect()
        }
    };
    // A prompt of no tokens leaves the model nothing to continue from.
    if prompts.is_empty() || prompts.iter().any(|(ids, _)| ids.is_empty()) {
        return Err(ServeError::invalid("prompt", "`prompt` must not be empty"));
    }
    Ok(prompts)
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<Arc<AppState>>,
    ApiJson(req): ApiJson<CompletionRequest>,
) -> Result<Response, ServeError> {
    state.engine.check_model(&req.model)?;
    reject_unsupported(&req.other, COMPLETION_UNSUPPORTED)?;
    if req.suffix.is_some() {
        return Err(ServeError::unsupported(
            "suffix",
            "insert mode is not implemented",
        ));
    }
    let n = check_choices(req.n, state.max_choices)?;
    if req
        .best_of
        .is_some_and(|best_of| best_of != 1 && best_of != n)
    {
        return Err(ServeError::unsupported(
            "best_of",
            "ranking candidates is not implemented; leave it at 1",
        ));
    }
    let stream = req.stream.unwrap_or(false);
    if req.stream_options.is_some() && !stream {
        return Err(ServeError::invalid(
            "stream_options",
            "`stream_options` can only be set when `stream` is true",
        ));
    }
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);
    let echo = req.echo.unwrap_or(false);
    if req.logprobs.is_some_and(|n| n > 5) {
        return Err(ServeError::invalid_with_code(
            "logprobs",
            "invalid_value",
            "`logprobs` is at most 5",
        ));
    }
    let permit = state.try_acquire_request_permit()?;

    let prompts = completion_prompts(&state.engine, &req.prompt)?;
    let stops =
        resolve_stop_sequences(req.stop.as_deref(), Some(MAX_STOP_SEQUENCES), &state.engine)?;
    let logprobs_top_n = req.logprobs.map(|n| n as usize);
    // An echoed prompt comes with its own log-probabilities.
    let prompt_logprobs_top_n = logprobs_top_n.filter(|_| echo);
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.engine.model_id().to_string();
    // Use request-time timestamp per OpenAI spec — not model creation time.
    let created = Utc::now().timestamp();

    // One generation per (prompt, choice); a raw completion has no chat
    // template, so no thinking.
    let mut streams = Vec::new();
    let mut echoes = Vec::new();
    let mut prompt_ids: Vec<Arc<Vec<u32>>> = Vec::new();
    let mut prompt_tokens = 0usize;
    for (ids, text) in &prompts {
        let params = sampling_params(
            &state.engine,
            &SamplingRequest {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                min_p: req.min_p,
                max_tokens: req.max_tokens,
                repetition_penalty: req.repetition_penalty,
                frequency_penalty: req.frequency_penalty,
                presence_penalty: req.presence_penalty,
                seed: req.seed,
                logit_bias: req.logit_bias.as_ref(),
                logprobs_top_n,
                prompt_logprobs_top_n,
            },
            &stops,
            false,
            ids.len(),
        )?;
        state.engine.validate_sampling_params(&params)?;
        prompt_tokens += ids.len();
        let prepared = PreparedPrompt::from_tokens(ids.clone());
        let make_assembler = |_: usize| {
            Assembler::new(AssemblerConfig {
                tokenizer: state.engine.tokenizer_arc(),
                prompt: ids.clone(),
                stops: stops.sequences.clone(),
                tools: None,
                single_tool_call: false,
            })
        };
        streams.extend(generate::start_choices(
            &state,
            &prepared,
            &params,
            n,
            make_assembler,
        ));
        echoes.extend(std::iter::repeat_n(text.clone(), n));
        prompt_ids.extend(std::iter::repeat_n(Arc::new(ids.clone()), n));
    }
    let total_choices = streams.len();

    if stream {
        let engine_state = Arc::clone(&state);
        let frame = {
            let (id, model) = (request_id.clone(), model_id.clone());
            move |index: usize,
                  text: String,
                  logprobs: Option<CompletionLogprobs>,
                  finish_reason: Option<String>,
                  usage: Option<Value>|
                  -> Event {
                let choices = if index == usize::MAX {
                    json!([])
                } else {
                    json!([{
                        "index": index,
                        "text": text,
                        "logprobs": logprobs,
                        "finish_reason": finish_reason,
                    }])
                };
                let mut chunk = json!({
                    "id": id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": choices,
                });
                if let Some(usage) = usage {
                    chunk["usage"] = usage;
                }
                Event::default().data(chunk.to_string())
            }
        };
        // With `logprobs` the echoed prompt goes out with its scores, when the
        // engine has them; without, it can open the stream at once.
        let opening: Vec<Event> = if echo && prompt_logprobs_top_n.is_none() {
            echoes
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    frame(
                        i,
                        text.clone(),
                        None,
                        None,
                        include_usage.then_some(Value::Null),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut offsets = vec![0usize; total_choices];
        let convert = {
            let frame = frame.clone();
            move |index: usize, event: ChoiceEvent, _totals: &mut StreamTotals| -> Vec<Event> {
                let usage = || include_usage.then_some(Value::Null);
                match event {
                    ChoiceEvent::PromptLogprobs(scored) => {
                        let payload = completion_logprobs(
                            &engine_state.engine,
                            Some((&prompt_ids[index][..], scored)),
                            Vec::new(),
                            &mut offsets[index],
                        );
                        vec![frame(index, echoes[index].clone(), Some(payload), None, usage())]
                    }
                    ChoiceEvent::Piece(Piece::Content { text, logprobs }) => {
                        let payload = logprobs_top_n.is_some().then(|| {
                            completion_logprobs(
                                &engine_state.engine,
                                None,
                                logprobs,
                                &mut offsets[index],
                            )
                        });
                        vec![frame(index, text, payload, None, usage())]
                    }
                    // A raw completion has no reasoning channel or tools.
                    ChoiceEvent::Piece(_) => Vec::new(),
                    ChoiceEvent::Done(done) => vec![frame(
                        index,
                        String::new(),
                        None,
                        Some(done.finish.reason.to_string()),
                        usage(),
                    )],
                    ChoiceEvent::Failed(failure) => vec![openai_error_event(&failure)],
                }
            }
        };
        let closing = {
            let frame = frame.clone();
            move |totals: &StreamTotals| -> Vec<Event> {
                if totals.failed {
                    return Vec::new();
                }
                let mut frames = Vec::new();
                if include_usage {
                    let usage = generate::usage_of(
                        prompt_tokens,
                        totals.cached_tokens,
                        [(totals.completion_tokens, 0)],
                    );
                    frames.push(frame(
                        usize::MAX,
                        String::new(),
                        None,
                        None,
                        serde_json::to_value(usage).ok(),
                    ));
                }
                frames.push(Event::default().data("[DONE]"));
                frames
            }
        };
        return Ok(sse_response(merged_sse(
            permit, opening, streams, convert, closing,
        )));
    }

    // ── Non-streaming completions ────────────────────────────────────────────
    let replies = generate::collect_all(streams).await?;
    let usage = generate::usage_of(
        prompt_tokens,
        replies.first().map_or(0, Collected::cached_tokens),
        replies.iter().map(|r| (r.completion_tokens(), 0)),
    );
    let choices = replies
        .into_iter()
        .zip(echoes)
        .zip(prompt_ids)
        .enumerate()
        .map(|(index, ((reply, echoed), ids))| {
            let finish_reason = reply.finish_reason().to_string();
            let scored = reply.prompt_logprobs.map(|scored| (&ids[..], scored));
            let logprobs = logprobs_top_n.is_some().then(|| {
                completion_logprobs(&state.engine, scored, reply.logprobs, &mut 0)
            });
            CompletionChoice {
                index,
                text: if echo {
                    format!("{echoed}{}", reply.content)
                } else {
                    reply.content
                },
                finish_reason: Some(finish_reason),
                logprobs,
            }
        })
        .collect();
    Ok(Json(CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created,
        model: model_id,
        choices,
        usage,
        system_fingerprint: None,
    })
    .into_response())
}

/// The most likely tokens at one position as the legacy completions API
/// reports them: the top `n` of the whole distribution, where `n` is how many
/// alternatives the engine found besides the chosen token, so the chosen
/// token is among them when it is one of them. Ties go to the lower id.
fn legacy_top(entry: &crate::engine::TokenLogprobEntry) -> Vec<(u32, f32)> {
    let n = entry.top_logprobs.len();
    let mut top = entry.top_logprobs.clone();
    top.push((entry.token, entry.logprob));
    top.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    top.truncate(n);
    top
}

/// OpenAI's four parallel arrays for the tokens of a piece of a completion.
///
/// `prompt` is an echoed prompt's token ids and its scores, one per token
/// after the first (see [`TokenEvent::PromptLogprobs`]): the arrays then start
/// with the prompt's tokens, the first of which has `null` log-probabilities.
/// `running_offset` is the caller-owned byte cursor into the concatenated
/// text: it advances by each token's text so `text_offset` entries align
/// across chunk boundaries.
fn completion_logprobs(
    engine: &InferenceEngine,
    prompt: Option<(&[u32], Vec<crate::engine::TokenLogprobEntry>)>,
    entries: Vec<crate::engine::TokenLogprobEntry>,
    running_offset: &mut usize,
) -> CompletionLogprobs {
    let vocab = engine.byte_vocab();
    let piece = |id: u32| token_piece(engine, vocab.as_deref(), id);
    let n = entries.len() + prompt.as_ref().map_or(0, |(ids, _)| ids.len());
    let mut tokens = Vec::with_capacity(n);
    let mut token_logprobs = Vec::with_capacity(n);
    let mut top_logprobs = Vec::with_capacity(n);
    let mut text_offset = Vec::with_capacity(n);
    let mut push = |id: u32, scored: Option<&crate::engine::TokenLogprobEntry>| {
        let (text, bytes) = piece(id);
        text_offset.push(*running_offset);
        *running_offset += bytes.len();
        tokens.push(text);
        token_logprobs.push(scored.map(|entry| entry.logprob));
        top_logprobs.push(scored.map(|entry| {
            let mut map = std::collections::HashMap::new();
            for (alt, logprob) in legacy_top(entry) {
                // Two tokens can spell the same text; the likelier one stays.
                map.entry(piece(alt).0).or_insert(logprob);
            }
            map
        }));
    };
    if let Some((ids, scored)) = &prompt {
        if let Some(first) = ids.first() {
            push(*first, None);
        }
        for (id, entry) in ids.iter().skip(1).zip(scored) {
            push(*id, Some(entry));
        }
    }
    for entry in &entries {
        push(entry.token, Some(entry));
    }
    CompletionLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// POST /v1/embeddings
// ────────────────────────────────────────────────────────────────────────────

/// The most inputs one embeddings request may hold (OpenAI's limit).
const MAX_EMBEDDING_INPUTS: usize = 2048;

/// The most tokens one embeddings request may hold in all (OpenAI's limit).
const MAX_EMBEDDING_TOKENS: usize = 300_000;

/// Scale `vector` to unit length (a zero vector stays zero).
fn l2_normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let scale = (1.0 / norm) as f32;
        vector.iter_mut().for_each(|v| *v *= scale);
    }
}

/// A vector as the response carries it: a list of floats, or (OpenAI's
/// `encoding_format: "base64"`) its little-endian f32 bytes in base64.
fn embedding_vector(vector: Vec<f32>, base64: bool) -> EmbeddingVector {
    use base64::Engine as _;
    if base64 {
        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        EmbeddingVector::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
    } else {
        EmbeddingVector::Float(vector)
    }
}

/// `POST /v1/embeddings` — OpenAI-compatible embeddings.
///
/// `input` is a string, a list of strings, token ids, or a list of token-id
/// lists; the reply holds one vector per input, in order. Text is tokenized
/// as the model's tokenizer does (its own special tokens included), pooled
/// the way the model's sentence-transformers files say (else by mean) and
/// scaled to unit length. `dimensions` keeps the leading components and
/// renormalizes (Matryoshka truncation, for a model trained so);
/// `encoding_format: "base64"` sends each vector as little-endian f32 bytes.
/// The model's pre-lm-head trunk is pulled through
/// [`InferenceEngine::embed`], which errors cleanly for architectures that
/// don't expose hidden states (see `DynamicModel::forward_hidden` for the
/// supported set).
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    ApiJson(req): ApiJson<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ServeError> {
    state.engine.check_model(&req.model)?;
    let base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(ServeError::invalid_with_code(
                "encoding_format",
                "invalid_value",
                format!("`encoding_format` is `float` or `base64`, not `{other}`"),
            ));
        }
    };
    if req.dimensions == Some(0) {
        return Err(ServeError::invalid_with_code(
            "dimensions",
            "invalid_value",
            "`dimensions` must be at least 1",
        ));
    }
    let defaults = state.engine.embedding_defaults();
    let mode = crate::types::parse_pooling_mode(req.pooling.as_deref())
        .ok_or_else(|| ServeError::invalid("pooling", "unknown pooling mode"))?
        .unwrap_or(defaults.pooling);
    let inputs = req.input.into_batch();
    if inputs.is_empty() || inputs.len() > MAX_EMBEDDING_INPUTS {
        return Err(ServeError::invalid(
            "input",
            format!("`input` holds 1 to {MAX_EMBEDDING_INPUTS} inputs"),
        ));
    }
    let permit = state.try_acquire_request_permit()?;

    let max_input = defaults
        .max_input_tokens
        .unwrap_or(usize::MAX)
        .min(state.engine.room_after(0).unwrap_or(usize::MAX));
    let mut tokenized = Vec::with_capacity(inputs.len());
    for (i, item) in inputs.into_iter().enumerate() {
        let ids = match item {
            EmbeddingItem::Text(text) => {
                if text.is_empty() {
                    return Err(ServeError::invalid(
                        format!("input.{i}"),
                        "an input must not be empty",
                    ));
                }
                state.engine.tokenize_for_embedding(&text)?
            }
            EmbeddingItem::Tokens(ids) => ids,
        };
        if ids.is_empty() {
            return Err(ServeError::invalid(
                format!("input.{i}"),
                "an input must not be empty",
            ));
        }
        if ids.len() > max_input {
            return Err(ServeError::invalid_with_code(
                format!("input.{i}"),
                "context_length_exceeded",
                format!(
                    "an input is {} tokens; this model takes at most {max_input}",
                    ids.len()
                ),
            ));
        }
        tokenized.push(ids);
    }
    let prompt_tokens: usize = tokenized.iter().map(Vec::len).sum();
    if prompt_tokens > MAX_EMBEDDING_TOKENS {
        return Err(ServeError::invalid(
            "input",
            format!("`input` holds {prompt_tokens} tokens in all; at most {MAX_EMBEDDING_TOKENS}"),
        ));
    }

    let mut vectors = state.engine.embed(&tokenized, mode).await?;
    drop(permit);
    if let Some(dimensions) = req.dimensions {
        let width = vectors.first().map_or(0, Vec::len);
        if dimensions > width {
            return Err(ServeError::invalid_with_code(
                "dimensions",
                "invalid_value",
                format!(
                    "this model's embeddings have {width} dimensions; `dimensions` cannot be {dimensions}"
                ),
            ));
        }
        vectors.iter_mut().for_each(|v| v.truncate(dimensions));
    }
    // Truncating cuts the length, so a truncated vector is scaled again.
    if defaults.normalize || req.dimensions.is_some() {
        vectors.iter_mut().for_each(|v| l2_normalize(v));
    }

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            index,
            object: "embedding",
            embedding: embedding_vector(embedding, base64),
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
