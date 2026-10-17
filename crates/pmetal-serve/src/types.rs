//! OpenAI-compatible request/response types.

use pmetal_data::chat_templates::{ChatTemplateKwargs, ToolCall, ToolDefinition};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

/// Deserialize a field that may be either a single string or an array of strings.
///
/// The OpenAI API accepts `"stop": "token"` (string) or `"stop": ["a", "b"]`
/// (array) — both map to `Option<Vec<String>>`.
fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        One(String),
        Many(Vec<String>),
    }
    Ok(
        Option::<StringOrVec>::deserialize(deserializer)?.map(|stop| match stop {
            StringOrVec::One(one) => vec![one],
            StringOrVec::Many(many) => many,
        }),
    )
}

// ────────────────────────────────────────────────────────────────────────────
// Messages
// ────────────────────────────────────────────────────────────────────────────

/// Chat message.
///
/// A request's `content` is a string, `null`, or a list of content parts
/// (`{"type": "text", "text": ...}`, `{"type": "image_url", ...}`, ...).
/// `content` holds the text parts joined; a list that carries anything else
/// is kept verbatim in [`parts`](Self::parts) for
/// [`crate::media`] to validate and place.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "WireChatMessage")]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Tool calls made by the assistant, as a request replays them. A call's
    /// `arguments` arrive as a JSON string (OpenAI's wire form); the chat
    /// template reads them as an object, which [`crate::media`] hands it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The request's content parts, in order, when they are more than text
    /// (an image, a video, or a part type this server may not accept). Never
    /// serialized: responses carry text.
    #[serde(skip_serializing)]
    pub parts: Option<Vec<Value>>,
    /// The thinking behind an assistant turn, sent back apart from its
    /// `content`. Chat templates that render thinking themselves read it,
    /// e.g. to keep earlier turns' thinking under Qwen3.8's
    /// `preserve_thinking`. A `<think>…</think>` left inside `content` is
    /// split out the same way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Which call a `tool` message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// A message whose content is plain text.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            parts: None,
            reasoning_content: None,
            tool_call_id: None,
        }
    }
}

/// [`ChatMessage`] as it arrives.
#[derive(Deserialize)]
struct WireChatMessage {
    role: String,
    #[serde(default)]
    content: WireContent,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

/// A message's `content`: a string, `null` (an assistant turn that only
/// calls tools), or a list of parts.
#[derive(Deserialize, Default)]
#[serde(untagged)]
enum WireContent {
    #[default]
    Null,
    Text(String),
    Parts(Vec<Value>),
}

impl From<WireChatMessage> for ChatMessage {
    fn from(wire: WireChatMessage) -> Self {
        let (content, parts) = match wire.content {
            WireContent::Null => (String::new(), None),
            WireContent::Text(text) => (text, None),
            WireContent::Parts(parts) => {
                let text = |part: &Value| -> Option<String> {
                    (part.get("type")?.as_str()? == "text")
                        .then(|| part.get("text")?.as_str().map(str::to_owned))?
                };
                let content = parts.iter().filter_map(text).collect::<String>();
                let text_only = parts.iter().all(|part| text(part).is_some());
                (content, (!text_only).then_some(parts))
            }
        };
        Self {
            role: wire.role,
            content,
            tool_calls: wire.tool_calls,
            parts,
            reasoning_content: wire.reasoning_content,
            tool_call_id: wire.tool_call_id,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Request options shared by the generation endpoints
// ────────────────────────────────────────────────────────────────────────────

/// `stream_options`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    /// Send a last chunk, before `[DONE]`, with the request's token usage.
    #[serde(default)]
    pub include_usage: Option<bool>,
    /// OpenAI pads streamed deltas with random characters against a
    /// side channel on the network path; a local server adds none, so the
    /// option is read and has nothing to switch.
    #[serde(default)]
    pub include_obfuscation: Option<bool>,
}

/// `tool_choice`: a mode, or one named function (or a set of allowed ones).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoiceParam {
    /// `"none"`, `"auto"` or `"required"`.
    Mode(String),
    /// `{"type": "function", "function": {"name": ...}}`, or
    /// `{"type": "allowed_tools", "allowed_tools": {...}}`.
    Object(Map<String, Value>),
}

/// `response_format`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Plain text (the default).
    Text,
    /// Any valid JSON object.
    JsonObject,
    /// JSON that conforms to a schema.
    JsonSchema { json_schema: JsonSchemaSpec },
}

/// The `json_schema` of a `response_format`.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonSchemaSpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub schema: Option<Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

/// Chat completion request (POST /v1/chat/completions).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Output budget; absent means the model's default (see
    /// [`InferenceEngine::default_max_tokens`](crate::engine::InferenceEngine::default_max_tokens)).
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// OpenAI's newer name for `max_tokens`; wins when both are set.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    /// How many replies to generate.
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Stop sequences — accepts either a single string or an array of strings.
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// A seed makes the draws repeatable (OpenAI sends a signed integer).
    #[serde(default)]
    pub seed: Option<i64>,
    /// OpenAI's `logit_bias`: token id (as a string) to a bias in
    /// `[-100, 100]` added to that token's logit before sampling. -100 all
    /// but bans a token, 100 all but forces it.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// OpenAI-compatible tool definitions. When present, the chat template
    /// injects them into the system prompt so the model is aware it may emit
    /// tool calls, and its calls come back as `tool_calls`.
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoiceParam>,
    /// With `false`, at most one tool call per reply.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Deprecated spelling of `tools` (one function each).
    #[serde(default)]
    pub functions: Option<Vec<pmetal_data::chat_templates::FunctionDefinition>>,
    /// Deprecated spelling of `tool_choice`: `"none"`, `"auto"` or
    /// `{"name": ...}`.
    #[serde(default)]
    pub function_call: Option<Value>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// When `true`, include per-generated-token log-probabilities in the
    /// response. Defaults to `false` — enabling adds one log-softmax
    /// reduction per decode step, so the hot path stays unchanged for
    /// callers that don't care about confidences.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top alternative logprobs to return per token when
    /// `logprobs = true`, up to 20.
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// Per-request processor overrides (`fps`, pixel budgets, ...), as some
    /// servers take them for vision models. Read only to be refused: media
    /// are preprocessed with the checkpoint's own processor settings, and an
    /// override silently ignored would change what the model sees.
    #[serde(default)]
    pub mm_processor_kwargs: Option<Value>,
    /// OpenAI `reasoning_effort`, passed to the chat template as the
    /// `reasoning_effort` kwarg (Qwen3.8: xhigh | medium | low; gpt-oss:
    /// low | medium | high). A value in `chat_template_kwargs` wins.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Keyword arguments for the chat template, as `apply_chat_template`
    /// takes them: `{"enable_thinking": false}`, `{"preserve_thinking":
    /// false}`, or anything else the model's template reads.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Every field above this list does not name. The ones OpenAI documents
    /// that change what is generated and this server does not implement
    /// (`audio`, `prediction`, ...) are refused by name; the ones that only
    /// label the request (`user`, `metadata`, `store`, ...) are accepted.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl ChatCompletionRequest {
    /// The chat template kwargs this request asks for: its
    /// `chat_template_kwargs`, plus the top-level `reasoning_effort` when
    /// those do not set one.
    pub fn template_kwargs(&self) -> ChatTemplateKwargs {
        let mut kwargs = self.chat_template_kwargs.clone().unwrap_or_default();
        if let Some(effort) = &self.reasoning_effort {
            kwargs.set_default(ChatTemplateKwargs::REASONING_EFFORT, effort.as_str());
        }
        kwargs
    }
}

/// Per-token logprob entry as it appears on the wire under
/// `ChatChoice.logprobs.content`. Matches OpenAI's shape exactly:
/// `{token, logprob, bytes, top_logprobs: [{token, logprob, bytes}]}`.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprobContent {
    /// The decoded token string.
    pub token: String,
    /// Natural-log probability of this token under the model's distribution.
    pub logprob: f32,
    /// The token's UTF-8 bytes, so a client can rebuild the byte stream
    /// when `token` holds a replacement character for half a character.
    pub bytes: Vec<u8>,
    /// Alternative tokens considered at this position, sorted descending
    /// by logprob. Excludes the chosen token itself.
    pub top_logprobs: Vec<TopLogprob>,
}

/// A single alternative in `TokenLogprobContent::top_logprobs`.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

/// Wrapper object attached to `ChatChoice.logprobs` when requested.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<TokenLogprobContent>,
    /// Refusals are not generated by this server.
    pub refusal: Option<Vec<TokenLogprobContent>>,
}

impl ChatLogprobs {
    pub fn new(content: Vec<TokenLogprobContent>) -> Self {
        Self {
            content,
            refusal: None,
        }
    }
}

/// A prompt of the legacy completions API: text, token ids, or a batch of
/// either.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Tokens(Vec<u32>),
    Texts(Vec<String>),
    TokenLists(Vec<Vec<u32>>),
}

/// Text completion request (POST /v1/completions).
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    /// Output budget; absent means the model's default (see
    /// [`InferenceEngine::default_max_tokens`](crate::engine::InferenceEngine::default_max_tokens)).
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Stop sequences — accepts either a single string or an array of strings.
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    /// OpenAI's `logit_bias`: token id (as a string) to a bias in
    /// `[-100, 100]` added to that token's logit before sampling. -100 all
    /// but bans a token, 100 all but forces it.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// OpenAI `/v1/completions` uses a single numeric field for logprobs:
    /// the number of top alternative tokens to include, per token (up to 5).
    /// `0` means only the chosen token's logprob. `None` disables logprob
    /// collection.
    ///
    /// Note this is distinct from `/v1/chat/completions` which uses a
    /// boolean `logprobs` + integer `top_logprobs` pair.
    #[serde(default)]
    pub logprobs: Option<u32>,
    /// Return the prompt in front of each completion.
    #[serde(default)]
    pub echo: Option<bool>,
    /// Generate this many candidates and return the best (OpenAI's
    /// `best_of`); only 1 (or `n`) is accepted, as picking the best needs
    /// the sequence scores this server does not rank by.
    #[serde(default)]
    pub best_of: Option<usize>,
    /// Text after the completion (insert mode): refused by name.
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

/// OpenAI `/v1/completions` logprobs object — four parallel arrays, one
/// entry per generated token, plus a `text_offset` so clients can align
/// logprobs against the substring positions in the returned `text`.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionLogprobs {
    /// Decoded token strings, in generation order.
    pub tokens: Vec<String>,
    /// Per-token log-probabilities, aligned 1:1 with `tokens`.
    pub token_logprobs: Vec<f32>,
    /// Per-token alternative maps: `{token_string: logprob}` for the top-N
    /// alternatives at each position. Empty map when the caller requested
    /// `logprobs: 0` (chosen-token logprob only).
    pub top_logprobs: Vec<std::collections::HashMap<String, f32>>,
    /// Byte offset into the returned `text` where each token starts.
    pub text_offset: Vec<usize>,
}

// ────────────────────────────────────────────────────────────────────────────
// Responses
// ────────────────────────────────────────────────────────────────────────────

/// Chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A tool call as a response carries it: `arguments` is the call's JSON as a
/// string, the form OpenAI's clients parse.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WireToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: WireFunctionCall,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WireFunctionCall {
    pub name: String,
    pub arguments: String,
}

impl WireToolCall {
    pub fn new(id: String, name: String, arguments: String) -> Self {
        Self {
            id,
            kind: "function",
            function: WireFunctionCall { name, arguments },
        }
    }
}

/// The assistant message of a chat completion.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    /// The answer; `null` when the reply is tool calls alone.
    pub content: Option<String>,
    /// Refusals are not generated by this server.
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Choice in a chat completion.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
    /// `null` unless the request had `logprobs: true`.
    pub logprobs: Option<ChatLogprobs>,
}

/// Text completion response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Choice in a text completion.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
    /// `null` unless the request set `logprobs`.
    pub logprobs: Option<CompletionLogprobs>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub prompt_tokens_details: PromptTokensDetails,
    pub completion_tokens_details: CompletionTokensDetails,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PromptTokensDetails {
    /// Prompt tokens whose keys and values came from the prefix cache.
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CompletionTokensDetails {
    /// Completion tokens spent on reasoning.
    pub reasoning_tokens: usize,
}

impl Usage {
    pub fn new(
        prompt_tokens: usize,
        completion_tokens: usize,
        cached_tokens: usize,
        reasoning_tokens: usize,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: PromptTokensDetails { cached_tokens },
            completion_tokens_details: CompletionTokensDetails { reasoning_tokens },
        }
    }
}

/// Streaming SSE chunk for chat completions.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    /// Present (`null` on every chunk but the last) when the request set
    /// `stream_options.include_usage`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
}

/// Delta in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
    /// Per-delta logprobs — one entry per token that contributed to this
    /// delta's `content`. `null` unless the request set `logprobs: true`.
    pub logprobs: Option<ChatLogprobs>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The model's reasoning, streamed apart from `content` as other
    /// OpenAI-compatible servers stream it: a thinking model's deltas carry
    /// this first, then `content` with the answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// A tool call opening (its `id`, `type` and `name`) or growing (a
    /// fragment of its `arguments`), by `index`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A step of a tool call in a stream.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    pub function: FunctionDelta,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ────────────────────────────────────────────────────────────────────────────
// Embeddings (OpenAI /v1/embeddings)
// ────────────────────────────────────────────────────────────────────────────

/// OpenAI `/v1/embeddings` accepts `input` as a string, an array of
/// strings, an array of token ids, or an array of token-id arrays — the
/// response shape is always a list, one entry per input item in order.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatch(Vec<Vec<u32>>),
}

/// One embedding input: text to tokenize, or token ids as given.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingItem {
    Text(String),
    Tokens(Vec<u32>),
}

impl EmbeddingsInput {
    /// Flatten to the list of inputs regardless of wire shape.
    pub fn into_batch(self) -> Vec<EmbeddingItem> {
        match self {
            Self::Single(s) => vec![EmbeddingItem::Text(s)],
            Self::Batch(v) => v.into_iter().map(EmbeddingItem::Text).collect(),
            Self::Tokens(t) => vec![EmbeddingItem::Tokens(t)],
            Self::TokenBatch(b) => b.into_iter().map(EmbeddingItem::Tokens).collect(),
        }
    }
}

/// `POST /v1/embeddings` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingsInput,
    /// Optional pooling strategy. When absent, the model's own: the one its
    /// sentence-embedding configuration names, else `mean`. Accepted values:
    /// `"mean"`, `"cls"`, `"max"`, `"last_token"`, `"weighted_mean"`.
    #[serde(default)]
    pub pooling: Option<String>,
    /// `"float"` (the default) or `"base64"`: little-endian f32s, base64.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Keep this many leading dimensions of the vector, renormalized
    /// (Matryoshka truncation), for a model trained for it.
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub user: Option<String>,
}

/// A vector as the response carries it.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

/// Single embedding entry in the response list.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingData {
    pub index: usize,
    pub object: &'static str,
    pub embedding: EmbeddingVector,
}

/// Token counts for an embeddings response. No completion tokens — embeddings
/// don't generate.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

/// Parse the wire `pooling` field into a `PoolingMode`. Unknown modes yield
/// `None` so the handler can 400 rather than silently applying a fallback;
/// an absent field yields `Some(None)`.
pub fn parse_pooling_mode(s: Option<&str>) -> Option<Option<pmetal_models::pooling::PoolingMode>> {
    use pmetal_models::pooling::PoolingMode;
    let Some(s) = s else {
        return Some(None);
    };
    match s.to_ascii_lowercase().as_str() {
        "mean" => Some(Some(PoolingMode::Mean)),
        "cls" => Some(Some(PoolingMode::Cls)),
        "max" => Some(Some(PoolingMode::Max)),
        "last_token" | "last" => Some(Some(PoolingMode::LastToken)),
        "weighted_mean" | "weighted" => Some(Some(PoolingMode::WeightedMean)),
        _ => None,
    }
}

/// Model info for /v1/models.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Model list response.
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_input_forms_deserialize() {
        let batch = |json: &str| -> Vec<EmbeddingItem> {
            serde_json::from_str::<EmbeddingsRequest>(&format!(r#"{{"model":"m","input":{json}}}"#))
                .unwrap()
                .input
                .into_batch()
        };
        assert_eq!(
            batch(r#""hello""#),
            vec![EmbeddingItem::Text("hello".into())]
        );
        assert_eq!(batch(r#"["a","b","c"]"#).len(), 3);
        assert_eq!(batch("[1,2,3]"), vec![EmbeddingItem::Tokens(vec![1, 2, 3])]);
        assert_eq!(
            batch("[[1],[2,3]]"),
            vec![
                EmbeddingItem::Tokens(vec![1]),
                EmbeddingItem::Tokens(vec![2, 3])
            ]
        );
    }

    #[test]
    fn pooling_mode_parse_accepts_common_spellings() {
        use pmetal_models::pooling::PoolingMode;
        assert_eq!(parse_pooling_mode(None), Some(None));
        assert_eq!(
            parse_pooling_mode(Some("mean")),
            Some(Some(PoolingMode::Mean))
        );
        assert_eq!(
            parse_pooling_mode(Some("CLS")),
            Some(Some(PoolingMode::Cls))
        );
        assert_eq!(
            parse_pooling_mode(Some("last")),
            Some(Some(PoolingMode::LastToken))
        );
        assert_eq!(
            parse_pooling_mode(Some("last_token")),
            Some(Some(PoolingMode::LastToken))
        );
        assert_eq!(
            parse_pooling_mode(Some("weighted")),
            Some(Some(PoolingMode::WeightedMean))
        );
        assert_eq!(parse_pooling_mode(Some("unknown")), None);
    }

    #[test]
    fn chat_request_template_kwargs() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[
                {"role":"assistant","content":"a","reasoning_content":"r"}],
                "reasoning_effort":"low",
                "chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":false}}"#,
        )
        .unwrap();
        let kwargs = req.template_kwargs();
        assert_eq!(kwargs.enable_thinking(), Some(false));
        assert_eq!(kwargs.reasoning_effort(), Some("low"));
        assert_eq!(
            kwargs.get("preserve_thinking"),
            Some(&serde_json::Value::Bool(false))
        );
        assert_eq!(req.messages[0].reasoning_content.as_deref(), Some("r"));

        // A reasoning_effort inside chat_template_kwargs beats the top-level one.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[],"reasoning_effort":"low",
                "chat_template_kwargs":{"reasoning_effort":"medium"}}"#,
        )
        .unwrap();
        assert_eq!(req.template_kwargs().reasoning_effort(), Some("medium"));

        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[]}"#).unwrap();
        assert!(req.template_kwargs().is_empty());
    }

    #[test]
    fn chat_request_parses_the_documented_shapes() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[],"logprobs":true,"top_logprobs":5,"n":2,
                "stop":"x","seed":-3,"stream":true,"stream_options":{"include_usage":true},
                "tool_choice":{"type":"function","function":{"name":"f"}},
                "response_format":{"type":"json_schema","json_schema":{"name":"s","schema":{},"strict":true}},
                "audio":{"voice":"x"}}"#,
        )
        .unwrap();
        assert_eq!(req.logprobs, Some(true));
        assert_eq!(req.top_logprobs, Some(5));
        assert_eq!(req.n, Some(2));
        assert_eq!(req.stop, Some(vec!["x".to_string()]));
        assert_eq!(req.seed, Some(-3));
        assert_eq!(req.stream_options.unwrap().include_usage, Some(true));
        assert!(matches!(req.tool_choice, Some(ToolChoiceParam::Object(_))));
        assert!(matches!(
            req.response_format,
            Some(ResponseFormat::JsonSchema { .. })
        ));
        assert!(req.other.contains_key("audio"));
    }

    #[test]
    fn a_tool_message_keeps_the_call_it_answers() {
        let msg: ChatMessage = serde_json::from_str(
            r#"{"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"72F"}]}"#,
        )
        .unwrap();
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(msg.content, "72F");
    }

    #[test]
    fn completion_prompts_take_all_four_forms() {
        let prompt = |json: &str| -> CompletionPrompt {
            serde_json::from_str::<CompletionRequest>(&format!(
                r#"{{"model":"m","prompt":{json}}}"#
            ))
            .unwrap()
            .prompt
        };
        assert!(matches!(prompt(r#""hi""#), CompletionPrompt::Text(_)));
        assert!(matches!(prompt("[1,2]"), CompletionPrompt::Tokens(_)));
        assert!(matches!(prompt(r#"["a","b"]"#), CompletionPrompt::Texts(_)));
        assert!(matches!(
            prompt("[[1],[2]]"),
            CompletionPrompt::TokenLists(_)
        ));
    }

    #[test]
    fn response_shapes_match_openai() {
        let message = ResponseMessage {
            role: "assistant",
            content: None,
            refusal: None,
            tool_calls: Some(vec![WireToolCall::new(
                "call_1".into(),
                "f".into(),
                r#"{"a":1}"#.into(),
            )]),
            reasoning_content: None,
        };
        let json = serde_json::to_value(&message).unwrap();
        assert!(json["content"].is_null());
        assert!(json["refusal"].is_null());
        assert_eq!(json["tool_calls"][0]["type"], "function");
        // `arguments` is a string of JSON, not an object.
        assert_eq!(json["tool_calls"][0]["function"]["arguments"], r#"{"a":1}"#);
        let usage = serde_json::to_value(Usage::new(3, 4, 1, 2)).unwrap();
        assert_eq!(usage["total_tokens"], 7);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 1);
        assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 2);
    }
}
