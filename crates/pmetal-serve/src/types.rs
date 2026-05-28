//! OpenAI-compatible request/response types.

use pmetal_data::chat_templates::{FunctionCall, ToolCall, ToolDefinition};
use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize a field that may be either a single string or an array of strings.
///
/// The OpenAI API accepts `"stop": "token"` (string) or `"stop": ["a", "b"]`
/// (array) — both map to `Option<Vec<String>>`.
fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct StringOrVec;

    impl<'de> Visitor<'de> for StringOrVec {
        type Value = Option<Vec<String>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or array of strings or null")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(Some(vec![v.to_owned()]))
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(Some(vec![v]))
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut vec = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                vec.push(s);
            }
            Ok(Some(vec))
        }
    }

    deserializer.deserialize_any(StringOrVec)
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Tool calls made by the assistant. `None` for user/system/tool messages.
    ///
    /// When the model emits a structured tool call (best-effort JSON parse of the
    /// generated text), this is populated and `content` is left empty. OpenAI
    /// clients that support tools read `tool_calls` and ignore `content`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Chat completion request (POST /v1/chat/completions).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
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
    pub seed: Option<u64>,
    /// OpenAI-compatible tool definitions. When present, the chat template
    /// injects them into the system prompt so the model is aware it may emit
    /// tool calls. Best-effort JSON parsing of the model's output after
    /// generation decides whether `tool_calls` or `content` is returned.
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    /// When `true`, include per-generated-token log-probabilities in the
    /// response. Defaults to `false` — enabling adds one log-softmax
    /// reduction per decode step, so the hot path stays unchanged for
    /// callers that don't care about confidences.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top alternative logprobs to return per token when
    /// `logprobs = true`. OpenAI caps this at 20 in their docs; we accept
    /// any `u8` value and clamp on the generation side. `None` / `Some(0)`
    /// means chosen-token logprob only.
    #[serde(default)]
    pub top_logprobs: Option<u8>,
}

/// Per-token logprob entry as it appears on the wire under
/// `ChatChoice.logprobs.content`. Matches OpenAI's shape exactly:
/// `{token, logprob, bytes?, top_logprobs: [{token, logprob, bytes?}]}`.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprobContent {
    /// The decoded token string.
    pub token: String,
    /// Natural-log probability of this token under the model's distribution.
    pub logprob: f32,
    /// UTF-8 byte representation. Present so clients can reconstruct the
    /// byte stream even when `token` contains a replacement character.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    /// Alternative tokens considered at this position, sorted descending
    /// by logprob. Excludes the chosen token itself.
    pub top_logprobs: Vec<TopLogprob>,
}

/// A single alternative in `TokenLogprobContent::top_logprobs`.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Wrapper object attached to `ChatChoice.logprobs` when requested.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<TokenLogprobContent>,
}

fn default_max_tokens() -> usize {
    256
}

/// Text completion request (POST /v1/completions).
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
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
    pub seed: Option<u64>,
    /// OpenAI `/v1/completions` uses a single numeric field for logprobs:
    /// the number of top alternative tokens to include, per token. `0` means
    /// only the chosen token's logprob. `None` disables logprob collection.
    ///
    /// Note this is distinct from `/v1/chat/completions` which uses a
    /// boolean `logprobs` + integer `top_logprobs` pair.
    #[serde(default)]
    pub logprobs: Option<u8>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_parse_bare_object() {
        let text = r#"{"name": "get_weather", "arguments": {"city": "SF"}}"#;
        let calls = try_parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["city"], "SF");
    }

    #[test]
    fn tool_parse_wrapped_object() {
        let text = r#"{"tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]}"#;
        let calls = try_parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "f");
    }

    #[test]
    fn tool_parse_qwen_xml_call() {
        let text = r#"<tool_call>
{"name": "web_search", "arguments": {"query": "qwen mtp", "limit": 5}}
</tool_call>"#;
        let calls = try_parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "web_search");
        assert_eq!(calls[0].function.arguments["query"], "qwen mtp");
        assert_eq!(calls[0].function.arguments["limit"], 5);
    }

    #[test]
    fn tool_parse_gemma4_custom_call() {
        let text = r#"<|tool_call>call:web_search{query:<|"|>latest Gemma 4 MTP<|"|>,limit:5}<tool_call|>"#;
        let calls = try_parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "web_search");
        assert_eq!(calls[0].function.arguments["query"], "latest Gemma 4 MTP");
        assert_eq!(calls[0].function.arguments["limit"], 5);
    }

    #[test]
    fn tool_parse_gemma4_multiple_calls_and_nested_args() {
        let text = concat!(
            r#"<|tool_call>call:web.search{query:<|"|>qwen mtp<|"|>}<tool_call|>"#,
            r#"<|tool_call>call:store_result{meta:{source:<|"|>web<|"|>,ok:true},scores:[1,2.5]}<turn|>"#
        );
        let calls = try_parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "web.search");
        assert_eq!(calls[1].function.name, "store_result");
        assert_eq!(calls[1].function.arguments["meta"]["source"], "web");
        assert_eq!(calls[1].function.arguments["meta"]["ok"], true);
        assert_eq!(calls[1].function.arguments["scores"][1], 2.5);
    }

    #[test]
    fn tool_parse_rejects_plain_text() {
        assert!(try_parse_tool_calls("Hello, world!").is_none());
        assert!(try_parse_tool_calls("{invalid json").is_none());
        assert!(try_parse_tool_calls(r#"{"greeting": "hi"}"#).is_none());
    }

    #[test]
    fn embeddings_input_single_string_deserializes() {
        let req: EmbeddingsRequest =
            serde_json::from_str(r#"{"model":"m","input":"hello"}"#).unwrap();
        assert_eq!(req.input.into_batch(), vec!["hello".to_string()]);
    }

    #[test]
    fn embeddings_input_array_deserializes() {
        let req: EmbeddingsRequest =
            serde_json::from_str(r#"{"model":"m","input":["a","b","c"]}"#).unwrap();
        assert_eq!(req.input.into_batch().len(), 3);
    }

    #[test]
    fn pooling_mode_parse_accepts_common_spellings() {
        use pmetal_models::pooling::PoolingMode;
        assert_eq!(parse_pooling_mode(None), Some(PoolingMode::Mean));
        assert_eq!(parse_pooling_mode(Some("mean")), Some(PoolingMode::Mean));
        assert_eq!(parse_pooling_mode(Some("CLS")), Some(PoolingMode::Cls));
        assert_eq!(
            parse_pooling_mode(Some("last")),
            Some(PoolingMode::LastToken)
        );
        assert_eq!(
            parse_pooling_mode(Some("last_token")),
            Some(PoolingMode::LastToken)
        );
        assert_eq!(
            parse_pooling_mode(Some("weighted")),
            Some(PoolingMode::WeightedMean)
        );
        assert_eq!(parse_pooling_mode(Some("unknown")), None);
    }

    #[test]
    fn chat_request_parses_logprobs_fields() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[],"logprobs":true,"top_logprobs":5}"#)
                .unwrap();
        assert_eq!(req.logprobs, Some(true));
        assert_eq!(req.top_logprobs, Some(5));
    }

    #[test]
    fn chat_logprobs_absent_serializes_without_field() {
        // When logprobs is None on ChatChoice, the JSON must not include
        // "logprobs": null — legacy clients would reject it.
        let choice = ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: "hi".into(),
                tool_calls: None,
            },
            finish_reason: Some("stop".into()),
            logprobs: None,
        };
        let json = serde_json::to_string(&choice).unwrap();
        assert!(
            !json.contains("logprobs"),
            "logprobs field should be omitted when None, got: {json}"
        );
    }

    #[test]
    fn chat_logprobs_serializes_when_present() {
        let choice = ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: "hi".into(),
                tool_calls: None,
            },
            finish_reason: Some("stop".into()),
            logprobs: Some(ChatLogprobs {
                content: vec![TokenLogprobContent {
                    token: "hi".into(),
                    logprob: -0.5,
                    bytes: None,
                    top_logprobs: vec![TopLogprob {
                        token: "hello".into(),
                        logprob: -1.2,
                        bytes: None,
                    }],
                }],
            }),
        };
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains(r#""logprobs""#));
        assert!(json.contains(r#""token":"hi""#));
        assert!(json.contains(r#""logprob":-0.5"#));
    }

    #[test]
    fn completion_request_parses_numeric_logprobs() {
        // /v1/completions uses a numeric logprobs field, not a bool.
        let req: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"p","logprobs":3}"#).unwrap();
        assert_eq!(req.logprobs, Some(3));
    }

    #[test]
    fn completion_request_logprobs_optional() {
        let req: CompletionRequest = serde_json::from_str(r#"{"model":"m","prompt":"p"}"#).unwrap();
        assert!(req.logprobs.is_none());
    }

    #[test]
    fn completion_logprobs_serializes_parallel_arrays() {
        use std::collections::HashMap;
        let mut top0 = HashMap::new();
        top0.insert("Hi".into(), -1.2_f32);
        let lp = CompletionLogprobs {
            tokens: vec!["Hello".into(), " world".into()],
            token_logprobs: vec![-0.5, -0.3],
            top_logprobs: vec![top0, HashMap::new()],
            text_offset: vec![0, 5],
        };
        let json = serde_json::to_string(&lp).unwrap();
        assert!(json.contains(r#""tokens":["Hello"," world"]"#));
        assert!(json.contains(r#""token_logprobs":[-0.5,-0.3]"#));
        assert!(json.contains(r#""text_offset":[0,5]"#));
    }

    #[test]
    fn tool_parse_allows_whitespace() {
        let text = "  \n\t{\n  \"name\": \"f\"\n}\n ";
        let calls = try_parse_tool_calls(text).expect("should parse");
        assert_eq!(calls[0].function.name, "f");
    }
}

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

/// Choice in a chat completion.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
    /// Populated only when the request had `logprobs: true`. The field is
    /// omitted from the response body in the default case so the wire shape
    /// stays byte-compatible with legacy clients that don't expect it.
    #[serde(skip_serializing_if = "Option::is_none")]
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
    /// Populated only when the request set `logprobs`. Same omit-when-None
    /// rule as `ChatChoice.logprobs` — legacy clients stay byte-compatible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming SSE chunk for chat completions.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// Delta in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls attached to the closing chunk when the model's output parses
    /// as a function call. Mid-stream token deltas leave this `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Per-delta logprobs payload — one `TokenLogprobContent` per token that
    /// contributed to this delta's `content`. Populated only when the request
    /// set `logprobs: true`. The default omit-when-None rule keeps the wire
    /// shape unchanged for callers that don't opt in.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

/// Best-effort parser that converts a raw assistant response into a tool call
/// when the text matches the OpenAI tool-call shape.
///
/// Accepted shapes:
/// 1. Bare object: `{"name": "...", "arguments": {...}}`
/// 2. Wrapped object: `{"tool_calls": [{"function": {"name": ..., "arguments": ...}}, ...]}`
/// 3. Qwen XML envelope: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
/// 4. Gemma 4 native envelope: `<|tool_call>call:name{arg:<|"|>value<|"|>}<tool_call|>`
///
/// Returns `None` when the text does not cleanly parse as a tool call. Callers
/// fall back to returning the raw text as `content`.
pub fn try_parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let trimmed = text.trim();
    if let Some(calls) = try_parse_gemma4_tool_calls(trimmed) {
        return Some(calls);
    }
    if let Some(calls) = try_parse_qwen_xml_tool_calls(trimmed) {
        return Some(calls);
    }
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;

    if let Some(wrapper) = value.get("tool_calls").and_then(|v| v.as_array()) {
        let calls: Vec<ToolCall> = wrapper
            .iter()
            .filter_map(|tc| {
                let func = tc.get("function")?;
                Some(ToolCall {
                    id: tc.get("id").and_then(|v| v.as_str()).map(str::to_owned),
                    tool_type: tc
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("function")
                        .to_owned(),
                    function: FunctionCall {
                        name: func.get("name")?.as_str()?.to_owned(),
                        arguments: func
                            .get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                    },
                })
            })
            .collect();
        return (!calls.is_empty()).then_some(calls);
    }

    let name = value.get("name").and_then(|v| v.as_str())?;
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    Some(vec![ToolCall {
        id: None,
        tool_type: "function".to_owned(),
        function: FunctionCall {
            name: name.to_owned(),
            arguments,
        },
    }])
}

fn try_parse_qwen_xml_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    const START: &str = "<tool_call>";
    const END: &str = "</tool_call>";

    let mut calls = Vec::new();
    let mut offset = 0usize;
    while let Some(start_rel) = text[offset..].find(START) {
        let body_start = offset + start_rel + START.len();
        let rest = &text[body_start..];
        let end_rel = rest.find(END)?;
        let body = rest[..end_rel].trim();
        let value: serde_json::Value = serde_json::from_str(body).ok()?;
        let name = value.get("name").and_then(|v| v.as_str())?;
        let arguments = value
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        calls.push(ToolCall {
            id: None,
            tool_type: "function".to_owned(),
            function: FunctionCall {
                name: name.to_owned(),
                arguments,
            },
        });
        offset = body_start + end_rel + END.len();
    }

    (!calls.is_empty()).then_some(calls)
}

fn try_parse_gemma4_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    const START: &str = "<|tool_call>call:";
    const END: &str = "<tool_call|>";
    const TURN_END: &str = "<turn|>";

    let mut calls = Vec::new();
    let mut offset = 0usize;
    while let Some(start_rel) = text[offset..].find(START) {
        let start = offset + start_rel + START.len();
        let after_start = &text[start..];
        let name_end_rel = after_start.find('{')?;
        let name = after_start[..name_end_rel].trim();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
        {
            offset = start + name_end_rel + 1;
            continue;
        }

        let args_start = start + name_end_rel + 1;
        let args_end_rel = find_gemma4_matching_brace(&text[args_start..])?;
        let args_end = args_start + args_end_rel;
        let after_args = &text[args_end + 1..];
        let consumed_end = if after_args.starts_with(END) {
            args_end + 1 + END.len()
        } else if after_args.starts_with(TURN_END) {
            args_end + 1 + TURN_END.len()
        } else {
            args_end + 1
        };

        calls.push(ToolCall {
            id: None,
            tool_type: "function".to_owned(),
            function: FunctionCall {
                name: name.to_owned(),
                arguments: parse_gemma4_arguments(&text[args_start..args_end]),
            },
        });
        offset = consumed_end;
    }

    (!calls.is_empty()).then_some(calls)
}

fn find_gemma4_matching_brace(input: &str) -> Option<usize> {
    const STRING_DELIM: &str = r#"<|"|>"#;
    let mut idx = 0usize;
    let mut depth = 0i32;
    let mut in_string = false;
    while idx < input.len() {
        let rest = &input[idx..];
        if rest.starts_with(STRING_DELIM) {
            in_string = !in_string;
            idx += STRING_DELIM.len();
            continue;
        }
        let byte = input.as_bytes()[idx];
        if !in_string {
            match byte {
                b'{' => depth += 1,
                b'}' if depth == 0 => return Some(idx),
                b'}' => depth -= 1,
                _ => {}
            }
        }
        idx += 1;
    }
    None
}

fn parse_gemma4_arguments(input: &str) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for part in split_gemma4_top_level(input, b',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(colon) = find_gemma4_top_level(trimmed, b':') {
            let key = trimmed[..colon].trim();
            if key.is_empty() {
                continue;
            }
            object.insert(
                key.to_owned(),
                parse_gemma4_value(trimmed[colon + 1..].trim()),
            );
        }
    }
    serde_json::Value::Object(object)
}

fn parse_gemma4_value(input: &str) -> serde_json::Value {
    const STRING_DELIM: &str = r#"<|"|>"#;
    let trimmed = input.trim();
    if let Some(inner) = trimmed
        .strip_prefix(STRING_DELIM)
        .and_then(|s| s.strip_suffix(STRING_DELIM))
    {
        return serde_json::Value::String(inner.to_owned());
    }
    if let Some(inner) = trimmed.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        return parse_gemma4_arguments(inner);
    }
    if let Some(inner) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let values = split_gemma4_top_level(inner, b',')
            .into_iter()
            .filter(|part| !part.trim().is_empty())
            .map(parse_gemma4_value)
            .collect();
        return serde_json::Value::Array(values);
    }
    match trimmed {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        "null" => serde_json::Value::Null,
        _ => {
            if let Ok(value) = trimmed.parse::<i64>() {
                serde_json::Value::Number(value.into())
            } else if let Ok(value) = trimmed.parse::<f64>() {
                serde_json::Number::from_f64(value)
                    .map(serde_json::Value::Number)
                    .unwrap_or_else(|| serde_json::Value::String(trimmed.to_owned()))
            } else {
                serde_json::Value::String(trimmed.to_owned())
            }
        }
    }
}

fn split_gemma4_top_level(input: &str, delimiter: u8) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut idx = 0usize;
    while idx < input.len() {
        let rest = &input[idx..];
        if rest.starts_with(r#"<|"|>"#) {
            in_string = !in_string;
            idx += r#"<|"|>"#.len();
            continue;
        }
        let byte = input.as_bytes()[idx];
        if !in_string {
            match byte {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                b if b == delimiter && depth == 0 => {
                    parts.push(&input[start..idx]);
                    start = idx + 1;
                }
                _ => {}
            }
        }
        idx += 1;
    }
    parts.push(&input[start..]);
    parts
}

fn find_gemma4_top_level(input: &str, needle: u8) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut idx = 0usize;
    while idx < input.len() {
        let rest = &input[idx..];
        if rest.starts_with(r#"<|"|>"#) {
            in_string = !in_string;
            idx += r#"<|"|>"#.len();
            continue;
        }
        let byte = input.as_bytes()[idx];
        if !in_string {
            match byte {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                b if b == needle && depth == 0 => return Some(idx),
                _ => {}
            }
        }
        idx += 1;
    }
    None
}

// ────────────────────────────────────────────────────────────────────────────
// Embeddings (OpenAI /v1/embeddings)
// ────────────────────────────────────────────────────────────────────────────

/// OpenAI `/v1/embeddings` accepts `input` as either a single string or an
/// array of strings — the response shape is always a list, one entry per
/// input item in order.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingsInput {
    /// Flatten to a single `Vec<String>` regardless of wire shape.
    pub fn into_batch(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Batch(v) => v,
        }
    }
}

/// `POST /v1/embeddings` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingsInput,
    /// Optional pooling strategy. When absent, defaults to `mean` pooling —
    /// the most common choice for sentence embeddings. Accepted values:
    /// `"mean"`, `"cls"`, `"max"`, `"last_token"`, `"weighted_mean"`.
    #[serde(default)]
    pub pooling: Option<String>,
    /// OpenAI compatibility fields — currently ignored but accepted so
    /// drop-in clients don't 400. `encoding_format` other than `"float"`
    /// would require output rewriting that's out of scope for Phase S3.
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
}

/// Single embedding entry in the response list.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingData {
    pub index: usize,
    pub object: &'static str,
    pub embedding: Vec<f32>,
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

/// Parse the wire `pooling` field into a [`PoolingMode`] with sensible
/// defaults. Unknown modes yield `None` so the handler can 400 rather than
/// silently applying a fallback.
pub fn parse_pooling_mode(s: Option<&str>) -> Option<pmetal_models::pooling::PoolingMode> {
    use pmetal_models::pooling::PoolingMode;
    match s.unwrap_or("mean").to_ascii_lowercase().as_str() {
        "mean" => Some(PoolingMode::Mean),
        "cls" => Some(PoolingMode::Cls),
        "max" => Some(PoolingMode::Max),
        "last_token" | "last" => Some(PoolingMode::LastToken),
        "weighted_mean" | "weighted" => Some(PoolingMode::WeightedMean),
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
