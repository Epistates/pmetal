//! A tiny seeded model behind the real router, driven in-process.
//!
//! The model is a 2-layer Llama with random (seeded) weights, so what it says
//! is noise, but the same noise every run. Tests that need exact text force it
//! (a grammar, `logit_bias`) or exercise the layers around generation on
//! synthetic token streams. The tokenizer is a byte-level BPE with the ChatML
//! markers, thinking and tool-call tags as special tokens.

#![allow(dead_code)]

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use pmetal_models::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_serve::{InferenceEngine, ServeConfig};
use serde_json::Value;
use std::path::Path;
use tower::ServiceExt as _;

pub const MODEL_ID: &str = "tiny-llama";

/// The special tokens of the test tokenizer, in id order after the 256 bytes
/// and the merges.
pub const SPECIALS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
];

pub const MERGES: &[&str] = &["h i", "\u{120} t", "t h", "th e"];

/// ChatML, with tools listed in the system turn and tool calls rendered back
/// the way Hermes-style templates do.
const CHAT_TEMPLATE: &str = "{%- if tools %}<|im_start|>system\nYou may call these tools:\n\
{%- for tool in tools %}\n{{ tool | tojson }}{%- endfor %}<|im_end|>\n{%- endif %}\
{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}\
{%- for call in m.tool_calls or [] %}<tool_call>\n{{ call.function | tojson }}\n</tool_call>{%- endfor %}\
<|im_end|>\n{%- endfor %}\
{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

pub fn tokenizer() -> pmetal_data::Tokenizer {
    pmetal_data::tokenizer::testing::byte_level_bpe(MERGES, SPECIALS, None)
}

/// Id of a special token.
pub fn special(name: &str) -> u32 {
    256 + MERGES.len() as u32 + SPECIALS.iter().position(|s| *s == name).unwrap() as u32
}

/// The files of the tiny model's directory, for a test that builds its own
/// engine.
pub fn write_model_dir_for_tests(dir: &Path) {
    write_model_dir(dir, 256);
}

fn write_model_dir(dir: &Path, max_positions: usize) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "model_type": "llama",
            "max_position_embeddings": max_positions,
            "vocab_size": 288,
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": CHAT_TEMPLATE,
            "eos_token": "<|im_end|>",
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        serde_json::json!({
            "eos_token_id": [special("<|im_end|>"), special("<|endoftext|>")],
            "max_new_tokens": 16,
        })
        .to_string(),
    )
    .unwrap();
}

pub fn tiny_llama() -> anyhow::Result<DynamicModel> {
    pmetal_runtime::compat::random::seed(7);
    let config = LlamaConfig {
        vocab_size: 288,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        max_position_embeddings: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        hidden_act: "silu".to_string(),
        tie_word_embeddings: true,
        ..Default::default()
    };
    Ok(DynamicModel::Llama(LlamaForCausalLM::new(config)?))
}

/// A running tiny engine and the directory its config lives in.
pub struct Tiny {
    pub engine: InferenceEngine,
    pub dir: tempfile::TempDir,
}

impl Tiny {
    /// The engine behind the server's router under `config`; keep the
    /// returned directory alive as long as the router.
    pub fn router(self, config: &ServeConfig) -> (Router, tempfile::TempDir) {
        (
            pmetal_serve::server::build_router_with(self.engine, config),
            self.dir,
        )
    }
}

/// The default server over a tiny engine with room for `max_seq_len` tokens.
pub fn app(max_seq_len: usize) -> (Router, tempfile::TempDir) {
    tiny(max_seq_len).router(&ServeConfig::default())
}

pub fn tiny(max_seq_len: usize) -> Tiny {
    tiny_with(max_seq_len, |_| {})
}

/// Like [`tiny`], with `setup` adding files to the model directory before
/// the engine reads it.
pub fn tiny_with(max_seq_len: usize, setup: impl FnOnce(&Path)) -> Tiny {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path(), max_seq_len);
    setup(dir.path());
    let engine = InferenceEngine::new_with_backend(
        tiny_llama,
        tokenizer(),
        MODEL_ID.into(),
        dir.path(),
        max_seq_len,
        false,
        1024,
    )
    .unwrap();
    Tiny { engine, dir }
}

/// A reply from the router.
pub struct Reply {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl Reply {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("not JSON ({e}): {}", String::from_utf8_lossy(&self.body)))
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// The server-sent events of the body: `(event name, data)`.
    pub fn sse(&self) -> Vec<(Option<String>, String)> {
        parse_sse(&self.text())
    }

    /// The JSON payloads of the `data:` lines, `[DONE]` excluded.
    pub fn sse_json(&self) -> Vec<Value> {
        self.sse()
            .into_iter()
            .filter(|(_, data)| data != "[DONE]")
            .map(|(_, data)| serde_json::from_str(&data).expect("an SSE data line of JSON"))
            .collect()
    }
}

pub fn parse_sse(text: &str) -> Vec<(Option<String>, String)> {
    text.split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .filter_map(|frame| {
            let mut event = None;
            let mut data = Vec::new();
            for line in frame.lines() {
                if let Some(rest) = line.strip_prefix("event:") {
                    event = Some(rest.trim().to_string());
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
                }
                // Lines starting with ':' are comments (keep-alives).
            }
            (!data.is_empty() || event.is_some()).then(|| (event, data.join("\n")))
        })
        .collect()
}

pub async fn send(router: &Router, request: Request<Body>) -> Reply {
    let response = router.clone().oneshot(request).await.unwrap();
    let (parts, body) = response.into_parts();
    let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    Reply {
        status: parts.status,
        headers: parts.headers,
        body: body.to_vec(),
    }
}

pub async fn post(router: &Router, path: &str, body: &Value) -> Reply {
    post_raw(router, path, body.to_string().into_bytes(), &[]).await
}

pub async fn post_with_headers(
    router: &Router,
    path: &str,
    body: &Value,
    headers: &[(&str, &str)],
) -> Reply {
    post_raw(router, path, body.to_string().into_bytes(), headers).await
}

pub async fn post_raw(
    router: &Router,
    path: &str,
    body: Vec<u8>,
    headers: &[(&str, &str)],
) -> Reply {
    let mut request = Request::post(path).header(header::CONTENT_TYPE, "application/json");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    send(router, request.body(Body::from(body)).unwrap()).await
}

pub async fn get(router: &Router, path: &str) -> Reply {
    send(router, Request::get(path).body(Body::empty()).unwrap()).await
}

pub async fn get_with_headers(router: &Router, path: &str, headers: &[(&str, &str)]) -> Reply {
    let mut request = Request::get(path);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    send(router, request.body(Body::empty()).unwrap()).await
}

/// Reads the engine's error channel: a runtime op that threw would otherwise
/// pass silently.
pub fn assert_no_unobserved_error() {
    pmetal_runtime::check_unobserved_error().expect("an op threw and nobody saw");
}
