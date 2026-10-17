//! The API over an engine that batches continuously: the replies are the ones
//! the single-request path gives, stop sequences and logprobs included, and a
//! request with a grammar (which the pump does not run) is still constrained.

mod support;

use axum::http::StatusCode;
use pmetal_serve::{BatcherConfig, ServeConfig};
use serde_json::{Value, json};
use support::*;

fn chat(extra: Value) -> Value {
    let mut bias = serde_json::Map::new();
    for byte in b'a'..=b'z' {
        bias.insert(byte.to_string(), json!(100));
    }
    let mut body = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 12,
        "temperature": 0,
        "logit_bias": bias,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

fn batched() -> (axum::Router, tempfile::TempDir) {
    let tiny = tiny(512);
    tiny.engine
        .enable_continuous_batching_auto(BatcherConfig {
            max_slots: 2,
            ..BatcherConfig::default()
        })
        .unwrap();
    tiny.router(&ServeConfig::default())
}

fn text(reply: &Value, i: usize) -> String {
    reply["choices"][i]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn batched_replies_match_the_single_request_path() {
    let (single, _a) = app(512);
    let (pumped, _b) = batched();
    for extra in [
        json!({}),
        json!({"n": 3}),
        json!({"logprobs": true, "top_logprobs": 2}),
        json!({"stop": ["e"], "max_tokens": 40}),
    ] {
        let want = post(&single, "/v1/chat/completions", &chat(extra.clone())).await;
        let got = post(&pumped, "/v1/chat/completions", &chat(extra.clone())).await;
        assert_eq!(got.status, StatusCode::OK, "{extra}: {}", got.text());
        let (want, got) = (want.json(), got.json());
        let n = want["choices"].as_array().unwrap().len();
        for i in 0..n {
            assert_eq!(text(&got, i), text(&want, i), "{extra}: choice {i}");
            assert_eq!(
                got["choices"][i]["finish_reason"], want["choices"][i]["finish_reason"],
                "{extra}"
            );
        }
        if extra.get("logprobs").is_some() {
            let lp = |r: &Value| -> Vec<f64> {
                r["choices"][0]["logprobs"]["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|e| e["logprob"].as_f64().unwrap())
                    .collect()
            };
            let (a, b) = (lp(&want), lp(&got));
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-3));
        }
    }
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_sequence_cuts_a_batched_stream_too() {
    let (pumped, _dir) = batched();
    let bias = json!({"259": 100, "120": 100});
    let body = |extra: Value| {
        let mut b = chat(json!({"logit_bias": bias, "max_tokens": 10}));
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        b
    };
    let full = post(&pumped, "/v1/chat/completions", &body(json!({})))
        .await
        .json();
    let full = text(&full, 0);
    let at = full.find("he").expect("the reply has the stop in it");
    let reply = post(
        &pumped,
        "/v1/chat/completions",
        &body(json!({"stop": "he", "stream": true})),
    )
    .await;
    let streamed: String = reply
        .sse_json()
        .iter()
        .filter_map(|f| {
            f["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    assert_eq!(streamed, &full[..at]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_grammar_still_binds_when_the_engine_batches() {
    let (pumped, _dir) = batched();
    let schema = json!({"type": "object", "properties": {"ok": {"enum": [true, false]}},
                        "required": ["ok"], "additionalProperties": false});
    let reply = post(
        &pumped,
        "/v1/chat/completions",
        &json!({"model": MODEL_ID, "max_tokens": 60, "temperature": 1.0, "seed": 2,
                "messages": [{"role": "user", "content": "json"}],
                "response_format": {"type": "json_schema",
                    "json_schema": {"name": "r", "schema": schema}}}),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let body = reply.json();
    let value: Value = serde_json::from_str(&text(&body, 0)).unwrap();
    assert!(value["ok"].is_boolean());
    assert_no_unobserved_error();
}
