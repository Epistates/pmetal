//! `/v1/messages` against Anthropic's Messages API: the response object, the
//! streaming event order, `stop_reason` and `stop_sequence`, errors, and the
//! request forms (system blocks, tool use and results).

mod support;

use axum::http::StatusCode;
use serde_json::{Value, json};
use support::*;

fn request(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "max_tokens": 10,
        "temperature": 0,
        "messages": [{"role": "user", "content": "hi"}],
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

async fn ok(router: &axum::Router, body: Value) -> Value {
    let reply = post(router, "/v1/messages", &body).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    reply.json()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_message_has_anthropic_s_shape() {
    let (router, _dir) = app(256);
    let reply = ok(&router, request(json!({}))).await;
    assert_eq!(reply["type"], "message");
    assert_eq!(reply["role"], "assistant");
    assert!(reply["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(reply["model"], MODEL_ID);
    assert_eq!(reply["content"][0]["type"], "text");
    assert_eq!(reply["stop_reason"], "max_tokens");
    assert!(reply["stop_sequence"].is_null());
    assert_eq!(reply["usage"]["output_tokens"], 10);
    assert!(reply["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(reply["usage"]["cache_creation_input_tokens"], 0);
    assert_eq!(reply["usage"]["cache_read_input_tokens"], 0);
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_sequence_is_reported_with_its_text() {
    let (router, _dir) = app(256);
    let full = ok(&router, request(json!({"max_tokens": 16}))).await;
    let text = full["content"][0]["text"].as_str().unwrap().to_string();
    // A stop sequence from the middle of the reply: two bytes at the start
    // of its text that are a whole character.
    let stop: String = text.chars().skip(3).take(1).collect();
    assert!(!stop.is_empty());
    let cut = ok(
        &router,
        request(json!({"max_tokens": 16, "stop_sequences": [stop.clone()]})),
    )
    .await;
    assert_eq!(cut["stop_reason"], "stop_sequence");
    assert_eq!(cut["stop_sequence"], stop);
    let cut_text = cut["content"][0]["text"].as_str().unwrap();
    assert_eq!(cut_text, &text[..text.find(&stop).unwrap()]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stream_follows_anthropic_s_event_order() {
    let (router, _dir) = app(256);
    let whole = ok(&router, request(json!({}))).await;
    let reply = post(&router, "/v1/messages", &request(json!({"stream": true}))).await;
    assert_eq!(reply.status, StatusCode::OK);
    let events = reply.sse();
    let names: Vec<&str> = events
        .iter()
        .map(|(name, _)| name.as_deref().expect("every frame is named"))
        .filter(|name| *name != "ping")
        .collect();
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(names.last(), Some(&"message_stop"));
    assert_eq!(names[1], "content_block_start");
    assert_eq!(names[names.len() - 2], "message_delta");
    assert_eq!(names[names.len() - 3], "content_block_stop");
    assert!(
        names[2..names.len() - 3]
            .iter()
            .all(|n| *n == "content_block_delta")
    );

    // Each frame's `type` is its event name.
    for (name, data) in &events {
        let payload: Value = serde_json::from_str(data).unwrap();
        assert_eq!(payload["type"], name.as_deref().unwrap());
    }
    let start: Value = serde_json::from_str(&events[0].1).unwrap();
    assert_eq!(start["message"]["content"], json!([]));
    assert!(start["message"]["stop_reason"].is_null());
    assert_eq!(
        start["message"]["usage"]["input_tokens"],
        whole["usage"]["input_tokens"]
    );
    let text: String = reply
        .sse_json()
        .iter()
        .filter_map(|e| e["delta"]["text"].as_str().map(str::to_string))
        .collect();
    assert_eq!(text, whole["content"][0]["text"].as_str().unwrap());
    let delta = reply
        .sse_json()
        .into_iter()
        .find(|e| e["type"] == "message_delta")
        .unwrap();
    assert_eq!(delta["delta"]["stop_reason"], "max_tokens");
    assert!(delta["delta"]["stop_sequence"].is_null());
    assert_eq!(delta["usage"]["output_tokens"], 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn system_blocks_and_tool_history_are_accepted() {
    let (router, _dir) = app(512);
    let reply = ok(
        &router,
        request(json!({
            "system": [{"type": "text", "text": "be brief"},
                       {"type": "text", "text": "be kind", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}]}
            ],
            "tools": [{"name": "get_weather", "description": "weather",
                       "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        })),
    )
    .await;
    assert_eq!(reply["type"], "message");
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_requests_are_anthropic_errors() {
    let (router, _dir) = app(256);
    let cases = [
        (request(json!({"max_tokens": 0})), "max_tokens"),
        (request(json!({"temperature": 1.5})), "temperature"),
        (
            request(json!({"thinking": {"type": "sometimes"}})),
            "thinking.type",
        ),
        (
            request(json!({"messages": [{"role": "system", "content": "x"}]})),
            "messages.0.role",
        ),
        (
            request(json!({"messages": [{"role": "user", "content": [{"type": "document"}]}]})),
            "messages.0.content.0.type",
        ),
        (
            request(json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]})),
            "tools.0.type",
        ),
        (request(json!({"mcp_servers": []})), ""),
    ];
    for (body, param) in cases {
        let reply = post(&router, "/v1/messages", &body).await;
        if param.is_empty() {
            // An empty list of connectors is no connector.
            assert_eq!(reply.status, StatusCode::OK);
            continue;
        }
        assert_eq!(
            reply.status,
            StatusCode::BAD_REQUEST,
            "{body}: {}",
            reply.text()
        );
        let error = reply.json();
        assert_eq!(error["type"], "error");
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(
            error["error"]["message"]
                .as_str()
                .is_some_and(|m| !m.is_empty()),
            "{error}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn max_tokens_is_required() {
    let (router, _dir) = app(256);
    let reply = post(
        &router,
        "/v1/messages",
        &json!({"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert_eq!(reply.json()["error"]["type"], "invalid_request_error");
    assert!(
        reply.json()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("max_tokens")
    );
}
