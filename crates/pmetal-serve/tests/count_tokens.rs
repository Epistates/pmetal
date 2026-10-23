//! `POST /v1/messages/count_tokens`: the count is the prompt `/v1/messages`
//! would run, and the request is refused where `/v1/messages` refuses it.

mod support;

use axum::http::StatusCode;
use serde_json::{Value, json};
use support::*;

/// The prompt tokens `/v1/messages` reports for `body`, cached ones included.
async fn messages_prompt_tokens(router: &axum::Router, mut body: Value) -> u64 {
    body["max_tokens"] = json!(1);
    let reply = post(router, "/v1/messages", &body).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let usage = reply.json()["usage"].clone();
    usage["input_tokens"].as_u64().unwrap() + usage["cache_read_input_tokens"].as_u64().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn count_tokens_is_the_prompt_messages_runs() {
    let (router, _dir) = app(1024);
    let tool = json!({"name": "get_weather", "description": "weather in a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}});
    let requests = [
        json!({"messages": [{"role": "user", "content": "hi"}]}),
        json!({
            "system": "be brief",
            "messages": [
                {"role": "user", "content": "hi there"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": [{"type": "text", "text": "and"}, {"type": "text", "text": "again"}]}
            ]
        }),
        // The system prompt as blocks.
        json!({
            "system": [{"type": "text", "text": "be brief"}, {"type": "text", "text": "be kind"}],
            "messages": [{"role": "user", "content": "hi"}]
        }),
        // Tool definitions, a tool call and its result.
        json!({
            "tools": [tool],
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}]}
            ]
        }),
    ];
    let mut counts = Vec::new();
    for mut body in requests {
        body["model"] = json!(MODEL_ID);
        let reply = post(&router, "/v1/messages/count_tokens", &body).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
        let counted = reply.json();
        assert_eq!(
            counted.as_object().unwrap().len(),
            1,
            "the response is {{input_tokens}} alone: {counted}"
        );
        let counted = counted["input_tokens"].as_u64().unwrap();
        assert_eq!(
            counted,
            messages_prompt_tokens(&router, body.clone()).await,
            "count_tokens and /v1/messages disagree about {body}"
        );
        counts.push(counted);
    }
    assert!(counts[1] > counts[0], "a longer conversation counts more");
    assert!(counts[3] > counts[0], "tools and their calls count");
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn count_tokens_refuses_what_messages_refuses() {
    let (router, _dir) = app(256);
    let base = |extra: Value| {
        let mut body = json!({"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };
    // max_tokens is not part of this request.
    let reply = post(&router, "/v1/messages/count_tokens", &base(json!({}))).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    for body in [
        base(json!({"messages": []})),
        base(json!({"thinking": {"type": "sometimes"}})),
        base(json!({"messages": [{"role": "system", "content": "x"}]})),
        base(json!({"model": "another-model"})),
        // An image for a model that reads none.
        base(json!({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]})),
    ] {
        let reply = post(&router, "/v1/messages/count_tokens", &body).await;
        assert!(
            reply.status.is_client_error(),
            "{body}: {} {}",
            reply.status,
            reply.text()
        );
        assert_eq!(reply.json()["type"], "error", "an Anthropic error: {body}");
    }
}
