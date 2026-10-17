//! What the server answers when a request is wrong: OpenAI's error object
//! (`{"error": {"message", "type", "param", "code"}}`) on its routes,
//! Anthropic's (`{"type": "error", "error": {"type", "message"}}`) on
//! `/v1/messages`, and the status each failure deserves.

mod support;

use axum::http::StatusCode;
use pmetal_serve::ServeConfig;
use serde_json::{Value, json};
use support::*;

fn chat(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

/// An OpenAI error object: `message` and `type` strings, `param` and `code`
/// strings or null (never a number).
fn assert_openai_error(reply: &Reply, status: StatusCode, kind: &str) -> Value {
    assert_eq!(reply.status, status, "{}", reply.text());
    let body = reply.json();
    let error = &body["error"];
    assert!(error["message"].as_str().is_some_and(|m| !m.is_empty()));
    assert_eq!(error["type"], kind, "{body}");
    for key in ["param", "code"] {
        assert!(
            error.get(key).is_some_and(|v| v.is_null() || v.is_string()),
            "error.{key} must be a string or null: {body}"
        );
    }
    body
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_json_is_an_openai_error_not_plain_text() {
    let (router, _dir) = app(64);
    let reply = post_raw(&router, "/v1/chat/completions", b"{not json".to_vec(), &[]).await;
    assert_openai_error(&reply, StatusCode::BAD_REQUEST, "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_missing_field_is_a_400_that_names_it() {
    let (router, _dir) = app(64);
    let reply = post(&router, "/v1/chat/completions", &json!({"model": MODEL_ID})).await;
    let body = assert_openai_error(&reply, StatusCode::BAD_REQUEST, "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("messages")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wrongly_typed_field_names_its_param() {
    let (router, _dir) = app(64);
    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"temperature": "hot"})),
    )
    .await;
    let body = assert_openai_error(&reply, StatusCode::BAD_REQUEST, "invalid_request_error");
    assert_eq!(body["error"]["param"], "temperature");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_body_that_is_not_json_content_type_is_415() {
    let (router, _dir) = app(64);
    let request = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "text/plain")
        .body(axum::body::Body::from("hello"))
        .unwrap();
    let reply = send(&router, request).await;
    assert_openai_error(
        &reply,
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "invalid_request_error",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_model_is_a_404_model_not_found() {
    let (router, _dir) = app(64);
    for path in ["/v1/chat/completions", "/v1/completions"] {
        let mut body = chat(json!({"model": "gpt-9"}));
        if path == "/v1/completions" {
            body = json!({"model": "gpt-9", "prompt": "hi", "max_tokens": 2});
        }
        let reply = post(&router, path, &body).await;
        let body = assert_openai_error(&reply, StatusCode::NOT_FOUND, "invalid_request_error");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
    }
    let reply = post(
        &router,
        "/v1/embeddings",
        &json!({"model": "gpt-9", "input": "hi"}),
    )
    .await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread")]
async fn models_are_listed_and_fetched_by_id() {
    let (router, _dir) = app(64);
    let listed = get(&router, "/v1/models").await.json();
    assert_eq!(listed["object"], "list");
    assert_eq!(listed["data"][0]["id"], MODEL_ID);
    let one = get(&router, &format!("/v1/models/{MODEL_ID}")).await;
    assert_eq!(one.status, StatusCode::OK);
    assert_eq!(one.json()["id"], MODEL_ID);
    assert_eq!(one.json()["object"], "model");
    let missing = get(&router, "/v1/models/nope").await;
    assert_openai_error(&missing, StatusCode::NOT_FOUND, "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_route_is_a_json_404() {
    let (router, _dir) = app(64);
    let reply = get(&router, "/v1/nothing").await;
    assert_openai_error(&reply, StatusCode::NOT_FOUND, "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_body_past_the_limit_is_a_413_error_object() {
    let config = ServeConfig {
        max_body_bytes: 256,
        ..ServeConfig::default()
    };
    let (router, _dir) = tiny(64).router(&config);
    let long = "x".repeat(1024);
    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"messages": [
            {"role": "user", "content": long}
        ]})),
    )
    .await;
    assert_openai_error(
        &reply,
        StatusCode::PAYLOAD_TOO_LARGE,
        "invalid_request_error",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_errors_use_anthropic_s_object() {
    let (router, _dir) = app(64);
    let reply = post_raw(&router, "/v1/messages", b"{nope".to_vec(), &[]).await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    let body = reply.json();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].as_str().is_some());

    let reply = post(
        &router,
        "/v1/messages",
        &json!({"model": "claude-9", "max_tokens": 4,
                "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
    let body = reply.json();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "not_found_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_api_key_gates_every_route_but_health() {
    let config = ServeConfig {
        api_keys: vec!["sk-test-1".into(), "sk-test-2".into()],
        ..ServeConfig::default()
    };
    let (router, _dir) = tiny(64).router(&config);

    let health = get(&router, "/health").await;
    assert_eq!(health.status, StatusCode::OK);

    let anonymous = post(&router, "/v1/chat/completions", &chat(json!({}))).await;
    let body = assert_openai_error(&anonymous, StatusCode::UNAUTHORIZED, "authentication_error");
    assert_eq!(body["error"]["code"], "invalid_api_key");
    assert!(anonymous.headers.contains_key("www-authenticate"));

    let wrong = post_with_headers(
        &router,
        "/v1/chat/completions",
        &chat(json!({})),
        &[("authorization", "Bearer sk-nope")],
    )
    .await;
    assert_eq!(wrong.status, StatusCode::UNAUTHORIZED);

    for key in ["sk-test-1", "sk-test-2"] {
        let ok = post_with_headers(
            &router,
            "/v1/chat/completions",
            &chat(json!({})),
            &[("authorization", &format!("Bearer {key}"))],
        )
        .await;
        assert_eq!(ok.status, StatusCode::OK, "{}", ok.text());
    }

    // Anthropic clients send the key in `x-api-key`.
    let messages = json!({"model": MODEL_ID, "max_tokens": 2,
                          "messages": [{"role": "user", "content": "hi"}]});
    let by_header = post_with_headers(
        &router,
        "/v1/messages",
        &messages,
        &[("x-api-key", "sk-test-2")],
    )
    .await;
    assert_eq!(by_header.status, StatusCode::OK, "{}", by_header.text());
    let denied = post(&router, "/v1/messages", &messages).await;
    assert_eq!(denied.status, StatusCode::UNAUTHORIZED);
    assert_eq!(denied.json()["error"]["type"], "authentication_error");

    let models = get(&router, "/v1/models").await;
    assert_eq!(models.status, StatusCode::UNAUTHORIZED);
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn cors_is_off_unless_an_origin_is_allowed() {
    let origin = ("origin", "https://app.example");
    let (router, _dir) = app(64);
    let reply = get_with_headers(&router, "/v1/models", &[origin]).await;
    assert!(!reply.headers.contains_key("access-control-allow-origin"));

    let config = ServeConfig {
        cors_allowed_origins: vec!["https://app.example".into()],
        ..ServeConfig::default()
    };
    let (router, _dir) = tiny(64).router(&config);
    let reply = get_with_headers(&router, "/v1/models", &[origin]).await;
    assert_eq!(
        reply.headers.get("access-control-allow-origin").unwrap(),
        "https://app.example"
    );
    let other =
        get_with_headers(&router, "/v1/models", &[("origin", "https://evil.example")]).await;
    assert!(!other.headers.contains_key("access-control-allow-origin"));
}
