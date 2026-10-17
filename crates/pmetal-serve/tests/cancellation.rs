//! A client that leaves ends its generation: the model thread is free for the
//! next request at once, not after the abandoned reply has run to its end.

mod support;

use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::Request;
use serde_json::{Value, json};
use support::*;
use tower::ServiceExt as _;

fn body(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 200,
        "temperature": 0,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

/// How long a one-token request takes now.
async fn quick_request(router: &axum::Router) -> Duration {
    let started = Instant::now();
    let reply = post(
        router,
        "/v1/chat/completions",
        &body(json!({"max_tokens": 1})),
    )
    .await;
    assert_eq!(reply.status, 200, "{}", reply.text());
    started.elapsed()
}

#[tokio::test(flavor = "multi_thread")]
async fn leaving_ends_the_generation_streamed_or_not() {
    let (router, _dir) = app(256);
    // The whole 200-token reply, once to warm the model and the prefix
    // cache, and again for the yardstick.
    let _ = post(&router, "/v1/chat/completions", &body(json!({}))).await;
    let started = Instant::now();
    let reply = post(&router, "/v1/chat/completions", &body(json!({}))).await;
    assert_eq!(reply.json()["usage"]["completion_tokens"], 200);
    let full = started.elapsed();
    let idle = quick_request(&router).await;

    // Non-streaming: the client gives up a quarter of the way through.
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body(json!({})).to_string()))
        .unwrap();
    let outcome = tokio::time::timeout(full / 4, router.clone().oneshot(request)).await;
    assert!(
        outcome.is_err(),
        "the reply should not have been ready in a quarter of its time"
    );
    let after_leaving = quick_request(&router).await;
    assert!(
        after_leaving < full / 2 + idle,
        "the model thread was still busy with a reply nobody wanted: {after_leaving:?} \
         (a whole reply takes {full:?}, an idle request {idle:?})"
    );

    // Streaming: read one frame, then drop the response.
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body(json!({"stream": true})).to_string()))
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let mut stream = response.into_body().into_data_stream();
    use futures::StreamExt as _;
    // Read until a token has been generated: the opening frame comes first
    // and holds an empty `content`.
    let has_token = |seen: &str| {
        seen.match_indices("\"content\":\"").any(|(at, key)| {
            seen[at + key.len()..]
                .chars()
                .next()
                .is_some_and(|c| c != '"')
        })
    };
    let mut seen = String::new();
    while !has_token(&seen) {
        let chunk = stream.next().await.expect("the stream ended").unwrap();
        seen.push_str(&String::from_utf8_lossy(&chunk));
        assert!(!seen.contains("[DONE]"), "the reply ended first: {seen}");
    }
    drop(stream);
    let after_leaving = quick_request(&router).await;
    assert!(
        after_leaving < full / 2 + idle,
        "a dropped stream kept the model thread busy: {after_leaving:?} (a whole reply takes {full:?})"
    );
    assert_no_unobserved_error();
}
