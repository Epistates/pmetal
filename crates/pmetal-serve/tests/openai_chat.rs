//! `/v1/chat/completions` and `/v1/completions` against OpenAI's documented
//! behavior: choices, usage, finish reasons, stop sequences, streaming frames,
//! logprobs, validation. Driven through the real router over the tiny seeded
//! model of `tests/support`.
//!
//! The model's replies are noise, but a `logit_bias` of +100 on a handful of
//! tokens makes them noise over a chosen alphabet, so text can be compared
//! exactly.

mod support;

use axum::http::StatusCode;
use pmetal_serve::ServeConfig;
use serde_json::{Value, json};
use support::*;

/// A bias that confines a reply to ASCII lowercase letters (a byte-level
/// vocabulary's ids for them are their byte values).
fn letters() -> Value {
    let mut bias = serde_json::Map::new();
    for byte in b'a'..=b'z' {
        bias.insert(byte.to_string(), json!(100));
    }
    Value::Object(bias)
}

fn chat(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 12,
        "temperature": 0,
        "logit_bias": letters(),
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

async fn ok(router: &axum::Router, body: Value) -> Value {
    let reply = post(router, "/v1/chat/completions", &body).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    reply.json()
}

fn text_of(reply: &Value, choice: usize) -> String {
    reply["choices"][choice]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

/// The text a stream's content deltas add up to, for choice `index`.
fn streamed_text(frames: &[Value], index: usize) -> String {
    frames
        .iter()
        .flat_map(|f| f["choices"].as_array().cloned().unwrap_or_default())
        .filter(|c| c["index"] == index)
        .filter_map(|c| c["delta"]["content"].as_str().map(str::to_string))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_reply_has_openai_s_shape() {
    let (router, _dir) = app(256);
    let reply = ok(&router, chat(json!({}))).await;
    assert_eq!(reply["object"], "chat.completion");
    assert!(reply["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(reply["model"], MODEL_ID);
    assert!(reply["created"].as_i64().unwrap() > 1_600_000_000);
    let choice = &reply["choices"][0];
    assert_eq!(choice["index"], 0);
    assert_eq!(choice["message"]["role"], "assistant");
    assert!(
        choice["message"]["content"]
            .as_str()
            .is_some_and(|t| !t.is_empty())
    );
    assert!(choice["message"]["refusal"].is_null());
    assert!(choice["logprobs"].is_null());
    assert_eq!(choice["finish_reason"], "length");
    let usage = &reply["usage"];
    assert_eq!(usage["completion_tokens"], 12);
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["prompt_tokens"].as_u64().unwrap() + 12
    );
    assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 0);
    assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 0);
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_conversation_that_continues_reads_its_prompt_from_the_cache() {
    let (router, _dir) = app(512);
    let first = ok(&router, chat(json!({"max_tokens": 4}))).await;
    assert_eq!(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    let prompt_tokens = first["usage"]["prompt_tokens"].as_u64().unwrap();

    let second = ok(
        &router,
        chat(json!({"max_tokens": 4, "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": text_of(&first, 0)},
            {"role": "user", "content": "and again"},
        ]})),
    )
    .await;
    let cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    assert!(
        cached > 0 && cached <= prompt_tokens,
        "cached {cached} of {prompt_tokens}"
    );
    assert!(cached < second["usage"]["prompt_tokens"].as_u64().unwrap());

    // And in Anthropic's terms: those tokens are read from the cache, not input.
    let reply = post(
        &router,
        "/v1/messages",
        &json!({"model": MODEL_ID, "max_tokens": 2, "temperature": 0, "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": text_of(&first, 0)},
            {"role": "user", "content": "once more"}]}),
    )
    .await
    .json();
    assert!(reply["usage"]["cache_read_input_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn n_choices_come_back_indexed_and_usage_adds_up() {
    let (router, _dir) = app(256);
    let one = ok(&router, chat(json!({"seed": 5}))).await;
    let reply = ok(&router, chat(json!({"seed": 5, "n": 3}))).await;
    let choices = reply["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 3);
    for (i, choice) in choices.iter().enumerate() {
        assert_eq!(choice["index"], i);
        assert_eq!(choice["finish_reason"], "length");
    }
    // Greedy: every choice is the single-choice reply.
    for i in 0..3 {
        assert_eq!(text_of(&reply, i), text_of(&one, 0));
    }
    // The prompt counts once, the completions three times.
    assert_eq!(reply["usage"]["completion_tokens"], 36);
    assert_eq!(
        reply["usage"]["prompt_tokens"],
        one["usage"]["prompt_tokens"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_seed_makes_sampling_repeatable() {
    let (router, _dir) = app(256);
    let sampled = |seed: i64| chat(json!({"temperature": 1.0, "seed": seed, "max_tokens": 16}));
    let a = text_of(&ok(&router, sampled(11)).await, 0);
    let b = text_of(&ok(&router, sampled(11)).await, 0);
    let c = text_of(&ok(&router, sampled(12)).await, 0);
    assert_eq!(a, b);
    assert_ne!(a, c);
    // A negative seed is a seed too.
    let d = text_of(&ok(&router, sampled(-4)).await, 0);
    assert_eq!(d, text_of(&ok(&router, sampled(-4)).await, 0));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_sequence_cuts_the_text_even_inside_a_token() {
    let (router, _dir) = app(256);
    // Only the tokens "the" (one token) and "x": the reply is a run of them.
    let bias = json!({"259": 100, "120": 100});
    let body = |extra: Value| {
        let mut b = chat(json!({"logit_bias": bias, "max_tokens": 10}));
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        b
    };
    let full = text_of(&ok(&router, body(json!({}))).await, 0);
    assert!(full.contains("he"), "{full:?}");
    let at = full.find("he").unwrap();

    let cut = ok(&router, body(json!({"stop": "he"}))).await;
    assert_eq!(text_of(&cut, 0), &full[..at], "full reply was {full:?}");
    assert_eq!(cut["choices"][0]["finish_reason"], "stop");

    // The same through a stream, and with the stop in a list.
    let reply = post(
        &router,
        "/v1/chat/completions",
        &body(json!({"stop": ["zzz", "he"], "stream": true})),
    )
    .await;
    let frames = reply.sse_json();
    assert_eq!(streamed_text(&frames, 0), &full[..at]);
    let last = frames.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert!(reply.text().trim_end().ends_with("[DONE]"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stream_is_role_deltas_finish_and_done() {
    let (router, _dir) = app(256);
    let whole = text_of(&ok(&router, chat(json!({}))).await, 0);
    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"stream": true})),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert!(
        reply.headers["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    let frames = reply.sse_json();
    assert_eq!(frames[0]["object"], "chat.completion.chunk");
    assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(streamed_text(&frames, 0), whole);
    // Exactly one finish_reason, on the last chunk, and no usage unless asked.
    let finishes: Vec<_> = frames
        .iter()
        .filter(|f| !f["choices"][0]["finish_reason"].is_null())
        .collect();
    assert_eq!(finishes.len(), 1);
    assert_eq!(finishes[0]["choices"][0]["finish_reason"], "length");
    assert!(std::ptr::eq(
        *finishes.last().unwrap(),
        frames.last().unwrap()
    ));
    assert!(frames.iter().all(|f| f.get("usage").is_none()));
    let events = reply.sse();
    assert_eq!(events.last().unwrap().1, "[DONE]");
}

#[tokio::test(flavor = "multi_thread")]
async fn include_usage_adds_a_last_chunk_with_no_choices() {
    let (router, _dir) = app(256);
    let whole = ok(&router, chat(json!({"n": 2}))).await;
    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"stream": true, "n": 2, "stream_options": {"include_usage": true}})),
    )
    .await;
    let frames = reply.sse_json();
    let (usage_chunk, others) = frames.split_last().unwrap();
    assert_eq!(usage_chunk["choices"], json!([]));
    // The same counts as the JSON reply (the earlier request left its prompt in
    // the cache, which the later one reports it read).
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        assert_eq!(usage_chunk["usage"][key], whole["usage"][key], "{key}");
    }
    assert!(
        usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    // Every other chunk says `usage: null`.
    assert!(
        others
            .iter()
            .all(|f| f["usage"].is_null() && f.get("usage").is_some())
    );
    assert_eq!(reply.sse().last().unwrap().1, "[DONE]");
    // Both choices streamed and finished.
    for i in 0..2 {
        assert_eq!(streamed_text(&frames, i), text_of(&whole, i));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn logprobs_carry_bytes_and_alternatives() {
    let (router, _dir) = app(256);
    let reply = ok(&router, chat(json!({"logprobs": true, "top_logprobs": 3}))).await;
    let content = reply["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap();
    assert_eq!(content.len(), 12);
    for entry in content {
        let token = entry["token"].as_str().unwrap();
        let bytes: Vec<u8> = entry["bytes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(bytes, token.as_bytes());
        assert!(entry["logprob"].as_f64().unwrap() <= 0.0);
        let top = entry["top_logprobs"].as_array().unwrap();
        assert!(top.len() <= 3);
        assert!(
            top.iter()
                .all(|t| t["bytes"].is_array() && t["token"].is_string())
        );
    }
    assert!(reply["choices"][0]["logprobs"]["refusal"].is_null());

    // Streamed, the logprobs ride the content deltas.
    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"logprobs": true, "stream": true})),
    )
    .await;
    let count: usize = reply
        .sse_json()
        .iter()
        .filter_map(|f| {
            f["choices"][0]["logprobs"]["content"]
                .as_array()
                .map(Vec::len)
        })
        .sum();
    assert_eq!(count, 12);
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_values_are_named() {
    let (router, _dir) = app(256);
    let cases = [
        (json!({"top_logprobs": 2}), "top_logprobs"),
        (
            json!({"logprobs": true, "top_logprobs": 21}),
            "top_logprobs",
        ),
        (json!({"temperature": 3.0}), "temperature"),
        (json!({"top_p": 0.0}), "top_p"),
        (json!({"n": 0}), "n"),
        (json!({"n": 999}), "n"),
        (json!({"stop": ["a", "b", "c", "d", "e"]}), "stop"),
        (json!({"max_tokens": 0}), "max_tokens"),
        (
            json!({"stream_options": {"include_usage": true}}),
            "stream_options",
        ),
        (json!({"logit_bias": {"x": 1}}), "logit_bias"),
        (json!({"logit_bias": {"5": 500}}), "logit_bias"),
        (json!({"frequency_penalty": 5.0}), "frequency_penalty"),
        (
            json!({"messages": [{"role": "wizard", "content": "hi"}]}),
            "messages.0.role",
        ),
        (json!({"messages": []}), "messages"),
        (
            json!({"tool_choice": "sometimes", "tools": [{"type": "function", "function": {"name": "f"}}]}),
            "tool_choice",
        ),
    ];
    for (extra, param) in cases {
        let reply = post(&router, "/v1/chat/completions", &chat(extra.clone())).await;
        assert_eq!(
            reply.status,
            StatusCode::BAD_REQUEST,
            "{extra}: {}",
            reply.text()
        );
        assert_eq!(reply.json()["error"]["param"], param, "{extra}");
        assert_eq!(reply.json()["error"]["type"], "invalid_request_error");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn documented_fields_the_server_cannot_honour_are_refused_by_name() {
    let (router, _dir) = app(256);
    for (extra, param) in [
        (
            json!({"audio": {"voice": "alloy", "format": "wav"}}),
            "audio",
        ),
        (json!({"modalities": ["text", "audio"]}), "modalities"),
        (
            json!({"prediction": {"type": "content", "content": "x"}}),
            "prediction",
        ),
        (json!({"web_search_options": {}}), "web_search_options"),
    ] {
        let reply = post(&router, "/v1/chat/completions", &chat(extra)).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST, "{}", reply.text());
        let error = &reply.json()["error"];
        assert_eq!(error["param"], param);
        assert_eq!(error["code"], "unsupported_parameter");
    }
    // Labels and text-only modalities are fine.
    let reply = ok(
        &router,
        chat(json!({"user": "u", "metadata": {"k": "v"}, "store": false,
                    "modalities": ["text"], "service_tier": "auto",
                    "safety_identifier": "s", "prompt_cache_key": "k"})),
    )
    .await;
    assert_eq!(reply["choices"].as_array().unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_that_fills_the_context_is_refused_and_output_is_clamped() {
    let (router, _dir) = app(64);
    let long = "word ".repeat(100);
    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"messages": [{"role": "user", "content": long}]})),
    )
    .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    let error = &reply.json()["error"];
    assert_eq!(error["code"], "context_length_exceeded");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("maximum context length is 64")
    );

    // A budget bigger than the room left is cut to the room, and says so.
    let reply = ok(&router, chat(json!({"max_tokens": 10_000}))).await;
    let prompt = reply["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(
        reply["usage"]["completion_tokens"].as_u64().unwrap(),
        64 - prompt
    );
    assert_eq!(reply["choices"][0]["finish_reason"], "length");
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn max_completion_tokens_wins_over_max_tokens() {
    let (router, _dir) = app(256);
    let reply = ok(
        &router,
        chat(json!({"max_tokens": 9, "max_completion_tokens": 5})),
    )
    .await;
    assert_eq!(reply["usage"]["completion_tokens"], 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_history_reaches_the_template() {
    let (router, _dir) = app(512);
    // An assistant turn that called a tool (arguments as OpenAI sends them, a
    // JSON string) and the tool's answer.
    let reply = ok(
        &router,
        chat(json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
            {"role": "developer", "content": "be brief"},
        ]})),
    )
    .await;
    assert_eq!(reply["choices"][0]["message"]["role"], "assistant");
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_past_the_time_limit_ends_with_a_timeout() {
    let config = ServeConfig {
        request_timeout: Some(std::time::Duration::from_millis(1)),
        ..ServeConfig::default()
    };
    let (router, _dir) = tiny(512).router(&config);
    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"max_tokens": 300})),
    )
    .await;
    assert_eq!(
        reply.status,
        StatusCode::GATEWAY_TIMEOUT,
        "{}",
        reply.text()
    );
    assert_eq!(reply.json()["error"]["type"], "timeout_error");

    // Mid-stream, the failure is a frame; `[DONE]` does not follow it.
    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"max_tokens": 300, "stream": true})),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert!(reply.text().contains("\"error\""), "{}", reply.text());
    assert!(!reply.text().contains("[DONE]"));
}

// ── /v1/completions ──────────────────────────────────────────────────────────

fn completion(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "prompt": "hi",
        "max_tokens": 8,
        "temperature": 0,
        "logit_bias": letters(),
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

#[tokio::test(flavor = "multi_thread")]
async fn completions_take_every_prompt_form() {
    let (router, _dir) = app(256);
    let text = |reply: &Value, i: usize| reply["choices"][i]["text"].as_str().unwrap().to_string();

    let one = post(&router, "/v1/completions", &completion(json!({})))
        .await
        .json();
    assert_eq!(one["object"], "text_completion");
    assert_eq!(one["choices"][0]["finish_reason"], "length");
    assert_eq!(one["usage"]["completion_tokens"], 8);

    // Token ids for "hi" give the same reply as the text.
    let ids = tokenizer().encode("hi").unwrap();
    let by_ids = post(
        &router,
        "/v1/completions",
        &completion(json!({"prompt": ids})),
    )
    .await
    .json();
    assert_eq!(text(&by_ids, 0), text(&one, 0));

    // A batch of prompts, two choices each: four choices, prompt-major.
    let batch = post(
        &router,
        "/v1/completions",
        &completion(json!({"prompt": ["hi", "hi"], "n": 2})),
    )
    .await
    .json();
    let choices = batch["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 4);
    for (i, choice) in choices.iter().enumerate() {
        assert_eq!(choice["index"], i);
        assert_eq!(text(&batch, i), text(&one, 0));
    }
    assert_eq!(batch["usage"]["completion_tokens"], 32);
    assert_eq!(
        batch["usage"]["prompt_tokens"].as_u64().unwrap(),
        2 * one["usage"]["prompt_tokens"].as_u64().unwrap()
    );

    // echo puts the prompt in front.
    let echoed = post(
        &router,
        "/v1/completions",
        &completion(json!({"echo": true})),
    )
    .await
    .json();
    assert_eq!(text(&echoed, 0), format!("hi{}", text(&one, 0)));
}

#[tokio::test(flavor = "multi_thread")]
async fn completions_logprobs_are_four_parallel_arrays() {
    let (router, _dir) = app(256);
    let reply = post(
        &router,
        "/v1/completions",
        &completion(json!({"logprobs": 2})),
    )
    .await
    .json();
    let lp = &reply["choices"][0]["logprobs"];
    let n = lp["tokens"].as_array().unwrap().len();
    assert_eq!(n, 8);
    assert_eq!(lp["token_logprobs"].as_array().unwrap().len(), n);
    assert_eq!(lp["top_logprobs"].as_array().unwrap().len(), n);
    let offsets: Vec<u64> = lp["text_offset"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_u64().unwrap())
        .collect();
    assert_eq!(offsets, (0..n as u64).collect::<Vec<_>>());
    assert_eq!(reply["choices"][0]["text"].as_str().unwrap().len(), n);

    let too_many = post(
        &router,
        "/v1/completions",
        &completion(json!({"logprobs": 6})),
    )
    .await;
    assert_eq!(too_many.status, StatusCode::BAD_REQUEST);
    assert_eq!(too_many.json()["error"]["param"], "logprobs");
}

#[tokio::test(flavor = "multi_thread")]
async fn completions_stream_with_usage_and_refuse_what_they_cannot_do() {
    let (router, _dir) = app(256);
    let whole = post(&router, "/v1/completions", &completion(json!({})))
        .await
        .json();
    let reply = post(
        &router,
        "/v1/completions",
        &completion(json!({"stream": true, "stream_options": {"include_usage": true}})),
    )
    .await;
    let frames = reply.sse_json();
    let text: String = frames
        .iter()
        .filter_map(|f| f["choices"][0]["text"].as_str().map(str::to_string))
        .collect();
    assert_eq!(text, whole["choices"][0]["text"].as_str().unwrap());
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        assert_eq!(
            frames.last().unwrap()["usage"][key],
            whole["usage"][key],
            "{key}"
        );
    }
    assert_eq!(frames.last().unwrap()["choices"], json!([]));
    assert_eq!(reply.sse().last().unwrap().1, "[DONE]");

    for (extra, param) in [
        (json!({"suffix": "tail"}), "suffix"),
        (json!({"best_of": 3}), "best_of"),
        (json!({"prompt": []}), "prompt"),
        (json!({"prompt": ""}), "prompt"),
        (json!({"prompt": ["hi", ""]}), "prompt"),
    ] {
        let reply = post(&router, "/v1/completions", &completion(extra)).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST, "{}", reply.text());
        assert_eq!(reply.json()["error"]["param"], param);
    }
}
