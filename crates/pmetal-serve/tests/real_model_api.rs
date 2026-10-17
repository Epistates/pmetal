//! The API on a real checkpoint: structured output, tool calls, stop
//! sequences and the cost of a grammar. Opt-in: needs a Qwen3 checkpoint
//! (Qwen3-0.6B is enough).
//!
//! ```text
//! PMETAL_QWEN3_DIR=<snapshot> cargo test --release -p pmetal-serve \
//!     --test real_model_api -- --nocapture --test-threads=1
//! ```
//!
//! Checked:
//!
//! * `response_format` json_schema and json_object: the reply parses and
//!   validates, with the model thinking (Qwen3's default) and without;
//! * `tool_choice: "required"` and a named function: the reply is a call to
//!   an offered tool whose arguments parse; an `auto` request that the model
//!   answers with a call comes back as `tool_calls`, its markup never
//!   `content`;
//! * a stop sequence cuts a real reply at the right character;
//! * what a grammar costs per token against the same reply unconstrained
//!   (printed, not asserted: it depends on the machine).

mod support;

use std::path::PathBuf;
use std::time::Instant;

use axum::Router;
use axum::http::StatusCode;
use pmetal_models::DynamicModel;
use pmetal_serve::{InferenceEngine, ServeConfig};
use serde_json::{Value, json};
use support::post;

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("PMETAL_QWEN3_DIR").map(PathBuf::from)
}

const MODEL: &str = "qwen3";

fn router(dir: &std::path::Path) -> Router {
    let load_dir = dir.to_path_buf();
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(dir).expect("tokenizer");
    let engine = InferenceEngine::new_with_backend(
        move || Ok(DynamicModel::load(&load_dir)?),
        tokenizer,
        MODEL.into(),
        dir,
        4096,
        false,
        1024,
    )
    .expect("engine");
    pmetal_serve::server::build_router_with(engine, &ServeConfig::default())
}

async fn chat(router: &Router, body: Value) -> Value {
    let reply = post(router, "/v1/chat/completions", &body).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    reply.json()
}

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

fn weather_tool() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string"},
                           "unit": {"enum": ["celsius", "fahrenheit"]}},
            "required": ["city"], "additionalProperties": false}}})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structured_output_on_a_real_model() {
    let Some(dir) = model_dir() else {
        eprintln!("PMETAL_QWEN3_DIR not set; skipping");
        return;
    };
    let router = router(&dir);
    let schema = json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population_millions": {"type": "number"},
            "landmarks": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3}
        },
        "required": ["city", "country", "population_millions", "landmarks"],
        "additionalProperties": false
    });
    for thinking in [false, true] {
        let reply = chat(
            &router,
            json!({
                "model": MODEL, "max_tokens": 1500, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": thinking},
                "messages": [{"role": "user", "content":
                    "Describe Paris as JSON: city, country, population in millions, landmarks."}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "city", "strict": true, "schema": schema}}
            }),
        )
        .await;
        drain("json_schema");
        let message = &reply["choices"][0]["message"];
        let text = message["content"].as_str().unwrap();
        let value: Value = serde_json::from_str(text)
            .unwrap_or_else(|e| panic!("thinking={thinking}: {e}: {text}"));
        println!("thinking={thinking}: {text}");
        assert!(value["city"].is_string() && value["country"].is_string());
        assert!(value["population_millions"].is_number());
        let landmarks = value["landmarks"].as_array().unwrap();
        assert!((2..=3).contains(&landmarks.len()));
        assert_eq!(value.as_object().unwrap().len(), 4);
        if thinking {
            assert!(
                message["reasoning_content"]
                    .as_str()
                    .is_some_and(|r| !r.is_empty())
            );
        }
    }

    let reply = chat(
        &router,
        json!({"model": MODEL, "max_tokens": 600, "temperature": 0.7, "seed": 3,
               "chat_template_kwargs": {"enable_thinking": false},
               "messages": [{"role": "user", "content": "Give me a JSON object about a dog."}],
               "response_format": {"type": "json_object"}}),
    )
    .await;
    let text = reply["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        serde_json::from_str::<Value>(text).unwrap().is_object(),
        "{text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_calls_on_a_real_model() {
    let Some(dir) = model_dir() else {
        eprintln!("PMETAL_QWEN3_DIR not set; skipping");
        return;
    };
    let router = router(&dir);
    let messages = json!([{"role": "user", "content": "What is the weather in Paris?"}]);
    for choice in [
        json!("required"),
        json!({"type": "function", "function": {"name": "get_weather"}}),
    ] {
        let reply = chat(
            &router,
            json!({"model": MODEL, "max_tokens": 800, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": false},
                   "messages": messages, "tools": [weather_tool()], "tool_choice": choice}),
        )
        .await;
        drain("tool_choice");
        let choice = &reply["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{reply}");
        let call = &choice["message"]["tool_calls"][0]["function"];
        assert_eq!(call["name"], "get_weather");
        let arguments: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
        println!("forced call: {arguments}");
        assert!(
            arguments["city"]
                .as_str()
                .is_some_and(|c| c.contains("Paris")),
            "{arguments}"
        );
    }

    // Left to itself the model calls the tool, and the call comes back parsed:
    // its markup is not content.
    let reply = chat(
        &router,
        json!({"model": MODEL, "max_tokens": 800, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": false},
               "messages": messages, "tools": [weather_tool()]}),
    )
    .await;
    let choice = &reply["choices"][0];
    println!("auto: {}", choice["message"]);
    assert_eq!(choice["finish_reason"], "tool_calls", "{reply}");
    let content = choice["message"]["content"].as_str().unwrap_or_default();
    assert!(!content.contains("<tool_call>"), "{content}");

    // The result goes back, and the model answers in text.
    let call = choice["message"]["tool_calls"][0].clone();
    let reply = chat(
        &router,
        json!({"model": MODEL, "max_tokens": 200, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": false},
               "messages": [
                   {"role": "user", "content": "What is the weather in Paris?"},
                   {"role": "assistant", "content": null, "tool_calls": [call.clone()]},
                   {"role": "tool", "tool_call_id": call["id"], "content": "18 degrees celsius, sunny"}],
               "tools": [weather_tool()]}),
    )
    .await;
    let answer = reply["choices"][0]["message"]["content"].as_str().unwrap();
    println!("answer: {answer}");
    assert!(answer.contains("18"), "{answer}");
    assert_eq!(reply["choices"][0]["finish_reason"], "stop");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stop_sequence_cuts_a_real_reply() {
    let Some(dir) = model_dir() else {
        eprintln!("PMETAL_QWEN3_DIR not set; skipping");
        return;
    };
    let router = router(&dir);
    let body = |extra: Value| {
        let mut body = json!({
            "model": MODEL, "max_tokens": 60, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": "Count from 1 to 20, separated by commas."}]
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };
    let full = chat(&router, body(json!({}))).await;
    let text = full["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    println!("full: {text}");
    // Cut inside a number: "1" of "12" never starts a token of its own in
    // this vocabulary's digit-by-digit tokens, so the stop starts mid-number.
    let at = text.find("12").expect("counts to 12");
    let cut = chat(&router, body(json!({"stop": ["2, 13"]}))).await;
    let got = cut["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(got, &text[..text.find("2, 13").unwrap()]);
    assert!(got.ends_with("11, 1"), "{got}");
    assert_eq!(cut["choices"][0]["finish_reason"], "stop");
    let _ = at;
    drain("stop");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn what_a_grammar_costs() {
    let Some(dir) = model_dir() else {
        eprintln!("PMETAL_QWEN3_DIR not set; skipping");
        return;
    };
    let router = router(&dir);
    let tokens = 160usize;
    let schema = json!({"type": "array", "items": {"type": "integer"}, "minItems": 100});
    let plain = json!({
        "model": MODEL, "max_tokens": tokens, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": "List the integers from 1 up as a JSON array."}]
    });
    let mut constrained = plain.clone();
    constrained["response_format"] =
        json!({"type": "json_schema", "json_schema": {"name": "n", "schema": schema}});

    // One request to warm the model, the prefix cache and the grammar
    // compiler, then interleaved runs.
    let _ = chat(&router, plain.clone()).await;
    let _ = chat(&router, constrained.clone()).await;
    let mut rates = [Vec::new(), Vec::new()];
    for _ in 0..12 {
        for (i, body) in [&plain, &constrained].into_iter().enumerate() {
            let started = Instant::now();
            let reply = chat(&router, body.clone()).await;
            let done = reply["usage"]["completion_tokens"].as_f64().unwrap();
            rates[i].push(done / started.elapsed().as_secs_f64());
        }
    }
    // Other work on the machine only ever slows a run, so the fastest of the
    // interleaved runs is the closest to the quiet figure; the median shows
    // how much the machine was disturbed.
    let sorted = |v: &mut Vec<f64>| v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted(&mut rates[0]);
    sorted(&mut rates[1]);
    let (best_plain, best_grammar) = (rates[0][11], rates[1][11]);
    let (median_plain, median_grammar) = (rates[0][6], rates[1][6]);
    println!(
        "decode, fastest of 12 interleaved runs: {best_plain:.1} tok/s unconstrained, \
         {best_grammar:.1} tok/s with a grammar ({:+.1}%); medians {median_plain:.1} and \
         {median_grammar:.1} ({:+.1}%)",
        (best_grammar / best_plain - 1.0) * 100.0,
        (median_grammar / median_plain - 1.0) * 100.0,
    );
    drain("grammar cost");
}
