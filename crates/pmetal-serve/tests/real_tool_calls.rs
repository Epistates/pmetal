//! Tool calls as a real model writes them, through every endpoint. Opt-in:
//! point `PMETAL_TOOL_MODEL_DIR` at any tool-trained checkpoint (Qwen3,
//! Qwen3.5 and Gemma 4 write three different syntaxes).
//!
//! ```text
//! PMETAL_TOOL_MODEL_DIR=<snapshot> cargo test --release -p pmetal-serve \
//!     --test real_tool_calls -- --nocapture
//! ```
//!
//! Checked, greedy, with thinking off where the template has the switch:
//!
//! * a question the tool answers is answered with a call, parsed into
//!   `tool_calls` (chat completions), a `tool_use` block (messages) and a
//!   `function_call` item (responses), and the call's markup is never text;
//! * streamed, the call's deltas add up to the same call;
//! * the tool's result goes back and the model answers in text.

mod support;

use std::path::PathBuf;

use axum::Router;
use axum::http::StatusCode;
use pmetal_models::DynamicModel;
use pmetal_serve::{InferenceEngine, ServeConfig};
use serde_json::{Value, json};
use support::post;

const MODEL: &str = "tool-model";

fn router() -> Option<Router> {
    let dir = PathBuf::from(std::env::var_os("PMETAL_TOOL_MODEL_DIR")?);
    let load_dir = dir.clone();
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&dir).expect("tokenizer");
    let engine = InferenceEngine::new_with_backend(
        move || Ok(DynamicModel::load(&load_dir)?),
        tokenizer,
        MODEL.into(),
        &dir,
        4096,
        false,
        1024,
    )
    .expect("engine");
    Some(pmetal_serve::server::build_router_with(
        engine,
        &ServeConfig::default(),
    ))
}

fn weather_tool() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string", "description": "The city"}},
            "required": ["city"]}}})
}

async fn json_ok(router: &Router, path: &str, body: Value) -> Value {
    let reply = post(router, path, &body).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    reply.json()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_real_model_calls_a_tool_through_every_endpoint() {
    let Some(router) = router() else {
        eprintln!("PMETAL_TOOL_MODEL_DIR not set; skipping");
        return;
    };
    let question = "What is the weather in Paris right now? Use the tool.";

    // Chat completions.
    let chat = |stream: bool| {
        json!({"model": MODEL, "max_tokens": 600, "temperature": 0, "stream": stream,
               "chat_template_kwargs": {"enable_thinking": false},
               "messages": [{"role": "user", "content": question}],
               "tools": [weather_tool()]})
    };
    let reply = json_ok(&router, "/v1/chat/completions", chat(false)).await;
    let choice = &reply["choices"][0];
    println!("chat completions: {}", choice["message"]);
    assert_eq!(choice["finish_reason"], "tool_calls", "{reply}");
    let calls = choice["message"]["tool_calls"].as_array().unwrap();
    let call = &calls[0]["function"];
    assert_eq!(call["name"], "get_weather");
    let arguments: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
    assert!(
        arguments["city"]
            .as_str()
            .is_some_and(|c| c.contains("Paris")),
        "{arguments}"
    );
    let content = choice["message"]["content"].as_str().unwrap_or_default();
    for marker in ["<tool_call>", "<|tool_call>", "<function=", "[TOOL_CALLS]"] {
        assert!(!content.contains(marker), "markup in content: {content}");
    }

    // The same, streamed: the call arrives as deltas that add up.
    let streamed = post(&router, "/v1/chat/completions", &chat(true)).await;
    let (mut name, mut args) = (String::new(), String::new());
    for frame in streamed.sse_json() {
        for tc in frame["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            name.push_str(tc["function"]["name"].as_str().unwrap_or_default());
            args.push_str(tc["function"]["arguments"].as_str().unwrap_or_default());
        }
        let delta = frame["choices"][0]["delta"]["content"]
            .as_str()
            .unwrap_or_default();
        for marker in ["<tool_call>", "<|tool_call>", "<function="] {
            assert!(
                !delta.contains(marker),
                "markup streamed as content: {delta}"
            );
        }
    }
    assert_eq!(name, "get_weather");
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), arguments);

    // Messages.
    let messages = json_ok(
        &router,
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 600, "temperature": 0,
               "messages": [{"role": "user", "content": question}],
               "tools": [{"name": "get_weather", "description": "Get the weather in a city.",
                          "input_schema": weather_tool()["function"]["parameters"].clone()}]}),
    )
    .await;
    println!("messages: {}", messages["content"]);
    assert_eq!(messages["stop_reason"], "tool_use", "{messages}");
    let block = messages["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .expect("a tool_use block");
    assert_eq!(block["name"], "get_weather");
    assert!(
        block["input"]["city"]
            .as_str()
            .is_some_and(|c| c.contains("Paris"))
    );

    // Responses.
    let response = json_ok(
        &router,
        "/v1/responses",
        json!({"model": MODEL, "max_output_tokens": 600, "temperature": 0,
               "input": question,
               "tools": [{"type": "function", "name": "get_weather",
                          "description": "Get the weather in a city.",
                          "parameters": weather_tool()["function"]["parameters"].clone()}]}),
    )
    .await;
    println!("responses: {}", response["output"]);
    let item = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["type"] == "function_call")
        .expect("a function_call item");
    assert_eq!(item["name"], "get_weather");

    // The result goes back; the model answers in text.
    let answer = json_ok(
        &router,
        "/v1/chat/completions",
        json!({"model": MODEL, "max_tokens": 200, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": false},
               "messages": [
                   {"role": "user", "content": question},
                   {"role": "assistant", "content": null, "tool_calls": calls},
                   {"role": "tool", "tool_call_id": calls[0]["id"],
                    "content": "18 degrees celsius, sunny"}],
               "tools": [weather_tool()]}),
    )
    .await;
    let text = answer["choices"][0]["message"]["content"].as_str().unwrap();
    println!("answer: {text}");
    assert!(text.contains("18"), "{text}");
    assert_eq!(answer["choices"][0]["finish_reason"], "stop");
    pmetal_runtime::check_unobserved_error().expect("no op threw");
}
