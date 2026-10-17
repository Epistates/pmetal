//! Constrained decoding through the server: `response_format` and
//! `tool_choice`. The tiny seeded model is noise, so a reply that always
//! parses, always validates and always names an offered tool can only be the
//! grammar's doing.

mod support;

use axum::http::StatusCode;
use serde_json::{Value, json};
use support::*;

fn chat(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "give me json"}],
        "max_tokens": 120,
        "temperature": 1.0,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

fn person_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 6},
            "age": {"type": "integer"},
            "kind": {"enum": ["cat", "dog"]},
            "tags": {"type": "array", "items": {"type": "string", "maxLength": 3}, "maxItems": 2}
        },
        "required": ["name", "age", "kind"],
        "additionalProperties": false
    })
}

/// The subset of JSON Schema the tests use, checked independently of the
/// grammar compiler.
fn validates(schema: &Value, value: &Value) -> Result<(), String> {
    if let Some(options) = schema.get("enum") {
        return options
            .as_array()
            .unwrap()
            .contains(value)
            .then_some(())
            .ok_or(format!("{value} is not one of {options}"));
    }
    if let Some(constant) = schema.get("const") {
        return (constant == value)
            .then_some(())
            .ok_or(format!("{value} != {constant}"));
    }
    match schema["type"].as_str() {
        Some("object") => {
            let object = value.as_object().ok_or("not an object")?;
            let props = schema["properties"]
                .as_object()
                .cloned()
                .unwrap_or_default();
            for required in schema["required"].as_array().into_iter().flatten() {
                let key = required.as_str().unwrap();
                if !object.contains_key(key) {
                    return Err(format!("missing {key}"));
                }
            }
            for (key, v) in object {
                match props.get(key) {
                    Some(sub) => validates(sub, v)?,
                    None if schema["additionalProperties"] == json!(false) => {
                        return Err(format!("unexpected {key}"));
                    }
                    None => {}
                }
            }
            Ok(())
        }
        Some("string") => value.is_string().then_some(()).ok_or("not a string".into()),
        // A 19-digit integer is beyond i64; it is still an integer.
        Some("integer") => value
            .as_f64()
            .is_some_and(|n| n.fract() == 0.0)
            .then_some(())
            .ok_or("not an integer".into()),
        Some("array") => {
            let items = value.as_array().ok_or("not an array")?;
            if let Some(max) = schema["maxItems"].as_u64() {
                if items.len() as u64 > max {
                    return Err("too many items".into());
                }
            }
            items
                .iter()
                .try_for_each(|v| validates(&schema["items"], v))
        }
        other => Err(format!("test validator: unsupported {other:?}")),
    }
}

async fn ok(router: &axum::Router, body: Value) -> Value {
    let reply = post(router, "/v1/chat/completions", &body).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    reply.json()
}

fn content(reply: &Value, choice: usize) -> String {
    reply["choices"][choice]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn json_object_mode_always_gives_an_object() {
    let (router, _dir) = app(512);
    for seed in 0..6 {
        let reply = ok(
            &router,
            chat(json!({"response_format": {"type": "json_object"}, "seed": seed})),
        )
        .await;
        if reply["choices"][0]["finish_reason"] == "length" {
            continue; // cut off by the budget
        }
        let value: Value = serde_json::from_str(&content(&reply, 0))
            .unwrap_or_else(|e| panic!("seed {seed}: {e}: {}", content(&reply, 0)));
        assert!(value.is_object(), "{value}");
    }
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_json_schema_is_always_met() {
    let (router, _dir) = app(512);
    let schema = person_schema();
    let mut finished = 0;
    for seed in 0..12 {
        let reply = ok(
            &router,
            chat(json!({
                "seed": seed,
                "max_tokens": 200,
                "response_format": {"type": "json_schema",
                    "json_schema": {"name": "person", "strict": true, "schema": schema}}
            })),
        )
        .await;
        if reply["choices"][0]["finish_reason"] == "length" {
            // A reply cut off by the budget is a valid prefix, not a value.
            continue;
        }
        finished += 1;
        let text = content(&reply, 0);
        let value: Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("seed {seed}: {e}: {text}"));
        validates(&schema, &value).unwrap_or_else(|e| panic!("seed {seed}: {e}: {text:?}"));
    }
    assert!(
        finished >= 3,
        "only {finished} of 12 finished within the budget"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn every_choice_and_a_stream_are_constrained() {
    let (router, _dir) = app(512);
    let schema = json!({"type": "object", "properties": {"ok": {"enum": [true, false]}},
                        "required": ["ok"], "additionalProperties": false});
    let format = json!({"type": "json_schema", "json_schema": {"name": "r", "schema": schema}});
    let reply = ok(
        &router,
        chat(json!({"n": 3, "seed": 4, "response_format": format.clone(), "max_tokens": 60})),
    )
    .await;
    for i in 0..3 {
        let text = content(&reply, i);
        let value: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
        assert!(value["ok"].is_boolean(), "{text}");
    }
    let streamed = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"stream": true, "seed": 4, "response_format": format, "max_tokens": 60})),
    )
    .await;
    let text: String = streamed
        .sse_json()
        .iter()
        .filter_map(|f| {
            f["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    let value: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
    assert!(value["ok"].is_boolean());
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_schema_the_compiler_refuses_is_a_400() {
    let (router, _dir) = app(512);
    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"response_format": {"type": "json_schema",
            "json_schema": {"name": "bad", "schema": {"type": "nonsense"}}}})),
    )
    .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST, "{}", reply.text());
    assert_eq!(reply.json()["error"]["code"], "invalid_json_schema");

    let reply = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}})),
    )
    .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        reply.json()["error"]["param"],
        "response_format.json_schema.schema"
    );
}

fn weather_tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object",
                "properties": {"city": {"type": "string", "maxLength": 8}, "days": {"type": "integer"}},
                "required": ["city"], "additionalProperties": false}}},
        {"type": "function", "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {"zone": {"enum": ["utc", "pst"]}},
                "required": ["zone"], "additionalProperties": false}}},
    ])
}

fn only_calls(reply: &Value) -> Vec<(String, Value)> {
    let message = &reply["choices"][0]["message"];
    assert!(
        message["content"].is_null() || message["content"] == "",
        "{message}"
    );
    message["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("no tool_calls: {reply}"))
        .iter()
        .map(|call| {
            assert_eq!(call["type"], "function");
            assert!(call["id"].as_str().unwrap().starts_with("call_"));
            let arguments: Value =
                serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
            (
                call["function"]["name"].as_str().unwrap().to_string(),
                arguments,
            )
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_required_tool_choice_always_calls_an_offered_tool() {
    let (router, _dir) = app(1024);
    let tools = weather_tools();
    let mut seen = std::collections::HashSet::new();
    for seed in 0..10 {
        let reply = ok(
            &router,
            chat(
                json!({"tools": tools, "tool_choice": "required", "seed": seed,
                        "max_tokens": 300, "parallel_tool_calls": false}),
            ),
        )
        .await;
        assert_eq!(
            reply["choices"][0]["finish_reason"], "tool_calls",
            "{reply}"
        );
        let calls = only_calls(&reply);
        assert_eq!(calls.len(), 1);
        let (name, arguments) = &calls[0];
        let schema = tools
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["function"]["name"] == name.as_str())
            .unwrap_or_else(|| panic!("a tool that was not offered: {name}"))["function"]["parameters"]
            .clone();
        validates(&schema, arguments).unwrap_or_else(|e| panic!("{name}({arguments}): {e}"));
        seen.insert(name.clone());
    }
    assert!(!seen.is_empty());
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_named_tool_choice_calls_that_tool_and_streams_its_call() {
    let (router, _dir) = app(1024);
    let tool_choice = json!({"type": "function", "function": {"name": "get_time"}});
    let reply = ok(
        &router,
        chat(
            json!({"tools": weather_tools(), "tool_choice": tool_choice.clone(),
                    "max_tokens": 300, "seed": 3, "parallel_tool_calls": false}),
        ),
    )
    .await;
    let calls = only_calls(&reply);
    assert!(
        calls.iter().all(|(name, _)| name == "get_time"),
        "{calls:?}"
    );
    assert!(["utc", "pst"].contains(&calls[0].1["zone"].as_str().unwrap()));

    // The same request streamed: the call arrives as indexed deltas that add
    // up to the same call.
    let streamed = post(
        &router,
        "/v1/chat/completions",
        &chat(json!({"tools": weather_tools(), "tool_choice": tool_choice,
                     "max_tokens": 300, "seed": 3, "stream": true,
                     "parallel_tool_calls": false})),
    )
    .await;
    let frames = streamed.sse_json();
    let mut name = String::new();
    let mut arguments = String::new();
    let mut ids = Vec::new();
    for frame in &frames {
        for call in frame["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            assert_eq!(call["index"], 0);
            if let Some(id) = call["id"].as_str() {
                ids.push(id.to_string());
                assert_eq!(call["type"], "function");
            }
            name.push_str(call["function"]["name"].as_str().unwrap_or_default());
            arguments.push_str(call["function"]["arguments"].as_str().unwrap_or_default());
        }
        let text = &frame["choices"][0]["delta"]["content"];
        assert!(
            text.is_null() || text == "",
            "tool-call markup leaked as content: {frame}"
        );
    }
    assert_eq!(ids.len(), 1);
    assert_eq!(name, "get_time");
    let streamed_args: Value = serde_json::from_str(&arguments).unwrap();
    assert_eq!(streamed_args, calls[0].1);
    let last = frames.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_choice_mistakes_are_named() {
    let (router, _dir) = app(512);
    for (extra, param) in [
        (json!({"tool_choice": "required"}), "tool_choice"),
        (
            json!({"tools": weather_tools(),
                   "tool_choice": {"type": "function", "function": {"name": "nope"}}}),
            "tool_choice",
        ),
        (
            json!({"tools": weather_tools(), "tool_choice": "required",
                   "response_format": {"type": "json_object"}}),
            "response_format",
        ),
        (
            json!({"tools": weather_tools(), "response_format": {"type": "json_object"}}),
            "response_format",
        ),
    ] {
        let reply = post(&router, "/v1/chat/completions", &chat(extra.clone())).await;
        assert_eq!(
            reply.status,
            StatusCode::BAD_REQUEST,
            "{extra}: {}",
            reply.text()
        );
        assert_eq!(reply.json()["error"]["param"], param, "{extra}");
    }
    // `none` offers no tools: the model answers in text.
    let reply = ok(
        &router,
        chat(json!({"tools": weather_tools(), "tool_choice": "none", "max_tokens": 6})),
    )
    .await;
    assert!(reply["choices"][0]["message"]["tool_calls"].is_null());
}

#[tokio::test(flavor = "multi_thread")]
async fn messages_any_and_tool_force_a_tool_use_block() {
    let (router, _dir) = app(1024);
    let tools = json!([{"name": "get_time", "description": "time",
        "input_schema": {"type": "object", "properties": {"zone": {"enum": ["utc", "pst"]}},
                         "required": ["zone"], "additionalProperties": false}}]);
    for choice in [
        json!({"type": "any"}),
        json!({"type": "tool", "name": "get_time"}),
    ] {
        let reply = post(
            &router,
            "/v1/messages",
            &json!({"model": MODEL_ID, "max_tokens": 300, "temperature": 1.0,
                    "tools": tools, "tool_choice": choice,
                    "messages": [{"role": "user", "content": "time?"}]}),
        )
        .await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
        let body = reply.json();
        assert_eq!(body["stop_reason"], "tool_use", "{body}");
        let block = body["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["type"] == "tool_use")
            .unwrap_or_else(|| panic!("no tool_use block: {body}"));
        assert_eq!(block["name"], "get_time");
        assert!(["utc", "pst"].contains(&block["input"]["zone"].as_str().unwrap()));
        assert!(block["id"].as_str().unwrap().starts_with("toolu_"));
    }
}
