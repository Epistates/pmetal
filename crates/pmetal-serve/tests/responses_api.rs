//! `/v1/responses` against OpenAI's Responses API: the response object, its
//! output items, the streaming events and their order, `incomplete`, function
//! tools, `text.format`, and what is refused.

mod support;

use axum::http::StatusCode;
use serde_json::{Value, json};
use support::*;

fn request(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL_ID,
        "input": "hi",
        "max_output_tokens": 10,
        "temperature": 0,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

async fn ok(router: &axum::Router, body: Value) -> Value {
    let reply = post(router, "/v1/responses", &body).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    reply.json()
}

fn output_text(response: &Value) -> String {
    response["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().unwrap().clone())
        .filter_map(|part| part["text"].as_str().map(str::to_string))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_response_has_openai_s_shape() {
    let (router, _dir) = app(256);
    let response = ok(&router, request(json!({"instructions": "be brief"}))).await;
    assert_eq!(response["object"], "response");
    assert!(response["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(response["model"], MODEL_ID);
    // Stopped by its budget, the response says so.
    assert_eq!(response["status"], "incomplete");
    assert_eq!(
        response["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(response["instructions"], "be brief");
    let item = &response["output"][0];
    assert_eq!(item["type"], "message");
    assert_eq!(item["role"], "assistant");
    assert_eq!(item["status"], "completed");
    assert!(item["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(item["content"][0]["type"], "output_text");
    assert!(item["content"][0]["annotations"].is_array());
    let usage = &response["usage"];
    assert_eq!(usage["output_tokens"], 10);
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["input_tokens"].as_u64().unwrap() + 10
    );
    assert!(usage["input_tokens_details"]["cached_tokens"].is_u64());
    assert!(usage["output_tokens_details"]["reasoning_tokens"].is_u64());
    assert_no_unobserved_error();

    // Finishing on its own, it is completed.
    let done = ok(
        &router,
        request(json!({"max_output_tokens": 200, "stop": null})),
    )
    .await;
    assert!(["completed", "incomplete"].contains(&done["status"].as_str().unwrap()));
}

#[tokio::test(flavor = "multi_thread")]
async fn input_is_a_string_or_items_and_means_the_same_conversation() {
    let (router, _dir) = app(512);
    let by_string = ok(&router, request(json!({"input": "hi"}))).await;
    let by_items = ok(
        &router,
        request(json!({"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]})),
    )
    .await;
    assert_eq!(output_text(&by_string), output_text(&by_items));
    assert_eq!(
        by_string["usage"]["input_tokens"],
        by_items["usage"]["input_tokens"]
    );

    // A whole exchange, with an earlier tool call and its result.
    let reply = ok(
        &router,
        request(json!({"input": [
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": "weather?"},
            {"type": "reasoning", "id": "rs_1", "summary": [],
             "content": [{"type": "reasoning_text", "text": "I should check."}]},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather",
             "arguments": "{\"city\": \"SF\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "72F"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "72F."}]},
            {"role": "user", "content": "and tomorrow?"},
        ]})),
    )
    .await;
    assert_eq!(reply["object"], "response");
    assert_no_unobserved_error();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stream_is_numbered_events_in_openai_s_order() {
    let (router, _dir) = app(256);
    let whole = ok(&router, request(json!({}))).await;
    let reply = post(&router, "/v1/responses", &request(json!({"stream": true}))).await;
    assert_eq!(reply.status, StatusCode::OK);
    let events = reply.sse();
    assert!(
        !reply.text().contains("[DONE]"),
        "a Responses stream has no [DONE]"
    );
    let names: Vec<&str> = events
        .iter()
        .map(|(name, _)| name.as_deref().expect("every frame is named"))
        .collect();
    assert_eq!(
        &names[..5],
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta"
        ]
    );
    let tail: Vec<&str> = names[names.len() - 4..].to_vec();
    assert_eq!(
        tail,
        [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.incomplete"
        ]
    );
    // Each frame's `type` is its event name, and sequence numbers count from 0.
    let payloads = reply.sse_json();
    for (i, (name, payload)) in events.iter().zip(&payloads).enumerate() {
        assert_eq!(payload["type"], name.0.as_deref().unwrap_or_default());
        assert_eq!(payload["sequence_number"], i);
    }
    let streamed: String = payloads
        .iter()
        .filter(|e| e["type"] == "response.output_text.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(streamed, output_text(&whole));
    let last = payloads.last().unwrap();
    assert_eq!(last["response"]["status"], "incomplete");
    assert_eq!(output_text(&last["response"]), streamed);
    assert_eq!(last["response"]["usage"]["output_tokens"], 10);
    let done = payloads
        .iter()
        .find(|e| e["type"] == "response.output_text.done")
        .unwrap();
    assert_eq!(done["text"], streamed);
}

fn weather_tool() -> Value {
    json!({"type": "function", "name": "get_weather", "description": "weather",
           "strict": true,
           "parameters": {"type": "object",
               "properties": {"city": {"type": "string", "maxLength": 8}},
               "required": ["city"], "additionalProperties": false}})
}

#[tokio::test(flavor = "multi_thread")]
async fn a_required_tool_choice_gives_a_function_call_item() {
    let (router, _dir) = app(1024);
    let response = ok(
        &router,
        request(json!({"tools": [weather_tool()], "tool_choice": "required",
                       "max_output_tokens": 300, "temperature": 1.0, "seed": 4,
                       "parallel_tool_calls": false})),
    )
    .await;
    assert_eq!(response["status"], "completed", "{response}");
    let call = &response["output"][0];
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["name"], "get_weather");
    assert_eq!(call["status"], "completed");
    assert!(call["call_id"].as_str().unwrap().starts_with("call_"));
    assert!(call["id"].as_str().unwrap().starts_with("fc_"));
    let arguments: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
    assert!(arguments["city"].is_string());
    assert_eq!(response["output"].as_array().unwrap().len(), 1);

    // Streamed: the call's events, in order.
    let reply = post(
        &router,
        "/v1/responses",
        &request(json!({"tools": [weather_tool()], "tool_choice": "required",
                        "max_output_tokens": 300, "temperature": 1.0, "seed": 4,
                        "parallel_tool_calls": false, "stream": true})),
    )
    .await;
    let names: Vec<String> = reply
        .sse()
        .into_iter()
        .map(|(name, _)| name.unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    let payloads = reply.sse_json();
    // The deltas add up to the arguments of the call the stream ends on.
    assert_eq!(
        payloads[3]["delta"],
        payloads[6]["response"]["output"][0]["arguments"]
    );
    assert_eq!(payloads[4]["arguments"], payloads[3]["delta"]);
    serde_json::from_str::<Value>(payloads[3]["delta"].as_str().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn text_format_json_schema_is_enforced() {
    let (router, _dir) = app(512);
    let schema = json!({"type": "object", "properties": {"ok": {"enum": [true, false]}},
                        "required": ["ok"], "additionalProperties": false});
    let response = ok(
        &router,
        request(
            json!({"max_output_tokens": 80, "temperature": 1.0, "seed": 8,
            "text": {"format": {"type": "json_schema", "name": "r", "strict": true,
                                "schema": schema}}}),
        ),
    )
    .await;
    let text = output_text(&response);
    let value: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
    assert!(value["ok"].is_boolean());
    assert_eq!(response["text"]["format"]["type"], "json_schema");
}

#[tokio::test(flavor = "multi_thread")]
async fn what_the_server_cannot_do_is_refused_by_name() {
    let (router, _dir) = app(256);
    for (extra, param) in [
        (
            json!({"previous_response_id": "resp_1"}),
            "previous_response_id",
        ),
        (json!({"conversation": "conv_1"}), "conversation"),
        (json!({"background": true}), "background"),
        (
            json!({"tools": [{"type": "web_search_preview"}]}),
            "tools.0.type",
        ),
        (
            json!({"include": ["message.output_text.logprobs"]}),
            "include",
        ),
        (
            json!({"input": [{"type": "web_search_call", "id": "ws_1"}]}),
            "input.0.type",
        ),
        (
            json!({"input": [{"role": "user", "content": [{"type": "input_file"}]}]}),
            "input.0.content.0.type",
        ),
        (
            json!({"tool_choice": "sometimes", "tools": [weather_tool()]}),
            "tool_choice",
        ),
        (
            json!({"text": {"format": {"type": "markdown"}}}),
            "text.format.type",
        ),
    ] {
        let reply = post(&router, "/v1/responses", &request(extra.clone())).await;
        assert_eq!(
            reply.status,
            StatusCode::BAD_REQUEST,
            "{extra}: {}",
            reply.text()
        );
        assert_eq!(reply.json()["error"]["param"], param, "{extra}");
    }
    // The model must exist.
    let reply = post(
        &router,
        "/v1/responses",
        &request(json!({"model": "gpt-9"})),
    )
    .await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
}
