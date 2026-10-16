//! The server as a client sees it: JSON-RPC requests through the real request
//! handling, answers checked against what the MCP specification requires.

mod common;

use common::{Harness, train_args};
use serde_json::{Value, json};

fn tools(list: &Value) -> &Vec<Value> {
    list["tools"].as_array().expect("tools/list returns tools")
}

#[tokio::test]
async fn initialize_names_the_crate_version_and_says_how_to_use_the_server() {
    let h = Harness::new("exit 0");
    let result = h
        .result(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"}
            }),
        )
        .await;
    assert_eq!(result["serverInfo"]["name"], "pmetal");
    // Not a literal that drifts from the release.
    assert_eq!(result["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));
    let instructions = result["instructions"].as_str().expect("instructions");
    for must_mention in ["job_status", "job_logs", "stop_job", "PMETAL_MCP_ROOTS"] {
        assert!(instructions.contains(must_mention), "{must_mention}");
    }
    let caps = &result["capabilities"];
    assert_eq!(caps["tools"]["listChanged"], json!(false));
    assert!(caps["resources"].is_object() && caps["prompts"].is_object());
    assert!(caps["logging"].is_object());
}

#[tokio::test]
async fn every_tool_states_every_hint_a_client_needs() {
    let h = Harness::new("exit 0");
    let list = h.result("tools/list", json!({})).await;
    let tools = tools(&list);
    assert!(tools.len() >= 50, "only {} tools", tools.len());
    let mut names = std::collections::BTreeSet::new();
    for tool in tools {
        let name = tool["name"].as_str().unwrap();
        assert!(names.insert(name), "{name} is listed twice");
        assert!(
            tool["description"].as_str().is_some_and(|d| d.len() > 20),
            "{name} has no description"
        );
        let a = &tool["annotations"];
        for hint in ["readOnlyHint", "idempotentHint", "openWorldHint"] {
            assert!(a[hint].is_boolean(), "{name} does not state {hint}");
        }
        // Destructive only means something for a tool that changes things.
        if a["readOnlyHint"] == json!(false) {
            assert!(a["destructiveHint"].is_boolean(), "{name}: destructiveHint");
        }
    }
}

#[tokio::test]
async fn every_input_schema_is_a_closed_object_whose_parameters_are_described() {
    let h = Harness::new("exit 0");
    let list = h.result("tools/list", json!({})).await;
    for tool in tools(&list) {
        let name = tool["name"].as_str().unwrap();
        let schema = &tool["inputSchema"];
        assert_eq!(schema["type"], "object", "{name}");
        // A parameter the tool does not take is refused, not dropped.
        assert_eq!(schema["additionalProperties"], json!(false), "{name}");
        let properties = schema["properties"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        for (param, definition) in &properties {
            let described = definition["description"]
                .as_str()
                .is_some_and(|d| !d.trim().is_empty());
            assert!(described, "{name}.{param} has no description");
        }
        for required in schema["required"].as_array().into_iter().flatten() {
            let required = required.as_str().unwrap();
            assert!(
                properties.contains_key(required),
                "{name} requires {required}"
            );
        }
    }
}

#[tokio::test]
async fn tools_that_start_jobs_declare_the_shape_of_what_they_return() {
    let h = Harness::new("exit 0");
    let list = h.result("tools/list", json!({})).await;
    let train = tools(&list).iter().find(|t| t["name"] == "train").unwrap();
    let output = &train["outputSchema"];
    assert_eq!(output["type"], "object");
    assert!(
        output["required"]
            .as_array()
            .unwrap()
            .contains(&json!("job_id"))
    );
    for name in ["job_status", "list_jobs", "job_logs"] {
        let _ = tools(&list).iter().find(|t| t["name"] == name).unwrap();
    }
}

#[tokio::test]
async fn what_the_job_tools_return_conforms_to_the_schema_they_declare() {
    use pmetal_mcp::schema_check::check_value;

    let h = Harness::new("echo hi; echo oops >&2; exit 4");
    let list = h.result("tools/list", json!({})).await;
    let declared = |name: &str| -> Value {
        tools(&list)
            .iter()
            .find(|t| t["name"] == name)
            .unwrap_or_else(|| panic!("{name}"))["outputSchema"]
            .clone()
    };
    let conforms = |name: &str, content: &Value| {
        let schema = declared(name);
        assert!(schema.is_object(), "{name} declares no output schema");
        check_value(name, content, &schema).unwrap_or_else(|e| panic!("{name}: {e}\n{content}"));
    };

    let started = h.call_ok("train", train_args(&h)).await;
    conforms("train", &started);
    let id = started["job_id"].as_str().unwrap();
    let failed = h
        .wait_for_job(id, |s| s["status"]["state"] == "failed")
        .await;
    conforms("job_status", &failed);
    conforms("list_jobs", &h.call_ok("list_jobs", json!({})).await);
    conforms(
        "job_logs",
        &h.call_ok("job_logs", json!({"job_id": id})).await,
    );

    // The declaration says something: a result of the wrong shape does not conform.
    assert!(check_value("job_status", &json!({"id": 5}), &declared("job_status")).is_err());
    let mut wrong = failed.clone();
    wrong["status"] = json!({"state": "exploded"});
    assert!(check_value("job_status", &wrong, &declared("job_status")).is_err());
}

#[tokio::test]
async fn an_unknown_tool_is_a_protocol_error() {
    let h = Harness::new("exit 0");
    let response = h
        .request(
            "tools/call",
            json!({"name": "no_such_tool", "arguments": {}}),
        )
        .await;
    assert_eq!(response["error"]["code"], json!(-32602), "{response}");
}

#[tokio::test]
async fn a_tool_that_fails_returns_a_result_the_model_can_read() {
    let h = Harness::new("exit 0");
    // Missing required parameters: the model should see which, and retry.
    let result = h.call("train", json!({})).await;
    assert_eq!(result["isError"], json!(true), "{result}");
    let message = result["content"][0]["text"].as_str().unwrap();
    assert!(
        message.contains("model") && message.contains("dataset"),
        "{message}"
    );
    assert_eq!(
        result["_meta"]["io.turbomcp/errorKind"], "invalid_params",
        "{result}"
    );
}

#[tokio::test]
async fn a_parameter_the_tool_does_not_take_is_refused_by_name() {
    let h = Harness::new("exit 0");
    let mut args = train_args(&h);
    args["output"] = json!("old-name");
    let message = h.call_err("train", args).await;
    assert!(message.contains("unknown parameter `output`"), "{message}");
    // And it says what the tool does take.
    assert!(message.contains("output_dir"), "{message}");
}

#[tokio::test]
async fn bad_values_say_which_parameter_and_what_was_wrong() {
    let h = Harness::new("exit 0");
    let mut args = train_args(&h);
    args["batch_size"] = json!("four");
    assert!(
        h.call_err("train", args.clone())
            .await
            .contains("`batch_size` must be an integer")
    );
    args["batch_size"] = json!(0);
    assert!(
        h.call_err("train", args.clone())
            .await
            .contains("`batch_size` must be at least 1")
    );
    args["batch_size"] = json!(2);
    args["optimizer"] = json!("sgdd");
    let message = h.call_err("train", args).await;
    assert!(message.contains("`optimizer` must be one of"), "{message}");
    assert!(message.contains("adamw"), "{message}");
}

#[tokio::test]
async fn native_tools_are_held_to_their_schemas_too() {
    let h = Harness::new("exit 0");
    for (tool, args, needle) in [
        (
            "search_models",
            json!({"query": "qwen", "sort": "newest"}),
            "`sort` must be one of downloads, likes, trending, updated",
        ),
        (
            "search_models",
            json!({"query": "qwen", "limit": 500}),
            "`limit` must be at most 100",
        ),
        (
            "search_models",
            json!({"limit": 5}),
            "missing required parameter: query",
        ),
        (
            "search_models",
            json!({"query": "q", "tags": "gguf"}),
            "`tags` must be an array",
        ),
        (
            "model_fit",
            json!({"model": "a/b", "context_length": 0}),
            "`context_length` must be at least 1",
        ),
        (
            "job_logs",
            json!({"job_id": "x", "tail": 0}),
            "`tail` must be at least 1",
        ),
        (
            "job_status",
            json!({"job_id": 7}),
            "`job_id` must be a string",
        ),
        (
            "device_info",
            json!({"verbose": true}),
            "unknown parameter `verbose`",
        ),
    ] {
        let message = h.call_err(tool, args).await;
        assert!(message.contains(needle), "{tool}: {message}");
    }
}

#[tokio::test]
async fn a_started_job_returns_structured_content_and_the_same_text() {
    let h = Harness::new("sleep 30");
    let result = h.call("train", train_args(&h)).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    let structured = &result["structuredContent"];
    assert_eq!(structured["command"], "train");
    assert_eq!(structured["status"], "running");
    let id = structured["job_id"].as_str().unwrap();
    // The text block carries the same JSON for clients without structured content.
    let text: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(&text, structured);
    h.call_ok("stop_job", json!({"job_id": id})).await;
}

#[tokio::test]
async fn resources_expose_jobs_and_prompts_guide_workflows() {
    let h = Harness::new("echo hello; echo oops >&2");
    let started = h.call_ok("train", train_args(&h)).await;
    let id = started["job_id"].as_str().unwrap().to_string();
    h.wait_for_job(&id, |s| s["status"]["state"] == "completed")
        .await;

    let resources = h.result("resources/list", json!({})).await;
    let uris: Vec<_> = resources["resources"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["uri"].as_str().unwrap().to_string())
        .collect();
    assert!(
        uris.contains(&"pmetal://jobs".to_string())
            && uris.contains(&"pmetal://device".to_string())
    );
    let templates = h.result("resources/templates/list", json!({})).await;
    assert_eq!(templates["resourceTemplates"].as_array().unwrap().len(), 2);

    let job = h
        .result(
            "resources/read",
            json!({"uri": format!("pmetal://jobs/{id}")}),
        )
        .await;
    let body: Value = serde_json::from_str(job["contents"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(body["id"], json!(id));
    let logs = h
        .result(
            "resources/read",
            json!({"uri": format!("pmetal://jobs/{id}/logs")}),
        )
        .await;
    let text = logs["contents"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("hello") && text.contains("[stderr] oops"),
        "{text}"
    );
    let missing = h
        .request("resources/read", json!({"uri": "pmetal://jobs/nope"}))
        .await;
    assert_eq!(missing["error"]["code"], json!(-32002), "{missing}");

    let prompts = h.result("prompts/list", json!({})).await;
    assert!(prompts["prompts"].as_array().unwrap().len() >= 3);
    let prompt = h
        .result(
            "prompts/get",
            json!({"name": "finetune", "arguments": {"model": "Qwen/Qwen3-0.6B", "dataset": "d.jsonl"}}),
        )
        .await;
    let text = prompt["messages"][0]["content"]["text"].as_str().unwrap();
    assert!(
        text.contains("model_fit") && text.contains("Qwen/Qwen3-0.6B"),
        "{text}"
    );
    // A prompt cannot name a tool that does not exist.
    let list = h.result("tools/list", json!({})).await;
    let names: Vec<&str> = tools(&list)
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for word in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if (word.starts_with("job_") || word.starts_with("dataset_")) && word != "job_id" {
            assert!(names.contains(&word), "the finetune prompt names {word}");
        }
    }
}
