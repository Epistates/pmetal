//! The dataset and Ollama tools: the command lines they build.

mod common;

use common::Harness;
use pmetal_mcp::cli_tools::CLI_TOOLS;
use serde_json::json;

/// The command line a tool ran: the fake `pmetal` prints its arguments, one per line.
fn harness() -> Harness {
    Harness::new(r#"for a in "$@"; do printf '%s\n' "$a"; done"#)
}

async fn argv(h: &Harness, tool: &str, args: serde_json::Value) -> Vec<String> {
    let result = h.call(tool, args).await;
    assert_ne!(result["isError"], json!(true), "{tool}: {result}");
    result["content"][0]["text"]
        .as_str()
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

#[tokio::test]
async fn positional_arguments_come_first_and_flags_carry_their_value_in_one_token() {
    let h = harness();
    let got = argv(
        &h,
        "dataset_download",
        json!({"dataset_id": "tatsu-lab/alpaca", "split": "test", "output": h.path("a.jsonl")}),
    )
    .await;
    assert_eq!(
        got,
        [
            "dataset".to_string(),
            "download".into(),
            "tatsu-lab/alpaca".into(),
            "--split=test".into(),
            format!("--output={}", h.path("a.jsonl")),
        ]
    );
}

#[tokio::test]
async fn split_and_prepare_use_the_flags_the_cli_defines() {
    let h = harness();
    let got = argv(
        &h,
        "dataset_split",
        json!({"input": h.path("in.jsonl"), "output_dir": h.path("splits"), "val_ratio": 0.2}),
    )
    .await;
    assert!(
        got.contains(&format!("--output-dir={}", h.path("splits"))),
        "{got:?}"
    );
    assert!(got.contains(&"--val-ratio=0.2".to_string()), "{got:?}");

    let got = argv(
        &h,
        "dataset_prepare",
        json!({"dataset": "tatsu-lab/alpaca", "output_dir": h.path("o"), "model": "Qwen/Qwen3-0.6B"}),
    )
    .await;
    assert_eq!(&got[..3], ["dataset", "prepare", "tatsu-lab/alpaca"]);
    assert!(
        got.contains(&"--model=Qwen/Qwen3-0.6B".to_string()),
        "{got:?}"
    );
}

#[tokio::test]
async fn merging_takes_a_list_of_inputs() {
    let h = harness();
    let got = argv(
        &h,
        "dataset_merge",
        json!({"inputs": [h.path("a.jsonl"), h.path("b.jsonl")], "output": h.path("m.jsonl"), "shuffle": true}),
    )
    .await;
    let at = got.iter().position(|a| a == "--inputs").unwrap();
    assert_eq!(got[at + 1], h.path("a.jsonl"));
    assert_eq!(got[at + 2], h.path("b.jsonl"));
    assert!(got.contains(&"--shuffle".to_string()));
    // A comma-joined string is not a list of paths.
    let message = h
        .call_err(
            "dataset_merge",
            json!({"inputs": format!("{},{}", h.path("a"), h.path("b")), "output": h.path("m")}),
        )
        .await;
    assert!(message.contains("inputs"), "{message}");
}

#[tokio::test]
async fn a_value_that_starts_with_a_dash_stays_a_value() {
    let h = harness();
    let got = argv(
        &h,
        "ollama_modelfile",
        json!({"base": "llama3", "system": "--output /etc/passwd"}),
    )
    .await;
    assert!(
        got.contains(&"--system=--output /etc/passwd".to_string()),
        "{got:?}"
    );
    // A positional cannot be turned into a flag.
    let message = h
        .call_err("dataset_download", json!({"dataset_id": "--output"}))
        .await;
    assert!(message.contains("dataset_id"), "{message}");
}

#[tokio::test]
async fn paths_are_checked_and_options_are_closed() {
    let h = harness();
    let message = h
        .call_err(
            "dataset_filter",
            json!({"input": h.path("in.jsonl"), "output": "/etc/shadow-copy"}),
        )
        .await;
    assert!(message.contains("outside the directories"), "{message}");
    let message = h
        .call_err(
            "dataset_analyze",
            json!({"path": h.path("a"), "verbose": true}),
        )
        .await;
    assert!(message.contains("unknown parameter `verbose`"), "{message}");
    let message = h
        .call_err(
            "dataset_convert",
            json!({"input": h.path("a"), "output": h.path("b"), "format": "xml"}),
        )
        .await;
    assert!(message.contains("one of"), "{message}");
    let message = h
        .call_err(
            "dataset_sample",
            json!({"input": h.path("a"), "output": h.path("b")}),
        )
        .await;
    assert!(
        message.contains("missing required parameter: num"),
        "{message}"
    );
}

#[test]
fn every_parameter_is_documented_and_every_required_one_is_listed() {
    let h = harness();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let list = rt.block_on(h.result("tools/list", json!({})));
    for cli in CLI_TOOLS {
        let tool = list["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == cli.name)
            .unwrap_or_else(|| panic!("{} is not listed", cli.name));
        let schema = &tool["inputSchema"];
        for p in cli.params {
            assert!(
                schema["properties"][p.name]["description"].is_string(),
                "{}.{}",
                cli.name,
                p.name
            );
            let required = schema["required"]
                .as_array()
                .is_some_and(|r| r.contains(&json!(p.name)));
            assert_eq!(required, p.required, "{}.{}", cli.name, p.name);
        }
        // Each example is a command line the server itself would build.
        assert!(!cli.example_invocations(&h.dir).is_empty());
    }
}
