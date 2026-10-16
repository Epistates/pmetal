//! What the tools let a client reach: paths stay under the allowed roots, an
//! inference server stays on this machine, and model references are Hub IDs or
//! checked paths.

mod common;

use common::{Harness, train_args};
use serde_json::json;

#[tokio::test]
async fn an_output_directory_outside_the_roots_is_refused_before_anything_runs() {
    let h = Harness::new("touch \"$(dirname \"$0\")/ran\"");
    for output in ["/etc/pmetal-out", "../../escape", "/tmp/../etc/x", "~/.ssh"] {
        let mut args = train_args(&h);
        args["output_dir"] = json!(output);
        let message = h.call_err("train", args).await;
        assert!(
            message.contains("outside the directories"),
            "{output}: {message}"
        );
        assert!(message.contains("PMETAL_MCP_ROOTS"), "{message}");
    }
    assert!(!h.dir.join("ran").exists(), "a job was started anyway");
}

#[tokio::test]
async fn a_dataset_or_adapter_outside_the_roots_is_refused() {
    let h = Harness::new("exit 0");
    let mut args = train_args(&h);
    args["dataset"] = json!("/etc/passwd");
    assert!(h.call_err("train", args).await.contains("dataset"));

    let message = h
        .call_err(
            "generate",
            json!({"model": "Qwen/Qwen3-0.6B", "prompt": "hi", "lora": "/Users/someone/adapter"}),
        )
        .await;
    assert!(
        message.contains("lora") && message.contains("outside"),
        "{message}"
    );
}

#[tokio::test]
async fn the_default_output_directory_is_checked_too() {
    // The working directory is not an allowed root here, so the default
    // `./output` must be refused rather than written wherever the server runs.
    let h = Harness::with_cwd_outside_the_roots("exit 0");
    let message = h
        .call_err(
            "train",
            json!({"model": "Qwen/Qwen3-0.6B", "dataset": h.path("d.jsonl")}),
        )
        .await;
    assert!(message.contains("output_dir"), "{message}");
}

#[tokio::test]
async fn paths_the_client_gives_become_absolute_and_normalized_in_the_command() {
    let h = Harness::new("sleep 5");
    let mut args = train_args(&h);
    args["output_dir"] = json!(format!("{}/a/../runs/one", h.dir.display()));
    let started = h.call_ok("train", args).await;
    assert_eq!(started["output"], json!(h.path("runs/one")));
    let id = started["job_id"].as_str().unwrap();
    let status = h.call_ok("job_status", json!({"job_id": id})).await;
    let argv: Vec<&str> = status["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect();
    let at = argv.iter().position(|a| *a == "--output").unwrap();
    assert_eq!(argv[at + 1], h.path("runs/one"));
    h.call_ok("stop_job", json!({"job_id": id})).await;
}

#[tokio::test]
async fn a_symlink_out_of_the_root_does_not_open_the_door() {
    let h = Harness::new("exit 0");
    let outside = common::scratch_dir("outside");
    std::os::unix::fs::symlink(&outside, h.dir.join("link")).unwrap();
    let mut args = train_args(&h);
    args["output_dir"] = json!(h.path("link/run"));
    assert!(
        h.call_err("train", args)
            .await
            .contains("outside the directories")
    );
}

#[tokio::test]
async fn model_references_must_be_hub_ids_or_paths() {
    let h = Harness::new("exit 0");
    for bad in ["a/b/c", "owner/..", "x;rm -rf /", "-rf", "owner/na me"] {
        let mut args = train_args(&h);
        args["model"] = json!(bad);
        let message = h.call_err("train", args).await;
        assert!(message.contains("model"), "{bad}: {message}");
    }
}

#[tokio::test]
async fn pretraining_shard_globs_are_checked_entry_by_entry() {
    let h = Harness::new("exit 0");
    let message = h
        .call_err(
            "pretrain",
            json!({
                "arch": "gpt-oss",
                "shards": format!("{}/shards/*.bin,/etc/*.bin", h.dir.display()),
                "output_dir": h.path("out"),
            }),
        )
        .await;
    assert!(
        message.contains("shards") && message.contains("/etc"),
        "{message}"
    );
}

#[tokio::test]
async fn an_inference_server_binds_to_this_machine_unless_the_operator_allows_more() {
    let h = Harness::new("sleep 5");
    let started = h
        .call_ok(
            "start_serve",
            json!({"model": "Qwen/Qwen3-0.6B", "port": 18081}),
        )
        .await;
    let id = started["job_id"].as_str().unwrap().to_string();
    let status = h.call_ok("job_status", json!({"job_id": id})).await;
    let args = status["args"].to_string();
    assert!(args.contains("127.0.0.1"), "{args}");
    h.call_ok("stop_job", json!({"job_id": id})).await;

    for host in ["0.0.0.0", "192.168.1.5", "example.com", "::"] {
        let message = h
            .call_err(
                "start_serve",
                json!({"model": "Qwen/Qwen3-0.6B", "host": host}),
            )
            .await;
        assert!(message.contains("unauthenticated"), "{host}: {message}");
    }
    for host in ["localhost", "127.0.0.1", "127.0.0.2", "::1"] {
        let started = h
            .call_ok(
                "start_serve",
                json!({"model": "Qwen/Qwen3-0.6B", "host": host}),
            )
            .await;
        h.call_ok("stop_job", json!({"job_id": started["job_id"]}))
            .await;
    }
    h.server.shutdown().await;
}
