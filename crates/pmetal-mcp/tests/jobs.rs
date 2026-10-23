//! Jobs: starting, following, steering, stopping and cleaning up after real
//! child processes (a script stands in for `pmetal`).

mod common;

use std::time::Duration;

use common::{Harness, train_args};
use pmetal_mcp::jobs::JobLimits;
use serde_json::{Value, json};

fn state(status: &Value) -> &str {
    status["status"]["state"].as_str().unwrap()
}

async fn start(h: &Harness) -> String {
    let started = h.call_ok("train", train_args(h)).await;
    started["job_id"].as_str().unwrap().to_string()
}

fn process_alive(pid: &str) -> bool {
    std::process::Command::new("kill")
        .args(["-0", pid.trim()])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Wait up to two seconds for `pid` to be gone.
async fn gone(pid: &str) -> bool {
    for _ in 0..40 {
        if !process_alive(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

#[tokio::test]
async fn a_job_runs_to_completion_and_keeps_its_output_in_order() {
    let h = Harness::new(
        "echo one; echo err1 >&2; sleep 0.2; echo two; echo err2 >&2; sleep 0.2; echo three",
    );
    let id = start(&h).await;
    let status = h.wait_for_job(&id, |s| state(s) == "completed").await;
    assert_eq!(status["status"]["exit_code"], 0);
    assert!(status["finished_at"].is_string());
    let logs = h.call_ok("job_logs", json!({"job_id": id})).await;
    // Both streams, in the order they were printed.
    assert_eq!(
        logs["lines"],
        json!(["one", "[stderr] err1", "two", "[stderr] err2", "three"])
    );
    assert_eq!(logs["total_lines"], 5);
}

#[tokio::test]
async fn a_failing_job_reports_the_last_line_it_wrote_to_stderr() {
    let h = Harness::new("echo starting; echo 'model not found' >&2; exit 3");
    let id = start(&h).await;
    let status = h.wait_for_job(&id, |s| state(s) == "failed").await;
    assert_eq!(status["status"]["exit_code"], 3);
    assert_eq!(status["status"]["error"], "model not found");
}

#[tokio::test]
async fn logs_can_be_read_incrementally() {
    let h = Harness::new("echo a; echo b; sleep 0.3; echo c; echo d; sleep 5");
    let id = start(&h).await;
    h.wait_for_job(&id, |_| true).await;
    for _ in 0..100 {
        let logs = h.call_ok("job_logs", json!({"job_id": id})).await;
        if logs["total_lines"] == 4 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let first = h
        .call_ok("job_logs", json!({"job_id": id, "tail": 2}))
        .await;
    assert_eq!(first["lines"], json!(["c", "d"]));
    let since = first["next_since"].as_u64().unwrap();
    let nothing_new = h
        .call_ok("job_logs", json!({"job_id": id, "since": since}))
        .await;
    assert_eq!(nothing_new["lines"], json!([]));
    let from_two = h
        .call_ok("job_logs", json!({"job_id": id, "since": 2}))
        .await;
    assert_eq!(from_two["lines"], json!(["c", "d"]));
    h.call_ok("stop_job", json!({"job_id": id})).await;
}

#[tokio::test]
async fn memory_stays_bounded_for_a_job_that_never_stops_printing() {
    // A megabyte with no newline, then enough lines to overflow the ring.
    let h = Harness::new(
        "head -c 1048576 /dev/zero | tr '\\0' x; echo; i=0; while [ $i -lt 12000 ]; do echo \"line $i\"; i=$((i+1)); done",
    );
    let id = start(&h).await;
    h.wait_for_job(&id, |s| state(s) == "completed").await;
    let logs = h
        .call_ok("job_logs", json!({"job_id": id, "tail": 10000}))
        .await;
    let lines = logs["lines"].as_array().unwrap();
    assert!(lines.len() <= 10_000);
    // The long line was cut where it was kept, and the oldest lines were dropped.
    assert!(lines.iter().all(|l| l.as_str().unwrap().len() < 20_000));
    assert_eq!(lines.last().unwrap(), "line 11999");
    assert_eq!(logs["total_lines"], 12_001);
}

#[tokio::test]
async fn a_token_in_a_jobs_output_never_reaches_the_model() {
    let h = Harness::new("echo 'fetching with hf_AbCdEfGhIjKlMnOpQrStUvWx'; echo done");
    let id = start(&h).await;
    h.wait_for_job(&id, |s| state(s) == "completed").await;
    let logs = h.call_ok("job_logs", json!({"job_id": id})).await;
    let text = logs.to_string();
    assert!(!text.contains("AbCdEfGh"), "{text}");
    assert!(text.contains("hf_[REDACTED]"), "{text}");
}

#[tokio::test]
async fn metrics_come_from_the_file_the_server_asked_the_job_to_write() {
    // The fake trainer writes steps to the path it was given as --log-metrics,
    // as the real one does besides its --log-events. The steps are read from the
    // events: they are not counted a second time from the metrics file.
    let h = Harness::new(
        r#"while [ $# -gt 0 ]; do [ "$1" = "--log-metrics" ] && f="$2"; [ "$1" = "--log-events" ] && e="$2"; shift; done
echo '{"event":"train_start"}' > "$f"
echo '{"step":1,"epoch":0,"total_epochs":2,"total_steps":40,"loss":3.0,"lr":0.0002,"tok_sec":900.0}' >> "$f"
echo '{"step":2,"epoch":0,"total_epochs":2,"total_steps":40,"loss":2.5,"lr":0.0002,"tok_sec":910.0}' >> "$f"
cp "$f" "$e"
sleep 5"#,
    );
    let id = start(&h).await;
    let status = h
        .wait_for_job(&id, |s| s["metrics"]["last_step"] == 2)
        .await;
    let metrics = &status["metrics"];
    assert_eq!(metrics["last_loss"], 2.5);
    assert_eq!(metrics["total_steps"], 40);
    assert_eq!(metrics["epoch"], 1.0);
    assert_eq!(metrics["total_epochs"], 2);
    // The file is where the TUI looks too: in the job's output directory.
    let args: Vec<&str> = status["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect();
    let at = args.iter().position(|a| *a == "--log-metrics").unwrap();
    assert_eq!(args[at + 1], h.path("out/metrics.jsonl"));
    let at = args.iter().position(|a| *a == "--log-events").unwrap();
    assert_eq!(args[at + 1], h.path("out/events.jsonl"));
    h.call_ok("stop_job", json!({"job_id": id})).await;
}

#[tokio::test]
async fn stopping_a_job_ends_what_it_started_too() {
    let h = Harness::new("sleep 300 &\necho $! > \"$(dirname \"$0\")/grandchild.pid\"\nwait");
    let id = start(&h).await;
    let pid_file = h.dir.join("grandchild.pid");
    for _ in 0..100 {
        if pid_file.exists()
            && !std::fs::read_to_string(&pid_file)
                .unwrap()
                .trim()
                .is_empty()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let grandchild = std::fs::read_to_string(&pid_file).unwrap();
    assert!(process_alive(&grandchild), "the helper never started");

    h.call_ok("stop_job", json!({"job_id": id})).await;
    let status = h.wait_for_job(&id, |s| state(s) == "stopped").await;
    assert!(status["finished_at"].is_string());
    assert!(gone(&grandchild).await, "the helper outlived the job");
}

#[tokio::test]
async fn a_job_that_ignores_sigterm_is_killed_after_the_grace_period() {
    let limits = JobLimits {
        stop_grace: Duration::from_millis(300),
        ..JobLimits::default()
    };
    let h = Harness::with_limits("trap '' TERM\nwhile true; do sleep 1; done", limits);
    let id = start(&h).await;
    h.call_ok("stop_job", json!({"job_id": id})).await;
    let status = h.wait_for_job(&id, |s| state(s) == "stopped").await;
    assert!(status["elapsed_seconds"].as_f64().unwrap() < 20.0);
}

#[tokio::test]
async fn stopping_a_finished_job_is_an_error_and_stopping_twice_is_not() {
    let h = Harness::new("exit 0");
    let id = start(&h).await;
    h.wait_for_job(&id, |s| state(s) == "completed").await;
    let message = h.call_err("stop_job", json!({"job_id": id})).await;
    assert!(message.contains("not running"), "{message}");
    assert!(
        h.call_err("stop_job", json!({"job_id": "nope"}))
            .await
            .contains("job not found")
    );

    let h = Harness::new("trap '' TERM\nwhile true; do sleep 1; done");
    let id = start(&h).await;
    h.call_ok("stop_job", json!({"job_id": id})).await;
    h.call_ok("stop_job", json!({"job_id": id})).await;
    h.server.shutdown().await;
}

#[tokio::test]
async fn jobs_beyond_the_limit_are_refused_until_one_ends() {
    let limits = JobLimits {
        max_running: 2,
        ..JobLimits::default()
    };
    let h = Harness::with_limits("sleep 30", limits);
    let a = start(&h).await;
    let _b = start(&h).await;
    let message = h.call_err("train", train_args(&h)).await;
    assert!(
        message.contains("limit is 2") && message.contains("PMETAL_MCP_MAX_JOBS"),
        "{message}"
    );
    h.call_ok("stop_job", json!({"job_id": a})).await;
    h.wait_for_job(&a, |s| state(s) == "stopped").await;
    let _c = start(&h).await;
    h.server.shutdown().await;
}

#[tokio::test]
async fn finished_jobs_are_forgotten_oldest_first_beyond_the_cap() {
    let limits = JobLimits {
        max_finished: 3,
        ..JobLimits::default()
    };
    let h = Harness::with_limits("exit 0", limits);
    let mut ids = Vec::new();
    for _ in 0..6 {
        let id = start(&h).await;
        h.wait_for_job(&id, |s| state(s) == "completed").await;
        ids.push(id);
    }
    let list = h.call_ok("list_jobs", json!({})).await;
    let kept: Vec<&str> = list["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j["id"].as_str().unwrap())
        .collect();
    assert!(kept.len() <= 4, "{kept:?}");
    assert!(kept.contains(&ids[5].as_str()));
    assert!(!kept.contains(&ids[0].as_str()));
}

#[tokio::test]
async fn shutting_the_server_down_stops_every_running_job() {
    let h = Harness::new("sleep 300 &\necho $! > \"$(dirname \"$0\")/helper.pid\"\nwait");
    let id = start(&h).await;
    let pid_file = h.dir.join("helper.pid");
    for _ in 0..100 {
        if pid_file.exists()
            && !std::fs::read_to_string(&pid_file)
                .unwrap()
                .trim()
                .is_empty()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let helper = std::fs::read_to_string(&pid_file).unwrap();
    h.server.shutdown().await;
    let status = h.call_ok("job_status", json!({"job_id": id})).await;
    assert!(
        !status["status"]["state"]
            .as_str()
            .unwrap()
            .starts_with("running")
    );
    assert!(gone(&helper).await, "a helper outlived the server");
}

#[tokio::test]
async fn learning_rate_control_reaches_a_training_job_and_only_that() {
    let h = Harness::new("sleep 30");
    let id = start(&h).await;
    let out = h.dir.join("out");
    // The job's directory exists once the server has prepared it.
    std::fs::create_dir_all(&out).unwrap();

    h.call_ok("job_set_lr", json!({"job_id": id, "lr": 1e-5}))
        .await;
    let written: Value =
        serde_json::from_slice(&std::fs::read(out.join(".lr_control.json")).unwrap()).unwrap();
    assert_eq!(written, json!({"action": "set_lr", "value": 1e-5}));

    // The job has not picked it up; a second command would replace it.
    let message = h
        .call_err("job_save_checkpoint", json!({"job_id": id}))
        .await;
    assert!(message.contains("not read the previous"), "{message}");
    std::fs::remove_file(out.join(".lr_control.json")).unwrap();
    h.call_ok("job_save_checkpoint", json!({"job_id": id}))
        .await;

    // The factor and the rate are checked before anything is written.
    std::fs::remove_file(out.join(".lr_control.json")).unwrap();
    assert!(
        h.call_err("job_reduce_lr", json!({"job_id": id, "factor": 1.5}))
            .await
            .contains("factor")
    );
    assert!(
        h.call_err("job_reduce_lr", json!({"job_id": id, "factor": -1.0}))
            .await
            .contains("factor")
    );
    assert!(
        h.call_err("job_set_lr", json!({"job_id": id, "lr": 0.0}))
            .await
            .contains("lr")
    );
    assert!(!out.join(".lr_control.json").exists());
    h.call_ok("stop_job", json!({"job_id": id})).await;

    // A job that does not read the control file is told so.
    let quantize = h
        .call_ok(
            "quantize",
            json!({"model": h.path("m"), "output": h.path("m.gguf")}),
        )
        .await;
    let message = h
        .call_err(
            "job_set_lr",
            json!({"job_id": quantize["job_id"], "lr": 1e-5}),
        )
        .await;
    assert!(
        message.contains("do not read learning-rate control"),
        "{message}"
    );
    h.server.shutdown().await;
}

#[tokio::test]
async fn control_commands_for_a_finished_job_are_refused() {
    let h = Harness::new("exit 0");
    let id = start(&h).await;
    h.wait_for_job(&id, |s| state(s) == "completed").await;
    let message = h.call_err("job_graceful_stop", json!({"job_id": id})).await;
    assert!(message.contains("not running"), "{message}");
}
