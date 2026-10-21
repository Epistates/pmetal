//! The hub against real child processes: a shell script stands in for `pmetal`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pmetal::core::SecretString;

use super::*;

/// A `pmetal` that records how it was called and reports what a trainer does
/// when told `--log-events`.
const TRAINER: &str = r#"
dir=$(dirname "$0")
printf '%s\n' "$@" > "$dir/argv.txt"
echo "${HF_TOKEN-unset}" > "$dir/token.txt"
events=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--log-events" ]; then events="$2"; fi
  shift
done
echo "loading"
if [ -n "$events" ]; then
  echo '{"event":"phase","job_id":"x","phase":{"phase":"loading_model"}}' >> "$events"
  sleep 0.2
  echo '{"event":"metric","job_id":"x","payload":{"kind":"step","step":1,"total_steps":4,"epoch":0,"total_epochs":1,"loss":3.0,"lr":0.0001,"tok_sec":500.0}}' >> "$events"
  echo '{"event":"metric","job_id":"x","payload":{"kind":"step","step":2,"total_steps":4,"epoch":0,"total_epochs":1,"loss":2.0,"lr":0.0001,"tok_sec":600.0}}' >> "$events"
  echo '{"event":"metric","job_id":"x","payload":{"kind":"step","step":3,"total_steps":4,"epoch":0,"total_epochs":1,"loss":2.5,"lr":0.0001,"tok_sec":600.0}}' >> "$events"
fi
echo "done"
"#;

fn fake(dir: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join("pmetal");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn hub(dir: &tempfile::TempDir, body: &str) -> JobHub {
    JobHub::new(Ok(fake(dir.path(), body)), dir.path().join("out"))
}

fn values(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn train_values() -> BTreeMap<String, String> {
    values(&[("model", "Qwen/Qwen3-0.6B"), ("dataset", "/data/d.jsonl")])
}

async fn until(what: &str, mut done: impl FnMut() -> bool) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for {what}");
}

async fn ended(hub: &JobHub, id: &str) -> JobView {
    until("the job to end", || {
        hub.view(id).is_some_and(|v| !v.state.is_active())
    })
    .await;
    hub.view(id).unwrap()
}

#[derive(Default)]
struct Recorder {
    updates: Mutex<Vec<JobView>>,
    output: Mutex<Vec<(String, Vec<OutputLine>)>>,
    resyncs: Mutex<usize>,
}

impl EventSink for Recorder {
    fn update(&self, job: &JobView) {
        self.updates.lock().unwrap().push(job.clone());
    }
    fn output(&self, id: &str, lines: &[OutputLine]) {
        self.output
            .lock()
            .unwrap()
            .push((id.to_string(), lines.to_vec()));
    }
    fn resync(&self) {
        *self.resyncs.lock().unwrap() += 1;
    }
}

#[tokio::test]
async fn a_training_job_runs_under_the_supervisor_and_reports_its_progress() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, TRAINER);
    let started = hub.start("train", &train_values(), None).unwrap();
    assert!(started.state.is_active());
    assert_eq!(started.kind, "train");
    assert_eq!(started.title, "Qwen/Qwen3-0.6B");

    let done = ended(&hub, &started.id).await;
    assert_eq!(done.state, State::Completed);
    assert_eq!(done.step, Some(3));
    assert_eq!(done.total_steps, Some(4));
    assert_eq!(done.loss, Some(2.5));
    // The best loss over the run, not the last.
    assert_eq!(done.best_loss, Some(2.0));
    assert_eq!(done.progress, Some(0.75));
    assert_eq!(done.tokens_per_second, Some(600.0));
    assert!(done.finished_at.is_some());

    // It was run the way the CLI spells it, with its progress file named.
    let argv = std::fs::read_to_string(dir.path().join("argv.txt")).unwrap();
    let argv: Vec<&str> = argv.lines().collect();
    assert_eq!(argv[0], "train");
    assert!(argv.windows(2).any(|w| w == ["--model", "Qwen/Qwen3-0.6B"]));
    let at = argv.iter().position(|a| *a == "--log-events").unwrap();
    assert!(argv[at + 1].ends_with("events.jsonl"));

    // Its output is kept, and the history for a chart.
    let output = hub.output(&done.id, 100, None).unwrap();
    assert!(output.lines.iter().any(|l| l.text == "loading"));
    let steps = hub.steps(&done.id, 0).unwrap();
    assert_eq!(steps.steps.len(), 3);
    assert_eq!(
        hub.steps(&done.id, steps.next_since).unwrap().steps.len(),
        0
    );
}

#[tokio::test]
async fn the_token_reaches_the_job_through_its_environment_not_its_command_line() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, TRAINER);
    let token = SecretString::new("hf_exampleTokenValue");
    let job = hub.start("train", &train_values(), Some(&token)).unwrap();
    let done = ended(&hub, &job.id).await;

    assert_eq!(
        std::fs::read_to_string(dir.path().join("token.txt"))
            .unwrap()
            .trim(),
        "hf_exampleTokenValue"
    );
    assert!(
        !done.command.contains("hf_exampleTokenValue"),
        "{}",
        done.command
    );
    let argv = std::fs::read_to_string(dir.path().join("argv.txt")).unwrap();
    assert!(!argv.contains("hf_exampleTokenValue"));
}

#[tokio::test]
async fn no_token_means_the_job_inherits_what_the_environment_has() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, TRAINER);
    let job = hub.start("train", &train_values(), None).unwrap();
    ended(&hub, &job.id).await;
    let seen = std::fs::read_to_string(dir.path().join("token.txt")).unwrap();
    // Nothing was added: whatever HF_TOKEN the test process has, or none.
    assert_eq!(
        seen.trim(),
        std::env::var("HF_TOKEN").unwrap_or_else(|_| "unset".into())
    );
}

#[tokio::test]
async fn a_bad_form_is_refused_before_anything_starts() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, TRAINER);
    let Err(StartError::Invalid(issues)) = hub.start("train", &values(&[("epochs", "x")]), None)
    else {
        panic!("expected the form to be refused");
    };
    assert!(issues.iter().any(|i| i.field == "epochs"));
    assert!(hub.list().is_empty());
    assert!(!dir.path().join("argv.txt").exists());
}

#[tokio::test]
async fn an_unknown_job_is_refused_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, TRAINER);
    let Err(StartError::Message(e)) = hub.start("telepathy", &values(&[]), None) else {
        panic!("expected a refusal");
    };
    assert!(e.message().contains("telepathy"));
}

#[tokio::test]
async fn a_missing_pmetal_says_what_to_do_and_starts_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let hub = JobHub::new(
        Err("The pmetal command-line tool was not found.".into()),
        dir.path().into(),
    );
    let Err(StartError::Message(e)) = hub.start("train", &train_values(), None) else {
        panic!("expected a refusal");
    };
    assert!(e.message().contains("not found"));
    assert!(hub.list().is_empty());
}

#[tokio::test]
async fn stopping_a_job_ends_it_as_stopped_and_it_is_asked_nicely_first() {
    let dir = tempfile::tempdir().unwrap();
    // Saves its work on SIGTERM, as `pmetal train` does.
    let hub = hub(
        &dir,
        r#"dir=$(dirname "$0"); trap 'echo saved > "$dir/saved.txt"; exit 130' TERM; while true; do sleep 0.05; done"#,
    );
    let job = hub.start("train", &train_values(), None).unwrap();
    until("the script to be running", || {
        hub.output(&job.id, 10, None).is_ok() && hub.running() == 1
    })
    .await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    hub.stop(&job.id).unwrap();
    let done = ended(&hub, &job.id).await;
    assert_eq!(done.state, State::Stopped);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("saved.txt"))
            .unwrap()
            .trim(),
        "saved"
    );
    // Stopping again is fine.
    hub.stop(&job.id).unwrap();
}

#[tokio::test]
async fn shutting_down_stops_every_job_and_waits_for_them() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, "sleep 300");
    let a = hub.start("train", &train_values(), None).unwrap();
    let b = hub
        .start("serve", &values(&[("model", "m"), ("port", "9301")]), None)
        .unwrap();
    assert_eq!(hub.running(), 2);

    hub.shutdown().await;

    assert_eq!(hub.running(), 0);
    for id in [&a.id, &b.id] {
        assert_eq!(hub.view(id).unwrap().state, State::Stopped);
    }
}

#[tokio::test]
async fn a_second_job_may_not_write_into_a_running_jobs_folder() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, "sleep 300");
    let out = dir.path().join("shared-run");
    let mut v = train_values();
    v.insert("output_dir".into(), out.display().to_string());
    hub.start("train", &v, None).unwrap();

    let Err(StartError::Message(e)) = hub.start("train", &v, None) else {
        panic!("expected a refusal");
    };
    assert!(e.message().contains("already writing"), "{e}");
    hub.shutdown().await;
}

#[tokio::test]
async fn two_servers_may_not_take_the_same_port() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, "sleep 300");
    let v = values(&[("model", "m"), ("port", "9302")]);
    hub.start("serve", &v, None).unwrap();
    let Err(StartError::Message(e)) = hub.start("serve", &v, None) else {
        panic!("expected a refusal");
    };
    assert!(e.message().contains("port"), "{e}");
    // Another port is fine.
    hub.start("serve", &values(&[("model", "m"), ("port", "9303")]), None)
        .unwrap();
    hub.shutdown().await;
}

#[tokio::test]
async fn too_many_jobs_say_how_many_and_what_to_do() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, "sleep 300");
    for port in 0..MAX_RUNNING {
        let port = (9400 + port).to_string();
        hub.start("serve", &values(&[("model", "m"), ("port", &port)]), None)
            .unwrap();
    }
    let Err(StartError::Message(e)) =
        hub.start("serve", &values(&[("model", "m"), ("port", "9499")]), None)
    else {
        panic!("expected a refusal");
    };
    assert!(e.message().contains(&MAX_RUNNING.to_string()), "{e}");
    assert!(e.message().contains("Stop one"), "{e}");
    hub.shutdown().await;
}

#[tokio::test]
async fn a_failed_job_carries_what_it_said_and_a_removed_one_leaves_the_list() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, "echo 'Error: model not found: nope/nope' >&2; exit 2");
    let job = hub.start("train", &train_values(), None).unwrap();
    let done = ended(&hub, &job.id).await;
    assert_eq!(done.state, State::Failed);
    assert_eq!(done.exit_code, Some(2));
    assert!(done.error.unwrap().contains("model not found"));

    hub.forget(&job.id).unwrap();
    assert!(hub.view(&job.id).is_none());
    assert!(hub.list().is_empty());
}

#[tokio::test]
async fn a_running_job_cannot_be_removed_from_the_list() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, "sleep 300");
    let job = hub.start("train", &train_values(), None).unwrap();
    assert!(hub.forget(&job.id).unwrap_err().message().contains("Stop"));
    hub.shutdown().await;
}

#[tokio::test]
async fn the_webview_hears_of_a_job_starting_its_output_and_its_end() {
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(&dir, TRAINER);
    let sink = Arc::new(Recorder::default());
    hub.pump(sink.clone());

    let job = hub.start("train", &train_values(), None).unwrap();
    ended(&hub, &job.id).await;
    until("the end to be sent", || {
        sink.updates
            .lock()
            .unwrap()
            .iter()
            .any(|u| u.id == job.id && u.state == State::Completed)
    })
    .await;

    let updates = sink.updates.lock().unwrap();
    // Told at once that it started, and later of its progress.
    assert!(updates.iter().any(|u| u.state == State::Running));
    assert!(updates.iter().any(|u| u.step.is_some()));
    // Updates are coalesced, not one per step event and line.
    assert!(updates.len() < 12, "{} updates", updates.len());
    let output: Vec<String> = sink
        .output
        .lock()
        .unwrap()
        .iter()
        .flat_map(|(_, lines)| lines.iter().map(|l| l.text.clone()))
        .collect();
    assert!(output.contains(&"loading".to_string()), "{output:?}");
    assert!(output.contains(&"done".to_string()), "{output:?}");
    assert_eq!(*sink.resyncs.lock().unwrap(), 0);
}

#[tokio::test]
async fn a_server_is_serving_once_something_answers_on_its_port() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(
        &dir,
        &format!("echo 'OpenAI-compatible API available at http://127.0.0.1:{port}/v1'; sleep 300"),
    );
    let sink = Arc::new(Recorder::default());
    hub.pump(sink.clone());
    let job = hub
        .start(
            "serve",
            &values(&[("model", "m"), ("port", &port.to_string())]),
            None,
        )
        .unwrap();

    until("the server to be reachable", || {
        hub.view(&job.id)
            .and_then(|v| v.serve)
            .is_some_and(|s| s.reachable)
    })
    .await;
    let serve = hub.view(&job.id).unwrap().serve.unwrap();
    assert_eq!(serve.url, format!("http://127.0.0.1:{port}"));
    // And the webview is told, not left to ask.
    until("the webview to hear it", || {
        sink.updates
            .lock()
            .unwrap()
            .iter()
            .any(|u| u.serve.as_ref().is_some_and(|s| s.reachable))
    })
    .await;
    hub.shutdown().await;
    drop(listener);
}

#[tokio::test]
async fn a_server_that_cannot_bind_is_never_shown_as_serving() {
    // It announces an address nothing listens on.
    let free = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let dir = tempfile::tempdir().unwrap();
    let hub = hub(
        &dir,
        &format!(
            "echo 'OpenAI-compatible API available at http://127.0.0.1:{free}/v1'; echo 'Address already in use' >&2; exit 1"
        ),
    );
    let job = hub
        .start(
            "serve",
            &values(&[("model", "m"), ("port", &free.to_string())]),
            None,
        )
        .unwrap();
    let done = ended(&hub, &job.id).await;
    assert_eq!(done.state, State::Failed);
    assert!(done.serve.is_none_or(|s| !s.reachable));
    assert!(done.error.unwrap().contains("Address already in use"));
}

#[test]
fn the_announcement_gives_the_address() {
    assert_eq!(
        announced_address("OpenAI-compatible API available at http://127.0.0.1:8080/v1"),
        Some("127.0.0.1:8080".to_string())
    );
    assert_eq!(announced_address("starting"), None);
}

#[test]
fn failures_a_person_can_act_on_say_so() {
    assert!(explain_failure("killed by signal 9").contains("memory"));
    assert!(explain_failure("HTTP 401 from huggingface.co").contains("Settings"));
    assert_eq!(explain_failure("shape mismatch"), "shape mismatch");
}
