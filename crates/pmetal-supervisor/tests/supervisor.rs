//! The supervisor against real child processes (a shell script stands in for
//! `pmetal`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use pmetal_supervisor::{
    Error, Event, EventKind, JobId, JobRequest, Limits, ProcessStatus, Stream, Supervisor,
};
use tokio::sync::broadcast::error::RecvError;

/// A script that stands in for `pmetal`: `body` runs with the subcommand and its
/// arguments as `$@`.
fn fake_pmetal(dir: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join("fake-pmetal");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn supervisor(dir: &Path, body: &str) -> Supervisor {
    Supervisor::new(fake_pmetal(dir, body), Limits::default())
}

fn request(argv: &[&str]) -> JobRequest {
    JobRequest {
        subcommand: "train".into(),
        argv: argv.iter().map(|s| s.to_string()).collect(),
        ..JobRequest::default()
    }
}

async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for {what}");
}

async fn finished(s: &Supervisor, id: &str) -> ProcessStatus {
    wait_until("the job to end", || {
        !s.summary(id).unwrap().status.is_active()
    })
    .await;
    s.summary(id).unwrap().status
}

fn alive(pid: &str) -> bool {
    std::process::Command::new("kill")
        .args(["-0", pid.trim()])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

async fn read_pid(file: &Path) -> String {
    for _ in 0..200 {
        if let Ok(text) = std::fs::read_to_string(file) {
            if !text.trim().is_empty() {
                return text.trim().to_string();
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("{} never appeared", file.display());
}

async fn gone(pid: &str) -> bool {
    for _ in 0..80 {
        if !alive(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

#[tokio::test]
async fn output_of_both_streams_keeps_the_order_it_was_printed_in() {
    let dir = tempfile::tempdir().unwrap();
    let s = supervisor(
        dir.path(),
        "echo one; echo err1 >&2; sleep 0.2; echo two; echo err2 >&2; sleep 0.2; echo three",
    );
    let id = s.spawn(request(&[])).unwrap();
    assert_eq!(
        finished(&s, &id).await,
        ProcessStatus::Completed { exit_code: 0 }
    );
    let page = s.logs(&id, 100, None).unwrap();
    assert_eq!(
        page.lines,
        ["one", "[stderr] err1", "two", "[stderr] err2", "three"]
    );
    assert_eq!(page.total_lines, 5);
    assert_eq!(page.entries[1].stream, Stream::Stderr);
}

#[tokio::test]
async fn a_failing_job_reports_the_last_line_it_wrote_to_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let s = supervisor(
        dir.path(),
        "echo starting; echo 'model not found' >&2; exit 3",
    );
    let id = s.spawn(request(&[])).unwrap();
    assert_eq!(
        finished(&s, &id).await,
        ProcessStatus::Failed {
            exit_code: 3,
            error: "model not found".into()
        }
    );
}

#[tokio::test]
async fn a_job_killed_by_a_signal_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let s = supervisor(dir.path(), "kill -9 $$");
    let id = s.spawn(request(&[])).unwrap();
    let ProcessStatus::Failed { error, .. } = finished(&s, &id).await else {
        panic!("a kill is a failure");
    };
    assert_eq!(error, "killed by signal 9");
}

#[tokio::test]
async fn a_missing_binary_is_an_error_not_a_hang() {
    let s = Supervisor::new("/definitely/not/pmetal", Limits::default());
    let err = s.spawn(request(&[])).unwrap_err();
    assert!(matches!(err, Error::Io(_)), "{err}");
    assert_eq!(s.running(), 0);
}

#[tokio::test]
async fn stopping_a_job_ends_the_helpers_it_spawned_too() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("helper.pid");
    let s = supervisor(
        dir.path(),
        &format!("sleep 300 & echo $! > {}; wait", pidfile.display()),
    );
    let id = s.spawn(request(&[])).unwrap();
    let helper = read_pid(&pidfile).await;
    assert!(alive(&helper));
    s.stop(&id).unwrap();
    assert_eq!(finished(&s, &id).await, ProcessStatus::Stopped);
    assert!(gone(&helper).await, "the helper outlived its job");
}

#[tokio::test]
async fn a_job_that_ignores_sigterm_is_killed_after_the_grace_period() {
    let dir = tempfile::tempdir().unwrap();
    let s = Supervisor::new(
        fake_pmetal(
            dir.path(),
            "trap '' TERM; echo ready; while :; do sleep 0.1; done",
        ),
        Limits {
            stop_grace: Duration::from_millis(300),
            ..Limits::default()
        },
    );
    let id = s.spawn(request(&[])).unwrap();
    wait_until("the job to say it is ready", || {
        s.logs(&id, 10, None).unwrap().lines == ["ready"]
    })
    .await;
    s.stop(&id).unwrap();
    // Stopping twice is not an error while it is going.
    s.stop(&id).unwrap();
    assert_eq!(finished(&s, &id).await, ProcessStatus::Stopped);
    // Stopped already: asking again is not an error.
    s.stop(&id).unwrap();

    // A job that ended on its own cannot be stopped.
    let done = supervisor(dir.path(), "true");
    let id = done.spawn(request(&[])).unwrap();
    finished(&done, &id).await;
    assert!(matches!(done.stop(&id), Err(Error::NotRunning)));
    assert!(matches!(done.stop("nope"), Err(Error::NotFound(_))));
}

#[tokio::test]
async fn dropping_the_last_handle_kills_the_jobs() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("helper.pid");
    let s = supervisor(
        dir.path(),
        &format!("sleep 300 & echo $! > {}; wait", pidfile.display()),
    );
    s.spawn(request(&[])).unwrap();
    let helper = read_pid(&pidfile).await;
    drop(s);
    assert!(gone(&helper).await, "the helper outlived the supervisor");
}

#[tokio::test]
async fn kill_all_now_does_not_wait_for_anyone() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("helper.pid");
    let s = supervisor(
        dir.path(),
        &format!(
            "trap '' TERM; sleep 300 & echo $! > {}; wait",
            pidfile.display()
        ),
    );
    let id = s.spawn(request(&[])).unwrap();
    let helper = read_pid(&pidfile).await;
    s.kill_all_now();
    assert!(gone(&helper).await);
    assert_eq!(finished(&s, &id).await, ProcessStatus::Stopped);
}

#[tokio::test]
async fn shutdown_stops_every_running_job() {
    let dir = tempfile::tempdir().unwrap();
    let s = supervisor(dir.path(), "sleep 300");
    let a = s.spawn(request(&[])).unwrap();
    let b = s.spawn(request(&[])).unwrap();
    assert_eq!(s.running(), 2);
    s.shutdown().await;
    assert_eq!(s.running(), 0);
    assert_eq!(s.summary(&a).unwrap().status, ProcessStatus::Stopped);
    assert_eq!(s.summary(&b).unwrap().status, ProcessStatus::Stopped);
}

#[tokio::test]
async fn starting_more_than_the_limit_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let s = Supervisor::new(
        fake_pmetal(dir.path(), "sleep 300"),
        Limits {
            max_running: 1,
            ..Limits::default()
        },
    );
    s.spawn(request(&[])).unwrap();
    let err = s.spawn(request(&[])).unwrap_err();
    assert!(
        matches!(
            err,
            Error::TooManyRunning {
                running: 1,
                limit: 1
            }
        ),
        "{err}"
    );
    s.shutdown().await;
}

#[tokio::test]
async fn finished_jobs_are_forgotten_oldest_first() {
    let dir = tempfile::tempdir().unwrap();
    let s = Supervisor::new(
        fake_pmetal(dir.path(), "true"),
        Limits {
            max_finished: 2,
            ..Limits::default()
        },
    );
    let mut ids = Vec::new();
    for _ in 0..4 {
        let id = s.spawn(request(&[])).unwrap();
        finished(&s, &id).await;
        ids.push(id);
    }
    // Spawning prunes: the oldest of the finished are gone.
    let last = s.spawn(request(&[])).unwrap();
    finished(&s, &last).await;
    let kept: Vec<JobId> = s.list().into_iter().map(|j| j.id).collect();
    assert!(kept.len() <= 3, "{kept:?}");
    assert!(kept.contains(&last));
    assert!(!kept.contains(&ids[0]));
    assert!(matches!(s.summary(&ids[0]), Err(Error::NotFound(_))));
    s.forget(&last).unwrap();
    assert!(matches!(s.forget(&last), Err(Error::NotFound(_))));
}

#[tokio::test]
async fn output_is_safe_to_draw_and_to_show() {
    let dir = tempfile::tempdir().unwrap();
    let s = supervisor(
        dir.path(),
        r#"printf '\033[31mred\033[0m text\n'
printf 'progress 10%%\rprogress 100%%\n'
printf 'bad \377\376 bytes\n'
printf '\033]0;retitled\007title\n'
echo 'fetching with hf_AbCdEfGhIjKlMnOpQrStUvWx'"#,
    );
    let id = s.spawn(request(&[])).unwrap();
    finished(&s, &id).await;
    let lines = s.logs(&id, 100, None).unwrap().lines;
    assert_eq!(
        lines,
        [
            "red text",
            "progress 100%",
            "bad \u{fffd}\u{fffd} bytes",
            "title",
            "fetching with hf_[REDACTED]"
        ]
    );
}

#[tokio::test]
async fn a_credential_in_the_arguments_is_not_in_the_summary() {
    let dir = tempfile::tempdir().unwrap();
    let s = supervisor(dir.path(), "true");
    let id = s
        .spawn(request(&[
            "--model",
            "m",
            "--hf-token",
            "hf_AbCdEfGhIjKlMnOpQrSt",
        ]))
        .unwrap();
    let args = s.summary(&id).unwrap().args.join(" ");
    assert!(!args.contains("AbCdEfGh"), "{args}");
    assert!(args.contains("--model m"), "{args}");
}

#[tokio::test]
async fn memory_stays_bounded_for_a_job_that_never_stops_printing() {
    let dir = tempfile::tempdir().unwrap();
    // A megabyte with no newline, then enough lines to overflow the ring.
    let s = supervisor(
        dir.path(),
        "head -c 1048576 /dev/zero | tr '\\0' x; echo; i=0; while [ $i -lt 12000 ]; do echo \"line $i\"; i=$((i+1)); done",
    );
    let id = s.spawn(request(&[])).unwrap();
    finished(&s, &id).await;
    let page = s.logs(&id, 20_000, None).unwrap();
    assert!(page.lines.len() <= 10_000);
    assert!(page.lines.iter().all(|l| l.len() < 20_000));
    assert_eq!(page.lines.last().unwrap(), "line 11999");
    assert_eq!(page.total_lines, 12_001);
}

#[tokio::test]
async fn events_arrive_in_order_and_carry_the_jobs_own_progress() {
    let dir = tempfile::tempdir().unwrap();
    let events_file = dir.path().join("out").join("events.jsonl");
    let s = supervisor(
        dir.path(),
        &format!(
            r#"echo '{{"event":"phase","job_id":"x","phase":{{"phase":"loading_model"}}}}' >> {f}
sleep 0.3
echo '{{"event":"metric","job_id":"x","payload":{{"kind":"step","step":1,"total_steps":2,"loss":2.5,"lr":0.001,"tok_sec":900.0}}}}' >> {f}
echo '{{"event":"metric","job_id":"x","payload":{{"kind":"step","step":2,"total_steps":2,"loss":2.0,"lr":0.001,"tok_sec":950.0}}}}' >> {f}
echo working
sleep 0.3"#,
            f = events_file.display()
        ),
    );
    let mut rx = s.subscribe();
    let id = s
        .spawn(JobRequest {
            events_file: Some(events_file),
            ..request(&[])
        })
        .unwrap();

    let mut seen: Vec<String> = Vec::new();
    loop {
        let Event { job, kind } = rx.recv().await.unwrap();
        assert_eq!(job, id);
        match kind {
            EventKind::Started => seen.push("started".into()),
            EventKind::Output(line) => seen.push(format!("out:{}", line.text)),
            EventKind::Job(event) => {
                assert_eq!(event.job_id(), id, "the job's id is the supervisor's");
                seen.push("job".into());
            }
            EventKind::Delta(_) => panic!("deltas were not asked for"),
            EventKind::Exited(status) => {
                seen.push(format!("exited:{status:?}"));
                break;
            }
        }
    }
    assert_eq!(seen.first().map(String::as_str), Some("started"));
    assert!(seen.contains(&"out:working".to_string()), "{seen:?}");
    assert_eq!(seen.iter().filter(|e| *e == "job").count(), 3, "{seen:?}");
    assert!(
        seen.last().unwrap().starts_with("exited:Completed"),
        "{seen:?}"
    );

    let summary = s.summary(&id).unwrap();
    assert_eq!(summary.metrics.last_step, Some(2));
    assert_eq!(summary.metrics.total_steps, Some(2));
    assert_eq!(summary.metrics.last_loss, Some(2.0));
    assert_eq!(summary.metrics.tokens_per_second, Some(950.0));
    // The phase is cleared by the first step.
    assert_eq!(summary.phase, None);

    let steps = s.steps(&id, 0).unwrap();
    assert_eq!(steps.steps.len(), 2);
    assert_eq!(steps.next_since, 2);
    assert!(s.steps(&id, 2).unwrap().steps.is_empty());
    assert_eq!(s.steps(&id, 1).unwrap().steps[0].step, 2);
}

#[tokio::test]
async fn the_phase_a_job_is_in_shows_in_its_summary() {
    let dir = tempfile::tempdir().unwrap();
    let events_file = dir.path().join("events.jsonl");
    let s = supervisor(
        dir.path(),
        &format!(
            r#"echo '{{"event":"phase","job_id":"x","phase":{{"phase":"loading_model"}}}}' >> {}; sleep 5"#,
            events_file.display()
        ),
    );
    let id = s
        .spawn(JobRequest {
            events_file: Some(events_file),
            ..request(&[])
        })
        .unwrap();
    wait_until("the phase", || {
        s.summary(&id).unwrap().phase.as_deref() == Some("Loading model…")
    })
    .await;
    s.shutdown().await;
}

#[tokio::test]
async fn standard_output_arrives_in_pieces_for_a_job_that_asks() {
    let dir = tempfile::tempdir().unwrap();
    let s = supervisor(
        dir.path(),
        "printf 'Hel'; sleep 0.2; printf 'lo, wor'; sleep 0.2; printf 'ld\\n'",
    );
    let mut rx = s.subscribe();
    let id = s
        .spawn(JobRequest {
            deltas: true,
            ..request(&[])
        })
        .unwrap();
    let mut deltas = Vec::new();
    let mut lines = Vec::new();
    loop {
        match rx.recv().await.unwrap().kind {
            EventKind::Delta(d) => deltas.push(d),
            EventKind::Output(l) => lines.push(l.text),
            EventKind::Exited(_) => break,
            _ => {}
        }
    }
    assert!(deltas.len() >= 2, "{deltas:?}");
    assert_eq!(deltas.concat(), "Hello, world\n");
    assert_eq!(lines, ["Hello, world"]);
    assert_eq!(s.logs(&id, 10, None).unwrap().lines, ["Hello, world"]);
}

#[tokio::test]
async fn a_subscriber_that_falls_behind_is_told_and_the_state_is_complete() {
    let dir = tempfile::tempdir().unwrap();
    let s = supervisor(
        dir.path(),
        "i=0; while [ $i -lt 6000 ]; do echo \"line $i\"; i=$((i+1)); done",
    );
    let mut idle = s.subscribe();
    let id = s.spawn(request(&[])).unwrap();
    finished(&s, &id).await;
    // Never read until now: the channel kept its bound, and says what it lost.
    match idle.recv().await {
        Err(RecvError::Lagged(missed)) => assert!(missed > 1000, "{missed}"),
        other => panic!("expected the receiver to have lagged, got {other:?}"),
    }
    // The job did not wait for it, and nothing was lost from the supervisor's own state.
    let page = s.logs(&id, 20_000, None).unwrap();
    assert_eq!(page.total_lines, 6000);
    assert_eq!(page.lines.last().unwrap(), "line 5999");
}

#[tokio::test]
async fn a_job_that_leaves_a_helper_holding_its_pipes_still_ends() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("helper.pid");
    // The job exits at once; its helper keeps standard output open.
    let s = supervisor(
        dir.path(),
        &format!("sleep 300 & echo $! > {}; echo done", pidfile.display()),
    );
    let id = s.spawn(request(&[])).unwrap();
    let helper = read_pid(&pidfile).await;
    assert_eq!(
        finished(&s, &id).await,
        ProcessStatus::Completed { exit_code: 0 }
    );
    assert!(gone(&helper).await, "the straggler was left running");
}

#[tokio::test]
async fn running_output_names_the_directory_of_a_running_job_only() {
    let dir = tempfile::tempdir().unwrap();
    let s = supervisor(dir.path(), "sleep 300");
    let with = s
        .spawn(JobRequest {
            output: Some(dir.path().join("out")),
            ..request(&[])
        })
        .unwrap();
    let without = s.spawn(request(&[])).unwrap();
    let (command, out) = s.running_output(&with).unwrap();
    assert_eq!((command.as_str(), out), ("train", dir.path().join("out")));
    assert!(matches!(
        s.running_output(&without),
        Err(Error::NoOutput(_))
    ));
    s.shutdown().await;
    assert!(matches!(s.running_output(&with), Err(Error::NotRunning)));
}
