//! Every trainer command stops gracefully.
//!
//! A supervisor stops a job with SIGTERM, and a person at a terminal presses
//! Ctrl-C. Either way the run must finish the step it is on, save what it has
//! and exit with status 130 (128 + SIGINT), instead of dying with its progress.
//! Each test starts the real `pmetal` binary on a tiny model, waits until the
//! run is training, sends SIGTERM and checks the exit status and the artifact.

mod fixtures;

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;
use tempfile::TempDir;

/// How long to wait for the run to start training, and to stop once asked.
/// Events, not wall-clock margins, decide when to signal; these are only the
/// limits after which a hung run is reported instead of waited for.
const LIMIT: Duration = Duration::from_secs(300);

/// A `pmetal` process and where its output goes.
struct Run {
    child: Child,
    stdout: PathBuf,
    stderr: PathBuf,
    events: PathBuf,
}

impl Run {
    /// Start `pmetal args...`, with `--log-events` pointing at a file for the
    /// commands that write an event log (the others refuse the flag).
    fn start(dir: &Path, args: &[&str]) -> Self {
        let (stdout, stderr, events) = (
            dir.join("stdout.log"),
            dir.join("stderr.log"),
            dir.join("events.jsonl"),
        );
        let logs_events = matches!(
            args.first().copied(),
            Some(
                "train"
                    | "distill"
                    | "grpo"
                    | "rlkd"
                    | "preference"
                    | "pretrain"
                    | "embed-train"
                    | "train-mtp"
                    | "train-draft"
                    | "train-diffusion"
            )
        );
        let mut command = Command::new(env!("CARGO_BIN_EXE_pmetal"));
        if logs_events {
            command.arg("--log-events").arg(&events);
        }
        let child = command
            .args(args)
            .env("RUST_LOG", "info")
            .stdin(Stdio::null())
            .stdout(std::fs::File::create(&stdout).unwrap())
            .stderr(std::fs::File::create(&stderr).unwrap())
            .spawn()
            .expect("start the pmetal binary");
        Self {
            child,
            stdout,
            stderr,
            events,
        }
    }

    /// Everything the run has printed and logged so far.
    fn output(&self) -> String {
        [&self.stdout, &self.stderr, &self.events]
            .iter()
            .map(|path| std::fs::read_to_string(path).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Wait until the run reports `progress`, or fail with its output if it
    /// exits first.
    fn wait_until(&mut self, what: &str, progress: impl Fn(&str) -> bool) {
        let started = Instant::now();
        loop {
            let output = self.output();
            if progress(&output) {
                return;
            }
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("the run ended ({status}) before {what}:\n{output}");
            }
            if started.elapsed() > LIMIT {
                let _ = self.child.kill();
                panic!("no {what} within {LIMIT:?}:\n{output}");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// SIGTERM, as a supervisor stops a job.
    fn terminate(&self) {
        let status = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }

    /// The exit code, once the run ends; a run that is still going after
    /// [`LIMIT`] is killed and reported.
    fn exit_code(&mut self) -> Option<i32> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status.code();
            }
            if started.elapsed() > LIMIT {
                let _ = self.child.kill();
                panic!("the run did not stop within {LIMIT:?}:\n{}", self.output());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// Whether the event log holds a training step.
fn event_log_has_a_step(output: &str) -> bool {
    output.lines().any(|line| {
        serde_json::from_str::<Value>(line)
            .is_ok_and(|event| event["event"] == "metric" && event["payload"]["kind"] == "step")
    })
}

/// Stop a run with SIGTERM once `progress` shows it training; it must exit
/// with status 130 and have written `artifact`.
fn stops_gracefully(mut run: Run, progress: impl Fn(&str) -> bool, artifact: &Path) {
    run.wait_until("a training step", progress);
    run.terminate();
    let code = run.exit_code();
    let output = run.output();
    assert_eq!(
        code,
        Some(130),
        "a stopped run exits with the interrupted status (None: killed by the signal):\n{output}"
    );
    assert!(
        artifact.exists(),
        "the run saved nothing before it exited ({}):\n{output}",
        artifact.display()
    );
    // A command that writes an event log wrote its steps to it.
    if let Ok(log) = std::fs::read_to_string(&run.events) {
        assert!(
            event_log_has_a_step(&log),
            "the event log holds no step:\n{output}"
        );
    }
}

/// A tiny model and a text dataset in a fresh directory.
fn workspace() -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    fixtures::write_tiny_model(&model, 1);
    (dir, model)
}

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn train_stops_after_its_step_and_keeps_the_adapter() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_text_dataset(&data);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "train",
            "--model",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--epochs",
            "1000",
            "--batch-size",
            "2",
            "--max-seq-len",
            "32",
            "--lora-r",
            "4",
        ],
    );
    stops_gracefully(
        run,
        event_log_has_a_step,
        &out.join("adapter_model.safetensors"),
    );
}

#[test]
fn full_fine_tune_stops_after_its_step_and_keeps_the_model() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_text_dataset(&data);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "train",
            "--model",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--full",
            "--learning-rate",
            "1e-5",
            "--epochs",
            "1000",
            "--batch-size",
            "2",
            "--max-seq-len",
            "32",
        ],
    );
    stops_gracefully(run, event_log_has_a_step, &out.join("model.safetensors"));
    // A whole model: what `infer` and `serve` load, not an adapter.
    for file in ["config.json", "tokenizer.json"] {
        assert!(out.join(file).exists(), "the saved model has no {file}");
    }
    assert!(!out.join("adapter_model.safetensors").exists());
}

#[test]
fn a_fully_fine_tuned_model_is_one_infer_loads() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_text_dataset(&data);
    let out = dir.path().join("out");
    let train = Command::new(env!("CARGO_BIN_EXE_pmetal"))
        .args([
            "train",
            "--model",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--full",
            "--freeze-embeddings",
            "--learning-rate",
            "1e-5",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--max-seq-len",
            "32",
        ])
        .output()
        .unwrap();
    assert!(
        train.status.success(),
        "{}",
        String::from_utf8_lossy(&train.stderr)
    );
    let infer = Command::new(env!("CARGO_BIN_EXE_pmetal"))
        .args([
            "infer",
            "--model",
            s(&out),
            "--prompt",
            "the cat",
            "--max-tokens",
            "4",
        ])
        .output()
        .unwrap();
    assert!(
        infer.status.success(),
        "infer could not load the fine-tuned model:\n{}",
        String::from_utf8_lossy(&infer.stderr)
    );
}

#[test]
fn a_full_fine_tune_that_will_not_fit_is_refused_before_it_loads() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_text_dataset(&data);
    // A sequence length no device holds the activations of.
    let refused = Command::new(env!("CARGO_BIN_EXE_pmetal"))
        .args([
            "train",
            "--model",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&dir.path().join("out")),
            "--full",
            "--learning-rate",
            "1e-5",
            "--batch-size",
            "1024",
            "--max-seq-len",
            "1000000",
        ])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("recommended working set") && stderr.contains("--no-memory-check"),
        "{stderr}"
    );
    assert!(!dir.path().join("out").join("model.safetensors").exists());
}

#[test]
fn grpo_stops_and_takes_its_reward_command_with_it() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_prompt_dataset(&data);
    let out = dir.path().join("out");
    // A reward command that starts a helper of its own and records its pid.
    let script = dir.path().join("reward.py");
    let pids = dir.path().join("pids.txt");
    std::fs::write(
        &script,
        format!(
            "import json, os, subprocess, sys\n\
             helper = subprocess.Popen(['sleep', '300'])\n\
             open({pids:?}, 'w').write(f'{{os.getpid()}} {{helper.pid}}')\n\
             for line in sys.stdin:\n\
             \x20   n = len(json.loads(line)['completions'])\n\
             \x20   print(json.dumps({{'rewards': [float(i) for i in range(n)]}}), flush=True)\n",
            pids = pids.display().to_string(),
        ),
    )
    .unwrap();
    let command = format!("python3 {}", script.display());
    let mut run = Run::start(
        dir.path(),
        &[
            "grpo",
            "--model",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--epochs",
            "1000",
            "--num-generations",
            "2",
            "--max-completion-length",
            "6",
            "--max-seq-len",
            "32",
            "--lora-r",
            "4",
            "--reward-cmd",
            &command,
        ],
    );
    run.wait_until("a training step", event_log_has_a_step);
    run.terminate();
    assert_eq!(run.exit_code(), Some(130), "{}", run.output());
    assert!(out.join("adapter_model.safetensors").exists());

    // The command and its helper are gone with the run.
    let recorded = std::fs::read_to_string(&pids).unwrap();
    for pid in recorded.split_whitespace() {
        let started = Instant::now();
        while Command::new("kill")
            .args(["-0", pid])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
        {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "process {pid} outlived the run"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

#[test]
fn distill_stops_after_its_step_and_keeps_the_student() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_text_dataset(&data);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "distill",
            "--teacher",
            s(&model),
            "--student",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--epochs",
            "1000",
            "--batch-size",
            "2",
            "--max-seq-len",
            "32",
            "--lora-r",
            "4",
        ],
    );
    stops_gracefully(
        run,
        event_log_has_a_step,
        &out.join("adapter_model.safetensors"),
    );
}

#[test]
fn grpo_stops_after_its_step_and_keeps_the_policy() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_prompt_dataset(&data);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "grpo",
            "--model",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--epochs",
            "1000",
            "--num-generations",
            "2",
            "--max-completion-length",
            "6",
            "--max-seq-len",
            "32",
            "--lora-r",
            "4",
            "--reasoning-rewards",
        ],
    );
    stops_gracefully(
        run,
        event_log_has_a_step,
        &out.join("adapter_model.safetensors"),
    );
}

#[test]
fn rlkd_stops_after_its_step_and_keeps_the_policy() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_prompt_dataset(&data);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "rlkd",
            "--model",
            s(&model),
            "--teacher-model",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--epochs",
            "1000",
            "--num-generations",
            "2",
            "--max-completion-length",
            "6",
            "--max-seq-len",
            "32",
            "--lora-r",
            "4",
            "--reasoning-rewards",
        ],
    );
    stops_gracefully(
        run,
        event_log_has_a_step,
        &out.join("adapter_model.safetensors"),
    );
}

#[test]
fn preference_stops_after_its_step_and_keeps_the_adapter() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_preference_dataset(&data);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "preference",
            "--model",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--loss",
            "dpo",
            "--epochs",
            "1000",
            "--batch-size",
            "2",
            "--max-length",
            "32",
            "--max-prompt-length",
            "16",
            "--lora-r",
            "4",
        ],
    );
    stops_gracefully(
        run,
        event_log_has_a_step,
        &out.join("adapter_model.safetensors"),
    );
}

#[test]
fn full_preference_stops_after_its_step_and_keeps_the_model() {
    let (dir, model) = workspace();
    let data = dir.path().join("data.jsonl");
    fixtures::write_preference_dataset(&data);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "preference",
            "--model",
            s(&model),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--full",
            "--loss",
            "dpo",
            "--epochs",
            "1000",
            "--batch-size",
            "2",
            "--max-length",
            "32",
            "--max-prompt-length",
            "16",
        ],
    );
    stops_gracefully(run, event_log_has_a_step, &out.join("model.safetensors"));
    assert!(out.join("config.json").exists());
    assert!(!out.join("adapter_model.safetensors").exists());
}

/// Tokenize the text dataset into shards with the real `pmetal tokenize`.
fn shards(dir: &Path, model: &Path) -> PathBuf {
    let data = dir.join("shard-source.jsonl");
    fixtures::write_text_dataset(&data);
    let out = dir.join("shards");
    let status = Command::new(env!("CARGO_BIN_EXE_pmetal"))
        .args(["tokenize", "--input", s(&data), "--output", s(&out)])
        .args(["--tokenizer", s(&model.join("tokenizer.json"))])
        .status()
        .unwrap();
    assert!(status.success(), "pmetal tokenize");
    std::fs::read_dir(&out)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().is_some_and(|ext| ext == "bin"))
        .expect("tokenize wrote a shard")
}

/// A training line on stderr: `step N | ... loss ...`.
fn stderr_has_a_step(output: &str) -> bool {
    output
        .lines()
        .any(|line| line.trim_start().starts_with("step") && line.contains("loss"))
}

#[test]
fn pretrain_stops_after_its_step_and_writes_its_final_checkpoint() {
    let (dir, model) = workspace();
    let shard = shards(dir.path(), &model);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "pretrain",
            "--arch",
            "qwen3",
            "--model-config",
            s(&model.join("config.json")),
            "--shards",
            s(&shard),
            "--seq-len",
            "16",
            "--batch-size",
            "2",
            "--steps",
            "100000",
            "--eos-token-id",
            "1",
            "--output",
            s(&out),
            "--log-every",
            "1",
            "--warmup-steps",
            "1",
        ],
    );
    stops_gracefully(
        run,
        stderr_has_a_step,
        &out.join("final").join("model.safetensors"),
    );
}

#[test]
fn train_draft_stops_after_its_step_and_writes_its_final_checkpoint() {
    let (dir, model) = workspace();
    let draft = dir.path().join("draft");
    fixtures::write_tiny_draft(&draft, 2);
    let shard = shards(dir.path(), &model);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "train-draft",
            "--target",
            s(&model),
            "--draft",
            s(&draft),
            "--shards",
            s(&shard),
            "--seq-len",
            "16",
            "--batch-size",
            "2",
            "--steps",
            "100000",
            "--eos-token-id",
            "1",
            "--output",
            s(&out),
            "--log-every",
            "1",
            "--warmup-steps",
            "1",
        ],
    );
    stops_gracefully(run, stderr_has_a_step, &out.join("model.safetensors"));
}

#[test]
fn embed_train_stops_after_its_step_and_saves_the_encoder() {
    let dir = tempfile::tempdir().unwrap();
    let encoder = dir.path().join("encoder");
    fixtures::write_tiny_encoder(&encoder);
    let data = dir.path().join("pairs.jsonl");
    fixtures::write_pair_dataset(&data);
    let out = dir.path().join("out");
    let run = Run::start(
        dir.path(),
        &[
            "embed-train",
            "--model",
            s(&encoder),
            "--dataset",
            s(&data),
            "--output",
            s(&out),
            "--batch-size",
            "2",
            "--epochs",
            "1000",
            "--max-seq-len",
            "8",
            "--log-every",
            "1",
        ],
    );
    let progress = |output: &str| output.lines().any(|line| line.contains("step=1 loss="));
    stops_gracefully(run, progress, &out.join("model.safetensors"));
}
