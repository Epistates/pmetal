//! The interface's jobs: what is running, and who asked for it.
//!
//! The processes belong to the shared [`pmetal_supervisor::Supervisor`] (the
//! same one the MCP server runs its jobs under): their process groups, their
//! bounded output, their metrics, their shutdown. This module adds what is the
//! interface's own: which tab (or background action) each job belongs to, so an
//! event for a job reaches the right place, and the pump that turns the
//! supervisor's events into interface messages.

use std::collections::HashMap;
use std::path::PathBuf;

use pmetal_supervisor::{Event, JobId, JobRequest, Limits, ProcessStatus, Supervisor};
use tokio::sync::{broadcast, mpsc};

use crate::tui::event::AppMsg;
use crate::tui::tabs::Tab;

/// Who a job belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Owner {
    /// A tab's own job (its form's Start, the inference chat).
    Tab(Tab),
    /// A model download started from the Models tab.
    Download(String),
    /// A dataset conversion started from the Datasets tab.
    Convert,
    /// Fusing a LoRA adapter, started from the Models tab.
    Fuse,
}

/// A command ready to run.
#[derive(Debug, Clone, Default)]
pub struct LaunchPlan {
    /// The `pmetal` subcommand.
    pub subcommand: String,
    /// Its arguments, as the spec spells them.
    pub argv: Vec<String>,
    /// Where it writes its results.
    pub output_dir: Option<PathBuf>,
    /// The `--log-events` file the argv asks for, so the supervisor follows it.
    pub events_file: Option<PathBuf>,
    /// Publish standard output as it arrives (a reply being typed).
    pub deltas: bool,
    /// Show this job's steps on the dashboard.
    pub dashboard: bool,
    /// The model the run starts from, to record next to its output.
    pub base_model: Option<String>,
}

impl LaunchPlan {
    /// Why this binary cannot run the command, if it cannot: `pmetal` is built
    /// with only the commands its features include, and the interface shows a
    /// form for every one of them.
    pub fn missing_command(&self) -> Option<String> {
        use clap::CommandFactory;
        crate::cli::Cli::command()
            .find_subcommand(&self.subcommand)
            .is_none()
            .then(|| {
                format!(
                    "This build of pmetal has no `{}` command. Rebuild with the cargo feature that includes it (for the server, `--features serve`).",
                    self.subcommand
                )
            })
    }

    /// The command line as a person would type it, for the log's first line.
    /// Credentials are hidden.
    pub fn command_line(&self) -> String {
        let mut words = vec!["pmetal".to_string(), self.subcommand.clone()];
        words.extend(self.argv.iter().cloned());
        pmetal_supervisor::redact_argv(&words).join(" ")
    }
}

/// Jobs the interface lets run at once.
pub const MAX_RUNNING: usize = 8;

/// The `pmetal` binary jobs run: the one running this interface, which is
/// `pmetal` itself, unless `PMETAL_BIN` says otherwise or the interface is
/// hosted by another program (then the `pmetal` beside it, or on `PATH`).
pub fn pmetal_binary() -> PathBuf {
    if let Some(bin) = std::env::var_os("PMETAL_BIN") {
        return PathBuf::from(bin);
    }
    if let Ok(exe) = std::env::current_exe() {
        if exe.file_stem().is_some_and(|name| name == "pmetal") {
            return exe;
        }
    }
    pmetal_supervisor::resolve_pmetal_binary()
}

/// The interface's view of the supervisor.
pub struct JobHub {
    supervisor: Supervisor,
    owners: HashMap<JobId, Owner>,
}

impl JobHub {
    /// A hub that runs `binary`, delivering the jobs' events to `tx`.
    /// Must be called from within a tokio runtime.
    pub fn new(binary: impl Into<PathBuf>, tx: mpsc::Sender<AppMsg>) -> Self {
        // A person at a terminal may train, serve, evaluate and chat at once;
        // the machine's memory is the real limit, and the jobs report when it
        // is reached.
        let supervisor = Supervisor::new(
            binary,
            Limits {
                max_running: MAX_RUNNING,
                ..Limits::default()
            },
        );
        spawn_pump(supervisor.subscribe(), tx);
        Self {
            supervisor,
            owners: HashMap::new(),
        }
    }

    pub fn supervisor(&self) -> &Supervisor {
        &self.supervisor
    }

    /// Start a job for `owner`.
    pub fn launch(&mut self, plan: &LaunchPlan, owner: Owner) -> Result<JobId, String> {
        let id = self
            .supervisor
            .spawn(JobRequest {
                subcommand: plan.subcommand.clone(),
                argv: plan.argv.clone(),
                output: plan.output_dir.clone(),
                events_file: plan.events_file.clone(),
                metrics_file: None,
                deltas: plan.deltas,
            })
            .map_err(|e| match e {
                pmetal_supervisor::Error::TooManyRunning { limit, .. } => format!(
                    "{limit} jobs are already running. Stop one, or wait for one to finish."
                ),
                other => other.to_string(),
            })?;
        self.owners.insert(id.clone(), owner);
        // The supervisor forgets old finished jobs; so do we.
        let supervisor = &self.supervisor;
        self.owners.retain(|id, _| supervisor.summary(id).is_ok());
        Ok(id)
    }

    /// Who a job belongs to.
    pub fn owner(&self, id: &str) -> Option<&Owner> {
        self.owners.get(id)
    }

    /// Ask a job to stop (SIGTERM to its group; SIGKILL after a grace period).
    pub fn stop(&self, id: &str) {
        // A job that already ended, or is already stopping, needs nothing.
        let _ = self.supervisor.stop(id);
    }

    /// Whether the job's process is still alive.
    pub fn is_active(&self, id: &str) -> bool {
        self.supervisor
            .summary(id)
            .is_ok_and(|s| s.status.is_active())
    }

    /// How the job's process ended, if it has.
    pub fn outcome(&self, id: &str) -> Option<ProcessStatus> {
        self.supervisor
            .summary(id)
            .ok()
            .map(|s| s.status)
            .filter(|s| !s.is_active())
    }

    /// Jobs whose process is still alive.
    pub fn running(&self) -> usize {
        self.supervisor.running()
    }
}

/// Forward the supervisor's events as interface messages. If the interface
/// falls behind, the broadcast tells the pump how many it missed, and the
/// interface is told to read the jobs' state instead.
fn spawn_pump(mut events: broadcast::Receiver<Event>, tx: mpsc::Sender<AppMsg>) {
    tokio::spawn(async move {
        loop {
            let message = match events.recv().await {
                Ok(event) => AppMsg::Job(event),
                Err(broadcast::error::RecvError::Lagged(_)) => AppMsg::JobsLagged,
                Err(broadcast::error::RecvError::Closed) => return,
            };
            if tx.send(message).await.is_err() {
                return;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn fake_pmetal(dir: &std::path::Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("fake-pmetal");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn plan(sub: &str) -> LaunchPlan {
        LaunchPlan {
            subcommand: sub.into(),
            ..LaunchPlan::default()
        }
    }

    #[tokio::test]
    async fn a_jobs_events_reach_the_interface_and_know_their_owner() {
        let dir = tempfile::tempdir().unwrap();
        let (tx, mut rx) = mpsc::channel(64);
        let mut hub = JobHub::new(fake_pmetal(dir.path(), "echo hello"), tx);
        let id = hub
            .launch(&plan("train"), Owner::Tab(Tab::Training))
            .unwrap();
        assert_eq!(hub.owner(&id), Some(&Owner::Tab(Tab::Training)));
        let mut saw_output = false;
        let mut exited = false;
        while !exited {
            let msg = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .unwrap()
                .unwrap();
            let AppMsg::Job(event) = msg else { continue };
            assert_eq!(event.job, id);
            match event.kind {
                pmetal_supervisor::EventKind::Output(line) => {
                    saw_output |= line.text == "hello";
                }
                pmetal_supervisor::EventKind::Exited(status) => {
                    assert_eq!(status, ProcessStatus::Completed { exit_code: 0 });
                    exited = true;
                }
                _ => {}
            }
        }
        assert!(saw_output);
        assert_eq!(
            hub.outcome(&id),
            Some(ProcessStatus::Completed { exit_code: 0 })
        );
    }

    #[tokio::test]
    async fn a_starting_failure_is_a_message_not_a_panic() {
        let (tx, _rx) = mpsc::channel(8);
        let mut hub = JobHub::new("/no/such/pmetal", tx);
        let err = hub
            .launch(&plan("train"), Owner::Tab(Tab::Training))
            .unwrap_err();
        assert!(err.contains("/no/such/pmetal"), "{err}");
        assert_eq!(hub.running(), 0);
    }

    #[tokio::test]
    async fn the_limit_is_explained_in_the_interfaces_words() {
        let dir = tempfile::tempdir().unwrap();
        let (tx, _rx) = mpsc::channel(8192);
        let mut hub = JobHub::new(fake_pmetal(dir.path(), "sleep 300"), tx);
        for _ in 0..MAX_RUNNING {
            hub.launch(&plan("train"), Owner::Tab(Tab::Training))
                .unwrap();
        }
        let err = hub
            .launch(&plan("train"), Owner::Tab(Tab::Grpo))
            .unwrap_err();
        assert!(err.contains("already running"), "{err}");
        hub.supervisor().shutdown().await;
    }

    /// The metrics the interface shows are read from what the trainer's own
    /// writers write: its JSONL metrics callback (`--log-metrics`) and its
    /// event sink (`--log-events`), not from a copy of their schema.
    #[cfg(feature = "trainer")]
    #[test]
    fn what_the_trainers_writers_write_is_what_the_interface_reads() {
        use pmetal_core::events::{JobEvent, MetricPayload};
        use pmetal_core::{
            EvalMetrics, JsonlSink, StepMetrics, TrainingCallback, TrainingCallbackToSink,
        };
        use pmetal_supervisor::{JobMetrics, parse_record};

        let dir = tempfile::tempdir().unwrap();
        let metrics_path = dir.path().join("metrics.jsonl");
        let events_path = dir.path().join("events.jsonl");
        let mut flat = pmetal_trainer::callbacks::MetricsJsonCallback::new(&metrics_path).unwrap();
        let mut events = TrainingCallbackToSink::new(
            "run",
            JsonlSink::new(std::fs::File::create(&events_path).unwrap()),
        );

        let steps: Vec<StepMetrics> = (1..=3)
            .map(|n| StepMetrics {
                step: n,
                epoch: 1,
                total_epochs: 3,
                total_steps: 30,
                loss: 3.0 / n as f64,
                lr: 2e-4,
                tok_sec: 1500.0 + n as f64,
                total_ms: 90.0,
                gpu_fwd_bwd_ms: 70.0,
                optimizer_ms: 12.0,
                grad_norm: Some(0.5),
                ..StepMetrics::default()
            })
            .collect();
        let eval = EvalMetrics {
            loss: 1.25,
            perplexity: 3.49,
            ..EvalMetrics::default()
        };
        for writer in [&mut flat as &mut dyn TrainingCallback, &mut events] {
            writer.on_train_start();
            for step in &steps {
                writer.on_step_end_with_metrics(step);
            }
            writer.on_epoch_end(1, &eval);
            writer.on_train_end();
        }
        drop(flat);
        drop(events);

        let read = |path: &std::path::Path| {
            let mut metrics = JobMetrics::default();
            let mut read_steps = Vec::new();
            for line in std::fs::read_to_string(path).unwrap().lines() {
                if let Some(event) = parse_record(line, "job") {
                    metrics.apply(&event);
                    if let JobEvent::Metric {
                        payload: MetricPayload::Step(step),
                        ..
                    } = event
                    {
                        read_steps.push(step);
                    }
                }
            }
            (metrics, read_steps)
        };
        for (what, path) in [
            ("metrics file", &metrics_path),
            ("event file", &events_path),
        ] {
            let (metrics, read_steps) = read(path);
            assert_eq!(read_steps.len(), 3, "{what}");
            assert_eq!(metrics.last_step, Some(3), "{what}");
            assert_eq!(metrics.total_steps, Some(30), "{what}");
            assert_eq!(metrics.last_loss, Some(1.0), "{what}");
            assert_eq!(metrics.last_lr, Some(2e-4), "{what}");
            assert_eq!(metrics.tokens_per_second, Some(1503.0), "{what}");
            assert_eq!(metrics.grad_norm, Some(0.5), "{what}");
            assert_eq!(metrics.eval_loss, Some(1.25), "{what}");
            // The trainer counts epochs from 0; the summary from 1.
            assert_eq!(metrics.epoch, Some(2.0), "{what}");
            assert_eq!(metrics.total_epochs, Some(3), "{what}");
            assert_eq!(read_steps[0].step, 1, "{what}");
        }
        // The event stream carries the timing the metrics file never had.
        let (_, from_events) = read(&events_path);
        assert_eq!(from_events[0].gpu_fwd_bwd_ms, 70.0);
        assert_eq!(from_events[0].optimizer_ms, 12.0);
    }

    #[test]
    fn a_command_this_build_lacks_is_named_as_missing() {
        let mut plan = plan("train");
        // `train` needs the trainer feature, `info` is always there.
        plan.subcommand = "info".into();
        assert_eq!(plan.missing_command(), None);
        plan.subcommand = "no-such-command".into();
        let why = plan.missing_command().unwrap();
        assert!(why.contains("`no-such-command`"), "{why}");
    }

    #[test]
    fn the_command_line_hides_credentials() {
        let plan = LaunchPlan {
            subcommand: "serve".into(),
            argv: vec![
                "--model".into(),
                "m".into(),
                "--api-key".into(),
                "sk-secret".into(),
            ],
            ..LaunchPlan::default()
        };
        let line = plan.command_line();
        assert!(line.starts_with("pmetal serve --model m"), "{line}");
        assert!(!line.contains("sk-secret"), "{line}");
    }
}
