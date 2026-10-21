//! `pmetal train`: supervised LoRA / QLoRA fine-tuning.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pmetal_core::TrainingCallback;
use pmetal_core::jobs::TrainSpec;
use pmetal_trainer::orchestrator::{self, TrainingJobConfig};

use crate::cli::train::TrainArgs;

/// Ends the run after the step in flight once a stop has been requested.
struct StopOnSignal(Arc<AtomicBool>);

impl TrainingCallback for StopOnSignal {
    fn should_stop(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Resolves when the process is asked to stop: Ctrl-C (SIGINT), or SIGTERM,
/// which is how the job supervisor (the terminal UI, the MCP server, the Python
/// bindings) stops a job.
async fn stop_requested() -> bool {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let Ok(mut term) = signal(SignalKind::terminate()) else {
            return tokio::signal::ctrl_c().await.is_ok();
        };
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.is_ok(),
            _ = term.recv() => true,
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.is_ok()
    }
}

/// Watch for a request to stop: the first asks the run to stop (it finishes the
/// step, saves the adapter and exits with status 130); a second, because a step
/// or a load can take long, exits at once.
fn watch_for_stop() -> Arc<AtomicBool> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&interrupted);
    tokio::spawn(async move {
        if stop_requested().await {
            flag.store(true, Ordering::SeqCst);
            eprintln!("Interrupted: stopping after this step (Ctrl-C again to quit now).");
            if stop_requested().await {
                std::process::exit(130);
            }
        }
    });
    interrupted
}

/// Run `pmetal train`. `events` is `--log-events`.
pub(crate) async fn run_train(args: TrainArgs, events: Option<&Path>) -> anyhow::Result<()> {
    let mut spec = TrainSpec::from(args);
    spec.normalize()
        .map_err(|errors| super::invalid_spec::<TrainSpec>(&errors))?;
    let job = TrainingJobConfig::from_spec(&spec).map_err(anyhow::Error::msg)?;
    let job_id = job.model_id.clone();
    let interrupted = watch_for_stop();
    let stop = Arc::clone(&interrupted);
    super::with_event_log(events, &job_id, |mut ctx| async move {
        ctx.callbacks.push(Box::new(StopOnSignal(stop)));
        orchestrator::run_training(job, None, ctx.callbacks)
            .await
            .map(|_| ())
    })
    .await?;
    if interrupted.load(Ordering::SeqCst) {
        // The adapter is saved; the status says the run did not finish.
        return Err(crate::Interrupted.into());
    }
    Ok(())
}
