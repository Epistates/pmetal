//! `pmetal train`: supervised LoRA / QLoRA fine-tuning.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pmetal_core::TrainingCallback;
use pmetal_core::jobs::TrainSpec;
use pmetal_trainer::orchestrator::{self, TrainingJobConfig};

use crate::cli::train::TrainArgs;

/// Ends the run after the step in flight once Ctrl-C has been pressed.
struct StopOnCtrlC(Arc<AtomicBool>);

impl TrainingCallback for StopOnCtrlC {
    fn should_stop(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Watch for Ctrl-C: the first asks the run to stop (it finishes the step,
/// saves the adapter and exits with status 130); a second, because a step or a
/// load can take long, exits at once.
fn watch_for_ctrl_c() -> Arc<AtomicBool> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&interrupted);
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            flag.store(true, Ordering::SeqCst);
            eprintln!("Interrupted: stopping after this step (Ctrl-C again to quit now).");
            if tokio::signal::ctrl_c().await.is_ok() {
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
    let interrupted = watch_for_ctrl_c();
    let stop = Arc::clone(&interrupted);
    super::with_event_log(events, &job_id, |mut ctx| async move {
        ctx.callbacks.push(Box::new(StopOnCtrlC(stop)));
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
