//! `pmetal train`: supervised LoRA / QLoRA fine-tuning.

use std::path::Path;

use pmetal_core::jobs::{Explicit, TrainSpec};
use pmetal_trainer::orchestrator::{self, TrainingJobConfig};

use crate::cli::train::TrainArgs;

/// Run `pmetal train`. `events` is `--log-events`.
///
/// A request to stop (Ctrl-C, SIGTERM) ends the run after the step in flight,
/// saves the adapter and exits with status 130 ([`super::with_run_context`]).
///
/// `given` is the flags the command line gave, as opposed to left at their
/// defaults: with `--config`, a setting is the flag's if given, else the
/// file's, else the default.
pub(crate) async fn run_train(
    args: TrainArgs,
    events: Option<&Path>,
    given: std::collections::BTreeSet<String>,
) -> anyhow::Result<()> {
    let mut spec = TrainSpec::from(args);
    spec.normalize()
        .map_err(|errors| super::invalid_spec::<TrainSpec>(&errors))?;
    let mut job = TrainingJobConfig::from_spec(&spec).map_err(anyhow::Error::msg)?;
    job.explicit = Explicit::Only(given);
    let job_id = job.model_id.clone();
    super::with_run_context(events, &job_id, |ctx| async move {
        orchestrator::run_training(job, None, ctx.callbacks)
            .await
            .map(|_| ())
    })
    .await
}
