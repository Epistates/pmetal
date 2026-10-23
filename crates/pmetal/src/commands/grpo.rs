use pmetal_core::jobs::GrpoSpec;
use pmetal_trainer::grpo::{GrpoJobOptions, run_grpo_job};

use super::RunContext;

/// Run GRPO (Group Relative Policy Optimization) for reasoning models.
///
/// The run itself is `pmetal_trainer::grpo::run_grpo_job`, which every surface
/// shares. A run a stop request ended is saved like a finished one; the
/// command exits with the interrupted status afterwards (see
/// [`with_run_context`](super::with_run_context)).
pub(crate) async fn run_grpo(spec: GrpoSpec, ctx: RunContext) -> anyhow::Result<()> {
    let RunContext {
        emit_console_output,
        callbacks,
    } = ctx;
    run_grpo_job(
        spec,
        GrpoJobOptions {
            emit_console_output,
            callbacks,
            extra_rewards: Vec::new(),
        },
    )
    .await?;
    Ok(())
}
