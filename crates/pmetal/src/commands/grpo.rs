use pmetal_core::jobs::GrpoSpec;
use pmetal_trainer::grpo::{GrpoJobOptions, run_grpo_job};

use super::RunContext;

/// The banner `pmetal grpo` opens with: what the run was asked to do.
fn print_banner(spec: &GrpoSpec) {
    println!("========================================");
    println!("  PMetal GRPO Reasoning Training");
    println!("========================================");
    println!("Model:         {}", spec.model);
    println!("Dataset:       {}", spec.dataset);
    println!("Output:        {}", spec.output_dir);
    println!("Generations:   {}", spec.num_generations);
    println!("Beta:          {}", spec.beta);
    println!("LR:            {:.2e}", spec.learning_rate);
    println!("DAPO:          {}", spec.dapo);
    println!("Loss type:     {}", spec.loss_type);
    println!("Updates/batch: {}", spec.num_iterations);
    println!("Reasoning Rew: {}", spec.reasoning_rewards);
    if spec.vlm {
        println!(
            "VLM Mode:      enabled (max_image_size={})",
            spec.max_image_size
        );
    }
    println!("========================================\n");
}

/// Run GRPO (Group Relative Policy Optimization) for reasoning models.
///
/// The run itself is `pmetal_trainer::grpo::run_grpo_job`, which every surface
/// shares; the banner and the closing line are the command line's own. A run
/// a stop request ended is saved like a finished one; the command exits with
/// the interrupted status afterwards (see
/// [`with_run_context`](super::with_run_context)).
pub(crate) async fn run_grpo(spec: GrpoSpec, ctx: RunContext) -> anyhow::Result<()> {
    let RunContext {
        emit_console_output,
        callbacks,
    } = ctx;
    if emit_console_output {
        print_banner(&spec);
    }
    let output_dir = spec.output_dir.clone();
    run_grpo_job(
        spec,
        GrpoJobOptions {
            callbacks,
            extra_rewards: Vec::new(),
        },
    )
    .await?;
    if emit_console_output {
        println!("\nGRPO training complete! Model weights saved to: {output_dir}");
    }
    Ok(())
}
