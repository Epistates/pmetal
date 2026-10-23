//! `pmetal pretrain`: full-parameter pretraining from scratch (no LoRA).

use pmetal_core::TrainingCallback;
use pmetal_data::streaming::{StreamConfig, StreamingShardReader};
use pmetal_runtime::compat::random;
use pmetal_trainer::CompositeCallback;
use pmetal_trainer::pretrain::{self, PretrainConfig};

use super::RunContext;
use crate::cli::pretrain::PretrainArgs;

/// Run `pmetal pretrain`.
///
/// A request to stop (Ctrl-C, SIGTERM) ends the run after the step in flight,
/// saves it and exits with status 130 ([`super::with_run_context`]).
pub(crate) async fn run_pretrain(args: PretrainArgs, ctx: RunContext) -> anyhow::Result<()> {
    let PretrainArgs {
        arch,
        shards,
        eval_shards,
        seq_len,
        batch_size,
        steps,
        learning_rate,
        min_lr,
        warmup_steps,
        lr_schedule,
        weight_decay,
        max_grad_norm,
        eos_token_id,
        output,
        checkpoint_every,
        resume,
        model_config: _model_config,
        z_loss,
        document_mask,
        gradient_accumulation_steps,
        log_every,
        eval_every,
        eval_batches,
        seed,
    } = args;

    random::seed(seed);
    let _output_dir = pmetal_core::paths::output_path(&output, "pretrain output")?;

    // Collect shard paths (each CLI value is a literal path)
    let shard_paths: Vec<std::path::PathBuf> =
        shards.iter().map(std::path::PathBuf::from).collect();
    anyhow::ensure!(!shard_paths.is_empty(), "no shard files specified");
    println!(
        "Pretraining on {} shards, {} steps",
        shard_paths.len(),
        steps
    );

    // Parse LR schedule
    let lr_sched = lr_schedule
        .parse::<pmetal_core::LrSchedulerType>()
        .map_err(|e| anyhow::anyhow!("--lr-schedule: {e}"))?;

    // Build model via factory (supports llama, qwen2, qwen3, gemma, gemma4, mistral, phi3, gpt-oss)
    use pmetal_runtime::compat::module::ModuleParameters;
    let config_path = _model_config.as_deref().map(std::path::Path::new);
    let mut model = pretrain::create_model(&arch, config_path)?;
    let n_layers = pretrain::n_layers(&model);
    println!(
        "Architecture: {arch}, layers: {n_layers}, params: {}",
        model.num_parameters()
    );

    let config = PretrainConfig {
        num_steps: steps,
        learning_rate,
        min_lr,
        warmup_steps,
        lr_schedule: lr_sched,
        weight_decay,
        betas: (0.9, 0.95),
        eps: 1e-8,
        max_grad_norm: if max_grad_norm > 0.0 {
            Some(max_grad_norm)
        } else {
            None
        },
        ignore_index: None,
        z_loss_coef: if z_loss > 0.0 { Some(z_loss) } else { None },
        n_layers,
        apply_init: resume.is_none(),
        checkpoint_every: if checkpoint_every > 0 {
            Some(checkpoint_every)
        } else {
            None
        },
        checkpoint_dir: Some(std::path::PathBuf::from(&output)),
        gradient_accumulation_steps,
        log_every,
        eval_every,
        eval_batches,
        stop: None,
    };

    let mut optimizer = pmetal_runtime::compat::optimizers::AdamWBuilder::new(learning_rate)
        .weight_decay(weight_decay)
        .betas((0.9, 0.95))
        .eps(1e-8)
        .build()?;
    let mut start_step = 0usize;
    let mut resume_position = None;

    // Resume from checkpoint if requested
    if let Some(ref ckpt_dir) = resume {
        let meta =
            pretrain::load_checkpoint(std::path::Path::new(ckpt_dir), &mut model, &mut optimizer)?;
        start_step = meta.step as usize;
        resume_position = meta.stream_position;
        println!("Resumed from step {}, loss {:.4}", meta.step, meta.loss);
    }

    // Streaming data pipeline
    let stream_config = StreamConfig {
        shard_paths,
        seq_len,
        batch_size,
        eos_token_id,
        resume_from: resume_position,
    };
    let mut reader = StreamingShardReader::new(stream_config)?;
    if resume.is_some() && resume_position.is_none() && start_step > 0 {
        let batches_to_skip = start_step.saturating_mul(gradient_accumulation_steps.max(1));
        for _ in 0..batches_to_skip {
            reader
                .try_next_batch()?
                .ok_or_else(|| anyhow::anyhow!("stream exhausted while seeking resume step"))?;
        }
    }

    // Convert streaming batches to MLX arrays. A shard that cannot be
    // read is an error that stops the run, not the end of the data.
    let batch_iter = std::iter::from_fn(move || {
        let (rows, pos) = match reader.try_next_rows() {
            Ok(Some(batch)) => batch,
            Ok(None) => return None,
            Err(e) => return Some(Err(pretrain::PretrainError::Data(e.to_string()))),
        };
        let flat: Vec<i32> = rows
            .iter()
            .flat_map(|row| row.tokens.iter().map(|&t| t as i32))
            .collect();
        let tokens = pmetal_runtime::InlineArray::from_i32_slice_shaped(
            &flat,
            &[rows.len() as i32, seq_len as i32],
        );
        Some(Ok(pretrain::PretrainBatch {
            tokens,
            position: Some(pos),
            doc_lengths: document_mask
                .then(|| rows.into_iter().map(|row| row.doc_lengths).collect()),
        }))
    });

    let eval_iter = if !eval_shards.is_empty() && eval_every > 0 {
        let eval_paths: Vec<std::path::PathBuf> =
            eval_shards.iter().map(std::path::PathBuf::from).collect();
        let eval_reader = StreamingShardReader::new(StreamConfig {
            shard_paths: eval_paths,
            seq_len,
            batch_size,
            eos_token_id,
            resume_from: None,
        })?;
        Some(eval_reader.map(move |(batch, _pos)| {
            let flat: Vec<i32> = batch
                .iter()
                .flat_map(|seq| seq.iter().map(|&t| t as i32))
                .collect();
            pmetal_runtime::InlineArray::from_i32_slice_shaped(
                &flat,
                &[batch.len() as i32, seq_len as i32],
            )
        }))
    } else {
        None
    };

    // Run
    let mut callbacks = CompositeCallback::from(ctx.callbacks);
    let losses = pretrain::run_pretrain_with_state(
        &mut model,
        &config,
        batch_iter,
        optimizer,
        start_step,
        eval_iter,
        &mut callbacks,
    )?;

    // Print summary
    if !losses.is_empty() {
        let last = losses.last().copied().unwrap_or(0.0);
        println!(
            "Pretraining {}: {} steps, final loss {:.4}",
            if callbacks.should_stop() {
                "stopped on request"
            } else {
                "complete"
            },
            losses.len(),
            last,
        );
    }
    Ok(())
}
