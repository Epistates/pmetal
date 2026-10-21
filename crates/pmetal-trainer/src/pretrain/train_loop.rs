//! Full-parameter pretraining loop with gradient accumulation, LR scheduling,
//! gradient clipping, periodic eval, and step logging.
//!
//! A step takes the shared machinery of the fine-tuning loop: micro-batches
//! add up in a `GradAccumulator` (the step's gradient is the mean over all
//! its tokens, and the one multiplication that divides by the count also
//! clips), and the loss, gradient norm, every parameter and the optimizer's
//! state are evaluated in one call.

use super::init::{apply_depth_scaled_init, zero_biases};
use super::loss::causal_lm_loss;
use super::{CausalLm, PretrainBatch, PretrainConfig, PretrainError, document_mask};

use pmetal_core::LearningRateScheduler;
use pmetal_data::streaming::StreamPosition;
use pmetal_runtime::compat::{
    Array, Exception, ModuleParametersExt,
    nn::value_and_grad,
    optimizers::{AdamW, AdamWBuilder, Optimizer, Updatable},
    transforms,
};

use crate::training_loop::step::{GradAccumulator, MicroOutput};

/// Compute forward + loss + grad for one micro-batch, as the sum of its
/// counted tokens' losses with the count. Does NOT step the optimizer.
fn forward_backward<M: CausalLm>(
    model: &mut M,
    input_ids: &Array,
    mask: Option<&Array>,
    ignore_index: Option<i32>,
    z_loss_coef: Option<f32>,
) -> Result<MicroOutput, PretrainError> {
    // The tokens a next-token loss scores: every position but the first,
    // less the ignored ones.
    let count = match ignore_index {
        Some(ignore) => input_ids
            .index((.., 1..))
            .ne(&Array::from_int(ignore))
            .as_type::<f32>()
            .sum(None)
            .stop_gradient(),
        None => Array::from_f32((input_ids.dim(0) * (input_ids.dim(1) - 1)) as f32),
    };
    let loss_count = count.clone();
    let mut vag = value_and_grad(
        move |model: &mut M, batch: &Array| -> Result<Array, Exception> {
            let logits = model.forward_logits_masked(batch, mask)?;
            // The loss is a mean over the counted tokens; times the count it is the sum.
            Ok(causal_lm_loss(&logits, batch, ignore_index, z_loss_coef)?.multiply(&loss_count))
        },
    );
    let (loss_sum, grads) =
        vag(model, input_ids).map_err(|e| PretrainError::Autograd(e.to_string()))?;
    Ok(MicroOutput {
        loss_sum,
        count,
        grads,
    })
}

/// One optimizer step with gradient accumulation over `micro_batches`.
///
/// Returns the mean loss across the accumulated micro-batches' tokens.
pub fn pretrain_step<M: CausalLm>(
    model: &mut M,
    optimizer: &mut AdamW,
    micro_batches: &[Array],
    ignore_index: Option<i32>,
    z_loss_coef: Option<f32>,
    max_grad_norm: Option<f32>,
) -> Result<f32, PretrainError> {
    let batches: Vec<PretrainBatch> = micro_batches
        .iter()
        .map(|tokens| PretrainBatch::new(tokens.clone()))
        .collect();
    pretrain_step_batches(
        model,
        optimizer,
        &batches,
        ignore_index,
        z_loss_coef,
        max_grad_norm,
    )
}

/// [`pretrain_step`] over batches that may carry the documents each row is
/// made of, in which case attention is masked to each document.
pub fn pretrain_step_batches<M: CausalLm>(
    model: &mut M,
    optimizer: &mut AdamW,
    batches: &[PretrainBatch],
    ignore_index: Option<i32>,
    z_loss_coef: Option<f32>,
    max_grad_norm: Option<f32>,
) -> Result<f32, PretrainError> {
    assert!(!batches.is_empty(), "pretrain_step: empty micro_batches");

    let mut accumulator = GradAccumulator::default();
    for batch in batches {
        let mask = match &batch.doc_lengths {
            Some(docs) => Some(document_mask(docs, batch.tokens.dim(1) as usize)?),
            None => None,
        };
        let out = forward_backward(
            model,
            &batch.tokens,
            mask.as_ref(),
            ignore_index,
            z_loss_coef,
        )?;
        accumulator.add(out);
        // Evaluating what is held frees each micro-batch's graph; the last
        // one is evaluated with the step.
        if accumulator.micro_batches() < batches.len() {
            accumulator
                .eval_held(&[])
                .map_err(|e| PretrainError::Autograd(e.to_string()))?;
        }
    }
    let finished = accumulator
        .finish(max_grad_norm.unwrap_or(0.0))
        .expect("at least one micro-batch was added");

    optimizer
        .update(model, finished.grads)
        .map_err(|e| PretrainError::Optimizer(e.to_string()))?;

    // One evaluation: the loss, the norm, every parameter, the optimizer's state.
    let params = model.flatten_trainable_params();
    let mut arrays: Vec<&Array> = vec![&finished.loss];
    arrays.extend(finished.grad_norm.iter());
    arrays.extend(params.values());
    arrays.extend(optimizer.updatable_states());
    transforms::eval(arrays).map_err(|e| PretrainError::Autograd(e.to_string()))?;
    Ok(finished.loss.item_f32())
}

/// Compute eval loss over `n_batches` from the eval iterator (forward only,
/// no grad). Returns mean loss.
pub fn eval_loss<M: CausalLm, I: Iterator<Item = Array>>(
    model: &mut M,
    eval_batches: &mut I,
    n_batches: usize,
    ignore_index: Option<i32>,
) -> Result<f32, PretrainError> {
    let mut total = 0.0f32;
    let mut count = 0;
    for _ in 0..n_batches {
        let batch = match eval_batches.next() {
            Some(b) => b,
            None => break,
        };
        let logits = model
            .forward_logits(&batch)
            .map_err(|e| PretrainError::Autograd(e.to_string()))?;
        let loss = causal_lm_loss(&logits, &batch, ignore_index, None)
            .map_err(|e| PretrainError::Autograd(e.to_string()))?;
        loss.eval();
        total += loss.item::<f32>();
        count += 1;
    }
    if count == 0 {
        return Ok(f32::NAN);
    }
    Ok(total / count as f32)
}

/// Run full pretraining with gradient accumulation, logging, eval, and
/// checkpointing.
pub fn run_pretrain<M, I>(
    model: &mut M,
    config: &PretrainConfig,
    batches: I,
) -> Result<Vec<f32>, PretrainError>
where
    M: CausalLm,
    I: Iterator<Item = Array>,
{
    let optimizer = build_optimizer(config)?;
    run_pretrain_with_state(
        model,
        config,
        batches.map(|tokens| Ok(PretrainBatch::new(tokens))),
        optimizer,
        0,
        Option::<std::iter::Empty<Array>>::None,
    )
}

fn build_optimizer(config: &PretrainConfig) -> Result<AdamW, PretrainError> {
    AdamWBuilder::new(config.learning_rate)
        .weight_decay(config.weight_decay)
        .betas(config.betas)
        .eps(config.eps)
        .build()
        .map_err(|e| PretrainError::Optimizer(e.to_string()))
}

/// Run pretraining from an explicit optimizer and global step.
///
/// `batches` yields `Result`s: a corrupt shard is an error that stops the
/// run, not the end of the data. A batch may carry a streaming position,
/// which is written into checkpoints so resume can restart at the exact
/// corpus boundary, and the documents its rows are made of, which attention
/// is then masked to.
pub fn run_pretrain_with_state<M, I, E>(
    model: &mut M,
    config: &PretrainConfig,
    mut batches: I,
    mut optimizer: AdamW,
    start_step: usize,
    mut eval_batches: Option<E>,
) -> Result<Vec<f32>, PretrainError>
where
    M: CausalLm,
    I: Iterator<Item = Result<PretrainBatch, PretrainError>>,
    E: Iterator<Item = Array>,
{
    config.validate()?;
    if config.apply_init && config.n_layers > 0 {
        apply_depth_scaled_init(model, config.n_layers)
            .map_err(|e| PretrainError::Autograd(e.to_string()))?;
        zero_biases(model);
    }

    let scheduler = LearningRateScheduler::new(
        config.learning_rate as f64,
        config.num_steps,
        config.warmup_steps,
        config.lr_schedule,
    )
    .with_min_lr(config.min_lr as f64);

    let gas = config.gradient_accumulation_steps.max(1);
    let remaining_steps = config.num_steps.saturating_sub(start_step);
    let mut losses = Vec::with_capacity(remaining_steps);
    let start = std::time::Instant::now();
    let mut last_stream_position: Option<StreamPosition> = None;

    for step in start_step..config.num_steps {
        let lr = scheduler.get_lr(step) as f32;
        optimizer.set_lr(lr);

        // Collect micro-batches for gradient accumulation
        let mut micro: Vec<PretrainBatch> = Vec::with_capacity(gas);
        for _ in 0..gas {
            let batch = batches
                .next()
                .ok_or(PretrainError::BatchIteratorExhausted { step })??;
            if batch.position.is_some() {
                last_stream_position = batch.position;
            }
            micro.push(batch);
        }

        let loss = pretrain_step_batches(
            model,
            &mut optimizer,
            &micro,
            config.ignore_index,
            config.z_loss_coef,
            config.max_grad_norm,
        )?;
        crate::step_check::check_step(step + 1, loss)
            .map_err(|e| PretrainError::Autograd(e.to_string()))?;
        losses.push(loss);

        // Logging
        if config.log_every > 0 && (step + 1) % config.log_every == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let tokens_per_batch = micro
                .first()
                .map(|b| b.tokens.dim(0) * b.tokens.dim(1))
                .unwrap_or(0);
            let total_tokens =
                (step + 1 - start_step) as f64 * gas as f64 * tokens_per_batch as f64;
            let tok_per_sec = total_tokens / elapsed;
            eprintln!(
                "step {:>6} | loss {:.4} | lr {:.2e} | {:.0} tok/s",
                step + 1,
                loss,
                lr,
                tok_per_sec,
            );
        }

        if config.eval_every > 0 && (step + 1) % config.eval_every == 0 {
            if let Some(eval_iter) = eval_batches.as_mut() {
                let eval = eval_loss(model, eval_iter, config.eval_batches, config.ignore_index)?;
                eprintln!("step {:>6} | eval_loss {:.4}", step + 1, eval);
            }
        }

        // Checkpoint
        if let (Some(every), Some(dir)) = (config.checkpoint_every, config.checkpoint_dir.as_ref())
        {
            if every > 0 && (step + 1) % every == 0 {
                let step_dir = dir.join(format!("step_{}", step + 1));
                super::save_checkpoint(
                    &step_dir,
                    model,
                    &optimizer,
                    &super::CheckpointMeta {
                        step: optimizer.step_count(),
                        loss,
                        learning_rate: lr,
                        stream_position: last_stream_position,
                    },
                )?;
                eprintln!("checkpoint saved: {}", step_dir.display());
            }
        }

        if config
            .stop
            .as_ref()
            .is_some_and(|stop| stop.load(std::sync::atomic::Ordering::SeqCst))
        {
            eprintln!(
                "stop requested: ending after step {} of {}",
                step + 1,
                config.num_steps
            );
            break;
        }
    }

    if let (Some(dir), Some(&loss)) = (config.checkpoint_dir.as_ref(), losses.last()) {
        // The step the run ended on: `num_steps` for a finished run, earlier
        // for one a stop request ended. A periodic checkpoint already holds
        // that step when it is a multiple of the interval.
        let last_step = start_step + losses.len();
        let should_save_final = match config.checkpoint_every {
            Some(every) if every > 0 => last_step % every != 0,
            _ => true,
        };
        if should_save_final {
            let final_dir = dir.join("final");
            super::save_checkpoint(
                &final_dir,
                model,
                &optimizer,
                &super::CheckpointMeta {
                    step: optimizer.step_count(),
                    loss,
                    learning_rate: scheduler.get_lr(last_step.saturating_sub(1)) as f32,
                    stream_position: last_stream_position,
                },
            )?;
            eprintln!("final checkpoint saved: {}", final_dir.display());
        }
    }
    Ok(losses)
}
