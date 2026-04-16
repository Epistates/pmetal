//! Full-parameter pretraining loop with gradient accumulation, LR scheduling,
//! gradient clipping, periodic eval, and step logging.

use std::rc::Rc;

use super::init::{apply_depth_scaled_init, zero_biases};
use super::loss::causal_lm_loss;
use super::{CausalLm, PretrainConfig, PretrainError};

use pmetal_bridge::compat::{
    Array, Exception,
    module::FlattenedModuleParam,
    nn::value_and_grad,
    ops,
    optimizers::{AdamW, AdamWBuilder, Optimizer},
};
use pmetal_core::LearningRateScheduler;

fn clip_grad_norm(
    mut grads: FlattenedModuleParam,
    max_norm: f32,
) -> Result<FlattenedModuleParam, Exception> {
    let mut total_norm_sq = Array::from_f32(0.0);
    for grad in grads.values() {
        total_norm_sq = total_norm_sq.add(&grad.multiply(grad).sum(None));
    }
    let total_norm = total_norm_sq.sqrt();
    let max_norm_arr = Array::from_f32(max_norm);
    let clip_coef = max_norm_arr.divide(&ops::maximum(&total_norm, &max_norm_arr));
    for grad in grads.values_mut() {
        *grad = grad.multiply(&clip_coef);
    }
    Ok(grads)
}

/// Accumulate `rhs` into `acc`, creating entries on first use.
fn accumulate_grads(acc: &mut FlattenedModuleParam, rhs: &FlattenedModuleParam) {
    for (key, grad) in rhs {
        acc.entry(key.clone())
            .and_modify(|existing| *existing = existing.add(grad))
            .or_insert_with(|| grad.clone());
    }
}

/// Scale every gradient by `1 / n`.
fn scale_grads(grads: &mut FlattenedModuleParam, n: f32) {
    let inv = Array::from_f32(1.0 / n);
    for grad in grads.values_mut() {
        *grad = grad.multiply(&inv);
    }
}

/// Compute forward + loss + grad for one micro-batch. Does NOT step the
/// optimizer — the caller accumulates and steps after K micro-batches.
fn forward_backward<M: CausalLm>(
    model: &mut M,
    input_ids: &Array,
    ignore_index: Option<i32>,
    z_loss_coef: Option<f32>,
) -> Result<(f32, FlattenedModuleParam), PretrainError> {
    let mut vag = value_and_grad(|model: &mut M, batch: &Array| -> Result<Array, Exception> {
        let logits = model.forward_logits(batch)?;
        causal_lm_loss(&logits, batch, ignore_index, z_loss_coef)
    });
    let (loss, grads) =
        vag(model, input_ids).map_err(|e| PretrainError::Autograd(e.to_string()))?;
    loss.eval();
    Ok((loss.item::<f32>(), grads))
}

/// One optimizer step with gradient accumulation over `micro_batches`.
///
/// Returns the mean loss across the accumulated micro-batches.
pub fn pretrain_step<M: CausalLm>(
    model: &mut M,
    optimizer: &mut AdamW,
    micro_batches: &[Array],
    ignore_index: Option<i32>,
    z_loss_coef: Option<f32>,
    max_grad_norm: Option<f32>,
) -> Result<f32, PretrainError> {
    let k = micro_batches.len();
    assert!(k > 0, "pretrain_step: empty micro_batches");

    let mut acc_grads = FlattenedModuleParam::new();
    let mut total_loss = 0.0f32;

    for batch in micro_batches {
        let (loss, grads) = forward_backward(model, batch, ignore_index, z_loss_coef)?;
        total_loss += loss;
        accumulate_grads(&mut acc_grads, &grads);
    }

    if k > 1 {
        scale_grads(&mut acc_grads, k as f32);
    }

    let acc_grads = if let Some(max_norm) = max_grad_norm {
        clip_grad_norm(acc_grads, max_norm).map_err(|e| PretrainError::Autograd(e.to_string()))?
    } else {
        acc_grads
    };

    optimizer
        .update(model, acc_grads)
        .map_err(|e| PretrainError::Optimizer(e.to_string()))?;

    Ok(total_loss / k as f32)
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
    mut batches: I,
) -> Result<Vec<f32>, PretrainError>
where
    M: CausalLm,
    I: Iterator<Item = Array>,
{
    if config.apply_init && config.n_layers > 0 {
        apply_depth_scaled_init(model, config.n_layers)
            .map_err(|e| PretrainError::Autograd(e.to_string()))?;
        zero_biases(model);
    }

    let mut optimizer = AdamWBuilder::new(config.learning_rate)
        .weight_decay(config.weight_decay)
        .betas(config.betas)
        .eps(config.eps)
        .build()
        .map_err(|e| PretrainError::Optimizer(e.to_string()))?;

    let scheduler = LearningRateScheduler::new(
        config.learning_rate as f64,
        config.num_steps,
        config.warmup_steps,
        config.lr_schedule,
    )
    .with_min_lr(config.min_lr as f64);

    let gas = config.gradient_accumulation_steps.max(1);
    let mut losses = Vec::with_capacity(config.num_steps);
    let start = std::time::Instant::now();

    for step in 0..config.num_steps {
        let lr = scheduler.get_lr(step) as f32;
        optimizer.set_lr(lr);

        // Collect micro-batches for gradient accumulation
        let mut micro: Vec<Array> = Vec::with_capacity(gas);
        for _ in 0..gas {
            let batch = batches
                .next()
                .ok_or(PretrainError::BatchIteratorExhausted { step })?;
            micro.push(batch);
        }

        let loss = pretrain_step(
            model,
            &mut optimizer,
            &micro,
            config.ignore_index,
            config.z_loss_coef,
            config.max_grad_norm,
        )?;
        losses.push(loss);

        // Logging
        if config.log_every > 0 && (step + 1) % config.log_every == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let tokens_per_batch = micro.first().map(|b| b.dim(0) * b.dim(1)).unwrap_or(0);
            let total_tokens = (step + 1) as f64 * gas as f64 * tokens_per_batch as f64;
            let tok_per_sec = total_tokens / elapsed;
            eprintln!(
                "step {:>6} | loss {:.4} | lr {:.2e} | {:.0} tok/s",
                step + 1,
                loss,
                lr,
                tok_per_sec,
            );
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
                    },
                )?;
                eprintln!("checkpoint saved: {}", step_dir.display());
            }
        }
    }
    Ok(losses)
}
