use pmetal_bridge::compat::{
    Array, Exception, FlattenedModuleParam,
    module::{ModuleParameters, ModuleParametersExt},
    nn,
    optimizers::{Optimizer, Updatable},
    transforms,
};
use pmetal_data::PackedTrainingBatch;
use pmetal_lora::TrainableModel;

/// Clip gradients by global L2 norm (GPU-based, lazy).
///
/// Same algorithm as `TrainingLoop::clip_gradients_gpu` but usable from
/// standalone step functions.
fn clip_grads(grads: &mut FlattenedModuleParam, max_norm: f32) {
    pmetal_bridge::training::clip_grad_norm_map(grads, max_norm);
}

/// Training step shared by the JIT step variants.
///
/// Defined at module level so it can access external functions and be used as
/// a function pointer (which is `Copy`).
///
/// When `neftune_alpha` is `Some(alpha)`, NEFTune embedding noise is applied via
/// `model.forward_noised()` instead of the regular `model.forward()`.
pub(crate) fn jit_training_step_inner<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    (input_ids, labels): (&Array, &Array),
    neftune_alpha: Option<f32>,
) -> std::result::Result<Array, Exception> {
    jit_training_step_inner_clipped(state, (input_ids, labels), neftune_alpha, 0.0)
}

/// Inner implementation with optional gradient clipping.
///
/// When `max_grad_norm > 0`, gradients are clipped by global L2 norm before
/// the optimizer step. Pass `0.0` to disable clipping.
pub(crate) fn jit_training_step_inner_clipped<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    (input_ids, labels): (&Array, &Array),
    neftune_alpha: Option<f32>,
    max_grad_norm: f32,
) -> std::result::Result<Array, Exception> {
    let (model, optimizer) = state;

    // Define loss function that will be used by value_and_grad
    let loss_fn = |model: &mut M,
                   (input_ids, labels): (&Array, &Array)|
     -> std::result::Result<Array, Exception> {
        let logits = if let Some(alpha) = neftune_alpha {
            model
                .forward_noised(input_ids, None, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?
        } else {
            model
                .forward(input_ids, None)
                .map_err(|e| Exception::custom(e.to_string()))?
        };

        Ok(pmetal_bridge::training::causal_lm_loss(
            &logits, labels, -100,
        ))
    };

    // Compute loss and gradients
    let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
    let (loss, mut grads) = loss_and_grad_fn(model, (input_ids, labels))?;

    // Clip gradients by global L2 norm
    if max_grad_norm > 0.0 {
        clip_grads(&mut grads, max_grad_norm);
    }

    // Apply gradients via optimizer
    optimizer.update(model, grads)?;

    Ok(loss)
}

/// Training step for packed sequences (variable-length, no padding).
///
/// This version handles packed sequences where multiple sequences are concatenated
/// into a single batch with block-diagonal attention masking and explicit position IDs.
pub(crate) fn jit_training_step_packed<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    packed_batch: &PackedTrainingBatch,
    max_grad_norm: f32,
) -> std::result::Result<Array, Exception> {
    let (model, optimizer) = state;

    // Reshape 1D packed input to 2D [1, total_tokens] for model forward
    let total_tokens = packed_batch.total_tokens as i32;
    let input_ids_2d = packed_batch.input_ids.reshape(&[1, total_tokens]);
    let labels_2d = packed_batch.labels.reshape(&[1, total_tokens]);

    // Keep explicit position IDs so packed-sequence models can still reset RoPE
    // at sequence boundaries when needed.
    let position_ids = packed_batch.position_ids.clone();
    let attn_mask_4d = if packed_batch.num_sequences > 1 {
        // Only materialize the expensive block-diagonal mask when this batch
        // actually contains multiple packed sequences. Single-sequence batches
        // can use the model's native causal path.
        Some(
            packed_batch
                .attention_mask()?
                .reshape(&[1, 1, total_tokens, total_tokens]),
        )
    } else {
        None
    };

    // Define loss function that will be used by value_and_grad
    // Use IDENTICAL loss computation as regular training for consistency
    let loss_fn = |model: &mut M,
                   (input_ids, labels): (&Array, &Array)|
     -> std::result::Result<Array, Exception> {
        // Preserve explicit position IDs while allowing single-sequence packed
        // batches to stay on the model's native causal-attention fast path.
        let logits = model
            .forward_with_positions(input_ids, attn_mask_4d.as_ref(), &position_ids)
            .map_err(|e| Exception::custom(e.to_string()))?;

        Ok(pmetal_bridge::training::causal_lm_loss(
            &logits, labels, -100,
        ))
    };

    // Compute loss and gradients.
    let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
    let (loss, mut grads) = loss_and_grad_fn(model, (&input_ids_2d, &labels_2d))?;

    if max_grad_norm > 0.0 {
        clip_grads(&mut grads, max_grad_norm);
    }

    // Apply gradients via optimizer
    optimizer.update(model, grads)?;

    Ok(loss)
}

/// Shared helper: the CCE loss of the next token at every position.
///
/// Every CCE step calls this, inside the function `value_and_grad`
/// differentiates, with what the model's hidden-state forward returned. The
/// head is read here too, not before the step: an adapter on the LM head is
/// in place as a traced parameter only inside that function, and a head read
/// outside it is a constant, which trained that adapter on nothing.
///
/// `hidden` is `[batch, seq, hidden]`; `labels` is `[batch, seq]`, unshifted.
/// A model that has no hidden-state forward, or no head CCE can use, is an
/// error rather than a quiet switch to the full logits: the caller decided on
/// CCE by asking the model ([`TrainingLoop::cut_cross_entropy_applies`]), so
/// either is a model that answered wrongly.
///
/// [`TrainingLoop::cut_cross_entropy_applies`]: super::TrainingLoop::cut_cross_entropy_applies
pub(crate) fn compute_cce_loss<M: TrainableModel>(
    model: &M,
    hidden: Option<std::result::Result<Array, pmetal_lora::LoraError>>,
    labels: &Array,
) -> std::result::Result<Array, Exception> {
    let hidden_states = hidden
        .ok_or_else(|| {
            Exception::custom("cut cross-entropy: the model has no hidden-state forward")
        })?
        .map_err(|e| Exception::custom(e.to_string()))?;
    let head = model
        .lm_head()
        .ok_or_else(|| Exception::custom("cut cross-entropy: the model has no LM head to use"))?;
    // Shift: hidden[:-1] predicts labels[1:].
    let seq_len = hidden_states.dim(1);
    let shift_hidden = hidden_states.index((.., ..seq_len - 1, ..));
    let shift_labels = labels.index((.., 1..)).reshape(&[-1]);
    head.cut_cross_entropy(&shift_hidden, &shift_labels, -100)
}

/// Training step using Cut Cross-Entropy (avoids materializing the full logits tensor).
///
/// Only for a model [`TrainingLoop::cut_cross_entropy_applies`] said yes to;
/// see [`compute_cce_loss`].
///
/// [`TrainingLoop::cut_cross_entropy_applies`]: super::TrainingLoop::cut_cross_entropy_applies
pub(crate) fn jit_training_step_cce<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    (input_ids, labels): (&Array, &Array),
) -> std::result::Result<Array, Exception> {
    jit_training_step_cce_clipped(state, (input_ids, labels), 0.0)
}

/// CCE training step with optional gradient clipping.
pub(crate) fn jit_training_step_cce_clipped<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    (input_ids, labels): (&Array, &Array),
    max_grad_norm: f32,
) -> std::result::Result<Array, Exception> {
    let (model, optimizer) = state;

    let loss_fn = |model: &mut M,
                   (input_ids, labels): (&Array, &Array)|
     -> std::result::Result<Array, Exception> {
        let hidden = model.forward_hidden(input_ids, None);
        compute_cce_loss(model, hidden, labels)
    };

    let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
    let (loss, mut grads) = loss_and_grad_fn(model, (input_ids, labels))?;
    if max_grad_norm > 0.0 {
        clip_grads(&mut grads, max_grad_norm);
    }
    optimizer.update(model, grads)?;
    Ok(loss)
}

/// Packed-sequence training step using Cut Cross-Entropy.
///
/// Mirrors `jit_training_step_packed` but feeds hidden states through CCE
/// to avoid materializing the full logits tensor, with the same positions and
/// block-diagonal mask. Only for a model
/// [`TrainingLoop::cut_cross_entropy_applies`] said yes to.
///
/// [`TrainingLoop::cut_cross_entropy_applies`]: super::TrainingLoop::cut_cross_entropy_applies
pub(crate) fn jit_training_step_packed_cce<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    packed_batch: &PackedTrainingBatch,
    max_grad_norm: f32,
) -> std::result::Result<Array, Exception> {
    let (model, optimizer) = state;

    let total_tokens = packed_batch.total_tokens as i32;
    let input_ids_2d = packed_batch.input_ids.reshape(&[1, total_tokens]);
    let labels_2d = packed_batch.labels.reshape(&[1, total_tokens]);
    let position_ids = packed_batch.position_ids.clone();
    let attn_mask_4d = if packed_batch.num_sequences > 1 {
        Some(
            packed_batch
                .attention_mask()?
                .reshape(&[1, 1, total_tokens, total_tokens]),
        )
    } else {
        None
    };

    let loss_fn = |model: &mut M,
                   (input_ids, labels): (&Array, &Array)|
     -> std::result::Result<Array, Exception> {
        let hidden =
            model.forward_hidden_with_positions(input_ids, attn_mask_4d.as_ref(), &position_ids);
        compute_cce_loss(model, hidden, labels)
    };

    let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
    let (loss, mut grads) = loss_and_grad_fn(model, (&input_ids_2d, &labels_2d))?;

    if max_grad_norm > 0.0 {
        clip_grads(&mut grads, max_grad_norm);
    }

    optimizer.update(model, grads)?;
    Ok(loss)
}

/// Evaluate all accumulated losses plus model params and optimizer states.
///
/// A single consolidated eval prevents the computation graph from growing
/// unbounded in deferred-eval mode. Evaluating only the losses (as prior
/// code did) leaves params and optimizer states (momentum, velocity) as
/// lazy nodes, causing Metal resource exhaustion on long runs.
pub(crate) fn eval_training_state<M: ModuleParameters, O: Updatable>(
    accumulated_losses: &[Array],
    state: &(M, O),
) -> std::result::Result<(), Exception> {
    let mut all_arrays: Vec<&Array> = accumulated_losses.iter().collect();

    // Keep flattened parameter clones alive while we evaluate the full state.
    let model_params = state.0.flatten_params();
    all_arrays.extend(model_params.values());

    // Optimizer states (momentum, velocity buffers)
    all_arrays.extend(state.1.updatable_states().into_iter());

    if !all_arrays.is_empty() {
        transforms::eval(all_arrays)?;
    }
    Ok(())
}
