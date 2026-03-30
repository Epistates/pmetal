use pmetal_bridge::compat::{
    Array, Exception,
    module::{ModuleParameters, ModuleParametersExt},
    nn, ops,
    optimizers::{Optimizer, Updatable},
    transforms,
};
use pmetal_data::PackedTrainingBatch;
use pmetal_lora::TrainableModel;

/// JIT-compiled training step for maximum throughput.
///
/// This function is defined at module level so it can access external functions
/// and be used as a function pointer (which is `Copy`).
///
/// When `neftune_alpha` is `Some(alpha)`, NEFTune embedding noise is applied via
/// `model.forward_noised()` instead of the regular `model.forward()`.
pub(crate) fn jit_training_step<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    (input_ids, labels): (&Array, &Array),
) -> std::result::Result<Array, Exception> {
    jit_training_step_inner(state, (input_ids, labels), None)
}

/// Inner implementation shared by JIT step variants.
pub(crate) fn jit_training_step_inner<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    (input_ids, labels): (&Array, &Array),
    neftune_alpha: Option<f32>,
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
    let (loss, grads) = loss_and_grad_fn(model, (input_ids, labels))?;

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

    // Apply gradient clipping if max_grad_norm > 0
    if max_grad_norm > 0.0 {
        // Compute global gradient norm (GPU-based, no sync required)
        let eps = Array::from_f32_slice(&[1e-6_f32], &[1]);
        let mut sq_sum = Array::from_f32_slice(&[0.0_f32], &[1]);
        for grad in grads.values() {
            sq_sum = sq_sum.add(&grad.square().sum(None));
        }
        let norm = sq_sum.sqrt();

        // Scale gradients: scale = max_norm / max(norm, max_norm)
        // This clamps scale to [0, 1], only reducing large gradients
        let max_norm_arr = Array::from_f32_slice(&[max_grad_norm], &[1]);
        let norm_clamped = ops::maximum(&norm, &max_norm_arr);
        let scale = max_norm_arr.divide(&norm_clamped.add(&eps));

        // Apply scale to all gradients (in-place via replace)
        for grad in grads.values_mut() {
            *grad = grad.multiply(&scale);
        }
    }

    // Apply gradients via optimizer
    optimizer.update(model, grads)?;

    Ok(loss)
}

/// Shared helper: compute CCE loss from hidden states + lm_head weight.
///
/// Both the standard and packed CCE steps call this after obtaining hidden states.
/// `hidden_states` must already be shifted/reshaped as appropriate.
/// `labels` must already be shifted and flattened to [n_tokens].
pub(crate) fn compute_cce_loss(
    hidden_states: &Array,
    lm_head_weight: &Array,
    labels: &Array,
) -> std::result::Result<Array, Exception> {
    use pmetal_mlx::kernels::cut_cross_entropy::cut_cross_entropy_loss;

    // Flatten hidden states to [n_tokens, hidden_dim]
    let hidden_dim = hidden_states.dim(-1);
    let n_tokens = hidden_states.size() as i32 / hidden_dim;
    let flat_hidden = hidden_states.reshape(&[n_tokens, hidden_dim]);

    cut_cross_entropy_loss(&flat_hidden, lm_head_weight, labels, -100)
        .map_err(|e| Exception::custom(e.to_string()))
}

/// Training step using Cut Cross-Entropy (avoids materializing the full logits tensor).
///
/// Falls back to standard cross-entropy when the model does not implement
/// `forward_hidden()` / `lm_head_weight()`, ensuring backward compatibility.
pub(crate) fn jit_training_step_cce<M: TrainableModel, O: Optimizer>(
    state: &mut (M, O),
    (input_ids, labels): (&Array, &Array),
    neftune_alpha: Option<f32>,
) -> std::result::Result<Array, Exception> {
    let (model, optimizer) = state;

    // Fetch the LM head weight once.  This both serves as a capability probe and
    // provides the weight to capture into the closure, avoiding a second call inside
    // the gradient computation graph where calling lm_head_weight() again would clone
    // the array a second time unnecessarily.
    let lm_weight_cached = model.lm_head_weight();
    let has_cce_support = lm_weight_cached.is_some();

    if has_cce_support && neftune_alpha.is_none() {
        let cached_weight = lm_weight_cached.expect("checked above");
        // CCE path: compute loss from hidden states without full logits.
        let loss_fn = |model: &mut M,
                       (input_ids, labels): (&Array, &Array)|
         -> std::result::Result<Array, Exception> {
            let hidden_opt = model.forward_hidden(input_ids, None);

            match hidden_opt {
                Some(Ok(hidden_states)) => {
                    let seq_len = hidden_states.dim(1);

                    // Shift: hidden[:-1] predicts labels[1:]
                    let shift_hidden = hidden_states.index((.., ..seq_len - 1, ..));
                    let shift_labels = labels.index((.., 1..));
                    let flat_labels = shift_labels.reshape(&[-1]);

                    compute_cce_loss(&shift_hidden, &cached_weight, &flat_labels)
                }
                _ => {
                    // Unexpected failure in hidden forward — fall through to standard CE.
                    let logits = model
                        .forward(input_ids, None)
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    Ok(pmetal_bridge::training::causal_lm_loss(
                        &logits, labels, -100,
                    ))
                }
            }
        };

        let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
        let (loss, grads) = loss_and_grad_fn(model, (input_ids, labels))?;
        optimizer.update(model, grads)?;
        Ok(loss)
    } else {
        // NEFTune active or model doesn't support CCE — use standard path.
        jit_training_step_inner(state, (input_ids, labels), neftune_alpha)
    }
}

/// Packed-sequence training step using Cut Cross-Entropy.
///
/// Mirrors `jit_training_step_packed` but feeds hidden states through CCE
/// to avoid materializing the full logits tensor.  Falls back to standard
/// cross-entropy when the model does not implement `forward_hidden_with_positions()`.
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

    // Fetch the LM head weight once — serves as capability probe and provides
    // the cached weight to capture into the closure (avoids a second call inside
    // the gradient graph).
    let lm_weight_cached = model.lm_head_weight();
    let has_cce_support = lm_weight_cached.is_some();

    if has_cce_support {
        let cached_weight = lm_weight_cached.expect("checked above");
        let loss_fn = |model: &mut M,
                       (input_ids, labels): (&Array, &Array)|
         -> std::result::Result<Array, Exception> {
            let hidden_opt = model.forward_hidden_with_positions(
                input_ids,
                attn_mask_4d.as_ref(),
                &position_ids,
            );

            match hidden_opt {
                Some(Ok(hidden_states)) => {
                    let seq_len = hidden_states.dim(1);
                    let shift_hidden = hidden_states.index((.., ..seq_len - 1, ..));
                    let shift_labels = labels.index((.., 1..));
                    let flat_labels = shift_labels.reshape(&[-1]);

                    compute_cce_loss(&shift_hidden, &cached_weight, &flat_labels)
                }
                _ => {
                    // Fall back to standard cross-entropy
                    let logits = model
                        .forward_with_positions(input_ids, attn_mask_4d.as_ref(), &position_ids)
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    Ok(pmetal_bridge::training::causal_lm_loss(
                        &logits, labels, -100,
                    ))
                }
            }
        };

        let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
        let (loss, mut grads) = loss_and_grad_fn(model, (&input_ids_2d, &labels_2d))?;

        // Gradient clipping (same as jit_training_step_packed)
        if max_grad_norm > 0.0 {
            let eps = Array::from_f32_slice(&[1e-6_f32], &[1]);
            let mut sq_sum = Array::from_f32_slice(&[0.0_f32], &[1]);
            for grad in grads.values() {
                sq_sum = sq_sum.add(&grad.square().sum(None));
            }
            let norm = sq_sum.sqrt();
            let max_norm_arr = Array::from_f32_slice(&[max_grad_norm], &[1]);
            let norm_clamped = ops::maximum(&norm, &max_norm_arr);
            let scale = max_norm_arr.divide(&norm_clamped.add(&eps));
            for grad in grads.values_mut() {
                *grad = grad.multiply(&scale);
            }
        }

        optimizer.update(model, grads)?;
        Ok(loss)
    } else {
        // Model doesn't support CCE — use standard packed step.
        jit_training_step_packed(state, packed_batch, max_grad_norm)
    }
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
        transforms::eval(all_arrays);
    }
    Ok(())
}
