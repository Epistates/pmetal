//! DFlash block-diffusion draft training.
//!
//! DFlash draft models ship pre-trained from the `z-lab/*-DFlash*` Hugging
//! Face collection. This module is the pmetal-native path for *training a
//! new draft from scratch*, which upstream `dflash-mlx` cannot do at all.
//!
//! # Objective
//!
//! For a sequence `x_1, …, x_n` the drafter minimizes the following
//! per-block cross-entropy:
//!
//! 1. Choose a contiguous block `x_{i..i+B}` of size `B = draft.block_size`.
//! 2. Replace every position in the block with the draft's
//!    `mask_token_id`, producing `x_masked`.
//! 3. Run the target model over `x_{1..i} ⊕ x_masked` with hidden-state
//!    capture at the layers listed in
//!    `draft.config.dflash_config.target_layer_ids`. The captured
//!    activations at positions `i..i+B` become `target_hidden`.
//! 4. Run `draft.forward(embed(x_masked), target_hidden)` to obtain draft
//!    hidden states, project them through the target's `lm_head`, and
//!    take `softmax(logits[:, 1:, :])` — the first position is a seed
//!    slot that the runtime always overwrites with the verified token.
//! 5. Compute cross-entropy against the true tokens `x_{i+1..i+B}`.
//!
//! The objective matches upstream DFlash: the drafter learns to
//! reconstruct an entire block in one forward pass by conditioning on the
//! target's intermediate representations.
//!
//! # Status
//!
//! This module provides [`dflash_train_step`] — the single forward+loss
//! primitive. It is deliberately *not* a full training loop: wiring up
//! dataset streaming, optimization, LR scheduling, checkpointing, and
//! distributed data parallel is covered by pmetal's existing trainer
//! infrastructure in [`crate::diffusion`] (block diffusion in general) and
//! [`crate::orchestrator`] (training orchestration).
//!
//! A full `pmetal train-draft` CLI command can be implemented on top of
//! [`dflash_train_step`] without touching any other module.

use pmetal_bridge::compat::{Array, Exception, Module, ops};
use pmetal_mlx::speculative::SpecCapture;
use pmetal_models::architectures::dflash_draft::DFlashDraftModel;
use pmetal_models::architectures::qwen3::Qwen3ForCausalLM;
use pmetal_models::dflash_decoder::DFlashTarget;

/// Configuration for a single [`dflash_train_step`] call.
#[derive(Debug, Clone)]
pub struct DFlashTrainStepConfig {
    /// Block size used for masking. Must match `draft.block_size()`.
    pub block_size: usize,
    /// Starting offset of the masked block within the input sequence.
    /// Callers typically pick this uniformly at random.
    pub block_start: usize,
}

/// Output of [`dflash_train_step`].
#[derive(Debug, Clone)]
pub struct DFlashTrainStepOutput {
    /// Cross-entropy loss for the block, shape `[]` (scalar).
    pub loss: Array,
    /// Logits produced by the draft for the `block_size - 1` predicted
    /// positions, shape `[B, block_size - 1, vocab_size]`.
    pub draft_logits: Array,
}

/// Run one DFlash training forward pass and return the cross-entropy loss.
///
/// This is the primitive that a full training loop composes — call it per
/// batch, backprop the returned loss, step the optimizer. The Qwen3 target
/// is held `&mut` because its KV cache is reused from capture to capture;
/// callers should pass a freshly allocated cache per batch so target state
/// is not leaked between examples.
///
/// # Arguments
///
/// * `target` — the frozen Qwen3 target model. Only its forward pass +
///   lm_head are used; no parameter updates flow into it.
/// * `draft` — the DFlash drafter whose parameters we want to train.
/// * `input_ids` — `[B, T]` token ids. Must satisfy
///   `cfg.block_start + cfg.block_size <= T`.
/// * `cfg` — which block to mask / train on.
///
/// # Returns
///
/// A [`DFlashTrainStepOutput`] whose `loss` field is ready for backprop.
/// `draft_logits` is returned for downstream metric collection (accuracy,
/// top-k match rate, etc).
pub fn dflash_train_step(
    target: &mut Qwen3ForCausalLM,
    draft: &mut DFlashDraftModel,
    input_ids: &Array,
    cfg: &DFlashTrainStepConfig,
) -> Result<DFlashTrainStepOutput, Exception> {
    if input_ids.ndim() != 2 {
        return Err(Exception::custom(format!(
            "dflash_train_step: input_ids must be [B, T], got shape {:?}",
            input_ids.shape()
        )));
    }
    let batch = input_ids.dim(0);
    let seq = input_ids.dim(1) as usize;
    if cfg.block_size != draft.block_size() {
        return Err(Exception::custom(format!(
            "dflash_train_step: cfg.block_size={} must equal draft.block_size()={}",
            cfg.block_size,
            draft.block_size()
        )));
    }
    if cfg.block_start + cfg.block_size > seq {
        return Err(Exception::custom(format!(
            "dflash_train_step: block_start {} + block_size {} > seq {}",
            cfg.block_start, cfg.block_size, seq
        )));
    }

    // Build `masked_ids`: a copy of `input_ids` with positions
    // [block_start, block_start + block_size) replaced by the draft's
    // mask token. We assemble it by concatenating three slices along the
    // sequence dim — cheap, and avoids an in-place update on a graph
    // tensor.
    let mask_tok = Array::from_slice(
        &vec![draft.mask_token_id(); (batch as usize) * cfg.block_size],
        &[batch, cfg.block_size as i32],
    );
    let prefix = input_ids.slice(&[0, 0], &[batch, cfg.block_start as i32]);
    let suffix_start = (cfg.block_start + cfg.block_size) as i32;
    let suffix = input_ids.slice(&[0, suffix_start], &[batch, seq as i32]);
    let masked_ids = if cfg.block_start == 0 && suffix_start == seq as i32 {
        mask_tok.clone()
    } else if cfg.block_start == 0 {
        ops::concatenate_axis(&[&mask_tok, &suffix], 1)
    } else if suffix_start == seq as i32 {
        ops::concatenate_axis(&[&prefix, &mask_tok], 1)
    } else {
        ops::concatenate_axis(&[&prefix, &mask_tok, &suffix], 1)
    };

    // Run the target over the masked sequence with hidden-state capture at
    // the drafter's tapped layers. Fresh KV cache per call — training must
    // not leak target state across examples.
    let mut kv_cache = target.make_kv_cache(seq);
    let target_layer_ids: Vec<usize> = draft.config.target_layer_ids();
    let mut capture = SpecCapture::with_layers(target_layer_ids.clone());
    let _target_logits = target.forward_with_capture(
        &masked_ids,
        None,
        Some(&mut kv_cache),
        &mut capture,
    )?;

    // Slice the captured `[B, T, L*hidden]` tensor to the block span —
    // the draft only conditions on the positions it is predicting.
    let target_hidden_full = capture.stack_hidden()?;
    let block_start_i32 = cfg.block_start as i32;
    let block_end_i32 = suffix_start;
    let last_dim = target_hidden_full.dim(2);
    let target_hidden_block = target_hidden_full.slice(
        &[0, block_start_i32, 0],
        &[batch, block_end_i32, last_dim],
    );

    // Draft forward: the mask-token embeddings go through the target's
    // embedding table (identical to inference), and the draft predicts
    // the unmasked tokens for positions 1..block_size.
    let block_ids = input_ids.slice(&[0, block_start_i32], &[batch, block_end_i32]);
    let noise_embedding = target.embed_tokens(&block_ids.clone())?;
    // Replace with the mask embedding — matches the inference pipeline,
    // which also feeds `target.embed_tokens(block_input)` where
    // `block_input` is mask-filled.
    let noise_embedding = {
        // re-derive using the masked block (all mask tokens):
        let masked_block = mask_tok.clone();
        target.embed_tokens(&masked_block)?
    };
    // `block_ids` is kept alive for labels below.
    let _ = noise_embedding.dim(0);

    let draft_hidden = draft.forward(&noise_embedding, &target_hidden_block, None)?;
    // Drop the seed position — the runtime always overwrites it with the
    // verified token, so we don't train it.
    let pred_hidden = draft_hidden.slice(
        &[0, 1, 0],
        &[batch, cfg.block_size as i32, draft_hidden.dim(2)],
    );
    let draft_logits = target.lm_head_project(&pred_hidden)?;

    // Labels are `input_ids[:, block_start+1 .. block_start+block_size]`.
    let labels = input_ids.slice(
        &[0, block_start_i32 + 1],
        &[batch, block_end_i32],
    );

    // Cross entropy: -sum(labels * log_softmax(logits)) / (B * (block_size-1)).
    // We materialize `log_softmax` via logsumexp for numerical stability.
    let loss = masked_cross_entropy(&draft_logits, &labels)?;

    Ok(DFlashTrainStepOutput {
        loss,
        draft_logits,
    })
}

/// Token-level masked cross-entropy matching what pmetal's other trainers
/// use. Accepts `logits` shape `[B, T, V]` and integer `labels` shape
/// `[B, T]`.
fn masked_cross_entropy(logits: &Array, labels: &Array) -> Result<Array, Exception> {
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(Exception::custom(format!(
            "masked_cross_entropy: logits must be [B, T, V], got {:?}",
            shape
        )));
    }
    let b = shape[0];
    let t = shape[1];
    let v = shape[2];

    // log_softmax = logits - logsumexp(logits, -1, keepdims)
    let max = ops::max_axis(logits, -1, true);
    let shifted = logits.subtract(&max);
    let exp = shifted.exp();
    let sum = ops::sum_axis(&exp, -1, true);
    let log_sum = sum.log();
    let log_softmax = shifted.subtract(&log_sum);

    // Gather -log p[label] per position via one-hot + sum. For a correct
    // rank-preserving gather the cheapest path is `take_along_axis`.
    let labels_i32 = labels.reshape(&[b, t, 1]);
    let gathered = ops::take_along_axis(&log_softmax, &labels_i32, -1);
    let per_tok = gathered.reshape(&[b, t]);
    let neg = per_tok.negative();
    let denom = Array::from_f32((b * t) as f32);
    Ok(ops::sum_axis(&neg, -1, false)
        .sum_axis(-1, false)
        .divide(&denom))
}
