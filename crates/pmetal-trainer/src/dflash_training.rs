//! DFlash block-diffusion draft training.
//!
//! DFlash draft models ship pre-trained from the `z-lab/*-DFlash*` Hugging
//! Face collection. This module is the pmetal-native path for *training a
//! new draft from scratch*, following the training recipe of DFlash (Chen,
//! Liang and Liu, *DFlash: Block Diffusion for Flash Speculative Decoding*,
//! arXiv 2602.06036, section 4.2).
//!
//! # Objective
//!
//! For a sequence `x_0, …, x_{n-1}` the drafter is trained on blocks that
//! look exactly like the ones it drafts at inference:
//!
//! 1. The **frozen target** runs once over the *clean* sequence, capturing the
//!    hidden states at the layers listed in
//!    `draft.config.dflash_config.target_layer_ids` for every position. (At
//!    inference the drafter's context is the target's states on tokens that
//!    were accepted, i.e. clean ones; causal attention makes the states at
//!    positions before an anchor the same whatever follows.)
//! 2. An **anchor** position `a ≥ 1` starts a block of `B = draft.block_size`
//!    tokens: the clean token `x_a` in slot 0 (at inference the bonus token
//!    the target just sampled) and the mask token in slots `1..B`. The block's
//!    embeddings come from the target's embedding table.
//! 3. The draft reads the target's states at positions `0..a` as context and
//!    the block, and its hidden states at slots `1..B` go through the target's
//!    `lm_head`: the predictions for `x_{a+1}, …, x_{a+B-1}`.
//! 4. The loss is the cross-entropy of those predictions, each slot `k`
//!    (1-based) weighted by `w_k = exp(−(k−1)/γ)`, because an early error
//!    invalidates the rest of a drafted block. The paper's decay is γ = 7 for
//!    blocks of 16, 5 for 10 and 4 for 8.
//!
//! The paper trains many anchors per sequence in one pass under a block-sparse
//! attention mask; here the target pass is shared and each anchor's block is
//! one draft forward, which has the same objective without the mask.
//!
//! [`dflash_train_step`] is the forward and loss, the primitive that
//! [`crate::run_dflash_draft_training`] steps an optimizer over.

use pmetal_cache::speculative::SpecCapture;
use pmetal_models::architectures::dflash_draft::DFlashDraftModel;
use pmetal_models::architectures::qwen3::Qwen3ForCausalLM;
use pmetal_models::dflash_decoder::DFlashTarget;
use pmetal_runtime::compat::{Array, Exception, ops};

/// Configuration for a single [`dflash_train_step`] call.
#[derive(Debug, Clone)]
pub struct DFlashTrainStepConfig {
    /// Block size. Must match `draft.block_size()`.
    pub block_size: usize,
    /// The anchors: the position of each block's clean first token. Each is at
    /// least 1 (the context is the tokens before it) and leaves a whole block
    /// inside the sequence. The loss is the mean over the blocks.
    pub block_starts: Vec<usize>,
    /// The decay `γ` of the position weights; `None` takes the paper's value for the
    /// block size ([`dflash_default_decay`]).
    pub position_decay: Option<f32>,
}

/// The paper's `γ` for `block_size`: 7 for 16, 5 for 10, 4 for 8, and a
/// proportional 0.45 per slot for other sizes.
pub fn dflash_default_decay(block_size: usize) -> f32 {
    match block_size {
        16 => 7.0,
        10 => 5.0,
        8 => 4.0,
        other => (0.45 * other as f32).max(1.0),
    }
}

/// The weights `w_k = exp(−(k−1)/γ)` of the `block_size − 1` predicted slots
/// (`k = 1..block_size`).
pub fn dflash_position_weights(block_size: usize, decay: f32) -> Vec<f32> {
    (0..block_size.saturating_sub(1))
        .map(|k| (-(k as f32) / decay).exp())
        .collect()
}

/// Output of [`dflash_train_step`].
#[derive(Debug, Clone)]
pub struct DFlashTrainStepOutput {
    /// Position-weighted cross-entropy loss, mean over the blocks, shape `[]`.
    pub loss: Array,
    /// Logits produced by the draft for the last block's `block_size - 1`
    /// predicted positions, shape `[B, block_size - 1, vocab_size]`.
    pub draft_logits: Array,
}

/// Run one DFlash training forward pass and return the weighted cross-entropy
/// loss.
///
/// This is the primitive that a full training loop composes — call it per
/// batch, backprop the returned loss, step the optimizer.
///
/// # Arguments
///
/// * `target` — the frozen Qwen3 target model. Its forward pass, embedding
///   table and `lm_head` are used; no gradient flows into it.
/// * `draft` — the DFlash drafter whose parameters we want to train.
/// * `input_ids` — `[B, T]` token ids; every anchor `a` of `cfg` satisfies
///   `1 <= a` and `a + block_size <= T`.
/// * `cfg` — which blocks to train on.
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
    if cfg.block_size < 2 {
        return Err(Exception::custom(
            "dflash_train_step: a block of one has nothing to predict",
        ));
    }
    if cfg.block_starts.is_empty() {
        return Err(Exception::custom(
            "dflash_train_step: no block anchors to train on",
        ));
    }
    for &anchor in &cfg.block_starts {
        if anchor < 1 || anchor + cfg.block_size > seq {
            return Err(Exception::custom(format!(
                "dflash_train_step: anchor {anchor} with block_size {} needs 1 <= anchor and \
                 anchor + block_size <= seq {seq}",
                cfg.block_size
            )));
        }
    }

    // The target over the clean sequence, once: its tapped states at every
    // position are the contexts of every block. It is frozen, so nothing
    // flows back into it, and it is not kept in the graph.
    let mut kv_cache = target.make_kv_cache(seq);
    let mut capture = SpecCapture::with_layers(draft.config.target_layer_ids());
    let _target_logits =
        target.forward_with_capture(input_ids, None, Some(&mut kv_cache), &mut capture)?;
    let target_hidden = capture.stack_hidden()?.stop_gradient();
    target_hidden.eval();

    let block = cfg.block_size as i32;
    let mask_slots = Array::from_slice(
        &vec![draft.mask_token_id(); batch as usize * (cfg.block_size - 1)],
        &[batch, block - 1],
    );
    let decay = cfg
        .position_decay
        .unwrap_or_else(|| dflash_default_decay(cfg.block_size));
    let weights = Array::from_slice(
        &dflash_position_weights(cfg.block_size, decay),
        &[1, block - 1],
    );
    let weight_sum = weights.sum(None).multiply(&Array::from_f32(batch as f32));

    let mut total: Option<Array> = None;
    let mut last_logits = None;
    for &anchor in &cfg.block_starts {
        let anchor_i = anchor as i32;
        // Context: the target's states on the tokens before the anchor.
        let context = target_hidden.slice(&[0, 0, 0], &[batch, anchor_i, target_hidden.dim(2)]);
        // Block: the clean anchor token, then the mask token.
        let anchor_token = input_ids.slice(&[0, anchor_i], &[batch, anchor_i + 1]);
        let block_ids = ops::concatenate_axis(&[&anchor_token, &mask_slots], 1);
        let noise_embedding = target.embed_tokens(&block_ids)?;

        let draft_hidden = draft.forward(&noise_embedding, &context, None)?;
        // Slot 0 holds the known anchor: it predicts nothing.
        let pred_hidden = draft_hidden.slice(&[0, 1, 0], &[batch, block, draft_hidden.dim(2)]);
        let logits = target.lm_head_project(&pred_hidden)?;

        // Slot k predicts x_{anchor + k}.
        let labels = input_ids.slice(&[0, anchor_i + 1], &[batch, anchor_i + block]);
        let per_token = pmetal_runtime::training::per_token_cross_entropy_loss(
            &logits.reshape(&[batch * (block - 1), logits.dim(2)]),
            &labels.reshape(&[batch * (block - 1)]),
            -100,
        )
        .reshape(&[batch, block - 1]);
        let loss = per_token.multiply(&weights).sum(None).divide(&weight_sum);
        total = Some(match total {
            Some(acc) => acc.add(&loss),
            None => loss,
        });
        last_logits = Some(logits);
    }

    let loss = total
        .expect("at least one anchor")
        .divide(&Array::from_f32(cfg.block_starts.len() as f32));
    Ok(DFlashTrainStepOutput {
        loss,
        draft_logits: last_logits.expect("at least one anchor"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::qwen3::Qwen3Config;
    use pmetal_models::architectures::{DFlashDraftConfig, DFlashExtras};
    use pmetal_runtime::compat::ModuleParameters as _;

    fn target() -> Qwen3ForCausalLM {
        Qwen3ForCausalLM::new(Qwen3Config {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: 8,
            max_position_embeddings: 64,
            tie_word_embeddings: true,
            ..Default::default()
        })
        .unwrap()
    }

    fn draft() -> DFlashDraftModel {
        DFlashDraftModel::new(DFlashDraftConfig {
            model_type: "dflash_qwen3".to_string(),
            hidden_size: 16,
            num_hidden_layers: 1,
            intermediate_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            vocab_size: 64,
            max_position_embeddings: 64,
            rope_theta: Some(10_000.0),
            head_dim: 8,
            block_size: 4,
            dflash_config: DFlashExtras {
                target_layer_ids: vec![0, 1],
                mask_token_id: 7,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap()
    }

    fn ids(rows: &[&[i32]]) -> Array {
        let seq = rows[0].len();
        let flat: Vec<i32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Array::from_slice(&flat, &[rows.len() as i32, seq as i32])
    }

    /// The paper's weights: `exp(−(k−1)/γ)` over the predicted slots, with
    /// its γ for 16, 10 and 8.
    #[test]
    fn the_position_weights_decay_as_in_the_paper() {
        assert_eq!(dflash_default_decay(16), 7.0);
        assert_eq!(dflash_default_decay(10), 5.0);
        assert_eq!(dflash_default_decay(8), 4.0);
        let w = dflash_position_weights(4, 2.0);
        assert_eq!(w.len(), 3);
        for (k, got) in w.iter().enumerate() {
            let want = (-(k as f64) / 2.0).exp();
            assert!((*got as f64 - want).abs() < 1e-6);
        }
    }

    /// The step is the paper's recipe, computed by hand: the target over the
    /// clean sequence, the context its states before the anchor, the block a
    /// clean anchor token and masks, the loss position-weighted.
    #[test]
    fn the_step_trains_on_blocks_that_look_like_inference() {
        pmetal_runtime::compat::random::seed(2);
        let mut target = target();
        let mut draft = draft();
        let input = ids(&[&[5, 9, 1, 33, 7, 12, 40, 3], &[2, 4, 6, 8, 10, 12, 14, 16]]);
        let anchor = 2usize;
        let out = dflash_train_step(
            &mut target,
            &mut draft,
            &input,
            &DFlashTrainStepConfig {
                block_size: 4,
                block_starts: vec![anchor],
                position_decay: Some(2.0),
            },
        )
        .unwrap();
        out.loss.eval();
        pmetal_runtime::check_unobserved_error().unwrap();
        let got = out.loss.item_f32() as f64;

        // By hand.
        let mut cache = target.make_kv_cache(8);
        let mut capture = SpecCapture::with_layers(vec![0, 1]);
        target
            .forward_with_capture(&input, None, Some(&mut cache), &mut capture)
            .unwrap();
        let states = capture.stack_hidden().unwrap();
        let context = states.slice(&[0, 0, 0], &[2, 2, states.dim(2)]);
        // Block ids per row: x_2 (clean), then the mask token 7.
        let block = ids(&[&[1, 7, 7, 7], &[6, 7, 7, 7]]);
        let noise = target.embed_tokens(&block).unwrap();
        let hidden = draft.forward(&noise, &context, None).unwrap();
        let logits = target
            .lm_head_project(&hidden.slice(&[0, 1, 0], &[2, 4, hidden.dim(2)]))
            .unwrap();
        logits.eval();
        let logits = logits.as_dtype(pmetal_runtime::dtype::F32);
        logits.eval();
        let rows: &[f32] = logits.as_slice();
        let labels = [[33usize, 7, 12], [8, 10, 12]];
        let w: Vec<f64> = (0..3).map(|k| (-(k as f64) / 2.0).exp()).collect();
        let mut sum = 0.0;
        for (b, row_labels) in labels.iter().enumerate() {
            for (k, &label) in row_labels.iter().enumerate() {
                let z = &rows[(b * 3 + k) * 64..(b * 3 + k + 1) * 64];
                let max = z.iter().cloned().fold(f32::MIN, f32::max) as f64;
                let lse = max + z.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln();
                sum += w[k] * (lse - z[label] as f64);
            }
        }
        let want = sum / (2.0 * w.iter().sum::<f64>());
        assert!((got - want).abs() < 1e-4, "{got} vs {want}");
    }

    #[test]
    fn anchors_are_checked() {
        let mut target = target();
        let mut draft = draft();
        let input = ids(&[&[5, 9, 1, 33, 7, 12]]);
        let mut step = |starts: Vec<usize>| {
            dflash_train_step(
                &mut target,
                &mut draft,
                &input,
                &DFlashTrainStepConfig {
                    block_size: 4,
                    block_starts: starts,
                    position_decay: None,
                },
            )
        };
        assert!(step(vec![]).is_err());
        // An anchor of 0 has no context; one at 3 leaves too few tokens.
        assert!(step(vec![0]).is_err());
        assert!(step(vec![3]).is_err());
        assert!(step(vec![1, 2]).is_ok());
    }

    /// The draft learns: on a fixed sequence the loss falls and the draft's
    /// parameters move, while the frozen target's do not.
    #[test]
    fn training_lowers_the_loss_and_moves_only_the_draft() {
        use pmetal_runtime::compat::{
            nn,
            optimizers::{AdamWBuilder, Optimizer},
        };
        pmetal_runtime::compat::random::seed(6);
        let mut target = target();
        let mut draft = draft();
        let input = ids(&[&[5, 9, 1, 33, 7, 12, 40, 3, 5, 9, 1, 33]]);
        let fc_before = draft.fc.weight.value.clone();
        let target_weight = target.model.embed_tokens.weight.value.clone();

        let mut optimizer = AdamWBuilder::new(3e-3).weight_decay(0.0).build().unwrap();
        let mut losses = Vec::new();
        for _ in 0..60 {
            let mut vag = nn::value_and_grad(|d: &mut DFlashDraftModel, _: ()| {
                dflash_train_step(
                    &mut target,
                    d,
                    &input,
                    &DFlashTrainStepConfig {
                        block_size: 4,
                        block_starts: vec![1, 4, 8],
                        position_decay: None,
                    },
                )
                .map(|o| o.loss)
            });
            let (loss, grads) = vag(&mut draft, ()).unwrap();
            optimizer.update(&mut draft, grads).unwrap();
            pmetal_runtime::compat::eval_params(draft.trainable_parameters()).unwrap();
            loss.eval();
            pmetal_runtime::check_unobserved_error().unwrap();
            losses.push(loss.item_f32());
        }
        assert!(losses[0].is_finite());
        assert!(
            *losses.last().unwrap() < 0.5 * losses[0],
            "loss {} -> {}",
            losses[0],
            losses.last().unwrap()
        );
        let moved = draft.fc.weight.value.subtract(&fc_before).abs().max(None);
        moved.eval();
        assert!(moved.item_f32() > 1e-5, "the draft did not move");
        let target_moved = target
            .model
            .embed_tokens
            .weight
            .value
            .subtract(&target_weight)
            .abs()
            .max(None);
        target_moved.eval();
        assert_eq!(target_moved.item_f32(), 0.0);
    }
}
