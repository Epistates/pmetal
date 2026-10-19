//! One micro-batch's loss and gradients, and the accumulation of several into
//! one optimizer step.
//!
//! Every micro-batch returns the *sum* of its tokens' losses, the number of
//! tokens that count, and the gradient of the sum. Accumulation adds those up
//! and divides by the total count once, so an optimizer step's gradient is
//! the gradient of the mean over all the tokens of the step, however unevenly
//! its micro-batches hold them. (The mean of per-micro-batch means, which
//! scaling each micro-batch by `1/n` computes, weights a token of a short
//! batch more than one of a long batch.) The division folds into the one
//! multiplication that also clips.

use pmetal_data::PackedTrainingBatch;
use pmetal_lora::TrainableModel;
use pmetal_models::dispatcher::CceLoss;
use pmetal_runtime::compat::{Array, Dtype, Exception, FlattenedModuleParam, nn, transforms};
use pmetal_runtime::nn::with_training_mode;
use pmetal_runtime::training::{CrossEntropy, global_grad_norm};

use super::source::BatchView;
use crate::Result;

/// The label that marks a token as not part of the loss.
pub(crate) const IGNORE_INDEX: i32 = -100;

/// How a micro-batch's loss is computed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LossSpec {
    /// Smooth the target distribution toward uniform by this share.
    pub(crate) label_smoothing: f32,
    /// Weight of the `lse²` auxiliary loss.
    pub(crate) z_loss: f32,
    /// NEFTune embedding noise, for the forward being differentiated.
    pub(crate) neftune_alpha: Option<f32>,
    /// Whether the loss comes from the LM head's cut cross-entropy, which
    /// never forms the logits, rather than from the logits.
    pub(crate) cut_cross_entropy: bool,
}

impl LossSpec {
    fn cross_entropy(&self) -> CrossEntropy {
        CrossEntropy::new()
            .with_ignore_index(IGNORE_INDEX)
            .with_label_smoothing(self.label_smoothing)
            .with_z_loss(self.z_loss)
    }

    fn cce_loss(&self) -> CceLoss {
        CceLoss::new(IGNORE_INDEX)
            .with_label_smoothing(self.label_smoothing)
            .with_z_loss(self.z_loss)
    }
}

/// What one micro-batch produced: all lazy.
pub(crate) struct MicroOutput {
    /// The sum of the counted tokens' losses.
    pub(crate) loss_sum: Array,
    /// How many tokens count (f32).
    pub(crate) count: Array,
    /// The gradients of `loss_sum`.
    pub(crate) grads: FlattenedModuleParam,
}

/// The number of labels at positions `1..` that are not ignored: the tokens a
/// causal language-model loss scores. An f32 scalar, outside any gradient.
fn counted_tokens(labels: &Array) -> Array {
    labels
        .index((.., 1..))
        .ne(&Array::from_int(IGNORE_INDEX))
        .as_type::<f32>()
        .sum(None)
        .stop_gradient()
}

/// The loss of the next token at every position, from the head's cut
/// cross-entropy: a mean over the counted tokens.
///
/// The head is read here, inside the function being differentiated, not
/// before it: an adapter on the LM head is in place as a traced parameter only
/// there, and a head read outside is a constant, which trained that adapter on
/// nothing. `hidden` is `[batch, seq, hidden]` and `labels` `[batch, seq]`,
/// unshifted. A model with no hidden-state forward or no usable head is an
/// error, not a quiet switch to the logits: the caller decided on cut
/// cross-entropy by asking the model, so either is a model that answered
/// wrongly.
pub(crate) fn compute_cce_loss<M: TrainableModel>(
    model: &M,
    hidden: Option<std::result::Result<Array, pmetal_lora::LoraError>>,
    labels: &Array,
    loss: CceLoss,
) -> std::result::Result<Array, Exception> {
    let hidden_states = hidden
        .ok_or_else(|| {
            Exception::custom("cut cross-entropy: the model has no hidden-state forward")
        })?
        .map_err(|e| Exception::custom(e.to_string()))?;
    let head = model
        .lm_head()
        .ok_or_else(|| Exception::custom("cut cross-entropy: the model has no LM head to use"))?;
    let seq_len = hidden_states.dim(1);
    let shift_hidden = hidden_states.index((.., ..seq_len - 1, ..));
    let shift_labels = labels.index((.., 1..)).reshape(&[-1]);
    head.cut_cross_entropy(&shift_hidden, &shift_labels, loss)
}

/// The shapes a micro-batch's tensors take at the model.
struct Inputs<'a> {
    input_ids: Array,
    labels: Array,
    forward: ForwardKind<'a>,
}

enum ForwardKind<'a> {
    Text,
    Images(&'a Array),
    Packed {
        mask: Option<Array>,
        positions: &'a Array,
    },
}

fn inputs(batch: BatchView<'_>) -> std::result::Result<Inputs<'_>, Exception> {
    Ok(match batch {
        BatchView::Padded(b) => Inputs {
            input_ids: b.input_ids.clone(),
            labels: b.labels.clone(),
            forward: match &b.pixel_values {
                Some(pixels) => ForwardKind::Images(pixels),
                None => ForwardKind::Text,
            },
        },
        BatchView::Packed(p) => packed_inputs(p)?,
    })
}

fn packed_inputs(p: &PackedTrainingBatch) -> std::result::Result<Inputs<'_>, Exception> {
    let total = p.total_tokens as i32;
    // Only a batch that combines sequences needs the block-diagonal mask; a
    // single sequence takes the model's causal path.
    let mask = if p.num_sequences > 1 {
        Some(p.attention_mask()?.reshape(&[1, 1, total, total]))
    } else {
        None
    };
    Ok(Inputs {
        input_ids: p.input_ids.reshape(&[1, total]),
        labels: p.labels.reshape(&[1, total]),
        forward: ForwardKind::Packed {
            mask,
            positions: &p.position_ids,
        },
    })
}

/// The loss and gradients of one micro-batch, in training mode, with NEFTune
/// noise when the spec asks for it.
pub(crate) fn loss_and_grads<M: TrainableModel>(
    model: &mut M,
    batch: BatchView<'_>,
    spec: &LossSpec,
) -> Result<MicroOutput> {
    let Inputs {
        input_ids,
        labels,
        forward,
    } = inputs(batch)?;
    let count = counted_tokens(&labels);
    let ce = spec.cross_entropy();
    let cce = spec.cce_loss();
    let use_cce = spec.cut_cross_entropy;
    let loss_count = count.clone();

    let loss_fn = |model: &mut M,
                   (input_ids, labels): (&Array, &Array)|
     -> std::result::Result<Array, Exception> {
        let to_exception = |e: pmetal_lora::LoraError| Exception::custom(e.to_string());
        if use_cce {
            let hidden = match &forward {
                ForwardKind::Text | ForwardKind::Images(_) => model.forward_hidden(input_ids, None),
                ForwardKind::Packed { mask, positions } => {
                    model.forward_hidden_with_positions(input_ids, mask.as_ref(), positions)
                }
            };
            // The head's loss is a mean; times the count it is the sum.
            return Ok(compute_cce_loss(model, hidden, labels, cce)?.multiply(&loss_count));
        }
        let logits = match &forward {
            ForwardKind::Text => model.forward(input_ids, None),
            ForwardKind::Images(pixels) => model.forward_with_images(input_ids, None, Some(pixels)),
            ForwardKind::Packed { mask, positions } => {
                model.forward_with_positions(input_ids, mask.as_ref(), positions)
            }
        }
        .map_err(to_exception)?;
        let seq_len = logits.dim(1);
        let shift_logits = logits.index((.., ..seq_len - 1, ..));
        let shift_labels = labels.index((.., 1..));
        Ok(ce.sum_and_count(&shift_logits, &shift_labels).0)
    };

    let noised = spec
        .neftune_alpha
        .is_some_and(|alpha| model.set_neftune_alpha(Some(alpha)));
    let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
    let result = with_training_mode(|| loss_and_grad_fn(model, (&input_ids, &labels)));
    if noised {
        model.set_neftune_alpha(None);
    }
    let (loss_sum, grads) = result?;
    Ok(MicroOutput {
        loss_sum,
        count,
        grads,
    })
}

/// What an optimizer step takes from its micro-batches.
pub(crate) struct Finished {
    /// The gradients of the step's mean loss, clipped.
    pub(crate) grads: FlattenedModuleParam,
    /// The step's mean loss over its counted tokens.
    pub(crate) loss: Array,
    /// The global norm of the unclipped gradients, when clipping is on.
    pub(crate) grad_norm: Option<Array>,
}

/// Micro-batch gradients on their way to one optimizer step.
#[derive(Default)]
pub(crate) struct GradAccumulator {
    grads: Option<FlattenedModuleParam>,
    loss_sum: Option<Array>,
    count: Option<Array>,
    micro_batches: usize,
}

impl GradAccumulator {
    /// Micro-batches held.
    pub(crate) fn micro_batches(&self) -> usize {
        self.micro_batches
    }

    /// Fold in a micro-batch. The sums are f32 whatever the gradients' dtype.
    pub(crate) fn add(&mut self, out: MicroOutput) {
        let f32_ = Dtype::Float32.as_i32();
        match (&mut self.grads, &mut self.loss_sum, &mut self.count) {
            (Some(grads), Some(loss_sum), Some(count)) => {
                for (key, grad) in out.grads {
                    match grads.get_mut(&key) {
                        Some(held) => *held = held.add(&grad),
                        None => {
                            grads.insert(key, grad.as_dtype(f32_));
                        }
                    }
                }
                *loss_sum = loss_sum.add(&out.loss_sum);
                *count = count.add(&out.count);
            }
            _ => {
                self.grads = Some(
                    out.grads
                        .into_iter()
                        .map(|(k, g)| (k, g.as_dtype(f32_)))
                        .collect(),
                );
                self.loss_sum = Some(out.loss_sum);
                self.count = Some(out.count);
            }
        }
        self.micro_batches += 1;
    }

    /// Evaluate what is held, which frees the micro-batches' graphs: without
    /// it each one's forward and backward stays alive until the step.
    pub(crate) fn eval_held(&self, extra: &[&Array]) -> Result<()> {
        let mut arrays: Vec<&Array> = extra.to_vec();
        if let Some(grads) = &self.grads {
            arrays.extend(grads.values());
        }
        arrays.extend(self.loss_sum.iter().chain(self.count.iter()));
        transforms::eval(arrays)?;
        Ok(())
    }

    /// End the step: divide by the token count and clip to `max_norm` (off
    /// at 0 or below) in one multiplication per gradient, and reset.
    pub(crate) fn finish(&mut self, max_norm: f32) -> Option<Finished> {
        let mut grads = self.grads.take()?;
        let loss_sum = self.loss_sum.take()?;
        let count = self.count.take()?;
        self.micro_batches = 0;

        let inv = Array::from_f32(1.0).divide(&count.maximum(&Array::from_f32(1.0)));
        let (scale, grad_norm) = if max_norm > 0.0 {
            // The norm of the sum; the mean's is that times `inv`. Clipping
            // the mean to `max_norm` scales by `min(1, max_norm / norm_mean)`
            // and the division by the count is another `inv`, so the one
            // factor is `min(inv, max_norm / norm_sum)`.
            let norm_sum = global_grad_norm(grads.values());
            let clip = Array::from_f32(max_norm).divide(&norm_sum);
            (inv.minimum(&clip), Some(norm_sum.multiply(&inv)))
        } else {
            (inv.clone(), None)
        };
        for grad in grads.values_mut() {
            *grad = grad.multiply(&scale);
        }
        Some(Finished {
            grads,
            loss: loss_sum.multiply(&inv),
            grad_norm,
        })
    }
}
