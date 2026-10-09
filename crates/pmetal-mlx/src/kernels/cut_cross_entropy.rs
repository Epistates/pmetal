//! Cut cross-entropy: the cross-entropy of an LM head's logits without ever
//! holding all of them, forward or backward.
//!
//! Wijmans et al., "Cut Your Losses in Large-Vocabulary Language Models"
//! (arXiv 2411.09009), with the reference implementation at
//! <https://github.com/apple/ml-cross-entropy>. The loss of a token is
//! `logsumexp(logits) - logits[target]`, and both terms can be had without the
//! `[tokens, vocab]` logits: build a vocabulary chunk's logits, reduce them to
//! the chunk's `logsumexp` and the target logit if the chunk holds it, drop
//! them, and combine the chunks' `[tokens]` results.
//!
//! Autograd would undo that: differentiating the chunked forward keeps every
//! chunk's intermediates for the backward pass, which is all the logits
//! again, in f32. So each chunk runs under [`checkpoint_apply`], which keeps
//! only its inputs and its `[tokens]` outputs and rebuilds the chunk during
//! the backward pass, as the reference implementation's backward kernel
//! does. The head is split rather than sliced, and the target logit read out
//! of its chunk rather than gathered from the head, so the head's gradient
//! comes back as one concatenation instead of a full-size array per chunk.
//!
//! # Precision
//!
//! As in the reference implementation: each chunk's logits come out of the
//! matmul in the model's dtype (the GEMM accumulates in f32 and rounds once,
//! which is exactly the logits the model's own head produces), and
//! everything after that (scale, softcap, max, exp, sum, the log) runs in
//! f32. The target logit is read out of the same chunk, so the two terms of
//! the loss see the same logits.

use pmetal_bridge::compat::{Array, Dtype, Exception, ops};
use pmetal_bridge::inline_array::checkpoint_apply;

/// Configuration for [`CutCrossEntropy`].
#[derive(Debug, Clone)]
pub struct CutCrossEntropyConfig {
    /// Vocabulary rows per chunk. The transient memory of a chunk is
    /// `tokens × vocab_chunk_size` f32 values, forward and backward.
    pub vocab_chunk_size: usize,
    /// Target id excluded from the loss.
    pub ignore_index: i32,
    /// `cap · tanh(logits / cap)` after the scale (Gemma 2, Gemma 4); 0
    /// disables it.
    pub softcap: f32,
    /// Multiplies the logits (Cohere, Granite); 1 disables it.
    pub logit_scale: f32,
}

impl Default for CutCrossEntropyConfig {
    fn default() -> Self {
        Self {
            vocab_chunk_size: 4096,
            ignore_index: -100,
            softcap: 0.0,
            logit_scale: 1.0,
        }
    }
}

impl CutCrossEntropyConfig {
    /// The default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vocabulary chunk size.
    pub fn with_vocab_chunk_size(mut self, size: usize) -> Self {
        self.vocab_chunk_size = size;
        self
    }

    /// Set the ignored target id.
    pub fn with_ignore_index(mut self, index: i32) -> Self {
        self.ignore_index = index;
        self
    }

    /// Enable logit softcapping.
    pub fn with_softcap(mut self, softcap: f32) -> Self {
        self.softcap = softcap;
        self
    }

    /// Scale the logits.
    pub fn with_logit_scale(mut self, scale: f32) -> Self {
        self.logit_scale = scale;
        self
    }
}

/// Cut cross-entropy over an LM head. See the [module docs](self).
pub struct CutCrossEntropy {
    config: CutCrossEntropyConfig,
}

impl CutCrossEntropy {
    /// A loss with the given configuration.
    pub fn new(config: CutCrossEntropyConfig) -> Self {
        Self { config }
    }

    /// Mean cross-entropy over the targets that are not `ignore_index`, as an
    /// f32 scalar.
    ///
    /// * `hidden` - `[tokens, hidden]`
    /// * `weight` - the head, `[vocab, hidden]`
    /// * `targets` - `[tokens]`
    /// * `bias` - `[vocab]`, when the head has one
    pub fn forward(
        &self,
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        bias: Option<&Array>,
    ) -> Result<Array, Exception> {
        if hidden.ndim() != 2 || weight.ndim() != 2 || hidden.dim(1) != weight.dim(1) {
            return Err(Exception::custom(format!(
                "cut cross-entropy: hidden {:?} and head {:?} don't multiply",
                hidden.shape(),
                weight.shape()
            )));
        }
        // Labels must never enter the trace: MLX ≥ 0.32 refuses a gather's
        // gradient with respect to its indices.
        let targets = targets.as_dtype(Dtype::Int32.as_i32()).stop_gradient();
        let valid = targets
            .not_equal(&Array::from_i32(self.config.ignore_index))
            .as_dtype(Dtype::Float32.as_i32());
        let transform = Transform {
            scale: self.config.logit_scale,
            softcap: self.config.softcap,
        };

        let (lse, target) = self.chunked(hidden, weight, bias, &targets, transform)?;

        let per_token = lse.subtract(&target).multiply(&valid);
        let count = valid.sum_all().maximum(&Array::from_f32(1.0));
        Ok(per_token.sum_all().divide(&count))
    }

    /// Each token's `logsumexp` and target logit, both `[tokens]` in f32, a
    /// chunk of the vocabulary at a time and each chunk rebuilt for its
    /// backward pass.
    ///
    /// The target logit is read out of the chunk that holds it, as the
    /// reference implementation's forward kernel does, rather than gathered
    /// from the head: a gather's gradient is a scatter into a zeroed copy of
    /// the whole head, a second full-size array beside the one the chunks
    /// already produce.
    fn chunked(
        &self,
        hidden: &Array,
        weight: &Array,
        bias: Option<&Array>,
        targets: &Array,
        transform: Transform,
    ) -> Result<(Array, Array), Exception> {
        let vocab = weight.dim(0);
        let chunk = self.config.vocab_chunk_size.clamp(1, i32::MAX as usize) as i32;
        let bounds: Vec<i32> = (1..)
            .map(|i| i * chunk)
            .take_while(|&b| b < vocab)
            .collect();
        let split = |a: &Array| -> Vec<Array> {
            if bounds.is_empty() {
                vec![a.clone()]
            } else {
                a.split(&bounds, 0)
            }
        };
        let weights = split(weight);
        let biases = bias.map(split);

        // The running logsumexp and target logit are threaded through the
        // chunks in order, and so are the hidden states: each chunk passes
        // them on as `h + 0·lse`, which is `h` exactly, but makes the next
        // chunk's matmul wait for this chunk's forward and this chunk's
        // backward wait for the next one's. Without that the chunks are
        // independent, and MLX schedules them side by side, which peaked
        // above the full logits' memory. Chained, a 4,212-token Qwen3-0.6B
        // step peaks at 1.5 GB against the full bf16 logits' 2.6 GB.
        let n = hidden.dim(0);
        let f32_ = Dtype::Float32.as_i32();
        let mut lse = Array::from_f32(f32::NEG_INFINITY).broadcast_to(&[n]);
        let mut target = ops::zeros(&[n], Dtype::Float32);
        let mut h = hidden.clone();
        let mut start = 0;
        for (i, w) in weights.into_iter().enumerate() {
            let len = w.dim(0);
            // Where each target sits in this chunk, and whether it does at
            // all. Constants, so captured rather than passed: the closure's
            // inputs are what the checkpoint differentiates.
            let local = targets.subtract(&Array::from_i32(start));
            let inside = local
                .greater_equal(&Array::from_i32(0))
                .logical_and(&local.less(&Array::from_i32(len)))
                .as_dtype(f32_);
            let local = local
                .maximum(&Array::from_i32(0))
                .minimum(&Array::from_i32(len - 1))
                .expand_dims(-1);
            start += len;

            let mut inputs = vec![h, w, lse, target];
            if let Some(biases) = &biases {
                inputs.push(biases[i].clone());
            }
            let out = checkpoint_apply(&inputs, move |a| {
                let mut z = a[0].matmul(&a[1].t());
                if let Some(b) = a.get(4) {
                    z = z.add(&b.as_dtype(z.dtype().as_i32()));
                }
                let z = transform.apply(z);
                let picked = z.take_along_axis(&local, -1).squeeze_axes(&[-1]);
                let lse = a[2].logaddexp(&z.logsumexp(-1, false));
                let h = a[0].add(
                    &lse.expand_dims(-1)
                        .multiply(&Array::from_f32(0.0))
                        .as_dtype(a[0].dtype().as_i32()),
                );
                vec![lse, a[3].add(&picked.multiply(&inside)), h]
            });
            pmetal_bridge::check_last_error()
                .map_err(|e| Exception::custom(format!("cut cross-entropy chunk {i}: {e}")))?;
            [lse, target, h] = out.try_into().map_err(|_| {
                Exception::custom("cut cross-entropy: a chunk returned the wrong outputs")
            })?;
        }
        Ok((lse, target))
    }
}

/// The logit transforms after the head, applied in f32.
#[derive(Debug, Clone, Copy)]
struct Transform {
    scale: f32,
    softcap: f32,
}

impl Transform {
    fn apply(self, logits: Array) -> Array {
        let mut z = logits.as_dtype(Dtype::Float32.as_i32());
        if self.scale != 1.0 {
            z = z.multiply(&Array::from_f32(self.scale));
        }
        if self.softcap > 0.0 {
            let cap = Array::from_f32(self.softcap);
            z = ops::tanh(&z.divide(&cap)).multiply(&cap);
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::inline_array::value_and_grad;

    fn drain() {
        if let Err(e) = pmetal_bridge::check_last_error() {
            panic!("a bridge op threw: {e}");
        }
    }

    /// The loss, and its gradients with respect to the hidden states and the
    /// head, by the full logits in f32.
    fn full(
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        transform: Transform,
    ) -> (f32, Vec<Array>) {
        let (loss, grads) = value_and_grad(
            |a| {
                let z = transform.apply(a[0].matmul(&a[1].t()));
                pmetal_bridge::training::cross_entropy_loss(&z, targets, -100)
            },
            &[hidden.clone(), weight.clone()],
            &[],
        );
        drain();
        (loss.item_f32(), grads)
    }

    fn chunked(
        config: CutCrossEntropyConfig,
        hidden: &Array,
        weight: &Array,
        targets: &Array,
    ) -> (f32, Vec<Array>) {
        let cce = CutCrossEntropy::new(config);
        let (loss, grads) = value_and_grad(
            |a| cce.forward(&a[0], &a[1], targets, None).unwrap(),
            &[hidden.clone(), weight.clone()],
            &[],
        );
        drain();
        (loss.item_f32(), grads)
    }

    fn max_abs(a: &Array) -> f32 {
        a.abs().max(None).item_f32()
    }

    /// The loss and both gradients match the full logits', over a vocabulary
    /// that doesn't divide into the chunks, with repeated and ignored targets
    /// and a softcap.
    #[test]
    fn matches_the_full_logits_loss_and_gradients() {
        pmetal_bridge::compat::random::seed(11);
        let (n, h, v) = (9, 16, 70);
        let hidden = pmetal_bridge::compat::random::normal(&[n, h], Dtype::Float32);
        let weight = pmetal_bridge::compat::random::normal(&[v, h], Dtype::Float32)
            .multiply(&Array::from_f32(0.5));
        let targets = Array::from_i32_slice(&[3, 3, 69, -100, 0, 41, 3, -100, 64]);

        for softcap in [0.0, 5.0] {
            let transform = Transform {
                scale: 1.0,
                softcap,
            };
            let (want, want_grads) = full(&hidden, &weight, &targets, transform);
            let config = CutCrossEntropyConfig::new()
                .with_vocab_chunk_size(16)
                .with_softcap(softcap);
            let (got, got_grads) = chunked(config, &hidden, &weight, &targets);

            assert!(
                (got - want).abs() < 1e-5 * want.abs().max(1.0),
                "softcap {softcap}: loss {got} vs {want}"
            );
            for (name, g, w) in [
                ("hidden", &got_grads[0], &want_grads[0]),
                ("head", &got_grads[1], &want_grads[1]),
            ] {
                let diff = max_abs(&g.subtract(w));
                let scale = max_abs(w);
                assert!(
                    diff <= 1e-5 * scale,
                    "softcap {softcap}: {name} gradient off by {diff} (scale {scale})"
                );
            }
        }
    }

    /// On a bf16 head the loss is the bf16 logits' loss taken in f32: the
    /// logsumexp and the target logit lose nothing beyond the head's own
    /// rounding. Reducing a chunk's logits in bf16 instead is off by 6e-3
    /// here.
    #[test]
    fn a_bf16_head_is_reduced_in_f32() {
        pmetal_bridge::compat::random::seed(13);
        let (n, h, v) = (16, 64, 1000);
        let bf16 = Dtype::Bfloat16.as_i32();
        let hidden = pmetal_bridge::compat::random::normal(&[n, h], Dtype::Float32)
            .multiply(&Array::from_f32(2.0))
            .as_dtype(bf16);
        let weight = pmetal_bridge::compat::random::normal(&[v, h], Dtype::Float32)
            .multiply(&Array::from_f32(0.5))
            .as_dtype(bf16);
        let ids: Vec<i32> = (0..n).map(|i| (i * 61) % v).collect();
        let targets = Array::from_i32_slice(&ids);

        let want = pmetal_bridge::training::cross_entropy_loss(
            &hidden.matmul(&weight.t()).as_dtype(Dtype::Float32.as_i32()),
            &targets,
            -100,
        )
        .item_f32();
        let got = CutCrossEntropy::new(CutCrossEntropyConfig::new().with_vocab_chunk_size(128))
            .forward(&hidden, &weight, &targets, None)
            .unwrap()
            .item_f32();
        drain();
        assert!(
            (got - want).abs() <= 1e-5 * want.abs(),
            "loss {got} vs the bf16 logits' f32 loss {want}"
        );
    }

    /// Ignored targets contribute neither loss nor gradient, and a batch of
    /// nothing but ignored targets is a zero loss rather than a NaN.
    #[test]
    fn ignored_targets_count_for_nothing() {
        pmetal_bridge::compat::random::seed(5);
        let hidden = pmetal_bridge::compat::random::normal(&[4, 8], Dtype::Float32);
        let weight = pmetal_bridge::compat::random::normal(&[16, 8], Dtype::Float32);
        let config = || CutCrossEntropyConfig::new().with_vocab_chunk_size(4);

        let some = Array::from_i32_slice(&[0, -100, 5, -100]);
        let (_, grads) = chunked(config(), &hidden, &weight, &some);
        let ignored_rows = grads[0].take_axis(&Array::from_i32_slice(&[1, 3]), 0);
        assert_eq!(max_abs(&ignored_rows), 0.0);

        let none = Array::from_i32_slice(&[-100, -100, -100, -100]);
        let (loss, _) = chunked(config(), &hidden, &weight, &none);
        assert_eq!(loss, 0.0);
    }
}
