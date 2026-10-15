//! Cut cross-entropy: the cross-entropy of an LM head's logits without ever
//! holding all of them, forward or backward.
//!
//! Wijmans et al., "Cut Your Losses in Large-Vocabulary Language Models"
//! (arXiv 2411.09009), with the reference implementation at
//! <https://github.com/apple/ml-cross-entropy>. The loss of a token is
//! `logsumexp(logits) - logits[target]`, and both terms can be had without the
//! `[tokens, vocab]` logits.
//!
//! # The fused kernels
//!
//! On a Metal GPU the loss runs as fused kernels
//! ([`pmetal_runtime::inline_array::cut_cross_entropy_fused`]): the forward
//! builds 64×64 tiles of logits in the GEMM's registers and folds them into
//! each token's logsumexp and target logit as it goes. The backward uses the
//! softmax's sparsity as the paper does (gradient filtering): where the
//! softmax of a group of eight tokens stays under
//! [`CutCrossEntropyConfig::filter_eps`] across a tile of the vocabulary, that
//! group and tile are left out, and the forward records enough (each group's
//! largest logit per tile) to leave most of them out without recomputing
//! them. The rest are rebuilt and reduced into the gradients of the hidden
//! states and the head. The whole loss is one op with a custom VJP, so the
//! gradients reach the hidden states and the head, and any adapter folded
//! into the head, as through the full logits.
//!
//! # The chunked fallback
//!
//! Where the kernels can't run (no Metal GPU, a dtype other than f32, f16 or
//! bf16) the loss is built from MLX ops a vocabulary chunk at a time: a
//! chunk's logits, reduced to the chunk's `logsumexp` and the target logit if
//! the chunk holds it, dropped, and the chunks' `[tokens]` results combined.
//! Autograd would undo that by keeping every chunk's intermediates for the
//! backward pass, so each chunk runs under [`checkpoint_apply`], which keeps
//! only its inputs and outputs and rebuilds the chunk during the backward
//! pass. The fallback is taken with a warning, once per process.
//!
//! # Precision
//!
//! As in the reference implementation: the logits come out of the GEMM in the
//! model's dtype (the GEMM accumulates in f32 and rounds once, which is
//! exactly the logits the model's own head produces), and everything after
//! that (bias, scale, softcap, max, exp, sum, the log) runs in f32. The target
//! logit is read out of the same tile, so the two terms of the loss see the
//! same logits.

use std::sync::Once;

use pmetal_runtime::compat::{Array, Dtype, Exception, ops};
use pmetal_runtime::inline_array::{FusedCceParams, checkpoint_apply, cut_cross_entropy_fused};

/// Configuration for [`CutCrossEntropy`].
#[derive(Debug, Clone)]
pub struct CutCrossEntropyConfig {
    /// Vocabulary rows per chunk: the fused backward's logit gradients, in the
    /// head's dtype, and the fallback's logits, in f32, forward and backward,
    /// are `tokens × vocab_chunk_size` at a time.
    pub vocab_chunk_size: usize,
    /// Target id excluded from the loss.
    pub ignore_index: i32,
    /// `cap · tanh(logits / cap)` after the scale (Gemma 2, Gemma 4); 0
    /// disables it.
    pub softcap: f32,
    /// Multiplies the logits (Cohere, Granite); 1 disables it.
    pub logit_scale: f32,
    /// Gradient filtering, which trades gradient precision for backward
    /// speed: the fused backward leaves out a tile of the vocabulary for a
    /// group of eight tokens when the softmax there is below this for each
    /// of them. A larger value leaves out more, so the backward is faster
    /// and the gradient further from f32; the loss itself is exact either
    /// way.
    ///
    /// `None` takes the head dtype's machine epsilon over 512 (2⁻¹⁶ for
    /// bf16, 2⁻¹⁹ for f16, 2⁻³² for f32), where the gradient is as close to
    /// f32 as the full logits'. `Some(2f32.powi(-12))` is the reference
    /// implementation's default for bf16 (epsilon over 32): on Qwen3-0.6B
    /// over 4,096 tokens of varied text it ran the step at the full logits'
    /// speed where the default is 16% slower, with the hidden states'
    /// gradient 1.25e-2 from f32 instead of 9.7e-3. `Some(0.0)` leaves out
    /// nothing.
    pub filter_eps: Option<f32>,
}

impl Default for CutCrossEntropyConfig {
    fn default() -> Self {
        Self {
            vocab_chunk_size: 4096,
            ignore_index: -100,
            softcap: 0.0,
            logit_scale: 1.0,
            filter_eps: None,
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

    /// Set the gradient filter's threshold (see
    /// [`filter_eps`](Self::filter_eps)).
    pub fn with_filter_eps(mut self, eps: f32) -> Self {
        self.filter_eps = Some(eps);
        self
    }
}

/// The default filter: the head dtype's machine epsilon over 512.
///
/// The reference implementation filters at epsilon over 32 (2⁻¹² for bf16),
/// each dropped value too small to change a sum of order one. The values a
/// token drops add up, though: on Qwen3-0.6B over 4,096 tokens of varied
/// text, filtering at 2⁻¹² left the hidden states' gradient 1.25e-2 away
/// from f32 against the bf16 full logits' 9.6e-3, and 2⁻¹⁴ still 1.06e-2.
/// At 2⁻¹⁶ it is 9.7e-3, the full logits' own precision.
fn auto_filter_eps(dtype: Dtype) -> f32 {
    let epsilon = match dtype {
        Dtype::Bfloat16 => 2f32.powi(-7),
        Dtype::Float16 => 2f32.powi(-10),
        _ => f32::EPSILON,
    };
    epsilon / 512.0
}

/// The dtype of `a · bᵀ` in MLX: the wider of the two, and f32 for a mix of
/// f16 and bf16.
fn product_dtype(a: Dtype, b: Dtype) -> Dtype {
    match (a, b) {
        (Dtype::Bfloat16, Dtype::Bfloat16) => Dtype::Bfloat16,
        (Dtype::Float16, Dtype::Float16) => Dtype::Float16,
        _ => Dtype::Float32,
    }
}

static FALLBACK: Once = Once::new();

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
        self.check_shapes(hidden, weight)?;
        let targets = Self::labels(targets);
        let (lse, target) = match self.fused(hidden, weight, &targets, bias)? {
            Some(terms) => terms,
            None => self.chunked(hidden, weight, bias, &targets)?,
        };
        Ok(self.mean(&lse, &target, &targets))
    }

    /// [`forward`](Self::forward) by the chunked fallback, whatever the
    /// device: for comparing the two.
    pub fn forward_chunked(
        &self,
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        bias: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.check_shapes(hidden, weight)?;
        let targets = Self::labels(targets);
        let (lse, target) = self.chunked(hidden, weight, bias, &targets)?;
        Ok(self.mean(&lse, &target, &targets))
    }

    fn check_shapes(&self, hidden: &Array, weight: &Array) -> Result<(), Exception> {
        if hidden.ndim() != 2 || weight.ndim() != 2 || hidden.dim(1) != weight.dim(1) {
            return Err(Exception::custom(format!(
                "cut cross-entropy: hidden {:?} and head {:?} don't multiply",
                hidden.shape(),
                weight.shape()
            )));
        }
        Ok(())
    }

    /// Labels must never enter the trace: MLX ≥ 0.32 refuses a gather's
    /// gradient with respect to its indices.
    fn labels(targets: &Array) -> Array {
        targets.as_dtype(Dtype::Int32.as_i32()).stop_gradient()
    }

    /// The mean of `lse - target` over the targets that aren't ignored.
    fn mean(&self, lse: &Array, target: &Array, targets: &Array) -> Array {
        let valid = targets
            .not_equal(&Array::from_i32(self.config.ignore_index))
            .as_dtype(Dtype::Float32.as_i32());
        let per_token = lse.subtract(target).multiply(&valid);
        let count = valid.sum_all().maximum(&Array::from_f32(1.0));
        per_token.sum_all().divide(&count)
    }

    /// Each token's `logsumexp` and target logit from the fused kernels, or
    /// `None` (after saying so, once) when they can't take these inputs.
    fn fused(
        &self,
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        bias: Option<&Array>,
    ) -> Result<Option<(Array, Array)>, Exception> {
        let filter_eps = self
            .config
            .filter_eps
            .unwrap_or_else(|| auto_filter_eps(product_dtype(hidden.dtype(), weight.dtype())));
        let params = FusedCceParams {
            logit_scale: self.config.logit_scale,
            softcap: self.config.softcap,
            filter_eps,
            vocab_chunk: self.config.vocab_chunk_size,
        };
        match cut_cross_entropy_fused(hidden, weight, targets, bias, params) {
            Ok(Ok(out)) => Ok(Some((out.lse, out.target_logit))),
            Ok(Err(reason)) => {
                FALLBACK.call_once(|| {
                    tracing::warn!(
                        "cut cross-entropy: the fused Metal kernels can't run here ({reason}); \
                         computing the loss a vocabulary chunk at a time instead, which is slower"
                    );
                });
                Ok(None)
            }
            Err(e) => Err(Exception::custom(format!("cut cross-entropy: {e}"))),
        }
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
    ) -> Result<(Array, Array), Exception> {
        let transform = Transform {
            scale: self.config.logit_scale,
            softcap: self.config.softcap,
        };
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
            pmetal_runtime::check_unobserved_error()
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
    use pmetal_runtime::compat::random;
    use pmetal_runtime::inline_array::value_and_grad;

    fn drain() {
        if let Err(e) = pmetal_runtime::check_unobserved_error() {
            panic!("a bridge op threw: {e}");
        }
    }

    fn max_abs(a: &Array) -> f32 {
        a.abs().max(None).item_f32()
    }

    fn f32_(a: &Array) -> Array {
        a.as_dtype(Dtype::Float32.as_i32())
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Path {
        Fused,
        Chunked,
    }

    /// The loss by one path. The fused one must run: a fallback here would
    /// test the chunked path twice.
    fn loss_by(
        path: Path,
        cce: &CutCrossEntropy,
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        bias: Option<&Array>,
    ) -> Array {
        match path {
            Path::Fused => {
                let labels = CutCrossEntropy::labels(targets);
                let (lse, target) = cce
                    .fused(hidden, weight, &labels, bias)
                    .unwrap()
                    .expect("the fused kernels run on a Metal GPU");
                cce.mean(&lse, &target, &labels)
            }
            Path::Chunked => cce.forward_chunked(hidden, weight, targets, bias).unwrap(),
        }
    }

    /// `softcap(scale · (h·Wᵀ + b))` in the inputs' dtype up to the head's
    /// output, then f32: the logits the model's own head produces.
    fn full_logits(
        config: &CutCrossEntropyConfig,
        h: &Array,
        w: &Array,
        b: Option<&Array>,
    ) -> Array {
        let mut z = h.matmul(&w.t());
        if let Some(b) = b {
            z = z.add(&b.as_dtype(z.dtype().as_i32()));
        }
        Transform {
            scale: config.logit_scale,
            softcap: config.softcap,
        }
        .apply(z)
    }

    /// The loss and its gradients with respect to the hidden states, the head
    /// and the bias (when there is one), by `loss`.
    fn with_grads(
        loss: impl Fn(&Array, &Array, Option<&Array>) -> Array,
        hidden: &Array,
        weight: &Array,
        bias: Option<&Array>,
    ) -> (f32, Vec<Array>) {
        let mut params = vec![hidden.clone(), weight.clone()];
        params.extend(bias.cloned());
        let (value, grads) = value_and_grad(|a| loss(&a[0], &a[1], a.get(2)), &params, &[]);
        let value = value.item_f32();
        drain();
        (value, grads)
    }

    fn by_path(
        path: Path,
        config: &CutCrossEntropyConfig,
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        bias: Option<&Array>,
    ) -> (f32, Vec<Array>) {
        let cce = CutCrossEntropy::new(config.clone());
        with_grads(
            |h, w, b| loss_by(path, &cce, h, w, targets, b),
            hidden,
            weight,
            bias,
        )
    }

    fn by_full_logits(
        config: &CutCrossEntropyConfig,
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        bias: Option<&Array>,
    ) -> (f32, Vec<Array>) {
        with_grads(
            |h, w, b| {
                pmetal_runtime::training::cross_entropy_loss(
                    &full_logits(config, h, w, b),
                    targets,
                    config.ignore_index,
                )
            },
            hidden,
            weight,
            bias,
        )
    }

    /// Problems with `got` against `want`, each gradient within `tol` of the
    /// reference's largest entry.
    fn compare(
        what: &str,
        (got, got_grads): &(f32, Vec<Array>),
        (want, want_grads): &(f32, Vec<Array>),
        tol: f32,
    ) -> Vec<String> {
        let mut problems = Vec::new();
        let gap = (got - want).abs();
        if gap.is_nan() || gap > tol * want.abs().max(1.0) {
            problems.push(format!("{what}: loss {got} vs {want}"));
        }
        for (name, g, w) in ["hidden", "head", "bias"]
            .iter()
            .zip(got_grads)
            .zip(want_grads)
            .map(|((n, g), w)| (n, g, w))
        {
            let diff = max_abs(&f32_(g).subtract(&f32_(w)));
            let scale = max_abs(&f32_(w));
            if diff.is_nan() || diff > tol * scale {
                problems.push(format!(
                    "{what}: {name} gradient off by {diff} (scale {scale})"
                ));
            }
        }
        problems
    }

    struct Case {
        hidden: Array,
        weight: Array,
        bias: Array,
        targets: Array,
    }

    /// 70 tokens (a full tile and a partial one), 200 vocabulary rows (three
    /// full tiles and a partial one), a width that is no multiple of the
    /// GEMM's depth, targets on the tiles' edges, repeated and ignored.
    fn ragged_case(dtype: Dtype) -> Case {
        random::seed(11);
        let (n, h, v) = (70, 40, 200);
        let mut ids: Vec<i32> = (0..n).map(|i| (i * 37 + 5) % v).collect();
        for (i, t) in [0, 63, 64, 127, 128, 191, 192, 199, 199, 3, 3, -100, -100]
            .into_iter()
            .enumerate()
        {
            ids[i * 5] = t;
        }
        Case {
            hidden: random::normal(&[n, h], Dtype::Float32).as_dtype(dtype.as_i32()),
            weight: random::normal(&[v, h], Dtype::Float32)
                .multiply(&Array::from_f32(0.4))
                .as_dtype(dtype.as_i32()),
            bias: random::normal(&[v], Dtype::Float32).as_dtype(dtype.as_i32()),
            targets: Array::from_i32_slice(&ids),
        }
    }

    fn transforms() -> Vec<(&'static str, CutCrossEntropyConfig)> {
        let base = || CutCrossEntropyConfig::new().with_vocab_chunk_size(64);
        vec![
            ("plain", base()),
            ("scaled", base().with_logit_scale(0.7)),
            ("softcapped", base().with_logit_scale(1.3).with_softcap(5.0)),
        ]
    }

    /// Both paths give the full logits' loss and gradients (hidden states,
    /// head, bias) in f32, with every logit transform, over ragged tiles and
    /// several backward chunks.
    #[test]
    fn both_paths_match_the_full_logits() {
        let case = ragged_case(Dtype::Float32);
        let mut problems = Vec::new();
        for (name, config) in transforms() {
            let config = config.with_filter_eps(0.0);
            for bias in [None, Some(&case.bias)] {
                let want = by_full_logits(&config, &case.hidden, &case.weight, &case.targets, bias);
                for path in [Path::Fused, Path::Chunked] {
                    let got = by_path(
                        path,
                        &config,
                        &case.hidden,
                        &case.weight,
                        &case.targets,
                        bias,
                    );
                    let what = format!("{path:?} {name} bias={}", bias.is_some());
                    problems.extend(compare(&what, &got, &want, 2e-5));
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    /// The fused path's gradients against a central finite difference of its
    /// own loss, along each gradient: the loss changes at the rate `|g|` along
    /// `g / |g|` exactly when the gradient is the loss's. Richardson over
    /// `H`, `H / 2` and `H / 4` cancels the difference's `h²` error.
    #[test]
    fn fused_gradients_match_finite_differences() {
        const H: f32 = 1e-2;
        let case = ragged_case(Dtype::Float32);
        let mut problems = Vec::new();
        for (name, config) in transforms() {
            let cce = CutCrossEntropy::new(config);
            let loss = |h: &Array, w: &Array, b: &Array| {
                let l = loss_by(Path::Fused, &cce, h, w, &case.targets, Some(b)).item_f32();
                drain();
                l
            };
            let (value, grads) = with_grads(
                |h, w, b| loss_by(Path::Fused, &cce, h, w, &case.targets, b),
                &case.hidden,
                &case.weight,
                Some(&case.bias),
            );
            let rounding = 128.0 * f32::EPSILON * value.abs().max(1.0) / H;
            for (i, which) in ["hidden", "head", "bias"].into_iter().enumerate() {
                let norm = grads[i].square().sum_all().sqrt().item_f32();
                let u = grads[i].divide(&Array::from_f32(norm));
                let at = |step: f32| {
                    let mut p = [case.hidden.clone(), case.weight.clone(), case.bias.clone()];
                    p[i] = p[i].add(&u.multiply(&Array::from_f32(step)));
                    loss(&p[0], &p[1], &p[2])
                };
                let central = |h: f32| (at(h) - at(-h)) / (2.0 * h);
                let d = [central(H), central(H / 2.0), central(H / 4.0)];
                let fd = [(4.0 * d[1] - d[0]) / 3.0, (4.0 * d[2] - d[1]) / 3.0];
                for fd in fd {
                    let gap = (fd - norm).abs();
                    if gap.is_nan() || gap > 2e-3 * norm + rounding {
                        problems.push(format!(
                            "{name}: {which} autograd |g| {norm} vs finite difference {fd}"
                        ));
                    }
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    /// On a bf16 head the loss is the bf16 logits' loss taken in f32: the
    /// logsumexp and the target logit lose nothing beyond the head's own
    /// rounding. Reducing the logits in bf16 instead is off by 6e-3 here.
    #[test]
    fn a_bf16_head_is_reduced_in_f32() {
        random::seed(13);
        let (n, h, v) = (16, 64, 1000);
        let bf16 = Dtype::Bfloat16.as_i32();
        let hidden = random::normal(&[n, h], Dtype::Float32)
            .multiply(&Array::from_f32(2.0))
            .as_dtype(bf16);
        let weight = random::normal(&[v, h], Dtype::Float32)
            .multiply(&Array::from_f32(0.5))
            .as_dtype(bf16);
        let ids: Vec<i32> = (0..n).map(|i| (i * 61) % v).collect();
        let targets = Array::from_i32_slice(&ids);

        let want = pmetal_runtime::training::cross_entropy_loss(
            &f32_(&hidden.matmul(&weight.t())),
            &targets,
            -100,
        )
        .item_f32();
        let cce = CutCrossEntropy::new(CutCrossEntropyConfig::new().with_vocab_chunk_size(128));
        for path in [Path::Fused, Path::Chunked] {
            let got = loss_by(path, &cce, &hidden, &weight, &targets, None).item_f32();
            drain();
            assert!(
                (got - want).abs() <= 1e-5 * want.abs(),
                "{path:?}: loss {got} vs the bf16 logits' f32 loss {want}"
            );
        }
    }

    /// In bf16, the fused gradients are the full logits' (with the softmax
    /// in f32) to bf16's precision, with the default filter on.
    #[test]
    fn a_bf16_head_trains_like_its_logits() {
        let case = ragged_case(Dtype::Bfloat16);
        let mut problems = Vec::new();
        for (name, config) in transforms() {
            let want = by_full_logits(
                &config,
                &case.hidden,
                &case.weight,
                &case.targets,
                Some(&case.bias),
            );
            let got = by_path(
                Path::Fused,
                &config,
                &case.hidden,
                &case.weight,
                &case.targets,
                Some(&case.bias),
            );
            problems.extend(compare(name, &got, &want, 1e-2));
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    /// Gradient filtering: a vocabulary tile whose softmax is under the
    /// threshold for every token (here about e⁻²⁰) gets no gradient at all,
    /// and leaving it out moves the hidden states' gradient by less than
    /// bf16 can show. With the filter off, the same tile gets its (tiny)
    /// gradient.
    #[test]
    fn filtering_drops_what_the_softmax_does_not_reach() {
        random::seed(17);
        let (n, h, v) = (64, 32, 128);
        let bf16 = Dtype::Bfloat16.as_i32();
        let hidden = random::normal(&[n, h], Dtype::Float32)
            .multiply(&Array::from_f32(0.3))
            .as_dtype(bf16);
        let weight = random::normal(&[v, h], Dtype::Float32)
            .multiply(&Array::from_f32(0.3))
            .as_dtype(bf16);
        // The second tile sits 20 under the first.
        let shift: Vec<f32> = (0..v).map(|c| if c < 64 { 0.0 } else { -20.0 }).collect();
        let bias = Array::from_f32_slice(&shift, &[v]).as_dtype(bf16);
        let ids: Vec<i32> = (0..n).map(|i| (i * 7) % 64).collect();
        let targets = Array::from_i32_slice(&ids);
        let second = Array::from_i32_slice(&(64..v).collect::<Vec<_>>());

        let run = |eps: f32| {
            let config = CutCrossEntropyConfig::new().with_filter_eps(eps);
            by_path(
                Path::Fused,
                &config,
                &hidden,
                &weight,
                &targets,
                Some(&bias),
            )
        };
        let (_, filtered) = run(2f32.powi(-12));
        let (_, exact) = run(0.0);
        assert_eq!(max_abs(&f32_(&filtered[1].take_axis(&second, 0))), 0.0);
        assert_eq!(max_abs(&f32_(&filtered[2].take_axis(&second, 0))), 0.0);
        assert!(max_abs(&f32_(&exact[1].take_axis(&second, 0))) > 0.0);
        let moved = max_abs(&f32_(&filtered[0]).subtract(&f32_(&exact[0])));
        assert!(
            moved <= 2f32.powi(-9) * max_abs(&f32_(&exact[0])),
            "filtering moved the hidden states' gradient by {moved}"
        );
    }

    /// The forward's record can only bound a group of eight tokens: its
    /// largest logit in a tile against its smallest logsumexp. Here one token
    /// has a small logsumexp and another large logits in the second tile, so
    /// the bound keeps that tile, yet each token's softmax there is about
    /// e⁻²⁵. The recomputed softmax is what decides, and the tile still gets
    /// no gradient.
    #[test]
    fn filtering_decides_on_the_recomputed_softmax() {
        let (n, h, v) = (8, 8, 128);
        let mut w = vec![0.0f32; (v * h) as usize];
        for c in 0..v as usize {
            // The first tile reads feature 0 three times over, the second
            // feature 1 once.
            w[c * h as usize + if c < 64 { 0 } else { 1 }] = if c < 64 { 3.0 } else { 1.0 };
        }
        let mut x = vec![0.0f32; (n * h) as usize];
        for r in 0..n as usize {
            // Row 0: logits 30 in the first tile, 10 in the second.
            // The rest: 6 in the first tile, -15 in the second.
            let (a, b) = if r == 0 { (10.0, 10.0) } else { (2.0, -15.0) };
            x[r * h as usize] = a;
            x[r * h as usize + 1] = b;
        }
        let hidden = Array::from_f32_slice(&x, &[n, h]);
        let weight = Array::from_f32_slice(&w, &[v, h]);
        let targets = Array::from_i32_slice(&(0..n).map(|i| i * 5).collect::<Vec<_>>());
        let second = Array::from_i32_slice(&(64..v).collect::<Vec<_>>());
        let run = |eps: f32| {
            let config = CutCrossEntropyConfig::new().with_filter_eps(eps);
            by_path(Path::Fused, &config, &hidden, &weight, &targets, None)
        };
        let (_, filtered) = run(2f32.powi(-12));
        let (_, exact) = run(0.0);
        assert_eq!(max_abs(&filtered[1].take_axis(&second, 0)), 0.0);
        assert!(max_abs(&exact[1].take_axis(&second, 0)) > 0.0);
    }

    /// Ignored targets contribute neither loss nor gradient, and a batch of
    /// nothing but ignored targets is a zero loss rather than a NaN.
    #[test]
    fn ignored_targets_count_for_nothing() {
        random::seed(5);
        let hidden = random::normal(&[4, 8], Dtype::Float32);
        let weight = random::normal(&[16, 8], Dtype::Float32);
        let config = CutCrossEntropyConfig::new().with_vocab_chunk_size(4);
        for path in [Path::Fused, Path::Chunked] {
            let some = Array::from_i32_slice(&[0, -100, 5, -100]);
            let (_, grads) = by_path(path, &config, &hidden, &weight, &some, None);
            let ignored_rows = grads[0].take_axis(&Array::from_i32_slice(&[1, 3]), 0);
            assert_eq!(max_abs(&ignored_rows), 0.0, "{path:?}");

            let none = Array::from_i32_slice(&[-100, -100, -100, -100]);
            let (loss, _) = by_path(path, &config, &hidden, &weight, &none, None);
            assert_eq!(loss, 0.0, "{path:?}");
        }
    }

    /// A fused call that fails (here, one target short) reports an error,
    /// on its own and from inside a traced loss: there the bridge can't
    /// panic in MLX's callback, and the error comes back to the loss as an
    /// `Err` rather than a result computed on placeholders.
    #[test]
    fn a_failing_call_reports_its_error() {
        random::seed(9);
        let hidden = random::normal(&[4, 8], Dtype::Float32);
        let weight = random::normal(&[16, 8], Dtype::Float32);
        let short = Array::from_i32_slice(&[1, 2, 3]);
        let cce = CutCrossEntropy::new(CutCrossEntropyConfig::new());

        let err = pmetal_runtime::expect_runtime_errors(|| {
            cce.forward(&hidden, &weight, &short, None).unwrap_err()
        });
        assert!(err.to_string().contains("3 targets for 4 tokens"), "{err}");

        let seen = std::cell::RefCell::new(None);
        let (loss, _) = value_and_grad(
            |a| match cce.forward(&a[0], &weight, &short, None) {
                Ok(loss) => loss,
                Err(e) => {
                    *seen.borrow_mut() = Some(e.to_string());
                    Array::from_f32(0.0)
                }
            },
            std::slice::from_ref(&hidden),
            &[],
        );
        assert_eq!(loss.item_f32(), 0.0);
        let seen = seen.into_inner().expect("the traced call reported nothing");
        assert!(seen.contains("3 targets for 4 tokens"), "{seen}");
        drain();
    }

    /// Inputs of fewer than eight elements, which MLX hands a kernel in
    /// constant rather than device memory.
    #[test]
    fn tiny_inputs() {
        random::seed(7);
        let hidden = random::normal(&[3, 2], Dtype::Float32);
        let weight = random::normal(&[3, 2], Dtype::Float32);
        let targets = Array::from_i32_slice(&[2, 0, 1]);
        let config = CutCrossEntropyConfig::new().with_filter_eps(0.0);
        let want = by_full_logits(&config, &hidden, &weight, &targets, None);
        let got = by_path(Path::Fused, &config, &hidden, &weight, &targets, None);
        let problems = compare("tiny", &got, &want, 2e-5);
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    /// A second derivative (the gradient of a function of the hidden states'
    /// gradient, with respect to the head) goes through MLX ops, since the
    /// kernels have no VJP of their own, and matches the full logits'.
    #[test]
    fn a_second_derivative_matches_the_full_logits() {
        let case = ragged_case(Dtype::Float32);
        let config = CutCrossEntropyConfig::new()
            .with_vocab_chunk_size(64)
            .with_softcap(5.0)
            .with_filter_eps(0.0);
        let cce = CutCrossEntropy::new(config.clone());
        let outer = |inner: &dyn Fn(&Array, &Array) -> Array| {
            let (value, grads) = value_and_grad(
                |a| {
                    let (_, g) = value_and_grad(
                        |b| inner(&b[0], &a[0]),
                        std::slice::from_ref(&case.hidden),
                        &[],
                    );
                    g[0].square().sum_all()
                },
                std::slice::from_ref(&case.weight),
                &[],
            );
            let value = value.item_f32();
            drain();
            (value, grads)
        };
        let got = outer(&|h, w| loss_by(Path::Fused, &cce, h, w, &case.targets, None));
        let want = outer(&|h, w| {
            pmetal_runtime::training::cross_entropy_loss(
                &full_logits(&config, h, w, None),
                &case.targets,
                -100,
            )
        });
        let problems = compare("second derivative", &got, &want, 1e-4);
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }
}
