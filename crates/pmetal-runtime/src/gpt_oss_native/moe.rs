//! MoE forward: biased top-k router with a softmax over the chosen logits,
//! then the routed experts ([`super::GptOssExperts`]), as transformers'
//! `GptOssMLP` and the gpt-oss reference implementation compute them.

use crate::InlineArray;
use crate::native_weight::LayerWeight;

use super::weights::LayerWeights;

/// GPT-OSS MoE forward pass.
///
/// ```text
///   logits  = x @ router_w + router_b                  [S, E]
///   top     = top_k(logits)                            [S, k]
///   weights = softmax(top)                             [S, k]
///   y       = Σ_k weights · (down(glu(gate(x) + b_g, up(x) + b_u)) + b_d)
/// ```
pub(super) fn moe_forward(lw: &LayerWeights, normed: &InlineArray, b: i32, s: i32) -> InlineArray {
    let hidden_size = normed.dim(2);
    let rows = b * s;
    let x = normed.reshape(&[rows, hidden_size]);
    let (inds, scores) = route(&x, &lw.moe_router, &lw.moe_router_b, lw.moe_top_k);
    lw.moe_experts
        .forward(&x, &inds, &scores, lw.swiglu_alpha, lw.swiglu_limit)
        .reshape(&[b, s, hidden_size])
}

/// gpt-oss's router: the `top_k` largest of `router(x) + b` (`router`
/// `hidden -> E`, dense or packed) and a softmax over just those. Only the
/// order of the logits picks the experts; the indices carry no gradient.
pub fn route(
    x_flat: &InlineArray,
    router: &LayerWeight,
    b: &InlineArray,
    top_k: i32,
) -> (InlineArray, InlineArray) {
    route_logits(&router.matmul_from(x_flat).add(b), top_k)
}

/// [`route`] from the biased router logits, `[rows, experts]`: the shared-path
/// model computes them with its own router layer and selects here, so there is
/// one top-k and one softmax.
pub fn route_logits(logits: &InlineArray, top_k: i32) -> (InlineArray, InlineArray) {
    let (rows, n_experts) = (logits.dim(0), logits.dim(1));
    let inds = logits
        .argpartition(-top_k, -1)
        .slice(&[0, n_experts - top_k], &[rows, n_experts])
        .stop_gradient();
    let scores = logits.take_along_axis(&inds, -1).softmax_precise(-1);
    (inds, scores)
}
