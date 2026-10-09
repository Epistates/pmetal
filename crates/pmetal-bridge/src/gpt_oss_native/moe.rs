//! MoE forward: biased top-k router with a softmax over the chosen logits,
//! then the routed experts ([`super::GptOssExperts`]), as transformers'
//! `GptOssMLP` and the gpt-oss reference implementation compute them.

use crate::InlineArray;

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
    let (inds, scores) = route(&x, &lw.moe_router_w, &lw.moe_router_b, lw.moe_top_k);
    lw.moe_experts
        .forward(&x, &inds, &scores, lw.swiglu_alpha, lw.swiglu_limit)
        .reshape(&[b, s, hidden_size])
}

/// gpt-oss's router: the `top_k` largest of `x @ w + b` (`w` `[hidden, E]`)
/// and a softmax over just those. Only the order of the logits picks the
/// experts; the indices carry no gradient.
pub fn route(
    x_flat: &InlineArray,
    w: &InlineArray,
    b: &InlineArray,
    top_k: i32,
) -> (InlineArray, InlineArray) {
    let logits = x_flat.matmul(w).add(b);
    let (rows, n_experts) = (logits.dim(0), logits.dim(1));
    let inds = logits
        .argpartition(-top_k, -1)
        .slice(&[0, n_experts - top_k], &[rows, n_experts])
        .stop_gradient();
    let scores = logits.take_along_axis(&inds, -1).softmax_precise(-1);
    (inds, scores)
}
