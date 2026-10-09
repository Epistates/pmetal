//! MoE forward: biased top-k router with a softmax over the chosen logits,
//! and the clamped GLU every expert runs, as transformers' `GptOssMLP` and
//! the gpt-oss reference implementation compute them.

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
///
/// The experts' gate and up projections are the even and odd columns of the
/// checkpoint's fused `gate_up_proj`, split at load time.
pub(super) fn moe_forward(lw: &LayerWeights, normed: &InlineArray, b: i32, s: i32) -> InlineArray {
    let hidden_size = normed.dim(2);
    let rows = b * s;
    let x = normed.reshape(&[rows, hidden_size]);

    // Router: [S, E]. Only the order of the logits picks the experts, and the
    // softmax runs over the selected ones alone (not over all E).
    let logits = x.matmul(&lw.moe_router_w).add(&lw.moe_router_b);
    let n_experts = lw.moe_num_experts;
    let top_k = lw.moe_top_k;
    let inds = logits
        .argpartition(-top_k, -1)
        .slice(&[0, n_experts - top_k], &[rows, n_experts])
        // Gradients flow through the selected logits, never the indices.
        .stop_gradient();
    let scores = logits.take_along_axis(&inds, -1).softmax_precise(-1);

    // Per-row expert biases: [S, k, 1, I] beside the `[S, k, 1, I]` gather
    // outputs `switch_glu` hands the activation.
    let inter = lw.moe_gate_b.dim(1);
    let bias_for = |bias: &InlineArray| bias.take_axis(&inds, 0).reshape(&[rows, top_k, 1, inter]);
    let (gate_b, up_b) = (bias_for(&lw.moe_gate_b), bias_for(&lw.moe_up_b));
    let (alpha, limit) = (lw.swiglu_alpha, lw.swiglu_limit);
    let routed = crate::native_moe::switch_glu(
        &x,
        &lw.moe_gate_w,
        &lw.moe_up_w,
        &lw.moe_down_w,
        &inds,
        &scores,
        |gate, up| gpt_oss_glu(&gate.add(&gate_b), &up.add(&up_b), alpha, limit),
    );

    // The down bias, weighted like the expert outputs it belongs to:
    // Σ_k weights[s, k] · b_d[inds[s, k]].
    let down_b = lw
        .moe_down_b
        .take_axis(&inds, 0)
        .multiply(&scores.reshape(&[rows, top_k, 1]))
        .sum_axis(1, false);

    routed.add(&down_b).reshape(&[b, s, hidden_size])
}

/// gpt-oss's clamped GLU, transformers' `GptOssExperts._apply_gate`:
///
/// ```text
///   gate = min(gate, limit)
///   up   = clip(up, -limit, limit)
///   out  = gate · σ(alpha · gate) · (up + 1)
/// ```
#[inline]
fn gpt_oss_glu(gate: &InlineArray, up: &InlineArray, alpha: f32, limit: f32) -> InlineArray {
    let hi = InlineArray::scalar_like(limit, gate);
    let lo = InlineArray::scalar_like(-limit, up);
    let gate = gate.minimum(&hi);
    let up = up.clip(Some(&lo), Some(&InlineArray::scalar_like(limit, up)));
    let glu = gate.multiply(
        &gate
            .multiply(&InlineArray::scalar_like(alpha, &gate))
            .sigmoid(),
    );
    glu.multiply(&up.add(&InlineArray::scalar_like(1.0, &up)))
}
