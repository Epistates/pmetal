//! MoE forward: sigmoid top-k routing + per-expert bias + clamped SwiGLU.

use crate::InlineArray;

use super::weights::LayerWeights;

/// GPT-OSS MoE forward pass with sigmoid routing and per-expert bias.
///
/// Routing:
///   1. `router_logits = hidden @ router_w`           [B*T, num_experts]
///   2. `scores = sigmoid(router_logits)`             [B*T, num_experts]
///   3. Top-k via argpartition(-k) → take_along_axis  [B*T, top_k]
///   4. Normalize: `weights = scores_topk / sum(scores_topk)`  (safe sum with 1e-8 floor)
///
/// Expert computation per slot s in [0..top_k):
///   1. Gather gate/up/down weight rows for expert_ids[:, s]  via take_axis
///   2. gate_out = batched_matmul(hidden_flat, gate_w[slot]) + gate_b[slot]
///   3. up_out   = batched_matmul(hidden_flat, up_w[slot])   + up_b[slot]
///   4. act      = gpt_oss_swiglu(gate_out, up_out)            (clamped + alpha-scaled)
///   5. slot_out = batched_matmul(act, down_w[slot])          + down_b[slot]
///   6. output  += slot_out * weights[:, s:s+1]
pub(super) fn moe_forward(lw: &LayerWeights, normed: &InlineArray, b: i32, s: i32) -> InlineArray {
    // Flatten to [B*T, hidden]
    let hidden_size = normed.dim(2);
    let bt = b * s;
    let hidden_flat = normed.reshape(&[bt, hidden_size]);

    // Router logits: [B*T, num_experts]
    let router_logits = hidden_flat.matmul(&lw.moe_router_w);

    // sigmoid scores
    let scores = router_logits.sigmoid();

    // Top-k: argpartition at kth = -top_k gives top-k indices in the last k slots
    let neg_k = -lw.moe_top_k;
    let partitioned = scores.argpartition(neg_k, -1); // [B*T, num_experts]
    let top_k_indices = partitioned
        .slice(
            &[0, lw.moe_num_experts - lw.moe_top_k],
            &[bt, lw.moe_num_experts],
        )
        // MLX >= 0.32: routing indices sit downstream of differentiable
        // scores, and gather VJP w.r.t. indices now raises — cut the trace
        // here (gradients flow through expert_weights, never the indices).
        .stop_gradient();
    // Re-cast to int32 for gather ops (argpartition returns int32 already, but ensure)
    let top_k_scores = scores.take_along_axis(&top_k_indices, -1); // [B*T, top_k]

    // Normalize: weights = scores / max(sum, 1e-8)
    let sum_scores = top_k_scores.sum_axis(-1, true); // [B*T, 1]
    let eps = InlineArray::scalar_like(1e-8, &sum_scores);
    let safe_sum = sum_scores.maximum(&eps);
    let expert_weights = top_k_scores.divide(&safe_sum); // [B*T, top_k]

    // Accumulate expert outputs
    let mut output = InlineArray::zeros(&[bt, hidden_size], hidden_flat.dtype_raw());

    for slot in 0..lw.moe_top_k {
        // Expert indices for this slot: [B*T]
        let slot_experts = top_k_indices
            .slice(&[0, slot], &[bt, slot + 1])
            .reshape(&[bt]);
        // Expert weights for this slot: [B*T, 1]
        let slot_weights = expert_weights.slice(&[0, slot], &[bt, slot + 1]);

        // Gather per-token expert weights from stacked tensors.
        // stacked shape: [num_experts, hidden, intermediate] (gate/up) or [num_experts, intermediate, hidden] (down)
        // take_axis(slot_experts, 0) → [B*T, hidden, intermediate]
        let gate_w = lw.moe_gate_w.take_axis(&slot_experts, 0); // [B*T, hidden, inter]
        let up_w = lw.moe_up_w.take_axis(&slot_experts, 0); // [B*T, hidden, inter]
        let down_w = lw.moe_down_w.take_axis(&slot_experts, 0); // [B*T, inter, hidden]
        let gate_b = lw.moe_gate_b.take_axis(&slot_experts, 0); // [B*T, inter]
        let up_b = lw.moe_up_b.take_axis(&slot_experts, 0); // [B*T, inter]
        let down_b = lw.moe_down_b.take_axis(&slot_experts, 0); // [B*T, hidden]

        // Batched matmul: [B*T, 1, hidden] @ [B*T, hidden, inter] → [B*T, 1, inter] → [B*T, inter]
        let h_exp = hidden_flat.reshape(&[bt, 1, hidden_size]);
        let gate_out = h_exp.matmul(&gate_w).reshape(&[bt, -1]).add(&gate_b);
        let up_out = h_exp.matmul(&up_w).reshape(&[bt, -1]).add(&up_b);

        // GPT-OSS SwiGLU activation with clamping
        let act = gpt_oss_swiglu(&gate_out, &up_out, lw.swiglu_alpha, lw.swiglu_limit);

        // Down projection: [B*T, 1, inter] @ [B*T, inter, hidden] → [B*T, hidden]
        let act_exp = act.reshape(&[bt, 1, -1]);
        let slot_out = act_exp
            .matmul(&down_w)
            .reshape(&[bt, hidden_size])
            .add(&down_b);

        // Weighted accumulation
        output = output.add(&slot_out.multiply(&slot_weights));
    }

    // Restore [B, S, hidden]
    output.reshape(&[b, s, hidden_size])
}

/// GPT-OSS custom SwiGLU activation (from Python `swiglu` compiled function):
///
///   x_glu  = clip(x_glu,    a_max=limit)
///   x_lin  = clip(x_linear, a_min=-limit, a_max=limit)
///   out    = x_glu * sigmoid(alpha * x_glu) * (x_linear + 1)
///
/// This differs from standard SwiGLU (`silu(gate) * up`) in two ways:
///   1. Clamping is applied to prevent FP16 overflow at large values.
///   2. The linear branch gets a bias of +1 before the gate multiply.
///   3. The gate uses a parametric alpha (1.702) instead of 1.0.
///
/// `InlineArray` exposes `maximum` but not `minimum`.  Upper-clamp is
/// implemented as `−maximum(−x, −limit)` (de Morgan's min/max identity).
#[inline]
fn gpt_oss_swiglu(
    x_linear: &InlineArray,
    x_glu: &InlineArray,
    alpha: f32,
    limit: f32,
) -> InlineArray {
    let neg_limit_arr = InlineArray::scalar_like(-limit, x_glu);
    let alpha_arr = InlineArray::scalar_like(alpha, x_glu);
    let one_arr = InlineArray::scalar_like(1.0, x_linear);

    // clip(x_glu, a_max=limit) = -maximum(-x_glu, -limit)
    let x_glu_clamped = x_glu.negative().maximum(&neg_limit_arr).negative();
    // clip(x_linear, a_min=-limit, a_max=limit):
    //   lower: maximum(x_linear, -limit)
    //   upper: -maximum(-result, -limit)
    let x_lin_lo = x_linear.maximum(&neg_limit_arr);
    let x_lin_clamped = x_lin_lo.negative().maximum(&neg_limit_arr).negative();

    // sigmoid(alpha * x_glu)
    let glu_scaled = x_glu_clamped.multiply(&alpha_arr);
    let sig = glu_scaled.sigmoid();

    // out_glu = x_glu * sigmoid(alpha * x_glu)
    let out_glu = x_glu_clamped.multiply(&sig);

    // (x_linear + 1)
    let lin_biased = x_lin_clamped.add(&one_arr);

    // out = out_glu * (x_linear + 1)
    out_glu.multiply(&lin_biased)
}
