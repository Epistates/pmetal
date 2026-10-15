//! Feed-forward: dense SwiGLU MLP for non-MoE layers + Llama 4's sigmoid-gated
//! top-k MoE with a shared expert.

use crate::InlineArray;
use crate::dtype;

use super::weights::{Mlp, MoeWeights};

pub(super) fn mlp_forward(mlp: &Mlp, x: &InlineArray) -> InlineArray {
    let gate = mlp.gate.matmul_from(x);
    let up = mlp.up.matmul_from(x);
    mlp.down.matmul_from(&InlineArray::fused_swiglu(&gate, &up))
}

/// Llama 4's router from its logits: each row's `top_k` experts by raw logit,
/// `[rows, top_k]` indices, and their gates `sigmoid(logit)`, `[rows, top_k]`
/// in `out_dtype`.
///
/// `argpartition(-logits, k - 1)[:, :k]` is the top k. The indices carry no
/// gradient (MLX rejects a gather VJP taken with respect to indices that
/// descend from differentiable scores). The sigmoid is taken in f32, as
/// transformers takes it, and then cast to `out_dtype` (dtype id 0 is bool,
/// which once turned every logit into 1).
pub fn route_logits(
    logits: &InlineArray,
    top_k: i32,
    out_dtype: i32,
) -> (InlineArray, InlineArray) {
    let rows = logits.dim(0);
    let inds = logits
        .negative()
        .argpartition(top_k - 1, -1)
        .slice(&[0, 0], &[rows, top_k])
        .stop_gradient();
    let gates = logits
        .take_along_axis(&inds, -1)
        .as_dtype(dtype::F32)
        .sigmoid()
        .as_dtype(out_dtype);
    (inds, gates)
}

/// Llama 4's MoE (`Llama4TextMoe`).
///
/// ```text
///   logits  = router(x)                                  [B, T, E]
///   experts = top_k(logits)                              [B, T, k]
///   gates   = sigmoid(logits[experts]) in f32, cast back [B, T, k]
///   out     = Σ_j expert_{experts[j]}(x · gates[j])  +  shared_expert(x)
/// ```
///
/// The gate scales an expert's *input*, not its output (the expert is not
/// linear), and the gated outputs are summed without normalisation. Every
/// release routes to one expert, so `k` is 1 in practice; the sum is the
/// transformers' for any `k`.
pub(super) fn moe_forward(moe: &MoeWeights, x: &InlineArray, b: i32, s: i32) -> InlineArray {
    let hidden = x.dim(2);
    let dtype = x.dtype_raw();
    let rows = b * s;
    let k = moe.top_k;

    // Router: [B, T, hidden] -> [rows, num_experts]
    let logits = moe.router.matmul_from(x);
    let logits = logits.reshape(&[rows, logits.dim(2)]);

    let (inds, gates) = route_logits(&logits, k, dtype);

    // The shared routed-expert dispatch, the gate applied to each expert's
    // input.
    let x_flat = x.reshape(&[rows, hidden]);
    let routed = crate::native_moe::switch_glu_gated_input(
        &x_flat,
        crate::native_moe::GateUp::Split {
            gate: &moe.experts_gate,
            up: &moe.experts_up,
        },
        &moe.experts_down,
        &inds,
        &gates,
        InlineArray::fused_swiglu,
    );

    let shared = mlp_forward(&moe.shared, &x_flat);
    routed.add(&shared).reshape(&[b, s, hidden])
}
