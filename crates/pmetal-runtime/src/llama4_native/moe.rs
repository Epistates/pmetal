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

    // `argpartition(-logits, k - 1)[:, :k]` is the top k. The indices carry
    // no gradient (MLX rejects a gather VJP taken with respect to indices
    // that descend from differentiable scores).
    let inds = logits
        .negative()
        .argpartition(k - 1, -1)
        .slice(&[0, 0], &[rows, k])
        .stop_gradient();

    // The gates: the sigmoid is taken in f32, as transformers takes it (dtype
    // id 0 is bool, which once turned every logit into 1).
    let gates = logits
        .take_along_axis(&inds, -1)
        .as_dtype(dtype::F32)
        .sigmoid()
        .as_dtype(dtype);

    // The shared routed-expert dispatch, one slot at a time, with the gate
    // already applied to the input (so the output weights are 1).
    let x_flat = x.reshape(&[rows, hidden]);
    let ones = InlineArray::ones(&[rows, 1], dtype);
    let mut routed: Option<InlineArray> = None;
    for j in 0..k {
        let slot = inds.slice(&[0, j], &[rows, j + 1]);
        let gated = x_flat.multiply(&gates.slice(&[0, j], &[rows, j + 1]));
        let y = crate::native_moe::switch_glu(
            &gated,
            &moe.experts_gate,
            &moe.experts_up,
            &moe.experts_down,
            &slot,
            &ones,
            InlineArray::fused_swiglu,
        );
        routed = Some(match routed {
            Some(sum) => sum.add(&y),
            None => y,
        });
    }
    let routed = routed.expect("top_k is at least 1");

    let shared = mlp_forward(&moe.shared, &x_flat);
    routed.add(&shared).reshape(&[b, s, hidden])
}
