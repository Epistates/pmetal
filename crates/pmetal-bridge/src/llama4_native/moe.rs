//! Feed-forward: dense SwiGLU MLP for non-MoE layers + top-1 SwitchGLU MoE
//! with shared expert.

use crate::InlineArray;
use crate::native_weight::LayerWeight;

use super::weights::{LayerWeights, MoeWeights};

pub(super) fn dense_mlp_forward(lw: &LayerWeights, x: &InlineArray) -> InlineArray {
    let gate = x.matmul(lw.mlp_gate_w.as_ref().unwrap());
    let up = x.matmul(lw.mlp_up_w.as_ref().unwrap());
    let act = InlineArray::fused_swiglu(&gate, &up);
    act.matmul(lw.mlp_down_w.as_ref().unwrap())
}

/// MoE forward pass matching Python's `MoE.__call__`.
///
/// Python:
/// ```python
/// logits = self.router(x)
/// indices = argpartition(-logits, kth=0, axis=-1)[..., :1]  # top-1
/// scores = take_along_axis(logits, indices, axis=-1)
/// scores = sigmoid(scores.astype(float32)).astype(x.dtype)
/// out = self.experts(x * scores, indices).squeeze(2)
/// return out + self.shared_expert(x)
/// ```
///
/// Weight layout in `moe` after sanitization:
///   - `experts_gate_w`: `[E, hidden, expert_h]` — used directly as gather_mm B matrix
///   - `experts_up_w`:   `[E, hidden, expert_h]`
///   - `experts_down_w`: `[E, expert_h, hidden]`
///
/// We reshape inputs to `[B*T, 1, hidden]` for gather_mm and `rhs_indices=[B*T, 1]`.
pub(super) fn moe_forward(moe: &MoeWeights, x: &InlineArray, b: i32, s: i32) -> InlineArray {
    let hidden_size = x.dim(2);
    let dtype = x.dtype_raw();

    // Router: [B, T, hidden] × [hidden, num_experts] → [B, T, num_experts]
    let logits = x.matmul(&moe.router_w);

    // Top-1 selection: argpartition(-logits, kth=0)  places the top-1 index
    // at position 0. Flatten to [B*T, num_experts], slice first column → [B*T, 1].
    let bt = b * s;
    let num_experts = logits.dim(2);
    let neg_logits = logits.negative();
    let partition = neg_logits.argpartition(0, -1);
    let part_flat = partition.reshape(&[bt, num_experts]);
    let indices_flat = part_flat.slice(&[0, 0], &[bt, 1]); // [bt, 1] top-1 expert index
    let indices = indices_flat.reshape(&[b, s, 1]); // [B, T, 1]

    // Gather scores for the top-1 expert, apply sigmoid in f32 for numerical stability.
    let scores_raw = logits.take_along_axis(&indices, -1); // [B, T, 1]
    // (Dtype id 0 is bool, which turned every logit into 1 and every gate
    // into sigmoid(1).)
    let scores_sig = scores_raw
        .as_dtype(crate::dtype::F32)
        .sigmoid()
        .as_dtype(dtype); // [B, T, 1]

    // Scale the routed input.
    let x_scaled = x.multiply(&scores_sig); // [B, T, hidden]

    // The shared routed-expert dispatch, with the gate already applied to
    // the input (so the output weights are 1).
    let x_flat = x_scaled.reshape(&[bt, hidden_size]);
    let ones = InlineArray::ones(&[bt, 1], dtype);
    let routed_out = crate::native_moe::switch_glu(
        &x_flat,
        &LayerWeight::Dense(moe.experts_gate_w.clone()),
        &LayerWeight::Dense(moe.experts_up_w.clone()),
        &LayerWeight::Dense(moe.experts_down_w.clone()),
        &indices_flat,
        &ones,
        InlineArray::fused_swiglu,
    )
    .reshape(&[b, s, hidden_size]);

    // Shared expert: standard dense SwiGLU MLP
    let sh_gate = x.matmul(&moe.shared_gate_w);
    let sh_up = x.matmul(&moe.shared_up_w);
    let sh_act = InlineArray::fused_swiglu(&sh_gate, &sh_up);
    let shared_out = sh_act.matmul(&moe.shared_down_w);

    routed_out.add(&shared_out)
}
