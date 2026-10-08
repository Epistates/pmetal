//! Routed-expert dispatch shared by the native engines.
//!
//! Every MoE here runs the same SwitchGLU shape as the reference `SwitchGLU`: each
//! row goes through its `top_k` experts' gate and up projections, an
//! activation joins them, the down projection brings each back to the hidden
//! size, and the router's scores weight the sum. Architectures differ in the
//! router and the activation, so those are the caller's.

use crate::InlineArray;
use crate::native_weight::LayerWeight;

/// `[S, hidden]` to the `[S, 1, 1, hidden]` layout `gather_mm` expects.
///
/// MLX's `SwitchGLU` does `mx.expand_dims(x, (-2, -3))`. Positive axes here
/// make the insertion order unambiguous and give the same layout for flattened
/// input.
#[inline]
pub fn switch_glu_input(x_flat: &InlineArray) -> InlineArray {
    debug_assert_eq!(x_flat.ndim(), 2);
    x_flat.expand_dims(1).expand_dims(2)
}

/// `Σ_k scores[s, k] · down(act(gate(x_s), up(x_s)))` over each row's experts
/// `inds[s, k]`.
///
/// `x_flat` is `[S, hidden]`, `inds` and `scores` are `[S, top_k]`, and the
/// result is `[S, hidden]`. The expert weights are stacked over a leading
/// expert axis, dense or packed.
///
/// ⚠️ The singleton axes from [`switch_glu_input`] are load-bearing: without
/// them the down projection can read the sequence axis as another batch
/// dimension and produce `[S, top_k, S, hidden]`, which then breaks the score
/// broadcast.
pub fn switch_glu(
    x_flat: &InlineArray,
    gate: &LayerWeight,
    up: &LayerWeight,
    down: &LayerWeight,
    inds: &InlineArray,
    scores: &InlineArray,
    activation: impl Fn(&InlineArray, &InlineArray) -> InlineArray,
) -> InlineArray {
    let s = x_flat.dim(0);
    let top_k = inds.dim(1);

    // [S, 1, 1, hidden] -> [S, top_k, 1, intermediate]
    let switch_in = switch_glu_input(x_flat);
    let gated = gate.gather_mm_from(&switch_in, None, Some(inds), false);
    let upped = up.gather_mm_from(&switch_in, None, Some(inds), false);
    let activated = activation(&gated, &upped);

    // [S, top_k, 1, hidden] -> [S, top_k, hidden]
    let per_expert = down
        .gather_mm_from(&activated, None, Some(inds), false)
        .squeeze(-2);

    per_expert
        .multiply(&scores.reshape(&[s, top_k, 1]))
        .sum_axis(-2, false)
}
