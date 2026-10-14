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
///
/// From [`GROUPING_THRESHOLD`] `(token, expert)` pairs up (a prefill) the pairs
/// are grouped by expert before the matmuls, see [`switch_glu_grouped`].
pub fn switch_glu(
    x_flat: &InlineArray,
    gate: &LayerWeight,
    up: &LayerWeight,
    down: &LayerWeight,
    inds: &InlineArray,
    scores: &InlineArray,
    activation: impl Fn(&InlineArray, &InlineArray) -> InlineArray,
) -> InlineArray {
    let grouped = x_flat.dim(0) * inds.dim(1) >= GROUPING_THRESHOLD;
    switch_glu_grouped(x_flat, gate, up, down, inds, scores, activation, grouped)
}

/// From this many `(token, expert)` pairs up, [`switch_glu`] groups the pairs
/// by expert. Below it (a decode step, a short draft block) the sort costs more
/// than the kernel it enables saves.
pub const GROUPING_THRESHOLD: i32 = 64;

/// [`switch_glu`] with the choice of order made by the caller.
///
/// With `grouped` the `(token, expert)` pairs are sorted by expert before the
/// gather matmuls and put back after, so each expert's rows form one
/// contiguous run. In router order every row is its own tiny matmul against a
/// different expert; grouped, MLX's gather kernels (`sorted_indices`) tile a
/// whole run of rows against one expert's weights, which is what makes a
/// prefill through a routed layer compute-bound rather than launch-bound. It
/// is `SwitchGLU`'s `_gather_sort` and `_scatter_unsort`. The two orders give
/// the same result.
#[allow(clippy::too_many_arguments)]
pub fn switch_glu_grouped(
    x_flat: &InlineArray,
    gate: &LayerWeight,
    up: &LayerWeight,
    down: &LayerWeight,
    inds: &InlineArray,
    scores: &InlineArray,
    activation: impl Fn(&InlineArray, &InlineArray) -> InlineArray,
    grouped: bool,
) -> InlineArray {
    let s = x_flat.dim(0);
    let top_k = inds.dim(1);

    let per_expert = if grouped {
        per_expert_grouped(x_flat, gate, up, down, inds, &activation)
    } else {
        per_expert_in_router_order(x_flat, gate, up, down, inds, &activation)
    };

    per_expert
        .multiply(&scores.reshape(&[s, top_k, 1]))
        .sum_axis(-2, false)
}

/// Each row through its experts, in the order the router chose them:
/// `[S, top_k, hidden]`.
fn per_expert_in_router_order(
    x_flat: &InlineArray,
    gate: &LayerWeight,
    up: &LayerWeight,
    down: &LayerWeight,
    inds: &InlineArray,
    activation: &impl Fn(&InlineArray, &InlineArray) -> InlineArray,
) -> InlineArray {
    // [S, 1, 1, hidden] -> [S, top_k, 1, intermediate]
    let switch_in = switch_glu_input(x_flat);
    let gated = gate.gather_mm_from(&switch_in, None, Some(inds), false);
    let upped = up.gather_mm_from(&switch_in, None, Some(inds), false);
    let activated = activation(&gated, &upped);

    // [S, top_k, 1, hidden] -> [S, top_k, hidden]
    down.gather_mm_from(&activated, None, Some(inds), false)
        .squeeze(-2)
}

/// [`per_expert_in_router_order`] with the `(token, expert)` pairs sorted by
/// expert, so the gather matmuls see each expert's rows as one contiguous run.
fn per_expert_grouped(
    x_flat: &InlineArray,
    gate: &LayerWeight,
    up: &LayerWeight,
    down: &LayerWeight,
    inds: &InlineArray,
    activation: &impl Fn(&InlineArray, &InlineArray) -> InlineArray,
) -> InlineArray {
    let (s, top_k) = (inds.dim(0), inds.dim(1));
    let pairs = s * top_k;

    // The expert and the token of every pair, in router order.
    let experts = inds.reshape(&[pairs]);
    let tokens = InlineArray::arange(s, crate::compat::Dtype::Int32.as_i32())
        .reshape(&[s, 1])
        .broadcast_to(&[s, top_k])
        .reshape(&[pairs]);

    // `order[i]` is the router-order slot of the i-th pair by expert; its
    // inverse puts the results back.
    let order = experts.argsort(-1);
    let inverse = order.argsort(-1);
    let sorted_experts = experts.take_axis(&order, 0);
    let sorted_tokens = tokens.take_axis(&order, 0);

    // [S * top_k, 1, hidden] -> [S * top_k, 1, intermediate] -> [S * top_k, 1, hidden]
    let rows = x_flat.take_axis(&sorted_tokens, 0).expand_dims(1);
    let gated = gate.gather_mm_from(&rows, None, Some(&sorted_experts), true);
    let upped = up.gather_mm_from(&rows, None, Some(&sorted_experts), true);
    let activated = activation(&gated, &upped);
    let down = down.gather_mm_from(&activated, None, Some(&sorted_experts), true);

    let hidden = down.dim(-1);
    down.take_axis(&inverse, 0).reshape(&[s, top_k, hidden])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuantizedMode;
    use crate::native_weight::QuantParams;

    fn values(mut a: InlineArray, n: usize) -> Vec<f32> {
        a.to_f32_vec(n).expect("read back")
    }

    /// `[E, rows, cols]` stacks of small distinct values.
    fn stack(experts: usize, rows: usize, cols: usize, salt: usize) -> (Vec<f32>, InlineArray) {
        let data: Vec<f32> = (0..experts * rows * cols)
            .map(|i| (((i * 37 + salt * 11) % 23) as f32 - 11.0) / 17.0)
            .collect();
        let array = InlineArray::from_f32_slice(&data, &[experts as i32, rows as i32, cols as i32]);
        (data, array)
    }

    /// Two distinct experts per token, spread unevenly so some get many rows
    /// and one gets none.
    fn routes(tokens: usize, experts: usize) -> Vec<[u32; 2]> {
        (0..tokens)
            .map(|s| {
                let first = (s * 7 + 3) % (experts - 1);
                let second = (first + 1 + s % (experts - 2)) % (experts - 1);
                [first as u32, second as u32]
            })
            .collect()
    }

    /// The layer, one row at a time in f64:
    /// `Σ_k score · down(silu(gate(x)) · up(x))`.
    fn reference(
        x: &[f32],
        [gate, up, down]: [&[f32]; 3],
        routes: &[[u32; 2]],
        scores: &[f32],
        (hidden, inter): (usize, usize),
    ) -> Vec<f32> {
        let mut out = vec![0f64; routes.len() * hidden];
        for (s, route) in routes.iter().enumerate() {
            for (k, &e) in route.iter().enumerate() {
                let e = e as usize;
                let project = |w: &[f32], rows: usize, cols: usize, input: &[f64]| -> Vec<f64> {
                    (0..cols)
                        .map(|c| {
                            (0..rows)
                                .map(|r| input[r] * f64::from(w[(e * rows + r) * cols + c]))
                                .sum()
                        })
                        .collect()
                };
                let row: Vec<f64> = x[s * hidden..(s + 1) * hidden]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                let g = project(gate, hidden, inter, &row);
                let u = project(up, hidden, inter, &row);
                let act: Vec<f64> = g
                    .iter()
                    .zip(&u)
                    .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                let y = project(down, inter, hidden, &act);
                for (h, y) in y.iter().enumerate() {
                    out[s * hidden + h] += f64::from(scores[s * 2 + k]) * y;
                }
            }
        }
        out.into_iter().map(|v| v as f32).collect()
    }

    fn run_dense(tokens: usize) {
        let (experts, hidden, inter) = (5usize, 6usize, 4usize);
        let (gate, gate_w) = stack(experts, hidden, inter, 1);
        let (up, up_w) = stack(experts, hidden, inter, 2);
        let (down, down_w) = stack(experts, inter, hidden, 3);
        let x: Vec<f32> = (0..tokens * hidden)
            .map(|i| ((i * 13 % 19) as f32 - 9.0) / 9.0)
            .collect();
        let routes = routes(tokens, experts);
        let scores: Vec<f32> = (0..tokens * 2)
            .map(|i| 0.25 + (i % 3) as f32 * 0.2)
            .collect();

        let flat_routes: Vec<u32> = routes.iter().flatten().copied().collect();
        let got = switch_glu(
            &InlineArray::from_f32_slice(&x, &[tokens as i32, hidden as i32]),
            &LayerWeight::Dense(gate_w),
            &LayerWeight::Dense(up_w),
            &LayerWeight::Dense(down_w),
            &InlineArray::from_u32_slice(&flat_routes, &[tokens as i32, 2]),
            &InlineArray::from_f32_slice(&scores, &[tokens as i32, 2]),
            InlineArray::fused_swiglu,
        );
        crate::check_unobserved_error().expect("switch_glu");
        let want = reference(&x, [&gate, &up, &down], &routes, &scores, (hidden, inter));
        for (i, (g, w)) in values(got, tokens * hidden).iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                "{tokens} tokens, element {i}: {g} against {w}"
            );
        }
    }

    /// Past the threshold the pairs are grouped by expert and put back after
    /// the matmuls: every token must still get its own experts' output, in its
    /// own row, weighted by its own scores, as the per-row sum says. Below the
    /// threshold nothing is reordered.
    #[test]
    fn grouping_by_expert_changes_no_row() {
        assert!(48 * 2 >= GROUPING_THRESHOLD && 16 * 2 < GROUPING_THRESHOLD);
        run_dense(48);
        run_dense(16);
    }

    /// A packed stack with one tensor scale per expert: the scale must follow
    /// the expert each grouped pair was routed to, not its router-order slot.
    #[test]
    fn grouped_packed_experts_keep_their_own_tensor_scale() {
        let (experts, hidden, inter, tokens) = (4usize, 64usize, 32usize, 40usize);
        let params = QuantParams::defaults_for(QuantizedMode::Nvfp4);
        let packed = |salt: usize, rows: usize, cols: usize| {
            // `[E, out, in]`, as the checkpoint stores it.
            let (_, w) = stack(experts, cols, rows, salt);
            let (weight, scales) =
                w.quantize_weights_mode(params.group_size, params.bits, params.mode);
            LayerWeight::new_scaled(
                weight,
                Some(scales),
                None,
                Some(InlineArray::from_f32_slice(
                    &[0.5, 1.5, 3.0, 0.75],
                    &[experts as i32],
                )),
                params,
            )
        };
        let (gate, up, down) = (
            packed(1, hidden, inter),
            packed(2, hidden, inter),
            packed(3, inter, hidden),
        );
        let x_values: Vec<f32> = (0..tokens * hidden)
            .map(|i| ((i * 13 % 19) as f32 - 9.0) / 9.0)
            .collect();
        let x = InlineArray::from_f32_slice(&x_values, &[tokens as i32, hidden as i32]);
        let flat_routes: Vec<u32> = routes(tokens, experts).iter().flatten().copied().collect();
        let inds = InlineArray::from_u32_slice(&flat_routes, &[tokens as i32, 2]);
        let scores = InlineArray::from_f32_slice(&vec![0.5; tokens * 2], &[tokens as i32, 2]);

        let grouped = switch_glu(
            &x,
            &gate,
            &up,
            &down,
            &inds,
            &scores,
            InlineArray::fused_swiglu,
        );
        let in_router_order =
            per_expert_in_router_order(&x, &gate, &up, &down, &inds, &InlineArray::fused_swiglu)
                .multiply(&scores.reshape(&[tokens as i32, 2, 1]))
                .sum_axis(-2, false);
        crate::check_unobserved_error().expect("packed switch_glu");

        let n = tokens * hidden;
        let (got, want) = (values(grouped, n), values(in_router_order, n));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs() / (1.0 + w.abs()))
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-3,
            "grouped and router-order products differ by {worst}"
        );
    }
}
