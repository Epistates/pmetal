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
    switch_glu_with(x_flat, gate, up, down, inds, scores, |gate, up, _| {
        activation(gate, up)
    })
}

/// Rows times experts from which the dispatch groups the rows by expert.
///
/// MLX's `SwitchGLU` does the same at 64: below it the indices are few enough
/// that gathering one expert matrix per row costs about what grouping would,
/// and above it `gather_mm` over sorted indices runs a grouped matmul instead
/// of one tiny matmul per row.
const SORT_THRESHOLD: i32 = 64;

/// [`switch_glu`] with an activation that sees which expert each row of its
/// inputs went through, for activations with per-expert terms (gpt-oss's
/// expert biases).
///
/// `expert_ids` has the leading shape of `gate` and `up` less their last two
/// axes: `[S, top_k]` as the router gave it, or `[S * top_k]` when the rows
/// were grouped by expert. An activation that is elementwise in its inputs
/// ignores it; one that adds a per-expert tensor takes
/// `bias.take_axis(expert_ids, 0).expand_dims(-2)` and so has the right
/// layout either way.
///
/// ⚠️ Grouping by expert is a 6 to 16 times faster prefill (measured: 512
/// tokens, 64 experts of 2048 by 1408, top-6, 92 ms against 5.8 ms a layer),
/// and a slower decode, which is why it applies from 64 (row, expert)
/// pairs and not always.
pub fn switch_glu_with(
    x_flat: &InlineArray,
    gate: &LayerWeight,
    up: &LayerWeight,
    down: &LayerWeight,
    inds: &InlineArray,
    scores: &InlineArray,
    activation: impl Fn(&InlineArray, &InlineArray, &InlineArray) -> InlineArray,
) -> InlineArray {
    let s = x_flat.dim(0);
    let top_k = inds.dim(1);
    let rows = s * top_k;
    // Positions are carried through the sort as `f32`s, exact below 2^24.
    if (SORT_THRESHOLD..1 << 24).contains(&rows) {
        return switch_glu_sorted(x_flat, gate, up, down, inds, scores, top_k, activation);
    }

    // [S, 1, 1, hidden] -> [S, top_k, 1, intermediate]
    let switch_in = switch_glu_input(x_flat);
    let gated = gate.gather_mm_from(&switch_in, None, Some(inds), false);
    let upped = up.gather_mm_from(&switch_in, None, Some(inds), false);
    let activated = activation(&gated, &upped, inds);

    // [S, top_k, 1, hidden] -> [S, top_k, hidden]
    let per_expert = down
        .gather_mm_from(&activated, None, Some(inds), false)
        .squeeze(-2);

    per_expert
        .multiply(&scores.reshape(&[s, top_k, 1]))
        .sum_axis(-2, false)
}

/// The same sum with the `S * top_k` (row, expert) pairs ordered by expert,
/// so each expert's rows are contiguous for `gather_mm`'s sorted-index path,
/// and the result put back in row order before the scores weight it.
#[allow(clippy::too_many_arguments)]
fn switch_glu_sorted(
    x_flat: &InlineArray,
    gate: &LayerWeight,
    up: &LayerWeight,
    down: &LayerWeight,
    inds: &InlineArray,
    scores: &InlineArray,
    top_k: i32,
    activation: impl Fn(&InlineArray, &InlineArray, &InlineArray) -> InlineArray,
) -> InlineArray {
    let s = x_flat.dim(0);
    // The ids carry no gradient; they only say where each row goes.
    let flat = inds.reshape(&[s * top_k]).stop_gradient();
    let order = flat.argsort(0);
    let inv_order = order.argsort(0);
    let expert_ids = flat.take_axis(&order, 0);
    // Pair `p` is row `p / top_k`'s `p % top_k`-th expert.
    let source_rows = order
        .divide(&InlineArray::scalar_with_dtype(
            top_k as f32,
            crate::dtype::I32,
        ))
        .as_dtype(crate::dtype::I32);

    // [S * top_k, 1, hidden] -> [S * top_k, 1, intermediate]
    let switch_in = x_flat.take_axis(&source_rows, 0).expand_dims(1);
    let gated = gate.gather_mm_from(&switch_in, None, Some(&expert_ids), true);
    let upped = up.gather_mm_from(&switch_in, None, Some(&expert_ids), true);
    let activated = activation(&gated, &upped, &expert_ids);

    // [S * top_k, 1, hidden] -> [S * top_k, hidden] -> [S, top_k, hidden]
    let per_expert = down
        .gather_mm_from(&activated, None, Some(&expert_ids), true)
        .squeeze(-2);
    let hidden = per_expert.dim(1);
    per_expert
        .take_axis(&inv_order, 0)
        .reshape(&[s, top_k, hidden])
        .multiply(&scores.reshape(&[s, top_k, 1]))
        .sum_axis(-2, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values in roughly [-0.5, 0.5], all different.
    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn array(shape: &[i32], seed: u64) -> InlineArray {
        let n = shape.iter().product::<i32>() as usize;
        InlineArray::from_f32_slice(&values(n, seed), shape)
    }

    fn to_vec(a: &InlineArray) -> Vec<f32> {
        let mut a = a.clone();
        a.eval();
        let n = a.size();
        a.to_f32_vec(n).expect("f32 readback")
    }

    /// `s` rows through `top_k` of 8 experts (hidden 12, intermediate 20,
    /// distinct weights and biases for gate and up so a swap or a mixed-up
    /// expert fails), each expert's rows weighted by its own score, against a
    /// loop that runs one expert on one row at a time.
    fn check(s: i32, top_k: i32) {
        let (e, h, i) = (8, 12, 20);
        let x = array(&[s, h], 1);
        let (gate_w, up_w, down_w) = (
            array(&[e, h, i], 2),
            array(&[e, h, i], 3),
            array(&[e, i, h], 4),
        );
        let (gate_b, up_b) = (array(&[e, i], 5), array(&[e, i], 6));
        // Skewed, distinct within a row: a row's first expert is `(7 s) % 8`.
        let ids: Vec<i32> = (0..s)
            .flat_map(|r| (0..top_k).map(move |j| (r * 7 + j * 3 + (r % 3)) % e))
            .collect();
        let inds = InlineArray::from_i32_slice(&ids).reshape(&[s, top_k]);
        let scores = array(&[s, top_k], 7);

        let (gate, up, down) = (
            LayerWeight::Dense(gate_w.clone()),
            LayerWeight::Dense(up_w.clone()),
            LayerWeight::Dense(down_w.clone()),
        );
        let got = switch_glu_with(&x, &gate, &up, &down, &inds, &scores, |g, u, expert_ids| {
            let bias = |b: &InlineArray| b.take_axis(expert_ids, 0).expand_dims(-2);
            InlineArray::fused_swiglu(&g.add(&bias(&gate_b)), &u.add(&bias(&up_b)))
        });
        assert_eq!(got.shape(), vec![s, h]);

        let score_values = to_vec(&scores);
        let mut want = vec![0f32; (s * h) as usize];
        for r in 0..s {
            let xr = x.slice(&[r, 0], &[r + 1, h]);
            for j in 0..top_k {
                let ex = ids[(r * top_k + j) as usize];
                let w = |a: &InlineArray, rows: i32, cols: i32| {
                    a.slice(&[ex, 0, 0], &[ex + 1, rows, cols])
                        .reshape(&[rows, cols])
                };
                let b = |a: &InlineArray| a.slice(&[ex, 0], &[ex + 1, i]);
                let g = xr.matmul(&w(&gate_w, h, i)).add(&b(&gate_b));
                let u = xr.matmul(&w(&up_w, h, i)).add(&b(&up_b));
                let y = InlineArray::fused_swiglu(&g, &u).matmul(&w(&down_w, i, h));
                let weight = score_values[(r * top_k + j) as usize];
                for (acc, v) in want[(r * h) as usize..((r + 1) * h) as usize]
                    .iter_mut()
                    .zip(to_vec(&y))
                {
                    *acc += weight * v;
                }
            }
        }
        crate::check_unobserved_error().expect("bridge ops succeeded");
        let worst = to_vec(&got)
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-5, "s={s} top_k={top_k}: max |diff| {worst:e}");
    }

    /// Training sends gradients through the grouped dispatch: the input's and
    /// every expert weight's match those of a loop that runs one expert on
    /// one row at a time.
    #[test]
    fn grouped_dispatch_passes_the_per_row_loops_gradients() {
        use crate::inline_array::value_and_grad;
        let (s, top_k, e, h, i) = (30, 3, 8, 12, 20);
        let ids: Vec<i32> = (0..s)
            .flat_map(|r| (0..top_k).map(move |j| (r * 5 + j * 3 + (r % 3)) % e))
            .collect();
        let inds = InlineArray::from_i32_slice(&ids).reshape(&[s, top_k]);
        let scores = array(&[s, top_k], 7);
        let probe = array(&[s, h], 8);
        let constants = [inds.clone(), scores.clone(), probe];
        let inputs = [
            array(&[s, h], 1),
            array(&[e, h, i], 2),
            array(&[e, h, i], 3),
            array(&[e, i, h], 4),
        ];

        let grouped = |a: &[InlineArray]| {
            let (gate, up, down) = (
                LayerWeight::Dense(a[1].clone()),
                LayerWeight::Dense(a[2].clone()),
                LayerWeight::Dense(a[3].clone()),
            );
            let y = switch_glu(
                &a[0],
                &gate,
                &up,
                &down,
                &a[4],
                &a[5],
                InlineArray::fused_swiglu,
            );
            y.multiply(&a[6]).sum_all()
        };
        let per_row = |a: &[InlineArray]| {
            let mut rows = Vec::new();
            for r in 0..s {
                let xr = a[0].slice(&[r, 0], &[r + 1, h]);
                let mut acc = InlineArray::zeros(&[1, h], crate::dtype::F32);
                for j in 0..top_k {
                    let ex = ids[(r * top_k + j) as usize];
                    let w = |m: &InlineArray, rows: i32, cols: i32| {
                        m.slice(&[ex, 0, 0], &[ex + 1, rows, cols])
                            .reshape(&[rows, cols])
                    };
                    let act = InlineArray::fused_swiglu(
                        &xr.matmul(&w(&a[1], h, i)),
                        &xr.matmul(&w(&a[2], h, i)),
                    );
                    let weight = a[5].slice(&[r, j], &[r + 1, j + 1]);
                    acc = acc.add(&act.matmul(&w(&a[3], i, h)).multiply(&weight));
                }
                rows.push(acc);
            }
            let y = rows[1..]
                .iter()
                .fold(rows[0].clone(), |all, row| all.concatenate_2(row, 0));
            y.multiply(&a[6]).sum_all()
        };

        let (want_loss, want) = value_and_grad(per_row, &inputs, &constants);
        let (got_loss, got) = value_and_grad(grouped, &inputs, &constants);
        crate::check_unobserved_error().expect("bridge ops succeeded");
        assert!((got_loss.item_f32() - want_loss.item_f32()).abs() < 1e-4);
        for (n, (g, w)) in got.iter().zip(&want).enumerate() {
            let worst = to_vec(g)
                .iter()
                .zip(to_vec(w))
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let scale = to_vec(w).iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(scale > 1e-3, "gradient {n} is all zeros");
            assert!(
                worst < 1e-4 * scale.max(1.0),
                "gradient {n}: max |diff| {worst:e}"
            );
        }
    }

    /// Few rows take the per-row gather, many take the grouped one; both give
    /// the per-row loop's sum.
    #[test]
    fn dispatch_matches_a_per_row_loop_below_and_above_the_grouping_threshold() {
        check(5, 3); // 15 pairs: per-row gather
        check(21, 3); // 63 pairs: still per-row
        check(22, 3); // 66 pairs: grouped
        check(41, 2); // 82 pairs, other top_k: grouped
    }
}
