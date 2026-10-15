//! Routed-expert dispatch shared by the native engines.
//!
//! Every MoE here runs the same SwitchGLU shape as the reference `SwitchGLU`: each
//! row goes through its `top_k` experts' gate and up projections, an
//! activation joins them, the down projection brings each back to the hidden
//! size, and the router's scores weight the sum. Architectures differ in the
//! router and the activation, so those are the caller's.

use crate::InlineArray;
use crate::native_weight::LayerWeight;

/// The top-k softmax router of Qwen3-MoE and the Qwen3.5 family, as
/// transformers' `Qwen3MoeTopKRouter` and `Qwen3_5MoeTopKRouter` compute it:
/// `[S, experts]` router logits to each row's `top_k` experts and their
/// weights, `([S, top_k] indices, [S, top_k] weights)`.
///
/// The softmax runs in f32 whatever the model's dtype
/// (`softmax(logits, dtype=torch.float)`), the experts are the `top_k` largest
/// probabilities, and with `norm_topk_prob` the weights are renormalized to
/// sum to one, still in f32; only then are they cast to `out_dtype`, the dtype
/// of the expert outputs they scale. Done in bf16 throughout, the
/// probabilities would round to 8 bits before the experts were picked and
/// again when renormalized.
///
/// The indices are not differentiable: gradients reach the router through the
/// weights alone (MLX >= 0.32 refuses a gather's VJP with respect to indices).
pub fn route_top_k(
    router_logits: &InlineArray,
    top_k: i32,
    norm_topk_prob: bool,
    out_dtype: i32,
) -> (InlineArray, InlineArray) {
    let (rows, experts) = (router_logits.dim(0), router_logits.dim(1));
    let probs = router_logits
        .as_dtype(crate::compat::Dtype::Float32.as_i32())
        .softmax_precise(-1);
    // `argpartition` puts the `top_k` largest in the last `top_k` slots.
    let inds = probs
        .argpartition(-top_k, -1)
        .slice(&[0, experts - top_k], &[rows, experts])
        .stop_gradient();
    let mut weights = probs.take_along_axis(&inds, -1);
    if norm_topk_prob {
        weights = weights.divide(&weights.sum_axis(-1, true));
    }
    (inds, weights.as_dtype(out_dtype))
}

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
    switch_glu_with(x_flat, gate, up, down, inds, scores, |gate, up, _| {
        activation(gate, up)
    })
}

/// From this many `(token, expert)` pairs up, [`switch_glu`] groups the pairs
/// by expert, as MLX's `SwitchGLU` does. Below it (a decode step, a short
/// draft block) the sort costs more than the kernel it enables saves.
pub const GROUPING_THRESHOLD: i32 = 64;

/// [`switch_glu`] with an activation that sees which expert each row of its
/// inputs went through, for activations with per-expert terms (gpt-oss's
/// expert biases).
///
/// `expert_ids` has the leading shape of `gate` and `up` less their last two
/// axes: `[S, top_k]` as the router gave it, or `[S * top_k]` when the pairs
/// were grouped by expert. An activation that is elementwise in its inputs
/// ignores it; one that adds a per-expert tensor takes
/// `bias.take_axis(expert_ids, 0).expand_dims(-2)` and so has the right
/// layout either way.
pub fn switch_glu_with(
    x_flat: &InlineArray,
    gate: &LayerWeight,
    up: &LayerWeight,
    down: &LayerWeight,
    inds: &InlineArray,
    scores: &InlineArray,
    activation: impl Fn(&InlineArray, &InlineArray, &InlineArray) -> InlineArray,
) -> InlineArray {
    let grouped = x_flat.dim(0) * inds.dim(1) >= GROUPING_THRESHOLD;
    switch_glu_grouped(x_flat, gate, up, down, inds, scores, activation, grouped)
}

/// [`switch_glu_with`] with the choice of order made by the caller.
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
    activation: impl Fn(&InlineArray, &InlineArray, &InlineArray) -> InlineArray,
    grouped: bool,
) -> InlineArray {
    let (s, top_k) = (x_flat.dim(0), inds.dim(1));
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
    activation: &impl Fn(&InlineArray, &InlineArray, &InlineArray) -> InlineArray,
) -> InlineArray {
    // [S, 1, 1, hidden] -> [S, top_k, 1, intermediate]
    let switch_in = switch_glu_input(x_flat);
    let gated = gate.gather_mm_from(&switch_in, None, Some(inds), false);
    let upped = up.gather_mm_from(&switch_in, None, Some(inds), false);
    let activated = activation(&gated, &upped, inds);

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
    activation: &impl Fn(&InlineArray, &InlineArray, &InlineArray) -> InlineArray,
) -> InlineArray {
    let (s, top_k) = (inds.dim(0), inds.dim(1));
    let pairs = s * top_k;

    // The expert and the token of every pair, in router order. The ids carry
    // no gradient; they only say where each row goes.
    let experts = inds.reshape(&[pairs]).stop_gradient();
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
    let activated = activation(&gated, &upped, &sorted_experts);
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

    /// A bf16 router, as transformers runs it: probabilities and the
    /// renormalization in f32, the weights cast to bf16 last. The same
    /// logits through bf16 probabilities and a bf16 sum land on other
    /// bf16 values (the test checks that they do, so it can tell the two
    /// apart), and the experts are the same top-2 either way here.
    #[test]
    fn a_bf16_router_softmaxes_and_renormalizes_in_f32() {
        let bf16 = crate::compat::Dtype::Bfloat16.as_i32();
        let (rows, experts, top_k) = (24usize, 8usize, 2usize);
        let logits: Vec<f32> = (0..rows * experts)
            .map(|i| (((i * 53 + 7) % 47) as f32 - 23.0) / 6.0)
            .collect();
        // What the model hands the router: logits already rounded to bf16.
        let rounded =
            InlineArray::from_f32_slice(&logits, &[rows as i32, experts as i32]).as_dtype(bf16);
        let host_logits = values(
            rounded.as_dtype(crate::compat::Dtype::Float32.as_i32()),
            rows * experts,
        );

        let (inds, weights) = route_top_k(&rounded, top_k as i32, true, bf16);
        crate::check_unobserved_error().expect("route");
        let inds = {
            let mut a = inds.as_dtype(crate::compat::Dtype::Int32.as_i32());
            a.to_f32_vec(rows * top_k).expect("indices")
        };
        let got = values(
            weights.as_dtype(crate::compat::Dtype::Float32.as_i32()),
            rows * top_k,
        );

        // f64 on the host, rounded to bf16 once at the end.
        let to_bf16 = |v: f64| {
            values(
                InlineArray::from_f32_slice(&[v as f32], &[1]).as_dtype(bf16),
                1,
            )[0]
        };
        let mut bf16_pipeline_differs = false;
        for r in 0..rows {
            let row = &host_logits[r * experts..(r + 1) * experts];
            let top = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f64> = row.iter().map(|&x| f64::from(x - top).exp()).collect();
            let total: f64 = exp.iter().sum();
            let probs: Vec<f64> = exp.iter().map(|e| e / total).collect();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let chosen = &order[..top_k];
            let kept: f64 = chosen.iter().map(|&e| probs[e]).sum();
            for k in 0..top_k {
                let (e, w) = (inds[r * top_k + k] as usize, got[r * top_k + k]);
                let want = chosen
                    .iter()
                    .position(|&c| c == e)
                    .map(|_| to_bf16(probs[e] / kept))
                    .unwrap_or_else(|| panic!("row {r}: expert {e} is not a top-{top_k} expert"));
                assert_eq!(w, want, "row {r}, expert {e}");
            }
            // The bf16 pipeline: probabilities rounded, then their sum.
            let rounded_probs: Vec<f64> = chosen
                .iter()
                .map(|&e| f64::from(to_bf16(probs[e])))
                .collect();
            let rounded_sum = f64::from(to_bf16(rounded_probs.iter().sum()));
            bf16_pipeline_differs |= chosen
                .iter()
                .zip(&rounded_probs)
                .any(|(&e, p)| to_bf16(p / rounded_sum) != to_bf16(probs[e] / kept));
        }
        assert!(
            bf16_pipeline_differs,
            "the logits do not tell an f32 router from a bf16 one"
        );
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
        const { assert!(48 * 2 >= GROUPING_THRESHOLD && 16 * 2 < GROUPING_THRESHOLD) };
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
            per_expert_in_router_order(&x, &gate, &up, &down, &inds, &|g, u, _| {
                InlineArray::fused_swiglu(g, u)
            })
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

/// The dispatch against loops that run one expert on one row at a time:
/// per-expert activation terms, gradients, and both sides of the threshold.
#[cfg(test)]
mod dispatch_tests {
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
