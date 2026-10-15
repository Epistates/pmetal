//! MLX MoE combine operations.
//!
//! Pure MLX ops for MoE expert output combination. The previous Metal side-channel
//! (`fused_moe_combine`) has been removed — the 6 MLX ops are already async on GPU
//! and adding synchronization barriers (4x eval() + waitUntilCompleted) made the
//! Metal path 5-20x slower than the MLX ops it replaced.

use pmetal_runtime::compat::{Array, Exception};

/// MoE combine: weighted expert sum + sigmoid-gated shared expert.
///
/// Computes:
/// ```text
/// y = (expert_outs * weights.unsqueeze(-1)).sum(-2)  // weighted sum
/// shared_gate = sigmoid(shared_gate_logit)
/// y += shared_gate * shared_out
/// ```
///
/// This is the MoE block's output alone. The decoder layer adds its own
/// residual, the same as it does for a dense MLP; adding the block's input here
/// as well counted the normalized hidden state twice.
///
/// All ops run asynchronously on GPU via MLX's lazy evaluation graph.
///
/// # Arguments
/// * `expert_outs` - Expert outputs `[batch_seq, K, D]`
/// * `expert_weights` - Routing weights `[batch_seq, K]`
/// * `shared_out` - Shared expert output `[batch_seq, D]`
/// * `shared_gate_logit` - Shared expert gate logit `[batch_seq, 1]` or scalar
/// * `k` - Number of active experts (top-k)
/// * `batch_seq` - Batch * sequence length
pub fn moe_combine_mlx(
    expert_outs: &Array,
    expert_weights: &Array,
    shared_out: &Array,
    shared_gate_logit: &Array,
    k: i32,
    batch_seq: i32,
) -> Result<Array, Exception> {
    let y = expert_outs
        .multiply(&expert_weights.reshape(&[batch_seq, k, 1]))
        .sum_axis(-2, false);

    // Shared expert with gate
    let shared_gate = shared_gate_logit.sigmoid();
    let shared_y = shared_gate.multiply(shared_out);

    Ok(y.add(&shared_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f64_reference::{Rng, arr, assert_close, drain, round32, vec_of};

    /// The combine in f64: `sum_k w_k e_k + sigmoid(g) s` per token.
    fn reference(
        outs: &[f64],
        weights: &[f64],
        shared: &[f64],
        gate: &[f64],
        (tokens, k, dim): (usize, usize, usize),
    ) -> Vec<f64> {
        let mut y = vec![0.0; tokens * dim];
        for t in 0..tokens {
            let g = 1.0 / (1.0 + (-gate[t % gate.len()]).exp());
            for d in 0..dim {
                let routed: f64 = (0..k)
                    .map(|j| weights[t * k + j] * outs[(t * k + j) * dim + d])
                    .sum();
                y[t * dim + d] = routed + g * shared[t * dim + d];
            }
        }
        y
    }

    /// The block's output is the weighted expert sum plus the gated shared
    /// expert, for a gate per token and for one scalar gate, and carries no
    /// residual (the decoder layer adds its own: adding the block's input here
    /// as well counted the normalized hidden state twice).
    #[test]
    fn the_combine_is_the_weighted_sum_plus_the_gated_shared_expert() {
        let (tokens, k, dim) = (5usize, 3usize, 8usize);
        let mut rng = Rng(3);
        let mut draw = |n: usize| (0..n).map(|_| round32(rng.normal())).collect::<Vec<_>>();
        let outs = draw(tokens * k * dim);
        let weights = draw(tokens * k);
        let shared = draw(tokens * dim);
        for (what, gate, gate_shape) in [
            ("per-token gate", draw(tokens), vec![tokens, 1]),
            ("scalar gate", draw(1), vec![]),
        ] {
            let got = moe_combine_mlx(
                &arr(&outs, &[tokens, k, dim]),
                &arr(&weights, &[tokens, k]),
                &arr(&shared, &[tokens, dim]),
                &arr(&gate, &gate_shape),
                k as i32,
                tokens as i32,
            )
            .unwrap();
            drain();
            let want = reference(&outs, &weights, &shared, &gate, (tokens, k, dim));
            assert_close(what, &vec_of(&got), &want, 1e-5);
        }

        // Silent experts give nothing: no residual leaks in.
        let zeros = vec![0.0; tokens * dim];
        let got = moe_combine_mlx(
            &arr(&vec![0.0; tokens * k * dim], &[tokens, k, dim]),
            &arr(&weights, &[tokens, k]),
            &arr(&zeros, &[tokens, dim]),
            &arr(&vec![0.0; tokens], &[tokens, 1]),
            k as i32,
            tokens as i32,
        )
        .unwrap();
        assert!(vec_of(&got).iter().all(|&v| v == 0.0));
    }
}
