//! The pieces of a mixture-of-experts block that every architecture shares.
//!
//! Each architecture routes tokens itself (top-k selection lives in
//! `pmetal_models::moe_routing`) and runs its experts through stacked or
//! gathered matmuls that keep the routing weights in the graph. What is shared
//! is the expert itself, a bias-free SwiGLU MLP whose weights a checkpoint
//! fills ([`Expert`]), and the step that sums the routed experts' outputs with
//! the gated shared expert ([`moe_combine`]).

use crate::compat::{Array, Dtype, Exception, ops};
use crate::impl_module_params;

/// Minimal linear layer: weight [out, in], no bias, zero-initialised.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix, shape `[out_features, in_features]`.
    pub weight: Array,
}
impl_module_params!(Linear; weight);

impl Linear {
    /// Create a zero-initialised linear layer.
    pub fn new(in_features: i32, out_features: i32) -> Self {
        Self {
            weight: ops::zeros(&[out_features, in_features], Dtype::Float32),
        }
    }

    /// `y = x @ W^T`
    pub fn forward(&self, x: &Array) -> Array {
        x.matmul(&self.weight.t())
    }
}

/// Single expert MLP (SwiGLU).
#[derive(Debug)]
pub struct Expert {
    /// Gate projection, `[intermediate, hidden]`.
    pub w1: Linear,
    /// Up projection, `[intermediate, hidden]`.
    pub w3: Linear,
    /// Down projection, `[hidden, intermediate]`.
    pub w2: Linear,
}
impl_module_params!(Expert; w1, w3, w2);

impl Expert {
    /// Create a zero-initialised expert; a checkpoint fills the weights.
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Self {
        Self {
            w1: Linear::new(hidden_size, intermediate_size),
            w3: Linear::new(hidden_size, intermediate_size),
            w2: Linear::new(intermediate_size, hidden_size),
        }
    }

    /// `w2(silu(w1(x)) * w3(x))`.
    pub fn forward(&self, x: &Array) -> Array {
        let gate = self.w1.forward(x);
        // SwiGLU: silu(gate) * up
        let gate_activated = gate.silu();
        let up = self.w3.forward(x);
        let hidden = gate_activated.multiply(&up);
        self.w2.forward(&hidden)
    }
}

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
pub fn moe_combine(
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

    /// `w2 (silu(w1 x) * w3 x)` in f64 for one token.
    fn expert_f64(x: &[f64], w1: &[f64], w3: &[f64], w2: &[f64], inter: usize) -> Vec<f64> {
        let hidden = x.len();
        let matvec = |w: &[f64], v: &[f64], rows: usize| -> Vec<f64> {
            (0..rows)
                .map(|r| (0..v.len()).map(|c| w[r * v.len() + c] * v[c]).sum())
                .collect()
        };
        let silu = |z: f64| z / (1.0 + (-z).exp());
        let gate = matvec(w1, x, inter);
        let up = matvec(w3, x, inter);
        let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        matvec(w2, &act, hidden)
    }

    #[test]
    fn an_expert_is_swiglu() {
        let (hidden, inter, tokens) = (6usize, 10usize, 3usize);
        let mut rng = Rng(11);
        let mut draw = |n: usize| (0..n).map(|_| round32(rng.normal())).collect::<Vec<_>>();
        let (w1, w3, w2) = (
            draw(inter * hidden),
            draw(inter * hidden),
            draw(hidden * inter),
        );
        let x = draw(tokens * hidden);

        let mut expert = Expert::new(hidden as i32, inter as i32);
        expert.w1.weight = arr(&w1, &[inter, hidden]);
        expert.w3.weight = arr(&w3, &[inter, hidden]);
        expert.w2.weight = arr(&w2, &[hidden, inter]);
        let got = vec_of(&expert.forward(&arr(&x, &[tokens, hidden])));
        drain();

        let want: Vec<f64> = (0..tokens)
            .flat_map(|t| expert_f64(&x[t * hidden..(t + 1) * hidden], &w1, &w3, &w2, inter))
            .collect();
        assert_close("expert", &got, &want, 1e-5);
    }

    #[test]
    fn a_new_expert_is_zero() {
        let expert = Expert::new(4, 8);
        let out = expert.forward(&Array::from_f32_slice(&[1.0; 8], &[2, 4]));
        assert_eq!(out.shape(), &[2, 4]);
        assert!(vec_of(&out).iter().all(|&v| v == 0.0));
    }

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
            let got = moe_combine(
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
        let got = moe_combine(
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
