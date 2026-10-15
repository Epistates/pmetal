//! Expert MLPs for mixture-of-experts architectures.
//!
//! Each architecture routes tokens itself (top-k selection lives in
//! `pmetal_models::moe_routing`) and runs its experts through stacked or
//! gathered matmuls that keep the routing weights in the graph. What is shared
//! is the expert itself: a bias-free SwiGLU MLP whose weights a checkpoint
//! fills.

use pmetal_runtime::compat::{Array, Dtype, ops};
use pmetal_runtime::impl_module_params;

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
}
