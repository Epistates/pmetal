//! Expert MLPs for mixture-of-experts architectures.
//!
//! Each architecture routes tokens itself (top-k selection lives in
//! `pmetal_models::moe_routing`) and runs its experts through stacked or
//! gathered matmuls that keep the routing weights in the graph. What is shared
//! is the expert itself: a bias-free SwiGLU MLP whose weights a checkpoint
//! fills.

use pmetal_bridge::compat::{Array, Dtype, ops};
use pmetal_bridge::impl_module_params;

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
    use pmetal_bridge::compat::random;

    #[test]
    fn test_expert_forward_shape() {
        let hidden = 16_i32;
        let intermediate = 32_i32;
        let t = 4_i32;

        let expert = Expert::new(hidden, intermediate);
        let input = random::uniform(&[t, hidden], Dtype::Float32);

        let output = expert.forward(&input);

        assert_eq!(
            output.shape(),
            &[t, hidden],
            "expert output shape mismatch: expected [{}, {}], got {:?}",
            t,
            hidden,
            output.shape()
        );
    }
}
