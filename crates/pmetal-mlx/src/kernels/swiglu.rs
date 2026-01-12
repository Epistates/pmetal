//! SwiGLU and GEGLU activation implementations.
//!
//! These are gated activation functions used in modern LLMs:
//! - SwiGLU: swish(gate) * up = gate * sigmoid(gate) * up
//! - GEGLU: gelu(gate) * up

use mlx_rs::Array;

/// Gated activation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GatedActivation {
    /// SwiGLU: swish(x) * y
    #[default]
    SwiGLU,
    /// GEGLU: gelu(x) * y
    GEGLU,
    /// ReGLU: relu(x) * y
    ReGLU,
}

/// Apply SwiGLU activation: swish(gate) * up.
///
/// swish(x) = x * sigmoid(x)
pub fn swiglu(gate: &Array, up: &Array) -> mlx_rs::error::Result<Array> {
    // swish = gate * sigmoid(gate)
    let sigmoid_gate = mlx_rs::ops::sigmoid(gate)?;
    let swish = gate.multiply(&sigmoid_gate)?;
    swish.multiply(up)
}

/// Apply GEGLU activation: gelu(gate) * up.
pub fn geglu(gate: &Array, up: &Array) -> mlx_rs::error::Result<Array> {
    let gelu_gate = mlx_rs::nn::gelu(gate)?;
    gelu_gate.multiply(up)
}

/// Apply ReGLU activation: relu(gate) * up.
pub fn reglu(gate: &Array, up: &Array) -> mlx_rs::error::Result<Array> {
    let relu_gate = mlx_rs::nn::relu(gate)?;
    relu_gate.multiply(up)
}

/// Apply gated activation based on type.
pub fn gated_activation(
    gate: &Array,
    up: &Array,
    activation: GatedActivation,
) -> mlx_rs::error::Result<Array> {
    match activation {
        GatedActivation::SwiGLU => swiglu(gate, up),
        GatedActivation::GEGLU => geglu(gate, up),
        GatedActivation::ReGLU => reglu(gate, up),
    }
}

/// Fused gated MLP forward pass (functional version).
///
/// Implements: down_proj(activation(gate_proj(x)) * up_proj(x))
pub fn gated_mlp_forward(
    x: &Array,
    gate_weight: &Array,
    up_weight: &Array,
    down_weight: &Array,
    activation: GatedActivation,
) -> mlx_rs::error::Result<Array> {
    // gate = x @ gate_weight.T
    let gate = x.matmul(&gate_weight.t())?;
    // up = x @ up_weight.T
    let up = x.matmul(&up_weight.t())?;
    // hidden = activation(gate) * up
    let hidden = gated_activation(&gate, &up, activation)?;
    // output = hidden @ down_weight.T
    hidden.matmul(&down_weight.t())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu() {
        let gate = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let up = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();

        let output = swiglu(&gate, &up).unwrap();
        assert_eq!(output.shape(), gate.shape());
    }

    #[test]
    fn test_geglu() {
        let gate = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let up = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();

        let output = geglu(&gate, &up).unwrap();
        assert_eq!(output.shape(), gate.shape());
    }

    #[test]
    fn test_reglu() {
        let gate = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let up = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();

        let output = reglu(&gate, &up).unwrap();
        assert_eq!(output.shape(), gate.shape());
    }
}
