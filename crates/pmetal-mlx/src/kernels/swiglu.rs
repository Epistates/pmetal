//! SwiGLU and GEGLU activation implementations.
//!
//! These are gated activation functions used in modern LLMs:
//! - SwiGLU: swish(gate) * up = gate * sigmoid(gate) * up
//! - GEGLU: gelu(gate) * up

use pmetal_bridge::compat::{Array, Exception, nn};

type Result<T> = std::result::Result<T, Exception>;

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
pub fn swiglu(gate: &Array, up: &Array) -> Result<Array> {
    // swish = gate * sigmoid(gate)
    let swish = gate.multiply(&gate.sigmoid());
    Ok(swish.multiply(up))
}

/// Apply GEGLU activation: gelu(gate) * up.
///
/// Uses the **tanh** GELU approximation, which is what every GEGLU
/// architecture names (`gelu_pytorch_tanh` for Gemma, `gelu_new` for T5).
/// `Array::gelu()` is the sigmoid fast-approx and is ~1.9e-2 off — far outside
/// parity tolerance. An architecture whose config says plain `"gelu"` wants the
/// exact erf definition and should not route through here.
pub fn geglu(gate: &Array, up: &Array) -> Result<Array> {
    Ok(nn::gelu_tanh_approximate(gate).multiply(up))
}

/// Apply ReGLU activation: relu(gate) * up.
pub fn reglu(gate: &Array, up: &Array) -> Result<Array> {
    Ok(gate.relu().multiply(up))
}

/// Apply gated activation based on type.
pub fn gated_activation(gate: &Array, up: &Array, activation: GatedActivation) -> Result<Array> {
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
) -> Result<Array> {
    // gate = x @ gate_weight.T
    let gate = x.matmul(&gate_weight.t());
    // up = x @ up_weight.T
    let up = x.matmul(&up_weight.t());
    // hidden = activation(gate) * up
    let hidden = gated_activation(&gate, &up, activation)?;
    // output = hidden @ down_weight.T
    Ok(hidden.matmul(&down_weight.t()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::Dtype;

    #[test]
    fn test_swiglu() {
        let gate = Array::random_normal(&[2, 4, 64], Dtype::Float32.as_i32());
        let up = Array::random_normal(&[2, 4, 64], Dtype::Float32.as_i32());

        let output = swiglu(&gate, &up).unwrap();
        assert_eq!(output.shape(), gate.shape());
    }

    #[test]
    fn test_geglu() {
        let gate = Array::random_normal(&[2, 4, 64], Dtype::Float32.as_i32());
        let up = Array::random_normal(&[2, 4, 64], Dtype::Float32.as_i32());

        let output = geglu(&gate, &up).unwrap();
        assert_eq!(output.shape(), gate.shape());
    }

    #[test]
    fn test_reglu() {
        let gate = Array::random_normal(&[2, 4, 64], Dtype::Float32.as_i32());
        let up = Array::random_normal(&[2, 4, 64], Dtype::Float32.as_i32());

        let output = reglu(&gate, &up).unwrap();
        assert_eq!(output.shape(), gate.shape());
    }
}
