//! RMS Layer Normalization.
//!
//! Wraps the pmetal-bridge RmsNorm implementation.

use pmetal_bridge::compat::{Array, Exception};

/// Apply RMS normalization to a tensor (functional version).
///
/// # Arguments
/// * `x` - Input tensor of shape [..., hidden_size]
/// * `weight` - Optional scale parameter of shape [hidden_size]
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// Normalized tensor of same shape as input.
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Array {
    pmetal_bridge::compat::fast::rms_norm_opt(x, weight, eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::{Dtype, ops, random};

    #[test]
    fn test_rms_norm_functional() {
        let x = random::normal(&[2, 4, 64], Dtype::Float32);
        let weight = ops::ones(&[64], Dtype::Float32);

        let output = rms_norm(&x, Some(&weight), 1e-6);
        assert_eq!(output.shape(), x.shape());
    }

    #[test]
    fn test_rms_norm_no_weight() {
        let x = random::normal(&[2, 4, 64], Dtype::Float32);
        let output = rms_norm(&x, None, 1e-6);
        assert_eq!(output.shape(), x.shape());
    }
}
