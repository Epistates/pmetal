//! RMS Layer Normalization.
//!
//! Re-exports the optimized mlx-rs RmsNorm implementation and provides
//! additional utilities for RMS normalization.

// Re-export the mlx-rs implementation
pub use mlx_rs::nn::{RmsNorm, RmsNormBuilder};

/// Apply RMS normalization to a tensor (functional version).
///
/// # Arguments
/// * `x` - Input tensor of shape [..., hidden_size]
/// * `weight` - Scale parameter of shape [hidden_size]
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// Normalized tensor of same shape as input.
pub fn rms_norm(
    x: &mlx_rs::Array,
    weight: &mlx_rs::Array,
    eps: f32,
) -> mlx_rs::error::Result<mlx_rs::Array> {
    mlx_rs::fast::rms_norm(x, weight, eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::builder::Builder;

    #[test]
    fn test_rms_norm_functional() {
        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let weight = mlx_rs::ops::ones::<f32>(&[64]).unwrap();

        let output = rms_norm(&x, &weight, 1e-6).unwrap();
        assert_eq!(output.shape(), x.shape());
    }

    #[test]
    fn test_rms_norm_module() {
        use mlx_rs::module::Module;

        let mut norm = RmsNormBuilder::new(64).build().unwrap();

        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.shape(), x.shape());
    }
}
