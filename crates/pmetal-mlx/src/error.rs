//! Error types for pmetal-mlx operations.

use thiserror::Error;

/// Errors that can occur in pmetal-mlx operations.
#[derive(Debug, Error)]
pub enum MlxError {
    /// MLX exception.
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// Metal kernel error.
    #[error("Metal error: {0}")]
    Metal(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Shape mismatch.
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        actual: String,
    },

    /// Unsupported operation.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}
