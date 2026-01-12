//! Error types for the vocoder crate.

use mlx_rs::error::Exception;
use thiserror::Error;

/// Result type for vocoder operations.
pub type Result<T> = std::result::Result<T, VocoderError>;

/// Error type for vocoder operations.
#[derive(Error, Debug)]
pub enum VocoderError {
    /// MLX operation error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Weight loading error.
    #[error("Weight loading error: {0}")]
    WeightLoad(String),

    /// Shape mismatch error.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<i32>,
        /// Actual shape.
        actual: Vec<i32>,
    },

    /// Audio processing error.
    #[error("Audio error: {0}")]
    Audio(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Hub error.
    #[error("Hub error: {0}")]
    Hub(String),
}
