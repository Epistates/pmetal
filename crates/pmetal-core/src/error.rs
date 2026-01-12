//! Error types for PMetal.

use thiserror::Error;

/// Result type alias for PMetal operations.
pub type Result<T> = std::result::Result<T, PMetalError>;

/// Main error type for PMetal operations.
#[derive(Error, Debug)]
pub enum PMetalError {
    /// Model loading errors.
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Model architecture not supported.
    #[error("Unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Configuration errors.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Tensor shape mismatch.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },

    /// Data type mismatch.
    #[error("Dtype mismatch: expected {expected}, got {actual}")]
    DtypeMismatch {
        /// Expected dtype.
        expected: String,
        /// Actual dtype.
        actual: String,
    },

    /// Out of memory.
    #[error("Out of memory: required {required_gb:.2}GB, available {available_gb:.2}GB")]
    OutOfMemory {
        /// Required memory in GB.
        required_gb: f64,
        /// Available memory in GB.
        available_gb: f64,
    },

    /// Quantization errors.
    #[error("Quantization error: {0}")]
    Quantization(String),

    /// Training errors.
    #[error("Training error: {0}")]
    Training(String),

    /// I/O errors.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// HuggingFace Hub errors.
    #[error("Hub error: {0}")]
    Hub(String),

    /// Tokenizer errors.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// MLX backend errors.
    #[error("MLX error: {0}")]
    Mlx(String),

    /// Invalid argument.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Operation not implemented.
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}
