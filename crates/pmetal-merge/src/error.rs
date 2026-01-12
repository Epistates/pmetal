//! Error types for model merging operations.

use thiserror::Error;

/// Errors that can occur during model merging.
#[derive(Debug, Error)]
pub enum MergeError {
    /// Model loading error.
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Tensor not found in model.
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    /// Shape mismatch between tensors.
    #[error("Shape mismatch for tensor '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Expected shape.
        expected: Vec<i32>,
        /// Actual shape.
        actual: Vec<i32>,
    },

    /// Architecture mismatch between models.
    #[error("Architecture mismatch: {0}")]
    ArchitectureMismatch(String),

    /// Invalid merge configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// MLX operation error.
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// Safetensors error.
    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// YAML parsing error.
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// HuggingFace Hub error.
    #[error("Hub error: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),

    /// Method requires base model but none provided.
    #[error("Base model required for {method} merge but not provided")]
    BaseModelRequired {
        /// The merge method that requires a base model.
        method: String,
    },

    /// Not enough models for merge operation.
    #[error("Expected at least {expected} models, got {actual}")]
    NotEnoughModels {
        /// Expected number of models.
        expected: usize,
        /// Actual number of models.
        actual: usize,
    },
}

/// Result type for merge operations.
pub type Result<T> = std::result::Result<T, MergeError>;
