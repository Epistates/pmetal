//! Error types for knowledge distillation operations.

use thiserror::Error;

/// Errors that can occur during knowledge distillation.
#[derive(Debug, Error)]
pub enum DistillError {
    /// Teacher model loading error.
    #[error("Failed to load teacher model: {0}")]
    TeacherLoad(String),

    /// Student model loading error.
    #[error("Failed to load student model: {0}")]
    StudentLoad(String),

    /// Logit cache error.
    #[error("Logit cache error: {0}")]
    LogitCache(String),

    /// Shape mismatch between teacher and student outputs.
    #[error("Shape mismatch: teacher {teacher:?}, student {student:?}")]
    ShapeMismatch {
        /// Teacher output shape.
        teacher: Vec<i32>,
        /// Student output shape.
        student: Vec<i32>,
    },

    /// Vocabulary size mismatch.
    #[error("Vocabulary size mismatch: teacher {teacher}, student {student}")]
    VocabMismatch {
        /// Teacher vocabulary size.
        teacher: usize,
        /// Student vocabulary size.
        student: usize,
    },

    /// Invalid temperature value.
    #[error("Invalid temperature: {0} (must be > 0)")]
    InvalidTemperature(f32),

    /// Invalid alpha value.
    #[error("Invalid alpha: {0} (must be in [0, 1])")]
    InvalidAlpha(f32),

    /// Invalid configuration.
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

    /// Bincode serialization error.
    #[error("Bincode error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),

    /// Layer mapping error.
    #[error("Layer mapping error: {0}")]
    LayerMapping(String),

    /// Metal GPU error.
    #[error("Metal GPU error: {0}")]
    Metal(String),

    /// Generic error for miscellaneous failures.
    #[error("{0}")]
    Other(String),
}

/// Result type for distillation operations.
pub type Result<T> = std::result::Result<T, DistillError>;
