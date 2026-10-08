//! Error and result types shared by the trainers.

use pmetal_bridge::compat::Exception;

/// Error type for SFT training.
#[derive(Debug, thiserror::Error)]
pub enum SftError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// LoRA error.
    #[error("LoRA error: {0}")]
    Lora(#[from] pmetal_lora::LoraError),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Training was cancelled by a callback.
    #[error("Training cancelled")]
    Cancelled,
}

/// Result type for SFT operations.
pub type Result<T> = std::result::Result<T, SftError>;
