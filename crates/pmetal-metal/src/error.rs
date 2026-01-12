//! Error types for Metal operations.

use std::fmt;
use std::time::Duration;

/// Result type for Metal operations.
pub type Result<T> = std::result::Result<T, MetalError>;

/// Errors that can occur during Metal operations.
#[derive(Debug, Clone)]
pub enum MetalError {
    /// No Metal device found on the system.
    NoDevice,

    /// Failed to create command queue.
    CommandQueueCreation,

    /// Failed to create command buffer.
    CommandBufferCreation,

    /// Failed to create compute encoder.
    EncoderCreation,

    /// Failed to load Metal library.
    LibraryLoad(String),

    /// Failed to find function in library.
    FunctionNotFound(String),

    /// Failed to create compute pipeline.
    PipelineCreation(String),

    /// Failed to create buffer.
    BufferCreation {
        /// Size of buffer that failed to allocate.
        size: usize,
        /// Reason for failure.
        reason: String,
    },

    /// Buffer size mismatch.
    BufferSizeMismatch {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    /// Invalid configuration.
    InvalidConfig(String),

    /// Shader compilation error.
    ShaderCompilation(String),

    /// Command execution failed.
    ExecutionFailed(String),

    /// Dimension mismatch in operation.
    DimensionMismatch {
        /// Name of the parameter.
        param: &'static str,
        /// Expected value.
        expected: usize,
        /// Actual value.
        actual: usize,
    },

    /// Unsupported data type.
    UnsupportedDtype(String),

    /// GPU command buffer didn't complete within timeout (potential GPU hang).
    GpuTimeout {
        /// Operation ID of the timed-out command buffer.
        operation_id: u64,
        /// Timeout duration that was exceeded.
        timeout: Duration,
    },

    /// Internal error (should not happen in normal operation).
    Internal(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::NoDevice => {
                write!(f, "No Metal device found. Ensure running on Apple Silicon or macOS with Metal support.")
            }
            MetalError::CommandQueueCreation => {
                write!(f, "Failed to create Metal command queue")
            }
            MetalError::CommandBufferCreation => {
                write!(f, "Failed to create Metal command buffer")
            }
            MetalError::EncoderCreation => {
                write!(f, "Failed to create Metal compute command encoder")
            }
            MetalError::LibraryLoad(msg) => {
                write!(f, "Failed to load Metal library: {}", msg)
            }
            MetalError::FunctionNotFound(name) => {
                write!(f, "Metal function '{}' not found in library", name)
            }
            MetalError::PipelineCreation(msg) => {
                write!(f, "Failed to create compute pipeline: {}", msg)
            }
            MetalError::BufferCreation { size, reason } => {
                write!(f, "Failed to create buffer of size {}: {}", size, reason)
            }
            MetalError::BufferSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Buffer size mismatch: expected {} bytes, got {} bytes",
                    expected, actual
                )
            }
            MetalError::InvalidConfig(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            MetalError::ShaderCompilation(msg) => {
                write!(f, "Shader compilation error: {}", msg)
            }
            MetalError::ExecutionFailed(msg) => {
                write!(f, "Command execution failed: {}", msg)
            }
            MetalError::DimensionMismatch {
                param,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Dimension mismatch for '{}': expected {}, got {}",
                    param, expected, actual
                )
            }
            MetalError::UnsupportedDtype(dtype) => {
                write!(f, "Unsupported data type: {}", dtype)
            }
            MetalError::GpuTimeout {
                operation_id,
                timeout,
            } => {
                write!(
                    f,
                    "GPU timeout: operation {} did not complete within {:?} (potential GPU hang)",
                    operation_id, timeout
                )
            }
            MetalError::Internal(msg) => {
                write!(f, "Internal error: {}", msg)
            }
        }
    }
}

impl std::error::Error for MetalError {}

impl From<String> for MetalError {
    fn from(s: String) -> Self {
        MetalError::Internal(s)
    }
}

impl From<&str> for MetalError {
    fn from(s: &str) -> Self {
        MetalError::Internal(s.to_string())
    }
}
