//! Common type definitions.

use serde::{Deserialize, Serialize};

/// Data type for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    /// 32-bit floating point.
    Float32,
    /// 16-bit floating point.
    Float16,
    /// Brain floating point (16-bit).
    #[default]
    BFloat16,
    /// 8-bit floating point (E4M3).
    Float8E4M3,
    /// 8-bit floating point (E5M2).
    Float8E5M2,
    /// 32-bit integer.
    Int32,
    /// 64-bit integer.
    Int64,
    /// 8-bit unsigned integer.
    UInt8,
    /// Boolean.
    Bool,
}

impl Dtype {
    /// Size of the dtype in bytes.
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
            Self::Float8E4M3 | Self::Float8E5M2 | Self::UInt8 | Self::Bool => 1,
            Self::Int64 => 8,
        }
    }
}

/// Compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    /// CPU computation.
    Cpu,
    /// GPU computation (Metal on macOS).
    #[default]
    Gpu,
}

/// Quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// No quantization (full precision).
    #[default]
    None,
    /// 4-bit Normal Float quantization.
    NF4,
    /// 4-bit Floating Point quantization.
    FP4,
    /// 8-bit integer quantization.
    Int8,
    /// 8-bit floating point quantization.
    FP8,
}

/// Memory statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total memory in bytes.
    pub total_bytes: u64,
    /// Used memory in bytes.
    pub used_bytes: u64,
    /// Peak memory usage in bytes.
    pub peak_bytes: u64,
}

impl MemoryStats {
    /// Used memory in gigabytes.
    #[must_use]
    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Total memory in gigabytes.
    #[must_use]
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Peak memory in gigabytes.
    #[must_use]
    pub fn peak_gb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Available memory in gigabytes.
    #[must_use]
    pub fn available_gb(&self) -> f64 {
        (self.total_bytes - self.used_bytes) as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Model output from forward pass.
#[derive(Debug, Clone)]
pub struct ModelOutput<T> {
    /// Logits tensor.
    pub logits: T,
    /// Hidden states (optional).
    pub hidden_states: Option<Vec<T>>,
    /// Attention weights (optional).
    pub attentions: Option<Vec<T>>,
    /// Past key-value cache (optional).
    pub past_key_values: Option<Vec<(T, T)>>,
}

/// Evaluation metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// Loss value.
    pub loss: f64,
    /// Perplexity.
    pub perplexity: f64,
    /// Accuracy (if applicable).
    pub accuracy: Option<f64>,
    /// Custom metrics.
    pub custom: std::collections::HashMap<String, f64>,
}
