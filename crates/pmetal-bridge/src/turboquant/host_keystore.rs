//! Host-side TurboQuant K/V stores backed by [`PackedBits`] for the
//! Uniform and Mixed-precision configs.
//!
//! These are the authoritative stores read by the C++ score kernels
//! (`mlx_inline_turboquant_attention_*`). The `gpu` / `gpu_mixed` Option
//! fields hold parallel GPU-resident copies for the cold-dequantize path
//! and any future fused score kernel.

use super::bits::PackedBits;
use super::config::TurboQuantTensorConfig;
use super::encode::{EncodedKeyRows, EncodedValueRows};
use super::gpu_keystore::{GpuKeyStore, GpuMixedKeyStore, GpuMixedValueStore, GpuValueStore};

/// Quantised key store for one attention layer.
#[derive(Debug, Clone)]
pub struct QuantizedKeyStore {
    // GPU-native store (Uniform path only).  When Some, dequantize uses GPU ops.
    pub(super) gpu: Option<GpuKeyStore>,

    // GPU-native store for the Mixed path. Populated alongside the CPU
    // `PackedBits` fields below when a Mixed config is active and the GPU
    // encode succeeded. `dequantize_keys` reads this directly so the cold
    // dequantize stays GPU-resident; a fused mixed-score kernel that
    // consumes it without going through dequantize is still future work.
    pub(super) gpu_mixed: Option<GpuMixedKeyStore>,

    // CPU fallback: regular (non-outlier) sub-vector data.
    pub regular_indices: PackedBits,
    pub regular_qjl_signs: PackedBits,
    pub regular_norms: Vec<f32>,
    pub regular_residual_norms: Vec<f32>,

    // Outlier sub-vector data (None when config is Uniform).
    pub outlier_mask: Option<PackedBits>,
    pub outlier_indices: Option<PackedBits>,
    pub outlier_qjl_signs: Option<PackedBits>,
    pub outlier_norms: Option<Vec<f32>>,
    pub outlier_residual_norms: Option<Vec<f32>>,
}

impl QuantizedKeyStore {
    pub(super) fn new(config: TurboQuantTensorConfig) -> Self {
        let regular_bits = match config {
            TurboQuantTensorConfig::Uniform { bits } => bits.saturating_sub(1),
            TurboQuantTensorConfig::Mixed { regular_bits, .. } => regular_bits.saturating_sub(1),
        };
        let outlier_bits: Option<u8> = match config {
            TurboQuantTensorConfig::Uniform { .. } => None,
            TurboQuantTensorConfig::Mixed { outlier_bits, .. } => {
                Some(outlier_bits.saturating_sub(1))
            }
        };

        Self {
            gpu: None,
            gpu_mixed: None,
            regular_indices: PackedBits::new(regular_bits),
            regular_qjl_signs: PackedBits::new(1),
            regular_norms: Vec::new(),
            regular_residual_norms: Vec::new(),
            outlier_mask: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_indices: outlier_bits.map(PackedBits::new),
            outlier_qjl_signs: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_norms: outlier_bits.map(|_| Vec::new()),
            outlier_residual_norms: outlier_bits.map(|_| Vec::new()),
        }
    }

    pub(super) fn extend(
        &mut self,
        encoded: &EncodedKeyRows,
        outlier_encoded: Option<&EncodedKeyRows>,
        outlier_mask: Option<&Vec<u16>>,
    ) {
        self.regular_indices.extend_from_slice(&encoded.mse_indices);
        self.regular_qjl_signs.extend_from_slice(&encoded.qjl_signs);
        self.regular_norms.extend_from_slice(&encoded.norms);
        self.regular_residual_norms
            .extend_from_slice(&encoded.residual_norms);

        if let Some(mask) = outlier_mask {
            self.outlier_mask
                .as_mut()
                .expect("TurboQuant key outlier mask missing")
                .extend_from_slice(mask);
        }
        if let Some(outlier) = outlier_encoded {
            self.outlier_indices
                .as_mut()
                .expect("TurboQuant key outlier indices missing")
                .extend_from_slice(&outlier.mse_indices);
            self.outlier_qjl_signs
                .as_mut()
                .expect("TurboQuant key outlier QJL signs missing")
                .extend_from_slice(&outlier.qjl_signs);
            self.outlier_norms
                .as_mut()
                .expect("TurboQuant key outlier norms missing")
                .extend_from_slice(&outlier.norms);
            self.outlier_residual_norms
                .as_mut()
                .expect("TurboQuant key outlier residual_norms missing")
                .extend_from_slice(&outlier.residual_norms);
        }
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.regular_indices.byte_len()
            + self.regular_qjl_signs.byte_len()
            + self.regular_norms.len() * 4
            + self.regular_residual_norms.len() * 4
            + self.outlier_mask.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_indices.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_qjl_signs.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_norms.as_ref().map_or(0, |v| v.len() * 4)
            + self
                .outlier_residual_norms
                .as_ref()
                .map_or(0, |v| v.len() * 4)
    }
}

/// Quantised value store for one attention layer.
#[derive(Debug, Clone)]
pub struct QuantizedValueStore {
    // GPU-native store (Uniform path only).
    pub(super) gpu: Option<GpuValueStore>,

    // GPU-native store for the Mixed path. See `QuantizedKeyStore.gpu_mixed`.
    pub(super) gpu_mixed: Option<GpuMixedValueStore>,

    pub regular_indices: PackedBits,
    pub regular_norms: Vec<f32>,

    pub outlier_mask: Option<PackedBits>,
    pub outlier_indices: Option<PackedBits>,
    pub outlier_norms: Option<Vec<f32>>,
}

impl QuantizedValueStore {
    pub(super) fn new(config: TurboQuantTensorConfig) -> Self {
        let regular_bits = match config {
            TurboQuantTensorConfig::Uniform { bits } => bits,
            TurboQuantTensorConfig::Mixed { regular_bits, .. } => regular_bits,
        };
        let outlier_bits: Option<u8> = match config {
            TurboQuantTensorConfig::Uniform { .. } => None,
            TurboQuantTensorConfig::Mixed { outlier_bits, .. } => Some(outlier_bits),
        };

        Self {
            gpu: None,
            gpu_mixed: None,
            regular_indices: PackedBits::new(regular_bits),
            regular_norms: Vec::new(),
            outlier_mask: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_indices: outlier_bits.map(PackedBits::new),
            outlier_norms: outlier_bits.map(|_| Vec::new()),
        }
    }

    pub(super) fn extend(
        &mut self,
        encoded: &EncodedValueRows,
        outlier_encoded: Option<&EncodedValueRows>,
        outlier_mask: Option<&Vec<u16>>,
    ) {
        self.regular_indices.extend_from_slice(&encoded.indices);
        self.regular_norms.extend_from_slice(&encoded.norms);

        if let Some(mask) = outlier_mask {
            self.outlier_mask
                .as_mut()
                .expect("TurboQuant value outlier mask missing")
                .extend_from_slice(mask);
        }
        if let Some(outlier) = outlier_encoded {
            self.outlier_indices
                .as_mut()
                .expect("TurboQuant value outlier indices missing")
                .extend_from_slice(&outlier.indices);
            self.outlier_norms
                .as_mut()
                .expect("TurboQuant value outlier norms missing")
                .extend_from_slice(&outlier.norms);
        }
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.regular_indices.byte_len()
            + self.regular_norms.len() * 4
            + self.outlier_mask.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_indices.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_norms.as_ref().map_or(0, |v| v.len() * 4)
    }
}

