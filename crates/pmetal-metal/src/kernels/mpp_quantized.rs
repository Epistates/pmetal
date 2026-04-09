#![allow(unsafe_code)]

//! Metal 4 / MPP quantized GEMM dispatch.
//!
//! Provides Apple10/M5-only dispatch for the Metal 4 quantized kernels in
//! `metal4/mpp_quantized.metal`. The currently wired format is 4-bit affine
//! quantization with fp16 activations and fp16 output.

use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLComputeCommandEncoder};

use crate::{
    buffer::AsMetalBuffer,
    context::MetalContext,
    error::{MetalError, Result},
    kernels::mpp_dispatch::encode_mpp_kernel,
};

/// Configuration for MPP quantized GEMM.
#[derive(Debug, Clone)]
pub struct MppQuantizedGemmConfig {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Quantization group size.
    pub group_size: usize,
    /// Quantization bits.
    pub bits: u8,
}

impl MppQuantizedGemmConfig {
    /// Create a new configuration for `Y[M, N] = X[M, K] @ W_q[N, K]^T`.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            group_size: 64,
            bits: 4,
        }
    }
}

/// Metal-side parameter block (must match `QuantGemmParams` in Metal).
#[repr(C)]
struct QuantGemmParams {
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
    num_tiles_m: u32,
    num_tiles_n: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchGeometry {
    bm: usize,
    bn: usize,
    bk: usize,
    num_tiles_m: usize,
    num_tiles_n: usize,
    threads_per_threadgroup: usize,
}

fn validate_config(config: &MppQuantizedGemmConfig) -> Result<()> {
    if config.m == 0 || config.n == 0 || config.k == 0 {
        return Err(MetalError::InvalidConfig(
            "MPP quantized GEMM dimensions must be non-zero".to_string(),
        ));
    }

    if config.group_size == 0 {
        return Err(MetalError::InvalidConfig(
            "MPP quantized GEMM group_size must be non-zero".to_string(),
        ));
    }

    if config.k % config.group_size != 0 {
        return Err(MetalError::InvalidConfig(format!(
            "MPP quantized GEMM requires K ({}) divisible by group_size ({})",
            config.k, config.group_size
        )));
    }

    match config.bits {
        4 => {
            if config.k % 8 != 0 {
                return Err(MetalError::InvalidConfig(format!(
                    "MPP 4-bit quantized GEMM requires K ({}) divisible by 8",
                    config.k
                )));
            }
        }
        8 => {}
        other => {
            return Err(MetalError::InvalidConfig(format!(
                "MPP quantized GEMM only supports 4-bit or 8-bit weights, got {}",
                other
            )));
        }
    }

    Ok(())
}

fn dispatch_geometry(config: &MppQuantizedGemmConfig) -> DispatchGeometry {
    let bm = 64usize;
    let bn = 64usize;
    let bk = match config.bits {
        4 => 32usize,
        8 => 64usize,
        _ => unreachable!("validate_config rejects unsupported bit-widths"),
    };

    DispatchGeometry {
        bm,
        bn,
        bk,
        num_tiles_m: config.m.div_ceil(bm),
        num_tiles_n: config.n.div_ceil(bn),
        threads_per_threadgroup: 4 * 32,
    }
}

fn expected_weight_len(config: &MppQuantizedGemmConfig) -> Result<usize> {
    match config.bits {
        4 => config
            .n
            .checked_mul(config.k / 8)
            .ok_or_else(|| MetalError::InvalidConfig("MPP 4-bit weight size overflow".to_string())),
        8 => config
            .n
            .checked_mul(config.k)
            .ok_or_else(|| MetalError::InvalidConfig("MPP 8-bit weight size overflow".to_string())),
        _ => Err(MetalError::InvalidConfig(
            "Unsupported MPP quantized bit-width".to_string(),
        )),
    }
}

fn expected_scales_len(config: &MppQuantizedGemmConfig) -> Result<usize> {
    config
        .n
        .checked_mul(config.k / config.group_size)
        .ok_or_else(|| MetalError::InvalidConfig("MPP quantized scales size overflow".to_string()))
}

fn validate_buffer_lengths(
    config: &MppQuantizedGemmConfig,
    x_len: usize,
    weight_len: usize,
    scales_len: usize,
    biases_len: Option<usize>,
    y_len: usize,
) -> Result<()> {
    let expected_x_len = config
        .m
        .checked_mul(config.k)
        .ok_or_else(|| MetalError::InvalidConfig("MPP quantized X size overflow".to_string()))?;
    if x_len != expected_x_len {
        return Err(MetalError::DimensionMismatch {
            param: "x",
            expected: expected_x_len,
            actual: x_len,
        });
    }

    let expected_weight_len = expected_weight_len(config)?;
    if weight_len != expected_weight_len {
        return Err(MetalError::DimensionMismatch {
            param: "weights",
            expected: expected_weight_len,
            actual: weight_len,
        });
    }

    let expected_scales_len = expected_scales_len(config)?;
    if scales_len != expected_scales_len {
        return Err(MetalError::DimensionMismatch {
            param: "scales",
            expected: expected_scales_len,
            actual: scales_len,
        });
    }

    match (config.bits, biases_len) {
        (4, Some(actual)) if actual == expected_scales_len => {}
        (4, Some(actual)) => {
            return Err(MetalError::DimensionMismatch {
                param: "biases",
                expected: expected_scales_len,
                actual,
            });
        }
        (4, None) => {
            return Err(MetalError::InvalidConfig(
                "MPP 4-bit quantized GEMM requires bias/zero-point buffer".to_string(),
            ));
        }
        (8, Some(_)) => {
            return Err(MetalError::InvalidConfig(
                "MPP 8-bit quantized GEMM expects symmetric int8 weights and no bias buffer"
                    .to_string(),
            ));
        }
        (8, None) => {}
        _ => unreachable!("validate_config rejects unsupported bit-widths"),
    }

    let expected_y_len = config
        .m
        .checked_mul(config.n)
        .ok_or_else(|| MetalError::InvalidConfig("MPP quantized Y size overflow".to_string()))?;
    if y_len != expected_y_len {
        return Err(MetalError::DimensionMismatch {
            param: "output",
            expected: expected_y_len,
            actual: y_len,
        });
    }

    Ok(())
}

fn kernel_name(config: &MppQuantizedGemmConfig) -> Result<&'static str> {
    match config.bits {
        4 => Ok("mpp_qmm_4bit_f16"),
        8 => Ok("mpp_qmm_8bit_f16"),
        _ => Err(MetalError::InvalidConfig(
            "Unsupported MPP quantized bit-width".to_string(),
        )),
    }
}

/// MPP quantized GEMM dispatcher.
pub struct MppQuantizedGemm {
    ctx: Arc<MetalContext>,
    config: MppQuantizedGemmConfig,
}

impl MppQuantizedGemm {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppQuantizedGemmConfig) -> Self {
        Self { ctx, config }
    }

    /// Whether the current device can execute the Metal 4 quantized kernels.
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute synchronously.
    pub fn execute(
        &self,
        x: &dyn AsMetalBuffer,
        weights: &dyn AsMetalBuffer,
        scales: &dyn AsMetalBuffer,
        biases: Option<&dyn AsMetalBuffer>,
        output: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let command_buffer = self.execute_async(x, weights, scales, biases, output)?;
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute asynchronously and return the submitted command buffer.
    pub fn execute_async(
        &self,
        x: &dyn AsMetalBuffer,
        weights: &dyn AsMetalBuffer,
        scales: &dyn AsMetalBuffer,
        biases: Option<&dyn AsMetalBuffer>,
        output: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP quantized GEMM not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        validate_config(&self.config)?;
        validate_buffer_lengths(
            &self.config,
            x.len(),
            weights.len(),
            scales.len(),
            biases.map(AsMetalBuffer::len),
            output.len(),
        )?;

        let geometry = dispatch_geometry(&self.config);
        let kernel_name = kernel_name(&self.config)?;

        let params = QuantGemmParams {
            m: self.config.m as u32,
            n: self.config.n as u32,
            k: self.config.k as u32,
            group_size: self.config.group_size as u32,
            bits: self.config.bits as u32,
            num_tiles_m: geometry.num_tiles_m as u32,
            num_tiles_n: geometry.num_tiles_n as u32,
        };

        let grid = objc2_metal::MTLSize {
            width: geometry.num_tiles_n,
            height: geometry.num_tiles_m,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: geometry.threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        // Resolve the biases buffer reference before entering the closure so
        // that the fallible `?` stays outside the infallible bind_buffers closure.
        let bias_buf = match self.config.bits {
            4 => Some(
                biases
                    .ok_or_else(|| {
                        MetalError::InvalidConfig(
                            "MPP 4-bit quantized GEMM requires biases".to_string(),
                        )
                    })?
                    .as_metal_buffer(),
            ),
            _ => None,
        };

        let x_buf = x.as_metal_buffer();
        let w_buf = weights.as_metal_buffer();
        let s_buf = scales.as_metal_buffer();
        let out_buf = output.as_metal_buffer();
        let bits = self.config.bits;

        encode_mpp_kernel(&self.ctx, kernel_name, grid, tg_size, |encoder| unsafe {
            match bits {
                4 => {
                    encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(s_buf), 0, 2);
                    encoder.setBuffer_offset_atIndex(bias_buf, 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 4);
                    let params_ptr = NonNull::from(&params).cast();
                    encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);
                }
                8 => {
                    encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(s_buf), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 3);
                    let params_ptr = NonNull::from(&params).cast();
                    encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
                }
                _ => unreachable!("validate_config rejects unsupported bit-widths"),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpp_quantized_config_defaults_to_4bit_group64() {
        let config = MppQuantizedGemmConfig::new(8, 256, 128);
        assert_eq!(config.group_size, 64);
        assert_eq!(config.bits, 4);
    }

    #[test]
    fn test_validate_config_rejects_unsupported_bits() {
        let mut config = MppQuantizedGemmConfig::new(8, 256, 128);
        config.bits = 3;
        let error = validate_config(&config).unwrap_err().to_string();
        assert!(error.contains("4-bit or 8-bit"));
    }

    #[test]
    fn test_validate_config_requires_group_divisibility() {
        let mut config = MppQuantizedGemmConfig::new(8, 256, 130);
        config.group_size = 64;
        let error = validate_config(&config).unwrap_err().to_string();
        assert!(error.contains("divisible by group_size"));
    }

    #[test]
    fn test_expected_lengths_for_4bit_config() {
        let config = MppQuantizedGemmConfig::new(8, 256, 128);
        assert_eq!(expected_weight_len(&config).unwrap(), 256 * 16);
        assert_eq!(expected_scales_len(&config).unwrap(), 256 * 2);
    }

    #[test]
    fn test_validate_buffer_lengths_requires_biases_for_4bit() {
        let config = MppQuantizedGemmConfig::new(8, 256, 128);
        let error = validate_buffer_lengths(&config, 8 * 128, 256 * 16, 256 * 2, None, 8 * 256)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires bias"));
    }

    #[test]
    fn test_validate_buffer_lengths_accepts_symmetric_8bit() {
        let mut config = MppQuantizedGemmConfig::new(8, 256, 128);
        config.bits = 8;

        validate_buffer_lengths(&config, 8 * 128, 256 * 128, 256 * 2, None, 8 * 256).unwrap();
    }

    #[test]
    fn test_dispatch_geometry_uses_bit_specific_k_tiles() {
        let config4 = MppQuantizedGemmConfig::new(8, 256, 128);
        let mut config8 = config4.clone();
        config8.bits = 8;

        assert_eq!(dispatch_geometry(&config4).bk, 32);
        assert_eq!(dispatch_geometry(&config8).bk, 64);
    }
}
