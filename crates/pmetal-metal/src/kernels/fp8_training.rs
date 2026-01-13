//! FP8 Training Kernels for Apple Silicon.
//!
//! This module provides GPU-accelerated FP8 operations for memory-efficient training:
//!
//! - Block-wise activation quantization (E4M3 format)
//! - Block-wise gradient quantization (E5M2 format)
//! - Weight dequantization (block and row-wise)
//! - Block FP8 GEMM with scaling
//! - Dynamic scale tracking for training
//!
//! FP8 enables ~2x memory reduction compared to BF16 with minimal accuracy loss.
//!
//! # Metal Shaders
//!
//! The Metal shaders are located in `metal/fp8_training.metal` and provide:
//! - `fp8_act_quant_block`: Block-wise activation quantization
//! - `fp8_grad_quant_block`: Block-wise gradient quantization
//! - `fp8_weight_dequant_block`: Block-wise weight dequantization
//! - `fp8_weight_dequant_row`: Row-wise weight dequantization
//! - `fp8_block_gemm`: Block FP8 GEMM with scaling
//! - `fp8_update_scale`: Dynamic scale update

use half::bf16;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use std::ptr::NonNull;

use crate::{
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
};

/// FP8 format type for quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// E4M3: 4-bit exponent, 3-bit mantissa. Range +-448. Best for weights.
    E4M3,
    /// E5M2: 5-bit exponent, 2-bit mantissa. Range +-57344. Best for activations/gradients.
    E5M2,
}

impl Fp8Format {
    /// Maximum representable value for this format.
    pub fn max_value(&self) -> f32 {
        match self {
            Self::E4M3 => 448.0,
            Self::E5M2 => 57344.0,
        }
    }

    /// Minimum representable non-zero value.
    pub fn min_value(&self) -> f32 {
        match self {
            Self::E4M3 => 1.0 / 448.0,   // 2^-9
            Self::E5M2 => 1.0 / 16384.0, // 2^-14
        }
    }

    /// Epsilon for numerical stability.
    pub fn epsilon(&self) -> f32 {
        match self {
            Self::E4M3 => 0.0625,       // 2^-4
            Self::E5M2 => 0.0009765625, // 2^-10
        }
    }
}

/// Configuration for FP8 training operations.
#[derive(Debug, Clone)]
pub struct Fp8TrainingConfig {
    /// Block size for quantization (must be power of 2).
    pub block_size: usize,
    /// Format for weight quantization.
    pub weight_format: Fp8Format,
    /// Format for activation quantization.
    pub activation_format: Fp8Format,
    /// Format for gradient quantization.
    pub gradient_format: Fp8Format,
    /// Window size for dynamic scale history.
    pub scale_window_size: usize,
}

impl Default for Fp8TrainingConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            weight_format: Fp8Format::E4M3,
            activation_format: Fp8Format::E4M3,
            gradient_format: Fp8Format::E5M2,
            scale_window_size: 1024,
        }
    }
}

/// Output of activation quantization.
#[derive(Debug)]
pub struct Fp8QuantOutput {
    /// Quantized data (stored as u8).
    pub data: MetalBuffer<u8>,
    /// Per-block scale factors.
    pub scales: MetalBuffer<f32>,
    /// Shape of the quantized tensor.
    pub shape: Vec<usize>,
    /// Block size used.
    pub block_size: usize,
}

/// Output of FP8 GEMM operation.
#[derive(Debug)]
pub struct Fp8GemmOutput {
    /// Result matrix.
    pub output: MetalBuffer<bf16>,
    /// Output shape [M, N].
    pub shape: Vec<usize>,
}

/// Dynamic scale tracker for FP8 training.
///
/// Tracks amax history over a sliding window to compute optimal scales.
#[derive(Debug)]
pub struct Fp8DynamicScale {
    /// History buffer for amax values.
    pub amax_history: Vec<f32>,
    /// Current scale value.
    pub scale: f32,
    /// Window size.
    pub window_size: usize,
    /// Current index in circular buffer.
    pub current_idx: usize,
    /// FP8 format for this scale.
    pub format: Fp8Format,
}

impl Fp8DynamicScale {
    /// Create a new dynamic scale tracker.
    pub fn new(window_size: usize, format: Fp8Format) -> Self {
        Self {
            amax_history: vec![0.0; window_size],
            scale: 1.0,
            window_size,
            current_idx: 0,
            format,
        }
    }

    /// Update scale with new amax value.
    pub fn update(&mut self, new_amax: f32) {
        // Update history (circular buffer)
        self.amax_history[self.current_idx] = new_amax;
        self.current_idx = (self.current_idx + 1) % self.window_size;

        // Find max in history
        let max_amax = self.amax_history.iter().cloned().fold(0.0f32, f32::max);

        // Compute new scale
        self.scale = self.format.max_value() / max_amax.max(1e-12);
    }

    /// Get current scale value.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get inverse scale for efficient computation.
    pub fn scale_inv(&self) -> f32 {
        1.0 / self.scale
    }
}

/// FP8 Training kernel manager.
#[derive(Debug)]
pub struct Fp8TrainingKernel {
    ctx: std::sync::Arc<MetalContext>,
    config: Fp8TrainingConfig,
}

impl Fp8TrainingKernel {
    /// Create a new FP8 training kernel.
    pub fn new(ctx: std::sync::Arc<MetalContext>, config: Fp8TrainingConfig) -> Self {
        Self { ctx, config }
    }

    /// Create with default configuration.
    pub fn with_defaults(ctx: std::sync::Arc<MetalContext>) -> Self {
        Self::new(ctx, Fp8TrainingConfig::default())
    }

    /// Quantize a tensor to FP8 with block-wise scaling (GPU).
    ///
    /// # Arguments
    /// * `input` - Input buffer (BF16)
    /// * `m` - Number of rows
    /// * `k` - Number of columns
    ///
    /// # Returns
    /// Fp8QuantOutput struct containing quantized data and scales.
    pub fn quantize_gpu(
        &self,
        input: &MetalBuffer<bf16>,
        m: usize,
        k: usize,
    ) -> Result<Fp8QuantOutput> {
        let block_size = self.config.block_size;
        let n_blocks = (k + block_size - 1) / block_size;

        let output_data = MetalBuffer::<u8>::new(&self.ctx, m * k, BufferUsage::Shared)?;
        let output_scales = MetalBuffer::<f32>::new(&self.ctx, m * n_blocks, BufferUsage::Shared)?;

        let pipeline = self.ctx.pipeline_cache_mut().get_or_create_pipeline(
            self.ctx.device(),
            "fp8_act_quant_block",
            None,
        )?;

        let command_buffer = self
            .ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_data.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output_scales.metal_buffer()), 0, 2);

            let m_u32 = m as u32;
            let k_u32 = k as u32;
            let bs_u32 = block_size as u32;

            let params = [m_u32, k_u32, bs_u32];
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);
        }

        let grid_size = MTLSize {
            width: n_blocks,
            height: m,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: block_size.min(256),
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(Fp8QuantOutput {
            data: output_data,
            scales: output_scales,
            shape: vec![m, k],
            block_size,
        })
    }

    /// Dequantize FP8 tensor back to BF16 (GPU).
    pub fn dequantize_gpu(&self, input: &Fp8QuantOutput) -> Result<MetalBuffer<bf16>> {
        let m = input.shape[0];
        let k = input.shape[1];
        let block_size = input.block_size;

        let output = MetalBuffer::<bf16>::new(&self.ctx, m * k, BufferUsage::Shared)?;

        let pipeline = self.ctx.pipeline_cache_mut().get_or_create_pipeline(
            self.ctx.device(),
            "fp8_weight_dequant_block",
            None,
        )?;

        let command_buffer = self
            .ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.data.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.scales.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 2);

            let m_u32 = m as u32;
            let k_u32 = k as u32;
            let bs_u32 = block_size as u32;

            let params = [m_u32, k_u32, bs_u32];
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);
        }

        let grid_size = MTLSize {
            width: k,
            height: m,
            depth: 1,
        };

        // Simple 1D dispatch
        let threadgroup_size = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        // Use dispatchThreads if available (Metal 2) or standard grid
        // Here we use standard grid for compatibility, though inefficient for this kernel
        // Optimized would be 2D blocking
        let tg_width = 32;
        let tg_height = 8;
        let grid_width = (k + tg_width - 1) / tg_width;
        let grid_height = (m + tg_height - 1) / tg_height;

        let grid_size_dispatch = MTLSize {
            width: grid_width,
            height: grid_height,
            depth: 1,
        };
        let tg_size_dispatch = MTLSize {
            width: tg_width,
            height: tg_height,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size_dispatch, tg_size_dispatch);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(output)
    }

    /// Perform Block FP8 GEMM (GPU).
    /// C = A @ B
    pub fn gemm_gpu(&self, a: &Fp8QuantOutput, b: &Fp8QuantOutput) -> Result<MetalBuffer<bf16>> {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[0]; // Assuming B is [N, K] layout for weights?
                            // Wait, standard matmul is [M, K] @ [K, N].
                            // If B is weights, it's often stored as [N, K] (transposed) or [K, N].
                            // Our kernel 'fp8_block_gemm' comment says:
                            // A: [M, K], B: [N, K] (Quantized weights (N, K))
                            // So B is transposed.

        if a.shape[1] != b.shape[1] {
            return Err(MetalError::DimensionMismatch {
                param: "K",
                expected: a.shape[1],
                actual: b.shape[1],
            });
        }

        let output = MetalBuffer::<bf16>::new(&self.ctx, m * n, BufferUsage::Shared)?;

        let pipeline = self.ctx.pipeline_cache_mut().get_or_create_pipeline(
            self.ctx.device(),
            "fp8_block_gemm",
            None,
        )?;

        let command_buffer = self
            .ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.data.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.data.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(a.scales.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(b.scales.metal_buffer()), 0, 4);

            // Params: M, N, K, group_n, group_k
            // group_k is block_size (usually 128)
            // group_n is block_size (usually 128)
            let m_u32 = m as u32;
            let n_u32 = n as u32;
            let k_u32 = k as u32;
            let gn_u32 = self.config.block_size as u32;
            let gk_u32 = self.config.block_size as u32;

            let params = [m_u32, n_u32, k_u32, gn_u32, gk_u32];
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);
        }

        // Grid setup from kernel:
        // Thread block computes BLOCK_M x BLOCK_N tile of C (64x64)
        let block_m = 64;
        let block_n = 64;

        let grid_width = (n + block_n - 1) / block_n;
        let grid_height = (m + block_m - 1) / block_m;

        let grid_size = MTLSize {
            width: grid_width,
            height: grid_height,
            depth: 1,
        };

        // Kernel uses 256 threads (16x16? No, linear index)
        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(output)
    }

    /// Allocate GPU buffers for FP8 quantization.
    pub fn allocate_quant_buffers(
        &self,
        m: usize,
        k: usize,
    ) -> Result<(MetalBuffer<u8>, MetalBuffer<f32>)> {
        let n_blocks = (k + self.config.block_size - 1) / self.config.block_size;

        let data_buf = MetalBuffer::<u8>::new(&self.ctx, m * k, BufferUsage::Shared)?;
        let scales_buf = MetalBuffer::<f32>::new(&self.ctx, m * n_blocks, BufferUsage::Shared)?;

        Ok((data_buf, scales_buf))
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Get the configuration.
    pub fn config(&self) -> &Fp8TrainingConfig {
        &self.config
    }

    /// Calculate memory savings from FP8 quantization.
    pub fn memory_savings(original_elements: usize, original_dtype_bits: usize) -> (usize, f32) {
        // FP8 = 8 bits + ~4 bits overhead for scales (amortized)
        let fp8_bits = 8 + 4;
        let original_bytes = original_elements * original_dtype_bits / 8;
        let fp8_bytes = original_elements * fp8_bits / 8;
        let savings = 1.0 - (fp8_bytes as f32 / original_bytes as f32);
        (fp8_bytes, savings)
    }
}
