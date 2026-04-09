#![allow(unsafe_code)]

//! Metal kernels for TurboQuant transform stages.
//!
//! The TurboQuant KV cache repeatedly applies fixed dense transforms to batches
//! of row vectors:
//! - random orthogonal rotations
//! - inverse rotations
//! - QJL projections
//! - transpose projections during decode
//!
//! This module provides a reusable batched row-transform wrapper for those
//! stages so higher-level code can keep the quantization logic in Rust while
//! moving the dense O(rows * dim^2) work onto Metal.

use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::MetalContext;
use crate::error::{MetalError, Result};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct TurboQuantTransformParams {
    dim: u32,
}

/// Fixed dense transform applied to a batch of row vectors.
///
/// The stored matrix is row-major `[dim, dim]`. Applying it to a batch of
/// inputs `[rows, dim]` produces:
///
/// `output[row, out_dim] = dot(matrix[out_dim, :], input[row, :])`
pub struct TurboQuantTransform {
    ctx: Arc<MetalContext>,
    dim: usize,
    matrix: MetalBuffer<f32>,
}

impl Clone for TurboQuantTransform {
    fn clone(&self) -> Self {
        Self {
            ctx: self.ctx.clone(),
            dim: self.dim,
            matrix: self.matrix.clone(),
        }
    }
}

impl std::fmt::Debug for TurboQuantTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurboQuantTransform")
            .field("dim", &self.dim)
            .finish()
    }
}

impl TurboQuantTransform {
    /// Create a new transform using the global Metal context.
    pub fn new(matrix: &[f32], dim: usize) -> Result<Self> {
        let ctx = MetalContext::global()?;
        Self::with_context(ctx, matrix, dim)
    }

    /// Create a new transform using a caller-provided Metal context.
    pub fn with_context(ctx: Arc<MetalContext>, matrix: &[f32], dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(MetalError::InvalidConfig(
                "TurboQuant transform dimension must be non-zero".to_string(),
            ));
        }
        let expected = dim * dim;
        if matrix.len() != expected {
            return Err(MetalError::BufferSizeMismatch {
                expected,
                actual: matrix.len(),
            });
        }

        let matrix = MetalBuffer::from_slice(&ctx, matrix, BufferUsage::GpuReadOnly)?;
        Ok(Self { ctx, dim, matrix })
    }

    /// Matrix dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Apply the transform to `rows` row vectors.
    ///
    /// `input` must be laid out as `[rows, dim]` in row-major order.
    pub fn apply_rows(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        if input.len() % self.dim != 0 {
            return Err(MetalError::DimensionMismatch {
                param: "input_len % dim",
                expected: 0,
                actual: input.len() % self.dim,
            });
        }

        let num_rows = input.len() / self.dim;
        let scratch_bytes = self.dim * std::mem::size_of::<f32>();
        if scratch_bytes > self.ctx.properties().max_threadgroup_memory_length as usize {
            return Err(MetalError::InvalidConfig(format!(
                "TurboQuant transform dim {} requires {} bytes of threadgroup scratch, device limit is {}",
                self.dim,
                scratch_bytes,
                self.ctx.properties().max_threadgroup_memory_length
            )));
        }
        let input_buffer = MetalBuffer::from_slice(&self.ctx, input, BufferUsage::GpuReadOnly)?;
        let output_buffer = MetalBuffer::new(&self.ctx, input.len(), BufferUsage::Shared)?;

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "turboquant_apply_rows", None)?
        };

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
            encoder.setBuffer_offset_atIndex(Some(input_buffer.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(self.matrix.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output_buffer.metal_buffer()), 0, 2);
        }

        let params = TurboQuantTransformParams {
            dim: self.dim as u32,
        };
        let params_ptr = NonNull::from(&params).cast();
        unsafe {
            encoder.setBytes_length_atIndex(
                params_ptr,
                std::mem::size_of::<TurboQuantTransformParams>(),
                3,
            );
        }
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(scratch_bytes, 0);
        }

        let threadgroup_width =
            choose_threadgroup_width(pipeline.maxTotalThreadsPerThreadgroup(), self.dim);
        let grid_size = MTLSize {
            width: num_rows,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: threadgroup_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        output_buffer.to_vec()
    }
}

fn choose_threadgroup_width(max_threads: usize, dim: usize) -> usize {
    let capped = dim.next_power_of_two().min(max_threads).min(256);
    capped.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_apply_rows(matrix: &[f32], dim: usize, input: &[f32]) -> Vec<f32> {
        let rows = input.len() / dim;
        let mut output = vec![0.0f32; input.len()];
        for row in 0..rows {
            let src = &input[row * dim..(row + 1) * dim];
            let dst = &mut output[row * dim..(row + 1) * dim];
            for out_dim in 0..dim {
                let matrix_row = &matrix[out_dim * dim..(out_dim + 1) * dim];
                dst[out_dim] = matrix_row.iter().zip(src.iter()).map(|(a, b)| a * b).sum();
            }
        }
        output
    }

    #[test]
    fn turboquant_transform_matches_cpu() {
        let dim = 8usize;
        let matrix: Vec<f32> = (0..dim * dim)
            .map(|index| ((index % dim) as f32 - 3.5) * 0.125)
            .collect();
        let input: Vec<f32> = (0..(dim * 3))
            .map(|index| ((index % 11) as f32 - 5.0) * 0.2)
            .collect();

        let transform = TurboQuantTransform::new(&matrix, dim).unwrap();
        let gpu = transform.apply_rows(&input).unwrap();
        let cpu = cpu_apply_rows(&matrix, dim, &input);

        assert_eq!(gpu.len(), cpu.len());
        for (lhs, rhs) in gpu.iter().zip(cpu.iter()) {
            assert!((lhs - rhs).abs() < 1e-4, "lhs={lhs} rhs={rhs}");
        }
    }
}
