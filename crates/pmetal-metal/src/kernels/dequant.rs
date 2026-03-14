//! Metal-accelerated dequantization kernels.

use crate::{
    context::MetalContext,
    error::Result,
};
use objc2_metal::{
    MTLSize, MTLCommandQueue, MTLComputeCommandEncoder, MTLCommandBuffer,
    MTLComputePipelineState, MTLCommandEncoder
};
use objc2::runtime::ProtocolObject;

/// Dequantization backend using Metal kernels.
pub struct DequantKernels;

impl DequantKernels {
    /// Create new dequantization kernels.
    pub fn new(_ctx: &MetalContext) -> Result<Self> {
        Ok(Self)
    }

    /// Dequantize Q4_0 data to a float buffer.
    pub fn dequantize_q4_0(
        &self,
        ctx: &MetalContext,
        input_buffer: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        output_buffer: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        n_elements: usize,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(ctx.device(), "dequantize_q4_0", None)?
        };

        let command_buffer = ctx.command_queue().commandBuffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buffer), 0, 1);
        }

        let grid_size = MTLSize {
            width: n_elements,
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: (pipeline.maxTotalThreadsPerThreadgroup() as usize).min(n_elements),
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(())
    }

    /// Dequantize IQ4_XS data to a float buffer.
    pub fn dequantize_iq4_xs(
        &self,
        ctx: &MetalContext,
        input_buffer: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        output_buffer: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        n_elements: usize,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(ctx.device(), "dequantize_iq4_xs", None)?
        };

        let command_buffer = ctx.command_queue().commandBuffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buffer), 0, 1);
        }

        let grid_size = MTLSize {
            width: n_elements,
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: (pipeline.maxTotalThreadsPerThreadgroup() as usize).min(n_elements),
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(())
    }
}
