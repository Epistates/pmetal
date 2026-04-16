#![allow(unsafe_code)]

//! Metal-accelerated dequantization kernels.
//!
//! Thin wrappers around [`dispatch_linear_kernel`] — each entry point just
//! names the Metal function to bind and forwards the input / output
//! buffers. Any failure to allocate a command buffer or encoder now
//! propagates as [`MetalError`] instead of panicking via `None.unwrap()`.

use crate::context::MetalContext;
use crate::error::Result;
use crate::kernels::dispatch::dispatch_linear_kernel;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

type MetalBufferRef = Retained<ProtocolObject<dyn MTLBuffer>>;

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
        input_buffer: &MetalBufferRef,
        output_buffer: &MetalBufferRef,
        n_elements: usize,
    ) -> Result<()> {
        dispatch_linear_kernel(
            ctx,
            "dequantize_q4_0",
            &[input_buffer, output_buffer],
            n_elements,
        )
    }

    /// Dequantize IQ4_XS data to a float buffer.
    pub fn dequantize_iq4_xs(
        &self,
        ctx: &MetalContext,
        input_buffer: &MetalBufferRef,
        output_buffer: &MetalBufferRef,
        n_elements: usize,
    ) -> Result<()> {
        dispatch_linear_kernel(
            ctx,
            "dequantize_iq4_xs",
            &[input_buffer, output_buffer],
            n_elements,
        )
    }
}
