#![allow(unsafe_code)]

//! Fused Gated Delta Network (GDN) Metal kernel.
//!
//! Provides a fused recurrent kernel for GDN decode/short-prefill that replaces
//! ~10 MLX op dispatches per timestep with a single Metal compute dispatch.
//!
//! # Performance
//!
//! For decode (T=1), this eliminates the dominant bottleneck: MLX op dispatch
//! overhead for the delta-rule recurrence. The kernel keeps the [Dk/32, BV]
//! state tile in registers and uses SIMD reductions for K-dimension sums.
//!
//! # Usage
//!
//! ```ignore
//! let config = FusedGdnConfig {
//!     key_dim: 64,
//!     value_dim: 128,
//!     value_block: 16,
//!     scalar_gate: true,
//! };
//! let gdn = FusedGdn::new(&ctx, config)?;
//!
//! // For decode (T=1):
//! gdn.forward(&q, &k, &v, &g, &beta, &mut state, &mut output)?;
//! ```

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    buffer::MetalBuffer,
    context::MetalContext,
    error::{MetalError, Result},
    pipeline::FunctionConstant,
};

/// Configuration for the fused GDN kernel.
#[derive(Debug, Clone)]
pub struct FusedGdnConfig {
    /// Key dimension (Dk). Must be divisible by 32 (SIMD width).
    pub key_dim: u32,

    /// Value dimension (Dv).
    pub value_dim: u32,

    /// Value block size (BV). Each threadgroup handles BV values.
    /// Smaller = more threadgroups = more parallelism but less work per group.
    /// Typical values: 8 or 16.
    pub value_block: u32,

    /// Whether gating is scalar (true) or vectorized (false).
    /// Qwen3.5 uses scalar gating.
    pub scalar_gate: bool,
}

impl FusedGdnConfig {
    /// Create config for a given model's GDN parameters.
    pub fn new(key_dim: u32, value_dim: u32) -> Self {
        // Default BV=16 balances parallelism vs per-group work.
        // For Dv=128: 8 threadgroups per head.
        let value_block = if value_dim <= 64 { 8 } else { 16 };
        Self {
            key_dim,
            value_dim,
            value_block,
            scalar_gate: true,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.key_dim == 0 || self.key_dim % 32 != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "key_dim must be a positive multiple of 32, got {}",
                self.key_dim
            )));
        }
        if self.key_dim > 256 {
            return Err(MetalError::InvalidConfig(format!(
                "key_dim must be ≤256 (K_PER_THREAD ≤ 8), got {}",
                self.key_dim
            )));
        }
        if self.value_block == 0 || self.value_block > 16 {
            return Err(MetalError::InvalidConfig(format!(
                "value_block must be in [1, 16], got {}",
                self.value_block
            )));
        }
        Ok(())
    }
}

/// Kernel parameters matching the Metal struct `GdnRecurrentParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GdnRecurrentParams {
    batch_size: u32,
    num_heads: u32,
    key_dim: u32,
    value_dim: u32,
    seq_len: u32,
}

/// Fused GDN recurrent kernel handle.
///
/// Caches the specialized Metal pipeline for a specific (Dk, BV, scalar_gate)
/// configuration. Thread-safe via Arc.
pub struct FusedGdn {
    ctx: Arc<MetalContext>,
    config: FusedGdnConfig,
}

impl FusedGdn {
    /// Create a new fused GDN kernel instance.
    ///
    /// This validates the configuration and eagerly creates the specialized
    /// Metal pipeline (caching it in the MetalContext's pipeline cache).
    pub fn new(ctx: Arc<MetalContext>, config: FusedGdnConfig) -> Result<Self> {
        config.validate()?;

        // Eagerly create the specialized pipeline to fail fast on shader errors.
        {
            let constants = Self::function_constants(&config);
            let mut cache = ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                ctx.device(),
                "gdn_fused_recurrent_fwd",
                &constants,
            )?;
        }

        Ok(Self { ctx, config })
    }

    /// Build function constants for pipeline specialization.
    fn function_constants(config: &FusedGdnConfig) -> HashMap<u64, FunctionConstant> {
        let mut constants = HashMap::new();
        constants.insert(0, FunctionConstant::UInt(config.key_dim));
        constants.insert(1, FunctionConstant::UInt(config.value_block));
        constants.insert(
            2,
            FunctionConstant::UInt(if config.scalar_gate { 1 } else { 0 }),
        );
        constants
    }

    /// Execute the fused GDN forward recurrence.
    ///
    /// # Arguments
    ///
    /// * `q` - Queries, shape `[B, T, Hv, Dk]` (GQA-expanded to Hv heads)
    /// * `k` - Keys, shape `[B, T, Hv, Dk]` (GQA-expanded to Hv heads)
    /// * `v` - Values, shape `[B, T, Hv, Dv]`
    /// * `g` - Gating decay factor in (0,1] from `compute_g()`, shape `[B, T, Hv]`
    /// * `beta` - Beta gate in (0,1) from `sigmoid()`, shape `[B, T, Hv]`
    /// * `state` - Recurrent state, shape `[B, Hv, Dv, Dk]`. Modified in-place.
    /// * `output` - Output buffer, shape `[B, T, Hv, Dv]`. Written by kernel.
    ///
    /// # Returns
    ///
    /// Ok(()) on success. State is updated in-place, output is written.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        batch_size: u32,
        num_heads: u32,
        seq_len: u32,
        q: &MetalBuffer<f32>,
        k: &MetalBuffer<f32>,
        v: &MetalBuffer<f32>,
        g: &MetalBuffer<f32>,
        beta: &MetalBuffer<f32>,
        state: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
    ) -> Result<()> {
        let constants = Self::function_constants(&self.config);
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                self.ctx.device(),
                "gdn_fused_recurrent_fwd",
                &constants,
            )?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        // Set buffers.
        // SAFETY: Metal buffers are valid and encoder is in correct state.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(g.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(beta.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(state.metal_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 6);

            let params = GdnRecurrentParams {
                batch_size,
                num_heads,
                key_dim: self.config.key_dim,
                value_dim: self.config.value_dim,
                seq_len,
            };
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 7);
        }

        // Grid: (NV, B * Hv) where NV = ceil(Dv / BV)
        let nv = self.config.value_dim.div_ceil(self.config.value_block);
        let grid_size = objc2_metal::MTLSize {
            width: nv as usize,
            height: (batch_size * num_heads) as usize,
            depth: 1,
        };

        // Threadgroup: 32 threads (1 SIMD group)
        let threadgroup_size = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Get the key dimension this kernel was specialized for.
    pub fn key_dim(&self) -> u32 {
        self.config.key_dim
    }

    /// Get the value dimension this kernel was specialized for.
    pub fn value_dim(&self) -> u32 {
        self.config.value_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::BufferUsage;

    #[test]
    fn test_fused_gdn_config_validation() {
        // Valid
        assert!(FusedGdnConfig::new(64, 128).validate().is_ok());
        assert!(FusedGdnConfig::new(128, 128).validate().is_ok());
        assert!(FusedGdnConfig::new(32, 64).validate().is_ok());

        // Invalid: not multiple of 32
        let mut c = FusedGdnConfig::new(64, 128);
        c.key_dim = 48;
        assert!(c.validate().is_err());

        // Invalid: too large
        c.key_dim = 512;
        assert!(c.validate().is_err());

        // Invalid: zero
        c.key_dim = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_fused_gdn_pipeline_creation() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let config = FusedGdnConfig::new(64, 128);
        let gdn = FusedGdn::new(ctx, config);
        assert!(gdn.is_ok(), "Pipeline creation failed: {:?}", gdn.err());
    }

    #[test]
    fn test_fused_gdn_decode_correctness() {
        // Test fused kernel against reference implementation for T=1.
        let ctx = Arc::new(MetalContext::new().unwrap());
        let config = FusedGdnConfig::new(64, 128);
        let gdn = FusedGdn::new(ctx.clone(), config).unwrap();

        let b: u32 = 1;
        let hv: u32 = 4;
        let dk: u32 = 64;
        let dv: u32 = 128;
        let t: u32 = 1;

        // Create test data with known values.
        let q_data: Vec<f32> = (0..b * t * hv * dk)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let k_data: Vec<f32> = (0..b * t * hv * dk)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();
        let v_data: Vec<f32> = (0..b * t * hv * dv)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();
        let g_data: Vec<f32> = (0..b * t * hv)
            .map(|i| -0.1 * (i as f32 + 1.0)) // Negative = decay
            .collect();
        let beta_data: Vec<f32> = (0..b * t * hv).map(|_| 0.5).collect();
        let state_data: Vec<f32> = (0..b * hv * dv * dk)
            .map(|i| (i as f32 * 0.005).cos() * 0.1)
            .collect();

        // Reference: compute step manually.
        let mut ref_state = state_data.clone();
        let mut ref_output = vec![0.0f32; (b * t * hv * dv) as usize];

        for batch in 0..b {
            for head in 0..hv {
                let g_val = g_data[(batch * t * hv + head) as usize];
                let beta_val = beta_data[(batch * t * hv + head) as usize];
                // Decay state (g is already the decay factor, not log-space)
                for vi in 0..dv {
                    for ki in 0..dk {
                        let idx = ((batch * hv + head) * dv + vi) * dk + ki;
                        ref_state[idx as usize] *= g_val;
                    }
                }

                // kv_mem[vi] = sum_k(state[vi,ki] * k[ki])
                let mut kv_mem = vec![0.0f32; dv as usize];
                for vi in 0..dv {
                    for ki in 0..dk {
                        let s_idx = ((batch * hv + head) * dv + vi) * dk + ki;
                        let k_idx = (batch * t * hv + head) * dk + ki;
                        kv_mem[vi as usize] += ref_state[s_idx as usize] * k_data[k_idx as usize];
                    }
                }

                // delta[vi] = beta * (v[vi] - kv_mem[vi])
                let mut delta = vec![0.0f32; dv as usize];
                for vi in 0..dv {
                    let v_idx = (batch * t * hv + head) * dv + vi;
                    delta[vi as usize] = beta_val * (v_data[v_idx as usize] - kv_mem[vi as usize]);
                }

                // state += k[:] * delta[:]  (outer product)
                for vi in 0..dv {
                    for ki in 0..dk {
                        let s_idx = ((batch * hv + head) * dv + vi) * dk + ki;
                        let k_idx = (batch * t * hv + head) * dk + ki;
                        ref_state[s_idx as usize] += k_data[k_idx as usize] * delta[vi as usize];
                    }
                }

                // output[vi] = sum_k(state[vi,ki] * q[ki])
                for vi in 0..dv {
                    let mut acc = 0.0f32;
                    for ki in 0..dk {
                        let s_idx = ((batch * hv + head) * dv + vi) * dk + ki;
                        let q_idx = (batch * t * hv + head) * dk + ki;
                        acc += ref_state[s_idx as usize] * q_data[q_idx as usize];
                    }
                    let o_idx = (batch * t * hv + head) * dv + vi;
                    ref_output[o_idx as usize] = acc;
                }
            }
        }

        // Run fused kernel.
        let q_buf = MetalBuffer::from_slice(&ctx, &q_data, BufferUsage::Shared).unwrap();
        let k_buf = MetalBuffer::from_slice(&ctx, &k_data, BufferUsage::Shared).unwrap();
        let v_buf = MetalBuffer::from_slice(&ctx, &v_data, BufferUsage::Shared).unwrap();
        let g_buf = MetalBuffer::from_slice(&ctx, &g_data, BufferUsage::Shared).unwrap();
        let beta_buf = MetalBuffer::from_slice(&ctx, &beta_data, BufferUsage::Shared).unwrap();
        let state_buf = MetalBuffer::from_slice(&ctx, &state_data, BufferUsage::Shared).unwrap();
        let output_buf = MetalBuffer::from_slice(
            &ctx,
            &vec![0.0f32; (b * t * hv * dv) as usize],
            BufferUsage::Shared,
        )
        .unwrap();

        gdn.forward(
            b,
            hv,
            t,
            &q_buf,
            &k_buf,
            &v_buf,
            &g_buf,
            &beta_buf,
            &state_buf,
            &output_buf,
        )
        .unwrap();

        // Compare output and state with combined absolute + relative tolerance.
        // ffast-math on Metal introduces minor FP32 differences (~1e-4 relative).
        let atol = 1e-5;
        let rtol = 1e-3;
        let gpu_output = output_buf.as_slice();
        let gpu_state = state_buf.as_slice();

        for i in 0..ref_output.len() {
            let diff = (gpu_output[i] - ref_output[i]).abs();
            let tol = atol + rtol * ref_output[i].abs();
            assert!(
                diff < tol,
                "Output mismatch at index {}: gpu={}, ref={}, diff={}, tol={}",
                i,
                gpu_output[i],
                ref_output[i],
                diff,
                tol
            );
        }

        for i in 0..ref_state.len() {
            let diff = (gpu_state[i] - ref_state[i]).abs();
            let tol = atol + rtol * ref_state[i].abs();
            assert!(
                diff < tol,
                "State mismatch at index {}: gpu={}, ref={}, diff={}, tol={}",
                i,
                gpu_state[i],
                ref_state[i],
                diff,
                tol
            );
        }
    }
}
