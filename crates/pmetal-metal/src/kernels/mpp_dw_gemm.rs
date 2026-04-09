#![allow(unsafe_code)]

//! Metal 4 / MPP weight gradient GEMM dispatch.
//!
//! Provides hardware-accelerated weight gradient computation via Metal Performance
//! Primitives on M5+ (Apple10) GPUs with NAX cores.
//!
//! Computes: C = alpha * A @ B^T + beta * C
//!
//! In ANE training: 20 layers × 7 GEMMs = 140 dispatches per step.
//! These run on the GPU while ANE handles dx propagation.

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

/// Configuration for MPP weight gradient GEMM.
#[derive(Debug, Clone)]
pub struct MppDwGemmConfig {
    /// Output rows (M: activations/dY rows).
    pub m: usize,
    /// Output columns (N: weight gradient columns).
    pub n: usize,
    /// Reduction dimension (K).
    pub k: usize,
    /// Scalar multiplier for A @ B^T (default 1.0).
    pub alpha: f32,
    /// Scalar multiplier for existing C (0 = overwrite, 1 = accumulate).
    pub beta: f32,
}

impl MppDwGemmConfig {
    /// Create a new config for `C[M,N] = alpha * A[M,K] @ B[N,K]^T + beta * C[M,N]`.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
        }
    }

    /// Create a config for in-place accumulation (`beta = 1.0`).
    pub fn new_accumulate(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Output buffer element count.
    pub fn output_size(&self) -> usize {
        self.m * self.n
    }
}

/// Metal-side parameter block (must match `DwGemmParams` in Metal).
#[repr(C)]
struct DwGemmParams {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    num_tiles_m: u32,
    num_tiles_n: u32,
}

#[derive(Debug, Clone, Copy)]
struct DispatchGeometry {
    num_tiles_m: usize,
    num_tiles_n: usize,
    /// Threads per threadgroup: 32 (single SIMD group).
    ///
    /// The shader uses `execution_simdgroup` (MPP single-simdgroup API).
    /// Thread management within each 64×64 tile is handled by MPP internally;
    /// dispatching more than one simdgroup wastes the extra threads.
    threads_per_threadgroup: usize,
}

fn dispatch_geometry(config: &MppDwGemmConfig) -> DispatchGeometry {
    const BM: usize = 64;
    const BN: usize = 64;
    DispatchGeometry {
        num_tiles_m: config.m.div_ceil(BM),
        num_tiles_n: config.n.div_ceil(BN),
        threads_per_threadgroup: 32,
    }
}

/// MPP weight gradient GEMM dispatcher.
///
/// Dispatches to `mpp_dw_gemm_accum` on M5+ hardware. The Metal shader handles
/// both the overwrite (`beta=0, alpha=1`) fast path and the general accumulate path.
pub struct MppDwGemm {
    ctx: Arc<MetalContext>,
    config: MppDwGemmConfig,
}

impl MppDwGemm {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppDwGemmConfig) -> Self {
        Self { ctx, config }
    }

    /// Check if MPP dW GEMM is available (requires M5+ with NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute synchronously.
    ///
    /// `a`: `[M, K]` fp32 (activations or dY), `b`: `[N, K]` fp32 (transposed weights),
    /// `c`: `[M, N]` fp32 (weight gradient, read/written when `beta != 0`, written otherwise).
    pub fn execute(
        &self,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        c: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let command_buffer = self.execute_async(a, b, c)?;
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute asynchronously and return the submitted command buffer.
    pub fn execute_async(
        &self,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        c: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP dW GEMM not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let geometry = dispatch_geometry(&self.config);

        let params = DwGemmParams {
            m: self.config.m as u32,
            n: self.config.n as u32,
            k: self.config.k as u32,
            alpha: self.config.alpha,
            beta: self.config.beta,
            num_tiles_m: geometry.num_tiles_m as u32,
            num_tiles_n: geometry.num_tiles_n as u32,
        };

        // 2D grid: [num_n_tiles, num_m_tiles, 1]
        // (Metal shader uses tgid.x for N tiles, tgid.y for M tiles)
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

        let a_buf = a.as_metal_buffer();
        let b_buf = b.as_metal_buffer();
        let c_buf = c.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, "mpp_dw_gemm_accum", grid, tg_size, |encoder| unsafe {
            // buffer(0): A, buffer(1): B, buffer(2): C, buffer(3): params
            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c_buf), 0, 2);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_output_size() {
        let config = MppDwGemmConfig::new(4096, 4096, 512);
        assert_eq!(config.output_size(), 4096 * 4096);
    }

    #[test]
    fn test_dispatch_geometry_tile_counts() {
        let config = MppDwGemmConfig::new(128, 256, 64);
        let geom = dispatch_geometry(&config);
        assert_eq!(geom.num_tiles_m, 2);
        assert_eq!(geom.num_tiles_n, 4);
        assert_eq!(geom.threads_per_threadgroup, 32);
    }

    #[test]
    fn test_dispatch_geometry_non_aligned() {
        let config = MppDwGemmConfig::new(65, 65, 32);
        let geom = dispatch_geometry(&config);
        assert_eq!(geom.num_tiles_m, 2);
        assert_eq!(geom.num_tiles_n, 2);
    }

    #[test]
    fn test_accumulate_constructor() {
        let config = MppDwGemmConfig::new_accumulate(128, 256, 64);
        assert!((config.alpha - 1.0).abs() < 1e-6);
        assert!((config.beta - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_overwrite_constructor() {
        let config = MppDwGemmConfig::new(128, 256, 64);
        assert!((config.alpha - 1.0).abs() < 1e-6);
        assert!(config.beta.abs() < 1e-6);
    }
}
