#![allow(unsafe_code)]

//! GPU weight-gradient GEMM kernel for ANE training backward pass.
//!
//! Replaces the per-layer cblas SGEMM dispatches (previously on a dedicated CPU worker
//! thread) with Metal GPU compute. All dW GEMMs per training step are encoded into a
//! single [`BatchedCommandBuffer`] for one GPU-CPU sync at the end.
//!
//! Kernel: `dw_gemm_accum` — `C = alpha * A @ B^T + beta * C`
//!
//! At 579M params (d=1024, h=2816, s=512): 20 layers × 7 GEMMs = 140 dispatches,
//! ~230 GFLOP total.
//!
//! The kernel is a 64 × 64 × 16 tiled SGEMM on `simdgroup_matrix` fragments. It
//! zero-fills partial tiles, so M, N and K may be anything, and with
//! `beta == 0` it never reads `C`, so a fresh buffer is fine as the output.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::MetalContext;
use crate::error::Result;
use crate::kernels::fused_training::BatchedCommandBuffer;

// Must match the Metal shader struct layout exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
struct DwGemmParams {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
}

// `DwGemmParams` in `metal/dw_gemm.metal`.
crate::msl_layout::msl_layout!(DwGemmParams, size = 20, {
    m: 0,
    n: 4,
    k: 8,
    alpha: 12,
    beta: 16,
});

/// Tile dimensions and threads per threadgroup — must match `dw_gemm.metal`.
const BM: usize = 64;
const BN: usize = 64;
const THREADS_PER_GROUP: usize = 128;

/// Minimum model dimension to use GPU path. Below this threshold the
/// Metal dispatch overhead exceeds the compute win over cblas.
pub const GPU_DW_MIN_DIM: usize = 256;

/// GPU weight-gradient GEMM dispatcher.
///
/// Encodes `C += alpha * A @ B^T` into an existing [`BatchedCommandBuffer`].
/// Typical usage: 7 per-layer dW GEMMs + 1 embedding GEMM, all in one buffer.
pub struct DwGemm {
    ctx: Arc<MetalContext>,
}

impl DwGemm {
    /// Create a new dispatcher, eagerly compiling the pipeline.
    pub fn new(ctx: Arc<MetalContext>) -> Result<Self> {
        // Eagerly compile to catch shader errors at init, not first dispatch.
        {
            let mut cache = ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(ctx.device(), "dw_gemm_accum", None)?;
        }
        Ok(Self { ctx })
    }

    /// Encode `C = alpha * A @ B^T + beta * C` into `batch`.
    ///
    /// - `a`: `[M, K]` row-major (gradient tensor)
    /// - `b`: `[N, K]` row-major (activation tensor, transposed on read)
    /// - `c`: `[M, N]` row-major (weight gradient accumulator, read-modify-write)
    ///
    /// All three buffers must be `Shared` (unified memory).
    #[allow(clippy::too_many_arguments)]
    pub fn queue_gemm_accum(
        &self,
        batch: &mut BatchedCommandBuffer,
        a: &MetalBuffer<f32>,
        b: &MetalBuffer<f32>,
        c: &MetalBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        debug_assert!(
            a.len() >= m * k,
            "A buffer too small: {} < {}",
            a.len(),
            m * k
        );
        debug_assert!(
            b.len() >= n * k,
            "B buffer too small: {} < {}",
            b.len(),
            n * k
        );
        debug_assert!(
            c.len() >= m * n,
            "C buffer too small: {} < {}",
            c.len(),
            m * n
        );

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "dw_gemm_accum", None)?
        };

        let encoder = batch.encoder_mut()?;
        encoder.setComputePipelineState(&pipeline);

        let params = DwGemmParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha,
            beta,
        };

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c.metal_buffer()), 0, 2);

            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<DwGemmParams>(), 3);
        }

        // Grid: width=ceil(N/BN) → tg_id.x (tile_col), height=ceil(M/BM) → tg_id.y (tile_row)
        let grid_size = MTLSize {
            width: n.div_ceil(BN),
            height: m.div_ceil(BM),
            depth: 1,
        };

        // Four simdgroups, 2 x 2 over the 64 x 64 tile.
        let threadgroup_size = MTLSize {
            width: THREADS_PER_GROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }
}

// =============================================================================
// Scratch Pool — size-bucketed MetalBuffer pool for activation copies
// =============================================================================

/// A pool of reusable `MetalBuffer<f32>` scratch buffers, bucketed by size.
///
/// Avoids per-dispatch allocation of temporary GPU buffers for activation
/// operands. Buffers are allocated once at the largest required size per bucket
/// and reused across training steps.
pub struct ScratchPool {
    ctx: Arc<MetalContext>,
    /// Cached buffers keyed by element count. Each key maps to a single buffer
    /// that is reused whenever a buffer of that exact size is requested.
    cache: HashMap<usize, MetalBuffer<f32>>,
}

impl ScratchPool {
    /// Create an empty pool.
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        Self {
            ctx,
            cache: HashMap::new(),
        }
    }

    /// Get or create a shared MetalBuffer of exactly `len` elements,
    /// and copy `data` into it.
    ///
    /// Returns a reference to the cached buffer. The buffer is valid until
    /// the next call to `get()` with the same `len`, or until `reset()`.
    pub fn get(&mut self, data: &[f32]) -> Result<&MetalBuffer<f32>> {
        let len = data.len();
        // Get or create buffer of this size
        if !self.cache.contains_key(&len) {
            let buf = MetalBuffer::new(&self.ctx, len, BufferUsage::Shared)?;
            self.cache.insert(len, buf);
        }
        let buf = self.cache.get_mut(&len).unwrap();
        // Copy activation data into the GPU-visible buffer
        buf.as_mut_slice_unchecked().copy_from_slice(data);
        // Return immutable ref — the data is now GPU-visible
        Ok(self.cache.get(&len).unwrap())
    }

    /// Pre-warm the pool with known sizes to avoid first-dispatch allocation.
    pub fn prewarm(&mut self, sizes: &[usize]) -> Result<()> {
        for &len in sizes {
            if !self.cache.contains_key(&len) {
                let buf = MetalBuffer::new(&self.ctx, len, BufferUsage::Shared)?;
                self.cache.insert(len, buf);
            }
        }
        Ok(())
    }

    /// Drop all cached buffers.
    pub fn reset(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare GPU dw_gemm_accum output against Accelerate cblas_sgemm for correctness.
    #[test]
    fn test_dw_gemm_correctness() {
        let ctx = match MetalContext::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return, // skip on non-Metal systems
        };

        let m = 128;
        let n = 64;
        let k = 32;

        // Random-ish deterministic data
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01)
            .collect();
        let b_data: Vec<f32> = (0..n * k)
            .map(|i| ((i * 11 + 5) % 100) as f32 * 0.01)
            .collect();
        let c_init: Vec<f32> = (0..m * n)
            .map(|i| ((i * 13 + 7) % 100) as f32 * 0.001)
            .collect();

        // ---- CPU reference: C_ref = 1.0 * A @ B^T + 1.0 * C ----
        let mut c_ref = c_init.clone();
        for row in 0..m {
            for col in 0..n {
                let mut dot = 0.0f32;
                for kk in 0..k {
                    dot += a_data[row * k + kk] * b_data[col * k + kk];
                }
                c_ref[row * n + col] += dot;
            }
        }

        // ---- GPU ----
        let a_buf = MetalBuffer::from_slice(&ctx, &a_data, BufferUsage::Shared).unwrap();
        let b_buf = MetalBuffer::from_slice(&ctx, &b_data, BufferUsage::Shared).unwrap();
        let c_buf = MetalBuffer::from_slice(&ctx, &c_init, BufferUsage::Shared).unwrap();

        let dw = DwGemm::new(ctx.clone()).unwrap();
        let mut batch = BatchedCommandBuffer::new(ctx).unwrap();
        dw.queue_gemm_accum(&mut batch, &a_buf, &b_buf, &c_buf, m, n, k, 1.0, 1.0)
            .unwrap();
        batch.execute().unwrap();

        let c_gpu = c_buf.as_slice();

        // Check accuracy (fp32 should be very close)
        let mut max_err = 0.0f32;
        for i in 0..m * n {
            let err = (c_gpu[i] - c_ref[i]).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-3,
                "Mismatch at index {}: gpu={}, ref={}, err={}",
                i,
                c_gpu[i],
                c_ref[i],
                err,
            );
        }
        eprintln!("dw_gemm correctness test passed, max_err={max_err:.2e}");
    }

    /// Test with non-tile-aligned dimensions (boundary handling).
    #[test]
    fn test_dw_gemm_boundary() {
        let ctx = match MetalContext::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };

        // Non-aligned: 70 not divisible by 64, 50 not divisible by 64, 19 not divisible by 16
        let m = 70;
        let n = 50;
        let k = 19;

        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 3 + 1) % 50) as f32 * 0.02)
            .collect();
        let b_data: Vec<f32> = (0..n * k)
            .map(|i| ((i * 5 + 2) % 50) as f32 * 0.02)
            .collect();
        let c_init: Vec<f32> = vec![0.0; m * n];

        // CPU reference (beta=0 this time)
        let mut c_ref = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut dot = 0.0f32;
                for kk in 0..k {
                    dot += a_data[row * k + kk] * b_data[col * k + kk];
                }
                c_ref[row * n + col] = 2.0 * dot; // alpha=2.0
            }
        }

        let a_buf = MetalBuffer::from_slice(&ctx, &a_data, BufferUsage::Shared).unwrap();
        let b_buf = MetalBuffer::from_slice(&ctx, &b_data, BufferUsage::Shared).unwrap();
        let c_buf = MetalBuffer::from_slice(&ctx, &c_init, BufferUsage::Shared).unwrap();

        let dw = DwGemm::new(ctx.clone()).unwrap();
        let mut batch = BatchedCommandBuffer::new(ctx).unwrap();
        dw.queue_gemm_accum(&mut batch, &a_buf, &b_buf, &c_buf, m, n, k, 2.0, 0.0)
            .unwrap();
        batch.execute().unwrap();

        let c_gpu = c_buf.as_slice();
        for i in 0..m * n {
            let err = (c_gpu[i] - c_ref[i]).abs();
            assert!(
                err < 1e-3,
                "Boundary mismatch at {}: gpu={}, ref={}, err={}",
                i,
                c_gpu[i],
                c_ref[i],
                err,
            );
        }
    }

    #[test]
    fn test_scratch_pool() {
        let ctx = match MetalContext::new() {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };

        let mut pool = ScratchPool::new(ctx);
        let data = vec![1.0f32, 2.0, 3.0, 4.0];

        // First get allocates
        let buf = pool.get(&data).unwrap();
        assert_eq!(buf.as_slice(), &data);

        // Second get with same size reuses
        let data2 = vec![5.0f32, 6.0, 7.0, 8.0];
        let buf2 = pool.get(&data2).unwrap();
        assert_eq!(buf2.as_slice(), &data2);
        assert_eq!(buf2.len(), 4); // same bucket

        // Different size creates new entry
        let data3 = vec![9.0f32, 10.0];
        let buf3 = pool.get(&data3).unwrap();
        assert_eq!(buf3.as_slice(), &data3);
        assert_eq!(buf3.len(), 2);
    }
}
