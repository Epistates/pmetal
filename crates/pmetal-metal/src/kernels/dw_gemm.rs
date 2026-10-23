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

use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use parking_lot::Mutex;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::MetalContext;
use crate::error::Result;
use crate::kernels::BatchedCommandBuffer;

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

    /// Stage host operands in `pool` and encode `C = alpha * A @ B^T + beta * C`
    /// into `batch`.
    ///
    /// `a` is `[M, K]` and `b` is `[N, K]`, row-major, copied into buffers of
    /// their own that stay untouched until `pool` is reset, so any number of
    /// these can share one batch and each reads its own operands. Reset the
    /// pool only after the batch has executed.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_gemm_accum_staged(
        &self,
        batch: &mut BatchedCommandBuffer,
        pool: &ScratchPool,
        a: &[f32],
        b: &[f32],
        c: &MetalBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let a = pool.stage(&a[..m * k])?;
        let b = pool.stage(&b[..n * k])?;
        self.queue_gemm_accum(batch, &a, &b, c, m, n, k, alpha, beta)
    }
}

// =============================================================================
// Scratch Pool — staging buffers for host operands
// =============================================================================

/// Staging buffers for the host-side operands of GPU GEMMs.
///
/// A command buffer runs only when it is executed, long after the operand
/// slices passed to [`DwGemm::queue_gemm_accum_staged`] were copied, so every
/// GEMM queued into it needs a buffer of its own: [`stage`](Self::stage) hands
/// out a different one on every call until [`reset`](Self::reset). The buffers
/// are kept and reused by position, so a step that stages the same sequence of
/// operands as the last one allocates nothing.
pub struct ScratchPool {
    ctx: Arc<MetalContext>,
    state: Mutex<PoolState>,
}

#[derive(Default)]
struct PoolState {
    /// Every buffer ever handed out, in the order it was first handed out.
    buffers: Vec<MetalBuffer<f32>>,
    /// Index of the next buffer to hand out.
    next: usize,
}

impl ScratchPool {
    /// Create an empty pool.
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        Self {
            ctx,
            state: Mutex::new(PoolState::default()),
        }
    }

    /// Copy `data` into a buffer no earlier call has returned since the last
    /// [`reset`](Self::reset), and return it.
    ///
    /// The buffer holds at least `data.len()` elements (a reused one may hold
    /// more) and stays untouched until the next `reset`.
    pub fn stage(&self, data: &[f32]) -> Result<MetalBuffer<f32>> {
        let mut state = self.state.lock();
        let slot = state.next;
        if slot == state.buffers.len() {
            let buffer = MetalBuffer::new(&self.ctx, data.len(), BufferUsage::Shared)?;
            state.buffers.push(buffer);
        } else if state.buffers[slot].len() < data.len() {
            state.buffers[slot] = MetalBuffer::new(&self.ctx, data.len(), BufferUsage::Shared)?;
        }
        state.next = slot + 1;
        let buffer = state.buffers[slot].clone();
        // The GPU is not using this buffer: it was not handed out since the
        // last reset, and a reset is only valid once the work has finished.
        buffer.as_mut_slice_unchecked()[..data.len()].copy_from_slice(data);
        Ok(buffer)
    }

    /// Make every buffer available again.
    ///
    /// Call this once the batches that use the staged operands have executed.
    pub fn reset(&self) {
        self.state.lock().next = 0;
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

        let pool = ScratchPool::new(ctx);
        let first = pool.stage(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        // The same length twice in a row must not share a buffer.
        let second = pool.stage(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        let third = pool.stage(&[9.0, 10.0]).unwrap();
        assert_eq!(&first.as_slice()[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&second.as_slice()[..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&third.as_slice()[..2], &[9.0, 10.0]);

        // After a reset the same positions are reused, grown when too small.
        pool.reset();
        let again = pool.stage(&[1.5; 6]).unwrap();
        assert!(again.len() >= 6);
        assert_eq!(&again.as_slice()[..6], &[1.5; 6]);
    }
}
