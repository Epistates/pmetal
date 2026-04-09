#![allow(unsafe_code)]

//! Metal 4 / MPP Grouped GEMM dispatch for MoE models.
//!
//! Provides hardware-accelerated grouped GEMM via Metal Performance Primitives
//! on M5+ (Apple10) GPUs with NAX cores.
//!
//! MoE decode bottleneck: 3 gather_mm × 28 layers × up to 8 active experts.
//! This kernel is on the hottest path in MoE model inference.
//!
//! Computes: Y[token, :] = X[token, :] @ W[expert, :, :]^T per expert bucket.

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

/// Configuration for MPP Grouped GEMM.
#[derive(Debug, Clone)]
pub struct MppGroupedGemmConfig {
    /// Total token-expert pairs (M).
    pub total_tokens: usize,
    /// Number of experts (E).
    pub num_experts: usize,
    /// Input hidden dimension (K).
    pub hidden_size: usize,
    /// Output dimension per expert (N).
    pub intermediate: usize,
    /// Top-k experts per token.
    pub topk: usize,
    /// Use fp16 (true) or fp32 (false).
    pub use_fp16: bool,
}

impl MppGroupedGemmConfig {
    /// Create a new config for grouped GEMM.
    pub fn new(
        total_tokens: usize,
        num_experts: usize,
        hidden_size: usize,
        intermediate: usize,
        topk: usize,
    ) -> Self {
        Self {
            total_tokens,
            num_experts,
            hidden_size,
            intermediate,
            topk,
            use_fp16: true,
        }
    }

    /// Total tiles across all experts for a dispatch.
    ///
    /// The exact total is data-dependent (depends on token routing) and is
    /// computed at dispatch time by reading `expert_offsets`. This returns
    /// a conservative upper bound for allocation purposes.
    pub fn max_tiles(&self) -> usize {
        let num_m_tiles = self.total_tokens.div_ceil(64);
        let num_n_tiles = self.intermediate.div_ceil(64);
        num_m_tiles * num_n_tiles * self.num_experts
    }
}

/// Metal-side parameter block (must match `GroupedGemmParams` in Metal).
#[repr(C)]
struct GroupedGemmParams {
    total_tokens: u32,
    num_experts: u32,
    hidden_size: u32,
    intermediate: u32,
    topk: u32,
    permute_x: u32,
    permute_y: u32,
    fuse_mul: u32,
}

/// Total threadgroup count for the dispatch.
///
/// This must be pre-computed from expert_offsets on the CPU before dispatch since the
/// Metal shader uses a flat 1D grid and iterates expert boundaries internally.
#[derive(Debug, Clone, Copy)]
pub struct GroupedGemmDispatch {
    /// Total tiles across all non-empty experts (1D grid width).
    pub total_tiles: usize,
    /// Threads per threadgroup: 32 (single SIMD group).
    ///
    /// The shader uses `execution_simdgroup` (MPP single-simdgroup API).
    /// Each 64×64 tile is handled by one simdgroup; additional simdgroups
    /// in the threadgroup would be idle.
    pub threads_per_threadgroup: usize,
}

impl GroupedGemmDispatch {
    /// Compute the dispatch geometry from a slice of per-expert token counts.
    ///
    /// `expert_token_counts[e]` is the number of tokens routed to expert `e`.
    pub fn from_token_counts(expert_token_counts: &[usize], intermediate: usize) -> Self {
        const BLOCK_M: usize = 64;
        const BLOCK_N: usize = 64;
        let num_n_tiles = intermediate.div_ceil(BLOCK_N);
        let total_tiles = expert_token_counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| count.div_ceil(BLOCK_M) * num_n_tiles)
            .sum();
        Self {
            total_tiles,
            threads_per_threadgroup: 32,
        }
    }
}

fn kernel_name(config: &MppGroupedGemmConfig) -> &'static str {
    if config.use_fp16 {
        "mpp_grouped_gemm_forward_f16"
    } else {
        "mpp_grouped_gemm_forward_f32"
    }
}

/// MPP Grouped GEMM dispatcher.
///
/// Dispatches to `mpp_grouped_gemm_forward_{f16,f32}` on M5+ hardware.
pub struct MppGroupedGemm {
    ctx: Arc<MetalContext>,
    config: MppGroupedGemmConfig,
}

impl MppGroupedGemm {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppGroupedGemmConfig) -> Self {
        Self { ctx, config }
    }

    /// Check if MPP Grouped GEMM is available (requires M5+ with NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute synchronously.
    ///
    /// `x`: `[total_tokens, hidden_size]` — tokens pre-sorted by expert,
    /// `w`: `[num_experts, intermediate, hidden_size]` — expert weight matrices,
    /// `y`: `[total_tokens, intermediate]` — output,
    /// `expert_offsets`: `[num_experts + 1]` u32 — exclusive prefix sums of per-expert token counts,
    /// `gather_indices`: `[total_tokens]` u32 — original token indices (for permutation),
    /// `scatter_indices`: `[total_tokens]` u32 — output placement indices,
    /// `topk_weights`: `[total_tokens]` — expert routing weights for optional weight fusion,
    /// `dispatch`: total tile count computed from `expert_offsets`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        x: &dyn AsMetalBuffer,
        w: &dyn AsMetalBuffer,
        y: &dyn AsMetalBuffer,
        expert_offsets: &dyn AsMetalBuffer,
        gather_indices: &dyn AsMetalBuffer,
        scatter_indices: &dyn AsMetalBuffer,
        topk_weights: &dyn AsMetalBuffer,
        dispatch: GroupedGemmDispatch,
    ) -> Result<()> {
        let command_buffer = self.execute_async(
            x,
            w,
            y,
            expert_offsets,
            gather_indices,
            scatter_indices,
            topk_weights,
            dispatch,
        )?;
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute asynchronously and return the submitted command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_async(
        &self,
        x: &dyn AsMetalBuffer,
        w: &dyn AsMetalBuffer,
        y: &dyn AsMetalBuffer,
        expert_offsets: &dyn AsMetalBuffer,
        gather_indices: &dyn AsMetalBuffer,
        scatter_indices: &dyn AsMetalBuffer,
        topk_weights: &dyn AsMetalBuffer,
        dispatch: GroupedGemmDispatch,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Grouped GEMM not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        if dispatch.total_tiles == 0 {
            return Err(MetalError::InvalidConfig(
                "MPP Grouped GEMM: total_tiles must be > 0".to_string(),
            ));
        }

        let kernel_name = kernel_name(&self.config);

        let params = GroupedGemmParams {
            total_tokens: self.config.total_tokens as u32,
            num_experts: self.config.num_experts as u32,
            hidden_size: self.config.hidden_size as u32,
            intermediate: self.config.intermediate as u32,
            topk: self.config.topk as u32,
            // permute and fuse_mul flags default to 0; callers enable via config extension.
            permute_x: 0,
            permute_y: 0,
            fuse_mul: 0,
        };

        // 1D flat grid: each threadgroup handles one tile from one expert.
        // The shader iterates expert_offsets to find which expert owns the tile.
        let grid = objc2_metal::MTLSize {
            width: dispatch.total_tiles,
            height: 1,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: dispatch.threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        let x_buf = x.as_metal_buffer();
        let w_buf = w.as_metal_buffer();
        let y_buf = y.as_metal_buffer();
        let eo_buf = expert_offsets.as_metal_buffer();
        let gi_buf = gather_indices.as_metal_buffer();
        let si_buf = scatter_indices.as_metal_buffer();
        let tw_buf = topk_weights.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, kernel_name, grid, tg_size, |encoder| unsafe {
            // buffer(0): x, buffer(1): w, buffer(2): y,
            // buffer(3): expert_offsets, buffer(4): gather_indices,
            // buffer(5): scatter_indices, buffer(6): topk_weights,
            // buffer(7): params
            encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(y_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(eo_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(gi_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(si_buf), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(tw_buf), 0, 6);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 7);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_from_token_counts_empty_experts_skipped() {
        // 2 experts: expert 0 has 64 tokens, expert 1 has 0 tokens, expert 2 has 128 tokens.
        let counts = vec![64, 0, 128];
        let dispatch = GroupedGemmDispatch::from_token_counts(&counts, 512);

        // num_n_tiles = 512/64 = 8
        // expert 0: 64 tokens → 1 m-tile × 8 n-tiles = 8
        // expert 1: 0 tokens → skipped = 0
        // expert 2: 128 tokens → 2 m-tiles × 8 n-tiles = 16
        // total = 24
        assert_eq!(dispatch.total_tiles, 24);
        assert_eq!(dispatch.threads_per_threadgroup, 32);
    }

    #[test]
    fn test_dispatch_from_token_counts_non_aligned_m() {
        let counts = vec![65]; // ceil(65/64) = 2 m-tiles
        let dispatch = GroupedGemmDispatch::from_token_counts(&counts, 64);
        // 2 m-tiles × 1 n-tile = 2
        assert_eq!(dispatch.total_tiles, 2);
    }

    #[test]
    fn test_config_max_tiles_upper_bound() {
        let config = MppGroupedGemmConfig::new(128, 8, 2048, 512, 2);
        // max_tiles = ceil(128/64) * ceil(512/64) * 8 = 2 * 8 * 8 = 128
        assert_eq!(config.max_tiles(), 128);
    }

    #[test]
    fn test_kernel_name_selects_dtype() {
        let mut config = MppGroupedGemmConfig::new(64, 8, 2048, 512, 2);
        assert_eq!(kernel_name(&config), "mpp_grouped_gemm_forward_f16");

        config.use_fp16 = false;
        assert_eq!(kernel_name(&config), "mpp_grouped_gemm_forward_f32");
    }
}
