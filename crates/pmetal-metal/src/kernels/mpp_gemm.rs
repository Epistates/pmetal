#![allow(unsafe_code)]

//! Metal 4 / MPP GEMM dispatch.
//!
//! Provides hardware-accelerated GEMM via Metal Performance Primitives
//! on M5+ (Apple10) GPUs with NAX cores.
//!
//! Falls back to standard Metal 3 kernels on older hardware.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};
use serde::{Deserialize, Serialize};

use crate::{
    buffer::{AsMetalBuffer, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
    pipeline::FunctionConstant,
    tuna::{MppGemmTuneRequest, MppGemmTunedConfig},
};

/// MPP GEMM kernel variants derived from the guide's 32x32 per-simdgroup
/// starting point on Apple10/M5 hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum MppGemmKernelVariant {
    /// Single simdgroup, 32x32 output tile.
    Sg1_32x32,
    /// Two simdgroups, 64x32 output tile.
    Sg2_64x32,
    /// Two simdgroups, 32x64 output tile.
    Sg2_32x64,
    /// Four simdgroups, 64x64 output tile.
    #[default]
    Sg4_64x64,
}

impl MppGemmKernelVariant {
    /// Return `(BM, BN, num_simdgroups)` for the variant.
    pub const fn tile_config(self) -> (usize, usize, usize) {
        match self {
            Self::Sg1_32x32 => (32, 32, 1),
            Self::Sg2_64x32 => (64, 32, 2),
            Self::Sg2_32x64 => (32, 64, 2),
            Self::Sg4_64x64 => (64, 64, 4),
        }
    }

    fn kernel_suffix(self) -> &'static str {
        match self {
            Self::Sg1_32x32 => "sg1_32x32",
            Self::Sg2_64x32 => "sg2_64x32",
            Self::Sg2_32x64 => "sg2_32x64",
            Self::Sg4_64x64 => "sg4_64x64",
        }
    }
}

/// Configuration for MPP GEMM.
#[derive(Debug, Clone)]
pub struct MppGemmConfig {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,

    /// Scalar multiplier for the matmul result.
    pub alpha: f32,
    /// Scalar multiplier for existing C (0 = overwrite, 1 = accumulate).
    pub beta: f32,

    /// Batch count (for batched GEMM).
    pub batch_size: usize,

    /// Use Morton ordering for threadgroup walk.
    pub use_morton: bool,

    /// Kernel tile/simdgroup variant.
    pub kernel_variant: MppGemmKernelVariant,

    /// Auto-tune the Morton walk order on M5/NAX hardware.
    ///
    /// This is a no-op on M1-M4, where Metal 4 / MPP is unavailable and the
    /// existing Metal 3 paths remain in use.
    pub auto_tune_morton: bool,

    /// Auto-tune the MPP kernel variant on M5/NAX hardware.
    pub auto_tune_variant: bool,

    /// Use fp16 (true) or fp32 (false).
    pub use_fp16: bool,
}

impl MppGemmConfig {
    /// Create a new MPP GEMM config for C = A[M,K] @ B[N,K]^T.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
            batch_size: 1,
            use_morton: true,
            kernel_variant: MppGemmKernelVariant::default(),
            auto_tune_morton: true,
            auto_tune_variant: true,
            use_fp16: true,
        }
    }
}

/// MPP GEMM kernel parameters (must match Metal struct layout).
#[repr(C)]
struct MppGemmParams {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    num_tiles_m: u32,
    num_tiles_n: u32,
}

// `MppGemmParams` in `metal4/mpp_gemm.metal`.
crate::msl_layout::msl_layout!(MppGemmParams, size = 28, {
    m: 0,
    n: 4,
    k: 8,
    alpha: 12,
    beta: 16,
    num_tiles_m: 20,
    num_tiles_n: 24,
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchGeometry {
    bm: usize,
    bn: usize,
    num_simdgroups: usize,
    num_tiles_m: usize,
    num_tiles_n: usize,
    total_tiles: usize,
    threads_per_threadgroup: usize,
}

fn dispatch_geometry(config: &MppGemmConfig, variant: MppGemmKernelVariant) -> DispatchGeometry {
    let (bm, bn, num_simdgroups) = variant.tile_config();
    let num_tiles_m = config.m.div_ceil(bm);
    let num_tiles_n = config.n.div_ceil(bn);

    DispatchGeometry {
        bm,
        bn,
        num_simdgroups,
        num_tiles_m,
        num_tiles_n,
        total_tiles: num_tiles_m * num_tiles_n,
        threads_per_threadgroup: num_simdgroups * 32,
    }
}

fn select_kernel_name(config: &MppGemmConfig, variant: MppGemmKernelVariant) -> Result<String> {
    if config.beta != 0.0 {
        if config.use_fp16 {
            Ok(format!(
                "mpp_gemm_accumulate_f16_{}",
                variant.kernel_suffix()
            ))
        } else {
            Err(MetalError::ExecutionFailed(
                "MPP GEMM accumulate not available for fp32 (use fp16)".to_string(),
            ))
        }
    } else if config.use_fp16 {
        Ok(format!("mpp_gemm_nn_f16_{}", variant.kernel_suffix()))
    } else {
        Ok(format!("mpp_gemm_nn_f32_{}", variant.kernel_suffix()))
    }
}

fn output_tile_alignment(config: &MppGemmConfig, geometry: &DispatchGeometry) -> (bool, bool) {
    (config.m % geometry.bm == 0, config.n % geometry.bn == 0)
}

fn build_function_constants(
    config: &MppGemmConfig,
    geometry: &DispatchGeometry,
    use_morton: bool,
) -> HashMap<u64, FunctionConstant> {
    let (m_aligned, n_aligned) = output_tile_alignment(config, geometry);
    let mut constants = HashMap::new();
    constants.insert(0u64, FunctionConstant::Bool(use_morton));
    constants.insert(1u64, FunctionConstant::Bool(m_aligned));
    constants.insert(2u64, FunctionConstant::Bool(n_aligned));
    constants
}

fn should_auto_tune_morton(
    config: &MppGemmConfig,
    has_nax: bool,
    has_metal4_library: bool,
) -> bool {
    (config.auto_tune_morton || config.auto_tune_variant) && has_nax && has_metal4_library
}

fn select_dispatch_choice(
    config: &MppGemmConfig,
    tuned: Option<MppGemmTunedConfig>,
) -> MppGemmTunedConfig {
    let mut choice = MppGemmTunedConfig {
        variant: config.kernel_variant,
        use_morton: config.use_morton,
    };

    if let Some(tuned) = tuned {
        if config.auto_tune_variant {
            choice.variant = tuned.variant;
        }
        if config.auto_tune_morton {
            choice.use_morton = tuned.use_morton;
        }
    }

    choice
}

/// MPP GEMM dispatcher.
///
/// Checks NAX availability and dispatches to Metal 4 MPP kernels when possible,
/// falling back to Metal 3 kernels otherwise.
pub struct MppGemm {
    ctx: Arc<MetalContext>,
    config: MppGemmConfig,
}

impl MppGemm {
    /// Create a new MPP GEMM dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppGemmConfig) -> Self {
        Self { ctx, config }
    }

    /// Check if MPP GEMM is available on this device (requires M5+ with NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    fn resolve_dispatch_choice(&self) -> Result<MppGemmTunedConfig> {
        let has_metal4_library = self.ctx.pipeline_cache().metal4_library().is_some();
        if !should_auto_tune_morton(
            &self.config,
            self.ctx.properties().has_nax(),
            has_metal4_library,
        ) {
            return Ok(select_dispatch_choice(&self.config, None));
        }

        let tuned = self.ctx.tuner().tune_mpp_gemm(
            &self.ctx,
            MppGemmTuneRequest {
                m: self.config.m,
                n: self.config.n,
                k: self.config.k,
                batch_size: self.config.batch_size,
                use_fp16: self.config.use_fp16,
                accumulate: self.config.beta != 0.0,
            },
        )?;

        Ok(select_dispatch_choice(&self.config, Some(tuned)))
    }

    /// Execute MPP GEMM: D = alpha * A @ B^T + beta * C
    ///
    /// Buffer element type is determined by `config.use_fp16`:
    /// - `use_fp16 = true`: buffers must contain fp16 data
    /// - `use_fp16 = false`: buffers must contain fp32 data
    ///
    /// Metal buffers are untyped at the API level — the caller is responsible
    /// for ensuring the buffer data matches the kernel's expected precision.
    ///
    /// For the overwrite case (beta=0): `c_or_d` is the output buffer D.
    /// For the accumulate case (beta!=0): `c_or_d` is used as both C (source)
    /// and D (destination), i.e. in-place accumulation.
    pub fn execute(
        &self,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        c_or_d: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let command_buffer = self.execute_async(a, b, c_or_d)?;
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            let err_str: String = error.to_string();
            return Err(MetalError::ExecutionFailed(err_str));
        }

        Ok(())
    }

    /// Execute MPP GEMM asynchronously and return the submitted command buffer.
    ///
    /// This allows callers to overlap GEMM execution with other GPU work and
    /// defer synchronization until a later point in the pipeline.
    pub fn execute_async(
        &self,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        c_or_d: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP GEMM not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let is_accumulate = self.config.beta != 0.0;
        let dispatch_choice = self.resolve_dispatch_choice()?;
        let kernel_name = select_kernel_name(&self.config, dispatch_choice.variant)?;
        let geometry = dispatch_geometry(&self.config, dispatch_choice.variant);

        // Function constants specialize walk order and aligned full-tile paths.
        let constants =
            build_function_constants(&self.config, &geometry, dispatch_choice.use_morton);

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_metal4_pipeline(self.ctx.device(), &kernel_name, &constants)?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        let params = MppGemmParams {
            m: self.config.m as u32,
            n: self.config.n as u32,
            k: self.config.k as u32,
            alpha: self.config.alpha,
            beta: self.config.beta,
            num_tiles_m: geometry.num_tiles_m as u32,
            num_tiles_n: geometry.num_tiles_n as u32,
        };

        unsafe {
            if is_accumulate {
                // mpp_gemm_accumulate_f16: A=0, B=1, C=2, D=3, params=4
                // c_or_d serves as both C (read) and D (write) for in-place accumulate
                encoder.setBuffer_offset_atIndex(Some(a.as_metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b.as_metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(c_or_d.as_metal_buffer()), 0, 2); // C
                encoder.setBuffer_offset_atIndex(Some(c_or_d.as_metal_buffer()), 0, 3); // D (alias)

                let params_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
            } else {
                // mpp_gemm_nn_{f16,f32}: A=0, B=1, D=2, params=3
                encoder.setBuffer_offset_atIndex(Some(a.as_metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b.as_metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(c_or_d.as_metal_buffer()), 0, 2); // D

                let params_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);
            }
        }

        let threadgroup_size = objc2_metal::MTLSize {
            width: geometry.threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        let grid_size = objc2_metal::MTLSize {
            width: geometry.total_tiles,
            height: 1,
            depth: self.config.batch_size,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();

        Ok(command_buffer)
    }

    /// Convenience: Execute with typed f32 buffers (dispatches to mpp_gemm_nn_f32).
    pub fn execute_f32(
        &self,
        a: &MetalBuffer<f32>,
        b: &MetalBuffer<f32>,
        d: &MetalBuffer<f32>,
    ) -> Result<()> {
        if self.config.use_fp16 {
            return Err(MetalError::ExecutionFailed(
                "execute_f32 called but config.use_fp16 is true".to_string(),
            ));
        }
        self.execute(a, b, d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpp_gemm_config_defaults_enable_auto_tuning() {
        let config = MppGemmConfig::new(64, 64, 64);
        assert!(config.use_morton);
        assert_eq!(config.kernel_variant, MppGemmKernelVariant::Sg4_64x64);
        assert!(config.auto_tune_morton);
        assert!(config.auto_tune_variant);
        assert!(config.use_fp16);
    }

    #[test]
    fn test_dispatch_geometry_uses_variant_tile_config() {
        let config = MppGemmConfig::new(130, 129, 256);
        let geometry = dispatch_geometry(&config, MppGemmKernelVariant::Sg2_64x32);

        assert_eq!(geometry.bm, 64);
        assert_eq!(geometry.bn, 32);
        assert_eq!(geometry.num_simdgroups, 2);
        assert_eq!(geometry.num_tiles_m, 3);
        assert_eq!(geometry.num_tiles_n, 5);
        assert_eq!(geometry.total_tiles, 15);
        assert_eq!(geometry.threads_per_threadgroup, 64);
    }

    #[test]
    fn test_select_kernel_name_for_non_accumulate_paths() {
        let mut config = MppGemmConfig::new(64, 64, 64);

        assert_eq!(
            select_kernel_name(&config, MppGemmKernelVariant::Sg4_64x64).unwrap(),
            "mpp_gemm_nn_f16_sg4_64x64"
        );

        config.use_fp16 = false;
        assert_eq!(
            select_kernel_name(&config, MppGemmKernelVariant::Sg2_32x64).unwrap(),
            "mpp_gemm_nn_f32_sg2_32x64"
        );
    }

    #[test]
    fn test_select_kernel_name_rejects_fp32_accumulate() {
        let mut config = MppGemmConfig::new(64, 64, 64);
        config.use_fp16 = false;
        config.beta = 1.0;

        let err = select_kernel_name(&config, MppGemmKernelVariant::Sg4_64x64).unwrap_err();
        assert!(
            err.to_string()
                .contains("MPP GEMM accumulate not available for fp32"),
        );
    }

    #[test]
    fn test_auto_tune_requires_explicit_support() {
        let config = MppGemmConfig::new(64, 64, 64);
        assert!(should_auto_tune_morton(&config, true, true));
        assert!(!should_auto_tune_morton(&config, false, true));
        assert!(!should_auto_tune_morton(&config, true, false));

        let mut disabled = config.clone();
        disabled.auto_tune_morton = false;
        disabled.auto_tune_variant = false;
        assert!(!should_auto_tune_morton(&disabled, true, true));
    }

    #[test]
    fn test_select_dispatch_choice_prefers_tuned_choice() {
        let mut config = MppGemmConfig::new(64, 64, 64);
        config.use_morton = true;
        config.kernel_variant = MppGemmKernelVariant::Sg2_64x32;

        let fallback = select_dispatch_choice(&config, None);
        assert_eq!(fallback.variant, MppGemmKernelVariant::Sg2_64x32);
        assert!(fallback.use_morton);

        let tuned = select_dispatch_choice(
            &config,
            Some(MppGemmTunedConfig {
                variant: MppGemmKernelVariant::Sg4_64x64,
                use_morton: false,
            }),
        );
        assert_eq!(tuned.variant, MppGemmKernelVariant::Sg4_64x64);
        assert!(!tuned.use_morton);
    }

    #[test]
    fn test_output_tile_alignment_detects_exact_multiples() {
        let aligned = MppGemmConfig::new(128, 192, 256);
        let aligned_geometry = dispatch_geometry(&aligned, MppGemmKernelVariant::Sg4_64x64);
        assert_eq!(
            output_tile_alignment(&aligned, &aligned_geometry),
            (true, true)
        );

        let unaligned = MppGemmConfig::new(129, 190, 256);
        let unaligned_geometry = dispatch_geometry(&unaligned, MppGemmKernelVariant::Sg4_64x64);
        assert_eq!(
            output_tile_alignment(&unaligned, &unaligned_geometry),
            (false, false)
        );
    }

    #[test]
    fn test_build_function_constants_include_alignment_specialization() {
        let config = MppGemmConfig::new(128, 64, 256);
        let geometry = dispatch_geometry(&config, MppGemmKernelVariant::Sg2_64x32);
        let constants = build_function_constants(&config, &geometry, false);

        assert!(matches!(
            constants.get(&0),
            Some(FunctionConstant::Bool(false))
        ));
        assert!(matches!(
            constants.get(&1),
            Some(FunctionConstant::Bool(true))
        ));
        assert!(matches!(
            constants.get(&2),
            Some(FunctionConstant::Bool(true))
        ));
    }

    #[test]
    fn test_variant_tile_config_matches_expected_shapes() {
        assert_eq!(MppGemmKernelVariant::Sg1_32x32.tile_config(), (32, 32, 1));
        assert_eq!(MppGemmKernelVariant::Sg2_64x32.tile_config(), (64, 32, 2));
        assert_eq!(MppGemmKernelVariant::Sg2_32x64.tile_config(), (32, 64, 2));
        assert_eq!(MppGemmKernelVariant::Sg4_64x64.tile_config(), (64, 64, 4));
    }
}
