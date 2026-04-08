//! Metal 4 / MPP kernel backend.
//!
//! [`Metal4Backend`] implements [`KernelBackend`] by delegating every operation
//! to the [`Metal3Backend`] fallback. Tasks 11–18 will replace individual
//! fallback calls with MPP kernel dispatch using the allocator pool and
//! residency manager stored in this struct.

use std::sync::Arc;

use half::f16;

use crate::{
    backend::{
        AdamWDescriptor, BackendCaps, GemmDescriptor, GroupedGemmDescriptor, KernelBackend,
        MoeExpertDescriptor, QuantizedGemmDescriptor,
    },
    buffer::{AsMetalBuffer, MetalBuffer},
    context::MetalContext,
    error::Result,
    kernels::{
        flash_attention::{FlashAttentionConfig, FlashAttentionOutput},
        fused_cross_entropy::{FusedCrossEntropyConfig, FusedCrossEntropyOutput},
        fused_distill::{DistillLossType, FusedDistillConfig, FusedDistillOutput},
        fused_lora::{FusedLoraConfig, FusedLoraOutput},
        fused_norm_lora::{FusedNormLoraConfig, FusedNormLoraOutput},
        fused_rope::FusedRoPEConfig,
        fused_swiglu::{FusedMLPOutput, FusedSwiGLUConfig, FusedSwiGLUOutput},
        fused_training::BatchedCommandBuffer,
        moe::{MoeConfig, MoeRouting},
    },
    metal3_backend::Metal3Backend,
    metal4::{allocator_pool::CommandAllocatorPool, residency::ResidencyManager},
};

// ============================================================================
// Metal4Backend
// ============================================================================

/// Metal 4 / MPP kernel backend for Apple M5+ (Apple10, NAX cores).
///
/// Holds the [`CommandAllocatorPool`] and [`ResidencyManager`] needed for MPP
/// kernel dispatch. Currently delegates all operations to [`Metal3Backend`];
/// Tasks 11–18 will replace individual fallback calls with MPP-accelerated
/// implementations using these resources.
///
/// # Runtime activation
///
/// [`Metal4Backend`] is constructed only when `has_nax()` returns `true`
/// (M5+ hardware) and the Metal 4 library is present in the pipeline cache.
/// On M4 and earlier it is never instantiated; `KernelDispatch` holds
/// `metal4: None` and routes everything to Metal 3.
pub struct Metal4Backend {
    ctx: Arc<MetalContext>,
    caps: BackendCaps,
    /// Triple-buffered command allocator pool — ready for MPP kernel encoding.
    pub(crate) pool: Arc<CommandAllocatorPool>,
    /// Residency set manager — tracks buffers that must be GPU-visible.
    pub(crate) residency: Arc<ResidencyManager>,
    /// Metal 3 fallback — receives every call until MPP kernels replace them.
    fallback: Metal3Backend,
}

impl Metal4Backend {
    /// Construct a new Metal 4 backend from the shared context.
    ///
    /// Creates a triple-buffered [`CommandAllocatorPool`] and a
    /// [`ResidencyManager`] backed by the device. The Metal 3 fallback is
    /// initialised so all 16 trait methods work immediately; they are replaced
    /// method-by-method in Tasks 11–18.
    ///
    /// # Note on `attach_to_queue`
    ///
    /// `ResidencyManager::attach_to_queue` requires an `MTL4CommandQueue`
    /// reference, which `MetalContext` does not yet expose. Attachment will be
    /// performed by the first MPP kernel dispatch in Task 11, when a typed
    /// Metal 4 queue is available.
    pub fn new(ctx: Arc<MetalContext>) -> Result<Self> {
        let pool = CommandAllocatorPool::new(ctx.device(), 3)?;
        let residency = ResidencyManager::new(ctx.device())?;
        let fallback = Metal3Backend::new(ctx.clone());
        Ok(Self {
            caps: BackendCaps::metal4(),
            ctx,
            pool,
            residency,
            fallback,
        })
    }

    /// Return a reference to the shared Metal context.
    pub fn ctx(&self) -> &Arc<MetalContext> {
        &self.ctx
    }

    /// Return a reference to the command allocator pool.
    pub fn pool(&self) -> &Arc<CommandAllocatorPool> {
        &self.pool
    }

    /// Return a reference to the residency manager.
    pub fn residency(&self) -> &Arc<ResidencyManager> {
        &self.residency
    }
}

// ============================================================================
// KernelBackend impl — full delegation to fallback
// ============================================================================

impl KernelBackend for Metal4Backend {
    // ---- Capabilities -------------------------------------------------------

    fn caps(&self) -> &BackendCaps {
        &self.caps
    }

    // ---- GEMM family --------------------------------------------------------

    fn gemm(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &GemmDescriptor,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        c_or_d: &dyn AsMetalBuffer,
    ) -> Result<()> {
        // Task 11: replace with MPP GEMM dispatch for M >= 2, K % 32 == 0.
        self.fallback.gemm(ctx, desc, a, b, c_or_d)
    }

    fn quantized_gemm(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &QuantizedGemmDescriptor,
        x: &dyn AsMetalBuffer,
        w_q: &dyn AsMetalBuffer,
        scales: &dyn AsMetalBuffer,
        biases: Option<&dyn AsMetalBuffer>,
        output: &dyn AsMetalBuffer,
    ) -> Result<()> {
        // Task 12: replace with MPP quantized GEMM (NAX FP4/FP8 path).
        self.fallback.quantized_gemm(ctx, desc, x, w_q, scales, biases, output)
    }

    fn dw_gemm_accum(
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
        // Metal 4 does not advertise has_dw_gemm; dispatch layer routes this to
        // Metal 3 directly. Fallback here for completeness.
        self.fallback.dw_gemm_accum(batch, a, b, c, m, n, k, alpha, beta)
    }

    fn grouped_gemm(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &GroupedGemmDescriptor,
        x: &MetalBuffer<f32>,
        w: &MetalBuffer<f32>,
        expert_offsets: &MetalBuffer<u32>,
        gather_indices: &MetalBuffer<u32>,
        scatter_indices: &MetalBuffer<u32>,
        topk_weights: &MetalBuffer<f32>,
    ) -> Result<MetalBuffer<f32>> {
        // Metal 4 does not advertise has_grouped_gemm; dispatch layer routes
        // this to Metal 3 directly. Fallback here for completeness.
        self.fallback.grouped_gemm(
            ctx,
            desc,
            x,
            w,
            expert_offsets,
            gather_indices,
            scatter_indices,
            topk_weights,
        )
    }

    // ---- Attention ----------------------------------------------------------

    fn flash_attention_forward(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FlashAttentionConfig,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
    ) -> Result<FlashAttentionOutput> {
        // Task 13: replace with MPP flash attention (has_mpp_flash_attention).
        // Metal 4 does NOT advertise has_flash_attention; MPP path is separate.
        self.fallback.flash_attention_forward(ctx, config, queries, keys, values)
    }

    fn flash_attention_backward(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FlashAttentionConfig,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
        output: &MetalBuffer<f16>,
        d_output: &MetalBuffer<f16>,
        logsumexp: &MetalBuffer<f32>,
    ) -> Result<(MetalBuffer<f16>, MetalBuffer<f16>, MetalBuffer<f16>)> {
        // Metal 4 has no backward flash attention path yet; route to Metal 3.
        self.fallback
            .flash_attention_backward(ctx, config, queries, keys, values, output, d_output, logsumexp)
    }

    // ---- Fused linear operations --------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn fused_swiglu(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedSwiGLUConfig,
        input: &MetalBuffer<f32>,
        gate_weight: &MetalBuffer<f32>,
        up_weight: &MetalBuffer<f32>,
        gate_lora_a: Option<&MetalBuffer<f32>>,
        gate_lora_b: Option<&MetalBuffer<f32>>,
        up_lora_a: Option<&MetalBuffer<f32>>,
        up_lora_b: Option<&MetalBuffer<f32>>,
    ) -> Result<FusedSwiGLUOutput> {
        // Metal 4 does not advertise has_swiglu; dispatch layer routes to Metal 3.
        self.fallback.fused_swiglu(
            ctx,
            config,
            input,
            gate_weight,
            up_weight,
            gate_lora_a,
            gate_lora_b,
            up_lora_a,
            up_lora_b,
        )
    }

    fn fused_mlp(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedSwiGLUConfig,
        input: &MetalBuffer<f32>,
        gate_weight: &MetalBuffer<f32>,
        up_weight: &MetalBuffer<f32>,
        down_weight: &MetalBuffer<f32>,
    ) -> Result<FusedMLPOutput> {
        self.fallback.fused_mlp(ctx, config, input, gate_weight, up_weight, down_weight)
    }

    fn fused_norm_lora(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedNormLoraConfig,
        input: &MetalBuffer<f32>,
        gamma: &MetalBuffer<f32>,
        weight: &MetalBuffer<f32>,
        lora_a: &MetalBuffer<f32>,
        lora_b: &MetalBuffer<f32>,
    ) -> Result<FusedNormLoraOutput> {
        self.fallback.fused_norm_lora(ctx, config, input, gamma, weight, lora_a, lora_b)
    }

    fn fused_lora_forward(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedLoraConfig,
        x: &dyn AsMetalBuffer,
        weight: &dyn AsMetalBuffer,
        lora_a: &dyn AsMetalBuffer,
        lora_b: &dyn AsMetalBuffer,
    ) -> Result<FusedLoraOutput> {
        self.fallback.fused_lora_forward(ctx, config, x, weight, lora_a, lora_b)
    }

    // ---- Training optimizers and losses -------------------------------------

    fn fused_adamw_step(
        &self,
        batch: &mut BatchedCommandBuffer,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        // Metal 4 does not advertise has_adamw; dispatch layer routes to Metal 3.
        self.fallback.fused_adamw_step(batch, desc)
    }

    fn fused_cross_entropy(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedCrossEntropyConfig,
        logits: &dyn AsMetalBuffer,
        targets: &MetalBuffer<i32>,
    ) -> Result<FusedCrossEntropyOutput> {
        // Metal 4 does not advertise has_cross_entropy; dispatch layer routes to Metal 3.
        self.fallback.fused_cross_entropy(ctx, config, logits, targets)
    }

    fn fused_rope(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedRoPEConfig,
        queries: &mut MetalBuffer<f32>,
        keys: Option<&mut MetalBuffer<f32>>,
        position_ids: Option<&MetalBuffer<i32>>,
    ) -> Result<()> {
        // Metal 4 does not advertise has_rope; dispatch layer routes to Metal 3.
        self.fallback.fused_rope(ctx, config, queries, keys, position_ids)
    }

    // ---- MoE ----------------------------------------------------------------

    fn moe_routing(
        &self,
        ctx: &Arc<MetalContext>,
        config: &MoeConfig,
        router_logits: &MetalBuffer<f32>,
    ) -> Result<MoeRouting> {
        // Metal 4 does not advertise has_moe; dispatch layer routes to Metal 3.
        self.fallback.moe_routing(ctx, config, router_logits)
    }

    fn fused_moe_expert(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &MoeExpertDescriptor<'_>,
    ) -> Result<MetalBuffer<f32>> {
        self.fallback.fused_moe_expert(ctx, desc)
    }

    // ---- Distillation -------------------------------------------------------

    fn fused_distill_loss(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedDistillConfig,
        teacher_logits: &dyn AsMetalBuffer,
        student_logits: &dyn AsMetalBuffer,
        loss_type: DistillLossType,
    ) -> Result<FusedDistillOutput> {
        // Metal 4 does not advertise has_distill; dispatch layer routes to Metal 3.
        self.fallback
            .fused_distill_loss(ctx, config, teacher_logits, student_logits, loss_type)
    }
}
