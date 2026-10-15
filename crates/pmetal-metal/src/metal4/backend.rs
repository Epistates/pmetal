//! Metal 4 / MPP kernel backend.
//!
//! [`Metal4Backend`] implements [`KernelBackend`] using MPP kernel dispatch
//! structs for operations that have NAX-accelerated implementations. Methods
//! that cannot be cleanly translated to the MPP API retain the [`Metal3Backend`]
//! fallback; each such delegation is annotated with the reason.

use std::sync::Arc;

use crate::{
    backend::{
        AdamWDescriptor, BackendCaps, GroupedGemmDescriptor, KernelBackend, MoeExpertDescriptor,
        QuantizedGemmDescriptor,
    },
    buffer::{AsMetalBuffer, BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
    kernels::fused_moe::ExpertBits,
    kernels::{
        fused_cross_entropy::{FusedCrossEntropyConfig, FusedCrossEntropyOutput},
        fused_training::BatchedCommandBuffer,
        moe::{MoeConfig, MoeRouting},
        mpp_fused_cross_entropy::{MppFusedCrossEntropy, MppFusedCrossEntropyConfig},
        mpp_fused_moe::{MppFusedMoEQuant, MppFusedMoEQuantConfig, MppGroupedGemmTileCount},
        mpp_fused_training::{MppFusedAdamW, MppFusedAdamWConfig, MppParamInfo},
        mpp_grouped_gemm::{GroupedGemmDispatch, MppGroupedGemm, MppGroupedGemmConfig},
        mpp_quantized::{MppQuantizedGemm, MppQuantizedGemmConfig},
    },
    metal3_backend::Metal3Backend,
    metal4::{allocator_pool::CommandAllocatorPool, residency::ResidencyManager},
};

// ============================================================================
// Metal4Backend
// ============================================================================

/// Metal 4 / MPP kernel backend for Apple M5+ (Apple10, NAX cores).
///
/// Most operations are dispatched through the dedicated MPP kernel structs
/// defined in `crates/pmetal-metal/src/kernels/mpp_*.rs`. A small number of
/// methods retain the [`Metal3Backend`] fallback where the MPP API is
/// structurally incompatible with the trait's parameter model:
///
/// - `fused_moe_expert` (2-bit): `MppFusedMoEQuant` handles 4-bit only;
///   2-bit weights fall back to Metal 3's dequant kernels.
/// - `moe_routing`: routing kernel is Metal 3 only; no MPP variant exists yet.
///
/// Previously falling back: `fused_adamw_step` now routes through
/// `fused_adamw_step_standalone` which dispatches `MppFusedAdamW` directly.
/// Quantized GEMM is now wired to `MppQuantizedGemm`.
pub struct Metal4Backend {
    ctx: Arc<MetalContext>,
    caps: BackendCaps,
    /// Triple-buffered command allocator pool — ready for MPP kernel encoding.
    pub(crate) pool: Arc<CommandAllocatorPool>,
    /// Residency set manager — tracks buffers that must be GPU-visible.
    pub(crate) residency: Arc<ResidencyManager>,
    /// Metal 3 fallback for operations without an MPP equivalent.
    fallback: Metal3Backend,
}

impl Metal4Backend {
    /// Construct a new Metal 4 backend from the shared context.
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
// KernelBackend impl
// ============================================================================

impl KernelBackend for Metal4Backend {
    // ---- Capabilities -------------------------------------------------------

    fn caps(&self) -> &BackendCaps {
        &self.caps
    }

    // ---- Quantized and grouped GEMM -----------------------------------------

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
        let mpp_config = MppQuantizedGemmConfig {
            m: desc.m,
            n: desc.n,
            k: desc.k,
            group_size: desc.group_size,
            bits: desc.bits,
        };
        let dispatcher = MppQuantizedGemm::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            return self
                .fallback
                .quantized_gemm(ctx, desc, x, w_q, scales, biases, output);
        }

        dispatcher.execute(x, w_q, scales, biases, output)
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
        // Use the GPU tile-count kernel to compute total_tiles from expert_offsets
        // without a CPU round-trip on the full offsets buffer. The tile count
        // kernel returns a single u32 — 4 bytes vs. (E+1)*4 bytes for the full
        // prefix-sum copy.
        let tile_counter = MppGroupedGemmTileCount::new(self.ctx.clone());

        if !tile_counter.is_available() {
            return self.fallback.grouped_gemm(
                ctx,
                desc,
                x,
                w,
                expert_offsets,
                gather_indices,
                scatter_indices,
                topk_weights,
            );
        }

        const BLOCK_M: usize = 64;
        const BLOCK_N: usize = 64;

        let total_tiles = tile_counter.compute(
            expert_offsets,
            desc.num_experts,
            desc.intermediate_size,
            BLOCK_M,
            BLOCK_N,
        )?;

        if total_tiles == 0 {
            // All experts have zero tokens — return empty output.
            return MetalBuffer::<f32>::new(
                ctx,
                desc.total_tokens * desc.intermediate_size,
                BufferUsage::Shared,
            );
        }

        let mpp_config = MppGroupedGemmConfig {
            total_tokens: desc.total_tokens,
            num_experts: desc.num_experts,
            hidden_size: desc.hidden_size,
            intermediate: desc.intermediate_size,
            topk: desc.topk,
            use_fp16: desc.use_fp16,
        };
        let dispatcher = MppGroupedGemm::new(self.ctx.clone(), mpp_config);

        let dispatch = GroupedGemmDispatch {
            total_tiles,
            threads_per_threadgroup: 32,
        };

        let y = MetalBuffer::<f32>::new(
            ctx,
            desc.total_tokens * desc.intermediate_size,
            BufferUsage::Shared,
        )?;
        dispatcher.execute(
            x,
            w,
            &y,
            expert_offsets,
            gather_indices,
            scatter_indices,
            topk_weights,
            dispatch,
        )?;
        Ok(y)
    }

    // ---- Training optimizers and losses -------------------------------------

    fn fused_adamw_step(
        &self,
        _batch: &mut BatchedCommandBuffer,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        // MppFusedAdamW creates its own command buffer and cannot encode into
        // an existing BatchedCommandBuffer. Delegate to the standalone path,
        // which executes synchronously and satisfies the caller's intent.
        self.fused_adamw_step_standalone(&self.ctx.clone(), desc)
    }

    fn fused_adamw_step_standalone(
        &self,
        _ctx: &Arc<MetalContext>,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        // Derive num_params and max_param_elements from the param_info buffer.
        let param_infos = desc.param_info.as_slice();
        let num_params = param_infos.len();
        let max_param_elements = param_infos
            .iter()
            .map(|p| p.size as usize)
            .max()
            .unwrap_or(0);

        let mpp_config = MppFusedAdamWConfig {
            num_params,
            max_param_elements,
            use_fp16: false,
        };

        // The MPP kernel has one set of scalars for every parameter.
        if let Some(p) = param_infos.iter().find(|p| p.group != 0) {
            return Err(MetalError::InvalidConfig(format!(
                "AdamWDescriptor carries one parameter group, but the parameter at \
                 offset {} is in group {}",
                p.offset, p.group
            )));
        }
        // Convert ParamInfo → MppParamInfo: offset, size and moment offsets,
        // into a buffer laid out as the MPP kernel reads it.
        let mpp_param_infos: Vec<MppParamInfo> = param_infos
            .iter()
            .map(|p| MppParamInfo {
                offset: p.offset,
                size: p.size,
                m_offset: p.m_offset,
                v_offset: p.v_offset,
            })
            .collect();

        let dispatcher = MppFusedAdamW::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            // NAX not present — fall back to Metal 3 path via a temporary batch.
            return self.fallback.fused_adamw_step_standalone(_ctx, desc);
        }

        let mpp_info_buf =
            MetalBuffer::from_slice(&self.ctx, &mpp_param_infos, BufferUsage::Shared)?;
        dispatcher.execute(
            desc.params,
            desc.grads,
            desc.m,
            desc.v,
            &mpp_param_infos,
            &mpp_info_buf,
            &desc.config,
        )
    }

    fn fused_cross_entropy(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedCrossEntropyConfig,
        logits: &dyn AsMetalBuffer,
        targets: &MetalBuffer<i32>,
    ) -> Result<FusedCrossEntropyOutput> {
        let mpp_config = MppFusedCrossEntropyConfig {
            num_tokens: config.num_tokens,
            vocab_size: config.vocab_size,
            ignore_index: config.ignore_index,
            use_fp16: config.use_fp16,
            forward_only: false,
        };
        let dispatcher = MppFusedCrossEntropy::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            return self
                .fallback
                .fused_cross_entropy(ctx, config, logits, targets);
        }

        let losses = MetalBuffer::<f32>::new(&self.ctx, config.num_tokens, BufferUsage::Shared)?;
        let logsumexp = MetalBuffer::<f32>::new(&self.ctx, config.num_tokens, BufferUsage::Shared)?;

        // MPP cross-entropy fwd+bwd kernel writes gradients into grad_logits.
        // For the forward-only output shape (losses + logsumexp), we pass logsumexp
        // as the grad_logits placeholder and a single-element loss accumulator.
        let loss_accum = MetalBuffer::<f32>::new(&self.ctx, 1, BufferUsage::Shared)?;
        dispatcher.execute(logits, targets, &logsumexp, &loss_accum)?;

        Ok(FusedCrossEntropyOutput { losses, logsumexp })
    }

    // ---- MoE ----------------------------------------------------------------

    fn moe_routing(
        &self,
        ctx: &Arc<MetalContext>,
        config: &MoeConfig,
        router_logits: &MetalBuffer<f32>,
    ) -> Result<MoeRouting> {
        // No MPP MoE routing kernel exists yet.
        self.fallback.moe_routing(ctx, config, router_logits)
    }

    fn fused_moe_expert(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &MoeExpertDescriptor<'_>,
    ) -> Result<MetalBuffer<f32>> {
        // Dispatch to the quantized MPP expert kernel for 4-bit weights.
        // 2-bit weights fall back to Metal 3 (MPP matmul2d needs fp16 inputs;
        // 2-bit requires the scalar qdot path).
        if desc.expert_config.bits != ExpertBits::Four {
            return self.fallback.fused_moe_expert(ctx, desc);
        }

        let mpp_config = MppFusedMoEQuantConfig::new(
            desc.num_tokens,
            desc.expert_config.hidden_dim as usize,
            desc.expert_config.intermediate_dim as usize,
            desc.expert_config.group_size as usize,
            4,
        );
        let dispatcher = MppFusedMoEQuant::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            return self.fallback.fused_moe_expert(ctx, desc);
        }

        // Intermediate buffer: SwiGLU output [num_tokens, intermediate_dim].
        let act_out = MetalBuffer::<f32>::new(
            ctx,
            desc.num_tokens * desc.expert_config.intermediate_dim as usize,
            BufferUsage::Shared,
        )?;

        dispatcher.execute_gate_up(
            desc.input,
            desc.gate_weight,
            desc.gate_scales,
            desc.gate_biases,
            desc.up_weight,
            desc.up_scales,
            desc.up_biases,
            &act_out,
        )?;

        // Down projection: [num_tokens, intermediate] @ down_W^T → [num_tokens, hidden]
        let output = MetalBuffer::<f32>::new(
            ctx,
            desc.num_tokens * desc.expert_config.hidden_dim as usize,
            BufferUsage::Shared,
        )?;

        dispatcher.execute_down(
            &act_out,
            desc.down_weight,
            desc.down_scales,
            desc.down_biases,
            &output,
        )?;

        Ok(output)
    }
}
