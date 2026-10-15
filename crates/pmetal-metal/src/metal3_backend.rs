//! Metal 3 kernel backend — thin delegation layer over existing kernel structs.
//!
//! [`Metal3Backend`] implements [`KernelBackend`] by forwarding every method to
//! the corresponding existing Metal 3 kernel struct. No logic lives here; this
//! file is pure glue.
//!
//! # Resolved stubs
//!
//! All `todo!()` stubs have been closed:
//!
//! - **`quantized_gemm`**: returns `Err(InvalidConfig)` — Metal 3 has no quantized
//!   GEMM path (`BackendCaps::metal3()` reports `has_quantized_gemm: false`).
//!
//! - **`grouped_gemm`**: returns `Err(InvalidConfig)` — [`MoeKernel`] requires
//!   `topk_ids` and `token_counts` buffers that are not present in
//!   [`GroupedGemmDescriptor`], making the adapter impossible without routing
//!   through the full `MoeKernel::route()` path.
//!
//! - **`fused_moe_expert`**: fully wired via [`MetalBuffer::reinterpret`] —
//!   the descriptor's `&MetalBuffer<f16>` scale/bias buffers are reinterpreted
//!   as `MetalBuffer<u16>` (same 2-byte layout, Metal does not distinguish them)
//!   and passed directly to [`FusedMoeExpert::forward_single_expert`].
//!
//! - **`fused_cross_entropy`** and **`fused_adamw_step`**: wired directly,
//!   with param-info extraction from the `AdamWDescriptor` for AdamW.

use std::sync::Arc;

use crate::{
    backend::{
        AdamWDescriptor, BackendCaps, GroupedGemmDescriptor, KernelBackend, MoeExpertDescriptor,
        QuantizedGemmDescriptor,
    },
    buffer::{AsMetalBuffer, BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
    kernels::{
        fused_cross_entropy::{
            FusedCrossEntropy, FusedCrossEntropyConfig, FusedCrossEntropyOutput,
        },
        fused_moe::{ExpertWeightBuffers, FusedMoeExpert},
        fused_training::{BatchedCommandBuffer, FusedAdamW},
        moe::{MoeConfig, MoeKernel, MoeRouting},
    },
};

// ============================================================================
// Metal3Backend
// ============================================================================

/// Metal 3 kernel backend.
///
/// A thin delegation layer over the existing Metal 3 kernel structs.
/// Implements [`KernelBackend`] so that [`KernelDispatch`](crate::KernelDispatch) can route
/// operations here without knowing the concrete kernel types.
///
/// Backends are `Send + Sync` — each method constructs lightweight kernel
/// structs on the fly from the shared `Arc<MetalContext>`.
pub struct Metal3Backend {
    ctx: Arc<MetalContext>,
    caps: BackendCaps,
}

impl Metal3Backend {
    /// Create a new Metal 3 backend from an existing context.
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        Self {
            ctx,
            caps: BackendCaps::metal3(),
        }
    }

    /// Return a reference to the shared Metal context.
    pub fn ctx(&self) -> &Arc<MetalContext> {
        &self.ctx
    }
}

// ============================================================================
// KernelBackend impl
// ============================================================================

impl KernelBackend for Metal3Backend {
    // ---- Capabilities -------------------------------------------------------

    fn caps(&self) -> &BackendCaps {
        &self.caps
    }

    // ---- Quantized and grouped GEMM -----------------------------------------

    /// Quantized GEMM — not available on Metal 3.
    ///
    /// `BackendCaps::metal3()` sets `has_quantized_gemm: false`; the dispatch
    /// layer must never route here. Calling this is a programming error.
    fn quantized_gemm(
        &self,
        _ctx: &Arc<MetalContext>,
        _desc: &QuantizedGemmDescriptor,
        _x: &dyn AsMetalBuffer,
        _w_q: &dyn AsMetalBuffer,
        _scales: &dyn AsMetalBuffer,
        _biases: Option<&dyn AsMetalBuffer>,
        _output: &dyn AsMetalBuffer,
    ) -> Result<()> {
        // Metal 3 has no quantized GEMM kernel. BackendCaps::metal3() reports
        // has_quantized_gemm: false, so the dispatch layer must never route here.
        // Return a proper error instead of panicking so the process can recover.
        Err(MetalError::InvalidConfig(
            "quantized_gemm is not available on the Metal 3 backend \
             (BackendCaps::metal3() has_quantized_gemm=false); \
             route quantized GEMM to Metal 4 or the MLX compute graph path"
                .into(),
        ))
    }

    /// Grouped GEMM — requires an adapter between [`GroupedGemmDescriptor`] and
    /// the [`MoeKernel`] API.
    ///
    /// [`MoeKernel::forward`] takes a [`MoeConfig`] + [`MoeRouting`] bundle, but
    /// the trait provides separate flat buffers. A proper adapter (building
    /// `MoeRouting` from the descriptor's pre-sorted index buffers) is deferred
    /// to Task 3 (KernelDispatch).
    fn grouped_gemm(
        &self,
        _ctx: &Arc<MetalContext>,
        _desc: &GroupedGemmDescriptor,
        _x: &MetalBuffer<f32>,
        _w: &MetalBuffer<f32>,
        _expert_offsets: &MetalBuffer<u32>,
        _gather_indices: &MetalBuffer<u32>,
        _scatter_indices: &MetalBuffer<u32>,
        _topk_weights: &MetalBuffer<f32>,
    ) -> Result<MetalBuffer<f32>> {
        // MoeKernel::forward() requires a MoeRouting, which bundles topk_ids and
        // token_counts in addition to the index buffers supplied here.
        // GroupedGemmDescriptor does not carry those fields — they are intermediate
        // routing state produced by MoeKernel::route() and not preserved in the
        // descriptor. Building the adapter is impossible without a full re-route,
        // which would duplicate work already done by the caller. Return a proper
        // error; the dispatch layer (KernelDispatch) should call moe_routing() +
        // MoeKernel::forward() directly rather than going through grouped_gemm().
        Err(MetalError::InvalidConfig(
            "grouped_gemm on Metal 3 cannot be bridged to MoeKernel::forward(): \
             MoeRouting requires topk_ids and token_counts buffers that are absent \
             from GroupedGemmDescriptor; use moe_routing() + MoeKernel::forward() \
             directly from the dispatch layer"
                .into(),
        ))
    }

    // ---- Training optimizers and losses -------------------------------------

    /// Fused AdamW step via [`FusedAdamW::queue_update`].
    ///
    /// [`FusedAdamW::new`] needs `&[usize]` param sizes to pre-compute the
    /// kernel grid (`max_param_size`, `num_params`). [`AdamWDescriptor`] does
    /// not carry that slice directly, but `param_info` is a `MetalBuffer<ParamInfo>`
    /// where each `ParamInfo::size` holds the per-parameter element count.
    /// We read the slice, extract sizes, build a temporary `FusedAdamW`, and
    /// delegate to `queue_update`.
    fn fused_adamw_step(
        &self,
        batch: &mut BatchedCommandBuffer,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        let param_sizes: Vec<usize> = desc
            .param_info
            .as_slice()
            .iter()
            .map(|p| p.size as usize)
            .collect();
        let adamw = FusedAdamW::new(self.ctx.clone(), &param_sizes);
        adamw.queue_update(
            batch,
            desc.params,
            desc.grads,
            desc.m,
            desc.v,
            desc.param_info,
            &[desc.config.scalars()?],
        )
    }

    /// Standalone AdamW step: creates a temporary [`BatchedCommandBuffer`],
    /// queues the update, and executes it synchronously.
    ///
    /// This satisfies the [`KernelBackend::fused_adamw_step_standalone`] contract
    /// for callers (e.g. [`Metal4Backend`] dispatch glue) that need a
    /// self-contained call without an externally owned batch.
    ///
    /// [`Metal4Backend`]: crate::metal4::Metal4Backend
    fn fused_adamw_step_standalone(
        &self,
        _ctx: &Arc<MetalContext>,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        let mut batch = BatchedCommandBuffer::new(self.ctx.clone())?;
        self.fused_adamw_step(&mut batch, desc)?;
        batch.execute()
    }

    /// Fused cross-entropy via `FusedCrossEntropy::forward_dyn`.
    ///
    /// The trait provides `logits: &dyn AsMetalBuffer` to support both f32
    /// and f16 logits without an extra type parameter. [`FusedCrossEntropy`]
    /// exposes `forward_dyn` which accepts the same type-erased buffer.
    fn fused_cross_entropy(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedCrossEntropyConfig,
        logits: &dyn AsMetalBuffer,
        targets: &MetalBuffer<i32>,
    ) -> Result<FusedCrossEntropyOutput> {
        let kernel = FusedCrossEntropy::new(ctx.clone(), config.clone())?;
        kernel.forward_dyn(logits, targets)
    }

    // ---- MoE ----------------------------------------------------------------

    /// MoE routing via [`MoeKernel::route`].
    fn moe_routing(
        &self,
        ctx: &Arc<MetalContext>,
        config: &MoeConfig,
        router_logits: &MetalBuffer<f32>,
    ) -> Result<MoeRouting> {
        let kernel = MoeKernel::new(ctx.clone(), config.clone())?;
        kernel.route(router_logits)
    }

    /// Fused MoE expert forward via [`FusedMoeExpert::forward_single_expert`].
    ///
    /// [`ExpertWeightBuffers`] stores scales/biases as `MetalBuffer<u16>` (raw
    /// 16-bit values), while [`MoeExpertDescriptor`] carries them as
    /// `&MetalBuffer<f16>` (typed half-precision). In Metal both are 2-byte
    /// values and the buffer layout is identical — no data movement is required.
    /// [`MetalBuffer::reinterpret`] aliases the same `MTLBuffer` allocation with
    /// the new Rust element type at zero cost.
    fn fused_moe_expert(
        &self,
        _ctx: &Arc<MetalContext>,
        desc: &MoeExpertDescriptor<'_>,
    ) -> Result<MetalBuffer<f32>> {
        let kernel = FusedMoeExpert::new(self.ctx.clone(), desc.expert_config.clone())?;

        // Reinterpret f16 scale/bias buffers as u16 — same 2-byte layout, Metal
        // does not distinguish half from uint16 at the buffer level.
        let weights = ExpertWeightBuffers {
            gate_weights: desc.gate_weight.clone(),
            gate_scales: desc.gate_scales.reinterpret::<u16>(),
            gate_biases: desc.gate_biases.reinterpret::<u16>(),
            up_weights: desc.up_weight.clone(),
            up_scales: desc.up_scales.reinterpret::<u16>(),
            up_biases: desc.up_biases.reinterpret::<u16>(),
            down_weights: desc.down_weight.clone(),
            down_scales: desc.down_scales.reinterpret::<u16>(),
            down_biases: desc.down_biases.reinterpret::<u16>(),
        };

        let hidden_dim = desc.expert_config.hidden_dim as usize;
        let intermediate_dim = desc.expert_config.intermediate_dim as usize;

        let output =
            MetalBuffer::<f32>::new(&self.ctx, desc.num_tokens * hidden_dim, BufferUsage::Shared)?;
        let intermediate = MetalBuffer::<f32>::new(
            &self.ctx,
            desc.num_tokens * intermediate_dim,
            BufferUsage::Shared,
        )?;

        kernel.forward_single_expert(desc.input, &weights, &output, &intermediate)?;

        Ok(output)
    }
}
