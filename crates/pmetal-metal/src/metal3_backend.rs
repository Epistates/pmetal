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
//! - **`fused_lora_forward`** and **`fused_adamw_step`**: wired directly, with param-info extraction from
//!   the `AdamWDescriptor` for AdamW.

use std::sync::Arc;

use half::f16;

use crate::{
    backend::{AdamWDescriptor, BackendCaps, GemmDescriptor, KernelBackend},
    buffer::{AsMetalBuffer, MetalBuffer},
    context::MetalContext,
    error::Result,
    kernels::{
        dw_gemm::DwGemm,
        flash_attention::{FlashAttention, FlashAttentionConfig, FlashAttentionOutput},
        fused_lora::{FusedLora, FusedLoraConfig, FusedLoraOutput},
        fused_norm_lora::{FusedNormLora, FusedNormLoraConfig, FusedNormLoraOutput},
        fused_rope::{FusedRoPE, FusedRoPEConfig},
        fused_swiglu::{
            FusedMLP, FusedMLPOutput, FusedSwiGLU, FusedSwiGLUConfig, FusedSwiGLUOutput,
        },
        fused_training::{BatchedCommandBuffer, FusedAdamW},
        mpp_gemm::MppGemm,
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
/// structs on the fly from the shared `Arc<MetalContext>`. For structs with
/// significant pipeline compilation cost (e.g., [`FlashAttention`]) a future
/// optimisation could cache them here, but correctness is unaffected either way.
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

    // ---- GEMM family --------------------------------------------------------

    /// Standard GEMM via [`MppGemm`].
    ///
    /// [`MppGemm`] checks NAX availability internally and falls back to the
    /// Metal 3 `steel_gemm` kernels when NAX is not present, making it correct
    /// for Metal 3 hardware.
    fn gemm(
        &self,
        _ctx: &Arc<MetalContext>,
        desc: &GemmDescriptor,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        c_or_d: &dyn AsMetalBuffer,
    ) -> Result<()> {
        use crate::kernels::mpp_gemm::MppGemmConfig;
        let mut config = MppGemmConfig::new(desc.m, desc.n, desc.k);
        config.alpha = desc.alpha;
        config.beta = desc.beta;
        config.batch_size = desc.batch_size;
        config.use_fp16 = desc.use_fp16;
        let kernel = MppGemm::new(self.ctx.clone(), config);
        kernel.execute(a, b, c_or_d)
    }

    /// Depthwise GEMM accumulate via [`DwGemm::queue_gemm_accum`].
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
        let kernel = DwGemm::new(self.ctx.clone())?;
        kernel.queue_gemm_accum(batch, a, b, c, m, n, k, alpha, beta)
    }

    // ---- Attention ----------------------------------------------------------

    /// FlashAttention forward via [`FlashAttention::forward`].
    fn flash_attention_forward(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FlashAttentionConfig,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
    ) -> Result<FlashAttentionOutput> {
        let kernel = FlashAttention::new(ctx.clone(), config.clone())?;
        kernel.forward(queries, keys, values)
    }

    /// FlashAttention backward via [`FlashAttention::backward`].
    ///
    /// Returns `(dQ, dK, dV)`.
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
        let kernel = FlashAttention::new(ctx.clone(), config.clone())?;
        kernel.backward(queries, keys, values, output, d_output, logsumexp)
    }

    // ---- Fused linear operations --------------------------------------------

    /// Fused SwiGLU via [`FusedSwiGLU`].
    ///
    /// Dispatches to `forward_with_lora` when any LoRA buffer is provided,
    /// otherwise calls the plain `forward` path.
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
        let kernel = FusedSwiGLU::new(ctx.clone(), config.clone())?;

        match (gate_lora_a, gate_lora_b, up_lora_a, up_lora_b) {
            (Some(ga), Some(gb), Some(ua), Some(ub)) => {
                kernel.forward_with_lora(input, gate_weight, up_weight, ga, gb, ua, ub)
            }
            _ => kernel.forward(input, gate_weight, up_weight),
        }
    }

    /// Fused full-MLP via [`FusedMLP::forward`].
    fn fused_mlp(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedSwiGLUConfig,
        input: &MetalBuffer<f32>,
        gate_weight: &MetalBuffer<f32>,
        up_weight: &MetalBuffer<f32>,
        down_weight: &MetalBuffer<f32>,
    ) -> Result<FusedMLPOutput> {
        let kernel = FusedMLP::new(ctx.clone(), config.clone())?;
        kernel.forward(input, gate_weight, up_weight, down_weight)
    }

    /// Fused RMSNorm + LoRA via [`FusedNormLora::forward`].
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
        let kernel = FusedNormLora::new(ctx.clone(), config.clone())?;
        kernel.forward(input, gamma, weight, lora_a, lora_b)
    }

    /// Fused LoRA forward via [`FusedLora::forward`].
    fn fused_lora_forward(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedLoraConfig,
        x: &dyn AsMetalBuffer,
        weight: &dyn AsMetalBuffer,
        lora_a: &dyn AsMetalBuffer,
        lora_b: &dyn AsMetalBuffer,
    ) -> Result<FusedLoraOutput> {
        let kernel = FusedLora::new(ctx.clone(), config.clone())?;
        kernel.forward(x, weight, lora_a, lora_b)
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

    /// Fused RoPE via [`FusedRoPE`].
    ///
    /// Dispatches to the most specific variant based on the presence of `keys`
    /// and `position_ids`:
    ///
    /// | keys | position_ids | method |
    /// |------|-------------|--------|
    /// | Some | Some        | `apply_qk_with_positions` |
    /// | Some | None        | `apply_qk_inplace` |
    /// | None | Some        | `apply_with_positions` |
    /// | None | None        | `apply_inplace` |
    fn fused_rope(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedRoPEConfig,
        queries: &mut MetalBuffer<f32>,
        keys: Option<&mut MetalBuffer<f32>>,
        position_ids: Option<&MetalBuffer<i32>>,
    ) -> Result<()> {
        let kernel = FusedRoPE::new(ctx.clone(), config.clone())?;

        match (keys, position_ids) {
            (Some(k), Some(pos)) => kernel.apply_qk_with_positions(queries, k, pos),
            (Some(k), None) => kernel.apply_qk_inplace(queries, k),
            (None, Some(pos)) => kernel.apply_with_positions(queries, pos),
            (None, None) => kernel.apply_inplace(queries),
        }
    }
}
