//! Metal 3 kernel backend — thin delegation layer over existing kernel structs.
//!
//! [`Metal3Backend`] implements [`KernelBackend`] by forwarding every method to
//! the corresponding existing Metal 3 kernel struct. No logic lives here; this
//! file is pure glue.
//!
//! # todo! stubs
//!
//! Several methods are stubbed with `todo!` because the existing kernel structs
//! have API shapes that don't align cleanly with the [`KernelBackend`] trait:
//!
//! - **`quantized_gemm`**: Metal 3 has no quantized GEMM path
//!   (`BackendCaps::metal3()` reports `has_quantized_gemm: false`).
//!
//! - **`grouped_gemm`**: [`MoeKernel`] takes [`MoeConfig`] + [`MoeRouting`],
//!   but the trait supplies a flat [`GroupedGemmDescriptor`]. An adapter is
//!   needed in Task 3 (KernelDispatch).
//!
//! - **`fused_lora_forward`**: [`FusedLora::forward`] is generic over
//!   `B: AsMetalBuffer` (static dispatch) while the trait provides
//!   `&dyn AsMetalBuffer`. A typed shim or a new `forward_dyn` method on
//!   [`FusedLora`] is needed.
//!
//! - **`fused_cross_entropy`**: [`FusedCrossEntropy::forward`] takes
//!   `&MetalBuffer<f32>`; the trait provides `&dyn AsMetalBuffer`. Same fix
//!   needed as `fused_lora_forward`.
//!
//! - **`fused_moe_expert`**: [`FusedMoeExpert::forward_single_expert`] takes
//!   `&ExpertWeightBuffers` (a pre-built buffer bundle) while
//!   [`MoeExpertDescriptor`] carries flat individual `MetalBuffer` references.
//!   An adapter that assembles `ExpertWeightBuffers` from the descriptor fields
//!   is required.
//!
//! - **`fused_distill_loss`**: [`FusedDistill::forward`] is generic over
//!   `impl AsMetalBuffer`; same static-vs-dynamic dispatch mismatch as
//!   `fused_lora_forward`.
//!
//! - **`fused_adamw_step`**: [`FusedAdamW::new`] requires a `&[usize]` of
//!   per-parameter sizes to size its internal grid, which isn't carried in
//!   [`AdamWDescriptor`]. A count shim or a `FusedAdamW::new_single` variant
//!   is needed.
//!
//! None of the stubs affect compilation correctness — they will panic at
//! runtime only if a caller routes to Metal3Backend for these operations,
//! which the `BackendCaps` flags prevent for the quantized path.

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
        dw_gemm::DwGemm,
        flash_attention::{FlashAttention, FlashAttentionConfig, FlashAttentionOutput},
        fused_cross_entropy::{FusedCrossEntropyConfig, FusedCrossEntropyOutput},
        fused_distill::{DistillLossType, FusedDistillConfig, FusedDistillOutput},
        fused_lora::{FusedLoraConfig, FusedLoraOutput},
        fused_norm_lora::{FusedNormLora, FusedNormLoraConfig, FusedNormLoraOutput},
        fused_rope::{FusedRoPE, FusedRoPEConfig},
        fused_swiglu::{FusedMLP, FusedMLPOutput, FusedSwiGLU, FusedSwiGLUConfig, FusedSwiGLUOutput},
        fused_training::BatchedCommandBuffer,
        moe::{MoeConfig, MoeKernel, MoeRouting},
        mpp_gemm::MppGemm,
    },
};

// ============================================================================
// Metal3Backend
// ============================================================================

/// Metal 3 kernel backend.
///
/// A thin delegation layer over the existing Metal 3 kernel structs.
/// Implements [`KernelBackend`] so that [`KernelDispatch`] can route
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
        todo!("Metal3 has no quantized GEMM path; BackendCaps::metal3() has_quantized_gemm=false")
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
        todo!(
            "MoeKernel API takes MoeConfig+MoeRouting, not GroupedGemmDescriptor; \
             adapter needed in KernelDispatch (Task 3)"
        )
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

    /// Fused LoRA forward — requires a typed shim.
    ///
    /// [`FusedLora::forward`] is generic over `B: AsMetalBuffer` (static
    /// dispatch). The trait provides `&dyn AsMetalBuffer` (dynamic dispatch).
    /// Rust does not allow calling a generic method through a trait object.
    /// A `forward_dyn` method on [`FusedLora`] (or a concrete-type accessor on
    /// [`FusedLoraConfig`]) is needed before this stub can be replaced.
    fn fused_lora_forward(
        &self,
        _ctx: &Arc<MetalContext>,
        _config: &FusedLoraConfig,
        _x: &dyn AsMetalBuffer,
        _weight: &dyn AsMetalBuffer,
        _lora_a: &dyn AsMetalBuffer,
        _lora_b: &dyn AsMetalBuffer,
    ) -> Result<FusedLoraOutput> {
        todo!(
            "FusedLora::forward<B: AsMetalBuffer> is statically dispatched; \
             trait provides &dyn AsMetalBuffer — add FusedLora::forward_dyn() \
             or expose execute_forward as pub"
        )
    }

    // ---- Training optimizers and losses -------------------------------------

    /// Fused AdamW step — requires per-parameter size metadata.
    ///
    /// [`FusedAdamW::new`] needs a `&[usize]` of per-parameter element counts
    /// to size its internal dispatch grid. [`AdamWDescriptor`] does not carry
    /// that slice. Either add a `param_count` field to the descriptor or expose
    /// a `FusedAdamW::new_single(ctx, max_param_size, num_params)` constructor.
    fn fused_adamw_step(
        &self,
        _batch: &mut BatchedCommandBuffer,
        _desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        todo!(
            "FusedAdamW::new() requires &[usize] param_sizes; AdamWDescriptor \
             does not carry per-parameter size metadata — add param_sizes field \
             or a FusedAdamW::queue_update_raw() accepting (num_params, max_param_size)"
        )
    }

    /// Fused cross-entropy — requires a typed shim.
    ///
    /// [`FusedCrossEntropy::forward`] takes `&MetalBuffer<f32>`; the trait
    /// provides `logits: &dyn AsMetalBuffer`. The same static-vs-dynamic
    /// dispatch gap as `fused_lora_forward`. Either expose `execute_forward`
    /// as public or add a `forward_dyn` variant to [`FusedCrossEntropy`].
    fn fused_cross_entropy(
        &self,
        _ctx: &Arc<MetalContext>,
        _config: &FusedCrossEntropyConfig,
        _logits: &dyn AsMetalBuffer,
        _targets: &MetalBuffer<i32>,
    ) -> Result<FusedCrossEntropyOutput> {
        todo!(
            "FusedCrossEntropy::forward takes &MetalBuffer<f32>; trait provides \
             &dyn AsMetalBuffer — add FusedCrossEntropy::forward_dyn() or \
             expose execute_forward as pub"
        )
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

    /// Fused MoE expert forward — requires an adapter struct.
    ///
    /// [`FusedMoeExpert::forward_single_expert`] takes an [`ExpertWeightBuffers`]
    /// bundle, but [`MoeExpertDescriptor`] carries flat individual buffer
    /// references. An adapter that assembles the bundle needs to be written
    /// before this stub can be replaced.
    ///
    /// [`ExpertWeightBuffers`]: crate::kernels::fused_moe::ExpertWeightBuffers
    fn fused_moe_expert(
        &self,
        _ctx: &Arc<MetalContext>,
        _desc: &MoeExpertDescriptor<'_>,
    ) -> Result<MetalBuffer<f32>> {
        todo!(
            "FusedMoeExpert::forward_single_expert takes ExpertWeightBuffers; \
             MoeExpertDescriptor carries flat buffers — write an adapter that \
             assembles ExpertWeightBuffers from the descriptor fields"
        )
    }

    // ---- Distillation -------------------------------------------------------

    /// Fused distillation loss — requires a typed shim.
    ///
    /// [`FusedDistill::forward`] is generic over `impl AsMetalBuffer`; the trait
    /// provides `&dyn AsMetalBuffer`. Same static-vs-dynamic dispatch gap as
    /// `fused_lora_forward`. Either expose `execute_forward` as public or add a
    /// `forward_dyn` variant to [`FusedDistill`].
    fn fused_distill_loss(
        &self,
        _ctx: &Arc<MetalContext>,
        _config: &FusedDistillConfig,
        _teacher_logits: &dyn AsMetalBuffer,
        _student_logits: &dyn AsMetalBuffer,
        _loss_type: DistillLossType,
    ) -> Result<FusedDistillOutput> {
        todo!(
            "FusedDistill::forward<impl AsMetalBuffer> is statically dispatched; \
             trait provides &dyn AsMetalBuffer — add FusedDistill::forward_dyn() \
             or expose execute_forward as pub"
        )
    }
}
