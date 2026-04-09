//! Metal 3 kernel backend — thin delegation layer over existing kernel structs.
//!
//! [`Metal3Backend`] implements [`KernelBackend`] by forwarding every method to
//! the corresponding existing Metal 3 kernel struct. No logic lives here; this
//! file is pure glue.
//!
//! # Remaining todo! stubs
//!
//! Two methods remain stubbed because the type impedance cannot be bridged
//! without a structural change to the underlying kernel:
//!
//! - **`quantized_gemm`**: Metal 3 has no quantized GEMM path
//!   (`BackendCaps::metal3()` reports `has_quantized_gemm: false`).
//!
//! - **`grouped_gemm`**: [`MoeKernel`] takes [`MoeConfig`] + [`MoeRouting`],
//!   but the trait supplies a flat [`GroupedGemmDescriptor`]. An adapter is
//!   needed in Task 3 (KernelDispatch).
//!
//! - **`fused_moe_expert`**: [`ExpertWeightBuffers`] stores scales/biases as
//!   `MetalBuffer<u16>` (raw bits), but [`MoeExpertDescriptor`] carries them
//!   as `&MetalBuffer<f16>`. A reinterpret-cast shim or a `MetalBuffer::reinterpret`
//!   constructor is needed before this can be wired without a copy.
//!
//! # Resolved stubs
//!
//! `fused_lora_forward`, `fused_cross_entropy`, `fused_distill_loss`, and
//! `fused_adamw_step` are now wired using [`DynBufRef`] (for the sampler-trait
//! mismatch) and param-info extraction from the `AdamWDescriptor` (for AdamW).

use std::sync::Arc;

use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

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
        fused_cross_entropy::{FusedCrossEntropy, FusedCrossEntropyConfig, FusedCrossEntropyOutput},
        fused_distill::{DistillLossType, FusedDistill, FusedDistillConfig, FusedDistillOutput},
        fused_lora::{FusedLora, FusedLoraConfig, FusedLoraOutput},
        fused_norm_lora::{FusedNormLora, FusedNormLoraConfig, FusedNormLoraOutput},
        fused_rope::{FusedRoPE, FusedRoPEConfig},
        fused_sampler::AsMetalBuffer as SamplerBuf,
        fused_swiglu::{FusedMLP, FusedMLPOutput, FusedSwiGLU, FusedSwiGLUConfig, FusedSwiGLUOutput},
        fused_training::{BatchedCommandBuffer, FusedAdamW},
        moe::{MoeConfig, MoeKernel, MoeRouting},
        mpp_gemm::MppGemm,
    },
};

// ============================================================================
// DynBufRef — bridges &dyn buffer::AsMetalBuffer to both AsMetalBuffer traits
// ============================================================================

/// Newtype wrapper that adapts a `&dyn buffer::AsMetalBuffer` reference into
/// types implementing both `buffer::AsMetalBuffer` and
/// `fused_sampler::AsMetalBuffer`.
///
/// The two traits are identically-shaped but distinct:
/// - `buffer::AsMetalBuffer` uses `as_metal_buffer()`
/// - `fused_sampler::AsMetalBuffer` uses `metal_buffer()`
///
/// Both return `&ProtocolObject<dyn MTLBuffer>`, so the adapter is
/// zero-overhead — it simply re-names the call. Implementing both lets
/// `DynBufRef` be passed to kernel methods regardless of which variant they
/// expect.
struct DynBufRef<'a>(&'a dyn AsMetalBuffer);

impl SamplerBuf for DynBufRef<'_> {
    fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.0.as_metal_buffer()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl AsMetalBuffer for DynBufRef<'_> {
    fn as_metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.0.as_metal_buffer()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

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

    /// Fused LoRA forward via [`FusedLora::forward`].
    ///
    /// [`FusedLora::forward`] is generic over `B: fused_sampler::AsMetalBuffer`.
    /// The trait provides `&dyn buffer::AsMetalBuffer`. We bridge the gap with
    /// [`DynBufRef`], which implements `fused_sampler::AsMetalBuffer` by
    /// delegating `metal_buffer()` to `buffer::AsMetalBuffer::as_metal_buffer()`.
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
        kernel.forward(&DynBufRef(x), &DynBufRef(weight), &DynBufRef(lora_a), &DynBufRef(lora_b))
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
        adamw.queue_update(batch, desc.params, desc.grads, desc.m, desc.v, desc.param_info, &desc.config)
    }

    /// Fused cross-entropy via [`FusedCrossEntropy::forward_dyn`].
    ///
    /// The trait provides `logits: &dyn AsMetalBuffer` to support both f32
    /// and f16 logits without an extra type parameter. [`FusedCrossEntropy`]
    /// exposes [`forward_dyn`] which accepts the same type-erased buffer.
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

    /// Fused MoE expert forward — blocked on dtype type mismatch.
    ///
    /// [`ExpertWeightBuffers`] stores scales/biases as `MetalBuffer<u16>`
    /// (raw 16-bit values), but [`MoeExpertDescriptor`] carries them as
    /// `&MetalBuffer<f16>` (typed half-precision). Assembling an
    /// `ExpertWeightBuffers` from the descriptor fields would require either:
    ///
    /// 1. A `MetalBuffer::reinterpret::<f16, u16>()` constructor that
    ///    returns an alias with the new element type; or
    /// 2. Changing `ExpertWeightBuffers` to use `MetalBuffer<f16>` throughout.
    ///
    /// Until one of those changes lands, this stub correctly panics when called.
    /// `BackendCaps::metal3()` does NOT set `has_moe: false`, so if a caller
    /// routes here for expert forward the panic is the correct signal.
    ///
    /// [`ExpertWeightBuffers`]: crate::kernels::fused_moe::ExpertWeightBuffers
    fn fused_moe_expert(
        &self,
        _ctx: &Arc<MetalContext>,
        _desc: &MoeExpertDescriptor<'_>,
    ) -> Result<MetalBuffer<f32>> {
        todo!(
            "ExpertWeightBuffers uses MetalBuffer<u16> for scales/biases but \
             MoeExpertDescriptor carries &MetalBuffer<f16> — add \
             MetalBuffer::reinterpret() or change ExpertWeightBuffers to f16 \
             (Task 3 adapter)"
        )
    }

    // ---- Distillation -------------------------------------------------------

    /// Fused distillation loss via [`FusedDistill::forward`].
    ///
    /// [`FusedDistill::forward`] is generic over `impl buffer::AsMetalBuffer`.
    /// [`DynBufRef`] implements `buffer::AsMetalBuffer` by delegating to
    /// `as_metal_buffer()`, so we wrap the `&dyn` references and pass them
    /// directly to the generic method.
    fn fused_distill_loss(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedDistillConfig,
        teacher_logits: &dyn AsMetalBuffer,
        student_logits: &dyn AsMetalBuffer,
        loss_type: DistillLossType,
    ) -> Result<FusedDistillOutput> {
        let kernel = FusedDistill::new(ctx.clone(), config.clone())?;
        kernel.forward(&DynBufRef(teacher_logits), &DynBufRef(student_logits), loss_type)
    }
}
