//! Trait-based kernel dispatch abstraction.
//!
//! This module defines [`KernelBackend`] — the single interface that all kernel
//! dispatch paths implement. [`Metal3Backend`](crate::metal3_backend::Metal3Backend) (Task 2) delegates to the existing
//! Metal 3 kernel structs. [`Metal4Backend`](crate::metal4::Metal4Backend) (Tasks 6+) uses MPP / NAX paths on
//! M5+ hardware.
//!
//! # Design
//!
//! Each operation family is represented by an input descriptor struct that bundles
//! the parameters the kernel needs. These descriptors mirror the existing config
//! structs so that [`Metal3Backend`](crate::metal3_backend::Metal3Backend) can delegate without translation overhead.
//!
//! All methods take `&self` — backends are wrapped in `Arc<dyn KernelBackend>` and
//! shared freely across threads.
//!
//! # What is not here
//!
//! Dense GEMM, attention, RoPE, SwiGLU and LoRA are not backend operations: the
//! tensor runtime's own ops run them in the lazy graph and beat every
//! hand-written kernel this crate carried for them, so they were removed.

use std::sync::Arc;

use half::f16;

use crate::{
    buffer::{AsMetalBuffer, MetalBuffer},
    context::MetalContext,
    error::Result,
    kernels::{
        fused_cross_entropy::{FusedCrossEntropyConfig, FusedCrossEntropyOutput},
        fused_moe::FusedMoeExpertConfig,
        fused_training::{AdamWConfig, BatchedCommandBuffer, ParamInfo},
        moe::MoeConfig,
    },
};

// ============================================================================
// BackendCaps — static capability descriptor
// ============================================================================

/// Static capability descriptor for a kernel backend.
///
/// Queried once at startup by [`KernelDispatch`](crate::KernelDispatch)
/// to decide which operations to route to which backend.
#[derive(Debug, Clone)]
pub struct BackendCaps {
    /// Backend name for diagnostics and logging.
    pub name: &'static str,

    /// Whether this backend has a quantized GEMM path.
    pub has_quantized_gemm: bool,

    /// Whether this backend provides fused AdamW optimizer.
    pub has_adamw: bool,

    /// Whether this backend provides fused cross-entropy loss.
    pub has_cross_entropy: bool,

    /// Whether this backend provides fused MoE kernels.
    pub has_moe: bool,

    /// Whether this backend provides grouped GEMM for MoE expert dispatch.
    pub has_grouped_gemm: bool,
}

impl BackendCaps {
    /// Capabilities for the Metal 3 fallback backend.
    pub const fn metal3() -> Self {
        Self {
            name: "Metal3",
            has_quantized_gemm: false,
            has_adamw: true,
            has_cross_entropy: true,
            has_moe: true,
            has_grouped_gemm: true,
        }
    }

    /// Capabilities for the Metal 4 / MPP backend (M5+ only).
    ///
    /// # Wired MPP operations
    ///
    /// The following flags are `true` because the corresponding trait methods
    /// in [`Metal4Backend`](crate::metal4::Metal4Backend) dispatch through MPP kernel structs:
    ///
    /// - `has_quantized_gemm`: `MppQuantizedGemm` via `fused_adamw_step_standalone`
    /// - `has_adamw`: `MppFusedAdamW` via `fused_adamw_step_standalone`
    /// - `has_cross_entropy`: `MppFusedCrossEntropy`
    ///
    /// The following remain `false` because their MPP APIs are structurally
    /// incompatible with the trait's parameter model:
    ///
    /// - `has_moe`: quantized u32 weights vs MPP's dense fp16 weights
    /// - `has_grouped_gemm`: GPU-side prefix-sum buffer, no CPU read
    pub const fn metal4() -> Self {
        Self {
            name: "Metal4",
            // Wired: MppQuantizedGemm dispatched directly from quantized_gemm().
            has_quantized_gemm: true,
            // Wired: MppFusedAdamW via fused_adamw_step_standalone().
            has_adamw: true,
            // Wired: MppFusedCrossEntropy.
            has_cross_entropy: true,
            // Fallback: quantized u32 weights in descriptor, MPP needs fp16.
            has_moe: false,
            // Fallback: GroupedGemmDispatch needs per-expert counts from GPU buffer.
            has_grouped_gemm: false,
        }
    }
}

// ============================================================================
// Input descriptor types
// ============================================================================

/// Descriptor for a quantized GEMM: `Y[M,N]` = `X[M,K]` @ `W_q[N,K]^T`
#[derive(Debug, Clone)]
pub struct QuantizedGemmDescriptor {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Quantization group size (typically 64).
    pub group_size: usize,
    /// Quantization bits (2 or 4).
    pub bits: u8,
}

impl QuantizedGemmDescriptor {
    /// Create a 4-bit group-64 descriptor.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            group_size: 64,
            bits: 4,
        }
    }
}

/// Descriptor for a grouped GEMM: `Y[M,N]` = `X[M,K]` @ `W[E,N,K]^T`, dispatched per expert.
///
/// Used by MoE models to route `total_tokens` token activations through `num_experts`
/// independent weight matrices. `expert_offsets[e]..expert_offsets[e+1]` gives the
/// contiguous range of sorted tokens assigned to expert `e`.
#[derive(Debug, Clone)]
pub struct GroupedGemmDescriptor {
    /// Total number of token-expert pairs (M).
    pub total_tokens: usize,
    /// Number of experts (E).
    pub num_experts: usize,
    /// Input / reduction dimension (K = hidden_size).
    pub hidden_size: usize,
    /// Output dimension per expert (N = intermediate_size).
    pub intermediate_size: usize,
    /// Top-k experts per token.
    pub topk: usize,
    /// Whether to permute input rows on load (gather).
    pub permute_x: bool,
    /// Whether to permute output rows on store (scatter).
    pub permute_y: bool,
    /// Whether to fuse top-k weight multiplication into the output store.
    pub fuse_mul: bool,
    /// Use fp16 precision (false = fp32).
    pub use_fp16: bool,
}

impl GroupedGemmDescriptor {
    /// Create a basic grouped GEMM descriptor with no permutation or weight fusion.
    pub fn new(
        total_tokens: usize,
        num_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
        topk: usize,
    ) -> Self {
        Self {
            total_tokens,
            num_experts,
            hidden_size,
            intermediate_size,
            topk,
            permute_x: false,
            permute_y: false,
            fuse_mul: false,
            use_fp16: false,
        }
    }
}

/// Descriptor for the AdamW optimizer step.
///
/// Mirrors `FusedAdamW::queue_update` parameters.
#[derive(Debug)]
pub struct AdamWDescriptor<'a> {
    /// Flattened parameter buffer.
    pub params: &'a MetalBuffer<f32>,
    /// Flattened gradient buffer.
    pub grads: &'a MetalBuffer<f32>,
    /// First moment buffer.
    pub m: &'a MetalBuffer<f32>,
    /// Second moment buffer.
    pub v: &'a MetalBuffer<f32>,
    /// Per-parameter metadata (offset, size, moment offsets), every
    /// parameter in group 0.
    pub param_info: &'a MetalBuffer<ParamInfo>,
    /// Optimizer hyperparameters, for every parameter.
    pub config: AdamWConfig,
}

/// Descriptor for fused MoE expert forward.
///
/// Mirrors `FusedMoeExpert` invocation parameters.
#[derive(Debug)]
pub struct MoeExpertDescriptor<'a> {
    /// Per-expert configuration (dims, group size, bit width).
    pub expert_config: FusedMoeExpertConfig,
    /// Input activations [num_tokens, hidden_dim].
    pub input: &'a MetalBuffer<f32>,
    /// Gate weight (quantized, packed).
    pub gate_weight: &'a MetalBuffer<u32>,
    /// Gate weight scales.
    pub gate_scales: &'a MetalBuffer<f16>,
    /// Gate weight biases.
    pub gate_biases: &'a MetalBuffer<f16>,
    /// Up weight (quantized, packed).
    pub up_weight: &'a MetalBuffer<u32>,
    /// Up weight scales.
    pub up_scales: &'a MetalBuffer<f16>,
    /// Up weight biases.
    pub up_biases: &'a MetalBuffer<f16>,
    /// Down weight (quantized, packed).
    pub down_weight: &'a MetalBuffer<u32>,
    /// Down weight scales.
    pub down_scales: &'a MetalBuffer<f16>,
    /// Down weight biases.
    pub down_biases: &'a MetalBuffer<f16>,
    /// Number of tokens to process.
    pub num_tokens: usize,
}

// ============================================================================
// KernelBackend trait
// ============================================================================

/// Trait-based kernel dispatch interface.
///
/// Both [`Metal3Backend`](crate::metal3_backend::Metal3Backend) and [`Metal4Backend`](crate::metal4::Metal4Backend) implement this trait, allowing
/// [`KernelDispatch`](crate::KernelDispatch) to route each operation to the appropriate backend without
/// runtime cost beyond the initial routing decision.
///
/// # Thread safety
///
/// Implementations must be `Send + Sync`. Backends are wrapped in `Arc<dyn KernelBackend>`
/// and shared across the thread pool.
pub trait KernelBackend: Send + Sync {
    // ---- Capabilities -------------------------------------------------------

    /// Return static capability descriptor for this backend.
    fn caps(&self) -> &BackendCaps;

    // ---- Quantized and grouped GEMM -----------------------------------------

    /// Execute quantized GEMM: Y = X @ W_q^T (dequant on the fly).
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the kernel's buffers one-to-one"
    )]
    fn quantized_gemm(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &QuantizedGemmDescriptor,
        x: &dyn AsMetalBuffer,
        w_q: &dyn AsMetalBuffer,
        scales: &dyn AsMetalBuffer,
        biases: Option<&dyn AsMetalBuffer>,
        output: &dyn AsMetalBuffer,
    ) -> Result<()>;

    /// Grouped GEMM forward: Y = `X[M,K]` @ `W[E,N,K]^T`, one sub-GEMM per expert.
    ///
    /// Tokens are pre-sorted by expert assignment. `expert_offsets[e]..expert_offsets[e+1]`
    /// gives the contiguous slice of tokens routed to expert `e`. `gather_indices` and
    /// `scatter_indices` handle the token permutation; `topk_weights` are fused into the
    /// output when `desc.fuse_mul` is set.
    ///
    /// Returns a `MetalBuffer<f32>` of shape `[total_tokens, intermediate_size]`.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the kernel's buffers one-to-one"
    )]
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
    ) -> Result<MetalBuffer<f32>>;

    // ---- Training optimizers and losses -------------------------------------

    /// Queue a fused AdamW optimizer step into a batched command buffer.
    ///
    /// This encodes the dispatch but does **not** execute it — callers must
    /// call `batch.execute()` after queuing all operations.
    fn fused_adamw_step(
        &self,
        batch: &mut BatchedCommandBuffer,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()>;

    /// Execute a fused AdamW optimizer step, creating its own command buffer.
    ///
    /// Unlike [`fused_adamw_step`], this method does not require an existing
    /// [`BatchedCommandBuffer`]. It creates a temporary command buffer internally,
    /// executes the step synchronously, and waits for completion before returning.
    ///
    /// # When to use
    ///
    /// Use this method from backends (like [`Metal4Backend`]) that cannot encode
    /// into a shared [`BatchedCommandBuffer`] because their MPP dispatch model
    /// owns its own command buffer per call.
    ///
    /// Metal 3 callers should prefer [`fused_adamw_step`] for batching efficiency.
    ///
    /// [`fused_adamw_step`]: KernelBackend::fused_adamw_step
    /// [`Metal4Backend`]: crate::metal4::Metal4Backend
    fn fused_adamw_step_standalone(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        // Default: fall through to error — backends that support this override it.
        let _ = ctx;
        let _ = desc;
        Err(crate::error::MetalError::ExecutionFailed(
            "fused_adamw_step_standalone not implemented for this backend".into(),
        ))
    }

    /// Fused cross-entropy loss forward pass.
    ///
    /// Returns per-token losses and cached logsumexp for backward.
    fn fused_cross_entropy(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedCrossEntropyConfig,
        logits: &dyn AsMetalBuffer,
        targets: &MetalBuffer<i32>,
    ) -> Result<FusedCrossEntropyOutput>;

    // ---- MoE ----------------------------------------------------------------

    /// MoE expert routing: compute top-k expert assignments from router logits.
    fn moe_routing(
        &self,
        ctx: &Arc<MetalContext>,
        config: &MoeConfig,
        router_logits: &MetalBuffer<f32>,
    ) -> Result<crate::kernels::moe::MoeRouting>;

    /// Fused MoE expert forward: gate+up (SwiGLU) + down projections for one expert.
    fn fused_moe_expert(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &MoeExpertDescriptor<'_>,
    ) -> Result<MetalBuffer<f32>>;
}
