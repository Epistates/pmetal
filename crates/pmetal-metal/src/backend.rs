//! Trait-based kernel dispatch abstraction.
//!
//! This module defines [`KernelBackend`] — the single interface that all kernel
//! dispatch paths implement. [`Metal3Backend`] (Task 2) delegates to the existing
//! Metal 3 kernel structs. [`Metal4Backend`] (Tasks 6+) uses MPP / NAX paths on
//! M5+ hardware.
//!
//! # Design
//!
//! Each operation family is represented by an input descriptor struct that bundles
//! the parameters the kernel needs. These descriptors mirror the existing config
//! structs so that `Metal3Backend` can delegate without translation overhead.
//!
//! All methods take `&self` — backends are wrapped in `Arc<dyn KernelBackend>` and
//! shared freely across threads.
//!
//! # Routing heuristics
//!
//! [`KernelBackend::should_handle_gemm`] lets the dispatch layer ask the backend
//! whether it wants to handle a particular GEMM shape. Metal 4 returns `false` for
//! decode (M=1) or K not divisible by 32, since the MPP tile constraints would
//! produce degenerate launch grids.

use std::sync::Arc;

use half::f16;

use crate::{
    buffer::{AsMetalBuffer, MetalBuffer},
    context::MetalContext,
    error::Result,
    kernels::{
        fused_cross_entropy::{FusedCrossEntropyConfig, FusedCrossEntropyOutput},
        fused_distill::{DistillLossType, FusedDistillConfig, FusedDistillOutput},
        fused_lora::{FusedLoraConfig, FusedLoraOutput},
        fused_moe::FusedMoeExpertConfig,
        fused_norm_lora::{FusedNormLoraConfig, FusedNormLoraOutput},
        fused_rope::FusedRoPEConfig,
        fused_swiglu::{FusedMLPOutput, FusedSwiGLUConfig, FusedSwiGLUOutput},
        fused_training::{AdamWConfig, BatchedCommandBuffer, ParamInfo},
        flash_attention::{FlashAttentionConfig, FlashAttentionOutput},
        moe::MoeConfig,
    },
};

// ============================================================================
// BackendCaps — static capability descriptor
// ============================================================================

/// Static capability descriptor for a kernel backend.
///
/// Queried once at startup by [`KernelDispatch`](crate::kernels::dispatch::KernelDispatch)
/// to decide which operations to route to which backend.
#[derive(Debug, Clone)]
pub struct BackendCaps {
    /// Backend name for diagnostics and logging.
    pub name: &'static str,

    /// Whether this backend can execute GEMM operations.
    pub has_gemm: bool,

    /// Whether this backend has a quantized GEMM path.
    pub has_quantized_gemm: bool,

    /// Whether this backend provides flash attention.
    pub has_flash_attention: bool,

    /// Whether this backend provides MPP flash attention (NAX-only).
    pub has_mpp_flash_attention: bool,

    /// Whether this backend provides fused SwiGLU.
    pub has_swiglu: bool,

    /// Whether this backend provides fused AdamW optimizer.
    pub has_adamw: bool,

    /// Whether this backend provides fused cross-entropy loss.
    pub has_cross_entropy: bool,

    /// Whether this backend provides fused RoPE.
    pub has_rope: bool,

    /// Whether this backend provides fused MoE kernels.
    pub has_moe: bool,

    /// Whether this backend provides fused distillation loss.
    pub has_distill: bool,

    /// Whether this backend provides fused norm+LoRA.
    pub has_norm_lora: bool,

    /// Whether this backend provides fused LoRA forward/backward.
    pub has_lora: bool,

    /// Whether this backend supports batched (depthwise) GEMM accumulate.
    pub has_dw_gemm: bool,

    /// Whether this backend provides grouped GEMM for MoE expert dispatch.
    pub has_grouped_gemm: bool,

    /// Minimum K dimension for GEMM to be efficient on this backend.
    ///
    /// Metal 4 MPP tile constraints require K divisible by 32.
    pub gemm_k_alignment: usize,

    /// Minimum M (output rows) for GEMM to be efficient on this backend.
    ///
    /// Metal 4 skips decode (M=1) and routes to Metal 3 instead.
    pub gemm_min_m: usize,
}

impl BackendCaps {
    /// Capabilities for the Metal 3 fallback backend.
    ///
    /// Supports every operation; no alignment requirements.
    pub const fn metal3() -> Self {
        Self {
            name: "Metal3",
            has_gemm: true,
            has_quantized_gemm: false,
            has_flash_attention: true,
            has_mpp_flash_attention: false,
            has_swiglu: true,
            has_adamw: true,
            has_cross_entropy: true,
            has_rope: true,
            has_moe: true,
            has_distill: true,
            has_norm_lora: true,
            has_lora: true,
            has_dw_gemm: true,
            has_grouped_gemm: true,
            gemm_k_alignment: 1,
            gemm_min_m: 1,
        }
    }

    /// Capabilities for the Metal 4 / MPP backend (M5+ only).
    ///
    /// Handles large GEMM shapes with NAX acceleration. Routes decode (M=1)
    /// and unaligned K back to Metal 3.
    ///
    /// # Wired MPP operations (Task 18)
    ///
    /// The following flags are `true` because the corresponding trait methods
    /// in [`Metal4Backend`] now dispatch through MPP kernel structs:
    ///
    /// - `has_swiglu`: `MppFusedSwiGLU` (no-LoRA path; LoRA falls back to Metal 3)
    /// - `has_cross_entropy`: `MppFusedCrossEntropy`
    /// - `has_rope`: `MppFusedRoPE`
    /// - `has_distill`: `MppFusedDistill`
    /// - `has_norm_lora`: `MppFusedNormLora`
    /// - `has_lora`: `MppFusedLora` (inference mode)
    /// - `has_dw_gemm`: `MppDwGemm`
    ///
    /// The following remain `false` because their MPP APIs are structurally
    /// incompatible with the trait's parameter model (see `backend.rs` doc):
    ///
    /// - `has_adamw`: `BatchedCommandBuffer` vs MPP's own-command-buffer model
    /// - `has_moe`: quantized u32 weights vs MPP's dense fp16 weights
    /// - `has_grouped_gemm`: GPU-side prefix-sum buffer, no CPU read
    pub const fn metal4() -> Self {
        Self {
            name: "Metal4",
            has_gemm: true,
            // Quantized GEMM path (mpp_quantized.rs) is wired in Task 12.
            // Leaving false while Metal3Backend's quantized_gemm() may panic.
            has_quantized_gemm: false,
            has_flash_attention: false,
            has_mpp_flash_attention: true,
            // Wired: MppFusedSwiGLU (no-LoRA path only; LoRA falls back).
            has_swiglu: true,
            // Fallback: BatchedCommandBuffer vs MPP own-command-buffer model.
            has_adamw: false,
            // Wired: MppFusedCrossEntropy.
            has_cross_entropy: true,
            // Wired: MppFusedRoPE.
            has_rope: true,
            // Fallback: quantized u32 weights in descriptor, MPP needs fp16.
            has_moe: false,
            // Wired: MppFusedDistill.
            has_distill: true,
            // Wired: MppFusedNormLora.
            has_norm_lora: true,
            // Wired: MppFusedLora (inference mode).
            has_lora: true,
            // Wired: MppDwGemm (synchronous dispatch, bypasses BatchedCommandBuffer).
            has_dw_gemm: true,
            // Fallback: GroupedGemmDispatch needs per-expert counts from GPU buffer.
            has_grouped_gemm: false,
            gemm_k_alignment: 32,
            gemm_min_m: 2,
        }
    }
}

// ============================================================================
// Input descriptor types
// ============================================================================

/// Descriptor for a standard GEMM: D = alpha * A[M,K] @ B[N,K]^T + beta * C
#[derive(Debug, Clone)]
pub struct GemmDescriptor {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Alpha scale.
    pub alpha: f32,
    /// Beta scale (0 = overwrite, non-zero = accumulate).
    pub beta: f32,
    /// Batch count.
    pub batch_size: usize,
    /// Use fp16 precision.
    pub use_fp16: bool,
}

impl GemmDescriptor {
    /// Create a simple C = A @ B^T descriptor.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
            batch_size: 1,
            use_fp16: true,
        }
    }
}

/// Descriptor for a quantized GEMM: Y[M,N] = X[M,K] @ W_q[N,K]^T
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

/// Descriptor for a grouped GEMM: Y[M,N] = X[M,K] @ W[E,N,K]^T, dispatched per expert.
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
/// Mirrors [`FusedAdamW::queue_update`] parameters.
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
    /// Per-parameter metadata (offset, size, moment offsets).
    pub param_info: &'a MetalBuffer<ParamInfo>,
    /// Optimizer hyperparameters.
    pub config: AdamWConfig,
}

/// Descriptor for fused MoE expert forward.
///
/// Mirrors [`FusedMoeExpert`] invocation parameters.
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
/// Both [`Metal3Backend`] and [`Metal4Backend`] implement this trait, allowing
/// [`KernelDispatch`] to route each operation to the appropriate backend without
/// runtime cost beyond the initial routing decision.
///
/// # Thread safety
///
/// Implementations must be `Send + Sync`. Backends are wrapped in `Arc<dyn KernelBackend>`
/// and shared across the thread pool.
///
/// # Routing
///
/// Before calling any GEMM method, callers should invoke [`should_handle_gemm`] to
/// confirm the backend can efficiently handle the shape. If it returns `false`, the
/// caller should fall back to the other backend.
#[allow(clippy::too_many_arguments)]
pub trait KernelBackend: Send + Sync {
    // ---- Capabilities -------------------------------------------------------

    /// Return static capability descriptor for this backend.
    fn caps(&self) -> &BackendCaps;

    // ---- Routing hints ------------------------------------------------------

    /// Return `true` if this backend should handle the given GEMM shape.
    ///
    /// Metal 4 returns `false` for:
    /// - M = 1 (decode / matvec — MPP tile constraints waste threads)
    /// - K % 32 != 0 (NAX alignment requirement)
    ///
    /// Metal 3 always returns `true`.
    fn should_handle_gemm(&self, m: usize, _n: usize, k: usize) -> bool {
        let caps = self.caps();
        m >= caps.gemm_min_m && (caps.gemm_k_alignment == 1 || k % caps.gemm_k_alignment == 0)
    }

    // ---- GEMM family --------------------------------------------------------

    /// Execute standard GEMM: D = alpha * A @ B^T + beta * C
    ///
    /// Buffers are untyped at the Metal API level; `desc.use_fp16` governs
    /// whether fp16 or fp32 kernels are dispatched.
    fn gemm(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &GemmDescriptor,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        c_or_d: &dyn AsMetalBuffer,
    ) -> Result<()>;

    /// Execute quantized GEMM: Y = X @ W_q^T (dequant on the fly).
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

    /// Encode a weight-gradient GEMM into an existing batched command buffer.
    ///
    /// C = alpha * A[M,K] @ B[N,K]^T + beta * C (read-modify-write).
    /// Used for the backward pass dW computation across all layers.
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
    ) -> Result<()>;

    /// Grouped GEMM forward: Y = X[M,K] @ W[E,N,K]^T, one sub-GEMM per expert.
    ///
    /// Tokens are pre-sorted by expert assignment. `expert_offsets[e]..expert_offsets[e+1]`
    /// gives the contiguous slice of tokens routed to expert `e`. `gather_indices` and
    /// `scatter_indices` handle the token permutation; `topk_weights` are fused into the
    /// output when `desc.fuse_mul` is set.
    ///
    /// Returns a `MetalBuffer<f32>` of shape `[total_tokens, intermediate_size]`.
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

    // ---- Attention ----------------------------------------------------------

    /// FlashAttention forward pass.
    ///
    /// Returns output and, if `config.is_training`, the log-sum-exp tensor.
    fn flash_attention_forward(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FlashAttentionConfig,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
    ) -> Result<FlashAttentionOutput>;

    /// FlashAttention backward pass.
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
    ) -> Result<(MetalBuffer<f16>, MetalBuffer<f16>, MetalBuffer<f16>)>;

    // ---- Fused linear operations --------------------------------------------

    /// Fused SwiGLU forward: output = silu(gate_proj(x)) * up_proj(x)
    ///
    /// If LoRA rank > 0 in `config`, all four LoRA weight buffers must be provided.
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
    ) -> Result<FusedSwiGLUOutput>;

    /// Fused full-MLP forward: output = down_proj(silu(gate(x)) * up(x))
    fn fused_mlp(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedSwiGLUConfig,
        input: &MetalBuffer<f32>,
        gate_weight: &MetalBuffer<f32>,
        up_weight: &MetalBuffer<f32>,
        down_weight: &MetalBuffer<f32>,
    ) -> Result<FusedMLPOutput>;

    /// Fused RMSNorm + LoRA projection.
    ///
    /// output = (norm(x) @ W^T) + scale * ((norm(x) @ A^T) @ B^T)
    fn fused_norm_lora(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedNormLoraConfig,
        input: &MetalBuffer<f32>,
        gamma: &MetalBuffer<f32>,
        weight: &MetalBuffer<f32>,
        lora_a: &MetalBuffer<f32>,
        lora_b: &MetalBuffer<f32>,
    ) -> Result<FusedNormLoraOutput>;

    /// Fused LoRA forward: output = x @ W^T + scale * (x @ A^T) @ B^T
    fn fused_lora_forward(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedLoraConfig,
        x: &dyn AsMetalBuffer,
        weight: &dyn AsMetalBuffer,
        lora_a: &dyn AsMetalBuffer,
        lora_b: &dyn AsMetalBuffer,
    ) -> Result<FusedLoraOutput>;

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

    /// Fused RoPE: apply rotary position embeddings in-place to Q (and optionally K).
    ///
    /// If `keys` is `Some`, this is a fused QK RoPE; if `None`, only Q is rotated.
    fn fused_rope(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedRoPEConfig,
        queries: &mut MetalBuffer<f32>,
        keys: Option<&mut MetalBuffer<f32>>,
        position_ids: Option<&MetalBuffer<i32>>,
    ) -> Result<()>;

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

    // ---- Distillation -------------------------------------------------------

    /// Fused distillation loss forward pass (KL, reverse-KL, JS, or soft-CE).
    fn fused_distill_loss(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedDistillConfig,
        teacher_logits: &dyn AsMetalBuffer,
        student_logits: &dyn AsMetalBuffer,
        loss_type: DistillLossType,
    ) -> Result<FusedDistillOutput>;
}
