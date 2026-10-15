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
        flash_attention::{FlashAttentionConfig, FlashAttentionOutput},
        fused_lora::{FusedLoraConfig, FusedLoraOutput},
        fused_norm_lora::{FusedNormLoraConfig, FusedNormLoraOutput},
        fused_rope::FusedRoPEConfig,
        fused_swiglu::{FusedMLPOutput, FusedSwiGLUConfig, FusedSwiGLUOutput},
        fused_training::{AdamWConfig, BatchedCommandBuffer, ParamInfo},
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

    /// Whether this backend can execute GEMM operations.
    pub has_gemm: bool,

    /// Whether this backend provides flash attention.
    pub has_flash_attention: bool,

    /// Whether this backend provides MPP flash attention (NAX-only).
    pub has_mpp_flash_attention: bool,

    /// Whether this backend provides fused SwiGLU.
    pub has_swiglu: bool,

    /// Whether this backend provides fused AdamW optimizer.
    pub has_adamw: bool,

    /// Whether this backend provides fused RoPE.
    pub has_rope: bool,

    /// Whether this backend provides fused norm+LoRA.
    pub has_norm_lora: bool,

    /// Whether this backend provides fused LoRA forward/backward.
    pub has_lora: bool,

    /// Whether this backend supports batched (depthwise) GEMM accumulate.
    pub has_dw_gemm: bool,

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
            has_flash_attention: true,
            has_mpp_flash_attention: false,
            has_swiglu: true,
            has_adamw: true,
            has_rope: true,
            has_norm_lora: true,
            has_lora: true,
            has_dw_gemm: true,
            gemm_k_alignment: 1,
            gemm_min_m: 1,
        }
    }

    /// Capabilities for the Metal 4 / MPP backend (M5+ only).
    ///
    /// Handles large GEMM shapes with NAX acceleration. Routes decode (M=1)
    /// and unaligned K back to Metal 3.
    ///
    /// # Wired MPP operations
    ///
    /// The following flags are `true` because the corresponding trait methods
    /// in [`Metal4Backend`](crate::metal4::Metal4Backend) now dispatch through MPP kernel structs:
    ///
    /// - `has_gemm`: `MppGemm` (large prefill / training shapes)
    /// - `has_swiglu`: `MppFusedSwiGLU` (base and LoRA paths both use MPP kernels)
    /// - `has_rope`: `MppFusedRoPE`
    /// - `has_norm_lora`: `MppFusedNormLora`
    /// - `has_lora`: `MppFusedLora` (inference mode)
    /// - `has_dw_gemm`: `MppDwGemm`
    pub const fn metal4() -> Self {
        Self {
            name: "Metal4",
            has_gemm: true,
            // Wired: MPP forward (MppFlashAttention) and backward (MppFlashAttentionBackward).
            // D=128 causal backward uses native MPP kernels; other dims fall back to Metal 3.
            has_flash_attention: true,
            has_mpp_flash_attention: true,
            // Wired: MppFusedSwiGLU (LoRA path now uses MPP variant).
            has_swiglu: true,
            has_adamw: true,
            // Wired: MppFusedRoPE.
            has_rope: true,
            // Wired: MppFusedNormLora.
            has_norm_lora: true,
            // Wired: MppFusedLora (inference mode).
            has_lora: true,
            // Wired: MppDwGemm (synchronous dispatch, bypasses BatchedCommandBuffer).
            has_dw_gemm: true,
            gemm_k_alignment: 32,
            gemm_min_m: 2,
        }
    }
}

// ============================================================================
// Input descriptor types
// ============================================================================

/// Descriptor for a standard GEMM: D = alpha * `A[M,K]` @ `B[N,K]^T` + beta * C
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
///
/// # Routing
///
/// Before calling any GEMM method, callers should invoke `should_handle_gemm` to
/// confirm the backend can efficiently handle the shape. If it returns `false`, the
/// caller should fall back to the other backend.
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

    /// Encode a weight-gradient GEMM into an existing batched command buffer.
    ///
    /// C = alpha * `A[M,K]` @ `B[N,K]^T` + beta * C (read-modify-write).
    /// Used for the backward pass dW computation across all layers.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the kernel's buffers one-to-one"
    )]
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
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the kernel's buffers one-to-one"
    )]
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
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the kernel's buffers one-to-one"
    )]
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
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the kernel's buffers one-to-one"
    )]
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
}
