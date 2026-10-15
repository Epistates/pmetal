//! Metal 4 / MPP kernel backend.
//!
//! [`Metal4Backend`] implements [`KernelBackend`] using MPP kernel dispatch
//! structs for operations that have NAX-accelerated implementations. Methods
//! that cannot be cleanly translated to the MPP API retain the [`Metal3Backend`]
//! fallback; each such delegation is annotated with the reason.

use std::sync::Arc;

use half::f16;
use objc2_metal::MTLCommandBuffer;

use crate::{
    backend::{AdamWDescriptor, BackendCaps, GemmDescriptor, KernelBackend},
    buffer::{AsMetalBuffer, BufferUsage, MetalBuffer},
    context::MetalContext,
    error::Result,
    kernels::{
        flash_attention::{FlashAttentionConfig, FlashAttentionOutput},
        fused_lora::{FusedLoraConfig, FusedLoraOutput},
        fused_norm_lora::{FusedNormLoraConfig, FusedNormLoraOutput},
        fused_rope::FusedRoPEConfig,
        fused_swiglu::{FusedMLPOutput, FusedSwiGLUConfig, FusedSwiGLUOutput},
        fused_training::BatchedCommandBuffer,
        mpp_dw_gemm::{MppDwGemm, MppDwGemmConfig},
        mpp_flash_attention::{MppFlashAttentionBackward, MppFlashAttentionConfig as MppFAConfig},
        mpp_fused_lora::{MppFusedLora, MppFusedLoraConfig},
        mpp_fused_norm_lora::{MppFusedNormLora, MppFusedNormLoraConfig},
        mpp_fused_rope::{MppFusedRoPE, MppFusedRoPEConfig},
        mpp_fused_swiglu::{MppFusedMLP, MppFusedMLPConfig, MppFusedSwiGLU, MppFusedSwiGLUConfig},
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
/// defined in `crates/pmetal-metal/src/kernels/mpp_*.rs`; the rest keep the
/// [`Metal3Backend`] fallback.
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

    // ---- Routing hints ------------------------------------------------------

    /// Override the default `should_handle_gemm` to add the problem-size
    /// heuristic from `DeviceProperties::should_consider_mpp_gemm`.
    fn should_handle_gemm(&self, m: usize, n: usize, k: usize) -> bool {
        let caps = self.caps();
        if m < caps.gemm_min_m || (caps.gemm_k_alignment > 1 && k % caps.gemm_k_alignment != 0) {
            return false;
        }
        self.ctx
            .properties()
            .should_consider_mpp_gemm(m, n, k, true)
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
        // MPP GEMM is handled by mpp_gemm.rs, wired in Task 11.
        self.fallback.gemm(ctx, desc, a, b, c_or_d)
    }

    fn dw_gemm_accum(
        &self,
        _batch: &mut BatchedCommandBuffer,
        a: &MetalBuffer<f32>,
        b: &MetalBuffer<f32>,
        c: &MetalBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        // MppDwGemm creates its own command buffer; it cannot encode into the
        // caller's BatchedCommandBuffer. We dispatch it synchronously here and
        // ignore the `_batch` parameter. This is semantically correct — dW
        // accumulation completes before the next layer's backward kernel starts.
        let config = MppDwGemmConfig {
            m,
            n,
            k,
            alpha,
            beta,
        };
        let dispatcher = MppDwGemm::new(self.ctx.clone(), config);
        if dispatcher.is_available() {
            dispatcher.execute(a, b, c)
        } else {
            self.fallback
                .dw_gemm_accum(_batch, a, b, c, m, n, k, alpha, beta)
        }
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
        // MPP flash attention is handled by mpp_flash_attention.rs (Task 13).
        // Metal 4 does NOT advertise has_flash_attention; the MPP path is separate.
        self.fallback
            .flash_attention_forward(ctx, config, queries, keys, values)
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
        // Attempt the MPP backward path (D=128 causal only; falls back otherwise).
        let mpp_config = MppFAConfig {
            batch_size: config.batch_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            query_seq_len: config.query_seq_len,
            kv_seq_len: config.kv_seq_len,
            head_dim: config.head_dim,
            scale: config.scale,
            is_causal: config.is_causal,
            sliding_window: config.sliding_window,
            softcap: config.softcap,
        };

        if let Ok(dispatcher) = MppFlashAttentionBackward::new(self.ctx.clone(), mpp_config) {
            if let Ok(Some(bwd)) =
                dispatcher.backward(queries, keys, values, output, d_output, logsumexp)
            {
                return Ok((bwd.d_queries, bwd.d_keys, bwd.d_values));
            }
        }

        // Fall back to Metal 3 for unsupported head dims / non-causal.
        self.fallback.flash_attention_backward(
            ctx, config, queries, keys, values, output, d_output, logsumexp,
        )
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
        // Route all cases (base and LoRA) through MPP. Build the config with
        // LoRA fields populated when all four adapter matrices are provided.
        let has_lora = gate_lora_a.is_some()
            && gate_lora_b.is_some()
            && up_lora_a.is_some()
            && up_lora_b.is_some();

        let mpp_config = MppFusedSwiGLUConfig {
            batch_size: config.batch_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            use_fp16: config.use_fp16,
            lora_rank: if has_lora { config.lora_rank } else { 0 },
            lora_scale: config.lora_scale,
        };
        let dispatcher = MppFusedSwiGLU::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            return self.fallback.fused_swiglu(
                ctx,
                config,
                input,
                gate_weight,
                up_weight,
                gate_lora_a,
                gate_lora_b,
                up_lora_a,
                up_lora_b,
            );
        }

        let output = MetalBuffer::<f32>::new(
            &self.ctx,
            config.batch_size * config.intermediate_size,
            BufferUsage::Shared,
        )?;

        if let (Some(gate_lora_a), Some(gate_lora_b), Some(up_lora_a), Some(up_lora_b)) =
            (gate_lora_a, gate_lora_b, up_lora_a, up_lora_b)
        {
            dispatcher.execute_lora(
                input,
                gate_weight,
                up_weight,
                gate_lora_a,
                gate_lora_b,
                up_lora_a,
                up_lora_b,
                &output,
            )?;
        } else {
            dispatcher.execute(input, gate_weight, up_weight, &output)?;
        }

        Ok(FusedSwiGLUOutput { output })
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
        let mpp_config = MppFusedMLPConfig::new(
            config.batch_size,
            config.hidden_size,
            config.intermediate_size,
        );
        let dispatcher = MppFusedMLP::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            return self.fallback.fused_mlp(
                ctx,
                config,
                input,
                gate_weight,
                up_weight,
                down_weight,
            );
        }

        let output = MetalBuffer::<f32>::new(
            &self.ctx,
            config.batch_size * config.hidden_size,
            BufferUsage::Shared,
        )?;
        dispatcher.execute(input, gate_weight, up_weight, down_weight, &output)?;
        Ok(FusedMLPOutput { output })
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
        let mpp_config = MppFusedNormLoraConfig {
            batch_size: config.batch_size,
            hidden_size: config.hidden_size,
            out_features: config.out_features,
            lora_rank: config.lora_rank,
            eps: config.eps,
            lora_scale: config.lora_scale,
        };
        let dispatcher = MppFusedNormLora::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            return self
                .fallback
                .fused_norm_lora(ctx, config, input, gamma, weight, lora_a, lora_b);
        }

        let output = MetalBuffer::<f32>::new(
            &self.ctx,
            config.batch_size * config.out_features,
            BufferUsage::Shared,
        )?;
        dispatcher.execute(input, gamma, weight, lora_a, lora_b, &output)?;
        Ok(FusedNormLoraOutput { output })
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
        let mpp_config = MppFusedLoraConfig::new_inference(
            config.batch_size,
            config.in_features,
            config.out_features,
            config.rank,
            config.scale,
        );
        let dispatcher = MppFusedLora::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            return self
                .fallback
                .fused_lora_forward(ctx, config, x, weight, lora_a, lora_b);
        }

        let output = MetalBuffer::<f16>::new(
            &self.ctx,
            config.batch_size * config.out_features,
            BufferUsage::Shared,
        )?;
        dispatcher.execute_inference(x, weight, lora_a, lora_b, &output)?;
        Ok(FusedLoraOutput {
            output,
            intermediate: None,
        })
    }

    // ---- Training optimizers and losses -------------------------------------

    fn fused_adamw_step(
        &self,
        batch: &mut BatchedCommandBuffer,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        self.fallback.fused_adamw_step(batch, desc)
    }

    fn fused_adamw_step_standalone(
        &self,
        ctx: &Arc<MetalContext>,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        self.fallback.fused_adamw_step_standalone(ctx, desc)
    }

    fn fused_rope(
        &self,
        ctx: &Arc<MetalContext>,
        config: &FusedRoPEConfig,
        queries: &mut MetalBuffer<f32>,
        keys: Option<&mut MetalBuffer<f32>>,
        position_ids: Option<&MetalBuffer<i32>>,
    ) -> Result<()> {
        let mpp_config = MppFusedRoPEConfig {
            batch_size: config.batch_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            seq_len: config.seq_len,
            head_dim: config.head_dim,
            base: config.base,
            scale: config.scale,
            use_fp16: config.use_fp16,
        };
        let dispatcher = MppFusedRoPE::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            return self
                .fallback
                .fused_rope(ctx, config, queries, keys, position_ids);
        }

        match (keys, position_ids) {
            (Some(k), Some(pos_ids)) => {
                // Fused QK RoPE with custom position IDs.
                dispatcher
                    .apply_qk_inplace_async(queries, k, Some(pos_ids as &dyn AsMetalBuffer))?
                    .waitUntilCompleted();
            }
            (Some(k), None) => {
                // Fused QK RoPE, sequential positions.
                dispatcher.apply_qk_inplace(queries, k)?;
            }
            (None, Some(pos_ids)) => {
                // Q-only RoPE with custom position IDs.
                dispatcher.apply_with_positions(queries, pos_ids)?;
            }
            (None, None) => {
                // Q-only RoPE, sequential positions.
                dispatcher.apply_inplace(queries)?;
            }
        }

        Ok(())
    }
}
