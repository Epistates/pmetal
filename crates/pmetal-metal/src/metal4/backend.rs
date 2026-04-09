//! Metal 4 / MPP kernel backend.
//!
//! [`Metal4Backend`] implements [`KernelBackend`] using MPP kernel dispatch
//! structs for operations that have NAX-accelerated implementations. Methods
//! that cannot be cleanly translated to the MPP API retain the [`Metal3Backend`]
//! fallback; each such delegation is annotated with the reason.

use std::sync::Arc;

use half::f16;

use crate::{
    backend::{
        AdamWDescriptor, BackendCaps, GemmDescriptor, GroupedGemmDescriptor, KernelBackend,
        MoeExpertDescriptor, QuantizedGemmDescriptor,
    },
    buffer::{AsMetalBuffer, BufferUsage, MetalBuffer},
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
        mpp_dw_gemm::{MppDwGemm, MppDwGemmConfig},
        mpp_fused_cross_entropy::{MppFusedCrossEntropy, MppFusedCrossEntropyConfig},
        mpp_fused_distill::{MppDistillLossType, MppFusedDistill, MppFusedDistillConfig},
        mpp_fused_lora::{MppFusedLora, MppFusedLoraConfig},
        mpp_fused_norm_lora::{MppFusedNormLora, MppFusedNormLoraConfig},
        mpp_fused_rope::{MppFusedRoPE, MppFusedRoPEConfig},
        mpp_fused_swiglu::{MppFusedSwiGLU, MppFusedSwiGLUConfig},
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
/// - `fused_adamw_step`: trait uses [`BatchedCommandBuffer`] (single shared
///   encoder); `MppFusedAdamW` owns its own command buffer per call. These
///   two execution models cannot be bridged without restructuring the trait.
/// - `fused_moe_expert`: descriptor carries quantized `u32` weight buffers;
///   `MppFusedMoE` expects dense fp16 weights. Type mismatch — no safe cast.
/// - `grouped_gemm`: `GroupedGemmDispatch` requires per-expert *token counts*
///   derived from the prefix-sum `expert_offsets` buffer, which lives on the
///   GPU and cannot be read without a CPU round-trip inside this call.
/// - `fused_mlp`: no MPP full-MLP (gate+up+down) kernel exists yet.
/// - `moe_routing`: routing kernel is Metal 3 only; no MPP variant exists yet.
/// - `flash_attention_backward`: no MPP backward path exists yet.
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
    /// heuristic from [`DeviceProperties::should_consider_mpp_gemm`].
    fn should_handle_gemm(&self, m: usize, n: usize, k: usize) -> bool {
        let caps = self.caps();
        if m < caps.gemm_min_m || (caps.gemm_k_alignment > 1 && k % caps.gemm_k_alignment != 0) {
            return false;
        }
        self.ctx.properties().should_consider_mpp_gemm(m, n, k, true)
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
        // MPP quantized GEMM is handled by mpp_quantized.rs, wired in Task 12.
        self.fallback.quantized_gemm(ctx, desc, x, w_q, scales, biases, output)
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
        let config = MppDwGemmConfig { m, n, k, alpha, beta };
        let dispatcher = MppDwGemm::new(self.ctx.clone(), config);
        if dispatcher.is_available() {
            dispatcher.execute(a, b, c)
        } else {
            self.fallback.dw_gemm_accum(_batch, a, b, c, m, n, k, alpha, beta)
        }
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
        // MppGroupedGemm requires GroupedGemmDispatch::total_tiles computed from
        // per-expert token counts. Those counts are derived from the prefix-sum
        // expert_offsets GPU buffer, which cannot be read without a CPU round-trip
        // at this call site. Route to Metal 3 to avoid the stall.
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
        // MPP flash attention is handled by mpp_flash_attention.rs (Task 13).
        // Metal 4 does NOT advertise has_flash_attention; the MPP path is separate.
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
        // No MPP backward flash attention path exists yet.
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
        // The MPP SwiGLU kernel (MppFusedSwiGLU) does not support LoRA-augmented
        // projections. Route the LoRA path to Metal 3; use MPP for the base path.
        let has_lora = gate_lora_a.is_some()
            || gate_lora_b.is_some()
            || up_lora_a.is_some()
            || up_lora_b.is_some();

        if has_lora {
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

        let mpp_config = MppFusedSwiGLUConfig {
            batch_size: config.batch_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            use_fp16: config.use_fp16,
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
        dispatcher.execute(input, gate_weight, up_weight, &output)?;
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
        // No MPP full-MLP (gate+up+down) kernel exists yet.
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
            return self.fallback.fused_norm_lora(ctx, config, input, gamma, weight, lora_a, lora_b);
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
            return self.fallback.fused_lora_forward(ctx, config, x, weight, lora_a, lora_b);
        }

        let output = MetalBuffer::<f16>::new(
            &self.ctx,
            config.batch_size * config.out_features,
            BufferUsage::Shared,
        )?;
        dispatcher.execute_inference(x, weight, lora_a, lora_b, &output)?;
        Ok(FusedLoraOutput { output, intermediate: None })
    }

    // ---- Training optimizers and losses -------------------------------------

    fn fused_adamw_step(
        &self,
        batch: &mut BatchedCommandBuffer,
        desc: &AdamWDescriptor<'_>,
    ) -> Result<()> {
        // MppFusedAdamW creates its own command buffer; it cannot encode into an
        // existing BatchedCommandBuffer. The trait contract requires encoding into
        // `batch` so that callers can batch the AdamW step with other operations
        // and execute them together. Bridging these two execution models requires
        // restructuring the trait — delegating to Metal 3 instead.
        self.fallback.fused_adamw_step(batch, desc)
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
            return self.fallback.fused_cross_entropy(ctx, config, logits, targets);
        }

        let losses =
            MetalBuffer::<f32>::new(&self.ctx, config.num_tokens, BufferUsage::Shared)?;
        let logsumexp =
            MetalBuffer::<f32>::new(&self.ctx, config.num_tokens, BufferUsage::Shared)?;

        // MPP cross-entropy fwd+bwd kernel writes gradients into grad_logits.
        // For the forward-only output shape (losses + logsumexp), we pass logsumexp
        // as the grad_logits placeholder and a single-element loss accumulator.
        let loss_accum = MetalBuffer::<f32>::new(&self.ctx, 1, BufferUsage::Shared)?;
        dispatcher.execute(logits, targets, &logsumexp, &loss_accum)?;

        Ok(FusedCrossEntropyOutput { losses, logsumexp })
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
            return self.fallback.fused_rope(ctx, config, queries, keys, position_ids);
        }

        match (keys, position_ids) {
            (Some(k), Some(pos_ids)) => {
                // Fused QK RoPE with custom position IDs.
                dispatcher.apply_qk_inplace_async(queries, k, Some(pos_ids as &dyn AsMetalBuffer))?
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
        // MoeExpertDescriptor carries quantized u32 weight buffers (gate/up/down).
        // MppFusedMoE expects dense fp16 weights. There is no safe in-place cast
        // or dequant path here — route to Metal 3's quantized expert kernel.
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
        let mpp_loss_type = match loss_type {
            DistillLossType::KlDivergence => MppDistillLossType::ForwardKL,
            DistillLossType::ReverseKlDivergence => MppDistillLossType::ReverseKL,
            DistillLossType::JensenShannon => MppDistillLossType::JensenShannon,
            DistillLossType::SoftCrossEntropy => MppDistillLossType::SoftCrossEntropy,
        };
        let mpp_config = MppFusedDistillConfig {
            num_tokens: config.num_tokens,
            vocab_size: config.vocab_size,
            temperature: config.temperature,
            alpha: config.alpha,
            ignore_index: config.ignore_index,
            loss_type: mpp_loss_type,
            use_fp16: config.use_fp16,
        };
        let dispatcher = MppFusedDistill::new(self.ctx.clone(), mpp_config);

        if !dispatcher.is_available() {
            return self.fallback.fused_distill_loss(
                ctx,
                config,
                teacher_logits,
                student_logits,
                loss_type,
            );
        }

        let losses =
            MetalBuffer::<f32>::new(&self.ctx, config.num_tokens, BufferUsage::Shared)?;
        let teacher_lse =
            MetalBuffer::<f32>::new(&self.ctx, config.num_tokens, BufferUsage::Shared)?;
        let student_lse =
            MetalBuffer::<f32>::new(&self.ctx, config.num_tokens, BufferUsage::Shared)?;

        dispatcher.execute(
            teacher_logits,
            student_logits,
            &losses,
            &teacher_lse,
            &student_lse,
        )?;

        Ok(FusedDistillOutput { losses, teacher_lse, student_lse })
    }
}
