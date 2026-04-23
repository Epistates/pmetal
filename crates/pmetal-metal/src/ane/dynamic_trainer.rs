#![allow(unsafe_code)]

//! Dynamic weight ANE trainer — compile once, train forever.
//!
//! Replaces the static trainer that recompiled all kernels every N steps.
//! Instead, 9 kernels compile once at startup; weight updates are `memcpy`
//! into fp32 IOSurfaces. Zero recompilation after init.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────┐    ┌───────────────────┐    ┌──────────┐
//! │  CPU (vDSP)    │    │ IOSurface (fp32)   │    │   ANE    │
//! │  RMSNorm fwd   │───►│ act + weights      │───►│ matmul   │
//! │  SiLU deriv    │    │ per-ch interleaved  │    │ softmax  │
//! │  CrossEntropy  │◄───│ output results      │◄───│ sigmoid  │
//! │  Adam          │    └───────────────────┘    └──────────┘
//! │  cblas dW      │
//! └────────────────┘
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tracing::{debug, info};

use crate::accelerate;
use crate::ane::dynamic_kernel::{self, DynamicKernelConfig, DynamicKernelOutput};
use crate::ane::iosurface::IoSurface;
use crate::ane::kernel::TransformerKernelConfig;
use crate::ane::loss::{AneTrainingLoss, CrossEntropyLoss};
use crate::ane::runtime::{AneModel, AneRuntime};
use crate::ane::scratch::{
    BackwardScratch, BackwardScratchIds, backward_scratch_entries, backward_scratch_ids,
};
use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::MetalContext;
use crate::error::{MetalError, Result};
use crate::kernels::dw_gemm::{DwGemm, GPU_DW_MIN_DIM};
use crate::kernels::fused_training::BatchedCommandBuffer;

/// Configuration for the dynamic ANE trainer.
///
/// No `max_compiles` or `exhaustion_strategy` — those are irrelevant when
/// kernels compile exactly once. `accum_steps` controls gradient accumulation
/// before each Adam update.
#[derive(Debug, Clone)]
pub struct DynamicAneTrainerConfig {
    /// Model dimension.
    pub dim: usize,
    /// FFN hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of key/value heads (GQA/MQA). Defaults to `n_heads` (MHA).
    pub n_kv_heads: usize,
    /// Per-head dimension. If `None`, computed as `dim / n_heads`.
    /// Models like Qwen3 use `head_dim=128` even when `dim/n_heads=64`.
    pub head_dim: Option<usize>,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Learning rate.
    pub learning_rate: f32,
    /// Adam beta1.
    pub adam_beta1: f32,
    /// Adam beta2.
    pub adam_beta2: f32,
    /// Adam epsilon.
    pub adam_eps: f32,
    /// Gradient clipping norm.
    pub gradient_clip_norm: f32,
    /// Number of gradient accumulation steps per batch.
    pub accum_steps: usize,
    /// Warmup steps for cosine LR schedule.
    pub warmup_steps: usize,
    /// Minimum LR ratio for cosine decay.
    pub min_lr_ratio: f32,
    /// RMSNorm epsilon (must match ANE kernel eps). Default: 1e-6.
    pub rms_norm_eps: f32,
    /// Loss scaling factor. Multiplies dlogits before backward and divides
    /// gradients after accumulation, preventing fp32 underflow for small
    /// gradient magnitudes at >350M params. Default: 1.0 (disabled).
    pub loss_scale: f32,
    /// Optional separate learning rate for embeddings. If `None`, uses base LR.
    /// Embeddings benefit from lower LR to prevent divergence from sparse
    /// token updates and magnitude drift through many layers.
    pub embedding_lr: Option<f32>,
}

impl Default for DynamicAneTrainerConfig {
    fn default() -> Self {
        Self {
            dim: 768,
            hidden_dim: 2048,
            n_heads: 12,
            n_kv_heads: 12,
            head_dim: None,
            n_layers: 12,
            vocab_size: 32000,
            seq_len: 256,
            learning_rate: 3e-4,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            gradient_clip_norm: 1.0,
            accum_steps: 10,
            warmup_steps: 100,
            min_lr_ratio: 0.1,
            rms_norm_eps: 1e-6,
            loss_scale: 1.0,
            embedding_lr: None,
        }
    }
}

/// Vocabulary compaction map for classifier speedup.
///
/// Reduces full vocab (e.g. 32K) to only tokens actually seen in training data
/// (e.g. ~9K), yielding ~3.5x speedup on the classifier matmul + cross-entropy.
/// Built once at training start from all batch tokens.
pub struct VocabMap {
    /// Number of active (compacted) tokens.
    pub compact_vocab: usize,
    /// Maps full vocab id → compact id (-1 if token unused).
    full_to_compact: Vec<i32>,
    /// Maps compact id → full vocab id.
    compact_to_full: Vec<usize>,
}

impl VocabMap {
    /// Build a VocabMap by scanning all tokens in the training batches.
    pub fn from_batches(batches: &[Vec<(Vec<u16>, Vec<u16>)>], full_vocab: usize) -> Self {
        let mut used = vec![false; full_vocab];
        for batch in batches {
            for (input, target) in batch {
                for &tok in input.iter().chain(target.iter()) {
                    let idx = tok as usize;
                    if idx < full_vocab {
                        used[idx] = true;
                    }
                }
            }
        }
        Self::from_used(used)
    }

    /// Build a VocabMap from u32 token IDs (supports vocab > 65536).
    ///
    /// Use this for models like Qwen3 (vocab_size=151936) where token IDs
    /// exceed the u16 range. After building the map, call `remap_u32` to
    /// convert u32 IDs to compact u16 IDs for ANE consumption.
    pub fn from_token_ids(token_ids: &[u32], full_vocab: usize) -> Self {
        let mut used = vec![false; full_vocab];
        for &tok in token_ids {
            let idx = tok as usize;
            if idx < full_vocab {
                used[idx] = true;
            }
        }
        Self::from_used(used)
    }

    fn from_used(used: Vec<bool>) -> Self {
        let full_vocab = used.len();
        let mut full_to_compact = vec![-1i32; full_vocab];
        let mut compact_to_full = Vec::new();
        for (i, &u) in used.iter().enumerate() {
            if u {
                full_to_compact[i] = compact_to_full.len() as i32;
                compact_to_full.push(i);
            }
        }
        let compact_vocab = compact_to_full.len();
        Self {
            compact_vocab,
            full_to_compact,
            compact_to_full,
        }
    }

    /// Remap a slice of full-vocab token ids to compact ids.
    pub fn remap_tokens(&self, tokens: &[u16]) -> Vec<u16> {
        tokens
            .iter()
            .map(|&t| {
                let c = self.full_to_compact[t as usize];
                debug_assert!(c >= 0, "Token {} not in VocabMap", t);
                c as u16
            })
            .collect()
    }

    /// Remap u32 full-vocab token ids to compact u16 ids.
    ///
    /// For large-vocab models (vocab > 65536), this avoids the u16 truncation
    /// that would corrupt token IDs if cast directly.
    pub fn remap_u32(&self, tokens: &[u32]) -> Vec<u16> {
        tokens
            .iter()
            .map(|&t| {
                let idx = t as usize;
                debug_assert!(
                    idx < self.full_to_compact.len(),
                    "Token {} out of range (vocab={})",
                    t,
                    self.full_to_compact.len()
                );
                let c = self.full_to_compact[idx];
                debug_assert!(c >= 0, "Token {} not in VocabMap", t);
                c as u16
            })
            .collect()
    }
}

impl DynamicAneTrainerConfig {
    /// Check if a model architecture is compatible with ANE training/inference.
    ///
    /// Rejects hybrid/recurrent architectures (GDN, Mamba, RG-LRU) and MoE models
    /// that cannot be mapped to ANE's transformer-only kernel set.
    pub fn is_ane_compatible(config_json: &serde_json::Value) -> std::result::Result<(), String> {
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Reject known incompatible architectures
        const INCOMPATIBLE: &[&str] = &[
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_next",     // GDN hybrid
            "nemotron_h",     // Mamba hybrid
            "recurrentgemma", // RG-LRU hybrid
            "jamba",          // Mamba hybrid
            "gemma4",         // extra per-layer norms / KV-sharing / PLI blocks
            "gemma4_text",
        ];
        if INCOMPATIBLE.contains(&model_type) {
            return Err(format!(
                "Model type '{}' uses hybrid/recurrent layers not supported by ANE. Use GPU training.",
                model_type
            ));
        }

        // Reject MoE (no expert routing on ANE)
        if config_json
            .get("num_experts")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            > 0
        {
            return Err("MoE models with routed experts are not supported by ANE training.".into());
        }
        if config_json
            .get("num_local_experts")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            > 0
        {
            return Err("MoE models with routed experts are not supported by ANE training.".into());
        }

        Ok(())
    }
}

/// Per-layer weight storage (f32, row-major).
struct LayerWeights {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    rms_att: Vec<f32>,
    rms_ffn: Vec<f32>,
    // Transposed weight buffers for backward kernels
    wq_t: Vec<f32>,
    wk_t: Vec<f32>,
    wv_t: Vec<f32>,
    wo_t: Vec<f32>,
    w1_t: Vec<f32>,
    w2_t: Vec<f32>,
    w3_t: Vec<f32>,
}

/// Adam state for a single parameter.
struct AdamParam {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamParam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }
}

/// Per-layer Adam optimizer state.
struct LayerAdamState {
    wq: AdamParam,
    wk: AdamParam,
    wv: AdamParam,
    wo: AdamParam,
    w1: AdamParam,
    w2: AdamParam,
    w3: AdamParam,
    rms_att: AdamParam,
    rms_ffn: AdamParam,
}

/// Per-layer gradient accumulators.
///
/// Weight gradient fields (wq..w3) are `MetalBuffer<f32>` with `Shared` storage,
/// enabling zero-copy access from both the Metal GPU (dW GEMM accumulation) and
/// CPU (scale_inplace, sum_of_squares, adam_update via `.as_mut_slice()`).
///
/// RMSNorm gradient fields remain `Vec<f32>` because they are tiny (dim elements)
/// and only written on CPU.
struct LayerGradients {
    wq: MetalBuffer<f32>,
    wk: MetalBuffer<f32>,
    wv: MetalBuffer<f32>,
    wo: MetalBuffer<f32>,
    w1: MetalBuffer<f32>,
    w2: MetalBuffer<f32>,
    w3: MetalBuffer<f32>,
    rms_att: Vec<f32>,
    rms_ffn: Vec<f32>,
}

impl LayerGradients {
    fn zero(&mut self) {
        self.wq.as_mut_slice().fill(0.0);
        self.wk.as_mut_slice().fill(0.0);
        self.wv.as_mut_slice().fill(0.0);
        self.wo.as_mut_slice().fill(0.0);
        self.w1.as_mut_slice().fill(0.0);
        self.w2.as_mut_slice().fill(0.0);
        self.w3.as_mut_slice().fill(0.0);
        self.rms_att.fill(0.0);
        self.rms_ffn.fill(0.0);
    }
}

/// Per-layer saved activations (f32, channel-first [D, S]).
struct LayerActivations {
    layer_in: Vec<f32>,
    /// Pre-norm for the attention block: RMSNorm(layer_in).
    xnorm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    /// Residual after attention: x2 = layer_in + o_out.
    x2: Vec<f32>,
    /// Pre-norm for the FFN block: RMSNorm(x2).
    ///
    /// Computed on CPU so the backward pass can form dW1 = dh1 @ x2norm^T
    /// and dW3 = dh3 @ x2norm^T correctly. The fused ANE FFN kernel also
    /// computes this norm internally but does not surface it as a tap.
    x2norm: Vec<f32>,
    h1: Vec<f32>,
    h3: Vec<f32>,
    silu_out: Vec<f32>,
    ffn_out: Vec<f32>,
}

/// Step timing metrics for dashboard integration.
#[derive(Debug, Clone, Default)]
pub struct StepTimings {
    /// Time spent in RMSNorm forward/backward (CPU vDSP).
    pub rmsnorm_us: u64,
    /// Time spent in ANE forward passes.
    pub ane_fwd_us: u64,
    /// Time spent in ANE backward passes.
    pub ane_bwd_us: u64,
    /// Time spent writing to IOSurfaces.
    pub io_write_us: u64,
    /// Time spent reading from IOSurfaces.
    pub io_read_us: u64,
    /// Time spent in async cblas dW accumulation.
    pub cblas_dw_us: u64,
    /// Time spent in Adam optimizer update.
    pub adam_us: u64,
    /// Total step time.
    pub total_us: u64,
}

/// Compiled ANE kernels (shared across all layers).
///
/// Uses decomposed single-projection kernels indexed by `(IC, OC)` shape,
/// plus attention-only and backward kernels. Each projection is a simple
/// single-matmul kernel that compiles reliably at any model scale.
struct DynamicKernels {
    /// Projection kernels indexed by (input_channels, output_channels).
    projections: HashMap<(usize, usize), AneModel>,
    /// Fused SDPA forward (RMSNorm + QKV + Attn + Wo).
    /// `None` when decomposed attention is used (attention matrix too large for ANE SRAM).
    sdpa_fwd: Option<AneModel>,
    /// Fused FFN forward (RMSNorm + W1 + W3 + SiLU + W2).
    /// `None` when decomposed FFN is used (hidden_dim too large for ANE SRAM).
    ffn_fwd: Option<AneModel>,
    /// SDPA backward part 1 (dV, probs, dp).
    /// `None` when decomposed attention is used.
    sdpa_bwd1: Option<AneModel>,
    /// SDPA backward part 2 (dQ, dK).
    /// `None` when decomposed attention is used.
    sdpa_bwd2: Option<AneModel>,
    /// FFN backward W2^T.
    ffn_bwd_w2t: AneModel,
    /// FFN backward W1^T + W3^T.
    ffn_bwd_w13t: AneModel,
    /// Softmax kernel for cross-entropy.
    softmax: Option<AneModel>,
}

/// IOSurface pool for the decomposed dynamic pipeline.
struct DynIoPool {
    /// Projection IO surfaces indexed by (IC, OC).
    proj_inputs: HashMap<(usize, usize), IoSurface>,
    proj_outputs: HashMap<(usize, usize), IoSurface>,
    /// Fused forward IOSurfaces.
    sdpa_fwd_in: IoSurface,
    sdpa_fwd_out: IoSurface,
    ffn_fwd_in: IoSurface,
    ffn_fwd_out: IoSurface,
    /// SDPA backward IO surfaces (fp16).
    sdpa_bwd1_in: IoSurface,
    sdpa_bwd1_out: IoSurface,
    sdpa_bwd2_in: IoSurface,
    sdpa_bwd2_out: IoSurface,
    /// FFN backward IO surfaces.
    ffn_bwd_w2t_in: IoSurface,
    ffn_bwd_w2t_out: IoSurface,
    ffn_bwd_w13t_in: IoSurface,
    ffn_bwd_w13t_out: IoSurface,
    /// Softmax IO surfaces (fp16, compact_vocab × seq).
    softmax_in: Option<IoSurface>,
    softmax_out: Option<IoSurface>,
}

/// Dynamic weight ANE trainer. Compiles 9+ kernels once, then trains forever.
pub struct DynamicAneTrainer {
    config: DynamicAneTrainerConfig,
    kernel_config: TransformerKernelConfig,
    layer_weights: Vec<LayerWeights>,
    kernels: Option<DynamicKernels>,
    io_pool: Option<DynIoPool>,
    layer_acts: Vec<LayerActivations>,
    layer_grads: Vec<LayerGradients>,
    layer_adam: Vec<LayerAdamState>,
    embed_weights: Vec<f32>,
    embed_grad: Vec<f32>,
    embed_adam: AdamParam,
    rms_final: Vec<f32>,
    rms_final_grad: Vec<f32>,
    rms_final_adam: AdamParam,
    adam_t: usize,
    compile_count: usize,
    /// GPU dW GEMM dispatcher (None if dim < GPU_DW_MIN_DIM or Metal init fails).
    gpu_dw: Option<DwGemm>,
    /// Pre-allocated scratch buffers for activation copies (A and B operands).
    /// Sized to max(h*s, d*s, qd*s, v*s) to fit any per-layer activation.
    scratch_a: Option<MetalBuffer<f32>>,
    scratch_b: Option<MetalBuffer<f32>>,
    /// Metal context for GPU dW path.
    metal_ctx: Option<Arc<MetalContext>>,
    /// Latest step timings.
    pub last_timings: StepTimings,
    /// Pre-allocated backward scratch pool (initialized in compile_kernels).
    bwd_scratch: Option<BackwardScratch>,
    /// Scratch region IDs for backward_layer.
    bwd_ids: Option<BackwardScratchIds>,
    /// Pluggable loss function (default: CrossEntropyLoss).
    loss_fn: Option<Box<dyn AneTrainingLoss>>,
    /// Use decomposed attention (ANE projections + Accelerate BLAS attention)
    /// instead of fused SDPA kernel. Enabled automatically when the attention
    /// matrix exceeds ANE SRAM capacity (~4 MB threshold).
    decomposed_attn: bool,
    /// Use decomposed FFN (ANE projections + CPU SiLU) instead of fused FFN
    /// kernel. Enabled automatically when hidden_dim × seq exceeds ANE SRAM.
    decomposed_ffn: bool,
    // --- Vocab compaction state ---
    /// Optional vocab compaction map (built from training data).
    vocab_map: Option<VocabMap>,
    /// Compact embedding matrix `[compact_vocab * dim]`.
    compact_embed: Vec<f32>,
    /// Compact embedding gradient accumulator.
    compact_embed_grad: Vec<f32>,
    /// Compact embedding Adam state.
    compact_embed_adam: AdamParam,
    // --- ANE throughput degradation detection ---
    /// Baseline hardware execution time (ns) from first 5 calibration steps.
    /// `None` until the calibration window is filled.
    baseline_hw_ns: Option<u64>,
    /// Ring buffer of the last 8 `hw_execution_time_ns` samples from the SDPA
    /// forward kernel (or largest fused projection kernel in decomposed mode).
    hw_ring: VecDeque<u64>,
    /// Total ANE dispatches recorded since the last kernel refresh.
    total_ane_evals: u64,
    /// Training step number at which kernels were last compiled / refreshed.
    last_refresh_step: usize,
    /// Fraction above baseline that triggers a kernel refresh (default 1.15 = 15%).
    refresh_threshold: f32,
    /// Maximum ANE dispatches before a proactive safety refresh (default 25 000).
    /// Doubled when dual-die alternation is active (Ultra chips).
    max_evals_before_refresh: u64,
    // --- Dual-die ANE alternation (UltraFusion / M3/M4/M5 Ultra) ---
    /// Second set of compiled kernels for dual-die alternation (Ultra only).
    ///
    /// The MIL programs in this set are trivially different (variant=1 no-op)
    /// from the primary set so the ANE daemon assigns them to the second die.
    kernels_b: Option<DynamicKernels>,
    /// Second IOSurface pool paired with `kernels_b`.
    io_pool_b: Option<DynIoPool>,
    /// Toggle: which kernel set to use this step (false = A, true = B).
    use_kernel_set_b: bool,
}

/// Encode a dW GEMM (`C += A @ B^T`) into a BatchedCommandBuffer (GPU path),
/// or fall back to cblas GEMM (CPU path).
///
/// Free function to avoid borrow conflicts with `backward_layer`, which holds
/// immutable borrows on `self.kernels` / `self.io_pool` while dispatching GEMMs.
#[allow(clippy::too_many_arguments)]
fn encode_dw_gemm(
    gpu_dw: Option<&DwGemm>,
    scratch_a: Option<&MetalBuffer<f32>>,
    scratch_b: Option<&MetalBuffer<f32>>,
    batch: Option<&mut BatchedCommandBuffer>,
    a_data: &[f32],
    b_data: &[f32],
    c_buf: &MetalBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
) {
    if let (Some(dw), Some(sa), Some(sb), Some(batch)) = (gpu_dw, scratch_a, scratch_b, batch) {
        // GPU path: copy activations to scratch, encode GEMM.
        // MetalBuffer has interior mutability via unified memory (as_mut_slice_unchecked).
        sa.as_mut_slice_unchecked()[..a_data.len()].copy_from_slice(a_data);
        sb.as_mut_slice_unchecked()[..b_data.len()].copy_from_slice(b_data);
        dw.queue_gemm_accum(batch, sa, sb, c_buf, m, n, k, 1.0, 1.0)
            .unwrap();
    } else {
        // CPU cblas fallback
        let c_slice = c_buf.as_mut_slice_unchecked();
        accelerate::gemm(a_data, b_data, c_slice, m, n, k, 1.0, 1.0, false, true);
    }
}

impl DynamicAneTrainer {
    /// Create a new dynamic ANE trainer.
    pub fn new(config: DynamicAneTrainerConfig) -> Self {
        let d = config.dim;
        let h = config.hidden_dim;
        let s = config.seq_len;
        let nl = config.n_layers;
        let v = config.vocab_size;
        let hd = config.head_dim.unwrap_or(d / config.n_heads);

        let nkv = if config.n_kv_heads > 0 {
            config.n_kv_heads
        } else {
            config.n_heads
        };

        let kernel_config = TransformerKernelConfig {
            dim: d,
            hidden_dim: h,
            n_heads: config.n_heads,
            n_kv_heads: nkv,
            head_dim: hd,
            seq_len: s,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        };

        // Try to initialize Metal GPU dW path
        let (metal_ctx, gpu_dw, scratch_a, scratch_b) = if d >= GPU_DW_MIN_DIM {
            match MetalContext::new() {
                Ok(ctx) => {
                    let ctx = Arc::new(ctx);
                    match DwGemm::new(ctx.clone()) {
                        Ok(dw) => {
                            // Pre-allocate two scratch buffers sized to the largest activation
                            let qd_val = config.n_heads * hd;
                            let max_scratch = [d * s, h * s, qd_val * s].into_iter().max().unwrap();
                            match (
                                MetalBuffer::new(&ctx, max_scratch, BufferUsage::Shared),
                                MetalBuffer::new(&ctx, max_scratch, BufferUsage::Shared),
                            ) {
                                (Ok(sa), Ok(sb)) => {
                                    info!(
                                        dim = d,
                                        scratch_size = max_scratch,
                                        "GPU dW path enabled (Metal GEMM)"
                                    );
                                    (Some(ctx.clone()), Some(dw), Some(sa), Some(sb))
                                }
                                _ => {
                                    info!("Scratch buffer alloc failed, falling back to cblas");
                                    (None, None, None, None)
                                }
                            }
                        }
                        Err(e) => {
                            info!("GPU dW init failed, falling back to cblas: {e}");
                            (None, None, None, None)
                        }
                    }
                }
                Err(e) => {
                    info!("MetalContext init failed, falling back to cblas: {e}");
                    (None, None, None, None)
                }
            }
        } else {
            info!(
                dim = d,
                threshold = GPU_DW_MIN_DIM,
                "dim below GPU dW threshold, using cblas"
            );
            (None, None, None, None)
        };

        let mut layer_weights = Vec::with_capacity(nl);
        let mut layer_acts = Vec::with_capacity(nl);
        let mut layer_grads = Vec::with_capacity(nl);
        let mut layer_adam = Vec::with_capacity(nl);

        let qd = kernel_config.q_dim(); // n_heads * head_dim
        let kvd = kernel_config.kv_dim(); // n_kv_heads * head_dim

        for _ in 0..nl {
            layer_weights.push(LayerWeights {
                wq: vec![0.0; qd * d],  // [q_dim, dim]
                wk: vec![0.0; kvd * d], // [kv_dim, dim]
                wv: vec![0.0; kvd * d], // [kv_dim, dim]
                wo: vec![0.0; d * qd],  // [dim, q_dim]
                w1: vec![0.0; h * d],
                w2: vec![0.0; d * h],
                w3: vec![0.0; h * d],
                rms_att: vec![0.0; d],
                rms_ffn: vec![0.0; d],
                wq_t: vec![0.0; d * qd],  // [dim, q_dim]
                wk_t: vec![0.0; d * kvd], // [dim, kv_dim]
                wv_t: vec![0.0; d * kvd], // [dim, kv_dim]
                wo_t: vec![0.0; qd * d],  // [q_dim, dim]
                w1_t: vec![0.0; h * d],
                w2_t: vec![0.0; d * h],
                w3_t: vec![0.0; h * d],
            });

            layer_acts.push(LayerActivations {
                layer_in: vec![0.0; d * s],
                xnorm: vec![0.0; d * s],
                q: vec![0.0; qd * s],        // [q_dim, seq]
                k: vec![0.0; kvd * s],       // [kv_dim, seq]
                v: vec![0.0; kvd * s],       // [kv_dim, seq]
                attn_out: vec![0.0; qd * s], // [q_dim, seq] (before Wo)
                o_out: vec![0.0; d * s],
                x2: vec![0.0; d * s],
                x2norm: vec![0.0; d * s],
                h1: vec![0.0; h * s],
                h3: vec![0.0; h * s],
                silu_out: vec![0.0; h * s],
                ffn_out: vec![0.0; d * s],
            });

            // Allocate gradient buffers as MetalBuffer (GPU-visible) or fallback Vec
            let lg = if let Some(ref ctx) = metal_ctx {
                LayerGradients {
                    wq: MetalBuffer::zeros(ctx, qd * d, BufferUsage::Shared).unwrap(),
                    wk: MetalBuffer::zeros(ctx, kvd * d, BufferUsage::Shared).unwrap(),
                    wv: MetalBuffer::zeros(ctx, kvd * d, BufferUsage::Shared).unwrap(),
                    wo: MetalBuffer::zeros(ctx, d * qd, BufferUsage::Shared).unwrap(),
                    w1: MetalBuffer::zeros(ctx, h * d, BufferUsage::Shared).unwrap(),
                    w2: MetalBuffer::zeros(ctx, d * h, BufferUsage::Shared).unwrap(),
                    w3: MetalBuffer::zeros(ctx, h * d, BufferUsage::Shared).unwrap(),
                    rms_att: vec![0.0; d],
                    rms_ffn: vec![0.0; d],
                }
            } else {
                // CPU-only fallback: create MetalBuffer from Vec (panics if no Metal,
                // but this should never happen on Apple Silicon)
                let fallback_ctx = MetalContext::new().expect("Metal required on Apple Silicon");
                let fallback_ctx = Arc::new(fallback_ctx);
                LayerGradients {
                    wq: MetalBuffer::zeros(&fallback_ctx, qd * d, BufferUsage::Shared).unwrap(),
                    wk: MetalBuffer::zeros(&fallback_ctx, kvd * d, BufferUsage::Shared).unwrap(),
                    wv: MetalBuffer::zeros(&fallback_ctx, kvd * d, BufferUsage::Shared).unwrap(),
                    wo: MetalBuffer::zeros(&fallback_ctx, d * qd, BufferUsage::Shared).unwrap(),
                    w1: MetalBuffer::zeros(&fallback_ctx, h * d, BufferUsage::Shared).unwrap(),
                    w2: MetalBuffer::zeros(&fallback_ctx, d * h, BufferUsage::Shared).unwrap(),
                    w3: MetalBuffer::zeros(&fallback_ctx, h * d, BufferUsage::Shared).unwrap(),
                    rms_att: vec![0.0; d],
                    rms_ffn: vec![0.0; d],
                }
            };
            layer_grads.push(lg);

            layer_adam.push(LayerAdamState {
                wq: AdamParam::new(qd * d),
                wk: AdamParam::new(kvd * d),
                wv: AdamParam::new(kvd * d),
                wo: AdamParam::new(d * qd),
                w1: AdamParam::new(h * d),
                w2: AdamParam::new(d * h),
                w3: AdamParam::new(h * d),
                rms_att: AdamParam::new(d),
                rms_ffn: AdamParam::new(d),
            });
        }

        Self {
            config,
            kernel_config,
            layer_weights,
            kernels: None,
            io_pool: None,
            layer_acts,
            layer_grads,
            layer_adam,
            embed_weights: vec![0.0; v * d],
            embed_grad: vec![0.0; v * d],
            embed_adam: AdamParam::new(v * d),
            rms_final: vec![0.0; d],
            rms_final_grad: vec![0.0; d],
            rms_final_adam: AdamParam::new(d),
            adam_t: 0,
            compile_count: 0,
            gpu_dw,
            scratch_a,
            scratch_b,
            metal_ctx,
            last_timings: StepTimings::default(),
            bwd_scratch: None,
            bwd_ids: None,
            loss_fn: None,
            decomposed_attn: false,
            decomposed_ffn: false,
            vocab_map: None,
            compact_embed: Vec::new(),
            compact_embed_grad: Vec::new(),
            compact_embed_adam: AdamParam::new(0),
            baseline_hw_ns: None,
            hw_ring: VecDeque::with_capacity(8),
            total_ane_evals: 0,
            last_refresh_step: 0,
            refresh_threshold: 1.15,
            max_evals_before_refresh: 25_000,
            kernels_b: None,
            io_pool_b: None,
            use_kernel_set_b: false,
        }
    }

    /// Get trainer config.
    pub fn config(&self) -> &DynamicAneTrainerConfig {
        &self.config
    }

    /// Current Adam step count.
    pub fn adam_t(&self) -> usize {
        self.adam_t
    }

    /// Number of ANE compilations performed (should be 9+ after compile_kernels).
    pub fn compile_count(&self) -> usize {
        self.compile_count
    }

    /// Check whether ANE throughput has degraded enough to warrant a kernel refresh.
    ///
    /// Returns `true` when either:
    /// - the 8-step moving average of `hw_execution_time_ns` exceeds
    ///   `baseline_hw_ns * refresh_threshold` (15% degradation), OR
    /// - `total_ane_evals` has reached `max_evals_before_refresh` (safety limit).
    ///
    /// Returns `false` if fewer than 8 samples have been collected (calibration
    /// window not yet filled) or no baseline has been established.
    pub fn check_degradation(&self) -> bool {
        // Safety limit: unconditional refresh after N total evaluations.
        if self.total_ane_evals >= self.max_evals_before_refresh {
            return true;
        }

        let Some(baseline) = self.baseline_hw_ns else {
            return false;
        };

        // Need a full ring buffer before triggering (avoids false positives during
        // early calibration).
        if self.hw_ring.len() < 8 {
            return false;
        }

        let moving_avg = self.hw_ring.iter().sum::<u64>() / self.hw_ring.len() as u64;
        let threshold = (baseline as f32 * self.refresh_threshold) as u64;
        moving_avg > threshold
    }

    /// Refresh ANE kernels to recover from throughput degradation.
    ///
    /// Recompiles all kernels via `compile_kernels()`, resets the hw_ring and
    /// `total_ane_evals`, and clears the baseline so it will be re-calibrated
    /// from the next 5 steps.
    pub fn refresh_kernels(&mut self) -> Result<()> {
        let step = self.adam_t;
        tracing::info!(
            step,
            total_evals = self.total_ane_evals,
            baseline_ns = self.baseline_hw_ns,
            "ANE kernel refresh: recompiling to recover throughput"
        );

        self.compile_kernels()?;

        // Reset degradation tracking so calibration restarts from scratch.
        self.hw_ring.clear();
        self.total_ane_evals = 0;
        self.baseline_hw_ns = None;
        self.last_refresh_step = step;

        tracing::info!(step, "ANE kernel refresh complete");
        Ok(())
    }

    /// Record a hardware execution time sample from a representative kernel.
    ///
    /// Pushes `hw_ns` into the 8-slot ring buffer, evicting the oldest value when
    /// full. After the first 5 samples the baseline is set to their median.
    /// `total_ane_evals` is incremented on every call.
    fn record_hw_sample(&mut self, hw_ns: u64) {
        // Ignore zero readings: perf stats class unavailable or hw timer not set.
        if hw_ns == 0 {
            return;
        }

        self.total_ane_evals += 1;

        if self.hw_ring.len() == 8 {
            self.hw_ring.pop_front();
        }
        self.hw_ring.push_back(hw_ns);

        // Establish baseline from the median of the first 5 valid samples.
        const CALIBRATION_WINDOW: usize = 5;
        if self.baseline_hw_ns.is_none() && self.hw_ring.len() >= CALIBRATION_WINDOW {
            let mut window: Vec<u64> = self
                .hw_ring
                .iter()
                .take(CALIBRATION_WINDOW)
                .copied()
                .collect();
            window.sort_unstable();
            let median = window[CALIBRATION_WINDOW / 2];
            self.baseline_hw_ns = Some(median);
            tracing::debug!(baseline_ns = median, "ANE throughput baseline established");
        }
    }

    /// Install a VocabMap for compact classifier/embedding operations.
    ///
    /// Must be called after `load_weights_*` and before `compile_kernels()`.
    /// Builds the compact embedding matrix from the full embedding table.
    pub fn install_vocab_map(&mut self, vocab_map: VocabMap) {
        let d = self.config.dim;
        let cv = vocab_map.compact_vocab;

        // Build compact embedding from full
        let mut compact = vec![0.0f32; cv * d];
        for c in 0..cv {
            let full_row = vocab_map.compact_to_full[c] * d;
            let compact_row = c * d;
            compact[compact_row..compact_row + d]
                .copy_from_slice(&self.embed_weights[full_row..full_row + d]);
        }

        info!(
            full_vocab = self.config.vocab_size,
            compact_vocab = cv,
            ratio = format!("{:.1}x", self.config.vocab_size as f64 / cv as f64),
            "Vocab compaction installed"
        );

        self.compact_embed = compact;
        self.compact_embed_grad = vec![0.0f32; cv * d];
        self.compact_embed_adam = AdamParam::new(cv * d);
        self.vocab_map = Some(vocab_map);
    }

    /// Whether vocab compaction is active.
    pub fn has_vocab_compaction(&self) -> bool {
        self.vocab_map.is_some()
    }

    /// Scatter compact embedding weights back to the full embedding table.
    fn scatter_compact_to_full(&mut self) {
        if let Some(ref vm) = self.vocab_map {
            let d = self.config.dim;
            for c in 0..vm.compact_vocab {
                let full_row = vm.compact_to_full[c] * d;
                let compact_row = c * d;
                self.embed_weights[full_row..full_row + d]
                    .copy_from_slice(&self.compact_embed[compact_row..compact_row + d]);
            }
        }
    }

    /// Compile all dynamic ANE kernels (called once at startup).
    ///
    /// Decomposed pipeline: one kernel per unique projection shape (IC, OC),
    /// plus attention-only and backward kernels. Each projection kernel is a
    /// simple single-matmul that compiles reliably at any model dimension.
    pub fn compile_kernels(&mut self) -> Result<()> {
        let rt = AneRuntime::global()?;
        let dkc = DynamicKernelConfig::new(self.kernel_config.clone());
        let d = self.kernel_config.dim;
        let h = self.kernel_config.hidden_dim;
        let s = self.kernel_config.seq_len;
        let qd = self.kernel_config.q_dim();
        let kvd = self.kernel_config.kv_dim();
        let score_ch = self.kernel_config.score_ch();
        let is_gqa = self.kernel_config.n_kv_heads != self.kernel_config.n_heads;

        // Compile helper with debug MIL dump on failure
        let compile =
            |out: &DynamicKernelOutput, rt: &AneRuntime, name: &str| -> Result<AneModel> {
                let wd = if out.static_weights.entries.is_empty() {
                    None
                } else {
                    Some(&out.static_weights)
                };
                debug!(
                    name,
                    ic = out.input_layout.ic,
                    sp = out.input_layout.total_spatial,
                    "Compiling dynamic kernel"
                );
                match rt.compile(out.mil_text.as_bytes(), wd) {
                    Ok(model) => Ok(model),
                    Err(e) => {
                        tracing::error!("Failed to compile {name}: {e}");
                        let mil_path = format!("/tmp/ane_debug_{name}.mil");
                        if let Err(we) = std::fs::write(&mil_path, &out.mil_text) {
                            tracing::error!("Could not write debug MIL to {mil_path}: {we}");
                        } else {
                            tracing::error!(
                                "Full MIL written to {mil_path} ({} bytes)",
                                out.mil_text.len()
                            );
                        }
                        Err(e)
                    }
                }
            };

        // 1. Determine unique projection shapes needed (Fallback/Hybrid)
        let mut proj_shapes: Vec<(usize, usize)> = vec![
            (d, qd),  // Q fwd, Wo^T bwd
            (d, kvd), // K fwd, V fwd
            (qd, d),  // Wo fwd, Wq^T bwd
            (d, h),   // W1 fwd, W3 fwd, W2^T bwd
            (h, d),   // W2 fwd, W1^T bwd, W3^T bwd
            (kvd, d), // Wk^T bwd, Wv^T bwd
        ];
        proj_shapes.sort();
        proj_shapes.dedup();

        let kernel_count = proj_shapes.len() + 6; // projections + fused(2) + bwd1 + bwd2 + bwd_ffn(2)
        info!(
            kernels = kernel_count,
            projection_shapes = proj_shapes.len(),
            gqa = is_gqa,
            "Compiling ANE performance frontier kernels..."
        );

        // Detect UltraFusion for dual-die alternation.
        let is_ultra = MetalContext::global()
            .map(|ctx| ctx.properties().is_ultra_fusion)
            .unwrap_or(false);

        // 2. Compile projection kernels (retained for flexibility/caching)
        let mut projections = HashMap::new();
        let mut proj_inputs = HashMap::new();
        let mut proj_outputs = HashMap::new();
        for &(ic, oc) in &proj_shapes {
            let out = dynamic_kernel::gen_dynamic_projection(ic, oc, s, 0);
            let model = compile(&out, rt, &format!("proj_{ic}x{oc}"))?;
            projections.insert((ic, oc), model);
            proj_inputs.insert((ic, oc), IoSurface::for_tensor_f32(ic, s + oc)?);
            proj_outputs.insert((ic, oc), IoSurface::for_tensor_f32(oc, s)?);
            self.compile_count += 1;
        }

        // 3. Compile Fused Forward Kernels
        //
        // The fused SDPA kernel materializes the full attention matrix [heads, seq, seq]
        // on ANE SRAM. For large head counts × seq lengths, this exceeds the ~40 MB SRAM
        // budget and causes hardware rejection (status=0x1d). Detect this upfront and
        // fall back to decomposed projections + CPU BLAS attention.
        let attn_matrix_bytes = self.kernel_config.n_heads * s * s * 2; // fp16
        let attn_sram_threshold = 4_000_000; // 4 MB — empirical safe limit

        let sdpa_fwd = if attn_matrix_bytes > attn_sram_threshold {
            info!(
                heads = self.kernel_config.n_heads,
                seq = s,
                attn_matrix_mb = attn_matrix_bytes / (1024 * 1024),
                "Attention matrix too large for fused ANE kernel — using decomposed projections + CPU attention"
            );
            self.decomposed_attn = true;
            None
        } else {
            let k1 = dynamic_kernel::gen_dynamic_sdpa_fwd(&dkc, 0);
            let model = compile(&k1, rt, "sdpa_fwd")?;
            self.compile_count += 1;
            Some(model)
        };

        // FFN fused kernel: input spatial = s + 1 + 2*h. For large hidden_dim
        // the IOSurface and intermediate activations (3 × [h, s] fp16) exceed ANE SRAM.
        // Threshold: 3 * h * s * 2 bytes > 8 MB → decompose to projection kernels + CPU SiLU.
        let ffn_intermediate_bytes = 3 * h * s * 2; // h1, h3, gate in fp16
        let ffn_sram_threshold = 8_000_000; // 8 MB

        let ffn_fwd = if ffn_intermediate_bytes > ffn_sram_threshold {
            info!(
                hidden_dim = h,
                seq = s,
                ffn_mb = ffn_intermediate_bytes / (1024 * 1024),
                "FFN intermediates too large for fused ANE kernel — using decomposed projections + CPU SiLU"
            );
            self.decomposed_ffn = true;
            None
        } else {
            let k2 = dynamic_kernel::gen_dynamic_ffn_w13(&dkc, 0);
            let model = compile(&k2, rt, "ffn_fwd")?;
            self.compile_count += 1;
            Some(model)
        };

        // 4. Compile Backward Kernels
        // Backward attention kernels also have O(seq^2) intermediates (score channels),
        // so skip them when decomposed attention is used.
        let (sdpa_bwd1, sdpa_bwd2) = if self.decomposed_attn {
            (None, None)
        } else {
            let k7 = dynamic_kernel::gen_dynamic_sdpa_bwd1(&dkc);
            let b1 = compile(&k7, rt, "sdpa_bwd1")?;
            self.compile_count += 1;

            let k8 = dynamic_kernel::gen_dynamic_sdpa_bwd2(&dkc);
            let b2 = compile(&k8, rt, "sdpa_bwd2")?;
            self.compile_count += 1;

            (Some(b1), Some(b2))
        };

        let k4 = dynamic_kernel::gen_dynamic_ffn_bwd_w2t(&dkc);
        let ffn_bwd_w2t = compile(&k4, rt, "ffn_bwd_w2t")?;
        self.compile_count += 1;

        let k5 = dynamic_kernel::gen_dynamic_ffn_bwd_w13t(&dkc);
        let ffn_bwd_w13t = compile(&k5, rt, "ffn_bwd_w13t")?;
        self.compile_count += 1;

        // 5. Softmax (optional)
        let softmax_vocab = self
            .vocab_map
            .as_ref()
            .map(|vm| vm.compact_vocab)
            .unwrap_or(self.config.vocab_size);
        let (softmax_kern, softmax_in, softmax_out) = {
            let sm_out = dynamic_kernel::gen_dynamic_softmax(softmax_vocab, s, 0);
            match compile(&sm_out, rt, "softmax") {
                Ok(model) => {
                    self.compile_count += 1;
                    let sm_in = IoSurface::for_tensor(softmax_vocab, s)?;
                    let sm_out_io = IoSurface::for_tensor(softmax_vocab, s)?;
                    (Some(model), Some(sm_in), Some(sm_out_io))
                }
                Err(_) => (None, None, None),
            }
        };

        // 6. Allocate IOSurface pool for fused kernels
        let sdpa_fwd_sp = s + 1 + 2 * qd + 2 * kvd;
        let ffn_fwd_sp = s + 1 + 2 * h;
        let bwd1_in_ch = qd + 2 * kvd + d;
        let bwd1_out_ch = kvd + 2 * score_ch;
        let bwd2_in_ch = 2 * score_ch + qd + kvd;
        let bwd2_out_ch = qd + kvd;

        let io_pool = DynIoPool {
            proj_inputs,
            proj_outputs,
            sdpa_fwd_in: IoSurface::for_tensor_f32(d, sdpa_fwd_sp)?,
            sdpa_fwd_out: IoSurface::for_tensor_f32(2 * d + 2 * qd + 2 * kvd, s)?,
            ffn_fwd_in: IoSurface::for_tensor_f32(d, ffn_fwd_sp)?,
            ffn_fwd_out: IoSurface::for_tensor_f32(3 * h, s)?,
            sdpa_bwd1_in: IoSurface::for_tensor_f32(bwd1_in_ch, s + qd)?,
            sdpa_bwd1_out: IoSurface::for_tensor(bwd1_out_ch, s)?,
            sdpa_bwd2_in: IoSurface::for_tensor(bwd2_in_ch, s)?,
            sdpa_bwd2_out: IoSurface::for_tensor(bwd2_out_ch, s)?,
            ffn_bwd_w2t_in: IoSurface::for_tensor_f32(d, s + h)?,
            ffn_bwd_w2t_out: IoSurface::for_tensor_f32(h, s)?,
            ffn_bwd_w13t_in: IoSurface::for_tensor_f32(h, 2 * s + 2 * d)?,
            ffn_bwd_w13t_out: IoSurface::for_tensor_f32(d, s)?,
            softmax_in,
            softmax_out,
        };

        self.kernels = Some(DynamicKernels {
            projections,
            sdpa_fwd,
            ffn_fwd,
            sdpa_bwd1,
            sdpa_bwd2,
            ffn_bwd_w2t,
            ffn_bwd_w13t,
            softmax: softmax_kern,
        });
        self.io_pool = Some(io_pool);

        // 7. Compile B kernel set for dual-die alternation on UltraFusion chips.
        //
        // Uses variant=1 to produce MIL programs with a different hash.  The ANE
        // daemon assigns programs with different hashes to different dies, so
        // alternating between A and B sets per training step distributes thermal
        // load across both dies of an M3/M4/M5 Ultra, extending the
        // degradation-free window from ~40 to ~80 steps.
        if is_ultra {
            info!("UltraFusion detected — compiling variant-B kernel set for dual-die alternation");

            let mut proj_b = HashMap::new();
            let mut proj_in_b = HashMap::new();
            let mut proj_out_b = HashMap::new();
            for &(ic, oc) in &proj_shapes {
                let out_b = dynamic_kernel::gen_dynamic_projection(ic, oc, s, 1);
                let model_b = compile(&out_b, rt, &format!("proj_{ic}x{oc}_b"))?;
                proj_b.insert((ic, oc), model_b);
                proj_in_b.insert((ic, oc), IoSurface::for_tensor_f32(ic, s + oc)?);
                proj_out_b.insert((ic, oc), IoSurface::for_tensor_f32(oc, s)?);
                self.compile_count += 1;
            }

            let sdpa_fwd_b = if self.decomposed_attn {
                None
            } else {
                let k1b = dynamic_kernel::gen_dynamic_sdpa_fwd(&dkc, 1);
                let m = compile(&k1b, rt, "sdpa_fwd_b")?;
                self.compile_count += 1;
                Some(m)
            };

            let ffn_fwd_b = if self.decomposed_ffn {
                None
            } else {
                let k2b = dynamic_kernel::gen_dynamic_ffn_w13(&dkc, 1);
                let m = compile(&k2b, rt, "ffn_fwd_b")?;
                self.compile_count += 1;
                Some(m)
            };

            // Backward kernels are shared (not part of die alternation):
            // use primary set's bwd kernels for B too.  Re-compile them as
            // separate AneModel instances pointing to the same MIL, so the
            // B IOSurface pool can be used independently.
            let ffn_bwd_w2t_b = {
                let k4b = dynamic_kernel::gen_dynamic_ffn_bwd_w2t(&dkc);
                compile(&k4b, rt, "ffn_bwd_w2t_b")?
            };
            let ffn_bwd_w13t_b = {
                let k5b = dynamic_kernel::gen_dynamic_ffn_bwd_w13t(&dkc);
                compile(&k5b, rt, "ffn_bwd_w13t_b")?
            };
            let (sdpa_bwd1_b, sdpa_bwd2_b) = if self.decomposed_attn {
                (None, None)
            } else {
                let b1 = compile(
                    &dynamic_kernel::gen_dynamic_sdpa_bwd1(&dkc),
                    rt,
                    "sdpa_bwd1_b",
                )?;
                let b2 = compile(
                    &dynamic_kernel::gen_dynamic_sdpa_bwd2(&dkc),
                    rt,
                    "sdpa_bwd2_b",
                )?;
                self.compile_count += 2;
                (Some(b1), Some(b2))
            };

            let sm_out_b = dynamic_kernel::gen_dynamic_softmax(softmax_vocab, s, 1);
            let softmax_b = match compile(&sm_out_b, rt, "softmax_b") {
                Ok(m) => {
                    self.compile_count += 1;
                    Some(m)
                }
                Err(_) => None,
            };

            // Allocate softmax IOSurfaces as a pair — both must succeed or both None.
            let sm_io_b = if softmax_b.is_some() {
                match (
                    IoSurface::for_tensor(softmax_vocab, s),
                    IoSurface::for_tensor(softmax_vocab, s),
                ) {
                    (Ok(sin), Ok(sout)) => (Some(sin), Some(sout)),
                    _ => {
                        tracing::warn!("Softmax B-set IOSurface pair allocation failed");
                        (None, None)
                    }
                }
            } else {
                (None, None)
            };

            let io_pool_b = DynIoPool {
                proj_inputs: proj_in_b,
                proj_outputs: proj_out_b,
                sdpa_fwd_in: IoSurface::for_tensor_f32(d, sdpa_fwd_sp)?,
                sdpa_fwd_out: IoSurface::for_tensor_f32(2 * d + 2 * qd + 2 * kvd, s)?,
                ffn_fwd_in: IoSurface::for_tensor_f32(d, ffn_fwd_sp)?,
                ffn_fwd_out: IoSurface::for_tensor_f32(3 * h, s)?,
                sdpa_bwd1_in: IoSurface::for_tensor_f32(bwd1_in_ch, s + qd)?,
                sdpa_bwd1_out: IoSurface::for_tensor(bwd1_out_ch, s)?,
                sdpa_bwd2_in: IoSurface::for_tensor(bwd2_in_ch, s)?,
                sdpa_bwd2_out: IoSurface::for_tensor(bwd2_out_ch, s)?,
                ffn_bwd_w2t_in: IoSurface::for_tensor_f32(d, s + h)?,
                ffn_bwd_w2t_out: IoSurface::for_tensor_f32(h, s)?,
                ffn_bwd_w13t_in: IoSurface::for_tensor_f32(h, 2 * s + 2 * d)?,
                ffn_bwd_w13t_out: IoSurface::for_tensor_f32(d, s)?,
                softmax_in: sm_io_b.0,
                softmax_out: sm_io_b.1,
            };

            self.kernels_b = Some(DynamicKernels {
                projections: proj_b,
                sdpa_fwd: sdpa_fwd_b,
                ffn_fwd: ffn_fwd_b,
                sdpa_bwd1: sdpa_bwd1_b,
                sdpa_bwd2: sdpa_bwd2_b,
                ffn_bwd_w2t: ffn_bwd_w2t_b,
                ffn_bwd_w13t: ffn_bwd_w13t_b,
                softmax: softmax_b,
            });
            self.io_pool_b = Some(io_pool_b);

            // Double the safety refresh window: with two dies, degradation
            // occurs at roughly twice the rate of evaluations.
            self.max_evals_before_refresh *= 2;

            info!(
                max_evals = self.max_evals_before_refresh,
                "Dual-die alternation ready; safety refresh window doubled"
            );
        }

        // Pre-allocate backward scratch pool (eliminates ~13 Vec allocs per layer per step)
        let entries = backward_scratch_entries(d, h, s, qd, kvd);
        let scratch = BackwardScratch::build(&entries);
        let ids = backward_scratch_ids(&entries);
        let mode = match (self.decomposed_attn, self.decomposed_ffn) {
            (false, false) => "fully fused (small model)",
            (true, false) => "decomposed attention + fused FFN",
            (false, true) => "fused attention + decomposed FFN",
            (true, true) => "fully decomposed (ANE projections + CPU attention/SiLU)",
        };
        info!(
            size_kb = scratch.size_bytes() / 1024,
            mode, "backward scratch pool allocated"
        );
        self.bwd_scratch = Some(scratch);
        self.bwd_ids = Some(ids);

        // Initialize default loss function if not already set
        let vocab = self
            .vocab_map
            .as_ref()
            .map_or(self.config.vocab_size, |vm| vm.compact_vocab);
        if self.loss_fn.is_none() {
            self.loss_fn = Some(Box::new(CrossEntropyLoss::new(vocab)));
        }

        Ok(())
    }

    /// Set a custom loss function for training.
    ///
    /// Must be called before `compile_kernels()` or after it (the loss function
    /// is independent of kernel compilation). Default is `CrossEntropyLoss`.
    pub fn set_loss(&mut self, loss: Box<dyn AneTrainingLoss>) {
        self.loss_fn = Some(loss);
    }

    /// Refresh transposed weight buffers after Adam update.
    fn refresh_transposed_weights(&mut self) {
        let d = self.config.dim;
        let h = self.config.hidden_dim;
        let qd = self.kernel_config.q_dim();
        let kvd = self.kernel_config.kv_dim();

        for lw in &mut self.layer_weights {
            transpose_weight(&lw.wq, &mut lw.wq_t, qd, d); // [qd, d] → [d, qd]
            transpose_weight(&lw.wk, &mut lw.wk_t, kvd, d); // [kvd, d] → [d, kvd]
            transpose_weight(&lw.wv, &mut lw.wv_t, kvd, d); // [kvd, d] → [d, kvd]
            transpose_weight(&lw.wo, &mut lw.wo_t, d, qd); // [d, qd] → [qd, d]
            transpose_weight(&lw.w1, &mut lw.w1_t, h, d);
            transpose_weight(&lw.w2, &mut lw.w2_t, d, h);
            transpose_weight(&lw.w3, &mut lw.w3_t, h, d);
        }
    }

    // dispatch_dw is a free function below (encode_dw_gemm) to avoid borrow conflicts.

    /// Execute a single projection kernel: act @ W → output.
    #[allow(clippy::too_many_arguments)]
    fn run_projection(
        kernels: &DynamicKernels,
        io: &DynIoPool,
        ic: usize,
        oc: usize,
        seq: usize,
        act: &[f32],
        weight: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let key = (ic, oc);
        let kernel = kernels.projections.get(&key).ok_or_else(|| {
            MetalError::InvalidConfig(format!("No projection kernel for ({ic}, {oc})"))
        })?;
        let io_in = io.proj_inputs.get(&key).unwrap();
        let io_out = io.proj_outputs.get(&key).unwrap();

        io_in.write_packed_f32(act, &[(weight, oc)], ic, seq);
        kernel.evaluate(&[io_in.as_ptr()], &[io_out.as_ptr()])?;
        io_out.read_f32(output, 0, oc, seq);
        Ok(())
    }

    /// Forward pass for a single layer using decomposed ANE kernels.
    ///
    /// Each projection (Q, K, V, Wo, W1, W3, W2) is a separate ANE kernel.
    /// Attention is a standalone weight-free kernel. SiLU gate runs on CPU.
    fn forward_layer(&mut self, l: usize, x: &mut [f32]) -> Result<()> {
        let d = self.config.dim;
        let h = self.config.hidden_dim;
        let s = self.config.seq_len;
        let qd = self.kernel_config.q_dim();
        let kvd = self.kernel_config.kv_dim();
        let lw = &self.layer_weights[l];
        let acts = &mut self.layer_acts[l];

        // hw_sample collects a single hw_execution_time_ns reading (layer 0 only).
        // We defer the call to record_hw_sample() until after all borrows are
        // released at the end of the function, to satisfy the borrow checker.
        let mut hw_sample: Option<u64> = None;

        // Dual-die selection: choose kernel set A or B based on the per-step toggle.
        // Both fields are borrowed as shared refs here.  The toggle is flipped once
        // per full step (not per layer) in train_step() after the forward loop.
        let (kernels, io) = if self.use_kernel_set_b {
            if let (Some(kb), Some(ib)) = (self.kernels_b.as_ref(), self.io_pool_b.as_ref()) {
                (kb, ib)
            } else {
                (
                    self.kernels
                        .as_ref()
                        .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?,
                    self.io_pool
                        .as_ref()
                        .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?,
                )
            }
        } else {
            (
                self.kernels
                    .as_ref()
                    .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?,
                self.io_pool
                    .as_ref()
                    .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?,
            )
        };

        if self.decomposed_attn {
            // ====== Decomposed Attention (ANE projections + CPU BLAS attention) ======
            // Used when the attention matrix [heads, seq, seq] exceeds ANE SRAM.

            // 1. RMSNorm on CPU
            accelerate::rmsnorm(
                &mut acts.xnorm,
                x,
                &lw.rms_att,
                d,
                s,
                self.config.rms_norm_eps,
            );

            // 2. Q, K, V projections via ANE (individual projection kernels).
            // On layer 0, use evaluate_with_stats on the Q projection to collect
            // hw timing for degradation detection.  The result is stored in
            // `hw_sample` and forwarded to record_hw_sample() at function exit,
            // after all mutable borrows of `self` fields are released.
            if l == 0 {
                let q_key = (d, qd);
                let q_kernel = kernels.projections.get(&q_key).ok_or_else(|| {
                    MetalError::InvalidConfig(format!("No projection kernel for ({d}, {qd})"))
                })?;
                let io_in = io.proj_inputs.get(&q_key).unwrap();
                let io_out = io.proj_outputs.get(&q_key).unwrap();
                io_in.write_packed_f32(&acts.xnorm, &[(&lw.wq, qd)], d, s);
                let stats = q_kernel.evaluate_with_stats(&[io_in.as_ptr()], &[io_out.as_ptr()])?;
                io_out.read_f32(&mut acts.q, 0, qd, s);
                hw_sample = Some(stats.hw_execution_time_ns);
            } else {
                Self::run_projection(kernels, io, d, qd, s, &acts.xnorm, &lw.wq, &mut acts.q)?;
            }
            Self::run_projection(kernels, io, d, kvd, s, &acts.xnorm, &lw.wk, &mut acts.k)?;
            Self::run_projection(kernels, io, d, kvd, s, &acts.xnorm, &lw.wv, &mut acts.v)?;

            // 3. Attention on CPU via Accelerate BLAS
            // Layout: Q [qd, s] channel-first, K [kvd, s], V [kvd, s]
            // Reshape to [heads, hd, s] → [heads, s, hd] for BLAS matmul
            let n_heads = self.kernel_config.n_heads;
            let n_kv_heads = self.kernel_config.n_kv_heads;
            let hd = self.kernel_config.head_dim;
            let scale = 1.0 / (hd as f32).sqrt();
            let gqa_ratio = n_heads / n_kv_heads;

            // Compute scores, softmax, and output per head (avoids materializing full [heads, s, s])
            acts.attn_out.fill(0.0);
            for h_idx in 0..n_heads {
                let kv_idx = h_idx / gqa_ratio;
                let q_off = h_idx * hd * s;
                let k_off = kv_idx * hd * s;
                let v_off = kv_idx * hd * s;

                // scores = Q_h^T @ K_h * scale: [s, hd] @ [hd, s] → [s, s]
                let mut scores = vec![0.0f32; s * s];
                accelerate::gemm(
                    &acts.q[q_off..q_off + hd * s],
                    &acts.k[k_off..k_off + hd * s],
                    &mut scores,
                    s,
                    s,
                    hd,
                    scale,
                    0.0,
                    true,
                    false,
                );

                // Causal mask: set upper triangle to -inf
                for row in 0..s {
                    for col in (row + 1)..s {
                        scores[row * s + col] = f32::NEG_INFINITY;
                    }
                }

                // Softmax per row
                for row in 0..s {
                    let row_slice = &mut scores[row * s..(row + 1) * s];
                    let max_val = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row_slice.iter_mut() {
                        *v = (*v - max_val).exp();
                        sum += *v;
                    }
                    let inv_sum = 1.0 / sum;
                    for v in row_slice.iter_mut() {
                        *v *= inv_sum;
                    }
                }

                // attn_out_h = scores @ V_h^T: [s, s] @ [s, hd] → [s, hd]
                // Output in channel-first: [hd, s] at offset h_idx * hd * s
                let attn_off = h_idx * hd * s;
                // V is [hd, s] channel-first; we need [s, hd] row-major for GEMM
                // scores @ V^T where V^T is [s, hd] = transpose of [hd, s]
                accelerate::gemm(
                    &scores,
                    &acts.v[v_off..v_off + hd * s],
                    &mut acts.attn_out[attn_off..attn_off + hd * s],
                    hd,
                    s,
                    s,
                    1.0,
                    0.0,
                    false,
                    true,
                );
            }

            // 4. Wo projection via ANE
            Self::run_projection(
                kernels,
                io,
                qd,
                d,
                s,
                &acts.attn_out,
                &lw.wo,
                &mut acts.o_out,
            )?;

            // 5. Residual
            accelerate::vadd(x, &acts.o_out, &mut acts.x2);
        } else {
            // ====== Fused Attention Block (ANE) ======
            // Input layout: x [d, s], rms_w [d, 1], wq [d, qd], wk [d, kvd], wv [d, kvd], wo [d, qd]
            io.sdpa_fwd_in.write_packed_f32(
                x,
                &[
                    (&lw.rms_att, 1),
                    (&lw.wq, qd),
                    (&lw.wk, kvd),
                    (&lw.wv, kvd),
                    (&lw.wo, qd),
                ],
                d,
                s,
            );

            let sdpa_model = kernels.sdpa_fwd.as_ref().ok_or_else(|| {
                MetalError::InvalidConfig("Fused SDPA kernel not compiled".into())
            })?;
            // Use evaluate_with_stats on layer 0 to collect hw timing for
            // degradation detection.  Result stored in `hw_sample` and forwarded
            // to record_hw_sample() at function exit after borrows are released.
            if l == 0 {
                let stats = sdpa_model
                    .evaluate_with_stats(&[io.sdpa_fwd_in.as_ptr()], &[io.sdpa_fwd_out.as_ptr()])?;
                hw_sample = Some(stats.hw_execution_time_ns);
            } else {
                sdpa_model.evaluate(&[io.sdpa_fwd_in.as_ptr()], &[io.sdpa_fwd_out.as_ptr()])?;
            }

            // Read back all taps for backward pass
            // Taps: [o_out, q, k, v, attn_flat, xnorm]
            let out_ch = 2 * d + 2 * qd + 2 * kvd;
            let mut taps = vec![0.0f32; out_ch * s];
            io.sdpa_fwd_out.read_f32(&mut taps, 0, out_ch, s);

            // De-interleave taps into acts
            let mut tap_off = 0;
            let mut copy_tap = |dst: &mut [f32], ch: usize| {
                dst.copy_from_slice(&taps[tap_off * s..(tap_off + ch) * s]);
                tap_off += ch;
            };

            copy_tap(&mut acts.o_out, d);
            copy_tap(&mut acts.q, qd);
            copy_tap(&mut acts.k, kvd);
            copy_tap(&mut acts.v, kvd);
            copy_tap(&mut acts.attn_out, qd);
            copy_tap(&mut acts.xnorm, d);

            // Residual: x2 = x + o_out
            accelerate::vadd(x, &acts.o_out, &mut acts.x2);
        }

        // Compute x2norm = RMSNorm(x2) on CPU for the backward pass.
        //
        // The fused FFN kernel computes this internally but does not surface it
        // as a tap.  We need it to form dW1 = dh1 @ x2norm^T and
        // dW3 = dh3 @ x2norm^T in backward_layer.  This is a tiny CPU cost
        // (one RMSNorm of [d, s]) — negligible compared to the ANE kernel.
        accelerate::rmsnorm(
            &mut acts.x2norm,
            &acts.x2,
            &lw.rms_ffn,
            d,
            s,
            self.config.rms_norm_eps,
        );

        if self.decomposed_ffn {
            // ====== Decomposed FFN (ANE projections + CPU SiLU) ======
            // Used when hidden_dim × seq exceeds ANE SRAM for the fused kernel.

            // W1, W3 projections via ANE: x2norm @ W1, x2norm @ W3
            Self::run_projection(kernels, io, d, h, s, &acts.x2norm, &lw.w1, &mut acts.h1)?;
            Self::run_projection(kernels, io, d, h, s, &acts.x2norm, &lw.w3, &mut acts.h3)?;

            // SiLU gate on CPU: silu_out = SiLU(h1) * h3
            for i in 0..(h * s) {
                let sig = 1.0 / (1.0 + (-acts.h1[i]).exp());
                acts.silu_out[i] = acts.h1[i] * sig * acts.h3[i];
            }

            // W2 projection via ANE
            Self::run_projection(
                kernels,
                io,
                h,
                d,
                s,
                &acts.silu_out,
                &lw.w2,
                &mut acts.ffn_out,
            )?;

            // Final residual
            accelerate::vadd(&acts.x2, &acts.ffn_out, x);
        } else {
            // ====== Fused FFN Block (ANE) ======
            // Input layout: x2 [d, s], rms_ffn [d, 1], w1 [d, h], w3 [d, h]
            io.ffn_fwd_in.write_packed_f32(
                &acts.x2,
                &[(&lw.rms_ffn, 1), (&lw.w1, h), (&lw.w3, h)],
                d,
                s,
            );

            let ffn_model = kernels
                .ffn_fwd
                .as_ref()
                .ok_or_else(|| MetalError::InvalidConfig("Fused FFN kernel not compiled".into()))?;
            ffn_model.evaluate(&[io.ffn_fwd_in.as_ptr()], &[io.ffn_fwd_out.as_ptr()])?;

            // Read back taps: [h1, h3, gate]
            let ffn_out_ch = 3 * h;
            let mut ffn_taps = vec![0.0f32; ffn_out_ch * s];
            io.ffn_fwd_out.read_f32(&mut ffn_taps, 0, ffn_out_ch, s);

            let mut ft_off = 0;
            let mut copy_ft = |dst: &mut [f32], ch: usize| {
                dst.copy_from_slice(&ffn_taps[ft_off * s..(ft_off + ch) * s]);
                ft_off += ch;
            };
            copy_ft(&mut acts.h1, h);
            copy_ft(&mut acts.h3, h);
            copy_ft(&mut acts.silu_out, h); // ANE gen_dynamic_ffn_w13 outputs 'gate' here

            // W2 projection (standalone)
            Self::run_projection(
                kernels,
                io,
                h,
                d,
                s,
                &acts.silu_out,
                &lw.w2,
                &mut acts.ffn_out,
            )?;

            // Final residual
            accelerate::vadd(&acts.x2, &acts.ffn_out, x);
        }

        // All borrows of self.layer_acts[l] and self.kernels/io_pool are released
        // here.  Record the hw sample now so that record_hw_sample can take &mut self.
        if let Some(ns) = hw_sample {
            self.record_hw_sample(ns);
        }

        Ok(())
    }

    /// Backward pass for a single layer using decomposed ANE kernels.
    ///
    /// Each weight-transpose projection is a separate ANE kernel dispatch.
    /// Weight gradient GEMMs are encoded into `batch` (GPU) or run inline (CPU fallback).
    /// Results are combined on CPU via vadd.
    fn backward_layer(
        &mut self,
        l: usize,
        dx: &mut [f32],
        batch: &mut Option<BatchedCommandBuffer>,
    ) -> Result<()> {
        let d = self.config.dim;
        let h = self.config.hidden_dim;
        let s = self.config.seq_len;
        let qd = self.kernel_config.q_dim();
        let kvd = self.kernel_config.kv_dim();
        let eps = self.config.rms_norm_eps;

        let kernels = self
            .kernels
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?;
        let io = self
            .io_pool
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?;

        // GPU dW references (immutable borrows of Option fields)
        let gpu_dw = self.gpu_dw.as_ref();
        let scratch_a = self.scratch_a.as_ref();
        let scratch_b = self.scratch_b.as_ref();

        let scratch = self
            .bwd_scratch
            .as_mut()
            .expect("backward scratch not initialized — call compile_kernels() first");
        let ids = self
            .bwd_ids
            .as_ref()
            .expect("backward scratch IDs not initialized");

        // ====== FFN Backward ======

        // 1. dffn @ W2^T → dsilu_raw (fused ANE kernel)
        io.ffn_bwd_w2t_in
            .write_packed_f32(dx, &[(&self.layer_weights[l].w2_t, h)], d, s);
        kernels.ffn_bwd_w2t.evaluate(
            &[io.ffn_bwd_w2t_in.as_ptr()],
            &[io.ffn_bwd_w2t_out.as_ptr()],
        )?;
        io.ffn_bwd_w2t_out
            .read_f32(scratch.get_mut(ids.dsilu_raw), 0, h, s);

        // 2. dW2 = dx @ silu_out^T
        encode_dw_gemm(
            gpu_dw,
            scratch_a,
            scratch_b,
            batch.as_mut(),
            &dx[..d * s],
            &self.layer_acts[l].silu_out,
            &self.layer_grads[l].w2,
            d,
            h,
            s,
        );

        // 3. SiLU derivative on CPU
        // Copy dsilu_raw out so we can mutably borrow scratch for dh1/dh3
        let acts = &self.layer_acts[l];
        let dsilu_data: Vec<f32> = scratch.get(ids.dsilu_raw).to_vec();
        {
            let dh1 = scratch.get_mut(ids.dh1);
            for i in 0..(h * s) {
                let h1_val = acts.h1[i];
                let sig = 1.0 / (1.0 + (-h1_val).exp());
                let silu_d = sig * (1.0 + h1_val * (1.0 - sig));
                dh1[i] = dsilu_data[i] * acts.h3[i] * silu_d;
            }
        }
        {
            let dh3 = scratch.get_mut(ids.dh3);
            for i in 0..(h * s) {
                let h1_val = acts.h1[i];
                let sig = 1.0 / (1.0 + (-h1_val).exp());
                dh3[i] = dsilu_data[i] * h1_val * sig;
            }
        }

        // dW1, dW3
        encode_dw_gemm(
            gpu_dw,
            scratch_a,
            scratch_b,
            batch.as_mut(),
            scratch.get(ids.dh1),
            &self.layer_acts[l].x2norm,
            &self.layer_grads[l].w1,
            h,
            d,
            s,
        );
        encode_dw_gemm(
            gpu_dw,
            scratch_a,
            scratch_b,
            batch.as_mut(),
            scratch.get(ids.dh3),
            &self.layer_acts[l].x2norm,
            &self.layer_grads[l].w3,
            h,
            d,
            s,
        );

        // 4. dh1@W1^T + dh3@W3^T → dx_ffn (fused ANE kernel)
        io.ffn_bwd_w13t_in.write_packed_f32(
            scratch.get(ids.dh1),
            &[
                (scratch.get(ids.dh3), s),
                (&self.layer_weights[l].w1_t, d),
                (&self.layer_weights[l].w3_t, d),
            ],
            h,
            s,
        );
        kernels.ffn_bwd_w13t.evaluate(
            &[io.ffn_bwd_w13t_in.as_ptr()],
            &[io.ffn_bwd_w13t_out.as_ptr()],
        )?;
        io.ffn_bwd_w13t_out
            .read_f32(scratch.get_mut(ids.dx_ffn), 0, d, s);

        // 5. FFN RMSNorm backward (CPU)
        // Copy dx_ffn out since rmsnorm_backward needs immutable ref while writing dx_ffn_norm
        let dx_ffn_copy: Vec<f32> = scratch.get(ids.dx_ffn).to_vec();
        accelerate::rmsnorm_backward(
            scratch.get_mut(ids.dx_ffn_norm),
            &mut self.layer_grads[l].rms_ffn,
            &dx_ffn_copy,
            &self.layer_acts[l].x2,
            &self.layer_weights[l].rms_ffn,
            d,
            s,
            eps,
        );

        // ====== Attention Backward ======

        if self.decomposed_attn {
            // Decomposed attention backward: ANE projections + CPU BLAS attention grad.
            let n_heads = self.kernel_config.n_heads;
            let n_kv_heads = self.kernel_config.n_kv_heads;
            let hd = self.kernel_config.head_dim;
            let gqa_ratio = n_heads / n_kv_heads;
            let scale = 1.0 / (hd as f32).sqrt();

            // 6a. da = dx_ffn_norm @ Wo^T via ANE projection
            let dx_ffn_norm_copy: Vec<f32> = scratch.get(ids.dx_ffn_norm).to_vec();
            let mut da = vec![0.0f32; qd * s];
            Self::run_projection(
                kernels,
                io,
                d,
                qd,
                s,
                &dx_ffn_norm_copy,
                &self.layer_weights[l].wo_t,
                &mut da,
            )?;

            // dWo
            encode_dw_gemm(
                gpu_dw,
                scratch_a,
                scratch_b,
                batch.as_mut(),
                scratch.get(ids.dx_ffn_norm),
                &self.layer_acts[l].attn_out,
                &self.layer_grads[l].wo,
                d,
                qd,
                s,
            );

            // 6b-6c. Per-head attention backward on CPU
            // Recompute: scores = Q @ K^T * scale, softmax → probs
            // Then: dV = probs^T @ da_h, dp = da_h @ V^T
            //        ds = probs * (dp - rowsum(probs * dp)) * scale
            //        dQ = ds @ K, dK = ds^T @ Q
            scratch.get_mut(ids.dq).fill(0.0);
            scratch.get_mut(ids.dk).fill(0.0);
            scratch.get_mut(ids.dv).fill(0.0);

            for h_idx in 0..n_heads {
                let kv_idx = h_idx / gqa_ratio;
                let q_off = h_idx * hd * s;
                let k_off = kv_idx * hd * s;
                let v_off = kv_idx * hd * s;
                let da_off = h_idx * hd * s;

                // Recompute scores = Q_h^T @ K_h * scale: [s, hd] @ [hd, s] → [s, s]
                let mut scores = vec![0.0f32; s * s];
                accelerate::gemm(
                    &acts.q[q_off..q_off + hd * s],
                    &acts.k[k_off..k_off + hd * s],
                    &mut scores,
                    s,
                    s,
                    hd,
                    scale,
                    0.0,
                    true,
                    false,
                );

                // Causal mask + softmax
                for row in 0..s {
                    for col in (row + 1)..s {
                        scores[row * s + col] = f32::NEG_INFINITY;
                    }
                    let max_val = scores[row * s..(row + 1) * s]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for col in 0..s {
                        scores[row * s + col] = (scores[row * s + col] - max_val).exp();
                        sum += scores[row * s + col];
                    }
                    let inv = 1.0 / sum;
                    for col in 0..s {
                        scores[row * s + col] *= inv;
                    }
                }
                // scores is now probs [s, s]

                // dp = da_h @ V_h^T: [s, hd] @ [hd, s]^T → but da is [hd, s] channel-first
                // Reshape: da_h [hd, s] → da_h^T [s, hd], V_h [hd, s] → V_h^T [s, hd]
                // dp = da_h^T @ V_h = gemm(da_h, V_h, s, s, hd, trans_a=true, trans_b=false)
                let mut dp = vec![0.0f32; s * s];
                accelerate::gemm(
                    &da[da_off..da_off + hd * s],
                    &acts.v[v_off..v_off + hd * s],
                    &mut dp,
                    s,
                    s,
                    hd,
                    1.0,
                    0.0,
                    true,
                    false,
                );

                // dV_h = probs^T @ da_h^T → [s, s]^T @ [s, hd] → gemm with trans_a
                // dV output is [hd, s] channel-first, so: dV = da_h @ probs = gemm(da_h [hd,s], probs [s,s], hd, s, s)
                let dv_slice = &mut scratch.get_mut(ids.dv)[kv_idx * hd * s..(kv_idx + 1) * hd * s];
                accelerate::gemm(
                    &da[da_off..da_off + hd * s],
                    &scores,
                    dv_slice,
                    hd,
                    s,
                    s,
                    1.0,
                    1.0,
                    false,
                    false,
                ); // accumulate for GQA

                // Softmax backward: ds = probs * (dp - rowsum(probs * dp)) * scale
                let mut ds = vec![0.0f32; s * s];
                for row in 0..s {
                    let mut dot = 0.0f32;
                    for col in 0..s {
                        dot += scores[row * s + col] * dp[row * s + col];
                    }
                    for col in 0..s {
                        ds[row * s + col] =
                            scores[row * s + col] * (dp[row * s + col] - dot) * scale;
                    }
                }

                // dQ_h = ds @ K_h: [s, s] @ [hd, s]^T → [s, hd] → channel-first [hd, s]
                // gemm: K_h [hd, s], ds^T [s, s] → K_h @ ds^T → [hd, s], or
                // dQ = K_h @ ds^T = gemm(K_h [hd,s], ds [s,s], hd, s, s, false, true)
                let dq_slice = &mut scratch.get_mut(ids.dq)[h_idx * hd * s..(h_idx + 1) * hd * s];
                accelerate::gemm(
                    &acts.k[k_off..k_off + hd * s],
                    &ds,
                    dq_slice,
                    hd,
                    s,
                    s,
                    1.0,
                    0.0,
                    false,
                    true,
                );

                // dK_h = ds^T @ Q_h: [s, s]^T @ [hd, s]^T → [s, hd] → channel-first [hd, s]
                // dK = Q_h @ ds = gemm(Q_h [hd,s], ds [s,s], hd, s, s, false, false)
                let dk_off = kv_idx * hd * s;
                let dk_slice = &mut scratch.get_mut(ids.dk)[dk_off..dk_off + hd * s];
                accelerate::gemm(
                    &acts.q[q_off..q_off + hd * s],
                    &ds,
                    dk_slice,
                    hd,
                    s,
                    s,
                    1.0,
                    1.0,
                    false,
                    false,
                ); // accumulate for GQA
            }
        } else {
            // Fused attention backward via ANE kernels
            let score_ch = self.kernel_config.n_heads * s;

            // 6. Fused sdpa_bwd1: computes dA (via Wo^T) and sets up for dV/probs/dp.
            let bwd1_sp = s + qd;
            let bwd1_in_ch = qd + 2 * kvd + d;
            let dy_ch_off = qd + 2 * kvd;

            io.sdpa_bwd1_in
                .zero_channel_range_f32(0, bwd1_in_ch, bwd1_sp);
            io.sdpa_bwd1_in
                .write_f32_strided_at(0, &acts.q, qd, s, bwd1_sp);
            io.sdpa_bwd1_in
                .write_f32_strided_at(qd, &acts.k, kvd, s, bwd1_sp);
            io.sdpa_bwd1_in
                .write_f32_strided_at(qd + kvd, &acts.v, kvd, s, bwd1_sp);
            io.sdpa_bwd1_in.write_f32_strided_at(
                dy_ch_off,
                scratch.get(ids.dx_ffn_norm),
                d,
                s,
                bwd1_sp,
            );
            io.sdpa_bwd1_in.write_f32_at_col_offset(
                dy_ch_off,
                &self.layer_weights[l].wo,
                d,
                qd,
                s,
                bwd1_sp,
            );

            let bwd1_model = kernels.sdpa_bwd1.as_ref().ok_or_else(|| {
                MetalError::InvalidConfig("Fused sdpa_bwd1 kernel not compiled".into())
            })?;
            bwd1_model.evaluate(&[io.sdpa_bwd1_in.as_ptr()], &[io.sdpa_bwd1_out.as_ptr()])?;

            io.sdpa_bwd1_out
                .read_fp16_as_f32(scratch.get_mut(ids.dv), 0, kvd, s);

            // dWo
            encode_dw_gemm(
                gpu_dw,
                scratch_a,
                scratch_b,
                batch.as_mut(),
                scratch.get(ids.dx_ffn_norm),
                &self.layer_acts[l].attn_out,
                &self.layer_grads[l].wo,
                d,
                qd,
                s,
            );

            // 7. SDPA backward part 2: Q, K, probs, dp → dQ, dK
            io.sdpa_bwd2_in
                .copy_from(0, &io.sdpa_bwd1_out, kvd, 2 * score_ch, s);
            io.sdpa_bwd2_in
                .write_f32_as_fp16_at(2 * score_ch, &acts.q, qd, s);
            io.sdpa_bwd2_in
                .write_f32_as_fp16_at(2 * score_ch + qd, &acts.k, kvd, s);

            let bwd2_model = kernels.sdpa_bwd2.as_ref().ok_or_else(|| {
                MetalError::InvalidConfig("Fused sdpa_bwd2 kernel not compiled".into())
            })?;
            bwd2_model.evaluate(&[io.sdpa_bwd2_in.as_ptr()], &[io.sdpa_bwd2_out.as_ptr()])?;

            io.sdpa_bwd2_out
                .read_fp16_as_f32(scratch.get_mut(ids.dq), 0, qd, s);
            io.sdpa_bwd2_out
                .read_fp16_as_f32(scratch.get_mut(ids.dk), qd, kvd, s);
        }

        // dWq, dWk, dWv
        encode_dw_gemm(
            gpu_dw,
            scratch_a,
            scratch_b,
            batch.as_mut(),
            scratch.get(ids.dq),
            &self.layer_acts[l].xnorm,
            &self.layer_grads[l].wq,
            qd,
            d,
            s,
        );
        encode_dw_gemm(
            gpu_dw,
            scratch_a,
            scratch_b,
            batch.as_mut(),
            scratch.get(ids.dk),
            &self.layer_acts[l].xnorm,
            &self.layer_grads[l].wk,
            kvd,
            d,
            s,
        );
        encode_dw_gemm(
            gpu_dw,
            scratch_a,
            scratch_b,
            batch.as_mut(),
            scratch.get(ids.dv),
            &self.layer_acts[l].xnorm,
            &self.layer_grads[l].wv,
            kvd,
            d,
            s,
        );

        // 8. QKV backward projections (dxq, dxk, dxv)
        // Copy inputs out to avoid borrow conflicts (scratch.get + scratch.get_mut)
        let dq_copy: Vec<f32> = scratch.get(ids.dq).to_vec();
        Self::run_projection(
            kernels,
            io,
            qd,
            d,
            s,
            &dq_copy,
            &self.layer_weights[l].wq_t,
            scratch.get_mut(ids.dxq),
        )?;
        let dk_copy: Vec<f32> = scratch.get(ids.dk).to_vec();
        Self::run_projection(
            kernels,
            io,
            kvd,
            d,
            s,
            &dk_copy,
            &self.layer_weights[l].wk_t,
            scratch.get_mut(ids.dxk),
        )?;
        let dv_copy: Vec<f32> = scratch.get(ids.dv).to_vec();
        Self::run_projection(
            kernels,
            io,
            kvd,
            d,
            s,
            &dv_copy,
            &self.layer_weights[l].wv_t,
            scratch.get_mut(ids.dxv),
        )?;

        // Combine dxq + dxk + dxv
        {
            let dxq_copy: Vec<f32> = scratch.get(ids.dxq).to_vec();
            let dxk_copy: Vec<f32> = scratch.get(ids.dxk).to_vec();
            accelerate::vadd(&dxq_copy, &dxk_copy, scratch.get_mut(ids.dx_attn));
        }
        {
            let dx_attn_copy: Vec<f32> = scratch.get(ids.dx_attn).to_vec();
            let dxv_copy: Vec<f32> = scratch.get(ids.dxv).to_vec();
            accelerate::vadd(&dx_attn_copy, &dxv_copy, scratch.get_mut(ids.dx_attn));
        }

        // 9. Attention RMSNorm backward
        let dx_attn_copy2: Vec<f32> = scratch.get(ids.dx_attn).to_vec();
        accelerate::rmsnorm_backward(
            scratch.get_mut(ids.dx_attn_norm),
            &mut self.layer_grads[l].rms_att,
            &dx_attn_copy2,
            &self.layer_acts[l].layer_in,
            &self.layer_weights[l].rms_att,
            d,
            s,
            eps,
        );

        // Final dx (residual)
        accelerate::vadd(
            scratch.get(ids.dx_ffn_norm),
            scratch.get(ids.dx_attn_norm),
            dx,
        );

        Ok(())
    }

    /// Run a single training step (forward + backward + grad accumulation).
    ///
    /// When vocab compaction is active, the classifier operates on the compact
    /// embedding (`compact_vocab * dim`) instead of the full one, giving ~3.5x
    /// speedup on the classifier matmul and cross-entropy.
    ///
    /// Weight gradient GEMMs are encoded into `batch` (GPU) or run inline (CPU).
    /// The caller (`train_batch`) creates the batch and calls `execute()`.
    pub fn train_step(
        &mut self,
        input_tokens: &[u16],
        target_tokens: &[u16],
        batch: &mut Option<BatchedCommandBuffer>,
    ) -> Result<f32> {
        let d = self.config.dim;
        let s = self.config.seq_len;

        assert_eq!(input_tokens.len(), s);
        assert_eq!(target_tokens.len(), s);

        // === Forward pass ===
        // Embedding lookup: if vocab compaction is active, input tokens are compact
        // IDs — use the compact embedding table directly. Otherwise use full table.
        let mut x = vec![0.0f32; d * s];
        if self.vocab_map.is_some() {
            let tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
            accelerate::embed_lookup(&mut x, &self.compact_embed, &tokens_u32, d, s);
        } else {
            let tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
            accelerate::embed_lookup(&mut x, &self.embed_weights, &tokens_u32, d, s);
        }

        let fwd_start = std::time::Instant::now();
        for l in 0..self.config.n_layers {
            self.layer_acts[l].layer_in.copy_from_slice(&x);
            self.forward_layer(l, &mut x)?;
        }
        // Flip the dual-die toggle after all layers have run.
        // All layers in a given step use the same kernel set (A or B), so we
        // flip once here rather than inside forward_layer.
        if self.kernels_b.is_some() {
            self.use_kernel_set_b = !self.use_kernel_set_b;
        }
        self.last_timings.ane_fwd_us += fwd_start.elapsed().as_micros() as u64;

        // Final RMSNorm (timed)
        let rms_start = std::time::Instant::now();
        let mut x_final = vec![0.0f32; d * s];
        accelerate::rmsnorm(
            &mut x_final,
            &x,
            &self.rms_final,
            d,
            s,
            self.config.rms_norm_eps,
        );
        self.last_timings.rmsnorm_us += rms_start.elapsed().as_micros() as u64;

        // Classifier + loss + backward: compact or full path
        let (loss, mut dx) = if let Some(ref vm) = self.vocab_map {
            // === Compact classifier path ===
            let cv = vm.compact_vocab;

            // Tokens are ALREADY compact u16 ids (remapped by orchestrator during
            // batch construction). No second remap needed.
            let compact_targets = target_tokens;

            // logits = compact_embed @ x_final: [cv, d] @ [d, s] → [cv, s]
            let mut logits = vec![0.0f32; cv * s];
            accelerate::gemm(
                &self.compact_embed,
                &x_final,
                &mut logits,
                cv,
                s,
                d,
                1.0,
                0.0,
                false,
                false,
            );

            // Cross-entropy loss on compact vocab
            // Try ANE softmax first, fall back to CPU
            let mut dlogits = vec![0.0f32; cv * s];
            let loss_scale = self.config.loss_scale;
            let loss = if let (Some(kernels), Some(io)) = (&self.kernels, &self.io_pool) {
                if let (Some(sm_kern), Some(sm_in), Some(sm_out)) =
                    (&kernels.softmax, &io.softmax_in, &io.softmax_out)
                {
                    // ANE softmax path: logits → ANE → probs → CPU NLL
                    sm_in.write_f32_as_fp16(&logits, cv, s);
                    if let Ok(()) = sm_kern.evaluate(&[sm_in.as_ptr()], &[sm_out.as_ptr()]) {
                        let mut probs = vec![0.0f32; cv * s];
                        sm_out.read_fp16_as_f32(&mut probs, 0, cv, s);
                        let l = accelerate::nll_loss_from_probs(
                            &mut dlogits,
                            &probs,
                            compact_targets,
                            cv,
                            s,
                        );
                        if loss_scale != 1.0 {
                            accelerate::scale_inplace(&mut dlogits, loss_scale);
                        }
                        l
                    } else {
                        // ANE eval failed, fall back to pluggable loss
                        self.loss_fn
                            .as_mut()
                            .unwrap()
                            .compute(&logits, compact_targets, cv, s, loss_scale, &mut dlogits)
                            .loss
                    }
                } else {
                    self.loss_fn
                        .as_mut()
                        .unwrap()
                        .compute(&logits, compact_targets, cv, s, loss_scale, &mut dlogits)
                        .loss
                }
            } else {
                self.loss_fn
                    .as_mut()
                    .unwrap()
                    .compute(&logits, compact_targets, cv, s, loss_scale, &mut dlogits)
                    .loss
            };

            // dx = compact_embed^T @ dlogits: [d, cv] @ [cv, s] → [d, s]
            let mut dx = vec![0.0f32; d * s];
            accelerate::gemm(
                &self.compact_embed,
                &dlogits,
                &mut dx,
                d,
                s,
                cv,
                1.0,
                0.0,
                true,
                false,
            );

            // Compact embed gradient: dE = dlogits @ x_final^T: [cv, s] @ [s, d] → [cv, d]
            // Stays on CPU (embed grads are Vec<f32>, not MetalBuffer — sparse update)
            accelerate::gemm(
                &dlogits,
                &x_final,
                &mut self.compact_embed_grad,
                cv,
                d,
                s,
                1.0,
                1.0,
                false,
                true,
            );

            (loss, dx)
        } else {
            // === Full vocab classifier path ===
            let v = self.config.vocab_size;

            let mut logits = vec![0.0f32; v * s];
            accelerate::gemm(
                &self.embed_weights,
                &x_final,
                &mut logits,
                v,
                s,
                d,
                1.0,
                0.0,
                false,
                false,
            );

            let mut dlogits = vec![0.0f32; v * s];
            let loss = self
                .loss_fn
                .as_mut()
                .unwrap()
                .compute(
                    &logits,
                    target_tokens,
                    v,
                    s,
                    self.config.loss_scale,
                    &mut dlogits,
                )
                .loss;

            let mut dx = vec![0.0f32; d * s];
            accelerate::gemm(
                &self.embed_weights,
                &dlogits,
                &mut dx,
                d,
                s,
                v,
                1.0,
                0.0,
                true,
                false,
            );

            // Full embed gradient accumulation (stays on CPU — too large for GPU scratch)
            accelerate::gemm(
                &dlogits,
                &x_final,
                &mut self.embed_grad,
                v,
                d,
                s,
                1.0,
                1.0,
                false,
                true,
            );

            (loss, dx)
        };

        // Final RMSNorm backward
        let x_before_final = x;
        let dy_final = dx.clone();
        accelerate::rmsnorm_backward(
            &mut dx,
            &mut self.rms_final_grad,
            &dy_final,
            &x_before_final,
            &self.rms_final,
            d,
            s,
            self.config.rms_norm_eps,
        );

        // Per-layer backward (reverse order, timed)
        let bwd_start = std::time::Instant::now();
        for l in (0..self.config.n_layers).rev() {
            self.backward_layer(l, &mut dx, batch)?;
        }
        self.last_timings.ane_bwd_us += bwd_start.elapsed().as_micros() as u64;

        // Embedding backward (scatter: stays CPU, sparse op)
        let tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
        if self.vocab_map.is_some() {
            // Compact tokens → scatter into compact embedding gradient
            accelerate::embed_backward(&mut self.compact_embed_grad, &dx, &tokens_u32, d, s);
        } else {
            accelerate::embed_backward(&mut self.embed_grad, &dx, &tokens_u32, d, s);
        }

        Ok(loss)
    }

    /// Compute learning rate with warmup + cosine decay.
    fn get_lr(&self, step: usize, max_steps: usize) -> f32 {
        let base_lr = self.config.learning_rate;
        let warmup = self.config.warmup_steps;
        let min_lr = base_lr * self.config.min_lr_ratio;

        if step < warmup {
            base_lr * (step as f32 / warmup as f32)
        } else if max_steps > warmup {
            let progress = (step - warmup) as f32 / (max_steps - warmup) as f32;
            min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        } else {
            base_lr
        }
    }

    /// Run a complete training batch (accumulate + Adam update).
    ///
    /// No recompilation needed — weights are injected via IOSurface writes.
    /// GPU dW GEMMs are batched per train_step and executed before gradient ops.
    pub fn train_batch(&mut self, data: &[(Vec<u16>, Vec<u16>)], max_steps: usize) -> Result<f32> {
        let use_compact = self.vocab_map.is_some();

        // Reset step timings
        self.last_timings = StepTimings::default();
        let step_start = std::time::Instant::now();

        // Zero gradients
        for lg in &mut self.layer_grads {
            lg.zero();
        }
        if use_compact {
            self.compact_embed_grad.fill(0.0);
        } else {
            self.embed_grad.fill(0.0);
        }
        self.rms_final_grad.fill(0.0);

        // Accumulate gradients
        let steps = data.len().min(self.config.accum_steps);
        let mut total_loss = 0.0f32;

        for (input, target) in data.iter().take(steps) {
            // Create a BatchedCommandBuffer for this step's dW GEMMs (GPU path).
            // CPU fallback: batch stays None, dispatch_dw runs cblas inline.
            let mut batch = if let Some(ref ctx) = self.metal_ctx {
                BatchedCommandBuffer::new(ctx.clone()).ok()
            } else {
                None
            };

            let loss = self.train_step(input, target, &mut batch)?;
            total_loss += loss;

            // Execute all encoded dW GEMMs for this step (single GPU-CPU sync)
            if let Some(b) = batch {
                if b.dispatch_count() > 0 {
                    b.execute()?;
                }
            }
        }

        // No wait_dw() needed — each step's batch.execute() is synchronous

        // Scale gradients: divide by accum_steps and undo loss scaling.
        // Combined into a single scale pass for efficiency.
        let scale = 1.0 / (steps as f32 * self.config.loss_scale);
        for lg in &mut self.layer_grads {
            accelerate::scale_inplace(lg.wq.as_mut_slice(), scale);
            accelerate::scale_inplace(lg.wk.as_mut_slice(), scale);
            accelerate::scale_inplace(lg.wv.as_mut_slice(), scale);
            accelerate::scale_inplace(lg.wo.as_mut_slice(), scale);
            accelerate::scale_inplace(lg.w1.as_mut_slice(), scale);
            accelerate::scale_inplace(lg.w2.as_mut_slice(), scale);
            accelerate::scale_inplace(lg.w3.as_mut_slice(), scale);
            accelerate::scale_inplace(&mut lg.rms_att, scale);
            accelerate::scale_inplace(&mut lg.rms_ffn, scale);
        }
        if use_compact {
            accelerate::scale_inplace(&mut self.compact_embed_grad, scale);
        } else {
            accelerate::scale_inplace(&mut self.embed_grad, scale);
        }
        accelerate::scale_inplace(&mut self.rms_final_grad, scale);

        // Gradient clipping
        let mut grad_norm_sq = 0.0f32;
        for lg in &self.layer_grads {
            grad_norm_sq += accelerate::sum_of_squares(lg.wq.as_slice());
            grad_norm_sq += accelerate::sum_of_squares(lg.wk.as_slice());
            grad_norm_sq += accelerate::sum_of_squares(lg.wv.as_slice());
            grad_norm_sq += accelerate::sum_of_squares(lg.wo.as_slice());
            grad_norm_sq += accelerate::sum_of_squares(lg.w1.as_slice());
            grad_norm_sq += accelerate::sum_of_squares(lg.w2.as_slice());
            grad_norm_sq += accelerate::sum_of_squares(lg.w3.as_slice());
            grad_norm_sq += accelerate::sum_of_squares(&lg.rms_att);
            grad_norm_sq += accelerate::sum_of_squares(&lg.rms_ffn);
        }
        if use_compact {
            grad_norm_sq += accelerate::sum_of_squares(&self.compact_embed_grad);
        } else {
            grad_norm_sq += accelerate::sum_of_squares(&self.embed_grad);
        }
        grad_norm_sq += accelerate::sum_of_squares(&self.rms_final_grad);

        let grad_norm = grad_norm_sq.sqrt();
        if grad_norm > self.config.gradient_clip_norm {
            let clip_scale = self.config.gradient_clip_norm / grad_norm;
            for lg in &mut self.layer_grads {
                accelerate::scale_inplace(lg.wq.as_mut_slice(), clip_scale);
                accelerate::scale_inplace(lg.wk.as_mut_slice(), clip_scale);
                accelerate::scale_inplace(lg.wv.as_mut_slice(), clip_scale);
                accelerate::scale_inplace(lg.wo.as_mut_slice(), clip_scale);
                accelerate::scale_inplace(lg.w1.as_mut_slice(), clip_scale);
                accelerate::scale_inplace(lg.w2.as_mut_slice(), clip_scale);
                accelerate::scale_inplace(lg.w3.as_mut_slice(), clip_scale);
                accelerate::scale_inplace(&mut lg.rms_att, clip_scale);
                accelerate::scale_inplace(&mut lg.rms_ffn, clip_scale);
            }
            if use_compact {
                accelerate::scale_inplace(&mut self.compact_embed_grad, clip_scale);
            } else {
                accelerate::scale_inplace(&mut self.embed_grad, clip_scale);
            }
            accelerate::scale_inplace(&mut self.rms_final_grad, clip_scale);
        }

        // Adam update (timed)
        let adam_start = std::time::Instant::now();
        self.adam_t += 1;
        let t = self.adam_t;
        let lr = self.get_lr(t, max_steps);
        let b1 = self.config.adam_beta1;
        let b2 = self.config.adam_beta2;
        let eps = self.config.adam_eps;

        for l in 0..self.config.n_layers {
            let lw = &mut self.layer_weights[l];
            let lg = &self.layer_grads[l];
            let la = &mut self.layer_adam[l];

            // Weight gradient fields are MetalBuffer — access via as_slice()
            macro_rules! adam_weight {
                ($w:ident, $g:ident, $a:ident) => {
                    accelerate::adam_update(
                        &mut lw.$w,
                        lg.$g.as_slice(),
                        &mut la.$a.m,
                        &mut la.$a.v,
                        t,
                        lr,
                        b1,
                        b2,
                        eps,
                    );
                };
            }
            // RMSNorm gradient fields are Vec<f32>
            macro_rules! adam_norm {
                ($w:ident, $g:ident, $a:ident) => {
                    accelerate::adam_update(
                        &mut lw.$w,
                        &lg.$g,
                        &mut la.$a.m,
                        &mut la.$a.v,
                        t,
                        lr,
                        b1,
                        b2,
                        eps,
                    );
                };
            }

            adam_weight!(wq, wq, wq);
            adam_weight!(wk, wk, wk);
            adam_weight!(wv, wv, wv);
            adam_weight!(wo, wo, wo);
            adam_weight!(w1, w1, w1);
            adam_weight!(w2, w2, w2);
            adam_weight!(w3, w3, w3);
            adam_norm!(rms_att, rms_att, rms_att);
            adam_norm!(rms_ffn, rms_ffn, rms_ffn);
        }

        // Embedding uses a separate LR if configured (prevents divergence
        // from sparse token updates and magnitude drift through many layers)
        let embed_lr = self.config.embedding_lr.unwrap_or(lr);

        if use_compact {
            // Adam update on compact embedding, then scatter back to full
            accelerate::adam_update(
                &mut self.compact_embed,
                &self.compact_embed_grad,
                &mut self.compact_embed_adam.m,
                &mut self.compact_embed_adam.v,
                t,
                embed_lr,
                b1,
                b2,
                eps,
            );
            self.scatter_compact_to_full();
        } else {
            accelerate::adam_update(
                &mut self.embed_weights,
                &self.embed_grad,
                &mut self.embed_adam.m,
                &mut self.embed_adam.v,
                t,
                embed_lr,
                b1,
                b2,
                eps,
            );
        }

        accelerate::adam_update(
            &mut self.rms_final,
            &self.rms_final_grad,
            &mut self.rms_final_adam.m,
            &mut self.rms_final_adam.v,
            t,
            lr,
            b1,
            b2,
            eps,
        );

        self.last_timings.adam_us = adam_start.elapsed().as_micros() as u64;

        // Refresh transposed weights for backward kernels
        self.refresh_transposed_weights();

        // Record total step time
        self.last_timings.total_us = step_start.elapsed().as_micros() as u64;

        Ok(total_loss / steps as f32)
    }

    /// Load weights from flat f32 arrays (llama2.c format).
    pub fn load_weights_flat(&mut self, weights: &[f32]) {
        let d = self.config.dim;
        let h = self.config.hidden_dim;
        let nl = self.config.n_layers;
        let v = self.config.vocab_size;
        let qd = self.kernel_config.q_dim();
        let kvd = self.kernel_config.kv_dim();

        let mut offset = 0;

        self.embed_weights[..v * d].copy_from_slice(&weights[offset..offset + v * d]);
        offset += v * d;

        for l in 0..nl {
            let lw = &mut self.layer_weights[l];

            lw.rms_att.copy_from_slice(&weights[offset..offset + d]);
            offset += d;
            lw.wq.copy_from_slice(&weights[offset..offset + qd * d]);
            offset += qd * d;
            lw.wk.copy_from_slice(&weights[offset..offset + kvd * d]);
            offset += kvd * d;
            lw.wv.copy_from_slice(&weights[offset..offset + kvd * d]);
            offset += kvd * d;
            lw.wo.copy_from_slice(&weights[offset..offset + d * qd]);
            offset += d * qd;
            lw.rms_ffn.copy_from_slice(&weights[offset..offset + d]);
            offset += d;
            lw.w1.copy_from_slice(&weights[offset..offset + h * d]);
            offset += h * d;
            lw.w2.copy_from_slice(&weights[offset..offset + d * h]);
            offset += d * h;
            lw.w3.copy_from_slice(&weights[offset..offset + h * d]);
            offset += h * d;
        }

        self.rms_final.copy_from_slice(&weights[offset..offset + d]);

        // Initialize transposed weights
        self.refresh_transposed_weights();
    }

    /// Load weights from SafeTensors files on disk.
    pub fn load_weights_safetensors(&mut self, path: &std::path::Path) -> Result<()> {
        use memmap2::Mmap;
        use safetensors::SafeTensors;

        let d = self.config.dim;
        let h = self.config.hidden_dim;

        fn st_to_f32(tensor: &safetensors::tensor::TensorView<'_>) -> Option<Vec<f32>> {
            use safetensors::Dtype;
            match tensor.dtype() {
                Dtype::F32 => {
                    let bytes = tensor.data();
                    if bytes.len() % 4 != 0 {
                        return None;
                    }
                    let n = bytes.len() / 4;
                    let mut out = vec![0.0f32; n];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            out.as_mut_ptr() as *mut u8,
                            n * 4,
                        );
                    }
                    Some(out)
                }
                Dtype::F16 => {
                    let bytes = tensor.data();
                    if bytes.len() % 2 != 0 {
                        return None;
                    }
                    let n = bytes.len() / 2;
                    let mut out = vec![0.0f32; n];
                    for i in 0..n {
                        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                        out[i] = half::f16::from_bits(bits).to_f32();
                    }
                    Some(out)
                }
                Dtype::BF16 => {
                    let bytes = tensor.data();
                    if bytes.len() % 2 != 0 {
                        return None;
                    }
                    let n = bytes.len() / 2;
                    let mut out = vec![0.0f32; n];
                    for i in 0..n {
                        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                        out[i] = f32::from_bits((bits as u32) << 16);
                    }
                    Some(out)
                }
                _ => None,
            }
        }

        fn copy_w(src: &[f32], dst: &mut [f32], expected: usize) {
            let n = src.len().min(expected).min(dst.len());
            dst[..n].copy_from_slice(&src[..n]);
        }

        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            let index_path = path.join("model.safetensors.index.json");
            if index_path.exists() {
                let index_text = std::fs::read_to_string(&index_path).map_err(|e| {
                    MetalError::InvalidConfig(format!("Failed to read index.json: {e}"))
                })?;
                let index: serde_json::Value = serde_json::from_str(&index_text).map_err(|e| {
                    MetalError::InvalidConfig(format!("Failed to parse index.json: {e}"))
                })?;
                let weight_map = index["weight_map"]
                    .as_object()
                    .ok_or_else(|| MetalError::InvalidConfig("Missing weight_map".into()))?;
                let mut unique_files: Vec<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                unique_files.sort();
                unique_files.dedup();
                unique_files.iter().map(|f| path.join(f)).collect()
            } else {
                let single = path.join("model.safetensors");
                if single.exists() {
                    vec![single]
                } else {
                    return Err(MetalError::InvalidConfig(
                        "No safetensors files found".into(),
                    ));
                }
            }
        };

        for file_path in &files {
            let file = std::fs::File::open(file_path).map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to open {:?}: {e}", file_path))
            })?;
            #[allow(unsafe_code)]
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to mmap {:?}: {e}", file_path))
            })?;
            let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to parse safetensors: {e}"))
            })?;

            for (name, tensor) in tensors.tensors() {
                let data = match st_to_f32(&tensor) {
                    Some(d) => d,
                    None => continue,
                };

                if name == "model.embed_tokens.weight" {
                    let expected = self.config.vocab_size * d;
                    copy_w(&data, &mut self.embed_weights, expected);
                    continue;
                }
                if name == "model.norm.weight" {
                    copy_w(&data, &mut self.rms_final, d);
                    continue;
                }

                if let Some(rest) = name.strip_prefix("model.layers.") {
                    let parts: Vec<&str> = rest.splitn(2, '.').collect();
                    if parts.len() < 2 {
                        continue;
                    }
                    let layer_idx: usize = match parts[0].parse() {
                        Ok(i) => i,
                        Err(_) => continue,
                    };
                    if layer_idx >= self.config.n_layers {
                        continue;
                    }

                    let qd = self.kernel_config.q_dim();
                    let kvd = self.kernel_config.kv_dim();
                    let lw = &mut self.layer_weights[layer_idx];
                    match parts[1] {
                        "self_attn.q_proj.weight" => copy_w(&data, &mut lw.wq, qd * d),
                        "self_attn.k_proj.weight" => copy_w(&data, &mut lw.wk, kvd * d),
                        "self_attn.v_proj.weight" => copy_w(&data, &mut lw.wv, kvd * d),
                        "self_attn.o_proj.weight" => copy_w(&data, &mut lw.wo, d * qd),
                        "mlp.gate_proj.weight" => copy_w(&data, &mut lw.w1, h * d),
                        "mlp.down_proj.weight" => copy_w(&data, &mut lw.w2, d * h),
                        "mlp.up_proj.weight" => copy_w(&data, &mut lw.w3, h * d),
                        "input_layernorm.weight" => copy_w(&data, &mut lw.rms_att, d),
                        "post_attention_layernorm.weight" => copy_w(&data, &mut lw.rms_ffn, d),
                        _ => {}
                    }
                }
            }
        }

        // Initialize transposed weights
        self.refresh_transposed_weights();

        Ok(())
    }
}

/// Transpose a [rows, cols] row-major matrix to [cols, rows].
fn transpose_weight(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    accelerate::matrix_transpose(dst, src, rows, cols);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_trainer_creation() {
        let config = DynamicAneTrainerConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 4,
            n_layers: 2,
            vocab_size: 100,
            seq_len: 16,
            ..Default::default()
        };
        let trainer = DynamicAneTrainer::new(config);
        assert_eq!(trainer.adam_t(), 0);
        assert_eq!(trainer.compile_count(), 0);
        assert_eq!(trainer.config().n_layers, 2);
    }

    #[test]
    fn test_dynamic_trainer_gqa_creation() {
        let config = DynamicAneTrainerConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 2,
            n_layers: 2,
            vocab_size: 100,
            seq_len: 16,
            ..Default::default()
        };
        let trainer = DynamicAneTrainer::new(config);
        assert_eq!(trainer.kernel_config.n_heads, 4);
        assert_eq!(trainer.kernel_config.n_kv_heads, 2);
        assert_eq!(trainer.kernel_config.n_groups(), 2);
    }

    #[test]
    fn test_dynamic_trainer_no_recompile_needed() {
        let config = DynamicAneTrainerConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 4,
            n_layers: 2,
            vocab_size: 100,
            seq_len: 16,
            ..Default::default()
        };
        let trainer = DynamicAneTrainer::new(config);
        assert_eq!(trainer.compile_count(), 0);
    }

    // ======== Vocab Compaction Tests ========

    #[test]
    fn test_vocab_map_from_batches() {
        let batches = vec![vec![
            (vec![0u16, 5, 10, 50], vec![5u16, 10, 50, 99]),
            (vec![1u16, 3, 5, 10], vec![3u16, 5, 10, 50]),
        ]];
        let vm = VocabMap::from_batches(&batches, 100);

        // Used tokens: 0, 1, 3, 5, 10, 50, 99 = 7 unique
        assert_eq!(vm.compact_vocab, 7);
        assert_eq!(vm.compact_to_full.len(), 7);

        // Verify roundtrip: full→compact→full
        for &full_id in &vm.compact_to_full {
            let compact_id = vm.full_to_compact[full_id];
            assert!(compact_id >= 0);
            assert_eq!(vm.compact_to_full[compact_id as usize], full_id);
        }

        // Verify unused tokens map to -1
        assert_eq!(vm.full_to_compact[2], -1);
        assert_eq!(vm.full_to_compact[4], -1);
        assert_eq!(vm.full_to_compact[98], -1);
    }

    #[test]
    fn test_vocab_map_remap_tokens() {
        let batches = vec![vec![(vec![0u16, 5, 10], vec![5u16, 10, 0])]];
        let vm = VocabMap::from_batches(&batches, 100);
        assert_eq!(vm.compact_vocab, 3);

        let remapped = vm.remap_tokens(&[0, 5, 10]);
        // All remapped ids should be in [0, 3)
        for &r in &remapped {
            assert!((r as usize) < vm.compact_vocab);
        }

        // Same full token → same compact token
        let r2 = vm.remap_tokens(&[5, 5, 0]);
        assert_eq!(r2[0], r2[1]); // both are token 5
    }

    #[test]
    fn test_vocab_compaction_install() {
        let config = DynamicAneTrainerConfig {
            dim: 8,
            hidden_dim: 16,
            n_heads: 2,
            n_kv_heads: 2,
            n_layers: 1,
            vocab_size: 100,
            seq_len: 4,
            ..Default::default()
        };
        let mut trainer = DynamicAneTrainer::new(config);

        // Set some embedding weights
        for i in 0..trainer.embed_weights.len() {
            trainer.embed_weights[i] = i as f32 * 0.01;
        }

        let batches = vec![vec![(vec![0u16, 5, 10, 50], vec![5u16, 10, 50, 0])]];
        let vm = VocabMap::from_batches(&batches, 100);
        assert_eq!(vm.compact_vocab, 4);

        trainer.install_vocab_map(vm);
        assert!(trainer.has_vocab_compaction());
        assert_eq!(trainer.compact_embed.len(), 4 * 8);

        // Verify compact embed contains correct rows
        let vm = trainer.vocab_map.as_ref().unwrap();
        for c in 0..vm.compact_vocab {
            let full_row = vm.compact_to_full[c] * 8;
            let compact_row = c * 8;
            assert_eq!(
                &trainer.compact_embed[compact_row..compact_row + 8],
                &trainer.embed_weights[full_row..full_row + 8]
            );
        }
    }

    // ======== Architecture Validation Tests ========

    #[test]
    fn test_ane_compatible_llama() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
        });
        assert!(DynamicAneTrainerConfig::is_ane_compatible(&config).is_ok());
    }

    #[test]
    fn test_ane_compatible_qwen3() {
        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
        });
        assert!(DynamicAneTrainerConfig::is_ane_compatible(&config).is_ok());
    }

    #[test]
    fn test_ane_incompatible_qwen3_5() {
        let config = serde_json::json!({
            "model_type": "qwen3_next",
            "hidden_size": 768,
            "num_attention_heads": 12,
        });
        let result = DynamicAneTrainerConfig::is_ane_compatible(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("hybrid/recurrent"));
    }

    #[test]
    fn test_ane_incompatible_nemotron_h() {
        let config = serde_json::json!({
            "model_type": "nemotron_h",
            "hidden_size": 768,
        });
        assert!(DynamicAneTrainerConfig::is_ane_compatible(&config).is_err());
    }

    #[test]
    fn test_ane_incompatible_jamba() {
        let config = serde_json::json!({
            "model_type": "jamba",
            "hidden_size": 768,
        });
        assert!(DynamicAneTrainerConfig::is_ane_compatible(&config).is_err());
    }

    #[test]
    fn test_ane_incompatible_gemma4() {
        let config = serde_json::json!({
            "model_type": "gemma4_text",
            "hidden_size": 768,
            "num_attention_heads": 12,
        });
        let result = DynamicAneTrainerConfig::is_ane_compatible(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("hybrid/recurrent"));
    }

    #[test]
    fn test_ane_incompatible_moe() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 768,
            "num_experts": 8,
        });
        let result = DynamicAneTrainerConfig::is_ane_compatible(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("MoE"));
    }

    #[test]
    fn test_ane_incompatible_moe_local_experts() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 768,
            "num_local_experts": 8,
        });
        assert!(DynamicAneTrainerConfig::is_ane_compatible(&config).is_err());
    }
}
