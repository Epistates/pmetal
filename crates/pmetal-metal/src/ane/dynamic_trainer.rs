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

use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;

use tracing::{debug, info};

use crate::accelerate;
use crate::ane::dynamic_kernel::{self, DynamicKernelConfig, DynamicKernelOutput};
use crate::ane::iosurface::IoSurface;
use crate::ane::kernel::TransformerKernelConfig;
use crate::ane::runtime::{AneModel, AneRuntime};
use crate::error::{MetalError, Result};

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
struct LayerGradients {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    rms_att: Vec<f32>,
    rms_ffn: Vec<f32>,
}

impl LayerGradients {
    fn zero(&mut self) {
        self.wq.fill(0.0);
        self.wk.fill(0.0);
        self.wv.fill(0.0);
        self.wo.fill(0.0);
        self.w1.fill(0.0);
        self.w2.fill(0.0);
        self.w3.fill(0.0);
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
    sdpa_fwd: AneModel,
    /// Fused FFN forward (RMSNorm + W1 + W3 + SiLU + W2).
    ffn_fwd: AneModel,
    /// SDPA backward part 1 (dV, probs, dp).
    sdpa_bwd1: AneModel,
    /// SDPA backward part 2 (dQ, dK).
    sdpa_bwd2: AneModel,
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
    /// Channel for dispatching async dW tasks to the cblas worker thread.
    dw_sender: mpsc::Sender<Box<dyn FnOnce() + Send>>,
    /// Worker thread handle.
    _dw_thread: thread::JoinHandle<()>,
    /// Latest step timings.
    pub last_timings: StepTimings,
    // --- Vocab compaction state ---
    /// Optional vocab compaction map (built from training data).
    vocab_map: Option<VocabMap>,
    /// Compact embedding matrix `[compact_vocab * dim]`.
    compact_embed: Vec<f32>,
    /// Compact embedding gradient accumulator.
    compact_embed_grad: Vec<f32>,
    /// Compact embedding Adam state.
    compact_embed_adam: AdamParam,
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

            layer_grads.push(LayerGradients {
                wq: vec![0.0; qd * d],
                wk: vec![0.0; kvd * d],
                wv: vec![0.0; kvd * d],
                wo: vec![0.0; d * qd],
                w1: vec![0.0; h * d],
                w2: vec![0.0; d * h],
                w3: vec![0.0; h * d],
                rms_att: vec![0.0; d],
                rms_ffn: vec![0.0; d],
            });

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

        // Spawn async dW worker thread (serial queue semantics)
        let (dw_sender, dw_receiver) = mpsc::channel::<Box<dyn FnOnce() + Send>>();

        let dw_thread = thread::Builder::new()
            .name("ane-dw-cblas".to_string())
            .spawn(move || {
                while let Ok(task) = dw_receiver.recv() {
                    task();
                }
            })
            .expect("Failed to spawn dW worker thread");

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
            dw_sender,
            _dw_thread: dw_thread,
            last_timings: StepTimings::default(),
            vocab_map: None,
            compact_embed: Vec::new(),
            compact_embed_grad: Vec::new(),
            compact_embed_adam: AdamParam::new(0),
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
                if let Err(e) = rt.compile(out.mil_text.as_bytes(), wd) {
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
                    return Err(e);
                }
                rt.compile(out.mil_text.as_bytes(), wd)
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

        // 2. Compile projection kernels (retained for flexibility/caching)
        let mut projections = HashMap::new();
        let mut proj_inputs = HashMap::new();
        let mut proj_outputs = HashMap::new();
        for &(ic, oc) in &proj_shapes {
            let out = dynamic_kernel::gen_dynamic_projection(ic, oc, s);
            let model = compile(&out, rt, &format!("proj_{ic}x{oc}"))?;
            projections.insert((ic, oc), model);
            proj_inputs.insert((ic, oc), IoSurface::for_tensor_f32(ic, s + oc)?);
            proj_outputs.insert((ic, oc), IoSurface::for_tensor_f32(oc, s)?);
            self.compile_count += 1;
        }

        // 3. Compile Fused Forward Kernels
        let k1 = dynamic_kernel::gen_dynamic_sdpa_fwd(&dkc);
        let sdpa_fwd = compile(&k1, rt, "sdpa_fwd")?;
        self.compile_count += 1;

        let k2 = dynamic_kernel::gen_dynamic_ffn_w13(&dkc);
        let ffn_fwd = compile(&k2, rt, "ffn_fwd")?;
        self.compile_count += 1;

        // 4. Compile Backward Kernels
        let k7 = dynamic_kernel::gen_dynamic_sdpa_bwd1(&dkc);
        let sdpa_bwd1 = compile(&k7, rt, "sdpa_bwd1")?;
        self.compile_count += 1;

        let k8 = dynamic_kernel::gen_dynamic_sdpa_bwd2(&dkc);
        let sdpa_bwd2 = compile(&k8, rt, "sdpa_bwd2")?;
        self.compile_count += 1;

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
            let sm_out = dynamic_kernel::gen_dynamic_softmax(softmax_vocab, s);
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

        Ok(())
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

    /// Dispatch an async weight gradient task to the cblas worker thread.
    fn dispatch_dw(&self, task: Box<dyn FnOnce() + Send>) {
        let _ = self.dw_sender.send(task);
    }

    /// Wait for all pending dW tasks to complete.
    fn wait_dw(&self) {
        let (done_tx, done_rx) = mpsc::channel();
        let _ = self.dw_sender.send(Box::new(move || {
            let _ = done_tx.send(());
        }));
        let _ = done_rx.recv();
    }

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

        let kernels = self
            .kernels
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?;
        let io = self
            .io_pool
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?;

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

        kernels
            .sdpa_fwd
            .evaluate(&[io.sdpa_fwd_in.as_ptr()], &[io.sdpa_fwd_out.as_ptr()])?;

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

        // Residual: x2 = x + o_out (CPU for now, or could be in ANE if we pass x)
        accelerate::vadd(x, &acts.o_out, &mut acts.x2);

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

        // ====== Fused FFN Block (ANE) ======
        // Input layout: x2 [d, s], rms_ffn [d, 1], w1 [d, h], w3 [d, h]
        io.ffn_fwd_in.write_packed_f32(
            &acts.x2,
            &[(&lw.rms_ffn, 1), (&lw.w1, h), (&lw.w3, h)],
            d,
            s,
        );

        kernels
            .ffn_fwd
            .evaluate(&[io.ffn_fwd_in.as_ptr()], &[io.ffn_fwd_out.as_ptr()])?;

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

        // W2 projection (standalone for now to keep things simple)
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

        Ok(())
    }

    /// Backward pass for a single layer using decomposed ANE kernels.
    ///
    /// Each weight-transpose projection is a separate ANE kernel dispatch.
    /// Results are combined on CPU via vadd.
    fn backward_layer(&mut self, l: usize, dx: &mut [f32]) -> Result<()> {
        let d = self.config.dim;
        let h = self.config.hidden_dim;
        let s = self.config.seq_len;
        let qd = self.kernel_config.q_dim();
        let kvd = self.kernel_config.kv_dim();
        let score_ch = self.kernel_config.score_ch();

        let kernels = self
            .kernels
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?;
        let io = self
            .io_pool
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?;

        // ====== FFN Backward ======

        // 1. dffn @ W2^T → dsilu_raw (fused ANE kernel)
        io.ffn_bwd_w2t_in
            .write_packed_f32(dx, &[(&self.layer_weights[l].w2_t, h)], d, s);
        kernels.ffn_bwd_w2t.evaluate(
            &[io.ffn_bwd_w2t_in.as_ptr()],
            &[io.ffn_bwd_w2t_out.as_ptr()],
        )?;
        let mut dsilu_raw = vec![0.0f32; h * s];
        io.ffn_bwd_w2t_out.read_f32(&mut dsilu_raw, 0, h, s);

        // 2. Async dW2 = dx @ silu_out^T (on cblas worker)
        {
            let dy_clone = dx[..d * s].to_vec();
            let silu_clone = self.layer_acts[l].silu_out.clone();
            let w2_grad_ptr = self.layer_grads[l].w2.as_mut_ptr() as usize;
            let w2_grad_len = self.layer_grads[l].w2.len();
            let d_val = d;
            let h_val = h;
            let s_val = s;
            self.dispatch_dw(Box::new(move || {
                let w2_grad =
                    unsafe { std::slice::from_raw_parts_mut(w2_grad_ptr as *mut f32, w2_grad_len) };
                accelerate::gemm(
                    &dy_clone,
                    &silu_clone,
                    w2_grad,
                    d_val,
                    h_val,
                    s_val,
                    1.0,
                    1.0,
                    false,
                    true,
                );
            }));
        }

        // 3. SiLU derivative on CPU
        let acts = &self.layer_acts[l];
        let mut dh1 = vec![0.0f32; h * s];
        let mut dh3 = vec![0.0f32; h * s];
        for i in 0..(h * s) {
            let h1_val = acts.h1[i];
            let sig = 1.0 / (1.0 + (-h1_val).exp());
            let silu_d = sig * (1.0 + h1_val * (1.0 - sig));
            dh1[i] = dsilu_raw[i] * acts.h3[i] * silu_d;
            dh3[i] = dsilu_raw[i] * h1_val * sig;
        }

        // Async dW1, dW3
        {
            let dh1_clone = dh1.clone();
            let dh3_clone = dh3.clone();
            let x2norm_clone = acts.x2norm.to_vec();
            let w1_grad_ptr = self.layer_grads[l].w1.as_mut_ptr() as usize;
            let w1_grad_len = self.layer_grads[l].w1.len();
            let w3_grad_ptr = self.layer_grads[l].w3.as_mut_ptr() as usize;
            let w3_grad_len = self.layer_grads[l].w3.len();
            let d_val = d;
            let h_val = h;
            let s_val = s;
            self.dispatch_dw(Box::new(move || {
                let w1_grad =
                    unsafe { std::slice::from_raw_parts_mut(w1_grad_ptr as *mut f32, w1_grad_len) };
                accelerate::gemm(
                    &dh1_clone,
                    &x2norm_clone,
                    w1_grad,
                    h_val,
                    d_val,
                    s_val,
                    1.0,
                    1.0,
                    false,
                    true,
                );
                let w3_grad =
                    unsafe { std::slice::from_raw_parts_mut(w3_grad_ptr as *mut f32, w3_grad_len) };
                accelerate::gemm(
                    &dh3_clone,
                    &x2norm_clone,
                    w3_grad,
                    h_val,
                    d_val,
                    s_val,
                    1.0,
                    1.0,
                    false,
                    true,
                );
            }));
        }

        // 4. dh1@W1^T + dh3@W3^T → dx_ffn (fused ANE kernel)
        io.ffn_bwd_w13t_in.write_packed_f32(
            &dh1,
            &[
                (&dh3, s),
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
        let mut dx_ffn = vec![0.0f32; d * s];
        io.ffn_bwd_w13t_out.read_f32(&mut dx_ffn, 0, d, s);

        // 5. FFN RMSNorm backward (CPU)
        let mut dx_ffn_norm = vec![0.0f32; d * s];
        accelerate::rmsnorm_backward(
            &mut dx_ffn_norm,
            &mut self.layer_grads[l].rms_ffn,
            &dx_ffn,
            &self.layer_acts[l].x2,
            &self.layer_weights[l].rms_ffn,
            d,
            s,
            self.config.rms_norm_eps,
        );

        // ====== Attention Backward ======

        // 6. Fused sdpa_bwd1: computes dA (via Wo^T) and sets up for dV/probs/dp.
        //
        // Surface layout: [in_ch = qd+2*kvd+d, sp = s+qd] fp32
        //   ch [0 .. qd]:            Q activations in sp[0..s], zeros in sp[s..s+qd]
        //   ch [qd .. qd+kvd]:       K activations in sp[0..s], zeros in sp[s..s+qd]
        //   ch [qd+kvd .. qd+2*kvd]: V activations in sp[0..s], zeros in sp[s..s+qd]
        //   ch [qd+2*kvd .. in_ch]:  dy activations in sp[0..s],
        //                            Wo^T rows in sp[s..s+qd]  (wo [d,qd] row-major)
        //
        // wo [d, qd] row-major gives exactly the Wo^T rows needed: row `ch` of `wo`
        // is `Wo^T[ch, 0..qd]` (output-projection weight row for input channel ch).
        let bwd1_sp = s + qd;
        let bwd1_in_ch = qd + 2 * kvd + d;
        let dy_ch_off = qd + 2 * kvd;

        // Zero the entire surface once to clear stale weight-column values in the
        // Q/K/V rows (those rows have activation data in sp[0..s] only; sp[s..s+qd]
        // must be zero so the kernel's Wo^T matmul sees zeros there).
        io.sdpa_bwd1_in
            .zero_channel_range_f32(0, bwd1_in_ch, bwd1_sp);

        // Q [qd, s] → ch 0..qd, sp[0..s]
        io.sdpa_bwd1_in
            .write_f32_strided_at(0, &acts.q, qd, s, bwd1_sp);

        // K [kvd, s] → ch qd..qd+kvd, sp[0..s]
        io.sdpa_bwd1_in
            .write_f32_strided_at(qd, &acts.k, kvd, s, bwd1_sp);

        // V [kvd, s] → ch qd+kvd..qd+2*kvd, sp[0..s]
        io.sdpa_bwd1_in
            .write_f32_strided_at(qd + kvd, &acts.v, kvd, s, bwd1_sp);

        // dy [d, s] → ch dy_ch_off..in_ch, sp[0..s]
        io.sdpa_bwd1_in
            .write_f32_strided_at(dy_ch_off, &dx_ffn_norm, d, s, bwd1_sp);

        // Wo^T = wo [d, qd] → ch dy_ch_off..in_ch, sp[s..s+qd]
        io.sdpa_bwd1_in.write_f32_at_col_offset(
            dy_ch_off,
            &self.layer_weights[l].wo,
            d,
            qd,
            s,
            bwd1_sp,
        );

        kernels
            .sdpa_bwd1
            .evaluate(&[io.sdpa_bwd1_in.as_ptr()], &[io.sdpa_bwd1_out.as_ptr()])?;

        let mut dv = vec![0.0f32; kvd * s];
        io.sdpa_bwd1_out.read_fp16_as_f32(&mut dv, 0, kvd, s);

        // Async dWo
        {
            let dx_clone = dx_ffn_norm.clone();
            let attn_clone = self.layer_acts[l].attn_out.to_vec();
            let wo_grad_ptr = self.layer_grads[l].wo.as_mut_ptr() as usize;
            let wo_grad_len = self.layer_grads[l].wo.len();
            let d_val = d;
            let qd_val = qd;
            let s_val = s;
            self.dispatch_dw(Box::new(move || {
                let wo_grad =
                    unsafe { std::slice::from_raw_parts_mut(wo_grad_ptr as *mut f32, wo_grad_len) };
                accelerate::gemm(
                    &dx_clone,
                    &attn_clone,
                    wo_grad,
                    d_val,
                    qd_val,
                    s_val,
                    1.0,
                    1.0,
                    false,
                    true,
                );
            }));
        }

        // 7. SDPA backward part 2: Q, K, probs, dp → dQ, dK
        io.sdpa_bwd2_in
            .copy_from(0, &io.sdpa_bwd1_out, kvd, 2 * score_ch, s);
        io.sdpa_bwd2_in
            .write_f32_as_fp16_at(2 * score_ch, &acts.q, qd, s);
        io.sdpa_bwd2_in
            .write_f32_as_fp16_at(2 * score_ch + qd, &acts.k, kvd, s);

        kernels
            .sdpa_bwd2
            .evaluate(&[io.sdpa_bwd2_in.as_ptr()], &[io.sdpa_bwd2_out.as_ptr()])?;

        let mut dq = vec![0.0f32; qd * s];
        let mut dk = vec![0.0f32; kvd * s];
        io.sdpa_bwd2_out.read_fp16_as_f32(&mut dq, 0, qd, s);
        io.sdpa_bwd2_out.read_fp16_as_f32(&mut dk, qd, kvd, s);

        // Async dWq, dWk, dWv
        {
            let dq_clone = dq.clone();
            let dk_clone = dk.clone();
            let dv_clone = dv.clone();
            let xnorm_clone = self.layer_acts[l].xnorm.to_vec();

            let wq_grad_ptr = self.layer_grads[l].wq.as_mut_ptr() as usize;
            let wq_grad_len = self.layer_grads[l].wq.len();
            let wk_grad_ptr = self.layer_grads[l].wk.as_mut_ptr() as usize;
            let wk_grad_len = self.layer_grads[l].wk.len();
            let wv_grad_ptr = self.layer_grads[l].wv.as_mut_ptr() as usize;
            let wv_grad_len = self.layer_grads[l].wv.len();

            let d_val = d;
            let qd_val = qd;
            let kvd_val = kvd;
            let s_val = s;

            self.dispatch_dw(Box::new(move || {
                let wq_grad =
                    unsafe { std::slice::from_raw_parts_mut(wq_grad_ptr as *mut f32, wq_grad_len) };
                accelerate::gemm(
                    &dq_clone,
                    &xnorm_clone,
                    wq_grad,
                    qd_val,
                    d_val,
                    s_val,
                    1.0,
                    1.0,
                    false,
                    true,
                );
                let wk_grad =
                    unsafe { std::slice::from_raw_parts_mut(wk_grad_ptr as *mut f32, wk_grad_len) };
                accelerate::gemm(
                    &dk_clone,
                    &xnorm_clone,
                    wk_grad,
                    kvd_val,
                    d_val,
                    s_val,
                    1.0,
                    1.0,
                    false,
                    true,
                );
                let wv_grad =
                    unsafe { std::slice::from_raw_parts_mut(wv_grad_ptr as *mut f32, wv_grad_len) };
                accelerate::gemm(
                    &dv_clone,
                    &xnorm_clone,
                    wv_grad,
                    kvd_val,
                    d_val,
                    s_val,
                    1.0,
                    1.0,
                    false,
                    true,
                );
            }));
        }

        // 8. QKV backward projections (dxq, dxk, dxv)
        let mut dxq = vec![0.0f32; d * s];
        Self::run_projection(
            kernels,
            io,
            qd,
            d,
            s,
            &dq,
            &self.layer_weights[l].wq_t,
            &mut dxq,
        )?;
        let mut dxk = vec![0.0f32; d * s];
        Self::run_projection(
            kernels,
            io,
            kvd,
            d,
            s,
            &dk,
            &self.layer_weights[l].wk_t,
            &mut dxk,
        )?;
        let mut dxv = vec![0.0f32; d * s];
        Self::run_projection(
            kernels,
            io,
            kvd,
            d,
            s,
            &dv,
            &self.layer_weights[l].wv_t,
            &mut dxv,
        )?;

        let mut dx_attn = vec![0.0f32; d * s];
        accelerate::vadd(&dxq, &dxk, &mut dx_attn);
        accelerate::vadd(&dx_attn, &dxv, &mut dxq); // reuse dxq
        dx_attn.copy_from_slice(&dxq);

        // 9. Attention RMSNorm backward
        let mut dx_attn_norm = vec![0.0f32; d * s];
        accelerate::rmsnorm_backward(
            &mut dx_attn_norm,
            &mut self.layer_grads[l].rms_att,
            &dx_attn,
            &self.layer_acts[l].layer_in,
            &self.layer_weights[l].rms_att,
            d,
            s,
            self.config.rms_norm_eps,
        );

        // Final dx (residual)
        accelerate::vadd(&dx_ffn_norm, &dx_attn_norm, dx);

        Ok(())
    }

    /// Run a single training step (forward + backward + grad accumulation).
    ///
    /// When vocab compaction is active, the classifier operates on the compact
    /// embedding (`compact_vocab * dim`) instead of the full one, giving ~3.5x
    /// speedup on the classifier matmul and cross-entropy.
    pub fn train_step(&mut self, input_tokens: &[u16], target_tokens: &[u16]) -> Result<f32> {
        let d = self.config.dim;
        let s = self.config.seq_len;

        assert_eq!(input_tokens.len(), s);
        assert_eq!(target_tokens.len(), s);

        // === Forward pass ===
        // Embedding lookup always uses the full table (input tokens are full-vocab ids)
        let tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
        let mut x = vec![0.0f32; d * s];
        accelerate::embed_lookup(&mut x, &self.embed_weights, &tokens_u32, d, s);

        for l in 0..self.config.n_layers {
            self.layer_acts[l].layer_in.copy_from_slice(&x);
            self.forward_layer(l, &mut x)?;
        }

        // Final RMSNorm
        let mut x_final = vec![0.0f32; d * s];
        accelerate::rmsnorm(
            &mut x_final,
            &x,
            &self.rms_final,
            d,
            s,
            self.config.rms_norm_eps,
        );

        // Classifier + loss + backward: compact or full path
        let (loss, mut dx) = if let Some(ref vm) = self.vocab_map {
            // === Compact classifier path ===
            let cv = vm.compact_vocab;

            // Remap target tokens to compact ids
            let compact_targets = vm.remap_tokens(target_tokens);

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
            let loss = if let (Some(kernels), Some(io)) = (&self.kernels, &self.io_pool) {
                if let (Some(sm_kern), Some(sm_in), Some(sm_out)) =
                    (&kernels.softmax, &io.softmax_in, &io.softmax_out)
                {
                    // ANE softmax path: logits → ANE → probs → CPU NLL
                    sm_in.write_f32_as_fp16(&logits, cv, s);
                    if let Ok(()) = sm_kern.evaluate(&[sm_in.as_ptr()], &[sm_out.as_ptr()]) {
                        let mut probs = vec![0.0f32; cv * s];
                        sm_out.read_fp16_as_f32(&mut probs, 0, cv, s);
                        accelerate::nll_loss_from_probs(
                            &mut dlogits,
                            &probs,
                            &compact_targets,
                            cv,
                            s,
                        )
                    } else {
                        // ANE eval failed, fall back to CPU
                        accelerate::cross_entropy_loss(
                            &mut dlogits,
                            &logits,
                            &compact_targets,
                            cv,
                            s,
                        )
                    }
                } else {
                    accelerate::cross_entropy_loss(&mut dlogits, &logits, &compact_targets, cv, s)
                }
            } else {
                accelerate::cross_entropy_loss(&mut dlogits, &logits, &compact_targets, cv, s)
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

            // Async compact embed gradient: dE = dlogits @ x_final^T: [cv, s] @ [s, d] → [cv, d]
            {
                let dlogits_clone = dlogits;
                let x_final_clone = x_final;
                let grad_ptr = self.compact_embed_grad.as_mut_ptr() as usize;
                let grad_len = self.compact_embed_grad.len();
                let cv_val = cv;
                let d_val = d;
                let s_val = s;
                self.dispatch_dw(Box::new(move || {
                    let grad =
                        unsafe { std::slice::from_raw_parts_mut(grad_ptr as *mut f32, grad_len) };
                    accelerate::gemm(
                        &dlogits_clone,
                        &x_final_clone,
                        grad,
                        cv_val,
                        d_val,
                        s_val,
                        1.0,
                        1.0,
                        false,
                        true,
                    );
                }));
            }

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
            let loss = accelerate::cross_entropy_loss(&mut dlogits, &logits, target_tokens, v, s);

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

            // Async embed gradient accumulation (full vocab)
            {
                let dlogits_clone = dlogits;
                let x_final_clone = x_final;
                let embed_grad_ptr = self.embed_grad.as_mut_ptr() as usize;
                let embed_len = self.embed_grad.len();
                let v_val = v;
                let d_val = d;
                let s_val = s;
                self.dispatch_dw(Box::new(move || {
                    let embed_grad = unsafe {
                        std::slice::from_raw_parts_mut(embed_grad_ptr as *mut f32, embed_len)
                    };
                    accelerate::gemm(
                        &dlogits_clone,
                        &x_final_clone,
                        embed_grad,
                        v_val,
                        d_val,
                        s_val,
                        1.0,
                        1.0,
                        false,
                        true,
                    );
                }));
            }

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

        // Per-layer backward (reverse order)
        for l in (0..self.config.n_layers).rev() {
            self.backward_layer(l, &mut dx)?;
        }

        // Embedding backward
        let tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
        accelerate::embed_backward(&mut self.embed_grad, &dx, &tokens_u32, d, s);

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
    pub fn train_batch(&mut self, data: &[(Vec<u16>, Vec<u16>)], max_steps: usize) -> Result<f32> {
        let use_compact = self.vocab_map.is_some();

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
            let loss = self.train_step(input, target)?;
            total_loss += loss;
        }

        // Wait for all async dW tasks
        self.wait_dw();

        // Scale gradients by 1/accum_steps
        let scale = 1.0 / steps as f32;
        for lg in &mut self.layer_grads {
            accelerate::scale_inplace(&mut lg.wq, scale);
            accelerate::scale_inplace(&mut lg.wk, scale);
            accelerate::scale_inplace(&mut lg.wv, scale);
            accelerate::scale_inplace(&mut lg.wo, scale);
            accelerate::scale_inplace(&mut lg.w1, scale);
            accelerate::scale_inplace(&mut lg.w2, scale);
            accelerate::scale_inplace(&mut lg.w3, scale);
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
            grad_norm_sq += accelerate::sum_of_squares(&lg.wq);
            grad_norm_sq += accelerate::sum_of_squares(&lg.wk);
            grad_norm_sq += accelerate::sum_of_squares(&lg.wv);
            grad_norm_sq += accelerate::sum_of_squares(&lg.wo);
            grad_norm_sq += accelerate::sum_of_squares(&lg.w1);
            grad_norm_sq += accelerate::sum_of_squares(&lg.w2);
            grad_norm_sq += accelerate::sum_of_squares(&lg.w3);
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
                accelerate::scale_inplace(&mut lg.wq, clip_scale);
                accelerate::scale_inplace(&mut lg.wk, clip_scale);
                accelerate::scale_inplace(&mut lg.wv, clip_scale);
                accelerate::scale_inplace(&mut lg.wo, clip_scale);
                accelerate::scale_inplace(&mut lg.w1, clip_scale);
                accelerate::scale_inplace(&mut lg.w2, clip_scale);
                accelerate::scale_inplace(&mut lg.w3, clip_scale);
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

        // Adam update
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

            macro_rules! adam {
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

            adam!(wq, wq, wq);
            adam!(wk, wk, wk);
            adam!(wv, wv, wv);
            adam!(wo, wo, wo);
            adam!(w1, w1, w1);
            adam!(w2, w2, w2);
            adam!(w3, w3, w3);
            adam!(rms_att, rms_att, rms_att);
            adam!(rms_ffn, rms_ffn, rms_ffn);
        }

        if use_compact {
            // Adam update on compact embedding, then scatter back to full
            accelerate::adam_update(
                &mut self.compact_embed,
                &self.compact_embed_grad,
                &mut self.compact_embed_adam.m,
                &mut self.compact_embed_adam.v,
                t,
                lr,
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
                lr,
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

        // Refresh transposed weights for backward kernels
        self.refresh_transposed_weights();

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
