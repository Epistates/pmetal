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
}

impl Default for DynamicAneTrainerConfig {
    fn default() -> Self {
        Self {
            dim: 768,
            hidden_dim: 2048,
            n_heads: 12,
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
        }
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
    xnorm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    x2: Vec<f32>,
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

/// The 9 compiled ANE kernels (shared across all layers).
struct DynamicKernels {
    sdpa_fwd: AneModel,
    ffn_w13: AneModel,
    ffn_w2: AneModel,
    ffn_bwd_w2t: AneModel,
    ffn_bwd_w13t: AneModel,
    wo_bwd: AneModel,
    sdpa_bwd1: AneModel,
    sdpa_bwd2: AneModel,
    qkv_bwd: AneModel,
}

/// IOSurface pool for the dynamic pipeline.
///
/// Each kernel has dedicated input/output surfaces sized for its dimensions.
struct DynIoPool {
    // Forward surfaces (fp32)
    sdpa_fwd_in: IoSurface,
    sdpa_fwd_out: IoSurface,
    ffn_w13_in: IoSurface,
    ffn_w13_out: IoSurface,
    ffn_w2_in: IoSurface,
    ffn_w2_out: IoSurface,
    // Backward surfaces (fp32 for dynamic, fp16 for bwd1/bwd2)
    ffn_bwd_w2t_in: IoSurface,
    ffn_bwd_w2t_out: IoSurface,
    ffn_bwd_w13t_in: IoSurface,
    ffn_bwd_w13t_out: IoSurface,
    wo_bwd_in: IoSurface,
    wo_bwd_out: IoSurface,
    sdpa_bwd1_in: IoSurface,
    sdpa_bwd1_out: IoSurface,
    sdpa_bwd2_in: IoSurface,
    sdpa_bwd2_out: IoSurface,
    qkv_bwd_in: IoSurface,
    qkv_bwd_out: IoSurface,
}

/// Dynamic weight ANE trainer. Compiles 9 kernels once, then trains forever.
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
}

impl DynamicAneTrainer {
    /// Create a new dynamic ANE trainer.
    pub fn new(config: DynamicAneTrainerConfig) -> Self {
        let d = config.dim;
        let h = config.hidden_dim;
        let s = config.seq_len;
        let nl = config.n_layers;
        let v = config.vocab_size;
        let hd = d / config.n_heads;

        let kernel_config = TransformerKernelConfig {
            dim: d,
            hidden_dim: h,
            n_heads: config.n_heads,
            n_kv_heads: config.n_heads,
            head_dim: hd,
            seq_len: s,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        };

        let mut layer_weights = Vec::with_capacity(nl);
        let mut layer_acts = Vec::with_capacity(nl);
        let mut layer_grads = Vec::with_capacity(nl);
        let mut layer_adam = Vec::with_capacity(nl);

        for _ in 0..nl {
            layer_weights.push(LayerWeights {
                wq: vec![0.0; d * d],
                wk: vec![0.0; d * d],
                wv: vec![0.0; d * d],
                wo: vec![0.0; d * d],
                w1: vec![0.0; h * d],
                w2: vec![0.0; d * h],
                w3: vec![0.0; h * d],
                rms_att: vec![0.0; d],
                rms_ffn: vec![0.0; d],
                wq_t: vec![0.0; d * d],
                wk_t: vec![0.0; d * d],
                wv_t: vec![0.0; d * d],
                wo_t: vec![0.0; d * d],
                w1_t: vec![0.0; h * d],
                w2_t: vec![0.0; d * h],
                w3_t: vec![0.0; h * d],
            });

            layer_acts.push(LayerActivations {
                layer_in: vec![0.0; d * s],
                xnorm: vec![0.0; d * s],
                q: vec![0.0; d * s],
                k: vec![0.0; d * s],
                v: vec![0.0; d * s],
                attn_out: vec![0.0; d * s],
                o_out: vec![0.0; d * s],
                x2: vec![0.0; d * s],
                x2norm: vec![0.0; d * s],
                h1: vec![0.0; h * s],
                h3: vec![0.0; h * s],
                silu_out: vec![0.0; h * s],
                ffn_out: vec![0.0; d * s],
            });

            layer_grads.push(LayerGradients {
                wq: vec![0.0; d * d],
                wk: vec![0.0; d * d],
                wv: vec![0.0; d * d],
                wo: vec![0.0; d * d],
                w1: vec![0.0; h * d],
                w2: vec![0.0; d * h],
                w3: vec![0.0; h * d],
                rms_att: vec![0.0; d],
                rms_ffn: vec![0.0; d],
            });

            layer_adam.push(LayerAdamState {
                wq: AdamParam::new(d * d),
                wk: AdamParam::new(d * d),
                wv: AdamParam::new(d * d),
                wo: AdamParam::new(d * d),
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
                loop {
                    match dw_receiver.recv() {
                        Ok(task) => task(),
                        Err(_) => break,
                    }
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

    /// Number of ANE compilations performed (should be 9 after compile_kernels).
    pub fn compile_count(&self) -> usize {
        self.compile_count
    }

    /// Compile all 9 dynamic kernels (called once at startup).
    ///
    /// After this call, no further compilations are needed.
    pub fn compile_kernels(&mut self) -> Result<()> {
        let rt = AneRuntime::global()?;
        let dkc = DynamicKernelConfig::new(self.kernel_config.clone());
        let d = self.kernel_config.dim;
        let h = self.kernel_config.hidden_dim;
        let s = self.kernel_config.seq_len;
        let score_ch = self.kernel_config.score_ch();

        info!("Compiling 9 dynamic ANE kernels (one-time)...");

        // Generate and compile each kernel
        let compile = |out: &DynamicKernelOutput, rt: &AneRuntime, name: &str| -> Result<AneModel> {
            let wd = if out.static_weights.entries.is_empty() {
                None
            } else {
                Some(&out.static_weights)
            };
            debug!("Compiling dynamic kernel: {name}");
            rt.compile(out.mil_text.as_bytes(), wd)
        };

        let k1 = dynamic_kernel::gen_dynamic_sdpa_fwd(&dkc);
        let sdpa_fwd = compile(&k1, rt, "sdpa_fwd")?;
        self.compile_count += 1;

        let k2 = dynamic_kernel::gen_dynamic_ffn_w13(&dkc);
        let ffn_w13 = compile(&k2, rt, "ffn_w13")?;
        self.compile_count += 1;

        let k3 = dynamic_kernel::gen_dynamic_ffn_w2(&dkc);
        let ffn_w2 = compile(&k3, rt, "ffn_w2")?;
        self.compile_count += 1;

        let k4 = dynamic_kernel::gen_dynamic_ffn_bwd_w2t(&dkc);
        let ffn_bwd_w2t = compile(&k4, rt, "ffn_bwd_w2t")?;
        self.compile_count += 1;

        let k5 = dynamic_kernel::gen_dynamic_ffn_bwd_w13t(&dkc);
        let ffn_bwd_w13t = compile(&k5, rt, "ffn_bwd_w13t")?;
        self.compile_count += 1;

        let k6 = dynamic_kernel::gen_dynamic_wo_bwd(&dkc);
        let wo_bwd = compile(&k6, rt, "wo_bwd")?;
        self.compile_count += 1;

        let k7 = dynamic_kernel::gen_dynamic_sdpa_bwd1(&dkc);
        let sdpa_bwd1 = compile(&k7, rt, "sdpa_bwd1")?;
        self.compile_count += 1;

        let k8 = dynamic_kernel::gen_dynamic_sdpa_bwd2(&dkc);
        let sdpa_bwd2 = compile(&k8, rt, "sdpa_bwd2")?;
        self.compile_count += 1;

        let k9 = dynamic_kernel::gen_dynamic_qkv_bwd(&dkc);
        let qkv_bwd = compile(&k9, rt, "qkv_bwd")?;
        self.compile_count += 1;

        info!("All 9 dynamic kernels compiled. Total compiles: {}", self.compile_count);

        // Allocate IOSurface pool
        let io_pool = DynIoPool {
            // Forward (fp32 surfaces)
            sdpa_fwd_in: IoSurface::for_tensor_f32(d, s + 4 * d)?,
            sdpa_fwd_out: IoSurface::for_tensor_f32(6 * d, s)?,
            ffn_w13_in: IoSurface::for_tensor_f32(d, s + 2 * h)?,
            ffn_w13_out: IoSurface::for_tensor_f32(3 * h, s)?,
            ffn_w2_in: IoSurface::for_tensor_f32(h, s + d)?,
            ffn_w2_out: IoSurface::for_tensor_f32(d, s)?,
            // Backward (fp32 for dynamic weight kernels)
            ffn_bwd_w2t_in: IoSurface::for_tensor_f32(d, s + h)?,
            ffn_bwd_w2t_out: IoSurface::for_tensor_f32(h, s)?,
            ffn_bwd_w13t_in: IoSurface::for_tensor_f32(h, 2 * s + 2 * d)?,
            ffn_bwd_w13t_out: IoSurface::for_tensor_f32(d, s)?,
            wo_bwd_in: IoSurface::for_tensor_f32(d, s + d)?,
            wo_bwd_out: IoSurface::for_tensor_f32(d, s)?,
            // Backward (fp16 for bwd1/bwd2 — no dynamic weights)
            sdpa_bwd1_in: IoSurface::for_tensor(4 * d, s)?,
            sdpa_bwd1_out: IoSurface::for_tensor(d + 2 * score_ch, s)?,
            sdpa_bwd2_in: IoSurface::for_tensor(2 * score_ch + 2 * d, s)?,
            sdpa_bwd2_out: IoSurface::for_tensor(2 * d, s)?,
            // QKV backward (fp32)
            qkv_bwd_in: IoSurface::for_tensor_f32(d, 3 * s + 3 * d)?,
            qkv_bwd_out: IoSurface::for_tensor_f32(d, s)?,
        };

        self.kernels = Some(DynamicKernels {
            sdpa_fwd,
            ffn_w13,
            ffn_w2,
            ffn_bwd_w2t,
            ffn_bwd_w13t,
            wo_bwd,
            sdpa_bwd1,
            sdpa_bwd2,
            qkv_bwd,
        });
        self.io_pool = Some(io_pool);

        Ok(())
    }

    /// Refresh transposed weight buffers after Adam update.
    fn refresh_transposed_weights(&mut self) {
        let d = self.config.dim;
        let h = self.config.hidden_dim;

        for lw in &mut self.layer_weights {
            transpose_weight(&lw.wq, &mut lw.wq_t, d, d);
            transpose_weight(&lw.wk, &mut lw.wk_t, d, d);
            transpose_weight(&lw.wv, &mut lw.wv_t, d, d);
            transpose_weight(&lw.wo, &mut lw.wo_t, d, d);
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

    /// Forward pass for a single layer using dynamic ANE kernels.
    fn forward_layer(&mut self, l: usize, x: &mut [f32]) -> Result<()> {
        let d = self.config.dim;
        let h = self.config.hidden_dim;
        let s = self.config.seq_len;
        let lw = &self.layer_weights[l];
        let acts = &mut self.layer_acts[l];

        let kernels = self.kernels.as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?;
        let io = self.io_pool.as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?;

        // 1. Attention Block
        // RMSNorm on CPU
        accelerate::rmsnorm(&mut acts.xnorm, x, &lw.rms_att, d, s);

        // Pack xnorm + Wq + Wk + Wv + Wo into sdpa_fwd input
        io.sdpa_fwd_in.write_packed_f32(
            &acts.xnorm,
            &[(&lw.wq, d), (&lw.wk, d), (&lw.wv, d), (&lw.wo, d)],
            d,
            s,
        );

        // ANE evaluate sdpa_fwd
        kernels.sdpa_fwd.evaluate(
            &[io.sdpa_fwd_in.as_ptr()],
            &[io.sdpa_fwd_out.as_ptr()],
        )?;

        // Read taps: [o_out, Q, K, V, attn_out, xnorm] each [D, S]
        io.sdpa_fwd_out.read_f32(&mut acts.o_out, 0, d, s);
        io.sdpa_fwd_out.read_f32(&mut acts.q, d, d, s);
        io.sdpa_fwd_out.read_f32(&mut acts.k, 2 * d, d, s);
        io.sdpa_fwd_out.read_f32(&mut acts.v, 3 * d, d, s);
        io.sdpa_fwd_out.read_f32(&mut acts.attn_out, 4 * d, d, s);

        // Residual: x2 = x + o_out
        accelerate::vadd(x, &acts.o_out, &mut acts.x2);

        // 2. FFN Block
        // RMSNorm on CPU
        accelerate::rmsnorm(&mut acts.x2norm, &acts.x2, &lw.rms_ffn, d, s);

        // Pack x2norm + W1 + W3 into ffn_w13 input
        io.ffn_w13_in.write_packed_f32(
            &acts.x2norm,
            &[(&lw.w1, h), (&lw.w3, h)],
            d,
            s,
        );

        // ANE evaluate ffn_w13
        kernels.ffn_w13.evaluate(
            &[io.ffn_w13_in.as_ptr()],
            &[io.ffn_w13_out.as_ptr()],
        )?;

        // Read taps: [h1, h3, gate] each [HIDDEN, S]
        io.ffn_w13_out.read_f32(&mut acts.h1, 0, h, s);
        io.ffn_w13_out.read_f32(&mut acts.h3, h, h, s);
        io.ffn_w13_out.read_f32(&mut acts.silu_out, 2 * h, h, s);

        // Pack gate + W2 into ffn_w2 input
        io.ffn_w2_in.write_packed_f32(
            &acts.silu_out,
            &[(&lw.w2, d)],
            h,
            s,
        );

        // ANE evaluate ffn_w2
        kernels.ffn_w2.evaluate(
            &[io.ffn_w2_in.as_ptr()],
            &[io.ffn_w2_out.as_ptr()],
        )?;

        // Read ffn_out
        io.ffn_w2_out.read_f32(&mut acts.ffn_out, 0, d, s);

        // Residual: x = x2 + ffn_out
        accelerate::vadd(&acts.x2, &acts.ffn_out, x);

        Ok(())
    }

    /// Backward pass for a single layer using dynamic ANE kernels.
    fn backward_layer(&mut self, l: usize, dx: &mut [f32]) -> Result<()> {
        let d = self.config.dim;
        let h = self.config.hidden_dim;
        let s = self.config.seq_len;
        let score_ch = self.kernel_config.score_ch();

        let kernels = self.kernels.as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?;
        let io = self.io_pool.as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?;

        // ====== FFN Backward ======

        // 1. dffn @ W2^T → dsilu_raw
        io.ffn_bwd_w2t_in.write_packed_f32(
            dx,
            &[(&self.layer_weights[l].w2_t, h)],
            d,
            s,
        );
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
                let w2_grad = unsafe {
                    std::slice::from_raw_parts_mut(w2_grad_ptr as *mut f32, w2_grad_len)
                };
                accelerate::gemm(
                    &dy_clone, &silu_clone, w2_grad,
                    d_val, h_val, s_val, 1.0, 1.0, false, true,
                );
            }));
        }

        // 3. SiLU derivative on CPU
        // dsilu_raw already has dffn @ W2^T
        // We need: dh1 = dsilu_raw * h3 * silu_deriv(h1)
        //          dh3 = dsilu_raw * sigmoid(h1)
        let acts = &self.layer_acts[l];
        let mut dh1 = vec![0.0f32; h * s];
        let mut dh3 = vec![0.0f32; h * s];
        for i in 0..(h * s) {
            let h1_val = acts.h1[i];
            let sig = 1.0 / (1.0 + (-h1_val).exp());
            let silu_d = sig * (1.0 + h1_val * (1.0 - sig));
            dh1[i] = dsilu_raw[i] * acts.h3[i] * silu_d;
            dh3[i] = dsilu_raw[i] * sig * acts.h1[i]; // dsilu * h1 * sig = dsilu * silu(h1) ... no
            // Actually: gate = silu(h1) * h3, dsilu = dgate * h3, dh3 = dgate * silu(h1)
            // dgate = dsilu_raw (from W2^T backward)
            // dh1 = dgate * h3 * silu_deriv = dsilu_raw * h3 * (sig + h1*sig*(1-sig))
            // dh3 = dgate * silu(h1) = dsilu_raw * h1 * sig
            dh3[i] = dsilu_raw[i] * h1_val * sig;
        }

        // Async dW1 = dh1 @ x2norm^T, dW3 = dh3 @ x2norm^T
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
                let w1_grad = unsafe {
                    std::slice::from_raw_parts_mut(w1_grad_ptr as *mut f32, w1_grad_len)
                };
                accelerate::gemm(
                    &dh1_clone, &x2norm_clone, w1_grad,
                    h_val, d_val, s_val, 1.0, 1.0, false, true,
                );
                let w3_grad = unsafe {
                    std::slice::from_raw_parts_mut(w3_grad_ptr as *mut f32, w3_grad_len)
                };
                accelerate::gemm(
                    &dh3_clone, &x2norm_clone, w3_grad,
                    h_val, d_val, s_val, 1.0, 1.0, false, true,
                );
            }));
        }

        // 4. dh1@W1^T + dh3@W3^T → dx_ffn
        io.ffn_bwd_w13t_in.write_packed_f32_multi(
            &[(&dh1, s), (&dh3, s)],
            &[(&self.layer_weights[l].w1_t, d), (&self.layer_weights[l].w3_t, d)],
            h,
        );
        kernels.ffn_bwd_w13t.evaluate(
            &[io.ffn_bwd_w13t_in.as_ptr()],
            &[io.ffn_bwd_w13t_out.as_ptr()],
        )?;
        let mut dx_ffn = vec![0.0f32; d * s];
        io.ffn_bwd_w13t_out.read_f32(&mut dx_ffn, 0, d, s);

        // 5. FFN RMSNorm backward
        let mut dx_ffn_norm = vec![0.0f32; d * s];
        accelerate::rmsnorm_backward(
            &mut dx_ffn_norm,
            &mut self.layer_grads[l].rms_ffn,
            &dx_ffn,
            &self.layer_acts[l].x2,
            &self.layer_weights[l].rms_ffn,
            d,
            s,
        );

        // ====== Attention Backward ======

        // 6. dy @ Wo^T → da (attention backward input)
        io.wo_bwd_in.write_packed_f32(
            &dx_ffn_norm,
            &[(&self.layer_weights[l].wo_t, d)],
            d,
            s,
        );
        kernels.wo_bwd.evaluate(
            &[io.wo_bwd_in.as_ptr()],
            &[io.wo_bwd_out.as_ptr()],
        )?;
        let mut da = vec![0.0f32; d * s];
        io.wo_bwd_out.read_f32(&mut da, 0, d, s);

        // Async dWo = dx_ffn_norm @ attn_out^T
        {
            let dx_clone = dx_ffn_norm.clone();
            let attn_clone = self.layer_acts[l].attn_out.to_vec();
            let wo_grad_ptr = self.layer_grads[l].wo.as_mut_ptr() as usize;
            let wo_grad_len = self.layer_grads[l].wo.len();
            let d_val = d;
            let s_val = s;
            self.dispatch_dw(Box::new(move || {
                let wo_grad = unsafe {
                    std::slice::from_raw_parts_mut(wo_grad_ptr as *mut f32, wo_grad_len)
                };
                accelerate::gemm(
                    &dx_clone, &attn_clone, wo_grad,
                    d_val, d_val, s_val, 1.0, 1.0, false, true,
                );
            }));
        }

        // 7. SDPA backward part 1: Q, K, V, da → dV, probs, dp (fp16 path)
        let acts = &self.layer_acts[l];
        io.sdpa_bwd1_in.write_f32_as_fp16_at(0, &acts.q, d, s);
        io.sdpa_bwd1_in.write_f32_as_fp16_at(d, &acts.k, d, s);
        io.sdpa_bwd1_in.write_f32_as_fp16_at(2 * d, &acts.v, d, s);
        io.sdpa_bwd1_in.write_f32_as_fp16_at(3 * d, &da, d, s);

        kernels.sdpa_bwd1.evaluate(
            &[io.sdpa_bwd1_in.as_ptr()],
            &[io.sdpa_bwd1_out.as_ptr()],
        )?;

        // Read dV, and copy probs+dp for bwd2 input
        let mut dv = vec![0.0f32; d * s];
        io.sdpa_bwd1_out.read_fp16_as_f32(&mut dv, 0, d, s);

        // 8. SDPA backward part 2: probs, dp, Q, K → dQ, dK (fp16 surface-to-surface copy)
        // Copy probs and dp from bwd1 output to bwd2 input
        io.sdpa_bwd2_in.copy_from(0, &io.sdpa_bwd1_out, d, 2 * score_ch, s);
        // Copy Q and K
        io.sdpa_bwd2_in.write_f32_as_fp16_at(2 * score_ch, &acts.q, d, s);
        io.sdpa_bwd2_in.write_f32_as_fp16_at(2 * score_ch + d, &acts.k, d, s);

        kernels.sdpa_bwd2.evaluate(
            &[io.sdpa_bwd2_in.as_ptr()],
            &[io.sdpa_bwd2_out.as_ptr()],
        )?;

        let mut dq = vec![0.0f32; d * s];
        let mut dk = vec![0.0f32; d * s];
        io.sdpa_bwd2_out.read_fp16_as_f32(&mut dq, 0, d, s);
        io.sdpa_bwd2_out.read_fp16_as_f32(&mut dk, d, d, s);

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
            let s_val = s;
            self.dispatch_dw(Box::new(move || {
                let wq_grad = unsafe {
                    std::slice::from_raw_parts_mut(wq_grad_ptr as *mut f32, wq_grad_len)
                };
                accelerate::gemm(
                    &dq_clone, &xnorm_clone, wq_grad,
                    d_val, d_val, s_val, 1.0, 1.0, false, true,
                );
                let wk_grad = unsafe {
                    std::slice::from_raw_parts_mut(wk_grad_ptr as *mut f32, wk_grad_len)
                };
                accelerate::gemm(
                    &dk_clone, &xnorm_clone, wk_grad,
                    d_val, d_val, s_val, 1.0, 1.0, false, true,
                );
                let wv_grad = unsafe {
                    std::slice::from_raw_parts_mut(wv_grad_ptr as *mut f32, wv_grad_len)
                };
                accelerate::gemm(
                    &dv_clone, &xnorm_clone, wv_grad,
                    d_val, d_val, s_val, 1.0, 1.0, false, true,
                );
            }));
        }

        // 9. QKV backward: dq@Wq^T + dk@Wk^T + dv@Wv^T → dx_attn
        io.qkv_bwd_in.write_packed_f32_multi(
            &[(&dq, s), (&dk, s), (&dv, s)],
            &[
                (&self.layer_weights[l].wq_t, d),
                (&self.layer_weights[l].wk_t, d),
                (&self.layer_weights[l].wv_t, d),
            ],
            d,
        );
        kernels.qkv_bwd.evaluate(
            &[io.qkv_bwd_in.as_ptr()],
            &[io.qkv_bwd_out.as_ptr()],
        )?;
        let mut dx_attn = vec![0.0f32; d * s];
        io.qkv_bwd_out.read_f32(&mut dx_attn, 0, d, s);

        // 10. Attention RMSNorm backward
        let mut dx_attn_norm = vec![0.0f32; d * s];
        accelerate::rmsnorm_backward(
            &mut dx_attn_norm,
            &mut self.layer_grads[l].rms_att,
            &dx_attn,
            &self.layer_acts[l].layer_in,
            &self.layer_weights[l].rms_att,
            d,
            s,
        );

        // Output: dx = dx_ffn_norm + dx_attn_norm (residual)
        // Actually both residual paths add: dx_attn flows through attention residual,
        // dx_ffn flows through FFN residual. Combined:
        accelerate::vadd(&dx_ffn_norm, &dx_attn_norm, dx);

        Ok(())
    }

    /// Run a single training step (forward + backward + grad accumulation).
    pub fn train_step(&mut self, input_tokens: &[u16], target_tokens: &[u16]) -> Result<f32> {
        let d = self.config.dim;
        let s = self.config.seq_len;
        let v = self.config.vocab_size;

        assert_eq!(input_tokens.len(), s);
        assert_eq!(target_tokens.len(), s);

        // === Forward pass ===
        let mut x = vec![0.0f32; d * s];
        accelerate::embed_lookup(&mut x, &self.embed_weights, input_tokens, d, s);

        for l in 0..self.config.n_layers {
            self.layer_acts[l].layer_in.copy_from_slice(&x);
            self.forward_layer(l, &mut x)?;
        }

        // Final RMSNorm
        let mut x_final = vec![0.0f32; d * s];
        accelerate::rmsnorm(&mut x_final, &x, &self.rms_final, d, s);

        // Classifier: logits = embed^T @ x_final
        let mut logits = vec![0.0f32; v * s];
        accelerate::gemm(
            &self.embed_weights, &x_final, &mut logits,
            v, s, d, 1.0, 0.0, false, false,
        );

        // Cross-entropy loss
        let mut dlogits = vec![0.0f32; v * s];
        let loss = accelerate::cross_entropy_loss(&mut dlogits, &logits, target_tokens, v, s);

        // === Backward pass ===

        // dEmbed (classifier gradient): dx = embed^T @ dlogits
        let mut dx = vec![0.0f32; d * s];
        accelerate::gemm(
            &self.embed_weights, &dlogits, &mut dx,
            d, s, v, 1.0, 0.0, true, false,
        );

        // Async embed gradient accumulation
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
                    &dlogits_clone, &x_final_clone, embed_grad,
                    v_val, d_val, s_val, 1.0, 1.0, false, true,
                );
            }));
        }

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
        );

        // Per-layer backward (reverse order)
        for l in (0..self.config.n_layers).rev() {
            self.backward_layer(l, &mut dx)?;
        }

        // Embedding backward
        accelerate::embed_backward(&mut self.embed_grad, &dx, input_tokens, d, s);

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
    pub fn train_batch(
        &mut self,
        data: &[(Vec<u16>, Vec<u16>)],
        max_steps: usize,
    ) -> Result<f32> {
        // Zero gradients
        for lg in &mut self.layer_grads {
            lg.zero();
        }
        self.embed_grad.fill(0.0);
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
        accelerate::scale_inplace(&mut self.embed_grad, scale);
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
        grad_norm_sq += accelerate::sum_of_squares(&self.embed_grad);
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
            accelerate::scale_inplace(&mut self.embed_grad, clip_scale);
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
                        &mut lw.$w, &lg.$g, &mut la.$a.m, &mut la.$a.v,
                        t, lr, b1, b2, eps,
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

        accelerate::adam_update(
            &mut self.embed_weights, &self.embed_grad,
            &mut self.embed_adam.m, &mut self.embed_adam.v,
            t, lr, b1, b2, eps,
        );
        accelerate::adam_update(
            &mut self.rms_final, &self.rms_final_grad,
            &mut self.rms_final_adam.m, &mut self.rms_final_adam.v,
            t, lr, b1, b2, eps,
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

        let mut offset = 0;

        self.embed_weights[..v * d].copy_from_slice(&weights[offset..offset + v * d]);
        offset += v * d;

        for l in 0..nl {
            let lw = &mut self.layer_weights[l];

            lw.rms_att.copy_from_slice(&weights[offset..offset + d]);
            offset += d;
            lw.wq.copy_from_slice(&weights[offset..offset + d * d]);
            offset += d * d;
            lw.wk.copy_from_slice(&weights[offset..offset + d * d]);
            offset += d * d;
            lw.wv.copy_from_slice(&weights[offset..offset + d * d]);
            offset += d * d;
            lw.wo.copy_from_slice(&weights[offset..offset + d * d]);
            offset += d * d;
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
                    if bytes.len() % 4 != 0 { return None; }
                    let n = bytes.len() / 4;
                    let mut out = vec![0.0f32; n];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(), out.as_mut_ptr() as *mut u8, n * 4,
                        );
                    }
                    Some(out)
                }
                Dtype::F16 => {
                    let bytes = tensor.data();
                    if bytes.len() % 2 != 0 { return None; }
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
                    if bytes.len() % 2 != 0 { return None; }
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
                let index_text = std::fs::read_to_string(&index_path)
                    .map_err(|e| MetalError::InvalidConfig(format!("Failed to read index.json: {e}")))?;
                let index: serde_json::Value = serde_json::from_str(&index_text)
                    .map_err(|e| MetalError::InvalidConfig(format!("Failed to parse index.json: {e}")))?;
                let weight_map = index["weight_map"].as_object()
                    .ok_or_else(|| MetalError::InvalidConfig("Missing weight_map".into()))?;
                let mut unique_files: Vec<String> = weight_map.values()
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
                    return Err(MetalError::InvalidConfig("No safetensors files found".into()));
                }
            }
        };

        for file_path in &files {
            let file = std::fs::File::open(file_path)
                .map_err(|e| MetalError::InvalidConfig(format!("Failed to open {:?}: {e}", file_path)))?;
            #[allow(unsafe_code)]
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| MetalError::InvalidConfig(format!("Failed to mmap {:?}: {e}", file_path)))?;
            let tensors = SafeTensors::deserialize(&mmap)
                .map_err(|e| MetalError::InvalidConfig(format!("Failed to parse safetensors: {e}")))?;

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
                    if parts.len() < 2 { continue; }
                    let layer_idx: usize = match parts[0].parse() {
                        Ok(i) => i,
                        Err(_) => continue,
                    };
                    if layer_idx >= self.config.n_layers { continue; }

                    let lw = &mut self.layer_weights[layer_idx];
                    match parts[1] {
                        "self_attn.q_proj.weight" => copy_w(&data, &mut lw.wq, d * d),
                        "self_attn.k_proj.weight" => copy_w(&data, &mut lw.wk, d * d),
                        "self_attn.v_proj.weight" => copy_w(&data, &mut lw.wv, d * d),
                        "self_attn.o_proj.weight" => copy_w(&data, &mut lw.wo, d * d),
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
    fn test_dynamic_trainer_no_recompile_needed() {
        let config = DynamicAneTrainerConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_layers: 2,
            vocab_size: 100,
            seq_len: 16,
            ..Default::default()
        };
        let trainer = DynamicAneTrainer::new(config);
        // compile_count stays at 0 until compile_kernels() is called
        assert_eq!(trainer.compile_count(), 0);
        // After compile_kernels(), it should be exactly 9
        // (can't test without actual ANE hardware)
    }
}
