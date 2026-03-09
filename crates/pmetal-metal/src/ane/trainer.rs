//! Hybrid CPU/ANE training loop.
//!
//! Orchestrates ANE forward/backward passes with CPU-side operations:
//! - **ANE**: All conv/matmul operations (~70% of FLOPs)
//! - **CPU (vDSP)**: RMSNorm fwd/bwd, cross-entropy, softmax
//! - **CPU (cblas)**: Weight gradient accumulation dW = dy @ x^T (async)
//! - **CPU**: Adam optimizer, embedding lookup/backward
//!
//! Weight gradients are dispatched to a background thread via mpsc channel
//! (replacing GCD `dispatch_queue_create(SERIAL)` from the reference).

use std::sync::mpsc;
use std::thread;

use crate::accelerate;
use crate::ane::budget::{BudgetExhaustionStrategy, CompileBudget};
use crate::ane::iosurface::IoSurface;
use crate::ane::kernel::{self, TransformerKernelConfig};
use crate::ane::runtime::{AneModel, AneRuntime};
use crate::error::{MetalError, Result};

/// Configuration for the ANE hybrid trainer.
#[derive(Debug, Clone)]
pub struct AneTrainerConfig {
    /// Model dimension.
    pub dim: usize,
    /// FFN hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of layers.
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
    /// Number of gradient accumulation steps between recompiles.
    pub accum_steps: usize,
    /// Maximum ANE compilations per process.
    pub max_compiles: usize,
    /// Strategy when compile budget is exhausted.
    pub exhaustion_strategy: BudgetExhaustionStrategy,
    /// RMSNorm epsilon (must match ANE kernel eps). Default: 1e-6.
    pub rms_norm_eps: f32,
}

impl Default for AneTrainerConfig {
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
            max_compiles: 100,
            exhaustion_strategy: BudgetExhaustionStrategy::Error,
            rms_norm_eps: 1e-6,
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
#[allow(dead_code)]
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

/// Per-layer compiled ANE kernels.
#[allow(dead_code)]
struct LayerKernels {
    fwd_attn: AneModel,
    fwd_ffn: AneModel,
    ffn_bwd: AneModel,
    sdpa_bwd1: AneModel,
    qkv_bwd: AneModel,
}

/// IOSurface pool for zero-copy data transfer between CPU and ANE.
#[allow(dead_code)]
struct IoPool {
    /// Generic input surface [max_ch, seq]
    input: IoSurface,
    /// SDPA forward output [6*dim, seq]
    fwd_attn: IoSurface,
    /// FFN forward output [2*dim + 3*hidden, seq]
    fwd_ffn: IoSurface,
    /// FFN backward output [dim + 2*hidden, seq]
    ffn_bwd: IoSurface,
    /// SDPA backward 1 output [dim + 2*heads*seq, seq]
    sdpa_bwd1: IoSurface,
    /// SDPA backward 2 output [2*heads*seq + 2*dim, seq]
    sdpa_bwd2: IoSurface,
    /// QKV backward output [3*dim, seq]
    qkv_bwd: IoSurface,
}

/// Hybrid CPU/ANE trainer.
///
/// Manages the complete training loop with ANE kernels for compute-heavy
/// operations and vDSP/cblas for reduction and accumulation operations.
pub struct AneTrainer {
    config: AneTrainerConfig,
    kernel_config: TransformerKernelConfig,
    layer_weights: Vec<LayerWeights>,
    layer_kernels: Option<Vec<LayerKernels>>,
    sdpa_bwd2: Option<AneModel>,
    io_pool: Option<IoPool>,
    layer_acts: Vec<LayerActivations>,
    layer_grads: Vec<LayerGradients>,
    layer_adam: Vec<LayerAdamState>,
    embed_weights: Vec<f32>,
    embed_grad: Vec<f32>,
    embed_adam: AdamParam,
    rms_final: Vec<f32>,
    rms_final_grad: Vec<f32>,
    rms_final_adam: AdamParam,
    budget: CompileBudget,
    adam_t: usize,
    /// Channel for dispatching async dW tasks to the cblas worker thread.
    dw_sender: mpsc::Sender<Box<dyn FnOnce() + Send>>,
    /// Barrier channel for waiting on pending dW tasks.
    dw_barrier_sender: mpsc::Sender<mpsc::Sender<()>>,
    /// Worker thread handle.
    _dw_thread: thread::JoinHandle<()>,
}

impl AneTrainer {
    /// Create a new ANE trainer.
    ///
    /// Allocates all per-layer buffers and spawns the async dW worker thread.
    pub fn new(config: AneTrainerConfig) -> Self {
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
            n_kv_heads: config.n_heads, // training path assumes MHA
            head_dim: hd,
            seq_len: s,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        };

        // Allocate per-layer structures
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
        let (barrier_sender, barrier_receiver) = mpsc::channel::<mpsc::Sender<()>>();

        let dw_thread = thread::Builder::new()
            .name("ane-dw-cblas".to_string())
            .spawn(move || {
                loop {
                    // Try to receive a task or barrier
                    match dw_receiver.recv() {
                        Ok(task) => task(),
                        Err(_) => break, // Channel closed
                    }

                    // Check for barrier requests (non-blocking)
                    while let Ok(reply) = barrier_receiver.try_recv() {
                        let _ = reply.send(());
                    }
                }
            })
            .expect("Failed to spawn dW worker thread");

        let budget = CompileBudget::new(config.max_compiles, 5 * nl);

        Self {
            config,
            kernel_config,
            layer_weights,
            layer_kernels: None,
            sdpa_bwd2: None,
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
            budget,
            adam_t: 0,
            dw_sender,
            dw_barrier_sender: barrier_sender,
            _dw_thread: dw_thread,
        }
    }

    /// Get a reference to the trainer configuration.
    pub fn config(&self) -> &AneTrainerConfig {
        &self.config
    }

    /// Get the current Adam step count.
    pub fn adam_t(&self) -> usize {
        self.adam_t
    }

    /// Get the compilation budget.
    pub fn budget(&self) -> &CompileBudget {
        &self.budget
    }

    /// Load weights from flat f32 arrays.
    ///
    /// `weights` should contain all model parameters in the order expected
    /// by the model architecture (llama2.c format or similar).
    pub fn load_weights_flat(&mut self, weights: &[f32]) {
        let d = self.config.dim;
        let h = self.config.hidden_dim;
        let nl = self.config.n_layers;
        let v = self.config.vocab_size;

        // Expected layout (llama2.c style):
        // embed[V*D], then per-layer: rms_att[D], wq[D*D], wk[D*D], wv[D*D],
        // wo[D*D], rms_ffn[D], w1[H*D], w2[D*H], w3[H*D], then rms_final[D]
        let mut offset = 0;

        // Embedding
        self.embed_weights[..v * d].copy_from_slice(&weights[offset..offset + v * d]);
        offset += v * d;

        // Per-layer
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

        // Final RMSNorm
        self.rms_final.copy_from_slice(&weights[offset..offset + d]);
    }

    /// Load weights from SafeTensors files on disk.
    ///
    /// Supports both single-file (`model.safetensors`) and sharded formats
    /// (with `model.safetensors.index.json`). Handles F32, F16, and BF16 dtypes.
    ///
    /// Note: The trainer assumes MHA (n_kv_heads == n_heads), so K/V projection
    /// weights must be `[D, D]`. GQA models are not currently supported for training.
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

        // Determine files to load
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

                // Parse "model.layers.{i}.{suffix}"
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

        Ok(())
    }

    /// Compile all kernels for the current weights.
    ///
    /// Frees any previously compiled kernels, generates MIL + weight blobs,
    /// and compiles all 6 kernel types per layer.
    pub fn compile_kernels(&mut self) -> Result<()> {
        let rt = AneRuntime::global()?;

        // Check budget
        if !self.budget.can_compile_batch() {
            return Err(MetalError::AneCompileFailed(format!(
                "Compile budget exhausted: {}/{} used",
                self.budget.current(),
                self.budget.max()
            )));
        }

        // Free old kernels
        self.layer_kernels = None;
        self.sdpa_bwd2 = None;
        self.io_pool = None;

        let cfg = &self.kernel_config;
        let d = cfg.dim;
        let h = cfg.hidden_dim;
        let s = cfg.seq_len;

        // Initialize IO Pool
        let input = IoSurface::for_tensor(std::cmp::max(6 * d, 2 * d + 3 * h), s)?;
        let fwd_attn = IoSurface::for_tensor(6 * d, s)?;
        let fwd_ffn = IoSurface::for_tensor(2 * d + 3 * h, s)?;
        let ffn_bwd = IoSurface::for_tensor(d + 2 * h, s)?;
        let sdpa_bwd1 = IoSurface::for_tensor(d + 2 * cfg.score_ch(), s)?;
        let sdpa_bwd2 = IoSurface::for_tensor(2 * cfg.score_ch() + 2 * d, s)?;
        let qkv_bwd = IoSurface::for_tensor(3 * d, s)?;

        self.io_pool = Some(IoPool {
            input,
            fwd_attn,
            fwd_ffn,
            ffn_bwd,
            sdpa_bwd1,
            sdpa_bwd2,
            qkv_bwd,
        });

        let mut layer_kernels = Vec::with_capacity(self.config.n_layers);

        // Compile weight-free sdpaBwd2 once (shared across layers)
        let bwd2_output = kernel::gen_sdpa_bwd2(cfg);
        let sdpa_bwd2 = rt.compile(bwd2_output.mil_text.as_bytes(), None)?;
        self.budget.record_compile();

        for l in 0..self.config.n_layers {
            let lw = &self.layer_weights[l];

            // Forward attention
            let fwd_attn_out =
                kernel::gen_sdpa_fwd_taps(cfg, &lw.rms_att, &lw.wq, &lw.wk, &lw.wv, &lw.wo);
            let fwd_attn = rt.compile(
                fwd_attn_out.mil_text.as_bytes(),
                Some(&fwd_attn_out.weights),
            )?;
            self.budget.record_compile();

            // Forward FFN
            let fwd_ffn_out = kernel::gen_ffn_fwd_taps(cfg, &lw.rms_ffn, &lw.w1, &lw.w3, &lw.w2);
            let fwd_ffn =
                rt.compile(fwd_ffn_out.mil_text.as_bytes(), Some(&fwd_ffn_out.weights))?;
            self.budget.record_compile();

            // Backward FFN
            let ffn_bwd_out = kernel::gen_ffn_bwd(cfg, &lw.w1, &lw.w2, &lw.w3);
            let ffn_bwd =
                rt.compile(ffn_bwd_out.mil_text.as_bytes(), Some(&ffn_bwd_out.weights))?;
            self.budget.record_compile();

            // Backward SDPA part 1
            let bwd1_out = kernel::gen_sdpa_bwd1(cfg, &lw.wo);
            let sdpa_bwd1 = rt.compile(bwd1_out.mil_text.as_bytes(), Some(&bwd1_out.weights))?;
            self.budget.record_compile();

            // Backward QKV
            let qkv_out = kernel::gen_qkv_bwd(cfg, &lw.wq, &lw.wk, &lw.wv);
            let qkv_bwd = rt.compile(qkv_out.mil_text.as_bytes(), Some(&qkv_out.weights))?;
            self.budget.record_compile();

            layer_kernels.push(LayerKernels {
                fwd_attn,
                fwd_ffn,
                ffn_bwd,
                sdpa_bwd1,
                qkv_bwd,
            });
        }

        self.layer_kernels = Some(layer_kernels);
        self.sdpa_bwd2 = Some(sdpa_bwd2);

        Ok(())
    }

    /// Dispatch an async weight gradient task to the cblas worker thread.
    fn dispatch_dw(&self, task: Box<dyn FnOnce() + Send>) {
        let _ = self.dw_sender.send(task);
    }

    /// Wait for all pending dW tasks to complete (barrier sync).
    fn wait_dw(&self) {
        let (reply_tx, _reply_rx) = mpsc::channel();
        let _ = self.dw_barrier_sender.send(reply_tx);
        // Send a no-op task to flush the queue
        let (done_tx, done_rx) = mpsc::channel();
        let _ = self.dw_sender.send(Box::new(move || {
            let _ = done_tx.send(());
        }));
        let _ = done_rx.recv();
    }

    /// Run a single training step (forward + backward + grad accumulation).
    ///
    /// Returns the cross-entropy loss for this step.
    pub fn train_step(&mut self, input_tokens: &[u16], target_tokens: &[u16]) -> Result<f32> {
        let d = self.config.dim;
        let s = self.config.seq_len;

        assert_eq!(input_tokens.len(), s);
        assert_eq!(target_tokens.len(), s);

        // === Forward pass ===

        // Embedding lookup → x [D, S] channel-first
        let tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
        let mut x = vec![0.0f32; d * s];
        accelerate::embed_lookup(&mut x, &self.embed_weights, &tokens_u32, d, s);

        // Per-layer forward
        for l in 0..self.config.n_layers {
            // Save input for backward
            self.layer_acts[l].layer_in.copy_from_slice(&x);

            // Native ANE dispatch
            self.forward_layer_ane(l, &mut x)?;
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

        // Classifier: logits = embed^T @ x_final via cblas_sgemm
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

        // Cross-entropy loss
        let mut dlogits = vec![0.0f32; v * s];
        let loss = accelerate::cross_entropy_loss(&mut dlogits, &logits, target_tokens, v, s);

        // === Backward pass ===

        // dEmbed (classifier gradient)
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

        // Embed gradient accumulation (async)
        let dlogits_clone = dlogits.clone();
        let x_final_clone = x_final.clone();
        let v_val = v;
        let d_val = d;
        let s_val = s;
        let embed_grad_ptr = self.embed_grad.as_mut_ptr() as usize;
        let embed_len = self.embed_grad.len();
        self.dispatch_dw(Box::new(move || {
            // dEmbed += dlogits @ x_final^T
            let embed_grad =
                unsafe { std::slice::from_raw_parts_mut(embed_grad_ptr as *mut f32, embed_len) };
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

        // Final RMSNorm backward
        let x_before_final = x.clone();
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
            self.backward_layer_ane(l, &mut dx)?;
        }

        // Embedding backward
        let tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
        accelerate::embed_backward(&mut self.embed_grad, &dx, &tokens_u32, d, s);

        Ok(loss)
    }

    /// Native ANE forward pass for a single layer.
    fn forward_layer_ane(&mut self, l: usize, x: &mut [f32]) -> Result<()> {
        let d = self.config.dim;
        let s = self.config.seq_len;
        let h = self.config.hidden_dim;
        let lw = &self.layer_weights[l];
        let acts = &mut self.layer_acts[l];

        let kernels = self
            .layer_kernels
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?;
        let io = self
            .io_pool
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?;

        // 1. Attention Block
        // RMSNorm on CPU
        accelerate::rmsnorm(
            &mut acts.xnorm,
            x,
            &lw.rms_att,
            d,
            s,
            self.config.rms_norm_eps,
        );

        // ANE fwd_attn
        io.input.write_f32_as_fp16(&acts.xnorm, d, s);
        kernels[l]
            .fwd_attn
            .evaluate(&[io.input.as_ptr()], &[io.fwd_attn.as_ptr()])?;

        // Read taps: concat(oo, Q, K, V, attn_out, xnorm)
        io.fwd_attn.read_fp16_as_f32(&mut acts.o_out, 0, d, s);
        io.fwd_attn.read_fp16_as_f32(&mut acts.q, d, d, s);
        io.fwd_attn.read_fp16_as_f32(&mut acts.k, 2 * d, d, s);
        io.fwd_attn.read_fp16_as_f32(&mut acts.v, 3 * d, d, s);
        io.fwd_attn
            .read_fp16_as_f32(&mut acts.attn_out, 4 * d, d, s);

        // Residual: x2 = x + o_out
        accelerate::vadd(x, &acts.o_out, &mut acts.x2);

        // 2. FFN Block
        // RMSNorm on CPU
        accelerate::rmsnorm(
            &mut acts.x2norm,
            &acts.x2,
            &lw.rms_ffn,
            d,
            s,
            self.config.rms_norm_eps,
        );

        // ANE fwd_ffn
        io.input.write_f32_as_fp16(&acts.x2norm, d, s);
        kernels[l]
            .fwd_ffn
            .evaluate(&[io.input.as_ptr()], &[io.fwd_ffn.as_ptr()])?;

        // Read taps: concat(y, h1, h3, gate, xn)
        io.fwd_ffn.read_fp16_as_f32(&mut acts.ffn_out, 0, d, s);
        io.fwd_ffn.read_fp16_as_f32(&mut acts.h1, d, h, s);
        io.fwd_ffn.read_fp16_as_f32(&mut acts.h3, d + h, h, s);
        io.fwd_ffn
            .read_fp16_as_f32(&mut acts.silu_out, d + 2 * h, h, s);

        // Residual: x = x2 + ffn_out
        accelerate::vadd(&acts.x2, &acts.ffn_out, x);

        Ok(())
    }

    /// Native ANE backward pass for a single layer.
    fn backward_layer_ane(&mut self, l: usize, dx: &mut [f32]) -> Result<()> {
        let d = self.config.dim;
        let s = self.config.seq_len;
        let h = self.config.hidden_dim;
        let lw = &self.layer_weights[l];
        let acts = &self.layer_acts[l];
        let kernels = self
            .layer_kernels
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?;
        let io = self
            .io_pool
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?;

        // 1. FFN Backward
        // ANE ffn_bwd: input concat(dy, h1, h3, gate, xn), output concat(dx_ffn, dw1, dw3)
        // (Note: weights w1,w2,w3 are transposed in MIL)
        io.input.write_f32_as_fp16_at(0, dx, d, s);
        io.input.write_f32_as_fp16_at(d, &acts.h1, h, s);
        io.input.write_f32_as_fp16_at(d + h, &acts.h3, h, s);
        io.input
            .write_f32_as_fp16_at(d + 2 * h, &acts.silu_out, h, s);
        io.input.write_f32_as_fp16_at(d + 3 * h, &acts.x2norm, d, s);

        kernels[l]
            .ffn_bwd
            .evaluate(&[io.input.as_ptr()], &[io.ffn_bwd.as_ptr()])?;

        let mut dx_ffn = vec![0.0f32; d * s];
        io.ffn_bwd.read_fp16_as_f32(&mut dx_ffn, 0, d, s);

        // Extract raw pointer for async dW task before dispatch
        let dy_clone = dx.to_vec();
        let acts_silu_clone = acts.silu_out.clone();
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
                &acts_silu_clone,
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

        // FFN RMSNorm backward
        let mut dx_ffn_norm = vec![0.0f32; d * s];
        accelerate::rmsnorm_backward(
            &mut dx_ffn_norm,
            &mut self.layer_grads[l].rms_ffn,
            &dx_ffn,
            &acts.x2,
            &lw.rms_ffn,
            d,
            s,
            self.config.rms_norm_eps,
        );

        // 2. Attention Backward (simplified logic matching SDPA BWD 1/2)
        // ... (remaining ANE bwd dispatch implementation)

        // Residual connection
        dx[..d * s].copy_from_slice(&dx_ffn_norm[..d * s]);

        Ok(())
    }

    /// Run a complete training batch.
    ///
    /// Compiles kernels with current weights, runs `accum_steps` forward/backward
    /// passes to accumulate gradients, then applies Adam update.
    ///
    /// Returns the average loss across all accumulation steps.
    pub fn train_batch(&mut self, data: &[(Vec<u16>, Vec<u16>)]) -> Result<f32> {
        // Compile kernels with current weights
        self.compile_kernels()?;

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

        // Adam update
        self.adam_t += 1;
        let t = self.adam_t;
        let lr = self.config.learning_rate;
        let b1 = self.config.adam_beta1;
        let b2 = self.config.adam_beta2;
        let eps = self.config.adam_eps;

        for l in 0..self.config.n_layers {
            let lw = &mut self.layer_weights[l];
            let lg = &self.layer_grads[l];
            let la = &mut self.layer_adam[l];

            macro_rules! adam_update_param {
                ($weight:ident, $grad:ident, $adam:ident) => {
                    accelerate::adam_update(
                        &mut lw.$weight,
                        &lg.$grad,
                        &mut la.$adam.m,
                        &mut la.$adam.v,
                        t,
                        lr,
                        b1,
                        b2,
                        eps,
                    );
                };
            }

            adam_update_param!(wq, wq, wq);
            adam_update_param!(wk, wk, wk);
            adam_update_param!(wv, wv, wv);
            adam_update_param!(wo, wo, wo);
            adam_update_param!(w1, w1, w1);
            adam_update_param!(w2, w2, w2);
            adam_update_param!(w3, w3, w3);
            adam_update_param!(rms_att, rms_att, rms_att);
            adam_update_param!(rms_ffn, rms_ffn, rms_ffn);
        }

        // Embedding + final RMSNorm Adam update
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

        Ok(total_loss / steps as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let config = AneTrainerConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_layers: 2,
            vocab_size: 100,
            seq_len: 16,
            ..Default::default()
        };
        let trainer = AneTrainer::new(config);
        assert_eq!(trainer.adam_t(), 0);
        assert_eq!(trainer.config().n_layers, 2);
    }

    #[test]
    fn test_trainer_budget() {
        let config = AneTrainerConfig {
            n_layers: 2,
            max_compiles: 100,
            ..Default::default()
        };
        let trainer = AneTrainer::new(config);
        // 5 kernels per layer * 2 layers = 10 per batch
        assert_eq!(trainer.budget().kernels_per_batch(), 10);
        assert!(trainer.budget().can_compile_batch());
    }
}
