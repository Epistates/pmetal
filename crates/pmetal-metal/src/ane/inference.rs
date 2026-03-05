//! ANE inference engine for autoregressive generation.
//!
//! Provides a hybrid ANE prefill + CPU decode architecture for efficient
//! autoregressive text generation:
//!
//! - **Prefill (ANE)**: Processes the full prompt in one shot, extracting
//!   K/V projections for the KV cache via 3-way concat output.
//! - **Decode (CPU)**: Generates one token per step using cached KV pairs
//!   with `cblas_sgemv` for matrix-vector multiplies.
//!
//! # Features
//!
//! - KV cache eliminates O(n²×L) recomputation per token
//! - GQA/MQA support via `n_kv_heads` config field
//! - SafeTensors weight loading (single and multi-file)
//! - LoRA adapter fusion (merge before ANE kernel compilation)
//! - Legacy full-recomputation path preserved for backward compatibility

use std::path::Path;

use rand::RngExt;

use crate::accelerate;
use crate::ane::budget::CompileBudget;
use crate::ane::iosurface::IoSurface;
use crate::ane::kernel::{self, TransformerKernelConfig};
use crate::ane::runtime::{AneModel, AneRuntime};
use crate::error::{MetalError, Result};

/// Configuration for ANE inference.
#[derive(Debug, Clone)]
pub struct AneInferenceConfig {
    /// Model dimension (e.g., 768).
    pub dim: usize,
    /// FFN hidden dimension (e.g., 2048).
    pub hidden_dim: usize,
    /// Number of attention heads (e.g., 12).
    pub n_heads: usize,
    /// Number of key/value heads for GQA/MQA (defaults to `n_heads`).
    pub n_kv_heads: usize,
    /// Number of transformer layers (e.g., 12).
    pub n_layers: usize,
    /// Vocabulary size (e.g., 32000).
    pub vocab_size: usize,
    /// Maximum sequence length — kernels compiled for this fixed shape.
    pub max_seq_len: usize,
    /// Maximum ANE compilations per process (default 100).
    pub max_compiles: usize,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// EOS token ID for early stopping.
    pub eos_token_id: Option<u32>,
    /// RoPE base frequency (Qwen3 default: 1_000_000.0).
    pub rope_theta: f32,
    /// RMSNorm epsilon (Qwen3 default: 1e-6).
    pub rms_norm_eps: f32,
    /// Per-head dimension. Defaults to `dim / n_heads` if `None`.
    ///
    /// Models like Qwen3 may specify `head_dim` explicitly in config.json
    /// when it differs from `dim / n_heads` (e.g., Qwen3-0.6B uses head_dim=128
    /// with dim=2048, n_heads=16).
    pub head_dim: Option<usize>,
}

impl Default for AneInferenceConfig {
    fn default() -> Self {
        Self {
            dim: 768,
            hidden_dim: 2048,
            n_heads: 12,
            n_kv_heads: 12,
            n_layers: 12,
            vocab_size: 32000,
            max_seq_len: 256,
            max_compiles: 100,
            temperature: 0.0,
            top_k: 0,
            max_tokens: 128,
            eos_token_id: None,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            head_dim: None,
        }
    }
}

/// Per-layer weight storage (f32, row-major).
struct InferenceLayerWeights {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    rms_att: Vec<f32>,
    rms_ffn: Vec<f32>,
    /// Per-head RMSNorm weights for Q, shape `[head_dim]`.
    q_norm: Vec<f32>,
    /// Per-head RMSNorm weights for K, shape `[head_dim]`.
    k_norm: Vec<f32>,
}

/// Per-layer compiled ANE kernels (inference-only, no backward).
struct InferenceLayerKernels {
    /// Prefill attention kernel: outputs concat(oo, K_proj, V_proj).
    fwd_attn_kv: AneModel,
    /// FFN kernel (unchanged between prefill/decode on ANE).
    fwd_ffn: AneModel,
}

/// IOSurface pool reused across layers and generation steps.
struct IoSurfacePool {
    input: IoSurface,
    /// Output surface sized for the larger KV-tapped attention output.
    output_attn: IoSurface,
    /// Output surface for FFN (same size as input).
    output_ffn: IoSurface,
}

/// Per-layer KV cache (f32, channel-first `[kv_dim, max_seq_len]`).
struct LayerKvCache {
    /// K cache: `[kv_dim, max_seq_len]` f32.
    k: Vec<f32>,
    /// V cache: `[kv_dim, max_seq_len]` f32.
    v: Vec<f32>,
}

/// KV cache pool across all transformer layers.
struct KvCachePool {
    layers: Vec<LayerKvCache>,
    /// Current position (number of tokens cached).
    pos: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
}

impl KvCachePool {
    fn new(n_layers: usize, kv_dim: usize, max_seq_len: usize) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(LayerKvCache {
                k: vec![0.0; kv_dim * max_seq_len],
                v: vec![0.0; kv_dim * max_seq_len],
            });
        }
        Self {
            layers,
            pos: 0,
            max_seq_len,
        }
    }
}

/// ANE inference engine for autoregressive text generation.
///
/// Supports two generation modes:
/// - **`generate()`**: Legacy full-recomputation path (backward compat)
/// - **`generate_cached()`**: Hybrid ANE prefill + CPU decode with KV cache
pub struct AneInferenceEngine {
    config: AneInferenceConfig,
    kernel_config: TransformerKernelConfig,
    layer_weights: Vec<InferenceLayerWeights>,
    layer_kernels: Option<Vec<InferenceLayerKernels>>,
    embed_weights: Vec<f32>,
    rms_final: Vec<f32>,
    budget: CompileBudget,
    io_pool: Option<IoSurfacePool>,
    /// Set to true after `compile_kernels()` is called.
    compiled: bool,
}

impl AneInferenceEngine {
    /// Create a new ANE inference engine.
    ///
    /// Returns an error if the configuration is invalid (e.g., `n_heads == 0`,
    /// `n_heads % n_kv_heads != 0`, `dim % n_heads != 0`).
    pub fn new(config: AneInferenceConfig) -> Result<Self> {
        if config.n_heads == 0 {
            return Err(MetalError::InvalidConfig("n_heads must be > 0".into()));
        }
        if config.n_kv_heads == 0 {
            return Err(MetalError::InvalidConfig("n_kv_heads must be > 0".into()));
        }
        if config.n_kv_heads > config.n_heads {
            return Err(MetalError::InvalidConfig(
                "n_kv_heads must be <= n_heads".into(),
            ));
        }
        if config.n_heads % config.n_kv_heads != 0 {
            return Err(MetalError::InvalidConfig(
                "n_heads must be divisible by n_kv_heads".into(),
            ));
        }
        // Resolve head_dim: use explicit value if provided, otherwise derive from dim/n_heads
        let hd = if let Some(explicit_hd) = config.head_dim {
            if explicit_hd == 0 {
                return Err(MetalError::InvalidConfig("head_dim must be > 0".into()));
            }
            explicit_hd
        } else {
            if config.dim == 0 || config.dim % config.n_heads != 0 {
                return Err(MetalError::InvalidConfig(
                    "dim must be > 0 and divisible by n_heads (or specify head_dim explicitly)"
                        .into(),
                ));
            }
            config.dim / config.n_heads
        };

        let d = config.dim;
        let h = config.hidden_dim;
        let nl = config.n_layers;
        let v = config.vocab_size;
        let q_dim = config.n_heads * hd; // May differ from d

        let n_kv_heads = config.n_kv_heads;
        let kernel_config = TransformerKernelConfig {
            dim: d,
            hidden_dim: h,
            n_heads: config.n_heads,
            n_kv_heads,
            head_dim: hd,
            seq_len: config.max_seq_len,
            rope_theta: config.rope_theta,
            rms_norm_eps: config.rms_norm_eps,
        };

        let kv_dim = n_kv_heads * hd;
        let mut layer_weights = Vec::with_capacity(nl);
        for _ in 0..nl {
            layer_weights.push(InferenceLayerWeights {
                wq: vec![0.0; q_dim * d],
                wk: vec![0.0; kv_dim * d],
                wv: vec![0.0; kv_dim * d],
                wo: vec![0.0; d * q_dim],
                w1: vec![0.0; h * d],
                w2: vec![0.0; d * h],
                w3: vec![0.0; h * d],
                rms_att: vec![0.0; d],
                rms_ffn: vec![0.0; d],
                q_norm: vec![1.0; hd],
                k_norm: vec![1.0; hd],
            });
        }

        // 2 kernels per layer (fwd_attn + fwd_ffn)
        let budget = CompileBudget::new(config.max_compiles, 2 * nl);

        Ok(Self {
            config,
            kernel_config,
            layer_weights,
            layer_kernels: None,
            embed_weights: vec![0.0; v * d],
            rms_final: vec![0.0; d],
            budget,
            io_pool: None,
            compiled: false,
        })
    }

    /// Get a reference to the engine configuration.
    pub fn config(&self) -> &AneInferenceConfig {
        &self.config
    }

    /// Get the compilation budget tracker.
    pub fn budget(&self) -> &CompileBudget {
        &self.budget
    }

    /// Load weights from a flat f32 array (llama2.c layout).
    ///
    /// Layout: `embed[V*D]`, then per-layer: `rms_att[D], wq[D*D], wk[D*D],
    /// wv[D*D], wo[D*D], rms_ffn[D], w1[H*D], w2[D*H], w3[H*D]`,
    /// then `rms_final[D]`.
    pub fn load_weights_flat(&mut self, weights: &[f32]) {
        let d = self.config.dim;
        let h = self.config.hidden_dim;
        let nl = self.config.n_layers;
        let v = self.config.vocab_size;
        let kv_dim = self.kernel_config.kv_dim();
        let q_dim = self.kernel_config.q_dim();

        let mut offset = 0;

        // Embedding
        self.embed_weights[..v * d].copy_from_slice(&weights[offset..offset + v * d]);
        offset += v * d;

        // Per-layer
        for l in 0..nl {
            let lw = &mut self.layer_weights[l];

            lw.rms_att.copy_from_slice(&weights[offset..offset + d]);
            offset += d;

            lw.wq.copy_from_slice(&weights[offset..offset + q_dim * d]);
            offset += q_dim * d;
            lw.wk.copy_from_slice(&weights[offset..offset + kv_dim * d]);
            offset += kv_dim * d;
            lw.wv.copy_from_slice(&weights[offset..offset + kv_dim * d]);
            offset += kv_dim * d;
            lw.wo.copy_from_slice(&weights[offset..offset + d * q_dim]);
            offset += d * q_dim;

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

    /// Compile forward-only kernels for all layers.
    ///
    /// Uses 2 compilations per layer (fwd_attn_kv + fwd_ffn).
    /// Must be called after loading weights and before any LoRA fusion.
    pub fn compile_kernels(&mut self) -> Result<()> {
        kernel::validate_config(&self.kernel_config)?;
        let rt = AneRuntime::global()?;

        if !self.budget.can_compile_batch() {
            return Err(MetalError::AneCompileFailed(format!(
                "Compile budget exhausted: {}/{} used",
                self.budget.current(),
                self.budget.max()
            )));
        }

        // Free old kernels
        self.layer_kernels = None;

        let cfg = &self.kernel_config;
        let d = cfg.dim;
        let s = cfg.seq_len;
        let kv_d = cfg.kv_dim();
        let mut layer_kernels = Vec::with_capacity(self.config.n_layers);

        for l in 0..self.config.n_layers {
            let lw = &self.layer_weights[l];

            // Forward attention with KV cache output (includes QK-norm + RoPE)
            let fwd_attn_out = kernel::gen_sdpa_fwd_kv(
                cfg,
                &lw.rms_att,
                &lw.wq,
                &lw.wk,
                &lw.wv,
                &lw.wo,
                &lw.q_norm,
                &lw.k_norm,
            );
            let fwd_attn_kv = match rt.compile(
                fwd_attn_out.mil_text.as_bytes(),
                Some(&fwd_attn_out.weights),
            ) {
                Ok(model) => model,
                Err(e) => {
                    let _ = std::fs::write(
                        format!("/tmp/ane_debug_layer{l}_attn.mil"),
                        &fwd_attn_out.mil_text,
                    );
                    tracing::error!("SDPA kernel compile failed layer {l}: {e}");
                    return Err(e);
                }
            };
            self.budget.record_compile();

            // Forward FFN (inference — no taps)
            let fwd_ffn_out = kernel::gen_ffn_fwd(cfg, &lw.rms_ffn, &lw.w1, &lw.w3, &lw.w2);
            let fwd_ffn =
                match rt.compile(fwd_ffn_out.mil_text.as_bytes(), Some(&fwd_ffn_out.weights)) {
                    Ok(model) => model,
                    Err(e) => {
                        let _ = std::fs::write(
                            format!("/tmp/ane_debug_layer{l}_ffn.mil"),
                            &fwd_ffn_out.mil_text,
                        );
                        tracing::error!("FFN kernel compile failed layer {l}: {e}");
                        return Err(e);
                    }
                };
            self.budget.record_compile();

            layer_kernels.push(InferenceLayerKernels {
                fwd_attn_kv,
                fwd_ffn,
            });
        }

        self.layer_kernels = Some(layer_kernels);

        // Allocate IOSurface pool
        let input_bytes = d * s * 2;
        let attn_output_bytes = (d + 2 * kv_d) * s * 2; // concat(oo, kf, vf)
        let ffn_output_bytes = d * s * 2;
        self.io_pool = Some(IoSurfacePool {
            input: IoSurface::new(input_bytes)?,
            output_attn: IoSurface::new(attn_output_bytes)?,
            output_ffn: IoSurface::new(ffn_output_bytes)?,
        });

        self.compiled = true;
        Ok(())
    }

    /// Run a forward pass and return logits `[V, S]` (channel-first).
    ///
    /// `token_ids` are padded/truncated to `max_seq_len`. Positions beyond
    /// input length are masked by the causal mask compiled into the kernels.
    ///
    /// This is the legacy full-recomputation path. Prefer `generate_cached()`
    /// for production use.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let d = self.config.dim;
        let s = self.config.max_seq_len;
        let v = self.config.vocab_size;

        let kernels = self
            .layer_kernels
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?;
        let io_pool = self
            .io_pool
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?;

        if token_ids.is_empty() || token_ids.len() > s {
            return Err(MetalError::InvalidConfig(format!(
                "Input length {} must be in [1, {}]",
                token_ids.len(),
                s
            )));
        }

        // Embedding lookup → x [D, S] channel-first, zero-padded
        let mut padded_tokens = vec![0u32; s];
        for (i, &tid) in token_ids.iter().enumerate() {
            padded_tokens[i] = tid;
        }

        let mut x = vec![0.0f32; d * s];
        accelerate::embed_lookup(&mut x, &self.embed_weights, &padded_tokens, d, s);

        // Per-layer forward
        let mut o_out = vec![0.0f32; d * s];
        let mut x2 = vec![0.0f32; d * s];
        let mut ffn_out = vec![0.0f32; d * s];

        for lk in kernels {
            // Attention: write x → ANE fwd_attn_kv → read oo (first D channels)
            io_pool.input.write_f32_as_fp16(&x, d, s);
            lk.fwd_attn_kv
                .evaluate(&[io_pool.input.as_ptr()], &[io_pool.output_attn.as_ptr()])?;
            io_pool.output_attn.read_fp16_as_f32(&mut o_out, 0, d, s);

            // Residual: x2 = x + o_out
            accelerate::vadd(&x, &o_out, &mut x2);

            // FFN: write x2 → ANE fwd_ffn → read ffn_out
            io_pool.input.write_f32_as_fp16(&x2, d, s);
            lk.fwd_ffn
                .evaluate(&[io_pool.input.as_ptr()], &[io_pool.output_ffn.as_ptr()])?;
            io_pool.output_ffn.read_fp16_as_f32(&mut ffn_out, 0, d, s);

            // Residual: x = x2 + ffn_out
            accelerate::vadd(&x2, &ffn_out, &mut x);
        }

        // Final RMSNorm
        let mut x_final = vec![0.0f32; d * s];
        accelerate::rmsnorm(&mut x_final, &x, &self.rms_final, d, s);

        // Classifier: logits = embed @ x_final → [V, S]
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

        Ok(logits)
    }

    /// Generate tokens autoregressively.
    ///
    /// Full-sequence recomputation per step (no KV cache).
    /// Stops on EOS, max_tokens, or when sequence fills `max_seq_len`.
    pub fn generate(&self, input_ids: &[u32]) -> Result<Vec<u32>> {
        self.generate_streaming(input_ids, |_| true)
    }

    /// Generate tokens with a streaming callback.
    ///
    /// `on_token` is called with each generated token. Return `false` to stop.
    pub fn generate_streaming<F>(&self, input_ids: &[u32], mut on_token: F) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        let s = self.config.max_seq_len;
        let v = self.config.vocab_size;

        if input_ids.is_empty() {
            return Err(MetalError::InvalidConfig("Input must not be empty".into()));
        }

        let mut sequence: Vec<u32> = input_ids.to_vec();

        for _ in 0..self.config.max_tokens {
            if sequence.len() >= s {
                break;
            }

            let logits = self.forward(&sequence)?;
            let pos = sequence.len() - 1;

            // Extract logits column at current position from [V, S] layout
            let mut logits_col = vec![0.0f32; v];
            for tok in 0..v {
                logits_col[tok] = logits[tok * s + pos];
            }

            let next_token = sample(&logits_col, self.config.temperature, self.config.top_k);

            if let Some(eos) = self.config.eos_token_id {
                if next_token == eos {
                    break;
                }
            }

            if !on_token(next_token) {
                break;
            }

            sequence.push(next_token);
        }

        Ok(sequence)
    }

    // ========================================================================
    // KV-cached generation (ANE prefill + CPU decode)
    // ========================================================================

    /// Generate tokens with KV cache (ANE prefill + CPU decode).
    ///
    /// This is the recommended generation method. Uses ANE for the initial
    /// prompt processing and CPU for single-token decode steps with cached
    /// KV pairs.
    pub fn generate_cached(&self, input_ids: &[u32]) -> Result<Vec<u32>> {
        self.generate_cached_streaming(input_ids, |_| true)
    }

    /// Generate tokens with KV cache and streaming callback.
    ///
    /// `on_token` is called with each generated token. Return `false` to stop.
    pub fn generate_cached_streaming<F>(
        &self,
        input_ids: &[u32],
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        let s = self.config.max_seq_len;
        let v = self.config.vocab_size;
        let kv_dim = self.kernel_config.kv_dim();

        if input_ids.is_empty() {
            return Err(MetalError::InvalidConfig("Input must not be empty".into()));
        }
        if input_ids.len() > s {
            return Err(MetalError::InvalidConfig(format!(
                "Input length {} exceeds max_seq_len {}",
                input_ids.len(),
                s
            )));
        }

        let mut kv_cache = KvCachePool::new(self.config.n_layers, kv_dim, s);

        // Prefill: process full prompt via ANE, populate KV cache
        let logits = self.prefill(input_ids, &mut kv_cache)?;
        let pos = input_ids.len() - 1;

        // Extract logits column at last input position
        let mut logits_col = vec![0.0f32; v];
        for tok in 0..v {
            logits_col[tok] = logits[tok * s + pos];
        }

        let next = sample(&logits_col, self.config.temperature, self.config.top_k);

        let mut sequence = input_ids.to_vec();

        if let Some(eos) = self.config.eos_token_id {
            if next == eos {
                return Ok(sequence);
            }
        }
        if !on_token(next) {
            return Ok(sequence);
        }
        sequence.push(next);

        // Decode loop: CPU-only with KV cache
        for _ in 1..self.config.max_tokens {
            if kv_cache.pos >= s {
                break;
            }

            let logits_vec = self.decode_step(*sequence.last().unwrap(), &mut kv_cache)?;
            let next = sample(&logits_vec, self.config.temperature, self.config.top_k);

            if let Some(eos) = self.config.eos_token_id {
                if next == eos {
                    break;
                }
            }
            if !on_token(next) {
                break;
            }
            sequence.push(next);
        }

        Ok(sequence)
    }

    /// ANE prefill: process full prompt, populate KV cache, return logits.
    fn prefill(&self, token_ids: &[u32], kv_cache: &mut KvCachePool) -> Result<Vec<f32>> {
        let d = self.config.dim;
        let s = self.config.max_seq_len;
        let v = self.config.vocab_size;
        let kv_dim = self.kernel_config.kv_dim();
        let input_len = token_ids.len();

        let kernels = self
            .layer_kernels
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("Kernels not compiled".into()))?;
        let io_pool = self
            .io_pool
            .as_ref()
            .ok_or_else(|| MetalError::InvalidConfig("IO pool not allocated".into()))?;

        // Embedding lookup → x [D, S] channel-first, zero-padded
        let mut padded_tokens = vec![0u32; s];
        for (i, &tid) in token_ids.iter().enumerate() {
            padded_tokens[i] = tid;
        }

        let mut x = vec![0.0f32; d * s];
        accelerate::embed_lookup(&mut x, &self.embed_weights, &padded_tokens, d, s);

        let mut o_out = vec![0.0f32; d * s];
        let mut kf_buf = vec![0.0f32; kv_dim * s];
        let mut vf_buf = vec![0.0f32; kv_dim * s];
        let mut x2 = vec![0.0f32; d * s];
        let mut ffn_out = vec![0.0f32; d * s];

        for (layer_idx, lk) in kernels.iter().enumerate() {
            // Attention: write x → ANE fwd_attn_kv → read concat(oo, kf, vf)
            io_pool.input.write_f32_as_fp16(&x, d, s);
            lk.fwd_attn_kv
                .evaluate(&[io_pool.input.as_ptr()], &[io_pool.output_attn.as_ptr()])?;

            // Read oo (channels 0..D)
            io_pool.output_attn.read_fp16_as_f32(&mut o_out, 0, d, s);
            // Read kf (channels D..D+kv_dim)
            io_pool
                .output_attn
                .read_fp16_as_f32(&mut kf_buf, d, kv_dim, s);
            // Read vf (channels D+kv_dim..D+2*kv_dim)
            io_pool
                .output_attn
                .read_fp16_as_f32(&mut vf_buf, d + kv_dim, kv_dim, s);

            // Cache K, V for positions 0..input_len
            // KV data is in channel-first [kv_dim, S] layout
            let lc = &mut kv_cache.layers[layer_idx];
            for ch in 0..kv_dim {
                for t in 0..input_len {
                    lc.k[ch * s + t] = kf_buf[ch * s + t];
                    lc.v[ch * s + t] = vf_buf[ch * s + t];
                }
            }

            // Residual: x2 = x + o_out
            accelerate::vadd(&x, &o_out, &mut x2);

            // FFN
            io_pool.input.write_f32_as_fp16(&x2, d, s);
            lk.fwd_ffn
                .evaluate(&[io_pool.input.as_ptr()], &[io_pool.output_ffn.as_ptr()])?;
            io_pool.output_ffn.read_fp16_as_f32(&mut ffn_out, 0, d, s);

            // Residual: x = x2 + ffn_out
            accelerate::vadd(&x2, &ffn_out, &mut x);
        }

        kv_cache.pos = input_len;

        // Final RMSNorm → logits
        let mut x_final = vec![0.0f32; d * s];
        accelerate::rmsnorm(&mut x_final, &x, &self.rms_final, d, s);

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

        Ok(logits)
    }

    /// CPU-only decode step: process single token with cached KV pairs.
    ///
    /// Returns logits vector `[V]` for the new token.
    #[allow(clippy::needless_range_loop)]
    fn decode_step(&self, token_id: u32, kv_cache: &mut KvCachePool) -> Result<Vec<f32>> {
        let d = self.config.dim;
        let v = self.config.vocab_size;
        let h = self.config.n_heads;
        let hd = self.kernel_config.head_dim;
        let kv_dim = self.kernel_config.kv_dim();
        let n_groups = self.kernel_config.n_groups();
        let s = kv_cache.max_seq_len;
        let pos = kv_cache.pos;
        let seq_through = pos + 1; // positions 0..=pos

        // Bounds check token ID
        if token_id as usize >= v {
            return Err(MetalError::InvalidConfig(format!(
                "Token ID {} exceeds vocab size {}",
                token_id, v
            )));
        }

        // Embed single token → x [D]
        let mut x = vec![0.0f32; d];
        for dim_i in 0..d {
            x[dim_i] = self.embed_weights[token_id as usize * d + dim_i];
        }

        let n_kv_heads = self.kernel_config.n_kv_heads;
        let q_dim = self.kernel_config.q_dim();
        let mut xnorm = vec![0.0f32; d];
        let mut q = vec![0.0f32; q_dim];
        let mut k_new = vec![0.0f32; kv_dim];
        let mut v_new = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut wo_out = vec![0.0f32; d];
        let mut x2 = vec![0.0f32; d];
        let mut h1 = vec![0.0f32; self.config.hidden_dim];
        let mut h3 = vec![0.0f32; self.config.hidden_dim];
        let mut ffn_out = vec![0.0f32; d];
        // Pooled scores buffer — allocated once, reused across heads/layers
        let mut scores = vec![0.0f32; s];

        for (layer_idx, lw) in self.layer_weights.iter().enumerate() {
            // RMSNorm (seq=1, channel-first layout collapses to simple vector)
            rmsnorm_vec(&mut xnorm, &x, &lw.rms_att, d);

            // Q, K, V projections via gemv: out = W @ xnorm
            gemv(&lw.wq, &xnorm, &mut q, q_dim, d);
            gemv(&lw.wk, &xnorm, &mut k_new, kv_dim, d);
            gemv(&lw.wv, &xnorm, &mut v_new, kv_dim, d);

            // Per-head QK-norm
            let mut q_normed = vec![0.0f32; q_dim];
            rmsnorm_per_head(
                &mut q_normed,
                &q,
                &lw.q_norm,
                h,
                hd,
                self.config.rms_norm_eps,
            );
            let mut k_normed = vec![0.0f32; kv_dim];
            rmsnorm_per_head(
                &mut k_normed,
                &k_new,
                &lw.k_norm,
                n_kv_heads,
                hd,
                self.config.rms_norm_eps,
            );

            // RoPE at position `pos`
            apply_rope_vec(&mut q_normed, h, hd, pos, self.config.rope_theta);
            apply_rope_vec(&mut k_normed, n_kv_heads, hd, pos, self.config.rope_theta);

            // Store RoPE'd K, raw V at cache position
            let lc = &mut kv_cache.layers[layer_idx];
            for ch in 0..kv_dim {
                lc.k[ch * s + pos] = k_normed[ch];
                lc.v[ch * s + pos] = v_new[ch];
            }

            // Multi-head attention with GQA
            // Q is [D] = [n_heads * head_dim], K/V cache is [kv_dim, max_seq_len]
            attn_out.fill(0.0);
            let scale = 1.0 / (hd as f32).sqrt();

            for head in 0..h {
                let kv_head = head / n_groups;
                let q_off = head * hd;
                let kv_off = kv_head * hd;

                // scores[t] = sum_i(Q[q_off+i] * K_cache[kv_off+i, t]) for t in 0..seq_through
                for t in 0..seq_through {
                    let mut dot = 0.0f32;
                    for i in 0..hd {
                        dot += q_normed[q_off + i] * lc.k[(kv_off + i) * s + t];
                    }
                    scores[t] = dot * scale;
                }

                // Softmax over scores
                let max_s = scores[..seq_through]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for sc in &mut scores[..seq_through] {
                    *sc = (*sc - max_s).exp();
                    sum += *sc;
                }
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    for sc in &mut scores[..seq_through] {
                        *sc *= inv_sum;
                    }
                } else {
                    let uniform = 1.0 / seq_through as f32;
                    scores[..seq_through].fill(uniform);
                }

                // attn_out[q_off+i] = sum_t(scores[t] * V_cache[kv_off+i, t])
                for i in 0..hd {
                    let mut val = 0.0f32;
                    for t in 0..seq_through {
                        val += scores[t] * lc.v[(kv_off + i) * s + t];
                    }
                    attn_out[q_off + i] = val;
                }
            }

            // Wo projection: wo_out = Wo @ attn_out (projects q_dim→dim)
            gemv(&lw.wo, &attn_out, &mut wo_out, d, q_dim);

            // Residual: x2 = x + wo_out
            for i in 0..d {
                x2[i] = x[i] + wo_out[i];
            }

            // FFN: RMSNorm → SwiGLU
            rmsnorm_vec(&mut xnorm, &x2, &lw.rms_ffn, d);

            // h1 = W1 @ xnorm, h3 = W3 @ xnorm
            gemv(&lw.w1, &xnorm, &mut h1, self.config.hidden_dim, d);
            gemv(&lw.w3, &xnorm, &mut h3, self.config.hidden_dim, d);

            // SiLU gate: gate = silu(h1) * h3
            accelerate::silu_inplace(&mut h1);
            for i in 0..self.config.hidden_dim {
                h1[i] *= h3[i];
            }

            // Down projection: ffn_out = W2 @ h1
            gemv(&lw.w2, &h1, &mut ffn_out, d, self.config.hidden_dim);

            // Residual: x = x2 + ffn_out
            for i in 0..d {
                x[i] = x2[i] + ffn_out[i];
            }
        }

        kv_cache.pos = pos + 1;

        // Final RMSNorm
        let mut x_final = vec![0.0f32; d];
        rmsnorm_vec(&mut x_final, &x, &self.rms_final, d);

        // Logits: embed^T @ x_final → [V]
        let mut logits = vec![0.0f32; v];
        accelerate::gemm(
            &self.embed_weights,
            &x_final,
            &mut logits,
            v,
            1,
            d,
            1.0,
            0.0,
            false,
            false,
        );

        Ok(logits)
    }

    // ========================================================================
    // SafeTensors loading
    // ========================================================================

    /// Load weights from a SafeTensors file (HuggingFace format).
    ///
    /// Supports single-file (`model.safetensors`) and multi-file formats
    /// (via `model.safetensors.index.json`).
    ///
    /// Must be called before `compile_kernels()`.
    pub fn load_weights_safetensors(&mut self, path: &Path) -> Result<()> {
        use memmap2::Mmap;
        use safetensors::SafeTensors;

        let d = self.config.dim;
        let kv_dim = self.kernel_config.kv_dim();
        let q_dim = self.kernel_config.q_dim();
        let hd = self.kernel_config.head_dim;

        // Determine files to load
        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            // Look for index.json for multi-file format
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
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to mmap {:?}: {e}", file_path))
            })?;
            let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to parse safetensors: {e}"))
            })?;

            for (name, tensor) in tensors.tensors() {
                let data_f32 = match safetensors_to_f32(&tensor) {
                    Ok(data) => data,
                    Err(_) => continue, // skip unsupported dtypes
                };

                if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
                    if name == "model.embed_tokens.weight" {
                        let expected = self.config.vocab_size * d;
                        if data_f32.len() >= expected {
                            self.embed_weights[..expected].copy_from_slice(&data_f32[..expected]);
                        }
                    }
                    continue;
                }

                if name == "model.norm.weight" {
                    if data_f32.len() == d {
                        self.rms_final.copy_from_slice(&data_f32);
                    }
                    continue;
                }

                // Parse layer index from "model.layers.{i}.xxx"
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
                    let suffix = parts[1];

                    match suffix {
                        "self_attn.q_proj.weight" => {
                            copy_weight(&data_f32, &mut lw.wq, q_dim * d);
                        }
                        "self_attn.k_proj.weight" => {
                            copy_weight(&data_f32, &mut lw.wk, kv_dim * d);
                        }
                        "self_attn.v_proj.weight" => {
                            copy_weight(&data_f32, &mut lw.wv, kv_dim * d);
                        }
                        "self_attn.o_proj.weight" => {
                            copy_weight(&data_f32, &mut lw.wo, d * q_dim);
                        }
                        "self_attn.q_norm.weight" => copy_weight(&data_f32, &mut lw.q_norm, hd),
                        "self_attn.k_norm.weight" => copy_weight(&data_f32, &mut lw.k_norm, hd),
                        "mlp.gate_proj.weight" => {
                            copy_weight(&data_f32, &mut lw.w1, self.config.hidden_dim * d);
                        }
                        "mlp.down_proj.weight" => {
                            copy_weight(&data_f32, &mut lw.w2, d * self.config.hidden_dim);
                        }
                        "mlp.up_proj.weight" => {
                            copy_weight(&data_f32, &mut lw.w3, self.config.hidden_dim * d);
                        }
                        "input_layernorm.weight" => copy_weight(&data_f32, &mut lw.rms_att, d),
                        "post_attention_layernorm.weight" => {
                            copy_weight(&data_f32, &mut lw.rms_ffn, d);
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    // LoRA fusion
    // ========================================================================

    /// Load and merge a LoRA adapter into the base weights.
    ///
    /// Reads `adapter_config.json` and `adapter_model.safetensors` from `adapter_dir`.
    /// Merges each target module's weights: `W += (alpha/rank) * B @ A`.
    ///
    /// **Must be called before `compile_kernels()`** — weights are baked into
    /// ANE kernels at compile time.
    pub fn load_lora_adapter(&mut self, adapter_dir: &Path) -> Result<()> {
        use memmap2::Mmap;
        use safetensors::SafeTensors;

        if self.compiled {
            return Err(MetalError::InvalidConfig(
                "LoRA adapter must be loaded before compile_kernels()".into(),
            ));
        }

        // Read adapter config
        let config_path = adapter_dir.join("adapter_config.json");
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
            MetalError::InvalidConfig(format!("Failed to read adapter_config.json: {e}"))
        })?;
        let config: serde_json::Value = serde_json::from_str(&config_text).map_err(|e| {
            MetalError::InvalidConfig(format!("Failed to parse adapter_config.json: {e}"))
        })?;

        let rank = config["r"]
            .as_u64()
            .ok_or_else(|| MetalError::InvalidConfig("adapter_config.json missing 'r'".into()))?
            as usize;
        if rank == 0 {
            return Err(MetalError::InvalidConfig("LoRA rank must be > 0".into()));
        }
        let alpha = config["lora_alpha"].as_f64().ok_or_else(|| {
            MetalError::InvalidConfig("adapter_config.json missing 'lora_alpha'".into())
        })? as f32;
        let scale = alpha / rank as f32;

        // Load adapter weights
        let adapter_path = adapter_dir.join("adapter_model.safetensors");
        let file = std::fs::File::open(&adapter_path).map_err(|e| {
            MetalError::InvalidConfig(format!("Failed to open adapter safetensors: {e}"))
        })?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| MetalError::InvalidConfig(format!("Failed to mmap adapter: {e}")))?;
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
            MetalError::InvalidConfig(format!("Failed to parse adapter safetensors: {e}"))
        })?;

        // Collect LoRA A/B pairs per layer and module
        // Tensor names: "base_model.model.model.layers.{i}.self_attn.{proj}.lora_A.weight"
        // or simpler: "model.layers.{i}.self_attn.{proj}.lora_A.weight"
        let tensor_map: std::collections::HashMap<String, Vec<f32>> = tensors
            .tensors()
            .into_iter()
            .filter_map(|(name, view)| safetensors_to_f32(&view).ok().map(|data| (name, data)))
            .collect();

        for layer_idx in 0..self.config.n_layers {
            let lw = &mut self.layer_weights[layer_idx];

            for (proj_name, weight_field, rows, cols) in lora_target_modules(
                self.config.dim,
                self.kernel_config.kv_dim(),
                self.config.hidden_dim,
            ) {
                // Try both naming conventions
                let a_key = find_lora_key(&tensor_map, layer_idx, proj_name, "lora_A");
                let b_key = find_lora_key(&tensor_map, layer_idx, proj_name, "lora_B");

                if let (Some(a_data), Some(b_data)) = (
                    a_key.and_then(|k| tensor_map.get(&k)),
                    b_key.and_then(|k| tensor_map.get(&k)),
                ) {
                    // A: [rank, cols], B: [rows, rank]
                    // Validate tensor shapes
                    if a_data.len() != rank * cols || b_data.len() != rows * rank {
                        continue;
                    }
                    // W += scale * B @ A
                    let target = match weight_field {
                        "wq" => &mut lw.wq,
                        "wk" => &mut lw.wk,
                        "wv" => &mut lw.wv,
                        "wo" => &mut lw.wo,
                        "w1" => &mut lw.w1,
                        "w2" => &mut lw.w2,
                        "w3" => &mut lw.w3,
                        _ => continue,
                    };

                    // Compute scale * B @ A and add to target
                    let mut ba = vec![0.0f32; rows * cols];
                    accelerate::gemm(
                        b_data, a_data, &mut ba, rows, cols, rank, scale, 0.0, false, false,
                    );
                    for i in 0..rows * cols {
                        target[i] += ba[i];
                    }
                }
            }
        }

        Ok(())
    }
}

/// RMSNorm for a single vector (seq=1).
#[allow(clippy::needless_range_loop)]
fn rmsnorm_vec(out: &mut [f32], x: &[f32], w: &[f32], dim: usize) {
    let mut ss = 0.0f32;
    for i in 0..dim {
        ss += x[i] * x[i];
    }
    ss = 1.0 / (ss / dim as f32 + 1e-5).sqrt();
    for i in 0..dim {
        out[i] = w[i] * x[i] * ss;
    }
}

/// Per-head RMSNorm: independent RMSNorm on each head's `[head_dim]` slice.
///
/// `x` has `n_heads * head_dim` elements. Each head's slice is normalized
/// independently using the shared `weights` of length `head_dim`.
#[allow(clippy::needless_range_loop)]
fn rmsnorm_per_head(
    out: &mut [f32],
    x: &[f32],
    weights: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    for head in 0..n_heads {
        let off = head * head_dim;
        let mut ss = 0.0f32;
        for i in 0..head_dim {
            ss += x[off + i] * x[off + i];
        }
        ss = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            out[off + i] = weights[i] * x[off + i] * ss;
        }
    }
}

/// Apply non-traditional split-half RoPE to a vector with `n_heads * head_dim` elements.
///
/// For each head, splits into first/second halves along head_dim and applies:
/// ```text
/// out[d]           = x[d] * cos(angle) - x[d+half] * sin(angle)
/// out[d+half]      = x[d] * sin(angle) + x[d+half] * cos(angle)
/// ```
fn apply_rope_vec(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, rope_theta: f32) {
    let half_dim = head_dim / 2;
    for head in 0..n_heads {
        let off = head * head_dim;
        for d in 0..half_dim {
            let inv_freq = 1.0 / rope_theta.powf(2.0 * d as f32 / head_dim as f32);
            let angle = pos as f32 * inv_freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x_first = x[off + d];
            let x_second = x[off + d + half_dim];
            x[off + d] = x_first * cos_a - x_second * sin_a;
            x[off + d + half_dim] = x_first * sin_a + x_second * cos_a;
        }
    }
}

/// Matrix-vector multiply: out = W @ x, where W is [rows, cols] row-major.
fn gemv(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    accelerate::gemm(w, x, out, rows, 1, cols, 1.0, 0.0, false, false);
}

/// Copy weight data, clamping to target size.
fn copy_weight(src: &[f32], dst: &mut [f32], expected: usize) {
    let len = src.len().min(dst.len()).min(expected);
    dst[..len].copy_from_slice(&src[..len]);
}

/// Convert safetensors tensor data to f32.
fn safetensors_to_f32(tensor: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    use safetensors::Dtype;
    match tensor.dtype() {
        Dtype::F32 => {
            let bytes = tensor.data();
            if bytes.len() % 4 != 0 {
                return Err(MetalError::UnsupportedDtype(format!(
                    "F32 tensor data length {} is not a multiple of 4",
                    bytes.len()
                )));
            }
            let n = bytes.len() / 4;
            let mut out = vec![0.0f32; n];
            // SAFETY: f32 is 4 bytes, no alignment requirement for byte copy
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, n * 4);
            }
            Ok(out)
        }
        Dtype::F16 => {
            let bytes = tensor.data();
            if bytes.len() % 2 != 0 {
                return Err(MetalError::UnsupportedDtype(format!(
                    "F16 tensor data length {} is not a multiple of 2",
                    bytes.len()
                )));
            }
            let n = bytes.len() / 2;
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                out[i] = half::f16::from_bits(bits).to_f32();
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let bytes = tensor.data();
            if bytes.len() % 2 != 0 {
                return Err(MetalError::UnsupportedDtype(format!(
                    "BF16 tensor data length {} is not a multiple of 2",
                    bytes.len()
                )));
            }
            let n = bytes.len() / 2;
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                out[i] = half::bf16::from_bits(bits).to_f32();
            }
            Ok(out)
        }
        dtype => Err(MetalError::UnsupportedDtype(format!("{dtype:?}"))),
    }
}

/// Return the list of LoRA target modules with their weight field name and dimensions.
fn lora_target_modules(
    dim: usize,
    kv_dim: usize,
    hidden_dim: usize,
) -> Vec<(&'static str, &'static str, usize, usize)> {
    vec![
        ("q_proj", "wq", dim, dim),
        ("k_proj", "wk", kv_dim, dim),
        ("v_proj", "wv", kv_dim, dim),
        ("o_proj", "wo", dim, dim),
        ("gate_proj", "w1", hidden_dim, dim),
        ("down_proj", "w2", dim, hidden_dim),
        ("up_proj", "w3", hidden_dim, dim),
    ]
}

/// Find a LoRA tensor key trying both naming conventions.
fn find_lora_key(
    tensor_map: &std::collections::HashMap<String, Vec<f32>>,
    layer_idx: usize,
    proj_name: &str,
    ab: &str,
) -> Option<String> {
    // Convention 1: "base_model.model.model.layers.{i}.self_attn.{proj}.lora_{AB}.weight"
    let key1 =
        format!("base_model.model.model.layers.{layer_idx}.self_attn.{proj_name}.{ab}.weight");
    if tensor_map.contains_key(&key1) {
        return Some(key1);
    }

    // Convention 2: "model.layers.{i}.self_attn.{proj}.lora_{AB}.weight"
    let key2 = format!("model.layers.{layer_idx}.self_attn.{proj_name}.{ab}.weight");
    if tensor_map.contains_key(&key2) {
        return Some(key2);
    }

    // MLP modules
    let key3 = format!("base_model.model.model.layers.{layer_idx}.mlp.{proj_name}.{ab}.weight");
    if tensor_map.contains_key(&key3) {
        return Some(key3);
    }
    let key4 = format!("model.layers.{layer_idx}.mlp.{proj_name}.{ab}.weight");
    if tensor_map.contains_key(&key4) {
        return Some(key4);
    }

    None
}

/// Sample a token from a logits vector.
///
/// - `temperature == 0.0`: greedy (argmax)
/// - `temperature > 0.0`: scale → softmax → optional top-k → categorical
pub fn sample(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    if temperature < 1e-6 {
        // Greedy: argmax
        return sample_greedy(logits);
    }

    let v = logits.len();

    // Apply temperature
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for x in &mut scaled {
        *x = (*x - max_val).exp();
    }
    let sum: f32 = scaled.iter().sum();
    for x in &mut scaled {
        *x /= sum;
    }

    // Top-k filter
    if top_k > 0 && top_k < v {
        // Find top-k threshold
        let mut sorted: Vec<f32> = scaled.clone();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[top_k - 1];

        // Zero out below threshold
        for x in &mut scaled {
            if *x < threshold {
                *x = 0.0;
            }
        }

        // Renormalize
        let sum: f32 = scaled.iter().sum();
        if sum > 0.0 {
            for x in &mut scaled {
                *x /= sum;
            }
        }
    }

    // Categorical sampling
    let mut rng = rand::rng();
    let r: f32 = rng.random();
    let mut cumulative = 0.0;
    for (i, &p) in scaled.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }

    // Fallback (floating-point rounding)
    (v - 1) as u32
}

/// Greedy sampling: return the index of the maximum value.
fn sample_greedy(logits: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> AneInferenceConfig {
        AneInferenceConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 4,
            n_layers: 2,
            vocab_size: 100,
            max_seq_len: 16,
            max_compiles: 100,
            temperature: 0.0,
            top_k: 0,
            max_tokens: 32,
            eos_token_id: None,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            head_dim: None,
        }
    }

    #[test]
    fn test_inference_engine_creation() {
        let config = small_config();
        let engine = AneInferenceEngine::new(config).unwrap();
        assert_eq!(engine.config().dim, 64);
        assert_eq!(engine.config().n_layers, 2);
        assert!(engine.layer_kernels.is_none());
        assert!(engine.io_pool.is_none());
    }

    #[test]
    fn test_inference_engine_budget() {
        let config = small_config();
        let engine = AneInferenceEngine::new(config).unwrap();
        // 2 kernels per layer * 2 layers = 4 per compile batch
        assert_eq!(engine.budget().kernels_per_batch(), 4);
        assert!(engine.budget().can_compile_batch());
    }

    #[test]
    fn test_load_weights_flat() {
        let config = small_config();
        let mut engine = AneInferenceEngine::new(config.clone()).unwrap();

        // Calculate expected total weight count
        let d = config.dim;
        let h = config.hidden_dim;
        let nl = config.n_layers;
        let v = config.vocab_size;
        let kv_dim = config.n_kv_heads * (d / config.n_heads);

        let total = v * d // embed
            + nl * (d + d * d + kv_dim * d * 2 + d * d + d + h * d + d * h + h * d) // per-layer
            + d; // rms_final

        let weights = vec![1.0f32; total];
        engine.load_weights_flat(&weights);

        // Verify embed weights loaded
        assert_eq!(engine.embed_weights[0], 1.0);
        // Verify final rms loaded
        assert_eq!(engine.rms_final[0], 1.0);
    }

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sample(&logits, 0.0, 0);
        assert_eq!(token, 3); // argmax at index 3
    }

    #[test]
    fn test_sample_temperature() {
        // With high temperature, distribution should be more spread
        let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0];

        // Greedy always picks index 0
        assert_eq!(sample(&logits, 0.0, 0), 0);

        // With very high temperature, other tokens should occasionally appear
        // Run many samples and check we get at least some diversity
        let mut counts = [0u32; 5];
        for _ in 0..1000 {
            let token = sample(&logits, 100.0, 0) as usize;
            if token < 5 {
                counts[token] += 1;
            }
        }
        // With temperature=100, logits are [0.1, 0, 0, 0, 0] → nearly uniform
        // All bins should get some samples
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > 0, "Token {i} never sampled at high temperature");
        }
    }

    #[test]
    fn test_input_length_validation() {
        let config = small_config();
        let mut engine = AneInferenceEngine::new(config.clone()).unwrap();

        // Allocate minimal weight data so engine is in a valid state
        let d = config.dim;
        let h = config.hidden_dim;
        let nl = config.n_layers;
        let v = config.vocab_size;
        let kv_dim = config.n_kv_heads * (d / config.n_heads);
        let total =
            v * d + nl * (d + d * d + kv_dim * d * 2 + d * d + d + h * d + d * h + h * d) + d;
        let weights = vec![0.0f32; total];
        engine.load_weights_flat(&weights);

        // Without compiled kernels, forward should fail with "not compiled"
        let result = engine.forward(&[1, 2, 3]);
        assert!(result.is_err());

        // Empty input should fail
        let result_empty = engine.forward(&[]);
        assert!(result_empty.is_err());

        // Input exceeding max_seq_len should fail
        let too_long: Vec<u32> = (0..=config.max_seq_len as u32).collect();
        let result_long = engine.forward(&too_long);
        assert!(result_long.is_err());
    }

    #[test]
    fn test_sample_greedy_known_logits() {
        // Deterministic: argmax on known distribution
        let logits = vec![-1.0, -2.0, 5.0, -3.0, 4.0, 0.0];
        assert_eq!(sample_greedy(&logits), 2);

        let logits2 = vec![0.0; 100];
        // All equal → should return index 0 (first max)
        assert_eq!(sample_greedy(&logits2), 0);
    }

    #[test]
    fn test_sample_top_k() {
        // With top_k=2, only the top 2 tokens should ever be sampled
        let logits = vec![10.0, 5.0, 0.0, 0.0, 0.0];

        let mut saw_outside_topk = false;
        for _ in 0..500 {
            let token = sample(&logits, 1.0, 2);
            if token > 1 {
                saw_outside_topk = true;
                break;
            }
        }
        assert!(
            !saw_outside_topk,
            "top_k=2 should never sample tokens outside top 2"
        );
    }

    #[test]
    fn test_rmsnorm_vec() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let mut out = vec![0.0f32; 4];
        rmsnorm_vec(&mut out, &x, &w, 4);

        let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-5f32).sqrt();
        assert!((out[0] - 1.0 / rms).abs() < 1e-4);
        assert!((out[3] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_kv_cache_pool_creation() {
        let pool = KvCachePool::new(2, 256, 128);
        assert_eq!(pool.layers.len(), 2);
        assert_eq!(pool.pos, 0);
        assert_eq!(pool.max_seq_len, 128);
        assert_eq!(pool.layers[0].k.len(), 256 * 128);
    }

    #[test]
    fn test_lora_merge_before_compile() {
        let config = small_config();
        let mut engine = AneInferenceEngine::new(config).unwrap();
        assert!(!engine.compiled);

        // After compile, LoRA should be rejected
        // (We can't actually compile without ANE, but we can test the flag)
        engine.compiled = true;
        let result = engine.load_lora_adapter(Path::new("/nonexistent"));
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("before compile_kernels"));
    }

    #[test]
    fn test_safetensors_name_mapping() {
        // Verify that our module name → weight field mapping is correct
        let modules = lora_target_modules(768, 256, 2048);
        assert_eq!(modules.len(), 7);
        assert_eq!(modules[0], ("q_proj", "wq", 768, 768));
        assert_eq!(modules[1], ("k_proj", "wk", 256, 768));
        assert_eq!(modules[2], ("v_proj", "wv", 256, 768));
        assert_eq!(modules[3], ("o_proj", "wo", 768, 768));
        assert_eq!(modules[4], ("gate_proj", "w1", 2048, 768));
        assert_eq!(modules[5], ("down_proj", "w2", 768, 2048));
        assert_eq!(modules[6], ("up_proj", "w3", 2048, 768));
    }

    #[test]
    fn test_lora_merge_identity() {
        // Test that scale * B @ A adds correctly to weights
        let dim = 4;
        let rank = 2;
        let scale = 1.0f32;

        // Identity-like merge: B = I_rows, A = I_cols (cropped)
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // [rank=2, cols=4]
        let b = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // [rows=4, rank=2]

        let mut target = vec![0.0f32; dim * dim];
        let mut ba = vec![0.0f32; dim * dim];
        accelerate::gemm(&b, &a, &mut ba, dim, dim, rank, scale, 0.0, false, false);
        for i in 0..dim * dim {
            target[i] += ba[i];
        }

        // B @ A should give a 4x4 matrix with 1s at (0,0) and (1,1)
        assert!((target[0] - 1.0).abs() < 1e-6); // (0,0)
        assert!((target[5] - 1.0).abs() < 1e-6); // (1,1)
        assert!((target[1] - 0.0).abs() < 1e-6); // off-diagonal
    }
}
