//! ANE inference engine for autoregressive generation.
//!
//! Provides forward-only ANE kernels with CPU-side embedding, RMSNorm,
//! sampling, and autoregressive generation. Uses lean forward kernels
//! (no concat taps) for ~6x smaller IOSurface transfers vs training.
//!
//! # Known Limitations (v0.2, deferred to v0.3)
//!
//! - No KV cache — full-sequence recomputation per token, O(n² × L)
//! - Fixed sequence length — compiled for `max_seq_len`, shorter inputs zero-padded
//! - Flat weight format only — requires `model.bin`
//! - No GQA/MQA — assumes `n_kv_heads == n_heads`
//! - No LoRA fusion — adapters must be merged before ANE inference

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
}

impl Default for AneInferenceConfig {
    fn default() -> Self {
        Self {
            dim: 768,
            hidden_dim: 2048,
            n_heads: 12,
            n_layers: 12,
            vocab_size: 32000,
            max_seq_len: 256,
            max_compiles: 100,
            temperature: 0.0,
            top_k: 0,
            max_tokens: 128,
            eos_token_id: None,
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
}

/// Per-layer compiled ANE kernels (inference-only, no backward).
struct InferenceLayerKernels {
    fwd_attn: AneModel,
    fwd_ffn: AneModel,
}

/// IOSurface pool reused across layers and generation steps.
struct IoSurfacePool {
    input: IoSurface,
    output: IoSurface,
}

/// ANE inference engine for autoregressive text generation.
///
/// Uses forward-only ANE kernels (no concat taps) with CPU-side
/// embedding, residual adds, final RMSNorm, and sampling.
pub struct AneInferenceEngine {
    config: AneInferenceConfig,
    kernel_config: TransformerKernelConfig,
    layer_weights: Vec<InferenceLayerWeights>,
    layer_kernels: Option<Vec<InferenceLayerKernels>>,
    embed_weights: Vec<f32>,
    rms_final: Vec<f32>,
    budget: CompileBudget,
    io_pool: Option<IoSurfacePool>,
}

impl AneInferenceEngine {
    /// Create a new ANE inference engine.
    pub fn new(config: AneInferenceConfig) -> Self {
        let d = config.dim;
        let h = config.hidden_dim;
        let nl = config.n_layers;
        let v = config.vocab_size;
        let hd = d / config.n_heads;

        let kernel_config = TransformerKernelConfig {
            dim: d,
            hidden_dim: h,
            n_heads: config.n_heads,
            head_dim: hd,
            seq_len: config.max_seq_len,
        };

        let mut layer_weights = Vec::with_capacity(nl);
        for _ in 0..nl {
            layer_weights.push(InferenceLayerWeights {
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
        }

        // 2 kernels per layer (fwd_attn + fwd_ffn)
        let budget = CompileBudget::new(config.max_compiles, 2 * nl);

        Self {
            config,
            kernel_config,
            layer_weights,
            layer_kernels: None,
            embed_weights: vec![0.0; v * d],
            rms_final: vec![0.0; d],
            budget,
            io_pool: None,
        }
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

    /// Compile forward-only kernels for all layers.
    ///
    /// Uses 2 compilations per layer (fwd_attn + fwd_ffn).
    pub fn compile_kernels(&mut self) -> Result<()> {
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
        let mut layer_kernels = Vec::with_capacity(self.config.n_layers);

        for l in 0..self.config.n_layers {
            let lw = &self.layer_weights[l];

            // Forward attention (inference — no taps)
            let fwd_attn_out =
                kernel::gen_sdpa_fwd(cfg, &lw.rms_att, &lw.wq, &lw.wk, &lw.wv, &lw.wo);
            let fwd_attn = rt.compile(
                fwd_attn_out.mil_text.as_bytes(),
                Some(&fwd_attn_out.weights),
            )?;
            self.budget.record_compile();

            // Forward FFN (inference — no taps)
            let fwd_ffn_out = kernel::gen_ffn_fwd(cfg, &lw.rms_ffn, &lw.w1, &lw.w3, &lw.w2);
            let fwd_ffn =
                rt.compile(fwd_ffn_out.mil_text.as_bytes(), Some(&fwd_ffn_out.weights))?;
            self.budget.record_compile();

            layer_kernels.push(InferenceLayerKernels { fwd_attn, fwd_ffn });
        }

        self.layer_kernels = Some(layer_kernels);

        // Allocate IOSurface pool (both input and output are [D, S] fp16)
        let surface_bytes = d * s * 2;
        self.io_pool = Some(IoSurfacePool {
            input: IoSurface::new(surface_bytes)?,
            output: IoSurface::new(surface_bytes)?,
        });

        Ok(())
    }

    /// Run a forward pass and return logits `[V, S]` (channel-first).
    ///
    /// `token_ids` are padded/truncated to `max_seq_len`. Positions beyond
    /// input length are masked by the causal mask compiled into the kernels.
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
        let mut padded_tokens = vec![0u16; s];
        for (i, &tid) in token_ids.iter().enumerate() {
            padded_tokens[i] = tid as u16;
        }

        let mut x = vec![0.0f32; d * s];
        accelerate::embed_lookup(&mut x, &self.embed_weights, &padded_tokens, d, s);

        // Per-layer forward
        let mut o_out = vec![0.0f32; d * s];
        let mut x2 = vec![0.0f32; d * s];
        let mut ffn_out = vec![0.0f32; d * s];

        for lk in kernels {
            // Attention: write x → ANE fwd_attn → read o_out
            io_pool.input.write_f32_as_fp16(&x, d, s);
            lk.fwd_attn
                .evaluate(&[io_pool.input.as_ptr()], &[io_pool.output.as_ptr()])?;
            io_pool.output.read_fp16_as_f32(&mut o_out, 0, d, s);

            // Residual: x2 = x + o_out
            accelerate::vadd(&x, &o_out, &mut x2);

            // FFN: write x2 → ANE fwd_ffn → read ffn_out
            io_pool.input.write_f32_as_fp16(&x2, d, s);
            lk.fwd_ffn
                .evaluate(&[io_pool.input.as_ptr()], &[io_pool.output.as_ptr()])?;
            io_pool.output.read_fp16_as_f32(&mut ffn_out, 0, d, s);

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
            n_layers: 2,
            vocab_size: 100,
            max_seq_len: 16,
            max_compiles: 100,
            temperature: 0.0,
            top_k: 0,
            max_tokens: 32,
            eos_token_id: None,
        }
    }

    #[test]
    fn test_inference_engine_creation() {
        let config = small_config();
        let engine = AneInferenceEngine::new(config);
        assert_eq!(engine.config().dim, 64);
        assert_eq!(engine.config().n_layers, 2);
        assert!(engine.layer_kernels.is_none());
        assert!(engine.io_pool.is_none());
    }

    #[test]
    fn test_inference_engine_budget() {
        let config = small_config();
        let engine = AneInferenceEngine::new(config);
        // 2 kernels per layer * 2 layers = 4 per compile batch
        assert_eq!(engine.budget().kernels_per_batch(), 4);
        assert!(engine.budget().can_compile_batch());
    }

    #[test]
    fn test_load_weights_flat() {
        let config = small_config();
        let mut engine = AneInferenceEngine::new(config.clone());

        // Calculate expected total weight count
        let d = config.dim;
        let h = config.hidden_dim;
        let nl = config.n_layers;
        let v = config.vocab_size;

        let total = v * d // embed
            + nl * (d + d * d * 4 + d + h * d + d * h + h * d) // per-layer
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
        let mut engine = AneInferenceEngine::new(config.clone());

        // Allocate minimal weight data so engine is in a valid state
        let d = config.dim;
        let h = config.hidden_dim;
        let nl = config.n_layers;
        let v = config.vocab_size;
        let total = v * d + nl * (d + d * d * 4 + d + h * d + d * h + h * d) + d;
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
}
