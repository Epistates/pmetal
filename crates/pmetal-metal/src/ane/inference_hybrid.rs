//! Qwen3.5 hybrid CPU GEMV decode engine.
//!
//! Implements CPU-only inference for Qwen3.5 hybrid architectures that combine
//! Gated Delta Net (GDN) linear attention with full attention layers. All
//! matrix-vector operations use `cblas_sgemv` via Accelerate.framework for
//! zero GPU kernel-launch overhead.
//!
//! # Architecture
//!
//! Qwen3.5 alternates between:
//! - **GDN layers** (75%): O(1) per-token decode via linear recurrence
//! - **Full attention layers** (25%): Gated output + partial RoPE + GQA
//!
//! The CPU decode path eliminates the ~144 Metal kernel launches per token
//! that dominate the GPU path at batch=1, achieving 3x+ throughput improvement.
//!
//! # Usage
//!
//! ```ignore
//! let mut engine = Qwen3NextInferenceEngine::new(config)?;
//! engine.load_weights_safetensors(model_dir)?;
//! let tokens = engine.generate_cached(&input_ids)?;
//! ```

use std::path::Path;

use crate::accelerate;
use crate::ane::inference::sample;
use crate::error::{MetalError, Result};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Qwen3.5 hybrid CPU inference.
#[derive(Debug, Clone)]
pub struct Qwen3NextInferenceConfig {
    /// Hidden dimension (e.g., 1024 for 0.8B).
    pub dim: usize,
    /// FFN intermediate size (e.g., 3584 for 0.8B).
    pub hidden_dim: usize,
    /// Number of full-attention heads.
    pub n_heads: usize,
    /// Number of KV heads for GQA in full attention.
    pub n_kv_heads: usize,
    /// Per-head dimension for full attention (e.g., 256).
    pub head_dim: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length for KV cache.
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Fraction of head_dim to apply RoPE to (default 0.25).
    pub partial_rotary_factor: f32,

    // --- GDN parameters ---
    /// Number of value heads for GDN layers.
    pub num_v_heads: usize,
    /// Number of key heads for GDN layers.
    pub num_k_heads: usize,
    /// Key head dimension for GDN.
    pub head_k_dim: usize,
    /// Value head dimension for GDN.
    pub head_v_dim: usize,
    /// Conv1d kernel size for GDN.
    pub conv_kernel_size: usize,

    // --- Hybrid layer control ---
    /// Every Nth layer is a full attention layer (default 4).
    pub full_attention_interval: usize,
    /// Optional explicit layer types.
    pub layer_types: Option<Vec<String>>,

    // --- Generation ---
    /// Whether to tie embed and lm_head weights.
    pub tie_word_embeddings: bool,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// EOS token ID for early stopping.
    pub eos_token_id: Option<u32>,
}

impl Qwen3NextInferenceConfig {
    /// Derived: key_dim = num_k_heads * head_k_dim.
    pub fn key_dim(&self) -> usize {
        self.num_k_heads * self.head_k_dim
    }
    /// Derived: value_dim = num_v_heads * head_v_dim.
    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.head_v_dim
    }
    /// Derived: conv_dim = key_dim * 2 + value_dim.
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
    /// Derived: q_dim for full attention = n_heads * head_dim.
    pub fn attn_q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    /// Derived: kv_dim for full attention = n_kv_heads * head_dim.
    pub fn attn_kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    /// Derived: number of GQA groups for full attention.
    pub fn attn_n_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
    /// Derived: rope_dims (number of dimensions that get RoPE).
    pub fn rope_dims(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    /// Check if a layer index is a GDN (linear attention) layer.
    pub fn is_gdn_layer(&self, layer_idx: usize) -> bool {
        if let Some(ref types) = self.layer_types {
            if layer_idx < types.len() {
                return types[layer_idx] == "linear_attention";
            }
        }
        // Default: every full_attention_interval-th layer is full attention
        ((layer_idx + 1) % self.full_attention_interval) != 0
    }
}

// ============================================================================
// Weight storage
// ============================================================================

/// Weights for a GDN (linear attention) layer.
struct GdnLayerWeights {
    /// Input RMSNorm weights [dim] — (1+w) applied at load time.
    rms_att: Vec<f32>,
    /// FFN RMSNorm weights [dim] — (1+w) applied at load time.
    rms_ffn: Vec<f32>,
    /// QKV projection [conv_dim, dim].
    in_proj_qkv: Vec<f32>,
    /// Z projection [value_dim, dim].
    in_proj_z: Vec<f32>,
    /// Beta projection [num_v_heads, dim].
    in_proj_b: Vec<f32>,
    /// Alpha projection [num_v_heads, dim].
    in_proj_a: Vec<f32>,
    /// Depthwise conv1d weights [conv_dim, kernel] (squeezed from [conv_dim, kernel, 1]).
    conv1d_w: Vec<f32>,
    /// Learnable dt bias [num_v_heads].
    dt_bias: Vec<f32>,
    /// Log decay rates [num_v_heads].
    a_log: Vec<f32>,
    /// GDN gated norm weights [head_v_dim] — NOT (1+w).
    norm_weight: Vec<f32>,
    /// Output projection [dim, value_dim].
    out_proj: Vec<f32>,
    /// Gate projection (SwiGLU) [hidden_dim, dim].
    w1: Vec<f32>,
    /// Down projection [dim, hidden_dim].
    w2: Vec<f32>,
    /// Up projection [hidden_dim, dim].
    w3: Vec<f32>,
}

/// Weights for a full attention layer (gated output + partial RoPE).
struct AttentionLayerWeights {
    /// Input RMSNorm weights [dim] — (1+w) applied at load time.
    rms_att: Vec<f32>,
    /// FFN RMSNorm weights [dim] — (1+w) applied at load time.
    rms_ffn: Vec<f32>,
    /// Q projection (2x width for gate) [n_heads*head_dim*2, dim].
    wq: Vec<f32>,
    /// K projection [kv_dim, dim].
    wk: Vec<f32>,
    /// V projection [kv_dim, dim].
    wv: Vec<f32>,
    /// Output projection [dim, n_heads*head_dim].
    wo: Vec<f32>,
    /// Per-head Q norm weights [head_dim] — (1+w) applied at load time.
    q_norm: Vec<f32>,
    /// Per-head K norm weights [head_dim] — (1+w) applied at load time.
    k_norm: Vec<f32>,
    /// Gate projection (SwiGLU) [hidden_dim, dim].
    w1: Vec<f32>,
    /// Down projection [dim, hidden_dim].
    w2: Vec<f32>,
    /// Up projection [hidden_dim, dim].
    w3: Vec<f32>,
}

/// Per-layer weights (enum over layer type).
enum HybridLayerWeights {
    Gdn(GdnLayerWeights),
    Attention(AttentionLayerWeights),
}

// ============================================================================
// State management
// ============================================================================

/// GDN layer decode state.
struct GdnLayerState {
    /// Conv1d state [conv_kernel-1, conv_dim], row-major [time, channel].
    conv_state: Vec<f32>,
    /// SSM recurrence state [num_v_heads * head_v_dim * head_k_dim], row-major.
    /// Layout: state[h * head_v_dim * head_k_dim + i * head_k_dim + j]
    ssm_state: Vec<f32>,
}

/// Attention layer KV cache.
struct AttentionLayerCache {
    /// Key cache [kv_dim, max_seq_len], channel-first f32.
    k: Vec<f32>,
    /// Value cache [kv_dim, max_seq_len], channel-first f32.
    v: Vec<f32>,
}

/// Per-layer state (enum over layer type).
enum HybridLayerState {
    Gdn(GdnLayerState),
    Attention(AttentionLayerCache),
}

/// Combined cache/state pool for all layers.
struct HybridStatePool {
    layers: Vec<HybridLayerState>,
    /// Current decode position (for attention KV cache).
    pos: usize,
    _max_seq_len: usize,
}

impl HybridStatePool {
    fn new(config: &Qwen3NextInferenceConfig) -> Self {
        let mut layers = Vec::with_capacity(config.n_layers);
        let conv_state_len = (config.conv_kernel_size - 1) * config.conv_dim();
        let ssm_state_len = config.num_v_heads * config.head_v_dim * config.head_k_dim;
        let kv_dim = config.attn_kv_dim();

        for i in 0..config.n_layers {
            if config.is_gdn_layer(i) {
                layers.push(HybridLayerState::Gdn(GdnLayerState {
                    conv_state: vec![0.0f32; conv_state_len],
                    ssm_state: vec![0.0f32; ssm_state_len],
                }));
            } else {
                layers.push(HybridLayerState::Attention(AttentionLayerCache {
                    k: vec![0.0f32; kv_dim * config.max_seq_len],
                    v: vec![0.0f32; kv_dim * config.max_seq_len],
                }));
            }
        }

        Self {
            layers,
            pos: 0,
            _max_seq_len: config.max_seq_len,
        }
    }
}

// ============================================================================
// Pre-allocated scratch buffers
// ============================================================================

/// Reusable scratch buffers for decode steps (avoids per-step allocation).
struct ScratchBuffers {
    xnorm: Vec<f32>,
    x2: Vec<f32>,
    h1: Vec<f32>,
    h3: Vec<f32>,
    ffn_out: Vec<f32>,

    // GDN-specific
    qkv_proj: Vec<f32>,
    z_proj: Vec<f32>,
    b_proj: Vec<f32>,
    a_proj: Vec<f32>,
    conv_out: Vec<f32>,
    gdn_y: Vec<f32>,
    gdn_normed: Vec<f32>,
    gdn_out: Vec<f32>,
    kv_mem: Vec<f32>,
    delta: Vec<f32>,
    y_head: Vec<f32>,

    // Attention-specific
    q_gate_proj: Vec<f32>,
    q: Vec<f32>,
    gate: Vec<f32>,
    k_new: Vec<f32>,
    v_new: Vec<f32>,
    q_normed: Vec<f32>,
    k_normed: Vec<f32>,
    attn_out: Vec<f32>,
    wo_out: Vec<f32>,
    scores: Vec<f32>,
}

impl ScratchBuffers {
    fn new(config: &Qwen3NextInferenceConfig) -> Self {
        let d = config.dim;
        let hd = config.hidden_dim;
        let conv_dim = config.conv_dim();
        let _key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let attn_q_dim = config.attn_q_dim();
        let attn_kv_dim = config.attn_kv_dim();

        Self {
            xnorm: vec![0.0; d],
            x2: vec![0.0; d],
            h1: vec![0.0; hd],
            h3: vec![0.0; hd],
            ffn_out: vec![0.0; d],

            qkv_proj: vec![0.0; conv_dim],
            z_proj: vec![0.0; value_dim],
            b_proj: vec![0.0; config.num_v_heads],
            a_proj: vec![0.0; config.num_v_heads],
            conv_out: vec![0.0; conv_dim],
            gdn_y: vec![0.0; value_dim],
            gdn_normed: vec![0.0; value_dim],
            gdn_out: vec![0.0; d],
            kv_mem: vec![0.0; config.head_v_dim],
            delta: vec![0.0; config.head_v_dim],
            y_head: vec![0.0; config.head_v_dim],

            q_gate_proj: vec![0.0; attn_q_dim * 2],
            q: vec![0.0; attn_q_dim],
            gate: vec![0.0; attn_q_dim],
            k_new: vec![0.0; attn_kv_dim],
            v_new: vec![0.0; attn_kv_dim],
            q_normed: vec![0.0; attn_q_dim],
            k_normed: vec![0.0; attn_kv_dim],
            attn_out: vec![0.0; attn_q_dim],
            wo_out: vec![0.0; d],
            scores: vec![0.0; config.max_seq_len],
        }
    }
}

// ============================================================================
// Engine
// ============================================================================

/// Qwen3.5 hybrid CPU inference engine.
///
/// Processes all layers on CPU using cblas_sgemv for matrix-vector multiplies.
/// GDN layers use O(1) linear recurrence; attention layers use standard
/// KV-cached multi-head attention with GQA.
pub struct Qwen3NextInferenceEngine {
    config: Qwen3NextInferenceConfig,
    layer_weights: Vec<HybridLayerWeights>,
    embed_weights: Vec<f32>,
    rms_final: Vec<f32>,
    /// Optional separate lm_head weights. None when tie_word_embeddings=true.
    lm_head_weights: Option<Vec<f32>>,
    scratch: ScratchBuffers,
}

impl Qwen3NextInferenceEngine {
    /// Create a new inference engine. Weights must be loaded separately.
    pub fn new(config: Qwen3NextInferenceConfig) -> Result<Self> {
        // Validate config
        if config.n_heads == 0 || config.n_kv_heads == 0 {
            return Err(MetalError::InvalidConfig(
                "n_heads and n_kv_heads must be > 0".into(),
            ));
        }
        if config.n_heads % config.n_kv_heads != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "n_heads ({}) must be divisible by n_kv_heads ({})",
                config.n_heads, config.n_kv_heads
            )));
        }

        let scratch = ScratchBuffers::new(&config);

        // Pre-allocate empty layer weights
        let mut layer_weights = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            if config.is_gdn_layer(i) {
                layer_weights.push(HybridLayerWeights::Gdn(GdnLayerWeights {
                    rms_att: vec![0.0; config.dim],
                    rms_ffn: vec![0.0; config.dim],
                    in_proj_qkv: vec![0.0; config.conv_dim() * config.dim],
                    in_proj_z: vec![0.0; config.value_dim() * config.dim],
                    in_proj_b: vec![0.0; config.num_v_heads * config.dim],
                    in_proj_a: vec![0.0; config.num_v_heads * config.dim],
                    conv1d_w: vec![0.0; config.conv_dim() * config.conv_kernel_size],
                    dt_bias: vec![0.0; config.num_v_heads],
                    a_log: vec![0.0; config.num_v_heads],
                    norm_weight: vec![0.0; config.head_v_dim],
                    out_proj: vec![0.0; config.dim * config.value_dim()],
                    w1: vec![0.0; config.hidden_dim * config.dim],
                    w2: vec![0.0; config.dim * config.hidden_dim],
                    w3: vec![0.0; config.hidden_dim * config.dim],
                }));
            } else {
                let q_dim = config.attn_q_dim();
                let kv_dim = config.attn_kv_dim();
                layer_weights.push(HybridLayerWeights::Attention(AttentionLayerWeights {
                    rms_att: vec![0.0; config.dim],
                    rms_ffn: vec![0.0; config.dim],
                    wq: vec![0.0; q_dim * 2 * config.dim],
                    wk: vec![0.0; kv_dim * config.dim],
                    wv: vec![0.0; kv_dim * config.dim],
                    wo: vec![0.0; config.dim * q_dim],
                    q_norm: vec![0.0; config.head_dim],
                    k_norm: vec![0.0; config.head_dim],
                    w1: vec![0.0; config.hidden_dim * config.dim],
                    w2: vec![0.0; config.dim * config.hidden_dim],
                    w3: vec![0.0; config.hidden_dim * config.dim],
                }));
            }
        }

        Ok(Self {
            config,
            layer_weights,
            embed_weights: Vec::new(),
            rms_final: Vec::new(),
            lm_head_weights: None,
            scratch,
        })
    }

    /// Get the engine configuration.
    pub fn config(&self) -> &Qwen3NextInferenceConfig {
        &self.config
    }

    // ========================================================================
    // Generation
    // ========================================================================

    /// Generate tokens using CPU-only decode (sequential prefill + cached decode).
    pub fn generate_cached(&mut self, input_ids: &[u32]) -> Result<Vec<u32>> {
        self.generate_cached_streaming(input_ids, |_| true)
    }

    /// Generate tokens with streaming callback.
    pub fn generate_cached_streaming<F>(
        &mut self,
        input_ids: &[u32],
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        let s = self.config.max_seq_len;
        let _v = self.config.vocab_size;

        if input_ids.is_empty() {
            return Err(MetalError::InvalidConfig("Input must not be empty".into()));
        }
        if input_ids.len() >= s {
            return Err(MetalError::InvalidConfig(format!(
                "Input length {} exceeds max_seq_len {}",
                input_ids.len(),
                s
            )));
        }

        let mut state = HybridStatePool::new(&self.config);

        // CPU sequential prefill: process each prompt token through decode_step
        // For 0.8B with 24 layers, this is fast (~2ms for 100 tokens)
        for &tok in &input_ids[..input_ids.len() - 1] {
            let _ = self.decode_step(tok, &mut state)?;
        }

        // Process last prompt token to get first generation logits
        let logits = self.decode_step(*input_ids.last().unwrap(), &mut state)?;
        let next = sample(&logits, self.config.temperature, self.config.top_k);

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

        // Decode loop
        for _ in 1..self.config.max_tokens {
            if state.pos >= s {
                break;
            }

            let logits_vec = self.decode_step(*sequence.last().unwrap(), &mut state)?;
            let next_tok = sample(&logits_vec, self.config.temperature, self.config.top_k);

            if let Some(eos) = self.config.eos_token_id {
                if next_tok == eos {
                    sequence.push(next_tok);
                    break;
                }
            }
            if !on_token(next_tok) {
                break;
            }
            sequence.push(next_tok);
        }

        Ok(sequence)
    }

    // ========================================================================
    // Decode step
    // ========================================================================

    /// Single-token decode step. Returns logits [vocab_size].
    #[allow(clippy::needless_range_loop)]
    fn decode_step(&mut self, token_id: u32, state: &mut HybridStatePool) -> Result<Vec<f32>> {
        let d = self.config.dim;
        let v = self.config.vocab_size;

        if token_id as usize >= v {
            return Err(MetalError::InvalidConfig(format!(
                "Token ID {} exceeds vocab size {}",
                token_id, v
            )));
        }

        // Embed single token → x [dim]
        let mut x = vec![0.0f32; d];
        for i in 0..d {
            x[i] = self.embed_weights[token_id as usize * d + i];
        }

        let pos = state.pos;
        let eps = self.config.rms_norm_eps;

        // Process each layer
        for layer_idx in 0..self.config.n_layers {
            match (&self.layer_weights[layer_idx], &mut state.layers[layer_idx]) {
                (HybridLayerWeights::Gdn(lw), HybridLayerState::Gdn(ls)) => {
                    decode_gdn_layer(&self.config, &mut self.scratch, &mut x, lw, ls, eps);
                }
                (HybridLayerWeights::Attention(lw), HybridLayerState::Attention(lc)) => {
                    decode_attention_layer(
                        &self.config,
                        &mut self.scratch,
                        &mut x,
                        lw,
                        lc,
                        pos,
                        eps,
                    );
                }
                _ => {
                    return Err(MetalError::InvalidConfig(format!(
                        "Layer {} weight/state type mismatch",
                        layer_idx
                    )));
                }
            }
        }

        state.pos = pos + 1;

        // Final RMSNorm
        let mut x_final = vec![0.0f32; d];
        rmsnorm_vec(&mut x_final, &x, &self.rms_final, d, eps);

        // Logits: W @ x_final → [vocab]
        let mut logits = vec![0.0f32; v];
        let lm_head = self.lm_head_weights.as_ref().unwrap_or(&self.embed_weights);
        accelerate::gemm(
            lm_head,
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
    // Weight loading
    // ========================================================================

    /// Load weights from SafeTensors files (HuggingFace format).
    pub fn load_weights_safetensors(&mut self, path: &Path) -> Result<()> {
        use memmap2::Mmap;
        use safetensors::SafeTensors;

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
                unique_files.into_iter().map(|f| path.join(f)).collect()
            } else {
                vec![path.join("model.safetensors")]
            }
        };

        // Detect (1+w) norm condition: check if conv1d weights are unsanitized
        let mut should_shift_norms = false;
        for file_path in &files {
            let file = std::fs::File::open(file_path).map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to open {}: {e}", file_path.display()))
            })?;
            let mmap = unsafe {
                Mmap::map(&file).map_err(|e| {
                    MetalError::InvalidConfig(format!(
                        "Failed to mmap {}: {e}",
                        file_path.display()
                    ))
                })?
            };
            let st = SafeTensors::deserialize(&mmap).map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to parse safetensors: {e}"))
            })?;

            for (name, info) in st.tensors() {
                if name.contains("conv1d.weight") {
                    let shape = info.shape();
                    if shape.len() == 3 && shape[2] != 1 {
                        should_shift_norms = true;
                    }
                }
                if name.contains("mtp.") {
                    should_shift_norms = true;
                }
            }
        }

        tracing::info!(
            "(1+w) norm shift: {}",
            if should_shift_norms { "yes" } else { "no" }
        );

        // Norm suffixes that get (1+w) offset
        let norm_1pw_suffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ];

        let conv_dim = self.config.conv_dim();
        let conv_kernel = self.config.conv_kernel_size;

        // Load tensors from all files
        for file_path in &files {
            let file = std::fs::File::open(file_path).map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to open {}: {e}", file_path.display()))
            })?;
            let mmap = unsafe {
                Mmap::map(&file).map_err(|e| {
                    MetalError::InvalidConfig(format!(
                        "Failed to mmap {}: {e}",
                        file_path.display()
                    ))
                })?
            };
            let st = SafeTensors::deserialize(&mmap).map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to parse safetensors: {e}"))
            })?;

            for (raw_name, tensor) in st.tensors() {
                // Canonicalize name: strip VLM prefix, A_log → a_log
                let mut name = raw_name.to_string();
                if name.starts_with("model.language_model.") {
                    name = name.replacen("model.language_model.", "model.", 1);
                }
                if name.contains(".A_log") {
                    name = name.replace(".A_log", ".a_log");
                }

                // Skip MTP weights
                if name.contains("mtp.") {
                    continue;
                }

                let mut data = safetensors_to_f32(&tensor)?;

                // Apply (1+w) offset where applicable
                if should_shift_norms && norm_1pw_suffixes.iter().any(|sfx| name.ends_with(sfx)) {
                    for v in &mut data {
                        *v += 1.0;
                    }
                }

                // Route to appropriate weight slot
                if name == "model.embed_tokens.weight" {
                    self.embed_weights = data;
                    continue;
                }
                if name == "model.norm.weight" {
                    self.rms_final = data;
                    continue;
                }
                if name == "lm_head.weight" {
                    self.lm_head_weights = Some(data);
                    continue;
                }

                if !name.starts_with("model.layers.") {
                    continue;
                }

                let after_prefix = &name["model.layers.".len()..];
                let dot_pos = match after_prefix.find('.') {
                    Some(p) => p,
                    None => continue,
                };
                let layer_idx: usize = match after_prefix[..dot_pos].parse() {
                    Ok(i) => i,
                    Err(_) => continue,
                };
                let rest = &after_prefix[dot_pos + 1..];

                if layer_idx >= self.config.n_layers {
                    continue;
                }

                match &mut self.layer_weights[layer_idx] {
                    HybridLayerWeights::Gdn(lw) => {
                        assign_gdn_weight(lw, rest, &data, &tensor, conv_dim, conv_kernel)?;
                    }
                    HybridLayerWeights::Attention(lw) => {
                        assign_attention_weight(lw, rest, &data);
                    }
                }
            }
        }

        // Validate essential weights loaded
        if self.embed_weights.is_empty() {
            return Err(MetalError::InvalidConfig(
                "embed_tokens.weight not found".into(),
            ));
        }
        if self.rms_final.is_empty() {
            return Err(MetalError::InvalidConfig(
                "model.norm.weight not found".into(),
            ));
        }

        tracing::info!(
            "Loaded weights: {} layers ({} GDN + {} attention), vocab={}",
            self.config.n_layers,
            (0..self.config.n_layers)
                .filter(|&i| self.config.is_gdn_layer(i))
                .count(),
            (0..self.config.n_layers)
                .filter(|&i| !self.config.is_gdn_layer(i))
                .count(),
            self.config.vocab_size,
        );

        Ok(())
    }
}

// ============================================================================
// Free-standing decode functions (avoid borrow conflicts with self)
// ============================================================================

/// Process one GDN (linear attention) layer for a single token.
#[allow(clippy::needless_range_loop)]
fn decode_gdn_layer(
    config: &Qwen3NextInferenceConfig,
    s: &mut ScratchBuffers,
    x: &mut [f32],
    lw: &GdnLayerWeights,
    ls: &mut GdnLayerState,
    eps: f32,
) {
    let d = config.dim;
    let conv_dim = config.conv_dim();
    let key_dim = config.key_dim();
    let value_dim = config.value_dim();
    let num_v_heads = config.num_v_heads;
    let num_k_heads = config.num_k_heads;
    let head_k_dim = config.head_k_dim;
    let head_v_dim = config.head_v_dim;
    let kernel = config.conv_kernel_size;
    let hidden_dim = config.hidden_dim;

    // 1. RMSNorm input (1+w weights already applied at load)
    rmsnorm_vec(&mut s.xnorm, x, &lw.rms_att, d, eps);

    // 2. Four projections via gemv
    gemv(&lw.in_proj_qkv, &s.xnorm, &mut s.qkv_proj, conv_dim, d);
    gemv(&lw.in_proj_z, &s.xnorm, &mut s.z_proj, value_dim, d);
    gemv(&lw.in_proj_b, &s.xnorm, &mut s.b_proj, num_v_heads, d);
    gemv(&lw.in_proj_a, &s.xnorm, &mut s.a_proj, num_v_heads, d);

    // 3. Depthwise conv1d with cached state
    for c in 0..conv_dim {
        let mut sum = 0.0f32;
        for j in 0..(kernel - 1) {
            sum += lw.conv1d_w[c * kernel + j] * ls.conv_state[j * conv_dim + c];
        }
        sum += lw.conv1d_w[c * kernel + (kernel - 1)] * s.qkv_proj[c];
        s.conv_out[c] = sum;
    }

    // Update conv state: shift left, append new input
    if kernel > 2 {
        let shift_len = (kernel - 2) * conv_dim;
        ls.conv_state.copy_within(conv_dim..conv_dim + shift_len, 0);
    }
    let last_slot = (kernel - 2) * conv_dim;
    ls.conv_state[last_slot..last_slot + conv_dim].copy_from_slice(&s.qkv_proj[..conv_dim]);

    // 4. SiLU activation on conv output
    accelerate::silu_inplace(&mut s.conv_out[..conv_dim]);

    // 5. Split conv output → Q[key_dim], K[key_dim], V[value_dim]
    // (conv_out layout: [0..key_dim]=Q, [key_dim..2*key_dim]=K, [2*key_dim..conv_dim]=V)

    // 6. Q/K RMSNorm with identity weights + scaling
    let inv_scale = (head_k_dim as f32).powf(-0.5);
    let q_scale = inv_scale * inv_scale;
    let k_scale = inv_scale;

    // Q norm + scale: per k-head (compute RMS first, then write back)
    for h in 0..num_k_heads {
        let off = h * head_k_dim;
        let mut ss = 0.0f32;
        for i in 0..head_k_dim {
            ss += s.conv_out[off + i] * s.conv_out[off + i];
        }
        ss = 1.0 / (ss / head_k_dim as f32 + 1e-6).sqrt();
        for i in 0..head_k_dim {
            s.conv_out[off + i] *= ss * q_scale;
        }
    }

    // K norm + scale: per k-head
    let k_off = key_dim;
    for h in 0..num_k_heads {
        let off = k_off + h * head_k_dim;
        let mut ss = 0.0f32;
        for i in 0..head_k_dim {
            ss += s.conv_out[off + i] * s.conv_out[off + i];
        }
        ss = 1.0 / (ss / head_k_dim as f32 + 1e-6).sqrt();
        for i in 0..head_k_dim {
            s.conv_out[off + i] *= ss * k_scale;
        }
    }

    // 7-8. Compute gating + GDN recurrence per value head
    let state_stride = head_v_dim * head_k_dim;

    for h in 0..num_v_heads {
        let a_biased = s.a_proj[h] + lw.dt_bias[h];
        let sp = softplus(a_biased);
        let decay_rate = lw.a_log[h].exp();
        let g = (-decay_rate * sp).exp();
        let beta = sigmoid(s.b_proj[h]);

        let state_h = &mut ls.ssm_state[h * state_stride..(h + 1) * state_stride];
        let q_h = &s.conv_out[h * head_k_dim..(h + 1) * head_k_dim];
        let k_h = &s.conv_out[key_dim + h * head_k_dim..key_dim + (h + 1) * head_k_dim];
        let v_h = &s.conv_out[2 * key_dim + h * head_v_dim..2 * key_dim + (h + 1) * head_v_dim];

        // Decay
        for val in state_h.iter_mut() {
            *val *= g;
        }

        // kv_mem = state_h @ k_h → [head_v_dim]
        accelerate::gemm(
            state_h,
            k_h,
            &mut s.kv_mem[..head_v_dim],
            head_v_dim,
            1,
            head_k_dim,
            1.0,
            0.0,
            false,
            false,
        );

        // delta = beta * (v_h - kv_mem)
        for i in 0..head_v_dim {
            s.delta[i] = beta * (v_h[i] - s.kv_mem[i]);
        }

        // Rank-1 update: state_h += outer(delta, k_h)
        accelerate::gemm(
            &s.delta[..head_v_dim],
            k_h,
            state_h,
            head_v_dim,
            head_k_dim,
            1,
            1.0,
            1.0,
            false,
            false,
        );

        // Output: y_h = state_h @ q_h → [head_v_dim]
        accelerate::gemm(
            state_h,
            q_h,
            &mut s.y_head[..head_v_dim],
            head_v_dim,
            1,
            head_k_dim,
            1.0,
            0.0,
            false,
            false,
        );

        s.gdn_y[h * head_v_dim..(h + 1) * head_v_dim].copy_from_slice(&s.y_head[..head_v_dim]);
    }

    // 9. Gated RMSNorm: rmsnorm(y, norm_weight) * silu(z) per head
    for h in 0..num_v_heads {
        let off = h * head_v_dim;
        let mut ss = 0.0f32;
        for i in 0..head_v_dim {
            ss += s.gdn_y[off + i] * s.gdn_y[off + i];
        }
        ss = 1.0 / (ss / head_v_dim as f32 + eps).sqrt();

        for i in 0..head_v_dim {
            let normed = lw.norm_weight[i] * s.gdn_y[off + i] * ss;
            let z_val = s.z_proj[off + i];
            let silu_z = z_val / (1.0 + (-z_val).exp());
            s.gdn_normed[off + i] = normed * silu_z;
        }
    }

    // 10. Output projection
    gemv(
        &lw.out_proj,
        &s.gdn_normed[..value_dim],
        &mut s.gdn_out,
        d,
        value_dim,
    );

    // 11. Residual + FFN
    for i in 0..d {
        s.x2[i] = x[i] + s.gdn_out[i];
    }

    rmsnorm_vec(&mut s.xnorm, &s.x2, &lw.rms_ffn, d, eps);
    gemv(&lw.w1, &s.xnorm, &mut s.h1, hidden_dim, d);
    gemv(&lw.w3, &s.xnorm, &mut s.h3, hidden_dim, d);
    accelerate::silu_inplace(&mut s.h1[..hidden_dim]);
    for i in 0..hidden_dim {
        s.h1[i] *= s.h3[i];
    }
    gemv(&lw.w2, &s.h1, &mut s.ffn_out, d, hidden_dim);

    for i in 0..d {
        x[i] = s.x2[i] + s.ffn_out[i];
    }
}

/// Process one full attention layer for a single token.
#[allow(clippy::needless_range_loop)]
fn decode_attention_layer(
    config: &Qwen3NextInferenceConfig,
    s: &mut ScratchBuffers,
    x: &mut [f32],
    lw: &AttentionLayerWeights,
    lc: &mut AttentionLayerCache,
    pos: usize,
    eps: f32,
) {
    let d = config.dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let n_groups = config.attn_n_groups();
    let q_dim = config.attn_q_dim();
    let kv_dim = config.attn_kv_dim();
    let rope_dims = config.rope_dims();
    let hidden_dim = config.hidden_dim;
    let max_seq = config.max_seq_len;
    let seq_through = pos + 1;

    // 1. RMSNorm input
    rmsnorm_vec(&mut s.xnorm, x, &lw.rms_att, d, eps);

    // 2. Q+gate, K, V projections
    gemv(&lw.wq, &s.xnorm, &mut s.q_gate_proj, q_dim * 2, d);
    gemv(&lw.wk, &s.xnorm, &mut s.k_new, kv_dim, d);
    gemv(&lw.wv, &s.xnorm, &mut s.v_new, kv_dim, d);

    // 3. Split Q+gate per head
    for h in 0..n_heads {
        let src_off = h * head_dim * 2;
        let dst_off = h * head_dim;
        s.q[dst_off..dst_off + head_dim]
            .copy_from_slice(&s.q_gate_proj[src_off..src_off + head_dim]);
        s.gate[dst_off..dst_off + head_dim]
            .copy_from_slice(&s.q_gate_proj[src_off + head_dim..src_off + 2 * head_dim]);
    }

    // 4. Per-head QK norm
    rmsnorm_per_head(&mut s.q_normed, &s.q, &lw.q_norm, n_heads, head_dim, eps);
    rmsnorm_per_head(
        &mut s.k_normed,
        &s.k_new,
        &lw.k_norm,
        n_kv_heads,
        head_dim,
        eps,
    );

    // 5. Partial RoPE
    apply_partial_rope_vec(
        &mut s.q_normed,
        n_heads,
        head_dim,
        rope_dims,
        pos,
        config.rope_theta,
    );
    apply_partial_rope_vec(
        &mut s.k_normed,
        n_kv_heads,
        head_dim,
        rope_dims,
        pos,
        config.rope_theta,
    );

    // 6. Store K, V in cache
    for ch in 0..kv_dim {
        lc.k[ch * max_seq + pos] = s.k_normed[ch];
        lc.v[ch * max_seq + pos] = s.v_new[ch];
    }

    // 7. Multi-head attention with GQA
    s.attn_out[..q_dim].fill(0.0);
    let scale = 1.0 / (head_dim as f32).sqrt();

    for head in 0..n_heads {
        let kv_head = head / n_groups;
        let q_off = head * head_dim;
        let kv_off = kv_head * head_dim;

        for t in 0..seq_through {
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += s.q_normed[q_off + i] * lc.k[(kv_off + i) * max_seq + t];
            }
            s.scores[t] = dot * scale;
        }

        let max_s = s.scores[..seq_through]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for sc in &mut s.scores[..seq_through] {
            *sc = (*sc - max_s).exp();
            sum += *sc;
        }
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for sc in &mut s.scores[..seq_through] {
                *sc *= inv_sum;
            }
        }

        for i in 0..head_dim {
            let mut val = 0.0f32;
            for t in 0..seq_through {
                val += s.scores[t] * lc.v[(kv_off + i) * max_seq + t];
            }
            s.attn_out[q_off + i] = val;
        }
    }

    // 8. Gated output: attn_out *= sigmoid(gate)
    for i in 0..q_dim {
        s.attn_out[i] *= sigmoid(s.gate[i]);
    }

    // 9. Output projection
    gemv(&lw.wo, &s.attn_out[..q_dim], &mut s.wo_out, d, q_dim);

    // 10. Residual + FFN
    for i in 0..d {
        s.x2[i] = x[i] + s.wo_out[i];
    }

    rmsnorm_vec(&mut s.xnorm, &s.x2, &lw.rms_ffn, d, eps);
    gemv(&lw.w1, &s.xnorm, &mut s.h1, hidden_dim, d);
    gemv(&lw.w3, &s.xnorm, &mut s.h3, hidden_dim, d);
    accelerate::silu_inplace(&mut s.h1[..hidden_dim]);
    for i in 0..hidden_dim {
        s.h1[i] *= s.h3[i];
    }
    gemv(&lw.w2, &s.h1, &mut s.ffn_out, d, hidden_dim);

    for i in 0..d {
        x[i] = s.x2[i] + s.ffn_out[i];
    }
}

// ============================================================================
// Free-standing weight assignment functions
// ============================================================================

/// Assign a weight tensor to a GDN layer.
fn assign_gdn_weight(
    lw: &mut GdnLayerWeights,
    key: &str,
    data: &[f32],
    tensor: &safetensors::tensor::TensorView<'_>,
    conv_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    match key {
        "input_layernorm.weight" => copy_weight(data, &mut lw.rms_att, data.len()),
        "post_attention_layernorm.weight" => copy_weight(data, &mut lw.rms_ffn, data.len()),
        "linear_attn.in_proj_qkv.weight" => copy_weight(data, &mut lw.in_proj_qkv, data.len()),
        "linear_attn.in_proj_z.weight" => copy_weight(data, &mut lw.in_proj_z, data.len()),
        "linear_attn.in_proj_b.weight" => copy_weight(data, &mut lw.in_proj_b, data.len()),
        "linear_attn.in_proj_a.weight" => copy_weight(data, &mut lw.in_proj_a, data.len()),
        "linear_attn.out_proj.weight" => copy_weight(data, &mut lw.out_proj, data.len()),
        "linear_attn.norm.weight" => copy_weight(data, &mut lw.norm_weight, data.len()),
        "linear_attn.dt_bias" => copy_weight(data, &mut lw.dt_bias, data.len()),
        "linear_attn.a_log" => copy_weight(data, &mut lw.a_log, data.len()),
        "linear_attn.conv1d.weight" => {
            let shape = tensor.shape();
            if shape.len() >= 2 {
                copy_weight(data, &mut lw.conv1d_w, conv_dim * conv_kernel);
            } else {
                return Err(MetalError::InvalidConfig(format!(
                    "Unexpected conv1d weight ndim: {}",
                    shape.len()
                )));
            }
        }
        "mlp.gate_proj.weight" => copy_weight(data, &mut lw.w1, data.len()),
        "mlp.down_proj.weight" => copy_weight(data, &mut lw.w2, data.len()),
        "mlp.up_proj.weight" => copy_weight(data, &mut lw.w3, data.len()),
        _ => {}
    }
    Ok(())
}

/// Assign a weight tensor to an attention layer.
fn assign_attention_weight(lw: &mut AttentionLayerWeights, key: &str, data: &[f32]) {
    match key {
        "input_layernorm.weight" => copy_weight(data, &mut lw.rms_att, data.len()),
        "post_attention_layernorm.weight" => copy_weight(data, &mut lw.rms_ffn, data.len()),
        "self_attn.q_proj.weight" => copy_weight(data, &mut lw.wq, data.len()),
        "self_attn.k_proj.weight" => copy_weight(data, &mut lw.wk, data.len()),
        "self_attn.v_proj.weight" => copy_weight(data, &mut lw.wv, data.len()),
        "self_attn.o_proj.weight" => copy_weight(data, &mut lw.wo, data.len()),
        "self_attn.q_norm.weight" => copy_weight(data, &mut lw.q_norm, data.len()),
        "self_attn.k_norm.weight" => copy_weight(data, &mut lw.k_norm, data.len()),
        "mlp.gate_proj.weight" => copy_weight(data, &mut lw.w1, data.len()),
        "mlp.down_proj.weight" => copy_weight(data, &mut lw.w2, data.len()),
        "mlp.up_proj.weight" => copy_weight(data, &mut lw.w3, data.len()),
        _ => {}
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Matrix-vector multiply: out = W @ x, where W is [rows, cols] row-major.
#[inline]
fn gemv(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    accelerate::gemm(w, x, out, rows, 1, cols, 1.0, 0.0, false, false);
}

/// Copy weight data, clamping to target size.
fn copy_weight(src: &[f32], dst: &mut [f32], expected: usize) {
    let len = src.len().min(dst.len()).min(expected);
    dst[..len].copy_from_slice(&src[..len]);
}

/// RMSNorm for a single vector (seq=1) with configurable epsilon.
#[allow(clippy::needless_range_loop)]
fn rmsnorm_vec(out: &mut [f32], x: &[f32], w: &[f32], dim: usize, eps: f32) {
    let mut ss = 0.0f32;
    for i in 0..dim {
        ss += x[i] * x[i];
    }
    ss = 1.0 / (ss / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = w[i] * x[i] * ss;
    }
}

/// Per-head RMSNorm: independent RMSNorm on each head's [head_dim] slice.
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

/// Apply partial RoPE: only rotate the first `rope_dims` dimensions of each head.
///
/// Uses split-half rotation on the rotary subspace, leaving the remaining
/// dimensions unchanged.
fn apply_partial_rope_vec(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dims: usize,
    pos: usize,
    rope_theta: f32,
) {
    let half_rope = rope_dims / 2;
    for head in 0..n_heads {
        let off = head * head_dim;
        for d in 0..half_rope {
            let inv_freq = 1.0 / rope_theta.powf(2.0 * d as f32 / rope_dims as f32);
            let angle = pos as f32 * inv_freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x_first = x[off + d];
            let x_second = x[off + d + half_rope];
            x[off + d] = x_first * cos_a - x_second * sin_a;
            x[off + d + half_rope] = x_first * sin_a + x_second * cos_a;
        }
        // Dims rope_dims..head_dim are left unchanged
    }
}

/// Softplus: log(1 + exp(x)).
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // Avoid overflow
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid: 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

/// Check if a config.json is compatible with the CPU hybrid engine.
///
/// Returns Ok(()) if compatible, Err(reason) if not.
pub fn is_hybrid_cpu_compatible(
    config_json: &serde_json::Value,
) -> std::result::Result<(), String> {
    let model_type = config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if model_type != "qwen3_5_text" && model_type != "qwen3_next" {
        return Err(format!(
            "CPU hybrid engine only supports qwen3_next/qwen3_5_text, got '{model_type}'"
        ));
    }

    // Reject MoE models (too many expert weights for CPU decode)
    let num_experts = config_json
        .get("num_experts")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    if num_experts > 0 {
        return Err(format!(
            "CPU hybrid engine does not support MoE ({num_experts} experts)"
        ));
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> Qwen3NextInferenceConfig {
        Qwen3NextInferenceConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            n_layers: 4,
            vocab_size: 100,
            max_seq_len: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            num_v_heads: 4,
            num_k_heads: 4,
            head_k_dim: 8,
            head_v_dim: 8,
            conv_kernel_size: 4,
            full_attention_interval: 4,
            layer_types: None,
            tie_word_embeddings: true,
            temperature: 0.0,
            top_k: 0,
            max_tokens: 16,
            eos_token_id: None,
        }
    }

    #[test]
    fn test_config_derived_dims() {
        let c = small_config();
        assert_eq!(c.key_dim(), 32); // 4 * 8
        assert_eq!(c.value_dim(), 32); // 4 * 8
        assert_eq!(c.conv_dim(), 96); // 32*2 + 32
        assert_eq!(c.attn_q_dim(), 64); // 4 * 16
        assert_eq!(c.attn_kv_dim(), 32); // 2 * 16
        assert_eq!(c.attn_n_groups(), 2); // 4 / 2
        assert_eq!(c.rope_dims(), 4); // 16 * 0.25
    }

    #[test]
    fn test_layer_type_detection() {
        let c = small_config();
        // full_attention_interval=4: layers 3 is full attention
        assert!(c.is_gdn_layer(0));
        assert!(c.is_gdn_layer(1));
        assert!(c.is_gdn_layer(2));
        assert!(!c.is_gdn_layer(3)); // full attention
    }

    #[test]
    fn test_layer_type_from_explicit() {
        let mut c = small_config();
        c.layer_types = Some(vec![
            "linear_attention".into(),
            "linear_attention".into(),
            "full_attention".into(),
            "linear_attention".into(),
        ]);
        assert!(c.is_gdn_layer(0));
        assert!(c.is_gdn_layer(1));
        assert!(!c.is_gdn_layer(2)); // explicit full_attention
        assert!(c.is_gdn_layer(3));
    }

    #[test]
    fn test_engine_creation() {
        let config = small_config();
        let engine = Qwen3NextInferenceEngine::new(config).unwrap();
        assert_eq!(engine.config().n_layers, 4);
    }

    #[test]
    fn test_state_pool_creation() {
        let config = small_config();
        let state = HybridStatePool::new(&config);
        assert_eq!(state.layers.len(), 4);
        assert!(matches!(state.layers[0], HybridLayerState::Gdn(_)));
        assert!(matches!(state.layers[3], HybridLayerState::Attention(_)));
    }

    #[test]
    fn test_rmsnorm_vec() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        let mut out = [0.0; 4];
        rmsnorm_vec(&mut out, &x, &w, 4, 1e-6);
        // rms = sqrt(mean([1,4,9,16])) = sqrt(7.5) ≈ 2.7386
        // inv_rms ≈ 0.3651
        let rms = (7.5f32 + 1e-6).sqrt();
        let inv = 1.0 / rms;
        assert!((out[0] - 1.0 * inv).abs() < 1e-4);
        assert!((out[1] - 2.0 * inv).abs() < 1e-4);
    }

    #[test]
    fn test_partial_rope() {
        // 4 dims per head, rope_dims=2 (first 2 rotated)
        let mut x = [1.0, 2.0, 3.0, 4.0];
        apply_partial_rope_vec(&mut x, 1, 4, 2, 0, 10000.0);
        // At pos=0, angle=0, cos=1, sin=0 → no change to rotary dims
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 2.0).abs() < 1e-6);
        // Non-rotary dims unchanged
        assert!((x[2] - 3.0).abs() < 1e-6);
        assert!((x[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 0.01);
        assert!((softplus(30.0) - 30.0).abs() < 0.01); // Large x → x
        assert!(softplus(-30.0).abs() < 0.01); // Very negative → 0
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_compatibility_check() {
        let config: serde_json::Value = serde_json::json!({
            "model_type": "qwen3_next",
            "num_experts": 0
        });
        assert!(is_hybrid_cpu_compatible(&config).is_ok());

        let config_moe: serde_json::Value = serde_json::json!({
            "model_type": "qwen3_next",
            "num_experts": 512
        });
        assert!(is_hybrid_cpu_compatible(&config_moe).is_err());

        let config_wrong: serde_json::Value = serde_json::json!({
            "model_type": "llama"
        });
        assert!(is_hybrid_cpu_compatible(&config_wrong).is_err());
    }
}
