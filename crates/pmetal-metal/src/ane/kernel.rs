//! Transformer kernel generators for ANE.
//!
//! Generates MIL programs and weight blobs for the 6 kernel types needed
//! for a complete transformer training step:
//!
//! | Kernel | Direction | Description |
//! |--------|-----------|-------------|
//! | `gen_sdpa_fwd_taps` | Forward | Attention with concat output taps |
//! | `gen_ffn_fwd_taps` | Forward | FFN (SwiGLU) with concat output taps |
//! | `gen_ffn_bwd` | Backward | FFN backward using transposed weights |
//! | `gen_sdpa_bwd1` | Backward | SDPA backward part 1 (dV, probs, dp) |
//! | `gen_sdpa_bwd2` | Backward | SDPA backward part 2 (dQ, dK) — weight-free |
//! | `gen_qkv_bwd` | Backward | QKV backward using transposed weights |

use crate::ane::mil::MilProgram;
use crate::ane::runtime::WeightDict;

/// Configuration for transformer kernel generation.
#[derive(Debug, Clone)]
pub struct TransformerKernelConfig {
    /// Model dimension (e.g., 768).
    pub dim: usize,
    /// FFN hidden dimension (e.g., 2048).
    pub hidden_dim: usize,
    /// Number of attention heads (e.g., 12).
    pub n_heads: usize,
    /// Number of key/value heads for GQA/MQA (defaults to `n_heads`).
    pub n_kv_heads: usize,
    /// Head dimension (dim / n_heads).
    pub head_dim: usize,
    /// Sequence length (e.g., 256).
    pub seq_len: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl TransformerKernelConfig {
    /// Q projection output dimension = n_heads * head_dim.
    ///
    /// Equals `dim` for standard architectures but differs for models like
    /// Qwen3 where `head_dim != dim / n_heads`.
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// KV dimension = n_kv_heads * head_dim.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Number of Q heads per KV head group (for GQA tiling).
    pub fn n_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Score channels = n_heads * seq_len (attention score tensor channels).
    pub fn score_ch(&self) -> usize {
        self.n_heads * self.seq_len
    }

    /// Attention forward output channels: o_out + Q + K + V + attn_out + xnorm
    ///
    /// oo(DIM) + qf(Q_DIM) + kf(KV_DIM) + vf(KV_DIM) + af(Q_DIM) + xn(DIM)
    /// where Q_DIM = n_heads * head_dim (equals DIM when head_dim = dim/n_heads).
    pub fn sdpa_fwd_output_ch(&self) -> usize {
        2 * self.dim + 2 * self.q_dim() + 2 * self.kv_dim()
    }

    /// FFN forward output channels: ffn_out + h1 + h3 + silu_out + x2norm
    pub fn ffn_fwd_output_ch(&self) -> usize {
        // y(DIM) + h1(HIDDEN) + h3(HIDDEN) + gate/silu(HIDDEN) + xn(DIM) = 2*DIM + 3*HIDDEN
        2 * self.dim + 3 * self.hidden_dim
    }

    /// FFN backward input channels.
    pub fn ffn_bwd_input_ch(&self) -> usize {
        self.dim + 2 * self.hidden_dim
    }

    /// SDPA backward part 1 input channels: Q(q_dim) + K(kv_dim) + V(kv_dim) + dy(dim)
    pub fn sdpa_bwd1_input_ch(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim() + self.dim
    }

    /// SDPA backward part 1 output channels: dV(kv_dim) + probs + dp
    pub fn sdpa_bwd1_output_ch(&self) -> usize {
        self.kv_dim() + 2 * self.score_ch()
    }

    /// SDPA backward part 2 input channels: probs + dp + Q(q_dim) + K(kv_dim)
    pub fn sdpa_bwd2_input_ch(&self) -> usize {
        2 * self.score_ch() + self.q_dim() + self.kv_dim()
    }
}

/// Result of kernel generation: MIL text + weight blobs.
pub struct KernelOutput {
    /// MIL program text (UTF-8).
    pub mil_text: String,
    /// Weight dictionary for compilation.
    pub weights: WeightDict,
    /// Input size in bytes (fp16).
    pub input_bytes: usize,
    /// Output size in bytes (fp16).
    pub output_bytes: usize,
}

// ============================================================================
// Weight blob format
// ============================================================================

/// ANE weight blob with 128-byte header + fp16 data.
///
/// Header format:
/// ```text
/// [0]:    0x01
/// [4]:    0x02
/// [64-67]: 0xDEADBEEF (little-endian magic)
/// [68]:   0x01
/// [72-75]: data_size (uint32 LE)
/// [80-83]: 128 (data offset, uint32 LE)
/// [128+]:  fp16 data
/// ```
pub struct WeightBlob;

impl WeightBlob {
    /// Build a weight blob from f32 weights (row-major → fp16).
    pub fn from_f32(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
        let n = rows * cols;
        debug_assert_eq!(weights.len(), n);
        let data_size = n * 2; // fp16 = 2 bytes
        let total = 128 + data_size;
        let mut blob = vec![0u8; total];

        // Header
        write_header(&mut blob, data_size);

        // Convert f32 → fp16 (row-major, no transpose)
        let fp16_ptr = blob[128..].as_mut_ptr() as *mut u16;
        let fp16_slice = unsafe { std::slice::from_raw_parts_mut(fp16_ptr, n) };
        crate::neon_convert::f32_to_f16_bulk(weights, fp16_slice);

        blob
    }

    /// Build a transposed weight blob from f32 weights.
    ///
    /// Input is `[rows, cols]` row-major. Output blob stores transposed fp16:
    /// `blob[j*rows + i] = weights[i*cols + j]`.
    pub fn from_f32_transposed(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
        let n = rows * cols;
        debug_assert_eq!(weights.len(), n);
        let data_size = n * 2;
        let total = 128 + data_size;
        let mut blob = vec![0u8; total];

        write_header(&mut blob, data_size);

        // Transpose and convert
        let fp16_ptr = blob[128..].as_mut_ptr() as *mut u16;
        for i in 0..rows {
            for j in 0..cols {
                let val = half::f16::from_f32(weights[i * cols + j]).to_bits();
                unsafe {
                    *fp16_ptr.add(j * rows + i) = val;
                }
            }
        }

        blob
    }

    /// Build a weight blob from raw fp16 data (no conversion).
    pub fn from_fp16(fp16_data: &[u16]) -> Vec<u8> {
        let data_size = fp16_data.len() * 2;
        let total = 128 + data_size;
        let mut blob = vec![0u8; total];

        write_header(&mut blob, data_size);

        // Copy raw fp16 data
        unsafe {
            std::ptr::copy_nonoverlapping(
                fp16_data.as_ptr() as *const u8,
                blob[128..].as_mut_ptr(),
                data_size,
            );
        }

        blob
    }

    /// Build the RMSNorm weight blob.
    ///
    /// Input is 1D `[dim]` f32 weights. Output is `[1, dim, 1, 1]` fp16 blob.
    pub fn from_rms_weights(weights: &[f32]) -> Vec<u8> {
        let n = weights.len();
        let data_size = n * 2;
        let total = 128 + data_size;
        let mut blob = vec![0u8; total];

        write_header(&mut blob, data_size);

        let fp16_ptr = blob[128..].as_mut_ptr() as *mut u16;
        let fp16_slice = unsafe { std::slice::from_raw_parts_mut(fp16_ptr, n) };
        crate::neon_convert::f32_to_f16_bulk(weights, fp16_slice);

        blob
    }
}

/// Write the 128-byte blob header.
fn write_header(blob: &mut [u8], data_size: usize) {
    blob[0] = 0x01;
    blob[4] = 0x02;
    // Magic: 0xDEADBEEF little-endian
    blob[64] = 0xEF;
    blob[65] = 0xBE;
    blob[66] = 0xAD;
    blob[67] = 0xDE;
    blob[68] = 0x01;
    // Data size (uint32 LE)
    let ds = data_size as u32;
    blob[72..76].copy_from_slice(&ds.to_le_bytes());
    // Data offset (uint32 LE) = 128
    blob[80..84].copy_from_slice(&128u32.to_le_bytes());
}

// ============================================================================
// Causal mask
// ============================================================================

/// Generate the causal mask blob for attention.
///
/// `mask[t, t2] = 0.0 if t2 <= t else -65504.0` (fp16 negative infinity).
/// Shape: `[1, 1, seq, seq]` stored as `[seq*seq]` fp16.
pub fn build_causal_mask(seq_len: usize) -> Vec<u8> {
    let mut mask = vec![0u16; seq_len * seq_len];
    let neg_inf = half::f16::from_f32(-65504.0).to_bits();
    let zero = half::f16::from_f32(0.0).to_bits();

    for t in 0..seq_len {
        for t2 in 0..seq_len {
            mask[t * seq_len + t2] = if t2 <= t { zero } else { neg_inf };
        }
    }

    WeightBlob::from_fp16(&mask)
}

// ============================================================================
// Kernel generators
// ============================================================================

/// Emit inline RMSNorm into a MIL program.
///
/// Reads from `x`, writes normalized output to `xn`.
fn emit_rmsnorm(p: &mut MilProgram, d: usize, s: usize, inv_d: f32, weight_path: &str) {
    p.emit_mul("sq", &[1, d, 1, s], "x", "x");
    p.emit_tensor_const("rax", &[1], "int32", "[1]");
    p.emit_scalar_const("kd", "bool", "true");
    p.emit_reduce_sum("ss", &[1, 1, 1, s], "sq", "rax", "kd");
    p.emit_scalar_const("invd", "fp16", &format!("{inv_d}"));
    p.emit_mul("ss2", &[1, 1, 1, s], "ss", "invd");
    p.emit_scalar_const("eps", "fp16", "0.00001");
    p.emit_add("ss3", &[1, 1, 1, s], "ss2", "eps");
    p.emit_scalar_const("nhalf", "fp16", "-0.5");
    p.emit_pow("rrms", &[1, 1, 1, s], "ss3", "nhalf");
    p.emit_mul("xr", &[1, d, 1, s], "x", "rrms");
    p.emit_weight_const("rw", &[1, d, 1, 1], weight_path);
    p.emit_mul("xn", &[1, d, 1, s], "xr", "rw");
}

/// Build precomputed cos/sin RoPE tables as weight blobs.
///
/// Returns `(cos_blob, sin_blob)` each of shape `[1, 1, half_dim, seq_len]`.
/// Uses non-traditional split-half RoPE frequencies:
/// `inv_freq[d] = 1 / rope_theta^(2d / head_dim)` for `d in 0..half_dim`.
fn build_rope_tables(head_dim: usize, seq_len: usize, rope_theta: f32) -> (Vec<u8>, Vec<u8>) {
    let half_dim = head_dim / 2;
    let n = half_dim * seq_len;
    let mut cos_data = vec![0.0f32; n];
    let mut sin_data = vec![0.0f32; n];

    for d in 0..half_dim {
        let inv_freq = 1.0 / rope_theta.powf(2.0 * d as f32 / head_dim as f32);
        for t in 0..seq_len {
            let angle = t as f32 * inv_freq;
            // Layout: [half_dim, seq_len] channel-first
            cos_data[d * seq_len + t] = angle.cos();
            sin_data[d * seq_len + t] = angle.sin();
        }
    }

    (
        WeightBlob::from_f32(&cos_data, half_dim, seq_len),
        WeightBlob::from_f32(&sin_data, half_dim, seq_len),
    )
}

/// Emit MIL for per-head RMSNorm on a `[1, n_heads, head_dim, seq]` tensor.
///
/// Applies independent RMSNorm per head: reduce along axis 2 (head_dim),
/// compute rsqrt, scale by broadcast weight `[1, 1, head_dim, 1]`.
#[allow(clippy::too_many_arguments)]
fn emit_per_head_rmsnorm(
    p: &mut MilProgram,
    input: &str,
    output: &str,
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
    eps: f32,
    weight_path: &str,
) {
    let pfx = p.next_var("phrn");

    // sq = x * x → [1, n_heads, head_dim, seq]
    let sq = p.next_var("sq");
    p.emit_mul(&sq, &[1, n_heads, head_dim, seq_len], input, input);

    // reduce_sum along axis 2 (head_dim) → [1, n_heads, 1, seq]
    let rax = p.next_var("rax");
    p.emit_tensor_const(&rax, &[1], "int32", "[2]");
    let kd = p.next_var("kd");
    p.emit_scalar_const(&kd, "bool", "true");
    let ss = p.next_var("ss");
    p.emit_reduce_sum(&ss, &[1, n_heads, 1, seq_len], &sq, &rax, &kd);

    // Multiply by 1/head_dim
    let inv_hd = p.next_var("inv");
    let inv_val = 1.0 / head_dim as f32;
    p.emit_scalar_const(&inv_hd, "fp16", &format!("{inv_val}"));
    let ss2 = p.next_var("ss2");
    p.emit_mul(&ss2, &[1, n_heads, 1, seq_len], &ss, &inv_hd);

    // Add eps
    let eps_c = p.next_var("eps");
    p.emit_scalar_const(&eps_c, "fp16", &format!("{eps}"));
    let ss3 = p.next_var("ss3");
    p.emit_add(&ss3, &[1, n_heads, 1, seq_len], &ss2, &eps_c);

    // rsqrt = pow(ss3, -0.5)
    let nhalf = p.next_var("nh");
    p.emit_scalar_const(&nhalf, "fp16", "-0.5");
    let rsqrt = p.next_var("rsq");
    p.emit_pow(&rsqrt, &[1, n_heads, 1, seq_len], &ss3, &nhalf);

    // xr = input * rsqrt (broadcasts over head_dim)
    let xr = p.next_var("xr");
    p.emit_mul(&xr, &[1, n_heads, head_dim, seq_len], input, &rsqrt);

    // Load weight [1, 1, head_dim, 1] and multiply (broadcasts over heads and seq)
    let rw = format!("{pfx}_rw");
    p.emit_weight_const(&rw, &[1, 1, head_dim, 1], weight_path);
    p.emit_mul(output, &[1, n_heads, head_dim, seq_len], &xr, &rw);
}

/// Emit MIL for non-traditional split-half RoPE on `[1, n_heads, head_dim, seq]`.
///
/// Splits the head_dim axis into first/second halves, applies rotation using
/// precomputed cos/sin tables of shape `[1, 1, half_dim, seq]`.
#[allow(clippy::too_many_arguments)]
fn emit_rope(
    p: &mut MilProgram,
    input: &str,
    output: &str,
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
    cos_path: &str,
    sin_path: &str,
) {
    let half_dim = head_dim / 2;
    let pfx = p.next_var("rope");

    // Load cos, sin tables: [1, 1, half_dim, seq]
    let cos_w = format!("{pfx}_cos");
    p.emit_weight_const(&cos_w, &[1, 1, half_dim, seq_len], cos_path);
    let sin_w = format!("{pfx}_sin");
    p.emit_weight_const(&sin_w, &[1, 1, half_dim, seq_len], sin_path);

    // Slice first half: begin=[0,0,0,0], size=[1,n_heads,half_dim,seq]
    let b0 = format!("{pfx}_b0");
    p.emit_tensor_const(&b0, &[4], "int32", "[0,0,0,0]");
    let sz_h = format!("{pfx}_szh");
    p.emit_tensor_const(
        &sz_h,
        &[4],
        "int32",
        &format!("[1,{n_heads},{half_dim},{seq_len}]"),
    );
    let x_first = format!("{pfx}_xf");
    p.emit_slice_by_size(
        &x_first,
        &[1, n_heads, half_dim, seq_len],
        input,
        &b0,
        &sz_h,
    );

    // Slice second half: begin=[0,0,half_dim,0]
    let b1 = format!("{pfx}_b1");
    p.emit_tensor_const(&b1, &[4], "int32", &format!("[0,0,{half_dim},0]"));
    let x_second = format!("{pfx}_xs");
    p.emit_slice_by_size(
        &x_second,
        &[1, n_heads, half_dim, seq_len],
        input,
        &b1,
        &sz_h,
    );

    // rot_first = x_first * cos - x_second * sin
    let fc = format!("{pfx}_fc");
    p.emit_mul(&fc, &[1, n_heads, half_dim, seq_len], &x_first, &cos_w);
    let ss = format!("{pfx}_ss");
    p.emit_mul(&ss, &[1, n_heads, half_dim, seq_len], &x_second, &sin_w);
    let rot_first = format!("{pfx}_rf");
    p.emit_sub(&rot_first, &[1, n_heads, half_dim, seq_len], &fc, &ss);

    // rot_second = x_first * sin + x_second * cos
    let fs = format!("{pfx}_fs");
    p.emit_mul(&fs, &[1, n_heads, half_dim, seq_len], &x_first, &sin_w);
    let sc = format!("{pfx}_sc");
    p.emit_mul(&sc, &[1, n_heads, half_dim, seq_len], &x_second, &cos_w);
    let rot_second = format!("{pfx}_rs");
    p.emit_add(&rot_second, &[1, n_heads, half_dim, seq_len], &fs, &sc);

    // Concat halves back on axis 2
    let cax = format!("{pfx}_cax");
    p.emit_scalar_const(&cax, "int32", "2");
    let cid = format!("{pfx}_cid");
    p.emit_scalar_const(&cid, "bool", "false");
    p.emit_concat(
        output,
        &[1, n_heads, head_dim, seq_len],
        &cax,
        &cid,
        &[&rot_first, &rot_second],
    );
}

/// Validate the kernel configuration.
pub fn validate_config(cfg: &TransformerKernelConfig) -> crate::error::Result<()> {
    use crate::error::MetalError;
    if cfg.n_heads == 0 {
        return Err(MetalError::InvalidConfig("n_heads must be > 0".into()));
    }
    if cfg.n_kv_heads == 0 {
        return Err(MetalError::InvalidConfig("n_kv_heads must be > 0".into()));
    }
    if cfg.n_kv_heads > cfg.n_heads {
        return Err(MetalError::InvalidConfig(
            "n_kv_heads must be <= n_heads".into(),
        ));
    }
    if cfg.n_heads % cfg.n_kv_heads != 0 {
        return Err(MetalError::InvalidConfig(
            "n_heads must be divisible by n_kv_heads".into(),
        ));
    }
    if cfg.dim == 0 || cfg.dim % cfg.n_heads != 0 {
        return Err(MetalError::InvalidConfig(
            "dim must be > 0 and divisible by n_heads".into(),
        ));
    }
    if cfg.head_dim != cfg.dim / cfg.n_heads {
        return Err(MetalError::InvalidConfig(
            "head_dim must equal dim / n_heads".into(),
        ));
    }
    Ok(())
}

/// Generate the SDPA forward kernel with feature taps.
///
/// Input: `[1, DIM, 1, SEQ]`
/// Output: `[1, 6*DIM, 1, SEQ]` = concat(o_out, Q, K, V, attn_out, xnorm)
///
/// Weights: rms1, Wq, Wk, Wv, Wo, causal mask
pub fn gen_sdpa_fwd_taps(
    cfg: &TransformerKernelConfig,
    rms_att: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
) -> KernelOutput {
    let d = cfg.dim;
    let s = cfg.seq_len;
    let h = cfg.n_heads;
    let hd = cfg.head_dim;
    let kv_h = cfg.n_kv_heads;
    let kv_d = cfg.kv_dim();
    let q_dim = cfg.q_dim();
    let n_groups = cfg.n_groups();
    let sc = 1.0 / (hd as f32).sqrt();
    let inv_d = 1.0 / d as f32;

    let mut p = MilProgram::new(d, s);

    // RMSNorm inline
    emit_rmsnorm(&mut p, d, s, inv_d, "@model_path/weights/rms1.bin");

    // QKV convolutions — Wq projects dim→q_dim, Wo projects q_dim→dim
    p.emit_conv_constants();
    p.emit_weight_const("Wq", &[q_dim, d, 1, 1], "@model_path/weights/wq.bin");
    p.emit_weight_const("Wk", &[kv_d, d, 1, 1], "@model_path/weights/wk.bin");
    p.emit_weight_const("Wv", &[kv_d, d, 1, 1], "@model_path/weights/wv.bin");
    p.emit_weight_const("Wo", &[d, q_dim, 1, 1], "@model_path/weights/wo.bin");
    p.emit_conv("qf", &[1, q_dim, 1, s], "Wq", "xn");
    p.emit_conv("kf", &[1, kv_d, 1, s], "Wk", "xn");
    p.emit_conv("vf", &[1, kv_d, 1, s], "Wv", "xn");

    // Reshape and transpose for multi-head attention
    p.emit_tensor_const("qsh", &[4], "int32", &format!("[1,{h},{hd},{s}]"));
    p.emit_tensor_const("pm", &[4], "int32", "[0,1,3,2]");
    p.emit_reshape("q4", &[1, h, hd, s], "qsh", "qf");
    p.emit_transpose("q", &[1, h, s, hd], "pm", "q4");

    // K, V: reshape to [1, kv_h, hd, S] then transpose, then tile if GQA
    p.emit_tensor_const("kvsh", &[4], "int32", &format!("[1,{kv_h},{hd},{s}]"));
    p.emit_reshape("k4", &[1, kv_h, hd, s], "kvsh", "kf");
    p.emit_transpose("k0", &[1, kv_h, s, hd], "pm", "k4");
    p.emit_reshape("v4", &[1, kv_h, hd, s], "kvsh", "vf");
    p.emit_transpose("v0", &[1, kv_h, s, hd], "pm", "v4");

    let (k_name, v_name) = if n_groups > 1 {
        p.emit_tensor_const("greps", &[4], "int32", &format!("[1,{n_groups},1,1]"));
        p.emit_tile("k", &[1, h, s, hd], "greps", "k0");
        p.emit_tile("v", &[1, h, s, hd], "greps", "v0");
        ("k", "v")
    } else {
        ("k0", "v0")
    };

    // Attention: scores = Q @ K^T * scale + mask
    p.emit_scalar_const("tx", "bool", "false");
    p.emit_scalar_const("ty", "bool", "true");
    p.emit_matmul("sc1", &[1, h, s, s], "tx", "ty", "q", k_name);
    p.emit_scalar_const("scv", "fp16", &format!("{sc}"));
    p.emit_mul("sc2", &[1, h, s, s], "sc1", "scv");
    p.emit_weight_const("cm", &[1, 1, s, s], "@model_path/weights/mask.bin");
    p.emit_add("ms", &[1, h, s, s], "sc2", "cm");

    // Softmax
    p.emit_scalar_const("sax", "int32", "-1");
    p.emit_softmax("aw", &[1, h, s, s], "sax", "ms");

    // Attention output: scores @ V, reshape back to [1, q_dim, 1, s], project to dim
    p.emit_matmul("a4", &[1, h, s, hd], "tx", "tx", "aw", v_name);
    p.emit_transpose("at", &[1, h, hd, s], "pm", "a4");
    p.emit_tensor_const("os", &[4], "int32", &format!("[1,{q_dim},1,{s}]"));
    p.emit_reshape("af", &[1, q_dim, 1, s], "os", "at");
    p.emit_conv("oo", &[1, d, 1, s], "Wo", "af");

    // Concat output taps: oo + qf + kf + vf + af + xn
    // oo=dim, qf=q_dim, kf=kv_dim, vf=kv_dim, af=q_dim, xn=dim
    let out_ch = 2 * d + 2 * q_dim + 2 * kv_d;
    p.emit_scalar_const("cax", "int32", "1");
    p.emit_scalar_const("cid", "bool", "false");
    p.emit_concat(
        "out",
        &[1, out_ch, 1, s],
        "cax",
        "cid",
        &["oo", "qf", "kf", "vf", "af", "xn"],
    );

    let mil_text = p.finalize("out");

    // Build weight dictionary
    let mut weights = WeightDict::new();
    weights.add(
        "@model_path/weights/rms1.bin",
        WeightBlob::from_rms_weights(rms_att),
    );
    weights.add(
        "@model_path/weights/wq.bin",
        WeightBlob::from_f32(wq, q_dim, d),
    );
    weights.add(
        "@model_path/weights/wk.bin",
        WeightBlob::from_f32(wk, kv_d, d),
    );
    weights.add(
        "@model_path/weights/wv.bin",
        WeightBlob::from_f32(wv, kv_d, d),
    );
    weights.add(
        "@model_path/weights/wo.bin",
        WeightBlob::from_f32(wo, d, q_dim),
    );
    weights.add("@model_path/weights/mask.bin", build_causal_mask(s));

    KernelOutput {
        mil_text,
        weights,
        input_bytes: d * s * 2,
        output_bytes: out_ch * s * 2,
    }
}

/// Generate the SDPA forward kernel for inference (no feature taps).
///
/// Input: `[1, DIM, 1, SEQ]`
/// Output: `[1, DIM, 1, SEQ]` = attention output only
///
/// Weights: rms1, Wq, Wk, Wv, Wo, causal mask
pub fn gen_sdpa_fwd(
    cfg: &TransformerKernelConfig,
    rms_att: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
) -> KernelOutput {
    let d = cfg.dim;
    let s = cfg.seq_len;
    let h = cfg.n_heads;
    let hd = cfg.head_dim;
    let kv_h = cfg.n_kv_heads;
    let kv_d = cfg.kv_dim();
    let q_dim = cfg.q_dim();
    let n_groups = cfg.n_groups();
    let sc = 1.0 / (hd as f32).sqrt();
    let inv_d = 1.0 / d as f32;

    let mut p = MilProgram::new(d, s);

    // RMSNorm inline
    emit_rmsnorm(&mut p, d, s, inv_d, "@model_path/weights/rms1.bin");

    // QKV convolutions — Wq projects dim→q_dim, Wo projects q_dim→dim
    p.emit_conv_constants();
    p.emit_weight_const("Wq", &[q_dim, d, 1, 1], "@model_path/weights/wq.bin");
    p.emit_weight_const("Wk", &[kv_d, d, 1, 1], "@model_path/weights/wk.bin");
    p.emit_weight_const("Wv", &[kv_d, d, 1, 1], "@model_path/weights/wv.bin");
    p.emit_weight_const("Wo", &[d, q_dim, 1, 1], "@model_path/weights/wo.bin");
    p.emit_conv("qf", &[1, q_dim, 1, s], "Wq", "xn");
    p.emit_conv("kf", &[1, kv_d, 1, s], "Wk", "xn");
    p.emit_conv("vf", &[1, kv_d, 1, s], "Wv", "xn");

    // Reshape and transpose for multi-head attention
    p.emit_tensor_const("qsh", &[4], "int32", &format!("[1,{h},{hd},{s}]"));
    p.emit_tensor_const("pm", &[4], "int32", "[0,1,3,2]");
    p.emit_reshape("q4", &[1, h, hd, s], "qsh", "qf");
    p.emit_transpose("q", &[1, h, s, hd], "pm", "q4");

    // K, V: reshape to [1, kv_h, hd, S] then transpose, then tile if GQA
    p.emit_tensor_const("kvsh", &[4], "int32", &format!("[1,{kv_h},{hd},{s}]"));
    p.emit_reshape("k4", &[1, kv_h, hd, s], "kvsh", "kf");
    p.emit_transpose("k0", &[1, kv_h, s, hd], "pm", "k4");
    p.emit_reshape("v4", &[1, kv_h, hd, s], "kvsh", "vf");
    p.emit_transpose("v0", &[1, kv_h, s, hd], "pm", "v4");

    // GQA: tile K, V along head axis to match n_heads
    let (k_name, v_name) = if n_groups > 1 {
        p.emit_tensor_const("greps", &[4], "int32", &format!("[1,{n_groups},1,1]"));
        p.emit_tile("k", &[1, h, s, hd], "greps", "k0");
        p.emit_tile("v", &[1, h, s, hd], "greps", "v0");
        ("k", "v")
    } else {
        ("k0", "v0")
    };

    // Attention: scores = Q @ K^T * scale + mask
    p.emit_scalar_const("tx", "bool", "false");
    p.emit_scalar_const("ty", "bool", "true");
    p.emit_matmul("sc1", &[1, h, s, s], "tx", "ty", "q", k_name);
    p.emit_scalar_const("scv", "fp16", &format!("{sc}"));
    p.emit_mul("sc2", &[1, h, s, s], "sc1", "scv");
    p.emit_weight_const("cm", &[1, 1, s, s], "@model_path/weights/mask.bin");
    p.emit_add("ms", &[1, h, s, s], "sc2", "cm");

    // Softmax
    p.emit_scalar_const("sax", "int32", "-1");
    p.emit_softmax("aw", &[1, h, s, s], "sax", "ms");

    // Attention output: scores @ V, reshape back to [1, q_dim, 1, s], project to dim
    p.emit_matmul("a4", &[1, h, s, hd], "tx", "tx", "aw", v_name);
    p.emit_transpose("at", &[1, h, hd, s], "pm", "a4");
    p.emit_tensor_const("os", &[4], "int32", &format!("[1,{q_dim},1,{s}]"));
    p.emit_reshape("af", &[1, q_dim, 1, s], "os", "at");
    p.emit_conv("oo", &[1, d, 1, s], "Wo", "af");

    // No concat — finalize directly on attention output
    let mil_text = p.finalize("oo");

    // Build weight dictionary
    let mut weights = WeightDict::new();
    weights.add(
        "@model_path/weights/rms1.bin",
        WeightBlob::from_rms_weights(rms_att),
    );
    weights.add(
        "@model_path/weights/wq.bin",
        WeightBlob::from_f32(wq, q_dim, d),
    );
    weights.add(
        "@model_path/weights/wk.bin",
        WeightBlob::from_f32(wk, kv_d, d),
    );
    weights.add(
        "@model_path/weights/wv.bin",
        WeightBlob::from_f32(wv, kv_d, d),
    );
    weights.add(
        "@model_path/weights/wo.bin",
        WeightBlob::from_f32(wo, d, q_dim),
    );
    weights.add("@model_path/weights/mask.bin", build_causal_mask(s));

    KernelOutput {
        mil_text,
        weights,
        input_bytes: d * s * 2,
        output_bytes: d * s * 2,
    }
}

/// Generate the SDPA forward kernel with KV cache output (prefill).
///
/// Input: `[1, DIM, 1, SEQ]`
/// Output: `[1, DIM + 2*KV_DIM, 1, SEQ]` = concat(o_out, K_roped, V_proj)
///
/// Includes per-head QK-norm and RoPE before attention computation.
/// The output K is post-RoPE so the CPU decode path can use it directly.
///
/// Weights: rms1, Wq, Wk, Wv, Wo, causal mask, q_norm, k_norm, cos, sin
#[allow(clippy::too_many_arguments)]
pub fn gen_sdpa_fwd_kv(
    cfg: &TransformerKernelConfig,
    rms_att: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
) -> KernelOutput {
    let d = cfg.dim;
    let s = cfg.seq_len;
    let h = cfg.n_heads;
    let hd = cfg.head_dim;
    let kv_h = cfg.n_kv_heads;
    let kv_d = cfg.kv_dim();
    let q_dim = cfg.q_dim(); // n_heads * head_dim — may differ from dim (e.g. Qwen3)
    let n_groups = cfg.n_groups();
    let sc = 1.0 / (hd as f32).sqrt();
    let inv_d = 1.0 / d as f32;

    let mut p = MilProgram::new(d, s);

    // RMSNorm inline
    emit_rmsnorm(&mut p, d, s, inv_d, "@model_path/weights/rms1.bin");

    // QKV convolutions — Wq projects dim→q_dim, Wo projects q_dim→dim
    p.emit_conv_constants();
    p.emit_weight_const("Wq", &[q_dim, d, 1, 1], "@model_path/weights/wq.bin");
    p.emit_weight_const("Wk", &[kv_d, d, 1, 1], "@model_path/weights/wk.bin");
    p.emit_weight_const("Wv", &[kv_d, d, 1, 1], "@model_path/weights/wv.bin");
    p.emit_weight_const("Wo", &[d, q_dim, 1, 1], "@model_path/weights/wo.bin");
    p.emit_conv("qf", &[1, q_dim, 1, s], "Wq", "xn");
    p.emit_conv("kf", &[1, kv_d, 1, s], "Wk", "xn");
    p.emit_conv("vf", &[1, kv_d, 1, s], "Wv", "xn");

    // Reshape Q to [1, h, hd, s], K to [1, kv_h, hd, s]
    p.emit_tensor_const("qsh", &[4], "int32", &format!("[1,{h},{hd},{s}]"));
    p.emit_reshape("q4", &[1, h, hd, s], "qsh", "qf");

    p.emit_tensor_const("kvsh", &[4], "int32", &format!("[1,{kv_h},{hd},{s}]"));
    p.emit_reshape("k4", &[1, kv_h, hd, s], "kvsh", "kf");

    // Per-head QK-norm on Q and K (in [1, heads, head_dim, seq] layout)
    emit_per_head_rmsnorm(
        &mut p,
        "q4",
        "qn",
        h,
        hd,
        s,
        cfg.rms_norm_eps,
        "@model_path/weights/qnorm.bin",
    );
    emit_per_head_rmsnorm(
        &mut p,
        "k4",
        "kn",
        kv_h,
        hd,
        s,
        cfg.rms_norm_eps,
        "@model_path/weights/knorm.bin",
    );

    // RoPE on Q and K (both in [1, heads, head_dim, seq] layout)
    emit_rope(
        &mut p,
        "qn",
        "qr",
        h,
        hd,
        s,
        "@model_path/weights/cos.bin",
        "@model_path/weights/sin.bin",
    );
    emit_rope(
        &mut p,
        "kn",
        "kr",
        kv_h,
        hd,
        s,
        "@model_path/weights/cos.bin",
        "@model_path/weights/sin.bin",
    );

    // Transpose Q to [1, h, s, hd], K to [1, kv_h, s, hd]
    p.emit_tensor_const("pm", &[4], "int32", "[0,1,3,2]");
    p.emit_transpose("q", &[1, h, s, hd], "pm", "qr");
    p.emit_transpose("k0", &[1, kv_h, s, hd], "pm", "kr");

    // V: reshape and transpose (no QK-norm or RoPE on V)
    p.emit_reshape("v4", &[1, kv_h, hd, s], "kvsh", "vf");
    p.emit_transpose("v0", &[1, kv_h, s, hd], "pm", "v4");

    // GQA: tile K, V along head axis to match n_heads
    let (k_name, v_name) = if n_groups > 1 {
        p.emit_tensor_const("greps", &[4], "int32", &format!("[1,{n_groups},1,1]"));
        p.emit_tile("k", &[1, h, s, hd], "greps", "k0");
        p.emit_tile("v", &[1, h, s, hd], "greps", "v0");
        ("k", "v")
    } else {
        ("k0", "v0")
    };

    // Attention: scores = Q @ K^T * scale + mask
    p.emit_scalar_const("tx", "bool", "false");
    p.emit_scalar_const("ty", "bool", "true");
    p.emit_matmul("sc1", &[1, h, s, s], "tx", "ty", "q", k_name);
    p.emit_scalar_const("scv", "fp16", &format!("{sc}"));
    p.emit_mul("sc2", &[1, h, s, s], "sc1", "scv");
    p.emit_weight_const("cm", &[1, 1, s, s], "@model_path/weights/mask.bin");
    p.emit_add("ms", &[1, h, s, s], "sc2", "cm");

    // Softmax
    p.emit_scalar_const("sax", "int32", "-1");
    p.emit_softmax("aw", &[1, h, s, s], "sax", "ms");

    // Attention output: scores @ V, reshape back to [1, q_dim, 1, s], project to dim
    p.emit_matmul("a4", &[1, h, s, hd], "tx", "tx", "aw", v_name);
    p.emit_transpose("at", &[1, h, hd, s], "pm", "a4");
    p.emit_tensor_const("os", &[4], "int32", &format!("[1,{q_dim},1,{s}]"));
    p.emit_reshape("af", &[1, q_dim, 1, s], "os", "at");
    p.emit_conv("oo", &[1, d, 1, s], "Wo", "af");

    // Output post-RoPE K (not raw kf) for the KV cache:
    // Reshape kr [1, kv_h, hd, s] back to [1, kv_d, 1, s]
    p.emit_tensor_const("kvfs", &[4], "int32", &format!("[1,{kv_d},1,{s}]"));
    p.emit_reshape("kr_flat", &[1, kv_d, 1, s], "kvfs", "kr");

    // 3-way concat: (oo, kr_flat, vf) — post-RoPE K, raw V
    let out_ch = d + 2 * kv_d;
    p.emit_scalar_const("cax", "int32", "1");
    p.emit_scalar_const("cid", "bool", "false");
    p.emit_concat(
        "out",
        &[1, out_ch, 1, s],
        "cax",
        "cid",
        &["oo", "kr_flat", "vf"],
    );

    let mil_text = p.finalize("out");

    // Build weight dictionary
    let mut weights = WeightDict::new();
    weights.add(
        "@model_path/weights/rms1.bin",
        WeightBlob::from_rms_weights(rms_att),
    );
    weights.add(
        "@model_path/weights/wq.bin",
        WeightBlob::from_f32(wq, q_dim, d),
    );
    weights.add(
        "@model_path/weights/wk.bin",
        WeightBlob::from_f32(wk, kv_d, d),
    );
    weights.add(
        "@model_path/weights/wv.bin",
        WeightBlob::from_f32(wv, kv_d, d),
    );
    weights.add(
        "@model_path/weights/wo.bin",
        WeightBlob::from_f32(wo, d, q_dim),
    );
    weights.add("@model_path/weights/mask.bin", build_causal_mask(s));
    weights.add(
        "@model_path/weights/qnorm.bin",
        WeightBlob::from_rms_weights(q_norm),
    );
    weights.add(
        "@model_path/weights/knorm.bin",
        WeightBlob::from_rms_weights(k_norm),
    );
    let (cos_blob, sin_blob) = build_rope_tables(hd, s, cfg.rope_theta);
    weights.add("@model_path/weights/cos.bin", cos_blob);
    weights.add("@model_path/weights/sin.bin", sin_blob);

    KernelOutput {
        mil_text,
        weights,
        input_bytes: d * s * 2,
        output_bytes: out_ch * s * 2,
    }
}

/// Generate the FFN forward kernel for inference (no feature taps).
///
/// Input: `[1, DIM, 1, SEQ]`
/// Output: `[1, DIM, 1, SEQ]` = FFN output only
///
/// Weights: rms2, W1, W3, W2
pub fn gen_ffn_fwd(
    cfg: &TransformerKernelConfig,
    rms_ffn: &[f32],
    w1: &[f32],
    w3: &[f32],
    w2: &[f32],
) -> KernelOutput {
    let d = cfg.dim;
    let h = cfg.hidden_dim;
    let s = cfg.seq_len;
    let inv_d = 1.0 / d as f32;

    let mut p = MilProgram::new(d, s);

    // RMSNorm inline
    p.emit_mul("sq", &[1, d, 1, s], "x", "x");
    p.emit_tensor_const("rax", &[1], "int32", "[1]");
    p.emit_scalar_const("kd", "bool", "true");
    p.emit_reduce_sum("ss", &[1, 1, 1, s], "sq", "rax", "kd");
    p.emit_scalar_const("invd", "fp16", &format!("{inv_d}"));
    p.emit_mul("ss2", &[1, 1, 1, s], "ss", "invd");
    p.emit_scalar_const("eps", "fp16", "0.00001");
    p.emit_add("ss3", &[1, 1, 1, s], "ss2", "eps");
    p.emit_scalar_const("nhalf", "fp16", "-0.5");
    p.emit_pow("rrms", &[1, 1, 1, s], "ss3", "nhalf");
    p.emit_mul("xr", &[1, d, 1, s], "x", "rrms");
    p.emit_weight_const("rw", &[1, d, 1, 1], "@model_path/weights/rms2.bin");
    p.emit_mul("xn", &[1, d, 1, s], "xr", "rw");

    // FFN: SwiGLU
    p.emit_conv_constants();
    p.emit_weight_const("W1", &[h, d, 1, 1], "@model_path/weights/w1.bin");
    p.emit_weight_const("W3", &[h, d, 1, 1], "@model_path/weights/w3.bin");
    p.emit_weight_const("W2", &[d, h, 1, 1], "@model_path/weights/w2.bin");
    p.emit_conv("h1", &[1, h, 1, s], "W1", "xn");
    p.emit_conv("h3", &[1, h, 1, s], "W3", "xn");
    p.emit_sigmoid("sig", &[1, h, 1, s], "h1");
    p.emit_mul("silu", &[1, h, 1, s], "h1", "sig");
    p.emit_mul("gate", &[1, h, 1, s], "silu", "h3");
    p.emit_conv("y", &[1, d, 1, s], "W2", "gate");

    // No concat — finalize directly on FFN output
    let mil_text = p.finalize("y");

    let mut weights = WeightDict::new();
    weights.add(
        "@model_path/weights/rms2.bin",
        WeightBlob::from_rms_weights(rms_ffn),
    );
    weights.add("@model_path/weights/w1.bin", WeightBlob::from_f32(w1, h, d));
    weights.add("@model_path/weights/w3.bin", WeightBlob::from_f32(w3, h, d));
    weights.add("@model_path/weights/w2.bin", WeightBlob::from_f32(w2, d, h));

    KernelOutput {
        mil_text,
        weights,
        input_bytes: d * s * 2,
        output_bytes: d * s * 2,
    }
}

/// Generate the FFN forward kernel with feature taps.
///
/// Input: `[1, DIM, 1, SEQ]`
/// Output: `[1, 2*DIM + 3*HIDDEN, 1, SEQ]` = concat(y, h1, h3, silu_out, xnorm)
///
/// Weights: rms2, W1, W3, W2
pub fn gen_ffn_fwd_taps(
    cfg: &TransformerKernelConfig,
    rms_ffn: &[f32],
    w1: &[f32],
    w3: &[f32],
    w2: &[f32],
) -> KernelOutput {
    let d = cfg.dim;
    let h = cfg.hidden_dim;
    let s = cfg.seq_len;
    let inv_d = 1.0 / d as f32;

    let mut p = MilProgram::new(d, s);

    // RMSNorm inline
    p.emit_mul("sq", &[1, d, 1, s], "x", "x");
    p.emit_tensor_const("rax", &[1], "int32", "[1]");
    p.emit_scalar_const("kd", "bool", "true");
    p.emit_reduce_sum("ss", &[1, 1, 1, s], "sq", "rax", "kd");
    p.emit_scalar_const("invd", "fp16", &format!("{inv_d}"));
    p.emit_mul("ss2", &[1, 1, 1, s], "ss", "invd");
    p.emit_scalar_const("eps", "fp16", "0.00001");
    p.emit_add("ss3", &[1, 1, 1, s], "ss2", "eps");
    p.emit_scalar_const("nhalf", "fp16", "-0.5");
    p.emit_pow("rrms", &[1, 1, 1, s], "ss3", "nhalf");
    p.emit_mul("xr", &[1, d, 1, s], "x", "rrms");
    p.emit_weight_const("rw", &[1, d, 1, 1], "@model_path/weights/rms2.bin");
    p.emit_mul("xn", &[1, d, 1, s], "xr", "rw");

    // FFN: SwiGLU
    p.emit_conv_constants();
    p.emit_weight_const("W1", &[h, d, 1, 1], "@model_path/weights/w1.bin");
    p.emit_weight_const("W3", &[h, d, 1, 1], "@model_path/weights/w3.bin");
    p.emit_weight_const("W2", &[d, h, 1, 1], "@model_path/weights/w2.bin");
    p.emit_conv("h1", &[1, h, 1, s], "W1", "xn");
    p.emit_conv("h3", &[1, h, 1, s], "W3", "xn");
    p.emit_sigmoid("sig", &[1, h, 1, s], "h1");
    p.emit_mul("silu", &[1, h, 1, s], "h1", "sig");
    p.emit_mul("gate", &[1, h, 1, s], "silu", "h3");
    p.emit_conv("y", &[1, d, 1, s], "W2", "gate");

    // Concat output taps
    let out_ch = 2 * d + 3 * h;
    p.emit_scalar_const("cax", "int32", "1");
    p.emit_scalar_const("cid", "bool", "false");
    p.emit_concat(
        "out",
        &[1, out_ch, 1, s],
        "cax",
        "cid",
        &["y", "h1", "h3", "gate", "xn"],
    );

    let mil_text = p.finalize("out");

    let mut weights = WeightDict::new();
    weights.add(
        "@model_path/weights/rms2.bin",
        WeightBlob::from_rms_weights(rms_ffn),
    );
    weights.add("@model_path/weights/w1.bin", WeightBlob::from_f32(w1, h, d));
    weights.add("@model_path/weights/w3.bin", WeightBlob::from_f32(w3, h, d));
    weights.add("@model_path/weights/w2.bin", WeightBlob::from_f32(w2, d, h));

    KernelOutput {
        mil_text,
        weights,
        input_bytes: d * s * 2,
        output_bytes: out_ch * s * 2,
    }
}

/// Generate the FFN backward kernel.
///
/// Input: `[1, DIM + 2*HIDDEN, 1, SEQ]` = concat(dffn, h1, h3)
/// Output: `[1, DIM + 2*HIDDEN, 1, SEQ]` = concat(dx, dh1, dh3)
///
/// Weights: W2^T, W1^T, W3^T
pub fn gen_ffn_bwd(
    cfg: &TransformerKernelConfig,
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
) -> KernelOutput {
    let d = cfg.dim;
    let h = cfg.hidden_dim;
    let s = cfg.seq_len;
    let in_ch = d + 2 * h;

    let mut p = MilProgram::new(in_ch, s);
    p.emit_conv_constants();

    // Slice input: dffn [DIM], h1 [HIDDEN], h3 [HIDDEN]
    p.emit_tensor_const("bd", &[4], "int32", "[0,0,0,0]");
    p.emit_tensor_const("sd", &[4], "int32", &format!("[1,{d},1,{s}]"));
    p.emit_slice_by_size("dffn", &[1, d, 1, s], "x", "bd", "sd");

    p.emit_tensor_const("b1", &[4], "int32", &format!("[0,{d},0,0]"));
    p.emit_tensor_const("s1", &[4], "int32", &format!("[1,{h},1,{s}]"));
    p.emit_slice_by_size("h1", &[1, h, 1, s], "x", "b1", "s1");

    let b3_off = d + h;
    p.emit_tensor_const("b3", &[4], "int32", &format!("[0,{b3_off},0,0]"));
    p.emit_slice_by_size("h3", &[1, h, 1, s], "x", "b3", "s1");

    // W2^T @ dffn → dsilu
    p.emit_weight_const("W2t", &[h, d, 1, 1], "@model_path/weights/w2t.bin");
    p.emit_conv("dsilu", &[1, h, 1, s], "W2t", "dffn");

    // SiLU backward
    p.emit_sigmoid("sig", &[1, h, 1, s], "h1");
    p.emit_scalar_const("one", "fp16", "1.0");
    p.emit_sub("oms", &[1, h, 1, s], "one", "sig");
    p.emit_mul("homs", &[1, h, 1, s], "h1", "oms");
    p.emit_add("brk", &[1, h, 1, s], "one", "homs");
    p.emit_mul("dsd", &[1, h, 1, s], "sig", "brk");
    p.emit_mul("t1", &[1, h, 1, s], "dsilu", "h3");
    p.emit_mul("dh1", &[1, h, 1, s], "t1", "dsd");
    p.emit_mul("slh", &[1, h, 1, s], "h1", "sig");
    p.emit_mul("dh3", &[1, h, 1, s], "dsilu", "slh");

    // W1^T @ dh1, W3^T @ dh3, sum → dx
    p.emit_weight_const("W1t", &[d, h, 1, 1], "@model_path/weights/w1t.bin");
    p.emit_weight_const("W3t", &[d, h, 1, 1], "@model_path/weights/w3t.bin");
    p.emit_conv("dx1", &[1, d, 1, s], "W1t", "dh1");
    p.emit_conv("dx3", &[1, d, 1, s], "W3t", "dh3");
    p.emit_add("dx", &[1, d, 1, s], "dx1", "dx3");

    // Output concat
    p.emit_scalar_const("cax", "int32", "1");
    p.emit_scalar_const("cid", "bool", "false");
    p.emit_concat(
        "out",
        &[1, in_ch, 1, s],
        "cax",
        "cid",
        &["dx", "dh1", "dh3"],
    );

    let mil_text = p.finalize("out");

    let mut weights = WeightDict::new();
    weights.add(
        "@model_path/weights/w2t.bin",
        WeightBlob::from_f32_transposed(w2, d, h),
    );
    weights.add(
        "@model_path/weights/w1t.bin",
        WeightBlob::from_f32_transposed(w1, h, d),
    );
    weights.add(
        "@model_path/weights/w3t.bin",
        WeightBlob::from_f32_transposed(w3, h, d),
    );

    KernelOutput {
        mil_text,
        weights,
        input_bytes: in_ch * s * 2,
        output_bytes: in_ch * s * 2,
    }
}

/// Generate the SDPA backward part 1 kernel.
///
/// Input: `[1, Q_DIM+2*KV_DIM+DIM, 1, SEQ]` = concat(Q, K, V, dy)
/// Output: `[1, KV_DIM + 2*SCORE_CH, 1, SEQ]` = concat(dV, probs, dp)
///
/// Weights: Wo^T, causal mask
///
/// Note: Requires MHA (n_kv_heads == n_heads). GQA backward would need
/// K/V tiling and gradient accumulation, which is not yet implemented.
pub fn gen_sdpa_bwd1(cfg: &TransformerKernelConfig, wo: &[f32]) -> KernelOutput {
    let d = cfg.dim;
    let qd = cfg.q_dim(); // n_heads * head_dim (may differ from dim)
    let kvd = cfg.kv_dim(); // n_kv_heads * head_dim (= qd for MHA)
    let s = cfg.seq_len;
    let h = cfg.n_heads;
    let hd = cfg.head_dim;
    let sc_ch = cfg.score_ch();
    let sc = 1.0 / (hd as f32).sqrt();
    let in_ch = qd + 2 * kvd + d; // Q(qd) + K(kvd) + V(kvd) + dy(d)
    let out_ch = kvd + 2 * sc_ch; // dV(kvd) + probs + dp

    debug_assert_eq!(
        cfg.n_kv_heads, cfg.n_heads,
        "Backward SDPA kernels require MHA (n_kv_heads == n_heads)"
    );

    let mut p = MilProgram::new(in_ch, s);
    p.emit_conv_constants();

    // Slice inputs: Q(qd), K(kvd), V(kvd), dy(d)
    p.emit_tensor_const("szq", &[4], "int32", &format!("[1,{qd},1,{s}]"));
    p.emit_tensor_const("b0", &[4], "int32", "[0,0,0,0]");
    p.emit_slice_by_size("qf", &[1, qd, 1, s], "x", "b0", "szq");
    p.emit_tensor_const("szkv", &[4], "int32", &format!("[1,{kvd},1,{s}]"));
    p.emit_tensor_const("b1", &[4], "int32", &format!("[0,{qd},0,0]"));
    p.emit_slice_by_size("kf", &[1, kvd, 1, s], "x", "b1", "szkv");
    p.emit_tensor_const("b2", &[4], "int32", &format!("[0,{},0,0]", qd + kvd));
    p.emit_slice_by_size("vf", &[1, kvd, 1, s], "x", "b2", "szkv");
    p.emit_tensor_const("szd", &[4], "int32", &format!("[1,{d},1,{s}]"));
    p.emit_tensor_const("b3", &[4], "int32", &format!("[0,{},0,0]", qd + 2 * kvd));
    p.emit_slice_by_size("dx2f", &[1, d, 1, s], "x", "b3", "szd");

    // Wo^T @ dy → df (attention gradient)
    // Wo is [dim, q_dim], so Wo^T is [q_dim, dim]
    p.emit_weight_const("Wot", &[qd, d, 1, 1], "@model_path/weights/wot.bin");
    p.emit_conv("df", &[1, qd, 1, s], "Wot", "dx2f");

    // Reshape to multi-head format [1, n_heads, head_dim, seq]
    // MHA: qd = kvd = n_heads * head_dim, so one reshape const works for all
    p.emit_tensor_const("rsh", &[4], "int32", &format!("[1,{h},{hd},{s}]"));
    p.emit_tensor_const("pm", &[4], "int32", "[0,1,3,2]");

    for (name, src) in &[("q", "qf"), ("k", "kf"), ("v", "vf"), ("da", "df")] {
        let r_name = format!("{name}r");
        p.emit_reshape(&r_name, &[1, h, hd, s], "rsh", src);
        p.emit_transpose(name, &[1, h, s, hd], "pm", &r_name);
    }

    // Recompute attention scores + softmax
    p.emit_scalar_const("bF", "bool", "false");
    p.emit_scalar_const("bT", "bool", "true");
    p.emit_matmul("sc1", &[1, h, s, s], "bF", "bT", "q", "k");
    p.emit_scalar_const("scv", "fp16", &format!("{sc}"));
    p.emit_mul("sc2", &[1, h, s, s], "sc1", "scv");
    p.emit_weight_const("cm", &[1, 1, s, s], "@model_path/weights/mask.bin");
    p.emit_add("ms", &[1, h, s, s], "sc2", "cm");
    p.emit_scalar_const("sax", "int32", "-1");
    p.emit_softmax("probs", &[1, h, s, s], "sax", "ms");

    // dV = probs^T @ da, dp = da @ V^T
    p.emit_matmul("dv4", &[1, h, s, hd], "bT", "bF", "probs", "da");
    p.emit_matmul("dp4", &[1, h, s, s], "bF", "bT", "da", "v");

    // Reshape outputs to flat format
    // dV has kv_dim channels (= q_dim for MHA)
    p.emit_transpose("dvt", &[1, h, hd, s], "pm", "dv4");
    p.emit_tensor_const("dvs", &[4], "int32", &format!("[1,{kvd},1,{s}]"));
    p.emit_reshape("dvf", &[1, kvd, 1, s], "dvs", "dvt");

    p.emit_tensor_const("scs", &[4], "int32", &format!("[1,{sc_ch},1,{s}]"));
    p.emit_reshape("pf", &[1, sc_ch, 1, s], "scs", "probs");
    p.emit_reshape("dpf", &[1, sc_ch, 1, s], "scs", "dp4");

    // Output concat
    p.emit_scalar_const("cax", "int32", "1");
    p.emit_scalar_const("cid", "bool", "false");
    p.emit_concat(
        "out",
        &[1, out_ch, 1, s],
        "cax",
        "cid",
        &["dvf", "pf", "dpf"],
    );

    let mil_text = p.finalize("out");

    let mut weights = WeightDict::new();
    weights.add(
        "@model_path/weights/wot.bin",
        // Wo is [dim, q_dim], transposed to [q_dim, dim]
        WeightBlob::from_f32_transposed(wo, d, qd),
    );
    weights.add("@model_path/weights/mask.bin", build_causal_mask(s));

    KernelOutput {
        mil_text,
        weights,
        input_bytes: in_ch * s * 2,
        output_bytes: out_ch * s * 2,
    }
}

/// Generate the SDPA backward part 2 kernel (weight-free).
///
/// Input: `[1, 2*SCORE_CH + Q_DIM + KV_DIM, 1, SEQ]` = concat(probs, dp, Q, K)
/// Output: `[1, Q_DIM + KV_DIM, 1, SEQ]` = concat(dQ, dK)
///
/// No weights — compiled once and shared across layers.
///
/// Note: Requires MHA (n_kv_heads == n_heads). GQA backward would need
/// gradient accumulation from n_heads back to n_kv_heads.
pub fn gen_sdpa_bwd2(cfg: &TransformerKernelConfig) -> KernelOutput {
    let qd = cfg.q_dim(); // n_heads * head_dim
    let kvd = cfg.kv_dim(); // n_kv_heads * head_dim (= qd for MHA)
    let s = cfg.seq_len;
    let h = cfg.n_heads;
    let hd = cfg.head_dim;
    let sc_ch = cfg.score_ch();
    let sc = 1.0 / (hd as f32).sqrt();
    let in_ch = 2 * sc_ch + qd + kvd;
    let out_ch = qd + kvd;

    debug_assert_eq!(
        cfg.n_kv_heads, cfg.n_heads,
        "Backward SDPA kernels require MHA (n_kv_heads == n_heads)"
    );

    let mut p = MilProgram::new(in_ch, s);

    // Slice inputs: probs(sc_ch), dp(sc_ch), Q(qd), K(kvd)
    p.emit_tensor_const("sz_sc", &[4], "int32", &format!("[1,{sc_ch},1,{s}]"));
    p.emit_tensor_const("b0", &[4], "int32", "[0,0,0,0]");
    p.emit_slice_by_size("pf", &[1, sc_ch, 1, s], "x", "b0", "sz_sc");
    p.emit_tensor_const("b1", &[4], "int32", &format!("[0,{sc_ch},0,0]"));
    p.emit_slice_by_size("dpf", &[1, sc_ch, 1, s], "x", "b1", "sz_sc");
    p.emit_tensor_const("szq", &[4], "int32", &format!("[1,{qd},1,{s}]"));
    p.emit_tensor_const("b2", &[4], "int32", &format!("[0,{},0,0]", 2 * sc_ch));
    p.emit_slice_by_size("qf", &[1, qd, 1, s], "x", "b2", "szq");
    p.emit_tensor_const("szkv", &[4], "int32", &format!("[1,{kvd},1,{s}]"));
    p.emit_tensor_const("b3", &[4], "int32", &format!("[0,{},0,0]", 2 * sc_ch + qd));
    p.emit_slice_by_size("kf", &[1, kvd, 1, s], "x", "b3", "szkv");

    // Reshape to multi-head
    p.emit_tensor_const("ssh", &[4], "int32", &format!("[1,{h},{s},{s}]"));
    p.emit_reshape("probs", &[1, h, s, s], "ssh", "pf");
    p.emit_reshape("dp", &[1, h, s, s], "ssh", "dpf");

    // MHA: qd = kvd = h*hd, so one reshape const works for both Q and K
    p.emit_tensor_const("rsh", &[4], "int32", &format!("[1,{h},{hd},{s}]"));
    p.emit_tensor_const("pm", &[4], "int32", "[0,1,3,2]");
    p.emit_reshape("qr", &[1, h, hd, s], "rsh", "qf");
    p.emit_transpose("q", &[1, h, s, hd], "pm", "qr");
    p.emit_reshape("kr", &[1, h, hd, s], "rsh", "kf");
    p.emit_transpose("k", &[1, h, s, hd], "pm", "kr");

    // Softmax gradient: ds = probs * (dp - sum(probs * dp)) * scale
    p.emit_mul("pdp", &[1, h, s, s], "probs", "dp");
    p.emit_tensor_const("rax", &[1], "int32", "[-1]");
    p.emit_scalar_const("kd", "bool", "true");
    p.emit_reduce_sum("spdp", &[1, h, s, 1], "pdp", "rax", "kd");
    p.emit_sub("dps", &[1, h, s, s], "dp", "spdp");
    p.emit_mul("ds0", &[1, h, s, s], "probs", "dps");
    p.emit_scalar_const("scv", "fp16", &format!("{sc}"));
    p.emit_mul("ds", &[1, h, s, s], "ds0", "scv");

    // dQ = ds @ K, dK = ds^T @ Q
    p.emit_scalar_const("bF", "bool", "false");
    p.emit_scalar_const("bT", "bool", "true");
    p.emit_matmul("dq4", &[1, h, s, hd], "bF", "bF", "ds", "k");
    p.emit_matmul("dk4", &[1, h, s, hd], "bT", "bF", "ds", "q");

    // Reshape back to flat
    p.emit_transpose("dqt", &[1, h, hd, s], "pm", "dq4");
    p.emit_transpose("dkt", &[1, h, hd, s], "pm", "dk4");
    p.emit_tensor_const("fsq", &[4], "int32", &format!("[1,{qd},1,{s}]"));
    p.emit_reshape("dqf", &[1, qd, 1, s], "fsq", "dqt");
    p.emit_tensor_const("fskv", &[4], "int32", &format!("[1,{kvd},1,{s}]"));
    p.emit_reshape("dkf", &[1, kvd, 1, s], "fskv", "dkt");

    // Output concat
    p.emit_scalar_const("cax", "int32", "1");
    p.emit_scalar_const("cid", "bool", "false");
    p.emit_concat("out", &[1, out_ch, 1, s], "cax", "cid", &["dqf", "dkf"]);

    let mil_text = p.finalize("out");

    KernelOutput {
        mil_text,
        weights: WeightDict::new(),
        input_bytes: in_ch * s * 2,
        output_bytes: out_ch * s * 2,
    }
}

/// Generate the QKV backward kernel.
///
/// Input: `[1, Q_DIM + 2*KV_DIM, 1, SEQ]` = concat(dQ, dK, dV)
/// Output: `[1, DIM, 1, SEQ]` = dx (sum of three transposed projections)
///
/// Weights: Wq^T, Wk^T, Wv^T
///
/// Wq is [q_dim, dim], so Wq^T is [dim, q_dim]. Conv(Wq^T, dQ) → [dim, s].
/// Same pattern for Wk^T [dim, kv_dim] and Wv^T [dim, kv_dim].
pub fn gen_qkv_bwd(
    cfg: &TransformerKernelConfig,
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
) -> KernelOutput {
    let d = cfg.dim;
    let qd = cfg.q_dim(); // n_heads * head_dim
    let kvd = cfg.kv_dim(); // n_kv_heads * head_dim
    let s = cfg.seq_len;
    let in_ch = qd + 2 * kvd; // dQ(qd) + dK(kvd) + dV(kvd)

    let mut p = MilProgram::new(in_ch, s);
    p.emit_conv_constants();

    // Slice inputs: dQ(qd), dK(kvd), dV(kvd)
    p.emit_tensor_const("szq", &[4], "int32", &format!("[1,{qd},1,{s}]"));
    p.emit_tensor_const("b0", &[4], "int32", "[0,0,0,0]");
    p.emit_slice_by_size("dq", &[1, qd, 1, s], "x", "b0", "szq");
    p.emit_tensor_const("szkv", &[4], "int32", &format!("[1,{kvd},1,{s}]"));
    p.emit_tensor_const("b1", &[4], "int32", &format!("[0,{qd},0,0]"));
    p.emit_slice_by_size("dk", &[1, kvd, 1, s], "x", "b1", "szkv");
    p.emit_tensor_const("b2", &[4], "int32", &format!("[0,{},0,0]", qd + kvd));
    p.emit_slice_by_size("dv", &[1, kvd, 1, s], "x", "b2", "szkv");

    // Transposed weight projections
    // Wq^T: [dim, q_dim, 1, 1], Wk^T: [dim, kv_dim, 1, 1], Wv^T: [dim, kv_dim, 1, 1]
    p.emit_weight_const("Wqt", &[d, qd, 1, 1], "@model_path/weights/wqt.bin");
    p.emit_weight_const("Wkt", &[d, kvd, 1, 1], "@model_path/weights/wkt.bin");
    p.emit_weight_const("Wvt", &[d, kvd, 1, 1], "@model_path/weights/wvt.bin");
    p.emit_conv("dxq", &[1, d, 1, s], "Wqt", "dq");
    p.emit_conv("dxk", &[1, d, 1, s], "Wkt", "dk");
    p.emit_conv("dxv", &[1, d, 1, s], "Wvt", "dv");

    // Sum all contributions to get dx
    p.emit_add("dxqk", &[1, d, 1, s], "dxq", "dxk");
    p.emit_add("out", &[1, d, 1, s], "dxqk", "dxv");

    let mil_text = p.finalize("out");

    let mut weights = WeightDict::new();
    weights.add(
        "@model_path/weights/wqt.bin",
        // Wq is [q_dim, dim], transposed to [dim, q_dim]
        WeightBlob::from_f32_transposed(wq, qd, d),
    );
    weights.add(
        "@model_path/weights/wkt.bin",
        // Wk is [kv_dim, dim], transposed to [dim, kv_dim]
        WeightBlob::from_f32_transposed(wk, kvd, d),
    );
    weights.add(
        "@model_path/weights/wvt.bin",
        // Wv is [kv_dim, dim], transposed to [dim, kv_dim]
        WeightBlob::from_f32_transposed(wv, kvd, d),
    );

    KernelOutput {
        mil_text,
        weights,
        input_bytes: in_ch * s * 2,
        output_bytes: d * s * 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> TransformerKernelConfig {
        TransformerKernelConfig {
            dim: 768,
            hidden_dim: 2048,
            n_heads: 12,
            n_kv_heads: 12,
            head_dim: 64,
            seq_len: 256,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    #[test]
    fn test_weight_blob_header() {
        let weights = vec![1.0f32; 4];
        let blob = WeightBlob::from_f32(&weights, 2, 2);

        assert_eq!(blob.len(), 128 + 8); // 4 elements * 2 bytes
        assert_eq!(blob[0], 0x01);
        assert_eq!(blob[4], 0x02);
        assert_eq!(blob[64], 0xEF);
        assert_eq!(blob[65], 0xBE);
        assert_eq!(blob[66], 0xAD);
        assert_eq!(blob[67], 0xDE);
        assert_eq!(blob[68], 0x01);
        assert_eq!(
            u32::from_le_bytes([blob[72], blob[73], blob[74], blob[75]]),
            8
        );
        assert_eq!(
            u32::from_le_bytes([blob[80], blob[81], blob[82], blob[83]]),
            128
        );
    }

    #[test]
    fn test_causal_mask() {
        let blob = build_causal_mask(4);
        // 128 header + 4*4*2 = 128 + 32 = 160
        assert_eq!(blob.len(), 160);
    }

    #[test]
    fn test_sdpa_fwd_infer_generates_valid_mil() {
        let cfg = test_config();
        let d = cfg.dim;
        let kv_d = cfg.kv_dim();
        let zeros_d = vec![0.0f32; d];
        let zeros_dd = vec![0.0f32; d * d];
        let zeros_kvd = vec![0.0f32; kv_d * d];

        let output = gen_sdpa_fwd(&cfg, &zeros_d, &zeros_dd, &zeros_kvd, &zeros_kvd, &zeros_dd);

        assert!(output.mil_text.contains("program(1.3)"));
        assert!(output.mil_text.contains("func main<ios18>"));
        assert!(
            !output.mil_text.contains("concat"),
            "inference kernel must not contain concat"
        );
        assert!(output.mil_text.contains("softmax"));
        assert!(output.mil_text.contains("} -> (oo);"));
        assert_eq!(output.input_bytes, d * 256 * 2);
        assert_eq!(output.output_bytes, d * 256 * 2);
        assert_eq!(output.weights.entries.len(), 6); // rms1, wq, wk, wv, wo, mask
    }

    #[test]
    fn test_ffn_fwd_infer_generates_valid_mil() {
        let cfg = test_config();
        let d = cfg.dim;
        let h = cfg.hidden_dim;
        let zeros_d = vec![0.0f32; d];
        let zeros_hd = vec![0.0f32; h * d];
        let zeros_dh = vec![0.0f32; d * h];

        let output = gen_ffn_fwd(&cfg, &zeros_d, &zeros_hd, &zeros_hd, &zeros_dh);

        assert!(output.mil_text.contains("sigmoid"));
        assert!(
            !output.mil_text.contains("concat"),
            "inference kernel must not contain concat"
        );
        assert!(output.mil_text.contains("} -> (y);"));
        assert_eq!(output.input_bytes, d * 256 * 2);
        assert_eq!(output.output_bytes, d * 256 * 2);
        assert_eq!(output.weights.entries.len(), 4); // rms2, w1, w3, w2
    }

    #[test]
    fn test_sdpa_fwd_taps_generates_valid_mil() {
        let cfg = test_config();
        let d = cfg.dim;
        let kv_d = cfg.kv_dim();
        let zeros_d = vec![0.0f32; d];
        let zeros_dd = vec![0.0f32; d * d];
        let zeros_kvd = vec![0.0f32; kv_d * d];

        let output =
            gen_sdpa_fwd_taps(&cfg, &zeros_d, &zeros_dd, &zeros_kvd, &zeros_kvd, &zeros_dd);

        assert!(output.mil_text.contains("program(1.3)"));
        assert!(output.mil_text.contains("func main<ios18>"));
        assert!(output.mil_text.contains("concat"));
        assert!(output.mil_text.contains("softmax"));
        assert!(output.mil_text.contains("} -> (out);"));
        assert_eq!(output.input_bytes, d * 256 * 2);
        let expected_out_ch = 4 * d + 2 * kv_d;
        assert_eq!(output.output_bytes, expected_out_ch * 256 * 2);
    }

    #[test]
    fn test_ffn_fwd_taps_generates_valid_mil() {
        let cfg = test_config();
        let d = cfg.dim;
        let h = cfg.hidden_dim;
        let zeros_d = vec![0.0f32; d];
        let zeros_hd = vec![0.0f32; h * d];
        let zeros_dh = vec![0.0f32; d * h];

        let output = gen_ffn_fwd_taps(&cfg, &zeros_d, &zeros_hd, &zeros_hd, &zeros_dh);

        assert!(output.mil_text.contains("sigmoid"));
        assert!(output.mil_text.contains("concat"));
        assert_eq!(output.output_bytes, (2 * d + 3 * h) * 256 * 2);
    }

    #[test]
    fn test_ffn_bwd_generates_valid_mil() {
        let cfg = test_config();
        let d = cfg.dim;
        let h = cfg.hidden_dim;
        let zeros_hd = vec![0.0f32; h * d];
        let zeros_dh = vec![0.0f32; d * h];

        let output = gen_ffn_bwd(&cfg, &zeros_hd, &zeros_dh, &zeros_hd);

        assert!(output.mil_text.contains("slice_by_size"));
        assert!(output.mil_text.contains("sigmoid"));
        assert!(output.weights.entries.len() == 3); // W2^T, W1^T, W3^T
    }

    #[test]
    fn test_sdpa_bwd2_weight_free() {
        let cfg = test_config();
        let output = gen_sdpa_bwd2(&cfg);

        assert!(output.weights.entries.is_empty());
        assert!(output.mil_text.contains("reduce_sum"));
        assert!(output.mil_text.contains("matmul"));
    }

    #[test]
    fn test_qkv_bwd_generates_valid_mil() {
        let cfg = test_config();
        let d = cfg.dim;
        let zeros_dd = vec![0.0f32; d * d];

        let output = gen_qkv_bwd(&cfg, &zeros_dd, &zeros_dd, &zeros_dd);

        assert!(output.mil_text.contains("slice_by_size"));
        assert_eq!(output.weights.entries.len(), 3);
        assert_eq!(output.output_bytes, d * 256 * 2);
    }

    #[test]
    fn test_sdpa_fwd_kv_generates_valid_mil() {
        let cfg = test_config();
        let d = cfg.dim;
        let hd = cfg.head_dim;
        let kv_d = cfg.kv_dim();
        let zeros_d = vec![0.0f32; d];
        let zeros_dd = vec![0.0f32; d * d];
        let zeros_kvd = vec![0.0f32; kv_d * d];
        let ones_hd = vec![1.0f32; hd];

        let output = gen_sdpa_fwd_kv(
            &cfg, &zeros_d, &zeros_dd, &zeros_kvd, &zeros_kvd, &zeros_dd, &ones_hd, &ones_hd,
        );

        assert!(output.mil_text.contains("program(1.3)"));
        assert!(output.mil_text.contains("concat"));
        assert!(output.mil_text.contains("softmax"));
        assert!(output.mil_text.contains("} -> (out);"));
        assert_eq!(output.input_bytes, d * 256 * 2);
        let expected_out_ch = d + 2 * kv_d;
        assert_eq!(output.output_bytes, expected_out_ch * 256 * 2);
        // 10 weights: rms1, wq, wk, wv, wo, mask, qnorm, knorm, cos, sin
        assert_eq!(output.weights.entries.len(), 10);
    }

    #[test]
    fn test_sdpa_fwd_kv_gqa() {
        // GQA config: 12 Q heads, 4 KV heads
        let cfg = TransformerKernelConfig {
            dim: 768,
            hidden_dim: 2048,
            n_heads: 12,
            n_kv_heads: 4,
            head_dim: 64,
            seq_len: 256,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        };
        let d = cfg.dim;
        let hd = cfg.head_dim;
        let kv_d = cfg.kv_dim(); // 4 * 64 = 256
        assert_eq!(kv_d, 256);

        let zeros_d = vec![0.0f32; d];
        let zeros_dd = vec![0.0f32; d * d];
        let zeros_kvd = vec![0.0f32; kv_d * d];
        let ones_hd = vec![1.0f32; hd];

        let output = gen_sdpa_fwd_kv(
            &cfg, &zeros_d, &zeros_dd, &zeros_kvd, &zeros_kvd, &zeros_dd, &ones_hd, &ones_hd,
        );

        assert!(output.mil_text.contains("tile"));
        let expected_out_ch = d + 2 * kv_d; // 768 + 512 = 1280
        assert_eq!(output.output_bytes, expected_out_ch * 256 * 2);
    }

    #[test]
    fn test_sdpa_fwd_gqa() {
        // GQA config: 12 Q heads, 4 KV heads
        let cfg = TransformerKernelConfig {
            dim: 768,
            hidden_dim: 2048,
            n_heads: 12,
            n_kv_heads: 4,
            head_dim: 64,
            seq_len: 256,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        };
        let d = cfg.dim;
        let kv_d = cfg.kv_dim();
        let zeros_d = vec![0.0f32; d];
        let zeros_dd = vec![0.0f32; d * d];
        let zeros_kvd = vec![0.0f32; kv_d * d];

        let output = gen_sdpa_fwd(&cfg, &zeros_d, &zeros_dd, &zeros_kvd, &zeros_kvd, &zeros_dd);

        assert!(output.mil_text.contains("tile"));
        assert!(!output.mil_text.contains("concat"));
        assert_eq!(output.output_bytes, d * 256 * 2);
    }

    #[test]
    fn test_transposed_blob() {
        // 2x3 matrix
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let blob = WeightBlob::from_f32_transposed(&w, 2, 3);

        // Read back fp16 data: should be 3x2 transposed
        let fp16_ptr = &blob[128..] as *const [u8] as *const u16;
        let fp16 = unsafe { std::slice::from_raw_parts(fp16_ptr, 6) };

        // Transposed: [1,4, 2,5, 3,6]
        assert_eq!(half::f16::from_bits(fp16[0]).to_f32(), 1.0);
        assert_eq!(half::f16::from_bits(fp16[1]).to_f32(), 4.0);
        assert_eq!(half::f16::from_bits(fp16[2]).to_f32(), 2.0);
        assert_eq!(half::f16::from_bits(fp16[3]).to_f32(), 5.0);
    }

    // ================================================================
    // Non-standard head_dim tests (Qwen3-style: head_dim != dim/n_heads)
    // ================================================================

    /// Qwen3-0.6B style config where q_dim (2048) != dim (1024).
    fn qwen3_config() -> TransformerKernelConfig {
        TransformerKernelConfig {
            dim: 1024,        // hidden_size
            hidden_dim: 3072, // intermediate_size
            n_heads: 16,      // num_attention_heads
            n_kv_heads: 16,   // MHA for Qwen3-0.6B
            head_dim: 128,    // explicit head_dim → q_dim = 16*128 = 2048
            seq_len: 64,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    #[test]
    fn test_qwen3_config_dimensions() {
        let cfg = qwen3_config();
        assert_eq!(cfg.q_dim(), 2048); // 16 * 128
        assert_eq!(cfg.kv_dim(), 2048); // 16 * 128 (MHA)
        assert_ne!(cfg.q_dim(), cfg.dim); // Key invariant: q_dim != dim
        assert_eq!(cfg.sdpa_fwd_output_ch(), 2 * 1024 + 2 * 2048 + 2 * 2048);
        assert_eq!(cfg.sdpa_bwd1_input_ch(), 2048 + 2 * 2048 + 1024);
        assert_eq!(cfg.sdpa_bwd2_input_ch(), 2 * (16 * 64) + 2048 + 2048);
    }

    #[test]
    fn test_qwen3_sdpa_fwd_taps_shapes() {
        let cfg = qwen3_config();
        let d = cfg.dim;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let wq = vec![0.01f32; qd * d];
        let wk = vec![0.01f32; kvd * d];
        let wv = vec![0.01f32; kvd * d];
        let wo = vec![0.01f32; d * qd];
        let rms = vec![0.01f32; d];

        let out = gen_sdpa_fwd_taps(&cfg, &rms, &wq, &wk, &wv, &wo);
        assert!(out.mil_text.contains("program(1.3)"));
        assert!(out.mil_text.contains("func main<ios18>"));
        // Output should have the correct total channels
        let expected_out_ch = cfg.sdpa_fwd_output_ch();
        assert_eq!(out.output_bytes, expected_out_ch * cfg.seq_len * 2);
    }

    #[test]
    fn test_qwen3_sdpa_bwd1_shapes() {
        let cfg = qwen3_config();
        let d = cfg.dim;
        let qd = cfg.q_dim();
        let wo = vec![0.01f32; d * qd]; // Wo: [dim, q_dim]

        let out = gen_sdpa_bwd1(&cfg, &wo);
        assert!(out.mil_text.contains("program(1.3)"));
        // Input: Q(qd) + K(kvd) + V(kvd) + dy(d)
        let expected_in_ch = cfg.sdpa_bwd1_input_ch();
        assert_eq!(out.input_bytes, expected_in_ch * cfg.seq_len * 2);
        // Output: dV(kvd) + 2*score_ch
        let expected_out_ch = cfg.sdpa_bwd1_output_ch();
        assert_eq!(out.output_bytes, expected_out_ch * cfg.seq_len * 2);
        // Wo^T weight blob should be q_dim * dim * 2 (fp16) + 128 header
        let wot_blob = &out.weights.entries.iter()
            .find(|(k, _)| k.contains("wot.bin")).unwrap().1;
        assert_eq!(wot_blob.len(), 128 + qd * d * 2);
    }

    #[test]
    fn test_qwen3_sdpa_bwd2_shapes() {
        let cfg = qwen3_config();
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();

        let out = gen_sdpa_bwd2(&cfg);
        assert!(out.mil_text.contains("program(1.3)"));
        // Input: probs + dp + Q(qd) + K(kvd)
        let expected_in_ch = cfg.sdpa_bwd2_input_ch();
        assert_eq!(out.input_bytes, expected_in_ch * cfg.seq_len * 2);
        // Output: dQ(qd) + dK(kvd)
        assert_eq!(out.output_bytes, (qd + kvd) * cfg.seq_len * 2);
        // No weights
        assert!(out.weights.entries.is_empty());
    }

    #[test]
    fn test_qwen3_qkv_bwd_shapes() {
        let cfg = qwen3_config();
        let d = cfg.dim;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let wq = vec![0.01f32; qd * d]; // Wq: [q_dim, dim]
        let wk = vec![0.01f32; kvd * d]; // Wk: [kv_dim, dim]
        let wv = vec![0.01f32; kvd * d]; // Wv: [kv_dim, dim]

        let out = gen_qkv_bwd(&cfg, &wq, &wk, &wv);
        assert!(out.mil_text.contains("program(1.3)"));
        // Input: dQ(qd) + dK(kvd) + dV(kvd)
        let expected_in_ch = qd + 2 * kvd;
        assert_eq!(out.input_bytes, expected_in_ch * cfg.seq_len * 2);
        // Output: dx (dim)
        assert_eq!(out.output_bytes, d * cfg.seq_len * 2);
        // Weight blobs: Wq^T (qd*d), Wk^T (kvd*d), Wv^T (kvd*d)
        let wqt = &out.weights.entries.iter()
            .find(|(k, _)| k.contains("wqt.bin")).unwrap().1;
        assert_eq!(wqt.len(), 128 + qd * d * 2);
        let wkt = &out.weights.entries.iter()
            .find(|(k, _)| k.contains("wkt.bin")).unwrap().1;
        assert_eq!(wkt.len(), 128 + kvd * d * 2);
    }

    #[test]
    fn test_qwen3_sdpa_fwd_kv_shapes() {
        let cfg = qwen3_config();
        let d = cfg.dim;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let hd = cfg.head_dim;
        let rms = vec![0.01f32; d];
        let wq = vec![0.01f32; qd * d];
        let wk = vec![0.01f32; kvd * d];
        let wv = vec![0.01f32; kvd * d];
        let wo = vec![0.01f32; d * qd];
        let q_norm = vec![0.01f32; hd];
        let k_norm = vec![0.01f32; hd];

        let out = gen_sdpa_fwd_kv(&cfg, &rms, &wq, &wk, &wv, &wo, &q_norm, &k_norm);
        assert!(out.mil_text.contains("program(1.3)"));
        // Output: o_out(dim) + k_cache(kv_dim) + v_cache(kv_dim) = d + 2*kvd
        let expected_out_ch = d + 2 * kvd;
        assert_eq!(out.output_bytes, expected_out_ch * cfg.seq_len * 2);
    }
}
