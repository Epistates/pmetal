//! Building blocks shared by the ANE kernel generators: the transformer shape
//! config, the weight blob format (fp16 and per-row int8), and RoPE tables
//! and MIL.

use crate::ane::mil::MilProgram;
use crate::ane::runtime::WeightDict;
use zerocopy::{FromBytes, IntoBytes};

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
        let fp16_slice: &mut [u16] = <[u16]>::mut_from_bytes(&mut blob[128..128 + data_size])
            .expect("blob data region is u16-aligned (offset 128)");
        crate::neon_convert::f32_to_f16_bulk(weights, fp16_slice);

        blob
    }

    /// Build a weight blob from raw fp16 data (no conversion).
    pub fn from_fp16(fp16_data: &[u16]) -> Vec<u8> {
        let data_size = fp16_data.len() * 2;
        let total = 128 + data_size;
        let mut blob = vec![0u8; total];

        write_header(&mut blob, data_size);

        // Copy raw fp16 data
        blob[128..128 + data_size].copy_from_slice(fp16_data.as_bytes());

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

        let fp16_slice: &mut [u16] = <[u16]>::mut_from_bytes(&mut blob[128..128 + data_size])
            .expect("blob data region is u16-aligned (offset 128)");
        crate::neon_convert::f32_to_f16_bulk(weights, fp16_slice);

        blob
    }
}

/// Blob dtype codes, the byte after the magic (as CoreML's `weight.bin` uses
/// them).
const BLOB_FP16: u8 = 1;
const BLOB_INT8: u8 = 4;

/// Quantize `[rows, cols]` row-major weights to int8, symmetric per row (per
/// output channel): `w[r, c] ≈ scale[r] * q[r, c]`. The scale is rounded to
/// fp16 before quantizing, so these are exactly the weights the ANE computes
/// with.
pub fn quantize_int8_rows(weights: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<half::f16>) {
    debug_assert_eq!(weights.len(), rows * cols);
    let mut q = vec![0i8; rows * cols];
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        let max = row.iter().fold(0f32, |m, w| m.max(w.abs()));
        let scale = half::f16::from_f32(if max > 0.0 { max / 127.0 } else { 1.0 });
        let inv = 1.0 / scale.to_f32();
        for (dst, w) in q[r * cols..(r + 1) * cols].iter_mut().zip(row) {
            *dst = (w * inv).round().clamp(-127.0, 127.0) as i8;
        }
        scales.push(scale);
    }
    (q, scales)
}

impl WeightBlob {
    /// int8 blobs for `[rows, cols]` weights quantized per row by
    /// [`quantize_int8_rows`]: the data, then the `[rows]` fp16 scales.
    pub fn int8_per_row(weights: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
        let (q, scales) = quantize_int8_rows(weights, rows, cols);
        let mut data = vec![0u8; 128 + q.len()];
        write_typed_header(&mut data, q.len(), BLOB_INT8);
        data[128..].copy_from_slice(q.as_bytes());
        let bits: Vec<u16> = scales.iter().map(|s| s.to_bits()).collect();
        (data, Self::from_fp16(&bits))
    }
}

/// Write the 128-byte blob header for fp16 data.
fn write_header(blob: &mut [u8], data_size: usize) {
    write_typed_header(blob, data_size, BLOB_FP16);
}

/// Write the 128-byte blob header for data of the given dtype code.
fn write_typed_header(blob: &mut [u8], data_size: usize, dtype: u8) {
    blob[0] = 0x01;
    blob[4] = 0x02;
    // Magic: 0xDEADBEEF little-endian
    blob[64] = 0xEF;
    blob[65] = 0xBE;
    blob[66] = 0xAD;
    blob[67] = 0xDE;
    blob[68] = dtype;
    // Data size (uint32 LE)
    let ds = data_size as u32;
    blob[72..76].copy_from_slice(&ds.to_le_bytes());
    // Data offset (uint32 LE) = 128
    blob[80..84].copy_from_slice(&128u32.to_le_bytes());
}

/// Build precomputed cos/sin RoPE tables as weight blobs.
///
/// Returns `(cos_blob, sin_blob)` each of shape `[1, 1, half_dim, seq_len]`.
/// Uses non-traditional split-half RoPE frequencies:
/// `inv_freq[d] = 1 / rope_theta^(2d / head_dim)` for `d in 0..half_dim`.
pub(crate) fn build_rope_tables(
    head_dim: usize,
    seq_len: usize,
    rope_theta: f32,
) -> (Vec<u8>, Vec<u8>) {
    let half_dim = head_dim / 2;
    let n = half_dim * seq_len;
    let mut cos_data = vec![0.0f32; n];
    let mut sin_data = vec![0.0f32; n];

    for d in 0..half_dim {
        let inv_freq = rope_inv_freq(d, head_dim, rope_theta);
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

/// RoPE's angular frequency for rotation pair `d` (`d < head_dim / 2`).
fn rope_inv_freq(d: usize, head_dim: usize, rope_theta: f32) -> f32 {
    1.0 / rope_theta.powf(2.0 * d as f32 / head_dim as f32)
}

/// Apply RoPE in place on the CPU to `x`, channel-first
/// `[n_heads * head_dim, seq]` with position `t` in column `t`: the same
/// split-half rotation [`emit_rope`] applies on the ANE.
///
/// `inverse` rotates by −θ. The rotation is orthogonal, so that is also its
/// transpose, which is how a gradient goes back through it.
pub(crate) fn rope_channel_first(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    rope_theta: f32,
    inverse: bool,
) {
    debug_assert_eq!(x.len(), n_heads * head_dim * seq);
    let half = head_dim / 2;
    for d in 0..half {
        let inv_freq = rope_inv_freq(d, head_dim, rope_theta);
        for t in 0..seq {
            let (sin, cos) = (t as f32 * inv_freq).sin_cos();
            let sin = if inverse { -sin } else { sin };
            for h in 0..n_heads {
                let first = (h * head_dim + d) * seq + t;
                let second = first + half * seq;
                let (a, b) = (x[first], x[second]);
                x[first] = a * cos - b * sin;
                x[second] = a * sin + b * cos;
            }
        }
    }
}

/// Emit MIL for non-traditional split-half RoPE on `[1, n_heads, head_dim, seq]`.
///
/// Splits the head_dim axis into first/second halves, applies rotation using
/// precomputed cos/sin tables of shape `[1, 1, half_dim, seq]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_rope(
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
    emit_rope_with(p, input, output, n_heads, head_dim, seq_len, &cos_w, &sin_w);
}

/// [`emit_rope`] with the cos/sin tables already in the program as `cos_w`
/// and `sin_w`, each `[1, 1, head_dim / 2, seq]`: constants, or inputs when
/// the positions change from call to call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_rope_with(
    p: &mut MilProgram,
    input: &str,
    output: &str,
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
    cos_w: &str,
    sin_w: &str,
) {
    let half_dim = head_dim / 2;
    let pfx = p.next_var("rope");

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
    p.emit_mul(&fc, &[1, n_heads, half_dim, seq_len], &x_first, cos_w);
    let ss = format!("{pfx}_ss");
    p.emit_mul(&ss, &[1, n_heads, half_dim, seq_len], &x_second, sin_w);
    let rot_first = format!("{pfx}_rf");
    p.emit_sub(&rot_first, &[1, n_heads, half_dim, seq_len], &fc, &ss);

    // rot_second = x_first * sin + x_second * cos
    let fs = format!("{pfx}_fs");
    p.emit_mul(&fs, &[1, n_heads, half_dim, seq_len], &x_first, sin_w);
    let sc = format!("{pfx}_sc");
    p.emit_mul(&sc, &[1, n_heads, half_dim, seq_len], &x_second, cos_w);
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// `[n_heads * head_dim, seq]` with every entry distinct.
    fn rope_input(n_heads: usize, head_dim: usize, seq: usize) -> Vec<f32> {
        (0..n_heads * head_dim * seq)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.02)
            .collect()
    }

    #[test]
    fn rope_inverse_undoes_the_rotation_and_position_zero_is_untouched() {
        let (nh, hd, s) = (3, 8, 5);
        let x = rope_input(nh, hd, s);
        let mut y = x.clone();
        rope_channel_first(&mut y, nh, hd, s, 10_000.0, false);
        assert_ne!(y, x);
        for ch in 0..nh * hd {
            assert_eq!(y[ch * s], x[ch * s], "position 0 rotates by 0");
        }
        rope_channel_first(&mut y, nh, hd, s, 10_000.0, true);
        let diff = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(diff < 1e-5, "{diff}");
    }

    #[test]
    fn rope_scores_depend_only_on_relative_position() {
        // One head, the same q and k at every position: after RoPE,
        // q_t . k_u is a function of t - u alone.
        let (hd, s) = (8, 6);
        let q: Vec<f32> = (0..hd).map(|i| 0.3 + i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..hd).map(|i| 0.9 - i as f32 * 0.07).collect();
        let spread = |v: &[f32]| -> Vec<f32> { (0..hd * s).map(|i| v[i / s]).collect() };
        let (mut qs, mut ks) = (spread(&q), spread(&k));
        rope_channel_first(&mut qs, 1, hd, s, 10_000.0, false);
        rope_channel_first(&mut ks, 1, hd, s, 10_000.0, false);
        let dot = |t: usize, u: usize| (0..hd).map(|i| qs[i * s + t] * ks[i * s + u]).sum::<f32>();
        assert!((dot(3, 1) - dot(4, 2)).abs() < 1e-5);
        assert!((dot(5, 5) - dot(0, 0)).abs() < 1e-5);
        assert!(
            (dot(3, 1) - dot(3, 2)).abs() > 1e-3,
            "and it does depend on it"
        );
    }

    #[test]
    fn rope_inverse_is_the_transpose_for_gradients() {
        // <R x, g> == <x, R^-1 g>, so R^-1 carries a gradient back through R.
        let (nh, hd, s) = (2, 4, 7);
        let x = rope_input(nh, hd, s);
        let g: Vec<f32> = rope_input(nh, hd, s).iter().rev().copied().collect();
        let (mut rx, mut rg) = (x.clone(), g.clone());
        rope_channel_first(&mut rx, nh, hd, s, 500.0, false);
        rope_channel_first(&mut rg, nh, hd, s, 500.0, true);
        let lhs: f32 = rx.iter().zip(&g).map(|(a, b)| a * b).sum();
        let rhs: f32 = x.iter().zip(&rg).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-4, "{lhs} vs {rhs}");
    }
}
