//! K-Quant (Q2K-Q8K) block structures and operations.
//!
//! K-Quants use 256-element blocks with hierarchical sub-block scaling for
//! superior compression quality compared to basic Q4_0/Q8_0.
//!
//! # Block Sizes
//!
//! | Type | Block Size | Bytes/Block | Bits/Element |
//! |------|------------|-------------|--------------|
//! | Q2K  | 256        | 84          | ~2.6         |
//! | Q3K  | 256        | 110         | ~3.4         |
//! | Q4K  | 256        | 144         | ~4.5         |
//! | Q5K  | 256        | 176         | ~5.5         |
//! | Q6K  | 256        | 210         | ~6.6         |
//! | Q8K  | 256        | 292         | ~9.1         |
//!
//! # Reference
//!
//! Follows the llama.cpp/GGML K-quant reference implementation.

use crate::quantize::{
    get_scale_min_k4, make_q3_quants, make_qkx1_quants, make_qx_quants, nearest_int,
};
use half::f16;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Block size for all K-quants (256 elements per block).
pub const QK_K: usize = 256;

/// Size of scale encoding for Q4K/Q5K (12 bytes for 8 scales).
pub const K_SCALE_SIZE: usize = 12;

// =============================================================================
// Block Structures
// =============================================================================

/// Q2K block: 2-bit quantization with dual scales.
///
/// Structure: 84 bytes for 256 elements (~2.6 bits/element)
/// - scales: 16 bytes (4-bit scale pairs for 16 sub-blocks of 16 elements)
/// - qs: 64 bytes (2-bit quantized values, 4 per byte)
/// - d: 2 bytes (f16 scale factor)
/// - dmin: 2 bytes (f16 minimum scale factor)
#[repr(C)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, KnownLayout, Immutable)]
pub struct BlockQ2K {
    /// 4-bit scale pairs for 16 sub-blocks.
    pub scales: [u8; QK_K / 16],
    /// 2-bit quantized values (4 values per byte).
    pub qs: [u8; QK_K / 4],
    /// Scale factor.
    pub d: f16,
    /// Minimum scale factor.
    pub dmin: f16,
}

/// Q3K block: 3-bit quantization with high-bit mask.
///
/// Structure: 110 bytes for 256 elements (~3.4 bits/element)
/// - hmask: 32 bytes (high bit for each element)
/// - qs: 64 bytes (lower 2 bits of quantized values)
/// - scales: 12 bytes (complex scale encoding)
/// - d: 2 bytes (f16 scale factor)
#[repr(C)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, KnownLayout, Immutable)]
pub struct BlockQ3K {
    /// High bit mask (1 bit per element, 8 per byte).
    pub hmask: [u8; QK_K / 8],
    /// Lower 2 bits of 3-bit quantized values.
    pub qs: [u8; QK_K / 4],
    /// Scale encoding (complex packing).
    pub scales: [u8; 12],
    /// Scale factor.
    pub d: f16,
}

/// Q4K block: 4-bit quantization with per-subblock scales.
///
/// Structure: 144 bytes for 256 elements (~4.5 bits/element)
/// - d: 2 bytes (f16 scale)
/// - dmin: 2 bytes (f16 minimum scale)
/// - scales: 12 bytes (6-bit scales for 8 sub-blocks of 32 elements)
/// - qs: 128 bytes (4-bit quantized values, 2 per byte)
#[repr(C)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, KnownLayout, Immutable)]
pub struct BlockQ4K {
    /// Scale factor.
    pub d: f16,
    /// Minimum scale factor.
    pub dmin: f16,
    /// 6-bit scale pairs for 8 sub-blocks (packed into 12 bytes).
    pub scales: [u8; K_SCALE_SIZE],
    /// 4-bit quantized values (2 values per byte).
    pub qs: [u8; QK_K / 2],
}

/// Q5K block: 5-bit quantization with high-bit array.
///
/// Structure: 176 bytes for 256 elements (~5.5 bits/element)
/// - d: 2 bytes (f16 scale)
/// - dmin: 2 bytes (f16 minimum scale)
/// - scales: 12 bytes (6-bit scales for 8 sub-blocks)
/// - qh: 32 bytes (high bit for each element)
/// - qs: 128 bytes (lower 4 bits of quantized values)
#[repr(C)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, KnownLayout, Immutable)]
pub struct BlockQ5K {
    /// Scale factor.
    pub d: f16,
    /// Minimum scale factor.
    pub dmin: f16,
    /// 6-bit scale pairs for 8 sub-blocks.
    pub scales: [u8; K_SCALE_SIZE],
    /// High bit for each element (1 bit per element).
    pub qh: [u8; QK_K / 8],
    /// Lower 4 bits of 5-bit quantized values.
    pub qs: [u8; QK_K / 2],
}

/// Q6K block: 6-bit quantization with split storage.
///
/// Structure: 210 bytes for 256 elements (~6.6 bits/element)
/// - ql: 128 bytes (lower 4 bits of quantized values)
/// - qh: 64 bytes (upper 2 bits of quantized values)
/// - scales: 16 bytes (signed 8-bit scales for 16 sub-blocks)
/// - d: 2 bytes (f16 scale factor)
#[repr(C)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, KnownLayout, Immutable)]
pub struct BlockQ6K {
    /// Lower 4 bits of 6-bit quantized values.
    pub ql: [u8; QK_K / 2],
    /// Upper 2 bits of 6-bit quantized values.
    pub qh: [u8; QK_K / 4],
    /// Signed 8-bit scales for 16 sub-blocks of 16 elements.
    pub scales: [i8; QK_K / 16],
    /// Scale factor.
    pub d: f16,
}

/// Q8K block: 8-bit quantization with block sums.
///
/// Structure: 292 bytes for 256 elements (~9.1 bits/element)
/// - d: 4 bytes (f32 scale - full precision)
/// - qs: 256 bytes (8-bit signed quantized values)
/// - bsums: 32 bytes (pre-computed sums for 16 sub-blocks, for fast dot product)
#[repr(C)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, KnownLayout, Immutable)]
pub struct BlockQ8K {
    /// Scale factor (f32 for higher precision).
    pub d: f32,
    /// 8-bit signed quantized values.
    pub qs: [i8; QK_K],
    /// Pre-computed sums for each 16-element sub-block.
    pub bsums: [i16; QK_K / 16],
}

// Verify struct sizes at compile time
const _: () = assert!(std::mem::size_of::<BlockQ2K>() == 84);
const _: () = assert!(std::mem::size_of::<BlockQ3K>() == 110);
const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);
const _: () = assert!(std::mem::size_of::<BlockQ5K>() == 176);
const _: () = assert!(std::mem::size_of::<BlockQ6K>() == 210);
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 292);

// =============================================================================
// Dequantization
// =============================================================================
//
// Every decoder works on the raw little-endian block bytes, so it neither
// depends on struct layout nor on the alignment of the tensor buffer. The
// arithmetic is formed in the same order as the reference so results match it
// to the last bit.

#[inline(always)]
fn read_f16(bytes: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32()
}

/// Decode one Q2_K block (84 bytes: `scales[16]`, `qs[64]`, `d`, `dmin`).
pub(crate) fn dequantize_block_q2_k(block: &[u8; 84], y: &mut [f32; QK_K]) {
    let scales = &block[..16];
    let qs = &block[16..80];
    let d = read_f16(block, 80);
    let min = read_f16(block, 82);

    let mut is = 0;
    for (half, q) in qs.as_chunks::<32>().0.iter().enumerate() {
        for pass in 0..4 {
            let shift = 2 * pass;
            for (sub, qsub) in q.as_chunks::<16>().0.iter().enumerate() {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0f);
                let ml = min * f32::from(sc >> 4);
                let out = &mut y[128 * half + 32 * pass + 16 * sub..][..16];
                for (o, &b) in out.iter_mut().zip(qsub) {
                    *o = dl * f32::from((b >> shift) & 3) - ml;
                }
            }
        }
    }
}

/// Decode one Q3_K block (110 bytes: `hmask[32]`, `qs[64]`, `scales[12]`, `d`).
pub(crate) fn dequantize_block_q3_k(block: &[u8; 110], y: &mut [f32; QK_K]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let hmask = &block[..32];
    let qs = &block[32..96];
    let d_all = read_f16(block, 108);

    // Unpack the sixteen 6-bit scales from the 12 packed bytes.
    let mut aux = [0u32; 4];
    for (a, chunk) in aux.iter_mut().zip(block[96..108].as_chunks::<4>().0) {
        *a = u32::from_le_bytes(*chunk);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i32; 16];
    for (i, word) in aux.iter().enumerate() {
        for (j, b) in word.to_le_bytes().iter().enumerate() {
            scales[4 * i + j] = i32::from(*b) - 32;
        }
    }

    let mut is = 0;
    let mut m = 1u8;
    for (half, q) in qs.as_chunks::<32>().0.iter().enumerate() {
        for pass in 0..4 {
            let shift = 2 * pass;
            for sub in 0..2 {
                let dl = d_all * scales[is] as f32;
                is += 1;
                let out = &mut y[128 * half + 32 * pass + 16 * sub..][..16];
                for (l, o) in out.iter_mut().enumerate() {
                    let idx = 16 * sub + l;
                    let low = i32::from((q[idx] >> shift) & 3);
                    let high = if hmask[idx] & m != 0 { 0 } else { 4 };
                    *o = dl * (low - high) as f32;
                }
            }
            m <<= 1;
        }
    }
}

/// Decode one Q4_K block (144 bytes: `d`, `dmin`, `scales[12]`, `qs[128]`).
pub(crate) fn dequantize_block_q4_k(block: &[u8; 144], y: &mut [f32; QK_K]) {
    let d = read_f16(block, 0);
    let min = read_f16(block, 2);
    let scales = &block[4..16];
    let qs = &block[16..144];

    for (j, q) in qs.as_chunks::<32>().0.iter().enumerate() {
        let (sc1, m1) = get_scale_min_k4(2 * j, scales);
        let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
        let (d1, m1) = (d * f32::from(sc1), min * f32::from(m1));
        let (d2, m2) = (d * f32::from(sc2), min * f32::from(m2));
        let out = &mut y[64 * j..64 * j + 64];
        for l in 0..32 {
            out[l] = d1 * f32::from(q[l] & 0x0f) - m1;
            out[l + 32] = d2 * f32::from(q[l] >> 4) - m2;
        }
    }
}

/// Decode one Q5_K block (176 bytes: `d`, `dmin`, `scales[12]`, `qh[32]`, `qs[128]`).
pub(crate) fn dequantize_block_q5_k(block: &[u8; 176], y: &mut [f32; QK_K]) {
    let d = read_f16(block, 0);
    let min = read_f16(block, 2);
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let (mut u1, mut u2) = (1u8, 2u8);
    for (j, q) in qs.as_chunks::<32>().0.iter().enumerate() {
        let (sc1, m1) = get_scale_min_k4(2 * j, scales);
        let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
        let (d1, m1) = (d * f32::from(sc1), min * f32::from(m1));
        let (d2, m2) = (d * f32::from(sc2), min * f32::from(m2));
        let out = &mut y[64 * j..64 * j + 64];
        for l in 0..32 {
            let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            out[l] = d1 * f32::from((q[l] & 0x0f) + h1) - m1;
            out[l + 32] = d2 * f32::from((q[l] >> 4) + h2) - m2;
        }
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Decode one Q6_K block (210 bytes: `ql[128]`, `qh[64]`, `scales[16]`, `d`).
pub(crate) fn dequantize_block_q6_k(block: &[u8; 210], y: &mut [f32; QK_K]) {
    let d = read_f16(block, 208);
    for half in 0..2 {
        let ql = &block[64 * half..64 * half + 64];
        let qh = &block[128 + 32 * half..128 + 32 * half + 32];
        let sc = &block[192 + 8 * half..192 + 8 * half + 8];
        let out = &mut y[128 * half..128 * half + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = i32::from((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) - 32;
            let q2 = i32::from((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l] = d * f32::from(sc[is] as i8) * q1 as f32;
            out[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
            out[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
            out[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
        }
    }
}

/// Decode one Q8_K block (292 bytes: `d: f32`, `qs[256]`, `bsums[16]`).
pub(crate) fn dequantize_block_q8_k(block: &[u8; 292], y: &mut [f32; QK_K]) {
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    for (o, &q) in y.iter_mut().zip(&block[4..260]) {
        *o = d * f32::from(q as i8);
    }
}

// =============================================================================
// Quantization Functions
// =============================================================================

/// Quantize f32 slice to Q8K blocks.
///
/// Q8K is the simplest K-quant: 8-bit signed values with a single scale per block.
pub fn quantize_q8k(xs: &[f32]) -> Vec<BlockQ8K> {
    let n_blocks = xs.len().div_ceil(QK_K);
    let mut blocks = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0i8; QK_K],
            bsums: [0i16; QK_K / 16],
        };
        n_blocks
    ];

    for (block_idx, block) in blocks.iter_mut().enumerate() {
        let start = block_idx * QK_K;
        let x_slice = if start + QK_K <= xs.len() {
            &xs[start..start + QK_K]
        } else {
            // Partial last block - pad with zeros
            continue;
        };

        // Find max absolute value
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &x in x_slice {
            if x.abs() > amax {
                amax = x.abs();
                max = x;
            }
        }

        if amax == 0.0 {
            block.d = 0.0;
            block.qs.fill(0);
            block.bsums.fill(0);
            continue;
        }

        let iscale = -128.0f32 / max;
        for (j, q) in block.qs.iter_mut().enumerate() {
            let v = (iscale * x_slice[j]).round();
            *q = v.min(127.0) as i8;
        }

        // Compute block sums for fast dot product
        for j in 0..(QK_K / 16) {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += block.qs[j * 16 + ii] as i32;
            }
            block.bsums[j] = sum as i16;
        }

        block.d = 1.0 / iscale;
    }

    blocks
}

/// Quantize f32 slice to Q4K blocks.
///
/// Q4K uses 4-bit values with per-subblock scales and minimums.
pub fn quantize_q4k(xs: &[f32]) -> Vec<BlockQ4K> {
    let n_blocks = xs.len().div_ceil(QK_K);
    let mut blocks = vec![
        BlockQ4K {
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.0),
            scales: [0u8; K_SCALE_SIZE],
            qs: [0u8; QK_K / 2],
        };
        n_blocks
    ];

    for (block_idx, block) in blocks.iter_mut().enumerate() {
        let start = block_idx * QK_K;
        if start + QK_K > xs.len() {
            continue;
        }
        let x = &xs[start..start + QK_K];

        let mut mins = [0.0f32; QK_K / 32];
        let mut scales = [0.0f32; QK_K / 32];

        // Compute per-subblock scales and mins
        for (j, chunk) in x.as_chunks::<32>().0.iter().enumerate() {
            (scales[j], mins[j]) = make_qkx1_quants(15, 5, chunk);
        }

        // Get max scale and max min
        let max_scale = scales.iter().fold(0.0f32, |max, &val| val.max(max));
        let max_min = mins.iter().fold(0.0f32, |max, &val| val.max(max));

        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        // Encode scales into 12-byte packed format
        for j in 0..(QK_K / 32) {
            let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
            if j < 4 {
                block.scales[j] = ls;
                block.scales[j + 4] = lm;
            } else {
                block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                block.scales[j - 4] |= (ls >> 4) << 6;
                block.scales[j] |= (lm >> 4) << 6;
            }
        }

        block.d = f16::from_f32(max_scale / 63.0);
        block.dmin = f16::from_f32(max_min / 63.0);

        // Quantize values
        let mut l = [0u8; QK_K];
        for j in 0..(QK_K / 32) {
            let (sc, m) = get_scale_min_k4(j, &block.scales);
            let d = block.d.to_f32() * sc as f32;
            if d != 0.0 {
                let dm = block.dmin.to_f32() * m as f32;
                for ii in 0..32 {
                    let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                    l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                }
            }
        }

        // Pack 4-bit values
        for j in (0..QK_K).step_by(64) {
            for l_val in 0..32 {
                let offset_index = (j / 64) * 32 + l_val;
                block.qs[offset_index] = l[j + l_val] | (l[j + l_val + 32] << 4);
            }
        }
    }

    blocks
}

/// Quantize f32 slice to Q5K blocks.
///
/// Q5K uses 5-bit values with per-subblock scales and minimums.
pub fn quantize_q5k(xs: &[f32]) -> Vec<BlockQ5K> {
    let n_blocks = xs.len().div_ceil(QK_K);
    let mut blocks = vec![
        BlockQ5K {
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.0),
            scales: [0u8; K_SCALE_SIZE],
            qh: [0u8; QK_K / 8],
            qs: [0u8; QK_K / 2],
        };
        n_blocks
    ];

    for (block_idx, block) in blocks.iter_mut().enumerate() {
        let start = block_idx * QK_K;
        if start + QK_K > xs.len() {
            continue;
        }
        let x = &xs[start..start + QK_K];

        let mut mins = [0.0f32; QK_K / 32];
        let mut scales = [0.0f32; QK_K / 32];

        for (j, chunk) in x.as_chunks::<32>().0.iter().enumerate() {
            (scales[j], mins[j]) = make_qkx1_quants(31, 5, chunk);
        }

        let max_scale = scales.iter().fold(0.0f32, |max, &val| val.max(max));
        let max_min = mins.iter().fold(0.0f32, |max, &val| val.max(max));

        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        for j in 0..(QK_K / 32) {
            let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
            if j < 4 {
                block.scales[j] = ls;
                block.scales[j + 4] = lm;
            } else {
                block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                block.scales[j - 4] |= (ls >> 4) << 6;
                block.scales[j] |= (lm >> 4) << 6;
            }
        }

        block.d = f16::from_f32(max_scale / 63.0);
        block.dmin = f16::from_f32(max_min / 63.0);

        let mut l = [0u8; QK_K];
        for j in 0..(QK_K / 32) {
            let (sc, m) = get_scale_min_k4(j, &block.scales);
            let d = block.d.to_f32() * sc as f32;
            if d != 0.0 {
                let dm = block.dmin.to_f32() * m as f32;
                for ii in 0..32 {
                    let ll = nearest_int((x[32 * j + ii] + dm) / d);
                    l[32 * j + ii] = ll.clamp(0, 31) as u8;
                }
            }
        }

        // Pack into qs (lower 4 bits) and qh (high bit)
        block.qh.fill(0);
        let mut m1 = 1u8;
        let mut m2 = 2u8;
        for n in (0..QK_K).step_by(64) {
            let offset = (n / 64) * 32;
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    block.qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    block.qh[j] |= m2;
                }
                block.qs[offset + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }
    }

    blocks
}

/// Quantize f32 slice to Q6K blocks.
///
/// Q6K uses 6-bit values with signed per-subblock scales.
pub fn quantize_q6k(xs: &[f32]) -> Vec<BlockQ6K> {
    let n_blocks = xs.len().div_ceil(QK_K);
    let mut blocks = vec![
        BlockQ6K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [0i8; QK_K / 16],
            d: f16::from_f32(0.0),
        };
        n_blocks
    ];

    for (block_idx, block) in blocks.iter_mut().enumerate() {
        let start = block_idx * QK_K;
        if start + QK_K > xs.len() {
            continue;
        }
        let x = &xs[start..start + QK_K];

        let mut l = [0i8; QK_K];
        let mut scales_f = [0.0f32; QK_K / 16];

        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;

        // Compute per-subblock scales
        for ib in 0..(QK_K / 16) {
            let x_slice = &x[ib * 16..(ib + 1) * 16];
            let (scale, ls) = make_qx_quants(16, 32, x_slice, 1);
            for ii in 0..16 {
                l[ib * 16 + ii] = ls[ii];
            }
            scales_f[ib] = scale;
            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        let iscale = -128.0f32 / max_scale;
        block.d = f16::from_f32(1.0 / iscale);

        // Quantize scales
        for (y_scale, &scale) in block.scales.iter_mut().zip(scales_f.iter()) {
            *y_scale = nearest_int(iscale * scale).min(127) as i8;
        }

        // Re-quantize values with final scales
        for j in 0..(QK_K / 16) {
            let d = block.d.to_f32() * block.scales[j] as f32;
            if d != 0.0 {
                for ii in 0..16 {
                    let ll = nearest_int(x[16 * j + ii] / d).clamp(-32, 31);
                    l[16 * j + ii] = (ll + 32) as i8;
                }
            }
        }

        // Pack into ql (lower 4 bits) and qh (upper 2 bits)
        for j in (0..QK_K).step_by(128) {
            for l_idx in 0..32 {
                let q1 = l[j + l_idx] & 0xF;
                let q2 = l[j + l_idx + 32] & 0xF;
                let q3 = l[j + l_idx + 64] & 0xF;
                let q4 = l[j + l_idx + 96] & 0xF;
                block.ql[(j / 2) + l_idx] = (q1 | (q3 << 4)) as u8;
                block.ql[(j / 2) + l_idx + 32] = (q2 | (q4 << 4)) as u8;
                block.qh[(j / 4) + l_idx] = ((l[j + l_idx] >> 4)
                    | ((l[j + l_idx + 32] >> 4) << 2)
                    | ((l[j + l_idx + 64] >> 4) << 4)
                    | ((l[j + l_idx + 96] >> 4) << 6))
                    as u8;
            }
        }
    }

    blocks
}

/// Quantize f32 slice to Q3K blocks.
///
/// Q3K uses 3-bit values with complex scale packing.
pub fn quantize_q3k(xs: &[f32]) -> Vec<BlockQ3K> {
    let n_blocks = xs.len().div_ceil(QK_K);
    let mut blocks = vec![
        BlockQ3K {
            hmask: [0u8; QK_K / 8],
            qs: [0u8; QK_K / 4],
            scales: [0u8; 12],
            d: f16::from_f32(0.0),
        };
        n_blocks
    ];

    for (block_idx, block) in blocks.iter_mut().enumerate() {
        let start = block_idx * QK_K;
        if start + QK_K > xs.len() {
            continue;
        }
        let x = &xs[start..start + QK_K];

        let mut scales_f = [0.0f32; QK_K / 16];
        for (j, chunk) in x.as_chunks::<16>().0.iter().enumerate() {
            scales_f[j] = make_q3_quants(chunk, 4, true);
        }

        // Get max scale by absolute value
        let mut max_scale = 0.0f32;
        for &scale in &scales_f {
            if scale.abs() > max_scale.abs() {
                max_scale = scale;
            }
        }

        block.scales.fill(0);

        if max_scale != 0.0 {
            let iscale = -32.0 / max_scale;
            for (j, &scale) in scales_f.iter().enumerate() {
                let l_val = nearest_int(iscale * scale).clamp(-32, 31) + 32;
                if j < 8 {
                    block.scales[j] = (l_val & 0xF) as u8;
                } else {
                    block.scales[j - 8] |= ((l_val & 0xF) << 4) as u8;
                }
                let l_val = l_val >> 4;
                block.scales[j % 4 + 8] |= (l_val << (2 * (j / 4))) as u8;
            }
            block.d = f16::from_f32(1.0 / iscale);
        } else {
            block.d = f16::from_f32(0.0);
        }

        // Quantize values
        let mut l = [0i8; QK_K];
        for j in 0..(QK_K / 16) {
            let sc = if j < 8 {
                block.scales[j] & 0xF
            } else {
                block.scales[j - 8] >> 4
            };
            let sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
            let d = block.d.to_f32() * sc as f32;
            if d != 0.0 {
                for ii in 0..16 {
                    let l_val = nearest_int(x[16 * j + ii] / d);
                    l[16 * j + ii] = (l_val.clamp(-4, 3) + 4) as i8;
                }
            }
        }

        // Set high mask and pack values
        block.hmask.fill(0);
        let mut m = 0usize;
        let mut hm = 1u8;

        for ll in l.iter_mut() {
            if *ll > 3 {
                block.hmask[m] |= hm;
                *ll -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }

        // Pack 2-bit values (4 per byte)
        for j in (0..QK_K).step_by(128) {
            for l_val in 0..32 {
                block.qs[j / 4 + l_val] = (l[j + l_val]
                    | (l[j + l_val + 32] << 2)
                    | (l[j + l_val + 64] << 4)
                    | (l[j + l_val + 96] << 6)) as u8;
            }
        }
    }

    blocks
}

/// Quantize f32 slice to Q2K blocks.
///
/// Q2K uses 2-bit values with per-subblock scales and minimums.
pub fn quantize_q2k(xs: &[f32]) -> Vec<BlockQ2K> {
    const Q4SCALE: f32 = 15.0;

    let n_blocks = xs.len().div_ceil(QK_K);
    let mut blocks = vec![
        BlockQ2K {
            scales: [0u8; QK_K / 16],
            qs: [0u8; QK_K / 4],
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.0),
        };
        n_blocks
    ];

    for (block_idx, block) in blocks.iter_mut().enumerate() {
        let start = block_idx * QK_K;
        if start + QK_K > xs.len() {
            continue;
        }
        let x = &xs[start..start + QK_K];

        let mut mins = [0.0f32; QK_K / 16];
        let mut scales = [0.0f32; QK_K / 16];

        for (j, chunk) in x.chunks(16).enumerate() {
            if chunk.len() == 16 {
                (scales[j], mins[j]) = make_qkx1_quants(3, 5, chunk);
            }
        }

        let max_scale = scales.iter().fold(0.0f32, |max, &val| val.max(max));
        let max_min = mins.iter().fold(0.0f32, |max, &val| val.max(max));

        if max_scale > 0.0 {
            let iscale = Q4SCALE / max_scale;
            for (j, &scale) in scales.iter().enumerate().take(QK_K / 16) {
                block.scales[j] = nearest_int(iscale * scale) as u8;
            }
            block.d = f16::from_f32(max_scale / Q4SCALE);
        } else {
            block.scales[..QK_K / 16].fill(0);
            block.d = f16::from_f32(0.0);
        }

        if max_min > 0.0 {
            let iscale = Q4SCALE / max_min;
            for (j, scale) in block.scales.iter_mut().enumerate() {
                let l = nearest_int(iscale * mins[j]) as u8;
                *scale |= l << 4;
            }
            block.dmin = f16::from_f32(max_min / Q4SCALE);
        } else {
            block.dmin = f16::from_f32(0.0);
        }

        // Quantize values
        let mut big_l = [0u8; QK_K];
        for j in 0..(QK_K / 16) {
            let d = block.d.to_f32() * (block.scales[j] & 0xF) as f32;
            if d == 0.0 {
                continue;
            }
            let dm = block.dmin.to_f32() * (block.scales[j] >> 4) as f32;
            for ii in 0..16 {
                let ll = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3);
                big_l[16 * j + ii] = ll as u8;
            }
        }

        // Pack 2-bit values (4 per byte)
        for j in (0..QK_K).step_by(128) {
            for ll in 0..32 {
                block.qs[j / 4 + ll] = big_l[j + ll]
                    | (big_l[j + ll + 32] << 2)
                    | (big_l[j + ll + 64] << 4)
                    | (big_l[j + ll + 96] << 6);
            }
        }
    }

    blocks
}

/// Convert K-quant blocks to bytes for GGUF export.
pub fn blocks_to_bytes<T: IntoBytes + Immutable>(blocks: &[T]) -> Vec<u8> {
    blocks.as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ2K>(), 84);
        assert_eq!(std::mem::size_of::<BlockQ3K>(), 110);
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
        assert_eq!(std::mem::size_of::<BlockQ8K>(), 292);
    }

    #[test]
    fn test_get_scale_min_k4() {
        let scales: [u8; K_SCALE_SIZE] = [
            0x3F, 0x3F, 0x3F, 0x3F, // scales 0-3 (lower 6 bits)
            0x3F, 0x3F, 0x3F, 0x3F, // mins 0-3 (lower 6 bits)
            0x00, 0x00, 0x00, 0x00, // upper bits
        ];

        let (sc, m) = get_scale_min_k4(0, &scales);
        assert_eq!(sc, 63);
        assert_eq!(m, 63);
    }

    #[test]
    fn test_blocks_to_bytes() {
        let blocks = quantize_q8k(&vec![0.0; QK_K]);
        let bytes = blocks_to_bytes(&blocks);
        assert_eq!(bytes.len(), 292); // Q8K block size
    }
}
