//! K-quant block formats (Q2_K to Q8_K).
//!
//! The K-quants store 256 weights per block in sub-blocks of 16 or 32, each
//! with its own small integer scale (and, for Q2_K, Q4_K and Q5_K, minimum),
//! which are in turn scaled by one or two `f16` super-block factors.
//!
//! Layouts, as defined by the ggml reference (`ggml-common.h`):
//!
//! | Type | Bytes | Layout                                                        |
//! |------|-------|---------------------------------------------------------------|
//! | Q2_K | 84    | `scales[16]` (4-bit scale, 4-bit min), `qs[64]`, `d`, `dmin`   |
//! | Q3_K | 110   | `hmask[32]`, `qs[64]`, `scales[12]` (6-bit, packed), `d`       |
//! | Q4_K | 144   | `d`, `dmin`, `scales[12]` (6-bit scale + min, packed), `qs[128]` |
//! | Q5_K | 176   | `d`, `dmin`, `scales[12]`, `qh[32]`, `qs[128]`                 |
//! | Q6_K | 210   | `ql[128]`, `qh[64]`, `scales[16]` (i8), `d`                    |
//! | Q8_K | 292   | `d: f32`, `qs[256]` (i8), `bsums[16]` (i16)                    |

// The quantizers follow the reference algorithms index for index; rewriting their
// loops as iterator chains over several parallel arrays would obscure that.
#![allow(clippy::needless_range_loop)]

use crate::quantize::{
    GROUP_MAX_EPS, ScaleSearch, make_q3_quants, make_qkx2_quants, make_qp_quants, make_qx_quants,
    nearest_int,
};
use half::f16;

/// Block size for all K-quants (256 elements per block).
pub const QK_K: usize = 256;

/// Size of scale encoding for Q4K/Q5K (12 bytes for 8 scales).
pub const K_SCALE_SIZE: usize = 12;

/// Extract the 6-bit scale and minimum of sub-block `j` (0..8) from the 12
/// packed bytes of a Q4_K or Q5_K block.
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

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
// Quantization
// =============================================================================
//
// Each `quantize_*` below quantizes whole blocks of one row: `x` holds the
// values, `qw` (when present) the importance of the same columns, `out` the
// bytes. Without importance the sub-block scales are fitted to the plain
// (lightly self-weighted) squared error; with it, to the importance-weighted
// error, and the sub-block scales are then jointly re-quantized under the
// importance of each sub-block.

/// Pack eight 6-bit scales and eight 6-bit minimums into the 12-byte layout
/// of Q4_K and Q5_K.
fn pack_scales_k4(ls: &[u8; 8], lm: &[u8; 8], out: &mut [u8; K_SCALE_SIZE]) {
    for j in 0..8 {
        if j < 4 {
            out[j] = ls[j];
            out[j + 4] = lm[j];
        } else {
            out[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
            out[j - 4] |= (ls[j] >> 4) << 6;
            out[j] |= (lm[j] >> 4) << 6;
        }
    }
}

/// Fit the eight 32-element sub-blocks of a Q4_K / Q5_K block (`nmax` = 15 or
/// 31). Returns the quantized values, `d`, `dmin` and the packed scales.
fn fit_k_block_45(
    x: &[f32],
    qw: Option<&[f32]>,
    nmax: i32,
) -> ([u8; QK_K], f16, f16, [u8; K_SCALE_SIZE]) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    let mut packed = [0u8; K_SCALE_SIZE];
    let (mut ls, mut lm) = ([0u8; 8], [0u8; 8]);
    let (d_h, dmin_h);

    if let Some(qw) = qw {
        let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;
        let mut sw = [0.0f32; 8];
        for j in 0..8 {
            let mut weights = [0.0f32; 32];
            for i in 0..32 {
                let v = x[32 * j + i];
                weights[i] = qw[32 * j + i] * (sigma2 + v * v).sqrt();
            }
            sw[j] = weights.iter().sum();
            (scales[j], mins[j]) = make_qkx2_quants(
                nmax,
                &x[32 * j..32 * j + 32],
                &weights,
                &mut l[32 * j..32 * j + 32],
                &mut laux,
                ScaleSearch::IMPORTANCE,
            );
        }
        let d_block = make_qp_quants(63, &scales, &mut ls, &sw);
        let m_block = make_qp_quants(63, &mins, &mut lm, &sw);
        d_h = f16::from_f32(d_block);
        dmin_h = f16::from_f32(m_block);
    } else {
        let (rmin, nstep) = if nmax == 15 { (-1.0, 20) } else { (-0.5, 15) };
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..8 {
            let xs = &x[32 * j..32 * j + 32];
            let av_x = (xs.iter().map(|v| v * v).sum::<f32>() / 32.0).sqrt();
            let mut weights = [0.0f32; 32];
            for i in 0..32 {
                weights[i] = av_x + xs[i].abs();
            }
            (scales[j], mins[j]) = make_qkx2_quants(
                nmax,
                xs,
                &weights,
                &mut l[32 * j..32 * j + 32],
                &mut laux,
                ScaleSearch {
                    rmin,
                    rdelta: 0.1,
                    nstep,
                    use_mad: false,
                },
            );
            max_scale = max_scale.max(scales[j]);
            max_min = max_min.max(mins[j]);
        }
        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        for j in 0..8 {
            ls[j] = nearest_int(inv_scale * scales[j]).clamp(0, 63) as u8;
            lm[j] = nearest_int(inv_min * mins[j]).clamp(0, 63) as u8;
        }
        d_h = f16::from_f32(max_scale / 63.0);
        dmin_h = f16::from_f32(max_min / 63.0);
    }
    pack_scales_k4(&ls, &lm, &mut packed);

    // Re-quantize the values with the scales as they will be stored.
    let d = d_h.to_f32();
    let dmin = dmin_h.to_f32();
    for j in 0..8 {
        let (sc, m) = get_scale_min_k4(j, &packed);
        let dl = d * f32::from(sc);
        if dl == 0.0 {
            continue;
        }
        let dm = dmin * f32::from(m);
        for i in 0..32 {
            l[32 * j + i] = nearest_int((x[32 * j + i] + dm) / dl).clamp(0, nmax) as u8;
        }
    }
    (l, d_h, dmin_h, packed)
}

/// Quantize to Q4_K.
pub(crate) fn quantize_q4_k(x: &[f32], qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (b, (xb, ob)) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<144>().0.iter_mut())
        .enumerate()
    {
        let qwb = qw.map(|w| &w[QK_K * b..QK_K * (b + 1)]);
        let (l, d, dmin, packed) = fit_k_block_45(xb, qwb, 15);
        ob[0..2].copy_from_slice(&d.to_le_bytes());
        ob[2..4].copy_from_slice(&dmin.to_le_bytes());
        ob[4..16].copy_from_slice(&packed);
        for j in 0..4 {
            for i in 0..32 {
                ob[16 + 32 * j + i] = l[64 * j + i] | (l[64 * j + i + 32] << 4);
            }
        }
    }
}

/// Quantize to Q5_K.
pub(crate) fn quantize_q5_k(x: &[f32], qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (b, (xb, ob)) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<176>().0.iter_mut())
        .enumerate()
    {
        let qwb = qw.map(|w| &w[QK_K * b..QK_K * (b + 1)]);
        let (l, d, dmin, packed) = fit_k_block_45(xb, qwb, 31);
        ob[0..2].copy_from_slice(&d.to_le_bytes());
        ob[2..4].copy_from_slice(&dmin.to_le_bytes());
        ob[4..16].copy_from_slice(&packed);
        ob[16..48].fill(0);
        let (mut m1, mut m2) = (1u8, 2u8);
        for n in 0..4 {
            for j in 0..32 {
                let mut l1 = l[64 * n + j];
                if l1 > 15 {
                    l1 -= 16;
                    ob[16 + j] |= m1;
                }
                let mut l2 = l[64 * n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    ob[16 + j] |= m2;
                }
                ob[48 + 32 * n + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }
    }
}

/// Quantize to Q6_K.
pub(crate) fn quantize_q6_k(x: &[f32], qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (b, (xb, ob)) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<210>().0.iter_mut())
        .enumerate()
    {
        let mut l = [0i8; QK_K];
        let mut scales = [0.0f32; 16];
        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;
        for ib in 0..16 {
            let qwb = qw.map(|w| &w[QK_K * b + 16 * ib..QK_K * b + 16 * ib + 16]);
            let scale = make_qx_quants(
                32,
                &xb[16 * ib..16 * ib + 16],
                &mut l[16 * ib..16 * ib + 16],
                1,
                qwb,
            );
            scales[ib] = scale;
            if scale.abs() > max_abs_scale {
                max_abs_scale = scale.abs();
                max_scale = scale;
            }
        }
        if max_abs_scale < GROUP_MAX_EPS {
            ob.fill(0);
            continue;
        }
        let iscale = -128.0 / max_scale;
        let d_h = f16::from_f32(1.0 / iscale);
        ob[208..210].copy_from_slice(&d_h.to_le_bytes());
        let d = d_h.to_f32();
        for ib in 0..16 {
            ob[192 + ib] = nearest_int(iscale * scales[ib]).min(127) as i8 as u8;
        }
        for j in 0..16 {
            let dl = d * f32::from(ob[192 + j] as i8);
            if dl == 0.0 {
                continue;
            }
            for i in 0..16 {
                let q = nearest_int(xb[16 * j + i] / dl).clamp(-32, 31);
                l[16 * j + i] = (q + 32) as i8;
            }
        }
        for half in 0..2 {
            for i in 0..32 {
                let at = 128 * half + i;
                let q1 = (l[at] & 0xF) as u8;
                let q2 = (l[at + 32] & 0xF) as u8;
                let q3 = (l[at + 64] & 0xF) as u8;
                let q4 = (l[at + 96] & 0xF) as u8;
                ob[64 * half + i] = q1 | (q3 << 4);
                ob[64 * half + i + 32] = q2 | (q4 << 4);
                ob[128 + 32 * half + i] = ((l[at] >> 4) as u8)
                    | (((l[at + 32] >> 4) as u8) << 2)
                    | (((l[at + 64] >> 4) as u8) << 4)
                    | (((l[at + 96] >> 4) as u8) << 6);
            }
        }
    }
}

/// Quantize to Q8_K.
pub(crate) fn quantize_q8_k(x: &[f32], _qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (xb, ob) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<292>().0.iter_mut())
    {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            ob.fill(0);
            continue;
        }
        // -127 rather than -128 keeps the quants symmetric.
        let iscale = -127.0 / max;
        let mut qs = [0i8; QK_K];
        for (q, &v) in qs.iter_mut().zip(xb) {
            *q = nearest_int(iscale * v).min(127) as i8;
        }
        ob[0..4].copy_from_slice(&(1.0 / iscale).to_le_bytes());
        for (i, &q) in qs.iter().enumerate() {
            ob[4 + i] = q as u8;
        }
        for j in 0..QK_K / 16 {
            let sum: i32 = qs[16 * j..16 * j + 16].iter().map(|&q| i32::from(q)).sum();
            ob[260 + 2 * j..262 + 2 * j].copy_from_slice(&(sum as i16).to_le_bytes());
        }
    }
}

/// Store a 6-bit scale index `l` (already biased by 32) for sub-block `j` of
/// a Q3_K block in its 12-byte packed form.
fn pack_scale_q3(scales: &mut [u8; 12], j: usize, l: u8) {
    if j < 8 {
        scales[j] = l & 0xF;
    } else {
        scales[j - 8] |= (l & 0xF) << 4;
    }
    scales[j % 4 + 8] |= (l >> 4) << (2 * (j / 4));
}

/// Read back the signed scale of sub-block `j` from a packed Q3_K block.
fn unpack_scale_q3(scales: &[u8; 12], j: usize) -> i32 {
    let low = if j < 8 {
        scales[j] & 0xF
    } else {
        scales[j - 8] >> 4
    };
    i32::from(low | (((scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32
}

/// Quantize to Q3_K.
pub(crate) fn quantize_q3_k(x: &[f32], qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (b, (xb, ob)) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<110>().0.iter_mut())
        .enumerate()
    {
        let mut l = [0i8; QK_K];
        let mut scales = [0.0f32; 16];
        let mut packed = [0u8; 12];
        let d_h;

        if let Some(qw) = qw {
            let sigma2 = 2.0 * xb.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;
            let mut sw = [0.0f32; 16];
            for j in 0..16 {
                let mut weight = [0.0f32; 16];
                for i in 0..16 {
                    let v = xb[16 * j + i];
                    weight[i] = qw[QK_K * b + 16 * j + i] * (sigma2 + v * v).sqrt();
                }
                sw[j] = weight.iter().sum();
                scales[j] = make_qx_quants(
                    4,
                    &xb[16 * j..16 * j + 16],
                    &mut l[16 * j..16 * j + 16],
                    1,
                    Some(&weight),
                );
            }
            let mut ls = [0i8; 16];
            let d_block = make_qx_quants(32, &scales, &mut ls, 1, Some(&sw));
            for j in 0..16 {
                pack_scale_q3(&mut packed, j, ls[j] as u8);
            }
            d_h = f16::from_f32(d_block);
        } else {
            let mut max_scale = 0.0f32;
            let mut amax = 0.0f32;
            for j in 0..16 {
                scales[j] = make_q3_quants(
                    4,
                    &xb[16 * j..16 * j + 16],
                    &mut l[16 * j..16 * j + 16],
                    true,
                );
                if scales[j].abs() > amax {
                    amax = scales[j].abs();
                    max_scale = scales[j];
                }
            }
            if max_scale != 0.0 {
                let iscale = -32.0 / max_scale;
                for j in 0..16 {
                    let q = nearest_int(iscale * scales[j]).clamp(-32, 31) + 32;
                    pack_scale_q3(&mut packed, j, q as u8);
                }
                d_h = f16::from_f32(1.0 / iscale);
            } else {
                d_h = f16::from_f32(0.0);
            }
        }

        // Re-quantize the values with the scales as they will be stored.
        let d = d_h.to_f32();
        for j in 0..16 {
            let dl = d * unpack_scale_q3(&packed, j) as f32;
            if dl == 0.0 {
                continue;
            }
            for i in 0..16 {
                let q = nearest_int(xb[16 * j + i] / dl).clamp(-4, 3);
                l[16 * j + i] = (q + 4) as i8;
            }
        }

        // High bits: the first 8 values' go in bit 0 of each hmask byte, and so on.
        ob[..32].fill(0);
        let (mut m, mut hm) = (0usize, 1u8);
        for v in l.iter_mut() {
            if *v > 3 {
                ob[m] |= hm;
                *v -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }
        for half in 0..2 {
            for i in 0..32 {
                let at = 128 * half + i;
                ob[32 + 32 * half + i] =
                    (l[at] | (l[at + 32] << 2) | (l[at + 64] << 4) | (l[at + 96] << 6)) as u8;
            }
        }
        ob[96..108].copy_from_slice(&packed);
        ob[108..110].copy_from_slice(&d_h.to_le_bytes());
    }
}

/// Quantize to Q2_K.
pub(crate) fn quantize_q2_k(x: &[f32], qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (b, (xb, ob)) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<84>().0.iter_mut())
        .enumerate()
    {
        let mut l = [0u8; QK_K];
        let mut laux = [0u8; 16];
        let mut scales = [0.0f32; 16];
        let mut mins = [0.0f32; 16];
        let mut packed = [0u8; 16];
        let (d_h, dmin_h);

        if let Some(qw) = qw {
            let sigma2 = xb.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;
            let mut sw = [0.0f32; 16];
            for j in 0..16 {
                let mut weight = [0.0f32; 16];
                for i in 0..16 {
                    let v = xb[16 * j + i];
                    weight[i] = qw[QK_K * b + 16 * j + i] * (sigma2 + v * v).sqrt();
                }
                sw[j] = weight.iter().sum();
                (scales[j], mins[j]) = make_qkx2_quants(
                    3,
                    &xb[16 * j..16 * j + 16],
                    &weight,
                    &mut l[16 * j..16 * j + 16],
                    &mut laux,
                    ScaleSearch::IMPORTANCE,
                );
            }
            let (mut ls, mut lm) = ([0u8; 16], [0u8; 16]);
            let dm = make_qp_quants(15, &scales, &mut ls, &sw);
            let mm = make_qp_quants(15, &mins, &mut lm, &sw);
            d_h = f16::from_f32(dm);
            dmin_h = f16::from_f32(mm);
            for j in 0..16 {
                packed[j] = ls[j] | (lm[j] << 4);
            }
        } else {
            const Q4SCALE: f32 = 15.0;
            let mut max_scale = 0.0f32;
            let mut max_min = 0.0f32;
            for j in 0..16 {
                let xs = &xb[16 * j..16 * j + 16];
                let mut weight = [0.0f32; 16];
                for i in 0..16 {
                    weight[i] = xs[i].abs();
                }
                (scales[j], mins[j]) = make_qkx2_quants(
                    3,
                    xs,
                    &weight,
                    &mut l[16 * j..16 * j + 16],
                    &mut laux,
                    ScaleSearch {
                        rmin: -0.5,
                        rdelta: 0.1,
                        nstep: 15,
                        use_mad: true,
                    },
                );
                max_scale = max_scale.max(scales[j]);
                max_min = max_min.max(mins[j]);
            }
            if max_scale > 0.0 {
                let iscale = Q4SCALE / max_scale;
                for j in 0..16 {
                    packed[j] = nearest_int(iscale * scales[j]).clamp(0, 15) as u8;
                }
                d_h = f16::from_f32(max_scale / Q4SCALE);
            } else {
                d_h = f16::from_f32(0.0);
            }
            if max_min > 0.0 {
                let iscale = Q4SCALE / max_min;
                for j in 0..16 {
                    packed[j] |= (nearest_int(iscale * mins[j]).clamp(0, 15) as u8) << 4;
                }
                dmin_h = f16::from_f32(max_min / Q4SCALE);
            } else {
                dmin_h = f16::from_f32(0.0);
            }
        }

        let d = d_h.to_f32();
        let dmin = dmin_h.to_f32();
        for j in 0..16 {
            let dl = d * f32::from(packed[j] & 0xF);
            if dl == 0.0 {
                continue;
            }
            let dm = dmin * f32::from(packed[j] >> 4);
            for i in 0..16 {
                l[16 * j + i] = nearest_int((xb[16 * j + i] + dm) / dl).clamp(0, 3) as u8;
            }
        }
        ob[..16].copy_from_slice(&packed);
        for half in 0..2 {
            for i in 0..32 {
                let at = 128 * half + i;
                ob[16 + 32 * half + i] =
                    l[at] | (l[at + 32] << 2) | (l[at + 64] << 4) | (l[at + 96] << 6);
            }
        }
        ob[80..82].copy_from_slice(&d_h.to_le_bytes());
        ob[82..84].copy_from_slice(&dmin_h.to_le_bytes());
    }
}
