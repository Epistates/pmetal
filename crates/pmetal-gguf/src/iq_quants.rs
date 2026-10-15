//! IQ block formats (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS).
//!
//! The IQ formats store, per group of 8 (or 4) weights, an index into a fixed
//! codebook plus a sign pattern, and scale each 16- or 32-weight sub-block by
//! a small integer on top of a per-block `f16` scale. IQ4_NL and IQ4_XS are the
//! exception: they use a 16-entry non-linear value table instead of a grid.
//!
//! Block layouts (all little endian, `QK_K = 256`), as defined by the ggml
//! reference (`ggml-common.h`):
//!
//! | Type    | Elements | Bytes | Layout                                                   |
//! |---------|----------|-------|----------------------------------------------------------|
//! | IQ1_S   | 256      | 50    | `d: f16`, `qs: [u8; 32]`, `qh: [u16; 8]`                 |
//! | IQ1_M   | 256      | 56    | `qs: [u8; 32]`, `qh: [u8; 16]`, `scales: [u16; 4]`       |
//! | IQ2_XXS | 256      | 66    | `d: f16`, `qs: [u16; 32]`                                |
//! | IQ2_XS  | 256      | 74    | `d: f16`, `qs: [u16; 32]`, `scales: [u8; 8]`             |
//! | IQ2_S   | 256      | 82    | `d: f16`, `qs: [u8; 64]`, `qh: [u8; 8]`, `scales: [u8; 8]` |
//! | IQ3_XXS | 256      | 98    | `d: f16`, `qs: [u8; 96]`                                 |
//! | IQ3_S   | 256      | 110   | `d: f16`, `qs: [u8; 64]`, `qh: [u8; 8]`, `signs: [u8; 32]`, `scales: [u8; 4]` |
//! | IQ4_NL  | 32       | 18    | `d: f16`, `qs: [u8; 16]`                                 |
//! | IQ4_XS  | 256      | 136   | `d: f16`, `scales_h: u16`, `scales_l: [u8; 4]`, `qs: [u8; 128]` |
//!
//! Every decoder here is bit-exact against the reference: all multiplications
//! are formed in the same order as the reference so that the f32 results match
//! to the last bit.

// The quantizers follow the reference algorithms index for index; rewriting their
// loops as iterator chains over several parallel arrays would obscure that.
#![allow(clippy::needless_range_loop)]

use half::f16;

use crate::iq_grids::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID};
use crate::quantize::{GROUP_MAX_EPS, best_index_int8, nearest_int};

/// Block size for IQ4_NL (32 elements per block, like Q4_0).
pub const QK4_NL: usize = 32;

/// Block size for the K-quant style IQ types (256 elements per block).
pub const QK_K: usize = 256;

/// Non-linear value table shared by IQ4_NL and IQ4_XS.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Offset added to every IQ1 grid point; the sign of the block's delta bit
/// selects `+` or `-`.
pub const IQ1S_DELTA: f32 = 0.125;

/// Sign patterns for IQ2/IQ3: a 7-bit index maps to an 8-bit mask whose top
/// bit is the parity of the index, so every pattern has an even number of
/// negative signs.
pub const KSIGNS_IQ2XS: [u8; 128] = ksigns_table();

const fn ksigns_table() -> [u8; 128] {
    let mut table = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        let parity = (i as u32).count_ones() & 1;
        table[i] = (i as u8) | ((parity as u8) << 7);
        i += 1;
    }
    table
}

#[inline(always)]
fn read_f16(bytes: &[u8]) -> f32 {
    f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

#[inline(always)]
fn sign(signs: u8, bit: usize) -> f32 {
    if signs & (1 << bit) != 0 { -1.0 } else { 1.0 }
}

#[inline(always)]
fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

#[inline(always)]
fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Decode one IQ4_NL block.
pub(crate) fn dequantize_block_iq4_nl(block: &[u8; 18], y: &mut [f32; 32]) {
    let d = read_f16(block);
    let qs = &block[2..];
    for j in 0..16 {
        y[j] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[j] & 0x0f)]);
        y[j + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[j] >> 4)]);
    }
}

/// Decode one IQ4_XS block.
pub(crate) fn dequantize_block_iq4_xs(block: &[u8; 136], y: &mut [f32; 256]) {
    let d = read_f16(block);
    let scales_h = read_u16(block, 2);
    let scales_l = &block[4..8];
    let qs = &block[8..];
    for ib in 0..QK_K / 32 {
        let ls = i32::from((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f)
            | (i32::from((scales_h >> (2 * ib)) & 3) << 4);
        let dl = d * (ls - 32) as f32;
        let qs = &qs[16 * ib..16 * ib + 16];
        let y = &mut y[32 * ib..32 * ib + 32];
        for j in 0..16 {
            y[j] = dl * f32::from(KVALUES_IQ4NL[usize::from(qs[j] & 0x0f)]);
            y[j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(qs[j] >> 4)]);
        }
    }
}

/// Decode one IQ2_XXS block.
pub(crate) fn dequantize_block_iq2_xxs(block: &[u8; 66], y: &mut [f32; 256]) {
    let d = read_f16(block);
    let qs = &block[2..];
    for ib32 in 0..QK_K / 32 {
        let group = &qs[8 * ib32..8 * ib32 + 8];
        let aux32 = read_u32(group, 4);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[usize::from(group[l])].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[32 * ib32 + 8 * l + j] = db * f32::from(grid[j]) * sign(signs, j);
            }
        }
    }
}

/// Decode one IQ2_XS block.
pub(crate) fn dequantize_block_iq2_xs(block: &[u8; 74], y: &mut [f32; 256]) {
    let d = read_f16(block);
    let qs = &block[2..66];
    let scales = &block[66..74];
    for ib32 in 0..QK_K / 32 {
        let db = [
            d * (0.5 + f32::from(scales[ib32] & 0x0f)) * 0.25,
            d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let q = read_u16(qs, 2 * (4 * ib32 + l));
            let grid = IQ2XS_GRID[usize::from(q & 511)].to_le_bytes();
            let signs = KSIGNS_IQ2XS[usize::from(q >> 9)];
            let dl = db[l / 2];
            for j in 0..8 {
                y[32 * ib32 + 8 * l + j] = dl * f32::from(grid[j]) * sign(signs, j);
            }
        }
    }
}

/// Decode one IQ2_S block.
pub(crate) fn dequantize_block_iq2_s(block: &[u8; 82], y: &mut [f32; 256]) {
    let d = read_f16(block);
    let qs = &block[2..34];
    let signs = &block[34..66];
    let qh = &block[66..74];
    let scales = &block[74..82];
    for ib32 in 0..QK_K / 32 {
        let db = [
            d * (0.5 + f32::from(scales[ib32] & 0x0f)) * 0.25,
            d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let hi = (usize::from(qh[ib32]) << (8 - 2 * l)) & 0x300;
            let grid = IQ2S_GRID[usize::from(qs[4 * ib32 + l]) | hi].to_le_bytes();
            let s = signs[4 * ib32 + l];
            let dl = db[l / 2];
            for j in 0..8 {
                y[32 * ib32 + 8 * l + j] = dl * f32::from(grid[j]) * sign(s, j);
            }
        }
    }
}

/// Decode one IQ3_XXS block.
pub(crate) fn dequantize_block_iq3_xxs(block: &[u8; 98], y: &mut [f32; 256]) {
    let d = read_f16(block);
    let qs = &block[2..66];
    let scales_and_signs = &block[66..98];
    for ib32 in 0..QK_K / 32 {
        let aux32 = read_u32(scales_and_signs, 4 * ib32);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l])].to_le_bytes();
            let g2 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l + 1])].to_le_bytes();
            let out = &mut y[32 * ib32 + 8 * l..32 * ib32 + 8 * l + 8];
            for j in 0..4 {
                out[j] = db * f32::from(g1[j]) * sign(signs, j);
                out[j + 4] = db * f32::from(g2[j]) * sign(signs, j + 4);
            }
        }
    }
}

/// Decode one IQ3_S block.
pub(crate) fn dequantize_block_iq3_s(block: &[u8; 110], y: &mut [f32; 256]) {
    let d = read_f16(block);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    for ib32 in 0..QK_K / 32 {
        let db = d * (1 + 2 * i32::from((scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0x0f)) as f32;
        for l in 0..4 {
            let hi1 = (usize::from(qh[ib32]) << (8 - 2 * l)) & 256;
            let hi2 = (usize::from(qh[ib32]) << (7 - 2 * l)) & 256;
            let g1 = IQ3S_GRID[usize::from(qs[8 * ib32 + 2 * l]) | hi1].to_le_bytes();
            let g2 = IQ3S_GRID[usize::from(qs[8 * ib32 + 2 * l + 1]) | hi2].to_le_bytes();
            let s = signs[4 * ib32 + l];
            let out = &mut y[32 * ib32 + 8 * l..32 * ib32 + 8 * l + 8];
            for j in 0..4 {
                out[j] = db * f32::from(g1[j]) * sign(s, j);
                out[j + 4] = db * f32::from(g2[j]) * sign(s, j + 4);
            }
        }
    }
}

/// Decode one IQ1_S block.
pub(crate) fn dequantize_block_iq1_s(block: &[u8; 50], y: &mut [f32; 256]) {
    let d = read_f16(block);
    let qs = &block[2..34];
    let qh = &block[34..50];
    for ib in 0..QK_K / 32 {
        let h = read_u16(qh, 2 * ib);
        let dl = d * (2 * i32::from((h >> 12) & 7) + 1) as f32;
        let delta = if h & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let idx = usize::from(qs[4 * ib + l]) | (usize::from((h >> (3 * l)) & 7) << 8);
            let grid = IQ1S_GRID[idx].to_le_bytes();
            for j in 0..8 {
                y[32 * ib + 8 * l + j] = dl * (f32::from(grid[j] as i8) + delta);
            }
        }
    }
}

/// Decode one IQ1_M block.
pub(crate) fn dequantize_block_iq1_m(block: &[u8; 56], y: &mut [f32; 256]) {
    let qs = &block[0..32];
    let qh = &block[32..48];
    let sc = [
        read_u16(block, 48),
        read_u16(block, 50),
        read_u16(block, 52),
        read_u16(block, 54),
    ];
    // The f16 super-scale is spread over the top nibble of the four scale words.
    let scale_bits =
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    let d = f16::from_bits(scale_bits).to_f32();
    for ib in 0..QK_K / 32 {
        let word = sc[ib / 2];
        let shift = 6 * (ib % 2);
        let dl = [
            d * (2 * i32::from((word >> shift) & 7) + 1) as f32,
            d * (2 * i32::from((word >> (shift + 3)) & 7) + 1) as f32,
        ];
        let h = [qh[2 * ib], qh[2 * ib + 1]];
        let idx = [
            usize::from(qs[4 * ib]) | ((usize::from(h[0]) << 8) & 0x700),
            usize::from(qs[4 * ib + 1]) | ((usize::from(h[0]) << 4) & 0x700),
            usize::from(qs[4 * ib + 2]) | ((usize::from(h[1]) << 8) & 0x700),
            usize::from(qs[4 * ib + 3]) | ((usize::from(h[1]) << 4) & 0x700),
        ];
        let delta = [
            if h[0] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if h[0] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if h[1] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if h[1] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
        ];
        for l in 0..4 {
            let grid = IQ1S_GRID[idx[l]].to_le_bytes();
            let dl = dl[l / 2];
            for j in 0..8 {
                y[32 * ib + 8 * l + j] = dl * (f32::from(grid[j] as i8) + delta[l]);
            }
        }
    }
}

// =============================================================================
// Quantization (IQ4_NL, IQ4_XS)
// =============================================================================

/// Quantize one super-block of `sb` values (32 for IQ4_NL, 256 for IQ4_XS) in
/// sub-blocks of `bs` to the non-linear 4-bit table.
///
/// Each sub-block's scale is found by trying `2 * ntry + 1` scales around the
/// one that maps its largest value to the end of the table and keeping the
/// least (importance-weighted) error; for IQ4_XS the sub-block scales are then
/// quantized to 6 bits against a shared `f16` scale, and the values
/// re-assigned under the quantized scales. Returns the `f16` scale; `q4`
/// receives the nibbles, `scales_h` / `scales_l` the packed 6-bit scales.
fn quantize_iq4_block(
    sb: usize,
    bs: usize,
    x: &[f32],
    qw: Option<&[f32]>,
    ntry: i32,
    q4: &mut [u8],
) -> (f16, u16, [u8; 4]) {
    let values = &KVALUES_IQ4NL;
    let mut scales_h = 0u16;
    let mut scales_l = [0u8; 4];
    let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / sb as f32;
    let nb = sb / bs;
    let mut l = [0u8; QK_K];
    let mut scales = [0.0f32; QK_K / 32];
    let mut weight = [0.0f32; 32];
    let mut max_scale = 0.0f32;
    let mut amax_scale = 0.0f32;

    for ib in 0..nb {
        let xb = &x[ib * bs..(ib + 1) * bs];
        for j in 0..bs {
            weight[j] = match qw {
                Some(qw) => qw[ib * bs + j] * (sigma2 + xb[j] * xb[j]).sqrt(),
                None => xb[j] * xb[j],
            };
        }
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let v0 = f32::from(values[0]);
        let mut d = if ntry > 0 { -max / v0 } else { max / v0 };
        let mut id = 1.0 / d;
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for j in 0..bs {
            let idx = best_index_int8(values, id * xb[j]);
            l[ib * bs + j] = idx as u8;
            let q = f32::from(values[idx]);
            sumqx += weight[j] * q * xb[j];
            sumq2 += weight[j] * q * q;
        }
        d = sumqx / sumq2;
        let mut best = d * sumqx;
        for itry in -ntry..=ntry {
            id = (itry as f32 + v0) / max;
            sumqx = 0.0;
            sumq2 = 0.0;
            for j in 0..bs {
                let q = f32::from(values[best_index_int8(values, id * xb[j])]);
                sumqx += weight[j] * q * xb[j];
                sumq2 += weight[j] * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        if d.abs() > amax_scale {
            amax_scale = d.abs();
            max_scale = d;
        }
    }

    let dh;
    if nb > 1 {
        let d = -max_scale / 32.0;
        dh = d;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for ib in 0..nb {
            let q = nearest_int(id * scales[ib]).clamp(-32, 31);
            let dl = d * q as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            for j in 0..bs {
                l[ib * bs + j] = best_index_int8(values, idl * x[ib * bs + j]) as u8;
            }
            let biased = (q + 32) as u8;
            if ib % 2 == 0 {
                scales_l[ib / 2] = biased & 0xf;
            } else {
                scales_l[ib / 2] |= (biased & 0xf) << 4;
            }
            scales_h |= u16::from(biased >> 4) << (2 * (ib % 8));
        }
    } else {
        dh = scales[0];
        if ntry > 0 {
            let id = if scales[0] != 0.0 {
                1.0 / scales[0]
            } else {
                0.0
            };
            for j in 0..sb {
                l[j] = best_index_int8(values, id * x[j]) as u8;
            }
        }
    }
    for i in 0..sb / 32 {
        for j in 0..16 {
            q4[16 * i + j] = l[32 * i + j] | (l[32 * i + 16 + j] << 4);
        }
    }
    (f16::from_f32(dh), scales_h, scales_l)
}

/// Quantize to IQ4_NL.
pub(crate) fn quantize_iq4_nl(x: &[f32], qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    let blocks = x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<18>().0.iter_mut());
    for (b, (xb, ob)) in blocks.enumerate() {
        let qwb = qw.map(|w| &w[32 * b..32 * b + 32]);
        let (d, _, _) = quantize_iq4_block(32, 32, xb, qwb, 7, &mut ob[2..]);
        ob[0..2].copy_from_slice(&d.to_le_bytes());
    }
}

/// Quantize to IQ4_XS.
pub(crate) fn quantize_iq4_xs(x: &[f32], qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    let blocks = x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<136>().0.iter_mut());
    for (b, (xb, ob)) in blocks.enumerate() {
        let qwb = qw.map(|w| &w[QK_K * b..QK_K * (b + 1)]);
        let (d, scales_h, scales_l) = quantize_iq4_block(QK_K, 32, xb, qwb, 7, &mut ob[8..]);
        ob[0..2].copy_from_slice(&d.to_le_bytes());
        ob[2..4].copy_from_slice(&scales_h.to_le_bytes());
        ob[4..8].copy_from_slice(&scales_l);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a over the little-endian bytes of a table.
    fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
        bytes.into_iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    fn table_hash<const N: usize>(table: &[u64; N]) -> u64 {
        fnv1a(table.iter().flat_map(|w| w.to_le_bytes()))
    }

    #[test]
    fn ksigns_match_the_reference_table() {
        // First and last rows of the reference `ksigns_iq2xs` table.
        assert_eq!(
            &KSIGNS_IQ2XS[..16],
            &[
                0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15
            ]
        );
        assert_eq!(
            &KSIGNS_IQ2XS[112..],
            &[
                240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
            ]
        );
        // Every pattern carries an even number of negative signs.
        assert!(KSIGNS_IQ2XS.iter().all(|s| s.count_ones() % 2 == 0));
    }

    #[test]
    fn iq4_nl_table_is_monotone_and_signed_around_zero() {
        assert!(KVALUES_IQ4NL.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(KVALUES_IQ4NL[0], -127);
        assert_eq!(KVALUES_IQ4NL[15], 113);
    }

    #[test]
    fn grid_tables_match_the_reference_checksums() {
        // Checksums of the reference codebooks, computed from the published
        // grid data. They pin every entry of every table.
        assert_eq!(table_hash(&IQ2XXS_GRID), 0xbb4e_e025_b5ac_6e8e);
        assert_eq!(table_hash(&IQ2XS_GRID), 0xc9b1_ee61_e799_09bd);
        assert_eq!(table_hash(&IQ2S_GRID), 0x123e_dd38_a3b6_2b90);
        assert_eq!(table_hash(&IQ1S_GRID), 0x6703_ed86_3501_ae2e);
        assert_eq!(
            fnv1a(IQ3XXS_GRID.iter().flat_map(|w| w.to_le_bytes())),
            0xdfa5_dc83_d6a1_55d5
        );
        assert_eq!(
            fnv1a(IQ3S_GRID.iter().flat_map(|w| w.to_le_bytes())),
            0xfa37_020c_25b4_4829
        );
    }
}
