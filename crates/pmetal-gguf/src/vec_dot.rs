//! SIMD-optimized vector dot product operations for K-quant blocks.
//!
//! This module provides high-performance dot product implementations for
//! computing `sum(quant[i] * f32[i])` efficiently using:
//!
//! - ARM NEON intrinsics (Apple Silicon, ARM64)
//! - Scalar fallback for unsupported platforms
//!
//! # Usage
//!
//! ```ignore
//! use pmetal_gguf::vec_dot::{vec_dot_q4k_q8k, vec_dot_q8k};
//!
//! let weights: &[BlockQ4K] = ...;
//! let activations: &[BlockQ8K] = ...;
//! let result = vec_dot_q4k_q8k(weights, activations);
//! ```
//!
//! # Performance
//!
//! On Apple Silicon (M1/M2/M3), NEON implementations provide ~4-8x speedup
//! over scalar code for typical transformer inference workloads.

use crate::k_quants::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K};

// =============================================================================
// Platform-specific SIMD implementations
// =============================================================================

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(unsafe_code)]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Custom dot product for int8x16_t vectors.
    ///
    /// NEON doesn't have a native i8->i32 dot product instruction,
    /// so we expand to i16, multiply, then accumulate.
    #[inline(always)]
    unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
        // Multiply low 8 elements (expanded to i16)
        let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        // Multiply high 8 elements (expanded to i16)
        let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        // Pairwise add i16->i32 and combine
        vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
    }

    /// Vec dot Q8K x Q8K (simplest case: both are 8-bit).
    pub fn vec_dot_q8k_q8k(xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
        assert_eq!(xs.len(), ys.len());
        if xs.is_empty() {
            return 0.0;
        }

        unsafe {
            let mut sumf = 0.0f32;

            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = x.d * y.d;
                let mut sumi = vdupq_n_s32(0);

                // Process 32 elements per iteration (2x int8x16_t)
                for j in (0..QK_K).step_by(32) {
                    let xv0 = vld1q_s8(x.qs.as_ptr().add(j));
                    let xv1 = vld1q_s8(x.qs.as_ptr().add(j + 16));
                    let yv0 = vld1q_s8(y.qs.as_ptr().add(j));
                    let yv1 = vld1q_s8(y.qs.as_ptr().add(j + 16));

                    sumi = vaddq_s32(sumi, vdotq_s32(xv0, yv0));
                    sumi = vaddq_s32(sumi, vdotq_s32(xv1, yv1));
                }

                sumf += d * vaddvq_s32(sumi) as f32;
            }

            sumf
        }
    }

    /// Vec dot Q4K x Q8K (4-bit weights, 8-bit activations).
    ///
    /// This is the most common operation in LLM inference.
    pub fn vec_dot_q4k_q8k(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
        assert_eq!(xs.len(), ys.len());
        if xs.is_empty() {
            return 0.0;
        }

        unsafe {
            let m4b = vdupq_n_u8(0x0F); // Mask for lower 4 bits
            let mut sumf = 0.0f32;

            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = x.d.to_f32() * y.d;
                let dmin = x.dmin.to_f32() * y.d;

                // Compute dmin contribution using bsums
                let mut mins_sum: i32 = 0;
                for j in 0..8 {
                    // Extract scales from packed format
                    let sc_idx = j;
                    let (_sc, m) = crate::quantize::get_scale_min_k4(sc_idx, &x.scales);
                    // Sum of 32 q8 values for this sub-block
                    let bsum = y.bsums[j * 2] as i32 + y.bsums[j * 2 + 1] as i32;
                    mins_sum += m as i32 * bsum;
                }
                sumf -= dmin * mins_sum as f32;

                let mut sumi = 0i32;
                let q4 = x.qs.as_ptr();
                let q8 = y.qs.as_ptr();

                // Process 64 elements per iteration
                for j in 0..QK_K / 64 {
                    let offset_q4 = j * 32;
                    let offset_q8 = j * 64;

                    // Get scales for this sub-block pair
                    let (sc0, _m0) = crate::quantize::get_scale_min_k4(j * 2, &x.scales);
                    let (sc1, _m1) = crate::quantize::get_scale_min_k4(j * 2 + 1, &x.scales);

                    // Load 32 bytes of 4-bit values
                    let q4_0 = vld1q_u8(q4.add(offset_q4));
                    let q4_1 = vld1q_u8(q4.add(offset_q4 + 16));

                    // Unpack lower nibbles
                    let q4l_0 = vreinterpretq_s8_u8(vandq_u8(q4_0, m4b));
                    let q4l_1 = vreinterpretq_s8_u8(vandq_u8(q4_1, m4b));

                    // Load 32 bytes of 8-bit activations (for lower nibbles)
                    let q8l_0 = vld1q_s8(q8.add(offset_q8));
                    let q8l_1 = vld1q_s8(q8.add(offset_q8 + 16));

                    // Dot product for lower nibbles
                    let p0 = vdotq_s32(q4l_0, q8l_0);
                    let p1 = vdotq_s32(q4l_1, q8l_1);
                    let sum_l = vaddvq_s32(vaddq_s32(p0, p1));
                    sumi += sum_l * sc0 as i32;

                    // Unpack upper nibbles
                    let q4h_0 = vreinterpretq_s8_u8(vshrq_n_u8(q4_0, 4));
                    let q4h_1 = vreinterpretq_s8_u8(vshrq_n_u8(q4_1, 4));

                    // Load 32 bytes of 8-bit activations (for upper nibbles)
                    let q8h_0 = vld1q_s8(q8.add(offset_q8 + 32));
                    let q8h_1 = vld1q_s8(q8.add(offset_q8 + 48));

                    // Dot product for upper nibbles
                    let p2 = vdotq_s32(q4h_0, q8h_0);
                    let p3 = vdotq_s32(q4h_1, q8h_1);
                    let sum_h = vaddvq_s32(vaddq_s32(p2, p3));
                    sumi += sum_h * sc1 as i32;
                }

                sumf += d * sumi as f32;
            }

            sumf
        }
    }

    /// Vec dot Q6K x Q8K (6-bit weights, 8-bit activations).
    pub fn vec_dot_q6k_q8k(xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
        assert_eq!(xs.len(), ys.len());
        if xs.is_empty() {
            return 0.0;
        }

        unsafe {
            let m4b = vdupq_n_u8(0x0F);
            let m2b = vdupq_n_u8(0x03);
            let m32s = vdupq_n_s8(-32);
            let mut sumf = 0.0f32;

            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = x.d.to_f32() * y.d;
                let ql = x.ql.as_ptr();
                let qh = x.qh.as_ptr();
                let q8 = y.qs.as_ptr();

                let mut sumi = 0i32;

                // Process 128 elements per iteration
                for j in 0..QK_K / 128 {
                    let offset_ql = j * 64;
                    let offset_qh = j * 32;
                    let offset_q8 = j * 128;

                    // Load scales for this group
                    let scale0 = x.scales[j * 8] as i32;
                    let scale1 = x.scales[j * 8 + 1] as i32;
                    let scale2 = x.scales[j * 8 + 2] as i32;
                    let scale3 = x.scales[j * 8 + 3] as i32;
                    let scale4 = x.scales[j * 8 + 4] as i32;
                    let scale5 = x.scales[j * 8 + 5] as i32;
                    let scale6 = x.scales[j * 8 + 6] as i32;
                    let scale7 = x.scales[j * 8 + 7] as i32;

                    // Load ql (lower 4 bits)
                    let ql0 = vld1q_u8(ql.add(offset_ql));
                    let ql1 = vld1q_u8(ql.add(offset_ql + 16));
                    let ql2 = vld1q_u8(ql.add(offset_ql + 32));
                    let ql3 = vld1q_u8(ql.add(offset_ql + 48));

                    // Load qh (upper 2 bits, packed 4 per byte)
                    let qh0 = vld1q_u8(qh.add(offset_qh));
                    let qh1 = vld1q_u8(qh.add(offset_qh + 16));

                    // Reconstruct 6-bit values for first 32 elements
                    let q6_0l = vorrq_u8(vandq_u8(ql0, m4b), vshlq_n_u8(vandq_u8(qh0, m2b), 4));
                    let q6_0h = vorrq_u8(
                        vshrq_n_u8(ql0, 4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), m2b), 4),
                    );

                    // Subtract 32 to center around zero
                    let q6_0l = vaddq_s8(vreinterpretq_s8_u8(q6_0l), m32s);
                    let q6_0h = vaddq_s8(vreinterpretq_s8_u8(q6_0h), m32s);

                    // Load q8
                    let q8_0 = vld1q_s8(q8.add(offset_q8));
                    let q8_1 = vld1q_s8(q8.add(offset_q8 + 16));

                    // Dot product
                    let p0 = vaddvq_s32(vdotq_s32(q6_0l, q8_0));
                    let p1 = vaddvq_s32(vdotq_s32(q6_0h, q8_1));
                    sumi += p0 * scale0 + p1 * scale1;

                    // Similar for remaining elements...
                    let q6_1l = vorrq_u8(
                        vandq_u8(ql1, m4b),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), m2b), 4),
                    );
                    let q6_1h = vorrq_u8(
                        vshrq_n_u8(ql1, 4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 6), m2b), 4),
                    );
                    let q6_1l = vaddq_s8(vreinterpretq_s8_u8(q6_1l), m32s);
                    let q6_1h = vaddq_s8(vreinterpretq_s8_u8(q6_1h), m32s);

                    let q8_2 = vld1q_s8(q8.add(offset_q8 + 32));
                    let q8_3 = vld1q_s8(q8.add(offset_q8 + 48));
                    let p2 = vaddvq_s32(vdotq_s32(q6_1l, q8_2));
                    let p3 = vaddvq_s32(vdotq_s32(q6_1h, q8_3));
                    sumi += p2 * scale2 + p3 * scale3;

                    // Second half using ql2, ql3, qh1
                    let q6_2l = vorrq_u8(vandq_u8(ql2, m4b), vshlq_n_u8(vandq_u8(qh1, m2b), 4));
                    let q6_2h = vorrq_u8(
                        vshrq_n_u8(ql2, 4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), m2b), 4),
                    );
                    let q6_2l = vaddq_s8(vreinterpretq_s8_u8(q6_2l), m32s);
                    let q6_2h = vaddq_s8(vreinterpretq_s8_u8(q6_2h), m32s);

                    let q8_4 = vld1q_s8(q8.add(offset_q8 + 64));
                    let q8_5 = vld1q_s8(q8.add(offset_q8 + 80));
                    let p4 = vaddvq_s32(vdotq_s32(q6_2l, q8_4));
                    let p5 = vaddvq_s32(vdotq_s32(q6_2h, q8_5));
                    sumi += p4 * scale4 + p5 * scale5;

                    let q6_3l = vorrq_u8(
                        vandq_u8(ql3, m4b),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), m2b), 4),
                    );
                    let q6_3h = vorrq_u8(
                        vshrq_n_u8(ql3, 4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 6), m2b), 4),
                    );
                    let q6_3l = vaddq_s8(vreinterpretq_s8_u8(q6_3l), m32s);
                    let q6_3h = vaddq_s8(vreinterpretq_s8_u8(q6_3h), m32s);

                    let q8_6 = vld1q_s8(q8.add(offset_q8 + 96));
                    let q8_7 = vld1q_s8(q8.add(offset_q8 + 112));
                    let p6 = vaddvq_s32(vdotq_s32(q6_3l, q8_6));
                    let p7 = vaddvq_s32(vdotq_s32(q6_3h, q8_7));
                    sumi += p6 * scale6 + p7 * scale7;
                }

                sumf += d * sumi as f32;
            }

            sumf
        }
    }

    /// Vec dot Q5K x Q8K (5-bit weights, 8-bit activations).
    pub fn vec_dot_q5k_q8k(xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
        assert_eq!(xs.len(), ys.len());
        if xs.is_empty() {
            return 0.0;
        }

        unsafe {
            let m4b = vdupq_n_u8(0x0F);
            let mone = vdupq_n_u8(0x10);
            let mut sumf = 0.0f32;

            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = x.d.to_f32() * y.d;
                let dmin = x.dmin.to_f32() * y.d;

                // Compute dmin contribution
                let mut mins_sum: i32 = 0;
                for j in 0..8 {
                    let (_, m) = crate::quantize::get_scale_min_k4(j, &x.scales);
                    let bsum = y.bsums[j * 2] as i32 + y.bsums[j * 2 + 1] as i32;
                    mins_sum += m as i32 * bsum;
                }
                sumf -= dmin * mins_sum as f32;

                let ql = x.qs.as_ptr();
                let qh = x.qh.as_ptr();
                let q8 = y.qs.as_ptr();

                let mut sumi = 0i32;

                // Process 64 elements per iteration
                for j in 0..QK_K / 64 {
                    let offset_ql = j * 32;
                    let offset_qh = j * 8;
                    let offset_q8 = j * 64;

                    let (sc0, _) = crate::quantize::get_scale_min_k4(j * 2, &x.scales);
                    let (sc1, _) = crate::quantize::get_scale_min_k4(j * 2 + 1, &x.scales);

                    // Load 32 bytes of lower 4 bits
                    let q5l_0 = vld1q_u8(ql.add(offset_ql));
                    let q5l_1 = vld1q_u8(ql.add(offset_ql + 16));

                    // Load high bits (1 bit per value, 8 per byte)
                    let qhbits = *qh.add(offset_qh);

                    // Reconstruct 5-bit values for lower nibbles
                    let hbit0 = if (qhbits & 0x01) != 0 {
                        mone
                    } else {
                        vdupq_n_u8(0)
                    };
                    let hbit1 = if (qhbits & 0x02) != 0 {
                        mone
                    } else {
                        vdupq_n_u8(0)
                    };

                    let q5_0l = vaddq_u8(vandq_u8(q5l_0, m4b), hbit0);
                    let q5_0h = vaddq_u8(vshrq_n_u8(q5l_0, 4), hbit1);

                    // Load q8
                    let q8_0 = vld1q_s8(q8.add(offset_q8));
                    let q8_1 = vld1q_s8(q8.add(offset_q8 + 16));

                    let p0 = vdotq_s32(vreinterpretq_s8_u8(q5_0l), q8_0);
                    let p1 = vdotq_s32(vreinterpretq_s8_u8(q5_0h), q8_1);
                    sumi += vaddvq_s32(vaddq_s32(p0, p1)) * sc0 as i32;

                    // Upper nibbles
                    let hbit2 = if (qhbits & 0x04) != 0 {
                        mone
                    } else {
                        vdupq_n_u8(0)
                    };
                    let hbit3 = if (qhbits & 0x08) != 0 {
                        mone
                    } else {
                        vdupq_n_u8(0)
                    };

                    let q5_1l = vaddq_u8(vandq_u8(q5l_1, m4b), hbit2);
                    let q5_1h = vaddq_u8(vshrq_n_u8(q5l_1, 4), hbit3);

                    let q8_2 = vld1q_s8(q8.add(offset_q8 + 32));
                    let q8_3 = vld1q_s8(q8.add(offset_q8 + 48));

                    let p2 = vdotq_s32(vreinterpretq_s8_u8(q5_1l), q8_2);
                    let p3 = vdotq_s32(vreinterpretq_s8_u8(q5_1h), q8_3);
                    sumi += vaddvq_s32(vaddq_s32(p2, p3)) * sc1 as i32;
                }

                sumf += d * sumi as f32;
            }

            sumf
        }
    }

    /// Vec dot Q3K x Q8K (3-bit weights, 8-bit activations).
    pub fn vec_dot_q3k_q8k(xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
        assert_eq!(xs.len(), ys.len());
        if xs.is_empty() {
            return 0.0;
        }

        unsafe {
            let m3b = vdupq_n_u8(0x03);
            let m4 = vdupq_n_s8(-4);
            let mut sumf = 0.0f32;

            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = x.d.to_f32() * y.d;
                let qs = x.qs.as_ptr();
                let hmask = x.hmask.as_ptr();
                let q8 = y.qs.as_ptr();

                let mut sumi = 0i32;

                // Unpack 12-byte scales to 16 6-bit values
                let mut scales = [0i8; 16];
                let aux0 = x.scales[0] as i32
                    | ((x.scales[1] as i32) << 8)
                    | ((x.scales[2] as i32) << 16)
                    | ((x.scales[3] as i32) << 24);
                let aux2 = x.scales[4] as i32
                    | ((x.scales[5] as i32) << 8)
                    | ((x.scales[6] as i32) << 16)
                    | ((x.scales[7] as i32) << 24);
                let aux4 = x.scales[8] as i32
                    | ((x.scales[9] as i32) << 8)
                    | ((x.scales[10] as i32) << 16)
                    | ((x.scales[11] as i32) << 24);

                #[allow(clippy::needless_range_loop)]
                for i in 0..8 {
                    scales[i] = ((aux0 >> (4 * i)) & 0xF) as i8 - 8;
                }
                #[allow(clippy::needless_range_loop)]
                for i in 0..8 {
                    scales[i + 8] = ((aux2 >> (4 * i)) & 0xF) as i8 - 8;
                }
                // Apply aux4 corrections
                for i in 0..4 {
                    let shift = (aux4 >> (8 * i)) & 0xFF;
                    scales[i] += ((shift & 0x03) << 4) as i8;
                    scales[i + 4] += (((shift >> 2) & 0x03) << 4) as i8;
                    scales[i + 8] += (((shift >> 4) & 0x03) << 4) as i8;
                    scales[i + 12] += (((shift >> 6) & 0x03) << 4) as i8;
                }

                // Process 128 elements per iteration
                for j in 0..2 {
                    let offset_qs = j * 32;
                    let offset_hm = j * 16;
                    let offset_q8 = j * 128;

                    // Load lower 2 bits
                    let q3l_0 = vld1q_u8(qs.add(offset_qs));
                    let q3l_1 = vld1q_u8(qs.add(offset_qs + 16));

                    // Load high bits
                    let hm0 = vld1q_u8(hmask.add(offset_hm));

                    // Reconstruct 3-bit values
                    let q3_0 = vandq_u8(q3l_0, m3b);
                    let q3_1 = vandq_u8(vshrq_n_u8(q3l_0, 2), m3b);
                    let q3_2 = vandq_u8(vshrq_n_u8(q3l_0, 4), m3b);
                    let q3_3 = vandq_u8(vshrq_n_u8(q3l_0, 6), m3b);

                    // Add high bit contribution
                    let h0 = vandq_u8(hm0, vdupq_n_u8(0x01));
                    let q3_0 = vaddq_u8(q3_0, vshlq_n_u8(h0, 2));

                    // Subtract 4 to center
                    let q3_0 = vaddq_s8(vreinterpretq_s8_u8(q3_0), m4);

                    // Load q8
                    let q8_0 = vld1q_s8(q8.add(offset_q8));

                    let scale = scales[j * 8] as i32;
                    sumi += vaddvq_s32(vdotq_s32(q3_0, q8_0)) * scale;

                    // Similar for remaining elements (simplified)
                    let q3_1 = vaddq_s8(vreinterpretq_s8_u8(q3_1), m4);
                    let q8_1 = vld1q_s8(q8.add(offset_q8 + 16));
                    sumi += vaddvq_s32(vdotq_s32(q3_1, q8_1)) * scales[j * 8 + 1] as i32;

                    let q3_2 = vaddq_s8(vreinterpretq_s8_u8(q3_2), m4);
                    let q8_2 = vld1q_s8(q8.add(offset_q8 + 32));
                    sumi += vaddvq_s32(vdotq_s32(q3_2, q8_2)) * scales[j * 8 + 2] as i32;

                    let q3_3 = vaddq_s8(vreinterpretq_s8_u8(q3_3), m4);
                    let q8_3 = vld1q_s8(q8.add(offset_q8 + 48));
                    sumi += vaddvq_s32(vdotq_s32(q3_3, q8_3)) * scales[j * 8 + 3] as i32;

                    // Second 16 bytes
                    let q3_4 = vandq_u8(q3l_1, m3b);
                    let q3_5 = vandq_u8(vshrq_n_u8(q3l_1, 2), m3b);
                    let q3_6 = vandq_u8(vshrq_n_u8(q3l_1, 4), m3b);
                    let q3_7 = vandq_u8(vshrq_n_u8(q3l_1, 6), m3b);

                    let q3_4 = vaddq_s8(vreinterpretq_s8_u8(q3_4), m4);
                    let q3_5 = vaddq_s8(vreinterpretq_s8_u8(q3_5), m4);
                    let q3_6 = vaddq_s8(vreinterpretq_s8_u8(q3_6), m4);
                    let q3_7 = vaddq_s8(vreinterpretq_s8_u8(q3_7), m4);

                    let q8_4 = vld1q_s8(q8.add(offset_q8 + 64));
                    let q8_5 = vld1q_s8(q8.add(offset_q8 + 80));
                    let q8_6 = vld1q_s8(q8.add(offset_q8 + 96));
                    let q8_7 = vld1q_s8(q8.add(offset_q8 + 112));

                    sumi += vaddvq_s32(vdotq_s32(q3_4, q8_4)) * scales[j * 8 + 4] as i32;
                    sumi += vaddvq_s32(vdotq_s32(q3_5, q8_5)) * scales[j * 8 + 5] as i32;
                    sumi += vaddvq_s32(vdotq_s32(q3_6, q8_6)) * scales[j * 8 + 6] as i32;
                    sumi += vaddvq_s32(vdotq_s32(q3_7, q8_7)) * scales[j * 8 + 7] as i32;
                }

                sumf += d * sumi as f32;
            }

            sumf
        }
    }

    /// Vec dot Q2K x Q8K (2-bit weights, 8-bit activations).
    pub fn vec_dot_q2k_q8k(xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
        assert_eq!(xs.len(), ys.len());
        if xs.is_empty() {
            return 0.0;
        }

        unsafe {
            let m3 = vdupq_n_u8(0x03);
            let mut sumf = 0.0f32;

            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = x.d.to_f32() * y.d;
                let dmin = x.dmin.to_f32() * y.d;

                // Compute dmin contribution
                let mut mins_sum: i32 = 0;
                for j in 0..16 {
                    let m = (x.scales[j] >> 4) as i32;
                    let bsum = y.bsums[j] as i32;
                    mins_sum += m * bsum;
                }
                sumf -= dmin * mins_sum as f32;

                let qs = x.qs.as_ptr();
                let q8 = y.qs.as_ptr();

                let mut sumi = 0i32;

                // Process 128 elements per iteration
                for j in 0..QK_K / 128 {
                    let offset_qs = j * 32;
                    let offset_q8 = j * 128;

                    // Load 32 bytes of 2-bit values (4 per byte = 128 values)
                    let q2_0 = vld1q_u8(qs.add(offset_qs));
                    let q2_1 = vld1q_u8(qs.add(offset_qs + 16));

                    // Extract 2-bit values
                    let q2_00 = vandq_u8(q2_0, m3);
                    let q2_01 = vandq_u8(vshrq_n_u8(q2_0, 2), m3);
                    let q2_02 = vandq_u8(vshrq_n_u8(q2_0, 4), m3);
                    let q2_03 = vandq_u8(vshrq_n_u8(q2_0, 6), m3);

                    let q2_10 = vandq_u8(q2_1, m3);
                    let q2_11 = vandq_u8(vshrq_n_u8(q2_1, 2), m3);
                    let q2_12 = vandq_u8(vshrq_n_u8(q2_1, 4), m3);
                    let q2_13 = vandq_u8(vshrq_n_u8(q2_1, 6), m3);

                    // Get scales for each 16-element group
                    for k in 0..8 {
                        let scale = (x.scales[j * 8 + k] & 0x0F) as i32;
                        let q8_k = vld1q_s8(q8.add(offset_q8 + k * 16));

                        let q2_k = match k {
                            0 => vreinterpretq_s8_u8(q2_00),
                            1 => vreinterpretq_s8_u8(q2_01),
                            2 => vreinterpretq_s8_u8(q2_02),
                            3 => vreinterpretq_s8_u8(q2_03),
                            4 => vreinterpretq_s8_u8(q2_10),
                            5 => vreinterpretq_s8_u8(q2_11),
                            6 => vreinterpretq_s8_u8(q2_12),
                            _ => vreinterpretq_s8_u8(q2_13),
                        };

                        sumi += vaddvq_s32(vdotq_s32(q2_k, q8_k)) * scale;
                    }
                }

                sumf += d * sumi as f32;
            }

            sumf
        }
    }
}

// =============================================================================
// Scalar fallback implementations
// =============================================================================

#[allow(dead_code)]
mod scalar {
    use super::*;

    /// Scalar vec_dot for Q8K x Q8K.
    pub fn vec_dot_q8k_q8k(xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
        let mut sum = 0.0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = x.d * y.d;
            let mut sumi: i32 = 0;
            for i in 0..QK_K {
                sumi += x.qs[i] as i32 * y.qs[i] as i32;
            }
            sum += d * sumi as f32;
        }
        sum
    }

    /// Scalar vec_dot for Q4K x Q8K.
    pub fn vec_dot_q4k_q8k(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
        let mut sum = 0.0f32;

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = x.d.to_f32() * y.d;
            let dmin = x.dmin.to_f32() * y.d;

            // Compute dmin contribution
            let mut mins_sum: i32 = 0;
            for j in 0..8 {
                let (_, m) = crate::quantize::get_scale_min_k4(j, &x.scales);
                let bsum = y.bsums[j * 2] as i32 + y.bsums[j * 2 + 1] as i32;
                mins_sum += m as i32 * bsum;
            }
            sum -= dmin * mins_sum as f32;

            let mut sumi: i32 = 0;
            for j in 0..QK_K / 64 {
                let (sc0, _) = crate::quantize::get_scale_min_k4(j * 2, &x.scales);
                let (sc1, _) = crate::quantize::get_scale_min_k4(j * 2 + 1, &x.scales);

                for k in 0..32 {
                    let q4_idx = j * 32 + k;
                    let q8_idx_l = j * 64 + k;
                    let q8_idx_h = j * 64 + k + 32;

                    let q4l = (x.qs[q4_idx] & 0x0F) as i32;
                    let q4h = (x.qs[q4_idx] >> 4) as i32;

                    sumi += q4l * y.qs[q8_idx_l] as i32 * sc0 as i32;
                    sumi += q4h * y.qs[q8_idx_h] as i32 * sc1 as i32;
                }
            }

            sum += d * sumi as f32;
        }

        sum
    }

    /// Scalar vec_dot for Q6K x Q8K.
    pub fn vec_dot_q6k_q8k(xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
        let mut sum = 0.0f32;

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = x.d.to_f32() * y.d;
            let mut sumi: i32 = 0;

            for j in 0..QK_K / 16 {
                let scale = x.scales[j] as i32;

                for k in 0..16 {
                    let idx = j * 16 + k;
                    let ql_idx = idx / 2;
                    let qh_idx = idx / 4;

                    // Extract 6-bit value
                    let ql = if idx % 2 == 0 {
                        x.ql[ql_idx] & 0x0F
                    } else {
                        x.ql[ql_idx] >> 4
                    };

                    let qh_shift = (idx % 4) * 2;
                    let qh = (x.qh[qh_idx] >> qh_shift) & 0x03;

                    let q6 = (ql | (qh << 4)) as i32 - 32;
                    sumi += q6 * y.qs[idx] as i32 * scale;
                }
            }

            sum += d * sumi as f32;
        }

        sum
    }

    /// Scalar vec_dot for Q5K x Q8K.
    pub fn vec_dot_q5k_q8k(xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
        let mut sum = 0.0f32;

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = x.d.to_f32() * y.d;
            let dmin = x.dmin.to_f32() * y.d;

            // Compute dmin contribution
            let mut mins_sum: i32 = 0;
            for j in 0..8 {
                let (_, m) = crate::quantize::get_scale_min_k4(j, &x.scales);
                let bsum = y.bsums[j * 2] as i32 + y.bsums[j * 2 + 1] as i32;
                mins_sum += m as i32 * bsum;
            }
            sum -= dmin * mins_sum as f32;

            let mut sumi: i32 = 0;
            for j in 0..QK_K {
                let ql_idx = j / 2;
                let qh_idx = j / 8;
                let qh_bit = j % 8;

                // Extract 4 lower bits
                let ql = if j % 2 == 0 {
                    x.qs[ql_idx] & 0x0F
                } else {
                    x.qs[ql_idx] >> 4
                };

                // Extract high bit
                let qh = ((x.qh[qh_idx] >> qh_bit) & 0x01) << 4;

                let q5 = (ql | qh) as i32;
                let (sc, _) = crate::quantize::get_scale_min_k4(j / 32, &x.scales);
                sumi += q5 * y.qs[j] as i32 * sc as i32;
            }

            sum += d * sumi as f32;
        }

        sum
    }

    /// Scalar vec_dot for Q3K x Q8K.
    pub fn vec_dot_q3k_q8k(xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
        let mut sum = 0.0f32;

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = x.d.to_f32() * y.d;
            let mut sumi: i32 = 0;

            // Unpack scales
            let mut scales = [0i8; 16];
            let aux0 = x.scales[0] as i32
                | ((x.scales[1] as i32) << 8)
                | ((x.scales[2] as i32) << 16)
                | ((x.scales[3] as i32) << 24);
            #[allow(clippy::needless_range_loop)]
            for i in 0..8 {
                scales[i] = ((aux0 >> (4 * i)) & 0xF) as i8 - 8;
            }

            for j in 0..QK_K {
                let qs_idx = j / 4;
                let qs_shift = (j % 4) * 2;
                let hm_idx = j / 8;
                let hm_bit = j % 8;

                // Extract 3-bit value
                let ql = (x.qs[qs_idx] >> qs_shift) & 0x03;
                let qh = ((x.hmask[hm_idx] >> hm_bit) & 0x01) << 2;
                let q3 = (ql | qh) as i32 - 4;

                let scale = scales[j / 16] as i32;
                sumi += q3 * y.qs[j] as i32 * scale;
            }

            sum += d * sumi as f32;
        }

        sum
    }

    /// Scalar vec_dot for Q2K x Q8K.
    pub fn vec_dot_q2k_q8k(xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
        let mut sum = 0.0f32;

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = x.d.to_f32() * y.d;
            let dmin = x.dmin.to_f32() * y.d;

            // Compute dmin contribution
            let mut mins_sum: i32 = 0;
            for j in 0..16 {
                let m = (x.scales[j] >> 4) as i32;
                let bsum = y.bsums[j] as i32;
                mins_sum += m * bsum;
            }
            sum -= dmin * mins_sum as f32;

            let mut sumi: i32 = 0;
            for j in 0..QK_K {
                let qs_idx = j / 4;
                let qs_shift = (j % 4) * 2;
                let q2 = ((x.qs[qs_idx] >> qs_shift) & 0x03) as i32;

                let scale = (x.scales[j / 16] & 0x0F) as i32;
                sumi += q2 * y.qs[j] as i32 * scale;
            }

            sum += d * sumi as f32;
        }

        sum
    }
}

// =============================================================================
// Public API: Dispatch to best implementation
// =============================================================================

/// Compute dot product of Q8K blocks.
pub fn vec_dot_q8k_q8k(xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::vec_dot_q8k_q8k(xs, ys)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        scalar::vec_dot_q8k_q8k(xs, ys)
    }
}

/// Compute dot product of Q4K x Q8K blocks.
pub fn vec_dot_q4k_q8k(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::vec_dot_q4k_q8k(xs, ys)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        scalar::vec_dot_q4k_q8k(xs, ys)
    }
}

/// Compute dot product of Q6K x Q8K blocks.
pub fn vec_dot_q6k_q8k(xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::vec_dot_q6k_q8k(xs, ys)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        scalar::vec_dot_q6k_q8k(xs, ys)
    }
}

/// Compute dot product of Q5K x Q8K blocks.
pub fn vec_dot_q5k_q8k(xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::vec_dot_q5k_q8k(xs, ys)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        scalar::vec_dot_q5k_q8k(xs, ys)
    }
}

/// Compute dot product of Q3K x Q8K blocks.
pub fn vec_dot_q3k_q8k(xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::vec_dot_q3k_q8k(xs, ys)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        scalar::vec_dot_q3k_q8k(xs, ys)
    }
}

/// Compute dot product of Q2K x Q8K blocks.
pub fn vec_dot_q2k_q8k(xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::vec_dot_q2k_q8k(xs, ys)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        scalar::vec_dot_q2k_q8k(xs, ys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_q8k(values: &[i8]) -> BlockQ8K {
        let mut block = BlockQ8K {
            d: 1.0,
            qs: [0i8; QK_K],
            bsums: [0i16; QK_K / 16],
        };
        for (i, &v) in values.iter().take(QK_K).enumerate() {
            block.qs[i] = v;
        }
        // Compute bsums
        for j in 0..QK_K / 16 {
            let mut sum = 0i16;
            for k in 0..16 {
                sum += block.qs[j * 16 + k] as i16;
            }
            block.bsums[j] = sum;
        }
        block
    }

    #[test]
    fn test_vec_dot_q8k_q8k_simple() {
        // Simple test: all 1s dot product
        let xs = vec![make_test_q8k(&[1i8; QK_K])];
        let ys = vec![make_test_q8k(&[1i8; QK_K])];

        let result = vec_dot_q8k_q8k(&xs, &ys);
        // With d=1.0 for both, result should be 256 (QK_K * 1 * 1)
        assert!(
            (result - 256.0).abs() < 1e-6,
            "Expected 256, got {}",
            result
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_vec_dot_q8k_q8k_alternating() {
        // Alternating signs should sum to 0
        let mut vals = [0i8; QK_K];
        for i in 0..QK_K {
            vals[i] = if i % 2 == 0 { 1 } else { -1 };
        }

        let xs = vec![make_test_q8k(&vals)];
        let ys = vec![make_test_q8k(&[1i8; QK_K])];

        let result = vec_dot_q8k_q8k(&xs, &ys);
        assert!(result.abs() < 1e-6, "Expected 0, got {}", result);
    }

    #[test]
    fn test_vec_dot_q8k_q8k_scaled() {
        let mut x = make_test_q8k(&[2i8; QK_K]);
        let mut y = make_test_q8k(&[3i8; QK_K]);
        x.d = 0.5;
        y.d = 2.0;

        let xs = vec![x];
        let ys = vec![y];

        let result = vec_dot_q8k_q8k(&xs, &ys);
        // d=0.5*2.0=1.0, sumi = 256 * (2*3) = 1536
        let expected = 1.0 * 1536.0;
        assert!(
            (result - expected).abs() < 1e-3,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_scalar_vs_neon_q8k() {
        // Create random-ish test data
        let mut vals_x = [0i8; QK_K];
        let mut vals_y = [0i8; QK_K];
        for i in 0..QK_K {
            vals_x[i] = ((i as i32 * 7 + 3) % 256 - 128) as i8;
            vals_y[i] = ((i as i32 * 11 + 17) % 256 - 128) as i8;
        }

        let xs = vec![make_test_q8k(&vals_x)];
        let ys = vec![make_test_q8k(&vals_y)];

        // Test scalar implementation
        let scalar_result = scalar::vec_dot_q8k_q8k(&xs, &ys);

        // Test dispatch (uses NEON on ARM64, scalar otherwise)
        let dispatch_result = vec_dot_q8k_q8k(&xs, &ys);

        // Results should match
        assert!(
            (scalar_result - dispatch_result).abs() < 1e-3,
            "Scalar: {}, Dispatch: {}",
            scalar_result,
            dispatch_result
        );
    }
}
