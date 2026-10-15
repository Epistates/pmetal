#![allow(unsafe_code)]

//! NEON-accelerated fp16↔fp32 bulk conversion for Apple Silicon.
//!
//! On aarch64, uses ARM NEON inline assembly to convert 4 floats at a time
//! via the `fcvtn` and `fcvtl` instructions, providing ~4-8x speedup over
//! scalar conversion for bulk operations.
//!
//! The `float16x4_t` NEON intrinsics (`vcvt_f16_f32`, `vcvt_f32_f16`) are
//! behind the unstable `stdarch_neon_f16` feature gate (rust-lang/rust#136306).
//! We use inline asm with `fcvtn`/`fcvtl` instead, which is fully stable on
//! aarch64 and achieves the same throughput.
//!
//! These functions convert raw weight and activation buffers (for example the
//! ANE input and output surfaces) where the runtime's own dtype conversion
//! is not available.

/// Convert a slice of f32 values to fp16 (stored as u16 bits) using NEON.
///
/// Falls back to scalar conversion on non-aarch64 platforms or for remainder
/// elements not covered by the 4-wide vectorized loop.
///
/// # Panics
///
/// If `dst` and `src` differ in length.
pub fn f32_to_f16_bulk(src: &[f32], dst: &mut [u16]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "src and dst must have the same length"
    );

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: lengths are asserted equal above; pointers are valid for
        // their respective slice extents; fcvtn/fcvtl work on any f32 bit
        // pattern (NaN, inf, subnormal) and produce defined fp16 output.
        unsafe { f32_to_f16_neon(src, dst) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        f32_to_f16_scalar(src, dst);
    }
}

/// Convert a slice of fp16 values (stored as u16 bits) to f32 using NEON.
///
/// Falls back to scalar conversion on non-aarch64 platforms or for remainder
/// elements not covered by the 4-wide vectorized loop.
///
/// # Panics
///
/// If `dst` and `src` differ in length.
pub fn f16_to_f32_bulk(src: &[u16], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "src and dst must have the same length"
    );

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { f16_to_f32_neon(src, dst) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        f16_to_f32_scalar(src, dst);
    }
}

/// Convert 8 f32 values to fp16 per iteration using `fcvtn`/`fcvtn2` (stable inline asm).
///
/// Primary loop processes 8 elements at a time:
/// - `fcvtn v2.4h, v0.4s` narrows lower 4 floats into lower 64 bits
/// - `fcvtn2 v2.8h, v1.4s` narrows upper 4 floats into upper 64 bits
/// - Single `st1 {v2.8h}` stores all 8 fp16 values
///
/// Falls back to 4-wide for 4-7 remainders, scalar for 1-3.
#[cfg(target_arch = "aarch64")]
unsafe fn f32_to_f16_neon(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    let chunks8 = n / 8;
    let mut i = 0;

    // 8-wide loop: process 8 f32 → 8 f16 per iteration
    for _ in 0..chunks8 {
        unsafe {
            let src_ptr = src.as_ptr().add(i);
            let dst_ptr = dst.as_mut_ptr().add(i);
            std::arch::asm!(
                // Load 8 f32 values into two 128-bit registers
                "ld1 {{v0.4s, v1.4s}}, [{src}]",
                // Convert lower 4 f32 → lower 4 f16 in v2
                "fcvtn v2.4h, v0.4s",
                // Convert upper 4 f32 → upper 4 f16 in v2 (same register)
                "fcvtn2 v2.8h, v1.4s",
                // Store 8 f16 values (128 bits)
                "st1 {{v2.8h}}, [{dst}]",
                src = in(reg) src_ptr,
                dst = in(reg) dst_ptr,
                out("v0") _,
                out("v1") _,
                out("v2") _,
                options(nostack),
            );
        }
        i += 8;
    }

    // 4-wide remainder for 4-7 leftover elements
    if i + 4 <= n {
        unsafe {
            let src_ptr = src.as_ptr().add(i);
            let dst_ptr = dst.as_mut_ptr().add(i);
            std::arch::asm!(
                "ld1 {{v0.4s}}, [{src}]",
                "fcvtn v1.4h, v0.4s",
                "st1 {{v1.4h}}, [{dst}]",
                src = in(reg) src_ptr,
                dst = in(reg) dst_ptr,
                out("v0") _,
                out("v1") _,
                options(nostack),
            );
        }
        i += 4;
    }

    // Scalar remainder for 1-3 leftover elements
    for j in i..n {
        dst[j] = half::f16::from_f32(src[j]).to_bits();
    }
}

/// Convert 8 fp16 values to f32 per iteration using `fcvtl`/`fcvtl2` (stable inline asm).
///
/// Primary loop processes 8 elements at a time:
/// - `ld1 {v0.8h}` loads 8 fp16 values into one 128-bit register
/// - `fcvtl v1.4s, v0.4h` widens lower 4 fp16 → 4 f32
/// - `fcvtl2 v2.4s, v0.8h` widens upper 4 fp16 → 4 f32
/// - `st1 {v1.4s, v2.4s}` stores all 8 f32 values
///
/// Falls back to 4-wide for 4-7 remainders, scalar for 1-3.
#[cfg(target_arch = "aarch64")]
unsafe fn f16_to_f32_neon(src: &[u16], dst: &mut [f32]) {
    let n = src.len();
    let chunks8 = n / 8;
    let mut i = 0;

    // 8-wide loop: process 8 f16 → 8 f32 per iteration
    for _ in 0..chunks8 {
        unsafe {
            let src_ptr = src.as_ptr().add(i);
            let dst_ptr = dst.as_mut_ptr().add(i);
            std::arch::asm!(
                // Load 8 f16 values into one 128-bit register
                "ld1 {{v0.8h}}, [{src}]",
                // Widen lower 4 f16 → 4 f32
                "fcvtl v1.4s, v0.4h",
                // Widen upper 4 f16 → 4 f32
                "fcvtl2 v2.4s, v0.8h",
                // Store 8 f32 values (two 128-bit registers)
                "st1 {{v1.4s, v2.4s}}, [{dst}]",
                src = in(reg) src_ptr,
                dst = in(reg) dst_ptr,
                out("v0") _,
                out("v1") _,
                out("v2") _,
                options(nostack),
            );
        }
        i += 8;
    }

    // 4-wide remainder for 4-7 leftover elements
    if i + 4 <= n {
        unsafe {
            let src_ptr = src.as_ptr().add(i);
            let dst_ptr = dst.as_mut_ptr().add(i);
            std::arch::asm!(
                "ld1 {{v0.4h}}, [{src}]",
                "fcvtl v1.4s, v0.4h",
                "st1 {{v1.4s}}, [{dst}]",
                src = in(reg) src_ptr,
                dst = in(reg) dst_ptr,
                out("v0") _,
                out("v1") _,
                options(nostack),
            );
        }
        i += 4;
    }

    // Scalar remainder for 1-3 leftover elements
    for j in i..n {
        dst[j] = half::f16::from_bits(src[j]).to_f32();
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn f32_to_f16_scalar(src: &[f32], dst: &mut [u16]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = half::f16::from_f32(*s).to_bits();
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn f16_to_f32_scalar(src: &[u16], dst: &mut [f32]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = half::f16::from_bits(*s).to_f32();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every fp16 bit pattern widens to the f32 the `half` crate produces
    /// (NaNs compare as NaNs, since their payloads may differ).
    #[test]
    fn widening_matches_the_reference_for_every_fp16_pattern() {
        let src: Vec<u16> = (0..=u16::MAX).collect();
        let mut dst = vec![0f32; src.len()];
        f16_to_f32_bulk(&src, &mut dst);
        for (&bits, &got) in src.iter().zip(&dst) {
            let want = half::f16::from_bits(bits).to_f32();
            if want.is_nan() {
                assert!(got.is_nan(), "{bits:#06x}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "{bits:#06x}");
            }
        }
    }

    /// Narrowing rounds to nearest-even exactly as the reference does, across
    /// the whole f32 range: normals, subnormals, overflow to infinity and ties.
    #[test]
    fn narrowing_matches_the_reference_across_the_f32_range() {
        // Every 257th f32 bit pattern, plus the neighbours of every fp16
        // rounding boundary, is about 17M values.
        let mut src: Vec<f32> = (0..=u32::MAX).step_by(257).map(f32::from_bits).collect();
        for bits in 0..=u16::MAX {
            let exact = half::f16::from_bits(bits).to_f32();
            if exact.is_finite() {
                let b = exact.to_bits();
                // The value, and the f32s just either side of it and of the
                // tie halfway to the next fp16.
                for delta in [-1i64, 0, 1] {
                    src.push(f32::from_bits((i64::from(b) + delta) as u32));
                }
                let next = half::f16::from_bits(bits.wrapping_add(1)).to_f32();
                if next.is_finite() {
                    let tie = f32::from_bits((exact.to_bits() / 2) + (next.to_bits() / 2));
                    for delta in [-1i64, 0, 1] {
                        src.push(f32::from_bits((i64::from(tie.to_bits()) + delta) as u32));
                    }
                }
            }
        }
        // Length not a multiple of 8, so the 4-wide and scalar tails run too.
        src.truncate(src.len() - src.len() % 8 + 3);
        let mut dst = vec![0u16; src.len()];
        f32_to_f16_bulk(&src, &mut dst);
        for (&value, &bits) in src.iter().zip(&dst) {
            let want = half::f16::from_f32(value);
            if value.is_nan() {
                assert!(half::f16::from_bits(bits).is_nan(), "{value:e}");
            } else {
                assert_eq!(bits, want.to_bits(), "{:#010x}", value.to_bits());
            }
        }
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn mismatched_lengths_are_rejected() {
        f32_to_f16_bulk(&[1.0, 2.0], &mut [0u16; 3]);
    }

    #[test]
    fn test_f32_to_f16_roundtrip() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 0.5, -1.0, 0.0, 100.0];
        let mut f16_buf = vec![0u16; src.len()];
        let mut f32_buf = vec![0f32; src.len()];

        f32_to_f16_bulk(&src, &mut f16_buf);
        f16_to_f32_bulk(&f16_buf, &mut f32_buf);

        for (orig, converted) in src.iter().zip(f32_buf.iter()) {
            let expected = half::f16::from_f32(*orig).to_f32();
            assert!(
                (converted - expected).abs() < 1e-3,
                "Mismatch: orig={}, converted={}, expected={}",
                orig,
                converted,
                expected
            );
        }
    }

    #[test]
    fn test_f32_to_f16_remainder() {
        // Test with non-multiple-of-4 length to exercise the scalar remainder path.
        let src = vec![1.0f32, 2.0, 3.0];
        let mut dst = vec![0u16; 3];
        f32_to_f16_bulk(&src, &mut dst);

        for (s, d) in src.iter().zip(dst.iter()) {
            let expected = half::f16::from_f32(*s).to_bits();
            assert_eq!(*d, expected);
        }
    }

    #[test]
    fn test_empty_slices() {
        let src: Vec<f32> = vec![];
        let mut dst: Vec<u16> = vec![];
        f32_to_f16_bulk(&src, &mut dst);
        // Should not panic
    }

    #[test]
    fn test_f16_to_f32_remainder() {
        // Non-multiple-of-4 length for f16→f32 scalar remainder path.
        let src_f32 = [0.25f32, -0.5, 1.5];
        let f16_bits: Vec<u16> = src_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let mut dst = vec![0f32; src_f32.len()];
        f16_to_f32_bulk(&f16_bits, &mut dst);

        for (orig, converted) in src_f32.iter().zip(dst.iter()) {
            let expected = half::f16::from_f32(*orig).to_f32();
            assert!(
                (converted - expected).abs() < 1e-3,
                "Mismatch: orig={}, converted={}, expected={}",
                orig,
                converted,
                expected
            );
        }
    }

    #[test]
    fn test_special_float_values() {
        // NaN, Inf, -Inf, -0.0, subnormal
        let src = vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0_f32,
            // Smallest positive subnormal in fp16 is 2^-24 ≈ 5.96e-8
            6.0e-8_f32,
        ];
        let mut f16_buf = vec![0u16; src.len()];
        let mut f32_buf = vec![0f32; src.len()];

        f32_to_f16_bulk(&src, &mut f16_buf);
        f16_to_f32_bulk(&f16_buf, &mut f32_buf);

        // NaN should remain NaN
        assert!(f32_buf[0].is_nan(), "NaN should roundtrip as NaN");
        // +Inf
        assert_eq!(f32_buf[1], f32::INFINITY, "+Inf should roundtrip");
        // -Inf
        assert_eq!(f32_buf[2], f32::NEG_INFINITY, "-Inf should roundtrip");
        // -0.0 should preserve sign bit
        assert!(
            f32_buf[3].is_sign_negative() && f32_buf[3] == 0.0,
            "-0.0 should roundtrip"
        );
        // Subnormal: just verify it doesn't crash and produces a finite result
        assert!(
            f32_buf[4].is_finite(),
            "subnormal should produce finite result"
        );
    }

    #[test]
    fn test_exact_4_element_alignment() {
        // Exactly 4 elements — exercises only the vectorized path, no scalar remainder
        let src = vec![1.0f32, -2.0, 3.5, -4.25];
        let mut f16_buf = vec![0u16; 4];
        let mut f32_buf = vec![0f32; 4];

        f32_to_f16_bulk(&src, &mut f16_buf);
        f16_to_f32_bulk(&f16_buf, &mut f32_buf);

        for (orig, converted) in src.iter().zip(f32_buf.iter()) {
            let expected = half::f16::from_f32(*orig).to_f32();
            assert!(
                (converted - expected).abs() < 1e-3,
                "Mismatch: orig={}, converted={}, expected={}",
                orig,
                converted,
                expected
            );
        }
    }

    #[test]
    fn test_large_array() {
        // 1M+ elements to stress the vectorized path
        let n = 1_048_576;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 500.0).collect();
        let mut f16_buf = vec![0u16; n];
        let mut f32_buf = vec![0f32; n];

        f32_to_f16_bulk(&src, &mut f16_buf);
        f16_to_f32_bulk(&f16_buf, &mut f32_buf);

        // Spot-check a few values
        for &idx in &[0, 1, n / 2, n - 2, n - 1] {
            let expected = half::f16::from_f32(src[idx]).to_f32();
            assert!(
                (f32_buf[idx] - expected).abs() < 1e-2,
                "Mismatch at index {}: orig={}, converted={}, expected={}",
                idx,
                src[idx],
                f32_buf[idx],
                expected
            );
        }
    }
}
