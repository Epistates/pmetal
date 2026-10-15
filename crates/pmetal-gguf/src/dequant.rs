//! Dequantization of GGUF tensor data to `f32`.
//!
//! Every tensor type a GGUF file can carry (see [`GgmlType`]) is decoded here
//! except the architecture-specific *repacked* layouts (`Q4_0_4_4` and
//! friends), which interleave rows for a particular kernel and are not
//! portable tensors.
//!
//! # Exactness
//!
//! Each decoder follows the reference block definition from the ggml sources
//! (`ggml-common.h` for the layouts, `ggml-quants.c` for the arithmetic) and
//! forms its products in the same order, so the results are bit-identical to
//! the reference rather than merely close. `tests/dequant_reference.rs` pins
//! this against values produced by the reference decoder.
//!
//! # Rows and blocks
//!
//! Quantized formats store whole blocks of 32, 64, 128 or 256 weights, and a
//! block never straddles a row. A tensor shape is therefore only valid for a
//! quantized type when its innermost dimension is a multiple of the block
//! size; [`dequantize`] rejects anything else with
//! [`DequantError::RowNotBlockAligned`] instead of silently truncating.
//!
//! # Performance
//!
//! Large tensors are decoded in parallel, one block per work item, writing
//! straight into the output buffer ([`dequantize_into`] avoids the
//! allocation entirely).

use rayon::prelude::*;

use crate::GgmlType;
use crate::fp4::{KVALUES_MXFP4, e8m0_to_fp32_half, ue4m3_to_fp32};
use crate::iq_quants as iq;
use crate::k_quants as kq;

/// Error type for dequantization.
#[derive(Debug, thiserror::Error)]
pub enum DequantError {
    /// Unsupported quantization type.
    #[error("Unsupported quantization type: {0:?}")]
    UnsupportedType(GgmlType),
    /// Invalid data size for the given shape.
    #[error("Invalid data size: expected {expected} bytes, got {actual}")]
    InvalidSize {
        /// Expected size in bytes.
        expected: usize,
        /// Actual size in bytes.
        actual: usize,
    },
    /// The output buffer does not hold exactly one `f32` per element.
    #[error("Output buffer holds {actual} values but the tensor has {expected} elements")]
    InvalidOutputLength {
        /// Element count of the tensor.
        expected: usize,
        /// Length of the output buffer.
        actual: usize,
    },
    /// Shape mismatch.
    #[error("Shape mismatch: cannot reshape {elements} elements to {shape:?}")]
    ShapeMismatch {
        /// Number of elements.
        elements: usize,
        /// Target shape.
        shape: Vec<i32>,
    },
    /// Invalid shape dimension.
    #[error("Invalid shape {shape:?}: dimensions must be non-negative")]
    InvalidShape {
        /// Target shape.
        shape: Vec<i32>,
    },
    /// Shape element count overflowed usize.
    #[error("Shape {shape:?} is too large")]
    ShapeTooLarge {
        /// Target shape.
        shape: Vec<i32>,
    },
    /// The innermost dimension is not a whole number of blocks.
    #[error(
        "Row length {row_len} is not a multiple of the {dtype:?} block size {block_size} (shape {shape:?})"
    )]
    RowNotBlockAligned {
        /// Tensor type.
        dtype: GgmlType,
        /// Block size of the type, in elements.
        block_size: usize,
        /// Length of the innermost dimension.
        row_len: usize,
        /// Tensor shape.
        shape: Vec<i32>,
    },
}

/// Below this many elements decoding runs on the calling thread.
const PARALLEL_MIN_ELEMENTS: usize = 1 << 16;
/// Blocks handed to one parallel task at minimum, to amortise scheduling.
const PARALLEL_MIN_BLOCKS: usize = 64;

/// Dequantize raw tensor data to f32 values.
///
/// # Arguments
///
/// * `data` - Raw quantized bytes from the GGUF file
/// * `dtype` - The tensor type
/// * `shape` - Tensor shape in row-major order (outermost dimension first)
///
/// # Returns
///
/// The tensor as `f32` values in row-major order.
///
/// # Example
///
/// ```ignore
/// let data = content.read_tensor_data(&mut file, "blk.0.attn_q.weight")?;
/// let info = content.get_tensor_info("blk.0.attn_q.weight").unwrap();
/// let shape: Vec<i32> = info.dimensions.iter().map(|&d| d as i32).collect();
/// let floats = dequantize(&data, info.dtype, &shape)?;
/// ```
pub fn dequantize(data: &[u8], dtype: GgmlType, shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = checked_element_count(dtype, shape)?;
    let mut out = vec![0.0f32; n_elements];
    dequantize_validated(data, dtype, shape, &mut out)?;
    Ok(out)
}

/// Dequantize into a caller-provided buffer.
///
/// `out` must hold exactly one `f32` per tensor element. The shape is given
/// as in [`dequantize`]; use this to decode straight into a pre-allocated
/// (for example pooled or memory-mapped) buffer.
pub fn dequantize_into(
    data: &[u8],
    dtype: GgmlType,
    shape: &[i32],
    out: &mut [f32],
) -> Result<(), DequantError> {
    let n_elements = checked_element_count(dtype, shape)?;
    if out.len() != n_elements {
        return Err(DequantError::InvalidOutputLength {
            expected: n_elements,
            actual: out.len(),
        });
    }
    dequantize_validated(data, dtype, shape, out)
}

/// Check if a tensor type can be dequantized.
pub fn is_supported(dtype: GgmlType) -> bool {
    !matches!(
        dtype,
        GgmlType::Q4_0_4_4
            | GgmlType::Q4_0_4_8
            | GgmlType::Q4_0_8_8
            | GgmlType::Iq4Nl_4_4
            | GgmlType::Iq4Nl_4_8
            | GgmlType::Iq4Nl_8_8
    )
}

/// Get the expected byte size for a tensor with given dtype and shape.
///
/// # Panics
/// Panics if the shape is invalid; use [`expected_byte_size_checked`] for
/// untrusted shapes.
pub fn expected_byte_size(dtype: GgmlType, shape: &[i32]) -> usize {
    expected_byte_size_checked(dtype, shape)
        .expect("invalid GGUF tensor shape for expected_byte_size")
}

/// Get the expected byte size for a tensor with checked shape arithmetic.
pub fn expected_byte_size_checked(dtype: GgmlType, shape: &[i32]) -> Result<usize, DequantError> {
    let n_elements = checked_element_count(dtype, shape)?;
    dtype
        .tensor_size_checked(n_elements)
        .ok_or_else(|| DequantError::ShapeTooLarge {
            shape: shape.to_vec(),
        })
}

/// Element count of `shape`, rejecting negative dimensions, overflow and
/// (for blocked types) rows that are not a whole number of blocks.
fn checked_element_count(dtype: GgmlType, shape: &[i32]) -> Result<usize, DequantError> {
    if shape.iter().any(|&d| d < 0) {
        return Err(DequantError::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    let n_elements = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim as usize)
            .ok_or_else(|| DequantError::ShapeTooLarge {
                shape: shape.to_vec(),
            })
    })?;
    let block_size = dtype.block_size();
    if block_size > 1 && n_elements > 0 {
        let row_len = shape.last().map_or(1, |&d| d as usize);
        if row_len % block_size != 0 {
            return Err(DequantError::RowNotBlockAligned {
                dtype,
                block_size,
                row_len,
                shape: shape.to_vec(),
            });
        }
    }
    Ok(n_elements)
}

fn dequantize_validated(
    data: &[u8],
    dtype: GgmlType,
    shape: &[i32],
    out: &mut [f32],
) -> Result<(), DequantError> {
    if !is_supported(dtype) {
        return Err(DequantError::UnsupportedType(dtype));
    }
    let expected =
        dtype
            .tensor_size_checked(out.len())
            .ok_or_else(|| DequantError::ShapeTooLarge {
                shape: shape.to_vec(),
            })?;
    if data.len() != expected {
        return Err(DequantError::InvalidSize {
            expected,
            actual: data.len(),
        });
    }

    match dtype {
        GgmlType::F32 => convert_elements::<4>(data, out, f32::from_le_bytes),
        GgmlType::F16 => convert_elements::<2>(data, out, |b| half::f16::from_le_bytes(b).to_f32()),
        GgmlType::Bf16 => {
            convert_elements::<2>(data, out, |b| half::bf16::from_le_bytes(b).to_f32())
        }
        GgmlType::F64 => convert_elements::<8>(data, out, |b| f64::from_le_bytes(b) as f32),
        GgmlType::I8 => convert_elements::<1>(data, out, |b| f32::from(i8::from_le_bytes(b))),
        GgmlType::I16 => convert_elements::<2>(data, out, |b| f32::from(i16::from_le_bytes(b))),
        GgmlType::I32 => convert_elements::<4>(data, out, |b| i32::from_le_bytes(b) as f32),
        GgmlType::I64 => convert_elements::<8>(data, out, |b| i64::from_le_bytes(b) as f32),

        GgmlType::Q4_0 => decode_blocks(data, out, dequantize_block_q4_0),
        GgmlType::Q4_1 => decode_blocks(data, out, dequantize_block_q4_1),
        GgmlType::Q5_0 => decode_blocks(data, out, dequantize_block_q5_0),
        GgmlType::Q5_1 => decode_blocks(data, out, dequantize_block_q5_1),
        GgmlType::Q8_0 => decode_blocks(data, out, dequantize_block_q8_0),
        GgmlType::Q8_1 => decode_blocks(data, out, dequantize_block_q8_1),
        GgmlType::Q1_0 => decode_blocks(data, out, dequantize_block_q1_0),
        GgmlType::Tq1_0 => decode_blocks(data, out, dequantize_block_tq1_0),
        GgmlType::Tq2_0 => decode_blocks(data, out, dequantize_block_tq2_0),
        GgmlType::Mxfp4 => decode_blocks(data, out, dequantize_block_mxfp4),
        GgmlType::Nvfp4 => decode_blocks(data, out, dequantize_block_nvfp4),

        GgmlType::Q2K => decode_blocks(data, out, kq::dequantize_block_q2_k),
        GgmlType::Q3K => decode_blocks(data, out, kq::dequantize_block_q3_k),
        GgmlType::Q4K => decode_blocks(data, out, kq::dequantize_block_q4_k),
        GgmlType::Q5K => decode_blocks(data, out, kq::dequantize_block_q5_k),
        GgmlType::Q6K => decode_blocks(data, out, kq::dequantize_block_q6_k),
        GgmlType::Q8K => decode_blocks(data, out, kq::dequantize_block_q8_k),

        GgmlType::Iq4Nl => decode_blocks(data, out, iq::dequantize_block_iq4_nl),
        GgmlType::Iq4Xs => decode_blocks(data, out, iq::dequantize_block_iq4_xs),
        GgmlType::Iq2Xxs => decode_blocks(data, out, iq::dequantize_block_iq2_xxs),
        GgmlType::Iq2Xs => decode_blocks(data, out, iq::dequantize_block_iq2_xs),
        GgmlType::Iq2S => decode_blocks(data, out, iq::dequantize_block_iq2_s),
        GgmlType::Iq3Xxs => decode_blocks(data, out, iq::dequantize_block_iq3_xxs),
        GgmlType::Iq3S => decode_blocks(data, out, iq::dequantize_block_iq3_s),
        GgmlType::Iq1S => decode_blocks(data, out, iq::dequantize_block_iq1_s),
        GgmlType::Iq1M => decode_blocks(data, out, iq::dequantize_block_iq1_m),

        GgmlType::Q4_0_4_4
        | GgmlType::Q4_0_4_8
        | GgmlType::Q4_0_8_8
        | GgmlType::Iq4Nl_4_4
        | GgmlType::Iq4Nl_4_8
        | GgmlType::Iq4Nl_8_8 => return Err(DequantError::UnsupportedType(dtype)),
    }
    Ok(())
}

/// Decode a buffer of fixed-width scalar elements.
fn convert_elements<const W: usize>(data: &[u8], out: &mut [f32], f: fn([u8; W]) -> f32) {
    let (src, _) = data.as_chunks::<W>();
    debug_assert_eq!(src.len(), out.len());
    if out.len() < PARALLEL_MIN_ELEMENTS {
        for (o, s) in out.iter_mut().zip(src) {
            *o = f(*s);
        }
    } else {
        out.par_iter_mut()
            .zip(src.par_iter())
            .with_min_len(PARALLEL_MIN_ELEMENTS / 4)
            .for_each(|(o, s)| *o = f(*s));
    }
}

/// Decode a buffer of `B`-byte blocks holding `E` elements each.
///
/// `data` and `out` have been validated to hold the same whole number of
/// blocks.
fn decode_blocks<const B: usize, const E: usize>(
    data: &[u8],
    out: &mut [f32],
    decode: fn(&[u8; B], &mut [f32; E]),
) {
    let (src, _) = data.as_chunks::<B>();
    let (dst, _) = out.as_chunks_mut::<E>();
    debug_assert_eq!(src.len(), dst.len());
    if dst.len() * E < PARALLEL_MIN_ELEMENTS {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            decode(s, d);
        }
    } else {
        src.par_iter()
            .zip(dst.par_iter_mut())
            .with_min_len(PARALLEL_MIN_BLOCKS)
            .for_each(|(s, d)| decode(s, d));
    }
}

#[inline(always)]
fn read_f16(block: &[u8], at: usize) -> f32 {
    half::f16::from_le_bytes([block[at], block[at + 1]]).to_f32()
}

/// Q1_0: `d: f16`, then 128 sign bits, LSB first. A set bit is `+d`.
fn dequantize_block_q1_0(block: &[u8; 18], y: &mut [f32; 128]) {
    let d = read_f16(block, 0);
    for (i, o) in y.iter_mut().enumerate() {
        let bit = (block[2 + i / 8] >> (i % 8)) & 1;
        *o = if bit == 1 { d } else { -d };
    }
}

/// Q4_0: `d: f16`, 16 bytes of nibbles. Low nibbles hold elements 0..16 and
/// high nibbles elements 16..32. `value = (q - 8) * d`.
fn dequantize_block_q4_0(block: &[u8; 18], y: &mut [f32; 32]) {
    let d = read_f16(block, 0);
    for j in 0..16 {
        let b = block[2 + j];
        y[j] = (i32::from(b & 0x0f) - 8) as f32 * d;
        y[j + 16] = (i32::from(b >> 4) - 8) as f32 * d;
    }
}

/// Q4_1: `d: f16`, `m: f16`, 16 bytes of nibbles. `value = q * d + m`.
fn dequantize_block_q4_1(block: &[u8; 20], y: &mut [f32; 32]) {
    let d = read_f16(block, 0);
    let m = read_f16(block, 2);
    for j in 0..16 {
        let b = block[4 + j];
        y[j] = f32::from(b & 0x0f) * d + m;
        y[j + 16] = f32::from(b >> 4) * d + m;
    }
}

/// Q5_0: `d: f16`, `qh: u32` (fifth bits), 16 bytes of nibbles.
/// `value = (q - 16) * d`.
fn dequantize_block_q5_0(block: &[u8; 22], y: &mut [f32; 32]) {
    let d = read_f16(block, 0);
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    for j in 0..16 {
        let b = block[6 + j];
        let lo = i32::from(b & 0x0f) | (((qh >> j) & 1) << 4) as i32;
        let hi = i32::from(b >> 4) | (((qh >> (j + 16)) & 1) << 4) as i32;
        y[j] = (lo - 16) as f32 * d;
        y[j + 16] = (hi - 16) as f32 * d;
    }
}

/// Q5_1: `d: f16`, `m: f16`, `qh: u32`, 16 bytes of nibbles.
/// `value = q * d + m`.
fn dequantize_block_q5_1(block: &[u8; 24], y: &mut [f32; 32]) {
    let d = read_f16(block, 0);
    let m = read_f16(block, 2);
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for j in 0..16 {
        let b = block[8 + j];
        let lo = i32::from(b & 0x0f) | (((qh >> j) & 1) << 4) as i32;
        let hi = i32::from(b >> 4) | (((qh >> (j + 16)) & 1) << 4) as i32;
        y[j] = lo as f32 * d + m;
        y[j + 16] = hi as f32 * d + m;
    }
}

/// Q8_0: `d: f16`, 32 signed bytes. `value = q * d`.
fn dequantize_block_q8_0(block: &[u8; 34], y: &mut [f32; 32]) {
    let d = read_f16(block, 0);
    for (o, &q) in y.iter_mut().zip(&block[2..]) {
        *o = f32::from(q as i8) * d;
    }
}

/// Q8_1: `d: f16`, `s: f16` (sum helper for dot kernels, unused here),
/// 32 signed bytes. `value = q * d`.
fn dequantize_block_q8_1(block: &[u8; 36], y: &mut [f32; 32]) {
    let d = read_f16(block, 0);
    for (o, &q) in y.iter_mut().zip(&block[4..]) {
        *o = f32::from(q as i8) * d;
    }
}

/// MXFP4: one E8M0 shared exponent, 16 bytes of E2M1 nibbles (low nibbles are
/// elements 0..16, high nibbles 16..32).
fn dequantize_block_mxfp4(block: &[u8; 17], y: &mut [f32; 32]) {
    let d = e8m0_to_fp32_half(block[0]);
    for j in 0..16 {
        let b = block[1 + j];
        y[j] = f32::from(KVALUES_MXFP4[usize::from(b & 0x0f)]) * d;
        y[j + 16] = f32::from(KVALUES_MXFP4[usize::from(b >> 4)]) * d;
    }
}

/// NVFP4: four UE4M3 scales (one per 16 elements) then 32 bytes of E2M1
/// nibbles; within a sub-block the low nibbles are elements 0..8 and the high
/// nibbles 8..16.
fn dequantize_block_nvfp4(block: &[u8; 36], y: &mut [f32; 64]) {
    for sub in 0..4 {
        let d = ue4m3_to_fp32(block[sub]);
        for j in 0..8 {
            let b = block[4 + 8 * sub + j];
            y[16 * sub + j] = f32::from(KVALUES_MXFP4[usize::from(b & 0x0f)]) * d;
            y[16 * sub + j + 8] = f32::from(KVALUES_MXFP4[usize::from(b >> 4)]) * d;
        }
    }
}

/// TQ1_0: 5 ternary digits per byte in `qs[48]`, 4 per byte in `qh[4]`, then
/// `d: f16`. A digit `t` in `{0,1,2}` decodes to `(t - 1) * d`.
fn dequantize_block_tq1_0(block: &[u8; 54], y: &mut [f32; 256]) {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let qs = &block[..48];
    let qh = &block[48..52];
    let d = read_f16(block, 52);

    #[inline(always)]
    fn digit(byte: u8, pow: u8) -> i32 {
        let q = byte.wrapping_mul(pow);
        ((u16::from(q) * 3) >> 8) as i32 - 1
    }

    let mut out = 0;
    // 32-byte groups (5 digits per byte), then the 16-byte remainder.
    for (start, width) in [(0usize, 32usize), (32, 16)] {
        for pow in &POW3[..5] {
            for m in 0..width {
                y[out] = digit(qs[start + m], *pow) as f32 * d;
                out += 1;
            }
        }
    }
    for pow in &POW3[..4] {
        for &b in qh {
            y[out] = digit(b, *pow) as f32 * d;
            out += 1;
        }
    }
    debug_assert_eq!(out, 256);
}

/// TQ2_0: four 2-bit ternary values per byte in `qs[64]`, then `d: f16`.
fn dequantize_block_tq2_0(block: &[u8; 66], y: &mut [f32; 256]) {
    let d = read_f16(block, 64);
    for j in 0..2 {
        for lane in 0..4 {
            for m in 0..32 {
                let q = i32::from((block[32 * j + m] >> (2 * lane)) & 3);
                y[128 * j + 32 * lane + m] = (q - 1) as f32 * d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    #[test]
    fn scalar_types_round_trip() {
        let f32s: Vec<u8> = [1.0f32, -2.5, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            dequantize(&f32s, GgmlType::F32, &[3]).unwrap(),
            vec![1.0, -2.5, 3.0]
        );

        let f16s: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|&v| f16_bytes(v)).collect();
        assert_eq!(
            dequantize(&f16s, GgmlType::F16, &[2]).unwrap(),
            vec![1.0, 2.0]
        );

        let bf16s: Vec<u8> = [1.0f32, -2.0]
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        assert_eq!(
            dequantize(&bf16s, GgmlType::Bf16, &[2]).unwrap(),
            vec![1.0, -2.0]
        );

        let i16s: Vec<u8> = [-3i16, 7].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            dequantize(&i16s, GgmlType::I16, &[2]).unwrap(),
            vec![-3.0, 7.0]
        );
        assert_eq!(
            dequantize(&[0xff, 0x02], GgmlType::I8, &[2]).unwrap(),
            vec![-1.0, 2.0]
        );
    }

    #[test]
    fn q4_0_places_low_nibbles_first() {
        let mut data = f16_bytes(0.5).to_vec();
        // byte j holds element j in the low nibble and element j+16 in the high nibble
        data.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        let out = dequantize(&data, GgmlType::Q4_0, &[32]).unwrap();
        for j in 0..16 {
            assert_eq!(out[j], (j as f32 - 8.0) * 0.5);
            assert_eq!(out[j + 16], ((15 - j) as f32 - 8.0) * 0.5);
        }
    }

    #[test]
    fn wrong_data_length_is_an_error() {
        assert!(matches!(
            dequantize(&[0u8; 10], GgmlType::F32, &[4]),
            Err(DequantError::InvalidSize {
                expected: 16,
                actual: 10
            })
        ));
        assert!(matches!(
            dequantize(&[0u8; 17], GgmlType::Q4_0, &[32]),
            Err(DequantError::InvalidSize {
                expected: 18,
                actual: 17
            })
        ));
    }

    #[test]
    fn rows_must_be_whole_blocks() {
        // 2 x 48 elements is 3 blocks of Q4_0 in total, but a row of 48 is
        // 1.5 blocks: blocks never straddle rows.
        let data = vec![0u8; 3 * 18];
        assert!(matches!(
            dequantize(&data, GgmlType::Q4_0, &[2, 48]),
            Err(DequantError::RowNotBlockAligned {
                row_len: 48,
                block_size: 32,
                ..
            })
        ));
        assert!(dequantize(&data, GgmlType::Q4_0, &[3, 32]).is_ok());
    }

    #[test]
    fn invalid_shapes_are_errors() {
        assert!(matches!(
            dequantize(&[], GgmlType::F32, &[-1]),
            Err(DequantError::InvalidShape { .. })
        ));
        assert!(matches!(
            dequantize(
                &[],
                GgmlType::F32,
                &[i32::MAX, i32::MAX, i32::MAX, i32::MAX]
            ),
            Err(DequantError::ShapeTooLarge { .. })
        ));
    }

    #[test]
    fn output_buffer_length_is_checked() {
        let mut out = [0.0f32; 3];
        assert!(matches!(
            dequantize_into(&[0u8; 16], GgmlType::F32, &[4], &mut out),
            Err(DequantError::InvalidOutputLength {
                expected: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn repacked_layouts_are_rejected() {
        assert!(!is_supported(GgmlType::Q4_0_4_4));
        assert!(!is_supported(GgmlType::Iq4Nl_8_8));
        assert!(matches!(
            dequantize(&[0u8; 18], GgmlType::Q4_0_4_4, &[32]),
            Err(DequantError::UnsupportedType(_))
        ));
    }

    #[test]
    fn every_other_type_is_supported() {
        for id in 0..=41u32 {
            if let Ok(t) = GgmlType::try_from(id) {
                let repacked = matches!(id, 31..=33 | 36..=38);
                assert_eq!(is_supported(t), !repacked, "{t:?}");
            }
        }
    }

    #[test]
    fn parallel_and_serial_decoding_agree() {
        // 2^17 elements is above the parallel threshold; compare with a
        // block-at-a-time decode.
        let n_blocks = (1 << 17) / 32;
        let mut data = Vec::with_capacity(n_blocks * 34);
        let mut state = 0x1234_5678u32;
        for _ in 0..n_blocks {
            data.extend_from_slice(&f16_bytes(0.25));
            for _ in 0..32 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                data.push((state >> 24) as u8);
            }
        }
        let parallel = dequantize(&data, GgmlType::Q8_0, &[n_blocks as i32, 32]).unwrap();
        let mut serial = vec![0.0f32; n_blocks * 32];
        for (s, d) in data
            .as_chunks::<34>()
            .0
            .iter()
            .zip(serial.as_chunks_mut::<32>().0)
        {
            dequantize_block_q8_0(s, d);
        }
        assert_eq!(parallel, serial);
    }
}
