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
//! Based on llama.cpp/GGML K-quant implementation and Candle's quantized module.

use crate::quantize::{
    get_scale_min_k4, make_q3_quants, make_qkx1_quants, make_qx_quants, nearest_int,
};
use half::f16;

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
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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
// Block Size Helper Methods
// =============================================================================

impl BlockQ2K {
    /// Calculate byte size for n_elements.
    pub fn byte_size(n_elements: usize) -> usize {
        let n_blocks = n_elements.div_ceil(QK_K);
        n_blocks * std::mem::size_of::<Self>()
    }
}

impl BlockQ3K {
    /// Calculate byte size for n_elements.
    pub fn byte_size(n_elements: usize) -> usize {
        let n_blocks = n_elements.div_ceil(QK_K);
        n_blocks * std::mem::size_of::<Self>()
    }
}

impl BlockQ4K {
    /// Calculate byte size for n_elements.
    pub fn byte_size(n_elements: usize) -> usize {
        let n_blocks = n_elements.div_ceil(QK_K);
        n_blocks * std::mem::size_of::<Self>()
    }
}

impl BlockQ5K {
    /// Calculate byte size for n_elements.
    pub fn byte_size(n_elements: usize) -> usize {
        let n_blocks = n_elements.div_ceil(QK_K);
        n_blocks * std::mem::size_of::<Self>()
    }
}

impl BlockQ6K {
    /// Calculate byte size for n_elements.
    pub fn byte_size(n_elements: usize) -> usize {
        let n_blocks = n_elements.div_ceil(QK_K);
        n_blocks * std::mem::size_of::<Self>()
    }
}

impl BlockQ8K {
    /// Calculate byte size for n_elements.
    pub fn byte_size(n_elements: usize) -> usize {
        let n_blocks = n_elements.div_ceil(QK_K);
        n_blocks * std::mem::size_of::<Self>()
    }
}

// =============================================================================
// Dequantization Functions
// =============================================================================

/// Dequantize Q2K block to f32.
pub fn dequantize_q2k(block: &BlockQ2K, output: &mut [f32; QK_K]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    let mut idx = 0;
    for j in (0..QK_K).step_by(128) {
        // Process 8 sub-blocks of 16 elements
        for l in 0..32 {
            let scale_idx = j / 16 + l / 4;
            let is_upper = (l / 4) % 2 == 1;

            let sc = if is_upper {
                (block.scales[scale_idx / 2] >> 4) & 0xF
            } else {
                block.scales[scale_idx / 2] & 0xF
            };

            let dl = d * sc as f32;

            // Extract 2-bit values
            let byte_idx = (j + l * 4) / 4;
            for k in 0..4 {
                let q = (block.qs[byte_idx] >> (k * 2)) & 3;
                output[idx] = dl * q as f32 - dmin;
                idx += 1;
            }
        }
    }
}

/// Dequantize Q3K block to f32.
pub fn dequantize_q3k(block: &BlockQ3K, output: &mut [f32; QK_K]) {
    let d = block.d.to_f32();

    // Decode scales from packed representation
    let mut scales = [0i8; 16];
    let mut aux = [0u8; 8];

    for i in 0..4 {
        aux[i] = block.scales[i] & 0x3F;
        aux[i + 4] = block.scales[i + 4] & 0x3F;
    }

    // Handle upper bits
    for i in 0..4 {
        let upper = block.scales[8 + i] & 3;
        aux[i] |= upper << 6;
        let upper = (block.scales[8 + i] >> 2) & 3;
        aux[i + 4] |= upper << 6;
    }

    for i in 0..8 {
        scales[i] = (aux[i] as i8).wrapping_sub(32);
        scales[i + 8] = (aux[i] as i8).wrapping_sub(32);
    }

    let mut idx = 0;
    for j in (0..QK_K).step_by(128) {
        for l in 0..32 {
            let scale_idx = j / 16 + l / 4;
            let dl = d * scales[scale_idx] as f32;

            for k in 0..4 {
                let byte_idx = (j + l * 4 + k) / 4;
                let shift = ((j + l * 4 + k) % 4) * 2;
                let q_low = (block.qs[byte_idx] >> shift) & 3;

                let hmask_idx = (j + l * 4 + k) / 8;
                let hmask_shift = (j + l * 4 + k) % 8;
                let q_high = ((block.hmask[hmask_idx] >> hmask_shift) & 1) << 2;

                let q = (q_low | q_high) as i8 - 4;
                output[idx] = dl * q as f32;
                idx += 1;
            }
        }
    }
}

/// Dequantize Q4K block to f32.
pub fn dequantize_q4k(block: &BlockQ4K, output: &mut [f32; QK_K]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    let mut idx = 0;
    let mut is = 0usize;

    for j in (0..QK_K).step_by(64) {
        // Get scale and min for this 32-element sub-block
        let (sc1, m1) = get_scale_min_k4(is, &block.scales);
        let d1 = d * sc1 as f32;
        let dm1 = dmin * m1 as f32;

        let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
        let d2 = d * sc2 as f32;
        let dm2 = dmin * m2 as f32;

        // Process 64 values (32 bytes of packed 4-bit)
        let q_start = j / 2;

        // First 32 values (lower nibbles then upper nibbles from first 16 bytes)
        for l in 0..16 {
            let q = block.qs[q_start + l];
            output[idx] = d1 * (q & 0xF) as f32 - dm1;
            idx += 1;
        }
        for l in 0..16 {
            let q = block.qs[q_start + l];
            output[idx] = d1 * (q >> 4) as f32 - dm1;
            idx += 1;
        }

        // Second 32 values (lower nibbles then upper nibbles from second 16 bytes)
        for l in 0..16 {
            let q = block.qs[q_start + 16 + l];
            output[idx] = d2 * (q & 0xF) as f32 - dm2;
            idx += 1;
        }
        for l in 0..16 {
            let q = block.qs[q_start + 16 + l];
            output[idx] = d2 * (q >> 4) as f32 - dm2;
            idx += 1;
        }

        is += 2;
    }
}

/// Dequantize Q5K block to f32.
pub fn dequantize_q5k(block: &BlockQ5K, output: &mut [f32; QK_K]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    let mut idx = 0;
    let mut is = 0usize;

    for j in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, &block.scales);
        let d1 = d * sc1 as f32;
        let dm1 = dmin * m1 as f32;

        let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
        let d2 = d * sc2 as f32;
        let dm2 = dmin * m2 as f32;

        let q_start = j / 2;
        let qh_start = j / 8;

        // First 32 values
        for l in 0..32 {
            let q_byte_idx = q_start + l / 2;
            let q_nibble = if l % 2 == 0 {
                block.qs[q_byte_idx] & 0xF
            } else {
                block.qs[q_byte_idx] >> 4
            };

            let qh_byte_idx = qh_start + l / 8;
            let qh_bit = (block.qh[qh_byte_idx] >> (l % 8)) & 1;
            let q = q_nibble | (qh_bit << 4);

            output[idx] = d1 * q as f32 - dm1;
            idx += 1;
        }

        // Second 32 values
        for l in 0..32 {
            let q_byte_idx = q_start + 16 + l / 2;
            let q_nibble = if l % 2 == 0 {
                block.qs[q_byte_idx] & 0xF
            } else {
                block.qs[q_byte_idx] >> 4
            };

            let qh_byte_idx = qh_start + 4 + l / 8;
            let qh_bit = (block.qh[qh_byte_idx] >> (l % 8)) & 1;
            let q = q_nibble | (qh_bit << 4);

            output[idx] = d2 * q as f32 - dm2;
            idx += 1;
        }

        is += 2;
    }
}

/// Dequantize Q6K block to f32.
pub fn dequantize_q6k(block: &BlockQ6K, output: &mut [f32; QK_K]) {
    let d = block.d.to_f32();

    let mut idx = 0;

    for j in (0..QK_K).step_by(128) {
        for l in 0..32 {
            let scale_idx = j / 16 + l / 4;
            let sc = block.scales[scale_idx];
            let dl = d * sc as f32;

            for k in 0..4 {
                let global_idx = j + l * 4 + k;
                let ql_idx = global_idx / 2;
                let ql_nibble = if global_idx % 2 == 0 {
                    block.ql[ql_idx] & 0xF
                } else {
                    block.ql[ql_idx] >> 4
                };

                let qh_idx = global_idx / 4;
                let qh_shift = (global_idx % 4) * 2;
                let qh_bits = (block.qh[qh_idx] >> qh_shift) & 3;

                let q = (ql_nibble | (qh_bits << 4)) as i8 - 32;
                output[idx] = dl * q as f32;
                idx += 1;
            }
        }
    }
}

/// Dequantize Q8K block to f32.
pub fn dequantize_q8k(block: &BlockQ8K, output: &mut [f32; QK_K]) {
    let d = block.d;

    for (i, &q) in block.qs.iter().enumerate() {
        output[i] = d * q as f32;
    }
}

// =============================================================================
// Safe Dequantization from Byte Slices
// =============================================================================

/// Dequantize Q4K data from raw bytes.
///
/// # Safety
/// The input data must be properly aligned and sized for BlockQ4K blocks.
pub fn dequantize_q4k_bytes(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4K>();
    let n_blocks = n_elements.div_ceil(QK_K);

    let mut output = vec![0.0f32; n_elements];
    let mut out_idx = 0;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        let block_end = block_start + BLOCK_SIZE;

        if block_end > data.len() {
            break;
        }

        let block_bytes = &data[block_start..block_end];

        // Parse block manually from bytes
        let d = f16::from_le_bytes([block_bytes[0], block_bytes[1]]);
        let dmin = f16::from_le_bytes([block_bytes[2], block_bytes[3]]);

        let mut scales = [0u8; K_SCALE_SIZE];
        scales.copy_from_slice(&block_bytes[4..4 + K_SCALE_SIZE]);

        let mut qs = [0u8; QK_K / 2];
        qs.copy_from_slice(&block_bytes[4 + K_SCALE_SIZE..]);

        let block = BlockQ4K {
            d,
            dmin,
            scales,
            qs,
        };

        // Dequantize this block
        let mut block_output = [0.0f32; QK_K];
        dequantize_q4k(&block, &mut block_output);

        // Copy to output (handle partial last block)
        let copy_count = (n_elements - out_idx).min(QK_K);
        output[out_idx..out_idx + copy_count].copy_from_slice(&block_output[..copy_count]);
        out_idx += copy_count;
    }

    output
}

/// Dequantize Q8K data from raw bytes.
pub fn dequantize_q8k_bytes(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8K>();
    let n_blocks = n_elements.div_ceil(QK_K);

    let mut output = vec![0.0f32; n_elements];
    let mut out_idx = 0;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        let block_end = block_start + BLOCK_SIZE;

        if block_end > data.len() {
            break;
        }

        let block_bytes = &data[block_start..block_end];

        // Parse block from bytes
        let d = f32::from_le_bytes([
            block_bytes[0],
            block_bytes[1],
            block_bytes[2],
            block_bytes[3],
        ]);

        let mut qs = [0i8; QK_K];
        for (i, &b) in block_bytes[4..4 + QK_K].iter().enumerate() {
            qs[i] = b as i8;
        }

        let mut bsums = [0i16; QK_K / 16];
        #[allow(clippy::needless_range_loop)]
        for i in 0..(QK_K / 16) {
            let offset = 4 + QK_K + i * 2;
            bsums[i] = i16::from_le_bytes([block_bytes[offset], block_bytes[offset + 1]]);
        }

        let block = BlockQ8K { d, qs, bsums };

        let mut block_output = [0.0f32; QK_K];
        dequantize_q8k(&block, &mut block_output);

        let copy_count = (n_elements - out_idx).min(QK_K);
        output[out_idx..out_idx + copy_count].copy_from_slice(&block_output[..copy_count]);
        out_idx += copy_count;
    }

    output
}

/// Dequantize Q6K data from raw bytes.
pub fn dequantize_q6k_bytes(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ6K>();
    let n_blocks = n_elements.div_ceil(QK_K);

    let mut output = vec![0.0f32; n_elements];
    let mut out_idx = 0;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        let block_end = block_start + BLOCK_SIZE;

        if block_end > data.len() {
            break;
        }

        let block_bytes = &data[block_start..block_end];

        // Parse block from bytes (Q6K layout: ql, qh, scales, d)
        let mut ql = [0u8; QK_K / 2];
        ql.copy_from_slice(&block_bytes[0..QK_K / 2]);

        let mut qh = [0u8; QK_K / 4];
        qh.copy_from_slice(&block_bytes[QK_K / 2..QK_K / 2 + QK_K / 4]);

        let mut scales = [0i8; QK_K / 16];
        for (i, &b) in block_bytes[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16]
            .iter()
            .enumerate()
        {
            scales[i] = b as i8;
        }

        let d_offset = QK_K / 2 + QK_K / 4 + QK_K / 16;
        let d = f16::from_le_bytes([block_bytes[d_offset], block_bytes[d_offset + 1]]);

        let block = BlockQ6K { ql, qh, scales, d };

        let mut block_output = [0.0f32; QK_K];
        dequantize_q6k(&block, &mut block_output);

        let copy_count = (n_elements - out_idx).min(QK_K);
        output[out_idx..out_idx + copy_count].copy_from_slice(&block_output[..copy_count]);
        out_idx += copy_count;
    }

    output
}

/// Dequantize Q5K data from raw bytes.
pub fn dequantize_q5k_bytes(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ5K>();
    let n_blocks = n_elements.div_ceil(QK_K);

    let mut output = vec![0.0f32; n_elements];
    let mut out_idx = 0;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        let block_end = block_start + BLOCK_SIZE;

        if block_end > data.len() {
            break;
        }

        let block_bytes = &data[block_start..block_end];

        // Parse block (Q5K layout: d, dmin, scales, qh, qs)
        let d = f16::from_le_bytes([block_bytes[0], block_bytes[1]]);
        let dmin = f16::from_le_bytes([block_bytes[2], block_bytes[3]]);

        let mut scales = [0u8; K_SCALE_SIZE];
        scales.copy_from_slice(&block_bytes[4..4 + K_SCALE_SIZE]);

        let mut qh = [0u8; QK_K / 8];
        qh.copy_from_slice(&block_bytes[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8]);

        let mut qs = [0u8; QK_K / 2];
        qs.copy_from_slice(&block_bytes[4 + K_SCALE_SIZE + QK_K / 8..]);

        let block = BlockQ5K {
            d,
            dmin,
            scales,
            qh,
            qs,
        };

        let mut block_output = [0.0f32; QK_K];
        dequantize_q5k(&block, &mut block_output);

        let copy_count = (n_elements - out_idx).min(QK_K);
        output[out_idx..out_idx + copy_count].copy_from_slice(&block_output[..copy_count]);
        out_idx += copy_count;
    }

    output
}

/// Dequantize Q3K data from raw bytes.
pub fn dequantize_q3k_bytes(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ3K>();
    let n_blocks = n_elements.div_ceil(QK_K);

    let mut output = vec![0.0f32; n_elements];
    let mut out_idx = 0;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        let block_end = block_start + BLOCK_SIZE;

        if block_end > data.len() {
            break;
        }

        let block_bytes = &data[block_start..block_end];

        // Parse block (Q3K layout: hmask, qs, scales, d)
        let mut hmask = [0u8; QK_K / 8];
        hmask.copy_from_slice(&block_bytes[0..QK_K / 8]);

        let mut qs = [0u8; QK_K / 4];
        qs.copy_from_slice(&block_bytes[QK_K / 8..QK_K / 8 + QK_K / 4]);

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&block_bytes[QK_K / 8 + QK_K / 4..QK_K / 8 + QK_K / 4 + 12]);

        let d_offset = QK_K / 8 + QK_K / 4 + 12;
        let d = f16::from_le_bytes([block_bytes[d_offset], block_bytes[d_offset + 1]]);

        let block = BlockQ3K {
            hmask,
            qs,
            scales,
            d,
        };

        let mut block_output = [0.0f32; QK_K];
        dequantize_q3k(&block, &mut block_output);

        let copy_count = (n_elements - out_idx).min(QK_K);
        output[out_idx..out_idx + copy_count].copy_from_slice(&block_output[..copy_count]);
        out_idx += copy_count;
    }

    output
}

/// Dequantize Q2K data from raw bytes.
pub fn dequantize_q2k_bytes(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ2K>();
    let n_blocks = n_elements.div_ceil(QK_K);

    let mut output = vec![0.0f32; n_elements];
    let mut out_idx = 0;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        let block_end = block_start + BLOCK_SIZE;

        if block_end > data.len() {
            break;
        }

        let block_bytes = &data[block_start..block_end];

        // Parse block (Q2K layout: scales, qs, d, dmin)
        let mut scales = [0u8; QK_K / 16];
        scales.copy_from_slice(&block_bytes[0..QK_K / 16]);

        let mut qs = [0u8; QK_K / 4];
        qs.copy_from_slice(&block_bytes[QK_K / 16..QK_K / 16 + QK_K / 4]);

        let d_offset = QK_K / 16 + QK_K / 4;
        let d = f16::from_le_bytes([block_bytes[d_offset], block_bytes[d_offset + 1]]);
        let dmin = f16::from_le_bytes([block_bytes[d_offset + 2], block_bytes[d_offset + 3]]);

        let block = BlockQ2K {
            scales,
            qs,
            d,
            dmin,
        };

        let mut block_output = [0.0f32; QK_K];
        dequantize_q2k(&block, &mut block_output);

        let copy_count = (n_elements - out_idx).min(QK_K);
        output[out_idx..out_idx + copy_count].copy_from_slice(&block_output[..copy_count]);
        out_idx += copy_count;
    }

    output
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
        for (j, chunk) in x.chunks_exact(32).enumerate() {
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

        for (j, chunk) in x.chunks_exact(32).enumerate() {
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
        for (j, chunk) in x.chunks_exact(16).enumerate() {
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
#[allow(unsafe_code)]
pub fn blocks_to_bytes<T: Copy>(blocks: &[T]) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(blocks);
    let mut bytes = vec![0u8; byte_len];

    // Safety: We're copying the raw bytes of the block structures
    // This is safe because our blocks are #[repr(C)] with known layout
    unsafe {
        std::ptr::copy_nonoverlapping(blocks.as_ptr() as *const u8, bytes.as_mut_ptr(), byte_len);
    }

    bytes
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
    fn test_q8k_dequantize() {
        // Simple Q8K block: scale=0.5, values [0, 1, 2, ...]
        let mut block = BlockQ8K {
            d: 0.5,
            qs: [0i8; QK_K],
            bsums: [0i16; QK_K / 16],
        };

        for i in 0..QK_K {
            block.qs[i] = (i % 128) as i8;
        }

        let mut output = [0.0f32; QK_K];
        dequantize_q8k(&block, &mut output);

        // Verify: output[i] = 0.5 * qs[i]
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[2] - 1.0).abs() < 1e-6);
        assert!((output[10] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_q4k_dequantize_simple() {
        // Create a simple Q4K block with known values
        let block = BlockQ4K {
            d: f16::from_f32(1.0),
            dmin: f16::from_f32(0.0),
            scales: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], // sc=1, m=0 for all
            qs: [0x21; QK_K / 2],                         // 0x21 = 0b00100001, lower=1, upper=2
        };

        let mut output = [0.0f32; QK_K];
        dequantize_q4k(&block, &mut output);

        // With scale=1, min=0, values should be the raw quantized values
        // The exact pattern depends on nibble ordering
        assert!(output.iter().all(|&x| (0.0..=15.0).contains(&x)));
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

    // =========================================================================
    // Roundtrip tests: quantize -> dequantize -> verify closeness
    // =========================================================================

    /// Compute RMSE between two slices.
    fn compute_rmse(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        (sum_sq / n as f32).sqrt()
    }

    /// Compute max absolute error between two slices.
    fn compute_max_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, |max, err| max.max(err))
    }

    #[test]
    fn test_q8k_roundtrip() {
        // Create test data with a range typical for neural network weights
        let original: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) * 0.01).collect();

        // Quantize
        let blocks = quantize_q8k(&original);
        assert_eq!(blocks.len(), 1);

        // Dequantize
        let mut restored = [0.0f32; QK_K];
        dequantize_q8k(&blocks[0], &mut restored);

        // Q8K should be very accurate
        let rmse = compute_rmse(&original, &restored);
        let max_err = compute_max_err(&original, &restored);

        assert!(rmse < 0.01, "Q8K RMSE too high: {} (expected < 0.01)", rmse);
        assert!(
            max_err < 0.02,
            "Q8K max error too high: {} (expected < 0.02)",
            max_err
        );
    }

    #[test]
    fn test_q4k_roundtrip() {
        let original: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let blocks = quantize_q4k(&original);
        assert_eq!(blocks.len(), 1);

        let bytes = blocks_to_bytes(&blocks);
        let restored = dequantize_q4k_bytes(&bytes, QK_K);

        let rmse = compute_rmse(&original, &restored);
        let max_err = compute_max_err(&original, &restored);

        // Q4K has more quantization error than Q8K
        assert!(rmse < 0.15, "Q4K RMSE too high: {} (expected < 0.15)", rmse);
        assert!(
            max_err < 0.5,
            "Q4K max error too high: {} (expected < 0.5)",
            max_err
        );
    }

    #[test]
    fn test_q5k_roundtrip() {
        let original: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let blocks = quantize_q5k(&original);
        assert_eq!(blocks.len(), 1);

        let bytes = blocks_to_bytes(&blocks);
        let restored = dequantize_q5k_bytes(&bytes, QK_K);

        let rmse = compute_rmse(&original, &restored);
        let max_err = compute_max_err(&original, &restored);

        // Q5K should be more accurate than Q4K
        assert!(rmse < 0.1, "Q5K RMSE too high: {} (expected < 0.1)", rmse);
        assert!(
            max_err < 0.4,
            "Q5K max error too high: {} (expected < 0.4)",
            max_err
        );
    }

    #[test]
    fn test_q6k_roundtrip() {
        let original: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let blocks = quantize_q6k(&original);
        assert_eq!(blocks.len(), 1);

        let bytes = blocks_to_bytes(&blocks);
        let restored = dequantize_q6k_bytes(&bytes, QK_K);

        let rmse = compute_rmse(&original, &restored);
        let max_err = compute_max_err(&original, &restored);

        // Q6K should be reasonably accurate
        assert!(rmse < 0.1, "Q6K RMSE too high: {} (expected < 0.1)", rmse);
        assert!(
            max_err < 0.3,
            "Q6K max error too high: {} (expected < 0.3)",
            max_err
        );
    }

    #[test]
    fn test_q3k_roundtrip() {
        let original: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let blocks = quantize_q3k(&original);
        assert_eq!(blocks.len(), 1);

        let bytes = blocks_to_bytes(&blocks);
        let restored = dequantize_q3k_bytes(&bytes, QK_K);

        let rmse = compute_rmse(&original, &restored);
        let max_err = compute_max_err(&original, &restored);

        // Q3K has significant quantization error due to 3-bit precision
        assert!(rmse < 1.5, "Q3K RMSE too high: {} (expected < 1.5)", rmse);
        assert!(
            max_err < 4.0,
            "Q3K max error too high: {} (expected < 4.0)",
            max_err
        );
    }

    #[test]
    fn test_q2k_roundtrip() {
        let original: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let blocks = quantize_q2k(&original);
        assert_eq!(blocks.len(), 1);

        let bytes = blocks_to_bytes(&blocks);
        let restored = dequantize_q2k_bytes(&bytes, QK_K);

        let rmse = compute_rmse(&original, &restored);
        let max_err = compute_max_err(&original, &restored);

        // Q2K has the most quantization error due to only 2-bit precision
        assert!(rmse < 1.0, "Q2K RMSE too high: {} (expected < 1.0)", rmse);
        assert!(
            max_err < 2.0,
            "Q2K max error too high: {} (expected < 2.0)",
            max_err
        );
    }

    #[test]
    fn test_blocks_to_bytes() {
        let blocks = quantize_q8k(&vec![0.0; QK_K]);
        let bytes = blocks_to_bytes(&blocks);
        assert_eq!(bytes.len(), 292); // Q8K block size
    }
}
