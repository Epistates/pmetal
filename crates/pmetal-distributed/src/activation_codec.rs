//! Activation compression for pipeline inference.
//!
//! Reduces bandwidth between pipeline stages by compressing activations.
//! Initial implementation: fp16 cast (same as dnet default wire dtype).

use half::f16;

/// Compression codec for activations transferred between pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivationCodec {
    /// No compression — transfer as-is.
    None,
    /// Cast to fp16 for transfer (2x compression for f32 activations).
    #[default]
    Float16,
    /// Column-wise sparsity: keep top-k% columns by L2 norm.
    SparseColumns {
        /// Fraction of columns to keep (0.0-1.0, default 0.1 = top 10%).
        keep_ratio: u32, // stored as ratio * 1000 to avoid float
    },
}

/// Compress f32 activations to fp16 bytes.
///
/// Input: `&[f32]` of hidden states
/// Output: `Vec<u8>` of fp16 bytes (half the size)
pub fn compress_f32_to_f16(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &val in data {
        let h = f16::from_f32(val);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Decompress fp16 bytes back to f32.
///
/// Input: `&[u8]` of fp16 bytes
/// Output: `Vec<f32>` of f32 values
pub fn decompress_f16_to_f32(data: &[u8]) -> Vec<f32> {
    assert!(
        data.len().is_multiple_of(2),
        "fp16 data must be even length"
    );
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.as_chunks::<2>().0 {
        out.push(f16::from_le_bytes(*chunk).to_f32());
    }
    out
}

/// Compress activations according to the codec.
///
/// Returns compressed bytes and a tag indicating the codec used.
pub fn compress_activation(data: &[u8], src_is_f32: bool, codec: ActivationCodec) -> Vec<u8> {
    match codec {
        ActivationCodec::None => data.to_vec(),
        ActivationCodec::Float16 => {
            if src_is_f32 {
                // Reinterpret bytes as f32 via zerocopy-safe conversion
                let f32_data: Vec<f32> = data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                compress_f32_to_f16(&f32_data)
            } else {
                data.to_vec()
            }
        }
        ActivationCodec::SparseColumns { .. } => {
            // Future: column-wise sparsity
            data.to_vec()
        }
    }
}

/// Decompress activations back to the original format.
pub fn decompress_activation(data: &[u8], codec: ActivationCodec, target_is_f32: bool) -> Vec<u8> {
    match codec {
        ActivationCodec::None => data.to_vec(),
        ActivationCodec::Float16 => {
            if target_is_f32 {
                let f32_vals = decompress_f16_to_f32(data);
                let mut bytes = Vec::with_capacity(f32_vals.len() * 4);
                for val in &f32_vals {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
                bytes
            } else {
                data.to_vec()
            }
        }
        ActivationCodec::SparseColumns { .. } => data.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        let original = vec![1.0f32, 2.0, 3.5, -0.5, 0.0, 100.0];
        let compressed = compress_f32_to_f16(&original);
        assert_eq!(compressed.len(), original.len() * 2);

        let decompressed = decompress_f16_to_f32(&compressed);
        assert_eq!(decompressed.len(), original.len());

        for (orig, decomp) in original.iter().zip(decompressed.iter()) {
            let diff = (orig - decomp).abs();
            // fp16 has ~3 decimal digits of precision
            assert!(diff < 0.1, "f16 roundtrip drift: {orig} -> {decomp}");
        }
    }
}
