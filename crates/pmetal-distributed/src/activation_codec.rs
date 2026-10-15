//! Activation compression for pipeline inference.
//!
//! Reduces bandwidth between pipeline stages by sending activations in half
//! precision. The sender hands over a tensor in some dtype; [`encode`] returns
//! the bytes to put on the wire together with the dtype *they* are in, so the
//! tag on the wire message always describes its payload, and [`decode`]
//! converts a received payload to whatever dtype the next stage computes in.

use crate::activation_transport::DtypeTag;
use crate::error::{DistributedError, DistributedResult};
use half::{bf16, f16};

/// Compression codec for activations transferred between pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivationCodec {
    /// No compression — transfer as-is.
    None,
    /// Send f32 activations as fp16 (half the bytes). Values beyond fp16's
    /// range saturate rather than overflow to infinity. Tensors already in
    /// 16 bits are sent as they are: bf16 has more range than fp16, so
    /// converting it could only lose.
    #[default]
    Float16,
}

/// The largest finite fp16 value.
const F16_MAX: f32 = 65504.0;

/// Compress f32 activations to fp16 bytes.
///
/// Input: `&[f32]` of hidden states
/// Output: `Vec<u8>` of fp16 bytes (half the size)
pub fn compress_f32_to_f16(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &val in data {
        let h = f16::from_f32(val.clamp(-F16_MAX, F16_MAX));
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Decompress fp16 bytes back to f32.
///
/// Input: `&[u8]` of fp16 bytes (a trailing odd byte is ignored)
/// Output: `Vec<f32>` of f32 values
pub fn decompress_f16_to_f32(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
        .collect()
}

fn elements(data: &[u8], dtype: DtypeTag) -> DistributedResult<usize> {
    if !data.len().is_multiple_of(dtype.element_size()) {
        return Err(DistributedError::Protocol(format!(
            "{} bytes is not a whole number of {dtype:?} elements",
            data.len()
        )));
    }
    Ok(data.len() / dtype.element_size())
}

/// Read a payload of `dtype` as f32 values.
fn to_f32(data: &[u8], dtype: DtypeTag) -> DistributedResult<Vec<f32>> {
    elements(data, dtype)?;
    Ok(match dtype {
        DtypeTag::Float32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DtypeTag::Float16 => decompress_f16_to_f32(data),
        DtypeTag::BFloat16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
    })
}

/// Prepare `data`, a tensor in `dtype`, for the wire under `codec`.
///
/// Returns the payload and the dtype it is in, which is what the wire
/// message must be tagged with.
pub fn encode(
    data: &[u8],
    dtype: DtypeTag,
    codec: ActivationCodec,
) -> DistributedResult<(Vec<u8>, DtypeTag)> {
    elements(data, dtype)?;
    match (codec, dtype) {
        (ActivationCodec::Float16, DtypeTag::Float32) => Ok((
            compress_f32_to_f16(&to_f32(data, dtype)?),
            DtypeTag::Float16,
        )),
        _ => Ok((data.to_vec(), dtype)),
    }
}

/// Convert a received payload (in `from`) to `to`.
///
/// Widening (fp16 or bf16 to f32) is exact. Narrowing rounds, and saturates
/// at fp16's range when the target is fp16.
pub fn decode(data: &[u8], from: DtypeTag, to: DtypeTag) -> DistributedResult<Vec<u8>> {
    if from == to {
        elements(data, from)?;
        return Ok(data.to_vec());
    }
    let values = to_f32(data, from)?;
    Ok(match to {
        DtypeTag::Float32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DtypeTag::Float16 => compress_f32_to_f16(&values),
        DtypeTag::BFloat16 => values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn f16_roundtrip() {
        let original = vec![1.0f32, 2.0, 3.5, -0.5, 0.0, 100.0];
        let compressed = compress_f32_to_f16(&original);
        assert_eq!(compressed.len(), original.len() * 2);

        let decompressed = decompress_f16_to_f32(&compressed);
        assert_eq!(decompressed, original, "these values are exact in fp16");
    }

    #[test]
    fn values_past_fp16_range_saturate() {
        let out = decompress_f16_to_f32(&compress_f32_to_f16(&[1.0e9, -1.0e9, f32::NAN]));
        assert_eq!(&out[..2], &[F16_MAX, -F16_MAX]);
        assert!(out[2].is_nan(), "NaN must stay NaN");
    }

    #[test]
    fn the_float16_codec_halves_f32_and_says_so() {
        let (payload, dtype) = encode(
            &f32_bytes(&[0.25, -3.0]),
            DtypeTag::Float32,
            ActivationCodec::Float16,
        )
        .unwrap();
        assert_eq!((payload.len(), dtype), (4, DtypeTag::Float16));
        let back = decode(&payload, dtype, DtypeTag::Float32).unwrap();
        assert_eq!(back, f32_bytes(&[0.25, -3.0]));
    }

    #[test]
    fn other_dtypes_and_the_none_codec_pass_through_with_their_own_tag() {
        let bf = vec![1u8, 2, 3, 4];
        assert_eq!(
            encode(&bf, DtypeTag::BFloat16, ActivationCodec::Float16).unwrap(),
            (bf.clone(), DtypeTag::BFloat16)
        );
        let f32s = f32_bytes(&[1.0]);
        assert_eq!(
            encode(&f32s, DtypeTag::Float32, ActivationCodec::None).unwrap(),
            (f32s, DtypeTag::Float32)
        );
    }

    #[test]
    fn a_payload_that_is_not_whole_elements_is_an_error() {
        assert!(encode(&[0u8; 5], DtypeTag::Float32, ActivationCodec::None).is_err());
        assert!(decode(&[0u8; 3], DtypeTag::Float16, DtypeTag::Float32).is_err());
    }

    #[test]
    fn bf16_widens_exactly() {
        let bits: Vec<u8> = [1.5f32, -2.0, 0.15625]
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect();
        let wide = decode(&bits, DtypeTag::BFloat16, DtypeTag::Float32).unwrap();
        assert_eq!(wide, f32_bytes(&[1.5, -2.0, 0.15625]));
    }
}
