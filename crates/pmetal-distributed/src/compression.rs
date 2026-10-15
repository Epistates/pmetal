//! Gradient compression for bandwidth optimization.
//!
//! Provides several compression strategies:
//! - TopK: Keep only the k largest gradients
//! - Random sparsification: Randomly sample gradients
//! - Quantization: Reduce precision (FP16, BF16, INT8, 1-bit)
//! - Error feedback: carry what a step failed to transmit into the next one
//!
//! # Error feedback
//!
//! With error feedback on, the compressor keeps a residual `e`. Each step
//! compresses `g + e` and keeps `e' = (g + e) - decompress(compress(g + e))`,
//! whatever the method, so nothing is lost, only delayed: the sum of what has
//! been transmitted plus the residual always equals the sum of the gradients
//! fed in. Sparsifiers and 1-bit quantizers need it to converge at all.
//!
//! Random sparsification is unbiased on its own (kept values are scaled by
//! `1/p`); with error feedback it keeps values as they are, because the
//! residual already restores what was skipped and rescaling too would count it
//! twice.
//!
//! # Reducing compressed gradients
//!
//! Compressed gradients cannot be summed on the wire by a ring all-reduce
//! (sparse indices and quantized scales do not add). [`all_reduce_compressed`]
//! gathers every rank's compressed gradient, decompresses them in rank order
//! and sums, so every rank ends with the same bits. It moves `N - 1`
//! compressed copies per rank against a ring's `2 (N - 1) / N` full copies,
//! which pays off when the compression ratio exceeds about `N / 2`.
//!
//! References:
//! - Deep Gradient Compression (Lin et al., 2018)
//! - 1-Bit SGD (Seide et al., 2014)

use crate::{DistributedBackend, ReduceOp};
use anyhow::Result;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Compression strategy.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// No compression.
    #[default]
    None,
    /// Keep only the top `ratio` (a fraction in `(0, 1]`) of gradients by
    /// magnitude.
    TopK { ratio: f32 },
    /// Random sparsification keeping each gradient with the given
    /// probability, in `(0, 1]`.
    Random { probability: f32 },
    /// Quantize to lower precision.
    Quantize(QuantizationType),
}

/// Quantization type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuantizationType {
    /// FP16 (IEEE half precision), saturating at its largest finite value.
    FP16,
    /// BF16 (Brain float).
    BF16,
    /// 8-bit integer with one scale for the whole tensor.
    INT8,
    /// 1-bit sign only (with scale).
    OneBit,
}

/// Compressed gradient representation.
#[derive(Debug, Clone)]
pub struct CompressedGradient {
    /// Original size.
    pub original_size: usize,
    /// Compression strategy used.
    pub strategy: CompressionStrategy,
    /// Compressed data.
    pub data: CompressedData,
}

/// Compressed data variants.
#[derive(Debug, Clone)]
pub enum CompressedData {
    /// Full precision (no compression).
    Full(Vec<f32>),
    /// Sparse representation (indices + values).
    Sparse { indices: Vec<u32>, values: Vec<f32> },
    /// FP16 quantized.
    FP16(Vec<u16>),
    /// BF16 quantized.
    BF16(Vec<u16>),
    /// INT8 quantized with scale.
    INT8 { data: Vec<i8>, scale: f32 },
    /// 1-bit with scale.
    OneBit { signs: Vec<u8>, scale: f32 },
}

impl CompressedGradient {
    /// Get the compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.original_size * 4;
        let compressed_bytes = self.compressed_bytes();
        original_bytes as f32 / compressed_bytes.max(1) as f32
    }

    /// Get compressed size in bytes.
    pub fn compressed_bytes(&self) -> usize {
        match &self.data {
            CompressedData::Full(v) => v.len() * 4,
            CompressedData::Sparse { indices, values } => indices.len() * 4 + values.len() * 4,
            CompressedData::FP16(v) => v.len() * 2,
            CompressedData::BF16(v) => v.len() * 2,
            CompressedData::INT8 { data, .. } => data.len() + 4,
            CompressedData::OneBit { signs, .. } => signs.len() + 4,
        }
    }

    /// Add the decompressed values to `out`, element by element.
    ///
    /// Indices past `out` (a malformed sparse packet) are ignored, as are
    /// elements past `original_size`.
    pub fn add_to(&self, out: &mut [f32]) {
        let n = self.original_size.min(out.len());
        match &self.data {
            CompressedData::Full(v) => out[..n].iter_mut().zip(v).for_each(|(o, x)| *o += x),
            CompressedData::Sparse { indices, values } => {
                for (&idx, &val) in indices.iter().zip(values) {
                    if let Some(slot) = out[..n].get_mut(idx as usize) {
                        *slot += val;
                    }
                }
            }
            CompressedData::FP16(v) => out[..n]
                .iter_mut()
                .zip(v)
                .for_each(|(o, &x)| *o += f16::from_bits(x).to_f32()),
            CompressedData::BF16(v) => out[..n]
                .iter_mut()
                .zip(v)
                .for_each(|(o, &x)| *o += bf16::from_bits(x).to_f32()),
            CompressedData::INT8 { data, scale } => out[..n]
                .iter_mut()
                .zip(data)
                .for_each(|(o, &x)| *o += f32::from(x) * scale),
            CompressedData::OneBit { signs, scale } => {
                for (i, o) in out[..n].iter_mut().enumerate() {
                    let positive = signs.get(i / 8).is_some_and(|b| (b >> (i % 8)) & 1 == 1);
                    *o += if positive { *scale } else { -*scale };
                }
            }
        }
    }

    /// The decompressed gradient.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.original_size];
        self.add_to(&mut out);
        out
    }
}

/// SplitMix64: a small, well-distributed generator, so masks are reproducible
/// from a seed and differ between ranks given different seeds.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Gradient compressor with error feedback.
pub struct GradientCompressor {
    /// Compression strategy.
    strategy: CompressionStrategy,
    /// Error feedback buffer (accumulated residuals).
    error_feedback: Option<Vec<f32>>,
    /// Whether to use error feedback.
    use_error_feedback: bool,
    /// Source of random masks.
    rng: SplitMix64,
}

impl GradientCompressor {
    /// Create a new compressor.
    ///
    /// A `ratio` or `probability` outside `(0, 1]` is clamped into it (a
    /// non-finite one means "keep everything"), so a bad setting can only
    /// cost bandwidth, never drop the whole gradient.
    pub fn new(strategy: CompressionStrategy, use_error_feedback: bool) -> Self {
        Self::with_seed(strategy, use_error_feedback, 0)
    }

    /// [`new`](Self::new) with a seed for the random masks. Give every rank
    /// its own: ranks that draw the same mask send the same coordinates and
    /// the sum of their sparse gradients is far noisier.
    pub fn with_seed(strategy: CompressionStrategy, use_error_feedback: bool, seed: u64) -> Self {
        let fraction = |x: f32| {
            if x.is_finite() {
                x.clamp(f32::MIN_POSITIVE, 1.0)
            } else {
                1.0
            }
        };
        let strategy = match strategy {
            CompressionStrategy::TopK { ratio } => CompressionStrategy::TopK {
                ratio: fraction(ratio),
            },
            CompressionStrategy::Random { probability } => CompressionStrategy::Random {
                probability: fraction(probability),
            },
            other => other,
        };
        Self {
            strategy,
            error_feedback: None,
            use_error_feedback,
            rng: SplitMix64(seed ^ 0x5DEE_CE66_D1CE_4E5B),
        }
    }

    /// The strategy in use.
    pub fn strategy(&self) -> &CompressionStrategy {
        &self.strategy
    }

    /// What has been left untransmitted so far, if error feedback is on and a
    /// step has run.
    pub fn residual(&self) -> Option<&[f32]> {
        self.error_feedback.as_deref()
    }

    /// Compress gradients.
    pub fn compress(&mut self, gradients: &[f32]) -> CompressedGradient {
        let original_size = gradients.len();
        let feedback =
            self.use_error_feedback && !matches!(self.strategy, CompressionStrategy::None);

        // Fold in what earlier steps left behind. A residual for a different
        // tensor length belongs to another buffer; mixing it in would add one
        // parameter's leftovers to another's gradient.
        let mut working = gradients.to_vec();
        if feedback && let Some(error) = self.error_feedback.take() {
            if error.len() == working.len() {
                working.iter_mut().zip(&error).for_each(|(w, e)| *w += e);
            } else {
                warn!(
                    "gradient length changed from {} to {}: discarding the error-feedback residual",
                    error.len(),
                    working.len()
                );
            }
        }

        let data = match &self.strategy {
            CompressionStrategy::None => CompressedData::Full(working.clone()),
            CompressionStrategy::TopK { ratio } => Self::compress_topk(&working, *ratio),
            CompressionStrategy::Random { probability } => {
                self.compress_random(&working, *probability, !feedback)
            }
            CompressionStrategy::Quantize(qtype) => Self::quantize(&working, *qtype),
        };
        let result = CompressedGradient {
            original_size,
            strategy: self.strategy.clone(),
            data,
        };

        if feedback {
            // What the receiver will reconstruct, subtracted from what we
            // meant to send: the same rule for every method.
            let mut residual = working;
            let sent = result.to_dense();
            residual.iter_mut().zip(&sent).for_each(|(r, s)| *r -= s);
            self.error_feedback = Some(residual);
        }

        debug!(
            "Compressed {} floats, ratio={:.2}x",
            original_size,
            result.compression_ratio()
        );

        result
    }

    /// Decompress gradients.
    pub fn decompress(&self, compressed: &CompressedGradient) -> Vec<f32> {
        compressed.to_dense()
    }

    /// Top-K sparsification: the `k` largest magnitudes, ties going to the
    /// lower index, in O(n) expected time.
    fn compress_topk(gradients: &[f32], ratio: f32) -> CompressedData {
        let n = gradients.len();
        let k = ((n as f64 * f64::from(ratio)) as usize).max(1).min(n);
        if k == n {
            return CompressedData::Sparse {
                indices: (0..n as u32).collect(),
                values: gradients.to_vec(),
            };
        }
        let mut order: Vec<u32> = (0..n as u32).collect();
        let by_magnitude = |a: &u32, b: &u32| {
            let (x, y) = (gradients[*a as usize].abs(), gradients[*b as usize].abs());
            // Larger first; NaN counts as largest so a poisoned gradient is
            // sent, not silently dropped; equal magnitudes go lower index first.
            y.total_cmp(&x).then(a.cmp(b))
        };
        order.select_nth_unstable_by(k - 1, by_magnitude);
        order.truncate(k);
        order.sort_unstable();
        let values = order.iter().map(|&i| gradients[i as usize]).collect();
        CompressedData::Sparse {
            indices: order,
            values,
        }
    }

    /// Random sparsification. `rescale` makes the kept values `1/p` times
    /// larger so the expectation is the input; with error feedback the
    /// residual already restores what was skipped, so it is off.
    fn compress_random(
        &mut self,
        gradients: &[f32],
        probability: f32,
        rescale: bool,
    ) -> CompressedData {
        let p = f64::from(probability);
        let factor = if rescale { (1.0 / p) as f32 } else { 1.0 };
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &val) in gradients.iter().enumerate() {
            if self.rng.next_unit() < p {
                indices.push(i as u32);
                values.push(val * factor);
            }
        }
        CompressedData::Sparse { indices, values }
    }

    /// Quantize gradients.
    fn quantize(gradients: &[f32], qtype: QuantizationType) -> CompressedData {
        match qtype {
            QuantizationType::FP16 => {
                // Saturate rather than overflow to infinity.
                let max = f16::MAX.to_f32();
                CompressedData::FP16(
                    gradients
                        .iter()
                        .map(|&x| f16::from_f32(x.clamp(-max, max)).to_bits())
                        .collect(),
                )
            }
            QuantizationType::BF16 => CompressedData::BF16(
                gradients
                    .iter()
                    .map(|&x| bf16::from_f32(x).to_bits())
                    .collect(),
            ),
            QuantizationType::INT8 => {
                // `fold(max)` skips NaN; a non-finite input must instead
                // surface, so the scale carries it to every element.
                let max_abs = gradients.iter().fold(0.0f32, |a, x| a.max(x.abs()));
                let poisoned = gradients.iter().any(|x| !x.is_finite());
                let scale = if poisoned {
                    f32::NAN
                } else if max_abs == 0.0 {
                    1.0
                } else {
                    max_abs / 127.0
                };
                let data = gradients
                    .iter()
                    .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
                    .collect();
                CompressedData::INT8 { data, scale }
            }
            QuantizationType::OneBit => {
                let mean_abs =
                    gradients.iter().map(|x| x.abs()).sum::<f32>() / gradients.len().max(1) as f32;
                let mut signs = vec![0u8; gradients.len().div_ceil(8)];
                for (i, &val) in gradients.iter().enumerate() {
                    if val > 0.0 {
                        signs[i / 8] |= 1 << (i % 8);
                    }
                }
                CompressedData::OneBit {
                    signs,
                    scale: mean_abs,
                }
            }
        }
    }

    /// Reset error feedback.
    pub fn reset_error_feedback(&mut self) {
        self.error_feedback = None;
    }
}

/// Reduce `data` across the ranks of `backend` using `compressor`, in place.
///
/// Every rank compresses its gradient, all ranks exchange the compressed
/// packets, and each decompresses every packet in rank order and sums them
/// (dividing by the world size for [`ReduceOp::Mean`]). The order is the same
/// on every rank, so the result is identical on every rank down to the bit.
/// A rank's own packet goes through the same decompression as the others', so
/// the error feedback its compressor keeps is exactly what the cluster is
/// missing from it.
///
/// With no compression configured, or a single rank, this is a plain
/// all-reduce and the compressor's state is left alone. Returns the bytes this
/// rank sent.
pub async fn all_reduce_compressed(
    backend: &dyn DistributedBackend,
    compressor: &mut GradientCompressor,
    data: &mut [f32],
    op: ReduceOp,
) -> Result<usize> {
    let world_size = backend.world_size();
    if world_size <= 1 || matches!(compressor.strategy, CompressionStrategy::None) {
        backend.all_reduce_f32(data, op).await?;
        return Ok(data.len() * 4);
    }

    let packet = serialize_compressed(&compressor.compress(data));
    let sent = packet.len();
    let packets = backend.all_gather_bytes(&packet).await?;
    if packets.len() != world_size {
        anyhow::bail!(
            "all_gather_bytes returned {} packets for {world_size} ranks",
            packets.len()
        );
    }

    data.fill(0.0);
    for (rank, bytes) in packets.iter().enumerate() {
        let compressed = deserialize_compressed(bytes).ok_or_else(|| {
            crate::error::DistributedError::Protocol(format!(
                "rank {rank} sent a malformed compressed gradient"
            ))
        })?;
        if compressed.original_size != data.len() {
            anyhow::bail!(
                "rank {rank} compressed {} gradients but this rank has {}",
                compressed.original_size,
                data.len()
            );
        }
        compressed.add_to(data);
    }
    if op == ReduceOp::Mean {
        let divisor = world_size as f32;
        data.iter_mut().for_each(|x| *x /= divisor);
    }
    Ok(sent)
}

/// First byte of a serialized gradient.
const WIRE_MAGIC: u8 = 0xC7;
/// Serialization format version.
const WIRE_VERSION: u8 = 1;
/// Magic, version, strategy tag and the element count.
const WIRE_HEADER: usize = 1 + 1 + 1 + 8;

/// Serialize compressed gradient to bytes.
pub fn serialize_compressed(compressed: &CompressedGradient) -> Vec<u8> {
    let mut result = Vec::with_capacity(WIRE_HEADER + compressed.compressed_bytes() + 8);
    result.push(WIRE_MAGIC);
    result.push(WIRE_VERSION);
    let tag_at = result.len();
    result.push(0);
    result.extend_from_slice(&(compressed.original_size as u64).to_le_bytes());

    let tag = match &compressed.data {
        CompressedData::Full(v) => {
            for f in v {
                result.extend_from_slice(&f.to_le_bytes());
            }
            0u8
        }
        CompressedData::Sparse { indices, values } => {
            result.extend_from_slice(&(indices.len() as u64).to_le_bytes());
            for &idx in indices {
                result.extend_from_slice(&idx.to_le_bytes());
            }
            for &val in values {
                result.extend_from_slice(&val.to_le_bytes());
            }
            1
        }
        CompressedData::FP16(v) => {
            for &x in v {
                result.extend_from_slice(&x.to_le_bytes());
            }
            2
        }
        CompressedData::BF16(v) => {
            for &x in v {
                result.extend_from_slice(&x.to_le_bytes());
            }
            3
        }
        CompressedData::INT8 { data, scale } => {
            result.extend_from_slice(&scale.to_le_bytes());
            result.extend(data.iter().map(|&x| x as u8));
            4
        }
        CompressedData::OneBit { signs, scale } => {
            result.extend_from_slice(&scale.to_le_bytes());
            result.extend_from_slice(signs);
            5
        }
    };
    result[tag_at] = tag;
    result
}

/// Deserialize compressed gradient from bytes.
///
/// Returns `None` for anything that is not exactly a well-formed packet; no
/// allocation is sized from an unchecked length field.
pub fn deserialize_compressed(bytes: &[u8]) -> Option<CompressedGradient> {
    if bytes.len() < WIRE_HEADER || bytes[0] != WIRE_MAGIC || bytes[1] != WIRE_VERSION {
        return None;
    }
    let tag = bytes[2];
    let original_size = usize::try_from(u64::from_le_bytes(bytes[3..11].try_into().ok()?)).ok()?;
    let body = &bytes[WIRE_HEADER..];

    let exact = |per_element: usize, extra: usize| {
        original_size
            .checked_mul(per_element)?
            .checked_add(extra)
            .filter(|&expected| expected == body.len())
    };
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    let u16s = |b: &[u8]| -> Vec<u16> {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect()
    };

    let data = match tag {
        0 => {
            exact(4, 0)?;
            CompressedData::Full(f32s(body))
        }
        1 => {
            let count =
                usize::try_from(u64::from_le_bytes(body.get(..8)?.try_into().ok()?)).ok()?;
            let table = count.checked_mul(8)?;
            if body.len() != 8usize.checked_add(table)? || count > original_size {
                return None;
            }
            let indices: Vec<u32> = body[8..8 + count * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            if indices.iter().any(|&idx| idx as usize >= original_size) {
                return None;
            }
            CompressedData::Sparse {
                indices,
                values: f32s(&body[8 + count * 4..]),
            }
        }
        2 => {
            exact(2, 0)?;
            CompressedData::FP16(u16s(body))
        }
        3 => {
            exact(2, 0)?;
            CompressedData::BF16(u16s(body))
        }
        4 => {
            exact(1, 4)?;
            CompressedData::INT8 {
                scale: f32::from_le_bytes(body[..4].try_into().ok()?),
                data: body[4..].iter().map(|&x| x as i8).collect(),
            }
        }
        5 => {
            let sign_bytes = original_size.div_ceil(8);
            if body.len() != 4usize.checked_add(sign_bytes)? {
                return None;
            }
            CompressedData::OneBit {
                scale: f32::from_le_bytes(body[..4].try_into().ok()?),
                signs: body[4..].to_vec(),
            }
        }
        _ => return None,
    };

    Some(CompressedGradient {
        original_size,
        strategy: CompressionStrategy::None, // Not tracked in serialization
        data,
    })
}

#[cfg(test)]
#[path = "compression_tests.rs"]
mod tests;
