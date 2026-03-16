//! Offline distillation with logit caching and compression.
//!
//! Offline distillation pre-computes teacher logits and stores them on disk,
//! allowing the student to train without running the teacher model during training.
//! This is memory-efficient and allows using larger teacher models.
//!
//! # Compression Methods
//!
//! - **TopK**: Only store the top-k logits per token (e.g., top-128 out of 32k vocab)
//! - **Int8**: Quantize full logits to 8-bit integers
//! - **Int4**: Quantize full logits to 4-bit integers
//!
//! TopK is typically the best choice, as most probability mass is in the top few tokens.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read as _, Write};
use std::path::{Path, PathBuf};

use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{CompressionMethod, DistillError, Result};

/// Cache for pre-computed teacher logits.
pub struct LogitCache {
    /// Path to the cache directory.
    cache_dir: PathBuf,
    /// Compression method used.
    compression: CompressionMethod,
    /// Number of top-k logits to keep.
    top_k: usize,
    /// Metadata about cached sequences.
    metadata: CacheMetadata,
}

/// Metadata for the logit cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Number of cached sequences.
    pub num_sequences: usize,
    /// Compression method.
    pub compression: String,
    /// Top-k value (if using TopK compression).
    pub top_k: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Model name/path used for generation.
    pub model: String,
}

impl Default for CacheMetadata {
    fn default() -> Self {
        Self {
            num_sequences: 0,
            compression: "none".to_string(),
            top_k: 128,
            vocab_size: 0,
            max_seq_len: 0,
            model: String::new(),
        }
    }
}

impl LogitCache {
    /// Create a new logit cache.
    pub fn new(
        cache_dir: impl AsRef<Path>,
        compression: CompressionMethod,
        top_k: usize,
    ) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)?;

        let metadata_path = cache_dir.join("metadata.json");
        let metadata = if metadata_path.exists() {
            let file = File::open(&metadata_path)?;
            serde_json::from_reader(file)?
        } else {
            CacheMetadata::default()
        };

        Ok(Self {
            cache_dir,
            compression,
            top_k,
            metadata,
        })
    }

    /// Load an existing cache.
    pub fn load(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        let metadata_path = cache_dir.join("metadata.json");

        if !metadata_path.exists() {
            return Err(DistillError::LogitCache(
                "Cache metadata not found".to_string(),
            ));
        }

        let file = File::open(&metadata_path)?;
        let metadata: CacheMetadata = serde_json::from_reader(file)?;

        let compression = match metadata.compression.as_str() {
            "none" => CompressionMethod::None,
            "top_k" => CompressionMethod::TopK,
            "int8" => CompressionMethod::Int8,
            "int4" => CompressionMethod::Int4,
            _ => CompressionMethod::None,
        };

        Ok(Self {
            cache_dir,
            compression,
            top_k: metadata.top_k,
            metadata,
        })
    }

    /// Cache logits for a sequence.
    pub fn cache_sequence(&mut self, sequence_id: usize, logits: &Array) -> Result<()> {
        let compressor = LogitCompressor::new(self.compression.clone(), self.top_k);
        let compressed = compressor.compress(logits)?;

        let path = self.cache_dir.join(format!("seq_{:06}.bin", sequence_id));
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Write compressed data
        writer.write_all(&bitcode::serialize(&compressed)?)?;
        writer.flush()?;

        self.metadata.num_sequences = self.metadata.num_sequences.max(sequence_id + 1);
        if self.metadata.vocab_size == 0 {
            self.metadata.vocab_size = logits.dim(-1) as usize;
        }

        Ok(())
    }

    /// Load cached logits for a sequence.
    pub fn load_sequence(&self, sequence_id: usize) -> Result<Array> {
        if sequence_id >= self.metadata.num_sequences {
            return Err(DistillError::LogitCache(format!(
                "sequence_id {} is out of range (cache contains {} sequences)",
                sequence_id, self.metadata.num_sequences,
            )));
        }
        let path = self.cache_dir.join(format!("seq_{:06}.bin", sequence_id));
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);

        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let compressed: CompressedLogits = bitcode::deserialize(&bytes)?;
        let compressor = LogitCompressor::new(self.compression.clone(), self.top_k);
        compressor.decompress(&compressed, self.metadata.vocab_size)
    }

    /// Check if a sequence is cached.
    pub fn has_sequence(&self, sequence_id: usize) -> bool {
        let path = self.cache_dir.join(format!("seq_{:06}.bin", sequence_id));
        path.exists()
    }

    /// Save metadata.
    pub fn save_metadata(&self) -> Result<()> {
        let metadata_path = self.cache_dir.join("metadata.json");
        let file = File::create(&metadata_path)?;
        serde_json::to_writer_pretty(file, &self.metadata)?;
        Ok(())
    }

    /// Update metadata.
    pub fn set_metadata(&mut self, model: String, vocab_size: usize, max_seq_len: usize) {
        self.metadata.model = model;
        self.metadata.vocab_size = vocab_size;
        self.metadata.max_seq_len = max_seq_len;
        self.metadata.compression = match self.compression {
            CompressionMethod::None => "none",
            CompressionMethod::TopK => "top_k",
            CompressionMethod::Int8 => "int8",
            CompressionMethod::Int4 => "int4",
        }
        .to_string();
        self.metadata.top_k = self.top_k;
    }

    /// Get cache metadata.
    pub fn metadata(&self) -> &CacheMetadata {
        &self.metadata
    }

    /// Get the cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

/// Compressed logits representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressedLogits {
    /// Full logits (no compression).
    Full {
        /// Flattened logit values.
        data: Vec<f32>,
        /// Shape of the original tensor.
        shape: Vec<i32>,
    },
    /// Top-k logits with indices.
    TopK {
        /// Top-k logit values per token.
        values: Vec<f32>,
        /// Indices of top-k logits.
        indices: Vec<i32>,
        /// Number of tokens.
        num_tokens: usize,
        /// K value.
        k: usize,
    },
    /// Quantized to int8 (stored as unsigned bytes; range [0, 255] via affine mapping).
    Int8 {
        /// Quantized values (unsigned 8-bit, zero_point-shifted affine encoding).
        data: Vec<u8>,
        /// Scale factor.
        scale: f32,
        /// Zero point.
        zero_point: f32,
        /// Shape.
        shape: Vec<i32>,
    },
    /// Quantized to int4 (packed as u8).
    Int4 {
        /// Packed nibbles (2 values per byte).
        data: Vec<u8>,
        /// Scale factor.
        scale: f32,
        /// Zero point.
        zero_point: f32,
        /// Shape.
        shape: Vec<i32>,
    },
}

impl CompressedLogits {
    /// Validate the internal invariants of this compressed representation.
    ///
    /// Returns an error with a descriptive message if any invariant is violated.
    pub fn validate(&self) -> Result<()> {
        match self {
            CompressedLogits::TopK {
                values,
                indices,
                num_tokens,
                k,
            } => {
                let expected = num_tokens * k;
                if values.len() != expected {
                    return Err(DistillError::LogitCache(format!(
                        "TopK values length mismatch: expected {} (num_tokens={} * k={}), got {}",
                        expected,
                        num_tokens,
                        k,
                        values.len(),
                    )));
                }
                if indices.len() != expected {
                    return Err(DistillError::LogitCache(format!(
                        "TopK indices length mismatch: expected {} (num_tokens={} * k={}), got {}",
                        expected,
                        num_tokens,
                        k,
                        indices.len(),
                    )));
                }
                Ok(())
            }
            // Full, Int8, Int4 shapes are checked during decompress; no extra validation needed.
            _ => Ok(()),
        }
    }
}

/// Compressor for teacher logits.
pub struct LogitCompressor {
    /// Compression method.
    method: CompressionMethod,
    /// Top-k value.
    top_k: usize,
}

impl LogitCompressor {
    /// Create a new compressor.
    pub fn new(method: CompressionMethod, top_k: usize) -> Self {
        Self { method, top_k }
    }

    /// Compress logits.
    pub fn compress(&self, logits: &Array) -> Result<CompressedLogits> {
        match &self.method {
            CompressionMethod::None => self.compress_none(logits),
            CompressionMethod::TopK => self.compress_topk(logits),
            CompressionMethod::Int8 => self.compress_int8(logits),
            CompressionMethod::Int4 => self.compress_int4(logits),
        }
    }

    /// Decompress logits.
    pub fn decompress(&self, compressed: &CompressedLogits, vocab_size: usize) -> Result<Array> {
        match compressed {
            CompressedLogits::Full { data, shape } => Ok(Array::from_slice(data, shape)),
            CompressedLogits::TopK {
                values,
                indices,
                num_tokens,
                k,
            } => self.decompress_topk(values, indices, *num_tokens, *k, vocab_size),
            CompressedLogits::Int8 {
                data,
                scale,
                zero_point,
                shape,
            } => self.decompress_int8(data, *scale, *zero_point, shape),
            CompressedLogits::Int4 {
                data,
                scale,
                zero_point,
                shape,
            } => self.decompress_int4(data, *scale, *zero_point, shape),
        }
    }

    fn compress_none(&self, logits: &Array) -> Result<CompressedLogits> {
        let data: Vec<f32> = logits.as_slice().to_vec();
        let shape = logits.shape().to_vec();
        Ok(CompressedLogits::Full { data, shape })
    }

    fn compress_topk(&self, logits: &Array) -> Result<CompressedLogits> {
        // logits shape: [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        let shape = logits.shape();
        let vocab_size = shape[shape.len() - 1] as usize;
        let k = self.top_k.min(vocab_size);

        // Flatten to [num_tokens, vocab_size]
        let flat = if shape.len() == 2 {
            logits.clone()
        } else {
            let num_tokens: i32 = shape[..shape.len() - 1].iter().product();
            logits.reshape(&[num_tokens, vocab_size as i32])?
        };
        let num_tokens = flat.dim(0) as usize;

        // CPU top-k selection: offline compression is a preprocessing step run once
        // per dataset, not a training hot path. CPU partial sort is simpler and more
        // predictable than navigating MLX lazy-eval semantics for index arrays.
        flat.eval()?;
        let data_slice: &[f32] = flat.as_slice();

        let mut all_values = Vec::with_capacity(num_tokens * k);
        let mut all_indices = Vec::with_capacity(num_tokens * k);

        for t in 0..num_tokens {
            let row = &data_slice[t * vocab_size..(t + 1) * vocab_size];
            // Collect (index, value) pairs and partial-sort to extract top-k.
            let mut order: Vec<usize> = (0..vocab_size).collect();
            // Use select_nth_unstable_by to partition in O(V) with a partial sort.
            // After this call, order[..k] are the k largest-value indices (unordered).
            if k < vocab_size {
                order.select_nth_unstable_by(k - 1, |&a, &b| {
                    row[b]
                        .partial_cmp(&row[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            // Sort the top-k window descending for deterministic output.
            order[..k].sort_unstable_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &idx in &order[..k] {
                all_values.push(row[idx]);
                all_indices.push(idx as i32);
            }
        }

        let values = all_values;
        let indices = all_indices;

        Ok(CompressedLogits::TopK {
            values,
            indices,
            num_tokens,
            k,
        })
    }

    fn decompress_topk(
        &self,
        values: &[f32],
        indices: &[i32],
        num_tokens: usize,
        k: usize,
        vocab_size: usize,
    ) -> Result<Array> {
        // Reconstruct sparse representation
        // For distillation, we often only need the top-k values (sparse softmax)
        // Initialize with very negative values (will become ~0 after softmax)
        let mut full = vec![-1e10_f32; num_tokens * vocab_size];

        for t in 0..num_tokens {
            for i in 0..k {
                let idx = indices[t * k + i] as usize;
                let val = values[t * k + i];
                if idx < vocab_size {
                    full[t * vocab_size + idx] = val;
                }
            }
        }

        Ok(Array::from_slice(
            &full,
            &[num_tokens as i32, vocab_size as i32],
        ))
    }

    fn compress_int8(&self, logits: &Array) -> Result<CompressedLogits> {
        let data: Vec<f32> = logits.as_slice().to_vec();
        let shape = logits.shape().to_vec();

        // Per-token quantization: compute min/max per row so that each token's
        // dynamic range is captured independently. This avoids a single outlier
        // token causing all other tokens to be squashed into a narrow band.
        //
        // Layout: `data` is stored as a flat Vec<u8> of quantised values followed
        // immediately by 2 * num_tokens f32s (scale_0, zp_0, scale_1, zp_1, …)
        // packed as raw little-endian bytes.
        //
        // The legacy global-scale format is detected at decompress time by
        // `scale.is_nan()` (the sentinel stored in the outer struct field).

        let num_tokens: usize = shape[..shape.len() - 1]
            .iter()
            .map(|&s| s as usize)
            .product();
        let token_size = data.len() / num_tokens.max(1);

        let mut quantized: Vec<u8> = Vec::with_capacity(data.len() + num_tokens * 8);
        let mut per_token_meta: Vec<f32> = Vec::with_capacity(num_tokens * 2);

        for t in 0..num_tokens {
            let token_data = &data[t * token_size..(t + 1) * token_size];
            let min = token_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = token_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let scale = if (max - min).abs() < 1e-12 {
                // Constant token; avoid division by zero
                1.0_f32
            } else {
                (max - min) / 255.0
            };
            let zero_point = min;

            per_token_meta.push(scale);
            per_token_meta.push(zero_point);

            for &v in token_data {
                let q = ((v - zero_point) / scale).round() as i32;
                quantized.push(q.clamp(0, 255) as u8);
            }
        }

        // Append per-token (scale, zero_point) pairs as raw f32 LE bytes.
        for &f in &per_token_meta {
            quantized.extend_from_slice(&f.to_le_bytes());
        }

        Ok(CompressedLogits::Int8 {
            data: quantized,
            // NaN sentinel signals that per-token metadata is embedded in `data`.
            scale: f32::NAN,
            zero_point: 0.0,
            shape,
        })
    }

    fn decompress_int8(
        &self,
        data: &[u8],
        scale: f32,
        zero_point: f32,
        shape: &[i32],
    ) -> Result<Array> {
        if scale.is_nan() {
            // New per-token format: parse embedded (scale, zp) pairs from the tail.
            let num_tokens: usize = shape[..shape.len() - 1]
                .iter()
                .map(|&s| s as usize)
                .product();
            let token_size: usize = shape[shape.len() - 1] as usize;
            let meta_bytes = num_tokens * 2 * 4; // 2 f32s per token

            if data.len() < meta_bytes {
                return Err(DistillError::LogitCache(
                    "Int8 compressed data too short to contain per-token metadata".to_string(),
                ));
            }

            let quant_len = data.len() - meta_bytes;
            let quant_data = &data[..quant_len];
            let meta_data = &data[quant_len..];

            let mut dequantized = Vec::with_capacity(num_tokens * token_size);
            for t in 0..num_tokens {
                let meta_off = t * 8;
                let sc = f32::from_le_bytes(meta_data[meta_off..meta_off + 4].try_into().unwrap());
                let zp =
                    f32::from_le_bytes(meta_data[meta_off + 4..meta_off + 8].try_into().unwrap());
                for &q in &quant_data[t * token_size..(t + 1) * token_size] {
                    dequantized.push((q as f32) * sc + zp);
                }
            }

            Ok(Array::from_slice(&dequantized, shape))
        } else {
            // Legacy global-scale format.
            let dequantized: Vec<f32> = data
                .iter()
                .map(|&q| (q as f32) * scale + zero_point)
                .collect();
            Ok(Array::from_slice(&dequantized, shape))
        }
    }

    fn compress_int4(&self, logits: &Array) -> Result<CompressedLogits> {
        let data: Vec<f32> = logits.as_slice().to_vec();
        let shape = logits.shape().to_vec();

        // Find min/max
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let scale = (max - min) / 15.0;
        let zero_point = min;

        // Quantize to 4-bit and pack into bytes
        let mut packed = Vec::with_capacity(data.len().div_ceil(2));

        for chunk in data.chunks(2) {
            let q0 = ((chunk[0] - zero_point) / scale).round() as u8;
            let q1 = if chunk.len() > 1 {
                ((chunk[1] - zero_point) / scale).round() as u8
            } else {
                0
            };

            let packed_byte = (q0.min(15)) | ((q1.min(15)) << 4);
            packed.push(packed_byte);
        }

        Ok(CompressedLogits::Int4 {
            data: packed,
            scale,
            zero_point,
            shape,
        })
    }

    fn decompress_int4(
        &self,
        data: &[u8],
        scale: f32,
        zero_point: f32,
        shape: &[i32],
    ) -> Result<Array> {
        let num_elements: usize = shape.iter().map(|&s| s as usize).product();
        let mut dequantized = Vec::with_capacity(num_elements);

        for &packed in data {
            let q0 = (packed & 0x0F) as f32;
            let q1 = ((packed >> 4) & 0x0F) as f32;

            dequantized.push(q0 * scale + zero_point);
            if dequantized.len() < num_elements {
                dequantized.push(q1 * scale + zero_point);
            }
        }

        dequantized.truncate(num_elements);
        Ok(Array::from_slice(&dequantized, shape))
    }
}

/// Compression statistics for reporting.
#[derive(Debug)]
pub struct CompressionStats {
    /// Original size in bytes.
    pub original_bytes: usize,
    /// Compressed size in bytes.
    pub compressed_bytes: usize,
    /// Compression ratio.
    pub ratio: f32,
}

impl CompressionStats {
    /// Calculate compression ratio.
    pub fn new(original_bytes: usize, compressed_bytes: usize) -> Self {
        let ratio = original_bytes as f32 / compressed_bytes.max(1) as f32;
        Self {
            original_bytes,
            compressed_bytes,
            ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_topk_compression() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);

        let compressor = LogitCompressor::new(CompressionMethod::TopK, 2);
        let compressed = compressor.compress(&logits).unwrap();

        match &compressed {
            CompressedLogits::TopK {
                values,
                indices,
                num_tokens,
                k,
            } => {
                assert_eq!(*num_tokens, 2);
                assert_eq!(*k, 2);
                assert_eq!(values.len(), 4); // 2 tokens * 2 top-k
                assert_eq!(indices.len(), 4);

                // First token: top-2 are indices 3 and 2 (values 4.0 and 3.0).
                // argpartition does not guarantee order within the top-k window,
                // so we check the set rather than the specific order.
                let token0_indices: std::collections::HashSet<i32> =
                    indices[..2].iter().cloned().collect();
                assert!(
                    token0_indices.contains(&3) && token0_indices.contains(&2),
                    "Expected top-2 indices {{2, 3}} for token 0, got {:?}",
                    &indices[..2],
                );
                let token0_values: std::collections::HashSet<i32> =
                    values[..2].iter().map(|&v| v.round() as i32).collect();
                assert!(
                    token0_values.contains(&4) && token0_values.contains(&3),
                    "Expected top-2 values {{3.0, 4.0}} for token 0, got {:?}",
                    &values[..2],
                );
            }
            _ => panic!("Wrong compression type"),
        }
    }

    #[test]
    #[serial]
    fn test_topk_roundtrip() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 10.0, 4.0, 5.0, 6.0, 20.0, 8.0], &[2, 4]);

        let compressor = LogitCompressor::new(CompressionMethod::TopK, 2);
        let compressed = compressor.compress(&logits).unwrap();
        let decompressed = compressor.decompress(&compressed, 4).unwrap();

        let data: Vec<f32> = decompressed.as_slice().to_vec();
        println!("decompressed: {:?}", &data);
        if let CompressedLogits::TopK {
            ref values,
            ref indices,
            ..
        } = compressed
        {
            println!("values: {:?}", values);
            println!("indices: {:?}", indices);
        }

        // Top values should be preserved, others should be very negative
        assert!((data[2] - 10.0).abs() < 1e-5); // Token 0, index 2
        assert!(data[0] < -1e5); // Token 0, index 0 (not in top-2)
        assert!((data[6] - 20.0).abs() < 1e-5); // Token 1, index 2
    }

    #[test]
    #[serial]
    fn test_int8_roundtrip() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);

        let compressor = LogitCompressor::new(CompressionMethod::Int8, 128);
        let compressed = compressor.compress(&logits).unwrap();
        let decompressed = compressor.decompress(&compressed, 4).unwrap();

        let original: Vec<f32> = logits.as_slice().to_vec();
        let recovered: Vec<f32> = decompressed.as_slice().to_vec();

        // Should be close (within quantization error)
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.1, "Int8 roundtrip error: {} vs {}", o, r);
        }
    }

    #[test]
    #[serial]
    fn test_int4_roundtrip() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);

        let compressor = LogitCompressor::new(CompressionMethod::Int4, 128);
        let compressed = compressor.compress(&logits).unwrap();
        let decompressed = compressor.decompress(&compressed, 4).unwrap();

        let original: Vec<f32> = logits.as_slice().to_vec();
        let recovered: Vec<f32> = decompressed.as_slice().to_vec();

        // Int4 has lower precision
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1.0, "Int4 roundtrip error: {} vs {}", o, r);
        }
    }

    #[test]
    #[serial]
    fn test_none_compression() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);

        let compressor = LogitCompressor::new(CompressionMethod::None, 128);
        let compressed = compressor.compress(&logits).unwrap();
        let decompressed = compressor.decompress(&compressed, 2).unwrap();

        let original: Vec<f32> = logits.as_slice().to_vec();
        let recovered: Vec<f32> = decompressed.as_slice().to_vec();

        assert_eq!(original, recovered);
    }
}
