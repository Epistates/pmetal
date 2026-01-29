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
use std::io::{BufReader, BufWriter, Write};
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
        bincode::serialize_into(&mut writer, &compressed)?;
        writer.flush()?;

        self.metadata.num_sequences = self.metadata.num_sequences.max(sequence_id + 1);
        if self.metadata.vocab_size == 0 {
            self.metadata.vocab_size = logits.dim(-1) as usize;
        }

        Ok(())
    }

    /// Load cached logits for a sequence.
    pub fn load_sequence(&self, sequence_id: usize) -> Result<Array> {
        let path = self.cache_dir.join(format!("seq_{:06}.bin", sequence_id));
        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        let compressed: CompressedLogits = bincode::deserialize_from(reader)?;
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
    /// Quantized to int8.
    Int8 {
        /// Quantized values.
        data: Vec<i8>,
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
        let data: Vec<f32> = flat.as_slice().to_vec();

        // Find top-k for each token
        let mut all_values = Vec::with_capacity(num_tokens * k);
        let mut all_indices = Vec::with_capacity(num_tokens * k);

        for t in 0..num_tokens {
            let token_logits: Vec<(usize, f32)> = (0..vocab_size)
                .map(|v| (v, data[t * vocab_size + v]))
                .collect();

            // Partial sort to get top-k
            let mut sorted = token_logits;
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (idx, val) in sorted.into_iter().take(k) {
                all_values.push(val);
                all_indices.push(idx as i32);
            }
        }

        Ok(CompressedLogits::TopK {
            values: all_values,
            indices: all_indices,
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

        // Find min/max for quantization
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let scale = (max - min) / 255.0;
        let zero_point = min;

        let quantized: Vec<i8> = data
            .iter()
            .map(|&v| {
                let q = ((v - zero_point) / scale).round() as i32;
                q.clamp(0, 255) as i8
            })
            .collect();

        Ok(CompressedLogits::Int8 {
            data: quantized,
            scale,
            zero_point,
            shape,
        })
    }

    fn decompress_int8(
        &self,
        data: &[i8],
        scale: f32,
        zero_point: f32,
        shape: &[i32],
    ) -> Result<Array> {
        let dequantized: Vec<f32> = data
            .iter()
            .map(|&q| (q as u8 as f32) * scale + zero_point)
            .collect();

        Ok(Array::from_slice(&dequantized, shape))
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

                // First token: top-2 are indices 3, 2 (values 4, 3)
                assert_eq!(indices[0], 3);
                assert_eq!(indices[1], 2);
                assert!((values[0] - 4.0).abs() < 1e-5);
                assert!((values[1] - 3.0).abs() < 1e-5);
            }
            _ => panic!("Wrong compression type"),
        }
    }

    #[test]
    fn test_topk_roundtrip() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 10.0, 4.0, 5.0, 6.0, 20.0, 8.0], &[2, 4]);

        let compressor = LogitCompressor::new(CompressionMethod::TopK, 2);
        let compressed = compressor.compress(&logits).unwrap();
        let decompressed = compressor.decompress(&compressed, 4).unwrap();

        let data: Vec<f32> = decompressed.as_slice().to_vec();

        // Top values should be preserved, others should be very negative
        assert!((data[2] - 10.0).abs() < 1e-5); // Token 0, index 2
        assert!(data[0] < -1e5); // Token 0, index 0 (not in top-2)
        assert!((data[6] - 20.0).abs() < 1e-5); // Token 1, index 2
    }

    #[test]
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
