//! Offline distillation with logit caching and compression.
//!
//! Offline distillation pre-computes teacher logits and stores them on disk,
//! allowing the student to train without running the teacher model during training.
//! This is memory-efficient and allows using larger teachers.
//!
//! # Compression Methods
//!
//! - **TopK**: Only store the top-k logits per token (e.g., top-128 out of 150k vocab),
//!   as `(value, index)` pairs. Read back sparse with [`LogitCache::load_sequence_sparse`]
//!   and fed to [`SparseKlLoss`](crate::losses::SparseKlLoss), no dense
//!   `[tokens, vocab]` teacher is ever formed.
//! - **Sampled**: Random-Sampling KD (Anshumann et al., ACL 2025). Draw a few
//!   tokens per position from the teacher's distribution and keep the distinct
//!   ones with their frequencies ([`SparseTargets`]). Their empirical
//!   distribution is an *unbiased* estimate of the teacher, where the top-k's
//!   is biased toward its head; about twelve draws match full distillation.
//!   The draws are taken at one temperature, which the loss must then use.
//! - **Int8**: Quantize full logits to 8-bit codes with a scale and offset per token.
//! - **Int4**: Quantize full logits to 4-bit codes with a scale and offset per token.
//!
//! TopK is typically the best choice, as most probability mass is in the top few tokens.
//!
//! Logits that are not finite are refused by the quantizers (a NaN means the
//! teacher forward diverged, and an infinity would set the scale of its row);
//! the top-k path rejects NaN and accepts `-inf` (a masked token is simply
//! never in the top k).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read as _, Write};
use std::path::{Path, PathBuf};

use pmetal_runtime::compat::{Array, Dtype, ops, random};
use serde::{Deserialize, Serialize};

use crate::{CompressionMethod, DistillError, Result};

/// Dense fill for the logits a top-k cache did not keep: after a softmax their
/// probability is exactly zero, so the kept entries are renormalized among
/// themselves.
const MASKED_LOGIT: f32 = -1e10;

/// Cache for pre-computed teacher logits.
pub struct LogitCache {
    /// Path to the cache directory.
    cache_dir: PathBuf,
    /// Compression method used.
    compression: CompressionMethod,
    /// Number of top-k logits to keep (or draws per token when sampling).
    top_k: usize,
    /// Temperature of a `sampled` cache's draws.
    sampling_temperature: f32,
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

fn compression_name(method: &CompressionMethod) -> &'static str {
    match method {
        CompressionMethod::None => "none",
        CompressionMethod::TopK => "top_k",
        CompressionMethod::Int8 => "int8",
        CompressionMethod::Int4 => "int4",
        CompressionMethod::Sampled => "sampled",
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
            sampling_temperature: 1.0,
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
            "sampled" => CompressionMethod::Sampled,
            other => {
                return Err(DistillError::LogitCache(format!(
                    "cache metadata names an unknown compression '{other}' \
                     (expected none, top_k, int8, int4 or sampled)"
                )));
            }
        };

        Ok(Self {
            cache_dir,
            compression,
            top_k: metadata.top_k,
            sampling_temperature: 1.0,
            metadata,
        })
    }

    /// The temperature a `sampled` cache draws its tokens at (default 1).
    ///
    /// The teacher's tokens are sampled from `softmax(logits / T)`, and the
    /// loss that reads them back must use the same `T`. Ignored by the other
    /// compressions.
    pub fn with_sampling_temperature(mut self, temperature: f32) -> Self {
        self.sampling_temperature = temperature;
        self
    }

    fn sequence_path(&self, sequence_id: usize) -> PathBuf {
        self.cache_dir.join(format!("seq_{:06}.bin", sequence_id))
    }

    /// Cache logits for a sequence.
    ///
    /// The file is written under a temporary name and renamed into place, so a
    /// crash cannot leave a truncated sequence that [`has_sequence`](Self::has_sequence)
    /// would accept.
    pub fn cache_sequence(&mut self, sequence_id: usize, logits: &Array) -> Result<()> {
        let compressor = LogitCompressor::new(self.compression.clone(), self.top_k)
            .with_sampling_temperature(self.sampling_temperature);
        let compressed = compressor.compress(logits)?;

        let path = self.sequence_path(sequence_id);
        let staging = path.with_extension("bin.tmp");
        {
            let mut writer = BufWriter::new(File::create(&staging)?);
            writer.write_all(&bitcode::serialize(&compressed)?)?;
            writer.flush()?;
        }
        std::fs::rename(&staging, &path)?;

        self.metadata.num_sequences = self.metadata.num_sequences.max(sequence_id + 1);
        if self.metadata.vocab_size == 0 {
            self.metadata.vocab_size = logits.dim(-1) as usize;
        }

        Ok(())
    }

    fn read_compressed(&self, sequence_id: usize) -> Result<CompressedLogits> {
        if sequence_id >= self.metadata.num_sequences {
            return Err(DistillError::LogitCache(format!(
                "sequence_id {} is out of range (cache contains {} sequences)",
                sequence_id, self.metadata.num_sequences,
            )));
        }
        let mut reader = BufReader::new(File::open(self.sequence_path(sequence_id))?);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let compressed: CompressedLogits = bitcode::deserialize(&bytes)?;
        compressed.validate()?;
        Ok(compressed)
    }

    /// Load cached logits for a sequence as a dense tensor.
    ///
    /// For a top-k cache this is `[.., vocab]` with the entries that were not
    /// kept at a very negative logit; prefer
    /// [`load_sequence_sparse`](Self::load_sequence_sparse) for those.
    pub fn load_sequence(&self, sequence_id: usize) -> Result<Array> {
        let compressed = self.read_compressed(sequence_id)?;
        let compressor = LogitCompressor::new(self.compression.clone(), self.top_k);
        compressor.decompress(&compressed, self.metadata.vocab_size)
    }

    /// Load a `sampled` sequence as the empirical teacher distribution of its
    /// draws: per position, the distinct tokens drawn and their frequencies.
    ///
    /// Errors if the cache holds anything but sampled draws.
    pub fn load_sequence_sampled(&self, sequence_id: usize) -> Result<SparseTargets> {
        match self.read_compressed(sequence_id)? {
            CompressedLogits::Sampled {
                probs,
                indices,
                slots,
                shape,
                temperature,
            } => {
                let mut out_shape = shape[..shape.len() - 1].to_vec();
                out_shape.push(slots as i32);
                SparseTargets::new(
                    Array::from_f32_slice(&probs, &out_shape),
                    Array::from_i32_slice_shaped(&indices, &out_shape),
                    shape[shape.len() - 1] as usize,
                    temperature,
                )
            }
            _ => Err(DistillError::LogitCache(
                "this cache does not hold sampled draws; \
                 use `load_sequence_sparse` (top-k) or `load_sequence`"
                    .to_string(),
            )),
        }
    }

    /// Load a top-k sequence without densifying it: the kept logits and their
    /// token ids, `[.., k]` each.
    ///
    /// Errors if the cache stores dense logits (`none`, `int8`, `int4`).
    pub fn load_sequence_sparse(&self, sequence_id: usize) -> Result<SparseLogits> {
        match self.read_compressed(sequence_id)? {
            CompressedLogits::TopKSparse {
                values,
                indices,
                k,
                shape,
            } => {
                let mut out_shape = shape[..shape.len() - 1].to_vec();
                out_shape.push(k as i32);
                SparseLogits::new(
                    Array::from_f32_slice(&values, &out_shape),
                    Array::from_i32_slice_shaped(&indices, &out_shape),
                    shape[shape.len() - 1] as usize,
                )
            }
            CompressedLogits::TopK {
                values,
                indices,
                num_tokens,
                k,
            } => SparseLogits::new(
                Array::from_f32_slice(&values, &[num_tokens as i32, k as i32]),
                Array::from_i32_slice_shaped(&indices, &[num_tokens as i32, k as i32]),
                self.metadata.vocab_size,
            ),
            _ => Err(DistillError::LogitCache(
                "this cache stores dense logits; load them with `load_sequence`".to_string(),
            )),
        }
    }

    /// Check if a sequence is cached.
    pub fn has_sequence(&self, sequence_id: usize) -> bool {
        self.sequence_path(sequence_id).exists()
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
        self.metadata.compression = compression_name(&self.compression).to_string();
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

/// A teacher's top-k logits for a run of tokens, as stored: `[.., k]` logit
/// values and the `[.., k]` token ids they belong to, sorted from the largest
/// logit down.
///
/// The teacher distribution this stands for is the softmax of the kept logits
/// among themselves (every other token has probability zero), at whatever
/// temperature the loss applies.
#[derive(Debug, Clone)]
pub struct SparseLogits {
    values: Array,
    indices: Array,
    vocab_size: usize,
}

impl SparseLogits {
    /// Wrap `[.., k]` values (f32) and `[.., k]` ids (int32) from a
    /// `vocab_size` vocabulary.
    pub fn new(values: Array, indices: Array, vocab_size: usize) -> Result<Self> {
        if values.shape() != indices.shape() || values.ndim() == 0 {
            return Err(DistillError::LogitCache(format!(
                "sparse logits need values and indices of one shape [.., k]; got {:?} and {:?}",
                values.shape(),
                indices.shape()
            )));
        }
        Ok(Self {
            values,
            indices,
            vocab_size,
        })
    }

    /// The kept logits, `[.., k]` f32.
    pub fn values(&self) -> &Array {
        &self.values
    }

    /// The token ids of the kept logits, `[.., k]` int32.
    pub fn indices(&self) -> &Array {
        &self.indices
    }

    /// Size of the vocabulary the ids index into.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// How many logits are kept per token.
    pub fn k(&self) -> usize {
        self.values.dim(-1) as usize
    }

    /// The tensor's shape, `[.., k]`.
    pub fn shape(&self) -> &[i32] {
        self.values.shape()
    }

    /// The dense `[.., vocab]` logits this stands for, the entries that were
    /// not kept at a very negative value. Materializes `vocab / k` times more
    /// than the sparse form holds.
    pub fn to_dense(&self) -> Array {
        let leading = self.values.shape().len() - 1;
        let mut shape = self.values.shape()[..leading].to_vec();
        shape.push(self.vocab_size as i32);
        Array::full(&shape, MASKED_LOGIT, Dtype::Float32.as_i32()).put_along_axis_op(
            &self.indices,
            &self.values,
            -1,
        )
    }

    /// Collate per-sequence logits into one `[batch, seq_len, k]` batch.
    ///
    /// Each row is `[tokens, k]` or `[1, tokens, k]`; rows shorter than
    /// `seq_len` are padded with zeros and longer ones truncated. Padded
    /// positions carry no teacher: mask them out with the loss's token
    /// weights. All rows must have the same `k` and vocabulary.
    pub fn stack(rows: &[SparseLogits], seq_len: usize) -> Result<SparseLogits> {
        let Some(first) = rows.first() else {
            return Err(DistillError::LogitCache(
                "cannot stack zero sequences".to_string(),
            ));
        };
        let k = first.k();
        let mut values = vec![0.0_f32; rows.len() * seq_len * k];
        let mut indices = vec![0_i32; rows.len() * seq_len * k];
        for (b, row) in rows.iter().enumerate() {
            if row.k() != k || row.vocab_size != first.vocab_size {
                return Err(DistillError::LogitCache(format!(
                    "cannot stack sequences with different k or vocabulary ({} / {} vs {} / {})",
                    row.k(),
                    row.vocab_size,
                    k,
                    first.vocab_size
                )));
            }
            let tokens = row.values.size() / k;
            let take = tokens.min(seq_len) * k;
            let dst = b * seq_len * k;
            values[dst..dst + take].copy_from_slice(&row.values.as_slice::<f32>()[..take]);
            indices[dst..dst + take].copy_from_slice(&row.indices.as_slice::<i32>()[..take]);
        }
        let shape = [rows.len() as i32, seq_len as i32, k as i32];
        SparseLogits::new(
            Array::from_f32_slice(&values, &shape),
            Array::from_i32_slice_shaped(&indices, &shape),
            first.vocab_size,
        )
    }
}

/// A teacher distribution as sampled draws: for each position the distinct
/// tokens drawn and the frequency of each (`[.., slots]` each, the frequencies
/// of a position summing to 1; unused slots hold frequency 0).
///
/// This is Random-Sampling KD (Anshumann et al., ACL 2025): `N` draws from
/// `softmax(logits / T)` make an unbiased estimate of the teacher distribution,
/// so the expected gradient of the loss against it is the full distillation
/// gradient, where truncating to the top-k is a biased estimate. About a dozen
/// draws match full distillation at a fraction of the storage.
#[derive(Debug, Clone)]
pub struct SparseTargets {
    probs: Array,
    indices: Array,
    vocab_size: usize,
    temperature: f32,
}

impl SparseTargets {
    /// Wrap `[.., slots]` frequencies (f32) and `[.., slots]` ids (int32) drawn
    /// at `temperature` from a `vocab_size` vocabulary.
    pub fn new(probs: Array, indices: Array, vocab_size: usize, temperature: f32) -> Result<Self> {
        if probs.shape() != indices.shape() || probs.shape().is_empty() {
            return Err(DistillError::LogitCache(format!(
                "sparse targets need frequencies and indices of one shape [.., slots]; got {:?} and {:?}",
                probs.shape(),
                indices.shape()
            )));
        }
        Ok(Self {
            probs,
            indices,
            vocab_size,
            temperature,
        })
    }

    /// The drawn tokens' frequencies, `[.., slots]` f32.
    pub fn probs(&self) -> &Array {
        &self.probs
    }

    /// The drawn tokens' ids, `[.., slots]` int32.
    pub fn indices(&self) -> &Array {
        &self.indices
    }

    /// Size of the vocabulary the ids index into.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The temperature the tokens were drawn at; the loss must use it too.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Slots per position.
    pub fn slots(&self) -> usize {
        self.probs.dim(-1) as usize
    }

    /// The tensor's shape, `[.., slots]`.
    pub fn shape(&self) -> &[i32] {
        self.probs.shape()
    }

    /// The dense `[.., vocab]` empirical distribution.
    pub fn to_dense_probs(&self) -> Array {
        let slots = self.slots();
        let positions = self.probs.size() / slots;
        let probs = self.probs.as_slice::<f32>();
        let indices = self.indices.as_slice::<i32>();
        let mut dense = vec![0.0_f32; positions * self.vocab_size];
        for p in 0..positions {
            for s in 0..slots {
                // Unused slots have frequency 0 and add nothing.
                dense[p * self.vocab_size + indices[p * slots + s] as usize] +=
                    probs[p * slots + s];
            }
        }
        let mut shape = self.probs.shape()[..self.probs.shape().len() - 1].to_vec();
        shape.push(self.vocab_size as i32);
        Array::from_f32_slice(&dense, &shape)
    }

    /// Collate per-sequence targets into one `[batch, seq_len, slots]` batch,
    /// as [`SparseLogits::stack`] does. Padded positions have frequency 0 and
    /// must be masked out by the loss's token weights. All rows must share the
    /// slots, vocabulary and temperature.
    pub fn stack(rows: &[SparseTargets], seq_len: usize) -> Result<SparseTargets> {
        let Some(first) = rows.first() else {
            return Err(DistillError::LogitCache(
                "cannot stack zero sequences".to_string(),
            ));
        };
        let slots = first.slots();
        let mut probs = vec![0.0_f32; rows.len() * seq_len * slots];
        let mut indices = vec![0_i32; rows.len() * seq_len * slots];
        for (b, row) in rows.iter().enumerate() {
            if row.slots() != slots
                || row.vocab_size != first.vocab_size
                || row.temperature != first.temperature
            {
                return Err(DistillError::LogitCache(
                    "cannot stack sampled sequences with different slots, vocabulary or \
                     temperature"
                        .to_string(),
                ));
            }
            let take = (row.probs.size() / slots).min(seq_len) * slots;
            let dst = b * seq_len * slots;
            probs[dst..dst + take].copy_from_slice(&row.probs.as_slice::<f32>()[..take]);
            indices[dst..dst + take].copy_from_slice(&row.indices.as_slice::<i32>()[..take]);
        }
        let shape = [rows.len() as i32, seq_len as i32, slots as i32];
        SparseTargets::new(
            Array::from_f32_slice(&probs, &shape),
            Array::from_i32_slice_shaped(&indices, &shape),
            first.vocab_size,
            first.temperature,
        )
    }
}

/// Compressed logits representation.
///
/// Variants are serialized by position: new ones go at the end, and the old
/// ones stay readable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressedLogits {
    /// Full logits (no compression).
    Full {
        /// Flattened logit values.
        data: Vec<f32>,
        /// Shape of the original tensor.
        shape: Vec<i32>,
    },
    /// Top-k logits with indices, flattened to `[num_tokens, k]`. Read-back
    /// only: the shape of the original tensor was not recorded. New caches
    /// use [`CompressedLogits::TopKSparse`].
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
    /// Legacy quantized-int8 representation kept for read-back compatibility.
    ///
    /// Two on-disk layouts encode through this variant:
    ///   1. *Pre-2026 global-scale*: `scale` is finite, `data.len() == n`,
    ///      decode is `(q[i] as f32) * scale + zero_point`.
    ///   2. *Per-token via NaN sentinel*: `scale.is_nan()` flagged that
    ///      `data` carried `[quant; n] ++ [(scale_t, zp_t); num_tokens]` —
    ///      the very pattern Phase 1.3 (Apr 2026) replaces because a real
    ///      input NaN could be misclassified as the sentinel.
    ///
    /// New caches must use [`CompressedLogits::Int8PerToken`] instead.
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
    /// Legacy quantized-int4 representation (one scale for the whole tensor),
    /// kept for read-back. New caches use [`CompressedLogits::Int4PerToken`].
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
    /// Quantized to int8 with explicit per-token (scale, zero_point) metadata.
    ///
    /// Layout: `quant` holds `n = product(shape)` u8 values; `per_token_meta`
    /// holds `2 * num_tokens` f32s laid out as `[scale_0, zp_0, scale_1, zp_1, …]`
    /// where `num_tokens = product(shape[..-1])`. This variant supersedes the
    /// NaN-sentinel scheme inside [`CompressedLogits::Int8`] and is robust
    /// against real NaN values in the input distribution.
    Int8PerToken {
        /// Quantized values, length `product(shape)`.
        quant: Vec<u8>,
        /// Per-token (scale, zero_point) pairs, length `2 * num_tokens`.
        per_token_meta: Vec<f32>,
        /// Shape of the original tensor; the trailing axis is the per-token axis.
        shape: Vec<i32>,
    },
    /// Top-k logits and their token ids, with the original tensor's shape.
    TopKSparse {
        /// The kept logits, `num_tokens * k` of them, each token's from the
        /// largest down.
        values: Vec<f32>,
        /// Their token ids, same layout.
        indices: Vec<i32>,
        /// How many were kept per token.
        k: usize,
        /// Shape of the original tensor, `[.., vocab]`.
        shape: Vec<i32>,
    },
    /// Quantized to 4 bits with a (scale, zero_point) pair per token.
    ///
    /// `packed` holds the `product(shape)` codes two to a byte (low nibble
    /// first, across token boundaries); `per_token_meta` is laid out as in
    /// [`CompressedLogits::Int8PerToken`].
    Int4PerToken {
        /// Packed 4-bit codes.
        packed: Vec<u8>,
        /// Per-token (scale, zero_point) pairs, length `2 * num_tokens`.
        per_token_meta: Vec<f32>,
        /// Shape of the original tensor; the trailing axis is the per-token axis.
        shape: Vec<i32>,
    },
    /// Random-sampling KD: draws from `softmax(logits / temperature)`, as the
    /// distinct tokens drawn per position and their frequencies.
    Sampled {
        /// Frequencies, `num_tokens * slots` of them, most frequent first;
        /// unused slots are 0.
        probs: Vec<f32>,
        /// The tokens' ids, same layout (unused slots hold 0).
        indices: Vec<i32>,
        /// Slots per position (the number of draws bounds it).
        slots: usize,
        /// Shape of the original tensor, `[.., vocab]`.
        shape: Vec<i32>,
        /// The temperature the draws were taken at.
        temperature: f32,
    },
}

/// `(num_tokens, token_size)` of a `[.., vocab]` shape.
fn token_layout(shape: &[i32]) -> Result<(usize, usize)> {
    let Some((&last, leading)) = shape.split_last() else {
        return Err(DistillError::LogitCache(
            "a logit tensor needs at least one axis".to_string(),
        ));
    };
    if last <= 0 || leading.iter().any(|&d| d <= 0) {
        return Err(DistillError::LogitCache(format!(
            "logit tensor shape {shape:?} has an empty axis"
        )));
    }
    let num_tokens = leading.iter().map(|&d| d as usize).product::<usize>();
    Ok((num_tokens, last as usize))
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
            CompressedLogits::TopKSparse {
                values,
                indices,
                k,
                shape,
            } => {
                let (num_tokens, _) = token_layout(shape)?;
                let expected = num_tokens * k;
                if values.len() != expected || indices.len() != expected {
                    return Err(DistillError::LogitCache(format!(
                        "TopKSparse payload mismatch: expected {expected} values and indices \
                         (num_tokens={num_tokens} * k={k}), got {} and {}",
                        values.len(),
                        indices.len(),
                    )));
                }
                Ok(())
            }
            CompressedLogits::Int8PerToken {
                quant,
                per_token_meta,
                shape,
            } => {
                let (num_tokens, token_size) = token_layout(shape)?;
                let expected_quant = num_tokens * token_size;
                if quant.len() != expected_quant {
                    return Err(DistillError::LogitCache(format!(
                        "Int8PerToken quant length mismatch: expected {} (num_tokens={} * token_size={}), got {}",
                        expected_quant,
                        num_tokens,
                        token_size,
                        quant.len(),
                    )));
                }
                let expected_meta = num_tokens * 2;
                if per_token_meta.len() != expected_meta {
                    return Err(DistillError::LogitCache(format!(
                        "Int8PerToken per_token_meta length mismatch: expected {} (2*num_tokens={}), got {}",
                        expected_meta,
                        num_tokens,
                        per_token_meta.len(),
                    )));
                }
                Ok(())
            }
            CompressedLogits::Sampled {
                probs,
                indices,
                slots,
                shape,
                ..
            } => {
                let (num_tokens, vocab) = token_layout(shape)?;
                let expected = num_tokens * slots;
                if probs.len() != expected || indices.len() != expected {
                    return Err(DistillError::LogitCache(format!(
                        "Sampled payload mismatch: expected {expected} frequencies and ids \
                         (num_tokens={num_tokens} * slots={slots}), got {} and {}",
                        probs.len(),
                        indices.len(),
                    )));
                }
                if indices.iter().any(|&i| i < 0 || i as usize >= vocab) {
                    return Err(DistillError::LogitCache(format!(
                        "Sampled ids outside the vocabulary of {vocab}"
                    )));
                }
                Ok(())
            }
            CompressedLogits::Int4PerToken {
                packed,
                per_token_meta,
                shape,
            } => {
                let (num_tokens, token_size) = token_layout(shape)?;
                let expected_packed = (num_tokens * token_size).div_ceil(2);
                if packed.len() != expected_packed {
                    return Err(DistillError::LogitCache(format!(
                        "Int4PerToken packed length mismatch: expected {expected_packed}, got {}",
                        packed.len(),
                    )));
                }
                if per_token_meta.len() != num_tokens * 2 {
                    return Err(DistillError::LogitCache(format!(
                        "Int4PerToken per_token_meta length mismatch: expected {}, got {}",
                        num_tokens * 2,
                        per_token_meta.len(),
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
    /// Top-k value (draws per token for `Sampled`).
    top_k: usize,
    /// Temperature of `Sampled` draws.
    sampling_temperature: f32,
}

/// The logits as host f32, with the finiteness the quantizers require.
fn host_logits(logits: &Array, require_finite: bool) -> Result<(Vec<f32>, Vec<i32>)> {
    let shape = logits.shape().to_vec();
    let n: usize = shape.iter().map(|&s| s as usize).product();
    let data = logits
        .clone()
        .to_f32_vec(n)
        .ok_or_else(|| DistillError::LogitCache("failed to read logits as f32".to_string()))?;
    if require_finite && data.iter().any(|v| !v.is_finite()) {
        return Err(DistillError::LogitCache(
            "teacher logits contain non-finite values; quantizing them would corrupt the \
             scale of their row (did the teacher forward diverge?)"
                .to_string(),
        ));
    }
    Ok((data, shape))
}

/// Per-token affine quantization: `(scale, zero_point, codes)` for one row,
/// with `levels` codes `0..=levels`. A constant row gets scale 1 and all-zero
/// codes, so it decodes to itself.
fn quantize_row(row: &[f32], levels: f32, mut emit: impl FnMut(u8)) -> (f32, f32) {
    let min = row.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scale = if max > min { (max - min) / levels } else { 1.0 };
    for &v in row {
        emit((((v - min) / scale).round()).clamp(0.0, levels) as u8);
    }
    (scale, min)
}

impl LogitCompressor {
    /// Create a new compressor. `top_k` is the number of logits kept per token
    /// for `TopK` and the number of draws per token for `Sampled`.
    pub fn new(method: CompressionMethod, top_k: usize) -> Self {
        Self {
            method,
            top_k,
            sampling_temperature: 1.0,
        }
    }

    /// The temperature `Sampled` draws at (default 1).
    pub fn with_sampling_temperature(mut self, temperature: f32) -> Self {
        self.sampling_temperature = temperature;
        self
    }

    /// Compress logits.
    pub fn compress(&self, logits: &Array) -> Result<CompressedLogits> {
        match &self.method {
            CompressionMethod::None => self.compress_none(logits),
            CompressionMethod::TopK => self.compress_topk(logits),
            CompressionMethod::Int8 => self.compress_int8(logits),
            CompressionMethod::Int4 => self.compress_int4(logits),
            CompressionMethod::Sampled => self.compress_sampled(logits),
        }
    }

    /// Decompress logits.
    pub fn decompress(&self, compressed: &CompressedLogits, vocab_size: usize) -> Result<Array> {
        compressed.validate()?;
        match compressed {
            CompressedLogits::Full { data, shape } => Ok(Array::from_f32_slice(data, shape)),
            CompressedLogits::TopK {
                values,
                indices,
                num_tokens,
                k,
            } => self.decompress_topk(values, indices, *num_tokens, *k, vocab_size),
            CompressedLogits::TopKSparse {
                values,
                indices,
                k,
                shape,
            } => {
                let mut sparse_shape = shape[..shape.len() - 1].to_vec();
                sparse_shape.push(*k as i32);
                let sparse = SparseLogits::new(
                    Array::from_f32_slice(values, &sparse_shape),
                    Array::from_i32_slice_shaped(indices, &sparse_shape),
                    shape[shape.len() - 1] as usize,
                )?;
                Ok(sparse.to_dense())
            }
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
            CompressedLogits::Int8PerToken {
                quant,
                per_token_meta,
                shape,
            } => self.decompress_int8_per_token(quant, per_token_meta, shape),
            CompressedLogits::Int4PerToken {
                packed,
                per_token_meta,
                shape,
            } => self.decompress_int4_per_token(packed, per_token_meta, shape),
            CompressedLogits::Sampled {
                probs,
                indices,
                slots,
                shape,
                temperature,
            } => {
                // Dense logits whose softmax at `temperature` is the empirical
                // distribution: `T · ln(frequency)` where drawn, very negative
                // elsewhere.
                let (num_tokens, vocab) = token_layout(shape)?;
                let mut dense = vec![MASKED_LOGIT; num_tokens * vocab];
                for t in 0..num_tokens {
                    for s in 0..*slots {
                        let p = probs[t * slots + s];
                        if p > 0.0 {
                            dense[t * vocab + indices[t * slots + s] as usize] =
                                temperature * p.ln();
                        }
                    }
                }
                Ok(Array::from_f32_slice(&dense, shape))
            }
        }
    }

    /// Random-sampling KD: `top_k` draws per token from
    /// `softmax(logits / T)`, kept as the distinct tokens and their counts.
    ///
    /// The draws are taken on the device a few rows at a time (each draw costs
    /// a pass over its rows' logits), from MLX's global key, so
    /// `random::seed` makes a cache reproducible.
    fn compress_sampled(&self, logits: &Array) -> Result<CompressedLogits> {
        const ROWS_PER_CHUNK: usize = 64;
        let shape = logits.shape().to_vec();
        let (num_tokens, vocab) = token_layout(&shape)?;
        let draws = self.top_k.max(1);
        let temperature = self.sampling_temperature;
        if !(temperature > 0.0 && temperature.is_finite()) {
            return Err(DistillError::LogitCache(format!(
                "the sampling temperature must be positive and finite, got {temperature}"
            )));
        }

        let flat = logits
            .reshape(&[num_tokens as i32, vocab as i32])
            .as_dtype(Dtype::Float32.as_i32())
            .multiply(&Array::from_f32(1.0 / temperature));
        if ops::item_bool(&flat.isnan().any(None, false)) {
            return Err(DistillError::LogitCache(
                "teacher logits contain NaN (did the teacher forward diverge?)".to_string(),
            ));
        }

        let slots = draws.min(vocab);
        let mut probs = vec![0.0_f32; num_tokens * slots];
        let mut indices = vec![0_i32; num_tokens * slots];
        for start in (0..num_tokens).step_by(ROWS_PER_CHUNK) {
            let rows = ROWS_PER_CHUNK.min(num_tokens - start);
            let chunk = flat.slice(&[start as i32, 0], &[(start + rows) as i32, vocab as i32]);
            let picks: Vec<Array> = (0..draws)
                .map(|_| random::categorical(&chunk, -1, None).reshape(&[rows as i32, 1]))
                .collect();
            let refs: Vec<&Array> = picks.iter().collect();
            let drawn = ops::concatenate_axis(&refs, 1).as_dtype(Dtype::Int32.as_i32());
            let drawn = drawn.as_slice::<i32>();

            for r in 0..rows {
                let mut row = drawn[r * draws..(r + 1) * draws].to_vec();
                row.sort_unstable();
                // Distinct tokens with their counts, most frequent first.
                let mut counts: Vec<(i32, usize)> = Vec::new();
                for id in row {
                    match counts.last_mut() {
                        Some((last, n)) if *last == id => *n += 1,
                        _ => counts.push((id, 1)),
                    }
                }
                counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                for (s, (id, n)) in counts.into_iter().take(slots).enumerate() {
                    probs[(start + r) * slots + s] = n as f32 / draws as f32;
                    indices[(start + r) * slots + s] = id;
                }
            }
        }

        Ok(CompressedLogits::Sampled {
            probs,
            indices,
            slots,
            shape,
            temperature,
        })
    }

    fn compress_none(&self, logits: &Array) -> Result<CompressedLogits> {
        let (data, shape) = host_logits(logits, false)?;
        Ok(CompressedLogits::Full { data, shape })
    }

    /// The `k` largest logits of every token, largest first, selected on the
    /// device (`argpartition` is `O(vocab)` per token) and ordered there, so a
    /// 150k-token vocabulary costs one pass over the logits and a copy of
    /// `k` entries per token to the host.
    fn compress_topk(&self, logits: &Array) -> Result<CompressedLogits> {
        let shape = logits.shape().to_vec();
        let (num_tokens, vocab) = token_layout(&shape)?;
        let k = self.top_k.clamp(1, vocab);

        let flat = logits
            .reshape(&[num_tokens as i32, vocab as i32])
            .as_dtype(Dtype::Float32.as_i32());
        if ops::item_bool(&flat.isnan().any(None, false)) {
            return Err(DistillError::LogitCache(
                "teacher logits contain NaN (did the teacher forward diverge?)".to_string(),
            ));
        }

        let (values, indices) = if k == vocab {
            let all =
                Array::arange(vocab as i32, Dtype::Int32.as_i32()).reshape(&[1, vocab as i32]);
            let ids = ops::broadcast_to(&all, &[num_tokens as i32, vocab as i32]);
            (flat.clone(), ids)
        } else {
            // The k smallest of -logits sit in the first k slots (the k-th
            // smallest at slot k-1).
            let part = ops::argpartition_axis(&flat.negative(), k as i32 - 1, -1);
            let ids = part.slice(&[0, 0], &[num_tokens as i32, k as i32]);
            (flat.take_along_axis(&ids, -1), ids)
        };
        let order = values.negative().argsort(-1);
        let values = values.take_along_axis(&order, -1);
        let indices = indices.take_along_axis(&order, -1);

        Ok(CompressedLogits::TopKSparse {
            values: values.as_slice::<f32>().to_vec(),
            indices: indices
                .as_dtype(Dtype::Int32.as_i32())
                .as_slice::<i32>()
                .to_vec(),
            k,
            shape,
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
        let sparse = SparseLogits::new(
            Array::from_f32_slice(values, &[num_tokens as i32, k as i32]),
            Array::from_i32_slice_shaped(indices, &[num_tokens as i32, k as i32]),
            vocab_size,
        )?;
        Ok(sparse.to_dense())
    }

    fn compress_int8(&self, logits: &Array) -> Result<CompressedLogits> {
        let (data, shape) = host_logits(logits, true)?;
        let (num_tokens, token_size) = token_layout(&shape)?;

        // Per-token quantization: min/max per row so each token's dynamic
        // range is captured independently, and one outlier token cannot
        // squash the rest into a narrow band.
        let mut quant = Vec::with_capacity(data.len());
        let mut per_token_meta = Vec::with_capacity(num_tokens * 2);
        for row in data.chunks(token_size) {
            let (scale, zero_point) = quantize_row(row, 255.0, |q| quant.push(q));
            per_token_meta.push(scale);
            per_token_meta.push(zero_point);
        }
        Ok(CompressedLogits::Int8PerToken {
            quant,
            per_token_meta,
            shape,
        })
    }

    /// Decompress the Int8PerToken variant. The (scale, zp) metadata travels
    /// alongside the quantized payload in a typed `Vec<f32>`, so there's no
    /// sentinel-overload risk for inputs containing real NaNs.
    fn decompress_int8_per_token(
        &self,
        quant: &[u8],
        per_token_meta: &[f32],
        shape: &[i32],
    ) -> Result<Array> {
        let (num_tokens, token_size) = token_layout(shape)?;
        let mut dequantized = Vec::with_capacity(num_tokens * token_size);
        for t in 0..num_tokens {
            let sc = per_token_meta[t * 2];
            let zp = per_token_meta[t * 2 + 1];
            for &q in &quant[t * token_size..(t + 1) * token_size] {
                dequantized.push((q as f32) * sc + zp);
            }
        }
        Ok(Array::from_f32_slice(&dequantized, shape))
    }

    fn decompress_int8(
        &self,
        data: &[u8],
        scale: f32,
        zero_point: f32,
        shape: &[i32],
    ) -> Result<Array> {
        if scale.is_nan() {
            // Per-token format: parse embedded (scale, zp) pairs from the tail.
            let (num_tokens, token_size) = token_layout(shape)?;
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

            Ok(Array::from_f32_slice(&dequantized, shape))
        } else {
            // Legacy global-scale format.
            let dequantized: Vec<f32> = data
                .iter()
                .map(|&q| (q as f32) * scale + zero_point)
                .collect();
            Ok(Array::from_f32_slice(&dequantized, shape))
        }
    }

    fn compress_int4(&self, logits: &Array) -> Result<CompressedLogits> {
        let (data, shape) = host_logits(logits, true)?;
        let (num_tokens, token_size) = token_layout(&shape)?;

        // One (scale, zero_point) per token, as for int8: sixteen levels are
        // few enough that sharing them across tokens of very different
        // ranges costs most of them.
        let mut codes = Vec::with_capacity(data.len());
        let mut per_token_meta = Vec::with_capacity(num_tokens * 2);
        for row in data.chunks(token_size) {
            let (scale, zero_point) = quantize_row(row, 15.0, |q| codes.push(q));
            per_token_meta.push(scale);
            per_token_meta.push(zero_point);
        }
        let packed = codes
            .chunks(2)
            .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(0) << 4))
            .collect();
        Ok(CompressedLogits::Int4PerToken {
            packed,
            per_token_meta,
            shape,
        })
    }

    fn decompress_int4_per_token(
        &self,
        packed: &[u8],
        per_token_meta: &[f32],
        shape: &[i32],
    ) -> Result<Array> {
        let (num_tokens, token_size) = token_layout(shape)?;
        let n = num_tokens * token_size;
        let mut dequantized = Vec::with_capacity(n);
        for i in 0..n {
            let byte = packed[i / 2];
            let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let t = i / token_size;
            dequantized.push(code as f32 * per_token_meta[t * 2] + per_token_meta[t * 2 + 1]);
        }
        Ok(Array::from_f32_slice(&dequantized, shape))
    }

    /// Legacy int4 payloads: a single scale for the whole tensor.
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
        Ok(Array::from_f32_slice(&dequantized, shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_topk_compression() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);

        let compressor = LogitCompressor::new(CompressionMethod::TopK, 2);
        let compressed = compressor.compress(&logits).unwrap();

        match &compressed {
            CompressedLogits::TopKSparse {
                values,
                indices,
                k,
                shape,
            } => {
                assert_eq!(*k, 2);
                assert_eq!(shape, &[2, 4]);
                assert_eq!(values.len(), 4); // 2 tokens * 2 top-k
                assert_eq!(indices.len(), 4);

                // Largest first: token 0's top-2 are indices 3 and 2.
                assert_eq!(&indices[..2], &[3, 2]);
                assert_eq!(&values[..2], &[4.0, 3.0]);
                assert_eq!(&indices[2..], &[3, 2]);
                assert_eq!(&values[2..], &[8.0, 7.0]);
            }
            _ => panic!("Wrong compression type"),
        }
    }

    #[test]
    #[serial]
    fn test_topk_roundtrip() {
        let logits =
            Array::from_f32_slice(&[1.0_f32, 2.0, 10.0, 4.0, 5.0, 6.0, 20.0, 8.0], &[2, 4]);

        let compressor = LogitCompressor::new(CompressionMethod::TopK, 2);
        let compressed = compressor.compress(&logits).unwrap();
        let decompressed = compressor.decompress(&compressed, 4).unwrap();

        let data: Vec<f32> = decompressed.clone().to_f32_vec(8).unwrap();

        // Top values should be preserved, others should be very negative
        assert!((data[2] - 10.0).abs() < 1e-5); // Token 0, index 2
        assert!(data[0] < -1e5); // Token 0, index 0 (not in top-2)
        assert!((data[6] - 20.0).abs() < 1e-5); // Token 1, index 2
    }

    #[test]
    #[serial]
    fn test_int8_roundtrip() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);

        let compressor = LogitCompressor::new(CompressionMethod::Int8, 128);
        let compressed = compressor.compress(&logits).unwrap();
        let decompressed = compressor.decompress(&compressed, 4).unwrap();

        let original: Vec<f32> = logits.clone().to_f32_vec(8).unwrap();
        let recovered: Vec<f32> = decompressed.clone().to_f32_vec(8).unwrap();

        // Should be close (within quantization error)
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.1, "Int8 roundtrip error: {} vs {}", o, r);
        }
    }

    /// New Int8 path always emits the per-token variant — no NaN sentinel.
    /// This is the regression test for Phase 1.3: prior to the fix, a NaN
    /// computed at write time (or already in the input) would silently flip
    /// decompress into the wrong code path.
    #[test]
    #[serial]
    fn int8_compression_emits_per_token_variant() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let compressor = LogitCompressor::new(CompressionMethod::Int8, 128);
        let compressed = compressor.compress(&logits).unwrap();
        match compressed {
            CompressedLogits::Int8PerToken {
                ref quant,
                ref per_token_meta,
                ref shape,
            } => {
                assert_eq!(shape, &vec![2, 4]);
                assert_eq!(quant.len(), 8);
                assert_eq!(per_token_meta.len(), 4); // 2 tokens * (scale, zp)
                assert!(
                    per_token_meta.iter().all(|f| f.is_finite()),
                    "per-token metadata must be finite"
                );
            }
            other => panic!("expected Int8PerToken, got {:?}", other),
        }
    }

    /// Legacy Int8 with a NaN scale must round-trip via the per-token tail
    /// of `data` (the pre-Phase-1.3 sentinel path) so existing on-disk caches
    /// still decode bit-exact.
    #[test]
    #[serial]
    fn legacy_int8_nan_sentinel_still_decodes() {
        // Build a synthetic "old format" cache by hand for [2, 4] logits.
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape: Vec<i32> = vec![2, 4];
        let token_size = 4_usize;
        let num_tokens = 2_usize;

        let mut quant = Vec::with_capacity(num_tokens * token_size);
        let mut meta = Vec::with_capacity(num_tokens * 2);
        for t in 0..num_tokens {
            let chunk = &values[t * token_size..(t + 1) * token_size];
            let lo = chunk.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = (hi - lo) / 255.0;
            for &v in chunk {
                quant.push((((v - lo) / scale).round() as i32).clamp(0, 255) as u8);
            }
            meta.push(scale);
            meta.push(lo);
        }
        let mut data = quant;
        for &f in &meta {
            data.extend_from_slice(&f.to_le_bytes());
        }

        let compressed = CompressedLogits::Int8 {
            data,
            scale: f32::NAN,
            zero_point: 0.0,
            shape,
        };
        let compressor = LogitCompressor::new(CompressionMethod::Int8, 128);
        let arr = compressor.decompress(&compressed, 4).unwrap();
        let got: Vec<f32> = arr.clone().to_f32_vec(8).unwrap();
        for (a, b) in values.iter().zip(got.iter()) {
            assert!((a - b).abs() < 0.05, "{} vs {}", a, b);
        }
    }

    /// Caches written before the per-token int4 and shaped top-k variants
    /// existed still decode.
    #[test]
    #[serial]
    fn legacy_top_k_and_int4_still_decode() {
        let compressor = LogitCompressor::new(CompressionMethod::TopK, 2);
        let legacy_topk = CompressedLogits::TopK {
            values: vec![4.0, 3.0, 8.0, 7.0],
            indices: vec![3, 2, 3, 2],
            num_tokens: 2,
            k: 2,
        };
        let dense = compressor.decompress(&legacy_topk, 4).unwrap();
        assert_eq!(dense.shape(), &[2, 4]);
        let v: Vec<f32> = dense.clone().to_f32_vec(8).unwrap();
        assert_eq!((v[3], v[2], v[7], v[6]), (4.0, 3.0, 8.0, 7.0));

        // Two values 0.0 and 15.0 quantized with one global scale of 1.0.
        let legacy_int4 = CompressedLogits::Int4 {
            data: vec![0xF0],
            scale: 1.0,
            zero_point: 0.0,
            shape: vec![1, 2],
        };
        let v: Vec<f32> = compressor
            .decompress(&legacy_int4, 2)
            .unwrap()
            .clone()
            .to_f32_vec(2)
            .unwrap();
        assert_eq!(v, vec![0.0, 15.0]);
    }

    #[test]
    #[serial]
    fn test_int4_roundtrip() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);

        let compressor = LogitCompressor::new(CompressionMethod::Int4, 128);
        let compressed = compressor.compress(&logits).unwrap();
        let decompressed = compressor.decompress(&compressed, 4).unwrap();

        let original: Vec<f32> = logits.clone().to_f32_vec(8).unwrap();
        let recovered: Vec<f32> = decompressed.clone().to_f32_vec(8).unwrap();

        // Int4 has lower precision
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1.0, "Int4 roundtrip error: {} vs {}", o, r);
        }
    }

    #[test]
    #[serial]
    fn test_none_compression() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);

        let compressor = LogitCompressor::new(CompressionMethod::None, 128);
        let compressed = compressor.compress(&logits).unwrap();
        let decompressed = compressor.decompress(&compressed, 2).unwrap();

        let original: Vec<f32> = logits.clone().to_f32_vec(4).unwrap();
        let recovered: Vec<f32> = decompressed.clone().to_f32_vec(4).unwrap();

        assert_eq!(original, recovered);
    }
}
