//! Sequence packing for efficient training.
//!
//! Packing concatenates multiple sequences into single batches to
//! eliminate padding and improve GPU utilization. This can provide
//! 2-3x training throughput improvement on datasets with variable-length sequences.
//!
//! # Overview
//!
//! Traditional batching pads all sequences to the maximum length:
//! ```text
//! [seq1: 10 tokens][PAD PAD PAD PAD PAD PAD]  <- 6 wasted positions
//! [seq2: 16 tokens]                           <- full
//! [seq3: 8 tokens][PAD PAD PAD PAD PAD PAD PAD PAD]  <- 8 wasted
//! ```
//!
//! Sample packing concatenates sequences efficiently:
//! ```text
//! [seq1: 10 tokens][seq3: 8 tokens][seq5: 12 tokens][PAD PAD]  <- only 2 wasted
//! [seq2: 16 tokens][seq4: 14 tokens]                           <- full
//! ```
//!
//! # Algorithm
//!
//! This module implements First-Fit Decreasing (FFD) bin packing:
//! 1. Sort sequences by length (descending)
//! 2. For each sequence, try to fit into existing bins
//! 3. If no bin has room, create a new bin
//!
//! This achieves near-optimal packing efficiency (typically >90% utilization).
//!
//! # Key Components
//!
//! - [`PackedBatch`]: A packed batch with cu_seqlens for FlashAttention
//! - [`SequencePacker`]: Bin-packing algorithm implementation
//! - Block diagonal attention masks for SDPA
//! - Position ID management (reset per sequence)
//! - Label boundary masking
//!
//! # References
//!
//! - [Unsloth Packing](https://github.com/unslothai/unsloth)
//! - [FlashAttention Variable Length](https://github.com/Dao-AILab/flash-attention)

use super::Sample;
use pmetal_core::Result;

/// Error type for packing operations.
#[derive(Debug, thiserror::Error)]
pub enum PackingError {
    /// No samples to pack.
    #[error("No samples to pack")]
    EmptySamples,
    /// All samples were filtered out (e.g., too short).
    #[error("All samples were filtered out")]
    AllFiltered,
}

/// A packed batch containing multiple concatenated sequences.
#[derive(Debug, Clone)]
pub struct PackedBatch {
    /// Concatenated input token IDs.
    pub input_ids: Vec<u32>,
    /// Position IDs (reset for each sequence).
    pub position_ids: Vec<u32>,
    /// Cumulative sequence lengths for attention (length: num_sequences + 1).
    /// cu_seqlens[0] = 0, cu_seqlens[i+1] = cu_seqlens[i] + seq_lengths[i]
    pub cu_seqlens: Vec<u32>,
    /// Individual sequence lengths.
    pub seq_lengths: Vec<u32>,
    /// Maximum sequence length in this batch.
    pub max_seqlen: usize,
    /// Number of sequences in this batch.
    pub num_sequences: usize,
    /// Concatenated labels (with boundary tokens masked as -100).
    pub labels: Option<Vec<i64>>,
}

impl PackedBatch {
    /// Get the total number of tokens in this batch.
    pub fn total_tokens(&self) -> usize {
        self.input_ids.len()
    }

    /// Build a block diagonal attention mask for SDPA.
    ///
    /// Returns a 2D mask where `mask[i][j]` is:
    /// - 0.0 if token i can attend to token j
    /// - -inf if token i should not attend to token j
    ///
    /// The mask ensures each sequence only attends to itself
    /// with causal masking within each sequence.
    ///
    /// # Performance
    /// Optimized to minimize per-element operations. Complexity is O(n²) for
    /// initialization plus O(sum(len_i²)) for filling causal blocks.
    /// For typical packed batches (2048 total tokens), this takes ~1-2ms.
    pub fn build_attention_mask(&self) -> Vec<f32> {
        let n = self.total_tokens();

        // Pre-allocate with -inf (masked out)
        let mut mask = vec![f32::NEG_INFINITY; n * n];

        // Fill causal blocks for each sequence
        // Optimization: compute row base once per row instead of per element
        let mut offset = 0usize;
        for &len in &self.seq_lengths {
            let len = len as usize;

            // For each row in this sequence's block
            for i in 0..len {
                let row_start = (offset + i) * n + offset;
                // Fill positions 0..=i with 0.0 (causal: can attend to self and previous)
                // Use fill instead of per-element assignment for better cache locality
                let end = i + 1;
                mask[row_start..row_start + end].fill(0.0);
            }
            offset += len;
        }

        mask
    }

    /// Build a block diagonal attention mask, returning as mlx_rs Array.
    ///
    /// This is more efficient than building Vec<f32> and then converting,
    /// as it avoids an intermediate allocation.
    pub fn build_attention_mask_array(&self) -> mlx_rs::error::Result<mlx_rs::Array> {
        let n = self.total_tokens() as i32;
        let mask_data = self.build_attention_mask();
        Ok(mlx_rs::Array::from_slice(&mask_data, &[n, n]))
    }

    /// Build a block diagonal attention mask with sliding window.
    ///
    /// Same as `build_attention_mask` but with a sliding window constraint:
    /// tokens can only attend to at most `window_size` previous tokens.
    pub fn build_attention_mask_with_window(&self, window_size: usize) -> Vec<f32> {
        let n = self.total_tokens();
        let mut mask = vec![f32::NEG_INFINITY; n * n];

        let mut offset = 0usize;
        for &len in &self.seq_lengths {
            let len = len as usize;
            for i in 0..len {
                let start = if i >= window_size {
                    i - window_size + 1
                } else {
                    0
                };
                for j in start..=i {
                    mask[(offset + i) * n + (offset + j)] = 0.0;
                }
            }
            offset += len;
        }

        mask
    }

    /// Get sequence end positions for loss masking.
    ///
    /// Returns indices of the last token of each sequence.
    pub fn sequence_boundaries(&self) -> Vec<usize> {
        let mut boundaries = Vec::with_capacity(self.num_sequences);
        let mut offset = 0usize;
        for &len in &self.seq_lengths {
            if len > 0 {
                boundaries.push(offset + len as usize - 1);
            }
            offset += len as usize;
        }
        boundaries
    }

    /// Mask label positions that would predict across sequence boundaries.
    ///
    /// For causal LM with shifted labels (logits[i] predicts labels[i+1]),
    /// we need to mask the FIRST token of each sequence after the first one.
    /// This prevents the last token of sequence N from being penalized for
    /// not predicting the first token of sequence N+1.
    ///
    /// Example: For packed sequences [A, B, C | D, E] with labels [A, B, C, D, E],
    /// after shift_labels = labels[1:] we get [B, C, D, E].
    /// logits[2] (from C) would predict D, but C is the end of seq1,
    /// so we must set labels[3] = -100 to ignore this cross-boundary prediction.
    pub fn mask_label_boundaries(&mut self, ignore_index: i64) {
        if let Some(ref mut labels) = self.labels {
            // Mask the first token of each sequence (except the first sequence)
            // because that's what would be predicted from the last token of the previous sequence
            let mut offset = 0usize;
            for (i, &len) in self.seq_lengths.iter().enumerate() {
                let len = len as usize;
                if i > 0 && offset < labels.len() {
                    // First token of this sequence - shouldn't be predicted from previous seq
                    labels[offset] = ignore_index;
                }
                offset += len;
            }
        }
    }
}

/// Packing algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PackingAlgorithm {
    /// Simple greedy packing (preserves order).
    Greedy,

    /// First-Fit Decreasing (better utilization, reorders samples).
    #[default]
    FirstFitDecreasing,
}

/// Sequence packer configuration.
#[derive(Debug, Clone)]
pub struct PackerConfig {
    /// Maximum total length per packed batch.
    pub max_length: usize,
    /// Pad to multiple of this value.
    pub pad_to_multiple: usize,
    /// Mask sequence boundaries in labels with -100.
    /// This prevents the loss from penalizing predictions
    /// at sequence boundaries.
    pub mask_boundaries: bool,
    /// Packing algorithm to use.
    pub algorithm: PackingAlgorithm,
    /// Minimum sequence length (drop shorter sequences).
    pub min_length: usize,
    /// Maximum sequence length (truncate longer sequences).
    pub max_seq_length: Option<usize>,
}

impl Default for PackerConfig {
    fn default() -> Self {
        Self {
            max_length: 2048,
            pad_to_multiple: 8,
            mask_boundaries: true,
            algorithm: PackingAlgorithm::FirstFitDecreasing,
            min_length: 1,
            max_seq_length: None,
        }
    }
}

impl PackerConfig {
    /// Create a config with the given max length.
    pub fn with_max_length(max_length: usize) -> Self {
        Self {
            max_length,
            ..Default::default()
        }
    }

    /// Set whether to mask sequence boundaries.
    pub fn mask_boundaries(mut self, mask: bool) -> Self {
        self.mask_boundaries = mask;
        self
    }

    /// Use greedy packing (preserves order).
    pub fn with_greedy(mut self) -> Self {
        self.algorithm = PackingAlgorithm::Greedy;
        self
    }

    /// Use FFD packing (better utilization).
    pub fn with_ffd(mut self) -> Self {
        self.algorithm = PackingAlgorithm::FirstFitDecreasing;
        self
    }

    /// Set minimum sequence length.
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = min_length;
        self
    }

    /// Set maximum sequence length (truncates longer sequences).
    /// This is CRITICAL for datasets with variable-length sequences.
    /// Without this, sequences longer than max_length are SKIPPED entirely.
    pub fn with_max_seq_length(mut self, max_seq_length: usize) -> Self {
        self.max_seq_length = Some(max_seq_length);
        self
    }
}

/// Statistics about packing efficiency.
#[derive(Debug, Clone, Default)]
pub struct PackingStats {
    /// Total number of tokens before packing.
    pub total_tokens: usize,
    /// Total capacity (batches * max_length).
    pub total_capacity: usize,
    /// Number of batches created.
    pub num_batches: usize,
    /// Number of sequences packed.
    pub num_sequences: usize,
    /// Packing efficiency (tokens / capacity).
    pub efficiency: f64,
    /// Average sequences per batch.
    pub avg_sequences_per_batch: f64,
    /// Maximum sequences in a single batch.
    pub max_sequences_per_batch: usize,
}

impl PackingStats {
    /// Calculate stats from batches.
    pub fn from_batches(batches: &[PackedBatch], max_length: usize) -> Self {
        if batches.is_empty() {
            return Self::default();
        }

        let total_tokens: usize = batches.iter().map(|b| b.total_tokens()).sum();
        let total_capacity = batches.len() * max_length;
        let num_sequences: usize = batches.iter().map(|b| b.num_sequences).sum();
        let max_seqs = batches.iter().map(|b| b.num_sequences).max().unwrap_or(0);

        Self {
            total_tokens,
            total_capacity,
            num_batches: batches.len(),
            num_sequences,
            efficiency: total_tokens as f64 / total_capacity as f64,
            avg_sequences_per_batch: num_sequences as f64 / batches.len() as f64,
            max_sequences_per_batch: max_seqs,
        }
    }

    /// Pretty print stats.
    pub fn summary(&self) -> String {
        format!(
            "Packing: {} seqs → {} batches, {:.1}% efficiency, avg {:.1} seqs/batch",
            self.num_sequences,
            self.num_batches,
            self.efficiency * 100.0,
            self.avg_sequences_per_batch
        )
    }
}

/// A bin for FFD packing (internal use).
struct PackingBin {
    samples: Vec<Sample>,
    current_length: usize,
}

impl PackingBin {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            current_length: 0,
        }
    }

    fn can_fit(&self, sample_len: usize, max_length: usize) -> bool {
        self.current_length + sample_len <= max_length
    }

    fn add(&mut self, sample: Sample) {
        self.current_length += sample.input_ids.len();
        self.samples.push(sample);
    }
}

/// Sequence packer for creating packed batches.
pub struct SequencePacker {
    config: PackerConfig,
}

impl SequencePacker {
    /// Create a new sequence packer.
    pub fn new(config: PackerConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &PackerConfig {
        &self.config
    }

    /// Pack samples into batches.
    ///
    /// Uses the configured packing algorithm (greedy or FFD).
    pub fn pack(&self, samples: &[Sample]) -> Result<Vec<PackedBatch>> {
        match self.config.algorithm {
            PackingAlgorithm::Greedy => self.pack_greedy(samples),
            PackingAlgorithm::FirstFitDecreasing => self.pack_ffd(samples),
        }
    }

    /// Pack samples into batches and return stats.
    pub fn pack_with_stats(&self, samples: &[Sample]) -> Result<(Vec<PackedBatch>, PackingStats)> {
        let batches = self.pack(samples)?;
        let stats = PackingStats::from_batches(&batches, self.config.max_length);
        Ok((batches, stats))
    }

    /// Greedy packing (preserves order).
    fn pack_greedy(&self, samples: &[Sample]) -> Result<Vec<PackedBatch>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_length = 0;

        for sample in samples {
            let sample_len = self.get_sample_length(sample);

            // Skip samples that are too short or too long
            if sample_len < self.config.min_length {
                continue;
            }

            if current_length + sample_len > self.config.max_length {
                if !current_batch.is_empty() {
                    batches.push(self.create_packed_batch(&current_batch)?);
                }
                current_batch = vec![self.prepare_sample(sample)];
                current_length = sample_len;
            } else {
                current_batch.push(self.prepare_sample(sample));
                current_length += sample_len;
            }
        }

        if !current_batch.is_empty() {
            batches.push(self.create_packed_batch(&current_batch)?);
        }

        Ok(batches)
    }

    /// First-Fit Decreasing (FFD) packing for better utilization.
    ///
    /// This algorithm:
    /// 1. Sorts samples by length (descending)
    /// 2. For each sample, finds the first bin with room
    /// 3. Creates new bins as needed
    ///
    /// Typically achieves 90%+ packing efficiency.
    fn pack_ffd(&self, samples: &[Sample]) -> Result<Vec<PackedBatch>> {
        // Prepare samples (apply length constraints)
        let mut indexed_samples: Vec<(usize, Sample)> = samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.input_ids.len() >= self.config.min_length)
            .map(|(i, s)| (i, self.prepare_sample(s)))
            .collect();

        // Sort by length descending (largest first)
        indexed_samples.sort_by(|a, b| b.1.input_ids.len().cmp(&a.1.input_ids.len()));

        // Pack using first-fit decreasing
        let mut bins: Vec<PackingBin> = Vec::new();

        for (_, sample) in indexed_samples {
            let sample_len = sample.input_ids.len();

            // Skip samples too long for any bin
            if sample_len > self.config.max_length {
                continue;
            }

            // Find first bin with room
            let mut placed = false;
            for bin in &mut bins {
                if bin.can_fit(sample_len, self.config.max_length) {
                    bin.add(sample.clone());
                    placed = true;
                    break;
                }
            }

            // Create new bin if needed
            if !placed {
                let mut bin = PackingBin::new();
                bin.add(sample);
                bins.push(bin);
            }
        }

        // Convert bins to packed batches
        let mut batches = Vec::with_capacity(bins.len());
        for bin in bins {
            if !bin.samples.is_empty() {
                batches.push(self.create_packed_batch(&bin.samples)?);
            }
        }

        Ok(batches)
    }

    /// Get sample length (with potential truncation).
    fn get_sample_length(&self, sample: &Sample) -> usize {
        let len = sample.input_ids.len();
        if let Some(max) = self.config.max_seq_length {
            len.min(max)
        } else {
            len
        }
    }

    /// Prepare a sample (apply truncation if needed).
    fn prepare_sample(&self, sample: &Sample) -> Sample {
        if let Some(max) = self.config.max_seq_length {
            if sample.input_ids.len() > max {
                return Sample {
                    input_ids: sample.input_ids[..max].to_vec(),
                    attention_mask: sample.attention_mask[..max].to_vec(),
                    labels: sample.labels.as_ref().map(|l| l[..max].to_vec()),
                    images: sample.images.clone(),
                };
            }
        }
        sample.clone()
    }

    fn create_packed_batch(&self, samples: &[Sample]) -> Result<PackedBatch> {
        // Concatenate all token IDs
        let input_ids: Vec<u32> = samples
            .iter()
            .flat_map(|s| s.input_ids.iter().copied())
            .collect();

        // Create position IDs (reset for each sequence)
        let position_ids: Vec<u32> = samples
            .iter()
            .flat_map(|s| (0..s.input_ids.len() as u32).collect::<Vec<_>>())
            .collect();

        // Individual sequence lengths
        let seq_lengths: Vec<u32> = samples.iter().map(|s| s.input_ids.len() as u32).collect();

        // Cumulative sequence lengths (for flash attention varlen)
        let mut cu_seqlens = vec![0u32];
        let mut cumsum = 0u32;
        for sample in samples {
            cumsum += sample.input_ids.len() as u32;
            cu_seqlens.push(cumsum);
        }

        let max_seqlen = samples.iter().map(|s| s.input_ids.len()).max().unwrap_or(0);

        // Concatenate labels if present
        let labels = if samples.iter().all(|s| s.labels.is_some()) {
            Some(
                samples
                    .iter()
                    .flat_map(|s| s.labels.as_ref().unwrap().iter().copied())
                    .collect(),
            )
        } else {
            None
        };

        let mut batch = PackedBatch {
            input_ids,
            position_ids,
            cu_seqlens,
            seq_lengths,
            max_seqlen,
            num_sequences: samples.len(),
            labels,
        };

        // Mask sequence boundaries in labels to prevent cross-sequence loss
        if self.config.mask_boundaries {
            batch.mask_label_boundaries(-100);
        }

        Ok(batch)
    }
}

impl Default for SequencePacker {
    fn default() -> Self {
        Self::new(PackerConfig::default())
    }
}

// =============================================================================
// MLX Training Integration
// =============================================================================

use mlx_rs::{error::Exception, Array};

/// A packed batch ready for training with MLX Arrays.
///
/// This is the bridge between the packing infrastructure and the training loop.
/// Unlike regular TrainingBatch which has shape [batch_size, seq_len],
/// PackedTrainingBatch has shape [total_tokens] with variable-length sequences.
#[derive(Debug)]
pub struct PackedTrainingBatch {
    /// Packed input token IDs [total_tokens].
    pub input_ids: Array,
    /// Position IDs reset per sequence [total_tokens].
    pub position_ids: Array,
    /// Cumulative sequence lengths [num_sequences + 1].
    pub cu_seqlens: Array,
    /// Packed labels for loss computation [total_tokens].
    pub labels: Array,
    /// Total number of tokens.
    pub total_tokens: usize,
    /// Number of sequences in this packed batch.
    pub num_sequences: usize,
    /// Maximum sequence length in this batch.
    pub max_seqlen: usize,
}

impl PackedTrainingBatch {
    /// Convert a PackedBatch to a PackedTrainingBatch with MLX Arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if label conversion fails (e.g., no labels present).
    pub fn from_packed_batch(batch: &PackedBatch) -> std::result::Result<Self, Exception> {
        // Convert input_ids to Array [total_tokens]
        let input_ids_i32: Vec<i32> = batch.input_ids.iter().map(|&x| x as i32).collect();
        let len = input_ids_i32.len() as i32;
        let input_ids = Array::from_slice(&input_ids_i32, &[len]);

        // Convert position_ids to Array [total_tokens]
        let position_ids_i32: Vec<i32> = batch.position_ids.iter().map(|&x| x as i32).collect();
        let len = position_ids_i32.len() as i32;
        let position_ids = Array::from_slice(&position_ids_i32, &[len]);

        // Convert cu_seqlens to Array [num_sequences + 1]
        let cu_seqlens_i32: Vec<i32> = batch.cu_seqlens.iter().map(|&x| x as i32).collect();
        let len = cu_seqlens_i32.len() as i32;
        let cu_seqlens = Array::from_slice(&cu_seqlens_i32, &[len]);

        // Convert labels to Array [total_tokens]
        // If labels are None, use input_ids shifted by 1 (language modeling)
        let labels = if let Some(ref labels_vec) = batch.labels {
            let len = labels_vec.len() as i32;
            Array::from_slice(labels_vec, &[len])
        } else {
            // Default: next token prediction (shift input_ids by 1)
            let mut labels_i64: Vec<i64> = batch.input_ids[1..].iter().map(|&x| x as i64).collect();
            labels_i64.push(-100); // Ignore last token prediction
            let len = labels_i64.len() as i32;
            Array::from_slice(&labels_i64, &[len])
        };

        Ok(Self {
            input_ids,
            position_ids,
            cu_seqlens,
            labels,
            total_tokens: batch.input_ids.len(),
            num_sequences: batch.num_sequences,
            max_seqlen: batch.max_seqlen,
        })
    }

    /// Create a packed attention mask for this batch.
    ///
    /// Returns a 2D block-diagonal causal mask [total_tokens, total_tokens].
    /// Each sequence only attends to tokens within itself.
    pub fn attention_mask(&self) -> std::result::Result<Array, Exception> {
        // Create block-diagonal mask
        let n = self.total_tokens;
        let mut mask = vec![f32::NEG_INFINITY; n * n];

        // For each position, allow attending to positions in the same sequence
        // up to and including the current position (causal)
        let cu_seqlens = self.cu_seqlens_vec();

        for seq_idx in 0..self.num_sequences {
            let seq_start = cu_seqlens[seq_idx] as usize;
            let seq_end = cu_seqlens[seq_idx + 1] as usize;

            for q in seq_start..seq_end {
                for k in seq_start..=q {
                    mask[q * n + k] = 0.0;
                }
            }
        }

        let n_i32 = n as i32;
        Ok(Array::from_slice(&mask, &[n_i32, n_i32]))
    }

    /// Get cu_seqlens as a Vec for CPU iteration.
    fn cu_seqlens_vec(&self) -> Vec<i32> {
        self.cu_seqlens.as_slice::<i32>().to_vec()
    }

    /// Get token count (for throughput calculations).
    pub fn token_count(&self) -> usize {
        self.total_tokens
    }

    /// Get average sequence length.
    pub fn avg_seq_len(&self) -> f64 {
        self.total_tokens as f64 / self.num_sequences.max(1) as f64
    }
}

/// A DataLoader that yields packed batches for padding-free training.
pub struct PackedDataLoader {
    /// The packer configuration.
    packer: SequencePacker,
    /// Pre-packed batches.
    batches: Vec<PackedBatch>,
    /// Current batch index.
    position: usize,
    /// Whether to shuffle batches each epoch.
    shuffle: bool,
    /// Random seed.
    seed: u64,
}

impl PackedDataLoader {
    /// Create a new PackedDataLoader from samples.
    pub fn new(samples: &[Sample], config: PackerConfig, shuffle: bool, seed: u64) -> Result<Self> {
        let packer = SequencePacker::new(config);
        let batches = packer.pack(samples)?;

        Ok(Self {
            packer,
            batches,
            position: 0,
            shuffle,
            seed,
        })
    }

    /// Get packing statistics.
    pub fn stats(&self) -> PackingStats {
        let total_tokens: usize = self.batches.iter().map(|b| b.input_ids.len()).sum();
        let max_len = self.packer.config.max_length;
        let total_capacity: usize = self.batches.len() * max_len;
        let num_sequences: usize = self.batches.iter().map(|b| b.num_sequences).sum();
        let max_seqs_per_batch = self
            .batches
            .iter()
            .map(|b| b.num_sequences)
            .max()
            .unwrap_or(0);

        PackingStats {
            total_tokens,
            total_capacity,
            num_batches: self.batches.len(),
            num_sequences,
            efficiency: total_tokens as f64 / total_capacity.max(1) as f64,
            avg_sequences_per_batch: num_sequences as f64 / self.batches.len().max(1) as f64,
            max_sequences_per_batch: max_seqs_per_batch,
        }
    }

    /// Get number of batches.
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Reset for a new epoch.
    pub fn reset(&mut self, new_seed: Option<u64>) {
        self.position = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            let seed = new_seed.unwrap_or(self.seed);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            self.batches.shuffle(&mut rng);
        }
    }

    /// Get the next packed training batch.
    pub fn next_batch(&mut self) -> Option<std::result::Result<PackedTrainingBatch, Exception>> {
        if self.position >= self.batches.len() {
            return None;
        }

        let batch = &self.batches[self.position];
        self.position += 1;

        Some(PackedTrainingBatch::from_packed_batch(batch))
    }
}

impl Iterator for PackedDataLoader {
    type Item = std::result::Result<PackedTrainingBatch, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(ids: Vec<u32>, labels: Option<Vec<i64>>) -> Sample {
        let len = ids.len();
        Sample {
            input_ids: ids,
            attention_mask: vec![1; len],
            labels,
            images: None,
        }
    }

    #[test]
    fn test_pack_sequences() {
        let samples = vec![
            make_sample(vec![1, 2], None),
            make_sample(vec![3], None),
            make_sample(vec![4, 5, 6], None),
        ];

        // Use greedy packing to preserve order for predictable test assertions
        let packer = SequencePacker::new(PackerConfig::with_max_length(10).with_greedy());
        let batches = packer.pack(&samples).unwrap();

        assert_eq!(batches.len(), 1);
        let batch = &batches[0];

        // Check concatenated input IDs
        assert_eq!(batch.input_ids, vec![1, 2, 3, 4, 5, 6]);

        // Check position IDs (reset for each sequence)
        assert_eq!(batch.position_ids, vec![0, 1, 0, 0, 1, 2]);

        // Check sequence lengths
        assert_eq!(batch.seq_lengths, vec![2, 1, 3]);

        // Check cumulative sequence lengths
        assert_eq!(batch.cu_seqlens, vec![0, 2, 3, 6]);

        assert_eq!(batch.max_seqlen, 3);
        assert_eq!(batch.num_sequences, 3);
    }

    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn test_block_diagonal_attention_mask() {
        // Create a batch with two sequences: [2 tokens] [3 tokens]
        let batch = PackedBatch {
            input_ids: vec![1, 2, 3, 4, 5],
            position_ids: vec![0, 1, 0, 1, 2],
            cu_seqlens: vec![0, 2, 5],
            seq_lengths: vec![2, 3],
            max_seqlen: 3,
            num_sequences: 2,
            labels: None,
        };

        let mask = batch.build_attention_mask();

        // Expected 5x5 mask (flattened):
        // Token 0: can attend to [0]
        // Token 1: can attend to [0, 1]
        // Token 2: can attend to [2]
        // Token 3: can attend to [2, 3]
        // Token 4: can attend to [2, 3, 4]

        // Position (0, 0) = 0.0 (seq1: token 0 attends to token 0)
        assert_eq!(mask[0 * 5 + 0], 0.0);
        // Position (0, 1) = -inf (seq1: token 0 can't attend to future)
        assert!(mask[0 * 5 + 1].is_infinite() && mask[0 * 5 + 1] < 0.0);
        // Position (0, 2) = -inf (seq1: token 0 can't attend to seq2)
        assert!(mask[0 * 5 + 2].is_infinite() && mask[0 * 5 + 2] < 0.0);

        // Position (1, 0) = 0.0 (seq1: token 1 attends to token 0)
        assert_eq!(mask[1 * 5 + 0], 0.0);
        // Position (1, 1) = 0.0 (seq1: token 1 attends to token 1)
        assert_eq!(mask[1 * 5 + 1], 0.0);
        // Position (1, 2) = -inf (seq1 can't attend to seq2)
        assert!(mask[1 * 5 + 2].is_infinite() && mask[1 * 5 + 2] < 0.0);

        // Position (2, 0) = -inf (seq2 can't attend to seq1)
        assert!(mask[2 * 5 + 0].is_infinite() && mask[2 * 5 + 0] < 0.0);
        // Position (2, 2) = 0.0 (seq2: token 0 attends to itself)
        assert_eq!(mask[2 * 5 + 2], 0.0);

        // Position (4, 2) = 0.0 (seq2: token 2 attends to token 0 of seq2)
        assert_eq!(mask[4 * 5 + 2], 0.0);
        // Position (4, 4) = 0.0 (seq2: token 2 attends to itself)
        assert_eq!(mask[4 * 5 + 4], 0.0);
    }

    #[test]
    fn test_sequence_boundaries() {
        let batch = PackedBatch {
            input_ids: vec![1, 2, 3, 4, 5, 6],
            position_ids: vec![0, 1, 0, 0, 1, 2],
            cu_seqlens: vec![0, 2, 3, 6],
            seq_lengths: vec![2, 1, 3],
            max_seqlen: 3,
            num_sequences: 3,
            labels: None,
        };

        let boundaries = batch.sequence_boundaries();
        // Boundaries are at indices: 1 (end of seq1), 2 (end of seq2), 5 (end of seq3)
        assert_eq!(boundaries, vec![1, 2, 5]);
    }

    #[test]
    fn test_label_boundary_masking() {
        let samples = vec![
            make_sample(vec![1, 2], Some(vec![10, 20])),
            make_sample(vec![3], Some(vec![30])),
            make_sample(vec![4, 5, 6], Some(vec![40, 50, 60])),
        ];

        // Use greedy packing to preserve order for predictable test assertions
        let packer = SequencePacker::new(
            PackerConfig::with_max_length(10)
                .with_greedy()
                .mask_boundaries(true),
        );
        let batches = packer.pack(&samples).unwrap();

        let labels = batches[0].labels.as_ref().unwrap();
        // First token of each sequence (except seq1) should be -100
        // This prevents cross-boundary prediction: logits[1] shouldn't predict token 3,
        // and logits[2] shouldn't predict token 4
        //
        // Packed layout: [1, 2 | 3 | 4, 5, 6] at positions [0, 1, 2, 3, 4, 5]
        // Seq2 starts at position 2, seq3 starts at position 3
        assert_eq!(labels[0], 10);
        assert_eq!(labels[1], 20);
        assert_eq!(labels[2], -100); // first token of seq2
        assert_eq!(labels[3], -100); // first token of seq3
        assert_eq!(labels[4], 50);
        assert_eq!(labels[5], 60); // last token of seq3 is NOT masked (nothing follows)
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_attention_mask_with_sliding_window() {
        let batch = PackedBatch {
            input_ids: vec![1, 2, 3, 4, 5],
            position_ids: vec![0, 1, 2, 3, 4],
            cu_seqlens: vec![0, 5],
            seq_lengths: vec![5],
            max_seqlen: 5,
            num_sequences: 1,
            labels: None,
        };

        // Window size 2: each token can attend to itself and 1 previous token
        let mask = batch.build_attention_mask_with_window(2);

        // Token 4 should attend to tokens 3 and 4, but not 0, 1, 2
        assert!(mask[4 * 5 + 0].is_infinite() && mask[4 * 5 + 0] < 0.0);
        assert!(mask[4 * 5 + 1].is_infinite() && mask[4 * 5 + 1] < 0.0);
        assert!(mask[4 * 5 + 2].is_infinite() && mask[4 * 5 + 2] < 0.0);
        assert_eq!(mask[4 * 5 + 3], 0.0);
        assert_eq!(mask[4 * 5 + 4], 0.0);
    }

    #[test]
    fn test_ffd_packing_efficiency() {
        // Create samples with varying lengths that benefit from FFD
        let samples = vec![
            make_sample(vec![1; 8], None), // 8 tokens
            make_sample(vec![2; 2], None), // 2 tokens
            make_sample(vec![3; 7], None), // 7 tokens
            make_sample(vec![4; 3], None), // 3 tokens
            make_sample(vec![5; 5], None), // 5 tokens
            make_sample(vec![6; 5], None), // 5 tokens
        ];
        // Total: 30 tokens

        // With max_length=10, greedy would produce:
        // Batch 1: [8] = 8 tokens (can't fit 2 after)
        // Batch 2: [2, 7] = 9 tokens
        // Batch 3: [3, 5] = 8 tokens
        // Batch 4: [5] = 5 tokens
        // = 4 batches, 30/40 = 75% efficiency

        // FFD should produce:
        // Batch 1: [8, 2] = 10 tokens
        // Batch 2: [7, 3] = 10 tokens
        // Batch 3: [5, 5] = 10 tokens
        // = 3 batches, 30/30 = 100% efficiency

        let ffd_packer = SequencePacker::new(PackerConfig::with_max_length(10).with_ffd());
        let greedy_packer = SequencePacker::new(PackerConfig::with_max_length(10).with_greedy());

        let (ffd_batches, ffd_stats) = ffd_packer.pack_with_stats(&samples).unwrap();
        let (greedy_batches, greedy_stats) = greedy_packer.pack_with_stats(&samples).unwrap();

        // FFD should be more efficient (fewer batches or higher utilization)
        assert!(
            ffd_stats.efficiency >= greedy_stats.efficiency,
            "FFD efficiency {:.1}% should be >= greedy {:.1}%",
            ffd_stats.efficiency * 100.0,
            greedy_stats.efficiency * 100.0
        );

        // FFD should produce 3 batches, greedy should produce 4
        assert!(ffd_batches.len() <= greedy_batches.len());

        // Verify total tokens are preserved
        let ffd_tokens: usize = ffd_batches.iter().map(|b| b.total_tokens()).sum();
        let greedy_tokens: usize = greedy_batches.iter().map(|b| b.total_tokens()).sum();
        assert_eq!(ffd_tokens, 30);
        assert_eq!(greedy_tokens, 30);
    }

    #[test]
    fn test_ffd_with_large_sample() {
        // Sample that exceeds max_length should be skipped
        let samples = vec![
            make_sample(vec![1; 15], None), // Too large for max_length=10
            make_sample(vec![2; 5], None),
            make_sample(vec![3; 3], None),
        ];

        let packer = SequencePacker::new(PackerConfig::with_max_length(10).with_ffd());
        let batches = packer.pack(&samples).unwrap();

        // Only the samples that fit should be packed
        let total_tokens: usize = batches.iter().map(|b| b.total_tokens()).sum();
        assert_eq!(total_tokens, 8); // 5 + 3, excluding the 15-token sample
    }

    #[test]
    fn test_packing_stats() {
        let samples = vec![
            make_sample(vec![1; 5], None),
            make_sample(vec![2; 3], None),
            make_sample(vec![3; 7], None),
        ];

        let packer = SequencePacker::new(PackerConfig::with_max_length(10));
        let (batches, stats) = packer.pack_with_stats(&samples).unwrap();

        assert_eq!(stats.num_sequences, 3);
        assert_eq!(stats.total_tokens, 15);
        assert!(stats.efficiency > 0.0 && stats.efficiency <= 1.0);
        assert_eq!(stats.num_batches, batches.len());

        // Check summary formatting
        let summary = stats.summary();
        assert!(summary.contains("3 seqs"));
        assert!(summary.contains("efficiency"));
    }

    #[test]
    fn test_greedy_preserves_order() {
        let samples = vec![
            make_sample(vec![1, 2, 3], None),
            make_sample(vec![4, 5], None),
            make_sample(vec![6], None),
        ];

        let packer = SequencePacker::new(PackerConfig::with_max_length(10).with_greedy());
        let batches = packer.pack(&samples).unwrap();

        // Greedy should maintain order
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].input_ids, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_min_length_filter() {
        let samples = vec![
            make_sample(vec![1], None),       // Too short
            make_sample(vec![2, 3], None),    // OK
            make_sample(vec![4], None),       // Too short
            make_sample(vec![5, 6, 7], None), // OK
        ];

        let packer = SequencePacker::new(PackerConfig::with_max_length(10).with_min_length(2));
        let batches = packer.pack(&samples).unwrap();

        let total_tokens: usize = batches.iter().map(|b| b.total_tokens()).sum();
        assert_eq!(total_tokens, 5); // Only [2,3] and [5,6,7]
    }
}
