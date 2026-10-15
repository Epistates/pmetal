//! Sequence packing for efficient training.
//!
//! Packing concatenates several short documents into one row so a batch carries
//! no padding to speak of. On a dataset of variable-length samples that is a
//! 2-3x throughput gain over padding every sample to the longest.
//!
//! ```text
//! padded:  [doc1: 10 tokens][PAD x6]   [doc2: 16 tokens]   [doc3: 8][PAD x8]
//! packed:  [doc1: 10][doc3: 8][PAD x6] [doc2: 16]
//! ```
//!
//! # What a packed row guarantees
//!
//! A packed row trains exactly what the same documents would train unpacked:
//!
//! - **Attention** is block-diagonal and causal: a token sees its own document
//!   and nothing else ([`PackedBatch::build_attention_mask`],
//!   [`PackedTrainingBatch::attention_mask`]).
//! - **Positions** restart at zero in every document ([`PackedBatch::position_ids`]).
//!   RoPE is relative, so a uniform shift inside a masked block would change
//!   nothing; the restart matters to the models that read absolute positions.
//! - **Labels** are unshifted, as everywhere in this crate: the loss compares
//!   the logits at `t` with the label at `t + 1`. The first token of every
//!   document after the first has its label masked, which is the one place the
//!   shift would otherwise score a document's last token against the next
//!   document's first.
//! - **Padding** (to [`PackerConfig::pad_to_multiple`]) is a block of its own:
//!   its tokens attend to each other only, carry label `-100`, and so neither
//!   see nor are seen by a document, and no attention row is ever empty (an
//!   all-`-inf` row is a NaN).
//!
//! # Algorithms
//!
//! [`PackingAlgorithm::FirstFitDecreasing`] (the default) and
//! [`PackingAlgorithm::BestFitDecreasing`] sort documents longest first and
//! place each in the first (or tightest) row it fits, in `O(n log n)`.
//! [`PackingAlgorithm::Greedy`] keeps the dataset's order and starts a new row
//! when the next document does not fit.
//!
//! A document longer than a row is truncated to the row; none is dropped.
//!
//! # References
//!
//! - [FlashAttention variable length](https://github.com/Dao-AILab/flash-attention)

use std::collections::BTreeSet;

use pmetal_core::{PMetalError, Result};
use pmetal_runtime::compat::{Array, Exception, ops as mlx_ops};
use rayon::prelude::*;

use super::Sample;
use crate::dataloader::{LoaderState, epoch_order};

/// Label of a position the loss ignores.
const IGNORE_INDEX: i64 = -100;

/// A packed batch containing multiple concatenated sequences.
#[derive(Debug, Clone)]
pub struct PackedBatch {
    /// Concatenated input token IDs, padding included.
    pub input_ids: Vec<u32>,
    /// Position IDs (restart at zero for each sequence, and for the padding).
    pub position_ids: Vec<u32>,
    /// Cumulative lengths of the attention blocks: the documents, then the
    /// padding when there is any. `cu_seqlens[0] = 0`,
    /// `cu_seqlens[i+1] = cu_seqlens[i] + seq_lengths[i]`.
    pub cu_seqlens: Vec<u32>,
    /// Length of each attention block (documents, then padding).
    pub seq_lengths: Vec<u32>,
    /// Longest attention block in this batch.
    pub max_seqlen: usize,
    /// Number of documents in this batch (the padding block is not one).
    pub num_sequences: usize,
    /// Labels, aligned with `input_ids` and unshifted. A document that came
    /// without labels trains on every token; the first token of every
    /// document after the first, and all padding, is `-100`.
    pub labels: Vec<i64>,
    /// Trailing padding tokens, counted in `input_ids`.
    pub padding: usize,
}

/// Fill the causal block of one document in an `n x n` additive mask.
fn fill_causal_block(mask: &mut [f32], n: usize, start: usize, len: usize, window: Option<usize>) {
    for i in 0..len {
        let first = window.map_or(0, |w| (i + 1).saturating_sub(w));
        let row = (start + i) * n + start;
        mask[row + first..=row + i].fill(0.0);
    }
}

impl PackedBatch {
    /// Get the total number of tokens in this batch, padding included.
    pub fn total_tokens(&self) -> usize {
        self.input_ids.len()
    }

    /// The tokens that belong to a document, i.e. without the padding.
    pub fn real_tokens(&self) -> usize {
        self.input_ids.len() - self.padding
    }

    /// Additive attention mask: `mask[i * n + j]` is `0.0` where token `i` may
    /// attend to token `j` and `-inf` elsewhere, with `n = total_tokens()`.
    ///
    /// Each attention block (a document, or the padding) is causal and sees
    /// nothing outside itself.
    pub fn build_attention_mask(&self) -> Vec<f32> {
        self.block_causal_mask(None)
    }

    /// [`build_attention_mask`](Self::build_attention_mask) with a sliding
    /// window: a token attends to at most `window_size` tokens, itself
    /// included.
    pub fn build_attention_mask_with_window(&self, window_size: usize) -> Vec<f32> {
        self.block_causal_mask(Some(window_size))
    }

    fn block_causal_mask(&self, window: Option<usize>) -> Vec<f32> {
        let n = self.total_tokens();
        let mut mask = vec![f32::NEG_INFINITY; n * n];
        let mut start = 0usize;
        for &len in &self.seq_lengths {
            fill_causal_block(&mut mask, n, start, len as usize, window);
            start += len as usize;
        }
        mask
    }

    /// The mask as an MLX array of shape `[total_tokens, total_tokens]`.
    pub fn build_attention_mask_array(&self) -> std::result::Result<Array, Exception> {
        let n = self.total_tokens() as i32;
        Ok(Array::from_f32_slice(&self.build_attention_mask(), &[n, n]))
    }

    /// Index of the last token of each document.
    pub fn sequence_boundaries(&self) -> Vec<usize> {
        let mut boundaries = Vec::with_capacity(self.num_sequences);
        let mut offset = 0usize;
        for &len in self.seq_lengths.iter().take(self.num_sequences) {
            if len > 0 {
                boundaries.push(offset + len as usize - 1);
            }
            offset += len as usize;
        }
        boundaries
    }

    /// Mask the label of the first token of every document after the first.
    ///
    /// Labels are unshifted, so the loss scores the logits at `t` against
    /// `labels[t + 1]`. Without this the last token of one document would be
    /// scored on predicting the first token of the next, which it cannot see.
    pub fn mask_label_boundaries(&mut self, ignore_index: i64) {
        let mut offset = 0usize;
        for (i, &len) in self.seq_lengths.iter().take(self.num_sequences).enumerate() {
            if i > 0 && offset < self.labels.len() {
                self.labels[offset] = ignore_index;
            }
            offset += len as usize;
        }
    }
}

/// Packing algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PackingAlgorithm {
    /// Keep the dataset's order; start a new row when the next sample does not
    /// fit.
    Greedy,

    /// First-Fit Decreasing: longest first, each into the first row with room
    /// (better utilization, reorders samples).
    #[default]
    FirstFitDecreasing,

    /// Best-Fit Decreasing: longest first, each into the row it leaves the
    /// least room in. Packs as tightly as first fit or tighter.
    BestFitDecreasing,
}

/// Sequence packer configuration.
#[derive(Debug, Clone)]
pub struct PackerConfig {
    /// Maximum number of real tokens per packed row.
    pub max_length: usize,
    /// Pad each row with a block of `pad_token_id` to a multiple of this
    /// (`0` and `1` leave rows unpadded). Padding is added on top of
    /// `max_length`, so a row is at most `pad_to_multiple - 1` tokens longer.
    pub pad_to_multiple: usize,
    /// Token the padding is made of. Its label is `-100` and nothing attends
    /// to it, so only needs to be a valid vocabulary id.
    pub pad_token_id: u32,
    /// Mask sequence boundaries in labels with -100.
    /// This prevents the loss from penalizing predictions
    /// at sequence boundaries.
    pub mask_boundaries: bool,
    /// Packing algorithm to use.
    pub algorithm: PackingAlgorithm,
    /// Minimum sequence length (drop shorter sequences).
    pub min_length: usize,
    /// Truncate sequences to this length before packing (never longer than
    /// `max_length`, to which a longer sequence is truncated regardless).
    pub max_seq_length: Option<usize>,
}

impl Default for PackerConfig {
    fn default() -> Self {
        Self {
            max_length: 2048,
            pad_to_multiple: 8,
            pad_token_id: 0,
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

    /// Use BFD packing (tightest fit).
    pub fn with_bfd(mut self) -> Self {
        self.algorithm = PackingAlgorithm::BestFitDecreasing;
        self
    }

    /// Set minimum sequence length.
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = min_length;
        self
    }

    /// Truncate every sequence to `max_seq_length` before packing.
    pub fn with_max_seq_length(mut self, max_seq_length: usize) -> Self {
        self.max_seq_length = Some(max_seq_length);
        self
    }

    /// Pad each row to a multiple of `multiple` with `pad_token_id`.
    pub fn with_padding(mut self, multiple: usize, pad_token_id: u32) -> Self {
        self.pad_to_multiple = multiple;
        self.pad_token_id = pad_token_id;
        self
    }

    /// The longest a sequence can be once packed.
    fn sequence_cap(&self) -> usize {
        self.max_seq_length
            .map_or(self.max_length, |m| m.min(self.max_length))
    }

    /// `len` rounded up to `pad_to_multiple`.
    fn padded(&self, len: usize) -> usize {
        match self.pad_to_multiple {
            0 | 1 => len,
            m => len.div_ceil(m) * m,
        }
    }
}

/// Statistics about packing efficiency.
#[derive(Debug, Clone, Default)]
pub struct PackingStats {
    /// Tokens in the packed rows that belong to a document.
    pub total_tokens: usize,
    /// Total capacity (rows * max_length).
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
    /// Padding tokens added to reach `pad_to_multiple`.
    pub padding_tokens: usize,
    /// Sequences longer than a row, cut to fit one.
    pub truncated_sequences: usize,
    /// Sequences shorter than `min_length`, left out.
    pub dropped_sequences: usize,
}

impl PackingStats {
    /// Calculate stats from batches.
    pub fn from_batches(batches: &[PackedBatch], max_length: usize) -> Self {
        if batches.is_empty() {
            return Self::default();
        }

        let padding_tokens: usize = batches.iter().map(|b| b.padding).sum();
        let total_tokens: usize = batches.iter().map(|b| b.real_tokens()).sum();
        let total_capacity = batches.len() * max_length;
        let num_sequences: usize = batches.iter().map(|b| b.num_sequences).sum();
        let max_seqs = batches.iter().map(|b| b.num_sequences).max().unwrap_or(0);

        Self {
            total_tokens,
            total_capacity,
            num_batches: batches.len(),
            num_sequences,
            efficiency: total_tokens as f64 / total_capacity.max(1) as f64,
            avg_sequences_per_batch: num_sequences as f64 / batches.len() as f64,
            max_sequences_per_batch: max_seqs,
            padding_tokens,
            truncated_sequences: 0,
            dropped_sequences: 0,
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

/// Which samples go in which row.
struct Plan {
    /// Sample indices of each row, in the order they are laid out.
    rows: Vec<Vec<u32>>,
    truncated: usize,
    dropped: usize,
}

/// Leftmost-fit search over the free space of an open-ended list of rows.
///
/// A segment tree of free capacity, grown by doubling, so placing one sample
/// is `O(log rows)` instead of a scan of every row opened so far.
struct FirstFitTree {
    capacity: u32,
    /// Leaves in the tree, a power of two.
    leaves: usize,
    /// `tree[1]` is the maximum free space; leaves start at `tree[leaves]`.
    tree: Vec<u32>,
}

impl FirstFitTree {
    fn new(capacity: u32) -> Self {
        let mut tree = Self {
            capacity,
            leaves: 0,
            tree: Vec::new(),
        };
        tree.grow(64);
        tree
    }

    /// Resize to `leaves` leaves, keeping the free space already recorded.
    /// Every new leaf is an unopened, empty row.
    fn grow(&mut self, leaves: usize) {
        let old = std::mem::take(&mut self.tree);
        let old_leaves = self.leaves;
        self.leaves = leaves;
        self.tree = vec![self.capacity; 2 * leaves];
        if old_leaves > 0 {
            self.tree[leaves..leaves + old_leaves].copy_from_slice(&old[old_leaves..]);
        }
        for node in (1..leaves).rev() {
            self.tree[node] = self.tree[2 * node].max(self.tree[2 * node + 1]);
        }
    }

    /// Place `len` tokens in the first row with room, opening one if none has.
    /// Returns its index.
    fn place(&mut self, len: u32) -> usize {
        if self.tree[1] < len {
            self.grow(self.leaves * 2);
        }
        let mut node = 1;
        while node < self.leaves {
            node = if self.tree[2 * node] >= len {
                2 * node
            } else {
                2 * node + 1
            };
        }
        self.tree[node] -= len;
        let row = node - self.leaves;
        while node > 1 {
            node /= 2;
            self.tree[node] = self.tree[2 * node].max(self.tree[2 * node + 1]);
        }
        row
    }
}

/// Order-preserving next-fit: a new row whenever the next sample overflows.
fn plan_greedy(lens: &[(u32, usize)], capacity: usize) -> Vec<Vec<u32>> {
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let mut used = 0;
    for &(index, len) in lens {
        match rows.last_mut() {
            Some(row) if used + len <= capacity => {
                row.push(index);
                used += len;
            }
            _ => {
                rows.push(vec![index]);
                used = len;
            }
        }
    }
    rows
}

/// First-fit over `lens` in the given order.
fn plan_first_fit(lens: &[(u32, usize)], capacity: usize) -> Vec<Vec<u32>> {
    let mut tree = FirstFitTree::new(capacity as u32);
    let mut rows: Vec<Vec<u32>> = Vec::new();
    for &(index, len) in lens {
        let row = tree.place(len as u32);
        if row == rows.len() {
            rows.push(Vec::new());
        }
        rows[row].push(index);
    }
    rows
}

/// Best-fit over `lens` in the given order: the row with the least free space
/// that still holds the sample, the lowest-numbered among ties.
fn plan_best_fit(lens: &[(u32, usize)], capacity: usize) -> Vec<Vec<u32>> {
    let mut rows: Vec<Vec<u32>> = Vec::new();
    // (free space, row) of every row with any left.
    let mut free: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(index, len) in lens {
        let fit = free.range((len, 0)..).next().copied();
        let (left, row) = match fit {
            Some((room, row)) => {
                free.remove(&(room, row));
                rows[row].push(index);
                (room - len, row)
            }
            None => {
                rows.push(vec![index]);
                (capacity - len, rows.len() - 1)
            }
        };
        if left > 0 {
            free.insert((left, row));
        }
    }
    rows
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
    /// Uses the configured packing algorithm.
    pub fn pack(&self, samples: &[Sample]) -> Result<Vec<PackedBatch>> {
        Ok(self.pack_with_stats(samples)?.0)
    }

    /// Pack samples into batches and return stats.
    pub fn pack_with_stats(&self, samples: &[Sample]) -> Result<(Vec<PackedBatch>, PackingStats)> {
        let plan = self.plan(samples)?;
        let batches = plan
            .rows
            .par_iter()
            .map(|row| self.create_packed_batch(samples, row))
            .collect::<Result<Vec<_>>>()?;
        let mut stats = PackingStats::from_batches(&batches, self.config.max_length);
        stats.truncated_sequences = plan.truncated;
        stats.dropped_sequences = plan.dropped;
        Ok((batches, stats))
    }

    /// Decide which sample goes in which row, touching lengths only.
    fn plan(&self, samples: &[Sample]) -> Result<Plan> {
        let config = &self.config;
        if config.max_length == 0 || config.max_length > u32::MAX as usize {
            return Err(PMetalError::InvalidArgument(format!(
                "packing max_length must be between 1 and {}, got {}",
                u32::MAX,
                config.max_length
            )));
        }
        if samples.len() > u32::MAX as usize {
            return Err(PMetalError::InvalidArgument(format!(
                "cannot pack {} samples at once",
                samples.len()
            )));
        }

        let cap = config.sequence_cap();
        let min_length = config.min_length.max(1);
        let (mut truncated, mut dropped) = (0, 0);
        let mut lens: Vec<(u32, usize)> = Vec::with_capacity(samples.len());
        for (index, sample) in samples.iter().enumerate() {
            let len = sample.input_ids.len();
            if len < min_length {
                dropped += 1;
                continue;
            }
            if len > cap {
                truncated += 1;
            }
            lens.push((index as u32, len.min(cap)));
        }

        // Stable, so equal lengths keep the dataset's order.
        let longest_first = |lens: &mut Vec<(u32, usize)>| {
            lens.sort_by_key(|&(_, len)| std::cmp::Reverse(len));
        };
        let rows = match config.algorithm {
            PackingAlgorithm::Greedy => plan_greedy(&lens, config.max_length),
            PackingAlgorithm::FirstFitDecreasing => {
                longest_first(&mut lens);
                plan_first_fit(&lens, config.max_length)
            }
            PackingAlgorithm::BestFitDecreasing => {
                longest_first(&mut lens);
                plan_best_fit(&lens, config.max_length)
            }
        };
        Ok(Plan {
            rows,
            truncated,
            dropped,
        })
    }

    fn create_packed_batch(&self, samples: &[Sample], row: &[u32]) -> Result<PackedBatch> {
        let cap = self.config.sequence_cap();
        let docs: Vec<&Sample> = row.iter().map(|&i| &samples[i as usize]).collect();
        let lens: Vec<usize> = docs.iter().map(|s| s.input_ids.len().min(cap)).collect();
        let real: usize = lens.iter().sum();
        let total = self.config.padded(real);
        let padding = total - real;

        let overflow = |what: &str| {
            PMetalError::InvalidArgument(format!(
                "Packed batch {what} exceed u32::MAX ({})",
                u32::MAX
            ))
        };
        if u32::try_from(total).is_err() {
            return Err(overflow("total tokens"));
        }

        let mut input_ids = Vec::with_capacity(total);
        let mut position_ids = Vec::with_capacity(total);
        let mut labels = Vec::with_capacity(total);
        let mut seq_lengths = Vec::with_capacity(docs.len() + 1);
        let mut cu_seqlens = Vec::with_capacity(docs.len() + 2);
        cu_seqlens.push(0u32);

        for (doc, &len) in docs.iter().zip(&lens) {
            input_ids.extend_from_slice(&doc.input_ids[..len]);
            position_ids.extend(0..len as u32);
            match &doc.labels {
                Some(l) if l.len() == doc.input_ids.len() => labels.extend_from_slice(&l[..len]),
                Some(l) => {
                    return Err(PMetalError::InvalidArgument(format!(
                        "a sample has {} labels for {} tokens",
                        l.len(),
                        doc.input_ids.len()
                    )));
                }
                None => labels.extend(doc.input_ids[..len].iter().map(|&t| i64::from(t))),
            }
            seq_lengths.push(len as u32);
            cu_seqlens.push(*cu_seqlens.last().unwrap_or(&0) + len as u32);
        }

        if padding > 0 {
            input_ids.resize(total, self.config.pad_token_id);
            position_ids.extend(0..padding as u32);
            labels.resize(total, IGNORE_INDEX);
            seq_lengths.push(padding as u32);
            cu_seqlens.push(total as u32);
        }

        let mut batch = PackedBatch {
            input_ids,
            position_ids,
            cu_seqlens,
            max_seqlen: seq_lengths.iter().map(|&l| l as usize).max().unwrap_or(0),
            seq_lengths,
            num_sequences: docs.len(),
            labels,
            padding,
        };
        if self.config.mask_boundaries {
            batch.mask_label_boundaries(IGNORE_INDEX);
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

/// A packed batch ready for training with MLX Arrays.
///
/// This is the bridge between the packing infrastructure and the training loop.
/// Unlike regular TrainingBatch which has shape [batch_size, seq_len],
/// PackedTrainingBatch has shape `[total_tokens]` with variable-length sequences.
#[derive(Debug)]
pub struct PackedTrainingBatch {
    /// Packed input token IDs `[total_tokens]`.
    pub input_ids: Array,
    /// Position IDs reset per sequence `[total_tokens]`.
    pub position_ids: Array,
    /// Cumulative lengths of the attention blocks, `[blocks + 1]`.
    pub cu_seqlens: Array,
    /// `cu_seqlens` on the CPU, for mask construction.
    pub cu_seqlens_raw: Vec<i32>,
    /// Packed labels for loss computation `[total_tokens]`.
    pub labels: Array,
    /// Total number of tokens, padding included.
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
    /// Returns an error if a token id or label does not fit in an `i32`.
    pub fn from_packed_batch(batch: &PackedBatch) -> std::result::Result<Self, Exception> {
        let to_i32 = |v: i64| {
            i32::try_from(v).map_err(|_| Exception::custom(format!("{v} does not fit an i32")))
        };
        let input_ids_i32 = batch
            .input_ids
            .iter()
            .map(|&x| to_i32(i64::from(x)))
            .collect::<std::result::Result<Vec<i32>, _>>()?;
        let position_ids_i32 = batch
            .position_ids
            .iter()
            .map(|&x| to_i32(i64::from(x)))
            .collect::<std::result::Result<Vec<i32>, _>>()?;
        let cu_seqlens_raw = batch
            .cu_seqlens
            .iter()
            .map(|&x| to_i32(i64::from(x)))
            .collect::<std::result::Result<Vec<i32>, _>>()?;
        let labels_i32 = batch
            .labels
            .iter()
            .map(|&x| to_i32(x))
            .collect::<std::result::Result<Vec<i32>, _>>()?;

        Ok(Self {
            input_ids: Array::from_i32_slice(&input_ids_i32),
            position_ids: Array::from_i32_slice(&position_ids_i32),
            cu_seqlens: Array::from_i32_slice(&cu_seqlens_raw),
            cu_seqlens_raw,
            labels: Array::from_i32_slice(&labels_i32),
            total_tokens: batch.input_ids.len(),
            num_sequences: batch.num_sequences,
            max_seqlen: batch.max_seqlen,
        })
    }

    /// The block-diagonal causal mask for this batch, `[total_tokens,
    /// total_tokens]`, `0.0` where a token may attend and `-inf` elsewhere.
    ///
    /// Identical to [`PackedBatch::build_attention_mask`], but built on the
    /// device from per-token block ids: the dense `n x n` matrix never exists
    /// on the host.
    pub fn attention_mask(&self) -> std::result::Result<Array, Exception> {
        let n = self.total_tokens as i32;
        let mut block = Vec::with_capacity(self.total_tokens);
        for (id, w) in self.cu_seqlens_raw.windows(2).enumerate() {
            block.extend(std::iter::repeat_n(
                id as i32,
                (w[1] - w[0]).max(0) as usize,
            ));
        }
        let block = Array::from_i32_slice(&block);
        let position = mlx_ops::arange_n(n);
        let same_block = mlx_ops::equal(&block.reshape(&[n, 1]), &block.reshape(&[1, n]));
        let causal = mlx_ops::greater_equal(&position.reshape(&[n, 1]), &position.reshape(&[1, n]));
        let allowed = mlx_ops::logical_and(&same_block, &causal);
        Ok(mlx_ops::where_fn(
            &allowed,
            &Array::from_f32(0.0),
            &Array::from_f32(f32::NEG_INFINITY),
        ))
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
///
/// The documents are packed once, up front; an epoch is those rows in an order
/// that depends only on the seed it was started with, so an interrupted run
/// can be resumed into the exact epoch with [`state`](Self::state) and
/// [`restore`](Self::restore). Under data parallelism
/// ([`with_sharding`](Self::with_sharding)) each rank takes an equal share of
/// the rows.
pub struct PackedDataLoader {
    config: PackerConfig,
    /// Every packed row, in packing order.
    batches: Vec<PackedBatch>,
    /// Rows to serve this epoch, in order, for this rank.
    order: Vec<usize>,
    /// Rows already served this epoch.
    position: usize,
    /// Whether to shuffle rows each epoch.
    shuffle: bool,
    /// Seed `reset(None)` starts an epoch from.
    seed: u64,
    /// Seed of the epoch being served.
    epoch_seed: u64,
    rank: usize,
    world_size: usize,
    stats: PackingStats,
}

impl PackedDataLoader {
    /// Pack `samples` and serve the rows, shuffled by `seed` when `shuffle`.
    pub fn new(samples: &[Sample], config: PackerConfig, shuffle: bool, seed: u64) -> Result<Self> {
        let (batches, stats) = SequencePacker::new(config.clone()).pack_with_stats(samples)?;
        let mut loader = Self {
            config,
            batches,
            order: Vec::new(),
            position: 0,
            shuffle,
            seed,
            epoch_seed: seed,
            rank: 0,
            world_size: 1,
            stats,
        };
        loader.start_epoch(seed);
        Ok(loader)
    }

    /// Serve this rank's share of the rows: every `world_size`-th row of the
    /// epoch's order starting at `rank`. Every rank gets the same number of
    /// rows (the remainder, fewer than `world_size` rows a different set each
    /// shuffled epoch, is left out) so the ranks stay in step.
    pub fn with_sharding(mut self, rank: usize, world_size: usize) -> Result<Self> {
        if world_size == 0 || rank >= world_size {
            return Err(PMetalError::InvalidArgument(format!(
                "rank {rank} is not in a world of {world_size}"
            )));
        }
        self.rank = rank;
        self.world_size = world_size;
        self.start_epoch(self.epoch_seed);
        Ok(self)
    }

    fn start_epoch(&mut self, seed: u64) {
        self.epoch_seed = seed;
        self.position = 0;
        self.order = epoch_order(
            self.batches.len(),
            self.shuffle,
            seed,
            self.rank,
            self.world_size,
        );
    }

    /// Packing statistics of all the rows, every rank's included.
    pub fn stats(&self) -> PackingStats {
        self.stats.clone()
    }

    /// Number of rows this rank serves per epoch.
    pub fn num_batches(&self) -> usize {
        self.order.len()
    }

    /// Start a new epoch, shuffled by `new_seed` (the construction seed when
    /// `None`). The order depends on the seed alone.
    pub fn reset(&mut self, new_seed: Option<u64>) {
        self.start_epoch(new_seed.unwrap_or(self.seed));
    }

    /// Where the loader is: enough to resume it exactly.
    pub fn state(&self) -> LoaderState {
        LoaderState {
            epoch_seed: self.epoch_seed,
            position: self.position,
        }
    }

    /// Resume at `state`, taken from a loader built over the same samples and
    /// configuration.
    pub fn restore(&mut self, state: LoaderState) -> Result<()> {
        self.start_epoch(state.epoch_seed);
        if state.position > self.order.len() {
            return Err(PMetalError::InvalidArgument(format!(
                "cannot resume at batch {} of an epoch of {}",
                state.position,
                self.order.len()
            )));
        }
        self.position = state.position;
        Ok(())
    }

    /// Get the next packed training batch.
    pub fn next_batch(&mut self) -> Option<std::result::Result<PackedTrainingBatch, Exception>> {
        let row = *self.order.get(self.position)?;
        self.position += 1;
        Some(PackedTrainingBatch::from_packed_batch(&self.batches[row]))
    }

    /// The packer configuration the rows were built with.
    pub fn config(&self) -> &PackerConfig {
        &self.config
    }
}

impl Iterator for PackedDataLoader {
    type Item = std::result::Result<PackedTrainingBatch, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

// =============================================================================
// Adaptive packing sequence length
// =============================================================================

/// Compute an adaptive packing sequence length from dataset statistics.
///
/// Uses the p99 of actual sequence lengths, rounded up to the next power of 2,
/// capped at `model_max_seq_len`. This avoids the O(n²) attention cost that
/// arises when `max_seq_len` auto-detects to the model's architectural maximum
/// (e.g., 8192) for datasets whose sequences are much shorter (e.g., 50–400
/// tokens).
///
/// # Example
///
/// For a dataset with 50–400 token sequences and a model max of 8192:
/// - p99 ≈ 390
/// - next_power_of_two(390) = 512
/// - min(512, 8192) = **512** ← used for packing instead of 8192
///
/// This reduces attention cost by (8192/512)² ≈ 256×.
pub fn compute_pack_seq_len(sequence_lengths: &[usize], model_max_seq_len: usize) -> usize {
    if sequence_lengths.is_empty() {
        return model_max_seq_len;
    }
    let mut sorted = sequence_lengths.to_vec();
    sorted.sort_unstable();
    // p99 index: ceiling of 99% of the sorted length, clamped to a valid index.
    let p99_idx = ((sorted.len() as f64) * 0.99).ceil() as usize;
    let p99 = sorted[p99_idx.saturating_sub(1).min(sorted.len() - 1)];
    let adaptive = p99.next_power_of_two().min(model_max_seq_len);
    tracing::info!(
        "Adaptive pack seq_len: p99={p99}, next_pow2={pow2}, model_max={model_max_seq_len}, using={adaptive}",
        pow2 = p99.next_power_of_two(),
    );
    adaptive
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
        let packer = SequencePacker::new(
            PackerConfig::with_max_length(10)
                .with_padding(1, 0)
                .with_greedy(),
        );
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
            labels: vec![0; 5],
            padding: 0,
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
            labels: vec![0; 6],
            padding: 0,
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
                .with_padding(1, 0)
                .with_greedy()
                .mask_boundaries(true),
        );
        let batches = packer.pack(&samples).unwrap();

        let labels = &batches[0].labels;
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
            labels: vec![0; 5],
            padding: 0,
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

        let ffd_packer = SequencePacker::new(
            PackerConfig::with_max_length(10)
                .with_padding(1, 0)
                .with_ffd(),
        );
        let greedy_packer = SequencePacker::new(
            PackerConfig::with_max_length(10)
                .with_padding(1, 0)
                .with_greedy(),
        );

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
        // A sample that exceeds max_length is cut to a full row, not dropped.
        let samples = vec![
            make_sample(vec![1; 15], None), // Too large for max_length=10
            make_sample(vec![2; 5], None),
            make_sample(vec![3; 3], None),
        ];

        let packer = SequencePacker::new(
            PackerConfig::with_max_length(10)
                .with_padding(1, 0)
                .with_ffd(),
        );
        let (batches, stats) = packer.pack_with_stats(&samples).unwrap();

        let total_tokens: usize = batches.iter().map(|b| b.total_tokens()).sum();
        assert_eq!(total_tokens, 18); // 10 + 5 + 3
        assert_eq!(stats.truncated_sequences, 1);
        assert_eq!(stats.dropped_sequences, 0);
    }

    #[test]
    fn test_packing_stats() {
        let samples = vec![
            make_sample(vec![1; 5], None),
            make_sample(vec![2; 3], None),
            make_sample(vec![3; 7], None),
        ];

        let packer = SequencePacker::new(PackerConfig::with_max_length(10).with_padding(1, 0));
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

        let packer = SequencePacker::new(
            PackerConfig::with_max_length(10)
                .with_padding(1, 0)
                .with_greedy(),
        );
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

        let packer = SequencePacker::new(
            PackerConfig::with_max_length(10)
                .with_padding(1, 0)
                .with_min_length(2),
        );
        let batches = packer.pack(&samples).unwrap();

        let total_tokens: usize = batches.iter().map(|b| b.total_tokens()).sum();
        assert_eq!(total_tokens, 5); // Only [2,3] and [5,6,7]
    }

    /// The row-major `i32` contents of an array, failing the test if an op
    /// threw while producing it.
    fn read_i32(a: &Array) -> Vec<i32> {
        let v = a.as_slice::<i32>().to_vec();
        pmetal_runtime::check_unobserved_error().expect("a bridge op threw");
        v
    }

    /// Labels are unshifted (the loss shifts them), so a batch built from
    /// samples that carry none must train on every token and still mask the
    /// first token of each later document. It used to shift `input_ids` by one
    /// itself, which the loss then shifted again, and let the last token of
    /// each document predict the first of the next.
    #[test]
    fn labels_default_to_the_input_ids_with_document_boundaries_masked() {
        let samples = vec![
            make_sample(vec![1, 2, 3], None),
            make_sample(vec![4, 5], None),
        ];
        let packer = SequencePacker::new(PackerConfig::with_max_length(8).with_greedy());
        let batches = packer.pack(&samples).unwrap();
        let training = PackedTrainingBatch::from_packed_batch(&batches[0]).unwrap();
        assert_eq!(read_i32(&training.labels)[..5], [1, 2, 3, -100, 5]);
    }

    /// `pad_to_multiple` was a config knob nothing read.
    #[test]
    fn batches_are_padded_to_the_configured_multiple() {
        let samples = vec![make_sample(vec![1, 2, 3], None), make_sample(vec![4], None)];
        let packer = SequencePacker::new(PackerConfig::with_max_length(64).with_greedy());
        let batch = &packer.pack(&samples).unwrap()[0];
        assert_eq!(
            batch.total_tokens() % 8,
            0,
            "{} tokens",
            batch.total_tokens()
        );
        assert_eq!(batch.num_sequences, 2);
    }

    /// A sequence longer than a batch is truncated to fit, as the unpacked
    /// loader truncates at `max_seq_len`; FFD used to drop it, silently,
    /// unless `max_seq_length` was also set.
    #[test]
    fn an_overlong_sequence_is_truncated_not_dropped() {
        let samples = vec![make_sample((1..=15).collect(), None)];
        for algorithm in [
            PackingAlgorithm::FirstFitDecreasing,
            PackingAlgorithm::Greedy,
        ] {
            let mut config = PackerConfig::with_max_length(10).with_padding(1, 0);
            config.algorithm = algorithm;
            let batches = SequencePacker::new(config).pack(&samples).unwrap();
            assert_eq!(batches.len(), 1, "{algorithm:?}");
            assert_eq!(batches[0].input_ids[..10], (1..=10).collect::<Vec<u32>>());
        }
    }

    /// The order a loader serves a shuffled epoch in is a function of the seed
    /// it was reset with, not of how many epochs it has already served:
    /// otherwise an interrupted run can never reproduce the epoch it stopped
    /// in.
    #[test]
    fn a_reset_epoch_order_depends_only_on_its_seed() {
        let samples: Vec<Sample> = (1..=40)
            .map(|i| make_sample(vec![i; (i as usize % 5) + 1], None))
            .collect();
        let order = |loader: &mut PackedDataLoader| -> Vec<Vec<i32>> {
            loader.map(|b| read_i32(&b.unwrap().input_ids)).collect()
        };
        let config = || PackerConfig::with_max_length(6).with_greedy();
        let mut fresh = PackedDataLoader::new(&samples, config(), true, 7).unwrap();
        fresh.reset(Some(11));
        let mut worn = PackedDataLoader::new(&samples, config(), true, 7).unwrap();
        worn.reset(Some(3));
        worn.reset(Some(5));
        worn.reset(Some(11));
        assert_eq!(order(&mut fresh), order(&mut worn));
    }

    /// Deterministic pseudo-random lengths in `1..=max`, so the property tests
    /// need no extra dependency and fail reproducibly.
    fn random_lengths(count: usize, max: usize, mut state: u64) -> Vec<usize> {
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                1 + ((state >> 33) as usize % max)
            })
            .collect()
    }

    /// Sample `i` is `[i * 1000, i * 1000 + 1, ...]`, so a token names its
    /// document and its position in it. A prompt of `prompt_len(i)` tokens is
    /// masked, as SFT masks it.
    fn tagged_samples(lengths: &[usize]) -> Vec<Sample> {
        lengths
            .iter()
            .enumerate()
            .map(|(i, &len)| {
                let ids: Vec<u32> = (0..len as u32).map(|t| i as u32 * 1000 + t).collect();
                let prompt = len / 3;
                let labels = ids
                    .iter()
                    .enumerate()
                    .map(|(t, &id)| if t < prompt { -100 } else { i64::from(id) })
                    .collect();
                make_sample(ids, Some(labels))
            })
            .collect()
    }

    /// What the loss scores for an unpacked sample: the label at `t + 1`
    /// against the logits at `t`, as `(document, t, label)`.
    fn unpacked_loss_targets(samples: &[Sample]) -> Vec<(usize, usize, i64)> {
        let mut targets = Vec::new();
        for (doc, s) in samples.iter().enumerate() {
            let labels = s.labels.as_ref().unwrap();
            for t in 0..labels.len() - 1 {
                if labels[t + 1] != -100 {
                    targets.push((doc, t, labels[t + 1]));
                }
            }
        }
        targets.sort_unstable();
        targets
    }

    /// The same, read off packed rows: what the loss scores at row position
    /// `p` is the label at `p + 1`.
    fn packed_loss_targets(batches: &[PackedBatch]) -> Vec<(usize, usize, i64)> {
        let mut targets = Vec::new();
        for b in batches {
            for p in 0..b.total_tokens() - 1 {
                if b.labels[p + 1] != -100 {
                    // The context token names its document and position.
                    let id = b.input_ids[p] as usize;
                    assert!(b.position_ids[p] as usize == id % 1000, "position restarts");
                    targets.push((id / 1000, id % 1000, b.labels[p + 1]));
                }
            }
        }
        targets.sort_unstable();
        targets
    }

    /// Packing changes nothing about what is trained: the (context, target)
    /// pairs the loss scores in the packed rows are exactly those of the
    /// unpacked samples, for every algorithm, padding, and truncation.
    #[test]
    fn packed_rows_train_exactly_what_the_unpacked_samples_train() {
        let lengths = random_lengths(200, 40, 3);
        let samples = tagged_samples(&lengths);
        let expected = unpacked_loss_targets(&samples);
        for algorithm in [
            PackingAlgorithm::Greedy,
            PackingAlgorithm::FirstFitDecreasing,
            PackingAlgorithm::BestFitDecreasing,
        ] {
            for pad in [1, 8, 64] {
                let mut config = PackerConfig::with_max_length(96).with_padding(pad, 7);
                config.algorithm = algorithm;
                let batches = SequencePacker::new(config).pack(&samples).unwrap();
                assert_eq!(
                    packed_loss_targets(&batches),
                    expected,
                    "{algorithm:?}, pad {pad}"
                );
                let docs: usize = batches.iter().map(|b| b.num_sequences).sum();
                assert_eq!(docs, samples.len());
            }
        }
    }

    /// Every token attends to exactly its own document's earlier tokens
    /// (itself included), padding to its own block, and no row is empty.
    #[test]
    fn the_attention_mask_confines_every_token_to_its_own_document() {
        let samples = tagged_samples(&random_lengths(30, 12, 9));
        let batches = SequencePacker::new(PackerConfig::with_max_length(40).with_padding(16, 0))
            .pack(&samples)
            .unwrap();
        assert!(batches.iter().any(|b| b.padding > 0));
        for b in &batches {
            let n = b.total_tokens();
            let mask = b.build_attention_mask();
            // Block of each position, from the cumulative lengths.
            let mut block = vec![0usize; n];
            for (id, w) in b.cu_seqlens.windows(2).enumerate() {
                block[w[0] as usize..w[1] as usize].fill(id);
            }
            for q in 0..n {
                let mut visible = 0;
                for k in 0..n {
                    let allowed = block[q] == block[k] && k <= q;
                    assert_eq!(mask[q * n + k] == 0.0, allowed, "q {q} k {k}");
                    visible += usize::from(allowed);
                }
                assert!(visible >= 1, "row {q} attends to nothing");
            }
        }
    }

    /// The device-built mask is the host-built one.
    #[test]
    fn the_device_mask_equals_the_host_mask() {
        let samples = tagged_samples(&random_lengths(12, 9, 5));
        let batches = SequencePacker::new(PackerConfig::with_max_length(24).with_padding(8, 0))
            .pack(&samples)
            .unwrap();
        for b in &batches {
            let device = PackedTrainingBatch::from_packed_batch(b)
                .unwrap()
                .attention_mask()
                .unwrap();
            pmetal_runtime::check_unobserved_error().expect("a bridge op threw");
            let host = b.build_attention_mask();
            assert_eq!(device.as_slice::<f32>(), host.as_slice());
        }
    }

    /// Padding is a block of its own, labelled `-100`, made of the pad token.
    #[test]
    fn padding_is_an_isolated_block_the_loss_ignores() {
        let samples = vec![make_sample(vec![1, 2, 3], None), make_sample(vec![4], None)];
        let batch = SequencePacker::new(
            PackerConfig::with_max_length(16)
                .with_greedy()
                .with_padding(8, 99),
        )
        .pack(&samples)
        .unwrap()
        .remove(0);
        assert_eq!(batch.input_ids, [1, 2, 3, 4, 99, 99, 99, 99]);
        assert_eq!(batch.labels, [1, 2, 3, -100, -100, -100, -100, -100]);
        assert_eq!(batch.position_ids, [0, 1, 2, 0, 0, 1, 2, 3]);
        assert_eq!(batch.seq_lengths, [3, 1, 4]);
        assert_eq!(batch.cu_seqlens, [0, 3, 4, 8]);
        assert_eq!((batch.num_sequences, batch.padding), (2, 4));
        assert_eq!(batch.sequence_boundaries(), [2, 3]);
        assert_eq!(batch.real_tokens(), 4);
    }

    /// The segment tree places every sample exactly where a scan of the open
    /// rows does.
    #[test]
    fn first_fit_by_segment_tree_equals_first_fit_by_scan() {
        for (count, max, cap, seed) in [(500, 30, 64, 1), (2000, 100, 128, 2), (300, 7, 8, 3)] {
            let lens: Vec<(u32, usize)> = random_lengths(count, max, seed)
                .into_iter()
                .enumerate()
                .map(|(i, l)| (i as u32, l.min(cap)))
                .collect();
            let mut scanned: Vec<(usize, Vec<u32>)> = Vec::new();
            for &(index, len) in &lens {
                match scanned.iter_mut().find(|(used, _)| used + len <= cap) {
                    Some((used, row)) => {
                        *used += len;
                        row.push(index);
                    }
                    None => scanned.push((len, vec![index])),
                }
            }
            let scanned: Vec<Vec<u32>> = scanned.into_iter().map(|(_, row)| row).collect();
            assert_eq!(plan_first_fit(&lens, cap), scanned);
        }
    }

    /// Best fit never needs more rows than first fit on the same input, and
    /// every row stays within the cap.
    #[test]
    fn best_fit_decreasing_is_valid_and_no_looser_than_first_fit() {
        let samples = tagged_samples(&random_lengths(1000, 50, 4));
        let rows = |algorithm| {
            let mut config = PackerConfig::with_max_length(128).with_padding(1, 0);
            config.algorithm = algorithm;
            SequencePacker::new(config).pack(&samples).unwrap()
        };
        let ffd = rows(PackingAlgorithm::FirstFitDecreasing);
        let bfd = rows(PackingAlgorithm::BestFitDecreasing);
        assert!(bfd.len() <= ffd.len(), "{} vs {}", bfd.len(), ffd.len());
        assert!(bfd.iter().all(|b| b.total_tokens() <= 128));
        let total: usize = bfd.iter().map(|b| b.total_tokens()).sum();
        let expected: usize = samples.iter().map(|s| s.input_ids.len()).sum();
        assert_eq!(total, expected);
    }

    /// Throughput of the planner and of batch construction, against the scan
    /// it replaced. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn packing_throughput() {
        use std::time::Instant;
        let cap = 4096;
        for count in [100_000usize, 1_000_000] {
            let lengths = random_lengths(count, 2048, 17);
            let lens: Vec<(u32, usize)> = {
                let mut v: Vec<(u32, usize)> = lengths
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (i as u32, l))
                    .collect();
                v.sort_by_key(|&(_, l)| std::cmp::Reverse(l));
                v
            };
            let start = Instant::now();
            let tree_rows = plan_first_fit(&lens, cap).len();
            let tree_time = start.elapsed();

            let start = Instant::now();
            let best_rows = plan_best_fit(&lens, cap).len();
            let best_time = start.elapsed();

            let scan_time = (count <= 100_000).then(|| {
                let start = Instant::now();
                let mut open: Vec<usize> = Vec::new();
                for &(_, len) in &lens {
                    match open.iter_mut().find(|used| **used + len <= cap) {
                        Some(used) => *used += len,
                        None => open.push(len),
                    }
                }
                assert_eq!(open.len(), tree_rows);
                start.elapsed()
            });

            let samples = tagged_samples(&lengths);
            let start = Instant::now();
            let (batches, stats) = SequencePacker::new(PackerConfig::with_max_length(cap))
                .pack_with_stats(&samples)
                .unwrap();
            let pack_time = start.elapsed();
            println!(
                "{count} samples: first-fit {tree_rows} rows in {tree_time:?} \
                 (scan {scan_time:?}), best-fit {best_rows} rows in {best_time:?}, \
                 end-to-end pack {} rows in {pack_time:?}, {:.2}% efficient",
                batches.len(),
                stats.efficiency * 100.0
            );
        }
    }

    fn row_signatures(loader: PackedDataLoader) -> Vec<Vec<i32>> {
        loader.map(|b| read_i32(&b.unwrap().input_ids)).collect()
    }

    /// Ranks serve disjoint, equally sized shares of the rows.
    #[test]
    fn data_parallel_ranks_share_the_rows_without_overlap() {
        let samples = tagged_samples(&random_lengths(101, 30, 6));
        let config = || PackerConfig::with_max_length(64).with_padding(1, 0);
        let whole = PackedDataLoader::new(&samples, config(), true, 5).unwrap();
        let rows = whole.num_batches();
        let all = row_signatures(whole);
        let world = 4;
        let shares: Vec<Vec<Vec<i32>>> = (0..world)
            .map(|rank| {
                let loader = PackedDataLoader::new(&samples, config(), true, 5)
                    .unwrap()
                    .with_sharding(rank, world)
                    .unwrap();
                assert_eq!(loader.num_batches(), rows / world);
                row_signatures(loader)
            })
            .collect();
        let mut seen: Vec<&Vec<i32>> = shares.iter().flatten().collect();
        assert_eq!(seen.len(), (rows / world) * world);
        seen.sort();
        seen.dedup();
        assert_eq!(
            seen.len(),
            (rows / world) * world,
            "a row went to two ranks"
        );
        assert!(seen.iter().all(|row| all.contains(row)));
    }

    /// A loader resumed from a saved state serves the rest of the epoch the
    /// first one would have.
    #[test]
    fn a_restored_packed_loader_serves_the_rest_of_the_epoch() {
        let samples = tagged_samples(&random_lengths(80, 30, 8));
        let config = || PackerConfig::with_max_length(64).with_padding(1, 0);
        let mut first = PackedDataLoader::new(&samples, config(), true, 21).unwrap();
        first.reset(Some(22));
        let head: Vec<_> = (&mut first).take(3).collect();
        assert_eq!(head.len(), 3);
        let state = first.state();
        let rest = row_signatures(first);

        let mut resumed = PackedDataLoader::new(&samples, config(), true, 21).unwrap();
        resumed.restore(state).unwrap();
        assert_eq!(row_signatures(resumed), rest);
    }

    #[test]
    fn test_compute_pack_seq_len_empty() {
        // Empty slice: fall back to model max unchanged.
        assert_eq!(compute_pack_seq_len(&[], 8192), 8192);
    }

    #[test]
    fn test_compute_pack_seq_len_short_dataset() {
        // Dataset with 50–400 token sequences, model max 8192.
        // p99 of 100 values in 50..=400 range → rounds to next power of 2, well below 8192.
        let lengths: Vec<usize> = (0..100).map(|i| 50 + i * 3).collect(); // 50, 53, ..., 347
        let result = compute_pack_seq_len(&lengths, 8192);
        // p99 ≈ 347, next_power_of_two(347) = 512
        assert_eq!(result, 512);
        assert!(result < 8192, "Should be far below model max");
    }

    #[test]
    fn test_compute_pack_seq_len_capped_at_model_max() {
        // Sequences longer than model max → result capped at model max.
        let lengths: Vec<usize> = vec![1000, 2000, 3000, 4000];
        let result = compute_pack_seq_len(&lengths, 2048);
        // p99 ≈ 4000, next_power_of_two = 4096, but capped at 2048
        assert_eq!(result, 2048);
    }

    #[test]
    fn test_compute_pack_seq_len_single_sequence() {
        let lengths = vec![300usize];
        let result = compute_pack_seq_len(&lengths, 8192);
        // next_power_of_two(300) = 512
        assert_eq!(result, 512);
    }

    #[test]
    fn test_compute_pack_seq_len_exact_power_of_two() {
        // If p99 is already a power of 2, result should be that power.
        let lengths: Vec<usize> = vec![512; 100];
        let result = compute_pack_seq_len(&lengths, 8192);
        assert_eq!(result, 512);
    }
}
