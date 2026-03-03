//! Vocabulary compaction for ANE training.
//!
//! Scans training data to identify active tokens, then builds a bidirectional
//! mapping (full_id ↔ compact_id) for ~3.5x reduction in embedding table,
//! classifier computation, and Adam state.
//!
//! Typical savings: 32K vocab → ~9K active tokens for TinyStories.

/// Bidirectional mapping between full vocabulary IDs and compact IDs.
#[derive(Debug, Clone)]
pub struct VocabCompactor {
    /// Full token ID → compact ID. -1 (None) if unused.
    full_to_compact: Vec<Option<u16>>,
    /// Compact ID → full token ID.
    compact_to_full: Vec<u32>,
    /// Number of active (compact) tokens.
    compact_vocab: usize,
}

impl VocabCompactor {
    /// Build a compactor by scanning all tokens in the dataset.
    ///
    /// `samples` is an iterator of token sequences (input or target).
    /// `full_vocab` is the original vocabulary size.
    pub fn from_dataset<'a, I>(samples: I, full_vocab: usize) -> Self
    where
        I: IntoIterator<Item = &'a [u16]>,
    {
        let mut seen = vec![false; full_vocab];
        for tokens in samples {
            for &t in tokens {
                if (t as usize) < full_vocab {
                    seen[t as usize] = true;
                }
            }
        }

        let mut full_to_compact = vec![None; full_vocab];
        let mut compact_to_full = Vec::new();
        let mut compact_id = 0u16;

        for (full_id, &is_active) in seen.iter().enumerate() {
            if is_active {
                full_to_compact[full_id] = Some(compact_id);
                compact_to_full.push(full_id as u32);
                compact_id += 1;
            }
        }

        let compact_vocab = compact_to_full.len();
        tracing::info!(
            full_vocab,
            compact_vocab,
            reduction = format!("{:.1}x", full_vocab as f32 / compact_vocab as f32),
            "Vocabulary compacted"
        );

        Self {
            full_to_compact,
            compact_to_full,
            compact_vocab,
        }
    }

    /// Number of active (compact) tokens.
    pub fn compact_vocab(&self) -> usize {
        self.compact_vocab
    }

    /// Map a full token ID to its compact ID.
    ///
    /// Returns `None` if the token was not seen in the training data.
    pub fn to_compact(&self, full_id: u16) -> Option<u16> {
        self.full_to_compact
            .get(full_id as usize)
            .copied()
            .flatten()
    }

    /// Map a compact token ID back to the full vocabulary ID.
    pub fn to_full(&self, compact_id: u16) -> u32 {
        self.compact_to_full[compact_id as usize]
    }

    /// Remap a token sequence to compact IDs.
    ///
    /// Tokens not in the compacted vocab are mapped to 0 (with a warning on first occurrence).
    pub fn compact_tokens(&self, tokens: &[u16]) -> Vec<u16> {
        tokens
            .iter()
            .map(|&t| self.to_compact(t).unwrap_or(0))
            .collect()
    }

    /// Extract compact embedding table from full embedding.
    ///
    /// `full_embed` is `[full_vocab, dim]` row-major.
    /// Returns `[compact_vocab, dim]` row-major.
    pub fn extract_compact_embedding(&self, full_embed: &[f32], dim: usize) -> Vec<f32> {
        let mut compact = vec![0.0f32; self.compact_vocab * dim];
        for (compact_id, &full_id) in self.compact_to_full.iter().enumerate() {
            let src_off = full_id as usize * dim;
            let dst_off = compact_id * dim;
            compact[dst_off..dst_off + dim].copy_from_slice(&full_embed[src_off..src_off + dim]);
        }
        compact
    }

    /// Scatter compact gradients back to full embedding gradient.
    ///
    /// `full_grad` is `[full_vocab, dim]`, `compact_grad` is `[compact_vocab, dim]`.
    /// Accumulates (+=) into `full_grad`.
    pub fn scatter_gradients(&self, full_grad: &mut [f32], compact_grad: &[f32], dim: usize) {
        for (compact_id, &full_id) in self.compact_to_full.iter().enumerate() {
            let src_off = compact_id * dim;
            let dst_off = full_id as usize * dim;
            for i in 0..dim {
                full_grad[dst_off + i] += compact_grad[src_off + i];
            }
        }
    }

    /// Write compact embedding back to full embedding after Adam update.
    ///
    /// `full_embed` is `[full_vocab, dim]`, `compact_embed` is `[compact_vocab, dim]`.
    pub fn update_full_embedding(&self, full_embed: &mut [f32], compact_embed: &[f32], dim: usize) {
        for (compact_id, &full_id) in self.compact_to_full.iter().enumerate() {
            let src_off = compact_id * dim;
            let dst_off = full_id as usize * dim;
            full_embed[dst_off..dst_off + dim]
                .copy_from_slice(&compact_embed[src_off..src_off + dim]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_compaction_basic() {
        // Full vocab of 100, but only tokens 5, 10, 20 appear
        let samples: Vec<Vec<u16>> = vec![vec![5, 10, 20], vec![10, 5]];
        let sample_refs: Vec<&[u16]> = samples.iter().map(|s| s.as_slice()).collect();
        let compactor = VocabCompactor::from_dataset(sample_refs, 100);

        assert_eq!(compactor.compact_vocab(), 3);
        assert_eq!(compactor.to_compact(5), Some(0));
        assert_eq!(compactor.to_compact(10), Some(1));
        assert_eq!(compactor.to_compact(20), Some(2));
        assert_eq!(compactor.to_compact(0), None);
        assert_eq!(compactor.to_full(0), 5);
        assert_eq!(compactor.to_full(1), 10);
        assert_eq!(compactor.to_full(2), 20);
    }

    #[test]
    fn test_compact_tokens() {
        let samples: Vec<Vec<u16>> = vec![vec![1, 3, 5]];
        let sample_refs: Vec<&[u16]> = samples.iter().map(|s| s.as_slice()).collect();
        let compactor = VocabCompactor::from_dataset(sample_refs, 10);

        let compacted = compactor.compact_tokens(&[1, 3, 5, 1]);
        assert_eq!(compacted, vec![0, 1, 2, 0]);
    }

    #[test]
    fn test_extract_scatter_roundtrip() {
        let dim = 4;
        let full_vocab = 8;
        let samples: Vec<Vec<u16>> = vec![vec![1, 3]];
        let sample_refs: Vec<&[u16]> = samples.iter().map(|s| s.as_slice()).collect();
        let compactor = VocabCompactor::from_dataset(sample_refs, full_vocab);

        // Full embedding: row[i] = [i, i, i, i]
        let full_embed: Vec<f32> = (0..full_vocab * dim).map(|i| (i / dim) as f32).collect();

        // Extract compact embedding
        let compact = compactor.extract_compact_embedding(&full_embed, dim);
        assert_eq!(compact.len(), 2 * dim); // tokens 1 and 3
        assert_eq!(&compact[..dim], &[1.0, 1.0, 1.0, 1.0]); // token 1
        assert_eq!(&compact[dim..], &[3.0, 3.0, 3.0, 3.0]); // token 3

        // Scatter gradients
        let compact_grad = vec![0.5; 2 * dim];
        let mut full_grad = vec![0.0f32; full_vocab * dim];
        compactor.scatter_gradients(&mut full_grad, &compact_grad, dim);

        // Only rows 1 and 3 should have gradients
        assert_eq!(&full_grad[0..dim], &[0.0; 4]); // row 0
        assert_eq!(&full_grad[dim..2 * dim], &[0.5; 4]); // row 1
        assert_eq!(&full_grad[2 * dim..3 * dim], &[0.0; 4]); // row 2
        assert_eq!(&full_grad[3 * dim..4 * dim], &[0.5; 4]); // row 3
    }
}
