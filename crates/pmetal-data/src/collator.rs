//! Data collation utilities.

use super::Sample;

/// Data collator for creating batches.
pub struct DataCollator {
    /// Padding token ID.
    pub pad_token_id: u32,
    /// Maximum sequence length.
    pub max_length: usize,
}

impl DataCollator {
    /// Create a new data collator.
    pub fn new(pad_token_id: u32, max_length: usize) -> Self {
        Self {
            pad_token_id,
            max_length,
        }
    }

    /// Collate samples into a batch.
    pub fn collate(&self, samples: &[Sample]) -> CollatedBatch {
        let batch_size = samples.len();
        let max_len = samples
            .iter()
            .map(|s| s.input_ids.len().min(self.max_length))
            .max()
            .unwrap_or(0);

        let mut input_ids = vec![vec![self.pad_token_id; max_len]; batch_size];
        let mut attention_mask = vec![vec![0u32; max_len]; batch_size];

        for (i, sample) in samples.iter().enumerate() {
            let len = sample.input_ids.len().min(self.max_length);
            input_ids[i][..len].copy_from_slice(&sample.input_ids[..len]);
            attention_mask[i][..len].fill(1);
        }

        CollatedBatch {
            input_ids,
            attention_mask,
            batch_size,
            seq_len: max_len,
        }
    }
}

/// A collated batch ready for the model.
#[derive(Debug, Clone)]
pub struct CollatedBatch {
    /// Input token IDs [batch_size, seq_len].
    pub input_ids: Vec<Vec<u32>>,
    /// Attention mask [batch_size, seq_len].
    pub attention_mask: Vec<Vec<u32>>,
    /// Batch size.
    pub batch_size: usize,
    /// Sequence length.
    pub seq_len: usize,
}
