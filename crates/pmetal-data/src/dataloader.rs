//! DataLoader for creating training batches.

use mlx_rs::Array;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::sync::Arc;

use super::{Sample, TrainingDataset};
use crate::image_processing::MllamaImageProcessor;

/// Errors produced while constructing a training batch.
#[derive(Debug, thiserror::Error)]
pub enum DataLoaderError {
    /// Failed to convert batch contents into MLX arrays.
    #[error("failed to create MLX arrays for batch: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// Failed to preprocess an image for a multimodal sample.
    #[error("failed to preprocess image for sample {sample_index} at {path}: {source}")]
    ImagePreprocess {
        /// Index of the failing sample within the batch.
        sample_index: usize,
        /// Path of the image that failed preprocessing.
        path: std::path::PathBuf,
        /// Underlying MLX/image-processing failure.
        source: mlx_rs::error::Exception,
    },
    /// Multimodal batches require image data for every sample.
    #[error("sample {sample_index} is missing images in a multimodal batch")]
    MissingImages {
        /// Index of the sample missing image data.
        sample_index: usize,
    },
}

/// A batch of training data ready for the model.
#[derive(Debug)]
pub struct TrainingBatch {
    /// Input token IDs [batch_size, seq_len].
    pub input_ids: Array,
    /// Labels for loss computation [batch_size, seq_len].
    pub labels: Array,
    /// Attention mask [batch_size, seq_len].
    pub attention_mask: Array,
    /// Pixel values for vision models [batch_size, channels, height, width].
    pub pixel_values: Option<Array>,
    /// Number of samples in this batch.
    pub batch_size: usize,
    /// Sequence length.
    pub seq_len: usize,
}

/// Configuration for the DataLoader.
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Whether to shuffle the data.
    pub shuffle: bool,
    /// Random seed for shuffling.
    pub seed: u64,
    /// Padding token ID.
    pub pad_token_id: u32,
    /// Whether to drop the last incomplete batch.
    pub drop_last: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            max_seq_len: 2048,
            shuffle: true,
            seed: 42,
            pad_token_id: 0,
            drop_last: false,
        }
    }
}

/// DataLoader that yields batches from a dataset.
pub struct DataLoader {
    /// The dataset.
    dataset: TrainingDataset,
    /// Configuration.
    config: DataLoaderConfig,
    /// Current index permutation.
    indices: Vec<usize>,
    /// Current position in the dataset.
    position: usize,
    /// Optional image processor for multimodal data.
    image_processor: Option<Arc<MllamaImageProcessor>>,
}

#[derive(Debug)]
struct BatchComponents {
    input_ids_flat: Vec<i32>,
    labels_flat: Vec<i64>,
    attention_mask_flat: Vec<i32>,
    pixel_tensors: Vec<Array>,
    batch_size: usize,
    seq_len: usize,
}

impl DataLoader {
    /// Create a new DataLoader.
    pub fn new(
        dataset: TrainingDataset,
        config: DataLoaderConfig,
        image_processor: Option<Arc<MllamaImageProcessor>>,
    ) -> Self {
        let n = dataset.len();
        let mut indices: Vec<usize> = (0..n).collect();

        if config.shuffle {
            let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
            indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            config,
            indices,
            position: 0,
            image_processor,
        }
    }

    /// Reset the DataLoader for a new epoch.
    pub fn reset(&mut self, new_seed: Option<u64>) {
        self.position = 0;
        if self.config.shuffle {
            let seed = new_seed.unwrap_or(self.config.seed);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get the number of batches.
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.config.drop_last {
            n / self.config.batch_size
        } else {
            n.div_ceil(self.config.batch_size)
        }
    }

    /// Get the total number of samples.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Check if the loader is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Get the next batch.
    pub fn next_batch(&mut self) -> Option<TrainingBatch> {
        match self.try_next_batch() {
            Ok(batch) => batch,
            Err(err) => {
                tracing::error!("Dropping invalid batch: {err}");
                None
            }
        }
    }

    /// Get the next batch with explicit error reporting.
    pub fn try_next_batch(&mut self) -> Result<Option<TrainingBatch>, DataLoaderError> {
        if self.position >= self.indices.len() {
            return Ok(None);
        }

        let batch_end = (self.position + self.config.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.position..batch_end];

        // Check if we should drop incomplete batch
        if self.config.drop_last && batch_indices.len() < self.config.batch_size {
            return Ok(None);
        }

        let batch = self.create_batch(batch_indices)?;
        self.position = batch_end;

        Ok(Some(batch))
    }

    /// Create a batch from sample indices.
    fn build_batch_components(&self, indices: &[usize]) -> Result<BatchComponents, DataLoaderError> {
        let batch_size = indices.len();

        // Collect samples
        let samples: Vec<&Sample> = indices
            .iter()
            .filter_map(|&i| self.dataset.get(i))
            .collect();

        // Find max sequence length in this batch
        let max_len = samples
            .iter()
            .map(|s| s.input_ids.len().min(self.config.max_seq_len))
            .max()
            .unwrap_or(1);

        // Create padded arrays
        let mut input_ids_flat = Vec::with_capacity(batch_size * max_len);
        let mut labels_flat = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_flat = Vec::with_capacity(batch_size * max_len);

        // Image processing
        let mut pixel_tensors = Vec::new();

        if let Some(ref processor) = self.image_processor {
            // Check if any sample has images - if so, ALL must have images
            let any_has_images = samples.iter().any(|s| s.images.is_some());
            if any_has_images {
                for (idx, sample) in samples.iter().enumerate() {
                    match &sample.images {
                        Some(images) if !images.is_empty() => {
                            let path = &images[0];
                            match processor.preprocess(path) {
                                Ok(tensor) => pixel_tensors.push(tensor),
                                Err(e) => {
                                    return Err(DataLoaderError::ImagePreprocess {
                                        sample_index: idx,
                                        path: path.clone(),
                                        source: e,
                                    });
                                }
                            }
                        }
                        _ => {
                            return Err(DataLoaderError::MissingImages { sample_index: idx });
                        }
                    }
                }
            }
        }

        for sample in &samples {
            let seq_len = sample.input_ids.len().min(self.config.max_seq_len);

            // Input IDs
            input_ids_flat.extend(sample.input_ids.iter().take(seq_len).map(|&x| x as i32));
            input_ids_flat.extend(std::iter::repeat_n(
                self.config.pad_token_id as i32,
                max_len - seq_len,
            ));

            // Labels
            if let Some(ref labels) = sample.labels {
                labels_flat.extend(labels.iter().take(seq_len).copied());
                labels_flat.extend(std::iter::repeat_n(-100_i64, max_len - seq_len));
            } else {
                // No labels - use input_ids shifted (causal LM)
                labels_flat.extend(sample.input_ids.iter().take(seq_len).map(|&x| x as i64));
                labels_flat.extend(std::iter::repeat_n(-100_i64, max_len - seq_len));
            }

            // Attention mask
            attention_mask_flat.extend(std::iter::repeat_n(1_i32, seq_len));
            attention_mask_flat.extend(std::iter::repeat_n(0_i32, max_len - seq_len));
        }

        Ok(BatchComponents {
            input_ids_flat,
            labels_flat,
            attention_mask_flat,
            pixel_tensors,
            batch_size,
            seq_len: max_len,
        })
    }

    fn create_batch(&self, indices: &[usize]) -> Result<TrainingBatch, DataLoaderError> {
        let components = self.build_batch_components(indices)?;

        let input_ids = Array::from_slice(
            &components.input_ids_flat,
            &[components.batch_size as i32, components.seq_len as i32],
        );
        let labels = Array::from_slice(
            &components.labels_flat,
            &[components.batch_size as i32, components.seq_len as i32],
        );
        let attention_mask = Array::from_slice(
            &components.attention_mask_flat,
            &[components.batch_size as i32, components.seq_len as i32],
        );

        let pixel_values = if !components.pixel_tensors.is_empty() {
            mlx_rs::ops::concatenate(&components.pixel_tensors).ok()
        } else {
            None
        };

        Ok(TrainingBatch {
            input_ids,
            labels,
            attention_mask,
            pixel_values,
            batch_size: components.batch_size,
            seq_len: components.seq_len,
        })
    }
}

impl Iterator for DataLoader {
    type Item = TrainingBatch;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

/// Create a simple iterator over (input_ids, labels) tuples for the training loop.
pub fn create_batch_iterator(
    dataset: TrainingDataset,
    batch_size: usize,
    max_seq_len: usize,
    pad_token_id: u32,
    shuffle: bool,
    seed: u64,
) -> impl Iterator<Item = (Array, Array)> {
    let config = DataLoaderConfig {
        batch_size,
        max_seq_len,
        shuffle,
        seed,
        pad_token_id,
        drop_last: false,
    };

    DataLoader::new(dataset, config, None).map(|batch| (batch.input_ids, batch.labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader_basic() {
        let samples: Vec<Sample> = (0..10)
            .map(|i| Sample::new(vec![i as u32, i as u32 + 1, i as u32 + 2]))
            .collect();
        let dataset = TrainingDataset::from_samples(samples);

        let config = DataLoaderConfig {
            batch_size: 3,
            max_seq_len: 10,
            shuffle: false,
            ..Default::default()
        };

        let loader = DataLoader::new(dataset, config, None);

        // Should have 4 batches (10 / 3 = 3 full + 1 partial)
        assert_eq!(loader.num_batches(), 4);
        let batch1 = loader.build_batch_components(&loader.indices[0..3]).unwrap();
        assert_eq!(batch1.batch_size, 3);
        assert_eq!(batch1.seq_len, 3);

        let batch2 = loader.build_batch_components(&loader.indices[3..6]).unwrap();
        assert_eq!(batch2.batch_size, 3);

        let batch3 = loader.build_batch_components(&loader.indices[6..9]).unwrap();
        assert_eq!(batch3.batch_size, 3);

        let batch4 = loader.build_batch_components(&loader.indices[9..10]).unwrap();
        assert_eq!(batch4.batch_size, 1); // Last incomplete batch
    }

    #[test]
    fn test_dataloader_drop_last() {
        let samples: Vec<Sample> = (0..10).map(|i| Sample::new(vec![i as u32])).collect();
        let dataset = TrainingDataset::from_samples(samples);

        let config = DataLoaderConfig {
            batch_size: 3,
            drop_last: true,
            shuffle: false,
            ..Default::default()
        };

        let loader = DataLoader::new(dataset, config, None);

        // Should only have 3 full batches (drops the last 1 sample)
        assert_eq!(loader.num_batches(), 3);
    }

    #[test]
    fn test_dataloader_iterator() {
        let samples: Vec<Sample> = (0..6)
            .map(|i| Sample::new(vec![i as u32, i as u32 + 10]))
            .collect();
        let dataset = TrainingDataset::from_samples(samples);

        let config = DataLoaderConfig {
            batch_size: 2,
            shuffle: false,
            ..Default::default()
        };

        let loader = DataLoader::new(dataset, config, None);
        assert_eq!(loader.num_batches(), 3);
    }

    #[test]
    fn test_dataloader_padding() {
        let samples = vec![
            Sample::new(vec![1, 2, 3]),
            Sample::new(vec![4, 5]), // Shorter sequence
        ];
        let dataset = TrainingDataset::from_samples(samples);

        let config = DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 10,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        };

        let loader = DataLoader::new(dataset, config, None);
        let batch = loader.build_batch_components(&[0, 1]).unwrap();

        assert_eq!(batch.seq_len, 3); // Max length in batch
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.input_ids_flat, vec![1, 2, 3, 4, 5, 0]);
        assert_eq!(batch.attention_mask_flat, vec![1, 1, 1, 1, 1, 0]);
    }

    #[test]
    fn test_batch_iterator() {
        let samples: Vec<Sample> = (0..8).map(|i| Sample::new(vec![i as u32])).collect();
        let dataset = TrainingDataset::from_samples(samples);

        let loader = DataLoader::new(
            dataset,
            DataLoaderConfig {
                batch_size: 2,
                max_seq_len: 10,
                shuffle: false,
                seed: 42,
                pad_token_id: 0,
                drop_last: false,
            },
            None,
        );

        assert_eq!(loader.num_batches(), 4);
    }
}
