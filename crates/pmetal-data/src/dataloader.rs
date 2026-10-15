//! DataLoader for creating training batches.

use pmetal_runtime::compat::{Array, ops as mlx_ops};
use std::sync::Arc;

use super::{IGNORE_INDEX, Sample, TrainingDataset};
use crate::image_processing::FixedSizeImageProcessor;

/// Errors produced while constructing a training batch.
#[derive(Debug, thiserror::Error)]
pub enum DataLoaderError {
    /// Failed to convert batch contents into MLX arrays.
    #[error("failed to create MLX arrays for batch: {0}")]
    Mlx(#[from] pmetal_runtime::compat::Exception),
    /// Failed to preprocess an image for a multimodal sample.
    #[error("failed to preprocess image for sample {sample_index} at {path}: {source}")]
    ImagePreprocess {
        /// Index of the failing sample within the batch.
        sample_index: usize,
        /// Path of the image that failed preprocessing.
        path: std::path::PathBuf,
        /// Underlying MLX/image-processing failure.
        source: pmetal_runtime::compat::Exception,
    },
    /// Multimodal batches require image data for every sample.
    #[error("sample {sample_index} is missing images in a multimodal batch")]
    MissingImages {
        /// Index of the sample missing image data.
        sample_index: usize,
    },
    /// A sample's labels are not one per token, so padding them would shift
    /// every label after the first missing one onto the wrong token.
    #[error("sample {sample_index} has {labels} labels for {tokens} tokens")]
    LabelLength {
        /// Dataset index of the sample.
        sample_index: usize,
        /// Number of tokens.
        tokens: usize,
        /// Number of labels.
        labels: usize,
    },
    /// A [`LoaderState`] that this loader's epoch cannot be at.
    #[error(
        "cannot resume at sample {position} of an epoch of {epoch_len} in batches of {batch_size}"
    )]
    BadState {
        /// Position the state asked for.
        position: usize,
        /// Samples this rank serves per epoch.
        epoch_len: usize,
        /// Batch size of this loader.
        batch_size: usize,
    },
}

/// Where a loader is in its epoch: enough to resume it exactly.
///
/// Take it with the loader's `state()` whenever a checkpoint is written, and
/// hand it to `restore()` on a loader built over the same data and
/// configuration: the rest of the epoch is the one the first run would have
/// served.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LoaderState {
    /// The seed the epoch was started with (the order is a function of it).
    pub epoch_seed: u64,
    /// How many of the epoch's items (samples for [`DataLoader`], packed rows
    /// for [`PackedDataLoader`](crate::PackedDataLoader)) were already served.
    pub position: usize,
}

/// A permutation of `0..n` that depends only on `seed`.
///
/// Fisher-Yates over SplitMix64: written out rather than taken from `rand` so
/// an epoch's order is the same in every build, which a checkpoint taken mid
/// epoch relies on to resume where it stopped.
pub fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        // Multiply-shift range reduction: uniform up to 2^-64 * n.
        let j = ((u128::from(next()) * (i as u128 + 1)) >> 64) as usize;
        indices.swap(i, j);
    }
    indices
}

/// The items one rank serves in an epoch, in order.
///
/// The epoch is `0..n` (shuffled by `seed` when `shuffle`), and rank `rank` of
/// `world_size` takes every `world_size`-th item from its own offset. Each
/// rank gets exactly `n / world_size` items: a rank that served one more would
/// wait forever on a collective its peers never join, so the last
/// `n % world_size` items of the epoch's order sit out (a different few each
/// epoch when shuffled). Fewer items than ranks is the exception: the order
/// wraps so no rank is left with nothing.
pub(crate) fn epoch_order(
    n: usize,
    shuffle: bool,
    seed: u64,
    rank: usize,
    world_size: usize,
) -> Vec<usize> {
    let order = if shuffle {
        shuffled_indices(n, seed)
    } else {
        (0..n).collect()
    };
    if world_size <= 1 {
        return order;
    }
    let share = n / world_size;
    if share == 0 {
        return order.get(rank % n.max(1)).copied().into_iter().collect();
    }
    order
        .into_iter()
        .take(share * world_size)
        .skip(rank)
        .step_by(world_size)
        .collect()
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
    /// Original dataset indices for each sample in the batch.
    pub sample_indices: Vec<usize>,
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
    /// Distributed data sharding: this node's rank (0-indexed).
    /// When set with `world_size > 1`, each node processes every
    /// `world_size`-th sample starting at offset `rank`.
    pub rank: usize,
    /// Distributed data sharding: total number of nodes.
    pub world_size: usize,
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
            rank: 0,
            world_size: 1,
        }
    }
}

/// DataLoader that yields batches from a dataset.
///
/// An epoch's order is a function of the seed it was started with
/// ([`new`](Self::new) uses `config.seed`, [`reset`](Self::reset) the seed it
/// is given), so [`state`](Self::state) and [`restore`](Self::restore) resume
/// a run mid-epoch exactly. Under data parallelism (`config.rank` /
/// `config.world_size`) each rank serves an equal, disjoint share of every
/// epoch.
pub struct DataLoader {
    /// The dataset.
    dataset: TrainingDataset,
    /// Configuration.
    config: DataLoaderConfig,
    /// This rank's samples for the current epoch, in serving order.
    indices: Vec<usize>,
    /// How many of `indices` were already served.
    position: usize,
    /// Seed of the epoch being served.
    epoch_seed: u64,
    /// Optional image processor for multimodal data.
    image_processor: Option<Arc<FixedSizeImageProcessor>>,
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
    /// Create a new DataLoader, its first epoch ordered by `config.seed`.
    pub fn new(
        dataset: TrainingDataset,
        config: DataLoaderConfig,
        image_processor: Option<Arc<FixedSizeImageProcessor>>,
    ) -> Self {
        let mut loader = Self {
            dataset,
            epoch_seed: config.seed,
            config,
            indices: Vec::new(),
            position: 0,
            image_processor,
        };
        loader.start_epoch(loader.config.seed);
        loader
    }

    /// Order the epoch started by `seed` for this rank and rewind to its
    /// first sample.
    fn start_epoch(&mut self, seed: u64) {
        self.epoch_seed = seed;
        self.position = 0;
        // Every rank derives the same permutation from the same seed, then
        // takes its own share of it.
        self.indices = epoch_order(
            self.dataset.len(),
            self.config.shuffle,
            seed,
            self.config.rank,
            self.config.world_size,
        );
    }

    /// Start a new epoch, ordered by `new_seed` (`config.seed` when `None`).
    ///
    /// The order is a function of the seed alone, not of the epochs served
    /// before: a loader built with `config.seed = s` serves what a loader that
    /// was `reset(Some(s))` serves.
    pub fn reset(&mut self, new_seed: Option<u64>) {
        self.start_epoch(new_seed.unwrap_or(self.config.seed));
    }

    /// Where the loader is in its epoch; see [`LoaderState`].
    pub fn state(&self) -> LoaderState {
        LoaderState {
            epoch_seed: self.epoch_seed,
            position: self.position,
        }
    }

    /// Resume at `state`, taken from a loader over the same dataset and
    /// configuration (the same shard, too: a checkpoint is per rank).
    ///
    /// # Errors
    /// If `state.position` is past the end of the epoch, or not on a batch
    /// boundary of this loader's batch size, which means the dataset or the
    /// configuration is not the one the state was taken from.
    pub fn restore(&mut self, state: LoaderState) -> Result<(), DataLoaderError> {
        self.start_epoch(state.epoch_seed);
        let batch_size = self.config.batch_size.max(1);
        if state.position > self.indices.len() || state.position % batch_size != 0 {
            return Err(DataLoaderError::BadState {
                position: state.position,
                epoch_len: self.indices.len(),
                batch_size,
            });
        }
        self.position = state.position;
        Ok(())
    }

    /// Number of batches this loader serves per epoch: this rank's share of
    /// the dataset, in batches.
    pub fn num_batches(&self) -> usize {
        let n = self.indices.len();
        let batch_size = self.config.batch_size.max(1);
        if self.config.drop_last {
            n / batch_size
        } else {
            n.div_ceil(batch_size)
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

        let batch_end = (self.position + self.config.batch_size.max(1)).min(self.indices.len());
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
    fn build_batch_components(
        &self,
        indices: &[usize],
    ) -> Result<BatchComponents, DataLoaderError> {
        let batch_size = indices.len();

        let samples: Vec<&Sample> = indices
            .iter()
            .map(|&i| &self.dataset.samples()[i])
            .collect();
        for (sample, &index) in samples.iter().zip(indices) {
            if let Some(labels) = &sample.labels {
                if labels.len() != sample.input_ids.len() {
                    return Err(DataLoaderError::LabelLength {
                        sample_index: index,
                        tokens: sample.input_ids.len(),
                        labels: labels.len(),
                    });
                }
            }
        }

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
                labels_flat.extend(std::iter::repeat_n(IGNORE_INDEX, max_len - seq_len));
            } else {
                // No labels: train on every token. The loss does the shifting.
                labels_flat.extend(sample.input_ids.iter().take(seq_len).map(|&x| x as i64));
                labels_flat.extend(std::iter::repeat_n(IGNORE_INDEX, max_len - seq_len));
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

        let shape = &[components.batch_size as i32, components.seq_len as i32];
        let input_ids = Array::from_i32_slice(&components.input_ids_flat).reshape(shape);
        let labels_i32: Vec<i32> = components.labels_flat.iter().map(|&x| x as i32).collect();
        let labels = Array::from_i32_slice(&labels_i32).reshape(shape);
        let attention_mask = Array::from_i32_slice(&components.attention_mask_flat).reshape(shape);

        let pixel_values = if !components.pixel_tensors.is_empty() {
            let refs: Vec<&Array> = components.pixel_tensors.iter().collect();
            Some(mlx_ops::concatenate_axis(&refs, 0))
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
            sample_indices: indices.to_vec(),
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
        rank: 0,
        world_size: 1,
    };

    DataLoader::new(dataset, config, None).map(|batch| (batch.input_ids, batch.labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(n: usize) -> TrainingDataset {
        TrainingDataset::from_samples((0..n).map(|i| Sample::new(vec![i as u32 + 1])).collect())
    }

    fn loader(n: usize, config: DataLoaderConfig) -> DataLoader {
        DataLoader::new(numbered(n), config, None)
    }

    /// Every sample the loader serves this epoch, in order.
    fn served(loader: &mut DataLoader) -> Vec<usize> {
        let mut out = Vec::new();
        while let Some(batch) = loader.try_next_batch().unwrap() {
            out.extend(batch.sample_indices);
        }
        out
    }

    /// The same epoch twice from one seed is the same order; different seeds
    /// give different ones; and the order is pinned, because a checkpoint
    /// taken in one build is resumed in another.
    #[test]
    fn an_epoch_order_is_a_pure_function_of_its_seed() {
        assert_eq!(shuffled_indices(10, 42), shuffled_indices(10, 42));
        assert_ne!(shuffled_indices(10, 42), shuffled_indices(10, 43));
        assert_eq!(shuffled_indices(10, 0), [4, 9, 2, 5, 1, 7, 6, 0, 3, 8]);
        let mut sorted = shuffled_indices(1000, 7);
        sorted.sort_unstable();
        assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
        assert!(shuffled_indices(0, 1).is_empty());
        assert_eq!(shuffled_indices(1, 1), [0]);
    }

    /// `reset(Some(s))` lands where a fresh loader seeded `s` starts, whatever
    /// epochs came before.
    #[test]
    fn reset_depends_on_the_seed_alone() {
        let config = |seed| DataLoaderConfig {
            batch_size: 3,
            seed,
            ..Default::default()
        };
        let mut worn = loader(20, config(1));
        served(&mut worn);
        worn.reset(Some(5));
        served(&mut worn);
        worn.reset(Some(9));
        let mut fresh = loader(20, config(9));
        assert_eq!(served(&mut worn), served(&mut fresh));
        worn.reset(None);
        assert_eq!(served(&mut worn), served(&mut loader(20, config(1))));
    }

    /// Ranks serve disjoint shares of equal size: a rank with one more batch
    /// than its peers would wait on a collective they never reach.
    #[test]
    fn ranks_serve_equal_disjoint_shares() {
        for (n, world) in [(10, 3), (11, 4), (12, 4), (7, 7), (100, 8)] {
            let mut union = Vec::new();
            for rank in 0..world {
                let mut l = loader(
                    n,
                    DataLoaderConfig {
                        batch_size: 2,
                        rank,
                        world_size: world,
                        ..Default::default()
                    },
                );
                let share = n / world;
                assert_eq!(l.num_batches(), share.div_ceil(2), "n {n} world {world}");
                let mine = served(&mut l);
                assert_eq!(mine.len(), share, "n {n} world {world} rank {rank}");
                union.extend(mine);
            }
            let count = union.len();
            union.sort_unstable();
            union.dedup();
            assert_eq!(union.len(), count, "a sample went to two ranks");
        }
    }

    /// Without a shuffle the shards partition the dataset, less the remainder.
    #[test]
    fn unshuffled_shards_partition_the_dataset() {
        let mut all = Vec::new();
        for rank in 0..3 {
            let config = DataLoaderConfig {
                shuffle: false,
                rank,
                world_size: 3,
                ..Default::default()
            };
            all.extend(served(&mut loader(9, config)));
        }
        all.sort_unstable();
        assert_eq!(all, (0..9).collect::<Vec<_>>());
    }

    /// Fewer samples than ranks: every rank still gets one.
    #[test]
    fn a_tiny_dataset_still_feeds_every_rank() {
        for rank in 0..4 {
            let config = DataLoaderConfig {
                rank,
                world_size: 4,
                ..Default::default()
            };
            assert_eq!(served(&mut loader(2, config)).len(), 1);
        }
    }

    /// Stop after some batches, save the state, resume in a new loader: the
    /// rest of the epoch is what the first loader would have served.
    #[test]
    fn a_restored_loader_serves_the_rest_of_the_epoch() {
        let config = || DataLoaderConfig {
            batch_size: 4,
            seed: 11,
            rank: 1,
            world_size: 2,
            ..Default::default()
        };
        let mut first = loader(50, config());
        first.reset(Some(12));
        for _ in 0..3 {
            first.try_next_batch().unwrap().unwrap();
        }
        let state = first.state();
        assert_eq!(state.position, 12);
        let rest = served(&mut first);

        let mut resumed = loader(50, config());
        resumed.restore(state).unwrap();
        assert_eq!(served(&mut resumed), rest);

        // The state survives a checkpoint round trip.
        let json = serde_json::to_string(&state).unwrap();
        assert_eq!(serde_json::from_str::<LoaderState>(&json).unwrap(), state);
    }

    #[test]
    fn a_state_from_another_epoch_shape_is_refused() {
        let mut l = loader(
            10,
            DataLoaderConfig {
                batch_size: 4,
                ..Default::default()
            },
        );
        let at = |position| LoaderState {
            epoch_seed: 1,
            position,
        };
        assert!(l.restore(at(11)).is_err(), "past the end");
        assert!(l.restore(at(3)).is_err(), "not on a batch boundary");
        assert!(l.restore(at(8)).is_ok());
    }

    /// Labels that don't line up with the tokens would shift every label
    /// after the gap; that is an error, not a padded batch.
    #[test]
    fn labels_that_do_not_match_the_tokens_are_an_error() {
        let bad = Sample {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            labels: Some(vec![1, 2]),
            images: None,
        };
        let mut l = DataLoader::new(
            TrainingDataset::from_samples(vec![bad]),
            DataLoaderConfig::default(),
            None,
        );
        assert!(matches!(
            l.try_next_batch(),
            Err(DataLoaderError::LabelLength {
                tokens: 3,
                labels: 2,
                ..
            })
        ));
    }

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
        let batch1 = loader
            .build_batch_components(&loader.indices[0..3])
            .unwrap();
        assert_eq!(batch1.batch_size, 3);
        assert_eq!(batch1.seq_len, 3);

        let batch2 = loader
            .build_batch_components(&loader.indices[3..6])
            .unwrap();
        assert_eq!(batch2.batch_size, 3);

        let batch3 = loader
            .build_batch_components(&loader.indices[6..9])
            .unwrap();
        assert_eq!(batch3.batch_size, 3);

        let batch4 = loader
            .build_batch_components(&loader.indices[9..10])
            .unwrap();
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
    fn test_batch_preserves_sample_indices() {
        let samples: Vec<Sample> = (0..4).map(|i| Sample::new(vec![i as u32 + 1])).collect();
        let dataset = TrainingDataset::from_samples(samples);

        let loader = DataLoader::new(
            dataset,
            DataLoaderConfig {
                batch_size: 2,
                shuffle: false,
                ..Default::default()
            },
            None,
        );

        let batch = loader.create_batch(&[3, 1]).unwrap();
        assert_eq!(batch.sample_indices, vec![3, 1]);
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
                ..Default::default()
            },
            None,
        );

        assert_eq!(loader.num_batches(), 4);
    }
}
