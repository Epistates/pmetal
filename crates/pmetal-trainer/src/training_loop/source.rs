//! Where a run's micro-batches come from: one source for padded batches and
//! one for packed rows, behind the one interface the loop reads them through.
//!
//! Both serve this rank's share of the data when training is data-parallel
//! (every rank the same number of batches, disjoint samples), and both can say
//! where they are in the epoch ([`LoaderState`]) and be put back there, which
//! is what a mid-epoch resume needs.

use pmetal_data::{
    DataLoader, DataLoaderConfig, LoaderState, PackedDataLoader, PackedTrainingBatch, PackerConfig,
    TrainingBatch, TrainingDataset, compute_pack_seq_len,
};
use pmetal_runtime::compat::Exception;
use std::sync::Arc;

use crate::{Result, SftError};

/// One micro-batch.
#[derive(Debug)]
pub(crate) enum MicroBatch {
    /// Rows padded to the longest in the batch, with an optional vision input.
    Padded(TrainingBatch),
    /// Documents packed end to end into one row, with position ids and a
    /// block-diagonal mask that keep them apart.
    Packed(PackedTrainingBatch),
}

/// A borrowed view of a micro-batch, which is what a step reads.
#[derive(Debug, Clone, Copy)]
pub(crate) enum BatchView<'a> {
    /// See [`MicroBatch::Padded`].
    Padded(&'a TrainingBatch),
    /// See [`MicroBatch::Packed`].
    Packed(&'a PackedTrainingBatch),
}

impl MicroBatch {
    /// This batch, borrowed.
    pub(crate) fn view(&self) -> BatchView<'_> {
        match self {
            Self::Padded(b) => BatchView::Padded(b),
            Self::Packed(b) => BatchView::Packed(b),
        }
    }
}

impl BatchView<'_> {
    /// Tokens the batch holds, padding included: what throughput counts.
    pub(crate) fn tokens(&self) -> usize {
        match self {
            Self::Padded(b) => b.batch_size.saturating_mul(b.seq_len),
            Self::Packed(b) => b.total_tokens,
        }
    }
}

/// A run's data.
pub(crate) enum BatchSource {
    /// Padded batches from a [`DataLoader`].
    Padded(Box<DataLoader>),
    /// Packed rows from a [`PackedDataLoader`].
    Packed(Box<PackedDataLoader>),
}

impl BatchSource {
    /// Padded batches of `dataloader`'s shape over `dataset`.
    pub(crate) fn padded(dataset: Arc<TrainingDataset>, dataloader: &DataLoaderConfig) -> Self {
        Self::Padded(Box::new(DataLoader::new(dataset, dataloader.clone(), None)))
    }

    /// Packed rows over `dataset`, `world_size` ranks sharing them.
    ///
    /// The packing length is `pack_max_seq_len` when the caller sets it, else
    /// the p99 of the samples' lengths rounded up to a power of two and capped
    /// at the data loader's own maximum: a length the model's architectural
    /// maximum would set makes attention cost for short samples the dominant
    /// term. Returns the loader and the packing's statistics.
    pub(crate) fn packed(
        dataset: &TrainingDataset,
        dataloader: &DataLoaderConfig,
        pack_max_seq_len: Option<usize>,
    ) -> Result<(Self, pmetal_data::PackingStats)> {
        let samples = dataset.samples();
        let max_seq_len = match pack_max_seq_len {
            Some(explicit) => {
                tracing::info!(
                    "Using explicit pack_max_seq_len={explicit} (overrides the p99 heuristic)"
                );
                explicit
            }
            None => {
                let lengths: Vec<usize> = samples.iter().map(|s| s.input_ids.len()).collect();
                compute_pack_seq_len(&lengths, dataloader.max_seq_len)
            }
        };
        // Long samples are truncated to the packing length, not skipped.
        let packer = PackerConfig::with_max_length(max_seq_len)
            .with_max_seq_length(max_seq_len)
            .mask_boundaries(true);
        let loader = PackedDataLoader::new(samples, packer, dataloader.shuffle, dataloader.seed)
            .and_then(|l| l.with_sharding(dataloader.rank, dataloader.world_size.max(1)))
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
        let stats = loader.stats();
        Ok((Self::Packed(Box::new(loader)), stats))
    }

    /// Start the epoch ordered by `seed`.
    pub(crate) fn reset(&mut self, seed: u64) {
        match self {
            Self::Padded(l) => l.reset(Some(seed)),
            Self::Packed(l) => l.reset(Some(seed)),
        }
    }

    /// Micro-batches this rank serves per epoch.
    pub(crate) fn num_batches(&self) -> usize {
        match self {
            Self::Padded(l) => l.num_batches(),
            Self::Packed(l) => l.num_batches(),
        }
    }

    /// The next micro-batch of the epoch, or `None` at its end. A corrupt
    /// sample is an error, not the end of the epoch.
    pub(crate) fn next(&mut self) -> Result<Option<MicroBatch>> {
        match self {
            Self::Padded(l) => Ok(l
                .try_next_batch()
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
                .map(MicroBatch::Padded)),
            Self::Packed(l) => l
                .next_batch()
                .transpose()
                .map(|b| b.map(MicroBatch::Packed))
                .map_err(SftError::Mlx),
        }
    }

    /// Where the source is in its epoch.
    pub(crate) fn state(&self) -> LoaderState {
        match self {
            Self::Padded(l) => l.state(),
            Self::Packed(l) => l.state(),
        }
    }

    /// Continue from `state`, taken from a source over the same data, in the
    /// same layout, for the same rank.
    pub(crate) fn restore(&mut self, state: LoaderState) -> Result<()> {
        let result = match self {
            Self::Padded(l) => l.restore(state).map_err(|e| e.to_string()),
            Self::Packed(l) => l.restore(state).map_err(|e| e.to_string()),
        };
        result.map_err(|e| SftError::Mlx(Exception::custom(format!("cannot resume the data: {e}"))))
    }
}
