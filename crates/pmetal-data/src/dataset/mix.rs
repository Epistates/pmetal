//! Combining datasets: mixing at set ratios, and dropping duplicates.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher, RandomState};

use pmetal_core::{PMetalError, Result};

use super::{Sample, TrainingDataset};
use crate::dataloader::shuffled_indices;

/// One dataset of a mixture, and its share.
#[derive(Clone)]
pub struct MixSource {
    /// The samples.
    pub dataset: TrainingDataset,
    /// Its share of the mixture, relative to the others' (they need not sum
    /// to one). Positive and finite.
    pub weight: f64,
}

impl MixSource {
    /// `dataset` with the share `weight`.
    pub fn new(dataset: TrainingDataset, weight: f64) -> Self {
        Self { dataset, weight }
    }
}

impl TrainingDataset {
    /// Mix datasets at set ratios into one of `total` samples (or, with
    /// `None`, as many as the ratios allow without repeating a sample of any
    /// source), in a random order fixed by `seed`.
    ///
    /// Each source contributes `weight / sum(weights)` of the mixture to
    /// within one sample (the sources are drawn from by largest shortfall, not
    /// by chance, so a small mixture matches its ratios as closely as a large
    /// one). A source asked for more samples than it has is cycled, in a
    /// fresh shuffled order each pass; one asked for fewer takes a random
    /// subset. Passing a `total` larger than the sources hold is how a
    /// small, valuable dataset is upsampled.
    ///
    /// # Errors
    /// No sources, a weight that is not positive and finite, or an empty
    /// source with a positive weight.
    pub fn mix(sources: Vec<MixSource>, total: Option<usize>, seed: u64) -> Result<Self> {
        if sources.is_empty() {
            return Err(PMetalError::InvalidArgument(
                "nothing to mix: no datasets".into(),
            ));
        }
        for (i, source) in sources.iter().enumerate() {
            if !(source.weight.is_finite() && source.weight > 0.0) {
                return Err(PMetalError::InvalidArgument(format!(
                    "mix weight of dataset {i} is {}, not a positive number",
                    source.weight
                )));
            }
            if source.dataset.is_empty() {
                return Err(PMetalError::InvalidArgument(format!(
                    "dataset {i} of the mixture is empty"
                )));
            }
        }
        let sum: f64 = sources.iter().map(|s| s.weight).sum();
        let shares: Vec<f64> = sources.iter().map(|s| s.weight / sum).collect();
        let total = total.unwrap_or_else(|| {
            sources
                .iter()
                .zip(&shares)
                .map(|(s, share)| (s.dataset.len() as f64 / share).floor() as usize)
                .min()
                .unwrap_or(0)
        });

        // Each source is read through its own shuffled order, reshuffled when
        // it runs out.
        let mut orders: Vec<Vec<usize>> = sources
            .iter()
            .enumerate()
            .map(|(i, s)| shuffled_indices(s.dataset.len(), seed ^ ((i as u64 + 1) << 32)))
            .collect();
        let mut cursor = vec![0usize; sources.len()];
        let mut passes = vec![0u64; sources.len()];
        let mut taken = vec![0usize; sources.len()];
        let mut samples = Vec::with_capacity(total);

        for step in 1..=total {
            // The source furthest below its share after `step` samples.
            let pick = (0..sources.len())
                .max_by(|&a, &b| {
                    let shortfall = |i: usize| shares[i] * step as f64 - taken[i] as f64;
                    shortfall(a).total_cmp(&shortfall(b)).then(b.cmp(&a))
                })
                .expect("there is at least one source");
            if cursor[pick] == orders[pick].len() {
                passes[pick] += 1;
                orders[pick] = shuffled_indices(
                    sources[pick].dataset.len(),
                    seed ^ ((pick as u64 + 1) << 32) ^ passes[pick].wrapping_mul(0x9E37_79B9),
                );
                cursor[pick] = 0;
            }
            samples.push(sources[pick].dataset.samples[orders[pick][cursor[pick]]].clone());
            cursor[pick] += 1;
            taken[pick] += 1;
        }

        let mut mixed = Self::from_samples(samples);
        mixed.shuffle(seed);
        Ok(mixed)
    }

    /// Drop samples identical to an earlier one (same tokens and same
    /// labels), keeping the first of each. Returns how many were dropped.
    ///
    /// Exact duplicates only: near-duplicates (a paraphrase, a boilerplate
    /// variant) are not found.
    pub fn dedup(&mut self) -> usize {
        let state = RandomState::new();
        let hash = |s: &Sample| -> u64 {
            let mut h = state.build_hasher();
            for id in &s.input_ids {
                h.write_u32(*id);
            }
            if let Some(labels) = &s.labels {
                for l in labels {
                    h.write_i64(*l);
                }
            }
            h.finish()
        };
        let same = |a: &Sample, b: &Sample| a.input_ids == b.input_ids && a.labels == b.labels;

        let mut seen: HashMap<u64, Vec<usize>> = HashMap::with_capacity(self.samples.len());
        let mut keep = vec![true; self.samples.len()];
        for (i, sample) in self.samples.iter().enumerate() {
            let bucket = seen.entry(hash(sample)).or_default();
            if bucket.iter().any(|&j| same(&self.samples[j], sample)) {
                keep[i] = false;
            } else {
                bucket.push(i);
            }
        }
        let dropped = keep.iter().filter(|k| !**k).count();
        if dropped > 0 {
            let mut flags = keep.iter();
            self.samples.retain(|_| *flags.next().unwrap_or(&true));
            if self.original_lengths.len() == keep.len() {
                let mut flags = keep.iter();
                self.original_lengths
                    .retain(|_| *flags.next().unwrap_or(&true));
            }
        }
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tagged(tag: u32, n: usize) -> TrainingDataset {
        TrainingDataset::from_samples((0..n).map(|i| Sample::new(vec![tag, i as u32])).collect())
    }

    fn counts(mixed: &TrainingDataset, sources: usize) -> Vec<usize> {
        let mut counts = vec![0; sources];
        for s in mixed.samples() {
            counts[s.input_ids[0] as usize] += 1;
        }
        counts
    }

    #[test]
    fn a_mixture_matches_its_ratios_to_within_a_sample() {
        for total in [7, 10, 100, 1001] {
            let mixed = TrainingDataset::mix(
                vec![
                    MixSource::new(tagged(0, 500), 5.0),
                    MixSource::new(tagged(1, 500), 3.0),
                    MixSource::new(tagged(2, 500), 2.0),
                ],
                Some(total),
                4,
            )
            .unwrap();
            assert_eq!(mixed.len(), total);
            let counts = counts(&mixed, 3);
            for (c, share) in counts.iter().zip([0.5, 0.3, 0.2]) {
                assert!(
                    (*c as f64 - share * total as f64).abs() <= 1.0,
                    "{counts:?} of {total}"
                );
            }
        }
    }

    #[test]
    fn a_small_source_is_cycled_and_a_big_one_sampled_without_repeats() {
        let mixed = TrainingDataset::mix(
            vec![
                MixSource::new(tagged(0, 1000), 1.0),
                MixSource::new(tagged(1, 4), 1.0),
            ],
            Some(40),
            9,
        )
        .unwrap();
        assert_eq!(counts(&mixed, 2), [20, 20]);
        // The big source's 20 are distinct samples; the small one repeats.
        let mut big: Vec<u32> = mixed
            .samples()
            .iter()
            .filter(|s| s.input_ids[0] == 0)
            .map(|s| s.input_ids[1])
            .collect();
        big.sort_unstable();
        big.dedup();
        assert_eq!(big.len(), 20);
        // Every pass over the small source uses each of its samples once.
        let mut small = [0usize; 4];
        for s in mixed.samples().iter().filter(|s| s.input_ids[0] == 1) {
            small[s.input_ids[1] as usize] += 1;
        }
        assert_eq!(small, [5, 5, 5, 5]);
    }

    #[test]
    fn by_default_nothing_is_repeated() {
        let mixed = TrainingDataset::mix(
            vec![
                MixSource::new(tagged(0, 30), 2.0),
                MixSource::new(tagged(1, 100), 1.0),
            ],
            None,
            1,
        )
        .unwrap();
        // 30 / (2/3) = 45 samples: 30 from the first, 15 from the second.
        assert_eq!(counts(&mixed, 2), [30, 15]);
    }

    #[test]
    fn a_mixture_is_a_function_of_its_seed() {
        let make = |seed| {
            TrainingDataset::mix(
                vec![
                    MixSource::new(tagged(0, 50), 1.0),
                    MixSource::new(tagged(1, 50), 1.0),
                ],
                Some(60),
                seed,
            )
            .unwrap()
        };
        let ids = |d: &TrainingDataset| -> Vec<Vec<u32>> {
            d.samples().iter().map(|s| s.input_ids.clone()).collect()
        };
        assert_eq!(ids(&make(3)), ids(&make(3)));
        assert_ne!(ids(&make(3)), ids(&make(4)));
    }

    #[test]
    fn bad_mixtures_are_refused() {
        let ok = || MixSource::new(tagged(0, 5), 1.0);
        assert!(TrainingDataset::mix(vec![], Some(1), 0).is_err());
        assert!(TrainingDataset::mix(vec![MixSource::new(tagged(0, 5), 0.0)], Some(1), 0).is_err());
        assert!(
            TrainingDataset::mix(vec![MixSource::new(tagged(0, 5), f64::NAN)], Some(1), 0).is_err()
        );
        assert!(
            TrainingDataset::mix(
                vec![ok(), MixSource::new(TrainingDataset::new(), 1.0)],
                Some(1),
                0
            )
            .is_err()
        );
    }

    #[test]
    fn dedup_drops_exact_repeats_and_keeps_the_first() {
        let sample = |ids: &[u32], labels: Option<&[i64]>| Sample {
            input_ids: ids.to_vec(),
            labels: labels.map(<[i64]>::to_vec),
            images: None,
        };
        let mut ds = TrainingDataset {
            samples: vec![
                sample(&[1, 2], Some(&[-100, 2])),
                sample(&[3], None),
                sample(&[1, 2], Some(&[-100, 2])), // repeat
                sample(&[1, 2], Some(&[1, 2])),    // same tokens, other mask: kept
                sample(&[3], None),                // repeat
            ],
            original_lengths: vec![2, 1, 2, 2, 1],
        };
        assert_eq!(ds.dedup(), 2);
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.original_lengths, [2, 1, 2]);
        assert_eq!(ds.get(2).unwrap().labels, Some(vec![1, 2]));
        assert_eq!(ds.dedup(), 0);
    }
}
