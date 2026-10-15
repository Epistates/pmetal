//! The flat parameter space a sharded optimizer divides between ranks.
//!
//! Every parameter is laid end to end, in name order, into one vector of
//! `f32`, and rank `r` owns the `r`-th of `world` contiguous chunks of it
//! ([`chunk_range`]). Dividing the flat vector, not the list of parameters,
//! is what makes the shards equal: one embedding matrix is a third of a
//! model, and handing a whole tensor to one rank leaves that rank holding a
//! third of the optimizer state however many ranks there are. A tensor that
//! straddles a shard boundary is simply updated in two pieces, by two ranks.

use crate::collective::chunk_range;
use crate::error::DistributedError;
use std::collections::HashMap;
use std::ops::Range;

/// Where each parameter sits in the flat vector, and which part is this
/// rank's.
#[derive(Debug, Clone)]
pub struct ZeroLayout {
    /// Parameters in layout order with their element ranges.
    entries: Vec<(String, Range<usize>)>,
    total: usize,
    rank: usize,
    world_size: usize,
}

impl ZeroLayout {
    /// Lay out `params` (name and element count) in name order for rank
    /// `rank` of `world_size`.
    ///
    /// Every rank must pass the same parameters; sorting makes the order
    /// independent of how each rank happened to list them.
    pub fn new(
        params: &[(String, usize)],
        rank: usize,
        world_size: usize,
    ) -> Result<Self, DistributedError> {
        if world_size == 0 || rank >= world_size {
            return Err(DistributedError::Config(format!(
                "rank {rank} outside a world of {world_size}"
            )));
        }
        let mut sorted: Vec<&(String, usize)> = params.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        if let Some(pair) = sorted.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(DistributedError::Config(format!(
                "parameter {:?} listed twice",
                pair[0].0
            )));
        }
        let mut entries = Vec::with_capacity(sorted.len());
        let mut at = 0;
        for (name, len) in sorted {
            entries.push((name.clone(), at..at + len));
            at += len;
        }
        Ok(Self {
            entries,
            total: at,
            rank,
            world_size,
        })
    }

    /// Number of elements across all parameters.
    pub fn total_elements(&self) -> usize {
        self.total
    }

    /// The elements of the flat vector this rank owns.
    pub fn shard_range(&self) -> Range<usize> {
        chunk_range(self.total, self.world_size, self.rank)
    }

    /// The elements rank `rank` owns.
    pub fn shard_range_of(&self, rank: usize) -> Range<usize> {
        chunk_range(self.total, self.world_size, rank)
    }

    /// Which rank owns flat element `index`.
    pub fn owner_of(&self, index: usize) -> usize {
        (0..self.world_size)
            .find(|&r| self.shard_range_of(r).contains(&index))
            .unwrap_or(self.world_size - 1)
    }

    /// Parameter names in layout order, with their ranges in the flat vector.
    pub fn entries(&self) -> &[(String, Range<usize>)] {
        &self.entries
    }

    /// The range of parameter `name` in the flat vector.
    pub fn range_of(&self, name: &str) -> Option<Range<usize>> {
        self.entries
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| r.clone())
    }

    /// Copy parameter values into one flat vector in layout order. Every
    /// laid-out parameter must be present with the right length.
    pub fn flatten(
        &self,
        values: &HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>, DistributedError> {
        let mut flat = Vec::with_capacity(self.total);
        for (name, range) in &self.entries {
            let v = values
                .get(name)
                .ok_or_else(|| DistributedError::Config(format!("missing parameter {name:?}")))?;
            if v.len() != range.len() {
                return Err(DistributedError::Config(format!(
                    "parameter {name:?} has {} elements, the layout expects {}",
                    v.len(),
                    range.len()
                )));
            }
            flat.extend_from_slice(v);
        }
        Ok(flat)
    }

    /// Split a flat vector back into parameters.
    pub fn unflatten(&self, flat: &[f32]) -> Result<HashMap<String, Vec<f32>>, DistributedError> {
        if flat.len() != self.total {
            return Err(DistributedError::Config(format!(
                "flat vector has {} elements, the layout expects {}",
                flat.len(),
                self.total
            )));
        }
        Ok(self
            .entries
            .iter()
            .map(|(name, range)| (name.clone(), flat[range.clone()].to_vec()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Vec<(String, usize)> {
        vec![
            ("b.weight".into(), 100),
            ("a.bias".into(), 3),
            ("c.embed".into(), 997),
        ]
    }

    #[test]
    fn shards_are_equal_however_unequal_the_tensors_are() {
        let world = 4;
        let total: usize = params().iter().map(|p| p.1).sum();
        let mut covered = 0;
        for rank in 0..world {
            let layout = ZeroLayout::new(&params(), rank, world).unwrap();
            let shard = layout.shard_range();
            assert_eq!(shard.start, covered);
            covered = shard.end;
            // Within one element of an equal share, though one tensor is
            // 90% of the parameters.
            assert!(shard.len().abs_diff(total / world) <= 1, "{shard:?}");
        }
        assert_eq!(covered, total);
    }

    #[test]
    fn layout_is_independent_of_listing_order() {
        let mut reversed = params();
        reversed.reverse();
        let a = ZeroLayout::new(&params(), 1, 3).unwrap();
        let b = ZeroLayout::new(&reversed, 1, 3).unwrap();
        assert_eq!(a.entries(), b.entries());
        assert_eq!(a.range_of("a.bias"), Some(0..3));
        assert_eq!(a.range_of("b.weight"), Some(3..103));
    }

    #[test]
    fn flatten_and_unflatten_round_trip() {
        let layout = ZeroLayout::new(&params(), 0, 2).unwrap();
        let values: HashMap<String, Vec<f32>> = params()
            .into_iter()
            .map(|(n, len)| {
                let v = (0..len).map(|i| i as f32 + n.len() as f32).collect();
                (n, v)
            })
            .collect();
        let flat = layout.flatten(&values).unwrap();
        assert_eq!(flat.len(), 1100);
        assert_eq!(layout.unflatten(&flat).unwrap(), values);
    }

    #[test]
    fn bad_inputs_are_errors() {
        assert!(ZeroLayout::new(&params(), 3, 3).is_err());
        assert!(ZeroLayout::new(&params(), 0, 0).is_err());
        let dup = vec![("x".to_string(), 1), ("x".to_string(), 2)];
        assert!(ZeroLayout::new(&dup, 0, 1).is_err());
        let layout = ZeroLayout::new(&params(), 0, 1).unwrap();
        assert!(layout.flatten(&HashMap::new()).is_err());
        assert!(layout.unflatten(&[0.0; 3]).is_err());
    }

    #[test]
    fn owner_of_agrees_with_shard_ranges() {
        let layout = ZeroLayout::new(&params(), 0, 3).unwrap();
        for index in 0..layout.total_elements() {
            let owner = layout.owner_of(index);
            assert!(layout.shard_range_of(owner).contains(&index));
        }
    }
}
