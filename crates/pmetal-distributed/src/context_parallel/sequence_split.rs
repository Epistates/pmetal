//! Sequence partitioning utilities for context parallelism.

use crate::collective::chunk_range;
use crate::mlx_dist::group::DistributedGroup;
use crate::mlx_dist::ops;
use pmetal_runtime::compat::{Array, Exception, ops as mlx_ops};

/// How a sequence of `total` tokens is cut between `world` ranks.
///
/// With causal attention a token only sees earlier ones, so under a
/// contiguous cut the last rank (whose tokens see everything) has about twice
/// the work of an average rank and the first about none. The zig-zag cut
/// (Brandon et al., "Striped Attention"; the layout used by Llama-3-scale
/// context parallelism) gives each rank one early and one late piece, so every
/// rank does the same work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SequenceLayout {
    /// Rank `r` holds the `r`-th contiguous run of tokens.
    #[default]
    Contiguous,
    /// The sequence is cut into `2 * world` runs and rank `r` holds run `r`
    /// followed by run `2 * world - 1 - r`.
    ZigZag,
}

impl SequenceLayout {
    /// Global positions of the tokens rank `rank` holds, in the order they
    /// sit in its local tensor.
    pub fn local_positions(self, total: usize, rank: usize, world: usize) -> Vec<u32> {
        let runs = match self {
            Self::Contiguous => vec![chunk_range(total, world, rank)],
            Self::ZigZag => vec![
                chunk_range(total, 2 * world, rank),
                chunk_range(total, 2 * world, 2 * world - 1 - rank),
            ],
        };
        runs.into_iter().flatten().map(|p| p as u32).collect()
    }

    /// Number of tokens rank `rank` holds.
    pub fn local_len(self, total: usize, rank: usize, world: usize) -> usize {
        match self {
            Self::Contiguous => chunk_range(total, world, rank).len(),
            Self::ZigZag => {
                chunk_range(total, 2 * world, rank).len()
                    + chunk_range(total, 2 * world, 2 * world - 1 - rank).len()
            }
        }
    }
}

fn position_index(positions: &[u32]) -> Array {
    let rows: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    Array::from_i32_slice(&rows)
}

/// Take rank `rank`'s share of `x` along `seq_axis` under `layout`.
pub fn split_sequence_with_layout(
    x: &Array,
    seq_axis: usize,
    layout: SequenceLayout,
    rank: usize,
    world: usize,
) -> Result<Array, Exception> {
    let shape = x.shape();
    if seq_axis >= shape.len() {
        return Err(Exception::custom(format!(
            "split_sequence: seq_axis {seq_axis} out of bounds for shape {shape:?}"
        )));
    }
    let total = shape[seq_axis] as usize;
    let positions = layout.local_positions(total, rank, world);
    if positions.is_empty() {
        return Err(Exception::custom(format!(
            "split_sequence: {total} tokens leave rank {rank} of {world} with none"
        )));
    }
    Ok(x.take_axis(&position_index(&positions), seq_axis as i32))
}

/// Split a tensor along the sequence dimension across ranks, contiguously.
///
/// Input `x` has shape `[B, S, D]` (or `[B, H, S, D]` for multi-head).
/// Each rank gets a contiguous chunk of about `S / world_size`, the first
/// `S % world_size` ranks one token more.
///
/// # Arguments
///
/// * `x` — Full tensor to split
/// * `seq_axis` — Which axis is the sequence dimension (typically 1 or 2)
/// * `rank` — This rank's index
/// * `world_size` — Total number of ranks
pub fn split_sequence(
    x: &Array,
    seq_axis: usize,
    rank: usize,
    world_size: usize,
) -> Result<Array, Exception> {
    split_sequence_with_layout(x, seq_axis, SequenceLayout::Contiguous, rank, world_size)
}

/// Reassemble the full sequence on every rank from each rank's share.
///
/// `local` is this rank's share along `seq_axis` under `layout` of a sequence
/// of `total` tokens. Shares may differ in length by one token, which
/// `all_gather` cannot carry, so they are padded to the longest, gathered, and
/// the padding dropped and the tokens put back in position order.
pub fn gather_sequence(
    local: &Array,
    seq_axis: usize,
    layout: SequenceLayout,
    total: usize,
    group: &DistributedGroup,
) -> Result<Array, Exception> {
    let world = group.size() as usize;
    let rank = group.rank() as usize;
    let ndim = local.ndim() as usize;
    if seq_axis >= ndim {
        return Err(Exception::custom(format!(
            "gather_sequence: seq_axis {seq_axis} out of bounds for {ndim} dimensions"
        )));
    }
    let lens: Vec<usize> = (0..world)
        .map(|r| layout.local_len(total, r, world))
        .collect();
    if local.shape()[seq_axis] as usize != lens[rank] {
        return Err(Exception::custom(format!(
            "gather_sequence: rank {rank} holds {} tokens along axis {seq_axis}, layout says {}",
            local.shape()[seq_axis],
            lens[rank]
        )));
    }
    let longest = *lens.iter().max().expect("world >= 1");

    // `all_gather` joins along axis 0, so bring the sequence axis there.
    let mut to_front: Vec<i32> = (0..ndim as i32).collect();
    to_front.swap(0, seq_axis);
    let moved = local.transpose_axes(&to_front);
    let mut padded = moved.clone();
    if lens[rank] < longest {
        let mut widths = vec![(0, 0); ndim];
        widths[0] = (0, (longest - lens[rank]) as i32);
        padded = mlx_ops::pad(&moved, &widths, None, Some(0.0));
    }
    let gathered = ops::all_gather(&padded, Some(group))?;

    // Row `r * longest + i` is rank `r`'s `i`-th token, which belongs at
    // position `positions[r][i]`.
    let mut source_of_position = vec![0i32; total];
    for r in 0..world {
        for (i, &p) in layout.local_positions(total, r, world).iter().enumerate() {
            source_of_position[p as usize] = (r * longest + i) as i32;
        }
    }
    let ordered = gathered.take_axis(&Array::from_i32_slice(&source_of_position), 0);
    Ok(ordered.transpose_axes(&to_front))
}

/// Compute the local sequence range for a rank.
///
/// Returns `(start, length)` for this rank's chunk of a sequence of total length `seq_len`.
pub fn local_seq_range(seq_len: usize, rank: usize, world_size: usize) -> (usize, usize) {
    let range = chunk_range(seq_len, world_size, rank);
    (range.start, range.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sequence_even() {
        // [1, 8, 4] → split seq_axis=1 into 2 → [1, 4, 4] each
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let x = Array::from_f32_slice(&data, &[1, 8, 4]);

        let chunk0 = split_sequence(&x, 1, 0, 2).unwrap();
        let chunk1 = split_sequence(&x, 1, 1, 2).unwrap();

        assert_eq!(chunk0.shape(), &[1, 4, 4]);
        assert_eq!(chunk1.shape(), &[1, 4, 4]);
    }

    #[test]
    fn split_sequence_uneven() {
        // [1, 7, 2] → split into 3 → [1,3,2], [1,2,2], [1,2,2]
        let data: Vec<f32> = (0..14).map(|i| i as f32).collect();
        let x = Array::from_f32_slice(&data, &[1, 7, 2]);

        let chunk0 = split_sequence(&x, 1, 0, 3).unwrap();
        let chunk1 = split_sequence(&x, 1, 1, 3).unwrap();
        let chunk2 = split_sequence(&x, 1, 2, 3).unwrap();

        assert_eq!(chunk0.shape(), &[1, 3, 2]); // gets remainder
        assert_eq!(chunk1.shape(), &[1, 2, 2]);
        assert_eq!(chunk2.shape(), &[1, 2, 2]);
    }

    #[test]
    fn local_seq_range_covers_all() {
        let seq_len = 100;
        let world_size = 3;
        let mut total = 0;
        let mut prev_end = 0;

        for rank in 0..world_size {
            let (start, len) = local_seq_range(seq_len, rank, world_size);
            assert_eq!(
                start, prev_end,
                "rank {rank} start must follow previous end"
            );
            total += len;
            prev_end = start + len;
        }

        assert_eq!(total, seq_len);
        assert_eq!(prev_end, seq_len);
    }

    /// Every layout puts each token on exactly one rank, in a list whose
    /// length agrees with `local_len`.
    #[test]
    fn layouts_partition_the_sequence() {
        for layout in [SequenceLayout::Contiguous, SequenceLayout::ZigZag] {
            for world in 1..=5 {
                for total in [world * 2, 17, 40, 101] {
                    let mut seen = vec![0u32; total];
                    for rank in 0..world {
                        let positions = layout.local_positions(total, rank, world);
                        assert_eq!(positions.len(), layout.local_len(total, rank, world));
                        for p in positions {
                            seen[p as usize] += 1;
                        }
                    }
                    assert!(
                        seen.iter().all(|&n| n == 1),
                        "{layout:?} world {world} total {total}: {seen:?}"
                    );
                }
            }
        }
    }

    /// The point of zig-zag: under a causal mask every rank sees about the
    /// same number of (query, key) pairs.
    #[test]
    fn zigzag_balances_causal_work_where_contiguous_does_not() {
        let (total, world) = (4096, 4);
        let work = |layout: SequenceLayout| -> Vec<u64> {
            (0..world)
                .map(|rank| {
                    layout
                        .local_positions(total, rank, world)
                        .iter()
                        .map(|&p| u64::from(p) + 1)
                        .sum()
                })
                .collect()
        };
        let spread =
            |w: Vec<u64>| *w.iter().max().unwrap() as f64 / *w.iter().min().unwrap() as f64;
        assert!(spread(work(SequenceLayout::Contiguous)) > 3.0);
        assert!(spread(work(SequenceLayout::ZigZag)) < 1.01);
    }
}
