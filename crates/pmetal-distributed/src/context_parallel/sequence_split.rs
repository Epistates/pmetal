//! Sequence partitioning utilities for context parallelism.

use crate::mlx_dist::group::DistributedGroup;
use crate::mlx_dist::ops;
use pmetal_runtime::compat::{Array, Exception};

/// Extract a contiguous slice along `axis` using `take_axis`.
///
/// Equivalent to `x[..., start:start+len, ...]` but using the bridge
/// compat `take_axis` API (there is no `narrow` method in the compat layer).
fn narrow(x: &Array, axis: i32, start: i32, len: i32) -> Result<Array, Exception> {
    let indices: Vec<i32> = (start..start + len).collect();
    let idx = Array::from_i32_slice(&indices);
    Ok(x.take_axis(&idx, axis))
}

/// Split a tensor along the sequence dimension across ranks.
///
/// Input `x` has shape `[B, S, D]` (or `[B, H, S, D]` for multi-head).
/// Each rank gets a contiguous chunk of size `S / world_size`.
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
    let shape = x.shape();
    if seq_axis >= shape.len() {
        return Err(Exception::custom(format!(
            "split_sequence: seq_axis {} out of bounds for shape {:?}",
            seq_axis, shape
        )));
    }

    let seq_len = shape[seq_axis] as usize;
    let chunk_size = seq_len / world_size;
    let remainder = seq_len % world_size;

    if chunk_size == 0 {
        return Err(Exception::custom(format!(
            "split_sequence: seq_len {} too small for {} ranks",
            seq_len, world_size
        )));
    }

    // First `remainder` ranks get one extra element.
    let start = if rank < remainder {
        rank * (chunk_size + 1)
    } else {
        remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
    };
    let len = if rank < remainder {
        chunk_size + 1
    } else {
        chunk_size
    };

    narrow(x, seq_axis as i32, start as i32, len as i32)
}

/// Gather sequence chunks from all ranks using all_gather.
///
/// Each rank provides its local chunk; the result is the concatenation
/// of all chunks in rank order along the sequence axis.
pub fn gather_sequence(local_chunk: &Array, group: &DistributedGroup) -> Result<Array, Exception> {
    ops::all_gather(local_chunk, Some(group))
}

/// Compute the local sequence range for a rank.
///
/// Returns `(start, length)` for this rank's chunk of a sequence of total length `seq_len`.
pub fn local_seq_range(seq_len: usize, rank: usize, world_size: usize) -> (usize, usize) {
    let chunk_size = seq_len / world_size;
    let remainder = seq_len % world_size;

    let start = if rank < remainder {
        rank * (chunk_size + 1)
    } else {
        remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
    };
    let len = if rank < remainder {
        chunk_size + 1
    } else {
        chunk_size
    };

    (start, len)
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
}
