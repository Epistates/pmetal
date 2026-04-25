//! Canonical expert-ID ↔ rank mapping for expert-parallel and tensor-parallel.
//!
//! # The invariant
//!
//! A single rule decides which experts live on which rank. Every module that
//! cares about "who owns expert `e`" — routing, weight sharding, ZeRO
//! optimizer partitioning — must derive its answer from the same formula:
//!
//! 1. Each rank `r` owns `base = total_experts / world_size` experts.
//! 2. The first `remainder = total_experts % world_size` ranks each get
//!    one extra expert, so rank 0 owns `[0..base+1)`, rank 1 owns
//!    `[base+1..2*(base+1))`, …, rank `remainder` owns `[offset..offset+base)`, etc.
//! 3. The local expert index is simply `expert_id - start_for_rank(r)`.
//!
//! If any module computes this mapping differently, expert-parallel routing
//! will send tokens to ranks whose weight shards don't include the
//! addressed experts — producing silent wrong-answers or shape errors.
//!
//! # Why "remainder to the first ranks"?
//!
//! This matches DeepSpeed's `distribute_experts` behavior and mlx-lm's
//! `_shard_experts`. A naïve `rank * base` indexing ignores the remainder
//! and leaves the tail experts unowned, which is the bug this module
//! exists to prevent.

/// Returns `(start_expert_id, count)` for the given rank.
///
/// `start..start+count` is the half-open range of global expert IDs
/// that rank owns. This is the one definition; every call site that asks
/// "which experts does this rank own" must come through here.
///
/// # Panics
///
/// - `world_size == 0` — call sites should never ask for a zero-rank plan.
/// - `rank >= world_size` — programmer error.
///
/// # Examples
///
/// Uniform split (divisible):
/// ```
/// # use pmetal_distributed::expert_shard::expert_range;
/// assert_eq!(expert_range(512, 0, 4), (0, 128));
/// assert_eq!(expert_range(512, 3, 4), (384, 128));
/// ```
///
/// Non-divisible — first ranks take the remainder:
/// ```
/// # use pmetal_distributed::expert_shard::expert_range;
/// // 10 experts across 3 ranks: 4, 3, 3.
/// assert_eq!(expert_range(10, 0, 3), (0, 4));
/// assert_eq!(expert_range(10, 1, 3), (4, 3));
/// assert_eq!(expert_range(10, 2, 3), (7, 3));
/// ```
pub fn expert_range(total_experts: usize, rank: usize, world_size: usize) -> (usize, usize) {
    assert!(world_size > 0, "expert_range: world_size must be > 0");
    assert!(
        rank < world_size,
        "expert_range: rank {rank} out of bounds for world_size {world_size}"
    );

    let base = total_experts / world_size;
    let remainder = total_experts % world_size;

    // First `remainder` ranks get `base + 1`, the rest get `base`.
    if rank < remainder {
        let start = rank * (base + 1);
        (start, base + 1)
    } else {
        let start = remainder * (base + 1) + (rank - remainder) * base;
        (start, base)
    }
}

/// Inverse of [`expert_range`]: which rank owns a given expert ID.
///
/// Equivalent to searching for the rank `r` such that
/// `expert_range(total_experts, r, world_size)` contains `expert_id`.
pub fn rank_for_expert(expert_id: usize, total_experts: usize, world_size: usize) -> usize {
    assert!(world_size > 0, "rank_for_expert: world_size must be > 0");
    assert!(
        expert_id < total_experts,
        "rank_for_expert: expert_id {expert_id} >= total_experts {total_experts}"
    );

    let base = total_experts / world_size;
    let remainder = total_experts % world_size;

    // Experts [0 .. remainder*(base+1)) live on ranks [0..remainder), each
    // holding `base+1` experts. After that, experts partition into `base`-sized
    // buckets over the remaining ranks.
    let big_block = remainder * (base + 1);
    if expert_id < big_block {
        expert_id / (base + 1)
    } else {
        // When base == 0 (world_size > total_experts), every expert lies
        // in the big-block region, so this branch is unreachable — the
        // assert above ensures expert_id < total_experts == big_block.
        match (expert_id - big_block).checked_div(base) {
            Some(extra) => remainder + extra,
            None => unreachable!("rank_for_expert: base == 0 but expert_id >= big_block"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_partition_total_experts() {
        for (total, world) in [(10, 3), (512, 4), (64, 8), (1, 1), (7, 7), (100, 3)] {
            let mut covered = 0usize;
            let mut last_end = 0usize;
            for rank in 0..world {
                let (start, count) = expert_range(total, rank, world);
                assert_eq!(start, last_end, "gap/overlap in rank {rank}");
                covered += count;
                last_end = start + count;
            }
            assert_eq!(covered, total, "total {total} world {world} not fully covered");
            assert_eq!(last_end, total);
        }
    }

    #[test]
    fn rank_for_expert_matches_range() {
        for (total, world) in [(10, 3), (512, 4), (7, 3), (100, 6)] {
            for rank in 0..world {
                let (start, count) = expert_range(total, rank, world);
                for eid in start..start + count {
                    assert_eq!(
                        rank_for_expert(eid, total, world),
                        rank,
                        "total {total} world {world} rank {rank} eid {eid}"
                    );
                }
            }
        }
    }

    #[test]
    fn remainder_concentrates_on_first_ranks() {
        // 10 experts, 3 ranks → 4, 3, 3 (DeepSpeed convention).
        assert_eq!(expert_range(10, 0, 3), (0, 4));
        assert_eq!(expert_range(10, 1, 3), (4, 3));
        assert_eq!(expert_range(10, 2, 3), (7, 3));
    }

    #[test]
    fn fewer_experts_than_ranks_is_ok_with_empty_tail() {
        // 3 experts, 4 ranks → 1, 1, 1, 0.
        assert_eq!(expert_range(3, 0, 4), (0, 1));
        assert_eq!(expert_range(3, 1, 4), (1, 1));
        assert_eq!(expert_range(3, 2, 4), (2, 1));
        assert_eq!(expert_range(3, 3, 4), (3, 0));

        assert_eq!(rank_for_expert(0, 3, 4), 0);
        assert_eq!(rank_for_expert(2, 3, 4), 2);
    }

    #[test]
    #[should_panic(expected = "world_size must be > 0")]
    fn zero_world_size_panics() {
        expert_range(10, 0, 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn rank_out_of_bounds_panics() {
        expert_range(10, 3, 3);
    }
}
