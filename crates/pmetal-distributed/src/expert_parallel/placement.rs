//! Expert placement solver: decides which experts live on which nodes.
//!
//! The `uniform` constructor is the canonical consumer of
//! [`crate::expert_shard::expert_range`] — the same helper that
//! [`crate::tensor_parallel::sharding::shard_weight`] uses for
//! `ExpertSharded` weights. Keeping both behind one helper is the
//! invariant that makes expert-parallel routing match weight sharding:
//! token for expert `e` is sent to rank `r`, and rank `r`'s weight shard
//! contains expert `e`.

use crate::expert_shard::expert_range;

/// Expert placement plan across nodes.
///
/// Maps each expert ID to the rank that owns its weights and computes
/// on its tokens.
#[derive(Debug, Clone)]
pub struct ExpertPlacement {
    /// experts_per_rank[rank] = list of expert IDs owned by that rank.
    pub experts_per_rank: Vec<Vec<usize>>,
    /// Total number of experts across all ranks.
    pub total_experts: usize,
    /// Top-k experts activated per token.
    pub top_k: usize,
    /// Reverse map: expert_id → rank.
    rank_for_expert: Vec<usize>,
}

impl ExpertPlacement {
    /// Uniform division: partition experts contiguously across ranks.
    ///
    /// Defers to [`crate::expert_shard::expert_range`] for the actual
    /// partitioning — the first `total_experts % world_size` ranks each
    /// get one extra expert. For 512 experts across 4 nodes: rank 0 gets
    /// `[0..128)`, rank 3 gets `[384..512)`. For 10 experts across 3 nodes:
    /// rank 0 gets `[0..4)`, ranks 1 and 2 get `[4..7)` and `[7..10)`.
    pub fn uniform(total_experts: usize, world_size: usize, top_k: usize) -> Self {
        let mut experts_per_rank = Vec::with_capacity(world_size);
        let mut rank_for_expert = vec![0usize; total_experts];

        for rank in 0..world_size {
            let (start, count) = expert_range(total_experts, rank, world_size);
            let experts: Vec<usize> = (start..start + count).collect();
            for &eid in &experts {
                rank_for_expert[eid] = rank;
            }
            experts_per_rank.push(experts);
        }

        Self {
            experts_per_rank,
            total_experts,
            top_k,
            rank_for_expert,
        }
    }

    /// Weighted division by available RAM on each node.
    ///
    /// Nodes with more RAM get more experts, proportional to their share
    /// of total cluster RAM.
    pub fn weighted(total_experts: usize, node_ram: &[u64], top_k: usize) -> Self {
        let world_size = node_ram.len();
        if node_ram.iter().sum::<u64>() == 0 {
            return Self::uniform(total_experts, world_size, top_k);
        }

        // Largest-remainder apportionment: every expert has exactly one owner
        // and no rank is more than one expert from its fair share.
        let counts = crate::apportion::largest_remainder(total_experts, node_ram);
        let mut experts_per_rank = Vec::with_capacity(world_size);
        let mut rank_for_expert = vec![0usize; total_experts];
        let mut assigned = 0;
        for (rank, &count) in counts.iter().enumerate() {
            let experts: Vec<usize> = (assigned..assigned + count).collect();
            for &eid in &experts {
                rank_for_expert[eid] = rank;
            }
            experts_per_rank.push(experts);
            assigned += count;
        }

        Self {
            experts_per_rank,
            total_experts,
            top_k,
            rank_for_expert,
        }
    }

    /// Which rank owns a given expert.
    pub fn rank_for_expert(&self, expert_id: usize) -> usize {
        self.rank_for_expert[expert_id]
    }

    /// Expert IDs owned by a given rank.
    pub fn local_expert_ids(&self, rank: usize) -> &[usize] {
        &self.experts_per_rank[rank]
    }

    /// Number of experts on a given rank.
    pub fn num_local_experts(&self, rank: usize) -> usize {
        self.experts_per_rank[rank].len()
    }

    /// Convert a global expert ID to a local index within its owning rank.
    pub fn global_to_local(&self, expert_id: usize) -> usize {
        let rank = self.rank_for_expert[expert_id];
        let first = self.experts_per_rank[rank][0];
        expert_id - first
    }

    /// Convert a local expert index on a rank to the global expert ID.
    pub fn local_to_global(&self, rank: usize, local_idx: usize) -> usize {
        self.experts_per_rank[rank][local_idx]
    }

    /// World size (number of ranks).
    pub fn world_size(&self) -> usize {
        self.experts_per_rank.len()
    }

    /// Check if a given expert is local to this rank.
    pub fn is_local(&self, expert_id: usize, rank: usize) -> bool {
        self.rank_for_expert[expert_id] == rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_placement_512_experts_4_nodes() {
        let p = ExpertPlacement::uniform(512, 4, 8);
        assert_eq!(p.world_size(), 4);
        assert_eq!(p.num_local_experts(0), 128);
        assert_eq!(p.num_local_experts(1), 128);
        assert_eq!(p.num_local_experts(2), 128);
        assert_eq!(p.num_local_experts(3), 128);

        assert_eq!(p.rank_for_expert(0), 0);
        assert_eq!(p.rank_for_expert(127), 0);
        assert_eq!(p.rank_for_expert(128), 1);
        assert_eq!(p.rank_for_expert(511), 3);
    }

    #[test]
    fn uniform_placement_remainder() {
        let p = ExpertPlacement::uniform(10, 3, 2);
        // 10 / 3 = 3 remainder 1 → ranks get 4, 3, 3
        assert_eq!(p.num_local_experts(0), 4);
        assert_eq!(p.num_local_experts(1), 3);
        assert_eq!(p.num_local_experts(2), 3);

        // Verify all experts assigned
        let total: usize = (0..3).map(|r| p.num_local_experts(r)).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn weighted_placement_proportional() {
        // Node 0 has 2x the RAM of nodes 1 and 2.
        let p = ExpertPlacement::weighted(12, &[200, 100, 100], 2);
        // Node 0 should get ~6, nodes 1 and 2 ~3 each.
        assert_eq!(p.num_local_experts(0), 6);
        assert_eq!(p.num_local_experts(1), 3);
        assert_eq!(p.num_local_experts(2), 3);
    }

    #[test]
    fn global_to_local_mapping() {
        let p = ExpertPlacement::uniform(8, 2, 2);
        assert_eq!(p.global_to_local(0), 0);
        assert_eq!(p.global_to_local(3), 3);
        assert_eq!(p.global_to_local(4), 0);
        assert_eq!(p.global_to_local(7), 3);
    }

    #[test]
    fn local_to_global_roundtrip() {
        let p = ExpertPlacement::uniform(8, 2, 2);
        for eid in 0..8 {
            let rank = p.rank_for_expert(eid);
            let local = p.global_to_local(eid);
            assert_eq!(p.local_to_global(rank, local), eid);
        }
    }

    #[test]
    fn is_local_check() {
        let p = ExpertPlacement::uniform(8, 2, 2);
        assert!(p.is_local(0, 0));
        assert!(p.is_local(3, 0));
        assert!(!p.is_local(4, 0));
        assert!(p.is_local(4, 1));
    }

    #[test]
    fn uniform_matches_canonical_expert_range() {
        // The invariant: ExpertPlacement::uniform must agree with
        // expert_shard::expert_range for every (total, world) pair. If
        // this drifts, expert-parallel routing will send tokens to ranks
        // whose weight shards don't include those experts.
        for (total, world) in [(1, 1), (10, 3), (512, 4), (64, 8), (100, 6), (3, 4)] {
            let p = ExpertPlacement::uniform(total, world, 2);
            for rank in 0..world {
                let (start, count) = crate::expert_shard::expert_range(total, rank, world);
                let owned = p.local_expert_ids(rank);
                assert_eq!(
                    owned.len(),
                    count,
                    "count mismatch total={total} world={world} rank={rank}"
                );
                for (i, &eid) in owned.iter().enumerate() {
                    assert_eq!(
                        eid,
                        start + i,
                        "id mismatch total={total} world={world} rank={rank} pos={i}"
                    );
                    assert_eq!(
                        p.rank_for_expert(eid),
                        rank,
                        "rank_for_expert mismatch eid={eid}"
                    );
                }
            }
        }
    }

    /// Whatever the weights, every expert gets exactly one owner and the
    /// shares follow the weights. Rounding each share on its own (and giving
    /// the last rank the rest) over-allocates when many shares round up:
    /// six equal ranks and three experts asked the last rank for `3 - 5`.
    #[test]
    fn weighted_placement_assigns_every_expert_exactly_once() {
        let cases: Vec<(usize, Vec<u64>)> = vec![
            (3, vec![1; 6]),
            (10, vec![3, 3, 3, 3]),
            (7, vec![1, 1, 1, 1]),
            (8, vec![10, 10, 1]),
            (5, vec![0, 4, 0, 4]),
            (1, vec![5, 5]),
            (64, vec![16, 32, 8, 8]),
        ];
        for (experts, ram) in cases {
            let p = ExpertPlacement::weighted(experts, &ram, 2);
            let mut seen = vec![0u32; experts];
            for rank in 0..ram.len() {
                for &e in p.local_expert_ids(rank) {
                    seen[e] += 1;
                    assert_eq!(p.rank_for_expert(e), rank);
                }
            }
            assert!(
                seen.iter().all(|&n| n == 1),
                "{experts} experts over {ram:?}: {seen:?}"
            );
            for (rank, &r) in ram.iter().enumerate() {
                let share = experts as f64 * r as f64 / ram.iter().sum::<u64>() as f64;
                let got = p.num_local_experts(rank) as f64;
                assert!(
                    (got - share).abs() < 1.0,
                    "{experts} experts over {ram:?}: rank {rank} got {got}, fair share {share}"
                );
            }
        }
    }
}
