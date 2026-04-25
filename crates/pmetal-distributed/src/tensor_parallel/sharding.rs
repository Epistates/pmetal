//! Core sharding abstractions for tensor parallelism.

use crate::expert_shard::expert_range;
use pmetal_bridge::compat::{Array, Exception};
use std::collections::HashMap;

/// Slice a contiguous range along `axis` using `take_axis`.
///
/// Equivalent to `x[..., start:start+len, ...]`. There is no `narrow` method
/// in the bridge compat layer; this helper builds the index array and calls
/// `take_axis`.
fn narrow(x: &Array, axis: i32, start: i32, len: i32) -> Result<Array, Exception> {
    let indices: Vec<i32> = (start..start + len).collect();
    let idx = Array::from_i32_slice(&indices);
    Ok(x.take_axis(&idx, axis))
}

/// Describes how a single weight tensor is distributed across ranks.
#[derive(Debug, Clone)]
pub enum ShardingDirective {
    /// Column-shard (split output dimension).
    ///
    /// Weight shape becomes `[output/N, input]` on each rank.
    /// Forward: `sum_gradients` barrier → local matmul → sharded output.
    /// Used for: Q/K/V projections, gate/up projections.
    AllToSharded {
        /// Axis to split along (typically 0 for weight matrices).
        axis: usize,
    },

    /// Row-shard (split input dimension).
    ///
    /// Weight shape becomes `[output, input/N]` on each rank.
    /// Forward: local matmul → `all_sum` → replicated output.
    /// Used for: O projection, down projection.
    ShardedToAll {
        /// Axis to split along (typically 1 for weight matrices).
        axis: usize,
    },

    /// Fully replicated across all ranks (no sharding).
    ///
    /// Used for: embeddings, layer norms, bias terms.
    Replicated,

    /// Expert-level sharding for MoE layers.
    ///
    /// Splits the expert dimension: each rank holds `total_experts/N` experts.
    /// Used for: routed expert gate/up/down projections.
    ExpertSharded {
        /// Total number of experts across all ranks.
        total_experts: usize,
    },
}

/// A complete sharding plan for an entire model.
///
/// Maps weight parameter names (e.g., `"model.layers.0.self_attn.q_proj.weight"`)
/// to their sharding directives. Weights not in the plan are treated as
/// [`ShardingDirective::Replicated`].
#[derive(Debug, Clone, Default)]
pub struct ShardingPlan {
    /// Maps weight name → sharding directive.
    pub directives: HashMap<String, ShardingDirective>,
}

impl ShardingPlan {
    /// Create an empty sharding plan (all weights replicated).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a directive for a weight parameter.
    pub fn add(&mut self, name: impl Into<String>, directive: ShardingDirective) {
        self.directives.insert(name.into(), directive);
    }

    /// Get the directive for a weight, defaulting to Replicated.
    pub fn get(&self, name: &str) -> &ShardingDirective {
        self.directives
            .get(name)
            .unwrap_or(&ShardingDirective::Replicated)
    }

    /// Total number of sharded weights in the plan.
    pub fn num_sharded(&self) -> usize {
        self.directives
            .values()
            .filter(|d| !matches!(d, ShardingDirective::Replicated))
            .count()
    }

    /// Merge another plan into this one (other's directives take precedence).
    pub fn merge(&mut self, other: &ShardingPlan) {
        for (name, directive) in &other.directives {
            self.directives.insert(name.clone(), directive.clone());
        }
    }
}

/// Slice a weight tensor according to its sharding directive.
///
/// Given a full weight tensor, returns the shard for `rank` out of `world_size`.
pub fn shard_weight(
    weight: &Array,
    directive: &ShardingDirective,
    rank: usize,
    world_size: usize,
) -> Result<Array, Exception> {
    match directive {
        ShardingDirective::Replicated => Ok(weight.clone()),

        ShardingDirective::AllToSharded { axis } | ShardingDirective::ShardedToAll { axis } => {
            let shape = weight.shape();
            let axis_idx = *axis;

            if axis_idx >= shape.len() {
                return Err(Exception::custom(format!(
                    "shard_weight: axis {} out of bounds for shape {:?}",
                    axis_idx, shape
                )));
            }

            let dim = shape[axis_idx] as usize;
            let shard_size = dim / world_size;
            let remainder = dim % world_size;

            if shard_size == 0 {
                return Err(Exception::custom(format!(
                    "shard_weight: dimension {} too small to split {} ways",
                    dim, world_size
                )));
            }

            // Compute start and end for this rank.
            // First `remainder` ranks get one extra element.
            let start = if rank < remainder {
                rank * (shard_size + 1)
            } else {
                remainder * (shard_size + 1) + (rank - remainder) * shard_size
            };
            let end = if rank < remainder {
                start + shard_size + 1
            } else {
                start + shard_size
            };

            // Use narrow to slice along the axis.
            let start_i = start as i32;
            let len_i = (end - start) as i32;
            narrow(weight, axis_idx as i32, start_i, len_i)
        }

        ShardingDirective::ExpertSharded { total_experts } => {
            let shape = weight.shape();
            if shape.is_empty() {
                return Err(Exception::custom(
                    "shard_weight: empty shape for ExpertSharded",
                ));
            }

            // Expert dimension is always axis 0: [num_experts, ...].
            //
            // Use the canonical `expert_range` helper so this shard
            // matches `ExpertPlacement::uniform` exactly. A divergence here
            // would route tokens to ranks whose weight shards don't
            // contain the addressed experts — see `crate::expert_shard`.
            let (start, len) = expert_range(*total_experts, rank, world_size);
            if len == 0 {
                return Err(Exception::custom(format!(
                    "shard_weight: rank {rank} has no experts for total_experts={total_experts}, \
                     world_size={world_size} (world_size must be <= total_experts)"
                )));
            }
            narrow(weight, 0, start as i32, len as i32)
        }
    }
}

/// Apply a sharding plan to a full weight map, producing sharded weights for one rank.
pub fn apply_sharding_plan(
    weights: &HashMap<String, Array>,
    plan: &ShardingPlan,
    rank: usize,
    world_size: usize,
) -> Result<HashMap<String, Array>, Exception> {
    let mut sharded = HashMap::with_capacity(weights.len());

    for (name, weight) in weights {
        let directive = plan.get(name);
        let shard = shard_weight(weight, directive, rank, world_size)?;
        sharded.insert(name.clone(), shard);
    }

    Ok(sharded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharding_plan_default_is_replicated() {
        let plan = ShardingPlan::new();
        assert!(matches!(
            plan.get("anything"),
            ShardingDirective::Replicated
        ));
        assert_eq!(plan.num_sharded(), 0);
    }

    #[test]
    fn sharding_plan_add_and_get() {
        let mut plan = ShardingPlan::new();
        plan.add("q_proj.weight", ShardingDirective::AllToSharded { axis: 0 });
        plan.add("o_proj.weight", ShardingDirective::ShardedToAll { axis: 1 });

        assert!(matches!(
            plan.get("q_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 }
        ));
        assert!(matches!(
            plan.get("o_proj.weight"),
            ShardingDirective::ShardedToAll { axis: 1 }
        ));
        assert_eq!(plan.num_sharded(), 2);
    }

    #[test]
    fn shard_weight_replicated() {
        let w = Array::from_f32_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let result = shard_weight(&w, &ShardingDirective::Replicated, 0, 2).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn shard_weight_column_split() {
        // [4, 2] weight split along axis 0 into 2 ranks → [2, 2] each
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let w = Array::from_f32_slice(&data, &[4, 2]);

        let shard0 = shard_weight(&w, &ShardingDirective::AllToSharded { axis: 0 }, 0, 2).unwrap();
        let shard1 = shard_weight(&w, &ShardingDirective::AllToSharded { axis: 0 }, 1, 2).unwrap();

        assert_eq!(shard0.shape(), &[2, 2]);
        assert_eq!(shard1.shape(), &[2, 2]);
    }

    #[test]
    fn shard_weight_row_split() {
        // [2, 4] weight split along axis 1 into 2 ranks → [2, 2] each
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let w = Array::from_f32_slice(&data, &[2, 4]);

        let shard0 = shard_weight(&w, &ShardingDirective::ShardedToAll { axis: 1 }, 0, 2).unwrap();
        let shard1 = shard_weight(&w, &ShardingDirective::ShardedToAll { axis: 1 }, 1, 2).unwrap();

        assert_eq!(shard0.shape(), &[2, 2]);
        assert_eq!(shard1.shape(), &[2, 2]);
    }

    #[test]
    fn shard_weight_expert_sharded() {
        // [8, 3, 4] → 8 experts, split into 4 ranks → [2, 3, 4] each
        let data: Vec<f32> = (0..96).map(|i| i as f32).collect();
        let w = Array::from_f32_slice(&data, &[8, 3, 4]);

        let shard = shard_weight(
            &w,
            &ShardingDirective::ExpertSharded { total_experts: 8 },
            2,
            4,
        )
        .unwrap();

        assert_eq!(shard.shape(), &[2, 3, 4]);
    }

    #[test]
    fn shard_weight_expert_sharded_non_divisible_covers_all() {
        // 10 experts across 3 ranks — previously the tail expert 9 went
        // unowned because of `total/world` truncation. Now every expert
        // must appear in exactly one shard.
        // Use expert dim (axis 0) = 10, per-expert trailing dim = 2 so
        // we can trivially check which experts ended up in each shard.
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        // Shape [10, 2]: expert e has values [2e, 2e+1].
        let w = Array::from_f32_slice(&data, &[10, 2]);

        let total_experts = 10usize;
        let world_size = 3usize;

        let mut covered = vec![false; total_experts];
        for rank in 0..world_size {
            let shard = shard_weight(
                &w,
                &ShardingDirective::ExpertSharded { total_experts },
                rank,
                world_size,
            )
            .unwrap();

            // Cross-reference against the canonical placement: the
            // experts in this shard must be exactly what `expert_range`
            // says rank owns.
            let (start, count) = crate::expert_shard::expert_range(
                total_experts,
                rank,
                world_size,
            );
            assert_eq!(shard.shape(), &[count as i32, 2]);
            for (e, slot) in covered
                .iter_mut()
                .enumerate()
                .skip(start)
                .take(count)
            {
                assert!(!*slot, "expert {e} covered twice");
                *slot = true;
            }
        }
        assert!(covered.iter().all(|c| *c), "some experts went unowned");
    }

    #[test]
    fn apply_sharding_plan_mixed() {
        let mut weights = HashMap::new();
        weights.insert(
            "q.weight".to_string(),
            Array::from_f32_slice(&[0.0f32; 8], &[4, 2]),
        );
        weights.insert(
            "norm.weight".to_string(),
            Array::from_f32_slice(&[1.0f32; 4], &[4]),
        );

        let mut plan = ShardingPlan::new();
        plan.add("q.weight", ShardingDirective::AllToSharded { axis: 0 });
        // norm.weight not in plan → replicated

        let sharded = apply_sharding_plan(&weights, &plan, 0, 2).unwrap();
        assert_eq!(sharded["q.weight"].shape(), &[2, 2]);
        assert_eq!(sharded["norm.weight"].shape(), &[4]);
    }
}
