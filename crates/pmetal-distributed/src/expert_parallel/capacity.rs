//! Load balancing and capacity management for expert parallelism.
//!
//! Controls how many tokens each expert can process, preventing
//! imbalanced workloads where some experts are overloaded while
//! others are idle.

use pmetal_runtime::compat::{Array, Dtype, Exception, ops};

/// Policy for handling tokens that exceed expert capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPolicy {
    /// Drop excess tokens (they produce zero output). Simple, used by Switch Transformer.
    Drop,
    /// Redistribute excess tokens to the least-loaded experts.
    Redistribute,
}

/// Configuration for capacity-aware expert routing.
#[derive(Debug, Clone)]
pub struct CapacityConfig {
    /// Capacity factor: how much overflow buffer each expert has.
    ///
    /// - `1.0`: Each expert can handle exactly `tokens / num_experts` tokens
    /// - `1.25`: 25% overflow buffer (recommended default)
    /// - `0.0`: No capacity limit (all tokens processed)
    pub capacity_factor: f32,

    /// What to do when an expert exceeds capacity.
    pub drop_policy: DropPolicy,
}

impl Default for CapacityConfig {
    fn default() -> Self {
        Self {
            capacity_factor: 1.25,
            drop_policy: DropPolicy::Drop,
        }
    }
}

impl CapacityConfig {
    /// No capacity limit — all tokens are processed.
    pub fn unlimited() -> Self {
        Self {
            capacity_factor: 0.0,
            drop_policy: DropPolicy::Drop,
        }
    }
}

/// Apply capacity constraints to routing indices.
///
/// Returns `(capped_indices, drop_mask)`:
/// - `capped_indices`: Same shape as input, but excess assignments are set to -1
/// - `drop_mask`: Boolean mask `[num_tokens, top_k]` where `true` = token is processed
///
/// # Arguments
///
/// * `routing_indices` — Expert indices `[num_tokens, top_k]`
/// * `config` — Capacity configuration
/// * `total_experts` — Total number of experts
pub fn apply_capacity(
    routing_indices: &Array,
    config: &CapacityConfig,
    total_experts: usize,
) -> Result<(Array, Array), Exception> {
    if config.capacity_factor <= 0.0 {
        // No capacity limit — all tokens pass through.
        let mask_shape = routing_indices.shape().to_vec();
        let mask = ops::ones(&mask_shape, Dtype::Bool);
        return Ok((routing_indices.clone(), mask));
    }

    let mut routing_indices = routing_indices.clone();
    routing_indices.eval();
    let shape = routing_indices.shape().to_vec();
    let num_tokens = shape[0] as usize;
    let top_k = shape[1] as usize;
    let total_assignments = num_tokens * top_k;

    // Capacity per expert: ceil(tokens * top_k / num_experts * capacity_factor)
    let avg_tokens_per_expert = (total_assignments as f32) / (total_experts as f32);
    let capacity = (avg_tokens_per_expert * config.capacity_factor).ceil() as usize;

    // Extract i32 data from the evaluated array.
    let idx_i32_opt = routing_indices.to_f32_vec(total_assignments);
    // The routing indices are i32 but to_f32_vec casts them; we need the raw
    // integer values so we use to_f32_vec (which evals and casts) and round.
    let idx_data: Vec<i32> = match idx_i32_opt {
        Some(v) => v.into_iter().map(|x| x as i32).collect(),
        None => {
            return Err(Exception::custom(
                "apply_capacity: failed to read routing indices",
            ));
        }
    };

    // Count assignments per expert and cap at capacity.
    let mut expert_counts = vec![0usize; total_experts];
    let mut capped = vec![0i32; total_assignments];
    let mut mask = vec![0.0f32; total_assignments]; // 1.0 = keep, 0.0 = drop

    for i in 0..total_assignments {
        let expert_id = idx_data[i] as usize;
        if expert_id < total_experts && expert_counts[expert_id] < capacity {
            expert_counts[expert_id] += 1;
            capped[i] = expert_id as i32;
            mask[i] = 1.0;
        } else {
            // Exceeds capacity — mark as dropped.
            capped[i] = -1;
            mask[i] = 0.0;
        }
    }

    // Build capped indices array (as f32 then cast to int32 on the GPU side,
    // or keep as f32 for compatibility — the downstream consumer uses as_slice).
    let capped_f32: Vec<f32> = capped.iter().map(|&x| x as f32).collect();
    let capped_arr = Array::from_f32_slice(&capped_f32, &[num_tokens as i32, top_k as i32]);
    // Cast to int32 so shape/dtype matches the original routing_indices contract.
    let capped_arr = capped_arr.as_dtype(Dtype::Int32.as_i32());

    let mask_arr = Array::from_f32_slice(&mask, &[num_tokens as i32, top_k as i32]);
    // Cast to bool.
    let mask_arr = mask_arr.as_dtype(Dtype::Bool.as_i32());

    Ok((capped_arr, mask_arr))
}

/// Compute load balancing auxiliary loss (Switch Transformer formula).
///
/// `aux_loss = N * sum_i(f_i * P_i)` where:
/// - `f_i` = fraction of tokens routed to expert i
/// - `P_i` = average routing probability for expert i
/// - `N` = number of experts
///
/// This loss encourages balanced expert utilization.
pub fn auxiliary_load_balance_loss(
    routing_indices: &Array,
    routing_probs: &Array,
    num_experts: usize,
) -> Result<f32, Exception> {
    let mut routing_indices = routing_indices.clone();
    let mut routing_probs = routing_probs.clone();
    routing_indices.eval();
    routing_probs.eval();

    let num_tokens = routing_indices.shape()[0] as usize;
    let top_k = routing_indices.shape()[1] as usize;
    let total_assignments = num_tokens * top_k;

    let idx_data: Vec<i32> = match routing_indices.to_f32_vec(total_assignments) {
        Some(v) => v.into_iter().map(|x| x as i32).collect(),
        None => {
            return Err(Exception::custom(
                "auxiliary_load_balance_loss: failed to read routing indices",
            ));
        }
    };

    let prob_total = num_tokens * num_experts;
    let prob_data: Vec<f32> = routing_probs.to_f32_vec(prob_total).ok_or_else(|| {
        Exception::custom("auxiliary_load_balance_loss: failed to read routing probs")
    })?;

    // Compute f_i (fraction of tokens routed to each expert).
    let mut token_counts = vec![0usize; num_experts];
    for &idx in idx_data.iter() {
        if (idx as usize) < num_experts {
            token_counts[idx as usize] += 1;
        }
    }

    let f: Vec<f32> = token_counts
        .iter()
        .map(|&c| c as f32 / total_assignments as f32)
        .collect();

    // Compute P_i (average routing probability for each expert).
    // routing_probs has shape [num_tokens, num_experts] (pre-softmax or post-softmax).
    let mut p = vec![0.0f32; num_experts];
    for token_idx in 0..num_tokens {
        for expert_id in 0..num_experts {
            p[expert_id] += prob_data[token_idx * num_experts + expert_id];
        }
    }
    for pi in &mut p {
        *pi /= num_tokens as f32;
    }

    // aux_loss = N * sum(f_i * P_i)
    let loss: f32 = f.iter().zip(p.iter()).map(|(fi, pi)| fi * pi).sum();
    Ok(num_experts as f32 * loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_unlimited_passes_all() {
        let indices = Array::from_f32_slice(&[0.0f32, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0], &[4, 2]);
        let indices = indices.as_dtype(Dtype::Int32.as_i32());
        let config = CapacityConfig::unlimited();
        let (capped, mask) = apply_capacity(&indices, &config, 4).unwrap();
        assert_eq!(capped.shape(), &[4, 2]);
        assert_eq!(mask.shape(), &[4, 2]);
    }

    #[test]
    fn capacity_drops_overloaded_experts() {
        // 4 tokens, top_k=1, 2 experts. All tokens go to expert 0.
        // capacity_factor=1.0 → capacity = ceil(4/2) = 2. Only 2 tokens pass.
        let indices = Array::from_f32_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4, 1]);
        let indices = indices.as_dtype(Dtype::Int32.as_i32());
        let config = CapacityConfig {
            capacity_factor: 1.0,
            drop_policy: DropPolicy::Drop,
        };
        let (capped, mask) = apply_capacity(&indices, &config, 2).unwrap();

        let mut capped = capped;
        let mut mask = mask;
        capped.eval();
        mask.eval();

        let capped_data: Vec<i32> = capped
            .to_f32_vec(4)
            .unwrap()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        let mask_data: Vec<f32> = mask.to_f32_vec(4).unwrap();

        // First 2 tokens should pass, last 2 should be dropped.
        assert_eq!(capped_data[0], 0);
        assert_eq!(capped_data[1], 0);
        assert_eq!(capped_data[2], -1);
        assert_eq!(capped_data[3], -1);
        assert!(mask_data[0] > 0.5);
        assert!(mask_data[1] > 0.5);
        assert!(mask_data[2] < 0.5);
        assert!(mask_data[3] < 0.5);
    }

    #[test]
    fn capacity_factor_increases_limit() {
        // Same setup but capacity_factor=2.0 → capacity = ceil(4/2 * 2) = 4.
        let indices = Array::from_f32_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4, 1]);
        let indices = indices.as_dtype(Dtype::Int32.as_i32());
        let config = CapacityConfig {
            capacity_factor: 2.0,
            drop_policy: DropPolicy::Drop,
        };
        let (_, mask) = apply_capacity(&indices, &config, 2).unwrap();

        let mut mask = mask;
        mask.eval();
        let mask_data: Vec<f32> = mask.to_f32_vec(4).unwrap();
        // All should pass with 2x capacity.
        assert!(mask_data.iter().all(|&m| m > 0.5));
    }
}
