//! All-to-all expert dispatch using MLX point-to-point operations.

use super::placement::ExpertPlacement;
use crate::mlx_dist::group::DistributedGroup;
use crate::mlx_dist::ops;
use pmetal_bridge::compat::{Array, Dtype, Exception, ops as mlx_ops};

/// Slice a contiguous range along `axis` using `take_axis`.
///
/// There is no `narrow` method in the bridge compat layer; this builds the
/// index array and calls `take_axis`.
fn narrow(x: &Array, axis: i32, start: i32, len: i32) -> Result<Array, Exception> {
    let indices: Vec<i32> = (start..start + len).collect();
    let idx = Array::from_i32_slice(&indices);
    Ok(x.take_axis(&idx, axis))
}

/// Manages routing tokens to expert-owning nodes and collecting results.
///
/// # Algorithm
///
/// 1. **Replicated routing**: Each rank computes the full routing scores locally.
/// 2. **Sort by destination**: Group tokens by the rank that owns their selected expert.
/// 3. **All-to-all exchange**: Send token batches to each rank via MLX `send()`/`recv()`.
/// 4. **Local compute**: Each rank runs its local experts on received tokens.
/// 5. **All-to-all return**: Send results back to originating ranks.
/// 6. **Reassemble**: Reorder results to match original token positions.
pub struct ExpertDispatcher {
    placement: ExpertPlacement,
    rank: usize,
    world_size: usize,
}

impl ExpertDispatcher {
    /// Create a new expert dispatcher.
    pub fn new(placement: ExpertPlacement, rank: usize) -> Self {
        let world_size = placement.world_size();
        Self {
            placement,
            rank,
            world_size,
        }
    }

    /// Get the expert placement plan.
    pub fn placement(&self) -> &ExpertPlacement {
        &self.placement
    }

    /// Dispatch tokens to expert-owning ranks, compute locally, collect results.
    ///
    /// This is the main entry point for expert-parallel MoE forward pass.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` — Input tensor `[num_tokens, hidden_dim]`
    /// * `routing_weights` — Routing weights `[num_tokens, top_k]` (from softmax)
    /// * `routing_indices` — Expert indices `[num_tokens, top_k]`
    /// * `group` — Distributed group for communication
    /// * `local_expert_fn` — Closure that computes local experts:
    ///   `fn(tokens: &Array, expert_ids: &[usize]) -> Result<Array>`
    ///
    /// # Returns
    ///
    /// Output tensor `[num_tokens, hidden_dim]` (weighted sum of expert outputs).
    pub fn dispatch_and_compute<F>(
        &self,
        hidden_states: &Array,
        routing_weights: &Array,
        routing_indices: &Array,
        group: &DistributedGroup,
        local_expert_fn: F,
    ) -> Result<Array, Exception>
    where
        F: Fn(&Array, &[usize]) -> Result<Array, Exception>,
    {
        let mut routing_indices = routing_indices.clone();
        let mut routing_weights = routing_weights.clone();
        let hidden_states = hidden_states.clone();
        routing_indices.eval();
        routing_weights.eval();
        hidden_states.eval();

        let num_tokens = hidden_states.shape()[0] as usize;
        let hidden_dim = hidden_states.shape()[1] as usize;
        let top_k = routing_indices.shape()[1] as usize;

        let total_idx = num_tokens * top_k;
        let idx_data: Vec<i32> = routing_indices
            .to_f32_vec(total_idx)
            .ok_or_else(|| Exception::custom("dispatch: failed to read routing indices"))?
            .into_iter()
            .map(|x| x as i32)
            .collect();
        let weight_data: Vec<f32> = routing_weights
            .to_f32_vec(total_idx)
            .ok_or_else(|| Exception::custom("dispatch: failed to read routing weights"))?;

        // Phase 1: Classify tokens by destination rank.
        //
        // See [`classify_by_destination_rank`] for the full rationale.
        // Extracted as a standalone pure fn so the (token, k) → rank
        // bookkeeping can be unit-tested without the distributed runtime.
        let Classification {
            tokens_for_rank,
            k_for_rank,
            expert_for_token,
        } = classify_by_destination_rank(
            &idx_data,
            num_tokens,
            top_k,
            self.world_size,
            &self.placement,
        );

        // Phase 2: Send tokens to each rank and receive tokens from each rank.
        let mut received_results: Vec<Option<Array>> = vec![None; self.world_size];

        for dest in 0..self.world_size {
            let token_indices = &tokens_for_rank[dest];
            let expert_ids = &expert_for_token[dest];

            if token_indices.is_empty() {
                continue;
            }

            if dest == self.rank {
                // Local computation — no network transfer needed.
                let local_tokens = gather_tokens(&hidden_states, token_indices)?;
                let result = local_expert_fn(&local_tokens, expert_ids)?;
                received_results[dest] = Some(result);
            } else {
                // Remote: send tokens, compute remotely, receive results.
                let local_tokens = gather_tokens(&hidden_states, token_indices)?;

                // Send tokens to remote rank.
                let send_sentinel = ops::send(&local_tokens, dest as i32, Some(group))?;
                send_sentinel.eval();

                // Receive computed results from remote rank.
                let result_shape = vec![token_indices.len() as i32, hidden_dim as i32];
                // Use Float32 as the receive dtype (matches the hidden_states dtype assumption).
                let result = ops::recv(&result_shape, Dtype::Float32, dest as i32, Some(group))?;
                result.eval();
                received_results[dest] = Some(result);
            }
        }

        // Phase 3: Serve remote requests.
        // Receive tokens from other ranks, compute, send back results.
        for src in 0..self.world_size {
            if src == self.rank {
                continue;
            }

            let expected_tokens = tokens_for_rank[src].len();
            if expected_tokens == 0 {
                continue;
            }

            let incoming_shape = vec![expected_tokens as i32, hidden_dim as i32];
            let incoming = ops::recv(&incoming_shape, Dtype::Float32, src as i32, Some(group))?;
            incoming.eval();

            let result = local_expert_fn(&incoming, &expert_for_token[src])?;

            let send_sentinel = ops::send(&result, src as i32, Some(group))?;
            send_sentinel.eval();
        }

        // Phase 4: Reassemble results in original token order with routing weights.
        // Initialize output as zeros.
        let out_shape = &[num_tokens as i32, hidden_dim as i32];
        let mut output = mlx_ops::zeros(out_shape, Dtype::Float32);

        for dest in 0..self.world_size {
            let token_indices = &tokens_for_rank[dest];
            let k_indices = &k_for_rank[dest];
            if token_indices.is_empty() {
                continue;
            }
            debug_assert_eq!(
                token_indices.len(),
                k_indices.len(),
                "tokens_for_rank and k_for_rank must stay parallel"
            );

            if let Some(ref result) = received_results[dest] {
                // Scatter results back with routing weights.
                //
                // `result` has one row per entry in `token_indices`
                // (in Phase 1 insertion order); `k_indices[local_idx]`
                // recovers the slot `k` so we pick the correct routing
                // weight from the `[num_tokens, top_k]` matrix.
                for (local_idx, (&token_idx, &k)) in
                    token_indices.iter().zip(k_indices.iter()).enumerate()
                {
                    let w = weight_data[token_idx * top_k + k];
                    let w_arr = Array::from_f32_slice(&[w], &[1, 1]);
                    let row = narrow(result, 0, local_idx as i32, 1)?;
                    let weighted = row.multiply(&w_arr);

                    // Accumulate into output at token_idx position.
                    let current = narrow(&output, 0, token_idx as i32, 1)?;
                    let updated = current.add(&weighted);
                    // Note: In production, this scatter-add would use a single
                    // index_put operation for efficiency. This loop is for clarity.
                    output = scatter_row(&output, token_idx, &updated)?;
                }
            }
        }

        Ok(output)
    }
}

/// Result of Phase-1 token/expert classification.
///
/// Three parallel lists keyed by destination rank. For rank `r`, every
/// index `i` refers to the same logical `(token, k)` assignment:
/// - `tokens_for_rank[r][i]` — the global token index.
/// - `k_for_rank[r][i]` — the slot `k` in `[0, top_k)` that produced
///   this assignment.
/// - `expert_for_token[r][i]` — the selected expert ID.
///
/// Tests: see `tests::classify_distributes_top_k_pairs_across_ranks`.
struct Classification {
    tokens_for_rank: Vec<Vec<usize>>,
    k_for_rank: Vec<Vec<usize>>,
    expert_for_token: Vec<Vec<usize>>,
}

/// Phase-1 classifier, extracted so it can be unit-tested without
/// allocating MLX arrays or constructing a distributed group.
///
/// `idx_data` is the flattened `[num_tokens, top_k]` routing matrix in
/// row-major order (token index varies fastest-slowest = slowest,
/// top-k axis fastest). For each `(token, k)` pair it looks up the
/// owning rank via `placement.rank_for_expert(expert_id)` and appends
/// one entry to that rank's lists.
///
/// The `k` value is preserved so Phase 4 can look up the correct
/// routing weight `weight_data[token * top_k + k]` when scattering
/// per-expert results back into the per-token output. Losing `k` here
/// was the bug fixed in the Phase-B audit follow-up.
fn classify_by_destination_rank(
    idx_data: &[i32],
    num_tokens: usize,
    top_k: usize,
    world_size: usize,
    placement: &ExpertPlacement,
) -> Classification {
    let mut tokens_for_rank: Vec<Vec<usize>> = vec![Vec::new(); world_size];
    let mut k_for_rank: Vec<Vec<usize>> = vec![Vec::new(); world_size];
    let mut expert_for_token: Vec<Vec<usize>> = vec![Vec::new(); world_size];

    for token_idx in 0..num_tokens {
        for k in 0..top_k {
            let expert_id = idx_data[token_idx * top_k + k] as usize;
            let dest_rank = placement.rank_for_expert(expert_id);
            tokens_for_rank[dest_rank].push(token_idx);
            k_for_rank[dest_rank].push(k);
            expert_for_token[dest_rank].push(expert_id);
        }
    }

    Classification {
        tokens_for_rank,
        k_for_rank,
        expert_for_token,
    }
}

/// Gather rows from a tensor by index list.
fn gather_tokens(x: &Array, indices: &[usize]) -> Result<Array, Exception> {
    let idx: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let idx_arr = Array::from_i32_slice(&idx);
    // take_axis gathers along a specific axis; take() operates on the flattened
    // array, so we use take_axis with axis=0 to gather rows.
    Ok(x.take_axis(&idx_arr, 0))
}

/// Replace a single row in a tensor (scatter by index).
fn scatter_row(x: &Array, row_idx: usize, value: &Array) -> Result<Array, Exception> {
    // Build the output by concatenating slices: [0..row_idx] + [value] + [row_idx+1..end]
    let num_rows = x.shape()[0] as usize;

    if row_idx == 0 && num_rows == 1 {
        return Ok(value.clone());
    }

    let mut parts: Vec<Array> = Vec::new();
    if row_idx > 0 {
        parts.push(narrow(x, 0, 0, row_idx as i32)?);
    }
    parts.push(value.clone());
    if row_idx + 1 < num_rows {
        parts.push(narrow(
            x,
            0,
            (row_idx + 1) as i32,
            (num_rows - row_idx - 1) as i32,
        )?);
    }

    let part_refs: Vec<&Array> = parts.iter().collect();
    Ok(mlx_ops::concatenate_axis(&part_refs, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_dispatcher_creation() {
        let placement = ExpertPlacement::uniform(8, 2, 2);
        let dispatcher = ExpertDispatcher::new(placement, 0);
        assert_eq!(dispatcher.rank, 0);
        assert_eq!(dispatcher.world_size, 2);
    }

    #[test]
    fn gather_tokens_basic() {
        let x = Array::from_f32_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let result = gather_tokens(&x, &[0, 2]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn classify_distributes_top_k_pairs_across_ranks() {
        // 4 experts, 2 ranks: rank 0 owns [0, 1], rank 1 owns [2, 3].
        // 3 tokens, top_k=2:
        //   token 0 → experts [0, 3] → ranks [0, 1]
        //   token 1 → experts [2, 1] → ranks [1, 0]
        //   token 2 → experts [1, 2] → ranks [0, 1]
        // idx_data row-major:
        let idx = vec![0, 3, 2, 1, 1, 2];
        let placement = ExpertPlacement::uniform(4, 2, 2);
        let c = classify_by_destination_rank(&idx, 3, 2, 2, &placement);

        // All three rank lists must stay parallel in length.
        assert_eq!(c.tokens_for_rank[0].len(), c.k_for_rank[0].len());
        assert_eq!(c.tokens_for_rank[0].len(), c.expert_for_token[0].len());
        assert_eq!(c.tokens_for_rank[1].len(), c.k_for_rank[1].len());
        assert_eq!(c.tokens_for_rank[1].len(), c.expert_for_token[1].len());

        // Rank 0 should hold exactly: (token 0, k 0, expert 0),
        // (token 1, k 1, expert 1), (token 2, k 0, expert 1).
        let rank0: Vec<_> = c.tokens_for_rank[0]
            .iter()
            .zip(c.k_for_rank[0].iter())
            .zip(c.expert_for_token[0].iter())
            .map(|((&t, &k), &e)| (t, k, e))
            .collect();
        assert_eq!(rank0, vec![(0, 0, 0), (1, 1, 1), (2, 0, 1)]);

        // Rank 1: (0, 1, 3), (1, 0, 2), (2, 1, 2).
        let rank1: Vec<_> = c.tokens_for_rank[1]
            .iter()
            .zip(c.k_for_rank[1].iter())
            .zip(c.expert_for_token[1].iter())
            .map(|((&t, &k), &e)| (t, k, e))
            .collect();
        assert_eq!(rank1, vec![(0, 1, 3), (1, 0, 2), (2, 1, 2)]);
    }

    #[test]
    fn classify_preserves_insertion_order_per_rank() {
        // Single rank case — order of (t, k) follows nested-loop order:
        // (0,0), (0,1), (1,0), (1,1), ...
        let idx = vec![0, 0, 0, 0]; // all experts = 0 → all on rank 0
        let placement = ExpertPlacement::uniform(4, 1, 2);
        let c = classify_by_destination_rank(&idx, 2, 2, 1, &placement);

        let seen: Vec<_> = c.tokens_for_rank[0]
            .iter()
            .zip(c.k_for_rank[0].iter())
            .map(|(&t, &k)| (t, k))
            .collect();
        assert_eq!(seen, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }
}
