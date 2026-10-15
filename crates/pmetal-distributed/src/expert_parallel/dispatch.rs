//! Expert dispatch: routing tokens to the ranks that own their experts.
//!
//! Two ways to run a mixture-of-experts layer whose experts are spread over
//! ranks, depending on what the ranks hold:
//!
//! - **Different tokens per rank** (data-parallel batches, the training case):
//!   [`ExpertDispatcher::dispatch_and_compute`]. Every rank routes its own
//!   tokens, an all-to-all carries each token to the rank that owns each of
//!   its experts, the owners compute, and a second all-to-all brings the
//!   results home to be combined with the routing weights.
//!
//! - **The same tokens on every rank** (tensor-parallel inference):
//!   [`ExpertDispatcher::compute_replicated`]. Nothing needs to move: each rank
//!   runs the tokens routed to its own experts, scales them by their routing
//!   weights, and one `all_sum` adds the ranks' contributions.
//!
//! # The all-to-all
//!
//! MLX's ring backend only connects neighbours, and has no all-to-all, so
//! buckets are relayed: in round `t` every rank sends its neighbour the
//! buckets that still have to travel `t` or more hops. Bucket `(from, to)`
//! rides `(to - from) mod N` rounds. What each rank sends and receives in each
//! round follows from the bucket sizes, which every rank learns first by an
//! all-gather of per-destination counts, so no round carries a header.
//! Neighbour exchanges are ordered by rank parity so they cannot deadlock
//! ([`crate::mlx_dist::exchange`]).

use super::placement::ExpertPlacement;
use crate::mlx_dist::exchange;
use crate::mlx_dist::group::DistributedGroup;
use crate::mlx_dist::ops;
use pmetal_runtime::compat::{Array, Dtype, Exception, ops as mlx_ops};
use std::collections::HashMap;

/// Manages routing tokens to expert-owning nodes and collecting results.
pub struct ExpertDispatcher {
    placement: ExpertPlacement,
    rank: usize,
    world_size: usize,
}

/// Which tokens go where: for each rank, the flat `(token, slot)` indices
/// (`token * top_k + slot`) routed to an expert it owns, in routing order, and
/// the experts they name.
struct Routing {
    slots_for_rank: Vec<Vec<usize>>,
    experts_for_rank: Vec<Vec<usize>>,
}

fn read_ints(a: &Array) -> Result<Vec<i32>, Exception> {
    let mut a = a.as_dtype(Dtype::Float32.as_i32());
    a.try_eval().map_err(|e| Exception::custom(e.to_string()))?;
    let n = a.size();
    Ok(a.to_f32_vec(n)
        .ok_or_else(|| Exception::custom("dispatch: could not read routing indices"))?
        .into_iter()
        .map(|x| x as i32)
        .collect())
}

fn index_array(indices: &[usize]) -> Array {
    Array::from_i32_slice(&indices.iter().map(|&i| i as i32).collect::<Vec<_>>())
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

    fn check_group(&self, group: &DistributedGroup) -> Result<(), Exception> {
        if group.size() as usize != self.world_size || group.rank() as usize != self.rank {
            return Err(Exception::custom(format!(
                "dispatch: placement is for rank {} of {} but the group is rank {} of {}",
                self.rank,
                self.world_size,
                group.rank(),
                group.size()
            )));
        }
        Ok(())
    }

    fn shapes(
        hidden: &Array,
        routing_weights: &Array,
        routing_indices: &Array,
    ) -> Result<(usize, usize, usize), Exception> {
        let (hs, ws, is) = (
            hidden.shape(),
            routing_weights.shape(),
            routing_indices.shape(),
        );
        if hs.len() != 2 || is.len() != 2 || ws != is || is[0] != hs[0] || hs[0] == 0 || is[1] == 0
        {
            return Err(Exception::custom(format!(
                "dispatch: hidden {hs:?} wants routing weights and indices of shape \
                 [{}, top_k] with at least one token",
                hs.first().copied().unwrap_or(0)
            )));
        }
        Ok((hs[0] as usize, hs[1] as usize, is[1] as usize))
    }

    fn route(
        &self,
        routing_indices: &Array,
        tokens: usize,
        top_k: usize,
    ) -> Result<Routing, Exception> {
        let indices = read_ints(routing_indices)?;
        let experts: Vec<usize> = indices
            .iter()
            .map(|&e| {
                usize::try_from(e)
                    .ok()
                    .filter(|&e| e < self.placement.total_experts)
                    .ok_or_else(|| {
                        Exception::custom(format!(
                            "dispatch: routing names expert {e} but there are {}",
                            self.placement.total_experts
                        ))
                    })
            })
            .collect::<Result<_, _>>()?;
        Ok(classify_by_destination_rank(
            &experts,
            tokens,
            top_k,
            self.world_size,
            &self.placement,
        ))
    }

    /// Dispatch tokens to expert-owning ranks, compute there, bring the
    /// results back and combine them: the mixture-of-experts layer when each
    /// rank holds different tokens.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` — this rank's tokens `[num_tokens, hidden_dim]`
    /// * `routing_weights` — routing weights `[num_tokens, top_k]`
    /// * `routing_indices` — chosen experts `[num_tokens, top_k]` (global ids)
    /// * `group` — the ranks
    /// * `local_expert_fn` — `fn(rows, expert_ids)`: each row of `rows`
    ///   `[n, hidden_dim]` passes through the expert named at the same
    ///   position of `expert_ids` (always one this rank owns), giving
    ///   `[n, hidden_dim]`
    ///
    /// Every rank must call this together; ranks may hold different numbers of
    /// tokens.
    ///
    /// # Returns
    ///
    /// `[num_tokens, hidden_dim]` in the dtype of `hidden_states`: for each
    /// token, the sum over its experts of weight times expert output.
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
        self.check_group(group)?;
        let (tokens, dim, top_k) = Self::shapes(hidden_states, routing_weights, routing_indices)?;
        let dtype = hidden_states.dtype();
        let (rank, size) = (self.rank, self.world_size);
        let routing = self.route(routing_indices, tokens, top_k)?;

        // Everyone learns how many rows every rank sends to every other.
        let my_counts: Vec<f32> = routing
            .slots_for_rank
            .iter()
            .map(|s| s.len() as f32)
            .collect();
        let counts = read_counts(
            &ops::all_gather(
                &Array::from_f32_slice(&my_counts, &[size as i32]),
                Some(group),
            )?,
            size,
        )?;

        // Outbound: each destination's tokens, and the experts they use.
        let mut rows_out: Vec<Option<Array>> = vec![None; size];
        let mut ids_out: Vec<Option<Array>> = vec![None; size];
        for d in 0..size {
            let slots = &routing.slots_for_rank[d];
            if slots.is_empty() {
                continue;
            }
            let token_rows: Vec<usize> = slots.iter().map(|s| s / top_k).collect();
            rows_out[d] = Some(hidden_states.take_axis(&index_array(&token_rows), 0));
            let ids: Vec<i32> = routing.experts_for_rank[d]
                .iter()
                .map(|&e| e as i32)
                .collect();
            ids_out[d] = Some(Array::from_i32_slice(&ids));
        }
        let rows_in = ring_all_to_all(group, rows_out, &counts, &[dim as i32], dtype)?;
        let ids_in = ring_all_to_all(group, ids_out, &counts, &[], Dtype::Int32)?;

        // Compute every received row at once, then split the results by origin.
        let origins: Vec<usize> = (0..size).filter(|&o| counts[o][rank] > 0).collect();
        let mut results_out: Vec<Option<Array>> = vec![None; size];
        if !origins.is_empty() {
            let rows = concat_rows(
                origins
                    .iter()
                    .map(|&o| rows_in[o].as_ref().expect("counted")),
            );
            let mut expert_ids = Vec::new();
            for &o in &origins {
                let ids = ids_in[o].as_ref().expect("counted");
                expert_ids.extend(read_ints(ids)?.into_iter().map(|e| e as usize));
            }
            if let Some(&bad) = expert_ids
                .iter()
                .find(|&&e| self.placement.rank_for_expert(e) != rank)
            {
                return Err(Exception::custom(format!(
                    "dispatch: expert {bad} was routed to rank {rank}, which does not own it"
                )));
            }
            let result = local_expert_fn(&rows, &expert_ids)?;
            if result.shape() != rows.shape() {
                return Err(Exception::custom(format!(
                    "dispatch: the expert function returned {:?} for {:?} rows",
                    result.shape(),
                    rows.shape()
                )));
            }
            let result = result.as_dtype(dtype.as_i32());
            let mut at = 0;
            for &o in &origins {
                let n = counts[o][rank] as i32;
                results_out[o] = Some(result.slice(&[at, 0], &[at + n, dim as i32]));
                at += n;
            }
        }

        // Results travel the opposite way: what s sent d, d returns to s.
        let transposed: Vec<Vec<usize>> = (0..size)
            .map(|s| (0..size).map(|d| counts[d][s]).collect())
            .collect();
        let results_in = ring_all_to_all(group, results_out, &transposed, &[dim as i32], dtype)?;

        // Put each result back at the (token, slot) that asked for it, then
        // weight and sum over slots.
        let mut slot_of_row = Vec::new();
        let mut pieces = Vec::new();
        for (piece, slots) in results_in.iter().zip(&routing.slots_for_rank) {
            if let Some(piece) = piece {
                pieces.push(piece);
                slot_of_row.extend_from_slice(slots);
            }
        }
        let gathered = concat_rows(pieces.into_iter());
        let mut row_of_slot = vec![0usize; tokens * top_k];
        for (row, &slot) in slot_of_row.iter().enumerate() {
            row_of_slot[slot] = row;
        }
        let by_slot = gathered
            .take_axis(&index_array(&row_of_slot), 0)
            .as_dtype(Dtype::Float32.as_i32())
            .reshape(&[tokens as i32, top_k as i32, dim as i32]);
        let weights = routing_weights.as_dtype(Dtype::Float32.as_i32()).reshape(&[
            tokens as i32,
            top_k as i32,
            1,
        ]);
        Ok(mlx_ops::sum_axis(&by_slot.multiply(&weights), 1, false).as_dtype(dtype.as_i32()))
    }

    /// The mixture-of-experts layer when every rank holds the same tokens and
    /// routing: each rank runs the tokens routed to its own experts, scales
    /// them by their routing weights, and one `all_sum` adds the ranks'
    /// contributions. Same arguments and result as
    /// [`dispatch_and_compute`](Self::dispatch_and_compute), but the ranks'
    /// tokens must be identical.
    pub fn compute_replicated<F>(
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
        self.check_group(group)?;
        let (tokens, dim, top_k) = Self::shapes(hidden_states, routing_weights, routing_indices)?;
        let dtype = hidden_states.dtype();
        let f32 = Dtype::Float32.as_i32();
        let routing = self.route(routing_indices, tokens, top_k)?;
        let mine = &routing.slots_for_rank[self.rank];

        // Contribution of each routed slot (zero where another rank's expert
        // handles it), laid out by slot.
        let zero_row = mlx_ops::zeros(&[1, dim as i32], Dtype::Float32);
        let by_slot = if mine.is_empty() {
            mlx_ops::zeros(&[(tokens * top_k) as i32, dim as i32], Dtype::Float32)
        } else {
            let token_rows: Vec<usize> = mine.iter().map(|s| s / top_k).collect();
            let rows = hidden_states.take_axis(&index_array(&token_rows), 0);
            let out = local_expert_fn(&rows, &routing.experts_for_rank[self.rank])?;
            if out.shape() != rows.shape() {
                return Err(Exception::custom(format!(
                    "dispatch: the expert function returned {:?} for {:?} rows",
                    out.shape(),
                    rows.shape()
                )));
            }
            let weights = routing_weights
                .as_dtype(f32)
                .reshape(&[(tokens * top_k) as i32])
                .take_axis(&index_array(mine), 0)
                .reshape(&[mine.len() as i32, 1]);
            let weighted = out.as_dtype(f32).multiply(&weights);
            let mut row_of_slot = vec![mine.len(); tokens * top_k];
            for (row, &slot) in mine.iter().enumerate() {
                row_of_slot[slot] = row;
            }
            mlx_ops::concatenate_axis(&[&weighted, &zero_row], 0)
                .take_axis(&index_array(&row_of_slot), 0)
        };
        let local = mlx_ops::sum_axis(
            &by_slot.reshape(&[tokens as i32, top_k as i32, dim as i32]),
            1,
            false,
        );
        Ok(ops::all_sum(&local, Some(group))?.as_dtype(dtype.as_i32()))
    }
}

/// Read the `[size, size]` count matrix an all-gather of count vectors made.
fn read_counts(gathered: &Array, size: usize) -> Result<Vec<Vec<usize>>, Exception> {
    let mut g = gathered.clone();
    g.try_eval().map_err(|e| Exception::custom(e.to_string()))?;
    let flat = g
        .to_f32_vec(size * size)
        .ok_or_else(|| Exception::custom("dispatch: could not read the gathered counts"))?;
    Ok(flat
        .chunks(size)
        .map(|row| row.iter().map(|&c| c as usize).collect())
        .collect())
}

/// Join arrays along the first axis.
fn concat_rows<'a>(parts: impl Iterator<Item = &'a Array>) -> Array {
    let parts: Vec<&Array> = parts.collect();
    mlx_ops::concatenate_axis(&parts, 0)
}

/// All-to-all over neighbour exchanges (see the module docs).
///
/// `buckets[d]` is what this rank sends rank `d` (`None` when
/// `counts[rank][d] == 0`); `counts[s][d]` is how many rows rank `s` sends
/// rank `d`, known on every rank. Each row is an array of `trailing` shape.
/// Returns, indexed by origin, what each rank sent this one.
fn ring_all_to_all(
    group: &DistributedGroup,
    mut buckets: Vec<Option<Array>>,
    counts: &[Vec<usize>],
    trailing: &[i32],
    dtype: Dtype,
) -> Result<Vec<Option<Array>>, Exception> {
    let (rank, size) = (group.rank() as usize, group.size() as usize);
    let mut received: Vec<Option<Array>> = vec![None; size];
    received[rank] = buckets[rank].take();

    // Buckets in transit through this rank, keyed by (origin, destination).
    let mut held: HashMap<(usize, usize), Array> = HashMap::new();
    for (d, bucket) in buckets.into_iter().enumerate() {
        if let Some(bucket) = bucket {
            held.insert((rank, d), bucket);
        }
    }

    // Round `t`: rank `q` hands on every bucket of origin `q - t + 1` that
    // still has somewhere to go, i.e. destinations `t` or more hops past its
    // origin. What arrives is the same list for origin `q - t`.
    let parts = |origin: usize, t: usize| -> Vec<(usize, usize)> {
        (t..size)
            .map(|k| (origin, (origin + k) % size))
            .filter(|&(o, d)| counts[o][d] > 0)
            .collect()
    };
    for t in 1..size {
        let out_parts = parts((rank + size + 1 - t) % size, t);
        let outgoing = (!out_parts.is_empty()).then(|| {
            concat_rows(out_parts.iter().map(|key| {
                held.get(key)
                    .expect("a bucket to hand on is held: it arrived the round before")
            }))
        });
        let in_parts = parts((rank + size - t) % size, t);
        let in_rows: usize = in_parts.iter().map(|&(o, d)| counts[o][d]).sum();
        let mut shape = vec![in_rows as i32];
        shape.extend_from_slice(trailing);

        let arrived = exchange::exchange(
            group,
            outgoing.as_ref(),
            (in_rows > 0).then_some((&shape[..], dtype)),
        )?;
        for key in &out_parts {
            held.remove(key);
        }
        if let Some(arrived) = arrived {
            let mut at = 0;
            for &(o, d) in &in_parts {
                let n = counts[o][d] as i32;
                let mut start = vec![0; shape.len()];
                let mut stop = shape.clone();
                (start[0], stop[0]) = (at, at + n);
                let piece = arrived.slice(&start, &stop);
                at += n;
                if d == rank {
                    received[o] = Some(piece);
                } else {
                    held.insert((o, d), piece);
                }
            }
        }
    }
    Ok(received)
}

/// Result of Phase-1 token/expert classification: for rank `r`, parallel
/// lists of the flat `(token, slot)` index (`token * top_k + slot`) and the
/// expert it selected.
fn classify_by_destination_rank(
    experts: &[usize],
    num_tokens: usize,
    top_k: usize,
    world_size: usize,
    placement: &ExpertPlacement,
) -> Routing {
    let mut slots_for_rank: Vec<Vec<usize>> = vec![Vec::new(); world_size];
    let mut experts_for_rank: Vec<Vec<usize>> = vec![Vec::new(); world_size];
    for (slot, &expert) in experts.iter().enumerate().take(num_tokens * top_k) {
        let dest = placement.rank_for_expert(expert);
        slots_for_rank[dest].push(slot);
        experts_for_rank[dest].push(expert);
    }
    Routing {
        slots_for_rank,
        experts_for_rank,
    }
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
    fn classify_distributes_top_k_pairs_across_ranks() {
        // 4 experts, 2 ranks: rank 0 owns [0, 1], rank 1 owns [2, 3].
        // 3 tokens, top_k=2:
        //   token 0 → experts [0, 3] → ranks [0, 1]
        //   token 1 → experts [2, 1] → ranks [1, 0]
        //   token 2 → experts [1, 2] → ranks [0, 1]
        let experts = vec![0, 3, 2, 1, 1, 2];
        let placement = ExpertPlacement::uniform(4, 2, 2);
        let c = classify_by_destination_rank(&experts, 3, 2, 2, &placement);

        // Flat slot = token * 2 + k. Rank 0 gets (t0,k0,e0), (t1,k1,e1), (t2,k0,e1).
        assert_eq!(c.slots_for_rank[0], vec![0, 3, 4]);
        assert_eq!(c.experts_for_rank[0], vec![0, 1, 1]);
        // Rank 1: (t0,k1,e3), (t1,k0,e2), (t2,k1,e2).
        assert_eq!(c.slots_for_rank[1], vec![1, 2, 5]);
        assert_eq!(c.experts_for_rank[1], vec![3, 2, 2]);
    }

    #[test]
    fn classify_preserves_insertion_order_per_rank() {
        // Single rank: slots stay in (token, k) nested-loop order.
        let experts = vec![0, 0, 0, 0];
        let placement = ExpertPlacement::uniform(4, 1, 2);
        let c = classify_by_destination_rank(&experts, 2, 2, 1, &placement);
        assert_eq!(c.slots_for_rank[0], vec![0, 1, 2, 3]);
    }
}
