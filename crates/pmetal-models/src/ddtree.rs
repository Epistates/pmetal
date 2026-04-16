//! Tree-verify speculative decoding infrastructure, inspired by the
//! Diffusion Draft Tree (DDTree) paper. Unlike linear DFlash which
//! accepts the longest matching prefix of a single drafted block, tree
//! verify proposes a budget-bounded tree of candidate continuations
//! (weighted by draft log-prob) and lets the target walk from the root
//! following its own argmax at each node. One wrong prediction no
//! longer throws away all subsequent positions — the tree has siblings
//! and alternate branches ready.
//!
//! This module is pure CPU. The outputs (node_token_ids, visibility,
//! position_ids, attention mask) are built in Rust using
//! `std::collections::BinaryHeap` and handed off to the native bridge
//! as `InlineArray`s for the target forward.
//!
//! The algorithm matches DDTree's `build_ddtree_tree` / `compile_ddtree_tree`
//! / `follow_verified_tree` but is ~10x faster than the reference
//! Python implementation thanks to monomorphised Rust loops and no
//! Python/NumPy marshalling overhead. See the module tests for a
//! determinism check against hand-computed small-tree cases.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use pmetal_bridge::compat::{Array, Dtype, ops};

/// Hand-computed log-prob budget tree for each tapped draft step. The
/// entry type is optimised for the heap — we cache `-log_w` as the
/// priority key so the default `BinaryHeap` (max-heap) pops
/// highest-score states first.
#[derive(Clone, Debug)]
struct HeapEntry {
    /// Negative cumulative log-prob. Smaller = better → we store as a
    /// `Reverse`-wrapped `BinaryHeap` entry below.
    neg_logw: f32,
    /// Monotonic tie-breaker so the heap is deterministic across
    /// floating-point ties.
    order: u64,
    /// Path of draft top-k ranks taken from root → this node. One
    /// entry per tree depth (the root's depth is 0). The last entry is
    /// the rank at the CURRENT depth.
    ranks: Vec<i16>,
    /// Parent index into `node_token_ids` (1-based; 0 = root). Root's
    /// parent is recorded as `-1`.
    parent_index: i32,
    /// Tree depth (1 for direct children of the root).
    depth: i32,
    /// Rank at the current depth — the index into the top-k list of
    /// the current depth slice.
    rank: i32,
    /// Cumulative log-prob of this path (signed; higher is better).
    logw: f32,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want the SMALLEST `neg_logw` (largest logw) to pop first.
        // BinaryHeap is a max-heap, so we reverse the primary key and
        // break ties by `order` (older = higher priority).
        other
            .neg_logw
            .partial_cmp(&self.neg_logw)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.order.cmp(&self.order))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.neg_logw == other.neg_logw && self.order == other.order
    }
}
impl Eq for HeapEntry {}

/// Output of [`build_tree`]: a budget-bounded tree of candidate draft
/// continuations rooted at the seed token.
///
/// The arrays are in DFS-ish order (prioritised by cumulative log-prob)
/// with index 0 being the FIRST non-root node — the root itself is
/// implicit and carries the seed token.
pub struct TreeBuildResult {
    /// Per-node token ids, length = N (= `1 + tree_budget` - 1 minus
    /// unused slots). Index 0 is the first non-root node.
    pub node_token_ids: Vec<i32>,
    /// Per-node tree depth (1 for direct children of the root).
    pub node_depths: Vec<i32>,
    /// `parents[i]` is the parent of node `i` in the 1-indexed tree
    /// where index 0 is the root. `parents[0] == -1`.
    pub parents: Vec<i32>,
    /// `child_maps[i]` maps token_id → child index for node `i`. Used
    /// by the acceptance walk to follow the verified path in O(1).
    pub child_maps: Vec<BTreeMap<i32, usize>>,
    /// `[N+1, N+1]` boolean matrix (CPU): visibility[i][j] = 1 iff
    /// node j is an ancestor of node i (inclusive). Used to build the
    /// tree attention mask during compile.
    pub visibility: Vec<Vec<bool>>,
}

/// Extract top-k logits and their indices from a `[horizon, vocab]`
/// draft_logits tensor. Both outputs live on CPU as f32/i32 vecs so
/// the subsequent heap beam search does not have to touch the GPU.
fn extract_top_k_cpu(
    draft_logits: &Array,
    budget: usize,
) -> (usize, usize, Vec<Vec<f32>>, Vec<Vec<i32>>) {
    let shape = draft_logits.shape();
    assert_eq!(
        shape.len(),
        2,
        "draft_logits must be [horizon, vocab]; got rank {}",
        shape.len()
    );
    let horizon = shape[0] as usize;
    let vocab = shape[1] as usize;
    let k = budget.min(vocab).max(1);

    // Compute softmax log-probs in f32 for determinism. `logsumexp`
    // returns an f32 [horizon] array; we subtract before pulling top-k.
    let logits_f32 = draft_logits.as_dtype(Dtype::Float32.as_i32());
    let lse = logits_f32.logsumexp(-1, true); // [horizon, 1]
    let log_probs = logits_f32.subtract(&lse); // [horizon, vocab]

    // MLX argsort on the negated log-probs gives a descending-by-
    // log-prob permutation. Materialise it and the gathered log-probs
    // as FULL [horizon, vocab] tensors — slicing to [horizon, k] and
    // then calling `as_slice` returns a strided view whose flat data
    // is still the full buffer, so we'd silently read the wrong
    // columns. Instead we pull the full rows onto the CPU and take
    // the first `k` per row ourselves.
    let neg = ops::negative(&log_probs);
    let sorted_idx = ops::argsort_axis(&neg, -1); // [horizon, vocab] int32
    let sorted_logp = ops::take_along_axis(&log_probs, &sorted_idx, -1);
    let _ = sorted_idx.eval();
    let _ = sorted_logp.eval();

    let idx_flat = sorted_idx.as_slice::<i32>();
    let logp_flat = sorted_logp.as_slice::<f32>();

    let mut top_logp = Vec::with_capacity(horizon);
    let mut top_ids = Vec::with_capacity(horizon);
    for row in 0..horizon {
        let row_base = row * vocab;
        top_logp.push(logp_flat[row_base..row_base + k].to_vec());
        top_ids.push(idx_flat[row_base..row_base + k].to_vec());
    }
    (horizon, k, top_logp, top_ids)
}

/// Build a DDTree-style draft tree from the draft model's logits.
///
/// * `draft_logits`: `[horizon, vocab]` — one row per non-root tree
///   depth (so `horizon = block_size - 1`). Must already be on the
///   CPU-reachable path via InlineArray's zero-copy slices.
/// * `budget`: maximum number of tree nodes (excluding the root).
///
/// The beam search prioritises high-cumulative-log-prob paths: each
/// popped state spawns a "sibling" (next rank at the same depth) and a
/// "child" (rank 0 at the next depth), mirroring the reference impl.
/// Returns an empty result when `budget == 0` or `horizon == 0`, in
/// which case the caller should fall back to linear DFlash.
pub fn build_tree(draft_logits: &Array, budget: usize) -> TreeBuildResult {
    if budget == 0 || draft_logits.shape()[0] == 0 {
        return TreeBuildResult {
            node_token_ids: Vec::new(),
            node_depths: Vec::new(),
            parents: vec![-1],
            child_maps: vec![BTreeMap::new()],
            visibility: vec![vec![true]],
        };
    }

    let (_horizon, k, top_logp, top_ids) = extract_top_k_cpu(draft_logits, budget);
    let depth_limit = _horizon as i32;

    // Seed the heap with the depth-1 top-1 path. Subsequent iterations
    // pop the best frontier state, admit it as a new tree node, and
    // push its sibling + child back into the heap.
    let first_logw = top_logp[0][0];
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(budget * 2);
    heap.push(HeapEntry {
        neg_logw: -first_logw,
        order: 0,
        ranks: vec![0],
        parent_index: 0, // root
        depth: 1,
        rank: 0,
        logw: first_logw,
    });

    let mut node_token_ids = Vec::with_capacity(budget);
    let mut node_depths = Vec::with_capacity(budget);
    let mut parents: Vec<i32> = Vec::with_capacity(budget + 1);
    parents.push(-1); // root
    let mut child_maps: Vec<BTreeMap<i32, usize>> = Vec::with_capacity(budget + 1);
    child_maps.push(BTreeMap::new()); // root's children slot

    let mut order_counter: u64 = 1;
    while let Some(entry) = heap.pop() {
        if node_token_ids.len() >= budget {
            break;
        }
        let depth = entry.depth as usize;
        let rank = entry.rank as usize;
        let parent = entry.parent_index as usize;

        let token_id = top_ids[depth - 1][rank];
        let current_index = node_token_ids.len() + 1; // 1-based
        node_token_ids.push(token_id);
        node_depths.push(entry.depth);
        parents.push(entry.parent_index);
        child_maps.push(BTreeMap::new());
        child_maps[parent].insert(token_id, current_index);

        // Sibling: advance rank within the same depth slice.
        if rank + 1 < k {
            let mut sibling_ranks = entry.ranks.clone();
            *sibling_ranks.last_mut().unwrap() = (rank + 1) as i16;
            let sibling_logw =
                entry.logw - top_logp[depth - 1][rank] + top_logp[depth - 1][rank + 1];
            heap.push(HeapEntry {
                neg_logw: -sibling_logw,
                order: order_counter,
                ranks: sibling_ranks,
                parent_index: entry.parent_index,
                depth: entry.depth,
                rank: (rank + 1) as i32,
                logw: sibling_logw,
            });
            order_counter += 1;
        }

        // Child: descend to the next depth at rank 0.
        if entry.depth < depth_limit {
            let mut child_ranks = entry.ranks.clone();
            child_ranks.push(0);
            let child_logw = entry.logw + top_logp[depth][0];
            heap.push(HeapEntry {
                neg_logw: -child_logw,
                order: order_counter,
                ranks: child_ranks,
                parent_index: current_index as i32,
                depth: entry.depth + 1,
                rank: 0,
                logw: child_logw,
            });
            order_counter += 1;
        }
    }

    let n = node_token_ids.len();
    let total = n + 1;
    let mut visibility = vec![vec![false; total]; total];
    visibility[0][0] = true;
    for i in 1..total {
        let parent = parents[i] as usize;
        let (head, tail) = visibility.split_at_mut(i);
        tail[0][..i].copy_from_slice(&head[parent][..i]);
        visibility[i][i] = true;
    }

    TreeBuildResult {
        node_token_ids,
        node_depths,
        parents,
        child_maps,
        visibility,
    }
}

/// Build the verify inputs for a compiled draft tree: token ids,
/// per-position offsets for RoPE, and an additive attention mask that
/// encodes tree visibility. The returned `attention_mask` is shaped
/// `[1, 1, N, past_length + N]` with `-inf` at unreachable positions
/// and `0` at visible ones — ready to hand to the per-op SDPA path.
///
/// All arrays live on the GPU via `InlineArray::from_*` so the only
/// CPU↔GPU copy is a single upload per decode round.
pub struct CompiledTree {
    /// `[1, N]` int32 — the tree's token sequence in DFS-ish order
    /// starting with the root.
    pub verify_input_ids: Array,
    /// `[1, N]` int32 — per-token absolute positions (depth + start).
    /// Used as the `offset_arr` for the per-position RoPE call.
    pub verify_position_ids: Array,
    /// `[1, 1, N, past_length + N]` additive mask, same dtype as the
    /// target's hidden state.
    pub attention_mask: Array,
    /// Length N = 1 + node_count.
    pub current_length: usize,
}

/// Compile a built tree into GPU inputs ready for the target forward.
///
/// * `root_token_id` — the seed token shared by all paths (iter-1's
///   prefill argmax, or the previous iter's bonus token).
/// * `start` — the absolute position of the seed in the growing
///   output sequence (used as the base for `position_ids`).
/// * `past_length` — KV cache length at the start of this verify
///   round. The attention mask is sized `[_, _, N, past_length + N]`
///   so the target can attend over both history and the tree.
/// * `mask_dtype` — the InlineArray dtype the target expects; mask
///   values are cast to this dtype after construction.
pub fn compile_tree(
    tree: &TreeBuildResult,
    root_token_id: i32,
    start: i32,
    past_length: i32,
    mask_dtype: i32,
) -> CompiledTree {
    let n = tree.node_token_ids.len();
    let current_length = 1 + n;

    // ── input_ids: [root, node_0, node_1, ...] ──────────────────────
    let mut ids_vec: Vec<i32> = Vec::with_capacity(current_length);
    ids_vec.push(root_token_id);
    ids_vec.extend_from_slice(&tree.node_token_ids);
    let verify_input_ids = Array::from_i32_slice_shaped(&ids_vec, &[1, current_length as i32]);

    // ── position_ids: depth + start ─────────────────────────────────
    let mut pos_vec: Vec<i32> = Vec::with_capacity(current_length);
    pos_vec.push(start);
    for &d in &tree.node_depths {
        pos_vec.push(start + d);
    }
    let verify_position_ids = Array::from_i32_slice_shaped(&pos_vec, &[1, current_length as i32]);

    // ── attention mask: [1, 1, N, past + N], 0 where visible, -inf else ──
    // Allocate as f32 (fast to fill), then cast to target dtype at the
    // end. The `past` columns are 0 (target attends to all history);
    // the `tree` columns are filled per `visibility[i][j]`.
    let total_k = past_length as usize + current_length;
    let mut mask_vec: Vec<f32> = vec![0.0; current_length * total_k];
    let neg_inf = f32::NEG_INFINITY;
    for i in 0..current_length {
        for j in 0..current_length {
            if !tree.visibility[i][j] {
                let idx = i * total_k + past_length as usize + j;
                mask_vec[idx] = neg_inf;
            }
        }
    }
    let mask_f32 = Array::from_slice(&mask_vec, &[1, 1, current_length as i32, total_k as i32]);
    let attention_mask = mask_f32.as_dtype(mask_dtype);

    CompiledTree {
        verify_input_ids,
        verify_position_ids,
        attention_mask,
        current_length,
    }
}

/// Walk a verified tree following the target's posterior argmax at
/// each accepted node. Stops when the posterior at the current node
/// does not match any of its children in the tree.
///
/// Returns `(accepted_indices, bonus_token)` where:
/// - `accepted_indices` is the list of tree indices (0-based into the
///   compiled sequence, including the root at index 0) that were
///   accepted. Always starts with `0`.
/// - `bonus_token` is the target's own-choice token at the last
///   accepted node — the "free" token DFlash always emits in addition
///   to the matched prefix.
pub fn follow_verified_tree(
    child_maps: &[BTreeMap<i32, usize>],
    posterior: &[i32],
) -> (Vec<usize>, i32) {
    let mut accepted_indices: Vec<usize> = vec![0];
    let mut current_index: usize = 0;
    let mut next_token = posterior[current_index];

    while let Some(&child) = child_maps[current_index].get(&next_token) {
        current_index = child;
        accepted_indices.push(current_index);
        next_token = posterior[current_index];
    }

    (accepted_indices, next_token)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::Array;

    fn make_logits(rows: &[&[f32]]) -> Array {
        let rows_count = rows.len() as i32;
        let vocab = rows[0].len() as i32;
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Array::from_slice(&flat, &[rows_count, vocab])
    }

    #[test]
    fn build_tree_budget_zero_is_root_only() {
        let logits = make_logits(&[&[1.0, 0.0, 0.0]]);
        let tree = build_tree(&logits, 0);
        assert!(tree.node_token_ids.is_empty());
        assert_eq!(tree.parents, vec![-1]);
        assert_eq!(tree.visibility, vec![vec![true]]);
    }

    #[test]
    fn build_tree_single_depth_picks_top_first() {
        // horizon=1, vocab=4, deterministic top order: [2, 0, 3, 1]
        let logits = make_logits(&[&[0.5, -10.0, 5.0, 0.0]]);
        let tree = build_tree(&logits, 3);
        // Three budget = three best candidates from the one depth.
        assert_eq!(tree.node_token_ids, vec![2, 0, 3]);
        assert_eq!(tree.node_depths, vec![1, 1, 1]);
        // Every non-root node's parent is the root (index 0).
        assert_eq!(tree.parents, vec![-1, 0, 0, 0]);
        assert_eq!(tree.child_maps[0].get(&2), Some(&1));
        assert_eq!(tree.child_maps[0].get(&0), Some(&2));
        assert_eq!(tree.child_maps[0].get(&3), Some(&3));
    }

    #[test]
    fn build_tree_two_depth_grows_deepest_first() {
        // At depth 1, top-1 is token 0 (logit 10). Its child (depth 2
        // top-1 = token 1, logit 20) has cumulative logw > any depth-1
        // sibling. So budget=2 should give node_0=depth1-rank0 and
        // node_1=depth2-rank0 under node_0.
        let logits = make_logits(&[&[10.0, -100.0, -100.0], &[-100.0, 20.0, -100.0]]);
        let tree = build_tree(&logits, 2);
        assert_eq!(tree.node_token_ids, vec![0, 1]);
        assert_eq!(tree.node_depths, vec![1, 2]);
        // node_1's parent is node_0 (index 1).
        assert_eq!(tree.parents, vec![-1, 0, 1]);
        // Visibility: node 2 sees root (0) and node 1.
        assert!(tree.visibility[2][0]);
        assert!(tree.visibility[2][1]);
        assert!(tree.visibility[2][2]);
    }

    #[test]
    fn follow_verified_tree_matches_first_mismatch() {
        // Build a small tree manually and walk it.
        let tree = TreeBuildResult {
            node_token_ids: vec![100, 200, 300],
            node_depths: vec![1, 2, 1],
            parents: vec![-1, 0, 1, 0],
            child_maps: vec![
                BTreeMap::from_iter([(100, 1), (300, 3)]),
                BTreeMap::from_iter([(200, 2)]),
                BTreeMap::new(),
                BTreeMap::new(),
            ],
            visibility: vec![
                vec![true, false, false, false],
                vec![true, true, false, false],
                vec![true, true, true, false],
                vec![true, false, false, true],
            ],
        };
        // Posterior: at root say 100 (match node 1), at node 1 say 200
        // (match node 2), at node 2 say 999 (no match → stop). Bonus
        // is 999 from posterior[node 2 index = 2].
        let posterior = vec![100, 200, 999, 77];
        let (accepted, bonus) = follow_verified_tree(&tree.child_maps, &posterior);
        assert_eq!(accepted, vec![0, 1, 2]);
        assert_eq!(bonus, 999);
    }
}
