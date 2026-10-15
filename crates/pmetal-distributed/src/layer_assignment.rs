//! Layer assignment solvers for pipeline-parallel inference.
//!
//! Determines how to partition a model's decoder layers across multiple nodes.
//!
//! - [`assign_layers`] is the general solver: contiguous stages in a fixed
//!   node order, each node's weights within its memory, the slowest stage as
//!   fast as it can be. Exact (dynamic programming), `O(N L^2)`.
//! - [`assign_layers_proportional`] splits in proportion to a weight per node
//!   (memory, say) when nothing more is known about the model.
//! - [`best_ring_order`] chooses which node sits next to which, which is where
//!   link bandwidth matters: with nodes in a fixed order the activation
//!   crossing each boundary is the same size whichever layers fall either
//!   side, so no split changes what a link carries, but the order decides
//!   which links the pipeline has to use.

use crate::apportion::largest_remainder;
use crate::error::{DistributedError, DistributedResult};
use std::ops::Range;

/// Divide `num_layers` across nodes proportionally to `available_ram`, every
/// node getting at least one layer.
///
/// Returns contiguous, non-overlapping layer ranges that cover `0..num_layers`.
/// Shares are exact (largest-remainder apportionment): no node is more than
/// one layer from its proportional share.
pub fn assign_layers_proportional(
    num_layers: usize,
    available_ram: &[u64],
) -> DistributedResult<Vec<Range<usize>>> {
    let world_size = available_ram.len();
    if world_size == 0 {
        return Err(DistributedError::Config("need at least one node".into()));
    }
    if num_layers < world_size {
        return Err(DistributedError::Config(format!(
            "{world_size} nodes cannot share {num_layers} layers: each needs at least one"
        )));
    }
    // One layer each, the rest in proportion.
    let extra = largest_remainder(num_layers - world_size, available_ram);
    let mut start = 0;
    Ok(extra
        .into_iter()
        .map(|extra| {
            let range = start..start + 1 + extra;
            start = range.end;
            range
        })
        .collect())
}

/// One node of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageNode {
    /// Memory available for this stage's weights, in bytes.
    pub available_ram: u64,
    /// Relative speed (any unit, only ratios matter). Must be positive.
    pub speed: f64,
}

/// What to partition and over what.
#[derive(Debug, Clone)]
pub struct LayerProblem<'a> {
    /// Weight bytes of each decoder layer, in order.
    pub layer_bytes: &'a [u64],
    /// Relative compute cost of each layer. `None`: all layers cost the same.
    pub layer_cost: Option<&'a [f64]>,
    /// Extra bytes the first stage holds (the embedding).
    pub first_stage_bytes: u64,
    /// Extra bytes the last stage holds (final norm and output head).
    pub last_stage_bytes: u64,
    /// The nodes in pipeline order.
    pub nodes: &'a [StageNode],
}

/// Partition the layers into one contiguous, non-empty range per node, in
/// node order, so that every node's weights fit its memory and the slowest
/// stage (sum of its layers' cost over its speed) is as fast as possible.
/// Among partitions equally fast, the most even.
///
/// Fails if the model cannot fit the nodes in this order, which is a fact
/// about the cluster that no assignment can hide.
pub fn assign_layers(problem: &LayerProblem<'_>) -> DistributedResult<Vec<Range<usize>>> {
    let layers = problem.layer_bytes.len();
    let nodes = problem.nodes;
    let n = nodes.len();
    let config = |msg: String| Err(DistributedError::Config(msg));
    if n == 0 {
        return config("need at least one node".into());
    }
    if layers < n {
        return config(format!(
            "{n} nodes cannot share {layers} layers: each needs at least one"
        ));
    }
    if let Some(cost) = problem.layer_cost
        && (cost.len() != layers || cost.iter().any(|c| !c.is_finite() || *c < 0.0))
    {
        return config("layer_cost must give a finite, non-negative cost for every layer".into());
    }
    if nodes
        .iter()
        .any(|s| !(s.speed.is_finite() && s.speed > 0.0))
    {
        return config("every node needs a positive, finite speed".into());
    }

    // Prefix sums make any stage's memory and cost O(1).
    let mut bytes = vec![0u128; layers + 1];
    let mut cost = vec![0f64; layers + 1];
    for l in 0..layers {
        bytes[l + 1] = bytes[l] + u128::from(problem.layer_bytes[l]);
        cost[l + 1] = cost[l] + problem.layer_cost.map_or(1.0, |c| c[l]);
    }
    let fits = |stage: usize, from: usize, to: usize| -> bool {
        let mut need = bytes[to] - bytes[from];
        if stage == 0 {
            need += u128::from(problem.first_stage_bytes);
        }
        if stage == n - 1 {
            need += u128::from(problem.last_stage_bytes);
        }
        need <= u128::from(nodes[stage].available_ram)
    };
    let time = |stage: usize, from: usize, to: usize| (cost[to] - cost[from]) / nodes[stage].speed;

    // best[s][l]: the best way to give the first `l` layers to the first
    // `s + 1` nodes, as (slowest stage, sum of squared stage times) compared
    // in that order.
    type Score = (f64, f64);
    let better = |a: Score, b: Score| {
        let eps = 1e-12 * a.0.abs().max(b.0.abs()).max(1.0);
        if (a.0 - b.0).abs() > eps {
            a.0 < b.0
        } else {
            a.1 < b.1 - 1e-12 * a.1.abs().max(b.1.abs()).max(1.0)
        }
    };
    let mut best: Vec<Vec<Option<(Score, usize)>>> = vec![vec![None; layers + 1]; n];
    for l in 1..=layers {
        if fits(0, 0, l) {
            let t = time(0, 0, l);
            best[0][l] = Some(((t, t * t), 0));
        }
    }
    for s in 1..n {
        for l in (s + 1)..=layers {
            for k in s..l {
                let Some((prev, _)) = best[s - 1][k] else {
                    continue;
                };
                if !fits(s, k, l) {
                    continue;
                }
                let t = time(s, k, l);
                let candidate = (prev.0.max(t), prev.1 + t * t);
                if best[s][l].is_none_or(|(cur, _)| better(candidate, cur)) {
                    best[s][l] = Some((candidate, k));
                }
            }
        }
    }

    if best[n - 1][layers].is_none() {
        let total: u128 = bytes[layers]
            + u128::from(problem.first_stage_bytes)
            + u128::from(problem.last_stage_bytes);
        let ram: u128 = nodes.iter().map(|s| u128::from(s.available_ram)).sum();
        return config(format!(
            "the model's {total} bytes of weights do not fit these {n} nodes in this order \
             ({ram} bytes in total, and every node must hold at least one layer)"
        ));
    }

    // Walk the choices back to ranges.
    let mut ranges = vec![0..0; n];
    let mut end = layers;
    for s in (0..n).rev() {
        let (_, start) = best[s][end].expect("every state on the optimal path was reached");
        ranges[s] = start..end;
        end = start;
    }
    Ok(ranges)
}

/// The order in which to arrange `n` nodes in a ring so that the slowest link
/// it uses is as fast as possible; among those, the highest total bandwidth;
/// among those, the lexicographically smallest order (so every node reaches
/// the same answer from the same matrix).
///
/// `bandwidth[a][b]` is the bytes per second from node `a` to node `b`. The
/// ring runs `order[0] -> order[1] -> ... -> order[n-1] -> order[0]`; node 0
/// is always first. Exhaustive up to 10 nodes; beyond that a nearest-neighbour
/// tour improved by 2-opt moves.
pub fn best_ring_order(bandwidth: &[Vec<u64>]) -> DistributedResult<Vec<usize>> {
    let n = bandwidth.len();
    if bandwidth.iter().any(|row| row.len() != n) {
        return Err(DistributedError::Config(
            "the bandwidth matrix must be square".into(),
        ));
    }
    if n <= 2 {
        return Ok((0..n).collect());
    }
    // (slowest link, total) of a ring, compared bigger-is-better.
    let score = |order: &[usize]| -> (u64, u128) {
        let mut slowest = u64::MAX;
        let mut total = 0u128;
        for i in 0..n {
            let bw = bandwidth[order[i]][order[(i + 1) % n]];
            slowest = slowest.min(bw);
            total += u128::from(bw);
        }
        (slowest, total)
    };

    if n <= 10 {
        // All orders with node 0 first, in lexicographic order, so the first
        // best found is the lexicographically smallest.
        let mut order: Vec<usize> = (0..n).collect();
        let mut best = (score(&order), order.clone());
        let mut c = vec![0usize; n - 1];
        // Heap's algorithm visits permutations in no lexicographic order, so
        // compare orders on a tie instead.
        let mut i = 0;
        while i < n - 1 {
            if c[i] < i {
                if i % 2 == 0 {
                    order.swap(1, 1 + i);
                } else {
                    order.swap(1 + c[i], 1 + i);
                }
                let s = score(&order);
                if s > best.0 || (s == best.0 && order < best.1) {
                    best = (s, order.clone());
                }
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        return Ok(best.1);
    }

    // Nearest neighbour from node 0, then 2-opt until nothing improves.
    let mut order = vec![0usize];
    let mut left: Vec<usize> = (1..n).collect();
    while !left.is_empty() {
        let last = *order.last().expect("non-empty");
        let (at, _) = left
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| (bandwidth[last][c], std::cmp::Reverse(c)))
            .expect("non-empty");
        order.push(left.remove(at));
    }
    let mut improved = true;
    while improved {
        improved = false;
        for i in 1..n - 1 {
            for j in i + 1..n {
                let mut trial = order.clone();
                trial[i..=j].reverse();
                if score(&trial) > score(&order) {
                    order = trial;
                    improved = true;
                }
            }
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_equal() {
        let r = assign_layers_proportional(32, &[16_000, 16_000]).unwrap();
        assert_eq!(r, vec![0..16, 16..32]);
    }

    #[test]
    fn proportional_3x1() {
        let r = assign_layers_proportional(32, &[48_000, 16_000]).unwrap();
        assert_eq!(r, vec![0..24, 24..32]);
    }

    #[test]
    fn proportional_three_equal() {
        let r = assign_layers_proportional(30, &[10_000, 10_000, 10_000]).unwrap();
        assert_eq!(r, vec![0..10, 10..20, 20..30]);
    }

    #[test]
    fn minimum_one_layer_per_node() {
        let r = assign_layers_proportional(4, &[100, 100, 100, 100]).unwrap();
        assert_eq!(r.len(), 4);
        for range in &r {
            assert!(!range.is_empty(), "each node must get at least one layer");
        }
        // A node with almost no memory still gets its one layer.
        let r = assign_layers_proportional(10, &[1_000_000, 1]).unwrap();
        assert_eq!(r.iter().map(Range::len).sum::<usize>(), 10);
        assert!(r[1].len() >= 1);
    }

    /// Rounding each share and giving the last node the rest drifted from
    /// the proportional split: 7 equal nodes over 100 layers gave the last
    /// node 100 - 6 * 14 = 16 while the others held 14.
    #[test]
    fn proportional_shares_stay_within_one_layer_of_exact() {
        for (layers, ram) in [
            (100usize, vec![1u64; 7]),
            (31, vec![5, 3, 3, 1]),
            (64, vec![16, 32, 8, 8]),
            (9, vec![1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ] {
            let r = assign_layers_proportional(layers, &ram).unwrap();
            assert_eq!(r.first().unwrap().start, 0);
            assert_eq!(r.last().unwrap().end, layers);
            assert!(r.windows(2).all(|w| w[0].end == w[1].start));
            let total: f64 = ram.iter().map(|&x| x as f64).sum();
            let spare = (layers - ram.len()) as f64;
            for (range, &x) in r.iter().zip(&ram) {
                let fair = 1.0 + spare * x as f64 / total;
                assert!(
                    (range.len() as f64 - fair).abs() < 1.0 + 1e-9,
                    "{layers} layers over {ram:?}: {range:?} vs fair {fair}"
                );
            }
        }
    }

    #[test]
    fn impossible_requests_are_errors_not_panics() {
        assert!(assign_layers_proportional(4, &[]).is_err());
        assert!(assign_layers_proportional(2, &[1, 1, 1]).is_err());
    }

    fn nodes(specs: &[(u64, f64)]) -> Vec<StageNode> {
        specs
            .iter()
            .map(|&(available_ram, speed)| StageNode {
                available_ram,
                speed,
            })
            .collect()
    }

    /// Every partition of `layers` into `n` non-empty contiguous ranges.
    fn all_partitions(layers: usize, n: usize) -> Vec<Vec<Range<usize>>> {
        fn go(
            start: usize,
            layers: usize,
            left: usize,
            acc: &mut Vec<Range<usize>>,
            out: &mut Vec<Vec<Range<usize>>>,
        ) {
            if left == 1 {
                acc.push(start..layers);
                out.push(acc.clone());
                acc.pop();
                return;
            }
            for end in start + 1..=layers - (left - 1) {
                acc.push(start..end);
                go(end, layers, left - 1, acc, out);
                acc.pop();
            }
        }
        let mut out = Vec::new();
        go(0, layers, n, &mut Vec::new(), &mut out);
        out
    }

    /// The solver finds the true optimum: no partition that fits has a
    /// faster slowest stage.
    #[test]
    fn the_solver_matches_exhaustive_search() {
        let mut seed = 0x1234_5678_9ABC_DEFu64;
        let mut next = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        for _ in 0..200 {
            let n = 1 + next(4) as usize;
            let layers = n + next(8) as usize;
            let layer_bytes: Vec<u64> = (0..layers).map(|_| 1 + next(9)).collect();
            let cost: Vec<f64> = (0..layers).map(|_| 0.5 + next(8) as f64).collect();
            let stage_nodes = nodes(
                &(0..n)
                    .map(|_| (8 + next(30), 0.5 + next(4) as f64))
                    .collect::<Vec<_>>(),
            );
            let (first, last) = (next(4), next(4));
            let problem = LayerProblem {
                layer_bytes: &layer_bytes,
                layer_cost: Some(&cost),
                first_stage_bytes: first,
                last_stage_bytes: last,
                nodes: &stage_nodes,
            };

            let stage_time = |ranges: &[Range<usize>]| -> f64 {
                ranges
                    .iter()
                    .enumerate()
                    .map(|(s, r)| cost[r.clone()].iter().sum::<f64>() / stage_nodes[s].speed)
                    .fold(0.0, f64::max)
            };
            let feasible = |ranges: &[Range<usize>]| -> bool {
                ranges.iter().enumerate().all(|(s, r)| {
                    let mut need: u64 = layer_bytes[r.clone()].iter().sum();
                    if s == 0 {
                        need += first;
                    }
                    if s == n - 1 {
                        need += last;
                    }
                    need <= stage_nodes[s].available_ram
                })
            };
            let want = all_partitions(layers, n)
                .iter()
                .filter(|p| feasible(p))
                .map(|p| stage_time(p))
                .fold(f64::INFINITY, f64::min);

            match assign_layers(&problem) {
                Ok(ranges) => {
                    assert!(want.is_finite(), "solver found a plan where none exists");
                    assert!(
                        feasible(&ranges),
                        "plan exceeds a node's memory: {ranges:?}"
                    );
                    assert_eq!(ranges.first().unwrap().start, 0);
                    assert_eq!(ranges.last().unwrap().end, layers);
                    assert!(ranges.windows(2).all(|w| w[0].end == w[1].start));
                    assert!(ranges.iter().all(|r| !r.is_empty()));
                    assert!(
                        (stage_time(&ranges) - want).abs() <= 1e-9 * want.max(1.0),
                        "slowest stage {} vs optimum {want}",
                        stage_time(&ranges)
                    );
                }
                Err(e) => assert!(
                    want.is_infinite(),
                    "solver gave up ({e}) though a plan exists"
                ),
            }
        }
    }

    #[test]
    fn a_faster_node_gets_proportionally_more_layers_and_ties_are_even() {
        let stage_nodes = nodes(&[(u64::MAX, 1.0), (u64::MAX, 2.0), (u64::MAX, 1.0)]);
        let bytes = vec![1u64; 40];
        let r = assign_layers(&LayerProblem {
            layer_bytes: &bytes,
            layer_cost: None,
            first_stage_bytes: 0,
            last_stage_bytes: 0,
            nodes: &stage_nodes,
        })
        .unwrap();
        assert_eq!(
            r.iter().map(Range::len).collect::<Vec<_>>(),
            vec![10, 20, 10]
        );

        // Equal nodes split evenly.
        let even = nodes(&[(u64::MAX, 1.0); 4]);
        let r = assign_layers(&LayerProblem {
            layer_bytes: &bytes,
            layer_cost: None,
            first_stage_bytes: 0,
            last_stage_bytes: 0,
            nodes: &even,
        })
        .unwrap();
        assert_eq!(r.iter().map(Range::len).collect::<Vec<_>>(), vec![10; 4]);
    }

    /// A node that cannot hold its proportional share gets fewer layers, and
    /// the embedding and head count against the first and last node.
    #[test]
    fn memory_limits_are_respected() {
        // Layers are 10 bytes each; node 0 holds only 3 layers' worth and the
        // 25-byte embedding, which leaves it no room for any.
        let bytes = vec![10u64; 12];
        let stage_nodes = nodes(&[(35, 1.0), (1000, 1.0), (1000, 1.0)]);
        let r = assign_layers(&LayerProblem {
            layer_bytes: &bytes,
            layer_cost: None,
            first_stage_bytes: 25,
            last_stage_bytes: 0,
            nodes: &stage_nodes,
        })
        .unwrap();
        assert_eq!(r[0].len(), 1, "{r:?}");

        // Together they cannot hold the model: an error that says so.
        let tiny = nodes(&[(50, 1.0), (50, 1.0)]);
        let err = assign_layers(&LayerProblem {
            layer_bytes: &bytes,
            layer_cost: None,
            first_stage_bytes: 0,
            last_stage_bytes: 0,
            nodes: &tiny,
        })
        .unwrap_err();
        assert!(err.to_string().contains("do not fit"), "{err}");
    }

    #[test]
    fn bad_problems_are_errors() {
        let ok = nodes(&[(100, 1.0), (100, 1.0)]);
        let bytes = vec![1u64; 4];
        let problem = |nodes: &'_ [StageNode], cost: Option<&'_ [f64]>| {
            assign_layers(&LayerProblem {
                layer_bytes: &bytes,
                layer_cost: cost,
                first_stage_bytes: 0,
                last_stage_bytes: 0,
                nodes,
            })
        };
        assert!(problem(&[], None).is_err());
        assert!(problem(&nodes(&[(100, 0.0), (100, 1.0)]), None).is_err());
        assert!(problem(&nodes(&[(100, f64::NAN), (100, 1.0)]), None).is_err());
        assert!(problem(&ok, Some(&[1.0, 1.0])).is_err());
        assert!(problem(&ok, Some(&[1.0, -1.0, 1.0, 1.0])).is_err());
        assert!(problem(&nodes(&[(100, 1.0); 5]), None).is_err());
        assert!(problem(&ok, None).is_ok());
    }

    fn matrix(n: usize, fast: &[(usize, usize)]) -> Vec<Vec<u64>> {
        let mut m = vec![vec![1u64; n]; n];
        for &(a, b) in fast {
            m[a][b] = 100;
            m[b][a] = 100;
        }
        m
    }

    #[test]
    fn the_ring_follows_the_fast_links() {
        // Thunderbolt cables join 0-2, 2-1, 1-3, 3-0; everything else is
        // Ethernet. Sorted order 0-1-2-3 would cross slow links; 0-2-1-3
        // (or its reverse) uses only fast ones.
        let bw = matrix(4, &[(0, 2), (2, 1), (1, 3), (3, 0)]);
        let order = best_ring_order(&bw).unwrap();
        let slowest = (0..4)
            .map(|i| bw[order[i]][order[(i + 1) % 4]])
            .min()
            .unwrap();
        assert_eq!(slowest, 100, "ring {order:?} uses a slow link");
        assert_eq!(order[0], 0);
    }

    #[test]
    fn ring_order_is_deterministic_and_complete() {
        for n in [1usize, 2, 3, 5, 8] {
            let bw = matrix(n, &[]);
            let a = best_ring_order(&bw).unwrap();
            let b = best_ring_order(&bw).unwrap();
            assert_eq!(a, b);
            let mut sorted = a.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
        }
        // Equal links: the smallest order wins.
        assert_eq!(
            best_ring_order(&matrix(5, &[])).unwrap(),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn a_large_ring_is_still_planned_well() {
        // 12 nodes in a Thunderbolt loop, shuffled labels.
        let order: Vec<usize> = vec![0, 7, 3, 9, 1, 11, 5, 2, 10, 4, 8, 6];
        let fast: Vec<(usize, usize)> = (0..12).map(|i| (order[i], order[(i + 1) % 12])).collect();
        let bw = matrix(12, &fast);
        let found = best_ring_order(&bw).unwrap();
        let slowest = (0..12)
            .map(|i| bw[found[i]][found[(i + 1) % 12]])
            .min()
            .unwrap();
        assert_eq!(slowest, 100, "heuristic missed the loop: {found:?}");
    }

    #[test]
    fn a_non_square_matrix_is_an_error() {
        assert!(best_ring_order(&[vec![1, 2], vec![3]]).is_err());
    }
}
