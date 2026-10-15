//! Splitting a whole number in proportion to weights.
//!
//! Experts over nodes and layers over nodes are the same problem: divide
//! `total` indivisible items in proportion to each node's weight (its memory,
//! say), so that every item is assigned exactly once and nobody is more than
//! one item from their fair share. Rounding each share on its own does not do
//! that, since many shares can round up at once, and handing the last node
//! "whatever is left" then over-allocates it.

/// Split `total` into whole parts proportional to `weights` that sum to
/// `total` (the largest-remainder method).
///
/// Each part is its exact share rounded down, and the items left over go one
/// each to the largest fractional remainders, the lower index on a tie.
/// Zero-weight entries get zero. If every weight is zero the items are shared
/// evenly instead, as if all were equal.
///
/// Arithmetic is exact (128-bit integers), so the result is reproducible on
/// every rank, which placement requires.
pub fn largest_remainder(total: usize, weights: &[u64]) -> Vec<usize> {
    if weights.is_empty() {
        return Vec::new();
    }
    let all_zero = weights.iter().all(|&w| w == 0);
    let weight = |i: usize| -> u128 { if all_zero { 1 } else { u128::from(weights[i]) } };
    let sum: u128 = (0..weights.len()).map(weight).sum();

    let mut parts = Vec::with_capacity(weights.len());
    let mut remainders = Vec::with_capacity(weights.len());
    for i in 0..weights.len() {
        let exact = total as u128 * weight(i);
        parts.push((exact / sum) as usize);
        remainders.push(exact % sum);
    }

    // Fewer items are left over than there are recipients, since each part
    // lost less than one item to rounding down.
    let leftover = total - parts.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| remainders[b].cmp(&remainders[a]).then(a.cmp(&b)));
    for &i in order.iter().take(leftover) {
        parts[i] += 1;
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_sum_to_the_total_and_follow_the_weights() {
        for total in [0usize, 1, 2, 3, 5, 10, 32, 100, 1001] {
            for weights in [
                vec![1u64],
                vec![1, 1],
                vec![1; 6],
                vec![3, 3, 3, 3],
                vec![10, 10, 1],
                vec![0, 4, 0, 4],
                vec![48, 16],
                vec![u64::MAX, u64::MAX / 3],
            ] {
                let parts = largest_remainder(total, &weights);
                assert_eq!(
                    parts.iter().sum::<usize>(),
                    total,
                    "{total} over {weights:?}"
                );
                let sum: f64 = weights.iter().map(|&w| w as f64).sum();
                for (i, (&p, &w)) in parts.iter().zip(&weights).enumerate() {
                    let fair = total as f64 * w as f64 / sum;
                    assert!(
                        (p as f64 - fair).abs() < 1.0 + 1e-9,
                        "{total} over {weights:?}: part {i} is {p}, fair share {fair}"
                    );
                    if w == 0 {
                        assert_eq!(p, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn ties_go_to_the_lower_index_and_zero_weights_share_evenly() {
        assert_eq!(largest_remainder(3, &[1; 6]), vec![1, 1, 1, 0, 0, 0]);
        assert_eq!(largest_remainder(7, &[0, 0, 0]), vec![3, 2, 2]);
        assert_eq!(largest_remainder(5, &[]), Vec::<usize>::new());
    }
}
