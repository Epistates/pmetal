//! Per-input-channel activation statistics of a model's weight matrices.
//!
//! A weight matrix `W [out, in]` computes `y = x W^T`, so the error that
//! quantizing `W` puts into `y` is `sum_j x_j^2 * sum_i dW_ij^2` (in
//! expectation): the columns of `W` that meet large activations matter most.
//! This module measures `sum_t x_tj^2` for each column `j` of each matrix over
//! the tokens `t` a calibration run feeds the model, which is what an
//! importance matrix holds.
//!
//! # Seam
//!
//! The model's forward is not changed or wrapped. The layers that multiply by
//! a weight ([`Linear`](crate::compat::Linear), the tied output head, the
//! routed experts) call one of the `observe_*` functions here with the weight
//! and the activation. With no [`Collector`] installed on the thread that call
//! is one thread-local flag read; nothing is computed, allocated or kept, so
//! the path costs nothing when not collecting.
//!
//! A weight is named by the identity of its array ([`InlineArray::id`]), which
//! is stable for the array's lifetime and valid on lazy arrays: the caller
//! registers `(name, weight)` for the matrices it wants measured, and an
//! observation of any other weight is ignored. The weights must outlive the
//! collector.
//!
//! # Accumulation
//!
//! The sums are built lazily on the GPU in `f32` (activations are squared after
//! the cast, so a `bf16` model loses nothing to the square), one `[in]` vector
//! per dense matrix and one `[experts, in]` matrix per stack of routed
//! experts, together with the number of tokens that went in (per expert for a
//! stack, counting only the tokens routed to it). Memory is bounded by the
//! model's widths, not by the amount of data: call [`Collector::flush`] after
//! each forward so the activations that feed the sums are released.
//!
//! ```no_run
//! use pmetal_runtime::{InlineArray, activation_stats::Collector};
//! # fn weights() -> Vec<(String, InlineArray)> { vec![] }
//! # fn forward() {}
//! let weights = weights();
//! let mut collector = Collector::install(weights.iter().map(|(n, w)| (n.clone(), w)))?;
//! for _chunk in 0..4 {
//!     forward();
//!     collector.flush();
//! }
//! let stats = collector.finish();
//! # Ok::<(), String>(())
//! ```

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::InlineArray;
use crate::dtype::{F32, I32};

thread_local! {
    /// Whether a collector is installed on this thread: the only thing the
    /// disabled path reads.
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// What a collector has seen of one weight matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelStats {
    /// The name the weight was registered under.
    pub name: String,
    /// Experts of a stacked weight, `None` for a plain matrix.
    pub experts: Option<usize>,
    /// Input channels (columns).
    pub in_features: usize,
    /// `sum_t x_tj^2` per channel: `in_features` values, or `in_features` per
    /// expert, expert after expert.
    pub sum_sq: Vec<f32>,
    /// Tokens accumulated: one count for a plain matrix, one per expert (the
    /// tokens routed to it) for a stack.
    pub counts: Vec<f64>,
}

impl ChannelStats {
    /// The mean squared activation per channel (per expert, for a stack): the
    /// sums divided by the tokens behind them. An expert no token reached has
    /// no mean and gets zeros.
    pub fn mean_sq(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.sum_sq.len());
        for (slice, &count) in self.counts.iter().enumerate() {
            let inv = if count > 0.0 { 1.0 / count } else { 0.0 };
            out.extend(
                self.sum_sq[slice * self.in_features..(slice + 1) * self.in_features]
                    .iter()
                    .map(|&v| (f64::from(v) * inv) as f32),
            );
        }
        out
    }
}

struct Target {
    name: String,
    /// `Some(E)` for a `[E, ..]` stack.
    experts: Option<usize>,
    in_features: Option<usize>,
    sum: Option<InlineArray>,
    /// `[E]` for a stack.
    counts: Option<InlineArray>,
    /// Tokens, for a plain matrix (known from shapes, no readback needed).
    rows: u64,
}

struct State {
    targets: HashMap<usize, Target>,
    /// Observations that did not fit the weight they named (a shape the
    /// accumulator could not take).
    rejected: u64,
}

/// Collects activation statistics for the registered weights of this thread
/// until it is finished or dropped. At most one per thread.
pub struct Collector {
    /// A collector belongs to the thread whose flag it set.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Collector {
    /// Start measuring `weights`, each registered under a name. A weight with
    /// three dimensions is a stack of experts `[E, out, in]`, one with two a
    /// plain matrix `[out, in]`.
    ///
    /// # Errors
    ///
    /// A collector is already installed on this thread, a weight has some
    /// other rank, or two registered weights are the same array.
    pub fn install<'a>(
        weights: impl IntoIterator<Item = (String, &'a InlineArray)>,
    ) -> Result<Self, String> {
        if ACTIVE.with(Cell::get) {
            return Err("a collector is already installed on this thread".into());
        }
        let mut targets = HashMap::new();
        for (name, weight) in weights {
            let experts = match weight.ndim() {
                2 => None,
                3 => Some(weight.dim(0) as usize),
                n => {
                    return Err(format!(
                        "{name}: a {n}-dimensional weight cannot be measured"
                    ));
                }
            };
            let target = Target {
                name: name.clone(),
                experts,
                in_features: None,
                sum: None,
                counts: None,
                rows: 0,
            };
            if let Some(previous) = targets.insert(weight.id(), target) {
                return Err(format!("{name} and {} are the same array", previous.name));
            }
        }
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                targets,
                rejected: 0,
            });
        });
        ACTIVE.with(|a| a.set(true));
        Ok(Self {
            _not_send: std::marker::PhantomData,
        })
    }

    /// Evaluate the sums gathered so far and drop the activations behind
    /// them. Call it after every forward: until then the sums hold the graph
    /// that feeds them.
    pub fn flush(&mut self) {
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            let Some(state) = s.as_mut() else { return };
            let mut arrays: Vec<&mut InlineArray> = Vec::new();
            for target in state.targets.values_mut() {
                arrays.extend(target.sum.as_mut());
                arrays.extend(target.counts.as_mut());
            }
            crate::inline_array::eval_and_detach_many(&mut arrays);
        });
    }

    /// Observations that did not fit the weight they named, so were not
    /// counted: a number other than zero means the statistics are incomplete.
    pub fn rejected(&self) -> u64 {
        STATE.with(|s| s.borrow().as_ref().map_or(0, |state| state.rejected))
    }

    /// Stop collecting and return what was seen, by name. A registered weight
    /// that never met an activation is not in the result.
    pub fn finish(mut self) -> Vec<ChannelStats> {
        self.flush();
        let state = STATE.with(|s| s.borrow_mut().take());
        ACTIVE.with(|a| a.set(false));
        let Some(state) = state else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (_, mut target) in state.targets {
            let (Some(in_features), Some(mut sum)) = (target.in_features, target.sum.take()) else {
                continue;
            };
            let n = sum.size();
            let Some(sum_sq) = sum.to_f32_vec(n) else {
                continue;
            };
            let counts = match (target.experts, target.counts.take()) {
                (None, _) => vec![target.rows as f64],
                (Some(_), Some(mut counts)) => {
                    let n = counts.size();
                    match counts.to_f32_vec(n) {
                        Some(c) => c.into_iter().map(f64::from).collect(),
                        None => continue,
                    }
                }
                (Some(_), None) => continue,
            };
            out.push(ChannelStats {
                name: target.name,
                experts: target.experts,
                in_features,
                sum_sq,
                counts,
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }
}

impl Drop for Collector {
    fn drop(&mut self) {
        // `finish` has already cleared both; a collector dropped any other way
        // (a panic in the forward) must not leave the flag set.
        if ACTIVE.with(Cell::get) {
            STATE.with(|s| *s.borrow_mut() = None);
            ACTIVE.with(|a| a.set(false));
        }
    }
}

/// Whether a collector is installed on this thread.
#[inline]
pub fn is_collecting() -> bool {
    ACTIVE.with(Cell::get)
}

/// `x [.., in]` met the plain matrix `weight [out, in]`.
#[inline]
pub fn observe_matmul(weight: &InlineArray, x: &InlineArray) {
    if ACTIVE.with(Cell::get) {
        observe_matmul_collecting(weight, x);
    }
}

/// Every token of `x [.., in]` went through each of the experts `ids [.., k]`
/// of the stack `weight [E, .., in]` picked for it: the input of a routed
/// layer's gate and up projections, which all see the layer's input.
#[inline]
pub fn observe_experts_per_token(weight: &InlineArray, x: &InlineArray, ids: &InlineArray) {
    if ACTIVE.with(Cell::get) {
        observe_experts_collecting(weight, x, ids, true);
    }
}

/// Row `r` of `rows [.., in]` went through the one expert `ids` names for it
/// (`rows` holds exactly one row per id, in the same order, however its
/// leading axes are laid out): the input of a routed layer's down
/// projection, which differs per (token, expert) pair.
#[inline]
pub fn observe_experts_per_slot(weight: &InlineArray, rows: &InlineArray, ids: &InlineArray) {
    if ACTIVE.with(Cell::get) {
        observe_experts_collecting(weight, rows, ids, false);
    }
}

fn accumulate(slot: &mut Option<InlineArray>, add: InlineArray) {
    *slot = Some(match slot.take() {
        Some(sum) => sum.add(&add),
        None => add,
    });
}

#[inline(never)]
fn observe_matmul_collecting(weight: &InlineArray, x: &InlineArray) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(state) = s.as_mut() else { return };
        let Some(target) = state.targets.get_mut(&weight.id()) else {
            return;
        };
        let in_features = x.dim(-1) as usize;
        if target.experts.is_some()
            || x.ndim() < 1
            || in_features == 0
            || target.in_features.is_some_and(|f| f != in_features)
        {
            state.rejected += 1;
            return;
        }
        let rows = x.size() / in_features;
        let sum_sq = x
            .as_dtype(F32)
            .square()
            .reshape(&[rows as i32, in_features as i32])
            .sum_axis(0, false);
        target.in_features = Some(in_features);
        target.rows += rows as u64;
        accumulate(&mut target.sum, sum_sq);
    });
}

#[inline(never)]
fn observe_experts_collecting(
    weight: &InlineArray,
    x: &InlineArray,
    ids: &InlineArray,
    per_token: bool,
) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(state) = s.as_mut() else { return };
        let Some(target) = state.targets.get_mut(&weight.id()) else {
            return;
        };
        let Some(experts) = target.experts else {
            state.rejected += 1;
            return;
        };
        let in_features = x.dim(-1) as usize;
        let n_ids = ids.size();
        // Per token: `ids` is `[tokens, k]`; per slot: one row per id.
        let (rows, k) = if per_token {
            let k = ids.dim(-1) as usize;
            (n_ids.checked_div(k).unwrap_or(0), k)
        } else {
            (n_ids, 1)
        };
        if in_features == 0
            || k == 0
            || x.size() != rows * in_features
            || target.in_features.is_some_and(|f| f != in_features)
        {
            state.rejected += 1;
            return;
        }

        // one_hot[r, k, e] = ids[r, k] == e
        let wanted = ids.as_dtype(I32).reshape(&[rows as i32, k as i32, 1]);
        let all = InlineArray::arange(experts as i32, I32).reshape(&[1, 1, experts as i32]);
        let one_hot = wanted.equal(&all).as_dtype(F32);
        let x2 = x
            .as_dtype(F32)
            .square()
            .reshape(&[rows as i32, in_features as i32]);

        // Per expert, the sum over the rows that reached it: a token counts once
        // per expert however it got there, so the k picks collapse first.
        let routed = one_hot.sum_axis(1, false); // [rows, E]
        let sum_sq = routed.t().matmul(&x2); // [E, in]
        let counts = routed.sum_axis(0, false); // [E]
        target.in_features = Some(in_features);
        accumulate(&mut target.sum, sum_sq);
        accumulate(&mut target.counts, counts);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array(data: &[f32], shape: &[i32]) -> InlineArray {
        InlineArray::from_f32_slice(data, shape)
    }

    fn ids(data: &[i32], shape: &[i32]) -> InlineArray {
        InlineArray::from_slice(data, shape)
    }

    fn only(stats: Vec<ChannelStats>) -> ChannelStats {
        assert_eq!(stats.len(), 1, "{stats:?}");
        stats.into_iter().next().unwrap()
    }

    #[test]
    fn nothing_is_collected_without_a_collector() {
        let w = array(&[0.0; 6], &[2, 3]);
        let x = array(&[1.0, 2.0, 3.0], &[1, 3]);
        assert!(!is_collecting());
        observe_matmul(&w, &x);
        // A collector installed afterwards starts empty.
        let collector = Collector::install([("w".to_string(), &w)]).unwrap();
        assert!(collector.finish().is_empty());
    }

    #[test]
    fn a_plain_matrix_sums_the_squares_of_its_inputs_per_column() {
        let w = array(&[0.0; 6], &[2, 3]);
        let mut collector = Collector::install([("w".to_string(), &w)]).unwrap();
        // Two forwards: [2, 3] and [1, 2, 3] (leading axes are tokens, however many).
        observe_matmul(&w, &array(&[1.0, 2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]));
        collector.flush();
        observe_matmul(&w, &array(&[0.5, 0.5, 2.0, 1.0, 1.0, 1.0], &[1, 2, 3]));
        let stats = only(collector.finish());
        assert_eq!(stats.name, "w");
        assert_eq!(stats.experts, None);
        assert_eq!(stats.in_features, 3);
        assert_eq!(stats.counts, vec![4.0]);
        // column 0: 1 + 16 + .25 + 1; column 1: 4 + 25 + .25 + 1; column 2: 9 + 36 + 4 + 1
        assert_eq!(stats.sum_sq, vec![18.25, 30.25, 50.0]);
        assert_eq!(stats.mean_sq(), vec![4.5625, 7.5625, 12.5]);
    }

    #[test]
    fn linear_layers_and_the_tied_head_report_their_inputs() {
        use crate::compat::Linear;
        use crate::compat::nn::Embedding;
        let linear = Linear::create(3, 2, true);
        let embedding = Embedding::new(5, 3).unwrap();
        let collector = Collector::install([
            ("linear".to_string(), &linear.weight.value),
            ("head".to_string(), &embedding.weight.value),
        ])
        .unwrap();
        let x = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = linear.forward(&x);
        let logits = embedding.as_linear(&y.as_dtype(F32).matmul(&array(&[1.0; 6], &[2, 3])));
        // Observing does not change the output's shape or defer anything the caller sees.
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(logits.shape(), &[2, 5]);
        let stats = collector.finish();
        let by_name = |n: &str| stats.iter().find(|s| s.name == n).unwrap();
        assert_eq!(by_name("linear").sum_sq, vec![17.0, 29.0, 45.0]);
        assert_eq!(by_name("head").in_features, 3);
        assert_eq!(by_name("head").counts, vec![2.0]);
    }

    #[test]
    fn a_half_precision_activation_is_squared_in_f32() {
        let w = array(&[0.0; 4], &[2, 2]);
        let collector = Collector::install([("w".to_string(), &w)]).unwrap();
        // 1 + 2^-7 is a bf16; its square, 1 + 2^-6 + 2^-14, is not (bf16 keeps 7
        // fraction bits and would round the 2^-14 away), but is exact in f32.
        let v = 1.0f32 + 1.0 / 128.0;
        let x = array(&[v, 1.0], &[1, 2]).as_dtype(crate::dtype::BF16);
        observe_matmul(&w, &x);
        let stats = only(collector.finish());
        assert_eq!(stats.sum_sq[0], v * v);
        assert_ne!(stats.sum_sq[0], 1.0 + 1.0 / 64.0);
    }

    #[test]
    fn other_weights_and_mismatched_shapes_are_ignored_or_counted() {
        let (w, other) = (array(&[0.0; 6], &[2, 3]), array(&[0.0; 6], &[2, 3]));
        let collector = Collector::install([("w".to_string(), &w)]).unwrap();
        observe_matmul(&other, &array(&[1.0, 1.0, 1.0], &[1, 3]));
        assert_eq!(collector.rejected(), 0);
        observe_matmul(&w, &array(&[1.0, 1.0, 1.0], &[1, 3]));
        // The same weight cannot suddenly take a different width.
        observe_matmul(&w, &array(&[1.0, 1.0], &[1, 2]));
        assert_eq!(collector.rejected(), 1);
        let stats = only(collector.finish());
        assert_eq!(stats.counts, vec![1.0]);
    }

    #[test]
    fn one_collector_per_thread_and_the_flag_is_released() {
        let w = array(&[0.0; 6], &[2, 3]);
        let first = Collector::install([("w".to_string(), &w)]).unwrap();
        assert!(Collector::install([("w".to_string(), &w)]).is_err());
        drop(first);
        assert!(!is_collecting());
        let again = Collector::install([("w".to_string(), &w)]).unwrap();
        drop(again);
        let err = Collector::install([("a".to_string(), &w), ("b".to_string(), &w)]).err();
        assert!(err.unwrap().contains("same array"));
        assert!(!is_collecting());
    }

    /// Tokens 0..3 over four experts, two experts each.
    fn routed_case() -> (InlineArray, InlineArray, InlineArray) {
        let w = array(&[0.0; 4 * 2 * 2], &[4, 2, 2]); // [E, out, in]
        let x = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let ids = ids(&[0, 2, 2, 1, 0, 1], &[3, 2]);
        (w, x, ids)
    }

    #[test]
    fn experts_count_only_the_tokens_routed_to_them() {
        let (w, x, ids) = routed_case();
        let collector = Collector::install([("e".to_string(), &w)]).unwrap();
        observe_experts_per_token(&w, &x, &ids);
        let stats = only(collector.finish());
        assert_eq!(stats.experts, Some(4));
        assert_eq!(stats.in_features, 2);
        // token 0 -> {0, 2}, token 1 -> {2, 1}, token 2 -> {0, 1}
        assert_eq!(stats.counts, vec![2.0, 2.0, 2.0, 0.0]);
        assert_eq!(
            stats.sum_sq,
            vec![
                1.0 + 25.0,
                4.0 + 36.0, // expert 0: tokens 0 and 2
                9.0 + 25.0,
                16.0 + 36.0, // expert 1: tokens 1 and 2
                1.0 + 9.0,
                4.0 + 16.0, // expert 2: tokens 0 and 1
                0.0,
                0.0, // expert 3: none
            ]
        );
        // An expert no token reached has no mean.
        assert_eq!(&stats.mean_sq()[6..], &[0.0, 0.0]);
        assert_eq!(&stats.mean_sq()[..2], &[13.0, 20.0]);
    }

    #[test]
    fn per_slot_rows_in_either_layout_give_the_same_sums() {
        let (w, _, ids_rt) = routed_case();
        // One activation row per (token, k) pair: row (t, k) = [10 t + k, 10 t + k + 0.5].
        let rows: Vec<f32> = (0..3)
            .flat_map(|t| {
                (0..2).flat_map(move |k| {
                    [10.0 * t as f32 + k as f32, 10.0 * t as f32 + k as f32 + 0.5]
                })
            })
            .collect();
        let router_order = array(&rows, &[3, 2, 1, 2]); // [S, k, 1, in], as the experts read it

        let collector = Collector::install([("e".to_string(), &w)]).unwrap();
        observe_experts_per_slot(&w, &router_order, &ids_rt);
        let in_router_order = only(collector.finish());

        // The same pairs sorted by expert, as a prefill groups them: [P, 1, in] with ids [P].
        let flat_ids: Vec<i32> = vec![0, 2, 2, 1, 0, 1];
        let mut order: Vec<usize> = (0..6).collect();
        order.sort_by_key(|&i| flat_ids[i]);
        let sorted_rows: Vec<f32> = order
            .iter()
            .flat_map(|&i| [rows[2 * i], rows[2 * i + 1]])
            .collect();
        let sorted_ids: Vec<i32> = order.iter().map(|&i| flat_ids[i]).collect();
        let collector = Collector::install([("e".to_string(), &w)]).unwrap();
        observe_experts_per_slot(
            &w,
            &array(&sorted_rows, &[6, 1, 2]),
            &ids(&sorted_ids, &[6]),
        );
        let grouped = only(collector.finish());

        assert_eq!(in_router_order.counts, vec![2.0, 2.0, 2.0, 0.0]);
        assert_eq!(in_router_order, grouped);
        // Expert 0 got pairs (t0,k0) = [0, 0.5] and (t2,k0) = [20, 20.5].
        assert_eq!(in_router_order.sum_sq[..2], [400.0, 0.25 + 420.25]);
    }

    #[test]
    fn a_plain_weight_rejects_expert_observations_and_the_reverse() {
        let (e, x, ids) = routed_case();
        let dense = array(&[0.0; 4], &[2, 2]);
        let collector =
            Collector::install([("e".to_string(), &e), ("d".to_string(), &dense)]).unwrap();
        observe_matmul(&e, &x);
        observe_experts_per_token(&dense, &x, &ids);
        assert_eq!(collector.rejected(), 2);
    }
}
