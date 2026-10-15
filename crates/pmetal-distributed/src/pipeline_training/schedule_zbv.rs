//! Zero-bubble pipeline schedule (the memory-neutral "ZB-H1" family).
//!
//! Splits the backward pass into two phases:
//! - **B** (BackwardInput): Computes activation gradients. Depends on
//!   gradients from the next stage — must execute in sequence.
//! - **W** (BackwardWeight): Computes weight gradients. Depends only on
//!   local activations — can fill pipeline bubble slots.
//!
//! With the backward split in two, the next stage's wait ends after B alone,
//! and W, which nobody waits for, is free to run whenever the stage would
//! otherwise idle. With equal forward, B and W times the idle time drops to a
//! third of 1F1B's: the makespan is `3 M + (P - 1)` unit steps for `P` stages
//! and `M` micro-batches, against `3 M + 3 (P - 1)` for 1F1B.
//!
//! Memory is bounded the way 1F1B bounds it: a stage never has more than
//! `P - stage` micro-batches in flight (forward done, backward not). The price
//! of deferring W is that up to `P` weight-gradient halves are owed at once,
//! each needing the saved inputs of its layer (not the full activations).
//!
//! The schedule is built by simulating the whole pipeline with a greedy rule,
//! one step at a time per stage: run a B if its gradient has arrived, else a
//! forward if the in-flight cap allows another, else a W; a W owed for more
//! than `P - 1` micro-batches goes ahead of the next forward. That is
//! deterministic, so every rank derives the same schedule for the same
//! `(stages, micro_batches)`, and [`super::simulate`] checks it against the
//! other stages' and measures its bubble.
//!
//! # Reference
//!
//! - Zero Bubble Pipeline Parallelism (Qi et al., ICLR 2024)

use std::collections::VecDeque;

/// An action in the zero-bubble pipeline schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZBAction {
    /// Run forward pass for micro-batch `id`.
    Forward(usize),
    /// Run backward-input pass for micro-batch `id`.
    /// Computes dL/dx (activation gradient). Depends on next stage's gradients.
    BackwardInput(usize),
    /// Run backward-weight pass for micro-batch `id`.
    /// Computes dL/dW (weight gradient). Local only — fills bubble slots.
    BackwardWeight(usize),
    /// Send the activations of micro-batch `id` to the next stage.
    SendActivation(usize),
    /// Receive the activations of micro-batch `id` from the previous stage.
    RecvActivation(usize),
    /// Send the input gradient of micro-batch `id` to the previous stage.
    SendGradient(usize),
    /// Receive the input gradient of micro-batch `id` from the next stage.
    RecvGradient(usize),
    /// Apply accumulated weight updates.
    WeightUpdate,
}

/// Compute actions of one stage, before the transfers are woven in.
#[derive(Debug, Clone, Copy)]
enum Step {
    F(usize),
    B(usize),
    W(usize),
}

/// Generate the zero-bubble schedule for one pipeline stage.
///
/// # Arguments
///
/// * `num_stages` — Total number of pipeline stages (P)
/// * `num_micro_batches` — Number of micro-batches (M, at least P)
/// * `stage` — This stage's index (0-indexed)
///
/// # Returns
///
/// Ordered list of actions this stage should execute.
///
/// # Panics
///
/// Panics if `num_micro_batches < num_stages` or `stage >= num_stages`.
pub fn schedule_zero_bubble(
    num_stages: usize,
    num_micro_batches: usize,
    stage: usize,
) -> Vec<ZBAction> {
    assert!(
        num_micro_batches >= num_stages,
        "zero-bubble requires num_micro_batches ({num_micro_batches}) >= num_stages ({num_stages})",
    );
    assert!(
        stage < num_stages,
        "stage {stage} >= num_stages {num_stages}"
    );

    let steps = plan(num_stages, num_micro_batches).swap_remove(stage);
    let (is_first, is_last) = (stage == 0, stage == num_stages - 1);
    let mut schedule = Vec::with_capacity(steps.len() * 2 + 1);
    for step in steps {
        match step {
            Step::F(m) => {
                if !is_first {
                    schedule.push(ZBAction::RecvActivation(m));
                }
                schedule.push(ZBAction::Forward(m));
                if !is_last {
                    schedule.push(ZBAction::SendActivation(m));
                }
            }
            Step::B(m) => {
                if !is_last {
                    schedule.push(ZBAction::RecvGradient(m));
                }
                schedule.push(ZBAction::BackwardInput(m));
                if !is_first {
                    schedule.push(ZBAction::SendGradient(m));
                }
            }
            Step::W(m) => schedule.push(ZBAction::BackwardWeight(m)),
        }
    }
    schedule.push(ZBAction::WeightUpdate);
    schedule
}

/// Simulate the whole pipeline with unit-time F, B and W and return each
/// stage's compute steps in the order the greedy rule chose them.
fn plan(stages: usize, micro_batches: usize) -> Vec<Vec<Step>> {
    // When each stage finished F / B for each micro-batch (None: not yet).
    let mut forward_done: Vec<Vec<Option<u64>>> = vec![vec![None; micro_batches]; stages];
    let mut backward_done: Vec<Vec<Option<u64>>> = vec![vec![None; micro_batches]; stages];
    let mut clock = vec![0u64; stages];
    let mut next_forward = vec![0usize; stages];
    let mut next_backward = vec![0usize; stages];
    let mut weight_queue: Vec<VecDeque<usize>> = vec![VecDeque::new(); stages];
    // Micro-batches in flight at a stage: forward done, backward not. 1F1B
    // never has more than `stages - stage` in flight; the same cap keeps the
    // forward activations to the same peak.
    let mut live = vec![0usize; stages];
    let mut steps: Vec<Vec<Step>> = vec![Vec::new(); stages];

    let total = stages * micro_batches * 3;
    let mut done = 0;
    while done < total {
        // The stage that can act soonest goes next, the lower stage on a tie.
        let mut best: Option<(u64, usize, Step)> = None;
        for s in 0..stages {
            let cap = stages - s;
            let mut options: Vec<(u64, u8, Step)> = Vec::new();

            let b = next_backward[s];
            if b < micro_batches
                && let Some(own) = forward_done[s][b]
            {
                let grad = if s + 1 == stages {
                    Some(0)
                } else {
                    backward_done[s + 1][b]
                };
                if let Some(grad) = grad {
                    options.push((clock[s].max(own).max(grad), 0, Step::B(b)));
                }
            }
            let f = next_forward[s];
            // A weight half may be put off by up to `stages - 1` micro-batches:
            // the idle time a stage meets as the pipeline drains is about that
            // many weight halves long, and these fill it. Past that, it goes
            // before the next forward, which bounds how many a stage holds.
            let overdue = weight_queue[s].len() > stages - 1;
            if f < micro_batches && live[s] < cap && !overdue {
                let input = if s == 0 {
                    Some(0)
                } else {
                    forward_done[s - 1][f]
                };
                if let Some(input) = input {
                    options.push((clock[s].max(input), 1, Step::F(f)));
                }
            }
            if let Some(&w) = weight_queue[s].front() {
                // An overdue weight half goes before anything but a backward.
                options.push((clock[s], if overdue { 0 } else { 2 }, Step::W(w)));
            }

            // The earliest-startable option, B before F before W on a tie.
            if let Some(&(start, _, step)) = options.iter().min_by_key(|(t, rank, _)| (*t, *rank))
                && best.is_none_or(|(t, ..)| start < t)
            {
                best = Some((start, s, step));
            }
        }
        let (start, s, step) = best.expect("a legal schedule always has a runnable step");
        clock[s] = start + 1;
        match step {
            Step::F(m) => {
                forward_done[s][m] = Some(clock[s]);
                next_forward[s] += 1;
                live[s] += 1;
            }
            Step::B(m) => {
                backward_done[s][m] = Some(clock[s]);
                next_backward[s] += 1;
                weight_queue[s].push_back(m);
                live[s] -= 1;
            }
            Step::W(_) => {
                weight_queue[s].pop_front();
            }
        }
        steps[s].push(step);
        done += 1;
    }
    steps
}

/// The pipeline bubble of the zero-bubble schedule for `num_stages` and
/// `num_micro_batches`, with equal forward, B and W times: the fraction of
/// stage-time not spent computing, as [`super::simulate`] measures it.
pub fn zbv_bubble_fraction(num_stages: usize, num_micro_batches: usize) -> f64 {
    use super::simulate::{Costs, Op, simulate};
    if num_micro_batches < num_stages || num_stages == 0 {
        return 1.0;
    }
    let schedules: Vec<Vec<Op>> = (0..num_stages)
        .map(|s| {
            schedule_zero_bubble(num_stages, num_micro_batches, s)
                .into_iter()
                .map(Op::from)
                .collect()
        })
        .collect();
    let unit = Costs {
        forward: 1.0,
        backward: 1.0,
        backward_weight: 1.0,
        update: 0.0,
    };
    simulate(&schedules, unit)
        .expect("zero-bubble schedules do not deadlock")
        .bubble_fraction()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zbv_schedule_counts() {
        let num_stages = 4;
        let num_micro_batches = 8;

        for stage in 0..num_stages {
            let schedule = schedule_zero_bubble(num_stages, num_micro_batches, stage);

            let count = |f: fn(&ZBAction) -> bool| schedule.iter().filter(|a| f(a)).count();
            assert_eq!(
                count(|a| matches!(a, ZBAction::Forward(_))),
                num_micro_batches,
                "stage {stage} forward count"
            );
            assert_eq!(
                count(|a| matches!(a, ZBAction::BackwardInput(_))),
                num_micro_batches,
                "stage {stage} backward-input count"
            );
            assert_eq!(
                count(|a| matches!(a, ZBAction::BackwardWeight(_))),
                num_micro_batches,
                "stage {stage} backward-weight count"
            );
            assert_eq!(
                count(|a| matches!(a, ZBAction::WeightUpdate)),
                1,
                "stage {stage} weight update count"
            );
        }
    }

    #[test]
    fn zbv_forward_before_backward() {
        let schedule = schedule_zero_bubble(3, 6, 1);
        for mb in 0..6 {
            let pos = |want: ZBAction| schedule.iter().position(|a| *a == want).unwrap();
            let (f, bi, bw) = (
                pos(ZBAction::Forward(mb)),
                pos(ZBAction::BackwardInput(mb)),
                pos(ZBAction::BackwardWeight(mb)),
            );
            assert!(f < bi, "mb {mb}: F at {f} must precede BI at {bi}");
            assert!(bi < bw, "mb {mb}: BI at {bi} must precede BW at {bw}");
        }
    }

    /// Transfers appear only where the stage has a neighbour to talk to, and
    /// pair up with the neighbour's.
    #[test]
    fn first_and_last_stages_skip_the_transfers_they_cannot_make() {
        let first = schedule_zero_bubble(3, 6, 0);
        let last = schedule_zero_bubble(3, 6, 2);
        assert!(
            !first
                .iter()
                .any(|a| matches!(a, ZBAction::RecvActivation(_) | ZBAction::SendGradient(_)))
        );
        assert!(
            !last
                .iter()
                .any(|a| matches!(a, ZBAction::SendActivation(_) | ZBAction::RecvGradient(_)))
        );
    }

    #[test]
    fn schedules_are_deterministic() {
        assert_eq!(
            schedule_zero_bubble(5, 12, 2),
            schedule_zero_bubble(5, 12, 2)
        );
    }
}
