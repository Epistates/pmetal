//! Execute pipeline schedules on a model clock.
//!
//! A schedule is a list of actions per stage; whether the lists fit together
//! (no stage waits for a message nobody sends, every micro-batch moves
//! forward through the stages and back through them in order) is a property of
//! all stages at once, and is easy to get wrong. [`simulate`] runs every
//! stage's list against the others with message passing and a cost per
//! compute action, and reports either the timeline or the deadlock, so a
//! schedule can be checked before a cluster waits on it, and its real bubble
//! measured instead of estimated from a formula.
//!
//! Messages are asynchronous: a `Send` never blocks, a `Recv` waits until the
//! matching `Send` has been executed.

use super::{MicroBatchAction, ZBAction};
use std::collections::HashMap;

/// One step of a stage's work, in the vocabulary all schedules share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Forward pass of a micro-batch.
    Forward(usize),
    /// Backward pass of a micro-batch (or only its input half, in a split
    /// backward).
    Backward(usize),
    /// The weight-gradient half of a split backward.
    BackwardWeight(usize),
    /// Send the activations of a micro-batch to the next stage.
    SendActivation(usize),
    /// Receive the activations of a micro-batch from the previous stage.
    RecvActivation(usize),
    /// Send the input gradient of a micro-batch to the previous stage.
    SendGradient(usize),
    /// Receive the input gradient of a micro-batch from the next stage.
    RecvGradient(usize),
    /// Optimizer step after all micro-batches.
    WeightUpdate,
}

impl From<MicroBatchAction> for Op {
    fn from(action: MicroBatchAction) -> Self {
        match action {
            MicroBatchAction::Forward(m) => Self::Forward(m),
            MicroBatchAction::Backward(m) => Self::Backward(m),
            MicroBatchAction::SendActivation(m) => Self::SendActivation(m),
            MicroBatchAction::RecvActivation(m) => Self::RecvActivation(m),
            MicroBatchAction::SendGradient(m) => Self::SendGradient(m),
            MicroBatchAction::RecvGradient(m) => Self::RecvGradient(m),
            MicroBatchAction::WeightUpdate => Self::WeightUpdate,
        }
    }
}

impl From<ZBAction> for Op {
    fn from(action: ZBAction) -> Self {
        match action {
            ZBAction::Forward(m) => Self::Forward(m),
            ZBAction::BackwardInput(m) => Self::Backward(m),
            ZBAction::BackwardWeight(m) => Self::BackwardWeight(m),
            ZBAction::SendActivation(m) => Self::SendActivation(m),
            ZBAction::RecvActivation(m) => Self::RecvActivation(m),
            ZBAction::SendGradient(m) => Self::SendGradient(m),
            ZBAction::RecvGradient(m) => Self::RecvGradient(m),
            ZBAction::WeightUpdate => Self::WeightUpdate,
        }
    }
}

/// What each compute action costs, in any consistent unit.
#[derive(Debug, Clone, Copy)]
pub struct Costs {
    /// Forward pass.
    pub forward: f64,
    /// Backward pass (the input half, when the weight half is separate).
    pub backward: f64,
    /// Weight-gradient half of a split backward.
    pub backward_weight: f64,
    /// Optimizer step.
    pub update: f64,
}

/// A compute action that ran, with when.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Event {
    /// The action.
    pub op: Op,
    /// When it started.
    pub start: f64,
    /// When it finished.
    pub end: f64,
}

/// The outcome of running a schedule.
#[derive(Debug, Clone)]
pub struct Timeline {
    /// Compute events per stage, in order.
    pub events: Vec<Vec<Event>>,
    /// When the last stage finished.
    pub makespan: f64,
}

impl Timeline {
    /// Fraction of stage-time spent not computing: the pipeline bubble.
    pub fn bubble_fraction(&self) -> f64 {
        let stages = self.events.len() as f64;
        let busy: f64 = self.events.iter().flatten().map(|e| e.end - e.start).sum();
        1.0 - busy / (stages * self.makespan)
    }

    /// The most micro-batches stage `stage` holds activations for at once:
    /// forward done, backward (or its input half) not yet.
    pub fn peak_in_flight(&self, stage: usize) -> usize {
        let mut live = 0usize;
        let mut peak = 0usize;
        for e in &self.events[stage] {
            match e.op {
                Op::Forward(_) => {
                    live += 1;
                    peak = peak.max(live);
                }
                Op::Backward(_) => live = live.saturating_sub(1),
                _ => {}
            }
        }
        peak
    }

    /// The most micro-batches whose weight-gradient half is still owed on
    /// `stage` at once: backward-input done, weight half not yet.
    pub fn peak_pending_weight(&self, stage: usize) -> usize {
        let mut pending = 0usize;
        let mut peak = 0usize;
        for e in &self.events[stage] {
            match e.op {
                Op::Backward(_) => {
                    pending += 1;
                    peak = peak.max(pending);
                }
                Op::BackwardWeight(_) => pending = pending.saturating_sub(1),
                _ => {}
            }
        }
        peak
    }

    /// When `op` finished on `stage`.
    pub fn end_of(&self, stage: usize, op: Op) -> Option<f64> {
        self.events[stage]
            .iter()
            .find(|e| e.op == op)
            .map(|e| e.end)
    }

    /// When `op` started on `stage`.
    pub fn start_of(&self, stage: usize, op: Op) -> Option<f64> {
        self.events[stage]
            .iter()
            .find(|e| e.op == op)
            .map(|e| e.start)
    }
}

/// A schedule that cannot run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScheduleError {
    /// Every unfinished stage waits for a message no stage will send.
    #[error("pipeline deadlock: {waiting:?}")]
    Deadlock {
        /// The action each stuck stage is waiting at.
        waiting: Vec<(usize, Op)>,
    },
}

/// Run `schedules` (one list of actions per stage, stage 0 first).
pub fn simulate(schedules: &[Vec<Op>], costs: Costs) -> Result<Timeline, ScheduleError> {
    let stages = schedules.len();
    let mut clock = vec![0.0f64; stages];
    let mut at = vec![0usize; stages];
    let mut events: Vec<Vec<Event>> = vec![Vec::new(); stages];
    // (kind, from-stage, micro-batch) -> time the message is available.
    let mut mailbox: HashMap<(bool, usize, usize), f64> = HashMap::new();

    loop {
        let mut progressed = false;
        for s in 0..stages {
            while let Some(&op) = schedules[s].get(at[s]) {
                let compute = |c: f64, clock: &mut Vec<f64>, events: &mut Vec<Vec<Event>>| {
                    let start = clock[s];
                    clock[s] += c;
                    events[s].push(Event {
                        op,
                        start,
                        end: clock[s],
                    });
                };
                match op {
                    Op::Forward(_) => compute(costs.forward, &mut clock, &mut events),
                    Op::Backward(_) => compute(costs.backward, &mut clock, &mut events),
                    Op::BackwardWeight(_) => {
                        compute(costs.backward_weight, &mut clock, &mut events)
                    }
                    Op::WeightUpdate => compute(costs.update, &mut clock, &mut events),
                    Op::SendActivation(mb) => {
                        mailbox.insert((true, s, mb), clock[s]);
                    }
                    Op::SendGradient(mb) => {
                        mailbox.insert((false, s, mb), clock[s]);
                    }
                    Op::RecvActivation(mb) => match s
                        .checked_sub(1)
                        .and_then(|from| mailbox.get(&(true, from, mb)))
                    {
                        Some(&ready) => clock[s] = clock[s].max(ready),
                        None => break,
                    },
                    Op::RecvGradient(mb) => match mailbox.get(&(false, s + 1, mb)) {
                        Some(&ready) => clock[s] = clock[s].max(ready),
                        None => break,
                    },
                }
                at[s] += 1;
                progressed = true;
            }
        }
        if (0..stages).all(|s| at[s] == schedules[s].len()) {
            let makespan = clock.iter().copied().fold(0.0, f64::max);
            return Ok(Timeline { events, makespan });
        }
        if !progressed {
            return Err(ScheduleError::Deadlock {
                waiting: (0..stages)
                    .filter(|&s| at[s] < schedules[s].len())
                    .map(|s| (s, schedules[s][at[s]]))
                    .collect(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{schedule_1f1b, schedule_zero_bubble};
    use super::*;

    fn one_f1b(stages: usize, micro_batches: usize) -> Vec<Vec<Op>> {
        (0..stages)
            .map(|s| {
                schedule_1f1b(stages, micro_batches, s)
                    .into_iter()
                    .map(Op::from)
                    .collect()
            })
            .collect()
    }

    fn zero_bubble(stages: usize, micro_batches: usize) -> Vec<Vec<Op>> {
        (0..stages)
            .map(|s| {
                schedule_zero_bubble(stages, micro_batches, s)
                    .into_iter()
                    .map(Op::from)
                    .collect()
            })
            .collect()
    }

    const F1B: Costs = Costs {
        forward: 1.0,
        backward: 2.0,
        backward_weight: 0.0,
        update: 0.0,
    };

    /// Whatever the shape, a micro-batch goes forward through the stages in
    /// order, backward through them in reverse, and a stage's backward comes
    /// after its own forward.
    fn assert_respects_dependencies(t: &Timeline, stages: usize, micro_batches: usize) {
        for m in 0..micro_batches {
            for s in 0..stages {
                let f = t.start_of(s, Op::Forward(m)).expect("forward ran");
                let b = t.start_of(s, Op::Backward(m)).expect("backward ran");
                assert!(f < b, "stage {s} mb {m}: backward before its forward");
                if s > 0 {
                    assert!(
                        f >= t.end_of(s - 1, Op::Forward(m)).unwrap(),
                        "stage {s} mb {m}: forward before the previous stage's"
                    );
                }
                if s + 1 < stages {
                    assert!(
                        b >= t.end_of(s + 1, Op::Backward(m)).unwrap(),
                        "stage {s} mb {m}: backward before the next stage's"
                    );
                }
            }
        }
    }

    #[test]
    fn one_f1b_runs_and_respects_dependencies() {
        for stages in 1..=6 {
            for micro_batches in stages..=4 * stages {
                let t = simulate(&one_f1b(stages, micro_batches), F1B)
                    .unwrap_or_else(|e| panic!("P={stages} M={micro_batches}: {e}"));
                assert_respects_dependencies(&t, stages, micro_batches);
            }
        }
    }

    /// With forward 1 and backward 2, the bubble of 1F1B is exactly
    /// (P - 1) / (M + P - 1), and a stage never holds more than `P - s`
    /// micro-batches of activations: the point of 1F1B over running every
    /// forward first.
    #[test]
    fn one_f1b_bubble_and_memory_match_theory() {
        for (stages, micro_batches) in [(2, 2), (4, 4), (4, 8), (4, 16), (6, 12)] {
            let t = simulate(&one_f1b(stages, micro_batches), F1B).unwrap();
            let want = (stages as f64 - 1.0) / (micro_batches as f64 + stages as f64 - 1.0);
            assert!(
                (t.bubble_fraction() - want).abs() < 1e-9,
                "P={stages} M={micro_batches}: bubble {} vs {want}",
                t.bubble_fraction()
            );
            for s in 0..stages {
                assert_eq!(
                    t.peak_in_flight(s),
                    (stages - s).min(micro_batches),
                    "P={stages} M={micro_batches} stage {s}"
                );
            }
        }
    }

    const SPLIT: Costs = Costs {
        forward: 1.0,
        backward: 1.0,
        backward_weight: 1.0,
        update: 0.0,
    };

    /// Splitting the backward into an input half (which the next stage waits
    /// for) and a weight half (which nobody waits for) lets the weight half
    /// fill the bubble: the makespan is `3 M + (P - 1)` with equal times,
    /// where 1F1B needs `3 M + 3 (P - 1)`.
    #[test]
    fn zero_bubble_reaches_its_bound_and_respects_dependencies() {
        for stages in 1..=8 {
            for micro_batches in stages..=6 * stages {
                let t = simulate(&zero_bubble(stages, micro_batches), SPLIT)
                    .unwrap_or_else(|e| panic!("P={stages} M={micro_batches}: {e}"));
                assert_respects_dependencies(&t, stages, micro_batches);
                for s in 0..stages {
                    for m in 0..micro_batches {
                        let input = t.end_of(s, Op::Backward(m)).unwrap();
                        let weight = t.start_of(s, Op::BackwardWeight(m)).expect("W ran");
                        assert!(weight >= input, "stage {s} mb {m}: W before B");
                    }
                }
                assert_eq!(
                    t.makespan,
                    (3 * micro_batches + stages - 1) as f64,
                    "P={stages} M={micro_batches}"
                );
                let one_f1b = simulate(&one_f1b(stages, micro_batches), F1B).unwrap();
                assert!(t.makespan <= one_f1b.makespan);
                if stages >= 2 {
                    assert!(t.makespan < one_f1b.makespan);
                    assert!(t.bubble_fraction() < one_f1b.bubble_fraction());
                }
            }
        }
    }

    /// Deferring W must not run memory away: the in-flight micro-batches stay
    /// within 1F1B's bound, and the weight halves owed within one per stage.
    #[test]
    fn zero_bubble_memory_stays_bounded() {
        for (stages, micro_batches) in [(2, 4), (4, 8), (4, 32), (8, 16), (8, 64)] {
            let t = simulate(&zero_bubble(stages, micro_batches), SPLIT).unwrap();
            for s in 0..stages {
                assert!(
                    t.peak_in_flight(s) <= stages - s,
                    "P={stages} M={micro_batches} stage {s}: {} in flight",
                    t.peak_in_flight(s)
                );
                assert!(
                    t.peak_pending_weight(s) <= stages,
                    "P={stages} M={micro_batches} stage {s}: {} weight halves owed",
                    t.peak_pending_weight(s)
                );
            }
        }
    }

    /// The bubble estimate is the simulation's, not a constant factor.
    #[test]
    fn the_bubble_estimate_is_the_measured_bubble() {
        use super::super::schedule_zbv::zbv_bubble_fraction;
        for (stages, micro_batches) in [(4, 8), (4, 16), (8, 32)] {
            let measured = simulate(&zero_bubble(stages, micro_batches), SPLIT)
                .unwrap()
                .bubble_fraction();
            assert!((zbv_bubble_fraction(stages, micro_batches) - measured).abs() < 1e-12);
        }
        assert_eq!(zbv_bubble_fraction(4, 2), 1.0);
    }

    #[test]
    fn a_schedule_that_waits_for_nothing_is_reported_as_a_deadlock() {
        let schedules = vec![
            vec![Op::RecvGradient(0), Op::Forward(0)],
            vec![Op::RecvActivation(0), Op::Forward(0)],
        ];
        let err = simulate(&schedules, F1B).unwrap_err();
        assert!(matches!(err, ScheduleError::Deadlock { .. }), "{err}");
    }
}
