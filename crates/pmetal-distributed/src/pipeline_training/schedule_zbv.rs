//! Zero Bubble V (ZBV) pipeline schedule.
//!
//! Splits the backward pass into two phases:
//! - **B** (BackwardInput): Computes activation gradients. Depends on
//!   gradients from the next stage — must execute in sequence.
//! - **W** (BackwardWeight): Computes weight gradients. Depends only on
//!   local activations — can fill pipeline bubble slots.
//!
//! By scheduling W into bubble time, ZBV achieves near-zero pipeline bubble
//! (15-30% throughput improvement over 1F1B).
//!
//! ```text
//! Stage 0: F0 F1 F2 F3 B0 F4 B1 W0 F5 B2 W1 ... B_last W_last  WU
//! Stage 1:    F0 F1 F2 B0 W0 F3 B1 W1 F4 B2 W2 ...  B_last W_last WU
//! ```
//!
//! W computations fill slots that would otherwise be idle.
//!
//! # Reference
//!
//! - Zero Bubble Pipeline Parallelism (Qi et al., ICLR 2024)

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
    /// Send activations or gradients for micro-batch `id`.
    Send(usize),
    /// Receive activations or gradients for micro-batch `id`.
    Recv(usize),
    /// Apply accumulated weight updates.
    WeightUpdate,
}

/// Generate a Zero Bubble V schedule for a given pipeline stage.
///
/// The key insight: BackwardWeight (W) has no cross-stage dependency,
/// so it can be deferred and scheduled into bubble slots.
///
/// # Arguments
///
/// * `num_stages` — Total number of pipeline stages (P)
/// * `num_micro_batches` — Number of micro-batches (M, must be >= 2*P)
/// * `stage` — This stage's index (0-indexed)
///
/// # Returns
///
/// Ordered list of actions this stage should execute.
pub fn schedule_zero_bubble(
    num_stages: usize,
    num_micro_batches: usize,
    stage: usize,
) -> Vec<ZBAction> {
    assert!(
        num_micro_batches >= num_stages,
        "ZBV requires num_micro_batches ({}) >= num_stages ({})",
        num_micro_batches,
        num_stages
    );
    assert!(
        stage < num_stages,
        "stage {stage} >= num_stages {num_stages}"
    );

    let mut schedule = Vec::new();
    let is_first = stage == 0;
    let is_last = stage == num_stages - 1;

    // Track which micro-batches have pending W computations.
    let mut pending_w: Vec<usize> = Vec::new();
    let mut next_fwd: usize = 0;
    let mut next_bwd: usize = 0;

    // Warmup phase: extra forwards (same as 1F1B).
    let num_warmup = (num_stages - stage - 1).min(num_micro_batches);

    for _ in 0..num_warmup {
        if !is_first && next_fwd < num_micro_batches {
            schedule.push(ZBAction::Recv(next_fwd));
        }
        if next_fwd < num_micro_batches {
            schedule.push(ZBAction::Forward(next_fwd));
            if !is_last {
                schedule.push(ZBAction::Send(next_fwd));
            }
            next_fwd += 1;
        }
    }

    // Steady state: F, B, then fill bubble with W from earlier micro-batches.
    // It lasts while forwards remain; the cooldown below takes the
    // backwards left after the last one. (Running it until every backward
    // was out spun forever: with no forward left, the gate on B never
    // opened again for the last `num_stages - stage - 1` micro-batches.)
    while next_fwd < num_micro_batches {
        // Forward for new micro-batch.
        if next_fwd < num_micro_batches {
            if !is_first {
                schedule.push(ZBAction::Recv(next_fwd));
            }
            schedule.push(ZBAction::Forward(next_fwd));
            if !is_last {
                schedule.push(ZBAction::Send(next_fwd));
            }
            next_fwd += 1;
        }

        // Backward-input for earlier micro-batch.
        if next_bwd < num_micro_batches && (next_bwd + num_stages - stage) <= next_fwd {
            if !is_last {
                schedule.push(ZBAction::Recv(next_bwd));
            }
            schedule.push(ZBAction::BackwardInput(next_bwd));
            if !is_first {
                schedule.push(ZBAction::Send(next_bwd));
            }
            pending_w.push(next_bwd);
            next_bwd += 1;
        }

        // Fill bubble with W from pending queue.
        if let Some(w_mb) = pending_w.first().copied() {
            schedule.push(ZBAction::BackwardWeight(w_mb));
            pending_w.remove(0);
        }
    }

    // Cooldown: remaining B passes.
    while next_bwd < num_micro_batches {
        if !is_last {
            schedule.push(ZBAction::Recv(next_bwd));
        }
        schedule.push(ZBAction::BackwardInput(next_bwd));
        if !is_first {
            schedule.push(ZBAction::Send(next_bwd));
        }
        pending_w.push(next_bwd);
        next_bwd += 1;

        // Fill with W.
        if let Some(w_mb) = pending_w.first().copied() {
            schedule.push(ZBAction::BackwardWeight(w_mb));
            pending_w.remove(0);
        }
    }

    // Drain remaining W computations.
    for w_mb in pending_w {
        schedule.push(ZBAction::BackwardWeight(w_mb));
    }

    // Weight update.
    schedule.push(ZBAction::WeightUpdate);

    schedule
}

/// Estimate bubble fraction for ZBV schedule.
///
/// ZBV reduces the bubble by filling it with W computations.
/// The theoretical bubble is close to zero when W time ≈ bubble time.
pub fn zbv_bubble_fraction(num_stages: usize, num_micro_batches: usize) -> f64 {
    // In the ideal case, W fills the bubble completely.
    // Residual bubble ≈ max(0, (P-1) * (F+B) - M * W) / (M * (F+B+W))
    // Simplified estimate: ~1/3 of the 1F1B bubble (W ≈ 1/3 of backward).
    let one_f1b = super::schedule_1f1b::bubble_fraction(num_stages, num_micro_batches);
    (one_f1b * 0.33).max(0.0)
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

            let fwd = schedule
                .iter()
                .filter(|a| matches!(a, ZBAction::Forward(_)))
                .count();
            let bi = schedule
                .iter()
                .filter(|a| matches!(a, ZBAction::BackwardInput(_)))
                .count();
            let bw = schedule
                .iter()
                .filter(|a| matches!(a, ZBAction::BackwardWeight(_)))
                .count();
            let wu = schedule
                .iter()
                .filter(|a| matches!(a, ZBAction::WeightUpdate))
                .count();

            assert_eq!(fwd, num_micro_batches, "stage {stage} forward count");
            assert_eq!(bi, num_micro_batches, "stage {stage} backward-input count");
            assert_eq!(bw, num_micro_batches, "stage {stage} backward-weight count");
            assert_eq!(wu, 1, "stage {stage} weight update count");
        }
    }

    #[test]
    fn zbv_forward_before_backward() {
        let schedule = schedule_zero_bubble(3, 6, 1);
        for mb in 0..6 {
            let fwd_pos = schedule
                .iter()
                .position(|a| matches!(a, ZBAction::Forward(id) if *id == mb));
            let bi_pos = schedule
                .iter()
                .position(|a| matches!(a, ZBAction::BackwardInput(id) if *id == mb));
            let bw_pos = schedule
                .iter()
                .position(|a| matches!(a, ZBAction::BackwardWeight(id) if *id == mb));

            if let (Some(f), Some(bi)) = (fwd_pos, bi_pos) {
                assert!(f < bi, "mb {mb}: F at {f} must precede BI at {bi}");
            }
            if let (Some(bi), Some(bw)) = (bi_pos, bw_pos) {
                assert!(bi < bw, "mb {mb}: BI at {bi} must precede BW at {bw}");
            }
        }
    }

    #[test]
    fn zbv_bubble_less_than_1f1b() {
        let zbv = zbv_bubble_fraction(4, 8);
        let one_f1b = super::super::schedule_1f1b::bubble_fraction(4, 8);
        assert!(
            zbv < one_f1b,
            "ZBV bubble {zbv} should be less than 1F1B {one_f1b}"
        );
    }
}
