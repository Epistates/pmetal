//! Learning rate scheduler implementations.
//!
//! This module provides the canonical learning rate scheduler implementations
//! used across all trainers. Use these instead of implementing schedulers
//! in individual trainer modules.

use crate::{LrSchedulerType, WsdDecayStyle};
use std::f64::consts::PI;

/// Learning rate scheduler that computes LR based on training progress.
///
/// This is the canonical scheduler implementation for PMetal. All trainers
/// should use this instead of implementing their own scheduler logic.
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    /// Base learning rate.
    base_lr: f64,
    /// Minimum learning rate (for warmup and decay).
    min_lr: f64,
    /// Total training steps.
    total_steps: usize,
    /// Warmup steps.
    warmup_steps: usize,
    /// Scheduler type.
    scheduler_type: LrSchedulerType,
    /// Number of restarts for cosine with restarts.
    num_restarts: usize,
    /// Fraction of post-warmup steps at stable (peak) LR for WSD (0.0-1.0).
    stable_ratio: f64,
    /// Shape of the WSD decay phase.
    wsd_decay_style: WsdDecayStyle,
    /// Exponent for polynomial decay (default 1.0 = linear).
    polynomial_power: f64,
    /// Current step.
    current_step: usize,
}

impl LearningRateScheduler {
    /// Create a new learning rate scheduler.
    pub fn new(
        base_lr: f64,
        total_steps: usize,
        warmup_steps: usize,
        scheduler_type: LrSchedulerType,
    ) -> Self {
        Self {
            base_lr,
            min_lr: 0.0,
            total_steps,
            warmup_steps,
            scheduler_type,
            num_restarts: 1,
            stable_ratio: 0.7,
            wsd_decay_style: WsdDecayStyle::default(),
            polynomial_power: 1.0,
            current_step: 0,
        }
    }

    /// The schedule a training config describes over `total_steps` optimizer
    /// steps: its scheduler type, warmup (`warmup_ratio` of the total when
    /// set, else `warmup_steps`), `min_lr`, restarts, WSD stable share and
    /// polynomial power.
    ///
    /// The knobs are taken as given, not clamped: a config that has not passed
    /// [`ConfigValidator::validate`](crate::ConfigValidator::validate) (a
    /// stable ratio of 3, a power of -1) gets the schedule those numbers
    /// describe, not a corrected one.
    pub fn for_training(config: &crate::TrainingConfig, total_steps: usize) -> Self {
        let mut scheduler = Self::new(
            config.learning_rate,
            total_steps,
            warmup_steps_for(config, total_steps),
            config.lr_scheduler,
        )
        .with_min_lr(config.min_lr.unwrap_or(0.0));
        if let Some(restarts) = config.cosine_num_restarts {
            scheduler = scheduler.with_num_restarts(restarts);
        }
        if let Some(ratio) = config.wsd_stable_ratio {
            scheduler = scheduler.with_stable_ratio(ratio);
        }
        if let Some(style) = config.wsd_decay_style {
            scheduler = scheduler.with_wsd_decay_style(style);
        }
        if let Some(power) = config.polynomial_power {
            scheduler = scheduler.with_polynomial_power(power);
        }
        scheduler
    }

    /// Set the shape of the WSD decay phase (default linear).
    pub fn with_wsd_decay_style(mut self, style: WsdDecayStyle) -> Self {
        self.wsd_decay_style = style;
        self
    }

    /// Set minimum learning rate.
    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Set the number of cosine cycles for cosine with restarts (1 is a
    /// single cosine; `n` cycles restart `n - 1` times).
    pub fn with_num_restarts(mut self, num_restarts: usize) -> Self {
        self.num_restarts = num_restarts;
        self
    }

    /// Set stable phase ratio for WSD scheduler (default 0.7).
    pub fn with_stable_ratio(mut self, ratio: f64) -> Self {
        self.stable_ratio = ratio;
        self
    }

    /// Set polynomial decay exponent (default 1.0 = linear, 2.0 = quadratic).
    pub fn with_polynomial_power(mut self, power: f64) -> Self {
        self.polynomial_power = power;
        self
    }

    /// Get learning rate for a specific step.
    #[must_use]
    pub fn get_lr(&self, step: usize) -> f64 {
        // Warmup phase
        if step < self.warmup_steps {
            let warmup_factor = step as f64 / self.warmup_steps.max(1) as f64;
            return self.min_lr + (self.base_lr - self.min_lr) * warmup_factor;
        }

        // Post-warmup phase
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        let current_decay_step = step.saturating_sub(self.warmup_steps);

        if decay_steps == 0 {
            return self.base_lr;
        }

        let progress = (current_decay_step as f64 / decay_steps as f64).min(1.0);

        match self.scheduler_type {
            LrSchedulerType::Constant => self.base_lr,

            LrSchedulerType::Linear => {
                self.min_lr + (self.base_lr - self.min_lr) * (1.0 - progress)
            }

            LrSchedulerType::Cosine => {
                self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + (PI * progress).cos())
            }

            LrSchedulerType::CosineWithRestarts => {
                // `num_restarts` hard-restarted cosine cycles over the decay
                // span, then the floor: the same curve as transformers'
                // `get_cosine_with_hard_restarts_schedule_with_warmup`.
                if progress >= 1.0 {
                    return self.min_lr;
                }
                let cycle_progress = (self.num_restarts.max(1) as f64 * progress) % 1.0;
                self.min_lr
                    + (self.base_lr - self.min_lr) * 0.5 * (1.0 + (PI * cycle_progress).cos())
            }

            LrSchedulerType::Polynomial => {
                let power = self.polynomial_power;
                self.min_lr + (self.base_lr - self.min_lr) * (1.0 - progress).powf(power)
            }

            LrSchedulerType::Wsd => {
                // Warmup-Stable-Decay: constant plateau then linear decay-to-zero.
                // stable_ratio controls how much of post-warmup is at peak LR.
                let stable_steps = (decay_steps as f64 * self.stable_ratio) as usize;
                if current_decay_step < stable_steps {
                    // Stable phase: hold at peak LR
                    self.base_lr
                } else {
                    // Decay phase: linear decay from base_lr to min_lr
                    let decay_phase_steps = decay_steps.saturating_sub(stable_steps);
                    let decay_progress = if decay_phase_steps > 0 {
                        ((current_decay_step - stable_steps) as f64 / decay_phase_steps as f64)
                            .min(1.0)
                    } else {
                        1.0
                    };
                    self.min_lr
                        + (self.base_lr - self.min_lr) * self.wsd_decay_style.factor(decay_progress)
                }
            }
        }
    }

    /// Get learning rate for current step.
    #[must_use]
    pub fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    /// Advance the scheduler by one step.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Set the current step.
    pub fn set_step(&mut self, step: usize) {
        self.current_step = step;
    }

    /// Get the current step.
    #[must_use]
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Check if training is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.total_steps
    }
}

/// Warmup steps for a run of `total_steps` optimizer steps: `warmup_ratio`
/// of the total, rounded up, when the config sets one, else `warmup_steps`.
pub fn warmup_steps_for(config: &crate::TrainingConfig, total_steps: usize) -> usize {
    match config.warmup_ratio {
        Some(ratio) => (total_steps as f64 * ratio).ceil() as usize,
        None => config.warmup_steps,
    }
}

/// Optimizer steps in a run, counted the way the reference trainers count
/// them: `max_steps` when set, else every epoch's micro-batches divided
/// into optimizer steps of `micro_batches_per_step`. `flush_each_epoch`
/// says whether a partial accumulation at the end of an epoch still takes
/// a step (as in the preference loop) or carries into the next epoch (as
/// in the SFT loop).
pub fn total_training_steps(
    config: &crate::TrainingConfig,
    micro_batches_per_epoch: usize,
    micro_batches_per_step: usize,
    flush_each_epoch: bool,
) -> usize {
    if let Some(max) = config.max_steps {
        return max.max(1);
    }
    let per_step = micro_batches_per_step.max(1);
    let epochs = config.num_epochs.max(1);
    let total = if flush_each_epoch {
        micro_batches_per_epoch.div_ceil(per_step) * epochs
    } else {
        micro_batches_per_epoch * epochs / per_step
    };
    total.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let scheduler = LearningRateScheduler::new(1e-4, 1000, 0, LrSchedulerType::Constant);

        assert!((scheduler.get_lr(0) - 1e-4).abs() < 1e-10);
        assert!((scheduler.get_lr(500) - 1e-4).abs() < 1e-10);
        assert!((scheduler.get_lr(999) - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_linear_decay() {
        let scheduler = LearningRateScheduler::new(1e-4, 1000, 0, LrSchedulerType::Linear);

        assert!((scheduler.get_lr(0) - 1e-4).abs() < 1e-10);
        assert!(scheduler.get_lr(500) < 1e-4);
        assert!(scheduler.get_lr(999) < scheduler.get_lr(500));
    }

    #[test]
    fn test_warmup() {
        let scheduler = LearningRateScheduler::new(1e-4, 1000, 100, LrSchedulerType::Cosine);

        // During warmup, LR should increase
        assert!(scheduler.get_lr(0) < scheduler.get_lr(50));
        assert!(scheduler.get_lr(50) < scheduler.get_lr(100));

        // At warmup end, should be at base LR
        assert!((scheduler.get_lr(100) - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_decay() {
        let scheduler = LearningRateScheduler::new(1e-4, 1000, 0, LrSchedulerType::Cosine);

        // Should start at base LR
        assert!((scheduler.get_lr(0) - 1e-4).abs() < 1e-10);

        // Should decay
        assert!(scheduler.get_lr(500) < 1e-4);

        // Should approach min at end
        assert!(scheduler.get_lr(999) < scheduler.get_lr(500));
    }

    #[test]
    fn total_steps_count_optimizer_steps_like_the_reference_trainers() {
        let config = crate::TrainingConfig {
            num_epochs: 3,
            max_steps: None,
            ..Default::default()
        };
        // 10 micro-batches an epoch, 4 per step. Flushing each epoch takes
        // ceil(10/4) = 3 steps an epoch; carrying the remainder takes 30/4.
        assert_eq!(total_training_steps(&config, 10, 4, true), 9);
        assert_eq!(total_training_steps(&config, 10, 4, false), 7);
        assert_eq!(total_training_steps(&config, 10, 1, false), 30);
        assert_eq!(total_training_steps(&config, 0, 4, false), 1);
        let capped = crate::TrainingConfig {
            max_steps: Some(5),
            ..config
        };
        assert_eq!(total_training_steps(&capped, 10, 4, true), 5);
    }

    #[test]
    fn for_training_applies_every_schedule_setting() {
        let config = crate::TrainingConfig {
            learning_rate: 1e-3,
            warmup_steps: 50,
            warmup_ratio: Some(0.1),
            min_lr: Some(1e-4),
            lr_scheduler: LrSchedulerType::Cosine,
            ..Default::default()
        };
        // Warmup is ceil(0.1 * 95) = 10 of the 95 steps, not the 50 steps.
        assert_eq!(warmup_steps_for(&config, 95), 10);
        let s = LearningRateScheduler::for_training(&config, 95);
        assert!((s.get_lr(5) - (1e-4 + (1e-3 - 1e-4) * 0.5)).abs() < 1e-12);
        assert!((s.get_lr(10) - 1e-3).abs() < 1e-12);
        // Cosine ends at the floor, not at zero.
        assert!((s.get_lr(95) - 1e-4).abs() < 1e-12);
        let wsd = crate::TrainingConfig {
            lr_scheduler: LrSchedulerType::Wsd,
            wsd_stable_ratio: Some(0.0),
            warmup_ratio: None,
            warmup_steps: 0,
            min_lr: None,
            ..config
        };
        let s = LearningRateScheduler::for_training(&wsd, 100);
        assert!(s.get_lr(50) < 0.6e-3, "no stable phase: decays from step 0");
    }

    #[test]
    fn test_wsd_scheduler() {
        // 1000 steps, 100 warmup, stable_ratio=0.7 → 630 stable steps, 270 decay steps
        let scheduler = LearningRateScheduler::new(1e-4, 1000, 100, LrSchedulerType::Wsd)
            .with_stable_ratio(0.7);

        // During warmup: increasing
        assert!(scheduler.get_lr(0) < scheduler.get_lr(50));
        assert!((scheduler.get_lr(100) - 1e-4).abs() < 1e-10);

        // During stable phase (steps 100-730): at base LR
        assert!((scheduler.get_lr(200) - 1e-4).abs() < 1e-10);
        assert!((scheduler.get_lr(500) - 1e-4).abs() < 1e-10);
        assert!((scheduler.get_lr(729) - 1e-4).abs() < 1e-10);

        // During decay phase (steps 730-1000): decreasing
        assert!(scheduler.get_lr(800) < 1e-4);
        assert!(scheduler.get_lr(900) < scheduler.get_lr(800));
        assert!(scheduler.get_lr(999) < scheduler.get_lr(900));
    }

    #[test]
    fn test_wsd_stable_ratio_zero() {
        // stable_ratio=0 → no stable phase, immediate decay after warmup
        let scheduler =
            LearningRateScheduler::new(1e-4, 100, 10, LrSchedulerType::Wsd).with_stable_ratio(0.0);

        // Warmup finishes at step 10
        assert!((scheduler.get_lr(10) - 1e-4).abs() < 1e-10);
        // Step 11 should already be in decay
        assert!(scheduler.get_lr(50) < 1e-4);
    }

    #[test]
    fn test_wsd_stable_ratio_one() {
        // stable_ratio=1.0 → all stable, no decay phase
        let scheduler =
            LearningRateScheduler::new(1e-4, 100, 10, LrSchedulerType::Wsd).with_stable_ratio(1.0);

        // All post-warmup steps at base LR
        assert!((scheduler.get_lr(50) - 1e-4).abs() < 1e-10);
        assert!((scheduler.get_lr(99) - 1e-4).abs() < 1e-10);
    }

    /// transformers' `_get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda`,
    /// term for term.
    fn reference_hard_restarts(step: usize, warmup: usize, total: usize, cycles: usize) -> f64 {
        if step < warmup {
            return step as f64 / warmup.max(1) as f64;
        }
        let progress = (step - warmup) as f64 / total.saturating_sub(warmup).max(1) as f64;
        if progress >= 1.0 {
            return 0.0;
        }
        (0.5 * (1.0 + (PI * ((cycles as f64 * progress) % 1.0)).cos())).max(0.0)
    }

    #[test]
    fn cosine_with_restarts_matches_the_reference_schedule() {
        for (total, warmup, cycles) in [
            (100, 0, 1),
            (100, 10, 2),
            (97, 7, 3),
            (50, 0, 5),
            (9, 0, 3),
            (10, 2, 4),
        ] {
            let s =
                LearningRateScheduler::new(1.0, total, warmup, LrSchedulerType::CosineWithRestarts)
                    .with_num_restarts(cycles);
            for step in 0..=total + 3 {
                let want = reference_hard_restarts(step, warmup, total, cycles);
                let got = s.get_lr(step);
                assert!(
                    (got - want).abs() < 1e-12,
                    "total={total} warmup={warmup} cycles={cycles} step={step}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn decay_schedules_match_the_reference_formulas() {
        // transformers' get_linear_schedule_with_warmup,
        // get_cosine_schedule_with_warmup (num_cycles = 0.5) and
        // get_polynomial_decay_schedule_with_warmup (lr_end = 0), as the
        // multiplier on the peak rate.
        for (total, warmup) in [(100usize, 0usize), (100, 10), (37, 5), (8, 7), (10, 1)] {
            let denom = total.saturating_sub(warmup).max(1) as f64;
            for step in 0..=total {
                let warm = step as f64 / warmup.max(1) as f64;
                let progress = step.saturating_sub(warmup) as f64 / denom;
                let (linear, cosine, quad) = if step < warmup {
                    (warm, warm, warm)
                } else {
                    (
                        ((total - step) as f64 / denom).max(0.0),
                        (0.5 * (1.0 + (PI * 2.0 * 0.5 * progress).cos())).max(0.0),
                        (1.0 - progress).powi(2),
                    )
                };
                for (kind, want) in [
                    (LrSchedulerType::Linear, linear),
                    (LrSchedulerType::Cosine, cosine),
                    (LrSchedulerType::Polynomial, quad),
                ] {
                    let mut s = LearningRateScheduler::new(1.0, total, warmup, kind);
                    if kind == LrSchedulerType::Polynomial {
                        s = s.with_polynomial_power(2.0);
                    }
                    assert!(
                        (s.get_lr(step) - want).abs() < 1e-12,
                        "{kind:?} total={total} warmup={warmup} step={step}: got {}, want {want}",
                        s.get_lr(step)
                    );
                }
            }
        }
    }

    #[test]
    fn wsd_decay_styles_match_the_reference_schedule() {
        // transformers' `_get_wsd_scheduler_lambda`, with min_lr_ratio 0.1.
        let (peak, ratio) = (2.0, 0.1);
        let (warmup, total, stable_ratio) = (10usize, 110usize, 0.5);
        let decay_span = total - warmup;
        let stable = (decay_span as f64 * stable_ratio) as usize;
        let decay = decay_span - stable;
        for (style, factor) in [
            (WsdDecayStyle::Linear, (|p: f64| 1.0 - p) as fn(f64) -> f64),
            (WsdDecayStyle::Cosine, |p| 0.5 * (1.0 + (PI * p).cos())),
            (WsdDecayStyle::OneMinusSqrt, |p| 1.0 - p.sqrt()),
        ] {
            let s = LearningRateScheduler::new(peak, total, warmup, LrSchedulerType::Wsd)
                .with_min_lr(peak * ratio)
                .with_stable_ratio(stable_ratio)
                .with_wsd_decay_style(style);
            for step in 0..=total {
                let want = if step < warmup {
                    let p = step as f64 / warmup as f64;
                    p * (1.0 - ratio) + ratio
                } else if step < warmup + stable {
                    1.0
                } else if step < warmup + stable + decay {
                    let p = (step - warmup - stable) as f64 / decay as f64;
                    factor(p) * (1.0 - ratio) + ratio
                } else {
                    ratio
                } * peak;
                assert!(
                    (s.get_lr(step) - want).abs() < 1e-12,
                    "{style:?} step={step}: got {}, want {want}",
                    s.get_lr(step)
                );
            }
        }
    }

    #[test]
    fn cosine_with_restarts_ends_at_the_floor_and_stays_there() {
        let s = LearningRateScheduler::new(1e-3, 100, 0, LrSchedulerType::CosineWithRestarts)
            .with_num_restarts(2)
            .with_min_lr(1e-5);
        // The last step of the run and everything past it sit at the floor
        // instead of jumping back to the peak for a restart nobody asked for.
        for step in [100, 101, 150, 1_000] {
            assert!(
                (s.get_lr(step) - 1e-5).abs() < 1e-12,
                "step {step}: {}",
                s.get_lr(step)
            );
        }
    }

    #[test]
    fn test_wsd_warmup_exceeds_total() {
        // warmup_steps > total_steps → always in warmup phase
        let scheduler = LearningRateScheduler::new(1e-4, 50, 100, LrSchedulerType::Wsd);

        // Should be partway through warmup, not crash
        let lr = scheduler.get_lr(25);
        assert!(lr > 0.0);
        assert!(lr < 1e-4);
    }
}
