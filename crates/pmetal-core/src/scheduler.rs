//! Learning rate scheduler implementations.
//!
//! This module provides the canonical learning rate scheduler implementations
//! used across all trainers. Use these instead of implementing schedulers
//! in individual trainer modules.

use crate::{LrSchedulerType, Result};
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
            current_step: 0,
        }
    }

    /// Set minimum learning rate.
    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Set number of restarts for cosine with restarts.
    pub fn with_num_restarts(mut self, num_restarts: usize) -> Self {
        self.num_restarts = num_restarts;
        self
    }

    /// Set stable phase ratio for WSD scheduler (default 0.7).
    pub fn with_stable_ratio(mut self, ratio: f64) -> Self {
        self.stable_ratio = ratio.clamp(0.0, 1.0);
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
                let cycle_length = decay_steps / self.num_restarts.max(1);
                let cycle_progress = if cycle_length > 0 {
                    (current_decay_step % cycle_length) as f64 / cycle_length as f64
                } else {
                    0.0
                };
                self.min_lr
                    + (self.base_lr - self.min_lr) * 0.5 * (1.0 + (PI * cycle_progress).cos())
            }

            LrSchedulerType::Polynomial => {
                let power = 2.0; // Quadratic decay
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
                    self.min_lr + (self.base_lr - self.min_lr) * (1.0 - decay_progress)
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

/// Builder for `LearningRateScheduler`.
#[derive(Debug, Clone)]
pub struct SchedulerBuilder {
    base_lr: f64,
    min_lr: f64,
    total_steps: usize,
    warmup_steps: usize,
    warmup_ratio: Option<f64>,
    scheduler_type: LrSchedulerType,
    num_restarts: usize,
    stable_ratio: f64,
}

impl Default for SchedulerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerBuilder {
    /// Create a new scheduler builder.
    pub fn new() -> Self {
        Self {
            base_lr: crate::defaults::LEARNING_RATE,
            min_lr: 0.0,
            total_steps: 1000,
            warmup_steps: crate::defaults::WARMUP_STEPS,
            warmup_ratio: None,
            scheduler_type: LrSchedulerType::Cosine,
            num_restarts: 1,
            stable_ratio: 0.7,
        }
    }

    /// Set base learning rate.
    pub fn base_lr(mut self, lr: f64) -> Self {
        self.base_lr = lr;
        self
    }

    /// Set minimum learning rate.
    pub fn min_lr(mut self, lr: f64) -> Self {
        self.min_lr = lr;
        self
    }

    /// Set total training steps.
    pub fn total_steps(mut self, steps: usize) -> Self {
        self.total_steps = steps;
        self
    }

    /// Set warmup steps.
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self.warmup_ratio = None;
        self
    }

    /// Set warmup ratio (alternative to warmup_steps).
    pub fn warmup_ratio(mut self, ratio: f64) -> Self {
        self.warmup_ratio = Some(ratio);
        self
    }

    /// Set scheduler type.
    pub fn scheduler_type(mut self, scheduler: LrSchedulerType) -> Self {
        self.scheduler_type = scheduler;
        self
    }

    /// Set number of restarts for cosine with restarts.
    pub fn num_restarts(mut self, restarts: usize) -> Self {
        self.num_restarts = restarts;
        self
    }

    /// Set WSD stable phase ratio (fraction of post-warmup at peak LR).
    pub fn stable_ratio(mut self, ratio: f64) -> Self {
        self.stable_ratio = ratio;
        self
    }

    /// Build the scheduler.
    pub fn build(self) -> Result<LearningRateScheduler> {
        let warmup_steps = if let Some(ratio) = self.warmup_ratio {
            (self.total_steps as f64 * ratio) as usize
        } else {
            self.warmup_steps
        };

        Ok(LearningRateScheduler::new(
            self.base_lr,
            self.total_steps,
            warmup_steps,
            self.scheduler_type,
        )
        .with_min_lr(self.min_lr)
        .with_num_restarts(self.num_restarts)
        .with_stable_ratio(self.stable_ratio))
    }
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
    fn test_builder() {
        let scheduler = SchedulerBuilder::new()
            .base_lr(2e-4)
            .warmup_ratio(0.1)
            .total_steps(1000)
            .scheduler_type(LrSchedulerType::Cosine)
            .build()
            .unwrap();

        // Warmup should be 10% of total steps = 100
        assert!(scheduler.get_lr(50) < scheduler.get_lr(99));
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
