//! Learning rate schedulers.

use pmetal_core::LrSchedulerType;
use std::f64::consts::PI;

/// Learning rate scheduler.
pub struct LrScheduler {
    scheduler_type: LrSchedulerType,
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl LrScheduler {
    /// Create a new learning rate scheduler.
    pub fn new(
        scheduler_type: LrSchedulerType,
        base_lr: f64,
        warmup_steps: usize,
        total_steps: usize,
    ) -> Self {
        Self {
            scheduler_type,
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Get the learning rate for the current step.
    pub fn get_lr(&self) -> f64 {
        let step = self.current_step;

        // Warmup phase
        if step < self.warmup_steps {
            return self.base_lr * (step as f64 / self.warmup_steps as f64);
        }

        let progress =
            (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;

        match self.scheduler_type {
            LrSchedulerType::Constant => self.base_lr,
            LrSchedulerType::Linear => self.base_lr * (1.0 - progress),
            LrSchedulerType::Cosine => self.base_lr * 0.5 * (1.0 + (PI * progress).cos()),
            LrSchedulerType::CosineWithRestarts => {
                // Single cycle for now
                self.base_lr * 0.5 * (1.0 + (PI * progress).cos())
            }
            LrSchedulerType::Polynomial => {
                let power = 2.0;
                self.base_lr * (1.0 - progress).powf(power)
            }
        }
    }

    /// Step the scheduler.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Get the current step.
    pub fn current_step(&self) -> usize {
        self.current_step
    }
}
