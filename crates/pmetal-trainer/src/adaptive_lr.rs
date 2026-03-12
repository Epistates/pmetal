//! Adaptive Learning Rate Controller.
//!
//! Provides intelligent, reactive LR management that responds to training dynamics:
//! - **Spike detection**: EMA-based z-score anomaly detection (inspired by ZClip)
//! - **Plateau detection**: Patience-based LR reduction when loss stalls
//! - **Divergence detection**: Detects sustained loss increase and aggressively reduces LR
//! - **Manual override**: Control file protocol for live LR adjustment from TUI
//! - **WSD scheduling**: Warmup-Stable-Decay as the modern default
//!
//! The controller wraps the base `LearningRateScheduler` and applies adaptive
//! adjustments on top. It acts as a filter: the scheduled LR is the ceiling,
//! and adaptive logic can only reduce from there (never increase beyond schedule).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// An LR adjustment event emitted by the controller.
#[derive(Debug, Clone)]
pub enum LrEvent {
    /// Normal scheduled LR (no intervention).
    Scheduled,
    /// Loss spike detected — LR temporarily reduced.
    SpikeDetected {
        scheduled_lr: f64,
        adjusted_lr: f64,
        loss: f64,
        ema_loss: f64,
        z_score: f64,
    },
    /// Loss plateau detected — LR permanently reduced.
    PlateauReduced {
        old_lr: f64,
        new_lr: f64,
        plateau_steps: usize,
    },
    /// Sustained loss increase detected — LR aggressively reduced.
    DivergenceReduced {
        old_lr: f64,
        new_lr: f64,
        trend_slope: f64,
    },
    /// User manually set LR via control file.
    ManualOverride { lr: f64 },
    /// LR restored after temporary spike reduction.
    SpikeRecovered,
}

impl std::fmt::Display for LrEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LrEvent::Scheduled => write!(f, "scheduled"),
            LrEvent::SpikeDetected {
                adjusted_lr,
                z_score,
                ..
            } => write!(f, "spike(z={z_score:.1}, lr={adjusted_lr:.2e})"),
            LrEvent::PlateauReduced { new_lr, .. } => write!(f, "plateau(lr={new_lr:.2e})"),
            LrEvent::DivergenceReduced { new_lr, .. } => write!(f, "diverge(lr={new_lr:.2e})"),
            LrEvent::ManualOverride { lr } => write!(f, "manual(lr={lr:.2e})"),
            LrEvent::SpikeRecovered => write!(f, "recovered"),
        }
    }
}

/// Configuration for the adaptive LR controller.
#[derive(Debug, Clone)]
pub struct AdaptiveLrConfig {
    /// Enable adaptive LR adjustments. When false, only the base scheduler runs.
    pub enabled: bool,

    /// EMA smoothing factor for loss tracking (0.95-0.99).
    /// Higher values = slower adaptation, more stable.
    pub ema_alpha: f64,

    /// Z-score threshold for spike detection (2.0-4.0).
    /// Lower = more sensitive to spikes.
    pub spike_threshold: f64,

    /// Number of steps to reduce LR after a spike before attempting recovery.
    pub spike_cooldown_steps: usize,

    /// Factor to reduce LR by during spike cooldown (0.1-0.5).
    pub spike_reduction_factor: f64,

    /// Number of steps without improvement before reducing LR (plateau detection).
    pub plateau_patience: usize,

    /// Factor to reduce LR by on plateau (0.3-0.7).
    pub plateau_factor: f64,

    /// Minimum relative improvement to reset plateau counter.
    pub plateau_min_delta: f64,

    /// Maximum number of plateau reductions before giving up.
    pub plateau_max_reductions: usize,

    /// Window size for divergence trend detection.
    pub divergence_window: usize,

    /// Factor to reduce LR by on divergence (0.2-0.5).
    pub divergence_factor: f64,

    /// Minimum slope (loss increase per step) to trigger divergence.
    pub divergence_slope_threshold: f64,

    /// Absolute minimum LR floor (never go below this).
    pub min_lr: f64,

    /// How often to poll the control file (in steps). 0 = disabled.
    pub control_poll_interval: usize,
}

impl Default for AdaptiveLrConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ema_alpha: 0.97,
            spike_threshold: 3.0,
            spike_cooldown_steps: 10,
            spike_reduction_factor: 0.3,
            plateau_patience: 50,
            plateau_factor: 0.5,
            plateau_min_delta: 1e-4,
            plateau_max_reductions: 5,
            divergence_window: 20,
            divergence_factor: 0.3,
            divergence_slope_threshold: 0.01,
            min_lr: 1e-7,
            control_poll_interval: 10,
        }
    }
}

impl AdaptiveLrConfig {
    /// Config tuned for knowledge distillation (more conservative).
    pub fn for_distillation() -> Self {
        Self {
            spike_threshold: 2.5, // More sensitive — distillation losses are smoother
            spike_cooldown_steps: 15,
            plateau_patience: 30, // React faster to plateaus
            plateau_factor: 0.5,
            divergence_window: 15,
            divergence_slope_threshold: 0.005, // More sensitive to divergence
            ..Self::default()
        }
    }
}

/// Control command from TUI → training subprocess.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum LrControlCommand {
    /// Set LR to an absolute value.
    SetLr { value: f64 },
    /// Reduce LR by a factor.
    ReduceLr { factor: f64 },
    /// Reset to the scheduled LR (clear all adaptive adjustments).
    ResetLr,
}

/// The adaptive LR controller.
///
/// Sits between the base scheduler and the optimizer, applying reactive
/// adjustments based on training dynamics.
pub struct AdaptiveLrController {
    config: AdaptiveLrConfig,

    // --- EMA loss tracking (spike detection) ---
    loss_ema: f64,
    loss_ema_var: f64, // EMA of squared deviation for variance estimation
    ema_initialized: bool,
    ema_step_count: usize, // Steps since EMA init, for bias correction
    warmup_samples: usize, // Steps before EMA is stable enough for detection

    // --- Spike state ---
    in_spike_cooldown: bool,
    spike_cooldown_remaining: usize,

    // --- Plateau detection ---
    best_loss: f64,
    plateau_counter: usize,
    plateau_reductions: usize,

    // --- Divergence detection ---
    loss_window: VecDeque<f64>,

    // --- LR multiplier (cumulative adaptive adjustments) ---
    /// Multiplicative factor applied to scheduled LR.
    /// Starts at 1.0, reduced by plateau/divergence events.
    lr_multiplier: f64,

    // --- Manual override ---
    manual_lr: Option<f64>,
    control_file: Option<PathBuf>,
    last_control_poll: usize,

    // --- Telemetry ---
    last_event: LrEvent,
    total_spikes: usize,
    total_plateau_reductions: usize,
    total_divergence_reductions: usize,
}

impl AdaptiveLrController {
    /// Create a new adaptive LR controller.
    pub fn new(config: AdaptiveLrConfig) -> Self {
        Self {
            warmup_samples: 20, // Need 20 loss samples before spike detection activates
            loss_ema: 0.0,
            loss_ema_var: 0.0,
            ema_initialized: false,
            ema_step_count: 0,
            in_spike_cooldown: false,
            spike_cooldown_remaining: 0,
            best_loss: f64::MAX,
            plateau_counter: 0,
            plateau_reductions: 0,
            loss_window: VecDeque::with_capacity(config.divergence_window + 1),
            lr_multiplier: 1.0,
            manual_lr: None,
            control_file: None,
            last_control_poll: 0,
            last_event: LrEvent::Scheduled,
            total_spikes: 0,
            total_plateau_reductions: 0,
            total_divergence_reductions: 0,
            config,
        }
    }

    /// Set the control file path for TUI → training communication.
    pub fn with_control_file(mut self, path: PathBuf) -> Self {
        self.control_file = Some(path);
        self
    }

    /// Process a training step and return the adjusted learning rate.
    ///
    /// Call this after each training step with the current loss and the
    /// scheduled LR from the base scheduler.
    ///
    /// Returns `(adjusted_lr, event)` where `adjusted_lr` is what should be
    /// applied to the optimizer for the *next* step.
    pub fn step(&mut self, step: usize, loss: f64, scheduled_lr: f64) -> (f64, LrEvent) {
        if !self.config.enabled {
            return (scheduled_lr, LrEvent::Scheduled);
        }

        // Guard against NaN/Inf loss — skip adaptive logic to avoid poisoning EMA
        if !loss.is_finite() {
            tracing::warn!("Non-finite loss ({loss}) at step {step}, skipping adaptive LR");
            let base_lr = scheduled_lr * self.lr_multiplier;
            return (base_lr.max(self.config.min_lr), LrEvent::Scheduled);
        }

        // 1. Check for manual override via control file
        if self.config.control_poll_interval > 0
            && step.saturating_sub(self.last_control_poll) >= self.config.control_poll_interval
        {
            self.last_control_poll = step;
            if let Some(cmd) = self.poll_control_file() {
                return self.apply_control_command(cmd, scheduled_lr);
            }
        }

        // If manual LR is set, use it directly (no adaptive logic)
        if let Some(manual) = self.manual_lr {
            return (manual, LrEvent::ManualOverride { lr: manual });
        }

        // 2. Update EMA loss tracking
        self.update_ema(loss);

        // 3. Update loss window for divergence detection
        self.loss_window.push_back(loss);
        if self.loss_window.len() > self.config.divergence_window {
            self.loss_window.pop_front();
        }

        // 4. Apply adaptive logic (in priority order)
        let base_lr = scheduled_lr * self.lr_multiplier;

        // 4a. Spike detection (temporary reduction)
        if self.in_spike_cooldown {
            self.spike_cooldown_remaining = self.spike_cooldown_remaining.saturating_sub(1);
            if self.spike_cooldown_remaining == 0 {
                self.in_spike_cooldown = false;
                self.last_event = LrEvent::SpikeRecovered;
                let lr = base_lr.max(self.config.min_lr);
                return (lr, LrEvent::SpikeRecovered);
            }
            let reduced = base_lr * self.config.spike_reduction_factor;
            let lr = reduced.max(self.config.min_lr);
            return (lr, self.last_event.clone());
        }

        if self.ema_initialized && self.warmup_samples == 0 {
            let z_score = self.compute_z_score(loss);

            // Spike detected
            if z_score > self.config.spike_threshold {
                self.in_spike_cooldown = true;
                self.spike_cooldown_remaining = self.config.spike_cooldown_steps;
                self.total_spikes += 1;

                let reduced = base_lr * self.config.spike_reduction_factor;
                let lr = reduced.max(self.config.min_lr);

                let event = LrEvent::SpikeDetected {
                    scheduled_lr,
                    adjusted_lr: lr,
                    loss,
                    ema_loss: self.loss_ema,
                    z_score,
                };
                self.last_event = event.clone();

                tracing::warn!(
                    "Loss spike detected (z={z_score:.2}, loss={loss:.4}, ema={:.4}). \
                     Reducing LR {base_lr:.2e} → {lr:.2e} for {} steps.",
                    self.loss_ema,
                    self.config.spike_cooldown_steps
                );

                return (lr, event);
            }

            // 4b. Divergence detection (permanent reduction)
            if self.loss_window.len() >= self.config.divergence_window {
                let slope = self.compute_trend_slope();
                if slope > self.config.divergence_slope_threshold {
                    let old_mult = self.lr_multiplier;
                    self.lr_multiplier *= self.config.divergence_factor;
                    self.total_divergence_reductions += 1;

                    let new_lr = (scheduled_lr * self.lr_multiplier).max(self.config.min_lr);
                    let old_lr = scheduled_lr * old_mult;

                    let event = LrEvent::DivergenceReduced {
                        old_lr,
                        new_lr,
                        trend_slope: slope,
                    };
                    self.last_event = event.clone();

                    tracing::warn!(
                        "Loss divergence detected (slope={slope:.4}). \
                         Reducing LR {old_lr:.2e} → {new_lr:.2e} (multiplier: {:.3}).",
                        self.lr_multiplier,
                    );

                    // Reset window after intervention
                    self.loss_window.clear();
                    // Reset plateau counter (we just intervened)
                    self.plateau_counter = 0;
                    self.best_loss = loss;

                    return (new_lr, event);
                }
            }

            // 4c. Plateau detection (permanent reduction)
            if self.plateau_reductions < self.config.plateau_max_reductions {
                if loss < self.best_loss - self.config.plateau_min_delta {
                    self.best_loss = loss;
                    self.plateau_counter = 0;
                } else {
                    self.plateau_counter += 1;

                    if self.plateau_counter >= self.config.plateau_patience {
                        let old_mult = self.lr_multiplier;
                        self.lr_multiplier *= self.config.plateau_factor;
                        self.plateau_reductions += 1;
                        self.total_plateau_reductions += 1;
                        self.plateau_counter = 0;

                        let new_lr = (scheduled_lr * self.lr_multiplier).max(self.config.min_lr);
                        let old_lr = scheduled_lr * old_mult;

                        let event = LrEvent::PlateauReduced {
                            old_lr,
                            new_lr,
                            plateau_steps: self.config.plateau_patience,
                        };
                        self.last_event = event.clone();

                        tracing::info!(
                            "Loss plateau detected ({} steps without improvement). \
                             Reducing LR {old_lr:.2e} → {new_lr:.2e} \
                             (reduction {}/{}).",
                            self.config.plateau_patience,
                            self.plateau_reductions,
                            self.config.plateau_max_reductions,
                        );

                        return (new_lr, event);
                    }
                }
            }
        }

        // 5. No intervention — return scheduled LR with multiplier
        let lr = base_lr.max(self.config.min_lr);
        self.last_event = LrEvent::Scheduled;
        (lr, LrEvent::Scheduled)
    }

    /// Get the last LR event.
    pub fn last_event(&self) -> &LrEvent {
        &self.last_event
    }

    /// Get cumulative LR multiplier from adaptive reductions.
    pub fn lr_multiplier(&self) -> f64 {
        self.lr_multiplier
    }

    /// Get current EMA loss (useful for logging).
    pub fn ema_loss(&self) -> f64 {
        self.loss_ema
    }

    /// Get summary statistics.
    pub fn stats_summary(&self) -> String {
        format!(
            "spikes={}, plateaus={}, divergences={}, multiplier={:.3}",
            self.total_spikes,
            self.total_plateau_reductions,
            self.total_divergence_reductions,
            self.lr_multiplier,
        )
    }

    // --- Internal methods ---

    fn update_ema(&mut self, loss: f64) {
        if !self.ema_initialized {
            self.loss_ema = loss;
            self.loss_ema_var = 0.0;
            self.ema_initialized = true;
            self.ema_step_count = 1;
            return;
        }

        if self.warmup_samples > 0 {
            self.warmup_samples -= 1;
        }

        self.ema_step_count += 1;

        let alpha = self.config.ema_alpha;
        let delta = loss - self.loss_ema;
        self.loss_ema = alpha * self.loss_ema + (1.0 - alpha) * loss;
        self.loss_ema_var = alpha * self.loss_ema_var + (1.0 - alpha) * delta * delta;
    }

    fn compute_z_score(&self, loss: f64) -> f64 {
        // Apply bias correction to the EMA variance estimate.
        // Raw EMA variance underestimates early (starts from 0, decays toward true var).
        // Correction: var_corrected = var_raw / (1 - alpha^n)
        let alpha = self.config.ema_alpha;
        let bias_correction = 1.0 - alpha.powi(self.ema_step_count as i32);
        let corrected_var = if bias_correction > 1e-12 {
            self.loss_ema_var / bias_correction
        } else {
            self.loss_ema_var
        };

        let std_dev = corrected_var.sqrt();
        if std_dev < 1e-8 {
            // Variance too small to estimate — use absolute deviation as fallback.
            // If loss deviates >50% from EMA, treat as extreme (z=10).
            let abs_dev = (loss - self.loss_ema).abs();
            let threshold = self.loss_ema.abs().max(0.1) * 0.5;
            return if abs_dev > threshold { 10.0 } else { 0.0 };
        }
        (loss - self.loss_ema) / std_dev
    }

    /// Compute linear regression slope over the loss window.
    fn compute_trend_slope(&self) -> f64 {
        let n = self.loss_window.len() as f64;
        if n < 3.0 {
            return 0.0;
        }

        // Normalize losses by the first value to get relative slope
        let first = self.loss_window.front().copied().unwrap_or(1.0);
        let normalizer = first.abs().max(1e-8);

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &loss) in self.loss_window.iter().enumerate() {
            let x = i as f64;
            let y = (loss - first) / normalizer; // Relative change
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }

    fn poll_control_file(&mut self) -> Option<LrControlCommand> {
        let path = self.control_file.as_ref()?;

        // Atomically claim the control file by renaming it before reading.
        // This prevents a race where the TUI writes a new file between our
        // read and delete operations.
        let claimed = path.with_extension("claimed");
        if std::fs::rename(path, &claimed).is_err() {
            // File doesn't exist or can't be claimed — nothing to do
            return None;
        }

        let content = std::fs::read_to_string(&claimed).ok()?;
        let _ = std::fs::remove_file(&claimed);

        let cmd: LrControlCommand = serde_json::from_str(content.trim()).ok()?;

        tracing::info!("Received LR control command: {:?}", cmd);
        Some(cmd)
    }

    fn apply_control_command(
        &mut self,
        cmd: LrControlCommand,
        scheduled_lr: f64,
    ) -> (f64, LrEvent) {
        match cmd {
            LrControlCommand::SetLr { value } => {
                let lr = value.max(self.config.min_lr);
                self.manual_lr = Some(lr);
                // Reset adaptive state since user is taking control
                self.plateau_counter = 0;
                self.in_spike_cooldown = false;
                let event = LrEvent::ManualOverride { lr };
                self.last_event = event.clone();
                tracing::info!("Manual LR override: {lr:.2e}");
                (lr, event)
            }
            LrControlCommand::ReduceLr { factor } => {
                let factor = factor.clamp(0.01, 1.0);
                self.lr_multiplier *= factor;
                self.manual_lr = None; // Clear manual, use adaptive with new multiplier
                let lr = (scheduled_lr * self.lr_multiplier).max(self.config.min_lr);
                let event = LrEvent::ManualOverride { lr };
                self.last_event = event.clone();
                tracing::info!("Manual LR reduction: factor={factor}, new lr={lr:.2e}");
                (lr, event)
            }
            LrControlCommand::ResetLr => {
                self.lr_multiplier = 1.0;
                self.manual_lr = None;
                self.plateau_counter = 0;
                self.plateau_reductions = 0;
                self.in_spike_cooldown = false;
                self.best_loss = f64::MAX;
                let event = LrEvent::Scheduled;
                self.last_event = event.clone();
                tracing::info!("LR reset to schedule: {scheduled_lr:.2e}");
                (scheduled_lr, event)
            }
        }
    }
}

/// Write an LR control command to the control file.
///
/// Called by the TUI to send LR adjustments to a running training job.
pub fn write_lr_control(output_dir: &Path, command: &LrControlCommand) -> std::io::Result<()> {
    let control_path = output_dir.join(".lr_control.json");
    let json = serde_json::to_string(command)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&control_path, json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_intervention_during_warmup() {
        let config = AdaptiveLrConfig::default();
        let mut ctrl = AdaptiveLrController::new(config);

        // During EMA warmup (first 20 steps), no adaptive logic should fire
        for i in 0..20 {
            let (lr, event) = ctrl.step(i, 10.0 + (i as f64) * 0.5, 1e-4);
            assert!(matches!(event, LrEvent::Scheduled));
            assert!((lr - 1e-4).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spike_detection() {
        let config = AdaptiveLrConfig {
            spike_threshold: 2.0,
            spike_cooldown_steps: 5,
            spike_reduction_factor: 0.1,
            ..Default::default()
        };
        let mut ctrl = AdaptiveLrController::new(config);

        // Feed stable losses to build EMA
        for i in 0..25 {
            ctrl.step(i, 5.0, 1e-4);
        }

        // Inject a massive spike
        let (lr, event) = ctrl.step(25, 50.0, 1e-4);
        assert!(matches!(event, LrEvent::SpikeDetected { .. }));
        assert!(lr < 1e-4); // LR should be reduced
    }

    #[test]
    fn test_plateau_detection() {
        let config = AdaptiveLrConfig {
            plateau_patience: 5,
            plateau_factor: 0.5,
            plateau_min_delta: 0.001,
            spike_threshold: 100.0, // High threshold so spike detection doesn't interfere
            ..Default::default()
        };
        let mut ctrl = AdaptiveLrController::new(config);
        // Manually set up stable EMA so spike detection doesn't trigger
        ctrl.warmup_samples = 0;
        ctrl.ema_initialized = true;
        ctrl.loss_ema = 4.5;
        ctrl.loss_ema_var = 0.01; // Small stable variance
        ctrl.best_loss = 4.5;

        // Feed flat losses (no improvement beyond min_delta)
        // Patience=5 means 5 flat steps before reduction: steps 0-4 increment counter, step 5 triggers
        let mut found_plateau = false;
        for i in 0..10 {
            let (lr, event) = ctrl.step(i, 4.5, 1e-4);
            if matches!(event, LrEvent::PlateauReduced { .. }) {
                assert!(lr < 1e-4, "LR should be reduced on plateau");
                found_plateau = true;
                break;
            }
        }
        assert!(
            found_plateau,
            "Plateau should have been detected within 10 steps"
        );
    }

    #[test]
    fn test_divergence_detection() {
        let config = AdaptiveLrConfig {
            divergence_window: 10,
            divergence_slope_threshold: 0.005,
            divergence_factor: 0.3,
            ..Default::default()
        };
        let mut ctrl = AdaptiveLrController::new(config);
        ctrl.warmup_samples = 0;
        ctrl.ema_initialized = true;

        // Feed steadily increasing losses
        for i in 0..15 {
            let loss = 5.0 + (i as f64) * 0.5; // Strong upward trend
            let (lr, event) = ctrl.step(i, loss, 1e-4);
            if matches!(event, LrEvent::DivergenceReduced { .. }) {
                assert!(lr < 1e-4);
                return; // Test passes
            }
        }
        panic!("Divergence should have been detected");
    }

    #[test]
    fn test_manual_override() {
        let config = AdaptiveLrConfig::default();
        let mut ctrl = AdaptiveLrController::new(config);
        ctrl.manual_lr = Some(5e-6);

        let (lr, event) = ctrl.step(0, 10.0, 1e-4);
        assert!(matches!(event, LrEvent::ManualOverride { .. }));
        assert!((lr - 5e-6).abs() < 1e-12);
    }

    #[test]
    fn test_lr_never_below_min() {
        let config = AdaptiveLrConfig {
            min_lr: 1e-7,
            plateau_patience: 1,
            plateau_factor: 0.01,
            plateau_max_reductions: 100,
            ..Default::default()
        };
        let mut ctrl = AdaptiveLrController::new(config);
        ctrl.warmup_samples = 0;
        ctrl.ema_initialized = true;
        ctrl.best_loss = 1.0;

        // Trigger many plateau reductions
        for i in 0..50 {
            let (lr, _) = ctrl.step(i, 1.0, 1e-4);
            assert!(lr >= 1e-7, "LR fell below min: {lr}");
        }
    }

    #[test]
    fn test_spike_recovery() {
        let config = AdaptiveLrConfig {
            spike_threshold: 2.0,
            spike_cooldown_steps: 3,
            spike_reduction_factor: 0.1,
            ..Default::default()
        };
        let mut ctrl = AdaptiveLrController::new(config);

        // Build stable EMA
        for i in 0..25 {
            ctrl.step(i, 5.0, 1e-4);
        }

        // Spike
        let (lr, _) = ctrl.step(25, 100.0, 1e-4);
        assert!(lr < 1e-4);

        // Cooldown steps
        ctrl.step(26, 5.0, 1e-4);
        ctrl.step(27, 5.0, 1e-4);

        // Recovery
        let (lr, event) = ctrl.step(28, 5.0, 1e-4);
        assert!(matches!(event, LrEvent::SpikeRecovered));
        // LR should be back to normal (possibly with multiplier but no spike reduction)
        assert!((lr - 1e-4).abs() < 1e-8);
    }

    #[test]
    fn test_nan_loss_does_not_poison_ema() {
        let config = AdaptiveLrConfig::default();
        let mut ctrl = AdaptiveLrController::new(config);

        // Build stable EMA
        for i in 0..25 {
            ctrl.step(i, 5.0, 1e-4);
        }
        let ema_before = ctrl.loss_ema;

        // Feed NaN — should be skipped
        let (lr, event) = ctrl.step(25, f64::NAN, 1e-4);
        assert!(matches!(event, LrEvent::Scheduled));
        assert!(lr.is_finite());

        // EMA should be unchanged
        assert!((ctrl.loss_ema - ema_before).abs() < 1e-12);

        // Feed Inf — should be skipped
        let (lr, event) = ctrl.step(26, f64::INFINITY, 1e-4);
        assert!(matches!(event, LrEvent::Scheduled));
        assert!(lr.is_finite());
        assert!((ctrl.loss_ema - ema_before).abs() < 1e-12);
    }

    #[test]
    fn test_zero_variance_spike_fallback() {
        let config = AdaptiveLrConfig {
            spike_threshold: 3.0,
            ..Default::default()
        };
        let mut ctrl = AdaptiveLrController::new(config);
        ctrl.warmup_samples = 0;
        ctrl.ema_initialized = true;
        ctrl.ema_step_count = 100; // Enough steps for bias correction to be ~1.0
        ctrl.loss_ema = 5.0;
        ctrl.loss_ema_var = 0.0; // Zero variance

        // Large deviation from EMA should trigger spike via fallback
        let (lr, event) = ctrl.step(0, 50.0, 1e-4);
        assert!(
            matches!(event, LrEvent::SpikeDetected { .. }),
            "Expected spike detection with zero-variance fallback, got {event:?}"
        );
        assert!(lr < 1e-4);
    }
}
