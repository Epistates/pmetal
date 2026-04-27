//! Measurement-only QJL ablation hooks (feature-gated, `tq-ablation`).
//!
//! When [`set_qjl_disabled(true)`] is in effect, the encoders zero the
//! per-row `residual_norms` they produce. The score kernel's QJL term is
//! `residual_scale * sign * qproj`; with `residual_scale == 0` that term
//! drops out *without* a kernel change, so this is a pure measurement of
//! whether the QJL residual stage contributes to attention quality on a
//! given workload.
//!
//! Used by `benches/turboquant_qjl_ablation.rs` to run an A/B perplexity
//! sweep that gates Phase C's default-flip decision (drop QJL → "Variant
//! F"). The reference's own ablation found ~0 contribution; we want to
//! reproduce or refute that on our targeted models before flipping the
//! production default.
//!
//! This module compiles to nothing in non-`tq-ablation` builds — the
//! [`crate::turboquant::ablation`] path simply doesn't exist, and callers
//! reach for the no-op [`super::should_zero_qjl`] shim defined in
//! `mod.rs` instead.

use std::sync::atomic::{AtomicBool, Ordering};

static QJL_DISABLE: AtomicBool = AtomicBool::new(false);

/// Set the global QJL-disable flag. After this returns, every subsequent
/// encode pass writes zeros into the per-row `residual_norms` field, which
/// makes the score kernel's QJL term collapse to 0.
///
/// The flag is process-global; concurrent inference paths share it. The
/// ablation harness sets it to `true`, runs a perplexity sweep, then sets
/// it back to `false`.
pub fn set_qjl_disabled(value: bool) {
    QJL_DISABLE.store(value, Ordering::Relaxed);
}

/// Returns the current value of the QJL-disable flag.
pub fn qjl_disabled() -> bool {
    QJL_DISABLE.load(Ordering::Relaxed)
}
