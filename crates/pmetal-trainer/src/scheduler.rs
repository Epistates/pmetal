//! Learning rate scheduler - re-exports the canonical implementation from pmetal-core.
//!
//! The canonical scheduler lives in `pmetal_core::LearningRateScheduler`.
//! All trainers should import from there directly or use this re-export.

pub use pmetal_core::{LearningRateScheduler, SchedulerBuilder};
