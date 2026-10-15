//! Pipeline training schedules for distributed training.
//!
//! Coordinates forward and backward passes across pipeline stages
//! to maximize GPU utilization and minimize the pipeline bubble.
//!
//! # Schedules
//!
//! - **1F1B**: One Forward, One Backward. Classic schedule from PipeDream.
//!   Warmup phase with (num_stages - rank - 1) forwards, then alternating
//!   forward/backward in steady state.
//!
//! - **Zero Bubble (ZB-H1)**: Splits backward into B (activation gradient)
//!   and W (weight gradient). W has no cross-stage dependency so it can fill
//!   bubble slots: a third of 1F1B's idle time at the same in-flight memory.
//!
//! [`simulate`] runs any schedule against its neighbours' on a model clock,
//! reporting a deadlock or the real timeline, so a schedule is checked and
//! its bubble measured rather than assumed.
//!
//! # Reference
//!
//! - PipeDream: Generalized Pipeline Parallelism (Narayanan et al., 2019)
//! - Zero Bubble Pipeline Parallelism (Qi et al., ICLR 2024)

pub mod schedule_1f1b;
pub mod schedule_zbv;
pub mod simulate;

pub use schedule_1f1b::{MicroBatchAction, schedule_1f1b};
pub use schedule_zbv::{ZBAction, schedule_zero_bubble};
