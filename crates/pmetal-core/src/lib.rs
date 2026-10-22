//! Core types, traits, and configuration for PMetal LLM fine-tuning.
//!
//! This crate provides the foundational abstractions used throughout the PMetal
//! framework, including:
//!
//! - Configuration types for training, LoRA, dataset and model loading, the
//!   strict loader for the config file ([`FullTrainingConfig`]) and their
//!   validation ([`ConfigValidator`])
//! - The job specs every surface builds jobs from ([`jobs`]) and the events
//!   a job streams back ([`events`])
//! - Learning rate schedulers
//! - Training callbacks and shared metric types
//! - Error handling infrastructure
//! - Secure handling of secrets (tokens, credentials)

#![warn(missing_docs)]

// Lets `#[derive(JobSpec)]` macros in this crate's own modules resolve their
// generated `::pmetal_core::*` paths correctly. Standard self-rename pattern
// for crates that consume their own derive macro.
extern crate self as pmetal_core;

mod config;
mod config_file;
pub mod constants;
pub mod control;
mod error;
pub mod events;
pub mod fields;
pub mod jobs;
pub mod model_type;
pub mod paths;
pub mod scheduler;
mod secrets;
mod traits;
mod types;
mod validation;

pub use config::*;
pub use config_file::{FullTrainingConfig, LoadedConfig};
pub use control::{LR_CONTROL_FILE, LrControlCommand, lr_control_path, write_lr_control};
pub use error::*;
pub use events::{
    BenchTrialMetrics, BroadcastSink, CompletionSummary, JobEvent, JobEventSink, JobKind,
    JobStatus, JsonlSink, LogLevel, MetricPayload, NullSink, ParseError, Phase, Progress,
    ServeRequestMetrics, TrainingCallbackToSink, parse_event, write_event,
};
pub use fields::{
    DefaultValue, FieldDescriptor, FieldError, FieldKind, FieldShape, JobFields, default_text,
    spec_from_text,
};
pub use scheduler::{LearningRateScheduler, total_training_steps, warmup_steps_for};
pub use secrets::SecretString;
pub use traits::*;
pub use types::*;

/// Re-export of `#[derive(JobSpec)]` from the companion proc-macro crate, so
/// users only ever import from `pmetal_core`.
pub use pmetal_core_derive::JobSpec;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::config::*;
    pub use crate::config_file::{FullTrainingConfig, LoadedConfig};
    pub use crate::error::{PMetalError, Result};
    pub use crate::events::{JobEvent, JobEventSink, JobKind, JobStatus, MetricPayload, Phase};
    pub use crate::fields::{
        DefaultValue, FieldDescriptor, FieldError, FieldKind, FieldShape, JobFields,
    };
    pub use crate::scheduler::LearningRateScheduler;
    pub use crate::secrets::SecretString;
    pub use crate::traits::*;
    pub use crate::types::*;
}

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

/// The same for `pmetal-core-derive`'s README. A proc-macro crate cannot run
/// its own doctests (the derive's output names `::pmetal_core`, and a
/// dev-dependency back on this crate would be a dependency cycle), so they run
/// here, where the derive can be used. The file is read only under
/// `cfg(doctest)`, so a packaged crate builds without it.
#[cfg(doctest)]
#[doc = include_str!("../../pmetal-core-derive/README.md")]
pub struct DeriveReadmeDoctests;
