//! Core types, traits, and configuration for PMetal LLM fine-tuning.
//!
//! This crate provides the foundational abstractions used throughout the PMetal
//! framework, including:
//!
//! - Core trait definitions for models, trainers, and quantizers
//! - Configuration types for training, LoRA, and model loading
//! - Common type definitions (Dtype, Device, etc.)
//! - Error handling infrastructure
//! - Learning rate schedulers
//! - Secure handling of secrets (tokens, credentials)

#![warn(missing_docs)]

// Lets `#[derive(JobSpec)]` macros in this crate's own modules resolve their
// generated `::pmetal_core::*` paths correctly. Standard self-rename pattern
// for crates that consume their own derive macro.
extern crate self as pmetal_core;

mod config;
pub mod constants;
mod error;
pub mod events;
pub mod fields;
pub mod jobs;
pub mod scheduler;
mod secrets;
mod traits;
mod types;

pub use config::*;
pub use error::*;
pub use events::{
    BenchTrialMetrics, BroadcastSink, CompletionSummary, JobEvent, JobEventSink, JobKind,
    JobStatus, JsonlSink, LogLevel, MetricPayload, NullSink, ParseError, Phase, Progress,
    ServeRequestMetrics, TrainingCallbackToSink, parse_event, write_event,
};
pub use fields::{DefaultValue, FieldDescriptor, FieldError, FieldKind, JobFields};
pub use scheduler::{LearningRateScheduler, SchedulerBuilder};
pub use secrets::SecretString;
pub use traits::*;
pub use types::*;

/// Re-export of `#[derive(JobSpec)]` from the companion proc-macro crate, so
/// users only ever import from `pmetal_core`.
pub use pmetal_core_derive::JobSpec;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::config::*;
    pub use crate::error::{PMetalError, Result};
    pub use crate::events::{JobEvent, JobEventSink, JobKind, JobStatus, MetricPayload, Phase};
    pub use crate::fields::{DefaultValue, FieldDescriptor, FieldError, FieldKind, JobFields};
    pub use crate::scheduler::{LearningRateScheduler, SchedulerBuilder};
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
