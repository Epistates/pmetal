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

mod config;
pub mod constants;
mod error;
pub mod events;
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
pub use scheduler::{LearningRateScheduler, SchedulerBuilder};
pub use secrets::SecretString;
pub use traits::*;
pub use types::*;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::config::*;
    pub use crate::error::{PMetalError, Result};
    pub use crate::events::{JobEvent, JobEventSink, JobKind, JobStatus, MetricPayload, Phase};
    pub use crate::scheduler::{LearningRateScheduler, SchedulerBuilder};
    pub use crate::secrets::SecretString;
    pub use crate::traits::*;
    pub use crate::types::*;
}
