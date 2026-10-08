//! OpenAI-compatible inference server for PMetal.
//!
//! Provides a drop-in local inference backend compatible with the OpenAI API
//! and the Anthropic Messages API.
//!
//! # Supported Endpoints
//!
//! - `POST /v1/chat/completions` — non-streaming and SSE streaming chat completions;
//!   Qwen3.5-family vision models also read images and videos, Llama 3.2
//!   Vision images (see [`media`])
//! - `POST /v1/completions` — raw text completions
//! - `POST /v1/messages` — Anthropic-compatible message generation
//! - `GET /v1/models` — list loaded models
//! - `GET /v1/metrics` — rolling serving metrics (tok/s, latencies, request counts)
//! - `GET /health` — liveness check
//!
//! A decision model (a Clef release) is served by its own router instead, with
//! `POST /v1/systemone`, `GET /v1/models` and `GET /health`; see [`decision`].

#![allow(clippy::too_many_arguments)]

pub mod anthropic;
pub mod continuous_batch;
pub mod continuous_driver;
pub mod continuous_pump;
pub mod decision;
pub mod engine;
pub mod error;
pub mod media;
pub mod prefix_cache;
pub mod routes;
pub mod server;
pub(crate) mod sse;
pub mod types;

pub use continuous_batch::{
    BatcherConfig, ContinuousBatcher, EnqueueError, FinishReason, SlotId, SlotParams, SlotState,
    StepInstruction,
};
pub use continuous_driver::{
    ContinuousEngineState, SlotForward, SlotIdxMap, SlotStepOutput, drive_decode_step,
    drive_prefill_step,
};
pub use continuous_pump::{ContinuousPump, Tick};
pub use decision::DecisionEngine;
pub use engine::{InferenceEngine, PreparedPrompt, RequestMetrics};
pub use prefix_cache::ServePrefixCache;
pub use routes::ServingMetrics;
pub use server::ServeConfig;

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
