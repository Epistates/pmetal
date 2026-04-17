//! OpenAI-compatible inference server for PMetal.
//!
//! Provides a drop-in local inference backend compatible with the OpenAI API
//! and the Anthropic Messages API.
//!
//! # Supported Endpoints
//!
//! - `POST /v1/chat/completions` — non-streaming and SSE streaming chat completions
//! - `POST /v1/completions` — raw text completions
//! - `POST /v1/messages` — Anthropic-compatible message generation
//! - `GET /v1/models` — list loaded models
//! - `GET /v1/metrics` — rolling serving metrics (tok/s, latencies, request counts)
//! - `GET /health` — liveness check

#![allow(clippy::too_many_arguments)]

pub mod anthropic;
pub mod engine;
pub mod error;
pub mod routes;
pub mod server;
pub mod types;

pub use engine::{InferenceEngine, RequestMetrics};
pub use routes::ServingMetrics;
pub use server::ServeConfig;
