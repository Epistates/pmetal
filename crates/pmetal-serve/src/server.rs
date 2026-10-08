//! Server setup and configuration.

use crate::anthropic;
use crate::continuous_batch::BatcherConfig;
use crate::engine::InferenceEngine;
use crate::routes::{self, AppState, ServingMetrics};
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Port to listen on.
    pub port: u16,
    /// Host to bind to.
    pub host: String,
    /// Maximum concurrent requests.
    pub max_concurrent: usize,
    /// When set, enable the continuous-batching engine at startup with
    /// the given slot-pool configuration. `None` (default) keeps the
    /// engine on the single-request `generate_streaming` path.
    pub continuous_batching: Option<BatcherConfig>,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            // Default to loopback only — callers that need external access should
            // explicitly set host to "0.0.0.0" or a specific interface address.
            host: "127.0.0.1".to_string(),
            max_concurrent: 16,
            continuous_batching: None,
        }
    }
}

/// Build the axum router with all routes.
pub fn build_router(engine: InferenceEngine, max_concurrent: usize) -> Router {
    let state = Arc::new(AppState {
        engine,
        metrics: ServingMetrics::default(),
        request_permits: Arc::new(tokio::sync::Semaphore::new(max_concurrent.max(1))),
    });

    Router::new()
        .route("/health", axum::routing::get(routes::health))
        .route("/v1/models", axum::routing::get(routes::list_models))
        .route("/v1/metrics", axum::routing::get(routes::serving_metrics))
        .route(
            "/v1/chat/completions",
            axum::routing::post(routes::chat_completions),
        )
        .route("/v1/completions", axum::routing::post(routes::completions))
        .route("/v1/embeddings", axum::routing::post(routes::embeddings))
        .route("/v1/messages", axum::routing::post(anthropic::messages))
        .route(
            "/v1/systemone",
            axum::routing::post(crate::decision::not_a_decision_model),
        )
        .layer(TraceLayer::new_for_http())
        // Reject request bodies larger than 2 MiB to prevent memory exhaustion.
        .layer(RequestBodyLimitLayer::new(2 * 1024 * 1024))
        .with_state(state)
}

/// Start the server.
pub async fn run_server(engine: InferenceEngine, config: ServeConfig) -> anyhow::Result<()> {
    // Opt-in continuous batching. Enabled here (before building the
    // router) so the driver thread is up and the route handler can
    // rely on `continuous_batching_enabled()` returning a stable
    // answer for the lifetime of the server.
    if let Some(batcher_config) = config.continuous_batching.clone() {
        tracing::info!(
            max_slots = batcher_config.max_slots,
            max_queue_depth = batcher_config.max_queue_depth,
            block_size = batcher_config.effective_block_size(),
            max_blocks = batcher_config.max_blocks,
            "Enabling continuous batching"
        );
        engine.enable_continuous_batching_auto(batcher_config)?;
    }

    let router = build_router(engine, config.max_concurrent);
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;

    if !addr.ip().is_loopback() {
        tracing::warn!(
            "binding the inference server to a non-loopback address without authentication; \
             expose this only on trusted networks"
        );
    }

    tracing::info!("Starting PMetal inference server on {}", addr);
    tracing::info!("OpenAI-compatible API available at http://{}/v1", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
