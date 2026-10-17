//! Server setup and configuration.

use crate::anthropic;
use crate::continuous_batch::BatcherConfig;
use crate::engine::InferenceEngine;
use crate::error::{ErrorInfo, ServeError};
use crate::routes::{self, AppState, ServingMetrics};
use axum::Router;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
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
    /// API keys that may call the server. Empty (the default) leaves it open,
    /// which is only sane on a loopback address. A request carries a key as
    /// `Authorization: Bearer <key>` (OpenAI clients) or `x-api-key: <key>`
    /// (Anthropic clients); `/health` needs none.
    pub api_keys: Vec<String>,
    /// Browser origins (`https://app.example`) whose pages may call the
    /// server, or `*` for any. Empty (the default) sends no CORS headers, so
    /// a browser lets only same-origin pages read the replies.
    pub cors_allowed_origins: Vec<String>,
    /// The largest request body accepted, in bytes.
    pub max_body_bytes: usize,
    /// How long one request may run, from arrival to its last byte; `None`
    /// lets a request run until it finishes or the client leaves.
    pub request_timeout: Option<Duration>,
    /// The most choices (`n`) a request may ask for.
    pub max_choices: usize,
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
            api_keys: Vec::new(),
            cors_allowed_origins: Vec::new(),
            max_body_bytes: MAX_REQUEST_BODY_BYTES,
            request_timeout: Some(Duration::from_secs(30 * 60)),
            max_choices: 16,
        }
    }
}

/// The largest request body the server reads by default: room for a few
/// base64-encoded photos or a short clip's frames (base64 adds a third to the
/// bytes).
pub const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Build the axum router with all routes and the default configuration.
pub fn build_router(engine: InferenceEngine, max_concurrent: usize) -> Router {
    build_router_with(
        engine,
        &ServeConfig {
            max_concurrent,
            ..ServeConfig::default()
        },
    )
}

/// Build the axum router with all routes, under `config`.
pub fn build_router_with(engine: InferenceEngine, config: &ServeConfig) -> Router {
    let state = Arc::new(AppState {
        engine,
        metrics: ServingMetrics::default(),
        request_permits: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent.max(1))),
        request_timeout: config.request_timeout,
        max_choices: config.max_choices.max(1),
    });

    let router = Router::new()
        .route("/health", axum::routing::get(routes::health))
        .route("/v1/models", axum::routing::get(routes::list_models))
        .route("/v1/models/{*id}", axum::routing::get(routes::get_model))
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
        );
    harden(router, config).with_state(state)
}

/// The layers every router of this server carries, outermost last: request
/// logging, CORS, spec-shaped errors, the body limit, the API key.
pub(crate) fn harden<S>(router: Router<S>, config: &ServeConfig) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    let router = router.fallback(|| async { StatusCode::NOT_FOUND });
    let router = if config.api_keys.is_empty() {
        router
    } else {
        let keys: Arc<[String]> = config.api_keys.clone().into();
        router.layer(middleware::from_fn_with_state(keys, require_api_key))
    };
    let router = limit_request_bodies(router, config.max_body_bytes)
        .layer(middleware::from_fn(shape_errors));
    let router = match cors_layer(&config.cors_allowed_origins) {
        Some(cors) => router.layer(cors),
        None => router,
    };
    router.layer(TraceLayer::new_for_http())
}

/// CORS for the origins configured, or none at all.
fn cors_layer(origins: &[String]) -> Option<CorsLayer> {
    if origins.is_empty() {
        return None;
    }
    let allow_origin = if origins.iter().any(|origin| origin == "*") {
        AllowOrigin::from(Any)
    } else {
        let list: Vec<HeaderValue> = origins
            .iter()
            .filter_map(|origin| HeaderValue::from_str(origin.trim_end_matches('/')).ok())
            .collect();
        AllowOrigin::list(list)
    };
    Some(
        CorsLayer::new()
            .allow_origin(allow_origin)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([
                header::AUTHORIZATION,
                header::CONTENT_TYPE,
                header::HeaderName::from_static("x-api-key"),
                header::HeaderName::from_static("anthropic-version"),
                header::HeaderName::from_static("anthropic-beta"),
            ])
            .max_age(Duration::from_secs(600)),
    )
}

/// Cap request bodies at `max_bytes`: images and video frames arrive
/// base64-encoded in the body, and anything past the cap is refused before it
/// is buffered.
///
/// axum's `Json` extractor carries a 2 MiB limit of its own
/// (`DefaultBodyLimit`), which refused any request holding a photo however
/// high the outer limit was set; it is lifted so this one limit holds.
pub(crate) fn limit_request_bodies<S>(router: Router<S>, max_bytes: usize) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(max_bytes))
}

/// Whether `candidate` is one of `keys`, without the time it takes depending
/// on how many leading bytes match.
fn key_matches(keys: &[String], candidate: &str) -> bool {
    let mut found = false;
    for key in keys {
        let (a, b) = (key.as_bytes(), candidate.as_bytes());
        let mut diff = a.len() ^ b.len();
        for i in 0..a.len().max(b.len()) {
            diff |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
        }
        found |= diff == 0;
    }
    found
}

/// Refuse a request without one of the configured API keys. `/health` and
/// CORS preflights are open: a load balancer and a browser ask before they
/// can know the key.
async fn require_api_key(
    State(keys): State<Arc<[String]>>,
    request: Request,
    next: Next,
) -> Response {
    if request.uri().path() == "/health" || request.method() == Method::OPTIONS {
        return next.run(request).await;
    }
    let headers = request.headers();
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            let (scheme, token) = v.split_once(' ')?;
            scheme
                .eq_ignore_ascii_case("bearer")
                .then_some(token.trim())
        });
    let api_key = headers.get("x-api-key").and_then(|v| v.to_str().ok());
    if bearer
        .into_iter()
        .chain(api_key)
        .any(|key| key_matches(&keys, key))
    {
        return next.run(request).await;
    }
    let mut response = ServeError::Unauthorized(
        "Incorrect API key provided. Send one as `Authorization: Bearer <key>` or `x-api-key`."
            .into(),
    )
    .into_response();
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

/// Give every error response the body its API's spec asks for: OpenAI's
/// error object, or Anthropic's on `/v1/messages`. Handlers' errors carry
/// their [`ErrorInfo`]; a failure the framework or a layer answered with an
/// empty body (a body over the limit, an unknown route, a wrong method) is
/// given one here.
async fn shape_errors(request: Request, next: Next) -> Response {
    let anthropic = request.uri().path().starts_with("/v1/messages");
    let mut response = next.run(request).await;
    let status = response.status();
    if !(status.is_client_error() || status.is_server_error()) {
        return response;
    }
    let info = match response.extensions_mut().remove::<ErrorInfo>() {
        Some(info) => info,
        None if response.headers().contains_key(header::CONTENT_TYPE) => return response,
        None => ErrorInfo::from_status(status),
    };
    let body = if anthropic {
        info.anthropic_body()
    } else {
        info.openai_body()
    };
    let (mut parts, _) = response.into_parts();
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    (parts, axum::Json(body)).into_response()
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

    let router = build_router_with(engine, &config);
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    warn_if_exposed(&addr, &config);

    tracing::info!("Starting PMetal inference server on {}", addr);
    tracing::info!("OpenAI-compatible API available at http://{}/v1", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}

/// A server reachable beyond this machine with no API key lets anyone on the
/// network spend the GPU.
pub(crate) fn warn_if_exposed(addr: &SocketAddr, config: &ServeConfig) {
    if !addr.ip().is_loopback() && config.api_keys.is_empty() {
        tracing::warn!(
            "binding the server to a non-loopback address without an API key; \
             anyone who can reach {addr} can use it. Set an API key or expose it only on \
             trusted networks"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::key_matches;

    #[test]
    fn keys_compare_whole() {
        let keys = vec!["sk-one".to_string(), "sk-two".to_string()];
        assert!(key_matches(&keys, "sk-one"));
        assert!(key_matches(&keys, "sk-two"));
        assert!(!key_matches(&keys, "sk-on"));
        assert!(!key_matches(&keys, "sk-onee"));
        assert!(!key_matches(&keys, ""));
        assert!(!key_matches(&[], "sk-one"));
    }
}
