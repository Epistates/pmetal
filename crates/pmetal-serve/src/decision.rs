//! `POST /v1/systemone`: typed decisions from a decision model (Clef).
//!
//! A decision model does not generate, so it is served by its own router:
//! `GET /health`, `GET /v1/models` and `POST /v1/systemone`. The request and
//! response bodies are those of [`pmetal_models::decision::systemone`]; a
//! malformed request is a 400 carrying the same message the release's
//! reference code raises.
//!
//! The model lives on one thread for its whole life, loaded there and run
//! there. MLX streams belong to the thread that made them, so arrays built
//! while loading cannot be evaluated from tokio's blocking pool ("There is no
//! Stream(gpu, 1) in current thread"); requests are queued to the model's
//! thread instead, and answered one at a time.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

use axum::Router;
use axum::extract::State;
use axum::response::Json;
use pmetal_models::decision::{DecisionError, DecisionModel};
use serde_json::Value;
use tokio::sync::{Semaphore, oneshot};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::error::{ServeError, ServeResult};
use crate::server::ServeConfig;
use crate::types::{ModelInfo, ModelListResponse};

type Job = (Value, oneshot::Sender<ServeResult<Value>>);

/// A decision model on its own thread, behind the server.
pub struct DecisionEngine {
    jobs: mpsc::Sender<Job>,
    model_id: String,
    created_at: i64,
}

impl DecisionEngine {
    /// Load the decision model in `model_path` on a new thread and serve it as
    /// `model_id`, encoding prompts of up to `max_length` tokens. Returns once
    /// the model is loaded, or with the reason it could not be.
    pub fn load(model_path: PathBuf, model_id: String, max_length: usize) -> anyhow::Result<Self> {
        let (jobs, queue) = mpsc::channel::<Job>();
        let (ready, loaded) = mpsc::sync_channel::<Result<(), DecisionError>>(1);
        std::thread::Builder::new()
            .name("pmetal-decision".into())
            .spawn(move || {
                let mut model = match DecisionModel::load(&model_path) {
                    Ok(model) => {
                        let _ = ready.send(Ok(()));
                        model
                    }
                    Err(error) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                };
                for (request, reply) in queue {
                    let answer = model.systemone(&request, max_length).map_err(serve_error);
                    let _ = reply.send(answer);
                }
            })?;
        loaded
            .recv()
            .map_err(|_| anyhow::anyhow!("the decision model thread exited while loading"))??;
        Ok(Self {
            jobs,
            model_id,
            created_at: chrono::Utc::now().timestamp(),
        })
    }

    /// The id `GET /v1/models` reports.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Answer a `/v1/systemone` request body.
    pub async fn systemone(&self, request: Value) -> ServeResult<Value> {
        pmetal_models::decision::systemone::validate_request(&request).map_err(serve_error)?;
        let (reply, answer) = oneshot::channel();
        self.jobs
            .send((request, reply))
            .map_err(|_| ServeError::ModelNotLoaded)?;
        answer.await.map_err(|_| ServeError::ModelNotLoaded)?
    }
}

fn serve_error(error: DecisionError) -> ServeError {
    match error {
        DecisionError::Request(message) | DecisionError::Unsupported(message) => {
            ServeError::BadRequest(message)
        }
        DecisionError::Tokenizer(message) => ServeError::Tokenizer(message),
        DecisionError::Model(exception) => ServeError::Model(exception),
        DecisionError::Load(message) => ServeError::Internal(message),
    }
}

struct DecisionAppState {
    engine: DecisionEngine,
    request_permits: Arc<Semaphore>,
}

/// `POST /v1/systemone`.
async fn systemone(
    State(state): State<Arc<DecisionAppState>>,
    Json(request): Json<Value>,
) -> ServeResult<Json<Value>> {
    let _permit = Arc::clone(&state.request_permits)
        .try_acquire_owned()
        .map_err(|_| ServeError::Busy)?;
    Ok(Json(state.engine.systemone(request).await?))
}

/// `GET /v1/models`.
async fn list_models(State(state): State<Arc<DecisionAppState>>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.engine.model_id.clone(),
            object: "model".to_string(),
            created: state.engine.created_at,
            owned_by: "pmetal".to_string(),
        }],
    })
}

/// `POST /v1/systemone` on a server whose model is not a decision model.
pub(crate) async fn not_a_decision_model() -> ServeError {
    ServeError::BadRequest(
        "the served model is not a decision model; /v1/systemone needs a Clef release \
         (joint_head_config.json + joint_head.safetensors)"
            .into(),
    )
}

/// The router for a decision model.
pub fn build_decision_router(engine: DecisionEngine, max_concurrent: usize) -> Router {
    let state = Arc::new(DecisionAppState {
        engine,
        request_permits: Arc::new(Semaphore::new(max_concurrent.max(1))),
    });
    Router::new()
        .route("/health", axum::routing::get(crate::routes::health))
        .route("/v1/models", axum::routing::get(list_models))
        .route("/v1/systemone", axum::routing::post(systemone))
        .layer(TraceLayer::new_for_http())
        .layer(RequestBodyLimitLayer::new(2 * 1024 * 1024))
        .with_state(state)
}

/// Serve a decision model until the process exits.
pub async fn run_decision_server(
    engine: DecisionEngine,
    config: ServeConfig,
) -> anyhow::Result<()> {
    let router = build_decision_router(engine, config.max_concurrent);
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    if !addr.ip().is_loopback() {
        tracing::warn!(
            "binding the decision server to a non-loopback address without authentication; \
             expose this only on trusted networks"
        );
    }
    tracing::info!("Decision API available at http://{}/v1/systemone", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}
