//! Error types for the inference server.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("model error: {0}")]
    Model(#[from] pmetal_runtime::compat::Exception),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("invalid request: {0}")]
    BadRequest(String),

    #[error("model not loaded")]
    ModelNotLoaded,

    #[error("engine busy")]
    Busy,

    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ServeError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ServeError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ServeError::ModelNotLoaded => {
                (StatusCode::SERVICE_UNAVAILABLE, "model not loaded".into())
            }
            ServeError::Busy => (StatusCode::TOO_MANY_REQUESTS, "engine busy".into()),
            ServeError::Tokenizer(e) => {
                tracing::error!("Tokenizer error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "tokenizer error".to_string(),
                )
            }
            ServeError::Model(e) => {
                tracing::error!("Model error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal model error".to_string(),
                )
            }
            ServeError::Internal(msg) => {
                tracing::error!("Internal error: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal server error".to_string(),
                )
            }
        };

        let body = serde_json::json!({
            "error": {
                "message": message,
                "type": "server_error",
                "code": status.as_u16()
            }
        });

        (status, axum::Json(body)).into_response()
    }
}

pub type ServeResult<T> = Result<T, ServeError>;
