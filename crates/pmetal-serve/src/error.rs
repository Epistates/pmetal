//! Error types for the inference server, and how they reach the wire.
//!
//! Every error is one [`ServeError`]. A handler returns it and the response
//! is the OpenAI error object
//!
//! ```json
//! {"error": {"message": "...", "type": "invalid_request_error", "param": "n", "code": null}}
//! ```
//!
//! with the HTTP status the failure deserves. `/v1/messages` speaks the
//! Anthropic error object (`{"type": "error", "error": {"type", "message"}}`)
//! instead; [`crate::server`] re-renders the error of a response in the shape
//! of the API the route belongs to, so a handler never picks one.

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

    /// A request field that is wrong, named by `param` (a dotted path such as
    /// `messages.1.content`), with an optional machine-readable `code`.
    #[error("invalid request: {message}")]
    Invalid {
        message: String,
        param: Option<String>,
        code: Option<&'static str>,
    },

    /// The request names a model this server does not serve.
    #[error("the model `{0}` does not exist")]
    ModelNotFound(String),

    /// The request carries no valid API key.
    #[error("{0}")]
    Unauthorized(String),

    /// The prompt and the requested output do not fit in the context window.
    #[error("{0}")]
    ContextLength(String),

    /// The request body is larger than the server accepts.
    #[error("the request body is larger than this server accepts")]
    PayloadTooLarge,

    /// The request body is not sent as JSON.
    #[error("the request body must be JSON (Content-Type: application/json)")]
    UnsupportedMediaType,

    /// The request ran past the server's time limit.
    #[error("the request took longer than {0} seconds")]
    Timeout(u64),

    #[error("model not loaded")]
    ModelNotLoaded,

    #[error("engine busy")]
    Busy,

    #[error("internal error: {0}")]
    Internal(String),
}

impl ServeError {
    /// A bad request whose offending field is `param`.
    pub fn invalid(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Invalid {
            message: message.into(),
            param: Some(param.into()),
            code: None,
        }
    }

    /// A bad request that names the field and a machine-readable code.
    pub fn invalid_with_code(
        param: impl Into<String>,
        code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self::Invalid {
            message: message.into(),
            param: Some(param.into()),
            code: Some(code),
        }
    }

    /// A field this server does not implement. Refusing it beats ignoring it:
    /// a request that sets it expects it to change the reply.
    pub fn unsupported(param: impl Into<String>, why: impl std::fmt::Display) -> Self {
        let param = param.into();
        Self::Invalid {
            message: format!("`{param}` is not supported: {why}"),
            param: Some(param),
            code: Some("unsupported_parameter"),
        }
    }

    /// What the response says: status and the error's parts.
    pub fn describe(&self) -> ErrorInfo {
        let plain = |status: StatusCode, kind: &'static str, message: String| ErrorInfo {
            status,
            kind,
            message,
            param: None,
            code: None,
        };
        match self {
            Self::BadRequest(msg) => plain(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                msg.clone(),
            ),
            Self::Invalid {
                message,
                param,
                code,
            } => ErrorInfo {
                status: StatusCode::BAD_REQUEST,
                kind: "invalid_request_error",
                message: message.clone(),
                param: param.clone(),
                code: code.map(str::to_owned),
            },
            Self::ModelNotFound(model) => ErrorInfo {
                status: StatusCode::NOT_FOUND,
                kind: "invalid_request_error",
                message: format!(
                    "The model `{model}` does not exist or this server does not serve it."
                ),
                param: Some("model".into()),
                code: Some("model_not_found".into()),
            },
            Self::Unauthorized(msg) => ErrorInfo {
                status: StatusCode::UNAUTHORIZED,
                kind: "authentication_error",
                message: msg.clone(),
                param: None,
                code: Some("invalid_api_key".into()),
            },
            Self::ContextLength(msg) => ErrorInfo {
                status: StatusCode::BAD_REQUEST,
                kind: "invalid_request_error",
                message: msg.clone(),
                param: Some("messages".into()),
                code: Some("context_length_exceeded".into()),
            },
            Self::PayloadTooLarge => ErrorInfo::from_status(StatusCode::PAYLOAD_TOO_LARGE),
            Self::UnsupportedMediaType => {
                ErrorInfo::from_status(StatusCode::UNSUPPORTED_MEDIA_TYPE)
            }
            Self::Timeout(_) => plain(
                StatusCode::GATEWAY_TIMEOUT,
                "timeout_error",
                self.to_string(),
            ),
            Self::ModelNotLoaded => plain(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "model not loaded".into(),
            ),
            Self::Busy => ErrorInfo {
                code: Some("rate_limit_exceeded".into()),
                ..plain(
                    StatusCode::TOO_MANY_REQUESTS,
                    "rate_limit_error",
                    "engine busy: every request slot is taken; retry shortly".into(),
                )
            },
            // The cause stays in the log: a client gets no model internals.
            Self::Tokenizer(e) => {
                tracing::error!("Tokenizer error: {e}");
                plain(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "tokenizer error".into(),
                )
            }
            Self::Model(e) => {
                tracing::error!("Model error: {e}");
                plain(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "internal model error".into(),
                )
            }
            Self::Internal(msg) => {
                tracing::error!("Internal error: {msg}");
                plain(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "internal server error".into(),
                )
            }
        }
    }
}

/// An error as the wire carries it.
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    pub status: StatusCode,
    /// OpenAI's `error.type`.
    pub kind: &'static str,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ErrorInfo {
    /// The OpenAI error object.
    pub fn openai_body(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": self.code,
            }
        })
    }

    /// The Anthropic error object. Its `type` vocabulary is its own.
    pub fn anthropic_body(&self) -> serde_json::Value {
        let kind = match self.status {
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_error",
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::FORBIDDEN => "permission_error",
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
            _ => "api_error",
        };
        serde_json::json!({
            "type": "error",
            "error": { "type": kind, "message": self.message }
        })
    }

    /// The error of a failure the framework reported before a handler ran
    /// (a body past the size limit, an unknown route).
    pub fn from_status(status: StatusCode) -> Self {
        let (kind, message) = match status {
            StatusCode::PAYLOAD_TOO_LARGE => (
                "invalid_request_error",
                "the request body is larger than this server accepts".to_string(),
            ),
            StatusCode::NOT_FOUND => (
                "invalid_request_error",
                "no such endpoint on this server".to_string(),
            ),
            StatusCode::METHOD_NOT_ALLOWED => (
                "invalid_request_error",
                "that method is not allowed on this endpoint".to_string(),
            ),
            StatusCode::UNSUPPORTED_MEDIA_TYPE => (
                "invalid_request_error",
                "the request body must be JSON (Content-Type: application/json)".to_string(),
            ),
            s if s.is_client_error() => ("invalid_request_error", "invalid request".to_string()),
            _ => ("server_error", "internal server error".to_string()),
        };
        Self {
            status,
            kind,
            message,
            param: None,
            code: None,
        }
    }
}

impl IntoResponse for ServeError {
    fn into_response(self) -> Response {
        let info = self.describe();
        let mut response = (info.status, axum::Json(info.openai_body())).into_response();
        if matches!(self, Self::Busy) {
            // Every request slot is taken; one frees as soon as a reply ends.
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
        }
        // Kept for [`crate::server`], which re-renders it for the Anthropic API.
        response.extensions_mut().insert(info);
        response
    }
}

pub type ServeResult<T> = Result<T, ServeError>;
