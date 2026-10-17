//! Request extraction that fails the way the APIs do.
//!
//! axum's own `Json` answers a bad body with a plain-text 422. [`ApiJson`]
//! answers with a [`ServeError`] instead: a 400 whose `param` names the field
//! that did not parse, a 413 for a body past the limit, a 415 for a body that
//! is not JSON.

use axum::body::Bytes;
use axum::extract::{FromRequest, Request};
use axum::http::{StatusCode, header};
use serde::de::DeserializeOwned;

use crate::error::ServeError;

/// A request body parsed as `T`, or the spec-shaped error that says why not.
pub struct ApiJson<T>(pub T);

impl<S, T> FromRequest<S> for ApiJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = ServeError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        if !is_json_content_type(request.headers().get(header::CONTENT_TYPE)) {
            return Err(ServeError::UnsupportedMediaType);
        }
        let bytes = Bytes::from_request(request, state)
            .await
            .map_err(|rejection| {
                if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                    ServeError::PayloadTooLarge
                } else {
                    ServeError::BadRequest(format!("could not read the request body: {rejection}"))
                }
            })?;
        parse_json(&bytes).map(ApiJson)
    }
}

/// Whether a `Content-Type` is `application/json` or a `+json` type.
fn is_json_content_type(value: Option<&header::HeaderValue>) -> bool {
    let Some(value) = value.and_then(|v| v.to_str().ok()) else {
        return false;
    };
    let essence = value.split(';').next().unwrap_or("").trim();
    let essence = essence.to_ascii_lowercase();
    essence == "application/json"
        || (essence.starts_with("application/") && essence.ends_with("+json"))
}

/// Parse `bytes` as `T`, naming the field of any error.
pub fn parse_json<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ServeError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
        let path = error.path().to_string();
        let inner = error.into_inner();
        if inner.is_syntax() || inner.is_eof() {
            return ServeError::BadRequest(format!("the request body is not valid JSON: {inner}"));
        }
        let message = inner.to_string();
        // serde reports a missing field at its parent's path.
        let param = match missing_field(&message) {
            Some(field) if path == "." => field.to_string(),
            Some(field) => format!("{path}.{field}"),
            None if path == "." => return ServeError::BadRequest(message),
            None => path,
        };
        ServeError::invalid(param, message)
    })
}

/// The field a "missing field `x`" error names.
fn missing_field(message: &str) -> Option<&str> {
    let rest = message.strip_prefix("missing field `")?;
    rest.split('`').next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, serde::Deserialize)]
    struct Body {
        #[allow(dead_code)]
        model: String,
        #[allow(dead_code)]
        n: Option<u32>,
    }

    #[test]
    fn errors_name_their_field() {
        let err = parse_json::<Body>(br#"{"model": "m", "n": "two"}"#).unwrap_err();
        assert!(
            matches!(&err, ServeError::Invalid { param: Some(p), .. } if p == "n"),
            "{err:?}"
        );
        let err = parse_json::<Body>(br#"{"n": 2}"#).unwrap_err();
        assert!(
            matches!(&err, ServeError::Invalid { param: Some(p), .. } if p == "model"),
            "{err:?}"
        );
        let err = parse_json::<Body>(b"{nope").unwrap_err();
        assert!(matches!(err, ServeError::BadRequest(_)));
    }

    #[test]
    fn json_content_types() {
        let ct = |s: &'static str| header::HeaderValue::from_static(s);
        assert!(is_json_content_type(Some(&ct("application/json"))));
        assert!(is_json_content_type(Some(&ct(
            "application/json; charset=utf-8"
        ))));
        assert!(is_json_content_type(Some(&ct("Application/Vnd.api+json"))));
        assert!(!is_json_content_type(Some(&ct("text/plain"))));
        assert!(!is_json_content_type(None));
    }
}
