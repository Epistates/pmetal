//! The error every command returns to the frontend.
//!
//! The frontend shows the message as it is, so each one says what failed and,
//! where there is something to do about it, what to do.

use serde::Serialize;

/// A message for the person using the app.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct AppError(String);

impl AppError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AppError {}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        Self(e.to_string())
    }
}

impl From<String> for AppError {
    fn from(message: String) -> Self {
        Self(message)
    }
}

pub type Result<T> = std::result::Result<T, AppError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_error_reaches_the_frontend_as_its_bare_message() {
        let json = serde_json::to_string(&AppError::new("No model named x")).unwrap();
        assert_eq!(json, r#""No model named x""#);
    }
}
