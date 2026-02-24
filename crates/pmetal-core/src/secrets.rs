//! Secure handling of sensitive data like API tokens and credentials.
//!
//! This module wraps the [`secrecy`] crate to provide types that prevent
//! accidental exposure of secrets in logs, error messages, and debug output.
//! Secrets are zeroized on drop via the `zeroize` crate (transitive dep).

use std::fmt;

use secrecy::ExposeSecret;

/// A string type that redacts its content in Debug and Display implementations.
///
/// Backed by [`secrecy::SecretBox<str>`], which zeroizes the inner string on drop.
/// The underlying value can only be accessed explicitly via
/// [`expose_secret`][SecretString::expose_secret].
///
/// # Example
///
/// ```
/// use pmetal_core::SecretString;
///
/// let token = SecretString::new("sk-secret-key-12345");
///
/// // Debug output shows [REDACTED], not the actual secret
/// assert_eq!(format!("{:?}", token), "SecretString([REDACTED])");
///
/// // Display also shows [REDACTED]
/// assert_eq!(format!("{}", token), "[REDACTED]");
///
/// // Access the secret explicitly when needed
/// assert_eq!(token.expose_secret(), "sk-secret-key-12345");
/// ```
pub struct SecretString {
    inner: secrecy::SecretString,
}

impl Clone for SecretString {
    fn clone(&self) -> Self {
        Self::new(self.expose_secret())
    }
}

impl Default for SecretString {
    fn default() -> Self {
        Self::new("")
    }
}

impl SecretString {
    /// Create a new `SecretString` from a string value.
    pub fn new(secret: impl Into<String>) -> Self {
        let s: String = secret.into();
        Self {
            inner: secrecy::SecretBox::new(s.into_boxed_str()),
        }
    }

    /// Create a `SecretString` from an optional string.
    ///
    /// Returns `None` if the input is `None`.
    pub fn from_option(secret: Option<impl Into<String>>) -> Option<Self> {
        secret.map(|s| Self::new(s))
    }

    /// Expose the secret value.
    ///
    /// This is the only way to access the actual secret content.
    /// Use sparingly and ensure the exposed value is not logged or displayed.
    #[inline]
    pub fn expose_secret(&self) -> &str {
        self.inner.expose_secret()
    }

    /// Check if the secret is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.expose_secret().is_empty()
    }

    /// Get the length of the secret.
    pub fn len(&self) -> usize {
        self.inner.expose_secret().len()
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretString([REDACTED])")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[REDACTED]")
    }
}

impl From<String> for SecretString {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for SecretString {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_string_debug_redacts() {
        let secret = SecretString::new("my-secret-token");
        assert_eq!(format!("{:?}", secret), "SecretString([REDACTED])");
    }

    #[test]
    fn test_secret_string_display_redacts() {
        let secret = SecretString::new("my-secret-token");
        assert_eq!(format!("{}", secret), "[REDACTED]");
    }

    #[test]
    fn test_secret_string_expose() {
        let secret = SecretString::new("my-secret-token");
        assert_eq!(secret.expose_secret(), "my-secret-token");
    }

    #[test]
    fn test_secret_string_from_option() {
        let some_secret = SecretString::from_option(Some("token"));
        assert!(some_secret.is_some());
        assert_eq!(some_secret.unwrap().expose_secret(), "token");

        let no_secret: Option<SecretString> = SecretString::from_option(None::<String>);
        assert!(no_secret.is_none());
    }

    #[test]
    fn test_secret_string_len() {
        let secret = SecretString::new("12345");
        assert_eq!(secret.len(), 5);
        assert!(!secret.is_empty());

        let empty = SecretString::default();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }
}
