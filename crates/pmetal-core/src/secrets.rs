//! Secure handling of sensitive data like API tokens and credentials.
//!
//! This module provides types that help prevent accidental exposure of secrets
//! in logs, error messages, and debug output.

use std::fmt;

/// A string type that redacts its content in Debug and Display implementations.
///
/// Use `SecretString` for handling sensitive data like API tokens, passwords,
/// or other credentials. The underlying value can only be accessed explicitly
/// via [`expose_secret`][SecretString::expose_secret].
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
#[derive(Clone)]
pub struct SecretString {
    inner: String,
}

impl SecretString {
    /// Create a new `SecretString` from a string value.
    pub fn new(secret: impl Into<String>) -> Self {
        Self {
            inner: secret.into(),
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
        &self.inner
    }

    /// Check if the secret is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the length of the secret.
    pub fn len(&self) -> usize {
        self.inner.len()
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

impl Default for SecretString {
    fn default() -> Self {
        Self {
            inner: String::new(),
        }
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

/// Zeroize the secret on drop to reduce the window of exposure.
impl Drop for SecretString {
    fn drop(&mut self) {
        // Overwrite the string contents with zeros before deallocating.
        // Note: This is a best-effort approach; the Rust optimizer or memory
        // allocator may not guarantee complete erasure. For high-security
        // applications, consider using the `secrecy` or `zeroize` crate.
        //
        // Safety: We're modifying our own string's bytes before it's deallocated.
        // We use write_volatile to prevent the compiler from optimizing this away.
        if !self.inner.is_empty() {
            // SAFETY:
            // 1. We own the String exclusively (Drop takes &mut self)
            // 2. The pointer from as_mut_ptr() is valid for self.inner.len() bytes
            // 3. We use write_volatile to prevent compiler optimization from eliding
            //    the zeroing operation
            // 4. The memory is properly aligned for u8 (1-byte alignment)
            // 5. After this, the String contains only zeros but maintains valid UTF-8
            //    (zeros are valid UTF-8 NUL characters)
            unsafe {
                let ptr = self.inner.as_mut_ptr();
                let len = self.inner.len();
                for i in 0..len {
                    std::ptr::write_volatile(ptr.add(i), 0);
                }
            }
        }
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
