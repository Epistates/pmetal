//! Error type for the mHC crate.

use crate::config::MhcConfigError;

/// Everything the fallible parts of the crate can report.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MhcError {
    /// The configuration is invalid.
    #[error(transparent)]
    Config(#[from] MhcConfigError),

    /// An array does not have the shape the configuration implies.
    #[error("shape mismatch for {what}: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// What was being checked (a parameter or an input).
        what: &'static str,
        /// The shape the configuration requires, e.g. `[4, 2560]`.
        expected: String,
        /// The shape that was given.
        actual: String,
    },

    /// A composite mapping needs at least one matrix.
    #[error("cannot compute the composite of an empty list of mappings")]
    EmptyComposite,

    /// Unknown collapse mode string.
    #[error("invalid collapse mode: '{0}' (expected 'first', 'average', or 'sum')")]
    InvalidCollapseMode(String),
}

impl MhcError {
    pub(crate) fn shape(what: &'static str, expected: impl Into<String>, actual: &[usize]) -> Self {
        Self::ShapeMismatch {
            what,
            expected: expected.into(),
            actual: format!("{actual:?}"),
        }
    }
}
