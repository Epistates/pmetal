//! Shared TurboQuant state: the key and value quantizers of a model.
//!
//! Building one costs a Haar-random rotation and a Lloyd-Max codebook per
//! distinct `(head dim, bits)`, so a model builds it once and shares it, in
//! an `Arc`, between all its layers.

use crate::RuntimeResult;
use std::sync::Arc;

use super::config::{TurboQuantConfig, TurboQuantTensorConfig};
use super::quantizer::Quantizer;

/// The quantizers for one pair of key and value head dimensions.
#[derive(Debug, Clone)]
pub struct TurboQuantState {
    pub(super) keys: Arc<Quantizer>,
    pub(super) values: Arc<Quantizer>,
}

impl TurboQuantState {
    /// Build the state for `key_dim`-wide keys and `value_dim`-wide values.
    /// Fails when the configuration asks for something the cache does not
    /// support.
    pub fn new(key_dim: usize, value_dim: usize, config: TurboQuantConfig) -> RuntimeResult<Self> {
        config.validate()?;
        let (
            TurboQuantTensorConfig::Uniform { bits: key_bits },
            TurboQuantTensorConfig::Uniform { bits: value_bits },
        ) = (config.keys, config.values)
        else {
            unreachable!("checked above");
        };
        let keys = Arc::new(Quantizer::new(key_dim, key_bits));
        let values = if value_dim == key_dim && value_bits == key_bits {
            Arc::clone(&keys)
        } else {
            Arc::new(Quantizer::new(value_dim, value_bits))
        };
        Ok(Self { keys, values })
    }

    /// The key quantizer.
    pub fn keys(&self) -> &Quantizer {
        &self.keys
    }

    /// The value quantizer.
    pub fn values(&self) -> &Quantizer {
        &self.values
    }
}
