//! Manifold-Constrained Hyper-Connections (mHC) for stable deep learning training.
//!
//! This crate implements the mHC framework from DeepSeek-AI, which addresses training
//! instability in Hyper-Connections by constraining residual connection matrices to
//! lie on a doubly stochastic manifold.
//!
//! It is a self-contained CPU reference implementation on `ndarray` (`f32`, with
//! the reductions that need it accumulated in `f64`): forward and backward
//! passes, written out and checked against finite differences and against
//! float64 evaluations of the paper's equations. Accelerated implementations
//! test themselves against it.
//!
//! # Overview
//!
//! mHC extends standard residual connections by:
//! 1. Expanding the residual stream from dimension C to n×C (expansion rate n)
//! 2. Learning mappings H^pre, H^post, H^res that mix information across streams
//! 3. Constraining H^res to be doubly stochastic via Sinkhorn-Knopp projection
//!    (and H^pre, H^post to be positive via sigmoids)
//!
//! The layer is `x_{l+1} = H^res x_l + H^postᵀ F(H^pre x_l)`.
//!
//! # Key Properties
//!
//! - **Identity Mapping**: a fresh layer ([`MhcParams::identity_init`]) is a plain
//!   residual connection
//! - **Compositional Closure**: Products of doubly stochastic matrices remain doubly stochastic
//! - **Stable Gradients**: Amax gain magnitude ≈ 1 throughout training
//!
//! # Example
//!
//! ```
//! use ndarray::Array3;
//! use pmetal_mhc::{MhcConfig, MhcLayer};
//!
//! # fn main() -> Result<(), pmetal_mhc::MhcError> {
//! let config = MhcConfig { hidden_dim: 16, ..MhcConfig::default() };
//! let layer = MhcLayer::new(config)?;
//!
//! // [batch, n streams, C]
//! let x = Array3::from_elem((2, 4, 16), 0.1f32);
//! let y = layer.forward(&x, |h_in| {
//!     // your attention / FFN computation on the aggregated [batch, C] input
//!     h_in.mapv(|v| 2.0 * v)
//! })?;
//! assert_eq!(y.shape(), x.shape());
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - [mHC Paper](https://arxiv.org/abs/2512.24880): "mHC: Manifold-Constrained Hyper-Connections"
//! - [HC Paper](https://arxiv.org/abs/2409.19606): "Hyper-Connections"

#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod layer;
pub mod mappings;
pub mod params;
pub mod sinkhorn;

// Re-exports for convenience
pub use config::{MhcConfig, MhcConfigError, MhcPreset};
pub use error::MhcError;
pub use layer::{CollapseMode, MhcLayer, MhcTransformerBlock, collapse_streams, expand_to_streams};
pub use mappings::{
    PostResGrads, apply_post_res_mapping, apply_post_res_mapping_backward, apply_pre_mapping,
    apply_pre_mapping_backward, compute_mappings, compute_mappings_backward,
};
pub use params::{MhcGradients, MhcMappings, MhcParams, RES_OFF_DIAGONAL_INIT};
pub use sinkhorn::{
    SinkhornConfig, SinkhornResult, amax_gain_backward, amax_gain_forward, composite_mapping,
    is_doubly_stochastic, sinkhorn_knopp, sinkhorn_knopp_backward, sinkhorn_knopp_backward_batch,
    sinkhorn_knopp_batch,
};

/// Version of the mHC implementation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default expansion rate (n=4 as per paper).
pub const DEFAULT_EXPANSION_RATE: usize = 4;

/// Default number of Sinkhorn iterations (t_max=20 as per paper).
pub const DEFAULT_SINKHORN_ITERATIONS: usize = 20;

/// Prelude module for convenient imports.
///
/// Convenient re-exports for common usage.
pub mod prelude {
    pub use crate::config::{MhcConfig, MhcPreset};
    pub use crate::error::MhcError;
    pub use crate::layer::{MhcLayer, MhcTransformerBlock};
    pub use crate::params::{MhcGradients, MhcMappings, MhcParams};
    pub use crate::sinkhorn::SinkhornConfig;
}

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
