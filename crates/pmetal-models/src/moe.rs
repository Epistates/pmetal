//! Mixture of Experts (MoE) re-exports from pmetal-mlx.
//!
//! This module re-exports the MoE components from `pmetal_mlx::moe`,
//! providing a unified interface for model architectures.
//!
//! # Available Components
//!
//! - [`MoEConfig`] - Configuration for MoE layers
//! - [`MoERouter`] - Router for expert selection
//! - [`Expert`] - Single expert MLP
//! - [`MoELayer`] - Complete MoE layer with router and experts
//! - [`SparseMoEWithShared`] - MoE with shared experts (DeepSeek style)
//!
//! # Supported Models
//!
//! - DeepSeek V2/V3
//! - Qwen3-MoE
//! - Mixtral
//! - GraniteMoE
//!
//! # Example
//!
//! ```ignore
//! use pmetal_models::moe::{MoEConfig, MoELayer};
//!
//! let config = MoEConfig::new(4096, 14336, 64)
//!     .with_num_experts_per_tok(8);
//!
//! let mut moe = MoELayer::new(config);
//! moe.eval();
//!
//! let (output, aux_loss) = moe.forward(&hidden_states)?;
//! ```

pub use pmetal_mlx::moe::*;
