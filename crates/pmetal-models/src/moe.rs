//! Mixture of Experts (MoE) re-exports from pmetal-mlx.
//!
//! The shared expert MLP ([`Expert`]) and its bias-free [`Linear`]. Routing
//! is in [`crate::moe_routing`]; each architecture runs its experts in its
//! own MoE block.

pub use pmetal_mlx::moe::*;
