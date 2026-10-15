//! ZeRO (Zero Redundancy Optimizer) style memory optimization.
//!
//! Shards the optimizer state, and with it the optimizer's work, across ranks
//! so a rank holds `1/N` of it instead of all of it. For Adam in f32 that is
//! two extra copies of the parameters, the largest part of training memory
//! after the model itself.
//!
//! # Stages
//!
//! - **Stage 1**: optimizer state is sharded. Each rank updates its own
//!   `1/N` of the parameters and the shards are all-gathered.
//! - **Stage 2**: and the gradient is reduce-scattered into a shard that
//!   replaces the full buffer.
//!
//! Stage 3 (parameters sharded, gathered layer by layer around the forward and
//! backward pass) needs hooks in the model's execution and is not here.
//!
//! [`ZeroLayout`] cuts the *flat* parameter vector into equal shards, not the
//! parameter list into whole tensors, so shards stay equal when one tensor
//! (the embedding) dominates. [`ZeroOptimizer::step`] is reduce-scatter,
//! local update, all-gather: the communication of one all-reduce, with the
//! state and update work divided by `N`, and the result bit-identical to a
//! single device.
//!
//! # Reference
//!
//! - ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)
//! - FSDP2 (PyTorch): Per-parameter sharding

pub mod layout;
pub mod optimizer;

pub use layout::ZeroLayout;
pub use optimizer::{ShardOptimizer, ShardedAdamW, ZeROStage, ZeroOptimizer, ZeroStepStats};
