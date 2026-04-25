//! Tensor parallelism for distributed inference and training.
//!
//! Splits model weights across multiple ranks so each rank holds a shard
//! of every layer. Communication happens within each layer via `all_sum`
//! (for row-sharded outputs) or implicit gradient synchronization
//! (for column-sharded inputs).
//!
//! # Sharding Pattern
//!
//! Following the Megatron-LM convention used by MLX:
//!
//! - **AllToSharded** (column parallel): Weight `[output/N, input]`.
//!   Each rank computes a slice of the output. No communication needed
//!   in forward; `sum_gradients` barrier needed for backward.
//!
//! - **ShardedToAll** (row parallel): Weight `[output, input/N]`.
//!   Each rank computes a partial result. `all_sum` reduces partial
//!   results to produce the full output.
//!
//! # Architecture Support
//!
//! The [`plan`] module provides architecture-agnostic plan builders
//! that read `config.json` to produce sharding plans for:
//! - Standard transformer attention (Llama, Mistral, etc.)
//! - SwiGLU FFN blocks
//! - GDN (gated delta net) blocks (Qwen 3.5)
//! - MoE layers (shared + routed experts)
//!
//! # Reference
//!
//! Based on mlx-lm's distributed linear layers and the Qwen 3.5
//! `shard()` implementation.

pub mod plan;
pub mod sharded_linear;
pub mod sharding;

pub use plan::{
    build_plan, plan_attention, plan_deepseek_moe, plan_ffn, plan_gdn, plan_gpt_oss_moe,
    plan_llama4_moe, plan_mla, plan_qwen_moe,
};
pub use sharded_linear::{all_to_sharded_forward, sharded_to_all_forward};
pub use sharding::{ShardingDirective, ShardingPlan, shard_weight};
