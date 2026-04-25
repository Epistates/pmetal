//! Shared building blocks reused across decoder architectures.
//!
//! `batched_attention` owns the fused `[N_active, 1, H]` GQA attention
//! block that every Tier-1 / Tier-2 fused-decode architecture delegates
//! to. Keeping it in one place avoids ~10 copies of the same q/k/v/rope/
//! sdpa sequence and lets future kernel upgrades land once.

pub mod batched_attention;

pub use batched_attention::{
    BatchedGqaAttnCfg, batched_gqa_attn, batched_parallel_block, batched_perinorm_layer,
    batched_prenorm_layer,
};
