//! Context parallelism for long-sequence inference.
//!
//! Splits sequences across ranks along the sequence dimension, enabling
//! inference on sequences longer than a single node's memory can hold.
//! Key/value (or query) blocks are exchanged between ranks in a ring.
//!
//! # Design
//!
//! - **Pass-KV** (prefill): each rank holds its share of Q, K and V and passes
//!   K/V blocks around the ring. Partial attention is computed at each step
//!   and merged with an online softmax, so the result is exactly full
//!   attention. Causal masking works from token positions, and the zig-zag
//!   [`SequenceLayout`](crate::context_parallel::SequenceLayout) keeps the causal work even across ranks.
//!
//! - **Pass-Q** (short queries over a long cache): query blocks travel, with
//!   their partial outputs, and arrive home after visiting every rank.
//!
//! # Hybrid Model Consideration
//!
//! Only attention layers participate in context parallelism.
//! GDN layers process sequences recurrently and keep state local —
//! no cross-rank communication needed for GDN.
//!
//! # Reference
//!
//! - Ring Attention (Liu et al., 2023)
//! - Striped Attention (Brandon et al., 2023)
//! - Context Parallelism for scalable million-token inference (Yang et al., 2024)

pub mod ring_attention;
pub mod sequence_split;

pub use ring_attention::{
    CPMode, RingAttentionConfig, merge_partial_attention, ring_attention_forward,
};
pub use sequence_split::{
    SequenceLayout, gather_sequence, local_seq_range, split_sequence, split_sequence_with_layout,
};
