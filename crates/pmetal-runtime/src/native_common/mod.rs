//! Shared building blocks for the native model architectures
//! (qwen3, gpt_oss, llama4, deepseek).
//!
//! Centralises patterns that previously lived as copy-pasted blocks in each
//! `attention.rs`/`cache.rs` and drifted in subtle ways. Today this module
//! covers KV cache allocation/growth; new shared helpers should land here so
//! the per-arch files stay focused on the parts that genuinely differ.

pub mod kv_cache;
