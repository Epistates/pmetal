//! Per-subcommand clap argument structs.
//!
//! Each module contains a thin `<Subcommand>Args` struct that carries only clap
//! metadata.  See `README.md` in this directory for the full migration pattern.

pub mod bench;
#[cfg(feature = "distributed")]
pub mod cluster;
pub mod dflash;
pub mod distill;
pub mod embed_train;
pub mod eval;
pub mod fuse;
pub mod grpo;
pub mod infer;
pub mod merge;
pub mod pack_experts;
pub mod pretrain;
pub mod quantize;
pub mod rlkd;
pub mod serve;
pub mod tokenize;
pub mod train;
#[cfg(feature = "trainer")]
pub mod train_draft;
#[cfg(feature = "trainer")]
pub mod train_mtp;
