//! Per-subcommand clap argument structs.
//!
//! Each module contains a thin `<Subcommand>Args` struct that carries only clap
//! metadata.  See `README.md` in this directory for the full migration pattern.

pub mod bench;
pub mod dflash;
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
pub mod distill;
