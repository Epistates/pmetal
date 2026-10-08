// Field-level docs in this module are carried by `#[job(label = "...", help = "...")]`
// attributes, which the missing-docs lint cannot see. Allow the lint here so
// specs stay declarative without duplicating every label as a doc comment.
#![allow(missing_docs)]

//! Canonical per-job specs — one struct per `pmetal` subcommand.
//!
//! Every spec derives [`crate::JobFields`] via `#[derive(JobSpec)]`, exposing
//! per-field metadata that the CLI, TUI, GUI, and MCP all consume the same
//! way. Validation lives on each spec via `normalize`; defaults come from
//! [`crate::defaults`].
//!
//! Specs are the SURFACE-FACING input contract — they hold the fields a user
//! sets when they invoke a job. Conversion to the orchestrator's internal
//! representation (e.g. `pmetal_trainer::orchestrator::TrainingJobConfig`)
//! happens at the call boundary in the consuming crate via
//! `From<*Spec> for *OrchConfig`.

pub mod bench;
pub mod decide;
pub mod dflash;
pub mod distill;
pub mod embed_train;
pub mod eval;
pub mod fuse;
pub mod grpo;
pub mod infer;
pub mod merge;
pub mod pack_experts;
pub mod preference;
pub mod pretrain;
pub mod quantize;
pub mod rlkd;
pub mod serve;
pub mod tokenize;
pub mod train;

pub use bench::BenchSpec;
pub use decide::DecideSpec;
pub use dflash::DflashSpec;
pub use distill::DistillSpec;
pub use embed_train::EmbedTrainSpec;
pub use eval::EvalSpec;
pub use fuse::FuseSpec;
pub use grpo::GrpoSpec;
pub use infer::InferSpec;
pub use merge::MergeSpec;
pub use pack_experts::PackExpertsSpec;
pub use preference::PreferenceSpec;
pub use pretrain::PretrainSpec;
pub use quantize::QuantizeSpec;
pub use rlkd::RlkdSpec;
pub use serve::ServeSpec;
pub use tokenize::TokenizeSpec;
pub use train::TrainSpec;

/// Default of every spec's `optimizer` field.
pub(crate) fn default_optimizer() -> String {
    crate::OptimizerType::default().as_str().to_string()
}

/// The optimizer a spec's `optimizer` field names. `normalize` has already
/// checked the name against the field's options, so this only fails on a
/// spec that skipped it.
pub fn parse_optimizer(name: &str) -> Result<crate::OptimizerType, String> {
    name.parse()
}
