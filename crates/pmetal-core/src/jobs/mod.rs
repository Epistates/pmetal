#![expect(
    missing_docs,
    reason = "field docs live in `#[job(label, help)]` attributes, which the lint cannot see"
)]

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
pub mod imatrix;
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
pub mod train_diffusion;
pub mod train_draft;
pub mod train_mtp;
pub mod vocode;

pub use bench::BenchSpec;
pub use decide::DecideSpec;
pub use dflash::DflashSpec;
pub use distill::DistillSpec;
pub use embed_train::EmbedTrainSpec;
pub use eval::EvalSpec;
pub use fuse::FuseSpec;
pub use grpo::GrpoSpec;
pub use imatrix::ImatrixSpec;
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
pub use train_diffusion::TrainDiffusionSpec;
pub use train_draft::TrainDraftSpec;
pub use train_mtp::TrainMtpSpec;
pub use vocode::VocodeSpec;

/// The highest learning rate `--full` accepts: above it a full fine-tune of a
/// pretrained model diverges, and LoRA's default (2e-4) is above it.
pub(crate) const FULL_FINETUNE_MAX_LEARNING_RATE: f64 = 1e-4;

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

/// The schedule a spec's `lr_schedule` field names. An unknown name is an
/// error, never a quiet fall back to cosine.
pub fn parse_lr_scheduler(name: &str) -> Result<crate::LrSchedulerType, String> {
    name.parse()
}

/// A job spec a surface can build a job from: the CLI's own, a TUI form's, an
/// MCP tool's. Everything that varies between specs is behind this trait
/// (descriptors and argv through [`JobFields`](crate::JobFields), defaults
/// through `Default`, the wire form through serde) except the cross-field
/// rules, which each spec's `normalize` holds and this trait calls.
pub trait Spec:
    crate::JobFields + Default + serde::Serialize + serde::de::DeserializeOwned + Send + Sync + 'static
{
    /// The spec's cross-field validation, which also runs the descriptor checks.
    fn normalize_spec(&mut self) -> Result<(), Vec<crate::FieldError>>;
}

macro_rules! impl_spec {
    ($($ty:ty),+ $(,)?) => {$(
        impl Spec for $ty {
            fn normalize_spec(&mut self) -> Result<(), Vec<crate::FieldError>> {
                self.normalize()
            }
        }
    )+};
}

impl_spec!(
    BenchSpec,
    DecideSpec,
    DflashSpec,
    DistillSpec,
    EmbedTrainSpec,
    EvalSpec,
    FuseSpec,
    GrpoSpec,
    ImatrixSpec,
    InferSpec,
    MergeSpec,
    PackExpertsSpec,
    PreferenceSpec,
    PretrainSpec,
    QuantizeSpec,
    RlkdSpec,
    ServeSpec,
    TokenizeSpec,
    TrainSpec,
    TrainDiffusionSpec,
    TrainDraftSpec,
    TrainMtpSpec,
    VocodeSpec,
);
