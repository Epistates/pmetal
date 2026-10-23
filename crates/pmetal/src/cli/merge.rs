//! Clap argument struct for `pmetal merge`.

use clap::Args;
use clap::builder::{PossibleValue, PossibleValuesParser};
use pmetal_core::jobs::MergeSpec;
use pmetal_core::{FieldKind, JobFields};

/// The merge methods the spec offers (its `method` field's options), each also
/// spelled with hyphens, so what `--method` takes and what its help lists are
/// the spec's.
fn merge_methods() -> Vec<PossibleValue> {
    let options = MergeSpec::field_descriptors()
        .iter()
        .find_map(|d| match d.kind {
            FieldKind::Enum { options } if d.name == "method" => Some(options),
            _ => None,
        })
        .unwrap_or_default();
    options
        .iter()
        .map(|name| {
            let mut value = PossibleValue::new(*name);
            if name.contains('_') {
                // Built once, for the life of the process.
                value = value.alias(&*Box::leak(name.replace('_', "-").into_boxed_str()));
            }
            if *name == "multislerp" {
                value = value.aliases(["multi_slerp", "multi-slerp"]);
            }
            value
        })
        .collect()
}

/// Merge two models into a new one.
///
/// Exactly two models are taken (`--model-a`, `--model-b`). The methods that
/// merge changes from a common ancestor need that ancestor as `--base`.
#[derive(Args, Debug)]
pub struct MergeArgs {
    /// First model path or HuggingFace ID
    #[arg(short = 'a', long = "model-a")]
    pub model_a: String,

    /// Second model path or HuggingFace ID
    #[arg(short = 'b', long = "model-b")]
    pub model_b: String,

    /// Output directory for merged model
    #[arg(short, long = "output")]
    pub output: String,

    /// Merge method
    #[arg(long = "method", default_value = "slerp", value_parser = PossibleValuesParser::new(merge_methods()))]
    pub method: String,

    /// Base model the other two descend from, for the task-vector methods (TIES, DARE,
    /// DELLA, task_arithmetic, breadcrumbs, model_stock, consensus, ram)
    #[arg(long = "base")]
    pub base: Option<String>,

    /// Interpolation parameter t for SLERP (0.0=model_a, 1.0=model_b)
    #[arg(long = "t", default_value = "0.5")]
    pub t: f32,

    /// Weight for model_a in linear/ties methods
    #[arg(long = "weight-a", default_value = "0.5")]
    pub weight_a: f32,

    /// Weight for model_b in linear/ties methods
    #[arg(long = "weight-b", default_value = "0.5")]
    pub weight_b: f32,

    /// Density for sparsification (TIES/DARE) — fraction of params to keep
    #[arg(long = "density", default_value = "0.5")]
    pub density: f32,

    /// Output dtype (float32, float16, bfloat16)
    #[arg(long = "dtype", default_value = "bfloat16")]
    pub dtype: String,
}

impl From<MergeArgs> for MergeSpec {
    fn from(a: MergeArgs) -> Self {
        MergeSpec {
            model_a: a.model_a,
            model_b: a.model_b,
            output: a.output,
            method: a.method,
            base: a.base,
            t: a.t,
            weight_a: a.weight_a,
            weight_b: a.weight_b,
            density: a.density,
            dtype: a.dtype,
        }
    }
}
