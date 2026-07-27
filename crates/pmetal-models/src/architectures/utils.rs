//! Shared utilities for model architecture implementations.

use std::collections::HashMap;

use pmetal_bridge::compat::{Array, Dtype, Exception, Param, nn, ops};

/// Outcome of a hand-written weight loader.
///
/// Architectures whose checkpoint layout the generic loader can't express (tied
/// trunks, transposed convolutions, per-arrangement embedding tables) assign
/// weights by explicit key and report what they found here, so a partial load
/// is visible instead of silently leaving random init in place.
#[derive(Debug, Default, Clone)]
pub struct LoadReport {
    /// Number of weights successfully assigned.
    pub loaded: usize,
    /// Keys that were looked for and not found.
    pub skipped: Vec<String>,
}

/// Assign `weights[key]` to a raw parameter slot, recording the outcome.
pub fn load_param(
    slot: &mut Param<Array>,
    weights: &HashMap<String, Array>,
    key: &str,
    report: &mut LoadReport,
) {
    match weights.get(key) {
        Some(w) => {
            *slot = Param::new(w.clone());
            report.loaded += 1;
        }
        None => report.skipped.push(key.to_string()),
    }
}

/// Assign `weights[key]` to an optional parameter slot (a bias, or a gate that
/// only gated layers carry), recording the outcome.
pub fn load_optional_param(
    slot: &mut Param<Option<Array>>,
    weights: &HashMap<String, Array>,
    key: &str,
    report: &mut LoadReport,
) {
    match weights.get(key) {
        Some(w) => {
            *slot = Param::new(Some(w.clone()));
            report.loaded += 1;
        }
        None => report.skipped.push(key.to_string()),
    }
}

/// Assign a linear layer's `weight` from `{prefix}.weight`, and its `bias` from
/// `{prefix}.bias` when the layer has one.
pub fn load_linear(
    linear: &mut nn::Linear,
    weights: &HashMap<String, Array>,
    prefix: &str,
    report: &mut LoadReport,
) {
    load_param(
        &mut linear.weight,
        weights,
        &format!("{prefix}.weight"),
        report,
    );
    if linear.bias.value.is_some() {
        load_optional_param(&mut linear.bias, weights, &format!("{prefix}.bias"), report);
    }
}

/// Assign a LayerNorm's `weight` and `bias` from `{prefix}.{weight,bias}`.
pub fn load_layer_norm(
    norm: &mut nn::LayerNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
    report: &mut LoadReport,
) {
    load_optional_param(
        &mut norm.weight,
        weights,
        &format!("{prefix}.weight"),
        report,
    );
    load_optional_param(&mut norm.bias, weights, &format!("{prefix}.bias"), report);
}

/// Cast an additive attention mask to the query dtype, if it isn't already.
///
/// MLX's scaled-dot-product attention requires the mask to promote to the
/// output dtype and errors out rather than upcasting, so an f32 mask against a
/// bf16 checkpoint is a hard failure. Every mask builder here produces f32
/// regardless of the model's dtype, so any code path that hands a mask straight
/// to `Array::sdpa_with_mask` has to coerce it first.
///
/// `pmetal_mlx::kernels::fused_sdpa` does this internally; this is for the
/// callers that go to the bridge directly.
pub fn coerce_mask_dtype(query: &Array, mask: Option<&Array>) -> Option<Array> {
    mask.map(|mask| {
        if mask.dtype() == query.dtype() {
            mask.clone()
        } else {
            mask.as_dtype(query.dtype().as_i32())
        }
    })
}

/// Create a causal attention mask of shape [seq_len, seq_len].
///
/// Returns an additive mask where masked (future) positions hold `-inf`
/// and valid (past/current) positions hold `0.0`, matching the convention
/// expected by all attention kernels in this crate.
///
/// # Arguments
/// * `seq_len` - Sequence length (mask will be square)
///
/// # Returns
/// Float32 array of shape [seq_len, seq_len]
pub fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    // Lower-triangular matrix: 1.0 where position is valid (past or current)
    let lower_tri = ops::tri(seq_len, seq_len, 0, Dtype::Float32);
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);
    // Where lower_tri == 0 (future positions), set -inf; otherwise 0.0
    let mask = lower_tri.equal(&zero);
    Ok(ops::where_fn(&mask, &neg_inf, &zero))
}
