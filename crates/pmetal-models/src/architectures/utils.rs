//! Shared utilities for model architecture implementations.

use std::collections::HashMap;

use pmetal_runtime::compat::{Array, Dtype, Exception, Param, nn, ops};

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

/// Combine per-shard reports, so a sharded load reads as one result.
impl std::ops::AddAssign for LoadReport {
    fn add_assign(&mut self, rhs: Self) {
        self.loaded += rhs.loaded;
        self.skipped.extend(rhs.skipped);
    }
}

impl LoadReport {
    /// Log what a load actually matched.
    ///
    /// A checkpoint whose layout the loader does not recognise assigns nothing
    /// and leaves the model on its random init, which downstream looks like a
    /// model that loads fine and emits noise. This is the line that tells you
    /// otherwise, so it is a warning when *nothing* matched.
    pub fn log_summary(&self, model_dir: &std::path::Path) {
        if self.loaded == 0 {
            tracing::warn!(
                dir = %model_dir.display(),
                skipped = self.skipped.len(),
                first_skipped = ?self.skipped.first(),
                "weight load matched no parameters — the model is running on random init; \
                 the checkpoint's key layout is probably not one this architecture expects"
            );
        } else if !self.skipped.is_empty() {
            tracing::info!(
                dir = %model_dir.display(),
                loaded = self.loaded,
                skipped = self.skipped.len(),
                first_skipped = ?self.skipped.first(),
                "weight load complete with unmatched checkpoint keys"
            );
        } else {
            tracing::debug!(loaded = self.loaded, "weight load complete");
        }
    }
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

/// A pointwise activation function.
///
/// Every activation in `compat::nn` has this shape, so an architecture can
/// resolve its config's activation name once at construction and store the
/// result, instead of re-matching a string on every forward pass.
pub type Activation = fn(&Array) -> Array;

/// Resolve a HuggingFace `ACT2FN` activation name to its implementation.
///
/// The GELU family is the reason this lives in one place. HuggingFace spells
/// **three different functions** with confusingly similar names, and reaching
/// for whichever one looks closest silently costs far more accuracy than a
/// parity test tolerates — the sigmoid approximation is ~1.9e-2 off exact and
/// the tanh approximation ~1.5e-4, against forward passes that agree to ~1e-6:
///
/// | name(s) | reference class | formula |
/// |---|---|---|
/// | `gelu`, `gelu_python` | `GELUActivation` | `0.5·x·(1 + erf(x/√2))` |
/// | `gelu_new`, `gelu_pytorch_tanh`, `gelu_python_tanh`, `gelu_fast`, `gelu_accurate` | `NewGELUActivation` & friends | `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` |
/// | `quick_gelu` | `QuickGELUActivation` | `x·sigmoid(1.702·x)` |
///
/// The four tanh spellings are separate classes upstream but algebraically the
/// same polynomial — `FastGELUActivation`'s `x·0.7978845608·(1 + 0.044715·x²)`
/// expands to `√(2/π)·(x + 0.044715·x³)` — so they collapse to one entry here.
///
/// Returns `None` for names no architecture in this crate needs yet, so callers
/// fail loudly at load time rather than silently substituting a default.
///
/// Every function returned keeps its input's dtype (see [`activate`]), as
/// PyTorch's activations do: a bf16 MLP stays bf16 from the gate to the down
/// projection.
pub fn resolve_activation(name: &str) -> Option<Activation> {
    Some(match name {
        "gelu" | "gelu_python" => gelu_erf,
        "gelu_new" | "gelu_pytorch_tanh" | "gelu_python_tanh" | "gelu_fast" | "gelu_accurate" => {
            gelu_tanh
        }
        "quick_gelu" => gelu_quick,
        "relu" => nn::relu,
        "silu" | "swish" => nn::silu,
        "tanh" => ops::tanh,
        "sigmoid" => nn::sigmoid,
        _ => return None,
    })
}

/// Apply `act` to `x`, keeping `x`'s dtype.
///
/// The GELUs evaluate a half-precision input in f32 and return f32 (see
/// `compat::nn::gelu`). A layer that fed that result on would run the rest of
/// the forward pass, and the backward pass's copy of the activation, in f32
/// against bf16 weights: twice the memory traffic for a result the reference
/// computes in bf16. Rounding back once is what PyTorch does inside its GELU
/// kernel. A no-op for an f32 model.
pub fn activate(act: Activation, x: &Array) -> Array {
    act(x).as_dtype(x.dtype().as_i32())
}

/// Exact (erf) GELU in `x`'s dtype: `ACT2FN["gelu"]`.
pub fn gelu_erf(x: &Array) -> Array {
    activate(nn::gelu_erf, x)
}

/// Tanh-approximation GELU in `x`'s dtype: `ACT2FN["gelu_pytorch_tanh"]`.
pub fn gelu_tanh(x: &Array) -> Array {
    activate(nn::gelu_tanh_approximate, x)
}

/// Sigmoid-approximation GELU in `x`'s dtype: `ACT2FN["quick_gelu"]`.
pub fn gelu_quick(x: &Array) -> Array {
    activate(nn::gelu_fast_approximate, x)
}

/// The most negative value `dtype` holds that a softmax can still subtract a
/// row maximum from without overflowing to `-inf` (`finfo.min` in the
/// reference). A row with no attendable key then stays finite instead of
/// turning into NaN.
fn lowest_finite(dtype: Dtype) -> f32 {
    match dtype {
        Dtype::Float16 => -65504.0,
        // `f32::MIN` rounds up to `-inf` in bf16, whose own largest finite
        // value is a few ulps lower.
        Dtype::Bfloat16 => -3.389_531_4e38,
        _ => f32::MIN,
    }
}

/// Which keys a padding mask lets a query attend to, as a boolean
/// `[batch, 1, 1, keys]`.
///
/// `attention_mask` is `[batch, keys]`, non-zero where a token is real.
pub fn padding_keep(attention_mask: &Array) -> Array {
    let (batch, keys) = (attention_mask.dim(0), attention_mask.dim(1));
    attention_mask
        .as_dtype(Dtype::Int32.as_i32())
        .gt(&Array::from_int(0))
        .reshape(&[batch, 1, 1, keys])
}

/// An additive attention mask from a boolean one that is true where a query
/// may attend: `0` there and the dtype's lowest finite value elsewhere, ready
/// to add to attention scores or hand to `fused_sdpa` as its mask.
///
/// Combine masks as booleans first ([`ops::logical_and`]) and convert once:
/// two additive masks of the lowest finite value sum to `-inf`.
pub fn additive_mask(keep: &Array, dtype: Dtype) -> Array {
    let zero = Array::from_f32(0.0).as_dtype(dtype.as_i32());
    let lowest = Array::from_f32(lowest_finite(dtype)).as_dtype(dtype.as_i32());
    ops::where_fn(keep, &zero, &lowest)
}

/// An additive attention mask from a padding mask, for bidirectional encoders:
/// `[batch, 1, 1, keys]` in `dtype` (see [`padding_keep`], [`additive_mask`]).
pub fn additive_padding_mask(attention_mask: &Array, dtype: Dtype) -> Array {
    additive_mask(&padding_keep(attention_mask), dtype)
}

/// Cast an additive attention mask to the query dtype, if it isn't already.
///
/// MLX's scaled-dot-product attention requires the mask to promote to the
/// output dtype and errors out rather than upcasting, so an f32 mask against a
/// bf16 checkpoint is a hard failure. Every mask builder here produces f32
/// regardless of the model's dtype, so any code path that hands a mask straight
/// to `Array::sdpa_with_mask` has to coerce it first.
///
/// `pmetal_runtime::nn::fused_sdpa` does this internally; this is for the
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Points chosen where the three GELUs disagree most; 7.0 is included as a
    /// saturation check where all three agree.
    const PROBES: [f32; 8] = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 7.0];

    fn eval(f: Activation) -> Vec<f32> {
        let mut out = f(&Array::from_f32_slice(&PROBES, &[PROBES.len() as i32]));
        out.eval();
        out.to_f32_vec(PROBES.len()).expect("to_f32_vec")
    }

    fn assert_resolves_to(name: &str, expected: Activation) {
        let got = resolve_activation(name)
            .unwrap_or_else(|| panic!("resolve_activation({name:?}) returned None"));
        assert_eq!(
            eval(got),
            eval(expected),
            "resolve_activation({name:?}) picked the wrong function"
        );
    }

    /// Every `hidden_act` string a supported checkpoint can carry, mapped to the
    /// function HuggingFace's `ACT2FN` would have picked. Verified against the
    /// `transformers.activations` source, not from recollection — the whole
    /// reason this table exists is that the names are easy to mix up.
    #[test]
    fn activation_names_resolve_to_the_reference_function() {
        assert_resolves_to("gelu", nn::gelu_erf);
        assert_resolves_to("gelu_python", nn::gelu_erf);

        for tanh_spelling in [
            "gelu_new",
            "gelu_pytorch_tanh",
            "gelu_python_tanh",
            "gelu_fast",
            "gelu_accurate",
        ] {
            assert_resolves_to(tanh_spelling, nn::gelu_tanh_approximate);
        }

        assert_resolves_to("quick_gelu", nn::gelu_fast_approximate);
        assert_resolves_to("relu", nn::relu);
        assert_resolves_to("silu", nn::silu);
        assert_resolves_to("swish", nn::silu);
        assert_resolves_to("tanh", ops::tanh);
    }

    /// The assertions above are only meaningful if the three GELUs are actually
    /// distinguishable on the probe points — otherwise a mis-mapping would pass
    /// silently, which is exactly the failure being guarded against.
    #[test]
    fn the_three_gelus_are_distinguishable_on_the_probe_points() {
        let variants = [
            ("gelu", nn::gelu_erf as Activation),
            ("gelu_new", nn::gelu_tanh_approximate),
            ("quick_gelu", nn::gelu_fast_approximate),
        ];
        for (i, (a_name, a)) in variants.iter().enumerate() {
            for (b_name, b) in &variants[i + 1..] {
                let gap = eval(*a)
                    .iter()
                    .zip(eval(*b))
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    gap > 1e-4,
                    "{a_name} and {b_name} agree to {gap:.2e} on the probe \
                     points — the mapping test above can no longer tell them apart"
                );
            }
        }
    }

    /// An unrecognized name must not fall back to a default. Architectures turn
    /// this `None` into a load-time error, so a checkpoint carrying an
    /// activation nothing implements fails loudly instead of running with the
    /// wrong one.
    #[test]
    fn unknown_activations_are_rejected_rather_than_defaulted() {
        assert!(resolve_activation("mish").is_none());
        assert!(resolve_activation("xielu").is_none());
        assert!(resolve_activation("").is_none());
    }
}
