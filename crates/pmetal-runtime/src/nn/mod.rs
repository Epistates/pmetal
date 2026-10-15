//! The operations a model's forward and its training loss are built from, over
//! the runtime's arrays.
//!
//! - [`attention`]: scaled dot-product attention with masks, sinks and logit
//!   softcapping
//! - [`gated_delta`]: the gated delta rule (Qwen 3.5's linear attention), with
//!   a chunkwise parallel form for training
//! - [`loss`]: the LM-head loss without materializing the logits
//! - [`mod@rope`]: rotary embeddings at contiguous or explicit positions
//! - [`moe`]: the expert MLP and the combine step of a mixture-of-experts block
//! - [`training`]: the training-mode flag the model forwards read

pub mod attention;
pub mod gated_delta;
pub mod loss;
pub mod moe;
pub mod rope;
pub mod training;

pub use attention::*;
pub use gated_delta::*;
pub use loss::*;
pub use moe::{Expert, moe_combine};
pub use rope::*;
pub use training::*;

use crate::compat::Array;

/// Logit softcapping, `cap · tanh(x / cap)`: Gemma 2's final logits and Gemma
/// 4's, and the form attention scores take under Gemma 2's attention softcap.
///
/// The result has `x`'s dtype, so a bf16 head stays bf16 (an f32 constant
/// would promote it, and the logits of a 262144-token vocabulary are the
/// largest tensor in a pass). It uses the `tanh` primitive: expanded as
/// `(exp(2x) - 1) / (exp(2x) + 1)` it overflows to `inf / inf = NaN` once
/// `x / cap` passes about 44, and softcapping exists to tame outliers, not to
/// blow up on them.
pub fn softcap(x: &Array, cap: f32) -> Array {
    crate::compat::ops::tanh(&x.div_scalar(cap)).mul_scalar(cap)
}

/// Whether autograd is tracing any of `arrays`, so whatever is computed from
/// them has to stay in MLX's graph.
///
/// A kernel that builds its output outside the graph (from a Metal buffer, or
/// from values read back to the host) hands autograd a constant, and one whose
/// primitive has no VJP (a custom Metal kernel, an op like `tri_inv`) fails or
/// returns NaN under it; either way everything upstream trains on nothing.
/// Each such kernel's dispatcher asks this and takes its graph path when the
/// answer is yes. A frozen model's forward (a distillation teacher, a
/// reference policy) isn't traced even inside `value_and_grad` and keeps the
/// fast kernels.
pub fn any_traced(arrays: &[&Array]) -> bool {
    arrays.iter().any(|a| a.is_tracer())
}

#[cfg(test)]
mod softcap_tests {
    use super::softcap;
    use crate::compat::{Array, Dtype};

    /// Against `cap · tanh(x / cap)` in f64, in the input's dtype, including
    /// inputs far past the point where a hand-expanded tanh overflows.
    #[test]
    fn softcap_matches_f64_and_keeps_the_dtype() {
        let xs = [
            -1.0e4f32, -90.0, -30.0, -1.5, 0.0, 0.25, 7.0, 30.0, 1.3e3, 1.0e4,
        ];
        let cap = 30.0f32;
        let want: Vec<f64> = xs
            .iter()
            .map(|&x| f64::from(cap) * (f64::from(x) / f64::from(cap)).tanh())
            .collect();
        for (dtype, tol) in [(Dtype::Float32, 1e-5), (Dtype::Bfloat16, 0.25)] {
            let x = Array::from_f32_slice(&xs, &[xs.len() as i32]).as_dtype(dtype.as_i32());
            let y = softcap(&x, cap);
            assert_eq!(y.dtype(), dtype, "a {dtype:?} head stays {dtype:?}");
            let got = y
                .as_dtype(Dtype::Float32.as_i32())
                .to_f32_vec(xs.len())
                .unwrap();
            crate::check_unobserved_error().expect("a runtime op threw");
            for ((x, got), want) in xs.iter().zip(&got).zip(&want) {
                assert!(got.is_finite(), "softcap({x}) = {got}");
                assert!(
                    (f64::from(*got) - want).abs() <= tol,
                    "softcap({x}) = {got}, want {want}"
                );
            }
        }
    }
}
