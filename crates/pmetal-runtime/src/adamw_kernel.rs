//! The fused AdamW update: one Metal dispatch per parameter.
//!
//! The op-based update in [`crate::optimizer::AdamW`] builds about a dozen
//! elementwise nodes per parameter, each reading and writing the whole tensor,
//! so an optimizer step is bound by far more memory traffic than the update
//! needs. This kernel reads the weight, the gradient and both moments once and
//! writes the weight and both moments once.
//!
//! It computes exactly what the op-based path computes, element by element and
//! in the same order, in f32 whatever the gradient's dtype:
//!
//! ```text
//! w = w·decay
//! m = m + (1−β₁)·(g − m)
//! v = β₂·v + (1−β₂)·g²
//! w = w − step_size · m / (√v / bias_correction2_sqrt + ε)
//! ```
//!
//! Fused multiply-add contraction is off, so the products and sums round as the
//! separate ops do, and the square root and the divisions are the precise ones:
//! the kernel and the reference agree to the last bit, not to a tolerance.
//!
//! The kernel returns the f32 weight (the master copy, for a half-precision
//! parameter) and both moments. Rounding the weight to the parameter's dtype
//! stays with the caller ([`crate::optimizer::MasterWeight::commit`]), which is
//! the same cast the op-based path uses.

use crate::compat::Dtype;
use crate::error::RuntimeResult;
use crate::optimizer::AdamWStepScalars;
use crate::{InlineArray, MetalKernel};
use std::sync::OnceLock;

/// Threads per threadgroup. The update is memory-bound; this is only enough
/// threads to keep the memory system busy.
const THREADGROUP: i32 = 256;

const HEADER: &str = r#"
#pragma METAL fp contract(off)
"#;

const SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    if (i >= w_shape[0]) {
        return;
    }
    float decay = hp[0];
    float one_minus_beta1 = hp[1];
    float beta2 = hp[2];
    float one_minus_beta2 = hp[3];
    float step_size = hp[4];
    float bias_correction2_sqrt = hp[5];
    float eps = hp[6];

    float gi = static_cast<float>(g[i]);
    float mi = m[i] + one_minus_beta1 * (gi - m[i]);
    float vi = v[i] * beta2 + (gi * gi) * one_minus_beta2;
    float denom = metal::precise::divide(metal::precise::sqrt(vi), bias_correction2_sqrt) + eps;
    float wi = w[i] * decay;
    w_out[i] = wi - metal::precise::divide(mi, denom) * step_size;
    m_out[i] = mi;
    v_out[i] = vi;
"#;

fn kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        MetalKernel::new(
            "pmetal_adamw_fused",
            &["w", "g", "m", "v", "hp"],
            &["w_out", "m_out", "v_out"],
            SOURCE,
            HEADER,
        )
        .expect("the fused AdamW kernel's source has no NUL byte")
    })
}

/// The step's scalars as the kernel reads them (`hp`), in the order of
/// [`AdamWStepScalars`]' fields.
pub(crate) fn scalars_array(s: &AdamWStepScalars) -> InlineArray {
    InlineArray::from_f32_slice(
        &[
            s.decay,
            s.one_minus_beta1,
            s.beta2,
            s.one_minus_beta2,
            s.step_size,
            s.bias_correction2_sqrt,
            s.eps,
        ],
        &[7],
    )
}

/// The elements a single dispatch can cover.
const MAX_ELEMENTS: usize = i32::MAX as usize;

/// Whether [`update`] takes a parameter of `n` elements: the grid is indexed
/// in `i32`.
pub(crate) fn fits(n: usize) -> bool {
    n > 0 && n <= MAX_ELEMENTS
}

/// One AdamW update of `weight` (f32) by `gradient`, with moments `m` and `v`
/// (f32, shaped like `weight`). Returns `(weight, m, v)`, lazily, each in
/// `weight`'s shape.
///
/// The caller has checked [`fits`] and that `gradient` has `weight`'s element
/// count; `hp` is [`scalars_array`].
pub(crate) fn update(
    weight: &InlineArray,
    gradient: &InlineArray,
    m: &InlineArray,
    v: &InlineArray,
    hp: &InlineArray,
) -> RuntimeResult<(InlineArray, InlineArray, InlineArray)> {
    let shape = weight.shape().to_vec();
    let n = i32::try_from(weight.size()).expect("`fits` was checked");
    let flat = |a: &InlineArray| a.reshape(&[n]);
    let mut out = kernel()
        .call(
            &[&flat(weight), &flat(gradient), &flat(m), &flat(v), hp],
            &[(&[n][..], Dtype::Float32); 3],
            [n, 1, 1],
            [THREADGROUP.min(n), 1, 1],
            &[],
        )?
        .into_iter();
    let (w, m, v) = (
        out.next().expect("three outputs"),
        out.next().expect("three outputs"),
        out.next().expect("three outputs"),
    );
    Ok((w.reshape(&shape), m.reshape(&shape), v.reshape(&shape)))
}
