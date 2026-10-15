//! The gradient of MLX's fused attention against the unfused path.
//!
//! Under autograd, `fast::scaled_dot_product_attention` with no array mask
//! (causal or none) runs a fused Metal forward that also outputs the
//! logsumexp, and a fused Metal backward reads it. With an array mask it
//! falls back to the unfused ops. A causal mask written out as an additive
//! array therefore gives the reference: the two must agree on the loss and
//! on the gradients of the queries, keys and values.
//!
//! Short queries (8 positions or fewer) take MLX's vector kernels in the
//! forward, which output no logsumexp, and the fused backward has no
//! attention sinks, so both are checked separately: the bridge has to keep
//! them off the fused backward (`bridge_sdpa` in `cpp/bridge_internal.h`).

use pmetal_runtime::InlineArray;
use pmetal_runtime::dtype::F32;
use pmetal_runtime::inline_array::{checkpoint_apply, random_seed, value_and_grad};

const HEADS: i32 = 2;
const HEAD_DIM: i32 = 64;

/// Additive causal mask: 0 on and below the diagonal, -1e9 above it.
fn causal_mask(len: i32) -> InlineArray {
    InlineArray::ones(&[len, len], F32)
        .tril(0)
        .sub_scalar(1.0)
        .mul_scalar(1e9)
}

/// `(loss, [dq, dk, dv])` of `sum(attention(q, k, v) * w)`, causal, with
/// per-head sinks when `sinks` is set. `fused` passes the mask as `"causal"`,
/// otherwise as an additive array.
fn loss_and_grads(len: i32, sinks: bool, fused: bool) -> (f32, Vec<InlineArray>) {
    random_seed(7);
    let shape = [1, HEADS, len, HEAD_DIM];
    let q = InlineArray::random_normal(&shape, F32);
    let k = InlineArray::random_normal(&shape, F32);
    let v = InlineArray::random_normal(&shape, F32);
    let w = InlineArray::random_normal(&shape, F32);
    let sink_logits = InlineArray::random_normal(&[HEADS], F32);
    let mask = causal_mask(len);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let (loss, grads) = value_and_grad(
        |a| {
            let (q, k, v, w, mask, sinks_arr) = (&a[0], &a[1], &a[2], &a[3], &a[4], &a[5]);
            let out = match (sinks, fused) {
                (false, true) => q.sdpa(k, v, scale, "causal"),
                (false, false) => q.sdpa_with_mask(k, v, scale, Some(mask)),
                (true, true) => q.sdpa_with_sinks(k, v, scale, "causal", None, sinks_arr),
                (true, false) => q.sdpa_with_sinks(k, v, scale, "", Some(mask), sinks_arr),
            };
            out.multiply(w).sum_all()
        },
        &[q, k, v],
        &[w, mask, sink_logits],
    );
    let loss = loss.item_f32();
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| {
        panic!("attention gradient at length {len} (sinks: {sinks}, fused: {fused}): {e}")
    });
    (loss, grads)
}

fn assert_fused_matches_unfused(len: i32, sinks: bool) {
    let (fused_loss, fused) = loss_and_grads(len, sinks, true);
    let (ref_loss, reference) = loss_and_grads(len, sinks, false);
    assert!(
        (fused_loss - ref_loss).abs() <= 1e-3 * ref_loss.abs().max(1.0),
        "length {len}: fused loss {fused_loss} vs unfused {ref_loss}"
    );
    for (name, (f, r)) in ["dq", "dk", "dv"].iter().zip(fused.iter().zip(&reference)) {
        let diff = f.subtract(r).abs().max(None).item_f32();
        let scale = r.abs().max(None).item_f32().max(1.0);
        assert!(
            diff <= 1e-3 * scale,
            "length {len}: {name} differs by {diff} (largest {scale})"
        );
    }
}

#[test]
fn fused_attention_gradient_matches_unfused_on_full_kernel() {
    assert_fused_matches_unfused(32, false);
}

#[test]
fn fused_attention_gradient_matches_unfused_on_short_queries() {
    for len in [1, 4, 8] {
        assert_fused_matches_unfused(len, false);
    }
}

/// A checkpointed layer rebuilds its forward inside the backward pass, so
/// the short-query attention it holds is built twice under the trace.
#[test]
fn short_query_attention_trains_under_checkpoint() {
    random_seed(7);
    let shape = [1, HEADS, 4, HEAD_DIM];
    let q = InlineArray::random_normal(&shape, F32);
    let k = InlineArray::random_normal(&shape, F32);
    let v = InlineArray::random_normal(&shape, F32);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let (loss, grads) = value_and_grad(
        |a| {
            let out = checkpoint_apply(&[a[0].clone(), a[1].clone(), a[2].clone()], move |x| {
                vec![x[0].sdpa(&x[1], &x[2], scale, "causal")]
            });
            out[0].sum_all()
        },
        &[q, k, v],
        &[],
    );
    assert!(loss.item_f32().is_finite());
    pmetal_runtime::check_unobserved_error().expect("checkpointed short-query attention");
    for g in &grads {
        assert!(g.abs().max(None).item_f32().is_finite());
    }
}

#[test]
fn attention_gradient_with_sinks_matches_unfused() {
    for len in [4, 32] {
        assert_fused_matches_unfused(len, true);
    }
}
