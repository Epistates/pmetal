//! The stochastic methods (DARE, DELLA) must be reproducible, and must not
//! drop the same entries in every model.

use pmetal_merge::{DareMerge, MergeMethod, MergeParameters};
use pmetal_runtime::compat::Array;

const N: usize = 8192;

fn values(array: &Array) -> Vec<f32> {
    let out = array.clone().to_f32_vec(N).expect("readable as f32");
    pmetal_runtime::check_unobserved_error().expect("runtime op failed");
    out
}

fn params(weight: f32, density: f32) -> MergeParameters {
    MergeParameters {
        weight: Some(weight.into()),
        density: Some(density.into()),
        ..Default::default()
    }
}

fn ones() -> Array {
    Array::from_f32_slice(&vec![1.0; N], &[N as i32])
}

#[test]
fn dare_without_a_configured_seed_is_reproducible() {
    let base = Array::from_f32_slice(&vec![0.0; N], &[N as i32]);
    let models = [ones()];
    let params = [params(1.0, 0.5)];
    let run = || {
        values(
            &DareMerge::linear()
                .merge(&models, Some(&base), &params, &MergeParameters::default())
                .unwrap(),
        )
    };
    assert_eq!(run(), run(), "the same merge twice must give the same bytes");
}

#[test]
fn dare_models_do_not_share_a_drop_pattern() {
    let base = Array::from_f32_slice(&vec![0.0; N], &[N as i32]);
    // Two models with the same all-ones task vector, summed without
    // normalization: independent masks give 0, 2 (one survivor) or 4 (both),
    // with P(2) = 0.5 at density 0.5. One shared mask gives only 0 and 4.
    let models = [ones(), ones()];
    let params = [params(1.0, 0.5), params(1.0, 0.5)];
    let global = MergeParameters {
        normalize: Some(false),
        ..Default::default()
    };
    let out = values(
        &DareMerge::linear()
            .with_seed(7)
            .merge(&models, Some(&base), &params, &global)
            .unwrap(),
    );
    let one_survivor = out.iter().filter(|&&v| v == 2.0).count() as f32 / N as f32;
    assert!(
        (one_survivor - 0.5).abs() < 0.05,
        "P(exactly one survivor) = {one_survivor}, expected 0.5"
    );
}
