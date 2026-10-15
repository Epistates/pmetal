//! The merge methods against the formulas of their papers, on tensors small
//! enough to work out by hand. Public API only.

use pmetal_merge::{
    BreadcrumbsMerge, DellaMerge, MergeMethod, MergeParameters, ModelStockMerge, MultiSlerpMerge,
    SlerpMerge, TaskArithmeticMerge,
};
use pmetal_runtime::compat::Array;

fn arr(values: &[f32]) -> Array {
    Array::from_f32_slice(values, &[values.len() as i32])
}

fn read(array: &Array) -> Vec<f32> {
    let n: usize = array.shape().iter().map(|&d| d as usize).product();
    let out = array.clone().to_f32_vec(n).expect("readable as f32");
    pmetal_runtime::check_unobserved_error().expect("runtime op failed");
    out
}

fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "index {i}: got {g}, want {w}\n got:  {got:?}\n want: {want:?}"
        );
    }
}

fn weight_density(weight: f32, density: f32) -> MergeParameters {
    MergeParameters {
        weight: Some(weight.into()),
        density: Some(density.into()),
        ..Default::default()
    }
}

/// Model Stock (Jang et al. 2024): w_H = t w_avg + (1 - t) w_0 with
/// t = N cos / (1 + (N - 1) cos). Orthogonal task vectors give t = 0.
#[test]
fn model_stock_orthogonal_task_vectors_stay_at_the_pretrained_weights() {
    let base = arr(&[0.0, 0.0]);
    let out = ModelStockMerge::new()
        .merge(
            &[arr(&[1.0, 0.0]), arr(&[0.0, 1.0])],
            Some(&base),
            &[],
            &MergeParameters::default(),
        )
        .unwrap();
    assert_close(&read(&out), &[0.0, 0.0], 1e-6);
}

#[test]
fn model_stock_two_models_at_forty_five_degrees() {
    // tau1 = [2, 0], tau2 = [1, 1]: cos = 1/sqrt 2, t = 2c/(1+c) = 0.828427,
    // average task vector [1.5, 0.5].
    let base = arr(&[0.0, 0.0]);
    let out = ModelStockMerge::new()
        .merge(
            &[arr(&[2.0, 0.0]), arr(&[1.0, 1.0])],
            Some(&base),
            &[],
            &MergeParameters::default(),
        )
        .unwrap();
    assert_close(&read(&out), &[1.242641, 0.414214], 1e-5);
}

/// DELLA's MAGPRUNE (Deep et al. 2024) keeps large entries *more* often: the
/// keep probability rises with magnitude rank. With density 0.5 and epsilon
/// 0.4 the smallest entry of a row is kept with probability 0.1 and the
/// largest with 0.9, so the top quarter survives far more often than the
/// bottom quarter.
#[test]
fn della_keeps_large_magnitudes_more_often_than_small_ones() {
    let n = 4000;
    let task: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let base = Array::from_f32_slice(&vec![0.0; n], &[1, n as i32]);
    let model = Array::from_f32_slice(&task, &[1, n as i32]);
    let global = MergeParameters {
        epsilon: Some(0.4),
        normalize: Some(false),
        ..Default::default()
    };
    let out = DellaMerge::linear()
        .merge(&[model], Some(&base), &[weight_density(1.0, 0.5)], &global)
        .unwrap();
    let out = read(&out);
    let rate = |range: std::ops::Range<usize>| {
        out[range.clone()].iter().filter(|&&v| v != 0.0).count() as f32 / range.len() as f32
    };
    let (bottom, top) = (rate(0..n / 4), rate(3 * n / 4..n));
    assert!(
        bottom < 0.35,
        "smallest quarter kept {bottom}, expected about 0.2"
    );
    assert!(top > 0.65, "largest quarter kept {top}, expected about 0.8");
}

/// Breadcrumbs (Davari and Belilovsky 2024): drop the smallest 1 - density
/// and the largest gamma of each layer's task vector; no rescaling.
#[test]
fn breadcrumbs_uses_density_and_gamma_and_does_not_rescale() {
    let task = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0];
    let base = arr(&[0.0; 10]);
    let global = MergeParameters {
        gamma: Some(0.1),
        normalize: Some(false),
        ..Default::default()
    };
    let out = BreadcrumbsMerge::new()
        .merge(
            &[arr(&task)],
            Some(&base),
            &[weight_density(1.0, 0.8)],
            &global,
        )
        .unwrap();
    assert_close(
        &read(&out),
        &[0.0, 0.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, 0.0],
        1e-6,
    );
}

/// Task arithmetic (Ilharco et al. 2023): theta = theta_0 + lambda sum w_i tau_i.
#[test]
fn task_arithmetic_sums_the_weighted_task_vectors() {
    let base = arr(&[1.0, 1.0]);
    let models = [arr(&[2.0, 3.0]), arr(&[4.0, 0.0])];
    let params = [
        MergeParameters {
            weight: Some(0.5_f32.into()),
            ..Default::default()
        },
        MergeParameters {
            weight: Some(2.0_f32.into()),
            ..Default::default()
        },
    ];
    let global = MergeParameters {
        lambda: Some(0.5_f32.into()),
        ..Default::default()
    };
    let out = TaskArithmeticMerge::new()
        .merge(&models, Some(&base), &params, &global)
        .unwrap();
    // tau = [1, 2] and [3, -1]: 0.5 * (0.5 * [1, 2] + 2 * [3, -1]) = [3.25, -0.5]
    assert_close(&read(&out), &[1.0 + 3.25, 1.0 - 0.5], 1e-6);
}

/// SLERP (Shoemake 1985): between orthogonal unit vectors at t = 0.25 the
/// weights are sin((1-t) pi/2) and sin(t pi/2).
#[test]
fn slerp_interpolates_along_the_arc() {
    let (a, b) = (arr(&[1.0, 0.0]), arr(&[0.0, 1.0]));
    let theta = std::f32::consts::FRAC_PI_2;
    let t = 0.25_f32;
    let want = [((1.0 - t) * theta).sin(), (t * theta).sin()];
    let out = SlerpMerge::slerp(&a, &b, t).unwrap();
    assert_close(&read(&out), &want, 1e-5);
}

/// Multi-SLERP of two models with weights (1 - t, t) is the SLERP point at t,
/// not the normalized weighted mean: orthogonal unit vectors at t = 0.25 sit
/// at 22.5 degrees, where the normalized mean would give 18.4.
#[test]
fn multislerp_of_two_models_is_the_slerp_point() {
    let (a, b) = (arr(&[1.0, 0.0]), arr(&[0.0, 1.0]));
    let params = [
        MergeParameters {
            weight: Some(0.75_f32.into()),
            ..Default::default()
        },
        MergeParameters {
            weight: Some(0.25_f32.into()),
            ..Default::default()
        },
    ];
    let out = MultiSlerpMerge::new()
        .merge(&[a, b], None, &params, &MergeParameters::default())
        .unwrap();
    let angle = std::f32::consts::FRAC_PI_8;
    assert_close(&read(&out), &[angle.cos(), angle.sin()], 1e-5);
}
