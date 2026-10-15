//! Snake / SnakeBeta / anti-aliased activation against the reference.

mod common;

use common::{Fixture, assert_close, assert_close_slice, assert_runtime_ok, to_vec};
use pmetal_runtime::compat::{Array, Dtype};
use pmetal_vocoder::nn::filter::kaiser_sinc_filter;
use pmetal_vocoder::nn::{
    Activation1d, DownSample1d, PeriodicActivation, Snake, SnakeBeta, UpSample1d,
};

const C: i32 = 5;

/// Fixtures hold PyTorch `[batch, channels, time]`; the layers are channels-last.
fn nlc(x: &Array) -> Array {
    x.transpose_axes(&[0, 2, 1])
}

fn ncl(x: &Array) -> Array {
    x.transpose_axes(&[0, 2, 1])
}

#[test]
fn kaiser_sinc_filters_match_reference() {
    let fx = Fixture::load("activation");
    for (ratio, ks) in [(2, 12usize), (2, 11), (3, 18), (4, 24), (1, 6)] {
        let taps = kaiser_sinc_filter(0.5 / f64::from(ratio), 0.6 / f64::from(ratio), ks).unwrap();
        let key = format!("filter_r{ratio}_k{ks}");
        assert_close_slice(&taps, fx.values(&key), 1e-5, 1e-6, &key);
    }
}

#[test]
fn snake_matches_reference() {
    let fx = Fixture::load("activation");
    let x = nlc(&fx.array("x"));
    for (tag, logscale) in [("snake_log", true), ("snake_lin", false)] {
        let mut act = Snake::new(C, logscale).unwrap();
        act.alpha.value = fx.array(&format!("{tag}.alpha"));
        let y = ncl(&act.forward(&x).unwrap());
        assert_close(
            &y,
            fx.shape(&format!("{tag}.y")),
            fx.values(&format!("{tag}.y")),
            1e-5,
            1e-5,
            tag,
        );
    }
}

#[test]
fn snakebeta_matches_reference() {
    let fx = Fixture::load("activation");
    let x = nlc(&fx.array("x"));
    for (tag, logscale) in [("snakebeta_log", true), ("snakebeta_lin", false)] {
        let mut act = SnakeBeta::new(C, logscale).unwrap();
        act.alpha.value = fx.array(&format!("{tag}.alpha"));
        act.beta.value = fx.array(&format!("{tag}.beta"));
        let y = ncl(&act.forward(&x).unwrap());
        assert_close(
            &y,
            fx.shape(&format!("{tag}.y")),
            fx.values(&format!("{tag}.y")),
            1e-5,
            1e-5,
            tag,
        );
    }
}

/// The reference divides by `beta + 1e-9`, so a zero beta is finite (a large
/// but finite correction) rather than `inf`/NaN.
#[test]
fn snakebeta_zero_beta_is_finite_like_the_reference() {
    let fx = Fixture::load("activation");
    let x = nlc(&fx.array("x"));
    let mut act = SnakeBeta::new(C, false).unwrap();
    act.alpha.value = Array::full(&[C], 1.5, Dtype::Float32);
    act.beta.value = Array::zeros(&[C], Dtype::Float32);
    let y = ncl(&act.forward(&x).unwrap());
    assert_close(
        &y,
        fx.shape("snakebeta_zero_beta.y"),
        fx.values("snakebeta_zero_beta.y"),
        1e-4,
        1e-3,
        "zero beta",
    );
}

#[test]
fn upsample_matches_reference() {
    let fx = Fixture::load("activation");
    let x = nlc(&fx.array("x"));
    // Ratio 1 is not an identity: the reference still low-pass filters.
    for (ratio, ks) in [(2, 12), (2, 11), (3, 18), (1, 6)] {
        let up = UpSample1d::new(ratio, ks, C).unwrap();
        let y = ncl(&up.forward(&x).unwrap());
        let key = format!("up_r{ratio}_k{ks}.y");
        assert_close(&y, fx.shape(&key), fx.values(&key), 1e-5, 1e-5, &key);
    }
}

#[test]
fn downsample_matches_reference() {
    let fx = Fixture::load("activation");
    let x = nlc(&fx.array("x"));
    for (ratio, ks) in [(2, 12), (2, 11), (3, 18), (1, 6)] {
        let down = DownSample1d::new(ratio, ks, C).unwrap();
        let y = ncl(&down.forward(&x).unwrap());
        let key = format!("down_r{ratio}_k{ks}.y");
        assert_close(&y, fx.shape(&key), fx.values(&key), 1e-5, 1e-5, &key);
    }
}

#[test]
fn activation1d_matches_reference() {
    let fx = Fixture::load("activation");
    let x = nlc(&fx.array("x"));
    let mut snake = SnakeBeta::new(C, true).unwrap();
    snake.alpha.value = fx.array("act1d.alpha");
    snake.beta.value = fx.array("act1d.beta");
    let act = Activation1d::new(PeriodicActivation::SnakeBeta(snake)).unwrap();
    let y = ncl(&act.forward(&x).unwrap());
    assert_close(
        &y,
        fx.shape("act1d.y"),
        fx.values("act1d.y"),
        1e-5,
        1e-5,
        "activation1d",
    );
}

/// A constant input must come out constant: the filters have unit DC gain and
/// replicate padding adds no edge artefacts.
#[test]
fn resamplers_preserve_a_constant_signal() {
    let x = Array::full(&[1, 40, 3], 0.75, Dtype::Float32);
    let up = UpSample1d::new(2, 12, 3).unwrap().forward(&x).unwrap();
    assert_eq!(up.shape(), &[1, 80, 3]);
    for v in to_vec(&up) {
        assert!((v - 0.75).abs() < 1e-5, "upsampled constant drifted: {v}");
    }
    let down = DownSample1d::new(2, 12, 3).unwrap().forward(&x).unwrap();
    assert_eq!(down.shape(), &[1, 20, 3]);
    for v in to_vec(&down) {
        assert!((v - 0.75).abs() < 1e-5, "downsampled constant drifted: {v}");
    }
}

/// Half-precision inputs must not lose the periodic term: the activation is
/// evaluated in float32 and returned in the input dtype.
#[test]
fn snake_keeps_the_input_dtype_and_stays_accurate_in_half_precision() {
    let fx = Fixture::load("activation");
    let x32 = nlc(&fx.array("x"));
    let mut act = SnakeBeta::new(C, true).unwrap();
    act.alpha.value = fx.array("snakebeta_log.alpha");
    act.beta.value = fx.array("snakebeta_log.beta");
    let reference = to_vec(&act.forward(&x32).unwrap());

    let x16 = x32.as_dtype(Dtype::Float16);
    let y16 = act.forward(&x16).unwrap();
    assert_eq!(y16.dtype(), Dtype::Float16);
    let got = to_vec(&y16);
    // Only the half-precision rounding of the input and output remains.
    let x_rounded = to_vec(&x16);
    assert_eq!(got.len(), reference.len());
    let worst = got
        .iter()
        .zip(&reference)
        .zip(&x_rounded)
        .map(|((g, r), _)| (g - r).abs() / (1.0 + r.abs()))
        .fold(0.0f32, f32::max);
    assert!(worst < 3e-2, "half-precision snake error {worst}");
    assert_runtime_ok();
}

/// Gradients of `sum(f(x) * w)` with respect to `x` and the parameters, with
/// `f` built from the given parameter values.
fn weighted_grads(
    fx: &Fixture,
    params: &[Array],
    f: impl Fn(&Array, &[Array]) -> Array,
) -> Vec<Array> {
    let w = nlc(&fx.array("grad_w"));
    let mut all = vec![nlc(&fx.array("x"))];
    all.extend(params.iter().cloned());
    let (loss, grads) = pmetal_runtime::compat::nn::value_and_grad_explicit(
        |p| f(&p[0], &p[1..]).multiply(&w).sum_all(),
        &all,
        &[],
    )
    .unwrap();
    loss.eval();
    grads
}

#[test]
fn snake_gradients_match_reference() {
    let fx = Fixture::load("activation");
    for (tag, logscale, has_beta) in [
        ("grad_snakebeta_log", true, true),
        ("grad_snake_lin", false, false),
    ] {
        let mut params = vec![fx.array(&format!("{tag}.alpha"))];
        if has_beta {
            params.push(fx.array(&format!("{tag}.beta")));
        }
        let grads = weighted_grads(&fx, &params, |x, p| {
            if has_beta {
                let mut act = SnakeBeta::new(C, logscale).unwrap();
                act.alpha.value = p[0].clone();
                act.beta.value = p[1].clone();
                act.forward(x).unwrap()
            } else {
                let mut act = Snake::new(C, logscale).unwrap();
                act.alpha.value = p[0].clone();
                act.forward(x).unwrap()
            }
        });
        assert_close(
            &ncl(&grads[0]),
            fx.shape(&format!("{tag}.dx")),
            fx.values(&format!("{tag}.dx")),
            1e-4,
            1e-4,
            &format!("{tag}.dx"),
        );
        assert_close_slice(
            &to_vec(&grads[1]),
            fx.values(&format!("{tag}.dalpha")),
            1e-4,
            1e-4,
            &format!("{tag}.dalpha"),
        );
        if has_beta {
            assert_close_slice(
                &to_vec(&grads[2]),
                fx.values(&format!("{tag}.dbeta")),
                1e-4,
                1e-4,
                &format!("{tag}.dbeta"),
            );
        }
    }
    assert_runtime_ok();
}

#[test]
fn anti_aliased_activation_gradients_match_reference() {
    let fx = Fixture::load("activation");
    let params = [fx.array("grad_act1d.alpha"), fx.array("grad_act1d.beta")];
    let grads = weighted_grads(&fx, &params, |x, p| {
        let mut snake = SnakeBeta::new(C, true).unwrap();
        snake.alpha.value = p[0].clone();
        snake.beta.value = p[1].clone();
        Activation1d::new(PeriodicActivation::SnakeBeta(snake))
            .unwrap()
            .forward(x)
            .unwrap()
    });
    assert_close(
        &ncl(&grads[0]),
        fx.shape("grad_act1d.dx"),
        fx.values("grad_act1d.dx"),
        1e-4,
        1e-4,
        "dx",
    );
    assert_close_slice(
        &to_vec(&grads[1]),
        fx.values("grad_act1d.dalpha"),
        1e-4,
        1e-4,
        "dalpha",
    );
    assert_close_slice(
        &to_vec(&grads[2]),
        fx.values("grad_act1d.dbeta"),
        1e-4,
        1e-4,
        "dbeta",
    );
    assert_runtime_ok();
}
