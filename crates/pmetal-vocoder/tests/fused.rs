//! The fused anti-aliased activation kernel against the op-based path and the
//! reference.

mod common;

use common::{Fixture, assert_close, assert_runtime_ok, max_abs_diff, to_vec};
use pmetal_runtime::compat::{Array, Dtype, random};
use pmetal_vocoder::nn::{Activation1d, PeriodicActivation, Snake, SnakeBeta};
use pmetal_vocoder::{BigVGAN, BigVGANConfig};

fn activation(
    snake_beta: bool,
    logscale: bool,
    channels: i32,
    ratio: (i32, i32),
    kernels: (i32, i32),
) -> Activation1d {
    let act = if snake_beta {
        let mut a = SnakeBeta::new(channels, logscale).unwrap();
        let scale = if logscale { 0.8 } else { 0.7 };
        let shift = if logscale { 0.0 } else { 1.0 };
        a.alpha.value = random::uniform(&[channels], Dtype::Float32)
            .multiply(&Array::from_f32(2.0 * scale))
            .add(&Array::from_f32(shift - scale));
        a.beta.value = random::uniform(&[channels], Dtype::Float32)
            .multiply(&Array::from_f32(2.0 * scale))
            .add(&Array::from_f32(shift - scale));
        PeriodicActivation::SnakeBeta(a)
    } else {
        let mut a = Snake::new(channels, logscale).unwrap();
        let (scale, shift) = if logscale { (0.8, 0.0) } else { (0.7, 1.0) };
        a.alpha.value = random::uniform(&[channels], Dtype::Float32)
            .multiply(&Array::from_f32(2.0 * scale))
            .add(&Array::from_f32(shift - scale));
        PeriodicActivation::Snake(a)
    };
    Activation1d::with_resampling(act, ratio.0, ratio.1, kernels.0, kernels.1).unwrap()
}

#[test]
fn fused_kernel_equals_the_op_path_across_shapes_ratios_and_variants() {
    random::seed(42);
    let ratios = [
        ((2, 2), (12, 12)),
        ((3, 3), (18, 18)),
        ((2, 2), (11, 11)),
        ((4, 4), (24, 24)),
        ((1, 1), (6, 6)),
    ];
    for (ratio, kernels) in ratios {
        for (snake_beta, logscale) in [(true, true), (false, true), (true, false), (false, false)] {
            for channels in [1, 5, 24, 96] {
                let mut act = activation(snake_beta, logscale, channels, ratio, kernels);
                for (batch, len) in [(1, 1), (1, 2), (2, 7), (1, 100)] {
                    let x = Array::random_normal(&[batch, len, channels], Dtype::Float32)
                        .multiply(&Array::from_f32(1.5));
                    act.unprepare();
                    let reference = to_vec(&act.forward(&x).unwrap());
                    act.prepare_inference();
                    assert!(act.is_prepared());
                    let fused = act.forward(&x).unwrap();
                    assert_eq!(
                        fused.shape(),
                        &[batch, len, channels][..],
                        "{ratio:?} {kernels:?}"
                    );
                    let diff = max_abs_diff(&to_vec(&fused), &reference);
                    assert!(
                        diff < 2e-5,
                        "ratio {ratio:?} kernels {kernels:?} beta {snake_beta} log {logscale} C {channels} len {len}: diff {diff}"
                    );
                }
            }
        }
    }
    assert_runtime_ok();
}

#[test]
fn fused_kernel_matches_the_reference_implementation() {
    let fx = Fixture::load("activation");
    let x = fx.array("x").transpose_axes(&[0, 2, 1]);
    let mut snake = SnakeBeta::new(5, true).unwrap();
    snake.alpha.value = fx.array("act1d.alpha");
    snake.beta.value = fx.array("act1d.beta");
    let mut act = Activation1d::new(PeriodicActivation::SnakeBeta(snake)).unwrap();
    act.prepare_inference();
    let y = act.forward(&x).unwrap().transpose_axes(&[0, 2, 1]);
    assert_close(
        &y,
        fx.shape("act1d.y"),
        fx.values("act1d.y"),
        1e-5,
        1e-5,
        "fused activation1d",
    );
    assert_runtime_ok();
}

#[test]
fn fused_kernel_keeps_half_precision_input_dtype_and_accuracy() {
    random::seed(7);
    let mut act = activation(true, true, 48, (2, 2), (12, 12));
    let x = Array::random_normal(&[1, 200, 48], Dtype::Float32);
    let reference = to_vec(&act.forward(&x).unwrap());
    act.prepare_inference();
    for dtype in [Dtype::Float16, Dtype::Bfloat16] {
        let y = act.forward(&x.as_dtype(dtype)).unwrap();
        assert_eq!(y.dtype(), dtype);
        let got = to_vec(&y);
        let worst = got
            .iter()
            .zip(&reference)
            .map(|(g, r)| (g - r).abs() / (1.0 + r.abs()))
            .fold(0.0f32, f32::max);
        let bound = if dtype == Dtype::Float16 { 3e-3 } else { 3e-2 };
        assert!(worst < bound, "{dtype:?}: worst relative error {worst}");
    }
}

#[test]
fn a_prepared_activation_rejects_the_wrong_channel_count() {
    let mut act = activation(true, true, 8, (2, 2), (12, 12));
    act.prepare_inference();
    // The kernel indexes per-channel constants by the input's channel count, so
    // a mismatch must be an error, not an out-of-bounds read.
    assert!(
        act.forward(&Array::zeros(&[1, 10, 9], Dtype::Float32))
            .is_err()
    );
    assert!(
        act.forward(&Array::zeros(&[10, 8], Dtype::Float32))
            .is_err()
    );
}

#[test]
fn changing_the_parameters_drops_back_to_the_op_path() {
    let mut act = activation(true, true, 8, (2, 2), (12, 12));
    act.prepare_inference();
    act.cast(Dtype::Float32);
    assert!(!act.is_prepared());
}

fn tiny_generator(name: &str) -> (BigVGAN, Fixture) {
    let fx = Fixture::load(name);
    let config = BigVGANConfig::from_json(fx.meta("config")).unwrap();
    let mut model = BigVGAN::new(config).unwrap();
    model
        .load_state_dict(fx.state_dict_except(&["mel", "audio"]))
        .unwrap();
    (model, fx)
}

#[test]
fn a_prepared_generator_still_matches_the_reference() {
    for name in ["generator_v2_tiny", "generator_v1_tiny"] {
        let (mut model, fx) = tiny_generator(name);
        model.prepare_for_inference();
        assert!(
            model
                .resblocks
                .iter()
                .all(|b| b.activations.iter().all(Activation1d::is_prepared))
        );
        let audio = model.forward(&fx.array("mel")).unwrap();
        assert_close(
            &audio,
            fx.shape("audio"),
            fx.values("audio"),
            1e-4,
            1e-5,
            name,
        );
    }
    assert_runtime_ok();
}

#[test]
fn streaming_a_prepared_generator_equals_one_shot_synthesis() {
    let (mut model, _) = tiny_generator("generator_v2_tiny");
    model.prepare_for_inference();
    random::seed(3);
    let mel = Array::random_normal(&[1, model.config.num_mels, 50], Dtype::Float32);
    let full = to_vec(&model.forward(&mel).unwrap());
    let chunked = to_vec(&model.synthesize_chunked(&mel, 6).unwrap());
    assert!(max_abs_diff(&chunked, &full) < 1e-5);
}
