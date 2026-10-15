//! The CQT building blocks (FIR design, resampler, transform) against
//! torchaudio and nnAudio.

mod common;

use common::{Fixture, assert_close, assert_close_slice, assert_runtime_ok, to_vec};
use pmetal_runtime::compat::{Array, Dtype};
use pmetal_vocoder::audio::{Cqt, CqtConfig, Resampler};
use pmetal_vocoder::nn::filter::{cqt_lowpass_filter, firwin2};
use serde_json::Value;

#[test]
fn lowpass_filters_match_scipy_firwin2() {
    let fx = Fixture::load("cqt");
    for name in ["lp_half", "lp_early", "lp_short"] {
        let args: Vec<f64> = serde_json::from_str(fx.meta(name)).unwrap();
        let taps = cqt_lowpass_filter(args[0], args[1] as usize, args[2]).unwrap();
        assert_close_slice(&taps, fx.values(name), 1e-4, 2e-6, name);
    }
}

#[test]
fn firwin2_rejects_unsupported_designs() {
    assert!(
        firwin2(8, &[0.0, 0.5, 1.0], &[1.0, 1.0, 1.0]).is_err(),
        "even length needs zero Nyquist gain"
    );
    assert!(
        firwin2(9, &[0.1, 1.0], &[1.0, 0.0]).is_err(),
        "freq must start at 0"
    );
    assert!(
        firwin2(9, &[0.0, 0.5, 0.5, 1.0], &[1.0, 1.0, 0.0, 0.0]).is_err(),
        "repeated frequencies"
    );
}

#[test]
fn resampler_matches_torchaudio() {
    let fx = Fixture::load("cqt");
    for name in ["rs_up2", "rs_44_48", "rs_down3"] {
        let rates: Vec<i32> = serde_json::from_str(fx.meta(name)).unwrap();
        let resampler = Resampler::new(rates[0], rates[1]).unwrap();
        let x = fx.array(&format!("{name}.x"));
        let y = resampler.forward(&x.reshape(&[1, -1])).unwrap();
        let key = format!("{name}.y");
        let expected_len = *fx.shape(&key).last().unwrap();
        assert_eq!(y.shape(), &[1, expected_len], "{name}: length");
        assert_close(
            &y.reshape(fx.shape(&key)),
            fx.shape(&key),
            fx.values(&key),
            1e-4,
            1e-5,
            name,
        );
    }
    assert_runtime_ok();
}

#[test]
fn resampler_round_trips_a_low_frequency_tone() {
    // 440 Hz survives 24 kHz -> 16 kHz -> 24 kHz away from the edges.
    let n = 2400;
    let tone: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
        .collect();
    let x = Array::from_f32_slice(&tone, &[n]);
    let down = Resampler::new(24000, 16000).unwrap().forward(&x).unwrap();
    assert_eq!(down.dim(0), 1600);
    let back = to_vec(
        &Resampler::new(16000, 24000)
            .unwrap()
            .forward(&down)
            .unwrap(),
    );
    let worst = back[200..n as usize - 200]
        .iter()
        .zip(&tone[200..n as usize - 200])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 5e-3, "round-trip error {worst}");
}

#[test]
fn resampler_rejects_nonsense() {
    assert!(Resampler::new(0, 16000).is_err());
    assert!(Resampler::with_options(16000, 8000, 6, 1.5).is_err());
}

fn cqt_case(fx: &Fixture, name: &str) -> (Cqt, Array) {
    let m: Value = serde_json::from_str(fx.meta(name)).unwrap();
    let geti = |k: &str| m[k].as_i64().unwrap() as i32;
    let cqt = Cqt::new(CqtConfig::new(
        geti("sr"),
        geti("hop_length"),
        geti("n_bins"),
        geti("bins_per_octave"),
    ))
    .unwrap();
    (cqt, fx.array(&format!("{name}.x")))
}

#[test]
fn cqt_matches_nnaudio_with_and_without_early_downsampling_and_a_partial_octave() {
    let fx = Fixture::load("cqt");
    for name in ["cqt_plain", "cqt_early", "cqt_partial"] {
        let (cqt, x) = cqt_case(&fx, name);
        let y = cqt.forward(&x.reshape(&[1, -1])).unwrap();
        // Reference: [batch, bins, frames, 2]; ours: [batch, frames, bins, 2].
        let ours = y.transpose_axes(&[0, 2, 1, 3]);
        let key = format!("{name}.y");
        let scale = fx.values(&key).iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert_eq!(ours.shape(), fx.shape(&key), "{name}: shape");
        assert_close_slice(&to_vec(&ours), fx.values(&key), 1e-3, 1e-4 * scale, name);
    }
    assert_runtime_ok();
}

#[test]
fn cqt_accepts_any_length_when_the_hop_halves_cleanly() {
    let fx = Fixture::load("cqt");
    let (cqt, x) = cqt_case(&fx, "cqt_plain");
    let x = x.reshape(&[1, -1]);
    for cut in [1, 3, 100] {
        let shorter = pmetal_runtime::compat::ops::slice_axis(&x, 1, 0, x.dim(1) - cut);
        let y = cqt.forward(&shorter).unwrap();
        assert_eq!((y.dim(0), y.dim(2), y.dim(3)), (1, 36, 2), "cut of {cut}");
        assert!(y.dim(1) >= 16, "cut of {cut}: {} frames", y.dim(1));
    }
}

#[test]
fn cqt_refuses_a_hop_that_does_not_halve_cleanly_and_a_top_bin_above_nyquist() {
    // Four octaves of hop 100: 100, 50, 25, 12 (the last is not 12.5).
    let cqt = Cqt::new(CqtConfig {
        early_downsample: false,
        ..CqtConfig::new(16000, 100, 48, 12)
    })
    .unwrap();
    let err = cqt
        .forward(&Array::zeros(&[1, 3200], Dtype::Float32))
        .unwrap_err();
    assert!(err.to_string().contains("frame counts"), "{err}");
    assert!(Cqt::new(CqtConfig::new(8000, 128, 100, 12)).is_err());
    assert_eq!(
        Cqt::new(CqtConfig::new(16000, 128, 36, 12))
            .unwrap()
            .lengths()
            .len(),
        36
    );
}
