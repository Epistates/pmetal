//! A released BigVGAN checkpoint against the reference implementation's output.
//!
//! These tests need a downloaded model and are skipped unless
//! `BIGVGAN_CHECKPOINT_DIR` names a directory holding, for each model under
//! test, a subdirectory with the release's `config.json` and
//! `bigvgan_generator.pt` plus `ref.safetensors` (the reference's mel and
//! waveform for a fixed signal). Run them with `--ignored`.

mod common;

use common::{Fixture, assert_runtime_ok, max_abs_diff, to_vec};
use pmetal_vocoder::BigVGAN;
use std::path::PathBuf;

fn model_dir(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("BIGVGAN_CHECKPOINT_DIR")?;
    let dir = PathBuf::from(root).join(name);
    dir.is_dir().then_some(dir)
}

fn check(name: &str, max_diff: f32, min_snr_db: f32) {
    let Some(dir) = model_dir(name) else {
        eprintln!("BIGVGAN_CHECKPOINT_DIR/{name} not found; skipping");
        return;
    };
    let model = BigVGAN::from_dir(&dir).expect("load the released checkpoint");
    let fx = Fixture::load_path(&dir.join("ref.safetensors"));

    let audio = model.forward(&fx.array("mel")).expect("forward");
    assert_eq!(audio.shape(), fx.shape("audio"));
    let got = to_vec(&audio);
    let expected = fx.values("audio");
    let diff = max_abs_diff(&got, expected);
    let signal: f64 = expected.iter().map(|&v| f64::from(v).powi(2)).sum();
    let noise: f64 = got
        .iter()
        .zip(expected)
        .map(|(g, e)| f64::from(g - e).powi(2))
        .sum();
    let snr = 10.0 * (signal / noise.max(1e-30)).log10();
    eprintln!("{name}: max |diff| {diff:.3e}, SNR {snr:.1} dB");
    assert!(diff <= max_diff, "{name}: max |diff| {diff}");
    assert!(snr >= f64::from(min_snr_db), "{name}: SNR {snr} dB");
    assert_runtime_ok();
}

#[test]
#[ignore = "needs a downloaded checkpoint (see the module docs)"]
fn bigvgan_base_24khz_matches_reference() {
    check("base24", 2e-4, 70.0);
}

#[test]
#[ignore = "needs a downloaded checkpoint (see the module docs)"]
fn bigvgan_v2_24khz_matches_reference() {
    check("v2_24", 2e-4, 70.0);
}
