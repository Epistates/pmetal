//! A released BigVGAN checkpoint against the reference implementation's output.
//!
//! These tests need a downloaded model and are skipped unless
//! `BIGVGAN_CHECKPOINT_DIR` names a directory holding, for each model under
//! test, a subdirectory with the release's `config.json` and
//! `bigvgan_generator.pt` plus `ref.safetensors` (the reference's mel and
//! waveform for a fixed signal). Run them with `--ignored`.

mod common;

use common::{Fixture, assert_runtime_ok, max_abs_diff, to_vec};
use pmetal_runtime::compat::Dtype;
use pmetal_vocoder::BigVGAN;
use std::path::PathBuf;

fn model_dir(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("BIGVGAN_CHECKPOINT_DIR")?;
    let dir = PathBuf::from(root).join(name);
    dir.is_dir().then_some(dir)
}

fn check(name: &str, max_diff: f32, min_snr_db: f32) {
    check_dtype(name, None, max_diff, min_snr_db);
}

/// Synthesise with the model cast to `dtype` (float32 when `None`) and compare
/// with the reference's float32 waveform.
fn check_dtype(name: &str, dtype: Option<Dtype>, max_diff: f32, min_snr_db: f32) {
    let Some(dir) = model_dir(name) else {
        eprintln!("BIGVGAN_CHECKPOINT_DIR/{name} not found; skipping");
        return;
    };
    // `from_dir` loads weights and prepares the fused inference path.
    let mut model = BigVGAN::from_dir(&dir).expect("load the released checkpoint");
    let fx = Fixture::load_path(&dir.join("ref.safetensors"));
    let mut mel = fx.array("mel");
    if let Some(dtype) = dtype {
        model.to_dtype(dtype);
        model.prepare_for_inference();
        mel = mel.as_dtype(dtype);
    }

    let audio = model.forward(&mel).expect("forward");
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

/// The released discriminators (`mpd` / `mrd` entries of the training
/// checkpoint, next to optimizer state) load and score audio like the reference.
#[test]
#[ignore = "needs a downloaded checkpoint (see the module docs)"]
fn bigvgan_base_discriminators_match_reference() {
    use pmetal_runtime::compat::Array;
    use pmetal_vocoder::discriminator::{BigVGANDiscriminator, DiscriminatorConfig};

    let Some(dir) = model_dir("base24") else {
        eprintln!("BIGVGAN_CHECKPOINT_DIR/base24 not found; skipping");
        return;
    };
    let checkpoint = dir.join("bigvgan_discriminator_optimizer.pt");
    if !checkpoint.is_file() {
        eprintln!("{} not found; skipping", checkpoint.display());
        return;
    }
    let mut disc = BigVGANDiscriminator::new(&DiscriminatorConfig::default()).unwrap();
    disc.load_checkpoint(&checkpoint)
        .expect("load the released discriminators");
    let fx = Fixture::load_path(&dir.join("ref_disc.safetensors"));

    let nchw = |x: &Array| x.transpose_axes(&[0, 3, 1, 2]);
    for (name, side, audio) in [("real", "real", "y"), ("fake", "fake", "yhat")] {
        let (mpd, mrd) = disc.forward(&fx.array(audio)).unwrap();
        for (bank, outputs) in [("mpd", &mpd), ("mrd", &mrd)] {
            for (i, out) in outputs.iter().enumerate() {
                let key = format!("{bank}.{side}.logits.{i}");
                let worst = max_abs_diff(&to_vec(&out.logits), fx.values(&key));
                let scale = fx.values(&key).iter().fold(0.0f32, |m, v| m.max(v.abs()));
                eprintln!("{name} {key}: max |diff| {worst:.2e} (scale {scale:.2})");
                assert!(worst <= 1e-3 * scale.max(1.0), "{key}: {worst}");
                for (j, feature) in out.features.iter().enumerate() {
                    let key = format!("{bank}.{side}.fmap.{i}.{j}");
                    let worst = max_abs_diff(&to_vec(&nchw(feature)), fx.values(&key));
                    let scale = fx.values(&key).iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    assert!(
                        worst <= 1e-3 * scale.max(1.0),
                        "{key}: {worst} (scale {scale})"
                    );
                }
            }
        }
    }
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

/// Half precision costs rounding noise, not artefacts: the waveform stays
/// within a few percent of the float32 one.
#[test]
#[ignore = "needs a downloaded checkpoint (see the module docs)"]
fn bigvgan_v2_24khz_in_half_precision_tracks_the_reference() {
    check_dtype("v2_24", Some(Dtype::Float16), 5e-2, 30.0);
    check_dtype("v2_24", Some(Dtype::Bfloat16), 2e-1, 20.0);
}
