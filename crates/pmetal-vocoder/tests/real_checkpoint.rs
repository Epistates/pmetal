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
    check_discriminators("base24");
}

/// BigVGAN-v2's discriminators: the MPD and the multi-scale sub-band CQT
/// discriminator, including its derived CQT / resampler buffers.
#[test]
#[ignore = "needs a downloaded checkpoint (see the module docs)"]
fn bigvgan_v2_discriminators_match_reference() {
    check_discriminators("v2_24");
}

fn check_discriminators(name: &str) {
    use pmetal_runtime::compat::Array;
    use pmetal_vocoder::discriminator::{BigVGANDiscriminator, DiscriminatorConfig};

    let Some(dir) = model_dir(name) else {
        eprintln!("BIGVGAN_CHECKPOINT_DIR/{name} not found; skipping");
        return;
    };
    let checkpoint = dir.join("bigvgan_discriminator_optimizer.pt");
    if !checkpoint.is_file() {
        eprintln!("{} not found; skipping", checkpoint.display());
        return;
    }
    let config = DiscriminatorConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("config.json"),
    )
    .expect("discriminator settings");
    let mut disc = BigVGANDiscriminator::new(&config).unwrap();
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

/// Fine-tune a released model with its released discriminators for a few
/// steps: the step runs at full size, stays finite and does not blow the
/// generator up.
fn fine_tune_smoke(name: &str) {
    use pmetal_runtime::compat::ops;
    use pmetal_vocoder::BigVGANConfig;
    use pmetal_vocoder::audio::MelFrontEnd;
    use pmetal_vocoder::discriminator::{BigVGANDiscriminator, DiscriminatorConfig};
    use pmetal_vocoder::train::{GanTrainer, TrainingConfig};

    let Some(dir) = model_dir(name) else {
        eprintln!("BIGVGAN_CHECKPOINT_DIR/{name} not found; skipping");
        return;
    };
    let discriminator_checkpoint = dir.join("bigvgan_discriminator_optimizer.pt");
    if !discriminator_checkpoint.is_file() {
        eprintln!("{} not found; skipping", discriminator_checkpoint.display());
        return;
    }
    let json = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let model_config = BigVGANConfig::from_json(&json).unwrap();
    let mut training = TrainingConfig::from_json(&json).unwrap();
    training.learning_rate = 1e-6; // a gentle fine-tune
    let mut discriminator =
        BigVGANDiscriminator::new(&DiscriminatorConfig::from_json(&json).unwrap()).unwrap();
    discriminator
        .load_checkpoint(&discriminator_checkpoint)
        .expect("load the released discriminators");
    let generator = BigVGAN::from_dir(&dir).expect("load the released generator");
    let mut trainer = GanTrainer::new(generator, discriminator, training).unwrap();

    // 8192 samples of the reference signal and their mel.
    let fx = Fixture::load_path(&dir.join("ref.safetensors"));
    let samples = 32 * model_config.hop_size;
    let wav = ops::slice_axis(&fx.array("signal"), 1, 0, samples);
    let mel = MelFrontEnd::new(&model_config)
        .unwrap()
        .forward(&wav)
        .unwrap();
    let wav = wav.reshape(&[1, 1, samples]);

    let started = std::time::Instant::now();
    let mut history = Vec::new();
    for _ in 0..3 {
        history.push(trainer.step(&mel, &wav).expect("training step"));
    }
    eprintln!(
        "{name}: {:.2} s per step; mel error {:?}",
        started.elapsed().as_secs_f64() / 3.0,
        history.iter().map(|m| m.mel_error).collect::<Vec<_>>()
    );
    for m in &history {
        assert!(m.generator_loss.is_finite() && m.mel_error.is_finite());
        assert!(m.discriminator_loss.is_some_and(f32::is_finite));
    }
    assert!(
        history[2].mel_error < 1.5 * history[0].mel_error,
        "the mel error grew: {:?}",
        history
    );
    assert_runtime_ok();
}

#[test]
#[ignore = "needs a downloaded checkpoint (see the module docs)"]
fn bigvgan_base_fine_tunes_with_its_released_discriminators() {
    fine_tune_smoke("base24");
}

#[test]
#[ignore = "needs a downloaded checkpoint (see the module docs)"]
fn bigvgan_v2_fine_tunes_with_its_released_discriminators() {
    fine_tune_smoke("v2_24");
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
