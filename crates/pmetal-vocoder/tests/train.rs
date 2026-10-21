//! The GAN training loop on tiny models.

mod common;

use common::{Fixture, assert_runtime_ok, max_abs_diff, to_vec};
use pmetal_runtime::compat::{Array, Dtype, ModuleParametersExt, random};
use pmetal_vocoder::audio::MelFrontEnd;
use pmetal_vocoder::discriminator::{BigVGANDiscriminator, DiscriminatorConfig};
use pmetal_vocoder::train::{GanTrainer, StepMetrics, TrainingConfig};
use pmetal_vocoder::{BigVGAN, BigVGANConfig};

fn tiny_config() -> BigVGANConfig {
    BigVGANConfig::from_json(Fixture::load("generator_v2_tiny").meta("config")).unwrap()
}

fn tiny_discriminator() -> BigVGANDiscriminator {
    BigVGANDiscriminator::new(&DiscriminatorConfig {
        mpd_reshapes: vec![2, 3],
        resolutions: vec![[64, 16, 48]],
        discriminator_channel_mult: 1.0 / 16.0,
        ..DiscriminatorConfig::default()
    })
    .unwrap()
}

/// Build with the global random seed set: the seed is shared by every test
/// thread, so seeding and drawing the initial weights happen under one lock.
fn seeded<T>(seed: u64, build: impl FnOnce() -> T) -> T {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    random::seed(seed);
    build()
}

fn trainer(config: TrainingConfig) -> GanTrainer {
    seeded(1234, || {
        GanTrainer::new(
            BigVGAN::new(tiny_config()).unwrap(),
            tiny_discriminator(),
            config,
        )
        .unwrap()
    })
}

/// A short harmonic tone and its mel spectrogram: `(mel, wav)`.
fn batch(config: &BigVGANConfig, frames: i32) -> (Array, Array) {
    let samples = (frames * config.hop_size) as usize;
    let sr = config.sampling_rate as f32;
    let wav: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / sr;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.25 * (2.0 * std::f32::consts::PI * 1320.0 * t + 0.5).sin()
        })
        .collect();
    let wav = Array::from_f32_slice(&wav, &[1, 1, samples as i32]);
    let mel = MelFrontEnd::new(config)
        .unwrap()
        .forward(&wav.reshape(&[1, samples as i32]))
        .unwrap();
    (mel, wav)
}

fn params(model: &impl ModuleParametersExt) -> Vec<(String, Vec<f32>)> {
    let mut all: Vec<(String, Vec<f32>)> = model
        .flatten_params()
        .into_iter()
        .map(|(k, v)| (k.to_string(), to_vec(&v)))
        .collect();
    all.sort_by(|a, b| a.0.cmp(&b.0));
    all
}

fn changed(before: &[(String, Vec<f32>)], after: &[(String, Vec<f32>)]) -> usize {
    assert_eq!(before.len(), after.len());
    before
        .iter()
        .zip(after)
        .filter(|(b, a)| max_abs_diff(&a.1, &b.1) > 0.0)
        .count()
}

#[test]
fn a_step_updates_both_networks_and_reports_sane_metrics() {
    let mut t = trainer(TrainingConfig::default());
    let (mel, wav) = batch(&t.generator.config, 24);
    let (g0, d0) = (params(&t.generator), params(&t.discriminator));

    let m = t.step(&mel, &wav).unwrap();
    assert_eq!(m.step, 0);
    assert_eq!(t.steps(), 1);
    assert!(m.generator_loss.is_finite() && m.generator_loss > 0.0);
    assert!(m.mel_error.is_finite() && m.mel_error > 0.0);
    assert!(m.discriminator_loss.is_some_and(f32::is_finite));
    let (mpd_norm, spectral_norm) = m.grad_norm_discriminator.unwrap();
    assert!(m.grad_norm_generator > 0.0 && mpd_norm > 0.0 && spectral_norm > 0.0);
    assert_eq!(m.learning_rate, 1e-4);

    // Every generator and discriminator tensor moved (AdamW's first step moves
    // each parameter by about the learning rate).
    let (g1, d1) = (params(&t.generator), params(&t.discriminator));
    assert_eq!(changed(&g0, &g1), g0.len(), "generator tensors that moved");
    assert_eq!(
        changed(&d0, &d1),
        d0.len(),
        "discriminator tensors that moved"
    );
    assert_runtime_ok();
}

#[test]
fn the_frozen_warmup_trains_the_generator_on_the_mel_loss_alone() {
    let mut t = trainer(TrainingConfig {
        freeze_step: 2,
        ..TrainingConfig::default()
    });
    let (mel, wav) = batch(&t.generator.config, 24);
    let d0 = params(&t.discriminator);

    for step in 0..2 {
        let m = t.step(&mel, &wav).unwrap();
        assert!(m.discriminator_loss.is_none() && m.grad_norm_discriminator.is_none());
        // The generator's whole objective is lambda_mel times the mel error.
        let expected = 45.0 * m.mel_error;
        assert!(
            (m.generator_loss - expected).abs() < 1e-3 * expected,
            "step {step}: {} vs {expected}",
            m.generator_loss
        );
    }
    assert_eq!(
        changed(&d0, &params(&t.discriminator)),
        0,
        "the discriminators are frozen"
    );

    let m = t.step(&mel, &wav).unwrap();
    assert!(
        m.discriminator_loss.is_some(),
        "adversarial training starts at the freeze step"
    );
    assert!(
        m.generator_loss > 45.0 * m.mel_error,
        "adversarial terms are added"
    );
    assert!(changed(&d0, &params(&t.discriminator)) > 0);
}

#[test]
fn the_learning_rate_decays_exponentially_per_step() {
    let config = TrainingConfig {
        learning_rate: 1e-3,
        lr_decay: 0.9,
        ..TrainingConfig::default()
    };
    assert!((config.learning_rate_at(3) - 1e-3 * 0.9f32.powi(3)).abs() < 1e-9);
    let mut t = trainer(config);
    let (mel, wav) = batch(&t.generator.config, 24);
    let rates: Vec<f32> = (0..3)
        .map(|_| t.step(&mel, &wav).unwrap().learning_rate)
        .collect();
    for (step, lr) in rates.iter().enumerate() {
        let want = 1e-3 * 0.9f32.powi(step as i32);
        assert!((lr - want).abs() < 1e-8, "step {step}: {lr} vs {want}");
    }
}

#[test]
fn the_generator_fits_a_fixed_batch() {
    let mut t = trainer(TrainingConfig {
        learning_rate: 2e-3,
        freeze_step: 0,
        ..TrainingConfig::default()
    });
    let (mel, wav) = batch(&t.generator.config, 24);
    let history: Vec<StepMetrics> = (0..60).map(|_| t.step(&mel, &wav).unwrap()).collect();
    let first = history[0].mel_error;
    // GPU reductions are not bit-reproducible, so judge the tail, not one step.
    let tail: f32 = history[55..].iter().map(|m| m.mel_error).sum::<f32>() / 5.0;
    assert!(tail < 0.85 * first, "mel error {first} -> {tail}");
    assert!(history.iter().all(|m| m.generator_loss.is_finite()));
}

#[test]
fn multi_scale_mel_loss_trains_too() {
    let mut t = trainer(TrainingConfig {
        use_multiscale_melloss: true,
        lambda_melloss: 15.0,
        learning_rate: 2e-3,
        ..TrainingConfig::default()
    });
    // Long enough for the 2048-point window (reflect padding needs > 1024 samples).
    let (mel, wav) = batch(&t.generator.config, 320);
    let first = t.step(&mel, &wav).unwrap();
    let mut last = first.clone();
    for _ in 0..4 {
        last = t.step(&mel, &wav).unwrap();
    }
    assert!(first.mel_error.is_finite() && last.mel_error.is_finite());
    assert!(
        last.mel_error < first.mel_error,
        "{} -> {}",
        first.mel_error,
        last.mel_error
    );
}

#[test]
fn a_checkpoint_resumes_training_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ckpt.safetensors");

    let mut a = trainer(TrainingConfig::default());
    let (mel, wav) = batch(&a.generator.config, 24);
    for _ in 0..3 {
        a.step(&mel, &wav).unwrap();
    }
    a.save_checkpoint(&path).unwrap();
    let continued: Vec<StepMetrics> = (0..2).map(|_| a.step(&mel, &wav).unwrap()).collect();

    // A fresh trainer with different initial weights picks up where `a` stopped.
    let mut b = seeded(99, || {
        GanTrainer::new(
            BigVGAN::new(tiny_config()).unwrap(),
            tiny_discriminator(),
            TrainingConfig::default(),
        )
        .unwrap()
    });
    b.load_checkpoint(&path).unwrap();
    assert_eq!(b.steps(), 3);
    let resumed: Vec<StepMetrics> = (0..2).map(|_| b.step(&mel, &wav).unwrap()).collect();

    for (c, r) in continued.iter().zip(&resumed) {
        assert_eq!(c.step, r.step);
        assert!(
            (c.generator_loss - r.generator_loss).abs() < 1e-3 * c.generator_loss.abs().max(1.0),
            "{c:?} vs {r:?}"
        );
        assert!((c.mel_error - r.mel_error).abs() < 1e-4 * c.mel_error.max(1e-3));
        assert_eq!(c.learning_rate, r.learning_rate);
    }
}

#[test]
fn a_model_prepared_for_inference_can_be_trained() {
    let mut t = seeded(5, || {
        let mut generator = BigVGAN::new(tiny_config()).unwrap();
        generator.prepare_for_inference();
        GanTrainer::new(generator, tiny_discriminator(), TrainingConfig::default()).unwrap()
    });
    let (mel, wav) = batch(&t.generator.config, 24);
    let g0 = params(&t.generator);
    let m = t.step(&mel, &wav).unwrap();
    assert!(m.grad_norm_generator > 0.0, "gradients reach the generator");
    assert!(changed(&g0, &params(&t.generator)) > 0);
}

#[test]
fn audio_that_does_not_match_the_mel_is_refused() {
    let mut t = trainer(TrainingConfig::default());
    let (mel, wav) = batch(&t.generator.config, 24);
    let short = pmetal_runtime::compat::ops::slice_axis(&wav, 2, 0, wav.dim(2) - 8);
    assert!(t.step(&mel, &short).is_err());
    assert!(
        t.step(&mel, &Array::zeros(&[1, 2, wav.dim(2)], Dtype::Float32))
            .is_err()
    );
    assert_eq!(t.steps(), 0);
}

#[test]
fn the_release_config_json_provides_the_training_settings() {
    let json = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/bigvgan_v2_24khz_100band_256x.json"),
    )
    .unwrap();
    let c = TrainingConfig::from_json(&json).unwrap();
    assert_eq!(c.learning_rate, 1e-4);
    assert_eq!((c.adam_b1, c.adam_b2), (0.8, 0.99));
    assert_eq!(c.lr_decay, 0.9999996);
    assert_eq!(c.clip_grad_norm, 500.0);
    assert!(c.use_multiscale_melloss);
    assert_eq!(c.lambda_melloss, 15.0);
}
