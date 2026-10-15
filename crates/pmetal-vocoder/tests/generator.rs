//! The BigVGAN generator against reference outputs, plus streaming.

mod common;

use common::{
    Fixture, assert_close, assert_close_slice, assert_runtime_ok, count_params, max_abs_diff,
    to_vec,
};
use pmetal_runtime::compat::{Array, Dtype, ops, random};
use pmetal_vocoder::{BigVGAN, BigVGANConfig, StreamingSynthesizer};

const NON_WEIGHTS: [&str; 2] = ["mel", "audio"];

fn tiny(name: &str) -> (BigVGAN, Fixture) {
    let fx = Fixture::load(name);
    let config = BigVGANConfig::from_json(fx.meta("config")).unwrap();
    let mut model = BigVGAN::new(config).unwrap();
    model
        .load_state_dict(fx.state_dict_except(&NON_WEIGHTS))
        .unwrap();
    (model, fx)
}

#[test]
fn generator_matches_reference_for_the_v2_layout() {
    // AMPBlock1, SnakeBeta in log scale, no final bias, clamped output.
    let (model, fx) = tiny("generator_v2_tiny");
    let audio = model.forward(&fx.array("mel")).unwrap();
    assert_close(
        &audio,
        fx.shape("audio"),
        fx.values("audio"),
        1e-4,
        1e-5,
        "v2 tiny",
    );
    assert_runtime_ok();
}

#[test]
fn generator_matches_reference_for_the_v1_layout() {
    // AMPBlock2, plain Snake, final bias, tanh output.
    let (model, fx) = tiny("generator_v1_tiny");
    let audio = model.forward(&fx.array("mel")).unwrap();
    assert_close(
        &audio,
        fx.shape("audio"),
        fx.values("audio"),
        1e-4,
        1e-5,
        "v1 tiny",
    );
    assert_runtime_ok();
}

#[test]
fn output_length_is_exactly_frames_times_hop() {
    for name in ["generator_v2_tiny", "generator_v1_tiny"] {
        let (model, fx) = tiny(name);
        for frames in [1, 2, 7, 13] {
            let mel = Array::random_normal(&[1, model.config.num_mels, frames], Dtype::Float32);
            let audio = model.forward(&mel).unwrap();
            assert_eq!(audio.shape(), &[1, 1, frames * model.hop_size()], "{name}");
        }
        drop(fx);
    }
    assert_runtime_ok();
}

#[test]
fn fusing_weight_norm_does_not_change_the_output() {
    let (mut model, fx) = tiny("generator_v2_tiny");
    model.fuse_weight_norm();
    let audio = model.forward(&fx.array("mel")).unwrap();
    assert_close(
        &audio,
        fx.shape("audio"),
        fx.values("audio"),
        1e-4,
        1e-5,
        "fused",
    );
    // The fused checkpoint exports `weight`, not `weight_v` / `weight_g`.
    let names: Vec<String> = model.state_dict().into_iter().map(|(k, _)| k).collect();
    assert!(names.iter().any(|n| n == "conv_pre.weight"));
    assert!(!names.iter().any(|n| n.ends_with("weight_g")));
}

#[test]
fn a_fused_checkpoint_loads_back_into_a_fresh_model() {
    let (mut model, fx) = tiny("generator_v1_tiny");
    model.fuse_weight_norm();
    let dir = tempdir();
    let path = dir.join("fused.safetensors");
    model.save_safetensors(&path).unwrap();

    let mut fresh = BigVGAN::new(model.config.clone()).unwrap();
    fresh.load_weights(&path).unwrap();
    let audio = fresh.forward(&fx.array("mel")).unwrap();
    assert_close(
        &audio,
        fx.shape("audio"),
        fx.values("audio"),
        1e-4,
        1e-5,
        "reloaded",
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn state_dict_uses_the_reference_names_and_layout() {
    let (model, fx) = tiny("generator_v2_tiny");
    let saved = model.state_dict();
    let mut saved_names: Vec<&str> = saved.iter().map(|(k, _)| k.as_str()).collect();
    let mut expected: Vec<&str> = fx
        .keys()
        .map(String::as_str)
        .filter(|k| !NON_WEIGHTS.contains(k))
        .collect();
    saved_names.sort_unstable();
    expected.sort_unstable();
    assert_eq!(saved_names, expected);
    for (name, tensor) in &saved {
        assert_eq!(tensor.shape(), fx.shape(name), "{name}");
        assert_close_slice(&to_vec(tensor), fx.values(name), 0.0, 1e-7, name);
    }
}

#[test]
fn parameter_count_matches_the_reference_architecture() {
    // BigVGAN-base: 14,025,154 parameters in the reference module.
    let model = BigVGAN::new(BigVGANConfig::base_24khz_100band()).unwrap();
    assert_eq!(count_params(&model), 14_025_154);
}

#[test]
fn rejects_a_mel_with_the_wrong_number_of_bins() {
    let (model, _) = tiny("generator_v2_tiny");
    let mel = Array::zeros(&[1, 5, 4], Dtype::Float32);
    assert!(model.forward(&mel).is_err());
    assert!(
        model
            .forward(&Array::zeros(&[8, 4], Dtype::Float32))
            .is_err()
    );
}

#[test]
fn generate_squeezes_the_batch_and_channel_axes() {
    let (model, fx) = tiny("generator_v2_tiny");
    let mel = fx.array("mel"); // [2, 8, 11]
    let batched = model.generate(&mel).unwrap();
    assert_eq!(batched.shape(), &[2, 88]);
    let single = model.generate(&ops::select_axis(&mel, 0, 0)).unwrap();
    assert_eq!(single.shape(), &[88]);
    assert_close_slice(
        &to_vec(&single),
        &fx.values("audio")[..88],
        1e-4,
        1e-5,
        "generate",
    );
}

#[test]
fn half_precision_tracks_float32() {
    let (mut model, fx) = tiny("generator_v2_tiny");
    model.fuse_weight_norm();
    let mel = fx.array("mel");
    let reference = to_vec(&model.forward(&mel).unwrap());
    model.to_dtype(Dtype::Bfloat16);
    // A float32 mel runs in the model's precision rather than promoting every
    // convolution back to float32.
    assert_eq!(model.dtype(), Dtype::Bfloat16);
    let audio = model.forward(&mel).unwrap();
    assert_eq!(audio.dtype(), Dtype::Bfloat16);
    let got = to_vec(&audio);
    let signal: f32 = reference.iter().map(|v| v * v).sum();
    let noise: f32 = got
        .iter()
        .zip(&reference)
        .map(|(g, r)| (g - r) * (g - r))
        .sum();
    let snr_db = 10.0 * (signal / noise.max(1e-30)).log10();
    assert!(snr_db > 20.0, "bf16 SNR {snr_db:.1} dB");
}

// --------------------------------------------------------------- receptive field

fn mel_with_bump(model: &BigVGAN, frames: i32, bump_at: i32) -> (Array, Array) {
    random::seed(7);
    let base = Array::random_normal(&[1, model.config.num_mels, frames], Dtype::Float32);
    let mut cols = vec![0.0f32; (model.config.num_mels * frames) as usize];
    for m in 0..model.config.num_mels {
        cols[(m * frames + bump_at) as usize] = 5.0;
    }
    let bumped = base.add(&Array::from_f32_slice(
        &cols,
        &[1, model.config.num_mels, frames],
    ));
    (base, bumped)
}

#[test]
fn nothing_outside_the_receptive_field_depends_on_a_mel_frame() {
    for name in ["generator_v2_tiny", "generator_v1_tiny"] {
        let (model, _) = tiny(name);
        let frames = 80;
        let bump_at = 40;
        let (base, bumped) = mel_with_bump(&model, frames, bump_at);
        let a = to_vec(&model.forward(&base).unwrap());
        let b = to_vec(&model.forward(&bumped).unwrap());
        let hop = model.hop_size() as usize;
        let radius = model.receptive_radius_samples() as usize;
        let centre = bump_at as usize * hop;

        let mut farthest_change = 0usize;
        for n in 0..a.len() {
            let distance = n.abs_diff(centre);
            if (a[n] - b[n]).abs() > 1e-6 {
                farthest_change = farthest_change.max(distance);
            }
        }
        assert!(
            farthest_change <= radius,
            "{name}: sample {farthest_change} away changed, receptive radius is only {radius}"
        );
        // The bound is useful only if it is not wildly loose.
        assert!(
            farthest_change * 4 >= radius,
            "{name}: radius {radius} overestimates the true reach {farthest_change} by more than 4x"
        );
    }
}

fn ramp_mel(model: &BigVGAN, frames: i32) -> Array {
    random::seed(11);
    Array::random_normal(&[2, model.config.num_mels, frames], Dtype::Float32)
}

#[test]
fn chunked_synthesis_equals_one_shot_synthesis() {
    for name in ["generator_v2_tiny", "generator_v1_tiny"] {
        let (model, _) = tiny(name);
        let mel = ramp_mel(&model, 57);
        let full = to_vec(&model.forward(&mel).unwrap());
        for chunk in [1, 4, 9, 57, 200] {
            let chunked = model.synthesize_chunked(&mel, chunk).unwrap();
            assert_eq!(chunked.shape(), &[2, 1, 57 * model.hop_size()]);
            let diff = max_abs_diff(&to_vec(&chunked), &full);
            assert!(
                diff < 1e-5,
                "{name}: chunk {chunk} differs from one-shot by {diff}"
            );
        }
    }
}

#[test]
fn streaming_with_uneven_pushes_equals_one_shot_and_is_causal_in_output_count() {
    let (model, _) = tiny("generator_v2_tiny");
    let mel = ramp_mel(&model, 40);
    let full = to_vec(&model.forward(&mel).unwrap());

    let mut stream = StreamingSynthesizer::new(&model, 2);
    let latency = stream.latency_frames();
    let hop = model.hop_size() as usize;
    let mut audio: Vec<Array> = Vec::new();
    let mut produced_frames = 0usize;
    let mut at = 0;
    for step in [1, 3, 2, 8, 1, 5, 9, 4, 7] {
        let end = (at + step).min(40);
        let piece = ops::slice_axis(&mel, 2, at, end);
        if let Some(a) = stream.push(&piece).unwrap() {
            produced_frames += a.dim(2) as usize / hop;
            audio.push(a);
        }
        at = end;
        // Output trails the input by exactly the latency, never more.
        assert_eq!(
            produced_frames,
            stream.frames_received().saturating_sub(latency)
        );
    }
    assert_eq!(at, 40);
    if let Some(a) = stream.finish().unwrap() {
        audio.push(a);
    }
    let refs: Vec<&Array> = audio.iter().collect();
    let joined = ops::concatenate_axis(&refs, 2);
    assert_eq!(joined.shape(), &[2, 1, 40 * hop as i32]);
    assert!(max_abs_diff(&to_vec(&joined), &full) < 1e-5);
}

#[test]
fn streaming_rejects_a_chunk_with_the_wrong_batch_or_bins() {
    let (model, _) = tiny("generator_v2_tiny");
    let mut stream = StreamingSynthesizer::new(&model, 1);
    assert!(
        stream
            .push(&Array::zeros(&[2, 8, 3], Dtype::Float32))
            .is_err()
    );
    assert!(
        stream
            .push(&Array::zeros(&[1, 7, 3], Dtype::Float32))
            .is_err()
    );
    assert!(stream.finish().unwrap().is_none());
}

// ------------------------------------------------------------------------ config

#[test]
fn a_released_config_json_parses_including_null_fmax() {
    // The BigVGAN-v2 24 kHz release's `config.json`: training-only keys are
    // ignored and `"fmax": null` means Nyquist.
    let json = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/bigvgan_v2_24khz_100band_256x.json"),
    )
    .unwrap();
    let config = BigVGANConfig::from_json(&json).unwrap();
    assert_eq!(config, BigVGANConfig::v2_24khz_100band());
    assert_eq!(config.fmax, None);
    assert_eq!(config.effective_fmax(), 12000.0);
    assert!(!config.use_tanh_at_final && !config.use_bias_at_final);
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pmetal-vocoder-test-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
