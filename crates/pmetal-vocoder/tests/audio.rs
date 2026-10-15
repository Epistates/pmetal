//! Mel front end, STFT and filterbank against the reference implementation.

mod common;

use common::{Fixture, assert_close, assert_close_slice, assert_runtime_ok, to_vec};
use pmetal_runtime::compat::{Array, ops};
use pmetal_vocoder::BigVGANConfig;
use pmetal_vocoder::audio::{
    MelConfig, MelFrontEnd, MelNorm, PadMode, StftConfig, Window, hann_window, hz_to_mel, istft,
    mel_filterbank, mel_filterbank_values, mel_to_hz, stft,
};
use serde_json::Value;

#[test]
fn hann_window_is_periodic_like_torch_by_default() {
    // torch.hann_window(4) = [0, 0.5, 1, 0.5]; the symmetric one is [0, .75, .75, 0].
    assert_close_slice(
        &to_vec(&hann_window(4, true).unwrap()),
        &[0.0, 0.5, 1.0, 0.5],
        0.0,
        1e-7,
        "periodic",
    );
    assert_close_slice(
        &to_vec(&hann_window(4, false).unwrap()),
        &[0.0, 0.75, 0.75, 0.0],
        0.0,
        1e-6,
        "symmetric",
    );
}

#[test]
fn mel_scale_anchor_values() {
    // Slaney: linear up to 1 kHz, where it reaches 15 mel.
    assert!((hz_to_mel(1000.0, false) - 15.0).abs() < 1e-5);
    assert!((hz_to_mel(500.0, false) - 7.5).abs() < 1e-5);
    // The scale is logarithmic above that: 6.4x in frequency is 27 mel steps.
    assert!((hz_to_mel(6400.0, false) - (15.0 + 27.0)).abs() < 1e-3);
    // HTK: 1000 Hz is about 1000 mel.
    assert!((hz_to_mel(1000.0, true) - 999.99).abs() < 0.05);
    for htk in [false, true] {
        for f in [0.0, 80.0, 999.0, 1000.0, 4321.0, 11025.0] {
            assert!(
                (mel_to_hz(hz_to_mel(f, htk), htk) - f).abs() < 1e-2 * (1.0 + f / 1000.0),
                "{f} {htk}"
            );
        }
    }
}

fn filterbank_config(meta: &Value) -> MelConfig {
    MelConfig {
        sr: meta["sr"].as_i64().unwrap() as i32,
        n_fft: meta["n_fft"].as_i64().unwrap() as i32,
        n_mels: meta["n_mels"].as_i64().unwrap() as i32,
        fmin: meta["fmin"].as_f64().unwrap() as f32,
        fmax: meta["fmax"].as_f64().map(|v| v as f32),
        htk: meta.get("htk").and_then(Value::as_bool).unwrap_or(false),
        norm: if meta.get("norm").is_some_and(Value::is_null) {
            MelNorm::None
        } else {
            MelNorm::Slaney
        },
    }
}

#[test]
fn mel_filterbanks_match_librosa() {
    let fx = Fixture::load("audio");
    for name in ["fb_slaney", "fb_htk", "fb_nonorm", "fb_few"] {
        let meta: Value = serde_json::from_str(fx.meta(name)).unwrap();
        let config = filterbank_config(&meta);
        let ours = mel_filterbank_values(&config).unwrap();
        assert_close_slice(&ours, fx.values(name), 1e-5, 1e-7, name);
        let fb = mel_filterbank(&config).unwrap();
        assert_eq!(fb.shape(), fx.shape(name));
    }
}

#[test]
fn mel_filterbank_rejects_impossible_ranges() {
    let bad = MelConfig {
        fmax: Some(20000.0),
        ..MelConfig::default()
    }; // above Nyquist at 24 kHz
    assert!(mel_filterbank(&bad).is_err());
    let inverted = MelConfig {
        fmin: 9000.0,
        fmax: Some(8000.0),
        ..MelConfig::default()
    };
    assert!(mel_filterbank(&inverted).is_err());
}

fn stft_case(fx: &Fixture, tag: &str) -> (StftConfig, Array) {
    let meta: Vec<Value> = serde_json::from_str(fx.meta(&format!("stft_{tag}"))).unwrap();
    let (n_fft, hop, win) = (
        meta[0].as_i64().unwrap() as i32,
        meta[1].as_i64().unwrap() as i32,
        meta[2].as_i64().unwrap() as i32,
    );
    let config = StftConfig {
        n_fft,
        hop_length: hop,
        win_length: Some(win),
        center: meta[3].as_bool().unwrap(),
        pad_mode: PadMode::Reflect,
        window: Window::Hann,
    };
    (config, fx.array("stft_signal"))
}

#[test]
fn stft_magnitude_matches_torch_for_centred_padded_and_short_windows() {
    let fx = Fixture::load("audio");
    for tag in ["a", "b", "c"] {
        let (config, signal) = stft_case(&fx, tag);
        let spec = stft(&signal, &config).unwrap();
        let key = format!("stft_{tag}.mag");
        assert_eq!(spec.shape(), fx.shape(&key), "{tag}: shape");
        assert_close(
            &spec.abs_val(),
            fx.shape(&key),
            fx.values(&key),
            1e-4,
            1e-4,
            &key,
        );
    }
    assert_runtime_ok();
}

#[test]
fn stft_of_a_batch_matches_each_signal() {
    let fx = Fixture::load("audio");
    let (config, signal) = stft_case(&fx, "a");
    let batch = ops::stack_axis(&[signal.clone(), signal.multiply(&Array::from_f32(0.5))], 0);
    let spec = stft(&batch, &config).unwrap();
    assert_eq!(spec.ndim(), 3);
    let first = ops::select_axis(&spec, 0, 0).abs_val();
    assert_close(
        &first,
        fx.shape("stft_a.mag"),
        fx.values("stft_a.mag"),
        1e-4,
        1e-4,
        "batch[0]",
    );
}

#[test]
fn istft_of_a_masked_spectrogram_matches_torch() {
    let fx = Fixture::load("audio");
    for tag in ["a", "b"] {
        let (config, signal) = stft_case(&fx, tag);
        let spec = stft(&signal, &config).unwrap();
        let masked = spec.multiply(&fx.array(&format!("stft_{tag}.mask")));
        let rec = istft(&masked, &config, Some(signal.dim(0))).unwrap();
        let key = format!("stft_{tag}.istft");
        assert_close(&rec, fx.shape(&key), fx.values(&key), 1e-3, 1e-4, &key);
    }
}

#[test]
fn stft_istft_round_trips_a_signal() {
    let fx = Fixture::load("audio");
    let (config, signal) = stft_case(&fx, "a");
    let rec = istft(
        &stft(&signal, &config).unwrap(),
        &config,
        Some(signal.dim(0)),
    )
    .unwrap();
    assert_close(
        &rec,
        signal.shape(),
        &to_vec(&signal),
        1e-3,
        1e-4,
        "round trip",
    );
}

#[test]
fn stft_rejects_signals_it_cannot_frame() {
    let config = StftConfig {
        n_fft: 64,
        hop_length: 16,
        center: false,
        ..StftConfig::default()
    };
    assert!(
        stft(
            &Array::zeros(&[32], pmetal_runtime::compat::Dtype::Float32),
            &config
        )
        .is_err()
    );
    let reflect = StftConfig {
        n_fft: 64,
        hop_length: 16,
        ..StftConfig::default()
    };
    assert!(
        stft(
            &Array::zeros(&[20], pmetal_runtime::compat::Dtype::Float32),
            &reflect
        )
        .is_err()
    );
}

fn front_end_case(fx: &Fixture, tag: &str) -> (MelFrontEnd, Array) {
    let meta: Value = serde_json::from_str(fx.meta(&format!("mel_{tag}"))).unwrap();
    let geti = |k: &str| meta[k].as_i64().unwrap() as i32;
    let config = BigVGANConfig {
        n_fft: geti("n_fft"),
        num_mels: geti("num_mels"),
        sampling_rate: geti("sampling_rate"),
        hop_size: geti("hop_size"),
        win_size: geti("win_size"),
        fmin: geti("fmin"),
        fmax: meta["fmax"].as_i64().map(|v| v as i32),
        ..BigVGANConfig::default()
    };
    (
        MelFrontEnd::new(&config).unwrap(),
        fx.array(&format!("mel_{tag}.wav")),
    )
}

#[test]
fn mel_front_end_matches_the_reference() {
    let fx = Fixture::load("audio");
    for tag in ["bigvgan", "small"] {
        let (front_end, wav) = front_end_case(&fx, tag);
        let mel = front_end.forward(&wav).unwrap();
        let key = format!("mel_{tag}.mel");
        assert_close(&mel, fx.shape(&key), fx.values(&key), 1e-4, 1e-4, &key);
    }
    assert_runtime_ok();
}

/// The vocoder is only usable if mel -> audio returns the samples that went in.
#[test]
fn mel_front_end_frame_count_is_samples_over_hop() {
    let fx = Fixture::load("audio");
    let (front_end, wav) = front_end_case(&fx, "bigvgan"); // 6000 samples, hop 256
    let mel = front_end.forward(&wav).unwrap();
    assert_eq!(mel.dim(2), 6000 / 256);
}
