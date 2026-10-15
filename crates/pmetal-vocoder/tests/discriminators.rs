//! MPD / MRD / MBD against the reference: logits, feature maps, losses and
//! the gradient that reaches the generator.

mod common;

use common::{Fixture, assert_close, assert_close_slice, assert_runtime_ok, count_params, to_vec};
use pmetal_runtime::compat::{Array, Dtype, nn};
use pmetal_vocoder::discriminator::{
    BandDiscriminator, BigVGANDiscriminator, Discriminator, DiscriminatorConfig,
    DiscriminatorOutput, MultiBandDiscriminator, MultiPeriodDiscriminator,
    MultiResolutionDiscriminator, PeriodDiscriminator, ResolutionDiscriminator,
};
use pmetal_vocoder::loss::{feature_matching_loss, generator_adversarial_loss};
use serde_json::Value;

fn meta(fx: &Fixture, name: &str) -> Value {
    serde_json::from_str(fx.meta(name)).unwrap()
}

fn build(name: &str, fx: &Fixture) -> Box<dyn DiscriminatorDyn> {
    let m = meta(fx, name);
    match name {
        "mpd" => {
            let periods: Vec<i32> = m["periods"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            let mut d = MultiPeriodDiscriminator::with_periods(
                &periods,
                m["channel_mult"].as_f64().unwrap() as f32,
            )
            .unwrap();
            let mut sd = fx.state_dict(&format!("{name}.state"));
            d.load_state(&mut sd, "").unwrap();
            // Every tensor of the reference module is a parameter we load or a
            // buffer we check; nothing is left over.
            sd.finish().unwrap();
            Box::new(d)
        }
        "mrd" => {
            let res: Vec<[i32; 3]> = m["resolutions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    let r = r.as_array().unwrap();
                    [
                        r[0].as_i64().unwrap() as i32,
                        r[1].as_i64().unwrap() as i32,
                        r[2].as_i64().unwrap() as i32,
                    ]
                })
                .collect();
            let mut d = MultiResolutionDiscriminator::with_resolutions(
                &res,
                m["channel_mult"].as_f64().unwrap() as f32,
            )
            .unwrap();
            let mut sd = fx.state_dict(&format!("{name}.state"));
            d.load_state(&mut sd, "").unwrap();
            // Every tensor of the reference module is a parameter we load or a
            // buffer we check; nothing is left over.
            sd.finish().unwrap();
            Box::new(d)
        }
        "mbd" => {
            let windows: Vec<i32> = m["window_lengths"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            let mut d =
                MultiBandDiscriminator::new(&windows, m["channels"].as_i64().unwrap() as i32)
                    .unwrap();
            let mut sd = fx.state_dict(&format!("{name}.state"));
            d.load_state(&mut sd, "").unwrap();
            // Every tensor of the reference module is a parameter we load or a
            // buffer we check; nothing is left over.
            sd.finish().unwrap();
            Box::new(d)
        }
        _ => unreachable!(),
    }
}

/// Object-safe view of the discriminators for the shared test body.
trait DiscriminatorDyn {
    fn run(&self, audio: &Array) -> Vec<DiscriminatorOutput>;
    fn saved(&self) -> Vec<(String, Array)>;
}

impl<D: Discriminator> DiscriminatorDyn for D {
    fn run(&self, audio: &Array) -> Vec<DiscriminatorOutput> {
        self.discriminate(audio).unwrap()
    }

    fn saved(&self) -> Vec<(String, Array)> {
        let mut out = Vec::new();
        self.save_state("", &mut out);
        out
    }
}

#[test]
fn saved_state_has_the_reference_names_values_and_layout() {
    let fx = Fixture::load("discriminators");
    for name in ["mpd", "mrd", "mbd"] {
        let saved = build(name, &fx).saved();
        let prefix = format!("{name}.state.");
        // The reference's derived buffers (the MBD's STFT window) are checked
        // on load but not part of what is saved.
        let mut expected: Vec<&str> = fx
            .keys()
            .filter_map(|k| k.strip_prefix(prefix.as_str()))
            .filter(|k| !k.ends_with("spec_fn.window"))
            .collect();
        let mut got: Vec<&str> = saved.iter().map(|(k, _)| k.as_str()).collect();
        expected.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, expected, "{name}: tensor names");
        for (key, tensor) in &saved {
            let full = format!("{prefix}{key}");
            assert_eq!(tensor.shape(), fx.shape(&full), "{full}");
            assert_close_slice(&to_vec(tensor), fx.values(&full), 0.0, 1e-7, &full);
        }
    }
}

fn nchw(x: &Array) -> Array {
    x.transpose_axes(&[0, 3, 1, 2])
}

fn check_outputs(fx: &Fixture, name: &str, side: &str, outputs: &[DiscriminatorOutput]) {
    let n_sub = serde_json::from_str::<Value>(fx.meta(name)).unwrap()["n_sub"]
        .as_u64()
        .unwrap() as usize;
    assert_eq!(outputs.len(), n_sub, "{name} sub-discriminators");
    for (i, out) in outputs.iter().enumerate() {
        let key = format!("{name}.{side}.logits.{i}");
        // The MBD reference leaves its scores unflattened; the values agree.
        assert_close_slice(&to_vec(&out.logits), fx.values(&key), 1e-4, 1e-4, &key);
        for (j, feature) in out.features.iter().enumerate() {
            let key = format!("{name}.{side}.fmap.{i}.{j}");
            assert_close(
                &nchw(feature),
                fx.shape(&key),
                fx.values(&key),
                1e-4,
                1e-4,
                &key,
            );
        }
    }
}

#[test]
fn discriminators_match_reference_logits_and_feature_maps() {
    let fx = Fixture::load("discriminators");
    for name in ["mpd", "mrd", "mbd"] {
        let disc = build(name, &fx);
        check_outputs(&fx, name, "real", &disc.run(&fx.array("y")));
        check_outputs(&fx, name, "fake", &disc.run(&fx.array("yhat")));
    }
    assert_runtime_ok();
}

#[test]
fn generator_side_gradients_match_autograd_in_the_reference() {
    // d/d(y_hat) of  L_adv + 2 * L_fm  through each discriminator bank.
    let fx = Fixture::load("discriminators");
    for name in ["mpd", "mrd", "mbd"] {
        let disc = build(name, &fx);
        let y = fx.array("y");
        let real = disc.run(&y);
        let loss_fn = |params: &[Array]| -> Array {
            let fake = disc.run(&params[0]);
            let adv = generator_adversarial_loss(&fake).unwrap();
            let fm = feature_matching_loss(&real, &fake).unwrap();
            adv.add(&fm.multiply(&Array::from_f32(2.0)))
        };
        let (loss, grads) = nn::value_and_grad_explicit(loss_fn, &[fx.array("yhat")], &[]).unwrap();

        let expected = fx.values(&format!("{name}.loss_g"))[0]
            + 2.0 * fx.values(&format!("{name}.loss_fm"))[0] / 2.0;
        assert!(
            (loss.item_f32() - expected).abs() < 2e-3 * expected.abs().max(1.0),
            "{name}: loss {} vs {expected}",
            loss.item_f32()
        );
        let key = format!("{name}.grad_yhat");
        let got = to_vec(&grads[0]);
        assert_eq!(got.len(), fx.values(&key).len());
        // Compare against the reference gradient with a tolerance relative to its scale.
        let scale = fx.values(&key).iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert_close_slice(&got, fx.values(&key), 1e-4, 1e-4 * scale, &key);
    }
    assert_runtime_ok();
}

#[test]
fn period_discriminator_pads_reflectively_to_a_whole_period() {
    // 400 samples with period 3 reflect-pad by 2 (402 / 3 = 134 rows); the first
    // convolution (kernel 5, stride 3, pad 2) then gives (134 + 4 - 5) / 3 + 1 = 45 rows.
    let fx = Fixture::load("discriminators");
    let d = PeriodDiscriminator::new(3, 1.0 / 32.0).unwrap();
    let out = d.forward(&fx.array("y")).unwrap();
    assert_eq!(out.features[0].shape()[1..3], [45, 3]);
    assert_eq!(
        out.features.len(),
        6,
        "five convolutions and the output layer"
    );
}

#[test]
fn period_discriminator_rejects_signals_shorter_than_the_pad() {
    let d = PeriodDiscriminator::new(11, 1.0 / 32.0).unwrap();
    assert!(
        d.forward(&Array::zeros(&[1, 1, 5], Dtype::Float32))
            .is_err()
    );
    assert!(
        d.forward(&Array::zeros(&[1, 2, 100], Dtype::Float32))
            .is_err()
    );
}

/// Digital silence is a legitimate input; its STFT magnitude is exactly zero
/// and a naive `sqrt(re^2 + im^2)` would backpropagate NaN through it.
#[test]
fn silent_input_gives_finite_gradients() {
    let d = ResolutionDiscriminator::new(64, 16, 48, 1.0 / 16.0).unwrap();
    let band = BandDiscriminator::new(64, 2).unwrap();
    let silence = Array::zeros(&[1, 1, 400], Dtype::Float32);
    for run in [
        Box::new(|x: &[Array]| generator_adversarial_loss(&[d.forward(&x[0]).unwrap()]).unwrap())
            as Box<dyn Fn(&[Array]) -> Array>,
        Box::new(|x: &[Array]| {
            generator_adversarial_loss(&[band.forward(&x[0]).unwrap()]).unwrap()
        }),
    ] {
        let (_, grads) =
            nn::value_and_grad_explicit(|p| run(p), std::slice::from_ref(&silence), &[]).unwrap();
        for g in to_vec(&grads[0]) {
            assert!(g.is_finite(), "gradient through silence is {g}");
        }
    }
}

#[test]
fn config_selects_the_spectral_discriminator_and_parses_the_release_json() {
    let json = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/bigvgan_v2_24khz_100band_256x.json"),
    )
    .unwrap();
    let config = DiscriminatorConfig::from_json(&json).unwrap();
    assert_eq!(config.mpd_reshapes, [2, 3, 5, 7, 11]);
    assert_eq!(config.discriminator_channel_mult, 1.0);
    assert!(!config.use_mbd_instead_of_mrd);

    let small = DiscriminatorConfig {
        mpd_reshapes: vec![2, 3],
        discriminator_channel_mult: 1.0 / 32.0,
        resolutions: vec![[64, 16, 48]],
        ..DiscriminatorConfig::default()
    };
    let both = BigVGANDiscriminator::new(&small).unwrap();
    let (mpd, mrd) = both
        .forward(&Array::random_normal(&[1, 1, 300], Dtype::Float32))
        .unwrap();
    assert_eq!((mpd.len(), mrd.len()), (2, 1));

    let spectral_norm = DiscriminatorConfig {
        use_spectral_norm: true,
        ..small
    };
    assert!(BigVGANDiscriminator::new(&spectral_norm).is_err());
}

#[test]
fn parameter_counts_match_the_reference_modules() {
    assert_eq!(
        count_params(&MultiPeriodDiscriminator::new().unwrap()),
        41_105_770
    );
    assert_eq!(
        count_params(&MultiResolutionDiscriminator::new().unwrap()),
        280_902
    );
    assert_eq!(
        count_params(&MultiBandDiscriminator::new(&[2048, 1024, 512], 32).unwrap()),
        1_413_990
    );
}
