//! The multi-scale sub-band CQT discriminator against the reference.

mod common;

use common::{Fixture, assert_close, assert_close_slice, assert_runtime_ok, to_vec};
use pmetal_runtime::compat::{Array, nn};
use pmetal_vocoder::audio::{Cqt, CqtConfig};
use pmetal_vocoder::discriminator::{
    CqtDiscriminatorConfig, Discriminator, DiscriminatorConfig, DiscriminatorOutput,
    MultiScaleSubbandCqtDiscriminator, SpectralKind,
};
use pmetal_vocoder::loss::{feature_matching_loss, generator_adversarial_loss};
use pmetal_vocoder::nn::StateDict;
use serde_json::Value;
use std::collections::HashMap;

fn disc_config(fx: &Fixture) -> DiscriminatorConfig {
    let meta: Value = serde_json::from_str(fx.meta("disc")).unwrap();
    let config = DiscriminatorConfig::from_json(&meta["cfg"].to_string()).unwrap();
    assert_eq!(config.spectral_kind(), SpectralKind::Cqt);
    config
}

fn small_discriminator(fx: &Fixture) -> MultiScaleSubbandCqtDiscriminator {
    let mut disc = MultiScaleSubbandCqtDiscriminator::new(&disc_config(fx).cqt_config()).unwrap();
    disc.load_state(&mut fx.state_dict("disc.state"), "")
        .unwrap();
    disc
}

fn nchw(x: &Array) -> Array {
    x.transpose_axes(&[0, 3, 1, 2])
}

fn check_outputs(fx: &Fixture, side: &str, outputs: &[DiscriminatorOutput]) {
    let meta: Value = serde_json::from_str(fx.meta("disc")).unwrap();
    assert_eq!(outputs.len(), meta["n_sub"].as_u64().unwrap() as usize);
    for (i, out) in outputs.iter().enumerate() {
        let key = format!("disc.{side}.logits.{i}");
        let scale = fx.values(&key).iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert_close_slice(
            &to_vec(&out.logits),
            fx.values(&key),
            1e-3,
            1e-4 * scale.max(1.0),
            &key,
        );
        assert_eq!(
            out.features.len(),
            meta["n_fmap"].as_u64().unwrap() as usize
        );
        for (j, feature) in out.features.iter().enumerate() {
            let key = format!("disc.{side}.fmap.{i}.{j}");
            let scale = fx.values(&key).iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert_close(
                &nchw(feature),
                fx.shape(&key),
                fx.values(&key),
                1e-3,
                1e-4 * scale.max(1.0),
                &key,
            );
        }
    }
}

#[test]
fn matches_the_reference_logits_and_feature_maps() {
    let fx = Fixture::load("cqt");
    let disc = small_discriminator(&fx);
    check_outputs(
        &fx,
        "real",
        &disc.discriminate(&fx.array("disc.y")).unwrap(),
    );
    check_outputs(
        &fx,
        "fake",
        &disc.discriminate(&fx.array("disc.yhat")).unwrap(),
    );
    assert_runtime_ok();
}

#[test]
fn generator_gradient_matches_autograd_in_the_reference() {
    let fx = Fixture::load("cqt");
    let disc = small_discriminator(&fx);
    let real = disc.discriminate(&fx.array("disc.y")).unwrap();
    let loss_fn = |p: &[Array]| -> Array {
        let fake = disc.discriminate(&p[0]).unwrap();
        generator_adversarial_loss(&fake).unwrap().add(
            &feature_matching_loss(&real, &fake)
                .unwrap()
                .multiply(&Array::from_f32(2.0)),
        )
    };
    let (loss, grads) =
        nn::value_and_grad_explicit(loss_fn, &[fx.array("disc.yhat")], &[]).unwrap();
    let expected = fx.values("disc.loss_g")[0] + fx.values("disc.loss_fm")[0];
    assert!(
        (loss.item_f32() - expected).abs() < 1e-3 * expected.abs().max(1.0),
        "{} vs {expected}",
        loss.item_f32()
    );
    let reference = fx.values("disc.grad_yhat");
    let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert_close_slice(
        &to_vec(&grads[0]),
        reference,
        1e-2,
        1e-3 * scale,
        "gradient",
    );
}

#[test]
fn checks_the_checkpoints_derived_buffers() {
    let fx = Fixture::load("cqt");
    let config = disc_config(&fx);

    // A checkpoint's `lenghts` buffer must equal the one this configuration computes.
    let cqt = Cqt::new(CqtConfig::new(
        2 * config.sampling_rate,
        config.cqtd_hop_lengths[0],
        config.cqtd_bins_per_octaves[0] * config.cqtd_n_octaves[0],
        config.cqtd_bins_per_octaves[0],
    ))
    .unwrap();
    let correct = Array::from_f32_slice(cqt.lengths(), &[cqt.lengths().len() as i32]);
    let wrong = correct.multiply(&Array::from_f32(1.5));
    for (buffer, ok) in [(correct, true), (wrong, false)] {
        let mut disc = MultiScaleSubbandCqtDiscriminator::new(&config.cqt_config()).unwrap();
        let mut sd = fx.state_dict("disc.state");
        let mut tensors = HashMap::new();
        for key in sd.keys().cloned().collect::<Vec<_>>() {
            tensors.insert(key.clone(), sd.take(&key).unwrap());
        }
        tensors.insert("discriminators.0.cqt_transform.lenghts".to_owned(), buffer);
        let result = disc.load_state(&mut StateDict::new(tensors), "");
        assert_eq!(result.is_ok(), ok, "{result:?}");
    }
}

#[test]
fn config_picks_the_spectral_discriminator_like_the_reference() {
    let both = DiscriminatorConfig {
        use_mbd_instead_of_mrd: true,
        use_cqtd_instead_of_mrd: true,
        ..DiscriminatorConfig::default()
    };
    assert_eq!(both.spectral_kind(), SpectralKind::MultiBand);
    let cqt = DiscriminatorConfig {
        use_cqtd_instead_of_mrd: true,
        ..DiscriminatorConfig::default()
    };
    assert_eq!(cqt.spectral_kind(), SpectralKind::Cqt);

    // The reference stack would feed a mismatched width if channels grew per
    // layer, so growth is refused up front.
    let scaled = CqtDiscriminatorConfig {
        filters_scale: 2,
        ..CqtDiscriminatorConfig::bigvgan_v2(24000)
    };
    assert!(MultiScaleSubbandCqtDiscriminator::new(&scaled).is_err());
}

#[test]
fn the_released_v2_configuration_builds_with_reference_sized_layers() {
    // The reference stack for BigVGAN-v2: 3 scales, 9 octave convolutions
    // each (27 in all), then 5 layers and an output layer per scale.
    let disc =
        MultiScaleSubbandCqtDiscriminator::new(&CqtDiscriminatorConfig::bigvgan_v2(24000)).unwrap();
    assert_eq!(disc.discriminators.len(), 3);
    for d in &disc.discriminators {
        assert_eq!(d.conv_pres.len(), 9);
        assert_eq!(d.convs.len(), 5);
    }
}
