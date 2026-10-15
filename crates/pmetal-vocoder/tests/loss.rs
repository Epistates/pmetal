//! GAN and mel losses against the reference training code.

mod common;

use common::{Fixture, assert_close_slice, assert_runtime_ok, to_vec};
use pmetal_runtime::compat::{Array, Dtype, nn};
use pmetal_vocoder::discriminator::DiscriminatorOutput;
use pmetal_vocoder::loss::{
    GeneratorLossConfig, MultiScaleMelLoss, discriminator_loss, feature_matching_loss,
    generator_adversarial_loss, generator_loss, mel_l1_loss,
};

/// Two sub-discriminators: the first with two feature maps, the second with one.
fn outputs(fx: &Fixture, logits: &str, features: &str) -> Vec<DiscriminatorOutput> {
    let layout = [2usize, 1];
    (0..2)
        .map(|i| DiscriminatorOutput {
            logits: fx.array(&format!("loss.{logits}.{i}")),
            features: (0..layout[i])
                .map(|j| fx.array(&format!("loss.{features}.{i}.{j}")))
                .collect(),
        })
        .collect()
}

#[test]
fn gan_losses_match_the_reference_reductions() {
    let fx = Fixture::load("discriminators");
    let real = outputs(&fx, "real", "fm_real");
    let fake = outputs(&fx, "fake", "fm_fake");

    let d = discriminator_loss(&real, &fake).unwrap();
    assert_close_slice(
        &[d.total.item_f32()],
        fx.values("loss.d"),
        1e-5,
        1e-6,
        "discriminator",
    );
    assert_close_slice(
        &[d.real.add(&d.fake).item_f32()],
        fx.values("loss.d"),
        1e-5,
        1e-6,
        "d = real + fake",
    );

    let g = generator_adversarial_loss(&fake).unwrap();
    assert_close_slice(
        &[g.item_f32()],
        fx.values("loss.g"),
        1e-5,
        1e-6,
        "generator",
    );

    // The reference folds its weight of 2 into the loss; ours is the plain sum.
    let fm = feature_matching_loss(&real, &fake).unwrap();
    assert_close_slice(
        &[2.0 * fm.item_f32()],
        fx.values("loss.fm"),
        1e-5,
        1e-6,
        "feature matching",
    );
    assert_runtime_ok();
}

#[test]
fn feature_matching_sums_rather_than_averages() {
    // Three identical feature maps with L1 distance 0.5 each: the reference
    // gives 1.5 (times its weight 2), not the mean 0.5.
    let make = |v: f32| DiscriminatorOutput {
        logits: Array::zeros(&[1, 1], Dtype::Float32),
        features: (0..3)
            .map(|_| Array::full(&[1, 2, 2, 1], v, Dtype::Float32))
            .collect(),
    };
    let fm = feature_matching_loss(&[make(1.0)], &[make(0.5)]).unwrap();
    assert!((fm.item_f32() - 1.5).abs() < 1e-6);
}

#[test]
fn combined_generator_loss_weights_its_terms() {
    let fx = Fixture::load("discriminators");
    let (real, fake) = (
        outputs(&fx, "real", "fm_real"),
        outputs(&fx, "fake", "fm_fake"),
    );
    let mel = Array::from_f32(0.25);
    let config = GeneratorLossConfig::v2();
    assert_eq!(
        (config.lambda_adv, config.lambda_fm, config.lambda_mel),
        (1.0, 2.0, 15.0)
    );
    assert_eq!(GeneratorLossConfig::v1().lambda_mel, 45.0);

    // Two banks (here: the same outputs twice) are summed, as the reference sums MPD and MRD.
    let out = generator_loss(&[&real, &real], &[&fake, &fake], mel, &config).unwrap();
    let adv = 2.0 * fx.values("loss.g")[0];
    let fm = 2.0 * fx.values("loss.fm")[0] / 2.0;
    let expected = adv + 2.0 * fm + 15.0 * 0.25;
    assert_close_slice(&[out.total.item_f32()], &[expected], 1e-5, 1e-5, "total");
    assert_close_slice(
        &[out.adversarial.item_f32()],
        &[adv],
        1e-5,
        1e-6,
        "adversarial",
    );
    assert_close_slice(
        &[out.feature_matching.item_f32()],
        &[fm],
        1e-5,
        1e-6,
        "feature matching",
    );
}

#[test]
fn mismatched_discriminator_outputs_are_an_error() {
    let fx = Fixture::load("discriminators");
    let real = outputs(&fx, "real", "fm_real");
    let fake = outputs(&fx, "fake", "fm_fake");
    assert!(discriminator_loss(&real, &fake[..1]).is_err());
    assert!(feature_matching_loss(&[], &[]).is_err());
    let mut short = outputs(&fx, "fake", "fm_fake");
    short[0].features.pop();
    assert!(feature_matching_loss(&real, &short).is_err());
}

#[test]
fn multi_scale_mel_loss_matches_the_reference_value_and_per_scale_terms() {
    let fx = Fixture::load("audio");
    let loss = MultiScaleMelLoss::bigvgan_v2(24000).unwrap();
    let value = loss
        .forward(&fx.array("msmel.yhat"), &fx.array("msmel.y"))
        .unwrap();
    assert_close_slice(
        &[value.item_f32()],
        fx.values("msmel.loss"),
        1e-3,
        1e-4,
        "msmel",
    );
    assert_runtime_ok();
}

#[test]
fn multi_scale_mel_loss_gradient_matches_autograd_in_the_reference() {
    let fx = Fixture::load("audio");
    let loss = MultiScaleMelLoss::bigvgan_v2(24000).unwrap();
    let y = fx.array("msmel.y");
    let (value, grads) = nn::value_and_grad_explicit(
        |p| loss.forward(&p[0], &y).unwrap(),
        &[fx.array("msmel.yhat")],
        &[],
    )
    .unwrap();
    assert_close_slice(
        &[value.item_f32()],
        fx.values("msmel.loss"),
        1e-3,
        1e-4,
        "value",
    );
    let expected = fx.values("msmel.grad");
    let got = to_vec(&grads[0]);
    let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert_close_slice(&got, expected, 2e-2, 2e-2 * scale, "gradient");
    assert_runtime_ok();
}

#[test]
fn multi_scale_mel_loss_is_zero_for_identical_audio_and_accepts_both_layouts() {
    let fx = Fixture::load("audio");
    let loss = MultiScaleMelLoss::bigvgan_v2(24000).unwrap();
    let y = fx.array("msmel.y"); // [1, 1, 4096]
    assert!(loss.forward(&y, &y).unwrap().item_f32().abs() < 1e-7);
    let flat = y.reshape(&[1, 4096]);
    let a = loss
        .forward(&flat, &fx.array("msmel.yhat"))
        .unwrap()
        .item_f32();
    assert!((a - fx.values("msmel.loss")[0]).abs() < 1e-3 * fx.values("msmel.loss")[0].abs());
    assert!(
        loss.forward(&Array::zeros(&[1, 2, 100], Dtype::Float32), &y)
            .is_err()
    );
}

#[test]
fn mel_l1_loss_is_the_mean_absolute_difference() {
    let a = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let b = Array::from_f32_slice(&[1.0, 0.0, 3.0, 8.0], &[1, 2, 2]);
    assert!((mel_l1_loss(&a, &b).unwrap().item_f32() - 1.5).abs() < 1e-6);
    assert!(mel_l1_loss(&a, &Array::zeros(&[1, 2, 3], Dtype::Float32)).is_err());
}
