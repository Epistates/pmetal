//! Memory use of the training-time backward passes.
//!
//! This file holds one test so that the process-wide peak-memory counter sees
//! only its work.

use pmetal_runtime::compat::{Array, nn};
use pmetal_runtime::inline_array::{get_peak_memory, reset_peak_memory};
use pmetal_vocoder::audio::{Cqt, CqtConfig};
use pmetal_vocoder::discriminator::{
    CqtDiscriminatorConfig, Discriminator, MultiScaleSubbandCqtDiscriminator,
};
use pmetal_vocoder::loss::generator_adversarial_loss;

/// Peak GPU memory, in MB, while running `f` (graph build, evaluation and the
/// gradient it returns), counting what was already live when it started.
fn peak_mb(f: impl FnOnce()) -> f64 {
    reset_peak_memory();
    f();
    get_peak_memory() as f64 / 1e6
}

fn tone(samples: i32) -> Array {
    let t: Vec<f32> = (0..samples)
        .map(|i| (i as f32 * 0.05).sin() * 0.5)
        .collect();
    Array::from_f32_slice(&t, &[1, samples])
}

#[test]
fn cqt_backward_does_not_materialise_the_kernel_bank_per_sample() {
    // Release configuration: 9 octaves, 48 bins per octave, hop 256, at 48 kHz.
    let cqt = Cqt::new(CqtConfig::new(48000, 256, 48 * 9, 48)).unwrap();
    let x = tone(15360);
    x.eval();
    let mut gradient_checked = false;
    let peak = peak_mb(|| {
        let f = |p: &[Array]| cqt.forward(&p[0]).unwrap().abs_val().mean_all();
        let (loss, grads) = nn::value_and_grad_explicit(f, std::slice::from_ref(&x), &[]).unwrap();
        loss.eval();
        grads[0].eval();
        gradient_checked = grads[0].shape() == x.shape();
    });
    assert!(gradient_checked);
    // The strided 512-tap convolutions' backward used to build an im2col
    // buffer of several GB for this 0.3 s of audio.
    assert!(peak < 300.0, "CQT forward+backward peaked at {peak:.0} MB");

    // And the discriminator built on it, at release width.
    let disc =
        MultiScaleSubbandCqtDiscriminator::new(&CqtDiscriminatorConfig::bigvgan_v2(24000)).unwrap();
    let wav = tone(7680).reshape(&[1, 1, 7680]);
    let peak = peak_mb(|| {
        let f =
            |p: &[Array]| generator_adversarial_loss(&disc.discriminate(&p[0]).unwrap()).unwrap();
        let (loss, grads) =
            nn::value_and_grad_explicit(f, std::slice::from_ref(&wav), &[]).unwrap();
        loss.eval();
        grads[0].eval();
    });
    // 12.8 GB before; the weights alone are 0.7 GB.
    assert!(
        peak < 2500.0,
        "CQT discriminator forward+backward peaked at {peak:.0} MB"
    );
}
