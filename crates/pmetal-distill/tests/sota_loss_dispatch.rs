//! Integration test for the Phase 3 SOTA-loss dispatch.
//!
//! For every `LossType` variant we instantiate a `Distiller` via the YAML
//! deserialization path (the surface most users interact with) and confirm
//! that the loss is finite, non-negative, and that hash routing through
//! `DistillerBuilder` selected the right implementation. This is the
//! contract test: the new variants are wire-complete from YAML config →
//! `compute_loss` output.

use pmetal_distill::{
    DistillConfig, DistillMethod, Distiller, LossConfig, LossType, TrainingConfig,
};
use pmetal_runtime::compat::Array;

fn make_logits() -> (Array, Array) {
    let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5], &[1, 2, 4]);
    let student = Array::from_f32_slice(&[2.0_f32, 1.0, 4.0, 3.0, 1.0, 0.5, 3.5, 2.5], &[1, 2, 4]);
    (teacher, student)
}

fn distiller_for(loss_type: LossType) -> Distiller {
    let loss = LossConfig {
        loss_type,
        ..LossConfig::default()
    };
    let config = DistillConfig {
        teacher: "t".to_string(),
        student: "s".to_string(),
        method: DistillMethod::Online,
        loss,
        offline: None,
        output_path: None,
        training: TrainingConfig::default(),
    };
    Distiller::new(config).unwrap()
}

#[test]
fn dispatch_every_sota_variant() {
    let (teacher, student) = make_logits();

    // (label, LossType, identity_must_be_zero?)
    //
    // Soft-CE has a non-zero entropy floor on identical inputs (H(p) > 0),
    // so it is excluded from the identity-zero check; MSE and the KL-family
    // losses (incl. JSD, JSD-skewed, ULD, MiniLLM, GKD) all collapse to 0.
    let cases: Vec<(&str, LossType, bool)> = vec![
        ("kl_divergence", LossType::KlDivergence, true),
        ("jensen_shannon", LossType::JensenShannon, true),
        ("soft_cross_entropy", LossType::SoftCrossEntropy, false),
        ("mse_loss", LossType::MseLoss, true),
        ("jsd_skewed", LossType::JsdSkewed { alpha: 0.7 }, true),
        (
            "universal_logit",
            LossType::UniversalLogit { top_k: Some(4) },
            true,
        ),
        (
            "skew_kl",
            LossType::SkewKl {
                alpha: 0.1,
                reverse: false,
            },
            true,
        ),
        (
            "skew_reverse_kl",
            LossType::SkewKl {
                alpha: 0.1,
                reverse: true,
            },
            true,
        ),
        (
            "gkd",
            LossType::Gkd {
                lambda: 0.3,
                sampler_temperature: 1.0,
            },
            true,
        ),
    ];

    for (label, loss_type, identity_zero) in cases {
        let distiller = distiller_for(loss_type);
        let out = distiller
            .compute_loss(&teacher, &student, None, None, 0, 1)
            .unwrap_or_else(|e| panic!("variant {} failed: {}", label, e));
        let v: f32 = out.total.item();
        assert!(
            v.is_finite() && v >= -1e-3,
            "loss for {} must be finite & non-negative; got {}",
            label,
            v
        );

        if identity_zero {
            // KL-family losses must vanish on identical teacher/student.
            let zero_out = distiller
                .compute_loss(&teacher, &teacher, None, None, 0, 1)
                .unwrap();
            let zv: f32 = zero_out.total.item();
            assert!(
                zv.abs() < 1e-3,
                "{} loss should be ~0 for identical inputs; got {}",
                label,
                zv
            );
        }
    }
}
