//! `LmHead::cut_cross_entropy` takes label smoothing and z-loss, and computes
//! what PyTorch's `label_smoothing` and the z-loss regulariser define, against
//! an f64 reference written out token by token.

use pmetal_models::dispatcher::{CceLoss, LmHead};
use pmetal_runtime::compat::Array;

const VOCAB: usize = 11;
const HIDDEN: usize = 6;
const TOKENS: usize = 5;

fn data(n: usize, seed: u64) -> Vec<f32> {
    // A deterministic spread in [-1, 1].
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn head(logit_scale: f32, softcap: Option<f32>) -> (LmHead, Vec<f32>, Vec<f32>) {
    let w = data(VOCAB * HIDDEN, 1);
    let h = data(TOKENS * HIDDEN, 2);
    let head = LmHead {
        weight: Array::from_f32_slice(&w, &[VOCAB as i32, HIDDEN as i32]),
        bias: None,
        logit_scale,
        softcap,
    };
    (head, w, h)
}

/// The mean over scored tokens of
/// `(1 - eps)(lse - z_y) + eps(lse - mean z) + lambda lse^2`.
fn reference(
    w: &[f32],
    h: &[f32],
    targets: &[i32],
    ignore: i32,
    scale: f32,
    cap: Option<f32>,
    loss: CceLoss,
) -> f64 {
    let (eps, lambda) = (f64::from(loss.label_smoothing), f64::from(loss.z_loss));
    let mut total = 0.0;
    let mut scored = 0usize;
    for (t, &target) in targets.iter().enumerate() {
        if target == ignore {
            continue;
        }
        let z: Vec<f64> = (0..VOCAB)
            .map(|v| {
                let dot: f64 = (0..HIDDEN)
                    .map(|k| f64::from(h[t * HIDDEN + k]) * f64::from(w[v * HIDDEN + k]))
                    .sum();
                let z = dot * f64::from(scale);
                match cap {
                    Some(cap) => f64::from(cap) * (z / f64::from(cap)).tanh(),
                    None => z,
                }
            })
            .collect();
        let max = z.iter().cloned().fold(f64::MIN, f64::max);
        let lse = max + z.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
        let mean = z.iter().sum::<f64>() / VOCAB as f64;
        total += (1.0 - eps) * (lse - z[target as usize]) + eps * (lse - mean) + lambda * lse * lse;
        scored += 1;
    }
    total / scored as f64
}

fn run(head: &LmHead, h: &[f32], targets: &[i32], loss: CceLoss) -> f64 {
    let hidden = Array::from_f32_slice(h, &[TOKENS as i32, HIDDEN as i32]);
    let targets = Array::from_i32_slice_shaped(targets, &[TOKENS as i32]);
    let out = head
        .cut_cross_entropy(&hidden, &targets, loss)
        .expect("the loss");
    let value = f64::from(out.item_f32());
    pmetal_runtime::check_unobserved_error().unwrap();
    value
}

#[test]
fn label_smoothing_and_z_loss_match_the_definition() {
    let targets = [3, 7, -100, 0, 10];
    for (scale, cap) in [(1.0, None), (0.7, None), (1.0, Some(5.0)), (0.7, Some(5.0))] {
        let (head, w, h) = head(scale, cap);
        for loss in [
            CceLoss::new(-100),
            CceLoss::new(-100).with_label_smoothing(0.1),
            CceLoss::new(-100).with_z_loss(0.01),
            CceLoss::new(-100)
                .with_label_smoothing(0.2)
                .with_z_loss(0.05),
        ] {
            let want = reference(&w, &h, &targets, -100, scale, cap, loss);
            let got = run(&head, &h, &targets, loss);
            assert!(
                (got - want).abs() < 1e-5 * want.abs().max(1.0),
                "scale {scale}, cap {cap:?}, {loss:?}: got {got}, want {want}"
            );
        }
    }
}

/// The options change the loss: a test that passed with them ignored would
/// pass the one above too if the reference happened to agree with the plain loss.
#[test]
fn the_options_actually_move_the_loss() {
    let targets = [3, 7, 1, 0, 10];
    let (head, _, h) = head(1.0, None);
    let plain = run(&head, &h, &targets, CceLoss::new(-100));
    let smoothed = run(
        &head,
        &h,
        &targets,
        CceLoss::new(-100).with_label_smoothing(0.2),
    );
    let z = run(&head, &h, &targets, CceLoss::new(-100).with_z_loss(0.1));
    assert!((smoothed - plain).abs() > 1e-3, "{smoothed} vs {plain}");
    assert!(z > plain + 1e-3, "{z} vs {plain}");
}

#[test]
fn out_of_range_options_are_errors_not_panics() {
    let (head, _, h) = head(1.0, None);
    let hidden = Array::from_f32_slice(&h, &[TOKENS as i32, HIDDEN as i32]);
    let targets = Array::from_i32_slice_shaped(&[0; TOKENS], &[TOKENS as i32]);
    for bad in [
        CceLoss::new(-100).with_label_smoothing(1.0),
        CceLoss::new(-100).with_label_smoothing(-0.1),
        CceLoss::new(-100).with_z_loss(-1.0),
        CceLoss::new(-100).with_z_loss(f32::NAN),
    ] {
        assert!(
            head.cut_cross_entropy(&hidden, &targets, bad).is_err(),
            "{bad:?} should be refused"
        );
    }
}
