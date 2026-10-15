//! Rationale (entropy-weighted) distillation and the reasoning-span mask.

use pmetal_distill::DistillLoss;
use pmetal_distill::RationaleLoss;
use pmetal_distill::reasoning::generate_reasoning_mask;
use pmetal_runtime::compat::Array;
use serial_test::serial;

const VOCAB: usize = 6;

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

fn logits(tokens: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..tokens * VOCAB)
        .map(|i| scale * ((i as f32) * 0.77 + seed).sin())
        .collect()
}

fn log_softmax(row: &[f64], t: f64) -> Vec<f64> {
    let s: Vec<f64> = row.iter().map(|x| x / t).collect();
    let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + s.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    s.iter().map(|x| x - lse).collect()
}

/// `sum(w_i · KL_i) / sum(w_i)` with `w_i = 1 + rw · H(teacher_i) / ln V`,
/// times the mask.
fn reference(teacher: &[f32], student: &[f32], mask: &[f64], rw: f64, t: f64) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (i, m) in mask.iter().enumerate() {
        let row = |x: &[f32]| -> Vec<f64> {
            x[i * VOCAB..(i + 1) * VOCAB]
                .iter()
                .map(|v| *v as f64)
                .collect()
        };
        let lp = log_softmax(&row(teacher), t);
        let lq = log_softmax(&row(student), t);
        let kl: f64 = lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum();
        let entropy: f64 = -lp.iter().map(|a| a.exp() * a).sum::<f64>();
        let w = m * (1.0 + rw * entropy / (VOCAB as f64).ln());
        num += w * kl;
        den += w;
    }
    num / den
}

/// The weight of a token must not depend on the other tokens of the batch:
/// it is its teacher entropy relative to the largest entropy a
/// `V`-way distribution has, `ln V`. Normalizing by the batch's own largest
/// entropy made a sample's loss change with whatever else (padding, junk)
/// shared its batch.
#[test]
#[serial]
fn a_tokens_weight_does_not_depend_on_the_rest_of_the_batch() {
    let loss = RationaleLoss::new(2.0);
    let teacher = logits(3, 0.3, 3.0);
    let student = logits(3, 1.7, 3.0);

    // The same two tokens alone and next to a masked-out token whose teacher
    // is uniform (the highest entropy there is).
    let alone = loss
        .compute(
            &Array::from_f32_slice(&teacher[..2 * VOCAB], &[1, 2, VOCAB as i32]),
            &Array::from_f32_slice(&student[..2 * VOCAB], &[1, 2, VOCAB as i32]),
            1.0,
        )
        .unwrap()
        .item::<f32>();

    let mut padded_teacher = teacher[..2 * VOCAB].to_vec();
    padded_teacher.extend([0.0; VOCAB]);
    let mut padded_student = student[..2 * VOCAB].to_vec();
    padded_student.extend(logits(1, 5.0, 3.0));
    let mask = Array::from_f32_slice(&[1.0, 1.0, 0.0], &[1, 3]);
    let with_junk = loss
        .compute_masked(
            &Array::from_f32_slice(&padded_teacher, &[1, 3, VOCAB as i32]),
            &Array::from_f32_slice(&padded_student, &[1, 3, VOCAB as i32]),
            1.0,
            &mask,
        )
        .unwrap()
        .item::<f32>();
    drain();
    assert!(
        (alone - with_junk).abs() <= 1e-5 * alone.abs(),
        "alone {alone}, with a masked neighbour {with_junk}"
    );
}

#[test]
#[serial]
fn matches_the_entropy_weighted_kl_in_float64() {
    let teacher = logits(4, 0.3, 3.0);
    let student = logits(4, 1.7, 3.0);
    let mask = [1.0, 1.0, 0.0, 1.0];
    for (rw, t) in [(0.0_f32, 1.0_f32), (1.0, 1.0), (3.0, 2.0)] {
        let loss = RationaleLoss::new(rw);
        let got = loss
            .compute_masked(
                &Array::from_f32_slice(&teacher, &[1, 4, VOCAB as i32]),
                &Array::from_f32_slice(&student, &[1, 4, VOCAB as i32]),
                t,
                &Array::from_f32_slice(&[1.0, 1.0, 0.0, 1.0], &[1, 4]),
            )
            .unwrap()
            .item::<f32>() as f64;
        drain();
        let want = reference(&teacher, &student, &mask, rw as f64, t as f64);
        assert!(
            (got - want).abs() <= 2e-5 * want.abs(),
            "rw={rw} T={t}: {got} vs {want}"
        );
    }
}

/// Tokens inside `<think> ... </think>` (the opening marker included, the
/// closing one not), for several blocks and for a block cut off by the end of
/// the sequence; a stray closing marker (a window that began mid-block)
/// opens nothing.
#[test]
#[serial]
fn the_reasoning_mask_covers_each_block() {
    const START: i32 = 90;
    const END: i32 = 91;
    let rows: [[i32; 10]; 3] = [
        [1, START, 2, 3, END, 4, START, 5, END, 6],
        [1, 2, START, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, END, 3, 4, START, 5, END, 6, 7],
    ];
    let tokens: Vec<i32> = rows.iter().flatten().copied().collect();
    let mask = generate_reasoning_mask(
        &Array::from_i32_slice_shaped(&tokens, &[3, 10]),
        START as u32,
        END as u32,
    )
    .unwrap();
    mask.eval();
    drain();
    let got: Vec<f32> = mask.clone().to_f32_vec(30).unwrap();
    let want: [[f32; 10]; 3] = [
        [0., 1., 1., 1., 0., 0., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
    ];
    let want: Vec<f32> = want.iter().flatten().copied().collect();
    assert_eq!(got, want);
}
