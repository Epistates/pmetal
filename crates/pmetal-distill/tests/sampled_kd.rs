//! Random-Sampling KD (Anshumann et al., ACL 2025): an offline cache of draws
//! from the teacher, whose empirical distribution is an unbiased estimate of
//! it.

use pmetal_distill::losses::{DistillLoss, KlDivergenceLoss, SparseKlLoss};
use pmetal_distill::{
    CompressedLogits, CompressionMethod, DistillConfig, DistillMethod, Distiller, LogitCache,
    LogitCompressor, LossConfig, SparseTargets,
};
use pmetal_runtime::compat::nn::value_and_grad_explicit;
use pmetal_runtime::compat::{Array, random};
use serial_test::serial;

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

fn softmax(z: &[f32], temperature: f64) -> Vec<f64> {
    let s: Vec<f64> = z.iter().map(|x| *x as f64 / temperature).collect();
    let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|x| x / sum).collect()
}

fn payload(c: CompressedLogits) -> (Vec<f32>, Vec<i32>, usize, f32) {
    match c {
        CompressedLogits::Sampled {
            probs,
            indices,
            slots,
            temperature,
            ..
        } => (probs, indices, slots, temperature),
        other => panic!("expected Sampled, got {other:?}"),
    }
}

/// The frequencies of many draws are the teacher's softmax, to within sampling
/// error (five standard errors per entry), at the temperature they were drawn.
#[test]
#[serial]
fn the_draws_estimate_the_teachers_distribution() {
    let row = [2.0_f32, 1.0, 0.0, -1.0, 0.5];
    let draws = 5_000;
    for temperature in [1.0_f32, 2.0] {
        random::seed(11);
        let compressor = LogitCompressor::new(CompressionMethod::Sampled, draws)
            .with_sampling_temperature(temperature);
        let (probs, indices, slots, stored_t) = payload(
            compressor
                .compress(&Array::from_f32_slice(&row, &[1, 5]))
                .unwrap(),
        );
        drain();
        assert_eq!(slots, 5);
        assert_eq!(stored_t, temperature);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);

        let want = softmax(&row, temperature as f64);
        let mut seen = [0.0_f64; 5];
        for (p, i) in probs.iter().zip(&indices) {
            seen[*i as usize] += *p as f64;
        }
        for i in 0..5 {
            let tol = 5.0 * (want[i] * (1.0 - want[i]) / draws as f64).sqrt();
            assert!(
                (seen[i] - want[i]).abs() <= tol,
                "T={temperature} token {i}: frequency {} vs probability {}",
                seen[i],
                want[i]
            );
        }
        // Most frequent first.
        assert!(probs.windows(2).all(|w| w[0] >= w[1]), "{probs:?}");
    }
}

/// The estimate is unbiased for the gradient: averaged over many small caches
/// (12 draws each, the paper's setting), the frequency of each token is its
/// probability, so the expected gradient `(q − p̂)/T` is the full one.
#[test]
#[serial]
fn twelve_draws_are_unbiased_on_average() {
    let row = [1.5_f32, 1.0, 0.2, -0.5, -1.0, 0.0];
    let trials = 400;
    let mut mean = [0.0_f64; 6];
    random::seed(5);
    let compressor = LogitCompressor::new(CompressionMethod::Sampled, 12);
    for _ in 0..trials {
        let (probs, indices, _, _) = payload(
            compressor
                .compress(&Array::from_f32_slice(&row, &[1, 6]))
                .unwrap(),
        );
        for (p, i) in probs.iter().zip(&indices) {
            mean[*i as usize] += *p as f64 / trials as f64;
        }
    }
    drain();
    let want = softmax(&row, 1.0);
    for i in 0..6 {
        // One cache's frequency has variance p(1-p)/12.
        let tol = 5.0 * (want[i] * (1.0 - want[i]) / (12.0 * trials as f64)).sqrt();
        assert!(
            (mean[i] - want[i]).abs() <= tol,
            "token {i}: mean frequency {} vs probability {}",
            mean[i],
            want[i]
        );
    }
}

#[test]
#[serial]
fn draws_are_reproducible_under_a_seed() {
    let row: Vec<f32> = (0..2 * 40).map(|i| ((i * 7) % 13) as f32 * 0.3).collect();
    let logits = Array::from_f32_slice(&row, &[1, 2, 40]);
    let compressor = LogitCompressor::new(CompressionMethod::Sampled, 12);
    let run = |seed: u64| {
        random::seed(seed);
        payload(compressor.compress(&logits).unwrap())
    };
    let (a, ai, ..) = run(3);
    let (b, bi, ..) = run(3);
    let (_, ci, ..) = run(4);
    assert_eq!((&a, &ai), (&b, &bi));
    assert_ne!(ai, ci);
}

/// Through a cache file: the shape, the temperature and the frequencies come
/// back, and stacking sequences pads with frequency zero.
#[test]
#[serial]
fn a_sampled_cache_round_trips() {
    let (seq, vocab) = (3_usize, 17_usize);
    let data: Vec<f32> = (0..seq * vocab)
        .map(|i| ((i * 5) % 11) as f32 * 0.4)
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let mut cache = LogitCache::new(dir.path(), CompressionMethod::Sampled, 8)
        .unwrap()
        .with_sampling_temperature(2.0);
    random::seed(1);
    cache
        .cache_sequence(
            0,
            &Array::from_f32_slice(&data, &[1, seq as i32, vocab as i32]),
        )
        .unwrap();
    cache.set_metadata("m".into(), vocab, seq);
    cache.save_metadata().unwrap();

    let cache = LogitCache::load(dir.path()).unwrap();
    let targets = cache.load_sequence_sampled(0).unwrap();
    assert_eq!(targets.shape(), &[1, seq as i32, 8]);
    assert_eq!(targets.vocab_size(), vocab);
    assert_eq!(targets.temperature(), 2.0);
    let sums = targets
        .probs()
        .sum_axes(&[-1], false)
        .to_f32_vec(seq)
        .unwrap();
    assert!(sums.iter().all(|s| (s - 1.0).abs() < 1e-5), "{sums:?}");

    // A top-k reader refuses it, a sampled reader refuses a top-k cache.
    assert!(cache.load_sequence_sparse(0).is_err());

    let stacked = SparseTargets::stack(&[targets.clone(), targets], 5).unwrap();
    assert_eq!(stacked.shape(), &[2, 5, 8]);
    let tail = stacked
        .probs()
        .slice(&[0, 3, 0], &[2, 5, 8])
        .sum_all()
        .item::<f32>();
    assert_eq!(tail, 0.0, "padding positions carry frequency zero");
}

/// The loss is `KL(empirical || student)`: against float64, and equal to the
/// dense KL on the densified empirical distribution.
#[test]
#[serial]
fn the_sampled_loss_is_the_kl_to_the_empirical_distribution() {
    let (vocab, temperature) = (7_usize, 2.0_f32);
    // Two positions: 4 draws {5:2, 1:1, 6:1} and 4 draws {0:3, 2:1}; unused
    // slots have frequency 0.
    let targets = SparseTargets::new(
        Array::from_f32_slice(&[0.5, 0.25, 0.25, 0.75, 0.25, 0.0], &[1, 2, 3]),
        Array::from_i32_slice_shaped(&[5, 1, 6, 0, 2, 0], &[1, 2, 3]),
        vocab,
        temperature,
    )
    .unwrap();
    let s: Vec<f32> = (0..2 * vocab)
        .map(|i| ((i * 3) % 7) as f32 * 0.5 - 1.0)
        .collect();
    let student = Array::from_f32_slice(&s, &[1, 2, vocab as i32]);

    let got = SparseKlLoss::new()
        .compute_sampled(&targets, &student, temperature, None)
        .unwrap()
        .item::<f32>() as f64;
    drain();

    let emp = [
        [(5, 0.5), (1, 0.25), (6, 0.25)],
        [(0, 0.75), (2, 0.25), (0, 0.0)],
    ];
    let mut total = 0.0;
    for (t, row) in emp.iter().enumerate() {
        let q = softmax(&s[t * vocab..(t + 1) * vocab], temperature as f64);
        for &(id, p) in row {
            if p > 0.0 {
                total += p * (p / q[id]).ln();
            }
        }
    }
    let want = total / 2.0;
    assert!(
        (got - want).abs() <= 2e-5 * want.abs(),
        "got {got}, float64 reference {want}"
    );

    let dense: f32 = KlDivergenceLoss::new()
        .compute(&dense_logits(&targets, temperature), &student, temperature)
        .unwrap()
        .item();
    assert!(
        (got as f32 - dense).abs() <= 2e-5 * dense.abs(),
        "{got} vs {dense}"
    );

    // And the gradient is (q - p̂)/(T N).
    let (value, grads) = value_and_grad_explicit(
        |a: &[Array]| {
            SparseKlLoss::new()
                .compute_sampled(&targets, &a[0], temperature, None)
                .unwrap()
        },
        std::slice::from_ref(&student),
        &[],
    )
    .unwrap();
    value.eval();
    grads[0].eval();
    drain();
    let grad = grads[0].clone().to_f32_vec(2 * vocab).unwrap();
    for (t, row) in emp.iter().enumerate() {
        let q = softmax(&s[t * vocab..(t + 1) * vocab], temperature as f64);
        for i in 0..vocab {
            let p: f64 = row.iter().filter(|(id, _)| *id == i).map(|(_, p)| p).sum();
            let want = (q[i] - p) / (temperature as f64 * 2.0);
            assert!(
                (grad[t * vocab + i] as f64 - want).abs() <= 1e-5 + 1e-3 * want.abs(),
                "token {t} entry {i}"
            );
        }
    }
}

/// Dense logits whose softmax at `temperature` is the empirical distribution.
fn dense_logits(targets: &SparseTargets, temperature: f32) -> Array {
    let mut probs = targets.to_dense_probs();
    let p = probs.to_f32_vec(probs.size()).unwrap();
    let logits: Vec<f32> = p
        .iter()
        .map(|&p| if p > 0.0 { temperature * p.ln() } else { -1e10 })
        .collect();
    Array::from_f32_slice(&logits, probs.shape())
}

/// Draws taken at one temperature are not an estimate of the teacher at
/// another: the loss refuses the mismatch rather than train toward the wrong
/// distribution.
#[test]
#[serial]
fn the_loss_temperature_must_be_the_sampling_temperature() {
    let targets = SparseTargets::new(
        Array::from_f32_slice(&[1.0], &[1, 1, 1]),
        Array::from_i32_slice_shaped(&[2], &[1, 1, 1]),
        4,
        1.0,
    )
    .unwrap();
    let student = Array::from_f32_slice(&[0.0; 4], &[1, 1, 4]);
    let err = SparseKlLoss::new()
        .compute_sampled(&targets, &student, 2.0, None)
        .expect_err("T=2 against draws taken at T=1");
    assert!(err.to_string().contains("temperature"), "{err}");
}

/// The distiller's sampled path equals its dense path on the densified draws.
#[test]
#[serial]
fn the_distiller_with_sampled_targets_equals_the_dense_distiller() {
    let (seq, vocab) = (3_usize, 11_usize);
    let teacher: Vec<f32> = (0..seq * vocab)
        .map(|i| ((i * 7) % 13) as f32 * 0.3)
        .collect();
    let student: Vec<f32> = (0..seq * vocab)
        .map(|i| 2.0 * ((i as f32) * 0.37 + 0.1).cos())
        .collect();
    let shape = [1, seq as i32, vocab as i32];

    let dir = tempfile::tempdir().unwrap();
    let mut cache = LogitCache::new(dir.path(), CompressionMethod::Sampled, 6)
        .unwrap()
        .with_sampling_temperature(2.0);
    random::seed(9);
    cache
        .cache_sequence(0, &Array::from_f32_slice(&teacher, &shape))
        .unwrap();
    cache.set_metadata("m".into(), vocab, seq);
    cache.save_metadata().unwrap();
    let cache = LogitCache::load(dir.path()).unwrap();
    let targets = cache.load_sequence_sampled(0).unwrap();
    let dense = cache.load_sequence(0).unwrap();

    let distiller = Distiller::new(DistillConfig {
        teacher: "t".to_string(),
        student: "s".to_string(),
        method: DistillMethod::Online,
        loss: LossConfig {
            temperature: 2.0,
            alpha: 0.6,
            ..LossConfig::default()
        },
        offline: None,
        output_path: None,
        training: Default::default(),
    })
    .unwrap();
    let student = Array::from_f32_slice(&student, &shape);
    let labels = Array::from_i32_slice_shaped(&[3, -100, 7], &[1, seq as i32]);
    let weights = Array::from_f32_slice(&[1.0, 0.0, 1.0], &[1, seq as i32]);

    let via_dense = distiller
        .compute_loss(&dense, &student, Some(&labels), Some(&weights), 0, 1)
        .unwrap();
    let via_sampled = distiller
        .compute_loss_sampled(&targets, &student, Some(&labels), Some(&weights), 0, 1)
        .unwrap();
    let (a, b) = (
        via_dense.total.item::<f32>(),
        via_sampled.total.item::<f32>(),
    );
    drain();
    assert!(
        (a - b).abs() <= 2e-5 * a.abs().max(1e-3),
        "dense {a} vs sampled {b}"
    );
}
