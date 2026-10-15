//! The offline logit cache: what goes to disk and what comes back.

use pmetal_distill::{CompressionMethod, LogitCache, LogitCompressor};
use pmetal_runtime::compat::Array;
use serial_test::serial;

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

fn read(a: &Array, n: usize) -> Vec<f32> {
    let mut a = a.clone();
    let v = a.to_f32_vec(n).unwrap();
    drain();
    v
}

fn logits(tokens: usize, vocab: usize, scale: f32) -> Vec<f32> {
    (0..tokens * vocab)
        .map(|i| {
            let t = (i / vocab) as f32;
            scale * ((i as f32) * 0.173 + t).sin() + 0.01 * ((i % vocab) as f32)
        })
        .collect()
}

/// Top-k keeps the leading axes: `[batch, seq, vocab]` in, `[batch, seq,
/// vocab]` out. It used to come back flattened to `[batch * seq, vocab]`.
#[test]
#[serial]
fn top_k_round_trip_keeps_the_shape() {
    let data = logits(6, 11, 4.0);
    let original = Array::from_f32_slice(&data, &[2, 3, 11]);
    let compressor = LogitCompressor::new(CompressionMethod::TopK, 4);
    let compressed = compressor.compress(&original).unwrap();
    let back = compressor.decompress(&compressed, 11).unwrap();
    assert_eq!(back.shape(), &[2, 3, 11]);
}

/// 4-bit codes have 16 levels per token, so the error of a decoded logit is
/// at most half of that token's own step, `(max - min) / 30`, however
/// different the tokens' ranges are. A single scale for the whole tensor
/// spends the same step on a token that spans 1 logit as on one that spans
/// 60.
#[test]
#[serial]
fn int4_error_is_bounded_by_each_tokens_own_range() {
    let vocab = 32;
    let mut data = logits(3, vocab, 0.4); // a narrow token ...
    for v in &mut data[vocab..2 * vocab] {
        *v *= 100.0; // ... and a wide one
    }
    let original = Array::from_f32_slice(&data, &[3, vocab as i32]);
    let compressor = LogitCompressor::new(CompressionMethod::Int4, 0);
    let back = compressor
        .decompress(&compressor.compress(&original).unwrap(), vocab)
        .unwrap();
    let decoded = read(&back, data.len());

    for t in 0..3 {
        let row = &data[t * vocab..(t + 1) * vocab];
        let min = row.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let half_step = (max - min) / 30.0;
        for (i, (a, b)) in row.iter().zip(&decoded[t * vocab..]).enumerate() {
            assert!(
                (a - b).abs() <= half_step * 1.001 + 1e-6,
                "token {t} entry {i}: {a} decoded as {b}, allowed error {half_step}"
            );
        }
    }
}

/// A constant row has no range to quantize: it must come back as itself, not
/// as NaN from a zero scale.
#[test]
#[serial]
fn quantizing_a_constant_row_is_exact() {
    let original = Array::from_f32_slice(&[2.5_f32; 8], &[2, 4]);
    for method in [CompressionMethod::Int8, CompressionMethod::Int4] {
        let compressor = LogitCompressor::new(method.clone(), 0);
        let back = compressor
            .decompress(&compressor.compress(&original).unwrap(), 4)
            .unwrap();
        let decoded = read(&back, 8);
        assert!(
            decoded.iter().all(|v| (*v - 2.5).abs() < 1e-6),
            "{method:?}: {decoded:?}"
        );
    }
}

/// A NaN logit means the teacher forward diverged; quantizing it would spread
/// the damage over the row's scale. The quantizers refuse it.
#[test]
#[serial]
fn quantizers_refuse_non_finite_logits() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let original = Array::from_f32_slice(&[1.0, bad, 3.0, 4.0], &[1, 4]);
        for method in [CompressionMethod::Int8, CompressionMethod::Int4] {
            let compressor = LogitCompressor::new(method.clone(), 0);
            assert!(
                compressor.compress(&original).is_err(),
                "{method:?} accepted {bad}"
            );
        }
    }
}

/// A cache whose metadata names a compression this build does not know is
/// corrupt, not a cache of uncompressed logits.
#[test]
#[serial]
fn unknown_compression_in_metadata_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("metadata.json"),
        r#"{"num_sequences":1,"compression":"zstd9","top_k":8,"vocab_size":4,"max_seq_len":2,"model":"m"}"#,
    )
    .unwrap();
    let err = LogitCache::load(dir.path())
        .err()
        .expect("must be rejected");
    assert!(err.to_string().contains("zstd9"), "{err}");
}

/// Distinct values, so the top-k set and its order are unambiguous.
fn distinct(tokens: usize, vocab: usize) -> Vec<f32> {
    (0..tokens * vocab)
        .map(|i| {
            // A permutation-driven spread: i * 37 mod p is injective for i < p.
            let p = 10007_usize;
            ((i * 37) % p) as f32 / 400.0 - 12.0
        })
        .collect()
}

/// The device top-k is the host top-k: the same ids and values, largest
/// first, for k below, at and above the vocabulary.
#[test]
#[serial]
fn device_top_k_matches_a_host_sort() {
    use pmetal_distill::CompressedLogits;

    let (batch, seq, vocab) = (2_usize, 3_usize, 9_usize);
    let data = distinct(batch * seq, vocab);
    let logits = Array::from_f32_slice(&data, &[batch as i32, seq as i32, vocab as i32]);

    for requested in [1_usize, 4, 9, 40] {
        let compressed = LogitCompressor::new(CompressionMethod::TopK, requested)
            .compress(&logits)
            .unwrap();
        let CompressedLogits::TopKSparse {
            values,
            indices,
            k,
            shape,
        } = compressed
        else {
            panic!("expected TopKSparse");
        };
        let want_k = requested.min(vocab);
        assert_eq!(k, want_k);
        assert_eq!(shape, vec![batch as i32, seq as i32, vocab as i32]);

        for t in 0..batch * seq {
            let row = &data[t * vocab..(t + 1) * vocab];
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            for j in 0..want_k {
                assert_eq!(
                    indices[t * k + j] as usize,
                    order[j],
                    "k={requested} t={t} j={j}"
                );
                assert_eq!(values[t * k + j], row[order[j]]);
            }
        }
    }
}

/// A NaN logit is refused by the top-k path too; `-inf` (a masked token) is
/// fine, it is just never in the top k.
#[test]
#[serial]
fn top_k_rejects_nan_and_tolerates_masked_tokens() {
    let compressor = LogitCompressor::new(CompressionMethod::TopK, 2);
    let nan = Array::from_f32_slice(&[1.0, f32::NAN, 3.0, 4.0], &[1, 4]);
    assert!(compressor.compress(&nan).is_err());

    let masked = Array::from_f32_slice(&[1.0, f32::NEG_INFINITY, 3.0, 4.0], &[1, 4]);
    let back = compressor
        .decompress(&compressor.compress(&masked).unwrap(), 4)
        .unwrap();
    let v = read(&back, 4);
    assert_eq!((v[2], v[3]), (3.0, 4.0));
}

/// Sparse cache -> sparse loss: the full path from `cache_sequence` to the
/// loss value, against a float64 evaluation of
/// `KL(softmax(top-k logits / T) || softmax(student / T))`.
#[test]
#[serial]
fn sparse_kl_from_a_cache_matches_float64() {
    use pmetal_distill::losses::SparseKlLoss;

    let (seq, vocab, k) = (4_usize, 17_usize, 5_usize);
    let teacher_logits = distinct(seq, vocab);
    let student_logits: Vec<f32> = (0..seq * vocab)
        .map(|i| 2.0 * ((i as f32) * 0.61 + 0.3).cos())
        .collect();

    let dir = tempfile::tempdir().unwrap();
    let mut cache = LogitCache::new(dir.path(), CompressionMethod::TopK, k).unwrap();
    cache
        .cache_sequence(
            0,
            &Array::from_f32_slice(&teacher_logits, &[1, seq as i32, vocab as i32]),
        )
        .unwrap();
    cache.set_metadata("m".into(), vocab, seq);
    cache.save_metadata().unwrap();

    let cache = LogitCache::load(dir.path()).unwrap();
    let sparse = cache.load_sequence_sparse(0).unwrap();
    assert_eq!(sparse.shape(), &[1, seq as i32, k as i32]);
    assert_eq!(sparse.vocab_size(), vocab);

    let student = Array::from_f32_slice(&student_logits, &[1, seq as i32, vocab as i32]);
    let weights = Array::from_f32_slice(&[1.0, 1.0, 1.0, 0.0], &[1, seq as i32]);
    for temperature in [1.0_f32, 2.5] {
        let got = SparseKlLoss::new()
            .compute(&sparse, &student, temperature, Some(&weights))
            .unwrap()
            .item::<f32>() as f64;
        drain();

        let t = temperature as f64;
        let mut total = 0.0;
        for tok in 0..3 {
            let row = &teacher_logits[tok * vocab..(tok + 1) * vocab];
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            let kept: Vec<usize> = order[..k].to_vec();
            let z: Vec<f64> = kept.iter().map(|&i| row[i] as f64 / t).collect();
            let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let zsum: f64 = z.iter().map(|x| (x - zmax).exp()).sum();
            let s: Vec<f64> = student_logits[tok * vocab..(tok + 1) * vocab]
                .iter()
                .map(|x| *x as f64 / t)
                .collect();
            let smax = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = smax + s.iter().map(|x| (x - smax).exp()).sum::<f64>().ln();
            for (j, &i) in kept.iter().enumerate() {
                let log_p = z[j] - zmax - zsum.ln();
                total += log_p.exp() * (log_p - (s[i] - lse));
            }
        }
        let want = total / 3.0;
        assert!(
            (got - want).abs() <= 2e-5 * want.abs().max(1e-3),
            "T={temperature}: got {got}, float64 reference {want}"
        );
    }
}

/// Its gradient is `(q - p) / (T * N)` on the kept tokens' logits and `q / (T * N)`
/// elsewhere, with `p` the renormalized top-k teacher and `q` the full student.
#[test]
#[serial]
fn sparse_kl_gradient_is_softmax_minus_teacher() {
    use pmetal_distill::SparseLogits;
    use pmetal_distill::losses::SparseKlLoss;
    use pmetal_runtime::compat::nn::value_and_grad_explicit;

    let (vocab, k, temperature) = (7_usize, 3_usize, 2.0_f32);
    let teacher = SparseLogits::new(
        Array::from_f32_slice(&[2.0, 1.0, -1.0, 0.5, 0.0, -0.5], &[1, 2, k as i32]),
        Array::from_i32_slice_shaped(&[5, 1, 6, 0, 3, 2], &[1, 2, k as i32]),
        vocab,
    )
    .unwrap();
    let student: Vec<f32> = (0..2 * vocab)
        .map(|i| ((i * 5) % 7) as f32 * 0.4 - 1.0)
        .collect();

    let (value, grads) = value_and_grad_explicit(
        |a: &[Array]| {
            SparseKlLoss::new()
                .compute(&teacher, &a[0], temperature, None)
                .unwrap()
        },
        &[Array::from_f32_slice(&student, &[1, 2, vocab as i32])],
        &[],
    )
    .unwrap();
    value.eval();
    grads[0].eval();
    drain();
    let grad = read(&grads[0], 2 * vocab);

    let kept = [[5_usize, 1, 6], [0, 3, 2]];
    let tvals = [[2.0_f64, 1.0, -1.0], [0.5, 0.0, -0.5]];
    for tok in 0..2 {
        let z: Vec<f64> = tvals[tok].iter().map(|x| x / temperature as f64).collect();
        let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let zsum: f64 = z.iter().map(|x| (x - zmax).exp()).sum();
        let s: Vec<f64> = student[tok * vocab..(tok + 1) * vocab]
            .iter()
            .map(|x| *x as f64 / temperature as f64)
            .collect();
        let smax = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ssum: f64 = s.iter().map(|x| (x - smax).exp()).sum();
        for i in 0..vocab {
            let q = (s[i] - smax).exp() / ssum;
            let p = kept[tok]
                .iter()
                .position(|&j| j == i)
                .map_or(0.0, |j| (z[j] - zmax).exp() / zsum);
            let want = (q - p) / (temperature as f64 * 2.0);
            let got = grad[tok * vocab + i] as f64;
            assert!(
                (got - want).abs() <= 1e-5 + 1e-3 * want.abs(),
                "token {tok} entry {i}: {got} vs {want}"
            );
        }
    }
}

/// The distiller's sparse path is its dense path on the densified teacher:
/// same total, soft and hard terms, hard labels included.
#[test]
#[serial]
fn the_distiller_with_a_sparse_teacher_equals_the_dense_distiller() {
    use pmetal_distill::{DistillConfig, DistillMethod, Distiller, LossConfig, LossType};

    let (seq, vocab, k) = (3_usize, 11_usize, 4_usize);
    let teacher_logits = distinct(seq, vocab);
    let student_logits: Vec<f32> = (0..seq * vocab)
        .map(|i| 2.0 * ((i as f32) * 0.37 + 0.1).cos())
        .collect();
    let shape = [1, seq as i32, vocab as i32];
    let compressor = LogitCompressor::new(CompressionMethod::TopK, k);
    let compressed = compressor
        .compress(&Array::from_f32_slice(&teacher_logits, &shape))
        .unwrap();
    let dense_teacher = compressor.decompress(&compressed, vocab).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let mut cache = LogitCache::new(dir.path(), CompressionMethod::TopK, k).unwrap();
    cache
        .cache_sequence(0, &Array::from_f32_slice(&teacher_logits, &shape))
        .unwrap();
    let sparse = cache.load_sequence_sparse(0).unwrap();

    let make = |loss_type: LossType, reverse_kl: bool| {
        Distiller::new(DistillConfig {
            teacher: "t".to_string(),
            student: "s".to_string(),
            method: DistillMethod::Online,
            loss: LossConfig {
                loss_type,
                reverse_kl,
                temperature: 2.0,
                alpha: 0.7,
                ..LossConfig::default()
            },
            offline: None,
            output_path: None,
            training: Default::default(),
        })
        .unwrap()
    };

    let student = Array::from_f32_slice(&student_logits, &shape);
    let labels = Array::from_i32_slice_shaped(&[3, -100, 7], &[1, seq as i32]);
    let weights = Array::from_f32_slice(&[1.0, 0.0, 1.0], &[1, seq as i32]);
    let distiller = make(LossType::KlDivergence, false);
    let dense = distiller
        .compute_loss(
            &dense_teacher,
            &student,
            Some(&labels),
            Some(&weights),
            0,
            1,
        )
        .unwrap();
    let sparse_out = distiller
        .compute_loss_sparse(&sparse, &student, Some(&labels), Some(&weights), 0, 1)
        .unwrap();
    for (name, a, b) in [
        ("total", &dense.total, &sparse_out.total),
        ("soft", &dense.soft, &sparse_out.soft),
        (
            "hard",
            dense.hard.as_ref().unwrap(),
            sparse_out.hard.as_ref().unwrap(),
        ),
    ] {
        let (a, b) = (a.item::<f32>(), b.item::<f32>());
        drain();
        assert!(
            (a - b).abs() <= 1e-5 * a.abs().max(1e-3),
            "{name}: dense {a} vs sparse {b}"
        );
    }

    // Losses that need the whole teacher are refused, not swapped.
    for unsupported in [
        make(LossType::KlDivergence, true),
        make(LossType::JensenShannon, false),
    ] {
        assert!(
            unsupported
                .compute_loss_sparse(&sparse, &student, None, None, 0, 1)
                .is_err()
        );
    }
}

/// Collating sequences of different lengths pads, truncates and keeps ids.
#[test]
#[serial]
fn stacking_pads_and_truncates() {
    use pmetal_distill::SparseLogits;

    let a = SparseLogits::new(
        Array::from_f32_slice(&[3.0, 2.0, 1.5, 1.0], &[2, 2]),
        Array::from_i32_slice_shaped(&[4, 1, 2, 0], &[2, 2]),
        6,
    )
    .unwrap();
    let b = SparseLogits::new(
        Array::from_f32_slice(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0], &[1, 3, 2]),
        Array::from_i32_slice_shaped(&[5, 3, 1, 0, 2, 4], &[1, 3, 2]),
        6,
    )
    .unwrap();
    let stacked = SparseLogits::stack(&[a, b], 3).unwrap();
    assert_eq!(stacked.shape(), &[2, 3, 2]);
    assert_eq!(
        read(stacked.values(), 12),
        vec![3.0, 2.0, 1.5, 1.0, 0.0, 0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
    );
    let ids = stacked.indices().as_slice::<i32>().to_vec();
    assert_eq!(ids, vec![4, 1, 2, 0, 0, 0, 5, 3, 1, 0, 2, 4]);
}

/// A sequence file appears whole or not at all: writes go through a temporary
/// name, so a crash cannot leave a truncated file that `has_sequence` accepts.
#[test]
#[serial]
fn writing_a_sequence_leaves_no_temporary_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut cache = LogitCache::new(dir.path(), CompressionMethod::TopK, 3).unwrap();
    let data = logits(2, 8, 3.0);
    cache
        .cache_sequence(0, &Array::from_f32_slice(&data, &[2, 8]))
        .unwrap();
    cache.set_metadata("m".into(), 8, 2);
    cache.save_metadata().unwrap();

    let names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        names.iter().all(|n| !n.contains("tmp")),
        "leftover temporary file: {names:?}"
    );
    assert!(cache.has_sequence(0));

    let reloaded = LogitCache::load(dir.path()).unwrap();
    let back = reloaded.load_sequence(0).unwrap();
    assert_eq!(back.shape(), &[2, 8]);
}
