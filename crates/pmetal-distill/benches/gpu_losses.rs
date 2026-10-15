//! Cost of the distillation losses: wall time and peak device memory of a
//! forward + backward pass, evaluated (a lazy graph that is never evaluated
//! measures nothing).
//!
//! Run with `cargo bench -p pmetal-distill`. Set `DISTILL_BENCH_TOKENS` and
//! `DISTILL_BENCH_VOCAB` to change the size (default 2048 tokens by 32768
//! entries); the largest realistic case is a few thousand tokens by a ~150k
//! vocabulary and needs several GB of device memory.
//!
//! Each case is timed `RUNS` times after a warm-up and reported by its median;
//! peak memory is the high-water mark above the live inputs. Compare cases
//! within one run: the machine is shared, and absolute times move with its load.

use std::time::Instant;

use pmetal_distill::losses::{
    DistillLoss, JensenShannonLoss, JsdSkewedLoss, KlDivergenceLoss, SoftCrossEntropyLoss,
    SparseKlLoss, TvdLoss,
};
use pmetal_distill::{CompressedLogits, CompressionMethod, LogitCompressor, SparseLogits};
use pmetal_runtime::compat::nn::value_and_grad_explicit;
use pmetal_runtime::compat::{Array, Dtype};
use pmetal_runtime::inline_array::{
    clear_cache, get_active_memory, get_peak_memory, reset_peak_memory,
};

const RUNS: usize = 7;
const TOP_K: usize = 128;
const TEMPERATURE: f32 = 2.0;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic logits with a realistic spread.
fn logits(tokens: usize, vocab: usize, seed: f32, dtype: Dtype) -> Array {
    let ids = Array::arange((tokens * vocab) as i32, Dtype::Float32.as_i32());
    let x = ids
        .multiply(&Array::from_f32(0.000_173 + seed * 1e-5))
        .sin()
        .multiply(&Array::from_f32(3.0));
    x.reshape(&[1, tokens as i32, vocab as i32])
        .as_dtype(dtype.as_i32())
}

/// One timed configuration: student logits in, `(loss, gradient)` out.
struct Case {
    name: &'static str,
    run: Box<dyn Fn(&Array) -> (Array, Array)>,
}

impl Case {
    fn new(name: &'static str, loss: impl Fn(&Array) -> Array + 'static) -> Self {
        Self {
            name,
            run: Box::new(move |student: &Array| {
                let (value, mut grads) = value_and_grad_explicit(
                    |a: &[Array]| loss(&a[0]),
                    std::slice::from_ref(student),
                    &[],
                )
                .unwrap();
                (value, grads.remove(0))
            }),
        }
    }

    fn dense(name: &'static str, loss: impl DistillLoss + 'static, teacher: &Array) -> Self {
        let teacher = teacher.clone();
        Self::new(name, move |student| {
            loss.compute(&teacher, student, TEMPERATURE).unwrap()
        })
    }
}

fn measure(case: &Case, student: &Array) {
    // Warm-up (kernel compilation, allocator).
    for _ in 0..2 {
        let (loss, grad) = (case.run)(student);
        loss.eval();
        grad.eval();
    }
    clear_cache();
    let base = get_active_memory();
    reset_peak_memory();

    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        let (loss, grad) = (case.run)(student);
        loss.eval();
        grad.eval();
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let peak = get_peak_memory().saturating_sub(base);
    println!(
        "{:<36} {:>9.1} ms (min {:>8.1})   peak +{:>7.0} MiB",
        case.name,
        times[RUNS / 2],
        times[0],
        peak as f64 / (1024.0 * 1024.0),
    );
}

/// The teacher's top-k, as an offline run would have cached it.
fn cached_teacher(teacher: &Array, tokens: usize) -> SparseLogits {
    let compressed = LogitCompressor::new(CompressionMethod::TopK, TOP_K)
        .compress(teacher)
        .unwrap();
    let CompressedLogits::TopKSparse {
        values,
        indices,
        k,
        shape,
    } = compressed
    else {
        unreachable!("top-k compression");
    };
    SparseLogits::new(
        Array::from_f32_slice(&values, &[1, tokens as i32, k as i32]),
        Array::from_i32_slice_shaped(&indices, &[1, tokens as i32, k as i32]),
        shape[2] as usize,
    )
    .unwrap()
}

fn main() {
    let tokens = env_usize("DISTILL_BENCH_TOKENS", 2048);
    let vocab = env_usize("DISTILL_BENCH_VOCAB", 32768);
    println!("{tokens} tokens x {vocab} vocab, bf16 logits, forward + backward, median of {RUNS}");

    let teacher = logits(tokens, vocab, 0.0, Dtype::Bfloat16);
    let student = logits(tokens, vocab, 1.0, Dtype::Bfloat16);
    teacher.eval();
    student.eval();

    let sparse = cached_teacher(&teacher, tokens);
    let cases = [
        Case::dense(
            "forward KL, dense teacher",
            KlDivergenceLoss::new(),
            &teacher,
        ),
        Case::dense(
            "reverse KL, dense teacher",
            KlDivergenceLoss::reverse(),
            &teacher,
        ),
        Case::dense(
            "Jensen-Shannon, dense teacher",
            JensenShannonLoss::new(),
            &teacher,
        ),
        Case::dense("JSD(0.1), dense teacher", JsdSkewedLoss::new(0.1), &teacher),
        Case::dense(
            "soft cross-entropy, dense teacher",
            SoftCrossEntropyLoss::new(),
            &teacher,
        ),
        Case::dense("TVD, dense teacher", TvdLoss::new(), &teacher),
        Case::new("forward KL, top-128 sparse teacher", move |student| {
            SparseKlLoss::new()
                .compute(&sparse, student, TEMPERATURE, None)
                .unwrap()
        }),
    ];
    for case in &cases {
        measure(case, &student);
    }

    compare_top_k_selection(&teacher, tokens, vocab);
}

/// Writing a top-k cache: the device selection against selecting on the host
/// (a partial sort of the vocabulary per token), which is what the cache did
/// before. One run each; the copy of the logits to the host is included in
/// both, as it is in the cache.
fn compare_top_k_selection(teacher: &Array, tokens: usize, vocab: usize) {
    let logits = teacher.as_dtype(Dtype::Float32.as_i32());
    logits.eval();
    println!("\ntop-{TOP_K} selection for the cache, {tokens} x {vocab}:");

    let start = Instant::now();
    let compressed = LogitCompressor::new(CompressionMethod::TopK, TOP_K)
        .compress(&logits)
        .unwrap();
    let device_ms = start.elapsed().as_secs_f64() * 1e3;

    let start = Instant::now();
    let mut data = logits.clone();
    let host = data.to_f32_vec(tokens * vocab).unwrap();
    let mut kept = Vec::with_capacity(tokens * TOP_K);
    for row in host.chunks(vocab) {
        let mut order: Vec<usize> = (0..vocab).collect();
        order.select_nth_unstable_by(TOP_K - 1, |&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        order[..TOP_K].sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        kept.extend(order[..TOP_K].iter().map(|&i| row[i]));
    }
    let host_ms = start.elapsed().as_secs_f64() * 1e3;

    // The two must agree on the kept logits (ties may be broken differently,
    // so the ids can differ where logits are equal).
    let CompressedLogits::TopKSparse { values, .. } = compressed else {
        unreachable!("top-k compression");
    };
    assert_eq!(values, kept, "device and host selections differ");
    println!("  device: {device_ms:>9.1} ms\n  host:   {host_ms:>9.1} ms");
}
