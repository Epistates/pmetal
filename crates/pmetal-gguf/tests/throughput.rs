//! Decode and quantize throughput. Ignored by default; run in release mode:
//! `cargo test --release -p pmetal-gguf --test throughput -- --ignored --nocapture`.

use std::time::Instant;

use pmetal_gguf::{GgmlType, dequant, quantize};

const ELEMENTS: usize = 1 << 25; // 32M weights

fn signal(n: usize) -> Vec<f32> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
        })
        .collect()
}

fn best_of<T>(runs: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::MAX;
    let mut last = None;
    for _ in 0..runs {
        let t = Instant::now();
        last = Some(f());
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, last.unwrap())
}

#[test]
#[ignore = "benchmark"]
fn throughput() {
    let data = signal(ELEMENTS);
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    println!(
        "{} threads available, {} M elements",
        rayon::current_num_threads(),
        ELEMENTS >> 20
    );

    for dtype in [
        GgmlType::Q4_0,
        GgmlType::Q8_0,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::Iq4Xs,
    ] {
        let (tq, bytes) = best_of(2, || quantize::quantize(&data, dtype).unwrap());
        let mb = bytes.len() as f64 / 1e6;
        let shape = [(ELEMENTS / 4096) as i32, 4096];

        let (t_par, out) = best_of(5, || dequant::dequantize(&bytes, dtype, &shape).unwrap());
        let (t_one, _) = best_of(3, || {
            pool1.install(|| dequant::dequantize(&bytes, dtype, &shape).unwrap())
        });
        let mut buf = vec![0.0f32; ELEMENTS];
        let (t_into, _) = best_of(5, || {
            dequant::dequantize_into(&bytes, dtype, &shape, &mut buf).unwrap()
        });
        assert_eq!(out.len(), ELEMENTS);
        println!(
            "{dtype:?}: dequantize {:.2} Gelem/s ({:.1} GB/s read) on all threads, {:.2} Gelem/s on one, into-buffer {:.2} Gelem/s; quantize {:.1} Melem/s",
            ELEMENTS as f64 / t_par / 1e9,
            mb / 1e3 / t_par,
            ELEMENTS as f64 / t_one / 1e9,
            ELEMENTS as f64 / t_into / 1e9,
            ELEMENTS as f64 / tq / 1e6,
        );
    }
}
