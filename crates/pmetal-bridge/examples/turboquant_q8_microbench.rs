use pmetal_bridge::InlineArray;
use pmetal_bridge::inline_array::reset_peak_memory;
use std::env;
use std::time::Instant;

const DIM: usize = 128;
const QJL_WORDS: usize = 4;

#[derive(Clone, Copy)]
struct Config {
    q_heads: u32,
    kv_heads: u32,
    n_seq: u32,
    cache_seq_capacity: u32,
    iters: usize,
    warmup: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            q_heads: 16,
            kv_heads: 8,
            n_seq: 2048,
            cache_seq_capacity: 2048,
            iters: 20,
            warmup: 5,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--q-heads" => cfg.q_heads = args.next().unwrap().parse().unwrap(),
            "--kv-heads" => cfg.kv_heads = args.next().unwrap().parse().unwrap(),
            "--seq" => {
                let seq: u32 = args.next().unwrap().parse().unwrap();
                cfg.n_seq = seq;
                cfg.cache_seq_capacity = seq;
            }
            "--cache-seq-capacity" => {
                cfg.cache_seq_capacity = args.next().unwrap().parse().unwrap()
            }
            "--iters" => cfg.iters = args.next().unwrap().parse().unwrap(),
            "--warmup" => cfg.warmup = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg: {other}"),
        }
    }
    cfg
}

fn build_f32(count: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i % 97) as f32) / 97.0 - 0.5) * scale)
        .collect()
}

fn build_u8(count: usize, modulo: usize) -> Vec<u8> {
    (0..count).map(|i| (i % modulo) as u8).collect()
}

fn build_u32_sign_words(rows: usize, seq: usize) -> Vec<u32> {
    let mut out = vec![0u32; rows * QJL_WORDS * seq];
    for row in 0..rows {
        for word in 0..QJL_WORDS {
            for s in 0..seq {
                let base_bit = ((row + s + word) & 1) as u32;
                let mut packed = 0u32;
                for bit in 0..32 {
                    let sign = (base_bit + bit as u32) & 1;
                    packed |= sign << bit;
                }
                out[(row * QJL_WORDS + word) * seq + s] = packed;
            }
        }
    }
    out
}

fn pack_q8_keybytes(indices: &[u8], qjl_signs: &[u32], rows: usize, seq: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows * DIM * seq];
    for row in 0..rows {
        for d in 0..DIM {
            let word = qjl_signs[(row * QJL_WORDS + (d >> 5)) * seq..][..seq].to_vec();
            for s in 0..seq {
                let sign = ((word[s] >> (d & 31)) & 1) as u8;
                let idx = indices[(row * DIM + d) * seq + s] & 0x7f;
                out[(row * DIM + d) * seq + s] = idx | (sign << 7);
            }
        }
    }
    out
}

fn bench_eval_ms(mut f: impl FnMut() -> InlineArray, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        let out = f();
        out.eval();
    }
    let mut total_ms = 0.0;
    for _ in 0..iters {
        let out = f();
        let t0 = Instant::now();
        out.eval();
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    total_ms / iters as f64
}

fn main() {
    let cfg = parse_args();
    assert!(cfg.q_heads % cfg.kv_heads == 0);
    assert!(cfg.cache_seq_capacity >= cfg.n_seq);

    let n_rows = cfg.q_heads as usize;
    let kv_rows = cfg.kv_heads as usize;
    let seq_cap = cfg.cache_seq_capacity as usize;

    let query_rot =
        InlineArray::from_f32_slice(&build_f32(n_rows * DIM, 0.2), &[n_rows as i32, DIM as i32]);
    let query_proj =
        InlineArray::from_f32_slice(&build_f32(n_rows * DIM, 0.1), &[n_rows as i32, DIM as i32]);
    let key_indices = InlineArray::from_u8_slice(
        &build_u8(kv_rows * DIM * seq_cap, 128),
        &[kv_rows as i32, DIM as i32, seq_cap as i32],
    );
    let key_qjl_signs = InlineArray::from_u32_slice(
        &build_u32_sign_words(kv_rows, seq_cap),
        &[kv_rows as i32, QJL_WORDS as i32, seq_cap as i32],
    );
    let key_bytes = InlineArray::from_u8_slice(
        &pack_q8_keybytes(
            &build_u8(kv_rows * DIM * seq_cap, 128),
            &build_u32_sign_words(kv_rows, seq_cap),
            kv_rows,
            seq_cap,
        ),
        &[kv_rows as i32, DIM as i32, seq_cap as i32],
    );
    let key_norms = InlineArray::from_f32_slice(
        &build_f32(kv_rows * seq_cap, 0.3),
        &[kv_rows as i32, seq_cap as i32],
    );
    let key_residual_norms = InlineArray::from_f32_slice(
        &build_f32(kv_rows * seq_cap, 0.05),
        &[kv_rows as i32, seq_cap as i32],
    );
    let key_slot_scale = InlineArray::from_f32_slice(
        &vec![1.0f32; kv_rows * seq_cap],
        &[kv_rows as i32, seq_cap as i32],
    );
    let key_codebook = InlineArray::from_f32_slice(&build_f32(128, 1.0), &[128]);
    let value_indices = InlineArray::from_u8_slice(
        &build_u8(kv_rows * DIM * seq_cap, 256),
        &[kv_rows as i32, DIM as i32, seq_cap as i32],
    );
    let value_norms = InlineArray::from_f32_slice(
        &build_f32(kv_rows * seq_cap, 0.4),
        &[kv_rows as i32, seq_cap as i32],
    );
    let value_codebook = InlineArray::from_f32_slice(&build_f32(256, 1.5), &[256]);

    let split = InlineArray::turboquant_attention_q8_d128_2pass(
        &query_rot,
        &query_proj,
        &key_indices,
        &key_qjl_signs,
        &key_norms,
        &key_residual_norms,
        &key_slot_scale,
        &key_codebook,
        &value_indices,
        &value_norms,
        &value_codebook,
        cfg.q_heads,
        cfg.n_seq,
        cfg.cache_seq_capacity,
        cfg.q_heads,
        cfg.kv_heads,
        0.0625,
    );

    let mut packed = InlineArray::turboquant_attention_q8_d128_packed_keys_2pass(
        &query_rot,
        &query_proj,
        &key_bytes,
        &key_norms,
        &key_residual_norms,
        &key_slot_scale,
        &key_codebook,
        &value_indices,
        &value_norms,
        &value_codebook,
        cfg.q_heads,
        cfg.n_seq,
        cfg.cache_seq_capacity,
        cfg.q_heads,
        cfg.kv_heads,
        0.0625,
    )
    .expect("packed q8 primitive");
    packed.eval();

    let packed_vals = packed.to_f32_vec(n_rows * DIM).expect("packed values");
    let (split_ms, max_abs_diff) = if let Some(mut split) = split {
        split.eval();
        let split_vals = split.to_f32_vec(n_rows * DIM).expect("split values");
        let max_abs_diff = split_vals
            .iter()
            .zip(packed_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let split_ms = bench_eval_ms(
            || {
                InlineArray::turboquant_attention_q8_d128_2pass(
                    &query_rot,
                    &query_proj,
                    &key_indices,
                    &key_qjl_signs,
                    &key_norms,
                    &key_residual_norms,
                    &key_slot_scale,
                    &key_codebook,
                    &value_indices,
                    &value_norms,
                    &value_codebook,
                    cfg.q_heads,
                    cfg.n_seq,
                    cfg.cache_seq_capacity,
                    cfg.q_heads,
                    cfg.kv_heads,
                    0.0625,
                )
                .expect("split bench")
            },
            cfg.warmup,
            cfg.iters,
        );
        (Some(split_ms), Some(max_abs_diff))
    } else {
        (None, None)
    };

    reset_peak_memory();
    let packed_ms = bench_eval_ms(
        || {
            InlineArray::turboquant_attention_q8_d128_packed_keys_2pass(
                &query_rot,
                &query_proj,
                &key_bytes,
                &key_norms,
                &key_residual_norms,
                &key_slot_scale,
                &key_codebook,
                &value_indices,
                &value_norms,
                &value_codebook,
                cfg.q_heads,
                cfg.n_seq,
                cfg.cache_seq_capacity,
                cfg.q_heads,
                cfg.kv_heads,
                0.0625,
            )
            .expect("packed bench")
        },
        cfg.warmup,
        cfg.iters,
    );

    match (split_ms, max_abs_diff) {
        (Some(split_ms), Some(max_abs_diff)) => println!(
            "q_heads={} kv_heads={} seq={} split_ms={:.3} packed_ms={:.3} speedup={:.3} max_abs_diff={:.6}",
            cfg.q_heads,
            cfg.kv_heads,
            cfg.n_seq,
            split_ms,
            packed_ms,
            split_ms / packed_ms,
            max_abs_diff,
        ),
        _ => println!(
            "q_heads={} kv_heads={} seq={} split_ms=unavailable packed_ms={:.3}",
            cfg.q_heads, cfg.kv_heads, cfg.n_seq, packed_ms,
        ),
    }
}
