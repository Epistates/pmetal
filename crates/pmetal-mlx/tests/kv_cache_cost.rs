//! What a decode step costs through each KV cache mode, at a realistic size.
//!
//! Not regression tests: they assert only that the work ran. Run by hand on a
//! quiet machine:
//!
//! ```text
//! cargo test -p pmetal-mlx --release --test kv_cache_cost -- --ignored --nocapture
//! ```
//!
//! One decode step is, for each of 8 layers, a cache update and attention of
//! 32 query heads over 8 KV heads of dimension 128 in bf16, evaluated before
//! the next step, after a prompt of 8192 tokens.

use std::time::Instant;

use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig, TurboQuantConfig};
use pmetal_runtime::compat::{Array, Dtype};

const LAYERS: usize = 8;
const HQ: i32 = 32;
const HKV: i32 = 8;
const DIM: i32 = 128;
const PROMPT: i32 = 8192;
const STEPS: usize = 60;

fn random(shape: &[i32]) -> Array {
    Array::random_normal(shape, Dtype::Bfloat16.as_i32())
}

fn attention_config() -> FusedAttentionConfig {
    FusedAttentionConfig::new(HQ, HKV, DIM).with_mask_type(AttentionMaskType::Causal)
}

/// Milliseconds per decode step of `config`, attending through `attend`.
fn decode_ms(
    config: KVCacheConfig,
    attend: impl Fn(&mut KVCache, usize, &Array, &Array, &Array) -> Array,
) -> f64 {
    let mut cache = KVCache::new(config.with_dtype(Dtype::Bfloat16));
    // The prompt, in the chunks a prefill uses.
    for chunk in 0..(PROMPT / 2048) {
        let _ = chunk;
        for layer in 0..LAYERS {
            let k = random(&[1, HKV, 2048, DIM]);
            let v = random(&[1, HKV, 2048, DIM]);
            let (keys, values) = cache.update_and_fetch(layer, &k, &v).unwrap();
            keys.eval();
            values.eval();
        }
    }
    let step = |cache: &mut KVCache| {
        let mut outputs = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let q = random(&[1, HQ, 1, DIM]);
            let k = random(&[1, HKV, 1, DIM]);
            let v = random(&[1, HKV, 1, DIM]);
            outputs.push(attend(cache, layer, &q, &k, &v));
        }
        for out in &outputs {
            out.eval();
        }
    };
    for _ in 0..8 {
        step(&mut cache);
    }
    let start = Instant::now();
    for _ in 0..STEPS {
        step(&mut cache);
    }
    start.elapsed().as_secs_f64() * 1000.0 / STEPS as f64
}

/// The path every architecture has: update, fetch dense keys, attend.
fn dense_attend(cache: &mut KVCache, layer: usize, q: &Array, k: &Array, v: &Array) -> Array {
    let (keys, values) = cache.update_and_fetch(layer, k, v).unwrap();
    fused_sdpa(q, &keys, &values, &attention_config(), None).unwrap()
}

/// What a model's attention layer does: ask the cache to attend to its
/// compressed rows, and fall back to the dense path when it declines.
fn compressed_attend(cache: &mut KVCache, layer: usize, q: &Array, k: &Array, v: &Array) -> Array {
    match cache
        .try_compressed_attention(layer, q, k, v, &attention_config())
        .unwrap()
    {
        Some(out) => out,
        None => dense_attend(cache, layer, q, k, v),
    }
}

#[test]
#[ignore = "timing; run by hand"]
fn dense_cache_decode() {
    let ms = decode_ms(
        KVCacheConfig::new(LAYERS, 16384, HKV as usize, DIM as usize),
        dense_attend,
    );
    eprintln!("dense cache, {PROMPT} tokens: {ms:.2} ms/step");
}

#[test]
#[ignore = "timing; run by hand"]
fn rotating_window_decode() {
    let config =
        KVCacheConfig::new(LAYERS, 16384, HKV as usize, DIM as usize).with_rotating(1024, 4);
    let ms = decode_ms(config, dense_attend);
    eprintln!("rotating cache (1024 + 4 sinks) after {PROMPT} tokens: {ms:.2} ms/step");
}

#[test]
#[ignore = "timing; run by hand"]
fn quantized_cache_decode() {
    let config =
        KVCacheConfig::new(LAYERS, 16384, HKV as usize, DIM as usize).with_quantized(4, 64);
    let ms = decode_ms(config, dense_attend);
    eprintln!("4-bit cache, {PROMPT} tokens, dequantize then attend: {ms:.2} ms/step");
}

#[test]
#[ignore = "timing; run by hand"]
fn quantized_cache_packed_decode() {
    let config =
        KVCacheConfig::new(LAYERS, 16384, HKV as usize, DIM as usize).with_quantized(4, 64);
    let ms = decode_ms(config, compressed_attend);
    eprintln!("4-bit cache, {PROMPT} tokens, attend to packed rows: {ms:.2} ms/step");
}

#[test]
#[ignore = "timing; run by hand"]
fn turboquant_decode() {
    // A 512-token recent window: nearly all of the prompt is quantized.
    let config = KVCacheConfig::new(LAYERS, 16384, HKV as usize, DIM as usize)
        .with_turboquant_config(TurboQuantConfig::uniform(4, 4).with_recent_window(Some(512)));
    let ms = decode_ms(config, compressed_attend);
    eprintln!("TurboQuant 4-bit (512 recent), {PROMPT} tokens: {ms:.2} ms/step");
}
