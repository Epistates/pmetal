//! Throughput benchmark for fused vs. serial continuous-batching decode.
//!
//! Run with: `cargo bench -p pmetal-serve --bench continuous_batch`.
//!
//! # What this measures
//!
//! For each `N_active` in {1, 2, 4, 8, 16}, the bench drives one decode
//! step (a single `[N_active, 1]` token forward) through two paths:
//!
//! 1. **Serial** — `N_active` independent `forward_with_cache` calls
//!    against per-slot `KVCache`s, mirroring the Phase-1 fallback.
//! 2. **Fused** — one `forward_batched_impl` call against a shared
//!    [`pmetal_cache::kv_cache::FusedBatchKVCache`].
//!
//! The decode-token sample size is intentionally small (synthetic Llama
//! weights, tiny hidden + few layers) to keep the bench self-contained
//! and to highlight the *kernel-dispatch* amortization the fused path is
//! built for. Absolute tok/s is not the goal — relative speedup at each
//! `N_active` is.
//!
//! Throughput is reported in `Throughput::Elements(N_active)` so
//! Criterion's tok/s readout already factors out the per-slot work.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pmetal_cache::kv_cache::{FusedBatchKVCache, KVCache, KVCacheConfig};
use pmetal_models::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use pmetal_runtime::compat::Array;

/// Tiny synthetic Llama config: enough layers to surface the per-layer
/// dispatch tax, small enough to bench fast on Apple Silicon.
fn bench_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 256,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 8,
        num_attention_heads: 8,
        num_key_value_heads: Some(2),
        head_dim: Some(16),
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        hidden_act: "silu".to_string(),
        tie_word_embeddings: true,
        ..Default::default()
    }
}

fn kv_cfg(num_layers: usize, max_seq: usize, kv_heads: usize, head_dim: usize) -> KVCacheConfig {
    KVCacheConfig::new(num_layers, max_seq, kv_heads, head_dim)
}

fn bench_decode_step(c: &mut Criterion) {
    let cfg = bench_config();
    let max_seq = cfg.max_position_embeddings as usize;
    let kv_heads = cfg.num_kv_heads() as usize;
    let head_dim = cfg.get_head_dim() as usize;
    let num_layers = cfg.num_hidden_layers as usize;

    let slot_counts: &[usize] = &[1, 2, 4, 8, 16];

    let mut group = c.benchmark_group("continuous_batch_decode_step");
    group.sample_size(20);

    for &n_active in slot_counts {
        // One model instance per N_active so weight init noise stays
        // consistent across configurations within a single bench run.
        let mut serial_model = LlamaForCausalLM::new(cfg.clone()).unwrap();
        let mut fused_model = LlamaForCausalLM::new(cfg.clone()).unwrap();

        // Per-slot serial caches (one KVCache per slot, freshly seeded).
        let mut serial_caches: Vec<KVCache> = (0..n_active)
            .map(|_| KVCache::new(kv_cfg(num_layers, max_seq, kv_heads, head_dim)))
            .collect();
        // Shared fused cache, all slots admitted.
        let mut fused_cache =
            FusedBatchKVCache::new(kv_cfg(num_layers, max_seq, kv_heads, head_dim), n_active)
                .unwrap();
        for slot in 0..n_active {
            fused_cache.admit(slot).unwrap();
        }

        // One real prefill step so caches contain at least one entry —
        // matches the actual continuous-batching workload (decode after
        // prompt prefill).
        for (slot, cache) in serial_caches.iter_mut().enumerate() {
            let tok = ((slot as i32) % 17) + 3;
            let prompt = Array::from_i32_slice(&[tok]).reshape(&[1, 1]);
            let _ = serial_model
                .forward_with_cache(&prompt, None, Some(cache))
                .unwrap();
        }
        let fused_prompt: Vec<i32> = (0..n_active as i32).map(|i| (i % 17) + 3).collect();
        let fused_input = Array::from_i32_slice(&fused_prompt).reshape(&[n_active as i32, 1]);
        let active: Vec<usize> = (0..n_active).collect();
        let _ = fused_model
            .forward_batched_impl(&fused_input, &active, &mut fused_cache)
            .unwrap();

        group.throughput(Throughput::Elements(n_active as u64));

        // Serial: `n_active` independent forward_with_cache calls.
        group.bench_with_input(BenchmarkId::new("serial", n_active), &n_active, |b, &n| {
            b.iter(|| {
                for (slot, cache) in serial_caches.iter_mut().enumerate().take(n) {
                    let tok = ((slot as i32) % 17) + 3;
                    let inp = Array::from_i32_slice(&[tok]).reshape(&[1, 1]);
                    let logits = serial_model
                        .forward_with_cache(&inp, None, Some(cache))
                        .unwrap();
                    logits.eval();
                    black_box(logits);
                }
            });
        });

        // Fused: one forward_batched_impl call.
        group.bench_with_input(BenchmarkId::new("fused", n_active), &n_active, |b, &n| {
            b.iter(|| {
                let toks: Vec<i32> = (0..n as i32).map(|i| (i % 17) + 3).collect();
                let inp = Array::from_i32_slice(&toks).reshape(&[n as i32, 1]);
                let active: Vec<usize> = (0..n).collect();
                let logits = fused_model
                    .forward_batched_impl(&inp, &active, &mut fused_cache)
                    .unwrap();
                logits.eval();
                black_box(logits);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_decode_step);
criterion_main!(benches);
