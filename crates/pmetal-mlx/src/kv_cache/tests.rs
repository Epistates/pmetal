//! Tests for all KV cache types.

use super::*;
use crate::test_utils::{max_abs_diff, to_f32_vec_eval};
use pmetal_runtime::compat::{Array, Dtype, ops};

fn seq_tensor(start: f32, len: usize) -> Array {
    let data: Vec<f32> = (0..len).map(|idx| start + idx as f32).collect();
    Array::from_f32_slice(&data, &[1, 1, len as i32, 1])
}

fn patterned_tensor(batch: usize, heads: usize, seq: usize, dim: usize, phase: f32) -> Array {
    let len = batch * heads * seq * dim;
    let data: Vec<f32> = (0..len)
        .map(|idx| {
            let x = idx as f32 + phase;
            (x * 0.113).sin() + 0.5 * (x * 0.037 + phase).cos()
        })
        .collect();
    Array::from_f32_slice(&data, &[batch as i32, heads as i32, seq as i32, dim as i32])
}

#[test]
fn test_kv_cache_config() {
    let config = KVCacheConfig::new(32, 2048, 8, 128)
        .with_dtype(Dtype::Float16)
        .with_sliding_window(512);

    assert_eq!(config.num_layers, 32);
    assert_eq!(config.max_seq_len, 2048);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.value_head_dim, 128);
    assert_eq!(config.dtype, Dtype::Float16);
    assert_eq!(config.mode, CacheMode::SlidingWindow { window_size: 512 });
}

#[test]
fn test_kv_cache_basic() {
    let config = KVCacheConfig::new(2, 100, 4, 64);
    let mut cache = KVCache::new(config);

    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);

    // First update - [B, heads, seq, head_dim] format
    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);

    let (cached_k, cached_v) = cache.update_and_fetch(0, &keys, &values).unwrap();

    // Seq is now axis 2
    assert_eq!(cached_k.dim(2), 10);
    assert_eq!(cached_v.dim(2), 10);
    assert_eq!(cache.seq_len(), 10);
    assert!(!cache.is_empty());
}

#[test]
fn test_kv_cache_asymmetric_head_dims() {
    let config = KVCacheConfig::new(1, 100, 2, 16).with_value_head_dim(8);
    let mut cache = KVCache::new(config);

    let keys = ops::zeros(&[1, 2, 3, 16], Dtype::Float32);
    let values = ops::zeros(&[1, 2, 3, 8], Dtype::Float32);

    let (cached_k, cached_v) = cache.update_and_fetch(0, &keys, &values).unwrap();

    assert_eq!(cached_k.shape(), &[1, 2, 3, 16]);
    assert_eq!(cached_v.shape(), &[1, 2, 3, 8]);
    assert_eq!(cache.seq_len(), 3);
}

#[test]
fn test_sanitize_cache_mode_adjusts_group_size_for_asymmetric_dims() {
    let config = KVCacheConfig::new(1, 32, 2, 96).with_value_head_dim(64);
    let safe = sanitize_cache_mode_for_config(
        &config,
        CacheMode::Quantized {
            bits: 4,
            group_size: 128,
        },
    );

    assert_eq!(
        safe,
        CacheMode::Quantized {
            bits: 4,
            group_size: 32
        }
    );
}

#[test]
fn test_sanitize_cache_mode_only_picks_group_sizes_mlx_can_quantize() {
    // MLX's affine quantizer packs groups of 32, 64 or 128 and nothing else.
    let mode = |bits, group_size| CacheMode::Quantized { bits, group_size };
    for (dim, preferred, expected) in [
        (128, 64, mode(4, 64)),
        (128, 128, mode(4, 128)),
        (96, 64, mode(4, 32)),
        (64, 128, mode(4, 64)),
        // Head dimensions no supported group divides cannot be quantized
        // this way: the cache stays dense rather than throwing on the
        // first token.
        (80, 64, CacheMode::Standard),
        (8, 64, CacheMode::Standard),
        (72, 32, CacheMode::Standard),
    ] {
        let config = KVCacheConfig::new(1, 32, 2, dim);
        assert_eq!(
            sanitize_cache_mode_for_config(&config, mode(4, preferred)),
            expected,
            "head dim {dim}, preferred group {preferred}"
        );
    }
    // The asymmetric mode follows the same rule, and so does a key/value
    // pair whose dimensions share no group.
    let config = KVCacheConfig::new(1, 32, 2, 96).with_value_head_dim(64);
    assert_eq!(
        sanitize_cache_mode_for_config(
            &config,
            CacheMode::AsymmetricQuantized {
                key_bits: 8,
                value_bits: 4,
                group_size: 64
            }
        ),
        CacheMode::AsymmetricQuantized {
            key_bits: 8,
            value_bits: 4,
            group_size: 32
        }
    );
    let config = KVCacheConfig::new(1, 32, 2, 96).with_value_head_dim(80);
    assert_eq!(
        sanitize_cache_mode_for_config(&config, mode(8, 64)),
        CacheMode::Standard
    );
}

#[test]
fn test_quantized_cache_refuses_a_group_size_mlx_cannot_pack() {
    // A cache built directly with a bad group is an error on the first
    // update, not a throw inside the quantizer.
    let mut cache = KVCache::new(KVCacheConfig::new(1, 32, 2, 16).with_quantized(4, 16));
    let k = ops::ones(&[1, 2, 3, 16], Dtype::Float32);
    assert!(cache.update_and_fetch(0, &k, &k).is_err());
}

#[test]
fn test_sanitize_cache_mode_clamps_turboquant_outliers_per_tensor_dim() {
    let config = KVCacheConfig::new(1, 32, 2, 8).with_value_head_dim(4);
    let safe = sanitize_cache_mode_for_config(
        &config,
        CacheMode::TurboQuant {
            config: TurboQuantConfig {
                keys: TurboQuantTensorConfig::mixed(2, 4, 99),
                values: TurboQuantTensorConfig::mixed(3, 5, 99),
                ..TurboQuantConfig::uniform(2, 3)
            },
        },
    );

    assert_eq!(
        safe,
        CacheMode::TurboQuant {
            config: TurboQuantConfig {
                keys: TurboQuantTensorConfig::mixed(2, 4, 7),
                values: TurboQuantTensorConfig::mixed(3, 5, 3),
                ..TurboQuantConfig::uniform(2, 3)
            }
        }
    );
}

#[test]
fn test_kv_cache_accumulation() {
    let config = KVCacheConfig::new(1, 100, 4, 64);
    let mut cache = KVCache::new(config);

    // First update: 10 tokens [B, heads, seq, head_dim]
    let k1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(0, &k1, &v1).unwrap();

    assert_eq!(cache.seq_len(), 10);

    // Second update: 5 more tokens
    let k2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let (cached_k, cached_v) = cache.update_and_fetch(0, &k2, &v2).unwrap();

    // Seq is axis 2
    assert_eq!(cached_k.dim(2), 15);
    assert_eq!(cached_v.dim(2), 15);
    assert_eq!(cache.seq_len(), 15);
    assert_eq!(cache.total_tokens(), 15);
}

#[test]
fn test_kv_cache_sliding_window() {
    let config = KVCacheConfig::new(1, 100, 4, 64).with_sliding_window(20);
    let mut cache = KVCache::new(config);

    // Add 15 tokens [B, heads, seq, head_dim]
    let k1 = ops::ones(&[1, 4, 15, 64], Dtype::Float32);
    let v1 = ops::ones(&[1, 4, 15, 64], Dtype::Float32);
    cache.update_and_fetch(0, &k1, &v1).unwrap();

    assert_eq!(cache.seq_len(), 15);

    // Add 10 more. The first of them still sees the 19 tokens before it, so
    // all 25 keys come back even though the window is 20.
    let k2 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let (cached_k, _) = cache.update_and_fetch(0, &k2, &v2).unwrap();
    assert_eq!(cached_k.dim(2), 25);
    assert_eq!(cache.seq_len(), 20);
    // But total tokens should reflect actual count
    assert_eq!(cache.total_tokens(), 25);

    // A single token attends to exactly the window.
    let k3 = ops::ones(&[1, 4, 1, 64], Dtype::Float32);
    let (cached_k, _) = cache.update_and_fetch(0, &k3, &k3).unwrap();
    assert_eq!(cached_k.dim(2), 20);
    assert_eq!(cache.total_tokens(), 26);
}

#[test]
fn test_kv_cache_reset() {
    let config = KVCacheConfig::new(2, 100, 4, 64);
    let mut cache = KVCache::new(config);

    // [B, heads, seq, head_dim] format
    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(0, &keys, &values).unwrap();
    cache.update_and_fetch(1, &keys, &values).unwrap();

    assert!(!cache.is_empty());

    cache.reset();

    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.total_tokens(), 0);
}

#[test]
fn test_kv_cache_speculative_rollback_preserves_accepted_prefix() {
    // Speculative-decoding scenario:
    //   1. prefill 4 tokens     (offset 4)
    //   2. verify appends 5 draft tokens → offset 9
    //   3. only 2 of the 5 are accepted → trim 3 → offset 6
    //   4. append 1 correction/bonus token → offset 7
    // After the final append we must observe a contiguous [prefill || accepted || bonus]
    // with the rejected tokens completely overwritten (the rewind semantics
    // DFlash verification needs).
    let config = KVCacheConfig::new(1, 64, 1, 1);
    let mut cache = KVCache::new(config);

    let prefill = seq_tensor(0.0, 4); // [0,1,2,3]
    let draft = seq_tensor(10.0, 5); // [10,11,12,13,14] (values the draft tried)
    let accepted_bonus = seq_tensor(100.0, 1); // [100] the corrective token

    cache.update_and_fetch(0, &prefill, &prefill).unwrap();
    cache.update_and_fetch(0, &draft, &draft).unwrap();
    assert_eq!(cache.seq_len(), 9);
    assert_eq!(cache.total_tokens(), 9);

    // 2 accepted out of 5 → rewind 3
    let trimmed = cache.rollback(3);
    assert_eq!(trimmed, 3);
    assert_eq!(cache.seq_len(), 6);
    assert_eq!(cache.total_tokens(), 6);

    cache
        .update_and_fetch(0, &accepted_bonus, &accepted_bonus)
        .unwrap();
    assert_eq!(cache.seq_len(), 7);

    let (cached_k, _) = cache.get(0).unwrap();
    let flat = to_f32_vec_eval(&cached_k);
    assert_eq!(
        flat,
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 100.0],
        "rollback must preserve accepted prefix and overwrite rejected tail"
    );
}

#[test]
fn test_kv_cache_rollback_uses_populated_later_layer() {
    // Hybrid models such as Qwen3Next can have GDN layers before the first
    // full-attention layer. Layer 0 may therefore be empty while later KV
    // layers hold speculative tokens. Rollback must trim those populated layers.
    let config = KVCacheConfig::new(4, 64, 1, 1);
    let mut cache = KVCache::new(config);

    let prefill = seq_tensor(0.0, 4);
    let draft = seq_tensor(10.0, 5);
    let correction = seq_tensor(100.0, 1);

    cache.update_and_fetch(3, &prefill, &prefill).unwrap();
    cache.update_and_fetch(3, &draft, &draft).unwrap();
    assert_eq!(cache.seq_len(), 9);

    let trimmed = cache.rollback(3);
    assert_eq!(trimmed, 3);
    assert_eq!(cache.seq_len(), 6);

    cache.update_and_fetch(3, &correction, &correction).unwrap();
    assert_eq!(cache.seq_len(), 7);

    let (cached_k, _) = cache.get(3).unwrap();
    let flat = to_f32_vec_eval(&cached_k);
    assert_eq!(flat, vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 100.0]);
}

#[test]
fn test_kv_cache_rollback_clamps_to_seq_len() {
    let config = KVCacheConfig::new(1, 100, 1, 1);
    let mut cache = KVCache::new(config);
    let tokens = seq_tensor(0.0, 3);
    cache.update_and_fetch(0, &tokens, &tokens).unwrap();

    // Asking to trim more than we have clamps to available.
    let trimmed = cache.rollback(99);
    assert_eq!(trimmed, 3);
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_kv_cache_rope_offset() {
    let config = KVCacheConfig::new(1, 100, 4, 64);
    let mut cache = KVCache::new(config);

    assert_eq!(cache.rope_offset_for(0), 0);

    // [B, heads, seq, head_dim] format
    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(0, &keys, &values).unwrap();

    assert_eq!(cache.rope_offset_for(0), 10);
}

/// Every layer advances its own offset during one forward, so the offset has
/// to be read per layer. A whole-cache reading is already a chunk ahead by the
/// time layer 1 asks, which puts every layer but the first one step past where
/// its own cached keys sit.
#[test]
fn rope_offset_is_per_layer_not_whichever_layer_ran_last() {
    let mut cache = KVCache::new(KVCacheConfig::new(3, 100, 1, 1));
    let prompt = seq_tensor(0.0, 5);

    for layer in 0..3 {
        assert_eq!(
            cache.rope_offset_for(layer),
            0,
            "layer {layer} starts the prefill at position 0, whatever the layers \
             before it have already cached"
        );
        cache.update_and_fetch(layer, &prompt, &prompt).unwrap();
    }

    let next = seq_tensor(5.0, 1);
    for layer in 0..3 {
        assert_eq!(
            cache.rope_offset_for(layer),
            5,
            "layer {layer} continues from the five tokens it cached itself"
        );
        cache.update_and_fetch(layer, &next, &next).unwrap();
    }
}

/// Same, for the sliding window, where `offset` stops being the absolute
/// position once the oldest entries start falling off the front.
#[test]
fn sliding_window_rope_offset_is_per_layer_and_absolute() {
    let mut cache = KVCache::new(KVCacheConfig::new(2, 100, 1, 1).with_sliding_window(4));
    let first = seq_tensor(0.0, 3);
    let second = seq_tensor(3.0, 3);

    for layer in 0..2 {
        cache.update_and_fetch(layer, &first, &first).unwrap();
    }
    for layer in 0..2 {
        assert_eq!(cache.rope_offset_for(layer), 3);
        cache.update_and_fetch(layer, &second, &second).unwrap();
    }
    for layer in 0..2 {
        // Six tokens in, four retained: the position is six, not four.
        assert_eq!(cache.rope_offset_for(layer), 6);
    }
}

#[test]
fn test_kv_cache_sliding_window_rope_offset_tracks_total_tokens() {
    let config = KVCacheConfig::new(1, 100, 1, 1).with_sliding_window(4);
    let mut cache = KVCache::new(config);

    let first = seq_tensor(0.0, 3);
    let second = seq_tensor(3.0, 3);
    cache.update_and_fetch(0, &first, &first).unwrap();
    cache.update_and_fetch(0, &second, &second).unwrap();

    assert_eq!(cache.seq_len(), 4);
    assert_eq!(cache.total_tokens(), 6);
    assert_eq!(cache.rope_offset_for(0), 6);
}

#[test]
fn test_kv_cache_rotating_window_preserves_keep_tokens() {
    let config = KVCacheConfig::new(1, 100, 1, 1).with_rotating(6, 2);
    let mut cache = KVCache::new(config);

    let first = seq_tensor(0.0, 4);
    let second = seq_tensor(4.0, 4);
    let _ = cache.update_and_fetch(0, &first, &first).unwrap();
    let (cached_k, _) = cache.update_and_fetch(0, &second, &second).unwrap();

    assert_eq!(cache.seq_len(), 6);
    assert_eq!(cache.total_tokens(), 8);
    assert_eq!(cache.rope_offset_for(0), 8);
    // The chunk's first query still needs the three tokens before it.
    assert_eq!(
        to_f32_vec_eval(&cached_k),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );

    // One more token attends to the two sinks and the newest four.
    let third = seq_tensor(8.0, 1);
    let (cached_k, cached_v) = cache.update_and_fetch(0, &third, &third).unwrap();
    assert_eq!(
        to_f32_vec_eval(&cached_k),
        vec![0.0, 1.0, 5.0, 6.0, 7.0, 8.0]
    );
    assert_eq!(
        to_f32_vec_eval(&cached_v),
        vec![0.0, 1.0, 5.0, 6.0, 7.0, 8.0]
    );
    assert_eq!(cache.rope_offset_for(0), 9);
}

#[test]
fn test_kv_cache_memory_estimation() {
    let config = KVCacheConfig::new(32, 2048, 8, 128).with_dtype(Dtype::Float16);
    let cache = KVCache::new(config);

    // 32 layers * 2048 seq * 8 heads * 128 dim * 2 bytes * 2 (K+V)
    let expected = 32 * 2048 * 8 * 128 * 2 * 2;
    assert_eq!(cache.max_memory_usage(), expected);
}

#[test]
fn test_kv_cache_multi_layer() {
    let config = KVCacheConfig::new(4, 100, 4, 64);
    let mut cache = KVCache::new(config);

    // Update all layers - [B, heads, seq, head_dim] format
    for layer in 0..4 {
        let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
        let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
        cache.update_and_fetch(layer, &keys, &values).unwrap();
    }

    // All layers should have same seq_len (axis 2)
    for layer in 0..4 {
        let (k, v) = cache.get(layer).expect("Cache should exist");
        assert_eq!(k.dim(2), 10);
        assert_eq!(v.dim(2), 10);
    }
}

#[test]
fn test_batch_kv_cache() {
    let config = KVCacheConfig::new(2, 100, 4, 64);
    let mut batch_cache = BatchKVCache::new(4, config);

    assert_eq!(batch_cache.batch_size(), 4);

    // Update one cache - [B, heads, seq, head_dim] format
    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    batch_cache
        .get_mut(0)
        .unwrap()
        .update_and_fetch(0, &keys, &values)
        .unwrap();

    assert!(!batch_cache.caches[0].is_empty());
    assert!(batch_cache.caches[1].is_empty());
}

#[test]
fn test_batch_kv_cache_reset_indices() {
    let config = KVCacheConfig::new(1, 100, 4, 64);
    let mut batch_cache = BatchKVCache::new(4, config);

    // [B, heads, seq, head_dim] format
    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);

    // Fill all caches
    for i in 0..4 {
        batch_cache
            .get_mut(i)
            .unwrap()
            .update_and_fetch(0, &keys, &values)
            .unwrap();
    }

    // Reset specific indices
    batch_cache.reset_indices(&[0, 2]);

    assert!(batch_cache.caches[0].is_empty());
    assert!(!batch_cache.caches[1].is_empty());
    assert!(batch_cache.caches[2].is_empty());
    assert!(!batch_cache.caches[3].is_empty());
}

// =========================================================================
// CacheMode Tests for New Modes
// =========================================================================

#[test]
fn test_cache_mode_rotating() {
    let config = KVCacheConfig::new(32, 2048, 8, 128).with_rotating(1024, 8);

    assert_eq!(
        config.mode,
        CacheMode::Rotating {
            max_size: 1024,
            keep: 8
        }
    );
}

#[test]
fn test_cache_mode_quantized() {
    let config = KVCacheConfig::new(32, 2048, 8, 128).with_quantized(4, 64);

    assert_eq!(
        config.mode,
        CacheMode::Quantized {
            bits: 4,
            group_size: 64
        }
    );
}

#[test]
fn test_cache_mode_turboquant() {
    let config = KVCacheConfig::new(32, 2048, 8, 128).with_turboquant(4, 3);

    assert_eq!(
        config.mode,
        CacheMode::TurboQuant {
            config: TurboQuantConfig::uniform(4, 3)
        }
    );
}

#[test]
fn test_cache_mode_turboquant_mixed() {
    let config = KVCacheConfig::new(32, 2048, 8, 128).with_turboquant_mixed(2, 4, 32, 3, 5, 32);

    assert_eq!(
        config.mode,
        CacheMode::TurboQuant {
            config: TurboQuantConfig::mixed(2, 4, 32, 3, 5, 32)
        }
    );
}

// =========================================================================
// Memory footprint
// =========================================================================

#[test]
fn test_kv_cache_memory_footprint() {
    // 32 layers × 2 (K+V) × 1 batch × 8 heads × 2048 seq × 128 dim × 2 bytes (fp16)
    let config = KVCacheConfig::new(32, 2048, 8, 128).with_dtype(Dtype::Float16);

    let expected = 32 * 2 * 8 * 2048 * 128 * 2; // layers * K+V * heads * seq * dim * sizeof(f16)
    assert_eq!(config.memory_footprint(), expected);
}

#[test]
fn test_kv_cache_memory_footprint_fp32() {
    // 16 layers × 2 (K+V) × 1 batch × 4 heads × 1024 seq × 64 dim × 4 bytes (fp32)
    let config = KVCacheConfig::new(16, 1024, 4, 64).with_dtype(Dtype::Float32);

    let expected = 16 * 2 * 4 * 1024 * 64 * 4; // layers * K+V * heads * seq * dim * sizeof(f32)
    assert_eq!(config.memory_footprint(), expected);
}

#[test]
fn test_kv_cache_memory_footprint_human_bytes() {
    let config = KVCacheConfig::new(1, 1, 1, 1).with_dtype(Dtype::Float32);

    // 1 × 2 × 1 × 1 × 1 × 1 × 4 = 8 bytes
    assert_eq!(config.memory_footprint_human(), "8 bytes");
}

#[test]
fn test_kv_cache_memory_footprint_human_kb() {
    let config = KVCacheConfig::new(1, 128, 1, 1).with_dtype(Dtype::Float32);

    // 1 × 2 × 1 × 1 × 128 × 1 × 4 = 1024 bytes = 1 KB
    assert_eq!(config.memory_footprint_human(), "1.00 KB");
}

#[test]
fn test_kv_cache_memory_footprint_human_mb() {
    let config = KVCacheConfig::new(1, 2048, 8, 64).with_dtype(Dtype::Float32);

    // 1 × 2 × 1 × 8 × 2048 × 64 × 4 = 8,388,608 bytes = 8 MB
    let human = config.memory_footprint_human();
    assert!(human.contains("MB"), "Expected MB, got: {}", human);
}

#[test]
fn test_kv_cache_memory_footprint_human_gb() {
    // Need to exceed 1 GB (1,073,741,824 bytes)
    // 32 layers × 2 × 1 batch × 8 heads × 8192 seq × 128 dim × 2 bytes = 1 GB
    let config = KVCacheConfig::new(32, 8192, 8, 128).with_dtype(Dtype::Float16);

    let human = config.memory_footprint_human();
    assert!(human.contains("GB"), "Expected GB, got: {}", human);
}

#[test]
fn test_kv_cache_lazy_reset_deallocates() {
    let config = KVCacheConfig::new(2, 64, 4, 32);
    let mut cache = KVCache::new(config);

    // Add some data
    let keys = ops::zeros(&[1, 4, 10, 32], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 32], Dtype::Float32);
    cache.update_and_fetch(0, &keys, &values).unwrap();
    assert!(!cache.is_empty());

    // Reset should deallocate in lazy mode
    cache.reset();

    assert!(cache.is_empty());
    // Verify buffers are deallocated by checking get returns None
    assert!(cache.get(0).is_none());
}

// =========================================================================
// MambaCache / GDN rollback tests
// =========================================================================

fn gdn_random_inputs(
    batch: usize,
    t: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    seed_phase: f32,
) -> (Array, Array, Array, Array) {
    // GDN input layout is [B, T, H, D]. `patterned_tensor(a, b, c, d)`
    // already returns [a, b, c, d], so we pass the axes in the target order
    // and skip the transpose.
    let k = patterned_tensor(batch, t, hv, dk, seed_phase);
    let v = patterned_tensor(batch, t, hv, dv, seed_phase + 0.31);
    // g ∈ (0,1) so the state doesn't explode; use small patterned values
    let g_len = batch * t * hv;
    let g_data: Vec<f32> = (0..g_len)
        .map(|i| 0.5 + 0.3 * ((i as f32 * 0.17 + seed_phase).sin()))
        .collect();
    let g = Array::from_f32_slice(&g_data, &[batch as i32, t as i32, hv as i32]);
    // beta ∈ (0,1)
    let beta_data: Vec<f32> = (0..g_len)
        .map(|i| 0.5 + 0.25 * ((i as f32 * 0.09 + seed_phase).cos()))
        .collect();
    let beta = Array::from_f32_slice(&beta_data, &[batch as i32, t as i32, hv as i32]);
    (k, v, g, beta)
}

/// Each cache, clone and reset is its own sequence, and a key outlives
/// neither the cache it names nor a reset.
#[test]
fn test_mamba_sequence_key_names_one_sequence() {
    let mut a = MambaCache::new(2);
    let b = MambaCache::new(2);
    let key = a.sequence_key();
    assert!(key.belongs_to(&a) && key.is_live());
    assert!(!key.belongs_to(&b));

    let clone = a.clone();
    assert!(!key.belongs_to(&clone), "a clone is a new sequence");

    a.reset();
    assert!(!key.belongs_to(&a), "a reset starts a new sequence");
    assert!(!key.is_live());

    let key = a.sequence_key();
    drop(a);
    assert!(!key.is_live());
    // The dead key's allocation is still held, so no new cache can reuse it.
    let c = MambaCache::new(2);
    assert!(!key.belongs_to(&c));
}

#[test]
fn test_mamba_snapshot_and_restore_roundtrip() {
    let mut cache = MambaCache::new(2);

    // Seed layer 0 with some fake state + conv_state
    {
        let entry = cache.get_mut(0).unwrap();
        entry.ssm_state = Some(patterned_tensor(1, 2, 8, 16, 0.5));
        entry.conv_state = Some(patterned_tensor(1, 3, 1, 32, 0.9).reshape(&[1, 3, 32]));
    }

    let snapshots = cache.snapshot();

    // Mutate the cache after taking the snapshot
    {
        let entry = cache.get_mut(0).unwrap();
        entry.ssm_state = Some(patterned_tensor(1, 2, 8, 16, 7.0));
    }

    // Restoring every layer verbatim brings the cache back to the snapshot
    let no_inputs: Vec<Option<GdnVerifyInputs>> = vec![None, None];
    cache
        .rewind_from_snapshots(&snapshots, &no_inputs, 0)
        .unwrap();

    let after = cache.get(0).unwrap();
    let before = snapshots[0].ssm_state.as_ref().unwrap();
    assert!(
        max_abs_diff(after.ssm_state.as_ref().unwrap(), before) < 1e-6,
        "restore must be a bitwise roundtrip"
    );
}

#[test]
fn test_gdn_rollback_matches_never_went_there() {
    // Speculative rollback invariant:
    //   advance(S0, inputs[..M])  ==  rewind(snapshot=S0, inputs[..K], accepted=M)
    //
    // Where `inputs[..K]` is the full verify batch and `inputs[..M]` is the
    // accepted prefix. The left-hand side is the "ground truth" — what we'd
    // compute if we had known up-front to only advance through M tokens.
    let batch = 1;
    let t_verify = 5;
    let accepted = 2;
    let hv = 2;
    let dk = 32;
    let dv = 16;

    let initial_state = patterned_tensor(batch, hv, dv, dk, 0.25);
    let (k_full, v_full, g_full, beta_full) = gdn_random_inputs(batch, t_verify, hv, dk, dv, 1.0);

    // Ground truth: advance through only the accepted prefix
    let slice_time = |arr: &Array, n: i32| -> Array {
        let shape = arr.shape();
        let rank = shape.len();
        let start = vec![0i32; rank];
        let mut stop: Vec<i32> = shape.to_vec();
        stop[1] = n;
        arr.slice(&start, &stop)
    };
    let k_trunc = slice_time(&k_full, accepted as i32);
    let v_trunc = slice_time(&v_full, accepted as i32);
    let g_trunc = slice_time(&g_full, accepted as i32);
    let beta_trunc = slice_time(&beta_full, accepted as i32);
    let expected_state = pmetal_runtime::nn::gated_delta_state_advance(
        &initial_state,
        &k_trunc,
        &v_trunc,
        &g_trunc,
        &beta_trunc,
    )
    .unwrap();

    // Rollback path: advance through the full verify batch, then rewind
    let mut entry = MambaCacheEntry {
        ssm_state: Some(initial_state.clone()),
        ..Default::default()
    };
    let snapshot = entry.snapshot();

    // Simulate verify advancing through all K tokens
    let post_verify = pmetal_runtime::nn::gated_delta_state_advance(
        &initial_state,
        &k_full,
        &v_full,
        &g_full,
        &beta_full,
    )
    .unwrap();
    entry.ssm_state = Some(post_verify);

    let conv_input = ops::zeros(&[batch as i32, t_verify as i32, 4], Dtype::Float32);
    let verify_inputs = GdnVerifyInputs {
        keys: k_full,
        values: v_full,
        g: g_full,
        beta: beta_full,
        conv_input,
        conv_kernel_size: 1, // no conv rewind path (we're testing SSM only)
    };

    entry
        .rewind(&snapshot, Some(&verify_inputs), accepted)
        .unwrap();

    let rolled_back = entry.ssm_state.as_ref().unwrap();
    let diff = max_abs_diff(rolled_back, &expected_state);
    assert!(
        diff < 1e-4,
        "rewound GDN state must equal never-went-there state, got max_diff={diff}"
    );
}

#[test]
fn test_gdn_rollback_zero_accepted_equals_snapshot() {
    // If zero tokens are accepted, rewind must restore the exact snapshot —
    // no replay through the verify inputs.
    let mut entry = MambaCacheEntry::default();
    let state = patterned_tensor(1, 2, 4, 8, 3.25);
    entry.ssm_state = Some(state.clone());
    let snapshot = entry.snapshot();

    // Advance state through some verify work
    let (k, v, g, beta) = gdn_random_inputs(1, 3, 2, 8, 4, 5.0);
    let post = pmetal_runtime::nn::gated_delta_state_advance(&state, &k, &v, &g, &beta).unwrap();
    entry.ssm_state = Some(post);

    let conv_input = ops::zeros(&[1, 3, 2], Dtype::Float32);
    let inputs = GdnVerifyInputs {
        keys: k,
        values: v,
        g,
        beta,
        conv_input,
        conv_kernel_size: 1,
    };

    entry.rewind(&snapshot, Some(&inputs), 0).unwrap();

    let diff = max_abs_diff(entry.ssm_state.as_ref().unwrap(), &state);
    assert!(
        diff < 1e-6,
        "zero-accepted rewind must be verbatim snapshot"
    );
}
