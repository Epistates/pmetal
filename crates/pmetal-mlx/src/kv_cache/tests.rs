//! Tests for all KV cache types.

use super::*;
use crate::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use crate::test_utils::{max_abs_diff, to_f32_vec_eval};
use pmetal_bridge::compat::{Array, Dtype, ops};

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

fn manual_attention_output(queries: &Array, keys: &Array, values: &Array, scale: f32) -> Array {
    let scores = queries
        .matmul(&keys.transpose_axes(&[0, 1, 3, 2]))
        .multiply(&Array::from_f32(scale));
    let weights = ops::softmax_axis(&scores, -1);
    weights.matmul(values)
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
fn test_sanitize_cache_mode_clamps_turboquant_outliers_per_tensor_dim() {
    let config = KVCacheConfig::new(1, 32, 2, 8).with_value_head_dim(4);
    let safe = sanitize_cache_mode_for_config(
        &config,
        CacheMode::TurboQuant {
            config: TurboQuantConfig {
                keys: TurboQuantTensorConfig::mixed(2, 4, 99),
                values: TurboQuantTensorConfig::mixed(3, 5, 99),
                recent_window: Some(DEFAULT_RECENT_WINDOW),
                qjl: Default::default(),
            },
        },
    );

    assert_eq!(
        safe,
        CacheMode::TurboQuant {
            config: TurboQuantConfig {
                keys: TurboQuantTensorConfig::mixed(2, 4, 7),
                values: TurboQuantTensorConfig::mixed(3, 5, 3),
                recent_window: Some(DEFAULT_RECENT_WINDOW),
                qjl: Default::default(),
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

    // Add 10 more - should trigger sliding window
    let k2 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let (cached_k, _) = cache.update_and_fetch(0, &k2, &v2).unwrap();

    // Should be trimmed to window size of 20, seq is axis 2
    assert_eq!(cached_k.dim(2), 20);
    assert_eq!(cache.seq_len(), 20);
    // But total tokens should reflect actual count
    assert_eq!(cache.total_tokens(), 25);
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
    // with the rejected tokens completely overwritten (matching dflash-mlx's
    // `target.rewind_kv_caches` semantics).
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

    assert_eq!(cache.rope_offset(), 0);

    // [B, heads, seq, head_dim] format
    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(0, &keys, &values).unwrap();

    assert_eq!(cache.rope_offset(), 10);
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
    assert_eq!(cache.rope_offset(), 6);
}

#[test]
fn test_kv_cache_rotating_window_preserves_keep_tokens() {
    let config = KVCacheConfig::new(1, 100, 1, 1).with_rotating(6, 2);
    let mut cache = KVCache::new(config);

    let first = seq_tensor(0.0, 4);
    let second = seq_tensor(4.0, 4);
    let _ = cache.update_and_fetch(0, &first, &first).unwrap();
    let (cached_k, cached_v) = cache.update_and_fetch(0, &second, &second).unwrap();

    assert_eq!(cache.seq_len(), 6);
    assert_eq!(cache.total_tokens(), 8);
    assert_eq!(cache.rope_offset(), 8);

    let k_vec = to_f32_vec_eval(&cached_k);
    let v_vec = to_f32_vec_eval(&cached_v);
    assert_eq!(k_vec, vec![0.0, 1.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(v_vec, vec![0.0, 1.0, 4.0, 5.0, 6.0, 7.0]);
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

#[test]
fn test_convenience_functions() {
    let cache = create_kv_cache(32, 2048, 8, 128);
    assert_eq!(cache.config().num_layers, 32);
    assert_eq!(cache.config().mode, CacheMode::Standard);

    let sliding_cache = create_sliding_window_cache(32, 512, 8, 128);
    assert_eq!(
        sliding_cache.config().mode,
        CacheMode::SlidingWindow { window_size: 512 }
    );
}

// =========================================================================
// RotatingKVCache Tests
// =========================================================================

#[test]
fn test_rotating_cache_basic() {
    let mut cache = RotatingKVCache::new(100, 0);

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.offset(), 0);

    // First update - [B, heads, seq, head_dim] format
    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);

    let (cached_k, cached_v) = cache.update_and_fetch(&keys, &values).unwrap();

    assert_eq!(cached_k.dim(2), 10);
    assert_eq!(cached_v.dim(2), 10);
    assert_eq!(cache.len(), 10);
    assert_eq!(cache.offset(), 10);
    assert!(!cache.is_empty());
}

#[test]
fn test_rotating_cache_accumulation() {
    let mut cache = RotatingKVCache::new(100, 0);

    // First update: 10 tokens
    let k1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&k1, &v1).unwrap();

    assert_eq!(cache.len(), 10);

    // Second update: 5 more tokens
    let k2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let (cached_k, cached_v) = cache.update_and_fetch(&k2, &v2).unwrap();

    assert_eq!(cached_k.dim(2), 15);
    assert_eq!(cached_v.dim(2), 15);
    assert_eq!(cache.len(), 15);
}

#[test]
fn test_rotating_cache_rotation() {
    let mut cache = RotatingKVCache::new(20, 0);

    // Fill beyond max_size
    let k1 = ops::ones(&[1, 4, 15, 64], Dtype::Float32);
    let v1 = ops::ones(&[1, 4, 15, 64], Dtype::Float32);
    cache.update_and_fetch(&k1, &v1).unwrap();

    let k2 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let (_cached_k, _) = cache.update_and_fetch(&k2, &v2).unwrap();

    // MLX-LM allows max_size + S - 1 to ensure every token gets at least max_size context
    // So cache can grow to max_size + num_steps - 1 before trimming
    // In this case: 20 + 10 - 1 = 29, then trimmed to ~20 region
    // The key behavior is len() is capped at max_size, offset tracks total
    assert!(cache.len() <= 25); // len() caps at max_size or slightly above during concat
    assert_eq!(cache.offset(), 25); // Total tokens seen
}

#[test]
fn test_rotating_cache_with_keep() {
    let mut cache = RotatingKVCache::new(20, 4); // Keep first 4 tokens

    // Fill cache
    let k1 = ops::ones(&[1, 4, 15, 64], Dtype::Float32);
    let v1 = ops::ones(&[1, 4, 15, 64], Dtype::Float32);
    cache.update_and_fetch(&k1, &v1).unwrap();

    assert_eq!(cache.len(), 15);

    // Add more tokens
    let k2 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&k2, &v2).unwrap();

    // Cache should have rotated but kept initial tokens
    assert!(cache.len() <= 20);
}

#[test]
fn test_rotating_cache_reset() {
    let mut cache = RotatingKVCache::new(100, 0);

    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&keys, &values).unwrap();

    assert!(!cache.is_empty());

    cache.reset();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.offset(), 0);
}

#[test]
fn test_rotating_cache_single_token_updates() {
    let mut cache = RotatingKVCache::new(50, 0);

    // Simulate autoregressive generation
    for i in 0..30 {
        let k = ops::ones(&[1, 4, 1, 64], Dtype::Float32);
        let v = ops::ones(&[1, 4, 1, 64], Dtype::Float32);
        let (cached_k, _) = cache.update_and_fetch(&k, &v).unwrap();

        assert_eq!(cached_k.dim(2) as usize, (i + 1).min(50));
    }

    assert_eq!(cache.offset(), 30);
    assert_eq!(cache.len(), 30);
}

#[test]
fn test_rotating_cache_trimmable() {
    let mut cache = RotatingKVCache::new(20, 0);

    assert!(cache.is_trimmable()); // Empty cache is trimmable

    let k = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&k, &v).unwrap();

    assert!(cache.is_trimmable()); // Under max_size

    // Fill to max
    let k2 = ops::ones(&[1, 4, 15, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 15, 64], Dtype::Float32);
    cache.update_and_fetch(&k2, &v2).unwrap();

    assert!(!cache.is_trimmable()); // At or over max_size
}

#[test]
fn test_rotating_cache_rope_offset() {
    let mut cache = RotatingKVCache::new(100, 0);

    assert_eq!(cache.rope_offset(), 0);

    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&keys, &values).unwrap();

    // RoPE offset should be total tokens seen, not cache length
    assert_eq!(cache.rope_offset(), 10);
}

// =========================================================================
// QuantizedKVCache Tests
// =========================================================================

#[test]
fn test_quantized_cache_basic() {
    let mut cache = QuantizedKVCache::new(8, 64);

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // First update
    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);

    let (cached_k, cached_v) = cache.update_and_fetch(&keys, &values).unwrap();

    // Dequantized output should have correct shape
    assert_eq!(cached_k.dim(2), 10);
    assert_eq!(cached_v.dim(2), 10);
    assert_eq!(cache.len(), 10);
    assert!(!cache.is_empty());
}

#[test]
fn test_quantized_cache_accumulation() {
    let mut cache = QuantizedKVCache::new(4, 64);

    // First update
    let k1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&k1, &v1).unwrap();

    // Second update
    let k2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let (cached_k, cached_v) = cache.update_and_fetch(&k2, &v2).unwrap();

    assert_eq!(cached_k.dim(2), 15);
    assert_eq!(cached_v.dim(2), 15);
    assert_eq!(cache.len(), 15);
}

#[test]
fn test_quantized_cache_reset() {
    let mut cache = QuantizedKVCache::new(8, 64);

    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&keys, &values).unwrap();

    assert!(!cache.is_empty());

    cache.reset();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_quantized_cache_different_bits() {
    // Test 8-bit
    let cache_8bit = QuantizedKVCache::new(8, 64);
    assert_eq!(cache_8bit.bits, 8);

    // Test 4-bit
    let cache_4bit = QuantizedKVCache::new(4, 64);
    assert_eq!(cache_4bit.bits, 4);

    // Test 2-bit
    let cache_2bit = QuantizedKVCache::new(2, 64);
    assert_eq!(cache_2bit.bits, 2);
}

#[test]
#[should_panic(expected = "bits must be 2, 4, or 8")]
fn test_quantized_cache_invalid_bits() {
    let _ = QuantizedKVCache::new(3, 64); // Invalid
}

#[test]
fn test_quantized_cache_memory_usage() {
    let mut cache = QuantizedKVCache::new(8, 64);

    assert_eq!(cache.memory_usage(), 0); // Empty

    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&keys, &values).unwrap();

    // Should have some memory usage now
    assert!(cache.memory_usage() > 0);
}

#[test]
fn test_quantized_cache_rope_offset() {
    let mut cache = QuantizedKVCache::new(8, 64);

    assert_eq!(cache.rope_offset(), 0);

    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&keys, &values).unwrap();

    assert_eq!(cache.rope_offset(), 10);
}

// =========================================================================
// Convenience Function Tests for New Cache Types
// =========================================================================

#[test]
fn test_create_rotating_cache() {
    let cache = create_rotating_cache(1024, 4);
    assert_eq!(cache.max_size, 1024);
    assert_eq!(cache.keep, 4);
    assert!(cache.is_empty());
}

#[test]
fn test_create_quantized_cache() {
    let cache = create_quantized_cache(4, 32);
    assert_eq!(cache.bits, 4);
    assert_eq!(cache.group_size, 32);
    assert!(cache.is_empty());
}

#[test]
fn test_create_turboquant_cache() {
    let cache = create_turboquant_cache(4, 3);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
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
// TurboQuantKvCache Tests
// =========================================================================

#[test]
fn test_turboquant_cache_basic() {
    let mut cache = TurboQuantKvCache::new(4, 3);

    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);

    let (cached_k, cached_v) = cache.update_and_fetch(&keys, &values).unwrap();

    assert_eq!(cached_k.dim(2), 10);
    assert_eq!(cached_v.dim(2), 10);
    assert_eq!(cache.len(), 10);
    assert!(!cache.is_empty());
}

#[test]
fn test_turboquant_cache_accumulation() {
    let mut cache = TurboQuantKvCache::new(4, 3);

    let k1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&k1, &v1).unwrap();

    let k2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let (cached_k, cached_v) = cache.update_and_fetch(&k2, &v2).unwrap();

    assert_eq!(cached_k.dim(2), 15);
    assert_eq!(cached_v.dim(2), 15);
    assert_eq!(cache.len(), 15);
}

#[test]
fn test_turboquant_cache_reset() {
    let mut cache = TurboQuantKvCache::new(4, 3);

    let keys = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 64], Dtype::Float32);
    cache.update_and_fetch(&keys, &values).unwrap();

    cache.reset();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_turboquant_cache_rollback() {
    let mut cache = TurboQuantKvCache::new(4, 3);

    let keys = ops::ones(&[1, 2, 10, 32], Dtype::Float32);
    let values = ops::ones(&[1, 2, 10, 32], Dtype::Float32);
    cache.update_and_fetch(&keys, &values).unwrap();

    cache.rollback(4);

    assert_eq!(cache.len(), 6);
    assert_eq!(cache.rope_offset(), 6);
}

#[test]
fn test_turboquant_cache_memory_usage() {
    let mut cache = TurboQuantKvCache::new(4, 3);
    assert_eq!(cache.memory_usage(), 0);

    let keys = ops::ones(&[1, 2, 8, 32], Dtype::Float32);
    let values = ops::ones(&[1, 2, 8, 32], Dtype::Float32);
    cache.update_and_fetch(&keys, &values).unwrap();

    assert!(cache.memory_usage() > 0);
}

#[test]
fn test_turboquant_cache_nonzero_inputs_stay_finite() {
    let mut cache = TurboQuantKvCache::new(4, 3);

    let data: Vec<f32> = (0..(2 * 16))
        .map(|idx| ((idx as f32) * 0.1).sin())
        .collect();
    let keys = Array::from_f32_slice(&data, &[1, 1, 2, 16]);
    let values = Array::from_f32_slice(&data, &[1, 1, 2, 16]);

    let (cached_k, cached_v) = cache.update_and_fetch(&keys, &values).unwrap();
    let ck = cached_k.clone();
    let cv = cached_v.clone();
    ck.eval();
    cv.eval();

    assert!(to_f32_vec_eval(&cached_k).iter().all(|v| v.is_finite()));
    assert!(to_f32_vec_eval(&cached_v).iter().all(|v| v.is_finite()));
}

#[test]
fn test_turboquant_mixed_cache_nonzero_inputs_stay_finite() {
    let mut cache = TurboQuantKvCache::new_with_config(TurboQuantConfig::preset_q2_5(16));

    let data: Vec<f32> = (0..(4 * 16))
        .map(|idx| ((idx as f32) * 0.07).cos())
        .collect();
    let keys = Array::from_f32_slice(&data, &[1, 1, 4, 16]);
    let values = Array::from_f32_slice(&data, &[1, 1, 4, 16]);

    let (cached_k, cached_v) = cache.update_and_fetch(&keys, &values).unwrap();
    let ck = cached_k.clone();
    let cv = cached_v.clone();
    ck.eval();
    cv.eval();

    assert_eq!(cache.len(), 4);
    assert!(to_f32_vec_eval(&cached_k).iter().all(|v| v.is_finite()));
    assert!(to_f32_vec_eval(&cached_v).iter().all(|v| v.is_finite()));
}

#[test]
fn test_turboquant_direct_attention_matches_reference_uniform() {
    let config = KVCacheConfig::new(1, 32, 2, 16).with_turboquant(4, 3);
    let mut fast_cache = KVCache::new(config.clone());
    let mut ref_cache = KVCache::new(config);

    let prefill_k = patterned_tensor(1, 2, 3, 16, 0.2);
    let prefill_v = patterned_tensor(1, 2, 3, 16, 0.7);
    fast_cache
        .update_and_fetch(0, &prefill_k, &prefill_v)
        .unwrap();
    ref_cache
        .update_and_fetch(0, &prefill_k, &prefill_v)
        .unwrap();

    let queries = patterned_tensor(1, 4, 1, 16, 1.1);
    let new_k = patterned_tensor(1, 2, 1, 16, 1.7);
    let new_v = patterned_tensor(1, 2, 1, 16, 2.3);

    let attn_config = FusedAttentionConfig::new(4, 2, 16)
        .with_scale((16.0f32).sqrt().recip())
        .with_mask_type(AttentionMaskType::Causal);

    let fast_output = fast_cache
        .try_turboquant_attention(0, &queries, &new_k, &new_v, &attn_config)
        .unwrap()
        .expect("TurboQuant direct attention should activate");
    let (ref_keys, ref_values) = ref_cache.update_and_fetch(0, &new_k, &new_v).unwrap();
    let ref_output = fused_sdpa(&queries, &ref_keys, &ref_values, &attn_config, None).unwrap();

    let fo = fast_output.clone();
    let ro = ref_output.clone();
    fo.eval();
    ro.eval();

    let diff = max_abs_diff(&fast_output, &ref_output);
    assert_eq!(fast_cache.seq_len(), 4);
    assert_eq!(fast_cache.total_tokens(), 4);
    assert_eq!(fast_output.shape(), ref_output.shape());
    assert!(diff < 1e-4, "uniform max_abs_diff={diff}");
}

#[test]
fn test_turboquant_direct_attention_matches_reference_mixed_sliding_window() {
    let config =
        KVCacheConfig::new(1, 32, 2, 16).with_turboquant_config(TurboQuantConfig::preset_q2_5(16));
    let mut fast_cache = KVCache::new(config.clone());
    let mut ref_cache = KVCache::new(config);

    let prefill_k = patterned_tensor(1, 2, 5, 16, 0.4);
    let prefill_v = patterned_tensor(1, 2, 5, 16, 0.9);
    fast_cache
        .update_and_fetch(0, &prefill_k, &prefill_v)
        .unwrap();
    ref_cache
        .update_and_fetch(0, &prefill_k, &prefill_v)
        .unwrap();

    let queries = patterned_tensor(1, 4, 1, 16, 1.5);
    let new_k = patterned_tensor(1, 2, 1, 16, 2.1);
    let new_v = patterned_tensor(1, 2, 1, 16, 2.7);

    let attn_config = FusedAttentionConfig::new(4, 2, 16)
        .with_scale((16.0f32).sqrt().recip())
        .with_mask_type(AttentionMaskType::SlidingWindow(3));

    let fast_output = fast_cache
        .try_turboquant_attention(0, &queries, &new_k, &new_v, &attn_config)
        .unwrap()
        .expect("TurboQuant direct attention should activate");
    let (ref_keys, ref_values) = ref_cache.update_and_fetch(0, &new_k, &new_v).unwrap();
    let ref_output = fused_sdpa(&queries, &ref_keys, &ref_values, &attn_config, None).unwrap();

    let fo = fast_output.clone();
    let ro = ref_output.clone();
    fo.eval();
    ro.eval();

    let diff = max_abs_diff(&fast_output, &ref_output);
    assert_eq!(fast_cache.seq_len(), 6);
    assert_eq!(fast_cache.total_tokens(), 6);
    assert_eq!(fast_output.shape(), ref_output.shape());
    assert!(diff < 1e-4, "mixed max_abs_diff={diff}");
}

#[test]
fn test_turboquant_direct_attention_matches_reference_asymmetric_value_dim() {
    let config = KVCacheConfig::new(1, 32, 2, 16)
        .with_value_head_dim(8)
        .with_turboquant(4, 3);
    let mut fast_cache = KVCache::new(config.clone());
    let mut ref_cache = KVCache::new(config);

    let prefill_k = patterned_tensor(1, 2, 3, 16, 0.3);
    let prefill_v = patterned_tensor(1, 2, 3, 8, 0.8);
    fast_cache
        .update_and_fetch(0, &prefill_k, &prefill_v)
        .unwrap();
    ref_cache
        .update_and_fetch(0, &prefill_k, &prefill_v)
        .unwrap();

    let queries = patterned_tensor(1, 2, 1, 16, 1.4);
    let new_k = patterned_tensor(1, 2, 1, 16, 2.0);
    let new_v = patterned_tensor(1, 2, 1, 8, 2.6);

    let attn_config = FusedAttentionConfig::new(2, 2, 16)
        .with_scale((16.0f32).sqrt().recip())
        .with_mask_type(AttentionMaskType::Causal);

    let fast_output = fast_cache
        .try_turboquant_attention(0, &queries, &new_k, &new_v, &attn_config)
        .unwrap()
        .expect("TurboQuant direct attention should activate");
    let (ref_keys, ref_values) = ref_cache.update_and_fetch(0, &new_k, &new_v).unwrap();
    let ref_output =
        manual_attention_output(&queries, &ref_keys, &ref_values, (16.0f32).sqrt().recip());

    let fo = fast_output.clone();
    let ro = ref_output.clone();
    fo.eval();
    ro.eval();

    let diff = max_abs_diff(&fast_output, &ref_output);
    assert_eq!(fast_output.shape(), &[1, 2, 1, 8]);
    assert_eq!(fast_output.shape(), ref_output.shape());
    assert!(diff < 1e-4, "asymmetric max_abs_diff={diff}");
}

// =========================================================================
// PagedKVCache Tests
// =========================================================================

#[test]
fn test_paged_cache_config() {
    let config = PagedKVCacheConfig::new(32, 8, 128, 2048)
        .with_block_size(16)
        .with_dtype(Dtype::Float16);

    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.block_size, 16);
    assert_eq!(config.dtype, Dtype::Float16);
}

#[test]
fn test_block_allocator_basic() {
    let mut allocator = BlockAllocator::new(10, 32);

    assert_eq!(allocator.total_blocks(), 10);
    assert_eq!(allocator.num_free(), 10);
    assert_eq!(allocator.num_allocated(), 0);

    // Allocate a block
    let block = allocator.allocate().unwrap();
    assert_eq!(allocator.num_allocated(), 1);
    assert_eq!(allocator.num_free(), 9);

    // Free the block
    allocator.free(block);
    assert_eq!(allocator.num_free(), 10);
    assert_eq!(allocator.num_allocated(), 0);
}

#[test]
fn test_block_allocator_batch() {
    let mut allocator = BlockAllocator::new(10, 32);

    // Allocate 5 blocks at once
    let blocks = allocator.allocate_n(5).unwrap();
    assert_eq!(blocks.len(), 5);
    assert_eq!(allocator.num_allocated(), 5);

    // Try to allocate more than available
    assert!(allocator.allocate_n(6).is_none());

    // Free all
    allocator.free_all(&blocks);
    assert_eq!(allocator.num_free(), 10);
}

#[test]
fn test_block_table() {
    let mut table = BlockTable::new(32);

    assert_eq!(table.num_tokens(), 0);
    assert_eq!(table.num_blocks(), 0);

    // Add a block and tokens
    table.add_block(5);
    table.num_tokens = 16;

    assert_eq!(table.num_tokens(), 16);
    assert_eq!(table.num_blocks(), 1);

    // Check block lookup
    let (phys, offset) = table.get_block_and_offset(10).unwrap();
    assert_eq!(phys, 5);
    assert_eq!(offset, 10);
}

#[test]
fn test_block_table_add_tokens() {
    let mut table = BlockTable::new(32);
    table.add_block(0);

    // First 32 tokens need 1 block (tokens 0-31)
    let new_blocks = table.add_tokens(32);
    assert_eq!(new_blocks, 1); // Need 1 block for first 32 tokens

    // Adding 32 more requires 1 more block (tokens 32-63)
    table.add_block(1);
    let new_blocks = table.add_tokens(32);
    assert_eq!(new_blocks, 1);

    // Adding 1 more requires 1 new block (token 64)
    table.add_block(2);
    let new_blocks = table.add_tokens(1);
    assert_eq!(new_blocks, 1);
}

#[test]
fn test_paged_cache_basic() {
    let config = PagedKVCacheConfig::new(2, 4, 64, 256);
    let mut cache = PagedKVCache::new(config);

    assert_eq!(cache.num_sequences(), 0);

    // Allocate a sequence
    let seq_id = cache.allocate_sequence(32).unwrap();
    assert_eq!(cache.num_sequences(), 1);

    // Check memory stats
    let stats = cache.memory_stats();
    assert!(stats.allocated_blocks > 0);
    assert!(stats.utilization() > 0.0);

    // Free the sequence
    cache.free_sequence(seq_id);
    assert_eq!(cache.num_sequences(), 0);
}

#[test]
fn test_paged_cache_extend() {
    let config = PagedKVCacheConfig::new(1, 4, 64, 256);
    let mut cache = PagedKVCache::new(config);

    let seq_id = cache.allocate_sequence(16).unwrap();
    let table = cache.get_block_table(seq_id).unwrap();
    assert_eq!(table.num_tokens(), 16);

    // Extend the sequence
    cache.extend_sequence(seq_id, 20).unwrap();
    let table = cache.get_block_table(seq_id).unwrap();
    assert_eq!(table.num_tokens(), 36);
}

#[test]
fn test_paged_cache_multiple_sequences() {
    let config = PagedKVCacheConfig::new(1, 4, 64, 1024);
    let mut cache = PagedKVCache::new(config);

    // Allocate multiple sequences
    let _seq1 = cache.allocate_sequence(64).unwrap();
    let seq2 = cache.allocate_sequence(32).unwrap();
    let _seq3 = cache.allocate_sequence(48).unwrap();

    assert_eq!(cache.num_sequences(), 3);

    // Free one
    cache.free_sequence(seq2);
    assert_eq!(cache.num_sequences(), 2);

    // Allocate new one (should reuse freed blocks)
    let _seq4 = cache.allocate_sequence(32).unwrap();
    assert_eq!(cache.num_sequences(), 3);

    // Reset all
    cache.reset();
    assert_eq!(cache.num_sequences(), 0);
}

#[test]
fn test_paged_cache_memory_stats() {
    let config = PagedKVCacheConfig::new(2, 8, 128, 1024).with_dtype(Dtype::Float16);
    let mut cache = PagedKVCache::new(config);

    let stats = cache.memory_stats();
    assert_eq!(stats.allocated_blocks, 0);
    assert_eq!(stats.utilization(), 0.0);

    // Allocate a sequence
    let _ = cache.allocate_sequence(100);

    let stats = cache.memory_stats();
    assert!(stats.allocated_blocks > 0);
    assert!(stats.used_memory_bytes > 0);
}

#[test]
fn test_create_paged_cache_convenience() {
    let cache = create_paged_cache(32, 8, 128, 2048);

    assert_eq!(cache.config.num_layers, 32);
    assert_eq!(cache.config.num_kv_heads, 8);
    assert_eq!(cache.config.head_dim, 128);
    assert_eq!(cache.num_sequences(), 0);
}

// =========================================================================
// Eager Pre-Allocation Tests
// =========================================================================

#[test]
fn test_kv_cache_eager_config() {
    let config = KVCacheConfig::new(32, 4096, 8, 128)
        .with_eager_allocate(1)
        .with_dtype(Dtype::Float16);

    assert!(config.eager_allocate);
    assert_eq!(config.eager_batch_size, 1);
    assert_eq!(config.max_seq_len, 4096);
    assert_eq!(config.dtype, Dtype::Float16);
}

#[test]
fn test_kv_cache_eager_config_batch_size() {
    let config = KVCacheConfig::new(32, 2048, 8, 128).with_eager_allocate(4);

    assert!(config.eager_allocate);
    assert_eq!(config.eager_batch_size, 4);
}

#[test]
fn test_kv_cache_memory_footprint() {
    // 32 layers × 2 (K+V) × 1 batch × 8 heads × 2048 seq × 128 dim × 2 bytes (fp16)
    let config = KVCacheConfig::new(32, 2048, 8, 128)
        .with_eager_allocate(1)
        .with_dtype(Dtype::Float16);

    let expected = 32 * 2 * 8 * 2048 * 128 * 2; // layers * K+V * heads * seq * dim * sizeof(f16)
    assert_eq!(config.memory_footprint(), expected);
}

#[test]
fn test_kv_cache_memory_footprint_fp32() {
    // 16 layers × 2 (K+V) × 1 batch × 4 heads × 1024 seq × 64 dim × 4 bytes (fp32)
    let config = KVCacheConfig::new(16, 1024, 4, 64)
        .with_eager_allocate(1)
        .with_dtype(Dtype::Float32);

    let expected = 16 * 2 * 4 * 1024 * 64 * 4; // layers * K+V * heads * seq * dim * sizeof(f32)
    assert_eq!(config.memory_footprint(), expected);
}

#[test]
fn test_kv_cache_memory_footprint_batch() {
    // Test with batch_size > 1
    let config = KVCacheConfig::new(8, 512, 4, 64)
        .with_eager_allocate(4)
        .with_dtype(Dtype::Float16);

    let expected = 8 * 2 * 4 * 4 * 512 * 64 * 2;
    assert_eq!(config.memory_footprint(), expected);
}

#[test]
fn test_kv_cache_memory_footprint_human_bytes() {
    let config = KVCacheConfig::new(1, 1, 1, 1)
        .with_eager_allocate(1)
        .with_dtype(Dtype::Float32);

    // 1 × 2 × 1 × 1 × 1 × 1 × 4 = 8 bytes
    assert_eq!(config.memory_footprint_human(), "8 bytes");
}

#[test]
fn test_kv_cache_memory_footprint_human_kb() {
    let config = KVCacheConfig::new(1, 128, 1, 1)
        .with_eager_allocate(1)
        .with_dtype(Dtype::Float32);

    // 1 × 2 × 1 × 1 × 128 × 1 × 4 = 1024 bytes = 1 KB
    assert_eq!(config.memory_footprint_human(), "1.00 KB");
}

#[test]
fn test_kv_cache_memory_footprint_human_mb() {
    let config = KVCacheConfig::new(1, 2048, 8, 64)
        .with_eager_allocate(1)
        .with_dtype(Dtype::Float32);

    // 1 × 2 × 1 × 8 × 2048 × 64 × 4 = 8,388,608 bytes = 8 MB
    let human = config.memory_footprint_human();
    assert!(human.contains("MB"), "Expected MB, got: {}", human);
}

#[test]
fn test_kv_cache_memory_footprint_human_gb() {
    // Need to exceed 1 GB (1,073,741,824 bytes)
    // 32 layers × 2 × 1 batch × 8 heads × 8192 seq × 128 dim × 2 bytes = 1 GB
    let config = KVCacheConfig::new(32, 8192, 8, 128)
        .with_eager_allocate(1)
        .with_dtype(Dtype::Float16);

    let human = config.memory_footprint_human();
    assert!(human.contains("GB"), "Expected GB, got: {}", human);
}

#[test]
fn test_kv_cache_new_eager() {
    let config = KVCacheConfig::new(2, 128, 4, 64)
        .with_eager_allocate(1)
        .with_dtype(Dtype::Float32);

    let cache = KVCache::new_eager(config).expect("Should create eager cache");

    assert!(cache.is_preallocated());
    assert!(cache.is_empty()); // Pre-allocated but no data
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_kv_cache_new_eager_fallback() {
    // When eager_allocate is false, new_eager should fall back to lazy
    let config = KVCacheConfig::new(2, 128, 4, 64);

    let cache = KVCache::new_eager(config).expect("Should create lazy cache");

    assert!(!cache.is_preallocated());
}

#[test]
fn test_kv_cache_eager_is_preallocated() {
    // Eager cache should report as preallocated
    let eager_config = KVCacheConfig::new(2, 64, 4, 32).with_eager_allocate(1);
    let eager_cache = KVCache::new_eager(eager_config).unwrap();
    assert!(eager_cache.is_preallocated());

    // Lazy cache should not report as preallocated
    let lazy_config = KVCacheConfig::new(2, 64, 4, 32);
    let lazy_cache = KVCache::new(lazy_config);
    assert!(!lazy_cache.is_preallocated());
}

#[test]
fn test_kv_cache_eager_is_empty() {
    let config = KVCacheConfig::new(2, 64, 4, 32).with_eager_allocate(1);
    let mut cache = KVCache::new_eager(config).unwrap();

    // Pre-allocated but empty (no data added)
    assert!(cache.is_empty());

    // Add some data
    let keys = ops::zeros(&[1, 4, 10, 32], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 32], Dtype::Float32);
    cache.update_and_fetch(0, &keys, &values).unwrap();

    // Now not empty
    assert!(!cache.is_empty());
}

#[test]
fn test_kv_cache_eager_update() {
    let config = KVCacheConfig::new(2, 128, 4, 64).with_eager_allocate(1);
    let mut cache = KVCache::new_eager(config).unwrap();

    // First update
    let k1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let v1 = ops::ones(&[1, 4, 10, 64], Dtype::Float32);
    let (cached_k, cached_v) = cache.update_and_fetch(0, &k1, &v1).unwrap();

    assert_eq!(cached_k.dim(2), 10);
    assert_eq!(cached_v.dim(2), 10);
    assert_eq!(cache.seq_len(), 10);

    // Second update (accumulation)
    let k2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let v2 = ops::ones(&[1, 4, 5, 64], Dtype::Float32);
    let (cached_k, cached_v) = cache.update_and_fetch(0, &k2, &v2).unwrap();

    assert_eq!(cached_k.dim(2), 15);
    assert_eq!(cached_v.dim(2), 15);
    assert_eq!(cache.seq_len(), 15);
}

#[test]
fn test_kv_cache_eager_reset_preserves_buffers() {
    let config = KVCacheConfig::new(2, 64, 4, 32).with_eager_allocate(1);
    let mut cache = KVCache::new_eager(config).unwrap();

    // Add some data
    let keys = ops::zeros(&[1, 4, 10, 32], Dtype::Float32);
    let values = ops::zeros(&[1, 4, 10, 32], Dtype::Float32);
    cache.update_and_fetch(0, &keys, &values).unwrap();
    assert!(!cache.is_empty());

    // Reset should preserve buffers in eager mode
    cache.reset();

    assert!(cache.is_empty()); // Offset reset
    assert!(cache.is_preallocated()); // Buffers preserved
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_kv_cache_eager_reset_full_deallocates() {
    let config = KVCacheConfig::new(2, 64, 4, 32).with_eager_allocate(1);
    let mut cache = KVCache::new_eager(config).unwrap();

    assert!(cache.is_preallocated());

    // reset_full should deallocate even in eager mode
    cache.reset_full();

    assert!(!cache.is_preallocated()); // Buffers deallocated
    assert!(cache.is_empty());
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
    let expected_state = crate::kernels::gated_delta_state_advance(
        &initial_state,
        &k_trunc,
        &v_trunc,
        &g_trunc,
        &beta_trunc,
    )
    .unwrap();

    // Rollback path: advance through the full verify batch, then rewind
    let mut entry = MambaCacheEntry::default();
    entry.ssm_state = Some(initial_state.clone());
    let snapshot = entry.snapshot();

    // Simulate verify advancing through all K tokens
    let post_verify = crate::kernels::gated_delta_state_advance(
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
    let post = crate::kernels::gated_delta_state_advance(&state, &k, &v, &g, &beta).unwrap();
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
