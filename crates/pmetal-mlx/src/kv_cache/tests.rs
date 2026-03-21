//! Tests for all KV cache types.

use super::*;
use mlx_rs::{Array, Dtype};

#[test]
fn test_kv_cache_config() {
    let config = KVCacheConfig::new(32, 2048, 8, 128)
        .with_dtype(Dtype::Float16)
        .with_sliding_window(512);

    assert_eq!(config.num_layers, 32);
    assert_eq!(config.max_seq_len, 2048);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 128);
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
    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();

    let (cached_k, cached_v) = cache.update_and_fetch(0, &keys, &values).unwrap();

    // Seq is now axis 2
    assert_eq!(cached_k.dim(2), 10);
    assert_eq!(cached_v.dim(2), 10);
    assert_eq!(cache.seq_len(), 10);
    assert!(!cache.is_empty());
}

#[test]
fn test_kv_cache_accumulation() {
    let config = KVCacheConfig::new(1, 100, 4, 64);
    let mut cache = KVCache::new(config);

    // First update: 10 tokens [B, heads, seq, head_dim]
    let k1 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    let v1 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    cache.update_and_fetch(0, &k1, &v1).unwrap();

    assert_eq!(cache.seq_len(), 10);

    // Second update: 5 more tokens
    let k2 = Array::ones::<f32>(&[1, 4, 5, 64]).unwrap();
    let v2 = Array::ones::<f32>(&[1, 4, 5, 64]).unwrap();
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
    let k1 = Array::ones::<f32>(&[1, 4, 15, 64]).unwrap();
    let v1 = Array::ones::<f32>(&[1, 4, 15, 64]).unwrap();
    cache.update_and_fetch(0, &k1, &v1).unwrap();

    assert_eq!(cache.seq_len(), 15);

    // Add 10 more - should trigger sliding window
    let k2 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    let v2 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
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
    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    cache.update_and_fetch(0, &keys, &values).unwrap();
    cache.update_and_fetch(1, &keys, &values).unwrap();

    assert!(!cache.is_empty());

    cache.reset();

    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.total_tokens(), 0);
}

#[test]
fn test_kv_cache_rope_offset() {
    let config = KVCacheConfig::new(1, 100, 4, 64);
    let mut cache = KVCache::new(config);

    assert_eq!(cache.rope_offset(), 0);

    // [B, heads, seq, head_dim] format
    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    cache.update_and_fetch(0, &keys, &values).unwrap();

    assert_eq!(cache.rope_offset(), 10);
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
        let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
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
    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
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
    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();

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
    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();

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
    let k1 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    let v1 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    cache.update_and_fetch(&k1, &v1).unwrap();

    assert_eq!(cache.len(), 10);

    // Second update: 5 more tokens
    let k2 = Array::ones::<f32>(&[1, 4, 5, 64]).unwrap();
    let v2 = Array::ones::<f32>(&[1, 4, 5, 64]).unwrap();
    let (cached_k, cached_v) = cache.update_and_fetch(&k2, &v2).unwrap();

    assert_eq!(cached_k.dim(2), 15);
    assert_eq!(cached_v.dim(2), 15);
    assert_eq!(cache.len(), 15);
}

#[test]
fn test_rotating_cache_rotation() {
    let mut cache = RotatingKVCache::new(20, 0);

    // Fill beyond max_size
    let k1 = Array::ones::<f32>(&[1, 4, 15, 64]).unwrap();
    let v1 = Array::ones::<f32>(&[1, 4, 15, 64]).unwrap();
    cache.update_and_fetch(&k1, &v1).unwrap();

    let k2 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    let v2 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
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
    let k1 = Array::ones::<f32>(&[1, 4, 15, 64]).unwrap();
    let v1 = Array::ones::<f32>(&[1, 4, 15, 64]).unwrap();
    cache.update_and_fetch(&k1, &v1).unwrap();

    assert_eq!(cache.len(), 15);

    // Add more tokens
    let k2 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    let v2 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    cache.update_and_fetch(&k2, &v2).unwrap();

    // Cache should have rotated but kept initial tokens
    assert!(cache.len() <= 20);
}

#[test]
fn test_rotating_cache_reset() {
    let mut cache = RotatingKVCache::new(100, 0);

    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
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
        let k = Array::ones::<f32>(&[1, 4, 1, 64]).unwrap();
        let v = Array::ones::<f32>(&[1, 4, 1, 64]).unwrap();
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

    let k = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    let v = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    cache.update_and_fetch(&k, &v).unwrap();

    assert!(cache.is_trimmable()); // Under max_size

    // Fill to max
    let k2 = Array::ones::<f32>(&[1, 4, 15, 64]).unwrap();
    let v2 = Array::ones::<f32>(&[1, 4, 15, 64]).unwrap();
    cache.update_and_fetch(&k2, &v2).unwrap();

    assert!(!cache.is_trimmable()); // At or over max_size
}

#[test]
fn test_rotating_cache_rope_offset() {
    let mut cache = RotatingKVCache::new(100, 0);

    assert_eq!(cache.rope_offset(), 0);

    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
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
    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();

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
    let k1 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    let v1 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    cache.update_and_fetch(&k1, &v1).unwrap();

    // Second update
    let k2 = Array::ones::<f32>(&[1, 4, 5, 64]).unwrap();
    let v2 = Array::ones::<f32>(&[1, 4, 5, 64]).unwrap();
    let (cached_k, cached_v) = cache.update_and_fetch(&k2, &v2).unwrap();

    assert_eq!(cached_k.dim(2), 15);
    assert_eq!(cached_v.dim(2), 15);
    assert_eq!(cache.len(), 15);
}

#[test]
fn test_quantized_cache_reset() {
    let mut cache = QuantizedKVCache::new(8, 64);

    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
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

    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    cache.update_and_fetch(&keys, &values).unwrap();

    // Should have some memory usage now
    assert!(cache.memory_usage() > 0);
}

#[test]
fn test_quantized_cache_rope_offset() {
    let mut cache = QuantizedKVCache::new(8, 64);

    assert_eq!(cache.rope_offset(), 0);

    let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
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
    let keys = Array::zeros::<f32>(&[1, 4, 10, 32]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 32]).unwrap();
    cache.update_and_fetch(0, &keys, &values).unwrap();

    // Now not empty
    assert!(!cache.is_empty());
}

#[test]
fn test_kv_cache_eager_update() {
    let config = KVCacheConfig::new(2, 128, 4, 64).with_eager_allocate(1);
    let mut cache = KVCache::new_eager(config).unwrap();

    // First update
    let k1 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    let v1 = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
    let (cached_k, cached_v) = cache.update_and_fetch(0, &k1, &v1).unwrap();

    assert_eq!(cached_k.dim(2), 10);
    assert_eq!(cached_v.dim(2), 10);
    assert_eq!(cache.seq_len(), 10);

    // Second update (accumulation)
    let k2 = Array::ones::<f32>(&[1, 4, 5, 64]).unwrap();
    let v2 = Array::ones::<f32>(&[1, 4, 5, 64]).unwrap();
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
    let keys = Array::zeros::<f32>(&[1, 4, 10, 32]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 32]).unwrap();
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
    let keys = Array::zeros::<f32>(&[1, 4, 10, 32]).unwrap();
    let values = Array::zeros::<f32>(&[1, 4, 10, 32]).unwrap();
    cache.update_and_fetch(0, &keys, &values).unwrap();
    assert!(!cache.is_empty());

    // Reset should deallocate in lazy mode
    cache.reset();

    assert!(cache.is_empty());
    // Verify buffers are deallocated by checking get returns None
    assert!(cache.get(0).is_none());
}

#[test]
fn test_kv_cache_eager_reuse_after_reset() {
    let config = KVCacheConfig::new(2, 64, 4, 32).with_eager_allocate(1);
    let mut cache = KVCache::new_eager(config).unwrap();

    // First generation
    let k1 = Array::ones::<f32>(&[1, 4, 20, 32]).unwrap();
    let v1 = Array::ones::<f32>(&[1, 4, 20, 32]).unwrap();
    cache.update_and_fetch(0, &k1, &v1).unwrap();
    assert_eq!(cache.seq_len(), 20);

    // Reset for new generation
    cache.reset();
    assert!(cache.is_empty());
    assert!(cache.is_preallocated());

    // Second generation (reuses pre-allocated buffers)
    let k2 = Array::ones::<f32>(&[1, 4, 15, 32]).unwrap();
    let v2 = Array::ones::<f32>(&[1, 4, 15, 32]).unwrap();
    cache.update_and_fetch(0, &k2, &v2).unwrap();
    assert_eq!(cache.seq_len(), 15);
}
