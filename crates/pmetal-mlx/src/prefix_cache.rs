//! Prefix caching for efficient RL training.
//!
//! This module provides prefix caching functionality that stores KV cache states
//! for prompt prefixes, enabling efficient generation of multiple completions
//! from the same prompt without recomputing the prompt's forward pass each time.
//!
//! ## Use Case: GRPO/DPO Training
//!
//! During RL training (GRPO, DPO), we often generate multiple completions from
//! the same prompt. Without prefix caching:
//!
//! ```text
//! Prompt: "What is 2+2?"
//! Completion 1: Full forward pass (prompt) + decode
//! Completion 2: Full forward pass (prompt) + decode  // Redundant!
//! Completion 3: Full forward pass (prompt) + decode  // Redundant!
//! ```
//!
//! With prefix caching:
//!
//! ```text
//! Prompt: "What is 2+2?"
//! Completion 1: Full forward pass (prompt) + decode, cache KV
//! Completion 2: Clone cached KV + decode  // 2-5x faster!
//! Completion 3: Clone cached KV + decode
//! ```
//!
//! ## Implementation
//!
//! The cache uses a hash of the prompt tokens as the key. When a prompt is
//! processed, we save a snapshot of the KV cache state (all layer K and V tensors).
//! For subsequent generations with the same prompt, we restore this snapshot
//! to avoid recomputing the prompt's forward pass.
//!
//! ## Memory Considerations
//!
//! The prefix cache stores full KV states which can be memory-intensive for:
//! - Long prompts
//! - Large models
//! - Many cached prefixes
//!
//! Use `max_entries` to limit cache size and `clear()` to free memory when needed.

use mlx_rs::{error::Exception, Array};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::kv_cache::{KVCache, KVCacheConfig};

/// A snapshot of the KV cache state at a particular point.
///
/// This stores all the K and V tensors from each layer, allowing us to
/// restore the cache to this exact state for subsequent generations.
#[derive(Debug, Clone)]
pub struct KVCacheSnapshot {
    /// Cached keys per layer: Vec<[B, heads, seq, head_dim]>
    keys: Vec<Array>,
    /// Cached values per layer: Vec<[B, heads, seq, head_dim]>
    values: Vec<Array>,
    /// Sequence length at snapshot time.
    seq_len: usize,
    /// Total tokens processed at snapshot time.
    total_tokens: usize,
}

impl KVCacheSnapshot {
    /// Create a snapshot from a KV cache.
    pub fn from_cache(cache: &KVCache) -> Self {
        let num_layers = cache.config().num_layers;
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            if let Some((k, v)) = cache.get(layer_idx) {
                keys.push(k.clone());
                values.push(v.clone());
            }
        }

        Self {
            keys,
            values,
            seq_len: cache.seq_len(),
            total_tokens: cache.total_tokens(),
        }
    }

    /// Restore a KV cache from this snapshot.
    ///
    /// Creates a new KV cache with the same configuration and restores
    /// all the cached K/V tensors to match this snapshot's state.
    pub fn restore(&self, config: KVCacheConfig) -> Result<KVCache, Exception> {
        let mut cache = KVCache::new(config);

        // Restore each layer's cache
        for (layer_idx, (k, v)) in self.keys.iter().zip(self.values.iter()).enumerate() {
            // We need to update the cache with these tensors
            // Since update_and_fetch expects "new" tensors, we just pass the full cached tensors
            // and reset first to clear any existing state
            cache.update_and_fetch(layer_idx, k, v)?;
        }

        Ok(cache)
    }

    /// Get the sequence length of this snapshot.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get the number of layers in this snapshot.
    pub fn num_layers(&self) -> usize {
        self.keys.len()
    }

    /// Estimate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;
        for k in &self.keys {
            let elements: usize = k.shape().iter().map(|&d| d as usize).product();
            total += elements * 4; // Assume f32
        }
        for v in &self.values {
            let elements: usize = v.shape().iter().map(|&d| d as usize).product();
            total += elements * 4;
        }
        total
    }
}

/// Hash function for prompt tokens.
fn hash_tokens(tokens: &[u32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

/// Prefix cache for efficient RL training.
///
/// Caches KV states for prompt prefixes, enabling fast generation of multiple
/// completions from the same prompt.
#[derive(Debug)]
pub struct PrefixCache {
    /// Cached snapshots keyed by prompt hash.
    cache: HashMap<u64, KVCacheSnapshot>,
    /// Maximum number of entries to keep.
    max_entries: usize,
    /// LRU order (most recently used at end).
    lru_order: Vec<u64>,
    /// Cache hit count.
    hits: usize,
    /// Cache miss count.
    misses: usize,
}

impl PrefixCache {
    /// Create a new prefix cache with the given maximum entry count.
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of unique prompts to cache
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries,
            lru_order: Vec::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Check if a prompt prefix is cached.
    pub fn contains(&self, tokens: &[u32]) -> bool {
        let hash = hash_tokens(tokens);
        self.cache.contains_key(&hash)
    }

    /// Get a cached snapshot for a prompt, if available.
    ///
    /// Returns None if the prompt is not cached.
    /// Updates LRU order on hit.
    pub fn get(&mut self, tokens: &[u32]) -> Option<&KVCacheSnapshot> {
        let hash = hash_tokens(tokens);

        if let Some(snapshot) = self.cache.get(&hash) {
            self.hits += 1;
            // Update LRU order
            self.lru_order.retain(|&h| h != hash);
            self.lru_order.push(hash);
            Some(snapshot)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Get a KV cache restored from the cached snapshot.
    ///
    /// This creates a new KV cache with the prompt's cached state,
    /// ready for continuation with new tokens.
    pub fn get_cache(
        &mut self,
        tokens: &[u32],
        config: KVCacheConfig,
    ) -> Result<Option<KVCache>, Exception> {
        if let Some(snapshot) = self.get(tokens) {
            Ok(Some(snapshot.restore(config)?))
        } else {
            Ok(None)
        }
    }

    /// Insert a prompt's KV cache state into the cache.
    ///
    /// If the cache is full, the least recently used entry is evicted.
    pub fn insert(&mut self, tokens: &[u32], cache: &KVCache) {
        let hash = hash_tokens(tokens);

        // Evict LRU if at capacity
        if self.cache.len() >= self.max_entries && !self.cache.contains_key(&hash) {
            if let Some(lru_hash) = self.lru_order.first().copied() {
                self.cache.remove(&lru_hash);
                self.lru_order.remove(0);
            }
        }

        // Insert new snapshot
        let snapshot = KVCacheSnapshot::from_cache(cache);
        self.cache.insert(hash, snapshot);

        // Update LRU order
        self.lru_order.retain(|&h| h != hash);
        self.lru_order.push(hash);
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
    }

    /// Get the number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache hit count.
    pub fn hits(&self) -> usize {
        self.hits
    }

    /// Get cache miss count.
    pub fn misses(&self) -> usize {
        self.misses
    }

    /// Get cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Reset hit/miss counters.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    /// Estimate total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.cache.values().map(|s| s.memory_usage()).sum()
    }
}

/// Builder for prefix-cached generation.
///
/// This helper manages the workflow of:
/// 1. Checking if a prompt is cached
/// 2. Either restoring from cache or doing full forward pass
/// 3. Caching the result for future use
#[derive(Debug)]
pub struct PrefixCachedGenerator {
    /// The prefix cache.
    cache: PrefixCache,
    /// KV cache configuration for restoration.
    kv_config: KVCacheConfig,
}

impl PrefixCachedGenerator {
    /// Create a new prefix-cached generator.
    pub fn new(max_cached_prompts: usize, kv_config: KVCacheConfig) -> Self {
        Self {
            cache: PrefixCache::new(max_cached_prompts),
            kv_config,
        }
    }

    /// Try to get a pre-filled KV cache for the given prompt.
    ///
    /// Returns `Some(cache)` if the prompt was cached, `None` otherwise.
    /// The caller should then do a full forward pass if None is returned.
    pub fn try_get_cache(&mut self, prompt_tokens: &[u32]) -> Result<Option<KVCache>, Exception> {
        self.cache.get_cache(prompt_tokens, self.kv_config.clone())
    }

    /// Cache the KV state after processing a prompt.
    ///
    /// Call this after doing a full forward pass on a prompt to enable
    /// future cache hits.
    pub fn cache_prompt(&mut self, prompt_tokens: &[u32], kv_cache: &KVCache) {
        self.cache.insert(prompt_tokens, kv_cache);
    }

    /// Clear all cached prompts.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> (usize, usize, f64) {
        (
            self.cache.hits(),
            self.cache.misses(),
            self.cache.hit_rate(),
        )
    }

    /// Get the underlying prefix cache for direct manipulation.
    pub fn cache(&self) -> &PrefixCache {
        &self.cache
    }

    /// Get mutable access to the underlying prefix cache.
    pub fn cache_mut(&mut self) -> &mut PrefixCache {
        &mut self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> KVCacheConfig {
        KVCacheConfig::new(2, 100, 4, 64)
    }

    #[test]
    fn test_prefix_cache_basic() {
        let mut cache = PrefixCache::new(10);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        // Create a dummy KV cache with some data
        let config = create_test_config();
        let mut kv_cache = KVCache::new(config);

        // Add some data to the cache
        let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        kv_cache.update_and_fetch(0, &keys, &values).unwrap();
        kv_cache.update_and_fetch(1, &keys, &values).unwrap();

        // Insert into prefix cache
        let tokens = vec![1, 2, 3, 4, 5];
        cache.insert(&tokens, &kv_cache);

        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&tokens));
        assert!(!cache.contains(&[1, 2, 3])); // Different tokens
    }

    #[test]
    fn test_prefix_cache_hit_miss() {
        let mut cache = PrefixCache::new(10);
        let config = create_test_config();
        let mut kv_cache = KVCache::new(config);

        let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        kv_cache.update_and_fetch(0, &keys, &values).unwrap();
        kv_cache.update_and_fetch(1, &keys, &values).unwrap();

        let tokens = vec![1, 2, 3];
        cache.insert(&tokens, &kv_cache);

        // Miss on different tokens
        assert!(cache.get(&[4, 5, 6]).is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);

        // Hit on same tokens
        assert!(cache.get(&tokens).is_some());
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);

        assert!((cache.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prefix_cache_lru_eviction() {
        let mut cache = PrefixCache::new(2); // Only 2 entries
        let config = create_test_config();

        // Create 3 different caches
        for i in 0..3 {
            let mut kv_cache = KVCache::new(config.clone());
            let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
            let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
            kv_cache.update_and_fetch(0, &keys, &values).unwrap();
            kv_cache.update_and_fetch(1, &keys, &values).unwrap();

            let tokens = vec![i as u32];
            cache.insert(&tokens, &kv_cache);
        }

        // Should have evicted the first entry
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains(&[0])); // Evicted
        assert!(cache.contains(&[1]));
        assert!(cache.contains(&[2]));
    }

    #[test]
    fn test_prefix_cache_restore() {
        let mut cache = PrefixCache::new(10);
        let config = create_test_config();
        let mut kv_cache = KVCache::new(config.clone());

        // Add data with specific values
        let keys = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
        let values = Array::ones::<f32>(&[1, 4, 10, 64]).unwrap();
        kv_cache.update_and_fetch(0, &keys, &values).unwrap();
        kv_cache.update_and_fetch(1, &keys, &values).unwrap();

        let tokens = vec![10, 20, 30];
        cache.insert(&tokens, &kv_cache);

        // Restore from cache
        let restored = cache.get_cache(&tokens, config.clone()).unwrap().unwrap();

        // Verify restored cache has same state
        assert_eq!(restored.seq_len(), 10);

        let (k, v) = restored.get(0).unwrap();
        assert_eq!(k.dim(2), 10); // seq_len
        assert_eq!(v.dim(2), 10);
    }

    #[test]
    fn test_kv_snapshot() {
        let config = create_test_config();
        let mut kv_cache = KVCache::new(config.clone());

        let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        kv_cache.update_and_fetch(0, &keys, &values).unwrap();
        kv_cache.update_and_fetch(1, &keys, &values).unwrap();

        let snapshot = KVCacheSnapshot::from_cache(&kv_cache);

        assert_eq!(snapshot.seq_len(), 10);
        assert_eq!(snapshot.num_layers(), 2);
        assert!(snapshot.memory_usage() > 0);
    }

    #[test]
    fn test_prefix_cached_generator() {
        let config = create_test_config();
        let mut generator = PrefixCachedGenerator::new(5, config.clone());

        // First generation - cache miss
        let tokens = vec![100, 200, 300];
        let cached = generator.try_get_cache(&tokens).unwrap();
        assert!(cached.is_none());

        // Simulate forward pass and cache
        let mut kv_cache = KVCache::new(config.clone());
        let keys = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        let values = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        kv_cache.update_and_fetch(0, &keys, &values).unwrap();
        kv_cache.update_and_fetch(1, &keys, &values).unwrap();
        generator.cache_prompt(&tokens, &kv_cache);

        // Second generation - cache hit
        let cached = generator.try_get_cache(&tokens).unwrap();
        assert!(cached.is_some());

        let (hits, misses, rate) = generator.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert!((rate - 0.5).abs() < 0.01);
    }
}
