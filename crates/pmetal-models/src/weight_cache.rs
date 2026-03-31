//! LRU weight cache for layer-at-a-time weight loading.
//!
//! Enables models larger than device memory on a single node by keeping
//! only a sliding window of layer weights resident. Layers outside the
//! window are evicted and reloaded from mmap on demand.
//!
//! # Status: Not yet integrated
//!
//! `WeightCache` needs to be wired into the model loading pipeline to enable
//! layer-at-a-time inference for models larger than device memory. The LRU
//! cache with refcounting is fully implemented and tested.

use pmetal_bridge::compat::{Array, Exception};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

/// LRU weight cache that keeps at most `max_resident` layers in memory.
pub struct WeightCache {
    /// Model directory for loading weights.
    model_dir: PathBuf,
    /// Maximum number of resident layers.
    max_resident: usize,
    /// Currently loaded layer weights: layer_idx → param_name → Array.
    resident: HashMap<usize, HashMap<String, Array>>,
    /// LRU order: front = least recently used.
    lru_order: VecDeque<usize>,
    /// Reference counts: layer_idx → number of active users.
    ///
    /// A layer with a refcount > 0 must not be evicted — it is currently
    /// being read by in-flight compute.  Callers must call
    /// `increase_reference` before using a layer and `decrease_reference`
    /// when they are finished.
    references: HashMap<usize, usize>,
    /// Total number of layers in the model.
    #[allow(dead_code)] // Kept for capacity planning and future eviction policy improvements
    num_layers: usize,
    /// Cache statistics.
    stats: CacheStats,
}

/// Cache hit/miss statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_loaded: u64,
    /// Number of times eviction was skipped because the candidate layer had
    /// active references (i.e. in-flight compute was using it).
    pub reference_held_eviction_skips: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl WeightCache {
    /// Create a new weight cache.
    ///
    /// `model_dir`: path to the model directory containing safetensors files.
    /// `max_resident`: maximum number of layers to keep in memory (the "window size").
    /// `num_layers`: total number of decoder layers in the model.
    pub fn new(model_dir: PathBuf, max_resident: usize, num_layers: usize) -> Self {
        assert!(max_resident > 0, "window size must be at least 1");
        Self {
            model_dir,
            max_resident,
            resident: HashMap::new(),
            lru_order: VecDeque::new(),
            references: HashMap::new(),
            num_layers,
            stats: CacheStats::default(),
        }
    }

    /// Get the model directory.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Number of currently resident layers.
    pub fn num_resident(&self) -> usize {
        self.resident.len()
    }

    /// Check if a layer's weights are currently resident.
    pub fn is_resident(&self, layer_idx: usize) -> bool {
        self.resident.contains_key(&layer_idx)
    }

    /// Increment the reference count for `layer_idx`.
    ///
    /// A layer whose refcount is greater than zero will be skipped during LRU
    /// eviction.  Callers must pair every `increase_reference` with exactly
    /// one `decrease_reference` once the compute that needs the weights
    /// completes.
    pub fn increase_reference(&mut self, layer_idx: usize) {
        let count = self.references.entry(layer_idx).or_insert(0);
        *count += 1;
        tracing::trace!(
            "Layer {} reference count increased to {}",
            layer_idx,
            *count
        );
    }

    /// Decrement the reference count for `layer_idx`.
    ///
    /// The count is clamped at zero; it is never decremented below zero.
    pub fn decrease_reference(&mut self, layer_idx: usize) {
        if let Some(count) = self.references.get_mut(&layer_idx) {
            if *count > 0 {
                *count -= 1;
            }
            tracing::trace!(
                "Layer {} reference count decreased to {}",
                layer_idx,
                *count
            );
            if *count == 0 {
                self.references.remove(&layer_idx);
            }
        }
    }

    /// Return the current reference count for `layer_idx` (0 if not tracked).
    pub fn reference_count(&self, layer_idx: usize) -> usize {
        self.references.get(&layer_idx).copied().unwrap_or(0)
    }

    /// Get weights for a layer, loading from disk if necessary.
    ///
    /// Returns a reference to the layer's weight HashMap.
    /// If the layer is not resident, it will be loaded and the LRU
    /// eviction policy will evict the oldest layer if at capacity.
    pub fn get_or_load(
        &mut self,
        layer_idx: usize,
    ) -> Result<&HashMap<String, Array>, pmetal_bridge::compat::Exception> {
        if self.resident.contains_key(&layer_idx) {
            // Cache hit — move to back of LRU
            self.stats.hits += 1;
            self.touch(layer_idx);
        } else {
            // Cache miss — load and possibly evict
            self.stats.misses += 1;
            self.load_layer(layer_idx)?;
        }

        Ok(self.resident.get(&layer_idx).unwrap())
    }

    /// Prefetch a layer's weights (hint: will be needed soon).
    ///
    /// If the layer is already resident, this is a no-op and returns
    /// immediately.  Otherwise, loads the layer synchronously (evicting an
    /// unreferenced layer if at capacity).
    ///
    /// Future work: when async I/O is available this should submit a
    /// non-blocking load so compute can overlap with I/O.
    pub fn prefetch(&mut self, layer_idx: usize) -> Result<(), pmetal_bridge::compat::Exception> {
        if self.resident.contains_key(&layer_idx) {
            tracing::trace!("Prefetch layer {}: already resident, skipping", layer_idx);
            return Ok(());
        }
        tracing::debug!(
            "Prefetch layer {}: not resident, loading synchronously (async not yet implemented)",
            layer_idx
        );
        self.load_layer(layer_idx)?;
        Ok(())
    }

    /// Explicitly evict a layer's weights from memory.
    ///
    /// If the layer has active references (refcount > 0), it is **not**
    /// evicted and this method returns without modifying the cache.
    pub fn evict(&mut self, layer_idx: usize) {
        if self.references.get(&layer_idx).copied().unwrap_or(0) > 0 {
            tracing::warn!(
                "Refused to evict layer {} — it has {} active reference(s)",
                layer_idx,
                self.references[&layer_idx]
            );
            self.stats.reference_held_eviction_skips += 1;
            return;
        }
        if self.resident.remove(&layer_idx).is_some() {
            self.lru_order.retain(|&idx| idx != layer_idx);
            self.stats.evictions += 1;
        }
    }

    /// Touch a layer (mark as most recently used).
    fn touch(&mut self, layer_idx: usize) {
        self.lru_order.retain(|&idx| idx != layer_idx);
        self.lru_order.push_back(layer_idx);
    }

    /// Load a layer's weights from safetensors and manage cache capacity.
    fn load_layer(&mut self, layer_idx: usize) -> Result<(), pmetal_bridge::compat::Exception> {
        // Evict LRU layers until we have room, skipping any that are referenced.
        while self.resident.len() >= self.max_resident {
            // Find the position of the first unreferenced layer in LRU order.
            let evict_pos = self
                .lru_order
                .iter()
                .position(|&idx| self.references.get(&idx).copied().unwrap_or(0) == 0);

            if let Some(pos) = evict_pos {
                let evict_idx = self.lru_order.remove(pos).unwrap();
                self.resident.remove(&evict_idx);
                self.stats.evictions += 1;
                tracing::debug!("Evicted layer {} from weight cache", evict_idx);
            } else {
                // All resident layers are referenced — cannot evict any.
                // Log and break to avoid an infinite loop; the layer will still
                // be inserted, temporarily exceeding max_resident.
                tracing::warn!(
                    "Weight cache at capacity ({} resident) but all layers are referenced; \
                     loading layer {} anyway (max_resident temporarily exceeded)",
                    self.resident.len(),
                    layer_idx
                );
                self.stats.reference_held_eviction_skips += 1;
                break;
            }
        }

        // Load the layer's weights from safetensors
        let weights = load_layer_weights(&self.model_dir, layer_idx)?;

        let bytes: usize = weights.values().map(|arr| arr.nbytes()).sum();
        self.stats.bytes_loaded += bytes as u64;

        tracing::debug!(
            "Loaded layer {} weights ({} tensors, {:.1} MB)",
            layer_idx,
            weights.len(),
            bytes as f64 / 1_048_576.0
        );

        self.resident.insert(layer_idx, weights);
        self.lru_order.push_back(layer_idx);

        Ok(())
    }
}

/// Load weights for a single decoder layer from safetensors files.
///
/// Uses the existing `load_weights` infrastructure and filters by layer prefix.
/// This avoids duplicating dtype conversion logic.
pub fn load_layer_weights(
    model_dir: &Path,
    layer_idx: usize,
) -> Result<HashMap<String, Array>, pmetal_bridge::compat::Exception> {
    let prefix = format!("model.layers.{layer_idx}.");
    let alt_prefix = format!("transformer.h.{layer_idx}.");

    // Load all weights using the existing loader (handles all dtypes safely)
    let all_weights = crate::loader::load_weights(model_dir)
        .map_err(|e| pmetal_bridge::compat::Exception::custom(format!("load_weights: {e}")))?;

    // Filter to just this layer's weights
    let layer_weights: HashMap<String, Array> = all_weights
        .into_iter()
        .filter(|(name, _)| name.starts_with(&prefix) || name.starts_with(&alt_prefix))
        .collect();

    Ok(layer_weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: insert a layer directly into the cache internals without I/O.
    fn insert_layer(cache: &mut WeightCache, layer_idx: usize) {
        cache.resident.insert(layer_idx, HashMap::new());
        cache.lru_order.push_back(layer_idx);
    }

    #[test]
    fn cache_eviction() {
        // Create a cache with max 2 resident layers
        let dir = PathBuf::from("/nonexistent"); // won't actually load
        let mut cache = WeightCache::new(dir, 2, 32);

        insert_layer(&mut cache, 0);
        insert_layer(&mut cache, 1);

        assert_eq!(cache.num_resident(), 2);
        assert!(cache.is_resident(0));
        assert!(cache.is_resident(1));

        // Evict layer 0 (no references held)
        cache.evict(0);
        assert_eq!(cache.num_resident(), 1);
        assert!(!cache.is_resident(0));
        assert!(cache.is_resident(1));
    }

    #[test]
    fn lru_order() {
        let dir = PathBuf::from("/nonexistent");
        let mut cache = WeightCache::new(dir, 2, 32);

        insert_layer(&mut cache, 0);
        insert_layer(&mut cache, 1);

        // Touch layer 0 (now 1 is LRU)
        cache.touch(0);
        assert_eq!(cache.lru_order[0], 1); // 1 is now LRU
        assert_eq!(cache.lru_order[1], 0); // 0 is MRU
    }

    #[test]
    fn evict_blocked_by_reference() {
        let dir = PathBuf::from("/nonexistent");
        let mut cache = WeightCache::new(dir, 2, 32);

        insert_layer(&mut cache, 0);

        // Acquire a reference — eviction must be refused.
        cache.increase_reference(0);
        cache.evict(0);
        assert!(
            cache.is_resident(0),
            "evict() must not remove a layer with active references"
        );
        assert_eq!(cache.stats.reference_held_eviction_skips, 1);

        // Release the reference — now eviction succeeds.
        cache.decrease_reference(0);
        cache.evict(0);
        assert!(
            !cache.is_resident(0),
            "evict() must remove layer once refcount is 0"
        );
    }

    #[test]
    fn reference_count_roundtrip() {
        let dir = PathBuf::from("/nonexistent");
        let mut cache = WeightCache::new(dir, 4, 32);

        assert_eq!(cache.reference_count(7), 0);

        cache.increase_reference(7);
        assert_eq!(cache.reference_count(7), 1);

        cache.increase_reference(7);
        assert_eq!(cache.reference_count(7), 2);

        cache.decrease_reference(7);
        assert_eq!(cache.reference_count(7), 1);

        cache.decrease_reference(7);
        assert_eq!(cache.reference_count(7), 0);

        // Extra decrease must not underflow.
        cache.decrease_reference(7);
        assert_eq!(cache.reference_count(7), 0);
    }

    #[test]
    fn lru_eviction_skips_referenced_layer() {
        // max_resident = 2; layers 0 and 1 are inserted; layer 0 is the LRU
        // but has a reference, so layer 1 (next in LRU) should be evicted instead.
        let dir = PathBuf::from("/nonexistent");
        let mut cache = WeightCache::new(dir, 2, 32);

        insert_layer(&mut cache, 0); // LRU
        insert_layer(&mut cache, 1); // MRU

        // Hold a reference on layer 0 (the LRU candidate).
        cache.increase_reference(0);

        // Simulate the eviction scan that load_layer performs.
        // Find first unreferenced layer in LRU order.
        let evict_pos = cache
            .lru_order
            .iter()
            .position(|&idx| cache.references.get(&idx).copied().unwrap_or(0) == 0);
        assert_eq!(
            evict_pos,
            Some(1),
            "layer 1 should be the first evictable candidate"
        );

        let evict_idx = cache.lru_order.remove(evict_pos.unwrap()).unwrap();
        assert_eq!(evict_idx, 1, "layer 1 should be chosen for eviction");
        cache.resident.remove(&evict_idx);

        assert!(cache.is_resident(0), "layer 0 (referenced) must remain");
        assert!(!cache.is_resident(1), "layer 1 must be evicted");
    }

    #[test]
    fn prefetch_noop_when_resident() {
        let dir = PathBuf::from("/nonexistent");
        let mut cache = WeightCache::new(dir, 4, 32);

        insert_layer(&mut cache, 3);

        // prefetch on a resident layer must not attempt I/O.
        // The real load_layer would fail on /nonexistent, so a successful
        // return here confirms the early-exit path was taken.
        let result = cache.prefetch(3);
        assert!(
            result.is_ok(),
            "prefetch of already-resident layer should succeed without I/O"
        );
        // stats must remain clean (no miss recorded through prefetch)
        assert_eq!(cache.stats.misses, 0);
    }
}
