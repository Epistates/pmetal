//! Cross-request KV prefix cache.
//!
//! When a chat client sends a multi-turn conversation, the incoming
//! prompt is almost always `previous_prompt + new_turn` — the leading
//! tokens repeat verbatim from the previous request. Re-running the
//! full prompt through the model discards GPU work that we already
//! paid for.
//!
//! This cache stores a bounded collection of `(tokens, KV-cache fork)`
//! pairs. On each new request, we scan for the cached entry whose
//! tokens are the longest prefix of the incoming sequence. If we
//! find one, we fork the cached KV state and prefill only the suffix.
//!
//! # Matching rule
//!
//! A cached entry `C` matches an incoming sequence `S` iff `C` is a
//! strict prefix of `S` — that is, `S[..C.len()] == C` **and**
//! `C.len() < S.len()`. The strict inequality is required: if the
//! cached sequence equals the incoming one, there are no suffix tokens
//! to prefill, and the engine's decode loop expects at least one
//! forward pass to produce the first-token logits.
//!
//! # Hybrid models
//!
//! Models with recurrent state (Mamba, GDN, RecurrentGemma, …) are
//! NOT supported here. Their recurrent state is separate from the KV cache
//! and can't be forked to a prefix cleanly. Returning wrong answers silently
//! is worse than the small prefill savings. The engine gates on
//! `mamba_cache.is_none()` before consulting this cache.
//!
//! # Eviction
//!
//! Entries are evicted LRU once either the entry count or the total
//! stored-byte budget is exceeded. KV entries can be multi-hundred-MB
//! for long contexts; the byte budget is the practical limit.

use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use std::time::Instant;

/// One cached prefix entry.
struct Entry {
    tokens: Vec<u32>,
    cache: KVCache,
    /// Monotonic counter; higher = more recently touched.
    last_used: u64,
    /// Memory estimate cached once at insert time.
    bytes: usize,
}

/// Outcome of a prefix-cache lookup.
pub struct PrefixHit {
    /// Number of leading tokens covered by the cached KV fork.
    pub prefix_len: usize,
    /// The restored KV cache, ready to continue with suffix tokens.
    pub restored_cache: KVCache,
}

/// Bounded cross-request KV prefix cache.
///
/// Thread-safety: not internally synchronised. The engine wraps a single
/// instance in `Arc<Mutex<_>>`.
pub struct ServePrefixCache {
    entries: Vec<Entry>,
    max_entries: usize,
    max_bytes: usize,
    current_bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl ServePrefixCache {
    /// Create a new prefix cache.
    ///
    /// The entry bound is always enforced (`max_entries = 0` disables the
    /// cache). The byte bound is enforced when `max_bytes > 0`.
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            max_bytes,
            current_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    /// Find the cached entry whose tokens are the longest strict prefix
    /// of `incoming` and fork it into a fresh KV cache.
    ///
    /// Returns `None` if no cached entry is a prefix of `incoming`, or
    /// if the best match would cover every incoming token (we require
    /// at least one suffix token so the decode path has something to
    /// prefill; otherwise it has no last-position logits to sample
    /// from).
    pub fn find_longest_prefix(
        &mut self,
        incoming: &[u32],
        kv_config: KVCacheConfig,
    ) -> Result<Option<PrefixHit>, pmetal_runtime::compat::Exception> {
        // Scan everything — the cache is small (typically <=32 entries)
        // so O(n * prefix_len) is fine.
        let mut best: Option<(usize, usize)> = None; // (index, prefix_len)
        for (i, entry) in self.entries.iter().enumerate() {
            let len = entry.tokens.len();
            // Strict prefix: cached must be shorter than incoming.
            if len >= incoming.len() {
                continue;
            }
            if incoming[..len] != entry.tokens[..] {
                continue;
            }
            match best {
                Some((_, best_len)) if best_len >= len => {}
                _ => best = Some((i, len)),
            }
        }

        match best {
            None => {
                self.misses = self.misses.saturating_add(1);
                Ok(None)
            }
            Some((idx, prefix_len)) => {
                self.hits = self.hits.saturating_add(1);
                let now = self.tick();
                self.entries[idx].last_used = now;
                let restored = self.entries[idx].cache.fork();
                if restored.config() != &kv_config {
                    return Err(pmetal_runtime::compat::Exception::custom(
                        "prefix-cache hit had a mismatched KVCacheConfig",
                    ));
                }
                Ok(Some(PrefixHit {
                    prefix_len,
                    restored_cache: restored,
                }))
            }
        }
    }

    /// Insert a new prefix into the cache.
    ///
    /// Forks the given KV cache; the caller must be at a clean
    /// end-of-prefill boundary (i.e., `cache.seq_len() == tokens.len()`).
    /// Mismatched lengths are skipped because restoring a prefix with a
    /// different RoPE/cache offset would corrupt the next suffix prefill.
    ///
    /// If an entry with the identical token sequence already exists,
    /// it is replaced (refreshed in LRU).
    pub fn insert(&mut self, tokens: &[u32], cache: &KVCache) {
        if tokens.is_empty() {
            return;
        }

        if cache.seq_len() != tokens.len() {
            tracing::debug!(
                target = "pmetal_serve::prefix_cache",
                cache_len = cache.seq_len(),
                token_len = tokens.len(),
                "skipping prompt-cache save: cache length does not match token prefix"
            );
            return;
        }

        // Replace existing entry with identical key.
        if let Some(pos) = self
            .entries
            .iter()
            .position(|e| e.tokens.as_slice() == tokens)
        {
            let old_bytes = self.entries[pos].bytes;
            self.current_bytes = self.current_bytes.saturating_sub(old_bytes);
            self.entries.swap_remove(pos);
        }

        let cached = cache.fork();
        let bytes = cached.memory_usage();
        let now = self.tick();
        let entry = Entry {
            tokens: tokens.to_vec(),
            cache: cached,
            last_used: now,
            bytes,
        };

        self.current_bytes = self.current_bytes.saturating_add(bytes);
        self.entries.push(entry);

        self.evict_to_budget();
    }

    fn evict_to_budget(&mut self) {
        // LRU eviction until both budgets satisfied. Finds the oldest
        // entry each pass (O(n²) worst case, but n is small).
        while self.entries.len() > self.max_entries
            || (self.max_bytes > 0 && self.current_bytes > self.max_bytes)
        {
            if self.entries.is_empty() {
                break;
            }
            let mut oldest_idx = 0;
            let mut oldest = self.entries[0].last_used;
            for (i, e) in self.entries.iter().enumerate().skip(1) {
                if e.last_used < oldest {
                    oldest = e.last_used;
                    oldest_idx = i;
                }
            }
            let removed = self.entries.swap_remove(oldest_idx);
            self.current_bytes = self.current_bytes.saturating_sub(removed.bytes);
        }
    }

    /// Clear every cached entry (e.g., on model swap).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_bytes = 0;
    }

    /// Number of cached prefixes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff no prefixes are cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total estimated bytes held by cached snapshots.
    pub fn bytes(&self) -> usize {
        self.current_bytes
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate in `[0.0, 1.0]`. Returns `0.0` if no lookups have
    /// occurred yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Guard-style helper: time a prefix-cache restore for logging.
pub fn time_restore<F, T>(label: &str, f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let out = f();
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    tracing::debug!(
        target = "pmetal_serve::prefix_cache",
        "{label} took {ms:.2}ms"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache_with_len(seq_len: usize) -> (KVCache, KVCacheConfig) {
        use pmetal_runtime::compat::Array;
        let cfg = KVCacheConfig::new(2, 128, 4, 64);
        let mut cache = KVCache::new(cfg.clone());
        if seq_len > 0 {
            let k = Array::zeros_f32(&[1, 4, seq_len as i32, 64]);
            let v = Array::zeros_f32(&[1, 4, seq_len as i32, 64]);
            cache.update_and_fetch(0, &k, &v).unwrap();
            cache.update_and_fetch(1, &k, &v).unwrap();
        }
        (cache, cfg)
    }

    fn make_cache() -> (KVCache, KVCacheConfig) {
        make_cache_with_len(8)
    }

    #[test]
    fn empty_cache_misses() {
        let mut pc = ServePrefixCache::new(8, 0);
        let cfg = KVCacheConfig::new(2, 128, 4, 64);
        let hit = pc.find_longest_prefix(&[1, 2, 3], cfg).unwrap();
        assert!(hit.is_none());
        assert_eq!(pc.misses(), 1);
        assert_eq!(pc.hits(), 0);
    }

    #[test]
    fn strict_prefix_match_hits() {
        let mut pc = ServePrefixCache::new(8, 0);
        let (cache, cfg) = make_cache();
        pc.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &cache);

        // Incoming is longer than cached — strict prefix match.
        let hit = pc
            .find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cfg.clone())
            .unwrap()
            .expect("expected hit");
        assert_eq!(hit.prefix_len, 8);
        assert_eq!(pc.hits(), 1);
    }

    #[test]
    fn exact_length_match_is_rejected_for_decode_safety() {
        // A cached entry whose length equals the incoming sequence
        // cannot be used — decode needs at least one suffix token to
        // produce first-sample logits.
        let mut pc = ServePrefixCache::new(8, 0);
        let (cache, cfg) = make_cache_with_len(3);
        pc.insert(&[1, 2, 3], &cache);

        let hit = pc.find_longest_prefix(&[1, 2, 3], cfg).unwrap();
        assert!(hit.is_none());
    }

    #[test]
    fn longest_strict_prefix_is_picked() {
        let mut pc = ServePrefixCache::new(8, 0);
        let cfg = KVCacheConfig::new(2, 128, 4, 64);
        pc.insert(&[1, 2], &make_cache_with_len(2).0);
        pc.insert(&[1, 2, 3, 4], &make_cache_with_len(4).0);
        pc.insert(&[1, 2, 3], &make_cache_with_len(3).0);

        let hit = pc
            .find_longest_prefix(&[1, 2, 3, 4, 5, 6], cfg)
            .unwrap()
            .expect("expected hit");
        assert_eq!(hit.prefix_len, 4);
    }

    #[test]
    fn non_prefix_misses() {
        let mut pc = ServePrefixCache::new(8, 0);
        let (cache, cfg) = make_cache_with_len(3);
        pc.insert(&[1, 2, 3], &cache);
        // Shares no leading tokens.
        let hit = pc.find_longest_prefix(&[9, 8, 7, 6], cfg).unwrap();
        assert!(hit.is_none());
    }

    #[test]
    fn lru_eviction_drops_oldest() {
        let mut pc = ServePrefixCache::new(2, 0);
        let (cache, cfg) = make_cache_with_len(1);
        pc.insert(&[1], &cache);
        pc.insert(&[2], &cache);
        pc.insert(&[3], &cache);
        assert_eq!(pc.len(), 2);

        // Oldest ([1]) should be gone.
        let hit = pc.find_longest_prefix(&[1, 99], cfg.clone()).unwrap();
        assert!(hit.is_none());
        let hit = pc.find_longest_prefix(&[3, 99], cfg).unwrap();
        assert!(hit.is_some());
    }

    #[test]
    fn reinsert_refreshes_lru() {
        let mut pc = ServePrefixCache::new(2, 0);
        let (cache, cfg) = make_cache_with_len(1);
        pc.insert(&[1], &cache);
        pc.insert(&[2], &cache);
        // Touch [1] by re-inserting — now [2] is the oldest.
        pc.insert(&[1], &cache);
        pc.insert(&[3], &cache);

        let hit = pc.find_longest_prefix(&[1, 9], cfg.clone()).unwrap();
        assert!(hit.is_some());
        let hit = pc.find_longest_prefix(&[2, 9], cfg).unwrap();
        assert!(hit.is_none(), "[2] should have been evicted");
    }

    #[test]
    fn hit_rate_tracks_lookups() {
        let mut pc = ServePrefixCache::new(8, 0);
        let (cache, cfg) = make_cache_with_len(3);
        pc.insert(&[1, 2, 3], &cache);

        let _ = pc.find_longest_prefix(&[1, 2, 3, 4], cfg.clone()); // hit
        let _ = pc.find_longest_prefix(&[9, 9, 9], cfg.clone()); // miss
        let _ = pc.find_longest_prefix(&[1, 2, 3, 5, 6], cfg); // hit

        assert_eq!(pc.hits(), 2);
        assert_eq!(pc.misses(), 1);
        assert!((pc.hit_rate() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn empty_tokens_is_ignored() {
        let mut pc = ServePrefixCache::new(8, 0);
        let (cache, _) = make_cache();
        pc.insert(&[], &cache);
        assert!(pc.is_empty());
    }

    #[test]
    fn mismatched_cache_length_is_skipped() {
        let mut pc = ServePrefixCache::new(8, 0);
        let (cache, _) = make_cache_with_len(8);
        pc.insert(&[1, 2, 3], &cache);
        assert!(pc.is_empty());
    }

    #[test]
    fn restored_fork_can_diverge_without_mutating_cached_prefix() {
        use pmetal_runtime::compat::Array;

        let mut pc = ServePrefixCache::new(8, 0);
        let (cache, cfg) = make_cache_with_len(2);
        pc.insert(&[1, 2], &cache);

        let mut hit = pc
            .find_longest_prefix(&[1, 2, 3], cfg.clone())
            .unwrap()
            .expect("expected prefix hit");
        let k = Array::zeros_f32(&[1, 4, 1, 64]);
        let v = Array::zeros_f32(&[1, 4, 1, 64]);
        hit.restored_cache.update_and_fetch(0, &k, &v).unwrap();
        hit.restored_cache.update_and_fetch(1, &k, &v).unwrap();
        assert_eq!(hit.restored_cache.seq_len(), 3);

        let hit_again = pc
            .find_longest_prefix(&[1, 2, 4], cfg)
            .unwrap()
            .expect("stored prefix should still be reusable");
        assert_eq!(hit_again.prefix_len, 2);
        assert_eq!(hit_again.restored_cache.seq_len(), 2);
    }

    #[test]
    fn turboquant_active_caches_are_forked_on_save() {
        // TurboQuant state lives in compressed per-layer stores. Serving
        // prefix cache must preserve that state with KVCache::fork() instead
        // of falling back to a dense snapshot.
        use pmetal_mlx::kv_cache::{CacheMode, TurboQuantConfig};
        use pmetal_runtime::compat::Array;

        let cfg = KVCacheConfig::new(2, 128, 4, 64).with_mode(CacheMode::TurboQuant {
            // Disable the hot window — we want eager compression so the test
            // can assert `turboquant_compression_active()` after a single
            // append.
            config: TurboQuantConfig::uniform(4, 3).with_recent_window(None),
        });
        let mut cache = KVCache::new(cfg.clone());
        let k = Array::zeros_f32(&[1, 4, 8, 64]);
        let v = Array::zeros_f32(&[1, 4, 8, 64]);
        cache.update_and_fetch(0, &k, &v).unwrap();
        cache.update_and_fetch(1, &k, &v).unwrap();
        assert!(cache.turboquant_compression_active());

        let mut pc = ServePrefixCache::new(8, 0);
        pc.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &cache);
        assert_eq!(pc.len(), 1);

        let hit = pc
            .find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9], cfg)
            .unwrap()
            .expect("expected TurboQuant prefix hit");
        assert_eq!(hit.prefix_len, 8);
        assert!(hit.restored_cache.turboquant_compression_active());
    }
}
