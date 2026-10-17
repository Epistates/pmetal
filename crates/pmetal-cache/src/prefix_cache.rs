//! Prefix caching: reuse the KV state of a prompt prefix that was already
//! prefilled.
//!
//! Chat clients resend the whole conversation every turn, an RL rollout
//! samples many completions of one prompt, and tool-calling loops branch from
//! a shared context. In each case the leading tokens of a request repeat
//! tokens some earlier request already ran through the model; running them
//! again discards work.
//!
//! # What is stored and what a hit returns
//!
//! The cache keeps a bounded set of `(tokens, KV cache)` pairs. The KV cache
//! is a [`KVCache::fork`]: MLX arrays are shared by handle, so a stored
//! prefix costs the arrays the original already held, and a hit costs a
//! handle per array however long the prefix is. A restored cache and the
//! stored one are independent: appending to one never changes the other.
//!
//! A request matches the entry that shares the most leading tokens with it
//! (the longest common prefix, as a radix tree over token sequences
//! gives), not only an entry that is a prefix of it: a request that
//! diverges from a cached conversation after a shared system prompt reuses
//! the system prompt. The restored cache is rolled back to the shared
//! length ([`KVCache::rollback`]), so what it holds is exactly the keys and
//! values a cold prefill of those tokens produces and the next token's
//! logits are the uncached run's.
//!
//! At least one token of the request is always left for the caller to run:
//! a hit that covered every token would leave nothing to produce the
//! logits the first sample is drawn from.
//!
//! # What can be reused
//!
//! - A dense cache can be cut at any token.
//! - A sliding or rotating cache can be cut only where its window still
//!   covers the cut; an entry that cannot be cut to the shared length is
//!   skipped, and the entry is still used whole when the request extends it.
//! - A quantized or TurboQuant cache can be cut anywhere. The history stays
//!   quantized as it was, so the logits equal a cold run's within the
//!   cache's quantization error, not bit for bit.
//! - Models with recurrent state (Mamba, GDN) hold state that is not
//!   per-token and cannot be cut: do not cache them here.
//!
//! # Eviction
//!
//! Entries are evicted least-recently-used first once either the entry
//! count or the byte budget is exceeded. Forks of one lineage that have
//! since diverged do not share storage, and each entry is counted at the
//! bytes its own arrays hold.

use pmetal_runtime::compat::Exception;

use crate::kv_cache::{KVCache, KVCacheConfig};

/// One cached prefix.
struct Entry {
    tokens: Vec<u32>,
    cache: KVCache,
    /// Monotonic counter; higher is more recently touched.
    last_used: u64,
    /// Memory held, measured once at insert.
    bytes: usize,
}

/// The result of a prefix-cache lookup.
#[derive(Debug)]
pub struct PrefixHit {
    /// Leading tokens of the request the restored cache already covers.
    pub prefix_len: usize,
    /// The restored cache, ready for the rest of the request. Its
    /// `total_tokens` equals `prefix_len`.
    pub restored_cache: KVCache,
}

/// Bounded prefix cache; see the module docs.
///
/// Not internally synchronised: share it behind a `Mutex`.
#[derive(Debug)]
pub struct PrefixCache {
    entries: Vec<Entry>,
    max_entries: usize,
    max_bytes: usize,
    current_bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field("tokens", &self.tokens.len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// Leading tokens `a` and `b` share.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl PrefixCache {
    /// A cache holding at most `max_entries` prefixes (0 disables it) and,
    /// when `max_bytes` is non-zero, at most that many bytes.
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

    /// Find the entry sharing the most leading tokens with `incoming` and
    /// restore a cache for that many of them.
    ///
    /// The shared length is capped at `incoming.len() - 1`, so the caller
    /// always has a token to run. Returns `None` when no entry shares a
    /// leading token with the request, or none can be cut to the length it
    /// shares. `kv_config` must be the configuration the entries were
    /// built with.
    pub fn find_longest_prefix(
        &mut self,
        incoming: &[u32],
        kv_config: &KVCacheConfig,
    ) -> Result<Option<PrefixHit>, Exception> {
        // Candidates by how many tokens they would serve, longest first.
        let mut candidates: Vec<(usize, usize)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, entry)| {
                let usable =
                    common_prefix(&entry.tokens, incoming).min(incoming.len().saturating_sub(1));
                (usable > 0).then_some((usable, i))
            })
            .collect();
        candidates.sort_by_key(|&(usable, _)| std::cmp::Reverse(usable));

        for (usable, idx) in candidates {
            let stored = &self.entries[idx];
            let mut restored = stored.cache.fork();
            if restored.config() != kv_config {
                return Err(Exception::custom(
                    "prefix-cache hit had a mismatched KVCacheConfig",
                ));
            }
            let cut = stored.tokens.len() - usable;
            if cut > 0 && restored.rollback(cut) != cut {
                // The cache cannot be cut that far back (a window moved
                // past it): another entry may serve.
                continue;
            }
            self.hits = self.hits.saturating_add(1);
            let now = self.tick();
            self.entries[idx].last_used = now;
            return Ok(Some(PrefixHit {
                prefix_len: usable,
                restored_cache: restored,
            }));
        }
        self.misses = self.misses.saturating_add(1);
        Ok(None)
    }

    /// Store the KV state of a prefilled prefix.
    ///
    /// Forks `cache`, which must be at a clean end-of-prefill boundary
    /// (`cache.total_tokens() == tokens.len()`): a cache that holds a
    /// different number of tokens than it was given would restore with the
    /// wrong RoPE offset and corrupt the next prefill, so it is skipped. An
    /// entry with the same tokens is replaced, which refreshes it.
    pub fn insert(&mut self, tokens: &[u32], cache: &KVCache) {
        if tokens.is_empty() || self.max_entries == 0 {
            return;
        }
        if cache.total_tokens() != tokens.len() {
            tracing::debug!(
                target = "pmetal_cache::prefix_cache",
                cache_tokens = cache.total_tokens(),
                token_len = tokens.len(),
                "skipping prefix save: the cache does not hold the tokens it was given"
            );
            return;
        }

        if let Some(pos) = self.entries.iter().position(|e| e.tokens == tokens) {
            let old = self.entries.swap_remove(pos);
            self.current_bytes = self.current_bytes.saturating_sub(old.bytes);
        }

        let cached = cache.fork();
        let bytes = cached.memory_usage();
        let last_used = self.tick();
        self.current_bytes = self.current_bytes.saturating_add(bytes);
        self.entries.push(Entry {
            tokens: tokens.to_vec(),
            cache: cached,
            last_used,
            bytes,
        });
        self.evict_to_budget();
    }

    fn evict_to_budget(&mut self) {
        while self.entries.len() > self.max_entries
            || (self.max_bytes > 0 && self.current_bytes > self.max_bytes)
        {
            let Some((oldest, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
            else {
                break;
            };
            let removed = self.entries.swap_remove(oldest);
            self.current_bytes = self.current_bytes.saturating_sub(removed.bytes);
        }
    }

    /// Drop every entry (a model swap, say).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_bytes = 0;
    }

    /// Change the limits, evicting down to them.
    pub fn set_limits(&mut self, max_entries: usize, max_bytes: usize) {
        self.max_entries = max_entries;
        self.max_bytes = max_bytes;
        self.evict_to_budget();
    }

    /// Number of cached prefixes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes the cached prefixes hold.
    pub fn bytes(&self) -> usize {
        self.current_bytes
    }

    /// The byte budget the cached prefixes may fill; 0 for none.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Lookups that found a prefix.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Lookups that found none.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hits over lookups, 0 before the first lookup.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::Array;

    fn config() -> KVCacheConfig {
        KVCacheConfig::new(2, 128, 4, 64)
    }

    fn cache_with_len(seq_len: usize) -> KVCache {
        let mut cache = KVCache::new(config());
        if seq_len > 0 {
            let k = Array::zeros_f32(&[1, 4, seq_len as i32, 64]);
            let v = Array::zeros_f32(&[1, 4, seq_len as i32, 64]);
            cache.update_and_fetch(0, &k, &v).unwrap();
            cache.update_and_fetch(1, &k, &v).unwrap();
        }
        cache
    }

    #[test]
    fn empty_cache_misses() {
        let mut pc = PrefixCache::new(8, 0);
        assert!(
            pc.find_longest_prefix(&[1, 2, 3], &config())
                .unwrap()
                .is_none()
        );
        assert_eq!((pc.hits(), pc.misses()), (0, 1));
    }

    #[test]
    fn a_request_extending_an_entry_reuses_all_of_it() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &cache_with_len(8));
        let hit = pc
            .find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], &config())
            .unwrap()
            .expect("hit");
        assert_eq!(hit.prefix_len, 8);
        assert_eq!(hit.restored_cache.total_tokens(), 8);
        assert_eq!(pc.hits(), 1);
    }

    #[test]
    fn a_request_diverging_after_a_shared_prefix_reuses_the_shared_part() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2, 3, 4, 5, 6], &cache_with_len(6));
        let hit = pc
            .find_longest_prefix(&[1, 2, 3, 9, 9], &config())
            .unwrap()
            .expect("the first three tokens are shared");
        assert_eq!(hit.prefix_len, 3);
        assert_eq!(hit.restored_cache.total_tokens(), 3);
        assert_eq!(hit.restored_cache.rope_offset_for(0), 3);
        assert_eq!(hit.restored_cache.rope_offset_for(1), 3);
    }

    #[test]
    fn a_request_equal_to_an_entry_leaves_one_token_to_run() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2, 3], &cache_with_len(3));
        let hit = pc
            .find_longest_prefix(&[1, 2, 3], &config())
            .unwrap()
            .expect("two of three tokens are reusable");
        assert_eq!(hit.prefix_len, 2);
        assert_eq!(hit.restored_cache.total_tokens(), 2);
        // A single-token request has nothing to reuse.
        assert!(pc.find_longest_prefix(&[1], &config()).unwrap().is_none());
    }

    #[test]
    fn the_entry_sharing_the_most_tokens_wins() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2], &cache_with_len(2));
        pc.insert(&[1, 2, 3, 4], &cache_with_len(4));
        pc.insert(&[1, 2, 3], &cache_with_len(3));
        let hit = pc
            .find_longest_prefix(&[1, 2, 3, 4, 5, 6], &config())
            .unwrap()
            .expect("hit");
        assert_eq!(hit.prefix_len, 4);
    }

    #[test]
    fn non_prefix_misses() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2, 3], &cache_with_len(3));
        assert!(
            pc.find_longest_prefix(&[9, 8, 7, 6], &config())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn lru_eviction_drops_the_oldest() {
        let mut pc = PrefixCache::new(2, 0);
        let cache = cache_with_len(1);
        pc.insert(&[1], &cache);
        pc.insert(&[2], &cache);
        pc.insert(&[3], &cache);
        assert_eq!(pc.len(), 2);
        assert!(
            pc.find_longest_prefix(&[1, 99], &config())
                .unwrap()
                .is_none()
        );
        assert!(
            pc.find_longest_prefix(&[3, 99], &config())
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn a_lookup_refreshes_an_entry() {
        let mut pc = PrefixCache::new(2, 0);
        let cache = cache_with_len(1);
        pc.insert(&[1], &cache);
        pc.insert(&[2], &cache);
        assert!(
            pc.find_longest_prefix(&[1, 9], &config())
                .unwrap()
                .is_some()
        );
        pc.insert(&[3], &cache);
        assert!(
            pc.find_longest_prefix(&[1, 9], &config())
                .unwrap()
                .is_some()
        );
        assert!(
            pc.find_longest_prefix(&[2, 9], &config())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn the_byte_budget_evicts() {
        let one = cache_with_len(8).memory_usage();
        let mut pc = PrefixCache::new(100, one * 2 + one / 2);
        for i in 0..4u32 {
            pc.insert(&[i; 8], &cache_with_len(8));
        }
        assert!(pc.len() <= 2, "{} entries over the byte budget", pc.len());
        assert!(pc.bytes() <= one * 2 + one / 2);
    }

    #[test]
    fn hit_rate_tracks_lookups() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2, 3], &cache_with_len(3));
        let _ = pc.find_longest_prefix(&[1, 2, 3, 4], &config());
        let _ = pc.find_longest_prefix(&[9, 9, 9], &config());
        let _ = pc.find_longest_prefix(&[1, 2, 3, 5, 6], &config());
        assert_eq!((pc.hits(), pc.misses()), (2, 1));
        assert!((pc.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn empty_tokens_and_a_disabled_cache_store_nothing() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[], &cache_with_len(3));
        assert!(pc.is_empty());
        let mut off = PrefixCache::new(0, 0);
        off.insert(&[1, 2, 3], &cache_with_len(3));
        assert!(off.is_empty());
    }

    #[test]
    fn a_cache_that_does_not_hold_its_tokens_is_skipped() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2, 3], &cache_with_len(8));
        assert!(pc.is_empty());
    }

    #[test]
    fn a_restored_fork_diverges_without_changing_the_stored_prefix() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2], &cache_with_len(2));
        let mut hit = pc
            .find_longest_prefix(&[1, 2, 3], &config())
            .unwrap()
            .expect("hit");
        let k = Array::zeros_f32(&[1, 4, 1, 64]);
        hit.restored_cache.update_and_fetch(0, &k, &k).unwrap();
        hit.restored_cache.update_and_fetch(1, &k, &k).unwrap();
        assert_eq!(hit.restored_cache.seq_len(), 3);

        let again = pc
            .find_longest_prefix(&[1, 2, 4], &config())
            .unwrap()
            .expect("the stored prefix is intact");
        assert_eq!(again.prefix_len, 2);
        assert_eq!(again.restored_cache.seq_len(), 2);
    }

    #[test]
    fn a_mismatched_config_is_an_error() {
        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2, 3], &cache_with_len(3));
        let other = KVCacheConfig::new(2, 256, 4, 64);
        assert!(pc.find_longest_prefix(&[1, 2, 3, 4], &other).is_err());
    }

    #[test]
    fn set_limits_evicts_down_to_them() {
        let mut pc = PrefixCache::new(8, 0);
        let cache = cache_with_len(1);
        for i in 0..5u32 {
            pc.insert(&[i], &cache);
        }
        pc.set_limits(2, 0);
        assert_eq!(pc.len(), 2);
    }

    #[test]
    fn turboquant_caches_are_forked_whole() {
        // The compressed history is part of the entry: a hit restores it
        // as it was, not a dense copy.
        use crate::kv_cache::{CacheMode, TurboQuantConfig};
        let cfg = KVCacheConfig::new(2, 128, 4, 64).with_mode(CacheMode::TurboQuant {
            // No recent window: the history is quantized as it arrives.
            config: TurboQuantConfig::uniform(4, 3).with_recent_window(None),
        });
        let mut cache = KVCache::new(cfg.clone());
        let k = Array::zeros_f32(&[1, 4, 8, 64]);
        cache.update_and_fetch(0, &k, &k).unwrap();
        cache.update_and_fetch(1, &k, &k).unwrap();
        assert!(cache.turboquant_compression_active());

        let mut pc = PrefixCache::new(8, 0);
        pc.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &cache);
        let hit = pc
            .find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &cfg)
            .unwrap()
            .expect("hit");
        assert_eq!(hit.prefix_len, 8);
        assert!(hit.restored_cache.turboquant_compression_active());
        // A shorter shared prefix cuts the compressed history.
        let hit = pc
            .find_longest_prefix(&[1, 2, 3, 4, 5, 0, 0], &cfg)
            .unwrap()
            .expect("hit");
        assert_eq!(hit.prefix_len, 5);
        assert_eq!(hit.restored_cache.total_tokens(), 5);
    }
}
