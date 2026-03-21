//! Rotating KV cache - MLX-LM parity implementation.

use mlx_rs::{Array, error::Exception, ops::concatenate_axis, ops::indexing::IndexOp};

/// Rotating KV cache that acts as a circular buffer.
///
/// This implementation matches MLX-LM's `RotatingKVCache` for full parity.
/// It maintains a fixed-size buffer that overwrites oldest entries when full,
/// while optionally preserving initial "keep" tokens (typically prompt tokens).
///
/// # Memory Efficiency
///
/// Unlike standard KV cache which grows unbounded, rotating cache maintains
/// constant memory usage regardless of generation length:
/// - Memory = max_size * num_heads * head_dim * 2 (K+V) * dtype_size
///
/// # Temporal Ordering
///
/// The cache internally tracks a write index (`_idx`) that wraps around.
/// When reading, entries are reordered to maintain temporal consistency
/// for attention computation.
#[derive(Debug)]
pub struct RotatingKVCache {
    /// Cached keys [batch, heads, seq, head_dim].
    keys: Option<Array>,
    /// Cached values [batch, heads, seq, head_dim].
    values: Option<Array>,
    /// Total offset (tokens seen, may exceed max_size).
    offset: usize,
    /// Current write index in circular buffer.
    _idx: usize,
    /// Maximum size of the cache.
    pub(crate) max_size: usize,
    /// Number of initial tokens to always preserve.
    pub(crate) keep: usize,
    /// Allocation step size.
    step: usize,
}

impl RotatingKVCache {
    /// Create a new rotating KV cache.
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of tokens to store
    /// * `keep` - Number of initial tokens to always preserve (default: 0)
    pub fn new(max_size: usize, keep: usize) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            _idx: 0,
            max_size,
            keep,
            step: 256,
        }
    }

    /// Get the current length of the cache (capped at max_size).
    pub fn len(&self) -> usize {
        self.offset.min(self.max_size)
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Get total tokens processed.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the RoPE offset for position encoding.
    pub fn rope_offset(&self) -> i32 {
        self.offset as i32
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self._idx = 0;
    }

    /// Trim entries from the front of the cache.
    fn trim_impl(
        &self,
        trim_size: usize,
        v: &Array,
        append: Option<&Array>,
    ) -> Result<Array, Exception> {
        if trim_size > 0 {
            // Keep initial tokens, then slice from after trim
            let kept = v.index((.., .., ..self.keep as i32, ..));
            let rest = v.index((.., .., (trim_size + self.keep) as i32.., ..));

            let parts: Vec<&Array> = if let Some(a) = append {
                vec![&kept, &rest, a]
            } else {
                vec![&kept, &rest]
            };
            concatenate_axis(&parts, 2)
        } else if let Some(a) = append {
            concatenate_axis(&[v, a], 2)
        } else {
            Ok(v.clone())
        }
    }

    /// Reorder cache into temporal order (for reading).
    fn temporal_order(&self, v: &Array) -> Array {
        let cache_len = v.dim(2) as usize;

        if self._idx == cache_len {
            // No wrap-around yet
            v.clone()
        } else if self._idx < self.offset {
            // Wrapped around: reorder [keep][idx..][keep..idx]
            let kept = v.index((.., .., ..self.keep as i32, ..));
            let after_idx = v.index((.., .., self._idx as i32.., ..));
            let before_idx = v.index((.., .., self.keep as i32..self._idx as i32, ..));

            concatenate_axis(&[&kept, &after_idx, &before_idx], 2).unwrap_or_else(|_| v.clone())
        } else {
            // Not full yet
            v.index((.., .., ..self._idx as i32, ..))
        }
    }

    /// Update with concatenation (for multi-token updates).
    fn update_concat(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array), Exception> {
        let num_steps = keys.dim(2) as usize;

        if self.keys.is_none() {
            self.keys = Some(keys.clone());
            self.values = Some(values.clone());
        } else {
            // Put in temporal order
            let ordered_k = self.temporal_order(self.keys.as_ref().unwrap());
            let ordered_v = self.temporal_order(self.values.as_ref().unwrap());
            self._idx = ordered_k.dim(2) as usize;

            // Trim to maintain max_size
            let trim_size = self._idx.saturating_sub(self.max_size - 1);
            self.keys = Some(self.trim_impl(trim_size, &ordered_k, Some(keys))?);
            self.values = Some(self.trim_impl(trim_size, &ordered_v, Some(values))?);
        }

        self.offset += num_steps;
        self._idx = self.keys.as_ref().unwrap().dim(2) as usize;

        Ok((
            self.keys.as_ref().unwrap().clone(),
            self.values.as_ref().unwrap().clone(),
        ))
    }

    /// Update in-place (for single-token updates).
    fn update_in_place(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> Result<(Array, Array), Exception> {
        let batch = keys.dim(0) as usize;
        let n_kv_heads = keys.dim(1) as usize;
        let num_steps = keys.dim(2) as usize;
        let k_head_dim = keys.dim(3) as usize;
        let v_head_dim = values.dim(3) as usize;

        // Grow cache if needed
        let needs_growth = self.keys.is_none()
            || (self.offset >= self.keys.as_ref().unwrap().dim(2) as usize
                && (self.keys.as_ref().unwrap().dim(2) as usize) < self.max_size);
        if needs_growth {
            let new_size = self.step.min(self.max_size - self.offset);
            let k_shape = [
                batch as i32,
                n_kv_heads as i32,
                new_size as i32,
                k_head_dim as i32,
            ];
            let v_shape = [
                batch as i32,
                n_kv_heads as i32,
                new_size as i32,
                v_head_dim as i32,
            ];

            let new_k = Array::zeros::<f32>(&k_shape)?;
            let new_v = Array::zeros::<f32>(&v_shape)?;

            if let Some(ref existing_k) = self.keys {
                self.keys = Some(concatenate_axis(&[existing_k, &new_k], 2)?);
                self.values = Some(concatenate_axis(
                    &[self.values.as_ref().unwrap(), &new_v],
                    2,
                )?);
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
            self._idx = self.offset;
        }

        // Trim if exceeding max_size
        let cache_len = self.keys.as_ref().unwrap().dim(2) as usize;
        if cache_len > self.max_size {
            let trim_size = cache_len - self.max_size;
            let trimmed_k = self.trim_impl(trim_size, self.keys.as_ref().unwrap(), None)?;
            let trimmed_v = self.trim_impl(trim_size, self.values.as_ref().unwrap(), None)?;
            self.keys = Some(trimmed_k);
            self.values = Some(trimmed_v);
            self._idx = self.max_size;
        }

        // Rotate if at max
        if self._idx == self.max_size {
            self._idx = self.keep;
        }

        // In-place update using scatter or slice assignment
        // For now, we reconstruct the array (MLX doesn't have true in-place slice assignment in Rust bindings)
        let k = self.keys.as_ref().unwrap();
        let v = self.values.as_ref().unwrap();

        // Build updated cache by concatenating before, new, and after
        let before_k = if self._idx > 0 {
            Some(k.index((.., .., ..self._idx as i32, ..)))
        } else {
            None
        };
        let after_k = if self._idx + num_steps < k.dim(2) as usize {
            Some(k.index((.., .., (self._idx + num_steps) as i32.., ..)))
        } else {
            None
        };

        let before_v = if self._idx > 0 {
            Some(v.index((.., .., ..self._idx as i32, ..)))
        } else {
            None
        };
        let after_v = if self._idx + num_steps < v.dim(2) as usize {
            Some(v.index((.., .., (self._idx + num_steps) as i32.., ..)))
        } else {
            None
        };

        // Assemble new cache
        let mut k_parts: Vec<&Array> = Vec::new();
        if let Some(ref bk) = before_k {
            k_parts.push(bk);
        }
        k_parts.push(keys);
        if let Some(ref ak) = after_k {
            k_parts.push(ak);
        }

        let mut v_parts: Vec<&Array> = Vec::new();
        if let Some(ref bv) = before_v {
            v_parts.push(bv);
        }
        v_parts.push(values);
        if let Some(ref av) = after_v {
            v_parts.push(av);
        }

        self.keys = Some(concatenate_axis(&k_parts, 2)?);
        self.values = Some(concatenate_axis(&v_parts, 2)?);

        self.offset += num_steps;
        self._idx += num_steps;

        // Return slice if not full yet
        if self.offset < self.max_size {
            Ok((
                self.keys
                    .as_ref()
                    .unwrap()
                    .index((.., .., ..self.offset as i32, ..)),
                self.values
                    .as_ref()
                    .unwrap()
                    .index((.., .., ..self.offset as i32, ..)),
            ))
        } else {
            Ok((
                self.keys.as_ref().unwrap().clone(),
                self.values.as_ref().unwrap().clone(),
            ))
        }
    }

    /// Update cache with new keys and values.
    ///
    /// Uses in-place update for single tokens, concatenation for multi-token.
    pub fn update_and_fetch(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> Result<(Array, Array), Exception> {
        if keys.dim(2) == 1 {
            self.update_in_place(keys, values)
        } else {
            self.update_concat(keys, values)
        }
    }

    /// Check if cache can be trimmed.
    pub fn is_trimmable(&self) -> bool {
        self.offset < self.max_size
    }

    /// Trim n tokens from the cache.
    pub fn trim(&mut self, n: usize) -> usize {
        let trimmed = n.min(self.offset);
        self.offset -= trimmed;
        self._idx = self._idx.saturating_sub(trimmed);
        trimmed
    }
}

/// Convenience function to create a rotating KV cache.
pub fn create_rotating_cache(max_size: usize, keep: usize) -> RotatingKVCache {
    RotatingKVCache::new(max_size, keep)
}
