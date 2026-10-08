//! Pre-allocated scratch memory pool for backward pass CPU operations.
//!
//! Eliminates per-step Vec allocations in `backward_layer()`. All buffers
//! are allocated once as a single contiguous block and carved into named
//! regions via [`ScratchId`] handles.
//!
//! For a 20-layer model this saves ~260 malloc+free cycles per training step.

/// Handle to a named scratch region. Zero-cost index wrapper.
#[derive(Clone, Copy, Debug)]
pub struct ScratchId(u32);

/// Pre-allocated f32 scratch memory for backward pass CPU operations.
///
/// All buffers live in a single contiguous `Vec<f32>`, improving cache
/// locality compared to separate heap allocations. Buffers are fully
/// overwritten before use — no zeroing needed between steps.
pub struct BackwardScratch {
    storage: Vec<f32>,
    regions: Vec<(usize, usize)>,
}

impl BackwardScratch {
    /// Plan and allocate the pool from a list of `(name, element_count)` entries.
    pub fn build(entries: &[(&'static str, usize)]) -> Self {
        let total: usize = entries.iter().map(|(_, len)| len).sum();
        let storage = vec![0.0f32; total];
        let mut regions = Vec::with_capacity(entries.len());
        let mut offset = 0;
        for &(_, len) in entries {
            regions.push((offset, len));
            offset += len;
        }
        Self { storage, regions }
    }

    /// Look up the `ScratchId` for a named entry.
    ///
    /// Panics if `name` is not found. Use at init time, not in hot loops.
    pub fn id_of(entries: &[(&'static str, usize)], name: &str) -> ScratchId {
        let idx = entries
            .iter()
            .position(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("unknown scratch region: {name}"));
        ScratchId(idx as u32)
    }

    /// Get a mutable slice for a scratch region.
    ///
    /// Callers must fully overwrite before reading — contents are stale
    /// from the previous step.
    #[inline]
    pub fn get_mut(&mut self, id: ScratchId) -> &mut [f32] {
        let (offset, len) = self.regions[id.0 as usize];
        &mut self.storage[offset..offset + len]
    }

    /// Get an immutable slice for a scratch region.
    #[inline]
    pub fn get(&self, id: ScratchId) -> &[f32] {
        let (offset, len) = self.regions[id.0 as usize];
        &self.storage[offset..offset + len]
    }

    /// Total allocated bytes.
    pub fn size_bytes(&self) -> usize {
        self.storage.len() * std::mem::size_of::<f32>()
    }
}

/// Scratch region IDs for `backward_layer()`.
///
/// Built once at `compile_kernels()` time, stored on the trainer.
/// Each field maps to a pre-allocated region in [`BackwardScratch`].
#[allow(missing_docs)]
pub struct BackwardScratchIds {
    pub dsilu_raw: ScratchId,
    pub dh1: ScratchId,
    pub dh3: ScratchId,
    pub dx_ffn: ScratchId,
    pub dx_ffn_norm: ScratchId,
    pub dv: ScratchId,
    pub dq: ScratchId,
    pub dk: ScratchId,
    pub dxq: ScratchId,
    pub dxk: ScratchId,
    pub dxv: ScratchId,
    pub dx_attn: ScratchId,
    pub dx_attn_norm: ScratchId,
}

/// Build the backward scratch entries list for a given model geometry.
pub fn backward_scratch_entries(
    dim: usize,
    hidden: usize,
    seq: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Vec<(&'static str, usize)> {
    debug_assert!(seq > 0 && dim > 0);
    vec![
        ("dsilu_raw", hidden * seq),
        ("dh1", hidden * seq),
        ("dh3", hidden * seq),
        ("dx_ffn", dim * seq),
        ("dx_ffn_norm", dim * seq),
        ("dv", kv_dim * seq),
        ("dq", q_dim * seq),
        ("dk", kv_dim * seq),
        ("dxq", dim * seq),
        ("dxk", dim * seq),
        ("dxv", dim * seq),
        ("dx_attn", dim * seq),
        ("dx_attn_norm", dim * seq),
    ]
}

/// Build `BackwardScratchIds` from entries.
pub fn backward_scratch_ids(entries: &[(&'static str, usize)]) -> BackwardScratchIds {
    BackwardScratchIds {
        dsilu_raw: BackwardScratch::id_of(entries, "dsilu_raw"),
        dh1: BackwardScratch::id_of(entries, "dh1"),
        dh3: BackwardScratch::id_of(entries, "dh3"),
        dx_ffn: BackwardScratch::id_of(entries, "dx_ffn"),
        dx_ffn_norm: BackwardScratch::id_of(entries, "dx_ffn_norm"),
        dv: BackwardScratch::id_of(entries, "dv"),
        dq: BackwardScratch::id_of(entries, "dq"),
        dk: BackwardScratch::id_of(entries, "dk"),
        dxq: BackwardScratch::id_of(entries, "dxq"),
        dxk: BackwardScratch::id_of(entries, "dxk"),
        dxv: BackwardScratch::id_of(entries, "dxv"),
        dx_attn: BackwardScratch::id_of(entries, "dx_attn"),
        dx_attn_norm: BackwardScratch::id_of(entries, "dx_attn_norm"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_pool_basic() {
        let entries: &[(&str, usize)] = &[("a", 100), ("b", 200), ("c", 50)];
        let mut pool = BackwardScratch::build(entries);

        let a = BackwardScratch::id_of(entries, "a");
        let b = BackwardScratch::id_of(entries, "b");
        let c = BackwardScratch::id_of(entries, "c");

        assert_eq!(pool.get_mut(a).len(), 100);
        assert_eq!(pool.get_mut(b).len(), 200);
        assert_eq!(pool.get_mut(c).len(), 50);
        assert_eq!(pool.size_bytes(), (100 + 200 + 50) * 4);
    }

    #[test]
    fn scratch_pool_contiguous() {
        let entries: &[(&str, usize)] = &[("x", 10), ("y", 20)];
        let pool = BackwardScratch::build(entries);

        // Single contiguous allocation
        assert_eq!(pool.size_bytes() / std::mem::size_of::<f32>(), 30);
    }
}
