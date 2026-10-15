//! SSD-offloaded routed experts of a sparse layer.
//!
//! A layer's experts live in a packed file per layer (see
//! [`crate::expert_layout`]), 4- or 2-bit affine groups in MLX's layout. A
//! forward pass routes first, then reads exactly the experts its tokens chose
//! into one page-aligned region (a *slab*), and runs them on MLX's quantized
//! gather kernels over views of that region: the same `switch_glu` the resident
//! and the packed (NVFP4) experts run, so prefill gets its sorted, tiled
//! matmuls and decode its gather kernel, with no kernel of our own.
//!
//! ```text
//! route (GPU) ──► unique experts ──► pread into slab ──► gather_qmm over the slab ──► combine
//!                                    (SSD, GPU idle)      (GPU, SSD idle)
//! ```
//!
//! The order is serial on purpose. SSD DMA and GPU compute share the memory
//! controller on Apple Silicon: reading ahead while the GPU computes makes the
//! GPU's work slower by about what the read saved, and a guess at the next
//! layer's experts is wrong often enough to read what nobody uses. What is
//! read twice is served by the OS page cache, which is also why no cache of
//! the experts' own is kept (see [`crate::expert_io`]).
//!
//! The slab is memory MLX reads in place ([`Array::from_external`]): the bytes
//! `pread` leaves in it are the bytes the kernel sees, with no copy between.
//! It returns to its pool when the last array over it is gone, which is when
//! the GPU has finished with it, so a layer's slab is never overwritten while
//! the previous layer's work still reads it.

use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use pmetal_runtime::QuantizedMode;
use pmetal_runtime::compat::ops::slice_axis;
use pmetal_runtime::compat::{Array, Dtype, Exception, nn, ops};
use pmetal_runtime::native_moe::switch_glu;
use pmetal_runtime::native_weight::{LayerWeight, QuantParams};

use crate::expert_io::ExpertOffloadContext;
use crate::expert_layout::{ExpertComponent, ExpertPackLayout};

/// Apple Silicon's page size: slabs are aligned to it and a whole number of
/// pages long, which is what lets MLX wrap them without a copy.
const PAGE: usize = 16384;

/// The most expert bytes one slab holds by default ([`SLAB_BYTES_ENV_VAR`] sets it). A pass whose routing
/// reaches more experts than fit runs in token windows, each with the experts
/// its own tokens chose; one token's experts always fit.
const DEFAULT_SLAB_BYTES: usize = 1 << 30;

/// Environment variable that sets the slab size, in bytes.
pub const SLAB_BYTES_ENV_VAR: &str = "PMETAL_EXPERT_SLAB_BYTES";

/// What the expert files have been asked for since the last reset.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ExpertReadStats {
    /// Experts read (each unique expert of each window of each layer).
    pub experts: u64,
    /// Bytes read.
    pub bytes: u64,
    /// Time spent reading, microseconds (the GPU idles through it).
    pub read_us: u64,
}

impl ExpertReadStats {
    /// Read throughput in bytes per second, if anything was read.
    pub fn bytes_per_second(&self) -> Option<f64> {
        (self.read_us > 0).then(|| self.bytes as f64 / (self.read_us as f64 / 1e6))
    }
}

#[derive(Debug, Default)]
struct ReadCounters {
    experts: AtomicU64,
    bytes: AtomicU64,
    read_us: AtomicU64,
}

/// Where the time of one offloaded layer went, for the profile.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct OffloadTimings {
    /// Reading the routed experts' bytes from the files.
    pub(crate) read_us: u64,
    /// Experts read (one per unique expert per window).
    pub(crate) experts_read: usize,
}

/// Page-aligned memory, returned to its pool when dropped.
struct Slab {
    ptr: NonNull<u8>,
    capacity: usize,
    pool: Arc<SlabPool>,
}

// SAFETY: plain heap memory; the pool and every array over it only read it
// once it has been filled, and it is not refilled until all of them are gone.
#[allow(unsafe_code)]
unsafe impl Send for Slab {}
#[allow(unsafe_code)]
unsafe impl Sync for Slab {}

impl Drop for Slab {
    fn drop(&mut self) {
        self.pool.give_back(RawSlab {
            ptr: self.ptr,
            capacity: self.capacity,
        });
    }
}

/// An allocation waiting in the pool.
struct RawSlab {
    ptr: NonNull<u8>,
    capacity: usize,
}

// SAFETY: as for `Slab`.
#[allow(unsafe_code)]
unsafe impl Send for RawSlab {}

impl RawSlab {
    /// Return the allocation to the system.
    fn free(self) {
        let layout = Layout::from_size_align(self.capacity, PAGE).expect("a valid slab layout");
        // SAFETY: allocated in `SlabPool::acquire` with this layout, and
        // consumed here.
        #[allow(unsafe_code)]
        unsafe {
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

/// Slabs not in use. A layer's slab goes back once the GPU is done with it, so
/// a few slabs cycle: one being read into, one or two still being computed on.
#[derive(Default)]
struct SlabPool {
    free: Mutex<Vec<RawSlab>>,
}

impl Drop for SlabPool {
    fn drop(&mut self) {
        let free = self.free.get_mut().unwrap_or_else(|e| e.into_inner());
        for raw in free.drain(..) {
            raw.free();
        }
    }
}

impl SlabPool {
    fn give_back(&self, slab: RawSlab) {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        free.push(slab);
        // A model whose layers differ in how many experts they route to would
        // otherwise keep every size it ever needed.
        const KEEP: usize = 4;
        if free.len() > KEEP {
            free.sort_by_key(|s| std::cmp::Reverse(s.capacity));
            for raw in free.split_off(KEEP) {
                raw.free();
            }
        }
    }

    /// A slab of at least `bytes`: the smallest free one that fits, else a new
    /// allocation.
    fn acquire(self: &Arc<Self>, bytes: usize) -> Result<Slab, Exception> {
        let capacity = bytes.max(1).div_ceil(PAGE) * PAGE;
        let reused = {
            let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
            let best = free
                .iter()
                .enumerate()
                .filter(|(_, s)| s.capacity >= capacity)
                .min_by_key(|(_, s)| s.capacity)
                .map(|(i, _)| i);
            best.map(|i| free.swap_remove(i))
        };
        let raw = match reused {
            Some(raw) => raw,
            None => {
                let layout = Layout::from_size_align(capacity, PAGE)
                    .map_err(|e| Exception::custom(format!("expert slab layout: {e}")))?;
                // SAFETY: non-zero size.
                #[allow(unsafe_code)]
                let ptr = unsafe { alloc(layout) };
                let ptr = NonNull::new(ptr).ok_or_else(|| {
                    Exception::custom(format!("could not allocate a {capacity} byte expert slab"))
                })?;
                RawSlab { ptr, capacity }
            }
        };
        Ok(Slab {
            ptr: raw.ptr,
            capacity: raw.capacity,
            pool: self.clone(),
        })
    }
}

/// The routed experts of every sparse layer, read from their packed files as
/// each layer routes.
#[derive(Debug)]
pub struct OffloadedExperts {
    ctx: ExpertOffloadContext,
    slabs: Arc<SlabPool>,
    /// Most expert bytes one slab holds; see [`DEFAULT_SLAB_BYTES`].
    slab_bytes: usize,
    params: QuantParams,
    counters: ReadCounters,
}

impl std::fmt::Debug for SlabPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlabPool").finish_non_exhaustive()
    }
}

impl OffloadedExperts {
    /// Open the packed experts in `dir`.
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        let ctx = ExpertOffloadContext::new(dir)?;
        let layout = &ctx.layout;
        let params = QuantParams {
            group_size: layout.group_size as i32,
            bits: layout.bits.bit_width() as i32,
            mode: QuantizedMode::Affine,
        };
        // The record's component sizes are what the views below index by.
        let (hidden, inter) = (layout.hidden_dim, layout.intermediate_dim);
        let pack = layout.bits.pack_factor();
        if hidden % pack != 0 || inter % pack != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "packed experts: hidden {hidden} and intermediate {inter} must be multiples \
                     of the {pack} values a word holds"
                ),
            ));
        }
        let slab_bytes = std::env::var(SLAB_BYTES_ENV_VAR)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SLAB_BYTES);
        Ok(Self {
            ctx,
            slabs: Arc::new(SlabPool::default()),
            slab_bytes,
            params,
            counters: ReadCounters::default(),
        })
    }

    /// What has been read since the last [`reset_stats`](Self::reset_stats).
    pub fn stats(&self) -> ExpertReadStats {
        ExpertReadStats {
            experts: self.counters.experts.load(Ordering::Relaxed),
            bytes: self.counters.bytes.load(Ordering::Relaxed),
            read_us: self.counters.read_us.load(Ordering::Relaxed),
        }
    }

    /// Zero the read counters.
    pub fn reset_stats(&self) {
        self.counters.experts.store(0, Ordering::Relaxed);
        self.counters.bytes.store(0, Ordering::Relaxed);
        self.counters.read_us.store(0, Ordering::Relaxed);
    }

    /// The layout of the packed files.
    pub fn layout(&self) -> &ExpertPackLayout {
        &self.ctx.layout
    }

    /// `Σ_k weights[s, k] · down(silu(gate(x_s)) · up(x_s))` over each token's
    /// routed experts of layer `layer_idx`: `x_flat` is `[S, hidden]`, `inds`
    /// and `weights` are `[S, top_k]`. The result is `[S, hidden]`, lazy.
    pub(crate) fn routed(
        &self,
        layer_idx: usize,
        x_flat: &Array,
        inds: &Array,
        weights: &Array,
    ) -> Result<(Array, OffloadTimings), Exception> {
        let (tokens, top_k) = (x_flat.dim(0) as usize, inds.dim(1) as usize);
        let inds_u32 = inds.as_dtype(Dtype::Uint32.as_i32());
        inds_u32.eval();
        let chosen: Vec<u32> = inds_u32.as_slice::<u32>().to_vec();
        if chosen.len() != tokens * top_k {
            return Err(Exception::custom(format!(
                "routing has {} entries for {tokens} tokens of {top_k} experts",
                chosen.len()
            )));
        }

        let expert_size = self.ctx.layout.expert_size;
        let max_slots = (self.slab_bytes / expert_size).max(top_k);
        let mut timings = OffloadTimings::default();
        let mut outputs = Vec::new();
        for (start, end) in token_windows(&chosen, top_k, max_slots) {
            let window = &chosen[start * top_k..end * top_k];
            let (experts, slot_of) = unique_in_order(window);

            let slab = Arc::new(self.slabs.acquire(experts.len() * expert_size)?);
            let read_start = Instant::now();
            let experts_usize: Vec<usize> = experts.iter().map(|&e| e as usize).collect();
            // SAFETY: the slab holds `experts.len() * expert_size` bytes, and
            // nothing else has it until the arrays below are built.
            #[allow(unsafe_code)]
            unsafe {
                self.ctx
                    .read_experts_into(layer_idx, &experts_usize, slab.ptr.as_ptr())
            }
            .map_err(|e| Exception::custom(format!("expert pread layer {layer_idx}: {e}")))?;
            let read_us = read_start.elapsed().as_micros() as u64;
            timings.read_us += read_us;
            timings.experts_read += experts.len();
            self.counters
                .experts
                .fetch_add(experts.len() as u64, Ordering::Relaxed);
            self.counters
                .bytes
                .fetch_add((experts.len() * expert_size) as u64, Ordering::Relaxed);
            self.counters.read_us.fetch_add(read_us, Ordering::Relaxed);

            let (gate, up, down) = self.views(&slab, experts.len());
            let slots: Vec<u32> = window.iter().map(|e| slot_of[e]).collect();
            let slots = Array::from_slice(&slots, &[(end - start) as i32, top_k as i32]);
            let (x_w, w_w) = if end - start == tokens {
                (x_flat.clone(), weights.clone())
            } else {
                (
                    slice_axis(x_flat, 0, start as i32, end as i32),
                    slice_axis(weights, 0, start as i32, end as i32),
                )
            };
            outputs.push(switch_glu(&x_w, &gate, &up, &down, &slots, &w_w, |g, u| {
                nn::silu(g).multiply(u)
            }));
        }
        let routed = match outputs.len() {
            1 => outputs.remove(0),
            _ => ops::concatenate_axis(&outputs.iter().collect::<Vec<_>>(), 0),
        };
        Ok((routed, timings))
    }

    /// The gate, up and down projections of the `slots` experts in `slab`, as
    /// stacked quantized weights `[slots, out, in / pack]`: views of the slab's
    /// bytes, no copy.
    fn views(&self, slab: &Arc<Slab>, slots: usize) -> (LayerWeight, LayerWeight, LayerWeight) {
        let layout = &self.ctx.layout;
        let record = &layout.record;
        let stride = layout.expert_size;
        let (hidden, inter) = (layout.hidden_dim as i32, layout.intermediate_dim as i32);
        let pack = layout.bits.pack_factor() as i32;
        let group = layout.group_size as i32;
        let s = slots as i32;

        // One owner, two typed views of the same bytes: the weights are words,
        // the scales and biases bf16.
        let words = external(slab, Dtype::Uint32, 4);
        let halves = external(slab, Dtype::Bfloat16, 2);
        // `component` of every slot's record, `[slots, rows, cols]`.
        let part = |source: &Array, unit: usize, c: &ExpertComponent, rows: i32, cols: i32| {
            let per_record = (stride / unit) as i32;
            let records = source
                .slice(&[0], &[s * per_record])
                .reshape(&[s, per_record]);
            let from = (c.offset / unit) as i32;
            let to = ((c.offset + c.size) / unit) as i32;
            slice_axis(&records, 1, from, to).reshape(&[s, rows, cols])
        };
        let projection = |w: &ExpertComponent,
                          sc: &ExpertComponent,
                          b: &ExpertComponent,
                          rows: i32,
                          cols: i32| LayerWeight::Quantized {
            weight: part(&words, 4, w, rows, cols / pack),
            scales: part(&halves, 2, sc, rows, cols / group),
            biases: Some(part(&halves, 2, b, rows, cols / group)),
            tensor_scale: None,
            params: self.params,
        };
        (
            projection(
                &record.gate_weight,
                &record.gate_scales,
                &record.gate_biases,
                inter,
                hidden,
            ),
            projection(
                &record.up_weight,
                &record.up_scales,
                &record.up_biases,
                inter,
                hidden,
            ),
            projection(
                &record.down_weight,
                &record.down_scales,
                &record.down_biases,
                hidden,
                inter,
            ),
        )
    }
}

/// A 1-D array of `dtype` over the whole of `slab`, which the array keeps alive.
fn external(slab: &Arc<Slab>, dtype: Dtype, itemsize: usize) -> Array {
    // SAFETY: the slab is `capacity` readable bytes for as long as the array's
    // owner (a clone of the `Arc`) lives, and it is not written again until the
    // pool hands it out, which is after every array over it is gone.
    #[allow(unsafe_code)]
    unsafe {
        Array::from_external(
            slab.ptr.as_ptr(),
            &[(slab.capacity / itemsize) as i32],
            dtype,
            slab.clone(),
        )
    }
}

/// The experts in `chosen`, each once, in order of first use, and the slot each
/// takes.
fn unique_in_order(chosen: &[u32]) -> (Vec<u32>, HashMap<u32, u32>) {
    let mut experts = Vec::new();
    let mut slot_of = HashMap::new();
    for &expert in chosen {
        slot_of.entry(expert).or_insert_with(|| {
            experts.push(expert);
            (experts.len() - 1) as u32
        });
    }
    (experts, slot_of)
}

/// Cut the tokens into runs, each reaching at most `max_slots` distinct
/// experts, so each run's experts fit one slab. `chosen` is `[tokens, top_k]`;
/// a token's own experts are at most `top_k`, which `max_slots` covers.
fn token_windows(chosen: &[u32], top_k: usize, max_slots: usize) -> Vec<(usize, usize)> {
    let tokens = chosen.len() / top_k.max(1);
    let mut windows = Vec::new();
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut start = 0;
    for token in 0..tokens {
        let row = &chosen[token * top_k..(token + 1) * top_k];
        let new: std::collections::HashSet<u32> =
            row.iter().copied().filter(|e| !seen.contains(e)).collect();
        if token > start && seen.len() + new.len() > max_slots {
            windows.push((start, token));
            start = token;
            seen.clear();
        }
        seen.extend(row.iter().copied());
    }
    if tokens > start || windows.is_empty() {
        windows.push((start, tokens));
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experts_are_numbered_in_order_of_first_use() {
        let (experts, slot_of) = unique_in_order(&[7, 3, 7, 5, 3]);
        assert_eq!(experts, vec![7, 3, 5]);
        assert_eq!(slot_of[&7], 0);
        assert_eq!(slot_of[&3], 1);
        assert_eq!(slot_of[&5], 2);
    }

    #[test]
    fn windows_cover_every_token_once_and_respect_the_slab() {
        // Six tokens, two experts each, all different: 12 experts.
        let chosen: Vec<u32> = (0..12).collect();
        for max_slots in [2, 4, 5, 6, 12, 100] {
            let windows = token_windows(&chosen, 2, max_slots);
            let mut next = 0;
            for &(start, end) in &windows {
                assert_eq!(start, next, "windows are contiguous");
                assert!(end > start);
                let (experts, _) = unique_in_order(&chosen[start * 2..end * 2]);
                assert!(
                    experts.len() <= max_slots.max(2),
                    "{max_slots}: window {start}..{end} reaches {} experts",
                    experts.len()
                );
                next = end;
            }
            assert_eq!(next, 6, "every token is in a window");
        }
    }

    #[test]
    fn tokens_that_share_experts_share_a_window() {
        // Every token routes to the same two experts.
        let chosen = [4u32, 9].repeat(50);
        assert_eq!(token_windows(&chosen, 2, 2), vec![(0, 50)]);
    }

    #[test]
    fn the_pool_reuses_a_slab_once_it_is_given_back() {
        let pool = Arc::new(SlabPool::default());
        let first = pool.acquire(100).unwrap();
        let address = first.ptr;
        let capacity = first.capacity;
        assert_eq!(capacity % PAGE, 0);
        assert_eq!(address.as_ptr() as usize % PAGE, 0, "page aligned");
        drop(first);
        let again = pool.acquire(capacity).unwrap();
        assert_eq!(again.ptr, address, "the freed slab is the one reused");
        // Still in use, so a second request allocates.
        let other = pool.acquire(10).unwrap();
        assert_ne!(other.ptr, again.ptr);
    }
}
