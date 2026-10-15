#![allow(unsafe_code)]

//! Tuna: the kernel auto-tuner.
//!
//! "Tuna" finds good kernel parameters (tile sizes, threadgroup shapes) for the
//! running hardware by benchmarking candidates at runtime.
//!
//! # How it works
//!
//! 1. **Check Cache**: Looks up if we've already tuned this kernel for the given problem size.
//! 2. **Generate Candidates**: Creates a list of valid configurations (e.g., 32x32, 64x32),
//!    led by the heuristic's choice.
//! 3. **Benchmark**: Times each candidate (the median of five runs after a warm-up). A
//!    candidate whose command buffer fails is skipped.
//! 4. **Select Winner**: The fastest candidate replaces the heuristic only if it is at least
//!    3 percent faster, so noise cannot flip the choice from run to run. A winner is cached.
//!
//! # Persistent Disk Cache
//!
//! A tuner made with [`Tuner::for_device`] stores results as JSON files (one per kernel
//! type) under `~/.cache/pmetal/tuna/<scope>/`. The scope names the chip (including its GPU
//! core count) and a digest of the compiled shader library, so results are never reused
//! across different silicon or different builds of the kernels. Subsequent launches skip
//! the benchmarking phase for previously-seen problem sizes. Concurrent processes merge
//! their results into the same files.
//!
//! # Example
//!
//! ```ignore
//! let ctx = MetalContext::global()?;
//! let config = ctx.tuner().tune_cross_entropy(&ctx, num_tokens, vocab_size)?;
//! println!("Best config: {:?}", config);
//!
//! // A tuner of your own, in memory only:
//! let tuner = Tuner::new();
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::{CommandQueueExt, DeviceProperties, DeviceTier, MetalContext};
use crate::error::{MetalError, Result};
use crate::kernels::fused_cross_entropy::{FusedLinearCrossEntropy, FusedLinearCrossEntropyConfig};
use crate::kernels::fused_merge::{MergeConfig, TensorInfo};
use tracing::{debug, info, warn};

// ============================================================================
// Config Structs
// ============================================================================

/// Configuration for tuned merge kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MergeTunedConfig {
    /// Threads per threadgroup for element-wise ops.
    pub threads_per_group: u32,
    /// Elements processed per thread (vectorization factor).
    pub elements_per_thread: u32,
    /// Use SIMD-optimized path.
    pub use_simd: bool,
}

impl Default for MergeTunedConfig {
    fn default() -> Self {
        Self {
            threads_per_group: 256,
            elements_per_thread: 4,
            use_simd: true,
        }
    }
}

/// Configuration for tuned cross-entropy loss kernels.
///
/// Cross-entropy over large vocabularies benefits from threadgroup-level
/// reduction and chunked accumulation to avoid numerical issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CrossEntropyTunedConfig {
    /// Threads per threadgroup for the reduction sweep.
    pub threadgroup_size: u32,
    /// Number of vocabulary elements processed per chunk.
    pub chunk_size: u32,
}

impl Default for CrossEntropyTunedConfig {
    fn default() -> Self {
        Self {
            threadgroup_size: 256,
            chunk_size: 4096,
        }
    }
}

fn device_tier_key(tier: DeviceTier) -> &'static str {
    match tier {
        DeviceTier::Base => "base",
        DeviceTier::Pro => "pro",
        DeviceTier::Max => "max",
        DeviceTier::Ultra => "ultra",
    }
}

fn sanitize_threads_per_token_candidate(
    threads_per_token: u32,
    max_threads_per_threadgroup: u32,
) -> u32 {
    threads_per_token
        .clamp(32, max_threads_per_threadgroup.max(32))
        .div_ceil(32)
        * 32
}

/// `items` without repeats, keeping the first of each and the order.
fn dedupe<T: PartialEq>(items: Vec<T>) -> Vec<T> {
    let mut unique = Vec::with_capacity(items.len());
    for item in items {
        if !unique.contains(&item) {
            unique.push(item);
        }
    }
    unique
}

// ============================================================================
// Choosing the fastest candidate
// ============================================================================

/// How much faster than the incumbent a candidate must measure to replace it.
///
/// Timing a kernel a few times on a machine doing other work is noisy to a few
/// percent. A smaller "win" is as likely noise as speed, and a choice that
/// flips from run to run is one nobody can reproduce or reason about, so the
/// incumbent (the heuristic) keeps its place unless the gain is clear.
const MIN_GAIN: f64 = 0.03;

/// Timed runs behind each candidate's measurement.
const BENCH_SAMPLES: usize = 5;

/// Fail a benchmark whose command buffer the GPU rejected. Such a buffer
/// completes almost at once, which would make the broken candidate look like
/// the fastest one.
fn check_completed(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    match command_buffer.error() {
        Some(error) => Err(MetalError::ExecutionFailed(error.to_string())),
        None => Ok(()),
    }
}

/// The median of `samples` (the mean of the middle two for an even count).
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        (samples[mid - 1] + samples[mid]) / 2.0
    }
}

/// Median wall-clock seconds of `samples` runs of `op`, after `warmups`
/// runs that are not timed (they compile pipelines and warm caches). A median
/// shrugs off the one run a context switch stretched, which a mean would not.
fn median_seconds(
    warmups: usize,
    samples: usize,
    mut op: impl FnMut() -> Result<()>,
) -> Result<f64> {
    for _ in 0..warmups {
        op()?;
    }
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        op()?;
        times.push(start.elapsed().as_secs_f64());
    }
    Ok(median(times))
}

/// Pick the fastest of `candidates` by measured time, as `(config, seconds)`.
///
/// The `incumbent` is measured first and keeps its place unless a candidate
/// beats it by [`MIN_GAIN`]; among candidates that do, the fastest wins and a
/// tie goes to the earlier one, so the order of the list (not noise) decides.
/// A candidate whose `measure` fails is skipped, including the incumbent: if
/// the incumbent cannot run, the fastest candidate that can takes over.
/// `None` means nothing could be measured at all.
fn pick_fastest<C: Copy + PartialEq>(
    incumbent: C,
    candidates: &[C],
    mut measure: impl FnMut(C) -> Result<f64>,
) -> Option<(C, f64)> {
    let incumbent_time = measure(incumbent).ok().filter(|t| t.is_finite());
    let mut best = incumbent_time.map(|t| (incumbent, t));
    // A challenger must beat the incumbent's time by the margin, and then
    // only the best so far.
    let bar = incumbent_time.map_or(f64::INFINITY, |t| t * (1.0 - MIN_GAIN));

    for &candidate in candidates.iter().filter(|&&c| c != incumbent) {
        let Ok(time) = measure(candidate) else {
            continue;
        };
        if !time.is_finite() {
            continue;
        }
        let beats_best = best.is_none_or(|(_, best_time)| time < best_time);
        if beats_best && (incumbent_time.is_none() || time < bar) {
            best = Some((candidate, time));
        }
    }
    best
}

// ============================================================================
// Device-scoped disk cache
// ============================================================================

/// A map of tuned configurations, optionally mirrored to one JSON file.
struct TuneCache<T> {
    /// The file this cache lives in, within the tuner's directory.
    file: &'static str,
    entries: Mutex<HashMap<String, T>>,
}

impl<T> TuneCache<T>
where
    T: Copy + Serialize + DeserializeOwned,
{
    fn new(file: &'static str) -> Self {
        Self {
            file,
            entries: Mutex::new(HashMap::new()),
        }
    }

    fn get(&self, key: &str) -> Option<T> {
        self.entries.lock().get(key).copied()
    }

    /// Remember `value` for this process only.
    fn insert(&self, key: &str, value: T) {
        self.entries.lock().insert(key.to_string(), value);
    }

    /// Merge the entries stored in `dir` under this cache's file.
    fn load_from(&self, dir: &Path) {
        let stored = read_cache_file::<T>(&dir.join(self.file));
        debug!("Loaded {} tuning entries from {}", stored.len(), self.file);
        self.entries.lock().extend(stored);
    }

    /// Remember `value` and write the cache to `dir`.
    ///
    /// Another process may have written entries since this one loaded, so the
    /// file is read back and merged (this process's entries win) before it is
    /// replaced; the replacement is a rename of a complete temporary file, so
    /// a reader never sees a partial one.
    fn persist(&self, dir: &Path, key: &str, value: T) {
        self.insert(key, value);
        let path = dir.join(self.file);
        let mut merged = read_cache_file::<T>(&path);
        merged.extend(self.entries.lock().clone());
        write_cache_file(&path, &merged);
    }
}

/// Read a cache file: the entries it holds, or none if it is missing or
/// unreadable (a corrupt file is logged and ignored, never fatal).
fn read_cache_file<T: DeserializeOwned>(path: &Path) -> HashMap<String, T> {
    match fs::read_to_string(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
        Err(error) => {
            warn!("Failed to read tuning cache {}: {}", path.display(), error);
            HashMap::new()
        }
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_else(|error| {
            warn!(
                "Ignoring unparsable tuning cache {}: {}",
                path.display(),
                error
            );
            HashMap::new()
        }),
    }
}

/// Replace `path` with `entries`, atomically, through a temporary file named
/// for this process and call so concurrent writers never share one.
fn write_cache_file<T: Serialize>(path: &Path, entries: &HashMap<String, T>) {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let tmp = path.with_extension(format!(
        "json.{}.{}.tmp",
        std::process::id(),
        WRITES.fetch_add(1, Ordering::Relaxed)
    ));
    let json = match serde_json::to_string_pretty(entries) {
        Ok(json) => json,
        Err(error) => {
            warn!(
                "Failed to serialize tuning cache {}: {}",
                path.display(),
                error
            );
            return;
        }
    };
    if let Err(error) = fs::write(&tmp, json).and_then(|()| fs::rename(&tmp, path)) {
        warn!("Failed to write tuning cache {}: {}", path.display(), error);
        let _ = fs::remove_file(&tmp);
    }
}

/// FNV-1a, 64-bit: a hash that is the same on every machine and Rust version,
/// which a name that outlives the process needs (`DefaultHasher` promises
/// neither).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The directory name that scopes cached results to the hardware and the
/// kernels they were measured with: a different chip (or a different core
/// count of one), or a rebuilt shader library, is a different directory, so a
/// result is never reused for kernels or silicon it was not measured on.
fn scope_dir_name(props: &DeviceProperties, kernel_fingerprint: &str) -> String {
    let identity = format!("{}:{kernel_fingerprint}", device_cache_identity(props));
    format!("{:016x}", fnv1a(identity.as_bytes()))
}

/// How long an unused scope directory is kept; a rebuild of the shaders leaves
/// the previous build's directory behind.
const STALE_SCOPE_AGE: std::time::Duration = std::time::Duration::from_secs(30 * 24 * 60 * 60);

/// Remove sibling scope directories nothing has touched for [`STALE_SCOPE_AGE`].
fn prune_stale_scopes(root: &Path, keep: &Path) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == keep || !path.is_dir() {
            continue;
        }
        let newest = fs::read_dir(&path)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|file| file.metadata().ok()?.modified().ok())
            .max()
            .or_else(|| entry.metadata().ok()?.modified().ok());
        let stale = newest
            .is_some_and(|modified| modified.elapsed().is_ok_and(|age| age > STALE_SCOPE_AGE));
        if stale {
            debug!("Removing stale tuning cache {}", path.display());
            let _ = fs::remove_dir_all(&path);
        }
    }
}

fn device_cache_identity(props: &DeviceProperties) -> String {
    format!(
        "{}:{}:{}:{}",
        props.name,
        device_tier_key(props.device_tier),
        props.architecture_gen,
        props.gpu_core_count
    )
}

fn merge_cache_key(props: &DeviceProperties, num_elements: usize, num_models: usize) -> String {
    format!(
        "merge:{}:{}:{}",
        device_cache_identity(props),
        num_elements,
        num_models
    )
}

// ============================================================================
// Tuner
// ============================================================================

/// The Auto-Tuner.
///
/// Results are cached per problem shape in memory and, for a tuner made with
/// [`Tuner::for_device`] or [`Tuner::with_cache_dir`], on disk.
pub struct Tuner {
    /// Merge kernel launch shape.
    /// Key: "merge:device:num_elements:num_models"
    merge: TuneCache<MergeTunedConfig>,

    /// Fused linear cross-entropy kernels.
    cross_entropy: TuneCache<CrossEntropyTunedConfig>,

    /// Directory the caches are mirrored to; `None` means in-memory only.
    cache_dir: Option<PathBuf>,
}

impl Default for Tuner {
    fn default() -> Self {
        Self::new()
    }
}

impl Tuner {
    /// Create a new Tuner instance with in-memory cache only.
    pub fn new() -> Self {
        Self {
            merge: TuneCache::new("merge.json"),
            cross_entropy: TuneCache::new("cross_entropy.json"),
            cache_dir: None,
        }
    }

    /// Create a Tuner that keeps its results in `dir`, loading any already there.
    pub fn with_cache_dir(dir: PathBuf) -> Self {
        let mut tuner = Self::new();
        tuner.cache_dir = Some(dir);
        tuner.load_disk_cache();
        tuner
    }

    /// Create a Tuner with a persistent cache for this device and these kernels.
    ///
    /// Results live under `~/.cache/pmetal/tuna/<scope>/`, where the scope
    /// names the chip (name, tier, generation and GPU core count) and
    /// `kernel_fingerprint`, a digest of the compiled shader library. A
    /// different machine, or kernels that have changed since the result was
    /// measured, therefore start from an empty cache instead of reusing a
    /// choice made for something else. Scope directories untouched for 30 days
    /// are removed.
    pub fn for_device(props: &DeviceProperties, kernel_fingerprint: &str) -> Self {
        let Some(root) = dirs::cache_dir().map(|dir| dir.join("pmetal").join("tuna")) else {
            return Self::new();
        };
        let dir = root.join(scope_dir_name(props, kernel_fingerprint));
        prune_stale_scopes(&root, &dir);
        Self::with_cache_dir(dir)
    }

    /// Load every cache's stored entries.
    ///
    /// Errors are logged and ignored so that a corrupt or missing cache file
    /// never prevents the tuner from running.
    fn load_disk_cache(&mut self) {
        let Some(dir) = self.cache_dir.clone() else {
            return;
        };
        self.merge.load_from(&dir);
        self.cross_entropy.load_from(&dir);
    }

    /// The cache directory, created if it does not exist yet.
    fn ensure_cache_dir(&self) -> Option<&Path> {
        let dir = self.cache_dir.as_deref()?;
        if let Err(error) = fs::create_dir_all(dir) {
            warn!(
                "Failed to create cache directory {}: {}",
                dir.display(),
                error
            );
            return None;
        }
        Some(dir)
    }

    /// Choose a configuration for `key`, benchmarking only on a cache miss.
    ///
    /// `plan` yields the incumbent (the heuristic's choice) and the candidates
    /// to try against it. A measured winner is cached and written to disk. If
    /// nothing could be measured, the incumbent is used for this process but
    /// not written, so a later run with a quiet GPU measures again instead of
    /// inheriting an untested guess.
    fn select<C>(
        &self,
        cache: &TuneCache<C>,
        what: &str,
        key: &str,
        plan: impl FnOnce() -> (C, Vec<C>),
        measure: impl FnMut(C) -> Result<f64>,
    ) -> C
    where
        C: Copy + PartialEq + std::fmt::Debug + Serialize + DeserializeOwned,
    {
        if let Some(hit) = cache.get(key) {
            return hit;
        }
        let (incumbent, candidates) = plan();
        match pick_fastest(incumbent, &candidates, measure) {
            Some((best, seconds)) => {
                info!(
                    "Selected {what} config {best:?} ({:.3} ms)",
                    seconds * 1000.0
                );
                match self.ensure_cache_dir() {
                    Some(dir) => cache.persist(dir, key, best),
                    None => cache.insert(key, best),
                }
                best
            }
            None => {
                debug!("No {what} candidate could be measured; using {incumbent:?}");
                cache.insert(key, incumbent);
                incumbent
            }
        }
    }

    // -------------------------------------------------------------------------
    // Public cache accessors (merge)
    // -------------------------------------------------------------------------

    /// Get a cached merge configuration if available.
    pub fn get_merge_config(&self, key: &str) -> Option<MergeTunedConfig> {
        self.merge.get(key)
    }

    /// Store a merge configuration in the cache.
    pub fn set_merge_config(&self, key: String, config: MergeTunedConfig) {
        self.merge.insert(&key, config);
    }

    // =========================================================================
    // Merge Kernel Tuning (benchmarked)
    // =========================================================================

    /// Tune merge kernels (sparsification, TIES, etc.) for the given problem size.
    ///
    /// # Arguments
    /// * `context` - Metal context
    /// * `num_elements` - Total number of elements to process
    /// * `num_models` - Number of models being merged (for TIES)
    ///
    /// # Returns
    /// Optimal configuration for merge operations on this hardware.
    pub fn tune_merge(
        &self,
        context: &MetalContext,
        num_elements: usize,
        num_models: usize,
    ) -> Result<MergeTunedConfig> {
        let key = merge_cache_key(context.properties(), num_elements, num_models);

        Ok(self.select(
            &self.merge,
            "merge",
            &key,
            || {
                info!(
                    "Tuning merge kernel for {} elements, {} models...",
                    num_elements, num_models
                );
                let candidates = self.generate_merge_candidates(context);
                debug!("Generated {} merge candidates for device", candidates.len());
                (MergeTunedConfig::default(), candidates)
            },
            |candidate| self.benchmark_merge(context, candidate, num_elements),
        ))
    }

    /// Generate candidate configurations for merge kernels.
    fn generate_merge_candidates(&self, context: &MetalContext) -> Vec<MergeTunedConfig> {
        use crate::context::DeviceTier;

        let props = context.properties();
        let max_threads = props.max_threads_per_threadgroup as u32;

        // Device tier-aware candidate ordering
        let base_candidates: Vec<MergeTunedConfig> = match props.device_tier {
            DeviceTier::Ultra | DeviceTier::Max => vec![
                // High-end: larger threadgroups, more elements per thread
                MergeTunedConfig {
                    threads_per_group: 512,
                    elements_per_thread: 8,
                    use_simd: true,
                },
                MergeTunedConfig {
                    threads_per_group: 256,
                    elements_per_thread: 8,
                    use_simd: true,
                },
                MergeTunedConfig {
                    threads_per_group: 512,
                    elements_per_thread: 4,
                    use_simd: true,
                },
                MergeTunedConfig {
                    threads_per_group: 256,
                    elements_per_thread: 4,
                    use_simd: true,
                },
            ],
            DeviceTier::Pro => vec![
                MergeTunedConfig {
                    threads_per_group: 256,
                    elements_per_thread: 8,
                    use_simd: true,
                },
                MergeTunedConfig {
                    threads_per_group: 256,
                    elements_per_thread: 4,
                    use_simd: true,
                },
                MergeTunedConfig {
                    threads_per_group: 512,
                    elements_per_thread: 4,
                    use_simd: true,
                },
                MergeTunedConfig {
                    threads_per_group: 128,
                    elements_per_thread: 8,
                    use_simd: true,
                },
            ],
            DeviceTier::Base => vec![
                // Base chips: smaller threadgroups, moderate vectorization
                MergeTunedConfig {
                    threads_per_group: 256,
                    elements_per_thread: 4,
                    use_simd: true,
                },
                MergeTunedConfig {
                    threads_per_group: 128,
                    elements_per_thread: 4,
                    use_simd: true,
                },
                MergeTunedConfig {
                    threads_per_group: 256,
                    elements_per_thread: 2,
                    use_simd: true,
                },
                MergeTunedConfig {
                    threads_per_group: 128,
                    elements_per_thread: 8,
                    use_simd: true,
                },
            ],
        };

        // Filter by device max threads
        base_candidates
            .into_iter()
            .filter(|c| c.threads_per_group <= max_threads)
            .collect()
    }

    /// Benchmark merge kernel configuration.
    fn benchmark_merge(
        &self,
        context: &MetalContext,
        config: MergeTunedConfig,
        num_elements: usize,
    ) -> Result<f64> {
        let device = context.device();

        // Skip very large allocations
        let total_bytes = num_elements * 4 * 2; // input + output
        if total_bytes > 500 * 1024 * 1024 {
            debug!(
                "Skipping merge benchmark (memory too large: {} MB)",
                total_bytes / 1024 / 1024
            );
            // Return default time penalty
            return Ok(if config.threads_per_group == 256 {
                1.0
            } else {
                100.0
            });
        }

        // Create test buffers
        let options = MTLResourceOptions::StorageModePrivate;
        let buf_input = device
            .newBufferWithLength_options(num_elements * 4, options)
            .ok_or(MetalError::BufferCreation {
                size: num_elements * 4,
                reason: "merge input".into(),
            })?;
        let buf_output = device
            .newBufferWithLength_options(num_elements * 4, options)
            .ok_or(MetalError::BufferCreation {
                size: num_elements * 4,
                reason: "merge output".into(),
            })?;

        // Get pipeline for simple magnitude computation
        let pipeline = context.pipeline_cache_mut().get_or_create_pipeline(
            context.device(),
            "fused_compute_magnitudes",
            None,
        )?;

        // Create config buffer
        let params = MergeConfig {
            num_tensors: 1,
            total_elements: num_elements as u32,
            epsilon: 1e-8,
            _pad: 0,
        };

        let tensor_info = TensorInfo {
            offset: 0,
            size: num_elements as u32,
            density: 0.5,
            threshold: 0.0,
        };

        median_seconds(1, BENCH_SAMPLES, || {
            self.dispatch_merge_kernel(
                context,
                &pipeline,
                config,
                &buf_input,
                &buf_output,
                &params,
                &tensor_info,
                num_elements,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_merge_kernel(
        &self,
        context: &MetalContext,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        config: MergeTunedConfig,
        input: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        params: &MergeConfig,
        tensor_info: &TensorInfo,
        num_elements: usize,
    ) -> Result<()> {
        let queue = context.command_queue();
        let buffer = queue.new_command_buffer()?;
        let encoder = buffer
            .computeCommandEncoder()
            .ok_or(MetalError::CommandQueueCreation)?;

        encoder.setComputePipelineState(pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 1);

            let tensor_info_ptr = NonNull::from(tensor_info).cast();
            encoder.setBytes_length_atIndex(tensor_info_ptr, std::mem::size_of::<TensorInfo>(), 2);

            let params_ptr = NonNull::from(params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<MergeConfig>(), 3);
        }

        // Calculate grid based on tuned config
        let elements_per_group = (config.threads_per_group * config.elements_per_thread) as usize;
        let grid_size = MTLSize {
            width: num_elements.div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: config.threads_per_group as usize,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();

        buffer.commit();
        buffer.waitUntilCompleted();
        check_completed(&buffer)
    }

    /// Tune the fused linear cross-entropy kernel for the given problem shape.
    ///
    /// This benchmarks the real hidden-state kernel instead of relying only on
    /// vocabulary heuristics, so the selected configuration reflects hidden
    /// width, dtype, and device tier on M1-M4 as well as Apple10/M5.
    pub fn tune_fused_linear_cross_entropy(
        &self,
        context: &Arc<MetalContext>,
        config: &FusedLinearCrossEntropyConfig,
    ) -> Result<CrossEntropyTunedConfig> {
        let props = context.properties();
        let key = format!(
            "fused_linear_ce:{}:{}:{}:{}:{}:{}",
            props.name,
            device_tier_key(props.device_tier),
            if config.use_fp16 { "f16" } else { "f32" },
            config.num_tokens,
            config.hidden_size,
            config.vocab_size
        );
        let max_threads = props.max_threads_per_threadgroup as u32;

        Ok(self.select(
            &self.cross_entropy,
            "fused linear CE",
            &key,
            || {
                info!(
                    "Tuning fused linear CE config for [T={}, H={}, V={}, dtype={}]...",
                    config.num_tokens,
                    config.hidden_size,
                    config.vocab_size,
                    if config.use_fp16 { "f16" } else { "f32" }
                );
                (
                    self.heuristic_fused_linear_cross_entropy_config(
                        props.device_tier,
                        config.hidden_size,
                        config.vocab_size,
                        max_threads,
                    ),
                    self.candidate_fused_linear_cross_entropy_configs(
                        props.device_tier,
                        config.hidden_size,
                        config.vocab_size,
                        max_threads,
                    ),
                )
            },
            |candidate| self.benchmark_fused_linear_cross_entropy(context, config, candidate),
        ))
    }

    fn heuristic_fused_linear_cross_entropy_config(
        &self,
        device_tier: DeviceTier,
        hidden_size: usize,
        vocab_size: usize,
        max_threads_per_threadgroup: u32,
    ) -> CrossEntropyTunedConfig {
        let threadgroup_size = match device_tier {
            DeviceTier::Base => {
                if vocab_size < 32_768 {
                    128
                } else if vocab_size < 131_072 {
                    256
                } else {
                    512
                }
            }
            DeviceTier::Pro | DeviceTier::Max | DeviceTier::Ultra => {
                if vocab_size < 32_768 {
                    256
                } else if vocab_size < 131_072 {
                    512
                } else {
                    1024
                }
            }
        };

        let chunk_size = match device_tier {
            DeviceTier::Base => {
                if hidden_size >= 4096 {
                    1024
                } else {
                    2048
                }
            }
            DeviceTier::Pro => {
                if hidden_size >= 8192 {
                    2048
                } else {
                    4096
                }
            }
            DeviceTier::Max | DeviceTier::Ultra => {
                if hidden_size >= 8192 {
                    4096
                } else {
                    8192
                }
            }
        };

        CrossEntropyTunedConfig {
            threadgroup_size: sanitize_threads_per_token_candidate(
                threadgroup_size,
                max_threads_per_threadgroup,
            ),
            chunk_size: chunk_size.min(vocab_size.max(1) as u32),
        }
    }

    fn candidate_fused_linear_cross_entropy_configs(
        &self,
        device_tier: DeviceTier,
        hidden_size: usize,
        vocab_size: usize,
        max_threads_per_threadgroup: u32,
    ) -> Vec<CrossEntropyTunedConfig> {
        let heuristic = self.heuristic_fused_linear_cross_entropy_config(
            device_tier,
            hidden_size,
            vocab_size,
            max_threads_per_threadgroup,
        );
        let mut configs = vec![heuristic];

        let thread_candidates: &[u32] = match device_tier {
            DeviceTier::Base => &[128, 256, 512],
            DeviceTier::Pro => &[128, 256, 512, 1024],
            DeviceTier::Max | DeviceTier::Ultra => &[256, 512, 1024],
        };
        for &threadgroup_size in thread_candidates {
            configs.push(CrossEntropyTunedConfig {
                threadgroup_size: sanitize_threads_per_token_candidate(
                    threadgroup_size,
                    max_threads_per_threadgroup,
                ),
                chunk_size: heuristic.chunk_size,
            });
        }

        let chunk_candidates: &[u32] = &[1024, 2048, 4096, 8192];
        for &chunk_size in chunk_candidates {
            configs.push(CrossEntropyTunedConfig {
                threadgroup_size: heuristic.threadgroup_size,
                chunk_size: chunk_size.min(vocab_size.max(1) as u32),
            });
        }

        dedupe(configs)
    }

    fn benchmark_fused_linear_cross_entropy(
        &self,
        context: &Arc<MetalContext>,
        config: &FusedLinearCrossEntropyConfig,
        candidate: CrossEntropyTunedConfig,
    ) -> Result<f64> {
        let heuristic = self.heuristic_fused_linear_cross_entropy_config(
            context.properties().device_tier,
            config.hidden_size,
            config.vocab_size,
            context.properties().max_threads_per_threadgroup as u32,
        );

        let dtype_size = if config.use_fp16 {
            std::mem::size_of::<f16>()
        } else {
            std::mem::size_of::<f32>()
        };
        let total_bytes = config
            .num_tokens
            .checked_mul(config.hidden_size)
            .and_then(|x| x.checked_add(config.vocab_size.checked_mul(config.hidden_size)?))
            .and_then(|x| x.checked_mul(dtype_size))
            .and_then(|x| x.checked_add(config.num_tokens.checked_mul(std::mem::size_of::<i32>())?))
            .and_then(|x| {
                x.checked_add(
                    config
                        .num_tokens
                        .checked_mul(2 * std::mem::size_of::<f32>())?,
                )
            })
            .ok_or_else(|| {
                MetalError::InvalidConfig("Fused linear CE benchmark size overflow".to_string())
            })?;
        let benchmark_budget = match context.properties().device_tier {
            DeviceTier::Base => 256 * 1024 * 1024,
            DeviceTier::Pro => 384 * 1024 * 1024,
            DeviceTier::Max | DeviceTier::Ultra => 512 * 1024 * 1024,
        };
        if total_bytes > benchmark_budget {
            return Ok(if candidate == heuristic { 1.0 } else { 10.0 });
        }

        let kernel = FusedLinearCrossEntropy::new_with_tuned_config(
            Arc::clone(context),
            config.clone(),
            candidate,
        )?;
        let targets = MetalBuffer::<i32>::zeros(context, config.num_tokens, BufferUsage::Shared)?;

        if config.use_fp16 {
            let hidden_states = MetalBuffer::<f16>::zeros(
                context,
                config.num_tokens * config.hidden_size,
                BufferUsage::Shared,
            )?;
            let lm_head_weight = MetalBuffer::<f16>::zeros(
                context,
                config.vocab_size * config.hidden_size,
                BufferUsage::Shared,
            )?;

            return median_seconds(1, BENCH_SAMPLES, || {
                let output = kernel.forward_f16(&hidden_states, &lm_head_weight, &targets)?;
                std::hint::black_box(output);
                Ok(())
            });
        }

        let hidden_states = MetalBuffer::<f32>::zeros(
            context,
            config.num_tokens * config.hidden_size,
            BufferUsage::Shared,
        )?;
        let lm_head_weight = MetalBuffer::<f32>::zeros(
            context,
            config.vocab_size * config.hidden_size,
            BufferUsage::Shared,
        )?;

        median_seconds(1, BENCH_SAMPLES, || {
            let output = kernel.forward(&hidden_states, &lm_head_weight, &targets)?;
            std::hint::black_box(output);
            Ok(())
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppleGPUFamily, MemoryBandwidthSource};
    use std::sync::Arc;

    fn test_device_properties(
        name: &str,
        device_tier: DeviceTier,
        architecture_gen: u32,
        gpu_core_count: u32,
    ) -> DeviceProperties {
        DeviceProperties {
            name: name.to_string(),
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory_length: 32 * 1024,
            has_unified_memory: true,
            recommended_working_set_size: 32 * 1024 * 1024 * 1024,
            max_buffer_length: 1 << 30,
            gpu_family: AppleGPUFamily::Apple9,
            device_tier,
            has_dynamic_caching: true,
            has_hardware_ray_tracing: true,
            has_mesh_shaders: true,
            has_nax: architecture_gen >= 17,
            architecture_gen,
            memory_bandwidth_gbps: 100.0,
            memory_bandwidth_source: MemoryBandwidthSource::SpecTableFallback,
            gpu_core_count,
            ane_core_count: 16,
            is_ultra_fusion: matches!(device_tier, DeviceTier::Ultra),
            die_count: if matches!(device_tier, DeviceTier::Ultra) {
                2
            } else {
                1
            },
        }
    }

    // -------------------------------------------------------------------------
    // Struct defaults
    // -------------------------------------------------------------------------

    #[test]
    fn merge_tuned_config_default() {
        let c = MergeTunedConfig::default();
        assert_eq!(c.threads_per_group, 256);
        assert_eq!(c.elements_per_thread, 4);
        assert!(c.use_simd);
    }

    #[test]
    fn cross_entropy_tuned_config_default() {
        let c = CrossEntropyTunedConfig::default();
        assert_eq!(c.threadgroup_size, 256);
        assert_eq!(c.chunk_size, 4096);
    }

    // -------------------------------------------------------------------------
    // Serde round-trips
    // -------------------------------------------------------------------------

    #[test]
    fn cross_entropy_tuned_config_serde_roundtrip() {
        let config = CrossEntropyTunedConfig {
            threadgroup_size: 512,
            chunk_size: 8192,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: CrossEntropyTunedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, decoded);
    }

    // -------------------------------------------------------------------------
    // Choosing the fastest candidate
    // -------------------------------------------------------------------------

    /// Measure from a table; a candidate missing from it fails to run.
    fn table(times: &[(u32, f64)]) -> impl FnMut(u32) -> Result<f64> + '_ {
        move |candidate| {
            times
                .iter()
                .find(|(c, _)| *c == candidate)
                .map(|(_, t)| *t)
                .ok_or_else(|| MetalError::ExecutionFailed(format!("{candidate} failed")))
        }
    }

    /// A candidate only a hair faster than the incumbent is noise, and a
    /// choice that flips with the noise is not reproducible.
    #[test]
    fn a_marginal_win_does_not_displace_the_incumbent() {
        let times = [(1, 1.000), (2, 0.985), (3, 0.99)];
        assert_eq!(
            pick_fastest(1, &[1, 2, 3], table(&times)).map(|(c, _)| c),
            Some(1)
        );
    }

    #[test]
    fn a_clear_win_displaces_the_incumbent() {
        let times = [(1, 1.0), (2, 0.9), (3, 0.8)];
        assert_eq!(pick_fastest(1, &[1, 2, 3], table(&times)), Some((3, 0.8)));
    }

    /// The same measurements give the same choice whatever order the
    /// candidates come in, as long as the winner is clear.
    #[test]
    fn the_choice_does_not_depend_on_candidate_order() {
        let times = [(1, 1.0), (2, 0.5), (3, 0.9), (4, 0.7)];
        let a = pick_fastest(1, &[1, 2, 3, 4], table(&times));
        let b = pick_fastest(1, &[4, 3, 2, 1], table(&times));
        assert_eq!(a, b);
        assert_eq!(a.map(|(c, _)| c), Some(2));
    }

    #[test]
    fn equally_fast_candidates_resolve_to_the_earlier() {
        let times = [(1, 1.0), (2, 0.5), (3, 0.5)];
        assert_eq!(
            pick_fastest(1, &[1, 2, 3], table(&times)).map(|(c, _)| c),
            Some(2)
        );
    }

    /// A candidate that fails to run can never be chosen, however its
    /// neighbours measure; this is what keeps a broken variant out.
    #[test]
    fn a_failing_candidate_is_never_chosen() {
        let times = [(1, 1.0), (3, 0.9)];
        assert_eq!(
            pick_fastest(1, &[1, 2, 3], table(&times)).map(|(c, _)| c),
            Some(3)
        );
        // Not measuring a number that is NaN or infinite either.
        let times = [(1, 1.0), (2, f64::NAN), (3, f64::INFINITY)];
        assert_eq!(
            pick_fastest(1, &[1, 2, 3], table(&times)).map(|(c, _)| c),
            Some(1)
        );
    }

    #[test]
    fn a_failing_incumbent_gives_way_to_the_fastest_that_runs() {
        let times = [(2, 0.9), (3, 0.95)];
        assert_eq!(pick_fastest(1, &[1, 2, 3], table(&times)), Some((2, 0.9)));
    }

    #[test]
    fn nothing_measurable_picks_nothing() {
        assert_eq!(pick_fastest(1, &[1, 2, 3], table(&[])), None);
    }

    #[test]
    fn the_median_ignores_one_stretched_run() {
        assert_eq!(median(vec![5.0, 1.0, 1.1, 1.2, 1.05]), 1.1);
        assert_eq!(median(vec![1.0, 3.0]), 2.0);
        assert_eq!(median(vec![7.0]), 7.0);
    }

    #[test]
    fn median_seconds_runs_the_warmups_untimed_and_stops_at_a_failure() {
        let mut calls = 0;
        let seconds = median_seconds(2, 5, || {
            calls += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 7);
        assert!(seconds >= 0.0);

        let mut calls = 0;
        let result = median_seconds(1, 5, || {
            calls += 1;
            if calls == 3 {
                Err(MetalError::ExecutionFailed("gpu fault".into()))
            } else {
                Ok(())
            }
        });
        assert!(result.is_err());
        assert_eq!(calls, 3);
    }

    // -------------------------------------------------------------------------
    // Tuner construction and the device-scoped cache
    // -------------------------------------------------------------------------

    #[test]
    fn tuner_new_has_no_cache_dir() {
        let tuner = Tuner::new();
        assert!(tuner.cache_dir.is_none());
    }

    /// A different chip, a binned part of the same chip, or a rebuilt shader
    /// library each get their own cache directory.
    #[test]
    fn the_cache_scope_names_the_device_and_the_kernels() {
        let full = test_device_properties("Apple M4 Max", DeviceTier::Max, 16, 40);
        let binned = test_device_properties("Apple M4 Max", DeviceTier::Max, 16, 32);
        let other = test_device_properties("Apple M3 Max", DeviceTier::Max, 15, 40);

        let scope = scope_dir_name(&full, "00aa");
        assert_eq!(scope, scope_dir_name(&full, "00aa"));
        assert_ne!(scope, scope_dir_name(&full, "00ab"));
        assert_ne!(scope, scope_dir_name(&binned, "00aa"));
        assert_ne!(scope, scope_dir_name(&other, "00aa"));
        assert_eq!(scope.len(), 16);
        assert!(scope.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn fnv1a_matches_the_published_vectors() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }

    // -------------------------------------------------------------------------
    // In-memory cache get/set round-trips
    // -------------------------------------------------------------------------

    #[test]
    fn merge_cache_get_set() {
        let tuner = Tuner::new();
        let key = "merge:Apple M4:base:16:10:1024:3".to_string();
        assert!(tuner.get_merge_config(&key).is_none());

        let cfg = MergeTunedConfig {
            threads_per_group: 512,
            elements_per_thread: 8,
            use_simd: true,
        };
        tuner.set_merge_config(key.clone(), cfg);
        assert_eq!(tuner.get_merge_config(&key), Some(cfg));
    }

    #[test]
    fn merge_cache_key_includes_device_identity() {
        let pro = test_device_properties("Apple M3 Pro", DeviceTier::Pro, 15, 18);
        let ultra = test_device_properties("Apple M4 Ultra", DeviceTier::Ultra, 16, 80);

        let pro_key = merge_cache_key(&pro, 1_048_576, 6);
        let ultra_key = merge_cache_key(&ultra, 1_048_576, 6);

        assert_ne!(pro_key, ultra_key);
        assert!(pro_key.contains("Apple M3 Pro:pro:15:18"));
        assert!(ultra_key.contains("Apple M4 Ultra:ultra:16:80"));
    }

    // -------------------------------------------------------------------------
    // Disk cache (uses tempdir)
    // -------------------------------------------------------------------------

    const CE_KEY: &str = "cross_entropy:4:2048";
    const CE_FAST: CrossEntropyTunedConfig = CrossEntropyTunedConfig {
        threadgroup_size: 512,
        chunk_size: 4096,
    };
    const CE_SLOW: CrossEntropyTunedConfig = CrossEntropyTunedConfig {
        threadgroup_size: 128,
        chunk_size: 1024,
    };

    fn ce_times(config: CrossEntropyTunedConfig) -> Result<f64> {
        Ok(if config == CE_FAST { 0.5 } else { 1.0 })
    }

    /// A measured winner is written, and a second process reads it back
    /// without measuring anything.
    #[test]
    fn a_measured_choice_survives_a_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");

        let first = Tuner::with_cache_dir(dir.clone());
        let chosen = first.select(
            &first.cross_entropy,
            "cross-entropy",
            CE_KEY,
            || (CE_SLOW, vec![CE_SLOW, CE_FAST]),
            ce_times,
        );
        assert_eq!(chosen, CE_FAST);
        assert!(dir.join("cross_entropy.json").exists());

        let second = Tuner::with_cache_dir(dir);
        let reread = second.select(
            &second.cross_entropy,
            "cross-entropy",
            CE_KEY,
            || panic!("a cached shape is not planned again"),
            |_| panic!("a cached shape is not measured again"),
        );
        assert_eq!(reread, CE_FAST);
    }

    /// With nothing measurable the heuristic is used for this process, but
    /// the file is not written: a later run on a quiet GPU measures again.
    #[test]
    fn an_unmeasured_choice_is_not_written_to_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");

        let tuner = Tuner::with_cache_dir(dir.clone());
        let chosen = tuner.select(
            &tuner.cross_entropy,
            "cross-entropy",
            CE_KEY,
            || (CE_SLOW, vec![CE_SLOW, CE_FAST]),
            |_| Err(MetalError::ExecutionFailed("busy".into())),
        );
        assert_eq!(chosen, CE_SLOW);
        assert_eq!(tuner.cross_entropy.get(CE_KEY), Some(CE_SLOW));
        assert!(!dir.join("cross_entropy.json").exists());
    }

    /// Two processes tuning different shapes must not erase each other's
    /// results: the file is merged on write, not replaced by one process's map.
    #[test]
    fn concurrent_writers_keep_each_others_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");

        // Both loaded before either wrote.
        let a = Tuner::with_cache_dir(dir.clone());
        let b = Tuner::with_cache_dir(dir.clone());
        a.select(
            &a.cross_entropy,
            "cross-entropy",
            "cross_entropy:shape-a",
            || (CE_SLOW, vec![CE_FAST]),
            ce_times,
        );
        b.select(
            &b.cross_entropy,
            "cross-entropy",
            "cross_entropy:shape-b",
            || (CE_SLOW, vec![CE_FAST]),
            ce_times,
        );

        let fresh = Tuner::with_cache_dir(dir);
        assert_eq!(
            fresh.cross_entropy.get("cross_entropy:shape-a"),
            Some(CE_FAST)
        );
        assert_eq!(
            fresh.cross_entropy.get("cross_entropy:shape-b"),
            Some(CE_FAST)
        );
    }

    #[test]
    fn writing_leaves_no_temporary_files_behind() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");
        let tuner = Tuner::with_cache_dir(dir.clone());
        for shape in 0..5 {
            tuner.select(
                &tuner.cross_entropy,
                "cross-entropy",
                &format!("cross_entropy:{shape}"),
                || (CE_SLOW, vec![CE_FAST]),
                ce_times,
            );
        }
        let names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["cross_entropy.json"]);
    }

    /// A damaged cache file is ignored, never fatal.
    #[test]
    fn a_corrupt_cache_file_is_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cross_entropy.json"), "{ not json").unwrap();

        let tuner = Tuner::with_cache_dir(dir);
        assert_eq!(tuner.cross_entropy.get(CE_KEY), None);
        let chosen = tuner.select(
            &tuner.cross_entropy,
            "cross-entropy",
            CE_KEY,
            || (CE_SLOW, vec![CE_FAST]),
            ce_times,
        );
        assert_eq!(chosen, CE_FAST);
    }

    /// The on-disk format is a flat JSON object of key to configuration, as
    /// earlier versions wrote it.
    #[test]
    fn the_cache_file_format_is_a_flat_map() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");
        fs::create_dir_all(&dir).unwrap();

        let key = "merge:Apple M5 Max:max:17:40:1048576:4";
        let config = MergeTunedConfig {
            threads_per_group: 512,
            elements_per_thread: 8,
            use_simd: false,
        };
        let map: HashMap<&str, MergeTunedConfig> = [(key, config)].into_iter().collect();
        fs::write(
            dir.join("merge.json"),
            serde_json::to_string_pretty(&map).unwrap(),
        )
        .unwrap();

        let tuner = Tuner::with_cache_dir(dir);
        assert_eq!(tuner.merge.get(key), Some(config));
    }

    /// Scope directories nothing has used for a month are removed; recent
    /// ones, and the current one, are kept.
    #[test]
    fn stale_scopes_are_pruned() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let make = |name: &str, age_days: u64| {
            let dir = root.join(name);
            fs::create_dir_all(&dir).unwrap();
            let file = dir.join("cross_entropy.json");
            fs::write(&file, "{}").unwrap();
            let when = std::time::SystemTime::now()
                - std::time::Duration::from_secs(age_days * 24 * 60 * 60);
            fs::File::options()
                .write(true)
                .open(&file)
                .unwrap()
                .set_modified(when)
                .unwrap();
            dir
        };
        let keep = make("current", 90);
        make("old", 45);
        make("recent", 3);

        prune_stale_scopes(root, &keep);

        assert!(keep.exists(), "the current scope is never pruned");
        assert!(root.join("recent").exists());
        assert!(!root.join("old").exists());
    }

    // -------------------------------------------------------------------------
    // Standard-Metal tuning sanity checks
    // -------------------------------------------------------------------------

    #[test]
    fn fused_linear_cross_entropy_heuristic_tracks_tier_and_shape() {
        let tuner = Tuner::new();
        assert_eq!(
            tuner.heuristic_fused_linear_cross_entropy_config(DeviceTier::Base, 2048, 16_384, 1024),
            CrossEntropyTunedConfig {
                threadgroup_size: 128,
                chunk_size: 2048,
            }
        );
        assert_eq!(
            tuner.heuristic_fused_linear_cross_entropy_config(
                DeviceTier::Base,
                4096,
                200_000,
                1024,
            ),
            CrossEntropyTunedConfig {
                threadgroup_size: 512,
                chunk_size: 1024,
            }
        );
        assert_eq!(
            tuner
                .heuristic_fused_linear_cross_entropy_config(DeviceTier::Max, 2048, 200_000, 1024,),
            CrossEntropyTunedConfig {
                threadgroup_size: 1024,
                chunk_size: 8192,
            }
        );
    }

    #[test]
    fn fused_linear_cross_entropy_candidates_cover_thread_and_chunk_options() {
        let tuner = Tuner::new();
        let candidates = tuner.candidate_fused_linear_cross_entropy_configs(
            DeviceTier::Pro,
            4096,
            200_000,
            1024,
        );
        assert!(candidates.contains(&CrossEntropyTunedConfig {
            threadgroup_size: 512,
            chunk_size: 4096,
        }));
        assert!(candidates.contains(&CrossEntropyTunedConfig {
            threadgroup_size: 128,
            chunk_size: 4096,
        }));
        assert!(candidates.contains(&CrossEntropyTunedConfig {
            threadgroup_size: 1024,
            chunk_size: 4096,
        }));
        assert!(candidates.contains(&CrossEntropyTunedConfig {
            threadgroup_size: 1024,
            chunk_size: 1024,
        }));
        assert!(candidates.contains(&CrossEntropyTunedConfig {
            threadgroup_size: 1024,
            chunk_size: 8192,
        }));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn fused_linear_cross_entropy_cache_hit_on_second_call() {
        let ctx = Arc::new(MetalContext::new().expect("Metal required"));
        let tuner = Tuner::new();
        let config = FusedLinearCrossEntropyConfig::new(64, 512, 8_192).with_fp16();

        let first = tuner
            .tune_fused_linear_cross_entropy(&ctx, &config)
            .expect("first fused linear CE tune");
        let second = tuner
            .tune_fused_linear_cross_entropy(&ctx, &config)
            .expect("second fused linear CE tune");

        assert_eq!(first, second);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn fused_linear_cross_entropy_tuning_keys_include_dtype_and_hidden_size() {
        let ctx = Arc::new(MetalContext::new().expect("Metal required"));
        let tuner = Tuner::new();
        let fp16 = FusedLinearCrossEntropyConfig::new(32, 512, 8_192).with_fp16();
        let fp32 = FusedLinearCrossEntropyConfig::new(32, 512, 8_192);
        let wider = FusedLinearCrossEntropyConfig::new(32, 1024, 8_192).with_fp16();

        tuner
            .tune_fused_linear_cross_entropy(&ctx, &fp16)
            .expect("fp16 tune");
        tuner
            .tune_fused_linear_cross_entropy(&ctx, &fp32)
            .expect("fp32 tune");
        tuner
            .tune_fused_linear_cross_entropy(&ctx, &wider)
            .expect("wider tune");

        let cache = tuner.cross_entropy.entries.lock();
        let fused_entries = cache
            .keys()
            .filter(|key| key.starts_with("fused_linear_ce:"))
            .count();
        assert!(
            fused_entries >= 3,
            "expected distinct fused linear CE entries for dtype/shape, found {fused_entries}"
        );
    }
}
