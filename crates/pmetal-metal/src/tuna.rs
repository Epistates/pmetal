#![allow(unsafe_code)]

//! Tuna: The "Tuna" Kernel Auto-Tuner.
//!
//! "Tuna" automatically finds the optimal kernel parameters (tile sizes, etc.)
//! for the running hardware (e.g., M1 vs M3 Max) by benchmarking candidates at runtime.
//!
//! # How it works
//!
//! 1. **Check Cache**: Looks up if we've already tuned this kernel for the given problem size.
//! 2. **Generate Candidates**: Creates a list of valid tile configurations (e.g., 32x32, 64x32).
//! 3. **Benchmark**: Runs each candidate for a few iterations, measuring execution time.
//! 4. **Select Winner**: Picks the fastest config and caches it.
//!
//! # Persistent Disk Cache
//!
//! When created via [`Tuner::with_persistent_cache`], tuning results are stored at
//! `~/.cache/pmetal/tuna/` as JSON files (one per kernel type). This means subsequent
//! process launches skip the benchmarking phase entirely for previously-seen problem sizes.
//!
//! # Example
//!
//! ```ignore
//! let tuner = Tuner::new();
//! let config = tuner.tune_lora_forward(&ctx, batch_size, in_features, out_features, rank)?;
//! println!("Best config: {:?}", config);
//!
//! // Or with persistent cache:
//! let tuner = Tuner::with_persistent_cache();
//! let config = tuner.tune_lora_forward(&ctx, batch_size, in_features, out_features, rank)?;
//! ```

use std::collections::HashMap;
use std::ffi::c_void;
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
use crate::kernels::flash_attention::{FlashAttention, FlashAttentionConfig};
use crate::kernels::fused_cross_entropy::{FusedLinearCrossEntropy, FusedLinearCrossEntropyConfig};
use crate::kernels::fused_lora::{FusedLoraConfig, FusedLoraParams};
use crate::kernels::fused_merge::{MergeConfig, TensorInfo};
use crate::kernels::fused_norm_lora::{FusedNormLora, FusedNormLoraConfig};
use crate::kernels::fused_swiglu::{FusedMLP, FusedSwiGLUConfig};
use crate::kernels::mpp_gemm::{MppGemm, MppGemmConfig, MppGemmKernelVariant};
use tracing::{debug, info, warn};

// ============================================================================
// Config Structs
// ============================================================================

/// Configuration for a tuned kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TunedConfig {
    /// M dimension tile size.
    pub tile_m: u32,
    /// N dimension tile size.
    pub tile_n: u32,
    /// K dimension tile size.
    pub tile_k: u32,
}

impl Default for TunedConfig {
    fn default() -> Self {
        Self {
            tile_m: 32,
            tile_n: 32,
            tile_k: 32,
        }
    }
}

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

/// Configuration for tuned SwiGLU activation kernels.
///
/// SwiGLU fuses gate and activation: `output = (gate * sigmoid(gate)) * x`.
/// The key tuning parameters are how many threads process each token and the
/// chunk size for data locality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SwiGLUTunedConfig {
    /// Threads assigned to process each output token.
    pub threads_per_token: u32,
    /// Number of elements processed per chunk (for cache reuse).
    pub chunk_size: u32,
}

impl Default for SwiGLUTunedConfig {
    fn default() -> Self {
        Self {
            threads_per_token: 256,
            chunk_size: 2048,
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

/// Configuration for tuned FlashAttention block sizes.
///
/// Standard Metal FlashAttention specializes query/key tile sizes at pipeline
/// creation time. These choices are safe to benchmark on first use and persist
/// per device/problem shape, rather than relying only on static tier tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct FlashAttentionTunedConfig {
    /// Number of query rows processed per block.
    pub block_q: u32,
    /// Number of key/value rows processed per block.
    pub block_k: u32,
}

impl Default for FlashAttentionTunedConfig {
    fn default() -> Self {
        Self {
            block_q: 32,
            block_k: 32,
        }
    }
}

/// Configuration for tuned Norm+LoRA fused kernels.
///
/// Fusing layer-norm with the LoRA projection saves a round-trip through
/// memory. The tiled path is beneficial when `out_features` is large enough
/// to amortize the shared-memory setup cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NormLoraTunedConfig {
    /// Threads assigned per output token during the LoRA projection.
    pub threads_per_token: u32,
    /// Use the tiled shared-memory code path (better for wide projections).
    pub use_tiled: bool,
}

impl Default for NormLoraTunedConfig {
    fn default() -> Self {
        Self {
            threads_per_token: 256,
            use_tiled: false,
        }
    }
}

/// Configuration for tuned MPP GEMM dispatch.
///
/// The tuner selects both the threadgroup tile/simdgroup variant and the tile
/// walk order for a given MPP GEMM problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct MppGemmTunedConfig {
    /// Kernel tile/simdgroup variant.
    pub variant: MppGemmKernelVariant,
    /// Whether to dispatch tiles in Morton Z-order rather than linear order.
    pub use_morton: bool,
}

impl Default for MppGemmTunedConfig {
    fn default() -> Self {
        Self {
            variant: MppGemmKernelVariant::default(),
            use_morton: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Problem description for tuning MPP GEMM dispatch.
pub struct MppGemmTuneRequest {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Batch count for batched GEMM dispatch.
    pub batch_size: usize,
    /// Whether the kernel uses fp16 buffers instead of fp32.
    pub use_fp16: bool,
    /// Whether the kernel performs in-place accumulation (`beta != 0`).
    pub accumulate: bool,
}

impl MppGemmTuneRequest {
    fn cache_key(self, device_name: &str, device_tier: DeviceTier) -> String {
        format!(
            "mpp_gemm:{}:{}:{}:{}:{}:{}:{}:{}",
            device_name,
            device_tier_key(device_tier),
            self.m,
            self.n,
            self.k,
            self.batch_size,
            if self.use_fp16 { "f16" } else { "f32" },
            if self.accumulate { "acc" } else { "plain" }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FlashAttentionTuneRequest {
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    query_seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
    is_causal: bool,
    has_sliding_window: bool,
    has_softcap: bool,
    is_training: bool,
}

fn bucket_flash_attention_decode_kv_seq_len(kv_seq_len: usize) -> usize {
    match kv_seq_len {
        0..=256 => kv_seq_len.div_ceil(32) * 32,
        257..=2048 => kv_seq_len.div_ceil(128) * 128,
        _ => kv_seq_len.div_ceil(256) * 256,
    }
}

impl FlashAttentionTuneRequest {
    fn from_config(config: &FlashAttentionConfig) -> Self {
        let is_bucketable_decode = !config.is_training
            && config.query_seq_len == 1
            && config.sliding_window.is_none()
            && config.softcap.is_none();
        let kv_seq_len = if is_bucketable_decode {
            bucket_flash_attention_decode_kv_seq_len(config.kv_seq_len)
        } else {
            config.kv_seq_len
        };
        Self {
            batch_size: config.batch_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            query_seq_len: config.query_seq_len,
            kv_seq_len,
            head_dim: config.head_dim,
            is_causal: config.is_causal,
            has_sliding_window: config.sliding_window.is_some(),
            has_softcap: config.softcap.is_some(),
            is_training: config.is_training,
        }
    }

    fn cache_key(self, device_name: &str, device_tier: DeviceTier) -> String {
        format!(
            "flash_attention:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            device_name,
            device_tier_key(device_tier),
            self.batch_size,
            self.num_heads,
            self.num_kv_heads,
            self.query_seq_len,
            self.kv_seq_len,
            self.head_dim,
            if self.is_causal { "causal" } else { "free" },
            if self.has_sliding_window {
                "window"
            } else {
                "full"
            },
            if self.has_softcap { "softcap" } else { "plain" },
            if self.is_training { "train" } else { "infer" }
        )
    }
}

/// The `fused_lora_forward` params a LoRA tuning run sends: the kernel's own
/// struct, with no LoRA+ scaling.
fn lora_bench_params(
    batch_size: usize,
    in_features: usize,
    out_features: usize,
    rank: usize,
) -> FusedLoraParams {
    FusedLoraParams::from(&FusedLoraConfig {
        batch_size,
        in_features,
        out_features,
        rank,
        scale: 1.0,
        lr_scale_a: None,
        lr_scale_b: None,
    })
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

fn lora_forward_cache_key(
    props: &DeviceProperties,
    batch_size: usize,
    in_features: usize,
    out_features: usize,
    rank: usize,
) -> String {
    format!(
        "fused_lora_forward:{}:{}:{}:{}:{}",
        device_cache_identity(props),
        batch_size,
        in_features,
        out_features,
        rank
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
    /// Fused LoRA forward tile sizes.
    /// Key: "fused_lora_forward:device:batch:in:out:rank"
    lora: TuneCache<TunedConfig>,

    /// Merge kernel launch shape.
    /// Key: "merge:device:num_elements:num_models"
    merge: TuneCache<MergeTunedConfig>,

    /// SwiGLU kernels.
    /// Key: "swiglu:device:batch:hidden:intermediate"
    swiglu: TuneCache<SwiGLUTunedConfig>,

    /// Fused linear cross-entropy kernels.
    cross_entropy: TuneCache<CrossEntropyTunedConfig>,

    /// FlashAttention block sizes.
    /// Key: "flash_attention:device:..."
    flash_attention: TuneCache<FlashAttentionTunedConfig>,

    /// Norm+LoRA fused kernels.
    /// Key: "norm_lora:device:batch:hidden:out_features:rank"
    norm_lora: TuneCache<NormLoraTunedConfig>,

    /// MPP GEMM dispatch.
    /// Key: "mpp_gemm:device:m:n:k:batch:dtype:accumulate"
    mpp_gemm: TuneCache<MppGemmTunedConfig>,

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
            lora: TuneCache::new("lora_forward.json"),
            merge: TuneCache::new("merge.json"),
            swiglu: TuneCache::new("swiglu.json"),
            cross_entropy: TuneCache::new("cross_entropy.json"),
            flash_attention: TuneCache::new("flash_attention.json"),
            norm_lora: TuneCache::new("norm_lora.json"),
            mpp_gemm: TuneCache::new("mpp_gemm.json"),
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
        self.lora.load_from(&dir);
        self.merge.load_from(&dir);
        self.swiglu.load_from(&dir);
        self.cross_entropy.load_from(&dir);
        self.flash_attention.load_from(&dir);
        self.norm_lora.load_from(&dir);
        self.mpp_gemm.load_from(&dir);
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
    // FlashAttention Block Tuning (benchmarked)
    // =========================================================================

    /// Tune standard Metal FlashAttention block sizes for the given problem.
    pub fn tune_flash_attention(
        &self,
        context: &Arc<MetalContext>,
        config: &FlashAttentionConfig,
    ) -> Result<FlashAttentionTunedConfig> {
        config.validate()?;

        let request = FlashAttentionTuneRequest::from_config(config);
        let props = context.properties();
        let key = request.cache_key(&props.name, props.device_tier);

        Ok(self.select(
            &self.flash_attention,
            "FlashAttention",
            &key,
            || {
                info!(
                    "Tuning FlashAttention blocks for [B={}, H={}, KVH={}, Q={}, KV={}, D={}, mode={}]...",
                    request.batch_size,
                    request.num_heads,
                    request.num_kv_heads,
                    request.query_seq_len,
                    request.kv_seq_len,
                    request.head_dim,
                    if request.is_training {
                        "train"
                    } else {
                        "infer"
                    }
                );
                (
                    self.heuristic_flash_attention_config(request.head_dim, props.device_tier),
                    self.candidate_flash_attention_configs(request.head_dim, props.device_tier),
                )
            },
            |candidate| self.benchmark_flash_attention(context, config, candidate),
        ))
    }

    fn heuristic_flash_attention_config(
        &self,
        head_dim: usize,
        tier: DeviceTier,
    ) -> FlashAttentionTunedConfig {
        match (head_dim, tier) {
            (64, DeviceTier::Base) => FlashAttentionTunedConfig {
                block_q: 64,
                block_k: 32,
            },
            (64, _) => FlashAttentionTunedConfig {
                block_q: 64,
                block_k: 64,
            },
            (80 | 96, DeviceTier::Base) => FlashAttentionTunedConfig {
                block_q: 32,
                block_k: 32,
            },
            (80 | 96, _) => FlashAttentionTunedConfig {
                block_q: 64,
                block_k: 32,
            },
            (128, DeviceTier::Max | DeviceTier::Ultra) => FlashAttentionTunedConfig {
                block_q: 64,
                block_k: 32,
            },
            (128, _) => FlashAttentionTunedConfig {
                block_q: 32,
                block_k: 32,
            },
            (256, DeviceTier::Max | DeviceTier::Ultra) => FlashAttentionTunedConfig {
                block_q: 32,
                block_k: 16,
            },
            (256, _) => FlashAttentionTunedConfig {
                block_q: 16,
                block_k: 16,
            },
            _ => FlashAttentionTunedConfig::default(),
        }
    }

    fn candidate_flash_attention_configs(
        &self,
        head_dim: usize,
        tier: DeviceTier,
    ) -> Vec<FlashAttentionTunedConfig> {
        let heuristic = self.heuristic_flash_attention_config(head_dim, tier);
        let mut candidates = vec![heuristic];

        let fallback = match head_dim {
            64 => vec![
                FlashAttentionTunedConfig {
                    block_q: 64,
                    block_k: 64,
                },
                FlashAttentionTunedConfig {
                    block_q: 64,
                    block_k: 32,
                },
                FlashAttentionTunedConfig {
                    block_q: 32,
                    block_k: 32,
                },
            ],
            80 | 96 => vec![
                FlashAttentionTunedConfig {
                    block_q: 64,
                    block_k: 32,
                },
                FlashAttentionTunedConfig {
                    block_q: 32,
                    block_k: 32,
                },
            ],
            128 => vec![
                FlashAttentionTunedConfig {
                    block_q: 64,
                    block_k: 32,
                },
                FlashAttentionTunedConfig {
                    block_q: 32,
                    block_k: 32,
                },
            ],
            256 => vec![
                FlashAttentionTunedConfig {
                    block_q: 32,
                    block_k: 16,
                },
                FlashAttentionTunedConfig {
                    block_q: 16,
                    block_k: 16,
                },
            ],
            _ => vec![FlashAttentionTunedConfig::default()],
        };

        for candidate in fallback {
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }

        candidates
    }

    fn benchmark_flash_attention(
        &self,
        context: &Arc<MetalContext>,
        config: &FlashAttentionConfig,
        candidate: FlashAttentionTunedConfig,
    ) -> Result<f64> {
        let heuristic = self
            .heuristic_flash_attention_config(config.head_dim, context.properties().device_tier);
        let total_bytes = config
            .query_size()
            .checked_mul(2)
            .and_then(|x| x.checked_add(config.kv_size().checked_mul(2)?))
            .and_then(|x| x.checked_add(config.kv_size().checked_mul(2)?))
            .and_then(|x| x.checked_add(config.output_size().checked_mul(2)?))
            .and_then(|x| {
                x.checked_add(if config.is_training {
                    config.logsumexp_size().checked_mul(4)?
                } else {
                    0
                })
            })
            .ok_or_else(|| {
                MetalError::InvalidConfig("FlashAttention benchmark size overflow".to_string())
            })?;
        if total_bytes > 256 * 1024 * 1024 {
            return Ok(if candidate == heuristic { 1.0 } else { 10.0 });
        }

        let queries = MetalBuffer::<f16>::zeros(context, config.query_size(), BufferUsage::Shared)?;
        let keys = MetalBuffer::<f16>::zeros(context, config.kv_size(), BufferUsage::Shared)?;
        let values = MetalBuffer::<f16>::zeros(context, config.kv_size(), BufferUsage::Shared)?;

        let flash =
            FlashAttention::new_with_tuned_blocks(Arc::clone(context), config.clone(), candidate)?;

        median_seconds(1, BENCH_SAMPLES, || {
            flash.forward(&queries, &keys, &values).map(drop)
        })
    }

    // =========================================================================
    // MPP GEMM Dispatch Tuning (benchmarked)
    // =========================================================================

    /// Tune the MPP GEMM dispatch shape and launch order for a specific problem size.
    pub fn tune_mpp_gemm(
        &self,
        context: &Arc<MetalContext>,
        request: MppGemmTuneRequest,
    ) -> Result<MppGemmTunedConfig> {
        let props = context.properties();
        let key = request.cache_key(&props.name, props.device_tier);

        // Benchmarking needs the Metal 4 kernels and a problem big enough to
        // be worth one; otherwise the heuristic is the answer and there is
        // nothing to measure.
        let measurable = if !props.has_nax() || context.pipeline_cache().metal4_library().is_none()
        {
            debug!(
                "Skipping MPP GEMM tuning on non-NAX or Metal 4-unavailable device: {}",
                props.name
            );
            false
        } else if !props.should_consider_mpp_gemm(request.m, request.n, request.k, request.use_fp16)
        {
            debug!(
                "Skipping MPP GEMM benchmarking for small problem [M={}, N={}, K={}, B={}]",
                request.m, request.n, request.k, request.batch_size
            );
            false
        } else {
            true
        };

        Ok(self.select(
            &self.mpp_gemm,
            "MPP GEMM",
            &key,
            || {
                let heuristic = self.heuristic_mpp_gemm_config(request, props.device_tier);
                if !measurable {
                    return (heuristic, Vec::new());
                }
                info!(
                    "Tuning MPP GEMM dispatch for [M={}, N={}, K={}, B={}, dtype={}, mode={}]...",
                    request.m,
                    request.n,
                    request.k,
                    request.batch_size,
                    if request.use_fp16 { "f16" } else { "f32" },
                    if request.accumulate {
                        "accumulate"
                    } else {
                        "overwrite"
                    }
                );
                (
                    heuristic,
                    self.candidate_mpp_gemm_configs(request, props.device_tier),
                )
            },
            |candidate| {
                if measurable {
                    self.benchmark_mpp_gemm(context, candidate, request)
                } else {
                    Err(MetalError::InvalidConfig("not benchmarked".into()))
                }
            },
        ))
    }

    fn heuristic_mpp_gemm_variant(
        &self,
        request: MppGemmTuneRequest,
        device_tier: DeviceTier,
    ) -> MppGemmKernelVariant {
        let wide = request.n >= request.m.saturating_mul(2);
        let tall = request.m >= request.n.saturating_mul(2);
        let tiny = request.m <= 32 || request.n <= 32;

        match device_tier {
            DeviceTier::Base => {
                if tiny {
                    MppGemmKernelVariant::Sg1_32x32
                } else if wide {
                    MppGemmKernelVariant::Sg2_32x64
                } else {
                    MppGemmKernelVariant::Sg2_64x32
                }
            }
            DeviceTier::Pro => {
                if tiny {
                    MppGemmKernelVariant::Sg1_32x32
                } else if wide {
                    MppGemmKernelVariant::Sg2_32x64
                } else if tall {
                    MppGemmKernelVariant::Sg2_64x32
                } else {
                    MppGemmKernelVariant::Sg4_64x64
                }
            }
            DeviceTier::Max | DeviceTier::Ultra => {
                if tiny {
                    MppGemmKernelVariant::Sg1_32x32
                } else if wide {
                    MppGemmKernelVariant::Sg2_32x64
                } else if tall {
                    MppGemmKernelVariant::Sg2_64x32
                } else {
                    MppGemmKernelVariant::Sg4_64x64
                }
            }
        }
    }

    fn heuristic_mpp_gemm_config(
        &self,
        request: MppGemmTuneRequest,
        device_tier: DeviceTier,
    ) -> MppGemmTunedConfig {
        let use_morton =
            request.accumulate || request.batch_size > 1 || request.m > 32 || request.n > 512;
        MppGemmTunedConfig {
            variant: self.heuristic_mpp_gemm_variant(request, device_tier),
            use_morton,
        }
    }

    fn candidate_mpp_gemm_variants(
        &self,
        request: MppGemmTuneRequest,
        device_tier: DeviceTier,
    ) -> Vec<MppGemmKernelVariant> {
        let mut variants = vec![self.heuristic_mpp_gemm_variant(request, device_tier)];
        match device_tier {
            DeviceTier::Base => variants.extend([
                MppGemmKernelVariant::Sg1_32x32,
                MppGemmKernelVariant::Sg2_64x32,
                MppGemmKernelVariant::Sg2_32x64,
            ]),
            DeviceTier::Pro => variants.extend([
                MppGemmKernelVariant::Sg1_32x32,
                MppGemmKernelVariant::Sg2_64x32,
                MppGemmKernelVariant::Sg2_32x64,
                MppGemmKernelVariant::Sg4_64x64,
            ]),
            DeviceTier::Max | DeviceTier::Ultra => variants.extend([
                MppGemmKernelVariant::Sg1_32x32,
                MppGemmKernelVariant::Sg2_64x32,
                MppGemmKernelVariant::Sg2_32x64,
                MppGemmKernelVariant::Sg4_64x64,
            ]),
        }
        let mut unique = Vec::with_capacity(variants.len());
        for variant in variants {
            if !unique.contains(&variant) {
                unique.push(variant);
            }
        }
        unique
    }

    fn candidate_mpp_gemm_configs(
        &self,
        request: MppGemmTuneRequest,
        device_tier: DeviceTier,
    ) -> Vec<MppGemmTunedConfig> {
        let preferred = self.heuristic_mpp_gemm_config(request, device_tier);
        let mut configs = vec![preferred];

        for variant in self.candidate_mpp_gemm_variants(request, device_tier) {
            configs.push(MppGemmTunedConfig {
                variant,
                use_morton: preferred.use_morton,
            });
            configs.push(MppGemmTunedConfig {
                variant,
                use_morton: !preferred.use_morton,
            });
        }

        let mut unique = Vec::with_capacity(configs.len());
        for config in configs {
            if !unique.contains(&config) {
                unique.push(config);
            }
        }
        unique
    }

    fn benchmark_mpp_gemm(
        &self,
        context: &Arc<MetalContext>,
        candidate: MppGemmTunedConfig,
        request: MppGemmTuneRequest,
    ) -> Result<f64> {
        let total_elements = request
            .batch_size
            .checked_mul(
                request
                    .m
                    .checked_mul(request.k)
                    .and_then(|x| x.checked_add(request.n.checked_mul(request.k)?))
                    .and_then(|x| x.checked_add(request.m.checked_mul(request.n)?))
                    .ok_or_else(|| {
                        MetalError::InvalidConfig("MPP GEMM benchmark size overflow".to_string())
                    })?,
            )
            .ok_or_else(|| {
                MetalError::InvalidConfig("MPP GEMM benchmark size overflow".to_string())
            })?;

        let bytes_per_element = if request.use_fp16 { 2usize } else { 4usize };
        let total_bytes = total_elements.saturating_mul(bytes_per_element);
        if total_bytes > 256 * 1024 * 1024 {
            let heuristic =
                self.heuristic_mpp_gemm_config(request, context.properties().device_tier);
            return Ok(if heuristic == candidate { 1.0 } else { 10.0 });
        }

        if request.use_fp16 {
            let a = MetalBuffer::<f16>::zeros(
                context,
                request.batch_size * request.m * request.k,
                BufferUsage::Shared,
            )?;
            let b = MetalBuffer::<f16>::zeros(
                context,
                request.batch_size * request.n * request.k,
                BufferUsage::Shared,
            )?;
            let d = MetalBuffer::<f16>::zeros(
                context,
                request.batch_size * request.m * request.n,
                BufferUsage::Shared,
            )?;
            let mut config = MppGemmConfig::new(request.m, request.n, request.k);
            config.batch_size = request.batch_size;
            config.use_fp16 = true;
            config.use_morton = candidate.use_morton;
            config.kernel_variant = candidate.variant;
            config.beta = if request.accumulate { 1.0 } else { 0.0 };
            config.auto_tune_morton = false;
            config.auto_tune_variant = false;
            let gemm = MppGemm::new(Arc::clone(context), config);
            median_seconds(1, BENCH_SAMPLES, || gemm.execute(&a, &b, &d))
        } else {
            let a = MetalBuffer::<f32>::zeros(
                context,
                request.batch_size * request.m * request.k,
                BufferUsage::Shared,
            )?;
            let b = MetalBuffer::<f32>::zeros(
                context,
                request.batch_size * request.n * request.k,
                BufferUsage::Shared,
            )?;
            let d = MetalBuffer::<f32>::zeros(
                context,
                request.batch_size * request.m * request.n,
                BufferUsage::Shared,
            )?;
            let mut config = MppGemmConfig::new(request.m, request.n, request.k);
            config.batch_size = request.batch_size;
            config.use_fp16 = false;
            config.use_morton = candidate.use_morton;
            config.kernel_variant = candidate.variant;
            config.beta = if request.accumulate { 1.0 } else { 0.0 };
            config.auto_tune_morton = false;
            config.auto_tune_variant = false;
            let gemm = MppGemm::new(Arc::clone(context), config);
            median_seconds(1, BENCH_SAMPLES, || gemm.execute(&a, &b, &d))
        }
    }

    // =========================================================================
    // LoRA Forward Kernel Tuning (benchmarked)
    // =========================================================================

    /// Tune the Fused LoRA Forward kernel.
    pub fn tune_lora_forward(
        &self,
        context: &MetalContext,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        rank: usize,
    ) -> Result<TunedConfig> {
        let key = lora_forward_cache_key(
            context.properties(),
            batch_size,
            in_features,
            out_features,
            rank,
        );

        Ok(self.select(
            &self.lora,
            "LoRA forward",
            &key,
            || {
                info!(
                    "Tuning LoRA Forward for [B={}, I={}, O={}, R={}]...",
                    batch_size, in_features, out_features, rank
                );
                let candidates = self.generate_lora_candidates(context);
                debug!(
                    "Generated {} valid candidates for device (max threads: {})",
                    candidates.len(),
                    context.properties().max_threads_per_threadgroup
                );
                (TunedConfig::default(), candidates)
            },
            |candidate| {
                self.benchmark_lora_forward(
                    context,
                    candidate,
                    batch_size,
                    in_features,
                    out_features,
                    rank,
                )
            },
        ))
    }

    /// Generate candidate configurations filtered by device capabilities.
    ///
    /// Filters candidates to only include tile sizes that fit within
    /// the device's max threads per threadgroup limit, and prioritizes
    /// based on device tier (M4 Max/Ultra get larger tiles first).
    fn generate_lora_candidates(&self, context: &MetalContext) -> Vec<TunedConfig> {
        use crate::context::DeviceTier;

        const SIMD_SIZE: u64 = 32;

        let props = context.properties();
        let max_threads = props.max_threads_per_threadgroup;

        // Device tier-aware candidate ordering
        // Higher tier devices benefit more from larger tiles
        let all_candidates: Vec<TunedConfig> = match props.device_tier {
            DeviceTier::Ultra | DeviceTier::Max => vec![
                // M4 Max/Ultra: Start with largest tiles (best for high bandwidth)
                TunedConfig {
                    tile_m: 64,
                    tile_n: 64,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 64,
                    tile_n: 32,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 32,
                    tile_n: 64,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 32,
                    tile_n: 32,
                    tile_k: 32,
                },
            ],
            DeviceTier::Pro => vec![
                // M4 Pro: Balance between tile size and occupancy
                TunedConfig {
                    tile_m: 64,
                    tile_n: 32,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 32,
                    tile_n: 64,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 64,
                    tile_n: 64,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 32,
                    tile_n: 32,
                    tile_k: 32,
                },
            ],
            DeviceTier::Base => vec![
                // M4 Base: Smaller tiles for better occupancy
                TunedConfig {
                    tile_m: 32,
                    tile_n: 32,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 32,
                    tile_n: 64,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 16,
                    tile_n: 64,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 64,
                    tile_n: 32,
                    tile_k: 32,
                },
                TunedConfig {
                    tile_m: 16,
                    tile_n: 32,
                    tile_k: 32,
                },
            ],
        };

        // Filter candidates by device capability
        // Threadgroup size: [TILE_N, TILE_M/SIMD_SIZE, 1]
        // Total threads = TILE_N * (TILE_M / SIMD_SIZE)
        all_candidates
            .into_iter()
            .filter(|config| {
                let threads = (config.tile_n as u64) * ((config.tile_m as u64) / SIMD_SIZE);
                threads <= max_threads
            })
            .collect()
    }

    /// Run a benchmark for a specific configuration.
    fn benchmark_lora_forward(
        &self,
        context: &MetalContext,
        config: TunedConfig,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        rank: usize,
    ) -> Result<f64> {
        // Create specialized pipeline
        let mut constants = HashMap::new();
        constants.insert(0, config.tile_m);
        constants.insert(1, config.tile_n);
        constants.insert(2, config.tile_k);

        let pipeline = context
            .pipeline_cache_mut()
            .get_or_create_specialized_pipeline(
                context.device(),
                "fused_lora_forward",
                &constants,
            )?;

        // Validate threadgroup size logic from kernel
        // Threadgroup: [TILE_N, TILE_M/SIMD_SIZE, 1]
        let _threads = (config.tile_n as u64) * ((config.tile_m as u64) / 32);
        if _threads > pipeline.maxTotalThreadsPerThreadgroup() as u64 {
            return Err(MetalError::PipelineCreation(
                "Threads exceed max threadgroup size".into(),
            ));
        }

        let device = context.device();

        // Estimate memory usage. If > 500MB, skip tuning or use smaller proxy.
        // x: f16
        let total_bytes =
            (batch_size * in_features + out_features * in_features + batch_size * out_features) * 2;
        if total_bytes > 500 * 1024 * 1024 {
            debug!(
                "Skipping benchmark (memory too large: {} MB)",
                total_bytes / 1024 / 1024
            );
            // Return dummy valid time to avoid failing, but high enough not to be picked
            // unless it's the default
            if config.tile_m == 32 && config.tile_n == 32 {
                return Ok(1.0); // Default penalty
            } else {
                return Ok(100.0);
            }
        }

        // Allocation (using unchecked createBuffer for speed)
        let options = MTLResourceOptions::StorageModePrivate;

        let x_size = batch_size * in_features * 2;
        let w_size = out_features * in_features * 2;
        let y_size = batch_size * out_features * 2;
        // Allocations
        // NOTE: newBufferWithLength_options takes usize in newer bindings
        let buf_x = device.newBufferWithLength_options(x_size, options).ok_or(
            MetalError::BufferCreation {
                size: x_size,
                reason: "x buffer".into(),
            },
        )?;
        let buf_w = device.newBufferWithLength_options(w_size, options).ok_or(
            MetalError::BufferCreation {
                size: w_size,
                reason: "w buffer".into(),
            },
        )?;
        let buf_a = device
            .newBufferWithLength_options(rank * in_features * 2, options)
            .ok_or(MetalError::BufferCreation {
                size: rank * in_features * 2,
                reason: "a buffer".into(),
            })?;
        let buf_b = device
            .newBufferWithLength_options(out_features * rank * 2, options)
            .ok_or(MetalError::BufferCreation {
                size: out_features * rank * 2,
                reason: "b buffer".into(),
            })?;
        let buf_y = device.newBufferWithLength_options(y_size, options).ok_or(
            MetalError::BufferCreation {
                size: y_size,
                reason: "y buffer".into(),
            },
        )?;
        let buf_xa = device
            .newBufferWithLength_options(batch_size * rank * 2, options)
            .ok_or(MetalError::BufferCreation {
                size: batch_size * rank * 2,
                reason: "xa buffer".into(),
            })?;

        let params = lora_bench_params(batch_size, in_features, out_features, rank);
        let params_size = std::mem::size_of::<FusedLoraParams>();
        let params_ptr = NonNull::from(&params).cast::<c_void>();

        let buf_params = unsafe {
            device.newBufferWithBytes_length_options(
                params_ptr,
                params_size,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            )
        }
        .ok_or(MetalError::BufferCreation {
            size: params_size,
            reason: "params buffer".into(),
        })?;

        median_seconds(1, BENCH_SAMPLES, || {
            self.dispatch_kernel(
                context,
                &pipeline,
                &config,
                &buf_x,
                &buf_w,
                &buf_a,
                &buf_b,
                &buf_y,
                &buf_xa,
                &buf_params,
                batch_size,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_kernel(
        &self,
        context: &MetalContext,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        config: &TunedConfig,
        x: &ProtocolObject<dyn MTLBuffer>,
        w: &ProtocolObject<dyn MTLBuffer>,
        a: &ProtocolObject<dyn MTLBuffer>,
        b: &ProtocolObject<dyn MTLBuffer>,
        y: &ProtocolObject<dyn MTLBuffer>,
        xa: &ProtocolObject<dyn MTLBuffer>,
        params: &ProtocolObject<dyn MTLBuffer>,
        batch_size: usize,
        out_features: usize,
    ) -> Result<()> {
        let queue = context.command_queue();
        let buffer = queue.new_command_buffer()?;
        let encoder = buffer
            .computeCommandEncoder()
            .ok_or(MetalError::CommandQueueCreation)?;

        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(w), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(a), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(b), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(xa), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(params), 0, 6);
        }

        // Calculate grid
        let grid_size = MTLSize {
            width: batch_size.div_ceil(config.tile_m as usize),
            height: out_features.div_ceil(config.tile_n as usize),
            depth: 1,
        };

        // Threadgroup: [TILE_N, TILE_M/SIMD_SIZE, 1]
        let threadgroup_size = MTLSize {
            width: config.tile_n as usize,
            height: (config.tile_m as usize) / 32, // SIMD_SIZE is 32
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();

        buffer.commit();
        buffer.waitUntilCompleted();
        check_completed(&buffer)
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

    // =========================================================================
    // SwiGLU Kernel Tuning (benchmarked)
    // =========================================================================

    /// Tune the SwiGLU activation kernel for the given problem size.
    ///
    /// # Arguments
    /// * `context` - Metal context
    /// * `batch_size` - Number of tokens in the batch
    /// * `hidden_size` - Model hidden dimension
    /// * `intermediate_size` - MLP intermediate dimension (typically 8/3 * hidden_size)
    ///
    /// # Returns
    /// Benchmarked optimal configuration for SwiGLU on this hardware, with
    /// heuristic fallback for oversized problems.
    pub fn tune_swiglu(
        &self,
        context: &Arc<MetalContext>,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<SwiGLUTunedConfig> {
        let props = context.properties();
        let key = format!(
            "swiglu:{}:{}:{}:{}:{}",
            props.name,
            device_tier_key(props.device_tier),
            batch_size,
            hidden_size,
            intermediate_size
        );
        let config = FusedSwiGLUConfig::new(batch_size, hidden_size, intermediate_size);
        let max_threads = props.max_threads_per_threadgroup as u32;

        Ok(self.select(
            &self.swiglu,
            "SwiGLU",
            &key,
            || {
                info!(
                    "Tuning SwiGLU config for [B={}, H={}, I={}]...",
                    batch_size, hidden_size, intermediate_size
                );
                (
                    self.heuristic_swiglu_config(props.device_tier, intermediate_size, max_threads),
                    self.candidate_swiglu_configs(
                        props.device_tier,
                        intermediate_size,
                        max_threads,
                    ),
                )
            },
            |candidate| self.benchmark_swiglu(context, &config, candidate),
        ))
    }

    /// Select the best SwiGLU config using device-tier heuristics.
    fn heuristic_swiglu_config(
        &self,
        device_tier: DeviceTier,
        intermediate_size: usize,
        max_threads_per_threadgroup: u32,
    ) -> SwiGLUTunedConfig {
        // For very small intermediate sizes, a smaller chunk avoids over-committing
        // threadgroup memory on any device tier.
        let small_intermediate = intermediate_size < 2048;

        let threads_per_token = match device_tier {
            DeviceTier::Ultra | DeviceTier::Max => 512,
            DeviceTier::Pro => 256,
            DeviceTier::Base => {
                if small_intermediate {
                    128
                } else {
                    256
                }
            }
        };

        SwiGLUTunedConfig {
            threads_per_token: sanitize_threads_per_token_candidate(
                threads_per_token,
                max_threads_per_threadgroup,
            ),
            chunk_size: match device_tier {
                DeviceTier::Ultra | DeviceTier::Max => {
                    if small_intermediate {
                        2048
                    } else {
                        4096
                    }
                }
                DeviceTier::Pro => {
                    if small_intermediate {
                        2048
                    } else {
                        4096
                    }
                }
                DeviceTier::Base => {
                    if small_intermediate {
                        1024
                    } else {
                        2048
                    }
                }
            },
        }
    }

    fn candidate_swiglu_configs(
        &self,
        device_tier: DeviceTier,
        intermediate_size: usize,
        max_threads_per_threadgroup: u32,
    ) -> Vec<SwiGLUTunedConfig> {
        let heuristic = self.heuristic_swiglu_config(
            device_tier,
            intermediate_size,
            max_threads_per_threadgroup,
        );
        let mut candidates = vec![heuristic];

        let thread_candidates: &[u32] = match device_tier {
            DeviceTier::Base => &[128, 256],
            DeviceTier::Pro => &[128, 256, 512],
            DeviceTier::Max | DeviceTier::Ultra => &[256, 512, 1024],
        };
        for &threads in thread_candidates {
            candidates.push(SwiGLUTunedConfig {
                threads_per_token: sanitize_threads_per_token_candidate(
                    threads,
                    max_threads_per_threadgroup,
                ),
                chunk_size: heuristic.chunk_size,
            });
        }

        let chunk_candidates: &[u32] = if intermediate_size < 2048 {
            &[1024, 2048]
        } else {
            &[1024, 2048, 4096]
        };
        for &chunk_size in chunk_candidates {
            candidates.push(SwiGLUTunedConfig {
                threads_per_token: heuristic.threads_per_token,
                chunk_size,
            });
        }

        dedupe(candidates)
    }

    fn benchmark_swiglu(
        &self,
        context: &Arc<MetalContext>,
        config: &FusedSwiGLUConfig,
        candidate: SwiGLUTunedConfig,
    ) -> Result<f64> {
        let heuristic = self.heuristic_swiglu_config(
            context.properties().device_tier,
            config.intermediate_size,
            context.properties().max_threads_per_threadgroup as u32,
        );
        let total_bytes = config
            .batch_size
            .checked_mul(config.hidden_size)
            .and_then(|x| x.checked_add(config.intermediate_size.checked_mul(config.hidden_size)?))
            .and_then(|x| x.checked_add(config.intermediate_size.checked_mul(config.hidden_size)?))
            .and_then(|x| x.checked_add(config.hidden_size.checked_mul(config.intermediate_size)?))
            .and_then(|x| x.checked_add(config.batch_size.checked_mul(config.hidden_size)?))
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                MetalError::InvalidConfig("SwiGLU benchmark size overflow".to_string())
            })?;
        if total_bytes > 256 * 1024 * 1024 {
            return Ok(if candidate == heuristic { 1.0 } else { 10.0 });
        }

        let input = MetalBuffer::<f32>::zeros(
            context,
            config.batch_size * config.hidden_size,
            BufferUsage::Shared,
        )?;
        let gate_weight = MetalBuffer::<f32>::zeros(
            context,
            config.intermediate_size * config.hidden_size,
            BufferUsage::Shared,
        )?;
        let up_weight = MetalBuffer::<f32>::zeros(
            context,
            config.intermediate_size * config.hidden_size,
            BufferUsage::Shared,
        )?;
        let down_weight = MetalBuffer::<f32>::zeros(
            context,
            config.hidden_size * config.intermediate_size,
            BufferUsage::Shared,
        )?;

        let kernel =
            FusedMLP::new_with_tuned_config(Arc::clone(context), config.clone(), candidate)?;

        median_seconds(1, BENCH_SAMPLES, || {
            kernel
                .forward(&input, &gate_weight, &up_weight, &down_weight)
                .map(drop)
        })
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

    // =========================================================================
    // Norm+LoRA Fused Kernel Tuning (benchmarked)
    // =========================================================================

    /// Tune the fused Layer-Norm + LoRA projection kernel.
    ///
    /// Fusing normalization with the LoRA forward pass eliminates an intermediate
    /// write-back to global memory. The tiled path (shared-memory accumulation)
    /// is beneficial when `out_features` is wide enough that the tiling
    /// amortises the shared-memory setup cost.
    ///
    /// # Arguments
    /// * `context` - Metal context
    /// * `batch_size` - Number of input tokens
    /// * `hidden_size` - Model hidden dimension (input width)
    /// * `out_features` - LoRA output dimension
    /// * `rank` - LoRA rank
    ///
    /// # Returns
    /// Benchmarked optimal configuration for Norm+LoRA on this hardware, with
    /// heuristic fallback for oversized problems.
    pub fn tune_norm_lora(
        &self,
        context: &Arc<MetalContext>,
        batch_size: usize,
        hidden_size: usize,
        out_features: usize,
        rank: usize,
    ) -> Result<NormLoraTunedConfig> {
        let props = context.properties();
        let key = format!(
            "norm_lora:{}:{}:{}:{}:{}:{}",
            props.name,
            device_tier_key(props.device_tier),
            batch_size,
            hidden_size,
            out_features,
            rank
        );
        let config = FusedNormLoraConfig::new(batch_size, hidden_size, out_features, rank, 16.0);
        let max_threads = props.max_threads_per_threadgroup as u32;

        Ok(self.select(
            &self.norm_lora,
            "Norm+LoRA",
            &key,
            || {
                info!(
                    "Tuning Norm+LoRA config for [B={}, H={}, O={}, R={}]...",
                    batch_size, hidden_size, out_features, rank
                );
                (
                    self.heuristic_norm_lora_config(props.device_tier, out_features, max_threads),
                    self.candidate_norm_lora_configs(props.device_tier, out_features, max_threads),
                )
            },
            |candidate| self.benchmark_norm_lora(context, &config, candidate),
        ))
    }

    /// Select the best Norm+LoRA config using heuristics.
    fn heuristic_norm_lora_config(
        &self,
        device_tier: DeviceTier,
        out_features: usize,
        max_threads_per_threadgroup: u32,
    ) -> NormLoraTunedConfig {
        // Tiled path is profitable when out_features is wide enough.
        let use_tiled = out_features > 256;

        let threads_per_token = match device_tier {
            DeviceTier::Ultra | DeviceTier::Max => 512,
            DeviceTier::Pro => 256,
            DeviceTier::Base => 128,
        };

        NormLoraTunedConfig {
            threads_per_token: sanitize_threads_per_token_candidate(
                threads_per_token,
                max_threads_per_threadgroup,
            ),
            use_tiled,
        }
    }

    fn candidate_norm_lora_configs(
        &self,
        device_tier: DeviceTier,
        out_features: usize,
        max_threads_per_threadgroup: u32,
    ) -> Vec<NormLoraTunedConfig> {
        let heuristic =
            self.heuristic_norm_lora_config(device_tier, out_features, max_threads_per_threadgroup);
        let mut configs = vec![heuristic];

        let thread_candidates: &[u32] = match device_tier {
            DeviceTier::Base => &[128, 256],
            DeviceTier::Pro => &[128, 256, 512],
            DeviceTier::Max | DeviceTier::Ultra => &[256, 512, 1024],
        };
        for &threads in thread_candidates {
            configs.push(NormLoraTunedConfig {
                threads_per_token: sanitize_threads_per_token_candidate(
                    threads,
                    max_threads_per_threadgroup,
                ),
                use_tiled: heuristic.use_tiled,
            });
        }

        configs.push(NormLoraTunedConfig {
            threads_per_token: heuristic.threads_per_token,
            use_tiled: false,
        });
        configs.push(NormLoraTunedConfig {
            threads_per_token: heuristic.threads_per_token,
            use_tiled: true,
        });

        dedupe(configs)
    }

    fn benchmark_norm_lora(
        &self,
        context: &Arc<MetalContext>,
        config: &FusedNormLoraConfig,
        candidate: NormLoraTunedConfig,
    ) -> Result<f64> {
        let heuristic = self.heuristic_norm_lora_config(
            context.properties().device_tier,
            config.out_features,
            context.properties().max_threads_per_threadgroup as u32,
        );
        let total_bytes = config
            .batch_size
            .checked_mul(config.hidden_size)
            .and_then(|x| x.checked_add(config.hidden_size))
            .and_then(|x| x.checked_add(config.out_features.checked_mul(config.hidden_size)?))
            .and_then(|x| x.checked_add(config.lora_rank.checked_mul(config.hidden_size)?))
            .and_then(|x| x.checked_add(config.out_features.checked_mul(config.lora_rank)?))
            .and_then(|x| x.checked_add(config.batch_size.checked_mul(config.out_features)?))
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                MetalError::InvalidConfig("Norm+LoRA benchmark size overflow".to_string())
            })?;
        if total_bytes > 256 * 1024 * 1024 {
            return Ok(if candidate == heuristic { 1.0 } else { 10.0 });
        }

        let input = MetalBuffer::<f32>::zeros(
            context,
            config.batch_size * config.hidden_size,
            BufferUsage::Shared,
        )?;
        let gamma = MetalBuffer::<f32>::zeros(context, config.hidden_size, BufferUsage::Shared)?;
        let weight = MetalBuffer::<f32>::zeros(
            context,
            config.out_features * config.hidden_size,
            BufferUsage::Shared,
        )?;
        let lora_a = MetalBuffer::<f32>::zeros(
            context,
            config.lora_rank * config.hidden_size,
            BufferUsage::Shared,
        )?;
        let lora_b = MetalBuffer::<f32>::zeros(
            context,
            config.out_features * config.lora_rank,
            BufferUsage::Shared,
        )?;

        let kernel =
            FusedNormLora::new_with_tuned_config(Arc::clone(context), config.clone(), candidate)?;

        median_seconds(1, BENCH_SAMPLES, || {
            kernel
                .forward(&input, &gamma, &weight, &lora_a, &lora_b)
                .map(drop)
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

    #[test]
    fn the_lora_tuning_run_sends_the_whole_shader_params_struct() {
        // `FusedLoraParams` in fused_lora.metal: four uints, then scale,
        // lr_scale_a and lr_scale_b as floats, 28 bytes. The tuner used to
        // send a 20-byte struct of its own without the two LoRA+ scales.
        let params = lora_bench_params(8, 64, 32, 4);
        assert_eq!(std::mem::size_of_val(&params), 28);
        // SAFETY: `FusedLoraParams` is a `#[repr(C)]` struct of seven 4-byte
        // scalars, so it is 7 initialized `u32`-sized words.
        let words: &[u32] =
            unsafe { std::slice::from_raw_parts(NonNull::from(&params).cast().as_ptr(), 7) };
        assert_eq!(&words[..4], &[8, 64, 32, 4]);
        let floats: Vec<f32> = words[4..].iter().map(|w| f32::from_bits(*w)).collect();
        assert_eq!(floats, [1.0, 1.0, 1.0], "scale, lr_scale_a, lr_scale_b");
    }

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
    fn tuned_config_default() {
        let c = TunedConfig::default();
        assert_eq!(c.tile_m, 32);
        assert_eq!(c.tile_n, 32);
        assert_eq!(c.tile_k, 32);
    }

    #[test]
    fn merge_tuned_config_default() {
        let c = MergeTunedConfig::default();
        assert_eq!(c.threads_per_group, 256);
        assert_eq!(c.elements_per_thread, 4);
        assert!(c.use_simd);
    }

    #[test]
    fn swiglu_tuned_config_default() {
        let c = SwiGLUTunedConfig::default();
        assert_eq!(c.threads_per_token, 256);
        assert_eq!(c.chunk_size, 2048);
    }

    #[test]
    fn cross_entropy_tuned_config_default() {
        let c = CrossEntropyTunedConfig::default();
        assert_eq!(c.threadgroup_size, 256);
        assert_eq!(c.chunk_size, 4096);
    }

    #[test]
    fn norm_lora_tuned_config_default() {
        let c = NormLoraTunedConfig::default();
        assert_eq!(c.threads_per_token, 256);
        assert!(!c.use_tiled);
    }

    #[test]
    fn mpp_gemm_tuned_config_default() {
        let c = MppGemmTunedConfig::default();
        assert_eq!(c.variant, MppGemmKernelVariant::Sg4_64x64);
        assert!(c.use_morton);
    }

    #[test]
    fn flash_attention_tuned_config_default() {
        let c = FlashAttentionTunedConfig::default();
        assert_eq!(c.block_q, 32);
        assert_eq!(c.block_k, 32);
    }

    // -------------------------------------------------------------------------
    // Serde round-trips
    // -------------------------------------------------------------------------

    #[test]
    fn tuned_config_serde_roundtrip() {
        let config = TunedConfig {
            tile_m: 64,
            tile_n: 32,
            tile_k: 16,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: TunedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, decoded);
    }

    #[test]
    fn swiglu_tuned_config_serde_roundtrip() {
        let config = SwiGLUTunedConfig {
            threads_per_token: 512,
            chunk_size: 4096,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: SwiGLUTunedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, decoded);
    }

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

    #[test]
    fn flash_attention_tuned_config_serde_roundtrip() {
        let config = FlashAttentionTunedConfig {
            block_q: 64,
            block_k: 32,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: FlashAttentionTunedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, decoded);
    }

    #[test]
    fn norm_lora_tuned_config_serde_roundtrip() {
        let config = NormLoraTunedConfig {
            threads_per_token: 512,
            use_tiled: true,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: NormLoraTunedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, decoded);
    }

    #[test]
    fn mpp_gemm_tuned_config_serde_roundtrip() {
        let config = MppGemmTunedConfig {
            variant: MppGemmKernelVariant::Sg2_32x64,
            use_morton: false,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: MppGemmTunedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, decoded);
    }

    #[test]
    fn mpp_gemm_tuned_config_deserializes_legacy_cache_entries() {
        let decoded: MppGemmTunedConfig = serde_json::from_str(r#"{"use_morton":false}"#).unwrap();
        assert_eq!(decoded.variant, MppGemmKernelVariant::Sg4_64x64);
        assert!(!decoded.use_morton);
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
    fn lora_forward_cache_key_includes_device_identity() {
        let base = test_device_properties("Apple M4", DeviceTier::Base, 16, 10);
        let max = test_device_properties("Apple M4 Max", DeviceTier::Max, 16, 40);

        let base_key = lora_forward_cache_key(&base, 64, 1024, 1024, 16);
        let max_key = lora_forward_cache_key(&max, 64, 1024, 1024, 16);

        assert_ne!(base_key, max_key);
        assert!(base_key.contains("Apple M4:base:16:10"));
        assert!(max_key.contains("Apple M4 Max:max:16:40"));
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

    const SWIGLU_KEY: &str = "swiglu:4:2048:8192";
    const SWIGLU_FAST: SwiGLUTunedConfig = SwiGLUTunedConfig {
        threads_per_token: 512,
        chunk_size: 4096,
    };
    const SWIGLU_SLOW: SwiGLUTunedConfig = SwiGLUTunedConfig {
        threads_per_token: 128,
        chunk_size: 1024,
    };

    fn swiglu_times(config: SwiGLUTunedConfig) -> Result<f64> {
        Ok(if config == SWIGLU_FAST { 0.5 } else { 1.0 })
    }

    /// A measured winner is written, and a second process reads it back
    /// without measuring anything.
    #[test]
    fn a_measured_choice_survives_a_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");

        let first = Tuner::with_cache_dir(dir.clone());
        let chosen = first.select(
            &first.swiglu,
            "SwiGLU",
            SWIGLU_KEY,
            || (SWIGLU_SLOW, vec![SWIGLU_SLOW, SWIGLU_FAST]),
            swiglu_times,
        );
        assert_eq!(chosen, SWIGLU_FAST);
        assert!(dir.join("swiglu.json").exists());

        let second = Tuner::with_cache_dir(dir);
        let reread = second.select(
            &second.swiglu,
            "SwiGLU",
            SWIGLU_KEY,
            || panic!("a cached shape is not planned again"),
            |_| panic!("a cached shape is not measured again"),
        );
        assert_eq!(reread, SWIGLU_FAST);
    }

    /// With nothing measurable the heuristic is used for this process, but
    /// the file is not written: a later run on a quiet GPU measures again.
    #[test]
    fn an_unmeasured_choice_is_not_written_to_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");

        let tuner = Tuner::with_cache_dir(dir.clone());
        let chosen = tuner.select(
            &tuner.swiglu,
            "SwiGLU",
            SWIGLU_KEY,
            || (SWIGLU_SLOW, vec![SWIGLU_SLOW, SWIGLU_FAST]),
            |_| Err(MetalError::ExecutionFailed("busy".into())),
        );
        assert_eq!(chosen, SWIGLU_SLOW);
        assert_eq!(tuner.swiglu.get(SWIGLU_KEY), Some(SWIGLU_SLOW));
        assert!(!dir.join("swiglu.json").exists());
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
            &a.swiglu,
            "SwiGLU",
            "swiglu:shape-a",
            || (SWIGLU_SLOW, vec![SWIGLU_FAST]),
            swiglu_times,
        );
        b.select(
            &b.swiglu,
            "SwiGLU",
            "swiglu:shape-b",
            || (SWIGLU_SLOW, vec![SWIGLU_FAST]),
            swiglu_times,
        );

        let fresh = Tuner::with_cache_dir(dir);
        assert_eq!(fresh.swiglu.get("swiglu:shape-a"), Some(SWIGLU_FAST));
        assert_eq!(fresh.swiglu.get("swiglu:shape-b"), Some(SWIGLU_FAST));
    }

    #[test]
    fn writing_leaves_no_temporary_files_behind() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");
        let tuner = Tuner::with_cache_dir(dir.clone());
        for shape in 0..5 {
            tuner.select(
                &tuner.swiglu,
                "SwiGLU",
                &format!("swiglu:{shape}"),
                || (SWIGLU_SLOW, vec![SWIGLU_FAST]),
                swiglu_times,
            );
        }
        let names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["swiglu.json"]);
    }

    /// A damaged cache file is ignored, never fatal.
    #[test]
    fn a_corrupt_cache_file_is_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("swiglu.json"), "{ not json").unwrap();

        let tuner = Tuner::with_cache_dir(dir);
        assert_eq!(tuner.swiglu.get(SWIGLU_KEY), None);
        let chosen = tuner.select(
            &tuner.swiglu,
            "SwiGLU",
            SWIGLU_KEY,
            || (SWIGLU_SLOW, vec![SWIGLU_FAST]),
            swiglu_times,
        );
        assert_eq!(chosen, SWIGLU_FAST);
    }

    /// The on-disk format is a flat JSON object of key to configuration, as
    /// earlier versions wrote it.
    #[test]
    fn the_cache_file_format_is_a_flat_map() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("scope");
        fs::create_dir_all(&dir).unwrap();

        let key = "mpp_gemm:Apple M5 Max:max:128:512:256:2:f16:acc";
        let config = MppGemmTunedConfig {
            variant: MppGemmKernelVariant::Sg4_64x64,
            use_morton: true,
        };
        let map: HashMap<&str, MppGemmTunedConfig> = [(key, config)].into_iter().collect();
        fs::write(
            dir.join("mpp_gemm.json"),
            serde_json::to_string_pretty(&map).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("flash_attention.json"),
            r#"{"flash_attention:k": {"block_q": 64}}"#,
        )
        .unwrap();

        let tuner = Tuner::with_cache_dir(dir);
        assert_eq!(tuner.mpp_gemm.get(key), Some(config));
        // Fields an older file lacks take their defaults.
        assert_eq!(
            tuner.flash_attention.get("flash_attention:k"),
            Some(FlashAttentionTunedConfig {
                block_q: 64,
                block_k: 32
            })
        );
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
            let file = dir.join("swiglu.json");
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

    #[test]
    fn flash_attention_heuristic_tracks_head_dim_and_tier() {
        let tuner = Tuner::new();
        assert_eq!(
            tuner.heuristic_flash_attention_config(64, DeviceTier::Base),
            FlashAttentionTunedConfig {
                block_q: 64,
                block_k: 32,
            }
        );
        assert_eq!(
            tuner.heuristic_flash_attention_config(128, DeviceTier::Pro),
            FlashAttentionTunedConfig {
                block_q: 32,
                block_k: 32,
            }
        );
        assert_eq!(
            tuner.heuristic_flash_attention_config(128, DeviceTier::Ultra),
            FlashAttentionTunedConfig {
                block_q: 64,
                block_k: 32,
            }
        );
        assert_eq!(
            tuner.heuristic_flash_attention_config(256, DeviceTier::Max),
            FlashAttentionTunedConfig {
                block_q: 32,
                block_k: 16,
            }
        );
    }

    #[test]
    fn flash_attention_candidates_cover_supported_configs_without_duplicates() {
        let tuner = Tuner::new();
        let candidates = tuner.candidate_flash_attention_configs(128, DeviceTier::Max);
        assert_eq!(
            candidates,
            vec![
                FlashAttentionTunedConfig {
                    block_q: 64,
                    block_k: 32,
                },
                FlashAttentionTunedConfig {
                    block_q: 32,
                    block_k: 32,
                },
            ]
        );

        let candidates = tuner.candidate_flash_attention_configs(64, DeviceTier::Base);
        assert_eq!(
            candidates,
            vec![
                FlashAttentionTunedConfig {
                    block_q: 64,
                    block_k: 32,
                },
                FlashAttentionTunedConfig {
                    block_q: 64,
                    block_k: 64,
                },
                FlashAttentionTunedConfig {
                    block_q: 32,
                    block_k: 32,
                },
            ]
        );
    }

    #[test]
    fn flash_attention_decode_tune_request_buckets_single_token_kv_lengths() {
        let mut config = FlashAttentionConfig::inference(1, 8, 2, 1, 256);
        config.kv_seq_len = 1025;
        let request = FlashAttentionTuneRequest::from_config(&config);
        assert_eq!(request.query_seq_len, 1);
        assert_eq!(request.kv_seq_len, 1152);

        config.kv_seq_len = 1032;
        let request = FlashAttentionTuneRequest::from_config(&config);
        assert_eq!(request.kv_seq_len, 1152);
    }

    #[test]
    fn flash_attention_tune_request_keeps_exact_lengths_outside_decode_case() {
        let training = FlashAttentionConfig {
            query_seq_len: 1,
            kv_seq_len: 1025,
            is_training: true,
            ..FlashAttentionConfig::inference(1, 8, 2, 1, 256)
        };
        let training_request = FlashAttentionTuneRequest::from_config(&training);
        assert_eq!(training_request.kv_seq_len, 1025);

        let prefill = FlashAttentionConfig::inference(1, 8, 2, 64, 256);
        let prefill_request = FlashAttentionTuneRequest::from_config(&prefill);
        assert_eq!(prefill_request.query_seq_len, 64);
        assert_eq!(prefill_request.kv_seq_len, 64);

        let decode_with_softcap = FlashAttentionConfig {
            query_seq_len: 1,
            kv_seq_len: 1025,
            softcap: Some(30.0),
            ..FlashAttentionConfig::inference(1, 8, 2, 1, 256)
        };
        let softcap_request = FlashAttentionTuneRequest::from_config(&decode_with_softcap);
        assert_eq!(softcap_request.kv_seq_len, 1025);
    }

    #[test]
    fn mpp_gemm_heuristic_prefers_linear_for_small_single_batch() {
        let tuner = Tuner::new();
        let config = tuner.heuristic_mpp_gemm_config(
            MppGemmTuneRequest {
                m: 16,
                n: 256,
                k: 128,
                batch_size: 1,
                use_fp16: true,
                accumulate: false,
            },
            DeviceTier::Base,
        );
        assert_eq!(config.variant, MppGemmKernelVariant::Sg1_32x32);
        assert!(!config.use_morton);
    }

    #[test]
    fn mpp_gemm_heuristic_prefers_morton_for_accumulate_or_batched_work() {
        let tuner = Tuner::new();
        assert!(
            tuner
                .heuristic_mpp_gemm_config(
                    MppGemmTuneRequest {
                        m: 16,
                        n: 256,
                        k: 128,
                        batch_size: 2,
                        use_fp16: true,
                        accumulate: false,
                    },
                    DeviceTier::Pro,
                )
                .use_morton
        );
        assert!(
            tuner
                .heuristic_mpp_gemm_config(
                    MppGemmTuneRequest {
                        m: 16,
                        n: 256,
                        k: 128,
                        batch_size: 1,
                        use_fp16: true,
                        accumulate: true,
                    },
                    DeviceTier::Pro,
                )
                .use_morton
        );
        assert!(
            tuner
                .heuristic_mpp_gemm_config(
                    MppGemmTuneRequest {
                        m: 64,
                        n: 256,
                        k: 128,
                        batch_size: 1,
                        use_fp16: true,
                        accumulate: false,
                    },
                    DeviceTier::Max,
                )
                .use_morton
        );
    }

    #[test]
    fn mpp_gemm_heuristic_variant_tracks_shape_and_tier() {
        let tuner = Tuner::new();
        let wide = MppGemmTuneRequest {
            m: 64,
            n: 256,
            k: 128,
            batch_size: 1,
            use_fp16: true,
            accumulate: false,
        };
        let tall = MppGemmTuneRequest {
            m: 256,
            n: 64,
            k: 128,
            batch_size: 1,
            use_fp16: true,
            accumulate: false,
        };
        let balanced = MppGemmTuneRequest {
            m: 128,
            n: 128,
            k: 128,
            batch_size: 1,
            use_fp16: true,
            accumulate: false,
        };

        assert_eq!(
            tuner.heuristic_mpp_gemm_variant(wide, DeviceTier::Base),
            MppGemmKernelVariant::Sg2_32x64
        );
        assert_eq!(
            tuner.heuristic_mpp_gemm_variant(tall, DeviceTier::Base),
            MppGemmKernelVariant::Sg2_64x32
        );
        assert_eq!(
            tuner.heuristic_mpp_gemm_variant(balanced, DeviceTier::Pro),
            MppGemmKernelVariant::Sg4_64x64
        );
    }

    #[test]
    fn mpp_gemm_candidates_cover_supported_variants_without_duplicates() {
        let tuner = Tuner::new();
        let request = MppGemmTuneRequest {
            m: 128,
            n: 128,
            k: 128,
            batch_size: 1,
            use_fp16: true,
            accumulate: false,
        };
        let variants = tuner.candidate_mpp_gemm_variants(request, DeviceTier::Pro);
        assert_eq!(
            variants,
            vec![
                MppGemmKernelVariant::Sg4_64x64,
                MppGemmKernelVariant::Sg1_32x32,
                MppGemmKernelVariant::Sg2_64x32,
                MppGemmKernelVariant::Sg2_32x64,
            ]
        );

        let configs = tuner.candidate_mpp_gemm_configs(request, DeviceTier::Pro);
        assert_eq!(configs.len(), 8);
        assert_eq!(
            configs.first(),
            Some(&MppGemmTunedConfig {
                variant: MppGemmKernelVariant::Sg4_64x64,
                use_morton: true,
            })
        );
    }

    // -------------------------------------------------------------------------
    // Standard-Metal tuning sanity checks
    // -------------------------------------------------------------------------

    #[test]
    fn norm_lora_heuristic_tracks_out_features_and_tier() {
        let tuner = Tuner::new();
        assert_eq!(
            tuner.heuristic_norm_lora_config(DeviceTier::Base, 128, 1024),
            NormLoraTunedConfig {
                threads_per_token: 128,
                use_tiled: false,
            }
        );
        assert_eq!(
            tuner.heuristic_norm_lora_config(DeviceTier::Pro, 512, 1024),
            NormLoraTunedConfig {
                threads_per_token: 256,
                use_tiled: true,
            }
        );
        assert_eq!(
            tuner.heuristic_norm_lora_config(DeviceTier::Max, 1024, 1024),
            NormLoraTunedConfig {
                threads_per_token: 512,
                use_tiled: true,
            }
        );
    }

    #[test]
    fn norm_lora_candidates_cover_thread_and_tiled_options() {
        let tuner = Tuner::new();
        let candidates = tuner.candidate_norm_lora_configs(DeviceTier::Pro, 512, 1024);
        assert!(candidates.contains(&NormLoraTunedConfig {
            threads_per_token: 256,
            use_tiled: true,
        }));
        assert!(candidates.contains(&NormLoraTunedConfig {
            threads_per_token: 256,
            use_tiled: false,
        }));
        assert!(candidates.contains(&NormLoraTunedConfig {
            threads_per_token: 128,
            use_tiled: true,
        }));
        assert!(candidates.contains(&NormLoraTunedConfig {
            threads_per_token: 512,
            use_tiled: true,
        }));
    }

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
    fn swiglu_heuristic_tracks_size_and_tier() {
        let tuner = Tuner::new();
        assert_eq!(
            tuner.heuristic_swiglu_config(DeviceTier::Base, 512, 1024),
            SwiGLUTunedConfig {
                threads_per_token: 128,
                chunk_size: 1024,
            }
        );
        assert_eq!(
            tuner.heuristic_swiglu_config(DeviceTier::Base, 8192, 1024),
            SwiGLUTunedConfig {
                threads_per_token: 256,
                chunk_size: 2048,
            }
        );
        assert_eq!(
            tuner.heuristic_swiglu_config(DeviceTier::Max, 8192, 1024),
            SwiGLUTunedConfig {
                threads_per_token: 512,
                chunk_size: 4096,
            }
        );
    }

    #[test]
    fn swiglu_candidates_cover_thread_and_chunk_options() {
        let tuner = Tuner::new();
        let candidates = tuner.candidate_swiglu_configs(DeviceTier::Pro, 8192, 1024);
        assert!(candidates.contains(&SwiGLUTunedConfig {
            threads_per_token: 256,
            chunk_size: 4096,
        }));
        assert!(candidates.contains(&SwiGLUTunedConfig {
            threads_per_token: 128,
            chunk_size: 4096,
        }));
        assert!(candidates.contains(&SwiGLUTunedConfig {
            threads_per_token: 512,
            chunk_size: 4096,
        }));
        assert!(candidates.contains(&SwiGLUTunedConfig {
            threads_per_token: 256,
            chunk_size: 2048,
        }));
    }

    /// Verify in-memory caching avoids re-computation (second call returns
    /// the exact same struct without hitting the heuristic again).
    #[test]
    #[cfg(target_os = "macos")]
    fn swiglu_cache_hit_on_second_call() {
        let ctx = Arc::new(MetalContext::new().expect("Metal required"));
        let tuner = Tuner::new();

        let first = tuner.tune_swiglu(&ctx, 4, 2048, 8192).unwrap();
        let second = tuner.tune_swiglu(&ctx, 4, 2048, 8192).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn norm_lora_cache_hit_on_second_call() {
        let ctx = Arc::new(MetalContext::new().expect("Metal required"));
        let tuner = Tuner::new();

        let first = tuner.tune_norm_lora(&ctx, 4, 2048, 2048, 16).unwrap();
        let second = tuner.tune_norm_lora(&ctx, 4, 2048, 2048, 16).unwrap();
        assert_eq!(first, second);
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
