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
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};
use serde::{Deserialize, Serialize};

use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::{DeviceProperties, DeviceTier, MetalContext};
use crate::error::{MetalError, Result};
use crate::kernels::flash_attention::{FlashAttention, FlashAttentionConfig};
use crate::kernels::fused_cross_entropy::{
    FusedLinearCrossEntropy, FusedLinearCrossEntropyConfig,
};
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

impl FlashAttentionTuneRequest {
    fn from_config(config: &FlashAttentionConfig) -> Self {
        Self {
            batch_size: config.batch_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            query_seq_len: config.query_seq_len,
            kv_seq_len: config.kv_seq_len,
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
            if self.has_sliding_window { "window" } else { "full" },
            if self.has_softcap { "softcap" } else { "plain" },
            if self.is_training { "train" } else { "infer" }
        )
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

fn dedupe_swiglu_configs(configs: Vec<SwiGLUTunedConfig>) -> Vec<SwiGLUTunedConfig> {
    let mut unique = Vec::with_capacity(configs.len());
    for config in configs {
        if !unique.contains(&config) {
            unique.push(config);
        }
    }
    unique
}

fn dedupe_norm_lora_configs(configs: Vec<NormLoraTunedConfig>) -> Vec<NormLoraTunedConfig> {
    let mut unique = Vec::with_capacity(configs.len());
    for config in configs {
        if !unique.contains(&config) {
            unique.push(config);
        }
    }
    unique
}

fn dedupe_cross_entropy_configs(
    configs: Vec<CrossEntropyTunedConfig>,
) -> Vec<CrossEntropyTunedConfig> {
    let mut unique = Vec::with_capacity(configs.len());
    for config in configs {
        if !unique.contains(&config) {
            unique.push(config);
        }
    }
    unique
}

// ============================================================================
// Disk cache free helpers (outside Tuner so borrowck is happy)
// ============================================================================

/// Read a JSON file at `path` and merge its entries into `cache`.
///
/// Errors are logged and silently ignored.
fn load_disk_cache_file<T>(path: &PathBuf, cache: &Mutex<HashMap<String, T>>)
where
    T: for<'de> Deserialize<'de>,
{
    if !path.exists() {
        debug!("Disk cache file not found, skipping: {}", path.display());
        return;
    }

    match fs::read_to_string(path) {
        Err(e) => warn!("Failed to read disk cache {}: {}", path.display(), e),
        Ok(contents) => match serde_json::from_str::<HashMap<String, T>>(&contents) {
            Err(e) => warn!("Failed to parse disk cache {}: {}", path.display(), e),
            Ok(map) => {
                let entries = map.len();
                match cache.lock() {
                    Err(e) => warn!("Mutex poisoned reading disk cache: {}", e),
                    Ok(mut guard) => {
                        guard.extend(map);
                        debug!(
                            "Loaded {} entries from disk cache: {}",
                            entries,
                            path.display()
                        );
                    }
                }
            }
        },
    }
}

fn cache_device_tier_label(tier: DeviceTier) -> &'static str {
    match tier {
        DeviceTier::Base => "base",
        DeviceTier::Pro => "pro",
        DeviceTier::Max => "max",
        DeviceTier::Ultra => "ultra",
    }
}

fn device_cache_identity(props: &DeviceProperties) -> String {
    format!(
        "{}:{}:{}:{}",
        props.name,
        cache_device_tier_label(props.device_tier),
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
pub struct Tuner {
    /// Cache of best configurations for matrix ops.
    /// Key: "kernel_name:M:N:K" (problem size hash)
    cache: Mutex<HashMap<String, TunedConfig>>,

    /// Cache of best configurations for merge ops.
    /// Key: "merge:device_name:device_tier:architecture_gen:gpu_core_count:num_elements:num_models"
    merge_cache: Mutex<HashMap<String, MergeTunedConfig>>,

    /// Cache of best configurations for SwiGLU kernels.
    /// Key: "swiglu:batch:hidden:intermediate"
    swiglu_cache: Mutex<HashMap<String, SwiGLUTunedConfig>>,

    /// Cache of best configurations for cross-entropy kernels.
    /// Keys cover both generic logits CE and fused-linear CE shapes.
    cross_entropy_cache: Mutex<HashMap<String, CrossEntropyTunedConfig>>,

    /// Cache of best configurations for FlashAttention block sizes.
    /// Key: "flash_attention:device_name:device_tier:..."
    flash_attention_cache: Mutex<HashMap<String, FlashAttentionTunedConfig>>,

    /// Cache of best configurations for Norm+LoRA fused kernels.
    /// Key: "norm_lora:device_name:device_tier:batch:hidden:out_features:rank"
    norm_lora_cache: Mutex<HashMap<String, NormLoraTunedConfig>>,

    /// Cache of best configurations for MPP GEMM dispatch.
    /// Key: "mpp_gemm:device_name:device_tier:m:n:k:batch:dtype:accumulate"
    mpp_gemm_cache: Mutex<HashMap<String, MppGemmTunedConfig>>,

    /// Path for persistent cache storage (`~/.cache/pmetal/tuna/` by default).
    /// `None` means no disk persistence (in-memory only).
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
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: None,
        }
    }

    /// Create a new Tuner with a persistent disk cache at `~/.cache/pmetal/tuna/`.
    ///
    /// Any previously-tuned configurations are loaded from disk immediately.
    /// New results are written through to disk as they are produced.
    pub fn with_persistent_cache() -> Self {
        let cache_dir = dirs::cache_dir().map(|p| p.join("pmetal").join("tuna"));

        let mut tuner = Self {
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: cache_dir.clone(),
        };

        if cache_dir.is_some() {
            tuner.load_disk_cache();
        }

        tuner
    }

    // -------------------------------------------------------------------------
    // Disk cache helpers
    // -------------------------------------------------------------------------

    /// Load all kernel caches from disk into the in-memory maps.
    ///
    /// Errors are logged and silently ignored so that a corrupt/missing cache
    /// file never prevents the tuner from running.
    fn load_disk_cache(&mut self) {
        // Clone the dir so we don't hold a borrow on self while accessing fields.
        let dir = match &self.cache_dir {
            Some(d) => d.clone(),
            None => return,
        };

        load_disk_cache_file::<TunedConfig>(&dir.join("lora_forward.json"), &self.cache);
        load_disk_cache_file::<MergeTunedConfig>(&dir.join("merge.json"), &self.merge_cache);
        load_disk_cache_file::<SwiGLUTunedConfig>(&dir.join("swiglu.json"), &self.swiglu_cache);
        load_disk_cache_file::<CrossEntropyTunedConfig>(
            &dir.join("cross_entropy.json"),
            &self.cross_entropy_cache,
        );
        load_disk_cache_file::<FlashAttentionTunedConfig>(
            &dir.join("flash_attention.json"),
            &self.flash_attention_cache,
        );
        load_disk_cache_file::<NormLoraTunedConfig>(
            &dir.join("norm_lora.json"),
            &self.norm_lora_cache,
        );
        load_disk_cache_file::<MppGemmTunedConfig>(
            &dir.join("mpp_gemm.json"),
            &self.mpp_gemm_cache,
        );
    }

    /// Ensure the cache directory exists, creating it if necessary.
    fn ensure_cache_dir(&self) -> Option<&PathBuf> {
        let dir = self.cache_dir.as_ref()?;
        if !dir.exists() {
            if let Err(e) = fs::create_dir_all(dir) {
                warn!("Failed to create cache directory {}: {}", dir.display(), e);
                return None;
            }
            debug!("Created cache directory: {}", dir.display());
        }
        Some(dir)
    }

    /// Persist the full lora_forward in-memory cache to `lora_forward.json`.
    fn save_to_disk(&self, key: &str, config: &TunedConfig) {
        let Some(dir) = self.ensure_cache_dir() else {
            return;
        };
        let path = dir.join("lora_forward.json");
        self.flush_cache_file(&path, &self.cache, key, config);
    }

    /// Persist the full swiglu in-memory cache to `swiglu.json`.
    fn save_swiglu_to_disk(&self, key: &str, config: &SwiGLUTunedConfig) {
        let Some(dir) = self.ensure_cache_dir() else {
            return;
        };
        let path = dir.join("swiglu.json");
        self.flush_cache_file(&path, &self.swiglu_cache, key, config);
    }

    /// Persist the full merge in-memory cache to `merge.json`.
    fn save_merge_to_disk(&self, key: &str, config: &MergeTunedConfig) {
        let Some(dir) = self.ensure_cache_dir() else {
            return;
        };
        let path = dir.join("merge.json");
        self.flush_cache_file(&path, &self.merge_cache, key, config);
    }

    /// Persist the full cross-entropy in-memory cache to `cross_entropy.json`.
    fn save_cross_entropy_to_disk(&self, key: &str, config: &CrossEntropyTunedConfig) {
        let Some(dir) = self.ensure_cache_dir() else {
            return;
        };
        let path = dir.join("cross_entropy.json");
        self.flush_cache_file(&path, &self.cross_entropy_cache, key, config);
    }

    /// Persist the full flash_attention in-memory cache to `flash_attention.json`.
    fn save_flash_attention_to_disk(&self, key: &str, config: &FlashAttentionTunedConfig) {
        let Some(dir) = self.ensure_cache_dir() else {
            return;
        };
        let path = dir.join("flash_attention.json");
        self.flush_cache_file(&path, &self.flash_attention_cache, key, config);
    }

    /// Persist the full norm-lora in-memory cache to `norm_lora.json`.
    fn save_norm_lora_to_disk(&self, key: &str, config: &NormLoraTunedConfig) {
        let Some(dir) = self.ensure_cache_dir() else {
            return;
        };
        let path = dir.join("norm_lora.json");
        self.flush_cache_file(&path, &self.norm_lora_cache, key, config);
    }

    /// Persist the full MPP GEMM in-memory cache to `mpp_gemm.json`.
    fn save_mpp_gemm_to_disk(&self, key: &str, config: &MppGemmTunedConfig) {
        let Some(dir) = self.ensure_cache_dir() else {
            return;
        };
        let path = dir.join("mpp_gemm.json");
        self.flush_cache_file(&path, &self.mpp_gemm_cache, key, config);
    }

    /// Generic helper: insert `key`→`config` into the mutex-guarded map, then
    /// atomically write the entire map to `path` as JSON.
    ///
    /// Writing the full map (not just appending) keeps the file valid JSON even
    /// if a previous write was interrupted.
    fn flush_cache_file<T>(
        &self,
        path: &PathBuf,
        cache: &Mutex<HashMap<String, T>>,
        key: &str,
        config: &T,
    ) where
        T: Serialize + Clone,
    {
        // Insert the new entry first.
        let snapshot: HashMap<String, T> = match cache.lock() {
            Err(e) => {
                warn!("Mutex poisoned writing disk cache: {}", e);
                return;
            }
            Ok(mut guard) => {
                guard.insert(key.to_string(), config.clone());
                guard.clone()
            }
        };

        // Serialize and write atomically via a temp file.
        let tmp_path = path.with_extension("json.tmp");
        match serde_json::to_string_pretty(&snapshot) {
            Err(e) => warn!("Failed to serialize cache for {}: {}", path.display(), e),
            Ok(json) => {
                if let Err(e) = fs::write(&tmp_path, &json) {
                    warn!(
                        "Failed to write tmp cache file {}: {}",
                        tmp_path.display(),
                        e
                    );
                    return;
                }
                if let Err(e) = fs::rename(&tmp_path, path) {
                    warn!("Failed to rename cache file {}: {}", path.display(), e);
                    // Clean up the orphaned tmp file if possible.
                    let _ = fs::remove_file(&tmp_path);
                    return;
                }
                debug!(
                    "Saved {} entries to disk cache: {}",
                    snapshot.len(),
                    path.display()
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Public cache accessors (merge)
    // -------------------------------------------------------------------------

    /// Get a cached merge configuration if available.
    pub fn get_merge_config(&self, key: &str) -> Option<MergeTunedConfig> {
        let cache = self
            .merge_cache
            .lock()
            .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))
            .ok()?;
        cache.get(key).copied()
    }

    /// Store a merge configuration in the cache.
    pub fn set_merge_config(&self, key: String, config: MergeTunedConfig) {
        match self
            .merge_cache
            .lock()
            .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))
        {
            Ok(mut cache) => {
                cache.insert(key, config);
            }
            Err(e) => {
                tracing::error!("Failed to acquire merge_cache lock: {}", e);
            }
        }
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

        {
            let cache = self
                .flash_attention_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            if let Some(&tuned) = cache.get(&key) {
                return Ok(tuned);
            }
        }

        let candidates = self.candidate_flash_attention_configs(request.head_dim, props.device_tier);
        let mut best_config = self.heuristic_flash_attention_config(request.head_dim, props.device_tier);
        let mut best_time = f64::INFINITY;

        info!(
            "Tuning FlashAttention blocks for [B={}, H={}, KVH={}, Q={}, KV={}, D={}, mode={}]...",
            request.batch_size,
            request.num_heads,
            request.num_kv_heads,
            request.query_seq_len,
            request.kv_seq_len,
            request.head_dim,
            if request.is_training { "train" } else { "infer" }
        );

        for candidate in candidates {
            match self.benchmark_flash_attention(context, config, candidate) {
                Ok(elapsed) => {
                    debug!(
                        "FlashAttention config {:?} took {:.3} ms",
                        candidate,
                        elapsed * 1000.0
                    );
                    if elapsed < best_time {
                        best_time = elapsed;
                        best_config = candidate;
                    }
                }
                Err(error) => {
                    debug!(
                        "FlashAttention config {:?} failed benchmarking: {}",
                        candidate,
                        error
                    );
                }
            }
        }

        if best_time.is_finite() {
            info!(
                "Selected FlashAttention block config {:?} ({:.3} ms)",
                best_config,
                best_time * 1000.0
            );
        } else {
            debug!(
                "Falling back to heuristic FlashAttention block config {:?}",
                best_config
            );
        }

        {
            let mut cache = self
                .flash_attention_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            cache.insert(key.clone(), best_config);
        }
        self.save_flash_attention_to_disk(&key, &best_config);

        Ok(best_config)
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
        let heuristic =
            self.heuristic_flash_attention_config(config.head_dim, context.properties().device_tier);
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
                MetalError::InvalidConfig(
                    "FlashAttention benchmark size overflow".to_string(),
                )
            })?;
        if total_bytes > 256 * 1024 * 1024 {
            return Ok(if candidate == heuristic { 1.0 } else { 10.0 });
        }

        let queries = MetalBuffer::<f16>::zeros(context, config.query_size(), BufferUsage::Shared)?;
        let keys = MetalBuffer::<f16>::zeros(context, config.kv_size(), BufferUsage::Shared)?;
        let values = MetalBuffer::<f16>::zeros(context, config.kv_size(), BufferUsage::Shared)?;

        let flash = FlashAttention::new_with_tuned_blocks(
            Arc::clone(context),
            config.clone(),
            candidate,
        )?;

        flash.forward(&queries, &keys, &values)?;

        let iterations = 3;
        let start = Instant::now();
        for _ in 0..iterations {
            flash.forward(&queries, &keys, &values)?;
        }

        Ok(start.elapsed().as_secs_f64() / iterations as f64)
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

        {
            let cache = self
                .mpp_gemm_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            if let Some(&config) = cache.get(&key) {
                return Ok(config);
            }
        }

        let mut best_config = self.heuristic_mpp_gemm_config(request, props.device_tier);
        let mut best_time = f64::INFINITY;

        if !props.has_nax() || context.pipeline_cache().metal4_library().is_none() {
            debug!(
                "Skipping MPP GEMM tuning on non-NAX or Metal 4-unavailable device: {}",
                props.name
            );
        } else if !props.should_consider_mpp_gemm(request.m, request.n, request.k, request.use_fp16)
        {
            debug!(
                "Skipping MPP GEMM benchmarking for small problem [M={}, N={}, K={}, B={}]",
                request.m, request.n, request.k, request.batch_size
            );
        } else {
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

            let candidates = self.candidate_mpp_gemm_configs(request, props.device_tier);

            for candidate in candidates {
                match self.benchmark_mpp_gemm(context, candidate, request) {
                    Ok(time) => {
                        debug!(
                            "MPP GEMM config {:?} took {:.3} ms",
                            candidate,
                            time * 1000.0
                        );
                        if time < best_time {
                            best_time = time;
                            best_config = candidate;
                        }
                    }
                    Err(error) => {
                        debug!("MPP GEMM config {:?} failed: {}", candidate, error);
                    }
                }
            }
        }

        {
            let mut cache = self
                .mpp_gemm_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            cache.insert(key.clone(), best_config);
        }
        self.save_mpp_gemm_to_disk(&key, &best_config);

        if best_time.is_finite() {
            info!(
                "Best MPP GEMM config: {:?} ({:.3} ms)",
                best_config,
                best_time * 1000.0
            );
        } else {
            debug!("Selected heuristic MPP GEMM config: {:?}", best_config);
        }

        Ok(best_config)
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

        let iterations = 3;

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
            gemm.execute(&a, &b, &d)?;
            let start = Instant::now();
            for _ in 0..iterations {
                gemm.execute(&a, &b, &d)?;
            }
            Ok(start.elapsed().as_secs_f64() / iterations as f64)
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
            gemm.execute(&a, &b, &d)?;
            let start = Instant::now();
            for _ in 0..iterations {
                gemm.execute(&a, &b, &d)?;
            }
            Ok(start.elapsed().as_secs_f64() / iterations as f64)
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

        // 1. Check in-memory cache (also populated from disk on startup)
        {
            let cache = self
                .cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            if let Some(&config) = cache.get(&key) {
                return Ok(config);
            }
        }

        info!(
            "Tuning LoRA Forward for [B={}, I={}, O={}, R={}]...",
            batch_size, in_features, out_features, rank
        );

        // 2. Generate candidates (filtered by device capabilities)
        let candidates = self.generate_lora_candidates(context);
        debug!(
            "Generated {} valid candidates for device (max threads: {})",
            candidates.len(),
            context.properties().max_threads_per_threadgroup
        );
        let mut best_config = TunedConfig::default();
        let mut best_time = f64::INFINITY;

        // 3. Benchmark candidates
        for config in candidates {
            match self.benchmark_lora_forward(
                context,
                config,
                batch_size,
                in_features,
                out_features,
                rank,
            ) {
                Ok(time) => {
                    debug!("Config {:?} took {:.3} ms", config, time * 1000.0);
                    if time < best_time {
                        best_time = time;
                        best_config = config;
                    }
                }
                Err(e) => {
                    debug!("Config {:?} failed: {}", config, e);
                }
            }
        }

        info!(
            "Best LoRA config: {:?} ({:.3} ms)",
            best_config,
            best_time * 1000.0
        );

        // 4. Update in-memory cache and persist to disk
        {
            let mut cache = self
                .cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            cache.insert(key.clone(), best_config);
        }

        // Write-through to disk (non-fatal if it fails)
        self.save_to_disk(&key, &best_config);

        Ok(best_config)
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

        // Create params buffer
        #[allow(dead_code)]
        struct FusedLoraParams {
            batch_size: u32,
            in_features: u32,
            out_features: u32,
            rank: u32,
            scale: f32,
        }
        let params = FusedLoraParams {
            batch_size: batch_size as u32,
            in_features: in_features as u32,
            out_features: out_features as u32,
            rank: rank as u32,
            scale: 1.0,
        };

        let params_size = std::mem::size_of::<FusedLoraParams>();
        let params_ptr = NonNull::new(&params as *const _ as *mut c_void).unwrap();

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

        // Warmup
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
        )?;

        // Measure
        let start = Instant::now();
        let iterations = 5;
        for _ in 0..iterations {
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
            )?;
        }
        let elapsed = start.elapsed();

        Ok(elapsed.as_secs_f64() / iterations as f64)
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
        let buffer = queue
            .commandBuffer()
            .ok_or(MetalError::CommandQueueCreation)?;
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

        Ok(())
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

        // Check cache
        if let Some(config) = self.get_merge_config(&key) {
            return Ok(config);
        }

        info!(
            "Tuning merge kernel for {} elements, {} models...",
            num_elements, num_models
        );

        // Generate candidates based on device tier
        let candidates = self.generate_merge_candidates(context);
        debug!("Generated {} merge candidates for device", candidates.len());

        let mut best_config = MergeTunedConfig::default();
        let mut best_time = f64::INFINITY;

        // Benchmark each candidate
        for config in candidates {
            match self.benchmark_merge(context, config, num_elements) {
                Ok(time) => {
                    debug!("Merge config {:?} took {:.3} ms", config, time * 1000.0);
                    if time < best_time {
                        best_time = time;
                        best_config = config;
                    }
                }
                Err(e) => {
                    debug!("Merge config {:?} failed: {}", config, e);
                }
            }
        }

        info!(
            "Best merge config: {:?} ({:.3} ms)",
            best_config,
            best_time * 1000.0
        );

        // Cache result
        self.set_merge_config(key.clone(), best_config);
        self.save_merge_to_disk(&key, &best_config);

        Ok(best_config)
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
        #[repr(C)]
        struct MergeConfigParams {
            num_tensors: u32,
            total_elements: u32,
            epsilon: f32,
            _pad: u32,
        }
        let params = MergeConfigParams {
            num_tensors: 1,
            total_elements: num_elements as u32,
            epsilon: 1e-8,
            _pad: 0,
        };

        // Tensor info
        #[repr(C)]
        struct TensorInfoParams {
            offset: u32,
            size: u32,
            density: f32,
            threshold: f32,
        }
        let tensor_info = TensorInfoParams {
            offset: 0,
            size: num_elements as u32,
            density: 0.5,
            threshold: 0.0,
        };

        // Warmup
        self.dispatch_merge_kernel(
            context,
            &pipeline,
            config,
            &buf_input,
            &buf_output,
            &params,
            &tensor_info,
            num_elements,
        )?;

        // Benchmark
        let start = Instant::now();
        let iterations = 5;
        for _ in 0..iterations {
            self.dispatch_merge_kernel(
                context,
                &pipeline,
                config,
                &buf_input,
                &buf_output,
                &params,
                &tensor_info,
                num_elements,
            )?;
        }
        let elapsed = start.elapsed();

        Ok(elapsed.as_secs_f64() / iterations as f64)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_merge_kernel<P, T>(
        &self,
        context: &MetalContext,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        config: MergeTunedConfig,
        input: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        params: &P,
        tensor_info: &T,
        num_elements: usize,
    ) -> Result<()> {
        let queue = context.command_queue();
        let buffer = queue
            .commandBuffer()
            .ok_or(MetalError::CommandQueueCreation)?;
        let encoder = buffer
            .computeCommandEncoder()
            .ok_or(MetalError::CommandQueueCreation)?;

        encoder.setComputePipelineState(pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 1);

            let tensor_info_ptr = NonNull::from(tensor_info).cast();
            encoder.setBytes_length_atIndex(tensor_info_ptr, std::mem::size_of::<T>(), 2);

            let params_ptr = NonNull::from(params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<P>(), 3);
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

        Ok(())
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

        // Check in-memory cache (populated from disk on startup)
        {
            let cache = self
                .swiglu_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            if let Some(&config) = cache.get(&key) {
                return Ok(config);
            }
        }

        debug!(
            "Selecting SwiGLU config for [B={}, H={}, I={}]",
            batch_size, hidden_size, intermediate_size
        );

        let heuristic = self.heuristic_swiglu_config(
            props.device_tier,
            intermediate_size,
            props.max_threads_per_threadgroup as u32,
        );
        let candidates = self.candidate_swiglu_configs(
            props.device_tier,
            intermediate_size,
            props.max_threads_per_threadgroup as u32,
        );
        let config = FusedSwiGLUConfig::new(batch_size, hidden_size, intermediate_size);
        let mut best_config = heuristic;
        let mut best_time = f64::INFINITY;

        info!(
            "Tuning SwiGLU config for [B={}, H={}, I={}]...",
            batch_size, hidden_size, intermediate_size
        );

        for candidate in candidates {
            match self.benchmark_swiglu(context, &config, candidate) {
                Ok(elapsed) => {
                    debug!(
                        "SwiGLU config {:?} took {:.3} ms",
                        candidate,
                        elapsed * 1000.0
                    );
                    if elapsed < best_time {
                        best_time = elapsed;
                        best_config = candidate;
                    }
                }
                Err(error) => {
                    debug!("SwiGLU config {:?} failed benchmarking: {}", candidate, error);
                }
            }
        }

        if best_time.is_finite() {
            info!(
                "Selected SwiGLU config: {:?} ({:.3} ms)",
                best_config,
                best_time * 1000.0
            );
        } else {
            debug!(
                "Falling back to heuristic SwiGLU config {:?} for device tier {:?}",
                best_config,
                props.device_tier
            );
        }

        // Update in-memory cache
        {
            let mut cache = self
                .swiglu_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            cache.insert(key.clone(), best_config);
        }

        // Write-through to disk
        self.save_swiglu_to_disk(&key, &best_config);

        Ok(best_config)
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
                    if small_intermediate { 2048 } else { 4096 }
                }
                DeviceTier::Pro => {
                    if small_intermediate { 2048 } else { 4096 }
                }
                DeviceTier::Base => {
                    if small_intermediate { 1024 } else { 2048 }
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
        let heuristic =
            self.heuristic_swiglu_config(device_tier, intermediate_size, max_threads_per_threadgroup);
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

        dedupe_swiglu_configs(candidates)
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
            .ok_or_else(|| MetalError::InvalidConfig("SwiGLU benchmark size overflow".to_string()))?;
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
        kernel.forward(&input, &gate_weight, &up_weight, &down_weight)?;

        let iterations = 3;
        let start = Instant::now();
        for _ in 0..iterations {
            kernel.forward(&input, &gate_weight, &up_weight, &down_weight)?;
        }

        Ok(start.elapsed().as_secs_f64() / iterations as f64)
    }

    // =========================================================================
    // Cross-Entropy Kernel Tuning (heuristic)
    // =========================================================================

    /// Tune the cross-entropy loss kernel for the given problem size.
    ///
    /// Cross-entropy over large vocabularies is dominated by the softmax
    /// reduction. The threadgroup size controls parallelism of that reduction
    /// sweep; larger threadgroups amortize overhead for wide vocabularies but
    /// require more threadgroup memory on tighter devices.
    ///
    /// # Arguments
    /// * `context` - Metal context
    /// * `num_tokens` - Number of tokens in the batch
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Heuristically-selected optimal configuration for cross-entropy on this hardware.
    pub fn tune_cross_entropy(
        &self,
        context: &MetalContext,
        num_tokens: usize,
        vocab_size: usize,
    ) -> Result<CrossEntropyTunedConfig> {
        let key = format!("cross_entropy:{}:{}", num_tokens, vocab_size);

        // Check in-memory cache
        {
            let cache = self
                .cross_entropy_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            if let Some(&config) = cache.get(&key) {
                return Ok(config);
            }
        }

        debug!(
            "Selecting cross-entropy config for [T={}, V={}]",
            num_tokens, vocab_size
        );

        let config = self.select_cross_entropy_config(context, vocab_size);

        info!(
            "Selected cross-entropy config: {:?} (heuristic, vocab={}, device tier: {:?})",
            config,
            vocab_size,
            context.properties().device_tier
        );

        // Update in-memory cache
        {
            let mut cache = self
                .cross_entropy_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            cache.insert(key.clone(), config);
        }

        // Write-through to disk
        self.save_cross_entropy_to_disk(&key, &config);

        Ok(config)
    }

    /// Select the best cross-entropy config using heuristics.
    ///
    /// Decision rationale:
    /// - `threadgroup_size`: Larger vocabularies need more parallel threads to
    ///   sweep the reduction without serialising too much. Cap at the device
    ///   max_threads_per_threadgroup.
    /// - `chunk_size`: Tile the vocabulary to fit output logits into cache.
    ///   High-bandwidth devices have larger effective caches so larger tiles pay off.
    fn select_cross_entropy_config(
        &self,
        context: &MetalContext,
        vocab_size: usize,
    ) -> CrossEntropyTunedConfig {
        use crate::context::DeviceTier;

        let props = context.properties();
        let max_threads = props.max_threads_per_threadgroup as u32;

        // Scale threadgroup size with vocab breadth:
        //   < 32k  → 256 threads (most LLM tokenizers)
        //   32k–128k → 512 threads (LLaMA-3, Mistral)
        //   > 128k → 1024 threads (Llama-3 extended, Falcon-180B)
        // Always clamp to the hardware maximum.
        let threadgroup_size = if vocab_size < 32_768 {
            256_u32
        } else if vocab_size < 131_072 {
            512_u32
        } else {
            1024_u32
        }
        .min(max_threads);

        // chunk_size: vocabulary tile for the softmax sweep.
        let chunk_size = match props.device_tier {
            DeviceTier::Ultra | DeviceTier::Max => 8192,
            DeviceTier::Pro => 4096,
            DeviceTier::Base => 2048,
        };

        CrossEntropyTunedConfig {
            threadgroup_size,
            chunk_size,
        }
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

        {
            let cache = self
                .cross_entropy_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            if let Some(&cached) = cache.get(&key) {
                return Ok(cached);
            }
        }

        debug!(
            "Selecting fused linear CE config for [T={}, H={}, V={}, dtype={}]",
            config.num_tokens,
            config.hidden_size,
            config.vocab_size,
            if config.use_fp16 { "f16" } else { "f32" }
        );

        let heuristic = self.heuristic_fused_linear_cross_entropy_config(
            props.device_tier,
            config.hidden_size,
            config.vocab_size,
            props.max_threads_per_threadgroup as u32,
        );
        let candidates = self.candidate_fused_linear_cross_entropy_configs(
            props.device_tier,
            config.hidden_size,
            config.vocab_size,
            props.max_threads_per_threadgroup as u32,
        );
        let mut best_config = heuristic;
        let mut best_time = f64::INFINITY;

        info!(
            "Tuning fused linear CE config for [T={}, H={}, V={}, dtype={}]...",
            config.num_tokens,
            config.hidden_size,
            config.vocab_size,
            if config.use_fp16 { "f16" } else { "f32" }
        );

        for candidate in candidates {
            match self.benchmark_fused_linear_cross_entropy(context, config, candidate) {
                Ok(elapsed) => {
                    debug!(
                        "Fused linear CE config {:?} took {:.3} ms",
                        candidate,
                        elapsed * 1000.0
                    );
                    if elapsed < best_time {
                        best_time = elapsed;
                        best_config = candidate;
                    }
                }
                Err(error) => {
                    debug!(
                        "Fused linear CE config {:?} failed benchmarking: {}",
                        candidate,
                        error
                    );
                }
            }
        }

        if best_time.is_finite() {
            info!(
                "Selected fused linear CE config: {:?} ({:.3} ms)",
                best_config,
                best_time * 1000.0
            );
        } else {
            debug!(
                "Falling back to heuristic fused linear CE config {:?} for [H={}, V={}] on {:?}",
                best_config,
                config.hidden_size,
                config.vocab_size,
                props.device_tier
            );
        }

        {
            let mut cache = self
                .cross_entropy_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            cache.insert(key.clone(), best_config);
        }

        self.save_cross_entropy_to_disk(&key, &best_config);

        Ok(best_config)
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
                if hidden_size >= 4096 { 1024 } else { 2048 }
            }
            DeviceTier::Pro => {
                if hidden_size >= 8192 { 2048 } else { 4096 }
            }
            DeviceTier::Max | DeviceTier::Ultra => {
                if hidden_size >= 8192 { 4096 } else { 8192 }
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

        dedupe_cross_entropy_configs(configs)
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
            .and_then(|x| {
                x.checked_add(config.num_tokens.checked_mul(std::mem::size_of::<i32>())?)
            })
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

            kernel.forward_f16(&hidden_states, &lm_head_weight, &targets)?;

            let iterations = 3;
            let start = Instant::now();
            for _ in 0..iterations {
                let output = kernel.forward_f16(&hidden_states, &lm_head_weight, &targets)?;
                std::hint::black_box(output);
            }
            return Ok(start.elapsed().as_secs_f64() / iterations as f64);
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

        kernel.forward(&hidden_states, &lm_head_weight, &targets)?;

        let iterations = 3;
        let start = Instant::now();
        for _ in 0..iterations {
            let output = kernel.forward(&hidden_states, &lm_head_weight, &targets)?;
            std::hint::black_box(output);
        }

        Ok(start.elapsed().as_secs_f64() / iterations as f64)
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

        // Check in-memory cache
        {
            let cache = self
                .norm_lora_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            if let Some(&config) = cache.get(&key) {
                return Ok(config);
            }
        }

        debug!(
            "Selecting Norm+LoRA config for [B={}, H={}, O={}, R={}]",
            batch_size, hidden_size, out_features, rank
        );

        let heuristic = self.heuristic_norm_lora_config(
            props.device_tier,
            out_features,
            props.max_threads_per_threadgroup as u32,
        );
        let candidates = self.candidate_norm_lora_configs(
            props.device_tier,
            out_features,
            props.max_threads_per_threadgroup as u32,
        );
        let config = FusedNormLoraConfig::new(batch_size, hidden_size, out_features, rank, 16.0);
        let mut best_config = heuristic;
        let mut best_time = f64::INFINITY;

        info!(
            "Tuning Norm+LoRA config for [B={}, H={}, O={}, R={}]...",
            batch_size, hidden_size, out_features, rank
        );

        for candidate in candidates {
            match self.benchmark_norm_lora(context, &config, candidate) {
                Ok(elapsed) => {
                    debug!(
                        "Norm+LoRA config {:?} took {:.3} ms",
                        candidate,
                        elapsed * 1000.0
                    );
                    if elapsed < best_time {
                        best_time = elapsed;
                        best_config = candidate;
                    }
                }
                Err(error) => {
                    debug!(
                        "Norm+LoRA config {:?} failed benchmarking: {}",
                        candidate,
                        error
                    );
                }
            }
        }

        if best_time.is_finite() {
            info!(
                "Selected Norm+LoRA config: {:?} ({:.3} ms)",
                best_config,
                best_time * 1000.0
            );
        } else {
            debug!(
                "Falling back to heuristic Norm+LoRA config {:?} for out_features={} on {:?}",
                best_config,
                out_features,
                props.device_tier
            );
        }

        // Update in-memory cache
        {
            let mut cache = self
                .norm_lora_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            cache.insert(key.clone(), best_config);
        }

        // Write-through to disk
        self.save_norm_lora_to_disk(&key, &best_config);

        Ok(best_config)
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

        dedupe_norm_lora_configs(configs)
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
        kernel.forward(&input, &gamma, &weight, &lora_a, &lora_b)?;

        let iterations = 3;
        let start = Instant::now();
        for _ in 0..iterations {
            kernel.forward(&input, &gamma, &weight, &lora_a, &lora_b)?;
        }

        Ok(start.elapsed().as_secs_f64() / iterations as f64)
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
    // Tuner construction
    // -------------------------------------------------------------------------

    #[test]
    fn tuner_new_has_no_cache_dir() {
        let tuner = Tuner::new();
        assert!(tuner.cache_dir.is_none());
    }

    #[test]
    fn tuner_with_persistent_cache_has_cache_dir() {
        let tuner = Tuner::with_persistent_cache();
        // dirs::cache_dir() should always resolve on macOS
        assert!(tuner.cache_dir.is_some());
        let dir = tuner.cache_dir.unwrap();
        assert!(dir.ends_with("pmetal/tuna"));
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
    // Disk cache round-trip (uses tempdir)
    // -------------------------------------------------------------------------

    #[test]
    fn disk_cache_write_read_roundtrip() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");

        // Construct a tuner whose cache_dir points at our temp location.
        let tuner = Tuner {
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path.clone()),
        };

        let key = "swiglu:4:2048:8192";
        let config = SwiGLUTunedConfig {
            threads_per_token: 512,
            chunk_size: 4096,
        };

        tuner.save_swiglu_to_disk(key, &config);

        // Verify the file exists and contains valid JSON.
        let json_path = cache_path.join("swiglu.json");
        assert!(json_path.exists(), "swiglu.json should have been created");

        let contents = fs::read_to_string(&json_path).unwrap();
        let map: HashMap<String, SwiGLUTunedConfig> = serde_json::from_str(&contents).unwrap();
        assert_eq!(map.get(key), Some(&config));
    }

    #[test]
    fn disk_cache_load_populates_in_memory() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");
        fs::create_dir_all(&cache_path).unwrap();

        // Write a hand-crafted JSON file for swiglu.
        let key = "swiglu:8:4096:16384";
        let config = SwiGLUTunedConfig {
            threads_per_token: 256,
            chunk_size: 2048,
        };
        let map: HashMap<&str, SwiGLUTunedConfig> = [(key, config)].into_iter().collect();
        let json = serde_json::to_string_pretty(&map).unwrap();
        fs::write(cache_path.join("swiglu.json"), &json).unwrap();

        // Build a tuner pointing at that directory and call load_disk_cache.
        let mut tuner = Tuner {
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path),
        };
        tuner.load_disk_cache();

        let guard = tuner.swiglu_cache.lock().unwrap();
        assert_eq!(guard.get(key), Some(&config));
    }

    #[test]
    fn mpp_gemm_disk_cache_write_read_roundtrip() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");

        let tuner = Tuner {
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path.clone()),
        };

        let key = "mpp_gemm:Apple M5 Pro:pro:64:256:128:1:f16:plain";
        let config = MppGemmTunedConfig {
            variant: MppGemmKernelVariant::Sg2_32x64,
            use_morton: false,
        };
        tuner.save_mpp_gemm_to_disk(key, &config);

        let json_path = cache_path.join("mpp_gemm.json");
        assert!(json_path.exists(), "mpp_gemm.json should have been created");

        let contents = fs::read_to_string(&json_path).unwrap();
        let map: HashMap<String, MppGemmTunedConfig> = serde_json::from_str(&contents).unwrap();
        assert_eq!(map.get(key), Some(&config));
    }

    #[test]
    fn merge_disk_cache_write_read_roundtrip() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");

        let tuner = Tuner {
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path.clone()),
        };

        let key = "merge:Apple M4 Max:max:16:40:1048576:8";
        let config = MergeTunedConfig {
            threads_per_group: 512,
            elements_per_thread: 8,
            use_simd: true,
        };
        tuner.save_merge_to_disk(key, &config);

        let json_path = cache_path.join("merge.json");
        assert!(json_path.exists(), "merge.json should have been created");

        let contents = fs::read_to_string(&json_path).unwrap();
        let map: HashMap<String, MergeTunedConfig> = serde_json::from_str(&contents).unwrap();
        assert_eq!(map.get(key), Some(&config));
    }

    #[test]
    fn merge_disk_cache_load_populates_in_memory() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");
        fs::create_dir_all(&cache_path).unwrap();

        let key = "merge:Apple M5 Pro:pro:17:28:2097152:6";
        let config = MergeTunedConfig {
            threads_per_group: 256,
            elements_per_thread: 8,
            use_simd: true,
        };
        let map: HashMap<&str, MergeTunedConfig> = [(key, config)].into_iter().collect();
        let json = serde_json::to_string_pretty(&map).unwrap();
        fs::write(cache_path.join("merge.json"), &json).unwrap();

        let mut tuner = Tuner {
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path),
        };
        tuner.load_disk_cache();

        let guard = tuner.merge_cache.lock().unwrap();
        assert_eq!(guard.get(key), Some(&config));
    }

    #[test]
    fn mpp_gemm_disk_cache_load_populates_in_memory() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");
        fs::create_dir_all(&cache_path).unwrap();

        let key = "mpp_gemm:Apple M5 Max:max:128:512:256:2:f16:acc";
        let config = MppGemmTunedConfig {
            variant: MppGemmKernelVariant::Sg4_64x64,
            use_morton: true,
        };
        let map: HashMap<&str, MppGemmTunedConfig> = [(key, config)].into_iter().collect();
        let json = serde_json::to_string_pretty(&map).unwrap();
        fs::write(cache_path.join("mpp_gemm.json"), &json).unwrap();

        let mut tuner = Tuner {
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path),
        };
        tuner.load_disk_cache();

        let guard = tuner.mpp_gemm_cache.lock().unwrap();
        assert_eq!(guard.get(key), Some(&config));
    }

    #[test]
    fn flash_attention_disk_cache_write_read_roundtrip() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");

        let tuner = Tuner {
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path.clone()),
        };

        let key = "flash_attention:Apple M4 Max:max:1:32:8:1:2048:128:causal:full:plain:infer";
        let config = FlashAttentionTunedConfig {
            block_q: 64,
            block_k: 32,
        };
        tuner.save_flash_attention_to_disk(key, &config);

        let json_path = cache_path.join("flash_attention.json");
        assert!(
            json_path.exists(),
            "flash_attention.json should have been created"
        );

        let contents = fs::read_to_string(&json_path).unwrap();
        let map: HashMap<String, FlashAttentionTunedConfig> =
            serde_json::from_str(&contents).unwrap();
        assert_eq!(map.get(key), Some(&config));
    }

    #[test]
    fn flash_attention_disk_cache_load_populates_in_memory() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");
        fs::create_dir_all(&cache_path).unwrap();

        let key = "flash_attention:Apple M5 Pro:pro:1:32:8:64:2048:128:causal:full:plain:infer";
        let config = FlashAttentionTunedConfig {
            block_q: 32,
            block_k: 32,
        };
        let map: HashMap<&str, FlashAttentionTunedConfig> = [(key, config)].into_iter().collect();
        let json = serde_json::to_string_pretty(&map).unwrap();
        fs::write(cache_path.join("flash_attention.json"), &json).unwrap();

        let mut tuner = Tuner {
            cache: Mutex::new(HashMap::new()),
            merge_cache: Mutex::new(HashMap::new()),
            swiglu_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            flash_attention_cache: Mutex::new(HashMap::new()),
            norm_lora_cache: Mutex::new(HashMap::new()),
            mpp_gemm_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path),
        };
        tuner.load_disk_cache();

        let guard = tuner.flash_attention_cache.lock().unwrap();
        assert_eq!(guard.get(key), Some(&config));
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

    /// Verify threadgroup_size scales with vocab_size.
    #[test]
    #[cfg(target_os = "macos")]
    fn cross_entropy_threadgroup_scales_with_vocab() {
        let ctx = MetalContext::new().expect("Metal required");
        let tuner = Tuner::new();

        let c_small = tuner
            .tune_cross_entropy(&ctx, 16, 8192)
            .expect("cross_entropy small vocab");
        let c_large = tuner
            .tune_cross_entropy(&ctx, 16, 200_000)
            .expect("cross_entropy large vocab");

        assert!(
            c_large.threadgroup_size >= c_small.threadgroup_size,
            "Larger vocab should have >= threadgroup_size"
        );
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
            tuner.heuristic_fused_linear_cross_entropy_config(
                DeviceTier::Max,
                2048,
                200_000,
                1024,
            ),
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

        let cache = tuner.cross_entropy_cache.lock().expect("cross_entropy cache");
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
