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
use std::sync::Mutex;
use std::time::Instant;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};
use serde::{Deserialize, Serialize};

use crate::context::MetalContext;
use crate::error::{MetalError, Result};
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

// ============================================================================
// Tuner
// ============================================================================

/// The Auto-Tuner.
pub struct Tuner {
    /// Cache of best configurations for matrix ops.
    /// Key: "kernel_name:M:N:K" (problem size hash)
    cache: Mutex<HashMap<String, TunedConfig>>,

    /// Cache of best configurations for merge ops.
    /// Key: "merge_kernel:num_elements:num_models"
    merge_cache: Mutex<HashMap<String, MergeTunedConfig>>,

    /// Cache of best configurations for SwiGLU kernels.
    /// Key: "swiglu:batch:hidden:intermediate"
    swiglu_cache: Mutex<HashMap<String, SwiGLUTunedConfig>>,

    /// Cache of best configurations for cross-entropy kernels.
    /// Key: "cross_entropy:num_tokens:vocab_size"
    cross_entropy_cache: Mutex<HashMap<String, CrossEntropyTunedConfig>>,

    /// Cache of best configurations for Norm+LoRA fused kernels.
    /// Key: "norm_lora:batch:hidden:out_features:rank"
    norm_lora_cache: Mutex<HashMap<String, NormLoraTunedConfig>>,

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
            norm_lora_cache: Mutex::new(HashMap::new()),
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
            norm_lora_cache: Mutex::new(HashMap::new()),
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
        load_disk_cache_file::<SwiGLUTunedConfig>(&dir.join("swiglu.json"), &self.swiglu_cache);
        load_disk_cache_file::<CrossEntropyTunedConfig>(
            &dir.join("cross_entropy.json"),
            &self.cross_entropy_cache,
        );
        load_disk_cache_file::<NormLoraTunedConfig>(
            &dir.join("norm_lora.json"),
            &self.norm_lora_cache,
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

    /// Persist the full cross-entropy in-memory cache to `cross_entropy.json`.
    fn save_cross_entropy_to_disk(&self, key: &str, config: &CrossEntropyTunedConfig) {
        let Some(dir) = self.ensure_cache_dir() else {
            return;
        };
        let path = dir.join("cross_entropy.json");
        self.flush_cache_file(&path, &self.cross_entropy_cache, key, config);
    }

    /// Persist the full norm-lora in-memory cache to `norm_lora.json`.
    fn save_norm_lora_to_disk(&self, key: &str, config: &NormLoraTunedConfig) {
        let Some(dir) = self.ensure_cache_dir() else {
            return;
        };
        let path = dir.join("norm_lora.json");
        self.flush_cache_file(&path, &self.norm_lora_cache, key, config);
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
        let key = format!(
            "fused_lora_forward:{}:{}:{}:{}",
            batch_size, in_features, out_features, rank
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
        let key = format!("merge:{}:{}", num_elements, num_models);

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
        self.set_merge_config(key, best_config);

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
    // SwiGLU Kernel Tuning (heuristic)
    // =========================================================================

    /// Tune the SwiGLU activation kernel for the given problem size.
    ///
    /// SwiGLU is memory-bandwidth-bound on Apple Silicon, so the optimal config
    /// is dominated by device memory bandwidth tier rather than by runtime
    /// benchmarking. A heuristic based on the device tier and problem size is
    /// therefore preferred over full benchmarking, which would require setting
    /// up the full weight buffers and is cost-prohibitive at startup.
    ///
    /// # Arguments
    /// * `context` - Metal context
    /// * `batch_size` - Number of tokens in the batch
    /// * `hidden_size` - Model hidden dimension
    /// * `intermediate_size` - MLP intermediate dimension (typically 8/3 * hidden_size)
    ///
    /// # Returns
    /// Heuristically-selected optimal configuration for SwiGLU on this hardware.
    pub fn tune_swiglu(
        &self,
        context: &MetalContext,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<SwiGLUTunedConfig> {
        let key = format!(
            "swiglu:{}:{}:{}",
            batch_size, hidden_size, intermediate_size
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

        let config = self.select_swiglu_config(context, batch_size, intermediate_size);

        info!(
            "Selected SwiGLU config: {:?} (heuristic, device tier: {:?})",
            config,
            context.properties().device_tier
        );

        // Update in-memory cache
        {
            let mut cache = self
                .swiglu_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            cache.insert(key.clone(), config);
        }

        // Write-through to disk
        self.save_swiglu_to_disk(&key, &config);

        Ok(config)
    }

    /// Select the best SwiGLU config using device-tier heuristics.
    ///
    /// Decision rationale:
    /// - `threads_per_token`: Apple Silicon SIMD width is 32. We scale up by
    ///   bandwidth tier so high-bandwidth devices can saturate more compute lanes
    ///   per token without stalling on memory.
    /// - `chunk_size`: Larger chunks improve arithmetic intensity but require
    ///   more threadgroup memory. High-end devices have larger L2 caches.
    fn select_swiglu_config(
        &self,
        context: &MetalContext,
        _batch_size: usize,
        intermediate_size: usize,
    ) -> SwiGLUTunedConfig {
        use crate::context::DeviceTier;

        let props = context.properties();

        // For very small intermediate sizes, a smaller chunk avoids over-committing
        // threadgroup memory on any device tier.
        let small_intermediate = intermediate_size < 2048;

        match props.device_tier {
            DeviceTier::Ultra | DeviceTier::Max => {
                // High-bandwidth: prefer wider threadgroups and larger chunks for
                // better arithmetic intensity, unless intermediate_size is tiny.
                SwiGLUTunedConfig {
                    threads_per_token: 512,
                    chunk_size: if small_intermediate { 2048 } else { 4096 },
                }
            }
            DeviceTier::Pro => SwiGLUTunedConfig {
                threads_per_token: 256,
                chunk_size: if small_intermediate { 2048 } else { 4096 },
            },
            DeviceTier::Base => {
                // Base chips: moderate thread count for better occupancy.
                SwiGLUTunedConfig {
                    threads_per_token: if small_intermediate { 128 } else { 256 },
                    chunk_size: if small_intermediate { 1024 } else { 2048 },
                }
            }
        }
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

    // =========================================================================
    // Norm+LoRA Fused Kernel Tuning (heuristic)
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
    /// Heuristically-selected optimal configuration for Norm+LoRA on this hardware.
    pub fn tune_norm_lora(
        &self,
        context: &MetalContext,
        batch_size: usize,
        hidden_size: usize,
        out_features: usize,
        rank: usize,
    ) -> Result<NormLoraTunedConfig> {
        let key = format!(
            "norm_lora:{}:{}:{}:{}",
            batch_size, hidden_size, out_features, rank
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

        let config = self.select_norm_lora_config(context, out_features);

        info!(
            "Selected Norm+LoRA config: {:?} (heuristic, out_features={}, device tier: {:?})",
            config,
            out_features,
            context.properties().device_tier
        );

        // Update in-memory cache
        {
            let mut cache = self
                .norm_lora_cache
                .lock()
                .map_err(|e| MetalError::Internal(format!("Mutex poisoned: {}", e)))?;
            cache.insert(key.clone(), config);
        }

        // Write-through to disk
        self.save_norm_lora_to_disk(&key, &config);

        Ok(config)
    }

    /// Select the best Norm+LoRA config using heuristics.
    ///
    /// Decision rationale:
    /// - `use_tiled`: Tiling shared memory pays off when `out_features > 256`.
    ///   Below that threshold the overhead of loading data into threadgroup
    ///   memory exceeds the bandwidth savings.
    /// - `threads_per_token`: Scales with device tier. The norm reduction over
    ///   `hidden_size` is the bottleneck; more threads reduce the serial portion.
    fn select_norm_lora_config(
        &self,
        context: &MetalContext,
        out_features: usize,
    ) -> NormLoraTunedConfig {
        use crate::context::DeviceTier;

        let props = context.properties();

        // Tiled path is profitable when out_features is wide enough.
        let use_tiled = out_features > 256;

        let threads_per_token = match props.device_tier {
            DeviceTier::Ultra | DeviceTier::Max => 512,
            DeviceTier::Pro => 256,
            DeviceTier::Base => 128,
        };

        NormLoraTunedConfig {
            threads_per_token,
            use_tiled,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
    fn norm_lora_tuned_config_serde_roundtrip() {
        let config = NormLoraTunedConfig {
            threads_per_token: 512,
            use_tiled: true,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: NormLoraTunedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, decoded);
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
        let key = "merge:1024:3".to_string();
        assert!(tuner.get_merge_config(&key).is_none());

        let cfg = MergeTunedConfig {
            threads_per_group: 512,
            elements_per_thread: 8,
            use_simd: true,
        };
        tuner.set_merge_config(key.clone(), cfg);
        assert_eq!(tuner.get_merge_config(&key), Some(cfg));
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
            norm_lora_cache: Mutex::new(HashMap::new()),
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
            norm_lora_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path),
        };
        tuner.load_disk_cache();

        let guard = tuner.swiglu_cache.lock().unwrap();
        assert_eq!(guard.get(key), Some(&config));
    }

    // -------------------------------------------------------------------------
    // Heuristic selection sanity checks (no Metal device needed)
    // -------------------------------------------------------------------------

    // The heuristic methods need a `MetalContext`, which requires a real GPU.
    // We test the pure-logic paths below by constructing synthetic properties
    // and calling the internal selection helpers via a wrapper that exposes them
    // for test use only. Because the selection functions are private we exercise
    // them through the public tuning API (which requires a Metal device).
    //
    // The tests below instead verify the public surface: tune_* methods with a
    // real context are exercised in the metal integration tests. Here we check
    // that the heuristic functions produce sensible values by directly matching
    // expected output against known device tier inputs using the integration
    // with the global Metal context (only run when Metal is available).

    /// Verify use_tiled=true when out_features > 256.
    #[test]
    #[cfg(target_os = "macos")]
    fn norm_lora_use_tiled_threshold() {
        let ctx = MetalContext::new().expect("Metal required");
        let tuner = Tuner::new();

        // out_features = 128 -> use_tiled should be false
        let c_small = tuner
            .tune_norm_lora(&ctx, 1, 512, 128, 8)
            .expect("tune_norm_lora small");
        assert!(
            !c_small.use_tiled,
            "Small out_features should not use tiled"
        );

        // out_features = 512 -> use_tiled should be true
        let c_large = tuner
            .tune_norm_lora(&ctx, 1, 512, 512, 8)
            .expect("tune_norm_lora large");
        assert!(c_large.use_tiled, "Large out_features should use tiled");
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

    /// Verify SwiGLU chunk_size is smaller for small intermediate_size.
    #[test]
    #[cfg(target_os = "macos")]
    fn swiglu_chunk_size_adapts_to_size() {
        let ctx = MetalContext::new().expect("Metal required");
        let tuner = Tuner::new();

        // intermediate_size = 512 -> small path
        let c_small = tuner.tune_swiglu(&ctx, 1, 512, 512).expect("swiglu small");
        // intermediate_size = 8192 -> large path
        let c_large = tuner.tune_swiglu(&ctx, 1, 512, 8192).expect("swiglu large");

        assert!(
            c_large.chunk_size >= c_small.chunk_size,
            "Larger intermediate_size should have >= chunk_size"
        );
    }

    /// Verify in-memory caching avoids re-computation (second call returns
    /// the exact same struct without hitting the heuristic again).
    #[test]
    #[cfg(target_os = "macos")]
    fn swiglu_cache_hit_on_second_call() {
        let ctx = MetalContext::new().expect("Metal required");
        let tuner = Tuner::new();

        let first = tuner.tune_swiglu(&ctx, 4, 2048, 8192).unwrap();
        let second = tuner.tune_swiglu(&ctx, 4, 2048, 8192).unwrap();
        assert_eq!(first, second);
    }
}
