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
//! let config = tuner.tune_cross_entropy(&ctx, num_tokens, vocab_size)?;
//! println!("Best config: {:?}", config);
//!
//! // Or with persistent cache:
//! let tuner = Tuner::with_persistent_cache();
//! let config = tuner.tune_cross_entropy(&ctx, num_tokens, vocab_size)?;
//! ```

use std::collections::HashMap;
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
    /// Cache of best configurations for merge ops.
    /// Key: "merge:device_name:device_tier:architecture_gen:gpu_core_count:num_elements:num_models"
    merge_cache: Mutex<HashMap<String, MergeTunedConfig>>,

    /// Cache of best configurations for cross-entropy kernels.
    /// Keys cover both generic logits CE and fused-linear CE shapes.
    cross_entropy_cache: Mutex<HashMap<String, CrossEntropyTunedConfig>>,

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
            merge_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
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
            merge_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
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

        load_disk_cache_file::<MergeTunedConfig>(&dir.join("merge.json"), &self.merge_cache);
        load_disk_cache_file::<CrossEntropyTunedConfig>(
            &dir.join("cross_entropy.json"),
            &self.cross_entropy_cache,
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

        Ok(())
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
                        candidate, error
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
                best_config, config.hidden_size, config.vocab_size, props.device_tier
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
            merge_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path.clone()),
        };

        let key = "cross_entropy:4:2048";
        let config = CrossEntropyTunedConfig {
            threadgroup_size: 512,
            chunk_size: 4096,
        };

        tuner.save_cross_entropy_to_disk(key, &config);

        // Verify the file exists and contains valid JSON.
        let json_path = cache_path.join("cross_entropy.json");
        assert!(
            json_path.exists(),
            "cross_entropy.json should have been created"
        );

        let contents = fs::read_to_string(&json_path).unwrap();
        let map: HashMap<String, CrossEntropyTunedConfig> =
            serde_json::from_str(&contents).unwrap();
        assert_eq!(map.get(key), Some(&config));
    }

    #[test]
    fn disk_cache_load_populates_in_memory() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");
        fs::create_dir_all(&cache_path).unwrap();

        // Write a hand-crafted JSON file for cross-entropy.
        let key = "cross_entropy:8:4096";
        let config = CrossEntropyTunedConfig {
            threadgroup_size: 256,
            chunk_size: 2048,
        };
        let map: HashMap<&str, CrossEntropyTunedConfig> = [(key, config)].into_iter().collect();
        let json = serde_json::to_string_pretty(&map).unwrap();
        fs::write(cache_path.join("cross_entropy.json"), &json).unwrap();

        // Build a tuner pointing at that directory and call load_disk_cache.
        let mut tuner = Tuner {
            merge_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path),
        };
        tuner.load_disk_cache();

        let guard = tuner.cross_entropy_cache.lock().unwrap();
        assert_eq!(guard.get(key), Some(&config));
    }

    #[test]
    fn merge_disk_cache_write_read_roundtrip() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let cache_path = tmp_dir.path().join("pmetal").join("tuna");

        let tuner = Tuner {
            merge_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
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
            merge_cache: Mutex::new(HashMap::new()),
            cross_entropy_cache: Mutex::new(HashMap::new()),
            cache_dir: Some(cache_path),
        };
        tuner.load_disk_cache();

        let guard = tuner.merge_cache.lock().unwrap();
        assert_eq!(guard.get(key), Some(&config));
    }

    // -------------------------------------------------------------------------
    // Standard-Metal tuning sanity checks
    // -------------------------------------------------------------------------

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

        let cache = tuner
            .cross_entropy_cache
            .lock()
            .expect("cross_entropy cache");
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
