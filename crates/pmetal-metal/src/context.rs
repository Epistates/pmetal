#![allow(unsafe_code)]

//! Metal device and command queue management.
//!
//! This module provides thread-safe access to the Metal device and command queue,
//! using a global singleton pattern for efficiency.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tracing::{debug, info};

use crate::error::{MetalError, Result};
use crate::pipeline::PipelineCache;
use crate::tuna::Tuner;

/// Global Metal context singleton using OnceLock for thread-safe lazy initialization.
static GLOBAL_CONTEXT: OnceLock<Result<Arc<MetalContext>>> = OnceLock::new();

/// Embedded Metal 3 library binary (compiled at build time).
const METAL_LIBRARY_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/pmetal_kernels.metallib"));

/// Embedded Metal 4 / MPP library binary (compiled at build time on macOS 26+).
/// Contains NAX-accelerated kernels for M5+ GPUs.
#[cfg(has_metal4)]
const METAL4_LIBRARY_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/pmetal_kernels_metal4.metallib"));

/// Metal execution context.
///
/// Provides access to the Metal device, command queue, and pipeline cache.
/// This is designed to be created once and shared across the application.
pub struct MetalContext {
    /// The Metal device (GPU).
    device: Retained<ProtocolObject<dyn MTLDevice>>,

    /// Command queue for submitting work.
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,

    /// Cached compute pipelines.
    pipeline_cache: RwLock<PipelineCache>,

    /// Auto-tuner for kernel parameters.
    tuner: Arc<Tuner>,

    /// Device properties.
    properties: DeviceProperties,
}

/// Apple GPU family classification.
///
/// Based on Metal GPU family enumeration, used for feature detection
/// and optimization selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AppleGPUFamily {
    /// Unknown family (ordered first so Unknown < Apple7 < ... < Apple10).
    Unknown,
    /// Apple7: M1 series
    Apple7,
    /// Apple8: M2 series
    Apple8,
    /// Apple9: M3/M4 series (Dynamic Caching, enhanced ray tracing)
    Apple9,
    /// Apple10: M5 series (Neural Accelerators in GPU cores)
    Apple10,
}

/// Device performance tier for optimization selection.
///
/// Determined by chip variant (base/Pro/Max/Ultra) and memory bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTier {
    /// Base chips (M1/M2/M3/M4): 100-120 GB/s bandwidth
    Base,
    /// Pro chips: 150-273 GB/s bandwidth
    Pro,
    /// Max chips: 400-546 GB/s bandwidth
    Max,
    /// Ultra chips: 800+ GB/s bandwidth
    Ultra,
}

/// Source of the detected memory bandwidth figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryBandwidthSource {
    /// Measured on this machine via a GPU copy benchmark and cached on disk.
    MeasuredGpuCopy,
    /// Fallback to the static Apple Silicon spec table.
    SpecTableFallback,
}

/// Properties of the Metal device.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name (e.g., "Apple M3 Max").
    pub name: String,

    /// Maximum threads per threadgroup.
    pub max_threads_per_threadgroup: u64,

    /// Maximum threadgroup memory length in bytes.
    pub max_threadgroup_memory_length: u64,

    /// Whether the device supports unified memory.
    pub has_unified_memory: bool,

    /// Recommended working set size in bytes.
    pub recommended_working_set_size: u64,

    /// Maximum buffer length in bytes.
    pub max_buffer_length: u64,

    /// Detected GPU family (Apple7-Apple10).
    pub gpu_family: AppleGPUFamily,

    /// Device performance tier.
    pub device_tier: DeviceTier,

    /// Whether Dynamic Caching is supported (Apple9+).
    pub has_dynamic_caching: bool,

    /// Whether hardware ray tracing is supported (Apple9+).
    pub has_hardware_ray_tracing: bool,

    /// Whether mesh shaders are supported (Apple9+).
    pub has_mesh_shaders: bool,

    /// Whether Neural Accelerators (NAX) are available in GPU cores (Apple10+/M5+).
    /// Enables optimized GEMM, quantization, and attention kernels via Metal 4.0.
    pub has_nax: bool,

    /// Architecture generation (e.g., 15=M3, 16=M4, 17=M5).
    /// Matches MLX's `get_architecture_gen()` convention.
    pub architecture_gen: u32,

    /// Estimated or measured memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,

    /// How `memory_bandwidth_gbps` was obtained.
    pub memory_bandwidth_source: MemoryBandwidthSource,

    /// Number of GPU cores (queried via Metal API, 0 if unknown).
    pub gpu_core_count: u32,

    /// Number of ANE (Neural Engine) cores (16 for Pro/Max, 32 for Ultra, 0 if unknown).
    pub ane_core_count: u32,

    /// Whether this is an UltraFusion (multi-die) chip.
    /// Detected via `sysctl hw.packages` (2 = dual-die UltraFusion).
    pub is_ultra_fusion: bool,

    /// Number of hardware packages (dies). 1 for single-die, 2 for UltraFusion.
    pub die_count: u32,
}

impl DeviceProperties {
    /// Check if this device supports Apple Family 9 features (M3/M4).
    #[inline]
    pub fn is_apple9_or_newer(&self) -> bool {
        self.gpu_family >= AppleGPUFamily::Apple9
    }

    /// Check if this device supports Apple Family 10 features (M5).
    #[inline]
    pub fn is_apple10_or_newer(&self) -> bool {
        self.gpu_family >= AppleGPUFamily::Apple10
    }

    /// Check if this device supports NAX (Neural Accelerators in GPU — M5+).
    #[inline]
    pub fn has_nax(&self) -> bool {
        self.has_nax
    }

    /// Get recommended batch size multiplier based on device tier.
    pub fn batch_size_multiplier(&self) -> usize {
        match self.device_tier {
            DeviceTier::Base => 1,
            DeviceTier::Pro => 2,
            DeviceTier::Max => 4,
            DeviceTier::Ultra => 8,
        }
    }

    /// Get recommended tile size for matrix operations (BM, BN, BK).
    pub fn recommended_tile_size(&self) -> (u32, u32, u32) {
        if self.has_nax {
            // M5/Apple10: NAX-optimized tile sizes. Larger tiles leverage
            // the neural accelerator units within GPU cores.
            match self.device_tier {
                DeviceTier::Ultra | DeviceTier::Max => (128, 64, 32),
                DeviceTier::Pro => (64, 64, 32),
                DeviceTier::Base => (64, 32, 32),
            }
        } else {
            match self.device_tier {
                DeviceTier::Ultra | DeviceTier::Max => (64, 64, 32),
                DeviceTier::Pro => (64, 32, 32),
                DeviceTier::Base => (32, 32, 32),
            }
        }
    }

    /// Get the default MPP GEMM threadgroup tile parameters for this tier.
    ///
    /// Returns `(BM, BN, BK, num_simdgroups)` as a starting-point heuristic for
    /// Apple10/M5 tuning. The actual dispatcher can now auto-tune among
    /// multiple kernel variants (`32x32`, `64x32`, `32x64`, `64x64`), so this
    /// method represents the preferred default before per-shape benchmarking.
    pub fn mpp_tile_config(&self) -> (u32, u32, u32, u32) {
        let (bm, bn, num_simdgroups) = match self.device_tier {
            DeviceTier::Base => (64, 32, 2),
            DeviceTier::Pro | DeviceTier::Max | DeviceTier::Ultra => (64, 64, 4),
        };
        (bm, bn, 128, num_simdgroups)
    }

    /// Heuristic gate for whether a GEMM is large enough to justify trying MPP.
    ///
    /// This is intentionally conservative. Small or poorly-aligned projections
    /// often do better on MLX's default kernels once dispatch overhead is
    /// included, so callers should only benchmark or dispatch MPP after this
    /// fast rejection step passes.
    pub fn should_consider_mpp_gemm(&self, m: usize, n: usize, k: usize, use_fp16: bool) -> bool {
        if !self.has_nax || m == 0 || n < 64 || k < 64 {
            return false;
        }

        let aligned = n % 64 == 0 && k % 128 == 0;
        let work = (m as u128) * (n as u128) * (k as u128);

        let base_threshold = match (self.device_tier, use_fp16) {
            (DeviceTier::Ultra | DeviceTier::Max, true) => 524_288_u128,
            (DeviceTier::Ultra | DeviceTier::Max, false) => 1_048_576_u128,
            (DeviceTier::Pro, true) => 1_048_576_u128,
            (DeviceTier::Pro, false) => 2_097_152_u128,
            (DeviceTier::Base, true) => 2_097_152_u128,
            (DeviceTier::Base, false) => 4_194_304_u128,
        };

        let threshold = if aligned {
            base_threshold / 2
        } else {
            base_threshold
        };

        work >= threshold
    }
}

/// Detect GPU family from device name.
///
/// Uses word-boundary-aware matching to avoid false positives
/// (e.g., "M10" should not match "M1").
fn detect_gpu_family(name: &str) -> AppleGPUFamily {
    // Check for "M5" not followed by another digit
    if has_chip_id(name, "M5") {
        return AppleGPUFamily::Apple10;
    }
    if has_chip_id(name, "M4") || has_chip_id(name, "M3") || has_chip_id(name, "A17") {
        return AppleGPUFamily::Apple9;
    }
    if has_chip_id(name, "M2") || has_chip_id(name, "A16") || has_chip_id(name, "A15") {
        return AppleGPUFamily::Apple8;
    }
    if has_chip_id(name, "M1") || has_chip_id(name, "A14") {
        return AppleGPUFamily::Apple7;
    }
    AppleGPUFamily::Unknown
}

/// Check if `name` contains `chip_id` not followed by another digit.
/// e.g., `has_chip_id("Apple M1 Max", "M1")` → true,
///       `has_chip_id("Apple M10", "M1")` → false.
fn has_chip_id(name: &str, chip_id: &str) -> bool {
    let mut start = 0;
    while let Some(pos) = name[start..].find(chip_id) {
        let abs_pos = start + pos;
        let after = abs_pos + chip_id.len();
        // Ensure the match is not followed by another digit
        if after >= name.len() || !name.as_bytes()[after].is_ascii_digit() {
            return true;
        }
        start = after;
    }
    false
}

/// Detect device tier from device name.
fn detect_device_tier(name: &str) -> DeviceTier {
    if name.contains("Ultra") {
        DeviceTier::Ultra
    } else if name.contains("Max") {
        DeviceTier::Max
    } else if name.contains("Pro") {
        DeviceTier::Pro
    } else {
        DeviceTier::Base
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryBandwidthCacheEntry {
    bandwidth_gbps: f64,
    source: MemoryBandwidthSource,
}

#[derive(Debug, Clone)]
struct MemoryBandwidthRequest {
    name: String,
    family: AppleGPUFamily,
    tier: DeviceTier,
    gpu_cores: u32,
    die_count: u32,
    recommended_working_set_size: u64,
}

impl MemoryBandwidthRequest {
    fn cache_key(&self) -> String {
        memory_bandwidth_cache_key(
            &self.name,
            self.family,
            self.tier,
            self.gpu_cores,
            self.die_count,
        )
    }

    fn fallback_bandwidth_gbps(&self) -> f64 {
        estimate_memory_bandwidth(self.tier, self.family)
    }
}

/// Estimate memory bandwidth in GB/s from tier and GPU family.
/// Uses known Apple Silicon specs per generation.
fn estimate_memory_bandwidth(tier: DeviceTier, family: AppleGPUFamily) -> f64 {
    match (family, tier) {
        // M5 (Apple10) — same memory subsystem as M4
        (AppleGPUFamily::Apple10, DeviceTier::Ultra) => 800.0,
        (AppleGPUFamily::Apple10, DeviceTier::Max) => 546.0,
        (AppleGPUFamily::Apple10, DeviceTier::Pro) => 273.0,
        (AppleGPUFamily::Apple10, DeviceTier::Base) => 120.0,

        // M3/M4 (Apple9)
        (AppleGPUFamily::Apple9, DeviceTier::Ultra) => 800.0,
        (AppleGPUFamily::Apple9, DeviceTier::Max) => 546.0,
        (AppleGPUFamily::Apple9, DeviceTier::Pro) => 273.0,
        (AppleGPUFamily::Apple9, DeviceTier::Base) => 120.0,

        // M2 (Apple8)
        (AppleGPUFamily::Apple8, DeviceTier::Ultra) => 800.0,
        (AppleGPUFamily::Apple8, DeviceTier::Max) => 400.0,
        (AppleGPUFamily::Apple8, DeviceTier::Pro) => 200.0,
        (AppleGPUFamily::Apple8, DeviceTier::Base) => 100.0,

        // M1 (Apple7) and Unknown
        (_, DeviceTier::Ultra) => 800.0,
        (_, DeviceTier::Max) => 400.0,
        (_, DeviceTier::Pro) => 200.0,
        (_, DeviceTier::Base) => 68.0,
    }
}

fn memory_bandwidth_cache_key(
    name: &str,
    family: AppleGPUFamily,
    tier: DeviceTier,
    gpu_cores: u32,
    die_count: u32,
) -> String {
    format!("{name}:{family:?}:{tier:?}:{gpu_cores}:{die_count}")
}

fn memory_bandwidth_cache_path() -> Option<PathBuf> {
    dirs::cache_dir().map(|path| {
        path.join("pmetal")
            .join("context")
            .join("memory_bandwidth.json")
    })
}

fn load_memory_bandwidth_cache(path: &Path) -> Option<HashMap<String, MemoryBandwidthCacheEntry>> {
    let contents = fs::read_to_string(path).ok()?;
    serde_json::from_str(&contents).ok()
}

fn load_cached_memory_bandwidth(path: &Path, cache_key: &str) -> Option<MemoryBandwidthCacheEntry> {
    let cache = load_memory_bandwidth_cache(path)?;
    let entry = cache.get(cache_key)?.clone();
    if measured_bandwidth_is_plausible(entry.bandwidth_gbps) {
        Some(entry)
    } else {
        None
    }
}

fn store_cached_memory_bandwidth(path: &Path, cache_key: &str, entry: &MemoryBandwidthCacheEntry) {
    let Some(parent) = path.parent() else {
        return;
    };
    if let Err(error) = fs::create_dir_all(parent) {
        debug!(
            "Skipping bandwidth cache write, failed to create {}: {}",
            parent.display(),
            error
        );
        return;
    }

    let mut cache = load_memory_bandwidth_cache(path).unwrap_or_default();
    cache.insert(cache_key.to_string(), entry.clone());

    let tmp_path = path.with_extension("json.tmp");
    let Ok(json) = serde_json::to_string_pretty(&cache) else {
        return;
    };
    if fs::write(&tmp_path, json).is_err() {
        return;
    }
    if fs::rename(&tmp_path, path).is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
}

fn measured_bandwidth_is_plausible(bandwidth_gbps: f64) -> bool {
    bandwidth_gbps.is_finite() && (25.0..=2_000.0).contains(&bandwidth_gbps)
}

fn measured_copy_buffer_size_bytes(recommended_working_set_size: u64) -> usize {
    let quarter_working_set = (recommended_working_set_size / 4) as usize;
    quarter_working_set.clamp(32 * 1024 * 1024, 128 * 1024 * 1024)
}

/// Map GPU family to architecture generation number.
/// Matches MLX convention: Apple7=14, Apple8=15, Apple9=16, Apple10=17.
fn architecture_gen(family: AppleGPUFamily) -> u32 {
    match family {
        AppleGPUFamily::Apple7 => 14,
        AppleGPUFamily::Apple8 => 15,
        AppleGPUFamily::Apple9 => 16,
        AppleGPUFamily::Apple10 => 17,
        AppleGPUFamily::Unknown => 0,
    }
}

/// Estimate GPU core count from device name and tier.
/// Uses known Apple Silicon specs. Returns 0 if unknown.
fn estimate_gpu_cores(name: &str, tier: DeviceTier) -> u32 {
    if has_chip_id(name, "M5") {
        return match tier {
            DeviceTier::Ultra => 80,
            DeviceTier::Max => 40,
            DeviceTier::Pro => 20,
            DeviceTier::Base => 10,
        };
    }
    if has_chip_id(name, "M4") {
        return match tier {
            DeviceTier::Ultra => 80,
            DeviceTier::Max => 40,
            DeviceTier::Pro => 20,
            DeviceTier::Base => 10,
        };
    }
    if has_chip_id(name, "M3") {
        return match tier {
            DeviceTier::Ultra => 60,
            DeviceTier::Max => 30,
            DeviceTier::Pro => 18,
            DeviceTier::Base => 10,
        };
    }
    if has_chip_id(name, "M2") {
        return match tier {
            DeviceTier::Ultra => 48,
            DeviceTier::Max => 30,
            DeviceTier::Pro => 16,
            DeviceTier::Base => 8,
        };
    }
    if has_chip_id(name, "M1") {
        return match tier {
            DeviceTier::Ultra => 48,
            DeviceTier::Max => 24,
            DeviceTier::Pro => 14,
            DeviceTier::Base => 8,
        };
    }
    0
}

/// Estimate ANE core count from tier.
/// All Pro/Max/Base variants have 16 NE cores; Ultra has 32 (UltraFusion 2x).
fn estimate_ane_cores(tier: DeviceTier) -> u32 {
    match tier {
        DeviceTier::Ultra => 32,
        _ => 16,
    }
}

fn measure_gpu_copy_bandwidth_gbps(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline_cache: &mut PipelineCache,
    recommended_working_set_size: u64,
) -> Result<f64> {
    let copy_bytes = measured_copy_buffer_size_bytes(recommended_working_set_size);
    let element_count = copy_bytes
        .checked_div(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::InvalidConfig("bandwidth probe size underflow".to_string()))?;
    let vec_count = element_count / 4;
    if vec_count == 0 {
        return Err(MetalError::InvalidConfig(
            "bandwidth probe buffer too small".to_string(),
        ));
    }

    let pipeline = pipeline_cache.get_or_create_pipeline(device, "bandwidth_probe_f32", None)?;

    let options =
        MTLResourceOptions::StorageModePrivate | MTLResourceOptions::HazardTrackingModeTracked;
    let src = device
        .newBufferWithLength_options(copy_bytes, options)
        .ok_or_else(|| MetalError::BufferCreation {
            size: copy_bytes,
            reason: "bandwidth probe src".to_string(),
        })?;
    let dst = device
        .newBufferWithLength_options(copy_bytes, options)
        .ok_or_else(|| MetalError::BufferCreation {
            size: copy_bytes,
            reason: "bandwidth probe dst".to_string(),
        })?;

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let threads_per_threadgroup = max_threads.clamp(32, 256).div_ceil(32) * 32;
    let threadgroups = vec_count.div_ceil(threads_per_threadgroup);

    let dispatch = |src: &Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
                    dst: &Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>|
     -> Result<()> {
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
            let count = vec_count as u32;
            let count_ptr = NonNull::from(&count).cast();
            encoder.setBytes_length_atIndex(count_ptr, std::mem::size_of_val(&count), 2);
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threadgroups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    };

    for _ in 0..2 {
        dispatch(&src, &dst)?;
        dispatch(&dst, &src)?;
    }

    let iterations = 6usize;
    let start = Instant::now();
    for i in 0..iterations {
        if i % 2 == 0 {
            dispatch(&src, &dst)?;
        } else {
            dispatch(&dst, &src)?;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    if elapsed <= 0.0 {
        return Err(MetalError::ExecutionFailed(
            "bandwidth probe elapsed time was zero".to_string(),
        ));
    }

    let total_bytes = copy_bytes as f64 * 2.0 * iterations as f64;
    Ok(total_bytes / elapsed / 1e9)
}

fn resolve_memory_bandwidth(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline_cache: &mut PipelineCache,
    request: &MemoryBandwidthRequest,
) -> (f64, MemoryBandwidthSource) {
    let fallback = request.fallback_bandwidth_gbps();
    let cache_key = request.cache_key();

    if let Some(path) = memory_bandwidth_cache_path() {
        if let Some(entry) = load_cached_memory_bandwidth(&path, &cache_key) {
            return (entry.bandwidth_gbps, entry.source);
        }

        if let Ok(measured) = measure_gpu_copy_bandwidth_gbps(
            device,
            command_queue,
            pipeline_cache,
            request.recommended_working_set_size,
        ) {
            if measured_bandwidth_is_plausible(measured) {
                let entry = MemoryBandwidthCacheEntry {
                    bandwidth_gbps: measured,
                    source: MemoryBandwidthSource::MeasuredGpuCopy,
                };
                store_cached_memory_bandwidth(&path, &cache_key, &entry);
                return (entry.bandwidth_gbps, entry.source);
            }
        }

        let entry = MemoryBandwidthCacheEntry {
            bandwidth_gbps: fallback,
            source: MemoryBandwidthSource::SpecTableFallback,
        };
        store_cached_memory_bandwidth(&path, &cache_key, &entry);
    }

    (fallback, MemoryBandwidthSource::SpecTableFallback)
}

/// Detect the number of hardware packages (dies) via `sysctl hw.packages`.
/// Returns 2 for UltraFusion chips, 1 for single-die chips.
fn detect_die_count() -> u32 {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.packages"]).output() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                if let Ok(count) = s.trim().parse::<u32>() {
                    return count;
                }
            }
        }
    }
    // Fallback: assume single-die
    1
}

impl MetalContext {
    /// Get the global Metal context, initializing it if necessary.
    ///
    /// This is the recommended way to obtain a Metal context, as it ensures
    /// only one context exists per process (which is optimal for Metal).
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available.
    pub fn global() -> Result<Arc<MetalContext>> {
        GLOBAL_CONTEXT
            .get_or_init(|| {
                MetalContext::new().map(Arc::new).map_err(|e| {
                    tracing::error!("Failed to initialize Metal context: {}", e);
                    e
                })
            })
            .clone()
    }

    /// Create a new Metal context.
    ///
    /// Prefer using [`MetalContext::global()`] unless you have a specific
    /// need for multiple contexts.
    pub fn new() -> Result<Self> {
        // Get the default Metal device
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;

        info!("Initialized Metal device: {}", device.name());

        // Create command queue
        let command_queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueCreation)?;

        // Query device properties
        let name = device.name().to_string();
        let gpu_family = detect_gpu_family(&name);
        let device_tier = detect_device_tier(&name);
        let gpu_core_count = estimate_gpu_cores(&name, device_tier);
        let ane_core_count = estimate_ane_cores(device_tier);
        let die_count = detect_die_count();
        let is_ultra_fusion = die_count > 1;
        let recommended_working_set_size = device.recommendedMaxWorkingSetSize();
        let has_nax = gpu_family >= AppleGPUFamily::Apple10;

        // Apple9+ features
        let is_apple9_plus = gpu_family >= AppleGPUFamily::Apple9;

        // Create pipeline cache and load the Metal library
        let mut cache = PipelineCache::new();
        cache.load_library(&device, METAL_LIBRARY_BYTES)?;
        info!(
            "Loaded Metal library ({} bytes, {} kernels available)",
            METAL_LIBRARY_BYTES.len(),
            cache
                .library()
                .map(|l| l.functionNames().len())
                .unwrap_or(0)
        );

        // Load Metal 4 / MPP library if available and device supports NAX
        #[cfg(has_metal4)]
        if has_nax {
            match cache.load_metal4_library(&device, METAL4_LIBRARY_BYTES) {
                Ok(()) => {
                    info!(
                        "Loaded Metal 4 / MPP library ({} bytes, {} NAX kernels)",
                        METAL4_LIBRARY_BYTES.len(),
                        cache
                            .metal4_library()
                            .map(|l| l.functionNames().len())
                            .unwrap_or(0)
                    );
                }
                Err(e) => {
                    info!("Metal 4 library available but failed to load: {e}");
                }
            }
        }

        let bandwidth_request = MemoryBandwidthRequest {
            name: name.clone(),
            family: gpu_family,
            tier: device_tier,
            gpu_cores: gpu_core_count,
            die_count,
            recommended_working_set_size,
        };
        let (memory_bandwidth_gbps, memory_bandwidth_source) =
            resolve_memory_bandwidth(&device, &command_queue, &mut cache, &bandwidth_request);

        let properties = DeviceProperties {
            name: name.clone(),
            max_threads_per_threadgroup: device.maxThreadsPerThreadgroup().width as u64,
            max_threadgroup_memory_length: device.maxThreadgroupMemoryLength() as u64,
            has_unified_memory: device.hasUnifiedMemory(),
            recommended_working_set_size,
            max_buffer_length: device.maxBufferLength() as u64,
            gpu_family,
            device_tier,
            has_dynamic_caching: is_apple9_plus,
            has_hardware_ray_tracing: is_apple9_plus,
            has_mesh_shaders: is_apple9_plus,
            has_nax,
            architecture_gen: architecture_gen(gpu_family),
            memory_bandwidth_gbps,
            memory_bandwidth_source,
            gpu_core_count,
            ane_core_count,
            die_count,
            is_ultra_fusion,
        };

        debug!("Device properties: {:?}", properties);
        if properties.is_ultra_fusion {
            info!(
                "GPU: {:?} {:?} (gen {}), {} cores ({} dies, UltraFusion), {:.0} GB/s [{:?}], NAX: {}, ANE: {} cores",
                properties.gpu_family,
                properties.device_tier,
                properties.architecture_gen,
                properties.gpu_core_count,
                properties.die_count,
                properties.memory_bandwidth_gbps,
                properties.memory_bandwidth_source,
                properties.has_nax,
                properties.ane_core_count,
            );
        } else {
            info!(
                "GPU: {:?} {:?} (gen {}), {} cores, {:.0} GB/s [{:?}], NAX: {}, ANE: {} cores",
                properties.gpu_family,
                properties.device_tier,
                properties.architecture_gen,
                properties.gpu_core_count,
                properties.memory_bandwidth_gbps,
                properties.memory_bandwidth_source,
                properties.has_nax,
                properties.ane_core_count,
            );
        }

        let pipeline_cache = RwLock::new(cache);
        let tuner = Arc::new(Tuner::with_persistent_cache());

        Ok(Self {
            device,
            command_queue,
            pipeline_cache,
            tuner,
            properties,
        })
    }

    /// Get a reference to the Metal device.
    #[inline]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Get a reference to the command queue.
    #[inline]
    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.command_queue
    }

    /// Get a reference to the auto-tuner.
    #[inline]
    pub fn tuner(&self) -> &Arc<Tuner> {
        &self.tuner
    }

    /// Get device properties.
    #[inline]
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Get read access to the pipeline cache.
    #[inline]
    pub fn pipeline_cache(&self) -> parking_lot::RwLockReadGuard<'_, PipelineCache> {
        self.pipeline_cache.read()
    }

    /// Get write access to the pipeline cache.
    #[inline]
    pub fn pipeline_cache_mut(&self) -> parking_lot::RwLockWriteGuard<'_, PipelineCache> {
        self.pipeline_cache.write()
    }

    /// Check if this device supports the Neural Engine.
    ///
    /// When the `ane` feature is enabled, this attempts to load the private
    /// `AppleNeuralEngine.framework` at runtime. Returns `true` on M1+ hardware.
    /// Without the `ane` feature, always returns `false`.
    #[inline]
    pub fn supports_neural_engine(&self) -> bool {
        #[cfg(feature = "ane")]
        {
            crate::ane::runtime::AneRuntime::global().is_ok()
        }
        #[cfg(not(feature = "ane"))]
        {
            false
        }
    }

    /// Get the maximum recommended batch size for a given operation.
    ///
    /// This takes into account the device's memory and compute capabilities.
    pub fn recommended_batch_size(&self, bytes_per_element: usize) -> usize {
        // Use about 25% of recommended working set for a single operation
        let available = self.properties.recommended_working_set_size / 4;
        (available as usize / bytes_per_element).max(1)
    }
}

// Implement Send and Sync for MetalContext
// SAFETY: Metal objects are thread-safe when used correctly
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl std::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalContext")
            .field("device", &self.properties.name)
            .field("unified_memory", &self.properties.has_unified_memory)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_device_properties(tier: DeviceTier, has_nax: bool) -> DeviceProperties {
        DeviceProperties {
            name: "Apple M5 Test".to_string(),
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory_length: 32 * 1024,
            has_unified_memory: true,
            recommended_working_set_size: 8 * 1024 * 1024 * 1024,
            max_buffer_length: 256 * 1024 * 1024,
            gpu_family: if has_nax {
                AppleGPUFamily::Apple10
            } else {
                AppleGPUFamily::Apple9
            },
            device_tier: tier,
            has_dynamic_caching: true,
            has_hardware_ray_tracing: true,
            has_mesh_shaders: true,
            has_nax,
            architecture_gen: if has_nax { 17 } else { 16 },
            memory_bandwidth_gbps: match tier {
                DeviceTier::Base => 120.0,
                DeviceTier::Pro => 273.0,
                DeviceTier::Max => 546.0,
                DeviceTier::Ultra => 800.0,
            },
            memory_bandwidth_source: MemoryBandwidthSource::SpecTableFallback,
            gpu_core_count: 20,
            ane_core_count: 16,
            is_ultra_fusion: matches!(tier, DeviceTier::Ultra),
            die_count: if matches!(tier, DeviceTier::Ultra) {
                2
            } else {
                1
            },
        }
    }

    #[test]
    fn test_context_creation() {
        let ctx = MetalContext::new();
        assert!(
            ctx.is_ok(),
            "Should be able to create Metal context on macOS"
        );

        let ctx = ctx.unwrap();
        assert!(!ctx.properties().name.is_empty());
        assert!(ctx.properties().has_unified_memory);
    }

    #[test]
    fn test_global_context() {
        let ctx1 = MetalContext::global().unwrap();
        let ctx2 = MetalContext::global().unwrap();

        // Should be the same instance
        assert!(Arc::ptr_eq(&ctx1, &ctx2));
    }

    #[test]
    fn test_detect_gpu_family() {
        assert_eq!(detect_gpu_family("Apple M5 Max"), AppleGPUFamily::Apple10);
        assert_eq!(detect_gpu_family("Apple M5 Pro"), AppleGPUFamily::Apple10);
        assert_eq!(detect_gpu_family("Apple M5 Ultra"), AppleGPUFamily::Apple10);
        assert_eq!(detect_gpu_family("Apple M4 Ultra"), AppleGPUFamily::Apple9);
        assert_eq!(detect_gpu_family("Apple M4 Max"), AppleGPUFamily::Apple9);
        assert_eq!(detect_gpu_family("Apple M3 Pro"), AppleGPUFamily::Apple9);
        assert_eq!(detect_gpu_family("Apple M2 Ultra"), AppleGPUFamily::Apple8);
        assert_eq!(detect_gpu_family("Apple M1 Max"), AppleGPUFamily::Apple7);
        assert_eq!(detect_gpu_family("Unknown GPU"), AppleGPUFamily::Unknown);
    }

    #[test]
    fn test_has_chip_id_no_false_positives() {
        // "M1" should not match "M10", "M12", etc.
        assert!(!has_chip_id("Apple M10", "M1"));
        assert!(!has_chip_id("Apple M12 Pro", "M1"));
        assert!(has_chip_id("Apple M1 Pro", "M1"));
        assert!(has_chip_id("Apple M1", "M1"));
        // "M5" should not match "M50"
        assert!(!has_chip_id("Apple M50", "M5"));
        assert!(has_chip_id("Apple M5 Max", "M5"));
    }

    #[test]
    fn test_detect_device_tier() {
        assert_eq!(detect_device_tier("Apple M4 Ultra"), DeviceTier::Ultra);
        assert_eq!(detect_device_tier("Apple M5 Max"), DeviceTier::Max);
        assert_eq!(detect_device_tier("Apple M3 Pro"), DeviceTier::Pro);
        assert_eq!(detect_device_tier("Apple M4"), DeviceTier::Base);
    }

    #[test]
    fn test_unknown_family_ordering() {
        // Unknown must be less than all known families
        assert!(AppleGPUFamily::Unknown < AppleGPUFamily::Apple7);
        assert!(AppleGPUFamily::Unknown < AppleGPUFamily::Apple10);
        // Known families are ordered correctly
        assert!(AppleGPUFamily::Apple7 < AppleGPUFamily::Apple8);
        assert!(AppleGPUFamily::Apple9 < AppleGPUFamily::Apple10);
    }

    #[test]
    fn test_m4_ultra_properties() {
        let family = detect_gpu_family("Apple M4 Ultra");
        let tier = detect_device_tier("Apple M4 Ultra");
        assert_eq!(family, AppleGPUFamily::Apple9);
        assert_eq!(tier, DeviceTier::Ultra);
        assert_eq!(estimate_gpu_cores("Apple M4 Ultra", tier), 80);
        assert_eq!(estimate_ane_cores(tier), 32);
        assert_eq!(estimate_memory_bandwidth(tier, family), 800.0);
        assert_eq!(architecture_gen(family), 16);
        // M4 Ultra should NOT have NAX (only Apple10+)
        assert!(family < AppleGPUFamily::Apple10);
    }

    #[test]
    fn test_m5_nax_properties() {
        let family = detect_gpu_family("Apple M5 Pro");
        let tier = detect_device_tier("Apple M5 Pro");
        assert_eq!(family, AppleGPUFamily::Apple10);
        assert_eq!(tier, DeviceTier::Pro);
        // M5 should have NAX
        assert!(family >= AppleGPUFamily::Apple10);
        assert_eq!(architecture_gen(family), 17);
    }

    #[test]
    fn test_memory_bandwidth_cache_key_tracks_device_shape() {
        let key = memory_bandwidth_cache_key(
            "Apple M4 Max",
            AppleGPUFamily::Apple9,
            DeviceTier::Max,
            40,
            1,
        );
        assert_eq!(key, "Apple M4 Max:Apple9:Max:40:1");
    }

    #[test]
    fn test_measured_bandwidth_plausibility_filter() {
        assert!(measured_bandwidth_is_plausible(300.0));
        assert!(!measured_bandwidth_is_plausible(0.0));
        assert!(!measured_bandwidth_is_plausible(10_000.0));
    }

    #[test]
    fn test_measured_copy_buffer_size_is_clamped() {
        assert_eq!(
            measured_copy_buffer_size_bytes(8 * 1024 * 1024),
            32 * 1024 * 1024
        );
        assert_eq!(
            measured_copy_buffer_size_bytes(1024 * 1024 * 1024),
            128 * 1024 * 1024
        );
        assert_eq!(
            measured_copy_buffer_size_bytes(256 * 1024 * 1024),
            64 * 1024 * 1024
        );
    }

    #[test]
    fn test_memory_bandwidth_cache_roundtrip() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("memory_bandwidth.json");
        let key = "Apple M4 Max:Apple9:Max:40:1";
        let entry = MemoryBandwidthCacheEntry {
            bandwidth_gbps: 412.5,
            source: MemoryBandwidthSource::MeasuredGpuCopy,
        };

        store_cached_memory_bandwidth(&path, key, &entry);
        let cached = load_cached_memory_bandwidth(&path, key).unwrap();
        assert_eq!(cached.bandwidth_gbps, entry.bandwidth_gbps);
        assert_eq!(cached.source, entry.source);
    }

    #[test]
    fn test_memory_bandwidth_cache_rejects_implausible_values() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("memory_bandwidth.json");
        let key = "Apple M4 Max:Apple9:Max:40:1";
        let entry = MemoryBandwidthCacheEntry {
            bandwidth_gbps: 5_000.0,
            source: MemoryBandwidthSource::MeasuredGpuCopy,
        };

        store_cached_memory_bandwidth(&path, key, &entry);
        assert!(load_cached_memory_bandwidth(&path, key).is_none());
    }

    #[test]
    fn test_mpp_gate_requires_nax() {
        let props = test_device_properties(DeviceTier::Max, false);
        assert!(!props.should_consider_mpp_gemm(8, 256, 128, true));
    }

    #[test]
    fn test_mpp_gate_rejects_tiny_problem() {
        let props = test_device_properties(DeviceTier::Max, true);
        assert!(!props.should_consider_mpp_gemm(1, 64, 64, true));
    }

    #[test]
    fn test_mpp_gate_prefers_large_aligned_problem() {
        let props = test_device_properties(DeviceTier::Max, true);
        assert!(props.should_consider_mpp_gemm(8, 256, 128, true));
    }

    #[test]
    fn test_mpp_gate_is_more_conservative_for_base_f32() {
        let props = test_device_properties(DeviceTier::Base, true);
        assert!(!props.should_consider_mpp_gemm(8, 256, 128, false));
        assert!(props.should_consider_mpp_gemm(64, 256, 128, false));
    }

    #[test]
    fn test_mpp_tile_config_defaults_by_tier() {
        assert_eq!(
            test_device_properties(DeviceTier::Base, true).mpp_tile_config(),
            (64, 32, 128, 2)
        );
        assert_eq!(
            test_device_properties(DeviceTier::Pro, true).mpp_tile_config(),
            (64, 64, 128, 4)
        );
    }

    #[test]
    fn test_context_reports_bandwidth_source() {
        let ctx = MetalContext::new().unwrap();
        assert!(ctx.properties().memory_bandwidth_gbps > 0.0);
        assert!(matches!(
            ctx.properties().memory_bandwidth_source,
            MemoryBandwidthSource::MeasuredGpuCopy | MemoryBandwidthSource::SpecTableFallback
        ));
    }
}
