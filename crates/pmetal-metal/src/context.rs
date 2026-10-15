#![allow(unsafe_code)]

//! Metal device and command queue management.
//!
//! This module provides thread-safe access to the Metal device and command queue,
//! using a global singleton pattern for efficiency.
//!
//! Device properties are asked of the device: the GPU family from
//! `supportsFamily`, the generation from the architecture name, the core count
//! from the IO registry. Memory bandwidth is the one figure Metal does not
//! report, so it is measured once per machine and checked against the
//! published peak for the chip.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};
use tracing::{debug, info};

use crate::chip;
use crate::error::{MetalError, Result};
use crate::pipeline::PipelineCache;
use crate::tuna::Tuner;

/// Global Metal context singleton using OnceLock for thread-safe lazy initialization.
static GLOBAL_CONTEXT: OnceLock<Result<Arc<MetalContext>>> = OnceLock::new();
static DEVICE_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// A digest of the compiled shader libraries, computed by the build script.
/// Cached tuning results are scoped to it.
const KERNEL_FINGERPRINT: &str = env!("PMETAL_KERNEL_FINGERPRINT");

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
/// Determined by the chip variant in the device's marketing name
/// (base/Pro/Max/Ultra).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTier {
    /// Base chips (M1/M2/M3/M4/M5): 68-153 GB/s bandwidth
    Base,
    /// Pro chips: 150-307 GB/s bandwidth
    Pro,
    /// Max chips: 300-614 GB/s bandwidth
    Max,
    /// Ultra chips: 800+ GB/s bandwidth
    Ultra,
}

/// Source of the detected memory bandwidth figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryBandwidthSource {
    /// Measured on this machine via a GPU copy benchmark and cached on disk.
    MeasuredGpuCopy,
    /// Apple's published peak for this chip (or, for a chip the table doesn't
    /// know, a figure for its tier).
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

    /// GPU family (Apple7-Apple10), from `supportsFamily`.
    pub gpu_family: AppleGPUFamily,

    /// Device performance tier.
    pub device_tier: DeviceTier,

    /// Whether Dynamic Caching is supported (Apple9+).
    pub has_dynamic_caching: bool,

    /// Whether hardware ray tracing is supported (Apple9+).
    pub has_hardware_ray_tracing: bool,

    /// Whether mesh shaders are supported (Apple9+).
    pub has_mesh_shaders: bool,

    /// Whether Neural Accelerators (NAX) are available in GPU cores and the OS
    /// can drive them (M5+ on macOS 26.2+). Enables optimized GEMM,
    /// quantization, and attention kernels via Metal 4.0.
    pub has_nax: bool,

    /// Architecture generation: the number in the Metal architecture name
    /// (`applegpu_g16s` is 16). 13=M1, 14=M2, 15=M3, 16=M4, 17=M5. The
    /// runtime reads the same number, 0 if the name doesn't parse.
    pub architecture_gen: u32,

    /// Memory bandwidth in GB/s: measured on this machine when a measurement
    /// is available and agrees with the chip's published peak, otherwise that
    /// peak.
    pub memory_bandwidth_gbps: f64,

    /// How `memory_bandwidth_gbps` was obtained.
    pub memory_bandwidth_source: MemoryBandwidthSource,

    /// Number of GPU cores, as the IO registry reports them (0 if unknown).
    pub gpu_core_count: u32,

    /// Number of ANE (Neural Engine) cores (16 for Pro/Max, 32 for Ultra, 0 if unknown).
    pub ane_core_count: u32,

    /// Whether this is an UltraFusion (multi-die) chip.
    pub is_ultra_fusion: bool,

    /// Number of dies. 1 for single-die, 2 for UltraFusion.
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
}

/// Creating a command buffer without leaking it.
///
/// Metal returns a command buffer autoreleased. A Rust thread has no
/// autorelease pool of its own, so `queue.commandBuffer()` called directly
/// leaves the buffer (and the encoders and resources it retains) alive until
/// the thread exits: about 1.8 KiB per dispatch, without bound. This creates
/// the buffer inside a pool and returns an owned handle, which is all the
/// caller needs.
pub trait CommandQueueExt {
    /// A new command buffer on this queue.
    ///
    /// # Errors
    ///
    /// [`MetalError::CommandBufferCreation`] if the queue declines.
    fn new_command_buffer(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>>;
}

impl CommandQueueExt for ProtocolObject<dyn MTLCommandQueue> {
    fn new_command_buffer(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        objc2::rc::autoreleasepool(|_| self.commandBuffer())
            .ok_or(MetalError::CommandBufferCreation)
    }
}

/// The GPU family the device reports supporting, highest first.
fn gpu_family_of(device: &ProtocolObject<dyn MTLDevice>) -> AppleGPUFamily {
    [
        (MTLGPUFamily::Apple10, AppleGPUFamily::Apple10),
        (MTLGPUFamily::Apple9, AppleGPUFamily::Apple9),
        (MTLGPUFamily::Apple8, AppleGPUFamily::Apple8),
        (MTLGPUFamily::Apple7, AppleGPUFamily::Apple7),
    ]
    .into_iter()
    .find(|&(family, _)| device.supportsFamily(family))
    .map_or(AppleGPUFamily::Unknown, |(_, ours)| ours)
}

/// Whether the OS is macOS 26.2 or later, where the Metal 4 tensor operations
/// the neural accelerators run are supported.
fn os_at_least_26_2() -> bool {
    NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
        majorVersion: 26,
        minorVersion: 2,
        patchVersion: 0,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory bandwidth
// ─────────────────────────────────────────────────────────────────────────────

/// Layout version of the bandwidth cache file. Bumping it ignores every entry
/// an older layout (or an older table) wrote.
const BANDWIDTH_CACHE_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryBandwidthCacheEntry {
    bandwidth_gbps: f64,
    source: MemoryBandwidthSource,
}

fn memory_bandwidth_cache_key(
    name: &str,
    arch_gen: u32,
    tier: DeviceTier,
    gpu_cores: u32,
    die_count: u32,
) -> String {
    format!("v{BANDWIDTH_CACHE_VERSION}:{name}:g{arch_gen}:{tier:?}:{gpu_cores}:{die_count}")
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

/// A measured figure is believable if it is finite and within reach of the
/// chip's peak: a copy can't beat the bus, and one far below it means the GPU
/// was busy with something else. Without a published peak, only a sanity range.
fn measured_bandwidth_is_plausible(bandwidth_gbps: f64, spec_gbps: Option<f64>) -> bool {
    if !bandwidth_gbps.is_finite() {
        return false;
    }
    match spec_gbps {
        Some(spec) => (0.25 * spec..=1.1 * spec).contains(&bandwidth_gbps),
        None => (25.0..=2_000.0).contains(&bandwidth_gbps),
    }
}

/// Only a measurement is worth keeping; a spec-table value is recomputed for free.
fn load_cached_memory_bandwidth(
    path: &Path,
    cache_key: &str,
    spec_gbps: Option<f64>,
) -> Option<MemoryBandwidthCacheEntry> {
    let entry = load_memory_bandwidth_cache(path)?.remove(cache_key)?;
    (entry.source == MemoryBandwidthSource::MeasuredGpuCopy
        && measured_bandwidth_is_plausible(entry.bandwidth_gbps, spec_gbps))
    .then_some(entry)
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

    // A temp file named for this process, so two processes writing at once
    // each rename a complete file.
    let tmp_path = path.with_extension(format!("json.{}.tmp", std::process::id()));
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

fn measured_copy_buffer_size_bytes(recommended_working_set_size: u64) -> usize {
    let quarter_working_set = (recommended_working_set_size / 4) as usize;
    quarter_working_set.clamp(32 * 1024 * 1024, 128 * 1024 * 1024)
}

/// The best copy bandwidth in GB/s over a few runs of a vectorized copy,
/// timed by the GPU's own clock rather than the host's, so command-buffer
/// submission and wake-up latency don't count against the bus.
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

    // One copy; returns the seconds the GPU spent on it.
    let copy = |src: &Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
                dst: &Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>|
     -> Result<f64> {
        let command_buffer = command_queue.new_command_buffer()?;
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

        Ok(command_buffer.GPUEndTime() - command_buffer.GPUStartTime())
    };

    for _ in 0..2 {
        copy(&src, &dst)?;
        copy(&dst, &src)?;
    }

    let mut best_seconds = f64::INFINITY;
    for i in 0..6 {
        let seconds = if i % 2 == 0 {
            copy(&src, &dst)?
        } else {
            copy(&dst, &src)?
        };
        if seconds > 0.0 {
            best_seconds = best_seconds.min(seconds);
        }
    }
    if !best_seconds.is_finite() {
        return Err(MetalError::ExecutionFailed(
            "bandwidth probe reported no GPU time".to_string(),
        ));
    }

    // Each copy reads the buffer once and writes it once.
    Ok(copy_bytes as f64 * 2.0 / best_seconds / 1e9)
}

/// Memory bandwidth for a chip: the cached measurement if there is one, else a
/// new measurement, else the published peak.
///
/// Only a believable measurement is cached. A failed or implausible one (the
/// GPU was busy, the probe errored) falls back to the peak *without* being
/// stored, so the next start measures again instead of keeping the fallback.
fn resolve_memory_bandwidth_with(
    cache_path: Option<&Path>,
    cache_key: &str,
    spec_gbps: Option<f64>,
    fallback_gbps: f64,
    measure: impl FnOnce() -> Result<f64>,
) -> (f64, MemoryBandwidthSource) {
    let peak = spec_gbps.unwrap_or(fallback_gbps);

    if let Some(path) = cache_path
        && let Some(entry) = load_cached_memory_bandwidth(path, cache_key, spec_gbps)
    {
        return (entry.bandwidth_gbps, entry.source);
    }

    match measure() {
        Ok(measured) if measured_bandwidth_is_plausible(measured, spec_gbps) => {
            if let Some(path) = cache_path {
                store_cached_memory_bandwidth(
                    path,
                    cache_key,
                    &MemoryBandwidthCacheEntry {
                        bandwidth_gbps: measured,
                        source: MemoryBandwidthSource::MeasuredGpuCopy,
                    },
                );
            }
            (measured, MemoryBandwidthSource::MeasuredGpuCopy)
        }
        Ok(measured) => {
            debug!("Ignoring implausible bandwidth measurement {measured:.1} GB/s");
            (peak, MemoryBandwidthSource::SpecTableFallback)
        }
        Err(error) => {
            debug!("Bandwidth measurement failed: {error}");
            (peak, MemoryBandwidthSource::SpecTableFallback)
        }
    }
}

impl MetalContext {
    /// Probe whether a Metal device is visible to this process without
    /// constructing the full context or loading kernel libraries.
    pub fn device_available() -> bool {
        *DEVICE_AVAILABLE.get_or_init(|| MTLCreateSystemDefaultDevice().is_some())
    }

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

        // Ask the device what it is.
        let name = device.name().to_string();
        let architecture = device.architecture().name().to_string();
        let architecture_gen = chip::architecture_gen(&architecture);
        let gpu_family = gpu_family_of(&device);
        let device_tier = chip::tier_from_name(&name);
        let gpu_core_count = chip::registry_gpu_core_count()
            .unwrap_or_else(|| chip::full_bin_gpu_cores(architecture_gen, device_tier));
        let ane_core_count = chip::ane_cores(device_tier);
        // An Ultra is two Max dies joined by UltraFusion.
        let die_count = if device_tier == DeviceTier::Ultra {
            2
        } else {
            1
        };
        let is_ultra_fusion = die_count > 1;
        let recommended_working_set_size = device.recommendedMaxWorkingSetSize();
        let has_nax = chip::has_neural_accelerators(&architecture, os_at_least_26_2());

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

        let spec_gbps = chip::spec_bandwidth_gbps(architecture_gen, device_tier, gpu_core_count);
        let (memory_bandwidth_gbps, memory_bandwidth_source) = resolve_memory_bandwidth_with(
            memory_bandwidth_cache_path().as_deref(),
            &memory_bandwidth_cache_key(
                &name,
                architecture_gen,
                device_tier,
                gpu_core_count,
                die_count,
            ),
            spec_gbps,
            chip::tier_bandwidth_gbps(device_tier),
            || {
                measure_gpu_copy_bandwidth_gbps(
                    &device,
                    &command_queue,
                    &mut cache,
                    recommended_working_set_size,
                )
            },
        );

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
            architecture_gen,
            memory_bandwidth_gbps,
            memory_bandwidth_source,
            gpu_core_count,
            ane_core_count,
            die_count,
            is_ultra_fusion,
        };

        debug!("Device properties: {:?}", properties);
        info!(
            "GPU: {:?} {:?} (gen {}), {} cores{}, {:.0} GB/s [{:?}], NAX: {}, ANE: {} cores",
            properties.gpu_family,
            properties.device_tier,
            properties.architecture_gen,
            properties.gpu_core_count,
            if properties.is_ultra_fusion {
                format!(" ({} dies, UltraFusion)", properties.die_count)
            } else {
                String::new()
            },
            properties.memory_bandwidth_gbps,
            properties.memory_bandwidth_source,
            properties.has_nax,
            properties.ane_core_count,
        );

        let pipeline_cache = RwLock::new(cache);
        let tuner = Arc::new(Tuner::for_device(&properties, KERNEL_FINGERPRINT));

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

    /// Whether the Metal 4 / MPP kernels can run on this device: the build
    /// compiled them, the device has neural accelerators the OS can drive, and
    /// the Metal 4 library loaded.
    pub fn supports_mpp(&self) -> bool {
        #[cfg(has_metal4)]
        {
            self.properties.has_nax && self.pipeline_cache().metal4_library().is_some()
        }
        #[cfg(not(has_metal4))]
        {
            false
        }
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

// SAFETY: the Metal device, command queue and pipeline states a context holds
// are documented as safe to use from any thread (Apple's Metal Programming
// Guide: "Command queues, devices, libraries and pipeline states are thread
// safe"). The pipeline cache is behind a lock and the tuner is internally
// synchronized.
unsafe impl Send for MetalContext {}
// SAFETY: see above.
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
    use std::cell::Cell;
    use tempfile::tempdir;

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

    /// Resident memory of this process, in KiB.
    fn resident_kib() -> i64 {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .expect("ps");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .expect("ps prints a number")
    }

    /// Command buffers made through the queue extension do not pile up on a
    /// thread with no autorelease pool; made with `queue.commandBuffer()`
    /// directly, 10,000 of them held about 18 MiB.
    #[test]
    fn new_command_buffers_do_not_accumulate() {
        let ctx = MetalContext::new().unwrap();
        let run = || {
            for _ in 0..10_000 {
                let command_buffer = ctx.command_queue().new_command_buffer().unwrap();
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
            }
        };
        run();
        let before = resident_kib();
        run();
        let growth_kib = resident_kib() - before;
        assert!(
            growth_kib < 6 * 1024,
            "10k command buffers grew resident memory by {growth_kib} KiB"
        );
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

    /// Every figure on a live context is the device's own answer, and agrees
    /// with a second way of asking.
    #[test]
    fn a_live_context_reports_what_the_device_says() {
        let ctx = MetalContext::new().unwrap();
        let props = ctx.properties();
        let device = ctx.device();

        // The family is what `supportsFamily` says, not a guess from the name.
        let expected = if device.supportsFamily(MTLGPUFamily::Apple10) {
            AppleGPUFamily::Apple10
        } else if device.supportsFamily(MTLGPUFamily::Apple9) {
            AppleGPUFamily::Apple9
        } else if device.supportsFamily(MTLGPUFamily::Apple8) {
            AppleGPUFamily::Apple8
        } else {
            AppleGPUFamily::Apple7
        };
        assert_eq!(props.gpu_family, expected);

        // The generation is the architecture name's, and fits the family.
        let architecture = device.architecture().name().to_string();
        assert_eq!(
            props.architecture_gen,
            chip::architecture_gen(&architecture)
        );
        assert!(props.architecture_gen >= 13, "{architecture}");
        let family_for_gen = match props.architecture_gen {
            13 => AppleGPUFamily::Apple7,
            14 => AppleGPUFamily::Apple8,
            15 | 16 => AppleGPUFamily::Apple9,
            _ => AppleGPUFamily::Apple10,
        };
        assert_eq!(props.gpu_family, family_for_gen, "{architecture}");

        // NAX only where the generation has the accelerators.
        if props.architecture_gen < 17 {
            assert!(!props.has_nax);
            assert!(!ctx.supports_mpp());
        }

        // The core count is the registry's, and the bandwidth is of that part.
        assert_eq!(Some(props.gpu_core_count), chip::registry_gpu_core_count());
        let spec = chip::spec_bandwidth_gbps(
            props.architecture_gen,
            props.device_tier,
            props.gpu_core_count,
        );
        assert!(
            measured_bandwidth_is_plausible(props.memory_bandwidth_gbps, spec),
            "{} GB/s on {} ({} cores)",
            props.memory_bandwidth_gbps,
            props.name,
            props.gpu_core_count
        );
        assert_eq!(
            props.is_ultra_fusion,
            props.device_tier == DeviceTier::Ultra
        );
        assert_eq!(
            props.recommended_working_set_size,
            device.recommendedMaxWorkingSetSize()
        );
    }

    #[test]
    fn test_memory_bandwidth_cache_key_tracks_device_shape() {
        let key = memory_bandwidth_cache_key("Apple M4 Max", 16, DeviceTier::Max, 32, 1);
        assert_eq!(key, "v2:Apple M4 Max:g16:Max:32:1");
        // A binned part is a different machine from the full one.
        assert_ne!(
            key,
            memory_bandwidth_cache_key("Apple M4 Max", 16, DeviceTier::Max, 40, 1)
        );
    }

    #[test]
    fn test_measured_bandwidth_plausibility_filter() {
        assert!(measured_bandwidth_is_plausible(300.0, None));
        assert!(!measured_bandwidth_is_plausible(0.0, None));
        assert!(!measured_bandwidth_is_plausible(10_000.0, None));
        assert!(!measured_bandwidth_is_plausible(f64::NAN, None));
        // Against a published peak: not above it, not a fraction of it.
        assert!(measured_bandwidth_is_plausible(480.0, Some(546.0)));
        assert!(!measured_bandwidth_is_plausible(700.0, Some(546.0)));
        assert!(!measured_bandwidth_is_plausible(60.0, Some(546.0)));
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
        let key = "v2:Apple M4 Max:g16:Max:40:1";
        let entry = MemoryBandwidthCacheEntry {
            bandwidth_gbps: 412.5,
            source: MemoryBandwidthSource::MeasuredGpuCopy,
        };

        store_cached_memory_bandwidth(&path, key, &entry);
        let cached = load_cached_memory_bandwidth(&path, key, Some(546.0)).unwrap();
        assert_eq!(cached.bandwidth_gbps, entry.bandwidth_gbps);
        assert_eq!(cached.source, entry.source);
    }

    #[test]
    fn test_memory_bandwidth_cache_rejects_implausible_values() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("memory_bandwidth.json");
        let key = "v2:Apple M4 Max:g16:Max:40:1";
        let entry = MemoryBandwidthCacheEntry {
            bandwidth_gbps: 5_000.0,
            source: MemoryBandwidthSource::MeasuredGpuCopy,
        };

        store_cached_memory_bandwidth(&path, key, &entry);
        assert!(load_cached_memory_bandwidth(&path, key, Some(546.0)).is_none());
    }

    /// A probe that fails once must not leave a fallback figure in the cache
    /// for every later start to read back as if it were the answer.
    #[test]
    fn a_failed_measurement_is_not_cached() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("memory_bandwidth.json");
        let key = "v2:chip";

        let (gbps, source) =
            resolve_memory_bandwidth_with(Some(&path), key, Some(546.0), 400.0, || {
                Err(MetalError::ExecutionFailed("probe failed".into()))
            });
        assert_eq!(gbps, 546.0);
        assert_eq!(source, MemoryBandwidthSource::SpecTableFallback);

        // The next start measures, and its answer is the one that sticks.
        let (gbps, source) =
            resolve_memory_bandwidth_with(Some(&path), key, Some(546.0), 400.0, || Ok(480.0));
        assert_eq!(
            (gbps, source),
            (480.0, MemoryBandwidthSource::MeasuredGpuCopy)
        );
    }

    /// An entry an older build wrote as a fallback is not a measurement.
    #[test]
    fn a_cached_fallback_is_measured_again() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("memory_bandwidth.json");
        let key = "v2:chip";
        store_cached_memory_bandwidth(
            &path,
            key,
            &MemoryBandwidthCacheEntry {
                bandwidth_gbps: 273.0,
                source: MemoryBandwidthSource::SpecTableFallback,
            },
        );

        let (gbps, source) =
            resolve_memory_bandwidth_with(Some(&path), key, Some(546.0), 400.0, || Ok(500.0));
        assert_eq!(
            (gbps, source),
            (500.0, MemoryBandwidthSource::MeasuredGpuCopy)
        );
    }

    #[test]
    fn a_cached_measurement_is_not_measured_again() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("memory_bandwidth.json");
        let key = "v2:chip";
        store_cached_memory_bandwidth(
            &path,
            key,
            &MemoryBandwidthCacheEntry {
                bandwidth_gbps: 490.0,
                source: MemoryBandwidthSource::MeasuredGpuCopy,
            },
        );

        let probes = Cell::new(0);
        let (gbps, source) =
            resolve_memory_bandwidth_with(Some(&path), key, Some(546.0), 400.0, || {
                probes.set(probes.get() + 1);
                Ok(1.0)
            });
        assert_eq!(
            (gbps, source),
            (490.0, MemoryBandwidthSource::MeasuredGpuCopy)
        );
        assert_eq!(probes.get(), 0);
    }

    /// A reading above the bus's published peak is a broken probe, not a fast chip.
    #[test]
    fn a_measurement_beyond_the_published_peak_is_discarded() {
        let (gbps, source) =
            resolve_memory_bandwidth_with(None, "k", Some(273.0), 200.0, || Ok(900.0));
        assert_eq!(
            (gbps, source),
            (273.0, MemoryBandwidthSource::SpecTableFallback)
        );
    }

    #[test]
    fn a_chip_with_no_published_peak_falls_back_to_its_tier() {
        let (gbps, source) = resolve_memory_bandwidth_with(None, "k", None, 400.0, || {
            Err(MetalError::ExecutionFailed("no".into()))
        });
        assert_eq!(
            (gbps, source),
            (400.0, MemoryBandwidthSource::SpecTableFallback)
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
